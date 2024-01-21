import functools
import glob
import os
import re
import sys
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import fiona
import fiona.transform
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import rasterio.merge
import shapely
import torch

# from torchvision.datasets.sentinel import Sentinel2
import torchgeo
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from rtree.index import Index, Property
from torch import Tensor
from torch.utils.data import Dataset
from torchgeo.datasets.geo import GeoDataset, RasterDataset
from torchgeo.datasets.utils import (
    BoundingBox,
    concat_samples,
    disambiguate_timestamp,
    merge_samples,
    path_is_vsi,
)
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader as pil_loader


try:
    import dapla as dp
except ImportError:
    pass

from ..io._is_dapla import is_dapla
from ..io.opener import opener
from .raster import get_from_regex


class DaplaRasterDataset(RasterDataset):
    def init__(
        self,
        root: str,
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
    ) -> None:
        self.transforms = transforms
        self.index = Index(interleaved=False, properties=Property(dimension=3))
        self.root = root
        self.cache = cache

        self.paths = paths
        self.bands = bands or self.all_bands
        self.cache = cache

        # Populate the dataset index
        i = 0
        pathname = os.path.join(root, "**")
        if is_dapla():
            filepaths = dp.FileClient.get_gcs_file_system().glob(
                pathname, recursive=True
            )
        else:
            filepaths = glob.iglob(pathname, recursive=True)
        for filepath in filepaths:
            print(filepath, self.name_regex)
            if get_from_regex(filepath, self.name_regex) is None:
                continue
            try:
                with opener(filepath) as f:
                    with rasterio.open(f) as src:
                        # See if file has a color map
                        if len(self.cmap) == 0:
                            try:
                                self.cmap = src.colormap(1)
                            except ValueError:
                                pass

                        if crs is None:
                            crs = src.crs
                        if res is None:
                            res = src.res[0]

                        with WarpedVRT(src, crs=crs) as vrt:
                            minx, miny, maxx, maxy = vrt.bounds
            except rasterio.errors.RasterioIOError:
                # Skip files that rasterio is unable to read
                continue

            date = get_from_regex(filepath, self.date_regex)
            if date:
                mint, maxt = disambiguate_timestamp(date, self.date_format)
            else:
                mint, maxt = 0, sys.maxsize

            coords = (minx, maxx, miny, maxy, mint, maxt)
            self.index.insert(i, coords, filepath)
            i += 1

        if i == 0:
            raise FileNotFoundError(
                f"No {self.__class__.__name__} data was found in '{root}'"
            )

        self._crs = cast(CRS, crs)
        self.res = cast(float, res)

    def init__(
        self,
        paths: Union[str, Iterable[str]] = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        bands: Optional[Sequence[str]] = None,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        cache: bool = True,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            FileNotFoundError: if no files are found in ``paths``

        .. versionchanged:: 0.5
           *root* was renamed to *paths*.
        """
        self.transforms = transforms
        self.index = Index(interleaved=False, properties=Property(dimension=3))
        self.paths = paths
        self.bands = bands or self.all_bands
        self.cache = cache

        # Populate the dataset index
        i = 0
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        for filepath in self.files:
            match = re.match(filename_regex, os.path.basename(filepath))
            if match is not None:
                try:
                    with rasterio.open(filepath) as src:
                        # See if file has a color map
                        if len(self.cmap) == 0:
                            try:
                                self.cmap = src.colormap(1)
                            except ValueError:
                                pass

                        if crs is None:
                            crs = src.crs
                        if res is None:
                            res = src.res[0]

                        with WarpedVRT(src, crs=crs) as vrt:
                            minx, miny, maxx, maxy = vrt.bounds
                except rasterio.errors.RasterioIOError:
                    # Skip files that rasterio is unable to read
                    continue
                else:
                    mint: float = 0
                    maxt: float = sys.maxsize
                    if "date" in match.groupdict():
                        date = match.group("date")
                        mint, maxt = disambiguate_timestamp(date, self.date_format)

                    coords = (minx, maxx, miny, maxy, mint, maxt)
                    self.index.insert(i, coords, filepath)
                    i += 1

        if i == 0:
            msg = (
                f"No {self.__class__.__name__} data was found "
                f"in `paths={self.paths!r}'`"
            )
            if self.bands:
                msg += f" with `bands={self.bands}`"
            raise FileNotFoundError(msg)

        if not self.separate_files:
            self.band_indexes = None
            if self.bands:
                if self.all_bands:
                    self.band_indexes = [
                        self.all_bands.index(i) + 1 for i in self.bands
                    ]
                else:
                    msg = (
                        f"{self.__class__.__name__} is missing an `all_bands` "
                        "attribute, so `bands` cannot be specified."
                    )
                    raise AssertionError(msg)

        self._crs = cast(CRS, crs)
        self._res = cast(float, res)

    @property
    def files(self) -> set[str]:
        """A list of all files in the dataset.

        Returns:
            All files in the dataset.

        .. versionadded:: 0.5
        """
        if isinstance(self.paths, str):
            paths: list[str] = [self.paths]
        else:
            paths = self.paths

        # Using set to remove any duplicates if directories are overlapping
        files: set[str] = set()
        for path in paths:
            if is_dapla():
                files |= {
                    x
                    for x in dp.FileClient.get_gcs_file_system().glob(
                        path + "**", recursive=True
                    )
                    if "." in x
                }
                continue

            if os.path.isdir(path):
                pathname = os.path.join(path, "**")  # , self.filename_glob)
                files |= set(glob.iglob(pathname, recursive=True))
            elif os.path.isfile(path) or path_is_vsi(path):
                files.add(path)
            else:
                warnings.warn(
                    f"Could not find any relevant files for provided path '{path}'. "
                    f"Path was ignored.",
                    UserWarning,
                )

        return files

    # def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
    #     """Retrieve image/mask and metadata indexed by query.

    #     Args:
    #         query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

    #     Returns:
    #         sample of image/mask and metadata at that index

    #     Raises:
    #         IndexError: if query is not found in the index
    #     """
    #     hits = self.index.intersection(tuple(query), objects=True)
    #     filepaths = [hit.object for hit in hits]

    #     if not filepaths:
    #         raise IndexError(
    #             f"query: {query} not found in index with bounds: {self.bounds}"
    #         )

    #     if self.separate_files:
    #         data_list: List[Tensor] = []
    #         for band in getattr(self, "bands", self.all_bands):
    #             band_filepaths = []
    #             for filepath in filepaths:
    #                 filename = os.path.basename(filepath)
    #                 directory = os.path.dirname(filepath)
    #                 if get_from_regex(filepath, self.name_regex) is None:
    #                     date = get_from_regex(filepath, self.date_regex)
    #                     if date:
    #                         start = match.start("band")
    #                         end = match.end("band")
    #                         filename = filename[:start] + band + filename[end:]
    #                     if "resolution" in match.groupdict():
    #                         start = match.start("resolution")
    #                         end = match.end("resolution")
    #                         filename = filename[:start] + "*" + filename[end:]
    #                 band_filepaths.append(filepath)
    #             data_list.append(self._merge_files(band_filepaths, query))
    #         data = torch.cat(data_list)
    #     else:
    #         data = self._merge_files(filepaths, query)

    #     key = "image" if self.is_image else "mask"
    #     sample = {key: data, "crs": self.crs, "bbox": query}

    #     if self.transforms is not None:
    #         sample = self.transforms(sample)

    #     return sample

    def _load_warp_file(self, filepath: str) -> DatasetReader:
        """Load and warp a file to the correct CRS and resolution.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        with opener(filepath) as f:
            src = rasterio.open(f)

            # Only warp if necessary
            if src.crs != self.crs:
                vrt = WarpedVRT(src, crs=self.crs)
                src.close()
                return vrt
            else:
                return src


class Sentinel2(DaplaRasterDataset):
    name_regex = r"B\d{1,2}A|B\d{1,2}"
    date_format: str = "%Y%m%d"
    date_regex = r"20\d{8}"
    filename_glob = None  # "T*_*_B02_*m.*"

    all_bands = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    ]
    rgb_bands = ["B04", "B03", "B02"]

    separate_files = True

    # name_regex = r"""
    #     ^SENTINEL2X_
    #     _(?P<date>\d{8})
    #     -(?P<tile>\d{2}[A-Z]{3})
    #     _(?P<band>B\d{1,2}[\dA])
    #     # _(?P<resolution>\d{2}m)
    #     \..*$
    # """

    # name_regex = r"""
    #     ^SENTINEL2X_
    #     _(?P<date>\d{8})
    #     -(?P<tile>\d{2}[A-Z]{3})
    #     _(?P<band>B\d{1,2}[\dA])
    #     \..*$
    # """


"SENTINEL2X_20230415-230437-251_L3A_T32VLL_C_V1-3_FRC_B11_clipped.tif"

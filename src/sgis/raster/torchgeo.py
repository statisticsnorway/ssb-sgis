import glob
import os
import re
import sys
import warnings
from typing import Any, Callable, Iterable, Optional, Sequence, Union, cast

import rasterio
import rasterio.merge
from pyproj import CRS
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from rtree.index import Index, Property
from torchgeo.datasets.geo import GeoDataset, RasterDataset
from torchgeo.datasets.sentinel import Sentinel2 as TorchgeoSentinel2
from torchgeo.datasets.utils import disambiguate_timestamp


try:
    import dapla as dp
except ImportError:
    pass

try:
    from gcsfs.core import GCSFile
except ImportError:

    class GCSFile:
        pass


from ..io._is_dapla import is_dapla
from ..io.opener import opener
from .bands import SENTINEL2_FILENAME_REGEX


class PathLikeGCSFile(GCSFile, os.PathLike):
    def __init__(self, *args, **kwargs):
        print(args)
        super().__init__(*args, **kwargs)

    def __fspath__(self):
        return self.full_name


class GCSRasterDataset(RasterDataset):
    """Wrapper around torchgeo's RasterDataset that works in and outside of Dapla (stat norway)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __init__(
        self,
        paths: Union[str, Iterable[str]] = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        bands: Optional[Sequence[str]] = None,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        cache: bool = True,
    ) -> None:
        """Copied from torcgeo version 0.5.2, but readable in GCS."""
        self.transforms = transforms
        self.index = Index(interleaved=False, properties=Property(dimension=3))

        self.paths = paths
        self.bands = bands or self.all_bands
        self.cache = cache

        file_system = dp.FileClient.get_gcs_file_system()

        # Populate the dataset index
        i = 0
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        for filepath in self.files:
            try:
                filepath = filepath.full_name
            except AttributeError:
                pass
            match = re.match(filename_regex, os.path.basename(filepath))
            if match is not None:
                try:
                    with opener(filepath, file_system=file_system) as f:
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
    def files(self) -> set[PathLikeGCSFile] | set[str]:
        """A list of all files in the dataset.

        Returns:
            All files in the dataset.

        .. versionadded:: 0.5
        """
        if isinstance(self.paths, str):
            paths: list[str] = [self.paths]
        else:
            paths = self.paths

        if is_dapla():
            fs = dp.FileClient.get_gcs_file_system()
            return _get_gcs_paths(
                paths, filename_glob=self.filename_glob, file_system=fs
            )

        # Using set to remove any duplicates if directories are overlapping
        files: set[str] = set()
        for path in paths:
            if os.path.isdir(path):
                pathname = os.path.join(path, "**", self.filename_glob)
                files |= {
                    x for x in glob.iglob(pathname, recursive=True) if os.path.isfile(x)
                }
            elif os.path.isfile(path):
                files.add(path)
            else:
                warnings.warn(
                    f"Could not find any relevant files for provided path '{path}'. "
                    f"Path was ignored.",
                    UserWarning,
                )

        return files

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


def _get_gcs_paths(
    paths: str | Iterable[str], filename_glob: str, file_system=None
) -> set[str]:
    if file_system is None:
        file_system = dp.FileClient.get_gcs_file_system()

    # Using set to remove any duplicates if directories are overlapping
    out_paths: set[str] = set()
    for path in paths:
        pathname = os.path.join(path, "**", filename_glob)
        if is_dapla():
            out_paths |= {
                x for x in file_system.glob(pathname, recursive=True) if "." in x
            }
    return out_paths


class Sentinel2(GCSRasterDataset):
    """Works like torchgeo's Sentinel2, with custom regexes."""

    date_format: str = "%Y%m%d"
    filename_glob = "SENTINEL2X_*_*.*"

    filename_regex = SENTINEL2_FILENAME_REGEX

    all_bands = [
        # "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        # "B9",
        # "B10",
        "B11",
        "B12",
    ]
    rgb_bands = ["B4", "B3", "B2"]

    separate_files = True

    cmap: dict[int, tuple[int, int, int, int]] = {}

    plot = TorchgeoSentinel2.plot

import os
import warnings
from collections.abc import Callable
from collections.abc import Iterable
from typing import ClassVar

import rasterio
import rasterio.merge
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from torchgeo.datasets.geo import RasterDataset
from torchgeo.datasets.sentinel import Sentinel2 as TorchgeoSentinel2

try:
    import dapla as dp
except ImportError:
    pass

try:
    from dapla.gcs import GCSFileSystem
except ImportError:

    class GCSFileSystem:
        """Placeholder."""


try:
    from gcsfs.core import GCSFile
except ImportError:

    class GCSFile:
        """Placeholder."""


from ..helpers import get_all_files
from ..io._is_dapla import is_dapla
from ..io.opener import opener

SENTINEL2_FILENAME_REGEX = r"""
    ^SENTINEL2X_
    (?P<date>\d{8})
    .*T(?P<tile>\d{2}[A-Z]{3})
    .*(?:_(?P<resolution>{}m))?
    .*(?P<band>B\d{1,2}A|B\d{1,2})
    .*\..*$
"""


SENTINEL_2_BANDS = [
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
    # "B10",
    "B11",
    "B12",
]
SENTINEL_2_RBG_BANDS = ["B4", "B3", "B2"]
SENTINEL_2_NDVI_BANDS = ["B08", "B04"]


class GCSRasterDataset(RasterDataset):
    """Wrapper around torchgeo's RasterDataset that works in and outside of Dapla (stat norway)."""

    def __init__(self, *args, product_level: str | None = None, **kwargs) -> None:
        """Initialiser. Args and kwargs passed to torchgeo.datasets.geo.RasterDataset."""
        self.product_level = product_level
        super().__init__(*args, **kwargs)
        if is_dapla():
            [file.close() for file in self.files]

    @property
    def files(self) -> set[GCSFile] | set[str]:
        """A list of all files in the dataset.

        Returns:
            All files in the dataset.
        """
        if isinstance(self.paths, str):
            paths: list[str] = [self.paths]
        else:
            paths = self.paths

        if is_dapla():
            fs = dp.FileClient.get_gcs_file_system()
            files: set[GCSFile] = {
                fs.open(x)
                for x in _get_gcs_paths(
                    paths,
                    filename_glob=self.filename_glob,
                    file_system=fs,
                    product_level=self.product_level,
                )
            }
            return files

        # Using set to remove any duplicates if directories are overlapping
        files: set[str] = set()
        for path in paths:
            if os.path.isdir(path):
                files |= {x for x in get_all_files(path, recursive=True)}
                # pathname = os.path.join(path, "**", self.filename_glob)
                # files |= {
                #     x for x in glob.iglob(pathname, recursive=True) if os.path.isfile(x)
                # }
            elif os.path.isfile(path):
                files.add(path)
            else:
                warnings.warn(
                    f"Could not find any relevant files for provided path '{path}'. "
                    f"Path was ignored.",
                    UserWarning,
                    stacklevel=1,
                )

            if self.product_level:
                files = {path for path in files if self.product_level in path}

        return files

    def _load_warp_file(self, filepath: str) -> DatasetReader:
        """Load and warp a file to the correct CRS and resolution.

        Args:
            filepath: file to load and warp.

        Returns:
            file handle of warped VRT.
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
    paths: str | Iterable[str],
    filename_glob: str,
    product_level: str | None = None,
    file_system: GCSFileSystem | None = None,
) -> set[str]:
    if file_system is None:
        file_system = dp.FileClient.get_gcs_file_system()

    # Using set to remove any duplicates if directories are overlapping
    out_paths: set[str] = set()
    for path in paths:
        pathname = os.path.join(path, "**", filename_glob)
        out_paths |= {x for x in file_system.glob(pathname, recursive=True) if "." in x}
    if not product_level:
        return out_paths
    return {path for path in out_paths if product_level in path}


class Sentinel2(GCSRasterDataset):
    """Works like torchgeo's Sentinel2, with custom regexes."""

    date_format: ClassVar[str] = "%Y%m%d"
    filename_glob = "T*_*_{}*.*"

    filename_regex: ClassVar[str] = SENTINEL2_FILENAME_REGEX
    all_bands: ClassVar[list[str]] = SENTINEL_2_BANDS
    rgb_bands: ClassVar[list[str]] = SENTINEL_2_RBG_BANDS
    ndvi_bands: ClassVar[list[str]] = SENTINEL_2_RBG_BANDS

    separate_files: ClassVar[bool] = True

    cmap: ClassVar[dict[int, tuple[int, int, int, int]]] = {}

    plot: Callable = TorchgeoSentinel2.plot

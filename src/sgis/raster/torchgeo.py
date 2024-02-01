import glob
import os
import warnings
from typing import Iterable

import rasterio
import rasterio.merge
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from torchgeo.datasets.geo import RasterDataset


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


class DaplaRasterDataset(RasterDataset):
    """Custom version of torchgeo's class that works in and outside of Dapla (stat norway)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if is_dapla():
            [file.close() for file in self.files]

    def _get_gcs_paths(self, paths: str | Iterable[str], fs=None) -> set[str]:
        if fs is None:
            fs = dp.FileClient.get_gcs_file_system()

        # Using set to remove any duplicates if directories are overlapping
        out_paths: set[str] = set()
        for path in paths:
            pathname = os.path.join(path, "**", self.filename_glob)
            if is_dapla():
                out_paths |= {x for x in fs.glob(pathname, recursive=True) if "." in x}
        return out_paths

    @property
    def files(self) -> set[str] | set[GCSFile]:
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
            files = {fs.open(x) for x in self._get_gcs_paths(paths, fs)}
            return files

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


class Sentinel2(DaplaRasterDataset):
    """Works like torchgeo's Sentinel2, with custom regexes."""

    date_format: str = "%Y%m%d"
    filename_glob = "SENTINEL2X_*_*.*"

    filename_regex = SENTINEL2_FILENAME_REGEX

    _indexes = 1
    _nodata = 0

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

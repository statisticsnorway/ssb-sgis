import functools
import itertools
import multiprocessing
import re
import warnings
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from copy import copy
from copy import deepcopy
from pathlib import Path
from typing import Any
from typing import ClassVar

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from pandas import DataFrame
from pandas import Series
from pandas.api.types import is_dict_like
from pandas.api.types import is_list_like
from rasterio import merge as rasterio_merge

try:
    import xarray as xr
    from xarray import Dataset
except ImportError:

    class Dataset:
        """Placeholder."""


from rtree.index import Index
from rtree.index import Property
from shapely import Geometry
from typing_extensions import Self  # TODO: imperter fra typing når python 3.11

from ..geopandas_tools.bounds import make_grid
from ..geopandas_tools.conversion import is_bbox_like
from ..geopandas_tools.conversion import to_bbox
from ..geopandas_tools.conversion import to_shapely
from ..geopandas_tools.general import get_common_crs
from ..geopandas_tools.overlay import clean_overlay
from ..helpers import get_all_files
from ..helpers import get_numpy_func
from ..io._is_dapla import is_dapla
from ..io.opener import opener
from ..parallel.parallel import Parallel
from .raster import Raster

try:
    from torchgeo.datasets.geo import RasterDataset
    from torchgeo.datasets.utils import BoundingBox
except ImportError:

    class BoundingBox:
        """Placeholder."""

        def __init__(self, *args, **kwargs) -> None:
            """Placeholder."""
            raise ImportError("missing optional dependency 'torchgeo'")

    class RasterDataset:
        """Placeholder."""

        def __init__(self, *args, **kwargs) -> None:
            """Placeholder."""
            raise ImportError("missing optional dependency 'torchgeo'")


try:
    import torch
    from torchgeo.datasets.utils import disambiguate_timestamp
except ImportError:

    class torch:
        """Placeholder."""

        class Tensor:
            """Placeholder to reference torch.Tensor."""


try:
    from ..io.dapla_functions import read_geopandas
except ImportError:
    pass

try:
    from dapla import FileClient
    from dapla import write_pandas
except ImportError:
    pass

from .base import ALLOWED_KEYS
from .base import NESSECARY_META
from .base import get_index_mapper
from .cubebase import _from_gdf_func
from .cubebase import _method_as_func
from .cubebase import _raster_from_path
from .cubebase import _write_func
from .indices import get_raster_pairs
from .indices import index_calc_pair
from .zonal import _make_geometry_iterrows
from .zonal import _prepare_zonal
from .zonal import _zonal_func
from .zonal import _zonal_post

TORCHGEO_RETURN_TYPE = dict[str, torch.Tensor | pyproj.CRS | BoundingBox]


class DataCube:
    """Experimental."""

    CUBE_DF_NAME: ClassVar[str] = "cube_df.parquet"

    separate_files: ClassVar[bool] = True
    is_image: ClassVar[bool] = True
    date_format: ClassVar[str | None] = None

    def __init__(
        self,
        data: Iterable[Raster] | None = None,
        crs: Any | None = None,
        res: int | None = None,
        nodata: int | None = None,
        copy: bool = False,
        parallelizer: Parallel | None = None,
    ) -> None:
        """Initialize a DataCube instance with optional Raster data.

        Args:
            data: Iterable of Raster objects or a single DataCube to copy data from.
            crs: Coordinate reference system to be applied to the images.
            res: Spatial resolution of the images, applied uniformly to all Rasters.
            nodata: Nodata value to unify across all Rasters within the cube.
            copy: If True, makes deep copies of Rasters provided.
            parallelizer: sgis.Parallel instance to handle concurrent operations.
        """
        warnings.warn(
            "This class is deprecated in favor of ImageCollection", stacklevel=1
        )

        self._arrays = None
        self._res = res
        self.parallelizer = parallelizer

        # hasattr check to allow class attribute
        if not hasattr(self, "_nodata"):
            self._nodata = nodata

        if isinstance(data, DataCube):
            for key, value in data.__dict__.items():
                setattr(self, key, value)
            return
        elif data is None:
            self.data = data
            self._crs = None
            return
        elif not is_list_like(data) and all(isinstance(r, Raster) for r in data):
            raise TypeError(
                "'data' must be a Raster instance or an iterable."
                f"Got {type(data)}: {data}"
            )
        else:
            data = list(data)

        if copy:
            data = [raster.copy() for raster in data]
        else:
            # take a copy only if there are gdfs with the same memory address
            if sum(r1 is r2 for r1 in data for r2 in data) < len(data):
                data = [raster.copy() for raster in data]

        self.data = data

        nodatas = {r.nodata for r in self}
        if self.nodata is None and len(nodatas) > 1:
            raise ValueError(
                "Must specify 'nodata' when the images have different nodata values. "
                f"Got {', '.join([str(x) for x in nodatas])}"
            )

        resolutions = {r.res for r in self}
        if self._res is None and len(resolutions) > 1:
            raise ValueError(
                "Must specify 'res' when the images have different resolutions. "
                f"Got {', '.join([str(x) for x in resolutions])}"
            )
        elif res is None and len(resolutions):
            self._res = resolutions.pop()

        if crs:
            self._crs = pyproj.CRS(crs)
            if not all(self._crs.equals(pyproj.CRS(r.crs)) for r in self.data):
                self = self.to_crs(self._crs)
        try:
            self._crs = get_common_crs(self.data)
        except (ValueError, IndexError):
            self._crs = None

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        *,
        res: int | None = None,
        check_for_df: bool = True,
        contains: str | None = None,
        endswith: str = ".tif",
        bands: str | list[str] | None = None,
        filename_regex: str | None = None,
        parallelizer: Parallel | None = None,
        file_system=None,
        **kwargs,
    ) -> "DataCube":
        """Construct a DataCube by searching for files starting from a root directory.

        Args:
            root: Root directory path to search for raster image files.
            res: Resolution to unify the data within the cube.
            check_for_df: Check for a parquet file in the root directory
                that holds metadata for the files in the directory.
            contains: Filter files containing specific substrings.
            endswith: Filter files that end with specific substrings.
            bands: One or more band ids to keep.
            filename_regex: Regular expression to match file names
                and attributes (date, band, tile, resolution).
            parallelizer: sgis.Parallel instance for concurrent file processing.
            file_system: File system to use for file operations, used in GCS environment.
            **kwargs: Additional keyword arguments to pass to 'from_path' method.

        Returns:
            An instance of DataCube containing the raster data from specified paths.
        """
        kwargs["res"] = res
        kwargs["filename_regex"] = filename_regex
        kwargs["contains"] = contains
        kwargs["bands"] = bands
        kwargs["endswith"] = endswith

        if is_dapla():
            if file_system is None:
                file_system = FileClient.get_gcs_file_system()
            glob_pattern = str(Path(root) / "**")
            paths: list[str] = file_system.glob(glob_pattern)
            if contains:
                paths = [path for path in paths if contains in path]

        else:
            paths = get_all_files(root)

        dfs = [path for path in paths if path.endswith(cls.CUBE_DF_NAME)]

        if not check_for_df or not len(dfs):
            return cls.from_paths(
                paths,
                parallelizer=parallelizer,
                **kwargs,
            )

        folders_with_df: set[Path] = {Path(path).parent for path in dfs if path}

        cubes: list[DataCube] = [cls.from_cube_df(df, res=res) for df in dfs]

        paths_in_folders_without_df = [
            path for path in paths if Path(path).parent not in folders_with_df
        ]

        if paths_in_folders_without_df:
            cubes += [
                cls.from_paths(
                    paths_in_folders_without_df,
                    parallelizer=parallelizer,
                    **kwargs,
                )
            ]

        return concat_cubes(cubes, res=res)

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[str | Path],
        *,
        res: int | None = None,
        parallelizer: Parallel | None = None,
        file_system=None,
        contains: str | None = None,
        bands: str | list[str] | None = None,
        endswith: str = ".tif",
        filename_regex: str | None = None,
        **kwargs,
    ) -> "DataCube":
        """Create a DataCube from a list of file paths.

        Args:
            paths: Iterable of file paths to raster files.
            res: Resolution to unify the data within the cube.
            parallelizer: Joblib Parallel instance for concurrent file processing.
            file_system: File system to use for file operations, used in Dapla environment.
            contains: Filter files containing specific substrings.
            endswith: Filter files that end with specific substrings.
            bands: One or more band ids to keep.
            filename_regex: Regular expression to match file names.
            **kwargs: Additional keyword arguments to pass to the raster loading function.

        Returns:
            An instance of DataCube containing the raster data from specified paths.
        """
        crs = kwargs.pop("crs", None)

        if contains:
            paths = [path for path in paths if contains in path]
        if endswith:
            paths = [path for path in paths if path.endswith(endswith)]
        if filename_regex:
            compiled = re.compile(filename_regex, re.VERBOSE)
            paths = [path for path in paths if re.search(compiled, Path(path).name)]
        if bands:
            if isinstance(bands, str):
                bands = [bands]
            paths = [path for path in paths if any(band in str(path) for band in bands)]

        if not paths:
            return cls(crs=crs, parallelizer=parallelizer, res=res)

        kwargs["res"] = res
        kwargs["filename_regex"] = filename_regex

        if file_system is None and is_dapla():
            kwargs["file_system"] = FileClient.get_gcs_file_system()

        if parallelizer is None:
            rasters: list[Raster] = [
                _raster_from_path(path, **kwargs) for path in paths
            ]
        else:
            rasters: list[Raster] = parallelizer.map(
                _raster_from_path,
                paths,
                kwargs=kwargs,
            )

        return cls(rasters, copy=False, crs=crs, res=res)

    @classmethod
    def from_gdf(
        cls,
        gdf: GeoDataFrame | Iterable[GeoDataFrame],
        columns: str | Iterable[str],
        res: int,
        parallelizer: Parallel | None = None,
        tile_size: int | None = None,
        grid: GeoSeries | None = None,
        **kwargs,
    ) -> "DataCube":
        """Create a DataCube from a GeoDataFrame or a set of them, tiling the spatial data as specified.

        Args:
            gdf: GeoDataFrame or an iterable of GeoDataFrames to rasterize.
            columns: The column(s) in the GeoDataFrame that will be used as values for the rasterization.
            res: Spatial resolution of the output rasters.
            parallelizer: Joblib Parallel instance for concurrent processing.
            tile_size: Size of each tile/grid cell in the output raster.
            grid: Predefined grid to align the rasterization.
            **kwargs: Additional keyword arguments passed to Raster.from_gdf.

        Returns:
            An instance of DataCube containing rasterized data from the GeoDataFrame(s).
        """
        if grid is None and tile_size is None:
            raise ValueError("Must specify either 'tile_size' or 'grid'.")

        if isinstance(gdf, GeoDataFrame):
            gdf = [gdf]
        elif not all(isinstance(frame, GeoDataFrame) for frame in gdf):
            raise TypeError("gdf must be one or more GeoDataFrames.")

        if grid is None:
            crs = get_common_crs(gdf)
            total_bounds = shapely.union_all(
                [shapely.box(*frame.total_bounds) for frame in gdf]
            )
            grid = make_grid(total_bounds, gridsize=tile_size, crs=crs)

        grid["tile_idx"] = range(len(grid))

        partial_func = functools.partial(
            _from_gdf_func,
            columns=columns,
            res=res,
            **kwargs,
        )

        def to_gdf_list(gdf: GeoDataFrame) -> list[GeoDataFrame]:
            return [gdf.loc[gdf["tile_idx"] == i] for i in gdf["tile_idx"].unique()]

        rasters = []

        if parallelizer.processes > 1:
            rasters = parallelizer.map(
                clean_overlay, gdf, args=(grid,), kwargs=dict(keep_geom_type=True)
            )
            with multiprocessing.get_context("spawn").Pool(parallelizer.processes) as p:
                for frame in gdf:
                    frame = frame.overlay(grid, keep_geom_type=True)
                    gdfs = to_gdf_list(frame)
                    rasters += p.map(partial_func, gdfs)
        elif parallelizer.processes < 1:
            raise ValueError("processes must be an integer 1 or greater.")
        else:
            for frame in gdf:
                frame = frame.overlay(grid, keep_geom_type=True)
                gdfs = to_gdf_list(frame)
                rasters += [partial_func(gdf) for gdf in gdfs]

        return cls(rasters, res=res)

    @classmethod
    def from_cube_df(
        cls, df: DataFrame | str | Path, res: int | None = None
    ) -> "DataCube":
        """Construct a DataCube from a DataFrame or path containing metadata or paths of rasters.

        Args:
            df: DataFrame, path to a DataFrame, or string path pointing to cube data.
            res: Optional resolution to standardize all rasters to this resolution.

        Returns:
            A DataCube instance containing the raster data described by the DataFrame.
        """
        if isinstance(df, (str, Path)):
            df = read_geopandas(df) if is_dapla() else gpd.read_parquet(df)

        # recursive
        if not is_dict_like(df) and all(
            isinstance(x, (str, Path, DataFrame)) for x in df
        ):
            cubes = [cls.from_cube_df(x) for x in df]
            cube = concat_cubes(cubes, res=res)
            return cube

        if isinstance(df, dict):
            df = DataFrame(df)
        elif not isinstance(df, DataFrame):
            raise TypeError("df must be DataFrame or file path to a parquet file.")

        rasters: list[Raster] = [
            Raster.from_dict(meta) for _, meta in (df[NESSECARY_META].iterrows())
        ]
        return cls(rasters)

    def to_gdf(
        self, column: str | None = None, ignore_index: bool = False, concat: bool = True
    ) -> GeoDataFrame:
        """Convert DataCube to GeoDataFrame."""
        gdfs = self.run_raster_method("to_gdf", column=column, return_self=False)

        if concat:
            return pd.concat(gdfs, ignore_index=ignore_index)
        return gdfs

    def to_xarray(self) -> Dataset:
        """Convert DataCube to an xarray.Dataset."""
        return xr.Dataset({i: r.to_xarray() for i, r in enumerate(self.data)})

    def zonal(
        self,
        polygons: GeoDataFrame,
        aggfunc: str | Callable | list[Callable | str],
        array_func: Callable | None = None,
        by_date: bool | None = None,
        dropna: bool = True,
    ) -> GeoDataFrame:
        """Calculate zonal statistics within polygons."""
        idx_mapper, idx_name = get_index_mapper(polygons)
        polygons, aggfunc, func_names = _prepare_zonal(polygons, aggfunc)
        poly_iter = _make_geometry_iterrows(polygons)

        if by_date is None:
            by_date: bool = all(r.date is not None for r in self)

        if not self.parallelizer:
            aggregated: list[DataFrame] = [
                _zonal_func(
                    poly,
                    cube=self,
                    array_func=array_func,
                    aggfunc=aggfunc,
                    func_names=func_names,
                    by_date=by_date,
                )
                for poly in poly_iter
            ]
        else:
            aggregated: list[DataFrame] = self.parallelizer.map(
                _zonal_func,
                poly_iter,
                kwargs=dict(
                    cube=self,
                    array_func=array_func,
                    aggfunc=aggfunc,
                    func_names=func_names,
                    by_date=by_date,
                ),
            )

        return _zonal_post(
            aggregated,
            polygons=polygons,
            idx_mapper=idx_mapper,
            idx_name=idx_name,
            dropna=dropna,
        )

    def gradient(self, degrees: bool = False) -> Self:
        """Get gradients in each image."""
        self.data = self.run_raster_method("gradient", degrees=degrees)
        return self

    def map(self, func: Callable, return_self: bool = True, **kwargs) -> Self:
        """Maps each raster array to a function.

        The function must take a numpy array as first positional argument,
        and return a single numpy array. The function should be defined in
        the leftmost indentation level. If in Jupyter, the function also
        have to be defined in and imported from another file.
        """
        self._check_for_array()
        if self.parallelizer:
            data = self.parallelizer.map(func, self.arrays, kwargs=kwargs)
        else:
            data = [func(arr, **kwargs) for arr in self.arrays]
        if not return_self:
            return data
        self.arrays = data
        return self

    def raster_map(self, func: Callable, return_self: bool = True, **kwargs) -> Self:
        """Maps each raster to a function.

        The function must take a Raster object as first positional argument,
        and return a single Raster object. The function should be defined in
        the leftmost indentation level. If in Jupyter, the function also
        have to be defined in and imported from another file.
        """
        if self.parallelizer:
            data = self.parallelizer.map(func, self, kwargs=kwargs)
        else:
            data = [func(r, **kwargs) for r in self]
        if not return_self:
            return data
        self.data = data
        return self

    def sample(self, n: int, copy: bool = True, **kwargs) -> Self:
        """Take n samples of the cube."""
        if self.crs is None:
            self._crs = get_common_crs(self.data)

        cube = self.copy() if copy else self

        cube.data = list(pd.Series(cube.data).sample(n))

        cube.data = cube.run_raster_method("load", **kwargs)

        return cube

    def load(self, copy: bool = True, **kwargs) -> Self:
        """Load all images as arrays into a DataCube copy."""
        if self.crs is None:
            self._crs = get_common_crs(self.data)

        cube = self.copy() if copy else self

        cube.data = cube.run_raster_method("load", **kwargs)

        return cube

    def intersection(self, other: Any, copy: bool = True) -> Self:
        """Select the images that intersect 'other'."""
        other = to_shapely(other)
        cube = self.copy() if copy else self
        cube = cube[cube.boxes.intersects(other)]
        return cube

    def sfilter(
        self, other: GeoDataFrame | GeoSeries | Geometry | tuple, copy: bool = True
    ) -> Self:
        """Spatially filter images by bounding box or geometry object."""
        other = to_shapely(other)
        cube = self.copy() if copy else self
        cube.data = [raster for raster in self if raster.union_all().intersects(other)]
        return cube

    def clip(
        self, mask: GeoDataFrame | GeoSeries | Geometry, copy: bool = True, **kwargs
    ) -> Self:
        """Clip the images by bounding box or geometry object."""
        if self.crs is None:
            self._crs = get_common_crs(self.data)

        if (
            hasattr(mask, "crs")
            and mask.crs
            and not pyproj.CRS(self.crs).equals(pyproj.CRS(mask.crs))
        ):
            raise ValueError("crs mismatch.")

        cube = self.copy() if copy else self

        cube = cube.sfilter(to_shapely(mask), copy=False)

        cube.data = cube.run_raster_method("clip", mask=mask, **kwargs)
        return cube

    def clipmerge(self, mask: GeoDataFrame | GeoSeries | Geometry, **kwargs) -> Self:
        """Clip the images and merge to one image."""
        return _clipmerge(self, mask, **kwargs)

    def merge_by_bounds(self, by: str | list[str] | None = None, **kwargs) -> Self:
        """Merge images with the same bounding box."""
        return _merge_by_bounds(self, by=by, **kwargs)

    def merge(self, by: str | list[str] | None = None, **kwargs) -> Self:
        """Merge all images to one."""
        return _merge(self, by=by, **kwargs)

    def explode(self) -> Self:
        """Convert from 3D to 2D arrays.

        Make multi-banded arrays (3d) into multiple single-banded arrays (2d).
        """

        def explode_one_raster(raster: Raster) -> list[Raster]:
            property_values = {key: getattr(raster, key) for key in raster.properties}

            all_meta = {
                key: value
                for key, value in (
                    raster.__dict__ | raster.meta | property_values
                ).items()
                if key in ALLOWED_KEYS and key not in ["array", "indexes"]
            }
            if raster.values is None:
                return [
                    raster.__class__.from_dict({"indexes": i} | all_meta)
                    for i in raster.indexes_as_tuple()
                ]
            else:
                return [
                    raster.__class__.from_dict(
                        {"array": array, "indexes": i + 1} | all_meta
                    )
                    for i, array in enumerate(raster.array_list())
                ]

        self.data = list(
            itertools.chain.from_iterable(
                [explode_one_raster(raster) for raster in self]
            )
        )
        return self

    def dissolve_bands(self, aggfunc: Callable | str, copy: bool = True) -> Self:
        """Aggregate values in 3D arrays to a single value in a 2D array."""
        self._check_for_array()
        if not callable(aggfunc) and not isinstance(aggfunc, str):
            raise TypeError("Can only supply a single aggfunc")

        cube = self.copy() if copy else self

        aggfunc = get_numpy_func(aggfunc)

        cube = cube.map(aggfunc, axis=0)
        return cube

    def write(
        self,
        root: str,
        file_format: str = "tif",
        **kwargs,
    ) -> None:
        """Writes arrays as tif files and df with file info.

        This method should be run after the rasters have been clipped, merged or
        its array values have been recalculated.

        Args:
            root: Directory path where the images will be written to.
            file_format: File extension.
            **kwargs: Keyword arguments passed to rasterio.open.

        """
        self._check_for_array()

        if any(raster.name is None for raster in self):
            raise ValueError("")

        paths = [
            (Path(root) / raster.name).with_suffix(f".{file_format}") for raster in self
        ]

        if self.parallelizer:
            self.parallelizer.starmap(
                _write_func, zip(self, paths, strict=False), kwargs=kwargs
            )
        else:
            [
                _write_func(raster, path, **kwargs)
                for raster, path in zip(self, paths, strict=False)
            ]

    def write_df(self, folder: str) -> None:
        """Write metadata DataFrame."""
        df = pd.DataFrame(self.meta)

        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError()

        if is_dapla():
            write_pandas(df, folder / self.CUBE_DF_NAME)
        else:
            df.to_parquet(folder / self.CUBE_DF_NAME)

    def calculate_index(
        self,
        index_func: Callable,
        band_name1: str,
        band_name2: str,
        copy: bool = True,
        **kwargs,
    ) -> Self:
        """Calculate an index based on a function."""
        cube = self.copy() if copy else self

        raster_pairs: list[tuple[Raster, Raster]] = get_raster_pairs(
            cube, band_name1=band_name1, band_name2=band_name2
        )

        kwargs = dict(index_formula=index_func) | kwargs

        if self.parallelizer:
            rasters = self.parallelizer.map(
                index_calc_pair, raster_pairs, kwargs=kwargs
            )
        else:
            rasters = [index_calc_pair(items, **kwargs) for items in raster_pairs]

        return cube.__class__(rasters)

    # def reproject_match(self) -> Self:
    #     pass

    def to_crs(self, crs: Any, copy: bool = True) -> Self:
        """Reproject the coordinates of each image."""
        cube = self.copy() if copy else self
        cube.data = [r.to_crs(crs) for r in cube]
        cube._warped_crs = crs
        return cube

    def set_crs(
        self, crs: Any, allow_override: bool = False, copy: bool = True
    ) -> Self:
        """Set the CRS of each image."""
        cube = self.copy() if copy else self
        cube.data = [r.set_crs(crs, allow_override=allow_override) for r in cube]
        cube._warped_crs = crs
        return cube

    def min(self) -> Series:
        """Get minimum array values for each image."""
        return Series(
            self.run_raster_method("min"),
            name="min",
        )

    def max(self) -> Series:
        """Get maximum array values for each image."""
        return Series(
            self.run_raster_method("max"),
            name="max",
        )

    def raster_attribute(self, attribute: str) -> Series | GeoSeries:
        """Get a Raster attribute returned as values in a pandas.Series."""
        data = [getattr(r, attribute) for r in self]
        if any(isinstance(x, Geometry) for x in data):
            return GeoSeries(data, name=attribute)
        return Series(data, name=attribute)

    def run_raster_method(
        self, method: str, *args, copy: bool = True, return_self: bool = False, **kwargs
    ) -> Self:
        """Run a Raster method for each raster in the cube."""
        if not all(hasattr(r, method) for r in self):
            raise AttributeError(f"Raster has no method {method!r}.")

        method_as_func = functools.partial(
            _method_as_func, *args, method=method, **kwargs
        )

        cube = self.copy() if copy else self

        return cube.raster_map(method_as_func, return_self=return_self)

    @property
    def meta(self) -> list[dict]:
        """Get metadata property of each raster."""
        return [raster.meta for raster in self]

    # @property
    # def cube_df_meta(self) -> dict[list]:
    #     return {
    #         "path": [r.path for r in self],
    #         "indexes": [r.indexes for r in self],
    #         "type": [r.__class__.__name__ for r in self],
    #         "bounds": [r.bounds for r in self],
    #         "crs": [crs_to_string(r.crs) for r in self],
    #     }

    @property
    def data(self) -> list[Raster]:
        """The Rasters as a list."""
        return self._data

    @data.setter
    def data(self, data: list[Raster]):
        self.index = Index(interleaved=False, properties=Property(dimension=3))

        if data is None or not len(data):
            self._data = []
            return
        if not all(isinstance(x, Raster) for x in data):
            types = {type(x).__name__ for x in data}
            raise TypeError(f"data must be Raster. Got {', '.join(types)}")
        self._data = list(data)

        for i, raster in enumerate(self._data):
            if raster.date:
                try:
                    mint, maxt = disambiguate_timestamp(raster.date, self.date_format)
                except (NameError, TypeError):
                    mint, maxt = 0, 1
            else:
                mint, maxt = 0, 1
            # important: torchgeo has a different order of the bbox than shapely and geopandas
            minx, miny, maxx, maxy = raster.bounds
            self.index.insert(i, (minx, maxx, miny, maxy, mint, maxt))

    @property
    def arrays(self) -> list[np.ndarray]:
        """The arrays of the images as a list."""
        return [raster.values for raster in self]

    @arrays.setter
    def arrays(self, new_arrays: list[np.ndarray]):
        if len(new_arrays) != len(self):
            raise ValueError(
                f"Number of arrays ({len(new_arrays)}) must be same as length as cube ({len(self)})."
            )
        if not all(isinstance(arr, np.ndarray) for arr in new_arrays):
            raise ValueError("Must be list of numpy ndarrays")

        self.data = [
            raster.update(array=arr)
            for raster, arr in zip(self, new_arrays, strict=False)
        ]

    @property
    def band(self) -> Series:
        """Get the 'band' attribute of the rasters."""
        return Series(
            [r.band for r in self],
            name="band",
        )

    @property
    def dtype(self) -> Series:
        """Get the 'dtype' attribute of the rasters."""
        return Series(
            [r.dtype for r in self],
            name="dtype",
        )

    @property
    def nodata(self) -> int | None:
        """No data value."""
        return self._nodata

    @property
    def path(self) -> Series:
        """Get the 'path' attribute of the rasters."""
        return self.raster_attribute("path")

    @property
    def name(self) -> Series:
        """Get the 'name' attribute of the rasters."""
        return self.raster_attribute("name")

    @property
    def date(self) -> Series:
        """Get the 'date' attribute of the rasters."""
        return self.raster_attribute("date")

    @property
    def indexes(self) -> Series:
        """Get the 'indexes' attribute of the rasters."""
        return self.raster_attribute("indexes")

    # @property
    # def raster_id(self) -> Series:
    #     return self.raster_attribute("raster_id")

    @property
    def area(self) -> Series:
        """Get the 'area' attribute of the rasters."""
        return self.raster_attribute("area")

    @property
    def length(self) -> Series:
        """Get the 'length' attribute of the rasters."""
        return self.raster_attribute("length")

    @property
    def height(self) -> Series:
        """Get the 'height' attribute of the rasters."""
        return self.raster_attribute("height")

    @property
    def width(self) -> Series:
        """Get the 'width' attribute of the rasters."""
        return self.raster_attribute("width")

    @property
    def shape(self) -> Series:
        """Get the 'shape' attribute of the rasters."""
        return self.raster_attribute("shape")

    @property
    def count(self) -> Series:
        """Get the 'count' attribute of the rasters."""
        return self.raster_attribute("count")

    @property
    def res(self) -> int:
        """Spatial resolution of the images."""
        return self._res

    @res.setter
    def res(self, value) -> None:
        self._res = value

    @property
    def crs(self) -> pyproj.CRS:
        """Coordinate reference system of the images."""
        crs = self._warped_crs if hasattr(self, "_warped_crs") else self._crs
        if crs is not None:
            return crs
        try:
            get_common_crs(self.data)
        except ValueError:
            return None

    @property
    def unary_union(self) -> Geometry:
        """Box polygon of the combined bounds of each image."""
        return shapely.union_all([shapely.box(*r.bounds) for r in self])

    @property
    def centroid(self) -> GeoSeries:
        """Get the 'centroid' attribute of the rasters."""
        return GeoSeries(
            [r.centroid for r in self],
            name="centroid",
            crs=self.crs,
        )

    @property
    def tile(self) -> Series:
        """Get the 'tile' attribute of the rasters."""
        return self.raster_attribute("tile")

    @property
    def boxes(self) -> GeoSeries:
        """Get the 'bounds' attribute of the rasters."""
        return GeoSeries(
            [shapely.box(*r.bounds) if r.bounds is not None else None for r in self],
            name="boxes",
            crs=self.crs,
        )

    @property
    def total_bounds(self) -> tuple[float, float, float, float]:
        """Combined minimum and maximum longitude and latitude."""
        return tuple(x for x in self.boxes.total_bounds)

    @property
    def bounds(self) -> BoundingBox:
        """Pytorch bounds of the index.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) of the dataset
        """
        return BoundingBox(*self.index.bounds)

    def copy(self, deep: bool = True) -> Self:
        """Returns a (deep) copy of the class instance and its rasters.

        Args:
            deep: Whether to return a deep or shallow copy. Defaults to True.
        """
        copied = deepcopy(self) if deep else copy(self)
        copied.data = [raster.copy() for raster in copied]
        return copied

    def _check_for_array(self, text: str = "") -> None:
        mess = "Arrays are not loaded. " + text
        if all(raster.values is None for raster in self):
            raise ValueError(mess)

    def __getitem__(
        self,
        item: (
            str
            | slice
            | int
            | Series
            | list
            | tuple
            | Callable
            | Geometry
            | BoundingBox
        ),
    ) -> Self | Raster | TORCHGEO_RETURN_TYPE:
        """Select one or more of the Rasters based on indexing or spatial or boolean predicates.

        Examples:
        ------------
        >>> import sgis as sg
        >>> root = 'https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/raster'
        >>> cube = sg.DataCube.from_root(root, filename_regex=sg.raster.SENTINEL2_FILENAME_REGEX, crs=25833).load()

        List slicing:

        >>> cube[1:3]
        >>> cube[3:]

        Single integer returns a Raster, not a cube.

        >>> cube[1]

        Boolean conditioning based on cube properties and pandas boolean Series:

        >>> cube[(cube.length > 0) & (cube.path.str.contains("FRC_B"))]
        >>> cube[lambda x: (x.length > 0) & (x.path.str.contains("dtm"))]

        """
        copy = self.copy()
        if isinstance(item, str) and copy.path is not None:
            copy.data = [raster for raster in copy if item in raster.path]
            if len(copy) == 1:
                return copy[0]
            elif not len(copy):
                return Raster()
            return copy

        if isinstance(item, slice):
            copy.data = copy.data[item]
            return copy
        elif isinstance(item, int):
            return copy.data[item]
        elif callable(item):
            item = item(copy)
        elif isinstance(item, BoundingBox):
            return cube_to_torchgeo(self, item)

        elif isinstance(item, (GeoDataFrame, GeoSeries, Geometry)) or is_bbox_like(
            item
        ):
            item = to_shapely(item)
            copy.data = [
                raster for raster in copy.data if raster.bounds.intersects(item)
            ]
            return copy

        copy.data = [
            raster
            for raster, condition in zip(copy.data, item, strict=True)
            if condition
        ]

        return copy

    def __setattr__(self, attr: str, value: Any) -> None:
        """Set an attribute of the cube."""
        if (
            attr in ["data", "_data"]
            or not is_list_like(value)
            or not hasattr(self, "data")
        ):
            return super().__setattr__(attr, value)
        if len(value) != len(self.data):
            raise ValueError(
                "custom cube attributes must be scalar or same length as number of rasters. "
                f"Got self.data {len(self)} and new attribute {len(value)}"
            )
        return super().__setattr__(attr, value)

    def __iter__(self) -> Iterator[Raster]:
        """Iterate over the Rasters."""
        return iter(self.data)

    def __len__(self) -> int:
        """Number of Rasters."""
        return len(self.data)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({len(self)})"

    # def __mul__(self, scalar) -> Self:
    #     return self.map(_mul, scalar=scalar)

    # def __add__(self, scalar) -> Self:
    #     return self.map(_add, scalar=scalar)

    # def __sub__(self, scalar) -> Self:
    #     return self.map(_sub, scalar=scalar)

    # def __truediv__(self, scalar) -> Self:
    #     return self.map(_truediv, scalar=scalar)

    # def __floordiv__(self, scalar) -> Self:
    #     return self.map(_floordiv, scalar=scalar)

    # def __pow__(self, scalar) -> Self:
    #     return self.map(_pow, scalar=scalar)


def concat_cubes(cubes: list[DataCube], res: int | None = None) -> DataCube:
    """Concatenate cubes to one.

    Args:
        cubes: A sequence of DataCubes.
        res: Spatial resolution.

    Returns:
        The cubes combined to one.
    """
    if not all(isinstance(cube, DataCube) for cube in cubes):
        raise TypeError("cubes must be of type DataCube.")

    return DataCube(
        list(itertools.chain.from_iterable([cube.data for cube in cubes])), res=res
    )


def _clipmerge(cube: DataCube, mask: Any, **kwargs) -> DataCube:
    return _merge(cube, bounds=mask, **kwargs)


def _merge(
    cube: DataCube,
    by: str | list[str] | None = None,
    bounds: Any | None = None,
    **kwargs,
) -> DataCube:
    if not all(r.values is None for r in cube):
        raise ValueError("Arrays can't be loaded when calling merge.")

    bounds = to_bbox(bounds) if bounds is not None else bounds

    if by is None:
        return _merge(
            cube,
            bounds=bounds,
            **kwargs,
        )

    elif isinstance(by, str):
        by = [by]
    elif not is_list_like(by):
        raise TypeError("'by' should be string or list like.", by)

    df = DataFrame(
        {"i": range(len(cube)), "tile": cube.tile} | {x: getattr(cube, x) for x in by}
    )

    grouped_indices = df.groupby(by)["i"].unique()
    indices = Series(range(len(cube)))

    return concat_cubes(
        [
            _merge(
                cube[indices.isin(idxs)],
                bounds=bounds,
            )
            for idxs in grouped_indices
        ],
        res=cube.res,
    )


def _merge_by_bounds(
    cube: DataCube,
    by: str | list[str] | None = None,
    bounds: Any | None = None,
    **kwargs,
) -> DataCube:
    if isinstance(by, str):
        by = [by, "tile"]
    elif by is None:
        by = ["tile"]
    else:
        by = list(by) + ["tile"]

    return _merge(
        cube,
        by=by,
        bounds=bounds,
        **kwargs,
    )


def _merge(cube: DataCube, **kwargs) -> DataCube:
    by = kwargs.pop("by")
    if cube.crs is None:
        cube._crs = get_common_crs(cube.data)

    indexes = cube[0].indexes_as_tuple()

    datasets = [_load_raster(raster.path) for raster in cube]
    array, transform = rasterio_merge.merge(datasets, indexes=indexes, **kwargs)
    cube.data = [Raster.from_array(array, crs=cube.crs, transform=transform)]

    return cube


def _load_raster(path: str | Path) -> rasterio.io.DatasetReader:
    with opener(path) as file:
        return rasterio.open(file)


def numpy_to_torch(array: np.ndarray) -> torch.Tensor:
    """Convert numpy array to a pytorch tensor."""
    # fix numpy dtypes which are not supported by pytorch tensors
    if array.dtype == np.uint16:
        array = array.astype(np.int32)
    elif array.dtype == np.uint32:
        array = array.astype(np.int64)

    return torch.tensor(array)


def cube_to_torchgeo(cube: DataCube, query: BoundingBox) -> TORCHGEO_RETURN_TYPE:
    """Convert a DayaCube to the type of dict returned from torchgeo datasets __getitem__."""
    bbox = shapely.box(*to_bbox(query))
    if cube.separate_files:
        cube = cube.sfilter(bbox).explode().load()
    else:
        cube = cube.clipmerge(bbox).explode()

    data: torch.Tensor = torch.cat([numpy_to_torch(array) for array in cube.arrays])

    key = "image" if cube.is_image else "mask"
    sample = {key: data, "crs": cube.crs, "bbox": query}
    return sample

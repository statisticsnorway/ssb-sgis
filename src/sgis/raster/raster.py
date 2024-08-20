import functools
import numbers
import os
import re
import warnings
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from copy import copy
from copy import deepcopy
from json import loads
from pathlib import Path
from typing import Any
from typing import ClassVar

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
import rasterio.windows
import shapely
from typing_extensions import Self  # TODO: imperter fra typing når python 3.11

try:
    import xarray as xr
    from xarray import DataArray
except ImportError:

    class DataArray:
        """Placeholder."""


try:
    from dapla.gcs import GCSFileSystem
except ImportError:

    class GCSFileSystem:
        """Placeholder."""


try:
    from rioxarray.rioxarray import _generate_spatial_coords
except ImportError:
    pass
from affine import Affine
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from pandas.api.types import is_list_like
from rasterio import features
from rasterio.enums import MergeAlg
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject
from shapely import Geometry
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import shape

from ..geopandas_tools.conversion import to_bbox
from ..geopandas_tools.conversion import to_gdf
from ..geopandas_tools.conversion import to_shapely
from ..geopandas_tools.general import is_bbox_like
from ..geopandas_tools.general import is_wkt
from ..helpers import is_property
from ..io.opener import opener
from .base import ALLOWED_KEYS
from .base import NESSECARY_META
from .base import get_index_mapper
from .base import memfile_from_array
from .zonal import _aggregate
from .zonal import _make_geometry_iterrows
from .zonal import _no_overlap_df
from .zonal import _prepare_zonal
from .zonal import _zonal_post

numpy_func_message = (
    "aggfunc must be functions or strings of numpy functions or methods."
)


class Raster:
    """For reading, writing and working with rasters.

    Raster instances should be created with the methods 'from_path', 'from_array' or
    'from_gdf'.


    Examples:
    ---------
    Read tif file.

    >>> import sgis as sg
    >>> path = 'https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/raster/dtm_10.tif'
    >>> raster = sg.Raster.from_path(path)
    >>> raster
    Raster(shape=(1, 201, 201), res=10, crs=ETRS89 / UTM zone 33N (N-E), path=https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/raster/dtm_10.tif)

    Load the entire image as an numpy ndarray.
    Operations are done in place to save memory.
    The array is stored in the array attribute.

    >>> raster.load()
    >>> raster.values[raster.values < 0] = 0
    >>> raster.values
    [[[  0.    0.    0.  ... 158.4 155.6 152.6]
    [  0.    0.    0.  ... 158.  154.8 151.9]
    [  0.    0.    0.  ... 158.5 155.1 152.3]
    ...
    [  0.  150.2 150.6 ...   0.    0.    0. ]
    [  0.  149.9 150.1 ...   0.    0.    0. ]
    [  0.  149.2 149.5 ...   0.    0.    0. ]]]

    Save as tif file.

    >>> raster.write("path/to/file.tif")

    Convert to GeoDataFrame.

    >>> gdf = raster.to_gdf(column="elevation")
    >>> gdf
           elevation                                           geometry  indexes
    0            1.9  POLYGON ((-25665.000 6676005.000, -25665.000 6...           1
    1           11.0  POLYGON ((-25655.000 6676005.000, -25655.000 6...           1
    2           18.1  POLYGON ((-25645.000 6676005.000, -25645.000 6...           1
    3           15.8  POLYGON ((-25635.000 6676005.000, -25635.000 6...           1
    4           11.6  POLYGON ((-25625.000 6676005.000, -25625.000 6...           1
    ...          ...                                                ...         ...
    25096       13.4  POLYGON ((-24935.000 6674005.000, -24935.000 6...           1
    25097        9.4  POLYGON ((-24925.000 6674005.000, -24925.000 6...           1
    25098        5.3  POLYGON ((-24915.000 6674005.000, -24915.000 6...           1
    25099        2.3  POLYGON ((-24905.000 6674005.000, -24905.000 6...           1
    25100        0.1  POLYGON ((-24895.000 6674005.000, -24895.000 6...           1

    The image can also be clipped by a mask while loading.

    >>> small_circle = raster_as_polygons.union_all().centroid.buffer(50)
    >>> raster = sg.Raster.from_path(path).clip(small_circle)
    Raster(shape=(1, 10, 10), res=10, crs=ETRS89 / UTM zone 33N (N-E), path=https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/raster/dtm_10.tif)

    Construct raster from GeoDataFrame.
    The arrays are put on top of each other in a 3 dimensional array.

    >>> raster_as_polygons["elevation_x2"] = raster_as_polygons["elevation"] * 2
    >>> raster_from_polygons = sg.Raster.from_gdf(raster_as_polygons, columns=["elevation", "elevation_x2"], res=20)
    >>> raster_from_polygons
    Raster(shape=(2, 100, 100), res=20, raster_id=-260056673995, crs=ETRS89 / UTM zone 33N (N-E), path=None)

    Calculate zonal statistics for each polygon in 'gdf'.

    >>> zonal = raster.zonal(raster_as_polygons, aggfunc=["sum", np.mean])
    >>> zonal
            sum  mean                                           geometry
    0       1.9   1.9  POLYGON ((-25665.000 6676005.000, -25665.000 6...
    1      11.0  11.0  POLYGON ((-25655.000 6676005.000, -25655.000 6...
    2      18.1  18.1  POLYGON ((-25645.000 6676005.000, -25645.000 6...
    3      15.8  15.8  POLYGON ((-25635.000 6676005.000, -25635.000 6...
    4      11.6  11.6  POLYGON ((-25625.000 6676005.000, -25625.000 6...
    ...     ...   ...                                                ...
    25096  13.4  13.4  POLYGON ((-24935.000 6674005.000, -24935.000 6...
    25097   9.4   9.4  POLYGON ((-24925.000 6674005.000, -24925.000 6...
    25098   5.3   5.3  POLYGON ((-24915.000 6674005.000, -24915.000 6...
    25099   2.3   2.3  POLYGON ((-24905.000 6674005.000, -24905.000 6...
    25100   0.1   0.1  POLYGON ((-24895.000 6674005.000, -24895.000 6...

    """

    # attributes concerning rasterio metadata
    _profile: ClassVar[dict[str, str | None]] = {
        "driver": "GTiff",
        "compress": "LZW",
        "nodata": None,
        "dtype": None,
        "crs": None,
        "tiled": None,
        "indexes": None,
    }

    def __init__(
        self,
        data: Self | str | np.ndarray | None = None,
        *,
        file_system: GCSFileSystem | None = None,
        filename_regex: str | None = None,
        **kwargs,
    ) -> None:
        """Note: use the classmethods from_path, from_array, from_gdf etc. instead of the initialiser.

        Args:
            data: A file path, an array or a Raster object.
            file_system: Optional GCSFileSystem.
            filename_regex: Regular expression to match file name attributes (date, band, tile, resolution).
            **kwargs: Arguments concerning file metadata or
                spatial properties of the image.
        """
        warnings.warn("This class is deprecated in favor of Band", stacklevel=1)
        self.filename_regex = filename_regex
        if filename_regex:
            self.filename_pattern = re.compile(self.filename_regex, re.VERBOSE)
        else:
            self.filename_pattern = None

        if isinstance(data, Raster):
            for key, value in data.__dict__.items():
                setattr(data, key, value)
            return

        if isinstance(data, (str | Path | os.PathLike)):
            self.path = data

        else:
            self.path = None

        if isinstance(data, (np.ndarray)):
            self.values = data
        else:
            self.values = None

        if self.path is None and not any(
            [kwargs.get("transform"), kwargs.get("bounds")]
        ):
            raise TypeError(
                "Must specify either bounds or transform when constructing raster from array."
            )

        # add class profile first, then override with args and kwargs
        self.update(**self._profile)

        self._crs = kwargs.pop("crs", self._crs if hasattr(self, "_crs") else None)
        self._bounds = None
        self.file_system = file_system
        self._indexes = self._get_indexes(kwargs.pop("indexes", self.indexes))

        # override the above with kwargs
        self.update(**kwargs)

        attributes = set(self.__dict__.keys()).difference(set(self.properties))

        if self.path is not None and not self._has_nessecary_attrs(attributes):
            self._add_meta()
            self._meta_added = True

        self._prev_crs = self._crs

    @classmethod
    def from_path(
        cls,
        path: str,
        res: int | None = None,
        file_system: GCSFileSystem | None = None,
        filename_regex: str | None = None,
        **kwargs,
    ) -> Self:
        """Construct Raster from file path.

        Args:
            path: Path to a raster image file.
            res: Spatial resolution when reading the image.
            file_system: Optional file system.
            filename_regex: Regular expression with optional match groups.
            **kwargs: Arguments concerning file metadata or
                spatial properties of the image.

        Returns:
            A Raster instance.
        """
        return cls(
            str(path),
            file_system=file_system,
            res=res,
            filename_regex=filename_regex,
            **kwargs,
        )

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        crs: Any,
        *,
        transform: Affine | None = None,
        bounds: tuple | Geometry | None = None,
        copy: bool = True,
        **kwargs,
    ) -> Self:
        """Construct Raster from numpy array.

        Must also specify nessecary spatial properties
        The necessary metadata is 'crs' and either 'transform' (Affine object)
        or 'bounds', which transform will then be created from.

        Args:
            array: 2d or 3d numpy ndarray.
            crs: Coordinate reference system.
            transform: Affine transform object. Can be specified instead
                of bounds.
            bounds: Minimum and maximum x and y coordinates. Can be specified instead
                of transform.
            copy: Whether to copy the array.
            **kwargs: Arguments concerning file metadata or
                spatial properties of the image.

        Returns:
            A Raster instance.
        """
        if array is None:
            raise TypeError("Must specify array.")

        if not any([transform, bounds]):
            raise TypeError(
                "Must specify either bounds or transform when constructing raster from array."
            )

        array = array.copy() if copy else array

        if len(array.shape) == 2:
            height, width = array.shape
        elif len(array.shape) == 3:
            height, width = array.shape[1:]
        else:
            raise ValueError("array must be 2 or 3 dimensional.")

        transform = Affine(*transform) if transform is not None else None

        if bounds is not None:
            bounds = to_bbox(bounds)

        if transform and not bounds:
            bounds = rasterio.transform.array_bounds(height, width, transform)

        crs = pyproj.CRS(crs) if crs else None

        return cls(array, crs=crs, transform=transform, bounds=bounds, **kwargs)

    @classmethod
    def from_gdf(
        cls,
        gdf: GeoDataFrame,
        columns: str | Iterable[str],
        res: int,
        fill: int = 0,
        all_touched: bool = False,
        merge_alg: Callable = MergeAlg.replace,
        default_value: int = 1,
        dtype: Any | None = None,
        **kwargs,
    ) -> Self:
        """Construct Raster from a GeoDataFrame.

        Args:
            gdf: The GeoDataFrame to rasterize.
            columns: Column(s) in the GeoDataFrame whose values are used to populate the raster.
                This can be a single column name or a list of column names.
            res: Resolution of the raster in units of the GeoDataFrame's coordinate reference system.
            fill: Fill value for areas outside of input geometries (default is 0).
            all_touched: Whether to consider all pixels touched by geometries,
                not just those whose center is within the polygon (default is False).
            merge_alg: Merge algorithm to use when combining geometries
                (default is 'MergeAlg.replace').
            default_value: Default value to use for the rasterized pixels
                (default is 1).
            dtype: Data type of the output array. If None, it will be
                determined automatically.
            **kwargs: Additional keyword arguments passed to the raster
                creation process, e.g., custom CRS or transform settings.

        Returns:
            A Raster instance based on the specified GeoDataFrame and parameters.

        Raises:
            TypeError: If 'transform' is provided in kwargs, as this is
            computed based on the GeoDataFrame bounds and resolution.
        """
        if not isinstance(gdf, GeoDataFrame):
            gdf = to_gdf(gdf)

        if "transform" in kwargs:
            raise TypeError("Unexpected argument 'transform'")

        kwargs["crs"] = gdf.crs or kwargs.get("crs")

        if kwargs["crs"] is None:
            raise TypeError("Must specify crs if the object doesn't have crs.")

        shape = get_shape_from_bounds(gdf.total_bounds, res=res)
        transform = get_transform_from_bounds(gdf.total_bounds, shape)
        kwargs["transform"] = transform

        def _rasterize(gdf, col):
            return features.rasterize(
                cls._gdf_to_geojson_with_col(gdf, col),
                out_shape=shape,
                transform=transform,
                fill=fill,
                all_touched=all_touched,
                merge_alg=merge_alg,
                default_value=default_value,
                dtype=dtype,
            )

        # make 2d array
        if isinstance(columns, str):
            array = _rasterize(gdf, columns)
            assert len(array.shape) == 2
            name = kwargs.get("name", columns)

        # 3d array even if single column in list/tuple
        elif hasattr(columns, "__iter__"):
            array = []
            for col in columns:
                arr = _rasterize(gdf, col)
                array.append(arr)
            array = np.array(array)
            assert len(array.shape) == 3
            name = kwargs.get("name", None)

        return cls.from_array(array, name=name, **kwargs)

    @classmethod
    def from_dict(cls, dictionary: dict) -> Self:
        """Construct Raster from metadata dict to fastpass the initializer.

        This is the fastest way to create a Raster since a metadata lookup is not
        needed.

        The dictionary must have all the keys ...
        and at least one of the keys 'transform' and 'bounds'.

        Args:
            dictionary: Dictionary with the nessecary and optional information
                about the raster. This can be fetched from an existing raster with
                the to_dict method.

        Returns:
            A Raster instance.
        """
        cls._validate_dict(dictionary)

        return cls(**dictionary)

    def update(self, **kwargs) -> Self:
        """Update attributes of the Raster."""
        for key, value in kwargs.items():
            self._validate_key(key)
            if is_property(self, key):
                key = "_" + key
            setattr(self, key, value)
        return self

    def write(
        self, path: str, window: rasterio.windows.Window | None = None, **kwargs
    ) -> None:
        """Write the raster as a single file.

        Multiband arrays will result in a multiband image file.

        Args:
            path: File path to write to.
            window: Optional window to clip the image to.
            **kwargs: Keyword arguments passed to rasterio.open.
                Thise will override the items in the Raster's profile,
                if overlapping.
        """
        if self.values is None:
            raise AttributeError("The image hasn't been loaded.")

        profile = self.profile | kwargs

        with opener(path, "wb", file_system=self.file_system) as file:
            with rasterio.open(file, "w", **profile) as dst:
                self._write(dst, window)

        self.path = str(path)

    def load(self, reload: bool = False, **kwargs) -> Self:
        """Load the entire image as an np.array.

        The array is stored in the 'array' attribute
        of the Raster.

        Args:
            reload: Whether to reload the array if already loaded.
            **kwargs: Keyword arguments passed to the rasterio read
                method.
        """
        if "mask" in kwargs:
            raise ValueError("Got an unexpected keyword argument 'mask'")
        if "window" in kwargs:
            raise ValueError("Got an unexpected keyword argument 'window'")

        if reload or self.values is None:
            self._read_tif(**kwargs)

        return self

    def clip(
        self,
        mask: Any,
        masked: bool = False,
        boundless: bool = True,
        **kwargs,
    ) -> Self:
        """Load the part of the image inside the mask.

        The returned array is stored in the 'array' attribute
        of the Raster.

        Args:
            mask: Geometry-like object or bounding box.
            masked: If 'masked' is True the return value will be a masked
                array. Otherwise (default) the return value will be a
                regular array. Masks will be exactly the inverse of the
                GDAL RFC 15 conforming arrays returned by read_masks().
            boundless: If True, windows that extend beyond the dataset's extent
                are permitted and partially or completely filled arrays will
                be returned as appropriate.
            **kwargs: Keyword arguments passed to the mask function
                from the rasterio.mask module.

        Returns:
            Self, but with the array loaded.
        """
        if not isinstance(mask, GeoDataFrame):
            mask = self._return_gdf(mask)

        try:
            mask = mask.to_crs(self.crs)
        except ValueError:
            mask = mask.set_crs(self.crs)

        self._read_with_mask(mask=mask, masked=masked, boundless=boundless, **kwargs)

        return self

    def intersects(self, other: Any) -> bool:
        """Returns True if the image bounds intersect with 'other'."""
        return self.union_all().intersects(to_shapely(other))

    def sample(
        self, n: int = 1, size: int = 20, mask: Any = None, copy: bool = True, **kwargs
    ) -> Self:
        """Take a random spatial sample of the image."""
        if mask is not None:
            points = GeoSeries(self.union_all()).clip(mask).sample_points(n)
        else:
            points = GeoSeries(self.union_all()).sample_points(n)
        buffered = points.buffer(size / self.res)
        boxes = to_gdf(
            [shapely.box(*arr) for arr in buffered.bounds.values], crs=self.crs
        )
        if copy:
            copy = self.copy()
            return copy.clip(boxes, **kwargs)
        return self.clip(boxes, **kwargs)

    def zonal(
        self,
        polygons: GeoDataFrame,
        aggfunc: str | Callable | list[Callable | str],
        array_func: Callable | None = None,
        dropna: bool = True,
    ) -> GeoDataFrame:
        """Calculate zonal statistics in polygons.

        Args:
            polygons: A GeoDataFrame of polygon geometries.
            aggfunc: Function(s) of which to aggregate the values
                within each polygon.
            array_func: Optional calculation of the raster
                array before calculating the zonal statistics.
            dropna: If True (default), polygons with all missing
                values will be removed.

        Returns:
            A GeoDataFrame with aggregated values per polygon.
        """
        idx_mapper, idx_name = get_index_mapper(polygons)
        polygons, aggfunc, func_names = _prepare_zonal(polygons, aggfunc)
        poly_iter = _make_geometry_iterrows(polygons)

        aggregated = []
        for i, poly in poly_iter:
            clipped = self.clip(poly)
            if not np.size(clipped.values):
                aggregated.append(_no_overlap_df(func_names, i, date=self.date))
            aggregated.append(
                _aggregate(
                    clipped.values, array_func, aggfunc, func_names, self.date, i
                )
            )

        return _zonal_post(
            aggregated,
            polygons=polygons,
            idx_mapper=idx_mapper,
            idx_name=idx_name,
            dropna=dropna,
        )

    def to_xarray(self) -> DataArray:
        """Convert the raster to  an xarray.DataArray."""
        self._check_for_array()
        self.name = self.name or self.__class__.__name__.lower()
        coords = _generate_spatial_coords(self.transform, self.width, self.height)
        if len(self.values.shape) == 2:
            dims = ["y", "x"]
            # dims = ["band", "y", "x"]
            # array = np.array([self.values])
            # assert len(array.shape) == 3
        elif len(self.values.shape) == 3:
            dims = ["band", "y", "x"]
            # array = self.values
        else:
            raise ValueError("Array must be 2 or 3 dimensional.")
        return xr.DataArray(
            self.values,
            coords=coords,
            dims=dims,
            name=self.name,
            attrs={"crs": self.crs},
        )  # .transpose("y", "x")

    def to_dict(self) -> dict:
        """Get a dictionary of Raster attributes."""
        out = {}
        for col in self.ALL_ATTRS:
            try:
                out[col] = self[col]
            except AttributeError:
                pass
        return out

    def to_gdf(self, column: str | list[str] | None = None) -> GeoDataFrame:
        """Create a GeoDataFrame from the raster.

        For multiband rasters, the bands are in separate rows with a "band" column
        value corresponding to the band indexes of the raster.

        Args:
            column: Name of resulting column(s) that holds the raster values.
                Can be a single string or an iterable with the same length as
                the number of raster bands.

        Returns:
            A GeoDataFrame with a geometry column, a 'band' column and a
            one or more value columns.
        """
        self._check_for_array()

        array_list = self.array_list()

        if is_list_like(column) and len(column) != len(array_list):
            raise ValueError(
                "columns should be a string or a list of same length as "
                f"layers in the array ({len(array_list)})."
            )

        if column is None:
            column = ["value"] * len(array_list)

        if isinstance(column, str):
            column = [column] * len(array_list)

        gdfs = []
        for i, (col, array) in enumerate(zip(column, array_list, strict=True)):
            gdf = gpd.GeoDataFrame(
                pd.DataFrame(
                    self._array_to_geojson(array, self.transform),
                    columns=[col, "geometry"],
                ),
                geometry="geometry",
                crs=self.crs,
            )
            gdf["indexes"] = i + 1
            gdfs.append(gdf)

        return pd.concat(gdfs, ignore_index=True)

    def set_crs(
        self,
        crs: pyproj.CRS | Any,
        allow_override: bool = False,
    ) -> Self:
        """Set coordinate reference system."""
        if not allow_override and self.crs is not None:
            raise ValueError("Cannot overwrite crs when allow_override is False.")

        if self.values is None:
            raise ValueError("array must be loaded/clipped before set_crs")

        self._crs = pyproj.CRS(crs)
        return self

    def to_crs(self, crs: pyproj.CRS | Any, **kwargs) -> Self:
        """Reproject the raster.

        Args:
            crs: The new coordinate reference system.
            **kwargs: Keyword arguments passed to the reproject function
                from the rasterio.warp module.
        """
        if self.crs is None:
            raise ValueError("Raster has no crs. Use set_crs.")

        # if pyproj.CRS(crs).equals(pyproj.CRS(self._crs)) and pyproj.CRS(crs).equals(
        #     pyproj.CRS(self._prev_crs)
        # ):
        #     return self

        if self.values is None:
            project = pyproj.Transformer.from_crs(
                pyproj.CRS(self._prev_crs), pyproj.CRS(crs), always_xy=True
            ).transform

            old_box = shapely.box(*self.bounds)
            new_box = shapely.ops.transform(project, old_box)
            self._bounds = to_bbox(new_box)

            # TODO: fix area changing... if possible
            # print("old/new:", shapely.area(old_box) / shapely.area(new_box))

            if pyproj.CRS(crs).equals(pyproj.CRS(self._crs)):
                self._warped_crs = self._crs
                return self

            # self._bounds = rasterio.warp.transform_bounds(
            #     pyproj.CRS(self._prev_crs), pyproj.CRS(crs), *to_bbox(self._bounds)
            # )
            # transformer = pyproj.Transformer.from_crs(
            #     pyproj.CRS(self._prev_crs), pyproj.CRS(crs), always_xy=True
            # )
            # minx, miny, maxx, maxy = self.bounds
            # xs, ys = transformer.transform(xx=[minx, maxx], yy=[miny, maxy])

            # minx, maxx = xs
            # miny, maxy = ys
            # self._bounds = minx, miny, maxx, maxy

            # self._bounds = shapely.transform(old_box, project)
        else:
            was_2d = len(self.shape) == 2
            self.values, transform = reproject(
                source=self.values,
                src_crs=self._prev_crs,
                src_transform=self.transform,
                dst_crs=pyproj.CRS(crs),
                **kwargs,
            )
            if was_2d and len(self.values.shape) == 3:
                assert self.values.shape[0] == 1
                self.values = self.values[0]

            self._bounds = rasterio.transform.array_bounds(
                self.height, self.width, transform
            )

        self._warped_crs = pyproj.CRS(crs)
        self._prev_crs = pyproj.CRS(crs)

        return self

    def plot(self, mask: Any | None = None) -> None:
        """Plot the images. One image per band."""
        self._check_for_array()
        if mask is not None:
            raster = self.copy().clip(mask)
        else:
            raster = self

        if len(raster.shape) == 2:
            array = np.array([raster.values])
        else:
            array = raster.values

        for arr in array:
            ax = plt.axes()
            ax.imshow(arr)
            ax.axis("off")
            plt.show()
            plt.close()

    def astype(self, dtype: type) -> Self:
        """Convert the datatype of the array."""
        if self.values is None:
            raise ValueError("Array is not loaded.")
        if not rasterio.dtypes.can_cast_dtype(self.values, dtype):
            min_dtype = rasterio.dtypes.get_minimum_dtype(self.values)
            raise ValueError(f"Cannot cast to dtype. Minimum dtype is {min_dtype}")
        self.values = self.values.astype(dtype)
        self._dtype = dtype
        return self

    def as_minimum_dtype(self) -> Self:
        """Convert the array to the minimum dtype without overflow."""
        min_dtype = rasterio.dtypes.get_minimum_dtype(self.values)
        self.values = self.values.astype(min_dtype)
        return self

    def min(self) -> int | None:
        """Minimum value in the array."""
        if np.size(self.values):
            return np.min(self.values)
        return None

    def max(self) -> int | None:
        """Maximum value in the array."""
        if np.size(self.values):
            return np.max(self.values)
        return None

    def _add_meta(self) -> Self:
        mess = "Cannot add metadata after image has been "
        if hasattr(self, "_clipped"):
            raise ValueError(mess + "clipped.")
        if hasattr(self, "_warped_crs"):
            raise ValueError(mess + "reprojected.")

        with opener(self.path, file_system=self.file_system) as file:
            with rasterio.open(file) as src:
                self._add_meta_from_src(src)

        return self

    def array_list(self) -> list[np.ndarray]:
        """Get a list of 2D arrays."""
        self._check_for_array()
        if len(self.values.shape) == 2:
            return [self.values]
        elif len(self.values.shape) == 3:
            return list(self.values)
        else:
            raise ValueError

    @property
    def indexes(self) -> int | tuple[int] | None:
        """Band indexes of the image."""
        return self._indexes

    @property
    def name(self) -> str | None:
        """Name of the file in the file path, if any."""
        try:
            return self._name
        except AttributeError:
            try:
                return Path(self.path).name
            except TypeError:
                return None

    @name.setter
    def name(self, value) -> None:
        self._name = value

    @property
    def date(self) -> str | None:
        """Date in the image file name, if filename_regex is present."""
        try:
            return re.match(self.filename_pattern, Path(self.path).name).group("date")
        except (AttributeError, TypeError):
            return None

    @property
    def band(self) -> str | None:
        """Band name of the image file name, if filename_regex is present."""
        try:
            return re.match(self.filename_pattern, Path(self.path).name).group("band")
        except (AttributeError, TypeError):
            return None

    @property
    def dtype(self) -> Any:
        """Data type of the array."""
        try:
            return self.values.dtype
        except AttributeError:
            try:
                return self._dtype
            except AttributeError:
                return None

    @dtype.setter
    def dtype(self, new_dtype: Any) -> None:
        self.values = self.values.astype(new_dtype)

    @property
    def nodata(self) -> int | None:
        """No data value."""
        try:
            return self._nodata
        except AttributeError:
            return None

    @property
    def tile(self) -> str | None:
        """Tile name from regex."""
        try:
            return re.match(self.filename_pattern, Path(self.path).name).group("tile")
        except (AttributeError, TypeError):
            return None

    @property
    def meta(self) -> dict:
        """Metadata dict."""
        return {
            "path": self.path,
            "type": self.__class__.__name__,
            "bounds": self.bounds,
            "indexes": self.indexes,
            "crs": self.crs,
        }

    @property
    def profile(self) -> dict:
        """Profile of the image file."""
        # TODO: .crs blir feil hvis warpa. Eller?
        return {
            "driver": self.driver,
            "compress": self.compress,
            "dtype": self.dtype,
            "crs": self.crs,
            "transform": self.transform,
            "nodata": self.nodata,
            "count": self.count,
            "height": self.height,
            "width": self.width,
            "indexes": self.indexes,
        }

    @property
    def read_kwargs(self) -> dict:
        """Keywords passed to the read method of rasterio.io.DatasetReader."""
        return {
            "indexes": self.indexes,
            "fill_value": self.nodata,
            "masked": False,
        }

    @property
    def res(self) -> float | None:
        """Get the spatial resolution of the image."""
        if hasattr(self, "_res") and self._res is not None:
            return self._res
        if self.width is None:
            return None
        diffx = self.bounds[2] - self.bounds[0]
        return diffx / self.width

    @property
    def height(self) -> int | None:
        """Get the height of the image as number of pixels."""
        if self.values is None:
            try:
                return self._height
            except AttributeError:
                return None
        i = 1 if len(self.values.shape) == 3 else 0
        return self.values.shape[i]

    @property
    def width(self) -> int | None:
        """Get the width of the image as number of pixels."""
        if self.values is None:
            try:
                return self._width
            except AttributeError:
                try:
                    heigth, width = get_shape_from_bounds(self, self.res)  # .res[0])
                    self._width = width
                    self._heigth = heigth
                    return self._width
                except Exception:
                    return None
        i = 2 if len(self.values.shape) == 3 else 1
        return self.values.shape[i]

    @property
    def count(self) -> int:
        """Get the number of bands in the image."""
        if self.values is not None:
            if len(self.values.shape) == 3:
                return self.values.shape[0]
            if len(self.values.shape) == 2:
                return 1
        if not hasattr(self._indexes, "__iter__"):
            return 1
        return len(self._indexes)

    @property
    def shape(self) -> tuple[int]:
        """Shape that is consistent with the array, whether it is loaded or not."""
        if self.values is not None:
            return self.values.shape
        if hasattr(self._indexes, "__iter__"):
            return self.count, self.width, self.height
        return self.width, self.height

    @property
    def transform(self) -> Affine | None:
        """Get the Affine transform of the image."""
        try:
            return rasterio.transform.from_bounds(*self.bounds, self.width, self.height)
        except (ZeroDivisionError, TypeError):
            if not self.width or not self.height:
                return None

    @property
    def bounds(self) -> tuple[float, float, float, float] | None:
        """Get the bounds of the image."""
        try:
            return to_bbox(self._bounds)
        except (AttributeError, TypeError):
            return None

    @property
    def crs(self) -> pyproj.CRS | None:
        """Get the coordinate reference system of the image."""
        try:
            return self._warped_crs
        except AttributeError:
            try:
                return self._crs
            except AttributeError:
                return None

    @property
    def area(self) -> float:
        """Get the area of the image."""
        return shapely.area(self.union_all())

    @property
    def length(self) -> float:
        """Get the circumfence of the image."""
        return shapely.length(self.union_all())

    @property
    def unary_union(self) -> Polygon:
        """Get the image bounds as a Polygon."""
        return shapely.box(*self.bounds)

    @property
    def centroid(self) -> Point:
        """Get the centerpoint of the image."""
        x = (self.bounds[0] + self.bounds[2]) / 2
        y = (self.bounds[1] + self.bounds[3]) / 2
        return Point(x, y)

    @property
    def properties(self) -> list[str]:
        """List of all properties of the class."""
        out = []
        for attr in dir(self):
            try:
                if is_property(self, attr):
                    out.append(attr)
            except AttributeError:
                pass
        return out

    def indexes_as_tuple(self) -> tuple[int, ...]:
        """Get the band index(es) as a tuple of integers."""
        if len(self.shape) == 2:
            return (1,)
        return tuple(i + 1 for i in range(self.shape[0]))

    def copy(self, deep: bool = True) -> "Raster":
        """Returns a (deep) copy of the class instance.

        Args:
            deep: Whether to return a deep or shallow copy. Defaults to True.
        """
        if deep:
            return deepcopy(self)
        else:
            return copy(self)

    def equals(self, other: Any) -> bool:
        """Check if the Raster is equal to another Raster."""
        if not isinstance(other, Raster):
            raise NotImplementedError("other must be of type Raster")
        if type(other) is not type(self):
            return False
        if self.values is None and other.values is not None:
            return False
        if self.values is not None and other.values is None:
            return False

        for method in dir(self):
            if not is_property(self, method):
                continue
            if getattr(self, method) != getattr(other, method):
                return False

        return np.array_equal(self.values, other.values)

    def __repr__(self) -> str:
        """The print representation."""
        shape = self.shape
        shp = ", ".join([str(x) for x in shape])
        try:
            res = int(self.res)
        except TypeError:
            res = None
        return f"{self.__class__.__name__}(shape=({shp}), res={res}, band={self.band})"

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over the arrays."""
        if len(self.values.shape) == 2:
            return iter([self.values])
        if len(self.values.shape) == 3:
            return iter(self.values)
        raise ValueError(
            f"Array should have shape length 2 or 3. Got {len(self.values.shape)}"
        )

    def __mul__(self, scalar: int | float) -> "Raster":
        """Multiply the array values with *."""
        self._check_for_array()
        self.values = self.values * scalar
        return self

    def __add__(self, scalar: int | float) -> "Raster":
        """Add to the array values with +."""
        self._check_for_array()
        self.values = self.values + scalar
        return self

    def __sub__(self, scalar: int | float) -> "Raster":
        """Subtract the array values with -."""
        self._check_for_array()
        self.values = self.values - scalar
        return self

    def __truediv__(self, scalar: int | float) -> "Raster":
        """Divide the array values with /."""
        self._check_for_array()
        self.values = self.values / scalar
        return self

    def __floordiv__(self, scalar: int | float) -> "Raster":
        """Floor divide the array values with //."""
        self._check_for_array()
        self.values = self.values // scalar
        return self

    def __pow__(self, exponent: int | float) -> "Raster":
        """Exponentiate the array values with **."""
        self._check_for_array()
        self.values = self.values**exponent
        return self

    def _has_nessecary_attrs(self, dict_like: dict) -> bool:
        """Check if Raster init got enough kwargs to not need to read src."""
        try:
            self._validate_dict(dict_like)
            return all(
                x is not None for x in [self.indexes, self.res, self.crs, self.bounds]
            )
        except AttributeError:
            return False

    def _return_self_or_copy(self, array: np.ndarray, copy: bool) -> "Raster":
        if not copy:
            self.values = array
            return self
        else:
            copy = self.copy()
            copy.values = array
            return copy

    @classmethod
    def _validate_dict(cls, dict_like: dict) -> None:
        missing = []
        for attr in NESSECARY_META:
            if any(
                [
                    attr in dict_like,
                    f"_{attr}" in dict_like,
                    attr.lstrip("_") in dict_like,
                ]
            ):
                continue
            missing.append(attr)
        if missing:
            raise AttributeError(f"Missing nessecary key(s) {', '.join(missing)}")

    @classmethod
    def _validate_key(cls, key: str) -> None:
        if key not in ALLOWED_KEYS:
            raise ValueError(
                f"Got an unexpected key {key!r}. Allowed keys are ",
                ", ".join(ALLOWED_KEYS),
            )

    def _get_shape_from_res(self, res: int) -> tuple[int] | None:
        if res is None:
            return None
        if hasattr(res, "__iter__") and len(res) == 2:
            res = res[0]
        diffx = self.bounds[2] - self.bounds[0]
        diffy = self.bounds[3] - self.bounds[1]
        width = int(diffx / res)
        height = int(diffy / res)
        if hasattr(self.indexes, "__iter__"):
            return len(self.indexes), width, height
        return width, height

    def _write(
        self, dst: rasterio.io.DatasetReader, window: rasterio.windows.Window
    ) -> None:
        if np.ma.is_masked(self.values):
            if len(self.values.shape) == 2:
                return dst.write(
                    self.values.filled(self.nodata), indexes=1, window=window
                )

            for i in range(len(self.indexes_as_tuple())):
                dst.write(
                    self.values[i].filled(self.nodata),
                    indexes=i + 1,
                    window=window,
                )

        else:
            if len(self.values.shape) == 2:
                return dst.write(self.values, indexes=1, window=window)

            for i, idx in enumerate(self.indexes_as_tuple()):
                dst.write(self.values[i], indexes=idx, window=window)

    def _get_indexes(self, indexes: int | tuple[int] | None) -> int | tuple[int] | None:
        if isinstance(indexes, numbers.Number):
            return int(indexes)
        if indexes is None:
            if self.values is not None and len(self.values.shape) == 3:
                return tuple(i + 1 for i in range(self.values.shape[0]))
            elif self.values is not None and len(self.values.shape) == 2:
                return 1
            elif self.values is not None:
                raise ValueError("Array must be 2 or 3 dimensional.")
            else:
                return None
        try:
            return tuple(int(x) for x in indexes)
        except Exception as e:
            raise TypeError(
                "indexes should be an integer or an iterable of integers."
                f"Got {type(indexes)}: {indexes}"
            ) from e

    def _return_gdf(self, obj: Any) -> GeoDataFrame:
        if isinstance(obj, str) and not is_wkt(obj):
            return self._read_tif(obj)
        elif isinstance(obj, Raster):
            return obj.to_gdf()
        elif is_bbox_like(obj):
            return to_gdf(shapely.box(*to_bbox(obj)), crs=self.crs)
        else:
            return to_gdf(obj, crs=self.crs)

    @staticmethod
    def _gdf_to_geojson(gdf: GeoDataFrame) -> list[dict]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return [x["geometry"] for x in loads(gdf.to_json())["features"]]

    @staticmethod
    def _gdf_to_geojson_with_col(gdf: GeoDataFrame, column: str) -> list[dict]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return [
                (feature["geometry"], val)
                for val, feature in zip(
                    gdf[column], loads(gdf.to_json())["features"], strict=False
                )
            ]

    @staticmethod
    def _array_to_geojson(array: np.ndarray, transform: Affine) -> list[tuple]:
        if np.ma.is_masked(array):
            array = array.data
        try:
            return [
                (value, shape(geom))
                for geom, value in features.shapes(
                    array, transform=transform, mask=None
                )
            ]
        except ValueError:
            array = array.astype(np.float32)
            return [
                (value, shape(geom))
                for geom, value in features.shapes(
                    array, transform=transform, mask=None
                )
            ]

    def _add_indexes_from_array(self, indexes: int | tuple[int]) -> int | tuple[int]:
        if indexes is not None:
            return indexes
        elif len(self.values.shape) == 3:
            return tuple(x + 1 for x in range(len(self.values)))
        elif len(self.values.shape) == 2:
            return 1
        else:
            raise ValueError

    def _add_meta_from_src(self, src: rasterio.io.DatasetReader) -> None:
        if not hasattr(self, "_bounds") or self._bounds is None:
            self._bounds = tuple(src.bounds)

        try:
            self._crs = pyproj.CRS(src.crs)
        except pyproj.exceptions.CRSError:
            self._crs = None

        self._width = src.width
        self._height = src.height

        # for attr in dir(self):
        #     try:
        #         if is_property(self, attr):
        #             continue
        #         if attr is None:
        #             new_value = getattr(src, attr)
        #             setattr(self, attr, new_value)
        #     except AttributeError:
        #         pass

        if not hasattr(self, "_indexes") or self._indexes is None:
            new_value = src.indexes
            if new_value == 1 or new_value == (1,):
                new_value = 1
            self._indexes = new_value

        if not hasattr(self, "_nodata") or self._nodata is None:
            new_value = src.nodata
            self._nodata = new_value

        # if not hasattr(self, "_indexes") or self._indexes is None:
        #     self._indexes = src.indexes

        # if not hasattr(self, "_nodata") or self._nodata is None:
        #     self._nodata = src.nodata

    def _load_warp_file(self) -> DatasetReader:
        """(from Torchgeo). Load and warp a file to the correct CRS and resolution.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        with opener(self.path, file_system=self.file_system) as file:
            src = rasterio.open(file)

        # Only warp if necessary
        if src.crs != self.crs:
            vrt = WarpedVRT(src, crs=self.crs)
            src.close()
            return vrt
        return src

    def _read_tif(self, **kwargs) -> None:
        return self._read(self.path, **kwargs)

    @functools.lru_cache(maxsize=128)
    def _read(self, path: str | Path, **kwargs) -> None:
        with opener(path, file_system=self.file_system) as file:
            with rasterio.open(file) as src:
                self._add_meta_from_src(src)
                out_shape = self._get_shape_from_res(self.res)

                if hasattr(self, "_warped_crs"):
                    src = WarpedVRT(src, crs=self.crs)

                self.values = src.read(
                    out_shape=out_shape,
                    **(self.read_kwargs | kwargs),
                )
                if self._dtype:
                    self = self.astype(self.dtype)
                else:
                    self = self.as_minimum_dtype()

    def _read_with_mask(
        self, mask: Any, masked: bool, boundless: bool, **kwargs
    ) -> None:
        kwargs["mask"] = mask

        def _read(self, src, mask, **kwargs):
            self._add_meta_from_src(src)
            if self.bounds is None:
                self._bounds = to_bbox(mask)

            window = rasterio.windows.from_bounds(
                *to_bbox(mask), transform=self.transform
            )

            out_shape = get_shape_from_bounds(mask, self.res)

            kwargs = (
                {"window": window, "boundless": boundless} | self.read_kwargs | kwargs
            )

            if hasattr(self, "_warped_crs"):
                src = WarpedVRT(src, crs=self.crs)

            self.values = src.read(out_shape=out_shape, **kwargs)

            if not masked:
                try:
                    self.values[self.values.mask] = self.nodata
                    self.values = self.values.data
                except AttributeError:
                    pass
                    # self.values = np.ma.masked_array(self.values, mask=mask)
                    # self.values[self.values.mask] = self.nodata
                    # self.values = self.values.data

            if boundless:
                self._bounds = src.window_bounds(window=window)
            else:
                intersected = to_shapely(self.bounds).intersection(to_shapely(mask))
                if intersected.is_empty:
                    self._bounds = None
                else:
                    self._bounds = intersected.bounds

            if not np.size(self.values):
                return

            if self._dtype:
                self = self.astype(self._dtype)
            else:
                self = self.as_minimum_dtype()

        if self.values is not None:
            with memfile_from_array(self.values, **self.profile) as src:
                _read(self, src, **kwargs)
        else:
            with opener(self.path, file_system=self.file_system) as file:
                with rasterio.open(file, **self.profile) as src:
                    _read(self, src, **kwargs)

    def _check_for_array(self, text=""):
        if self.values is None:
            raise ValueError("Arrays are not loaded. " + text)


def get_transform_from_bounds(
    obj: GeoDataFrame | GeoSeries | Geometry | tuple, shape: tuple[float, ...]
) -> Affine:
    minx, miny, maxx, maxy = to_bbox(obj)
    if len(shape) == 2:
        width, height = shape
    elif len(shape) == 3:
        _, width, height = shape
    else:
        raise ValueError
    return rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)


def get_shape_from_bounds(
    obj: GeoDataFrame | GeoSeries | Geometry | tuple, res: int
) -> tuple[int, int]:
    resx, resy = (res, res) if isinstance(res, numbers.Number) else res

    minx, miny, maxx, maxy = to_bbox(obj)
    diffx = maxx - minx
    diffy = maxy - miny
    width = int(diffx / resx)
    heigth = int(diffy / resy)
    return heigth, width

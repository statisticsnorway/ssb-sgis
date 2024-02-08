import functools
import numbers
import re
import warnings
from collections.abc import Callable, Iterable
from copy import copy, deepcopy
from json import loads
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely
from typing_extensions import Self  # TODO: imperter fra typing når python 3.11


try:
    import xarray as xr
    from xarray import DataArray
except ImportError:

    class DataArray:
        pass


try:
    from rioxarray.rioxarray import _generate_spatial_coords
except ImportError:
    pass
from affine import Affine
from geopandas import GeoDataFrame, GeoSeries
from pandas.api.types import is_list_like
from rasterio import features
from rasterio.enums import MergeAlg
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject
from shapely import Geometry
from shapely.geometry import Point, Polygon, shape

from ..geopandas_tools.conversion import to_bbox, to_gdf, to_shapely
from ..geopandas_tools.general import is_bbox_like, is_wkt
from ..helpers import is_property
from ..io.opener import opener
from .base import ALLOWED_KEYS, NESSECARY_META, get_index_mapper, memfile_from_array
from .gradient import get_gradient
from .zonal import (
    _aggregate,
    _no_overlap_df,
    make_geometry_iterrows,
    prepare_zonal,
    zonal_post,
)


numpy_func_message = (
    "aggfunc must be functions or strings of numpy functions or methods."
)


class Raster:
    """For reading, writing and working with rasters.

    Raster instances should be created with the methods 'from_path', 'from_array' or
    'from_gdf'.


    Examples
    --------

    Read tif file.

    >>> path = 'https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/raster/dtm_10.tif'
    >>> raster = sg.Raster.from_path(path)
    >>> raster
    Raster(shape=(1, 201, 201), res=10, crs=ETRS89 / UTM zone 33N (N-E), path=https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/raster/dtm_10.tif)

    Load the entire image as an numpy ndarray.
    Operations are done in place to save memory.
    The array is stored in the array attribute.

    >>> raster.load()
    >>> raster.array[raster.array < 0] = 0
    >>> raster.array
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

    >>> small_circle = raster_as_polygons.unary_union.centroid.buffer(50)
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

    # attributes conserning file path
    filename_regex: str | None = None
    date_format: str | None = None
    contains: str | None = None
    endswith: str = ".tif"

    # attributes conserning rasterio metadata
    _profile = {
        "driver": "GTiff",
        "compress": "LZW",
        "nodata": None,
        "dtype": None,
        "crs": None,
        "tiled": None,
        "indexes": None,
    }

    # driver: str = "GTiff"
    # compress: str = "LZW"
    # _nodata: int | float | None = None
    # _dtype: type | None = None

    def __init__(
        self,
        raster=None,
        *,
        path: str | None = None,
        # indexes: int | list[int] | None = None,
        array: np.ndarray | None = None,
        file_system=None,
        **kwargs,
    ):
        if raster is not None:
            if not isinstance(raster, Raster):
                raise TypeError(
                    "Raster should be constructed with the classmethods (from_...)."
                )
            for key, value in raster.__dict__.items():
                setattr(raster, key, value)
            return

        if path is None and not any([kwargs.get("transform"), kwargs.get("bounds")]):
            raise TypeError(
                "Must specify either bounds or transform when constructing raster from array."
            )

        # add class profile first to override with args and kwargs
        self.update(**self._profile)

        self._crs = kwargs.pop("crs", self._crs if hasattr(self, "_crs") else None)
        self._bounds = None

        self.path = path
        self.array = array
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
        file_system=None,
        **kwargs,
    ):
        """Construct Raster from file path.

        Args:
            path: Path to a raster image file.

        Returns:
            A Raster instance.
        """
        return cls(
            path=str(path),
            file_system=file_system,
            res=res,
            **kwargs,
        )

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        crs,
        *,
        transform: Affine | None = None,
        bounds: tuple | Geometry | None = None,
        copy: bool = True,
        **kwargs,
    ):
        """Construct Raster from numpy array.

        Metadata must be specified, either individually or in a dictionary
        (hint: use the 'meta' attribute of an existing Raster object if applicable).
        The necessary metadata is 'crs' and either 'transform' (Affine object) or 'bounds',
        which transform will then be created from.

        Args:
            array: 2d or 3d numpy ndarray.
            crs: Coordinate reference system.
            transform: Affine transform object. Can be specified instead
                of bounds.
            bounds: Minimum and maximum x and y coordinates. Can be specified instead
                of transform.
            name: Optional name to give the raster.

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

        return cls(array=array, crs=crs, transform=transform, bounds=bounds, **kwargs)

    @classmethod
    def from_gdf(
        cls,
        gdf: GeoDataFrame,
        columns: str | Iterable[str],
        res: int,
        fill=0,
        all_touched=False,
        merge_alg=MergeAlg.replace,
        default_value=1,
        dtype=None,
        **kwargs,
    ):
        """Construct Raster from a GeoDataFrame.

        Args:
            gdf: The GeoDataFrame.
            column: The column to be used as values for the array.
            res: Resolution of the raster in units of gdf's coordinate
                reference system.

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

        return cls.from_array(array=array, name=name, **kwargs)

    @classmethod
    def from_dict(cls, dictionary: dict):
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
        for key, value in kwargs.items():
            self._validate_key(key)
            if is_property(self, key):
                key = "_" + key
            setattr(self, key, value)
        return self

    def write(self, path: str, window=None, **kwargs) -> None:
        """Write the raster as a single file.

        Multiband arrays will result in a multiband image file.

        Args:
            path: File path to write to.
            window: Optional window to clip the image to.
        """

        if self.array is None:
            raise AttributeError("The image hasn't been loaded.")

        profile = self.profile | kwargs

        with opener(path, file_system=self.file_system) as file:
            with rasterio.open(file, "w", **profile) as dst:
                self._write(dst, window)

        self.path = str(path)

    def load(self, **kwargs) -> Self:
        """Load the entire image as an np.array.

        The array is stored in the 'array' attribute
        of the Raster.

        Args:
            **kwargs: Keyword arguments passed to the rasterio read
                method.
        """
        if "mask" in kwargs:
            raise ValueError("Got an unexpected keyword argument 'mask'")
        if "window" in kwargs:
            raise ValueError("Got an unexpected keyword argument 'window'")

        self._read_tif(**kwargs)

        return self

    def clip(
        self,
        mask,
        masked: bool = False,
        boundless: bool = True,
        **kwargs,
    ) -> Self:
        """Load the part of the image inside the mask.

        The returned array is stored in the 'array' attribute
        of the Raster.

        Args:
            mask: Geometry-like object or bounding box.
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

        # if not self.crs.equals(pyproj.CRS(mask.crs)):
        #     raise ValueError("crs mismatch.")

        self._read_with_mask(mask=mask, masked=masked, boundless=boundless, **kwargs)

        return self

    def intersects(self, other) -> bool:
        return self.unary_union.intersects(to_shapely(other))

    def sample(self, n=1, size=20, mask=None, copy=True, **kwargs) -> Self:
        if mask is not None:
            points = GeoSeries(self.unary_union).clip(mask).sample_points(n)
        else:
            points = GeoSeries(self.unary_union).sample_points(n)
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
        polygons, aggfunc, func_names = prepare_zonal(polygons, aggfunc)
        poly_iter = make_geometry_iterrows(polygons)

        aggregated = []
        for i, poly in poly_iter:
            clipped = self.clip(poly)
            if not np.size(clipped.array):
                aggregated.append(_no_overlap_df(func_names, i, date=self.date))
            aggregated.append(
                _aggregate(clipped.array, array_func, aggfunc, func_names, self.date, i)
            )

        return zonal_post(
            aggregated,
            polygons=polygons,
            idx_mapper=idx_mapper,
            idx_name=idx_name,
            dropna=dropna,
        )

    def gradient(self, degrees: bool = False, copy: bool = False) -> Self:
        """Get the slope of an elevation raster.

        Calculates the absolute slope between the grid cells
        based on the image resolution.

        For multiband images, the calculation is done for each band.

        Args:
            degrees: If False (default), the returned values will be in ratios,
                where a value of 1 means 1 meter up per 1 meter forward. If True,
                the values will be in degrees from 0 to 90.
            copy: Whether to copy or overwrite the original Raster.
                Defaults to False to save memory.

        Returns:
            The class instance with new array values, or a copy if copy is True.

        Examples
        --------
        Making an array where the gradient to the center is always 10.

        >>> import sgis as sg
        >>> import numpy as np
        >>> arr = np.array(
        ...         [
        ...             [100, 100, 100, 100, 100],
        ...             [100, 110, 110, 110, 100],
        ...             [100, 110, 120, 110, 100],
        ...             [100, 110, 110, 110, 100],
        ...             [100, 100, 100, 100, 100],
        ...         ]
        ...     )

        Now let's create a Raster from this array with a resolution of 10.

        >>> r = sg.Raster.from_array(arr, crs=None, bounds=(0, 0, 50, 50))

        The gradient will be 1 (1 meter up for every meter forward).
        The calculation is by default done in place to save memory.

        >>> r.gradient()
        >>> r.array
        array([[0., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1.],
            [1., 1., 0., 1., 1.],
            [1., 1., 1., 1., 1.],
            [0., 1., 1., 1., 0.]])
        """
        return get_gradient(self, degrees=degrees, copy=copy)

    def to_xarray(self) -> DataArray:
        self._check_for_array()
        self.name = self.name or self.__class__.__name__.lower()
        coords = _generate_spatial_coords(self.transform, self.width, self.height)
        if len(self.array.shape) == 2:
            dims = ["y", "x"]
            # dims = ["band", "y", "x"]
            # array = np.array([self.array])
            # assert len(array.shape) == 3
        elif len(self.array.shape) == 3:
            dims = ["band", "y", "x"]
            # array = self.array
        else:
            raise ValueError("Array must be 2 or 3 dimensional.")
        return xr.DataArray(
            self.array,
            coords=coords,
            dims=dims,
            name=self.name,
            attrs={"crs": self.crs},
        )  # .transpose("y", "x")

    def to_dict(self) -> dict:
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
        for i, (column, array) in enumerate(zip(column, array_list, strict=True)):
            gdf = gpd.GeoDataFrame(
                pd.DataFrame(
                    self._array_to_geojson(array, self.transform),
                    columns=[column, "geometry"],
                ),
                geometry="geometry",
                crs=self.crs,
            )
            gdf["indexes"] = i + 1
            gdfs.append(gdf)

        return pd.concat(gdfs, ignore_index=True)

    def set_crs(
        self,
        crs,
        allow_override: bool = False,
    ) -> Self:
        """Set coordinate reference system."""
        if not allow_override and self.crs is not None:
            raise ValueError("Cannot overwrite crs when allow_override is False.")

        if self.array is None:
            raise ValueError("array must be loaded/clipped before set_crs")

        self._crs = pyproj.CRS(crs)
        return self

    def to_crs(self, crs, **kwargs) -> Self:
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

        if self.array is None:
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
            self.array, transform = reproject(
                source=self.array,
                src_crs=self._prev_crs,
                src_transform=self.transform,
                dst_crs=pyproj.CRS(crs),
                **kwargs,
            )
            if was_2d and len(self.array.shape) == 3:
                assert self.array.shape[0] == 1
                self.array = self.array[0]

            self._bounds = rasterio.transform.array_bounds(
                self.height, self.width, transform
            )

        self._warped_crs = pyproj.CRS(crs)
        self._prev_crs = pyproj.CRS(crs)

        return self

    def plot(self, mask=None) -> None:
        self._check_for_array()
        """Plot the images. One image per band."""
        if mask is not None:
            raster = self.copy().clip(mask)
        else:
            raster = self

        if len(raster.shape) == 2:
            array = np.array([raster.array])
        else:
            array = raster.array

        for arr in array:
            ax = plt.axes()
            ax.imshow(arr)
            ax.axis("off")
            plt.show()
            plt.close()

    def astype(self, dtype: type) -> Self:
        if self.array is None:
            raise ValueError("Array is not loaded.")
        if not rasterio.dtypes.can_cast_dtype(self.array, dtype):
            min_dtype = rasterio.dtypes.get_minimum_dtype(self.array)
            raise ValueError(f"Cannot cast to dtype. Minimum dtype is {min_dtype}")
        self.array = self.array.astype(dtype)
        self._dtype = dtype
        return self

    def as_minimum_dtype(self) -> Self:
        min_dtype = rasterio.dtypes.get_minimum_dtype(self.array)
        self.array = self.array.astype(min_dtype)
        return self

    def min(self) -> int | None:
        if np.size(self.array):
            return np.min(self.array)
        return None

    def max(self) -> int | None:
        if np.size(self.array):
            return np.max(self.array)
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
        self._check_for_array()
        if len(self.array.shape) == 2:
            return [self.array]
        elif len(self.array.shape) == 3:
            return list(self.array)
        else:
            raise ValueError

    @property
    def indexes(self) -> int | tuple[int] | None:
        return self._indexes

    @property
    def name(self) -> str | None:
        try:
            return self._name
        except AttributeError:
            try:
                return Path(self.path).name
            except TypeError:
                return None

    @name.setter
    def name(self, value):
        self._name = value
        return self._name

    @property
    def date(self):
        try:
            pattern = re.compile(self.filename_regex, re.VERBOSE)
            return re.match(pattern, Path(self.path).name).group("date")
        except (AttributeError, TypeError):
            return None

    @property
    def band(self) -> str | None:
        try:
            pattern = re.compile(self.filename_regex, re.VERBOSE)
            return re.match(pattern, Path(self.path).name).group("band")
        except (AttributeError, TypeError):
            return None

    # @property
    # def band_color(self):
    #     """To be implemented in subclasses."""
    #     pass

    @property
    def dtype(self):
        try:
            return self.array.dtype
        except AttributeError:
            try:
                return self._dtype
            except AttributeError:
                return None

    @dtype.setter
    def dtype(self, new_dtype):
        self.array = self.array.astype(new_dtype)
        return self.array.dtype

    @property
    def nodata(self) -> int | None:
        try:
            return self._nodata
        except AttributeError:
            return None

    @property
    def tile(self) -> str | None:
        if self.bounds is None:
            return None
        return f"{int(self.bounds[0])}_{int(self.bounds[1])}"

    @property
    def meta(self) -> dict:
        return {
            "path": self.path,
            "type": self.__class__.__name__,
            "bounds": self.bounds,
            "indexes": self.indexes,
            "crs": self.crs,
        }

    @property
    def profile(self) -> dict:
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
        return {
            "indexes": self.indexes,
            "fill_value": self.nodata,
            "masked": True,
        }

    @property
    def res(self) -> float | None:
        if hasattr(self, "_res") and self._res is not None:
            return self._res
        if self.width is None:
            return None
        diffx = self.bounds[2] - self.bounds[0]
        return diffx / self.width

    @property
    def height(self) -> int | None:
        if self.array is None:
            try:
                return self._height
            except AttributeError:
                return None
        i = 1 if len(self.array.shape) == 3 else 0
        return self.array.shape[i]

    @property
    def width(self) -> int | None:
        if self.array is None:
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
        i = 2 if len(self.array.shape) == 3 else 1
        return self.array.shape[i]

    @property
    def count(self) -> int:
        if self.array is not None:
            if len(self.array.shape) == 3:
                return self.array.shape[0]
            if len(self.array.shape) == 2:
                return 1
        if not hasattr(self._indexes, "__iter__"):
            return 1
        return len(self._indexes)

    @property
    def shape(self) -> tuple[int]:
        """Shape that is consistent with the array, whether it is loaded or not."""
        if self.array is not None:
            return self.array.shape
        if hasattr(self._indexes, "__iter__"):
            return self.count, self.width, self.height
        return self.width, self.height

    @property
    def transform(self) -> Affine | None:
        try:
            return rasterio.transform.from_bounds(*self.bounds, self.width, self.height)
        except (ZeroDivisionError, TypeError):
            if not self.width or not self.height:
                return None

    @property
    def bounds(self) -> tuple[float, float, float, float] | None:
        try:
            return to_bbox(self._bounds)
        except (AttributeError, TypeError):
            return None

    @property
    def crs(self):
        try:
            return self._warped_crs
        except AttributeError:
            try:
                return self._crs
            except AttributeError:
                return None

    @property
    def area(self) -> float:
        return shapely.area(self.unary_union)

    @property
    def length(self) -> float:
        return shapely.length(self.unary_union)

    @property
    def unary_union(self) -> Polygon:
        return shapely.box(*self.bounds)

    @property
    def centroid(self) -> Point:
        x = (self.bounds[0] + self.bounds[2]) / 2
        y = (self.bounds[1] + self.bounds[3]) / 2
        return Point(x, y)

    @property
    def properties(self) -> list[str]:
        out = []
        for attr in dir(self):
            try:
                if is_property(self, attr):
                    out.append(attr)
            except AttributeError:
                pass
        return out

    def indexes_as_tuple(self) -> tuple[int, ...]:
        if len(self.shape) == 2:
            return (1,)
        return tuple(i + 1 for i in range(self.shape[0]))

    def copy(self, deep=True):
        """Returns a (deep) copy of the class instance.

        Args:
            deep: Whether to return a deep or shallow copy. Defaults to True.
        """
        if deep:
            return deepcopy(self)
        else:
            return copy(self)

    def equals(self, other) -> bool:
        if not isinstance(other, Raster):
            raise NotImplementedError("other must be of type Raster")
        if type(other) != type(self):
            return False
        if self.array is None and other.array is not None:
            return False
        if self.array is not None and other.array is None:
            return False

        for method in dir(self):
            if not is_property(self, method):
                continue
            if getattr(self, method) != getattr(other, method):
                return False

        return np.array_equal(self.array, other.array)

    def __repr__(self) -> str:
        """The print representation."""
        shape = self.shape
        shp = ", ".join([str(x) for x in shape])
        try:
            res = int(self.res)
        except TypeError:
            res = None
        return f"{self.__class__.__name__}(shape=({shp}), res={res}, name={self.name}, path={self.path})"

    def __mul__(self, scalar):
        self._check_for_array()
        self.array = self.array * scalar
        return self

    def __add__(self, scalar):
        self._check_for_array()
        self.array = self.array + scalar
        return self

    def __sub__(self, scalar):
        self._check_for_array()
        self.array = self.array - scalar
        return self

    def __truediv__(self, scalar):
        self._check_for_array()
        self.array = self.array / scalar
        return self

    def __floordiv__(self, scalar):
        self._check_for_array()
        self.array = self.array // scalar
        return self

    def __pow__(self, exponent):
        self._check_for_array()
        self.array = self.array**exponent
        return self

    def _has_nessecary_attrs(self, dict_like) -> bool:
        """Check if Raster init got enough kwargs to not need to read src."""
        try:
            self._validate_dict(dict_like)
            return all(
                x is not None for x in [self.indexes, self.res, self.crs, self.bounds]
            )
        except AttributeError:
            return False

    def _return_self_or_copy(self, array, copy: bool):
        if not copy:
            self.array = array
            return self
        else:
            copy = self.copy()
            copy.array = array
            return copy

    @classmethod
    def _validate_dict(cls, dict_like) -> None:
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
    def _validate_key(cls, key) -> None:
        if key not in ALLOWED_KEYS:
            raise ValueError(
                f"Got an unexpected key {key!r}. Allowed keys are ",
                ", ".join(ALLOWED_KEYS),
            )

    def _get_shape_from_res(self, res) -> tuple[int] | None:
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

    def _write(self, dst, window):
        if np.ma.is_masked(self.array):
            if len(self.array.shape) == 2:
                return dst.write(
                    self.array.filled(self.nodata), indexes=1, window=window
                )

            for i in range(len(self.indexes_as_tuple())):
                dst.write(
                    self.array[i].filled(self.nodata),
                    indexes=i + 1,
                    window=window,
                )

        else:
            if len(self.array.shape) == 2:
                return dst.write(self.array, indexes=1, window=window)

            for i, idx in enumerate(self.indexes_as_tuple()):
                dst.write(self.array[i], indexes=idx, window=window)

    def _get_indexes(self, indexes):
        if isinstance(indexes, numbers.Number):
            return int(indexes)
        if indexes is None:
            if self.array is not None and len(self.array.shape) == 3:
                return tuple(i + 1 for i in range(self.array.shape[0]))
            elif self.array is not None and len(self.array.shape) == 2:
                return 1
            elif self.array is not None:
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

    def _return_gdf(self, obj) -> GeoDataFrame:
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
                for val, feature in zip(gdf[column], loads(gdf.to_json())["features"])
            ]

    @staticmethod
    def _array_to_geojson(array: np.ndarray, transform: Affine):
        try:
            return [
                (value, shape(geom))
                for geom, value in features.shapes(array, transform=transform)
            ]
        except ValueError:
            array = array.astype(np.float32)
            return [
                (value, shape(geom))
                for geom, value in features.shapes(array, transform=transform)
            ]

    def _add_indexes_from_array(self, indexes):
        if indexes is not None:
            return indexes
        elif len(self.array.shape) == 3:
            return tuple(x + 1 for x in range(len(self.array)))
        elif len(self.array.shape) == 2:
            return 1
        else:
            raise ValueError

    def _add_meta_from_src(self, src):
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

        for attr in ["_indexes", "_nodata"]:
            if not hasattr(self, attr) or getattr(self, attr) is None:
                new_value = getattr(src, attr.replace("_", ""))
                setattr(self, attr, new_value)

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
    def _read(self, path, **kwargs):
        with opener(path, file_system=self.file_system) as file:
            with rasterio.open(file) as src:
                self._add_meta_from_src(src)
                out_shape = self._get_shape_from_res(self.res)

                if hasattr(self, "_warped_crs"):
                    src = WarpedVRT(src, crs=self.crs)

                self.array = src.read(
                    out_shape=out_shape,
                    **(self.read_kwargs | kwargs),
                )
                if self._dtype:
                    self = self.astype(self.dtype)
                else:
                    self = self.as_minimum_dtype()

    def _read_with_mask(self, mask, masked, boundless, **kwargs):
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

            self.array = src.read(out_shape=out_shape, **kwargs)

            if not masked:
                self.array[self.array.mask] = self.nodata
                self.array = self.array.data

            if boundless:
                self._bounds = src.window_bounds(window=window)
            else:
                intersected = to_shapely(self.bounds).intersection(to_shapely(mask))
                if intersected.is_empty:
                    self._bounds = None
                else:
                    self._bounds = intersected.bounds

            if not np.size(self.array):
                return

            if self._dtype:
                self = self.astype(self._dtype)
            else:
                self = self.as_minimum_dtype()

        if self.array is not None:
            with memfile_from_array(self.array, **self.profile) as src:
                _read(self, src, **kwargs)
        else:
            with opener(self.path, file_system=self.file_system) as file:
                with rasterio.open(file, **self.profile) as src:
                    _read(self, src, **kwargs)

    def _check_for_array(self, text=""):
        if self.array is None:
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

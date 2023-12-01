import numbers
import re
import uuid
import warnings
from collections.abc import Callable
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


try:
    import xarray as xr
except ImportError:
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
from rasterio.mask import mask as rast_mask
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject
from shapely import Geometry, box
from shapely.geometry import Point, Polygon, shape

from ..geopandas_tools.bounds import to_bbox
from ..geopandas_tools.conversion import to_gdf
from ..geopandas_tools.general import is_bbox_like, is_wkt
from ..helpers import get_non_numpy_func_name, get_numpy_func, is_property
from ..io.opener import opener
from .base import (
    RasterBase,
    RasterHasChangedError,
    get_index_mapper,
    memfile_from_array,
)
from .zonal import (
    _aggregate,
    _no_overlap_df,
    make_geometry_iterrows,
    prepare_zonal,
    zonal_post,
)


numpy_func_message = (
    "aggfunc must be functions or " "strings of numpy functions or methods."
)


class Raster(RasterBase):
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
           elevation                                           geometry  band_index
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

    def __init__(
        self,
        raster=None,
        *,
        path: str | None = None,
        array: np.ndarray | None = None,
        band_index: int | list[int] | None = None,
        name: str | None = None,
        name_regex: str | None = None,
        date: str | None = None,
        date_regex: str | None = None,
        shortname: str | None = None,
        dtype: type | None = None,
        nodata: int | float | None = None,
        **kwargs,
    ):
        if raster is not None:
            if not isinstance(raster, Raster):
                raise TypeError(
                    "Raster should be constructed with the from-classmethods."
                )
            for key, value in raster.__dict__.items():
                self[key] = value
            return

        self.path = path
        self.array = array

        # hasattr check to allow class attributes in subclasses
        if not hasattr(self, "shortname"):
            self.shortname = shortname
        if not hasattr(self, "name_regex"):
            self.name_regex = name_regex
        if not hasattr(self, "date_regex"):
            self.date_regex = date_regex
        if not hasattr(self, "_dtype") or dtype is not None:
            self._dtype = dtype
        if not hasattr(self, "nodata"):
            self.nodata = nodata

        if not hasattr(self, "driver"):
            self.driver = kwargs.pop("driver", "GTiff")
        if not hasattr(self, "compress"):
            self.compress = kwargs.pop("compress", "LZW")

        if not hasattr(self, "_band_index") or band_index is not None:
            self._band_index = self._get_band_index(band_index)

        # underscore to allow properties without setters in subclasses (e.g. Sentinel2)
        if date:
            self._date = date
        if name:
            self._name = name

        self.root = kwargs.pop("root", None)

        self._hash = uuid.uuid4()

        # override the above with kwargs
        self.update(**kwargs)

        if self.path is None and self.array is None:
            return

        if self.path is not None and self.array is not None:
            raise ValueError("Cannot supply both 'path' and 'array'.")

        # the important attributes must be of correct type
        crs = kwargs.get("crs")
        transform = kwargs.get("transform")
        bounds = kwargs.get("bounds")
        if bounds is not None:
            bounds = to_bbox(bounds)

        if not any([path, transform, bounds]):
            raise TypeError(
                "Must specify either bounds or transform when constructing raster from array."
            )

        self._crs = pyproj.CRS(crs) if crs else None
        transform = Affine(*transform) if transform is not None else None

        if transform and not bounds:
            self._bounds = rasterio.transform.array_bounds(
                self.height, self.width, transform
            )
        elif not path:
            self._bounds = bounds

        attributes = set(self.__dict__.keys()).difference(set(self.properties))

        if self.path is not None and not self.has_nessecary_attrs(attributes):
            self.add_meta()
            self._meta_added = True

        if self.path is None:
            if not isinstance(self.array, np.ndarray):
                raise TypeError

            if not hasattr(self, "crs"):
                raise TypeError("Must specify crs when constructing raster from array.")

            self._band_index = self._add_band_index_from_array(band_index)

        self._prev_crs = self._crs

    @classmethod
    def from_path(
        cls,
        path: str,
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
            **kwargs,
        )

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        *,
        crs=None,
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

        array = array.copy() if copy else array

        return cls(
            array=array,
            crs=crs,
            transform=transform,
            bounds=bounds,
            **kwargs,
        )

    @classmethod
    def from_gdf(
        cls,
        gdf: GeoDataFrame,
        columns: str | list[str],
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

        shape = cls.get_shape_from_bounds(gdf.total_bounds, res=res)
        transform = cls.get_transform_from_bounds(gdf.total_bounds, shape)
        kwargs["transform"] = transform

        def _rasterize(gdf, col):
            return features.rasterize(
                cls._to_geojson_geom_val(gdf, col),
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

        cls.validate_dict(dictionary)

        return cls(**dictionary)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.validate_key(key)
            if key == "indexes":
                self._band_index = value
            if is_property(self, key):
                self["_" + key] = value
            else:
                self[key] = value
        return self

    def load(self, res: int | None = None, **kwargs):
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

        self._read_tif(res=res, **kwargs)

        return self

    def clip(self, mask, boundless: bool = True, **kwargs):
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

        if not self.crs.equals(pyproj.CRS(mask.crs)):
            raise ValueError("crs mismatch.")

        self._read_with_mask(mask=mask, boundless=boundless, **kwargs)

        return self

    def sample(self, n=1, size=20, mask=None, copy=True, **kwargs):
        if mask is not None:
            points = GeoSeries(self.unary_union).clip(mask).sample_points(n)
        else:
            points = GeoSeries(self.unary_union).sample_points(n)
        buffered = points.buffer(size / self.res[0])
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

    def write(self, path: str, window=None, **kwargs):
        """Write the raster as a single file.

        Multiband arrays will result in a multiband image file.

        Args:
            path: File path to write to.
            window: Optional window to clip the image to.
        """

        if self.array is None:
            raise AttributeError("The image hasn't been loaded yet.")

        profile = self.profile | kwargs

        with opener(path) as file:
            with rasterio.open(file, "w", **profile) as dst:
                self._write(dst, window)

        self.path = str(path)

    def to_xarray(self) -> xr.DataArray:
        self.check_for_array()
        coords = _generate_spatial_coords(self.transform, self.width, self.height)
        if len(self.array.shape) == 2:
            dims = ["y", "x"]
        elif len(self.array.shape) == 3:
            dims = ["band", "y", "x"]
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
        self.check_for_array()

        array_list = self._to_2d_array_list(self.array)

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
            gdf["band_index"] = i + 1

            if hasattr(self, "_datadict"):
                for name, value in self._datadict.items():
                    try:
                        gdf[name] = value
                    except Exception:
                        # in case of iterable columns (band_index...)
                        # reset index in case of duplicate index
                        index = gdf.index
                        gdf = gdf.reset_index(drop=True)
                        gdf[name] = pd.Series(
                            [value for _ in range(len(gdf))], index=gdf.index
                        )
                        gdf.index = index

            gdfs.append(gdf)

        return pd.concat(gdfs, ignore_index=True)

    def set_crs(
        self,
        crs,
        allow_override: bool = False,
    ):
        """Set coordinate reference system."""
        if not allow_override and self.crs is not None:
            raise ValueError("Cannot overwrite crs when allow_override is False.")

        if self.array is None:
            raise ValueError("array must be loaded/clipped before set_crs")

        self._crs = pyproj.CRS(crs)
        return self

    def to_crs(self, crs, **kwargs):
        """Reproject the raster.

        Args:
            crs: The new coordinate reference system.
            **kwargs: Keyword arguments passed to the reproject function
                from the rasterio.warp module.
        """
        if self.crs is None:
            raise ValueError("Raster has no crs. Use set_crs.")

        if pyproj.CRS(crs).equals(pyproj.CRS(self.crs)):
            return self

        if self.array is None:
            project = pyproj.Transformer.from_crs(
                pyproj.CRS(self._prev_crs), pyproj.CRS(crs), always_xy=True
            ).transform

            old_box = shapely.box(*self.bounds)
            new_box = shapely.ops.transform(project, old_box)
            self._bounds = to_bbox(new_box)

            # TODO: fix this
            print("old/new:", shapely.area(old_box) / shapely.area(new_box))

            """self._bounds = rasterio.warp.transform_bounds(
                pyproj.CRS(self._prev_crs), pyproj.CRS(crs), *to_bbox(self._bounds)
            )
            transformer = pyproj.Transformer.from_crs(
                pyproj.CRS(self._prev_crs), pyproj.CRS(crs), always_xy=True
            )
            minx, miny, maxx, maxy = self.bounds
            xs, ys = transformer.transform(xx=[minx, maxx], yy=[miny, maxy])

            minx, maxx = xs
            miny, maxy = ys
            self._bounds = minx, miny, maxx, maxy"""

            # self._bounds = shapely.transform(old_box, project)
        else:
            self.array, transform = reproject(
                source=self.array,
                src_crs=self.crs,
                src_transform=self.transform,
                dst_crs=pyproj.CRS(crs),
                **kwargs,
            )

            self._bounds = rasterio.transform.array_bounds(
                self.height, self.width, transform
            )

        self._warped_crs = pyproj.CRS(crs)
        self._prev_crs = pyproj.CRS(crs)

        return self

    def plot(self, mask=None) -> None:
        self.check_for_array()
        """Plot the images. One image per band."""
        if len(self.shape) == 3:
            for arr in self.array:
                self._plot_2d(arr)

    def astype(self, dtype: type):
        if self.array is None:
            raise ValueError("Array is not loaded.")
        if not rasterio.dtypes.can_cast_dtype(self.array, dtype):
            min_dtype = rasterio.dtypes.get_minimum_dtype(self.array)
            raise ValueError(f"Cannot cast to dtype. Minimum dtype is {min_dtype}")
        self.array = self.array.astype(dtype)
        self._dtype = dtype
        return self

    def as_minimum_dtype(self):
        min_dtype = rasterio.dtypes.get_minimum_dtype(self.array)
        self.array = self.array.astype(min_dtype)
        return self

    def add_meta(self):
        mess = "Cannot add metadata after image has been "
        if hasattr(self, "_clipped"):
            raise ValueError(mess + "clipped.")
        if hasattr(self, "_warped_crs"):
            raise ValueError(mess + "reprojected.")

        with opener(self.path) as file:
            with rasterio.open(file) as src:
                self._add_meta_from_src(src)

        return self

    def get_coords(self):
        # TODO: droppe
        self.check_for_array()
        return _generate_spatial_coords(self.transform, self.width, self.height)

    @property
    def subfolder(self):
        try:
            folder = Path(Path(self.path).parent).name
            if self.root:
                return folder.difference(self.root)
            else:
                return folder
        except TypeError:
            return None

    @property
    def tile(self) -> str | None:
        if self.bounds is None:
            return None
        return f"{int(self.bounds[0])}{int(self.bounds[1])}"

    @property
    def raster_id(self) -> str:
        shortname = self.shortname if self.shortname else ""
        date = self.date if self.date else ""
        name = self.name if self.name else ""
        return f"{shortname}_{self.tile}_{date}_{name}".replace("__", "_").strip("_")

    @property
    def band_index(self):
        return self._band_index

    @property
    def name(self) -> str | None:
        if hasattr(self, "_name"):
            return self._name
        if not self.name_regex and self.path:
            return Path(self.path).stem
        try:
            return re.search(self.name_regex, Path(self.path).name).group()
        except (AttributeError, TypeError):
            return None

    @name.setter
    def name(self, value):
        self._name = value
        return self._name

    @property
    def date(self):
        if hasattr(self, "_date"):
            return self._date
        try:
            return re.search(self.date_regex, Path(self.path).name).group()
        except (AttributeError, TypeError):
            return None

    @date.setter
    def date(self, value):
        self._date = value
        return self._date

    @property
    def band_color(self):
        """To be implemented in subclasses."""
        pass

    @property
    def is_mask(self):
        """To be implemented in subclasses."""
        pass

    def array_list(self):
        return self._to_2d_array_list(self.array)

    @property
    def dtype(self):
        try:
            return self.array.dtype
        except Exception:
            return getattr(self, "_dtype", None)

    @dtype.setter
    def dtype(self, new_dtype):
        self.array = self.array.astype(new_dtype)
        return self.array.dtype

    def read_kwargs(self, kwargs):
        # if kwargs is None:
        #   kwargs = {}
        return {
            "indexes": self.band_index,
            "masked": False,
            "fill_value": self.nodata,
        } | kwargs or {}

    @property
    def profile(self):
        # TODO: .crs blir feil hvis warpa. Eller?
        return {
            "driver": self.driver,
            "dtype": self.dtype,
            "crs": self.crs,
            "transform": self.transform,
            "nodata": self.nodata,
            "count": self.count,
            "height": self.height,
            "width": self.width,
            "compress": self.compress,
            "indexes": self.band_index,
        }

    @property
    def raster_type(self) -> str:
        return self.__class__.__name__

    @property
    def area(self):
        return shapely.area(self.unary_union)

    @property
    def length(self):
        return shapely.length(self.unary_union)

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
    def unary_union(self) -> Polygon:
        return shapely.box(*self.bounds)

    @property
    def centroid(self) -> Point:
        x = (self.bounds[0] + self.bounds[2]) / 2
        y = (self.bounds[1] + self.bounds[3]) / 2
        return Point(x, y)

    @property
    def res(self) -> tuple[float, float]:
        if self.width is None:
            return None
        diffx = self.bounds[2] - self.bounds[0]
        diffy = self.bounds[3] - self.bounds[1]
        resx = diffx / self.width
        resy = diffy / self.height
        return resx, resy

    @property
    def height(self):
        if self.array is None:
            try:
                return self._height
            except AttributeError:
                return None
        i = 1 if len(self.array.shape) == 3 else 0
        return self.array.shape[i]

    @property
    def width(self):
        if self.array is None:
            try:
                return self._width
            except AttributeError:
                return None
        i = 2 if len(self.array.shape) == 3 else 1
        return self.array.shape[i]

    @property
    def count(self):
        if self.array is not None:
            if len(self.array.shape) == 3:
                return self.array.shape[0]
            if len(self.array.shape) == 2:
                return 1
        if not hasattr(self._band_index, "__iter__"):
            return 1
        return len(self._band_index)

    @property
    def shape(self):
        """Shape that is consistent with the array, whether it is loaded or not."""
        if self.array is not None:
            return self.array.shape
        if self._band_index is None or hasattr(self._band_index, "__iter__"):
            return self.count, self.width, self.height
        return self.width, self.height

    @property
    def transform(self) -> Affine:
        return rasterio.transform.from_bounds(*self.bounds, self.width, self.height)

    @property
    def bounds(self) -> tuple[float, float, float, float] | None:
        if not hasattr(self, "_bounds"):
            return None
        return to_bbox(self._bounds)

    @property
    def total_bounds(self) -> tuple[float, float, float, float]:
        return self.bounds

    @classmethod
    def has_nessecary_attrs(cls, dict_like):
        """Check if Raster init got enough kwargs to not need to read src."""
        try:
            cls.validate_dict(dict_like)
            return True
        except AttributeError:
            return False

    def band_index_tuple(self) -> tuple[int, ...]:
        if len(self.shape) == 2:
            return (1,)
        return tuple(i + 1 for i in range(self.shape[0]))

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, ValueError, IndexError):
            return default

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
        if self.array is None and other.array is not None:
            return False
        if self.array is not None and other.array is None:
            return False

        for method in dir(self):
            if not is_property(self, method):
                continue
            if self[method] != other[method]:
                return False

        return np.array_equal(self.array, other.array)

    @staticmethod
    def _plot_2d(array, mask=None) -> None:
        ax = plt.axes()
        ax.imshow(array)
        ax.axis("off")
        plt.show()
        plt.close()

    def check_for_array(self, text=""):
        mess = "Arrays are not loaded. " + text
        if self.array is None:
            raise ValueError(mess)

    def _add_band_index_from_array(self, band_index):
        if band_index is not None:
            return band_index
        elif len(self.array.shape) == 3:
            return tuple(x + 1 for x in range(len(self.array)))
        elif len(self.array.shape) == 2:
            return 1
        else:
            raise ValueError

    def _add_meta_from_src(self, src):
        if not hasattr(self, "_bounds"):
            self._bounds = tuple(src.bounds)
        self._width = src.width
        self._height = src.height

        if not hasattr(self, "_band_index") or self._band_index is None:
            self._band_index = src.indexes

        if not hasattr(self, "nodata") or self.nodata is None:
            self.nodata = src.nodata

        try:
            self._crs = pyproj.CRS(src.crs)
        except pyproj.exceptions.CRSError:
            self._crs = None

    def _load_warp_file(self) -> DatasetReader:
        """(from Torchgeo). Load and warp a file to the correct CRS and resolution.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        with opener(self.path) as file:
            src = rasterio.open(file)

        # Only warp if necessary
        if src.crs != self.crs:
            vrt = WarpedVRT(src, crs=self.crs)
            src.close()
            return vrt
        return src

    def _read_tif(self, **kwargs) -> None:
        def _read(self, src):
            self._add_meta_from_src(src)
            out_shape = self.get_shape_from_res(kwargs.pop("res", None))

            if hasattr(self, "_warped_crs"):
                src = WarpedVRT(src, crs=self.crs)

            self.array = src.read(
                out_shape=out_shape,
                **self.read_kwargs(kwargs),
            )
            if self._dtype:
                self = self.astype(self._dtype)
            else:
                self = self.as_minimum_dtype()

        with opener(self.path) as file:
            with rasterio.open(file) as src:
                _read(self, src)

    def _read_with_mask(self, mask, boundless, **kwargs):
        kwargs = {"mask": mask, "boundless": boundless} | kwargs

        def _read(self, src, mask, boundless, **kwargs):
            # to non-warped crs
            # mask = mask.to_crs(self._crs)

            self._add_meta_from_src(src)
            transform = self.transform
            window = rasterio.windows.from_bounds(*to_bbox(mask), transform=transform)
            kwargs = {
                "window": window,
                "boundless": boundless,
                **self.read_kwargs(kwargs),
            }
            if hasattr(self, "_warped_crs"):
                src = WarpedVRT(src, crs=self.crs)
                kwargs.pop("boundless")
            self.array = src.read(**kwargs)

            self._bounds = src.window_bounds(window=window)

            if not np.size(self.array):
                return

            if self._dtype:
                self = self.astype(self._dtype)
            else:
                self = self.as_minimum_dtype()

        if self.array is not None:
            with memfile_from_array(self.array, **self.profile) as src:
                _read(self, src, **kwargs)
            return

        with opener(self.path) as file:
            with rasterio.open(file, **self.profile) as src:
                _read(self, src, **kwargs)

    def get_shape_from_res(self, res):
        if res is None:
            return None
        if hasattr(res, "__iter__") and len(res) == 2:
            res, res = res
        diffx = self.bounds[2] - self.bounds[0]
        diffy = self.bounds[3] - self.bounds[1]
        width = int(diffx / res)
        height = int(diffy / res)
        if hasattr(self.band_index, "__iter__"):
            return len(self.band_index), width, height
        return width, height

    def _write(self, dst, window):
        if np.ma.is_masked(self.array):
            if len(self.array.shape) == 2:
                return dst.write(
                    self.array.filled(self.nodata), indexes=1, window=window
                )

            for i in range(len(self.band_index_tuple())):
                dst.write(
                    self.array[i].filled(self.nodata),
                    indexes=i + 1,
                    window=window,
                )

        else:
            if len(self.array.shape) == 2:
                return dst.write(self.array, indexes=1, window=window)

            for i, idx in enumerate(self.band_index_tuple()):
                dst.write(self.array[i], indexes=idx, window=window)

    def _get_band_index(self, band_index):
        if isinstance(band_index, numbers.Number):
            return int(band_index)
        if band_index is None:
            return None
        try:
            return tuple(int(x) for x in band_index)
        except Exception as e:
            raise TypeError(
                "band_index should be an integer or an iterable of integers."
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
    def _to_geojson(gdf: GeoDataFrame) -> list[dict]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return [x["geometry"] for x in loads(gdf.to_json())["features"]]

    @staticmethod
    def _to_geojson_geom_val(gdf: GeoDataFrame, column: str) -> list[dict]:
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

    def min(self):
        if np.size(self.array):
            return np.min(self.array)
        return None

    def max(self):
        if np.size(self.array):
            return np.max(self.array)
        return None

    def __hash__(self):
        return hash(self._hash)

    def __eq__(self, other):
        if isinstance(other, Raster):
            return np.all(np.array_equal(self.array, other.array))
        return NotImplemented

    def __repr__(self) -> str:
        """The print representation."""
        shape = self.shape
        shp = ", ".join([str(x) for x in shape])
        try:
            res = int(self.res[0])
        except TypeError:
            res = None
        crs = str(self.crs_to_string(self.crs))
        raster_id = self.raster_id
        path = self.path
        return f"{self.__class__.__name__}(shape=({shp}), res={res}, raster_id={raster_id}, crs={crs}, path={path})"

    def __setattr__(self, __name: str, __value) -> None:
        return super().__setattr__(__name, __value)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __mul__(self, scalar):
        self.check_for_array()
        self.array = self.array * scalar
        return self

    def __add__(self, scalar):
        self.check_for_array()
        self.array = self.array + scalar
        return self

    def __sub__(self, scalar):
        self.check_for_array()
        self.array = self.array - scalar
        return self

    def __truediv__(self, scalar):
        self.check_for_array()
        self.array = self.array / scalar
        return self

    def __floordiv__(self, scalar):
        self.check_for_array()
        self.array = self.array // scalar
        return self

    def __pow__(self, exponent):
        self.check_for_array()
        self.array = self.array**exponent
        return self

    def _return_self_or_copy(self, array, copy: bool):
        if not copy:
            self.array = array
            return self
        else:
            copy = self.copy()
            copy.array = array
            return copy

import numbers
import uuid
import warnings
from copy import copy, deepcopy
from json import loads
from pathlib import Path
from typing import Callable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely
from affine import Affine
from geopandas import GeoDataFrame, GeoSeries
from pandas.api.types import is_dict_like, is_list_like
from rasterio import features
from rasterio.mask import mask as rast_mask
from rasterio.warp import reproject
from shapely import Geometry, box
from shapely.geometry import shape

from ..geopandas_tools.bounds import is_bbox_like
from ..geopandas_tools.general import is_wkt
from ..geopandas_tools.to_geodataframe import to_gdf


class RasterHasChangedError(ValueError):
    def __init__(self, method: str):
        self.method = method

    def __str__(self):
        return (
            f"{self.method} requires reading of tif files, but the "
            "current file paths are outdated. "
            "Use the to_tifs method to save new tif files. "
            "This also updates the file paths of the rasters."
        )


class RasterBase:
    @staticmethod
    def _crs_to_string(crs):
        if crs is None:
            return "None"
        crs = pyproj.CRS(crs)
        return str(crs.to_json_dict()["name"])


class Raster(RasterBase):
    """For reading, writing and working with rasters.

    Raster instances should be created with the methods 'from_path', 'from_array' or
    'from_gdf'.


    Examples
    --------

    Read tif file.

    >>> path = 'https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/dtm_10.tif'
    >>> raster = sg.Raster.from_path(path)
    >>> raster
    Raster(shape=(1, 201, 201), res=10, crs=ETRS89 / UTM zone 33N (N-E), path=https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/dtm_10.tif)

    Load the entire image as an numpy ndarray.
    Operations are done in place to save memory.
    The array is stored in the array attribute.

    >>> r.load()
    >>> r.array[r.array < 0] = 0
    >>> r.array
    [[[  0.    0.    0.  ... 158.4 155.6 152.6]
    [  0.    0.    0.  ... 158.  154.8 151.9]
    [  0.    0.    0.  ... 158.5 155.1 152.3]
    ...
    [  0.  150.2 150.6 ...   0.    0.    0. ]
    [  0.  149.9 150.1 ...   0.    0.    0. ]
    [  0.  149.2 149.5 ...   0.    0.    0. ]]]

    Save as tif file.

    >>> r.write("path/to/file.tif")

    Convert to GeoDataFrame.

    >>> gdf = r.to_gdf(column="elevation")
    >>> gdf
           elevation                                           geometry  band
    0            1.9  POLYGON ((-25665.000 6676005.000, -25665.000 6...     1
    1           11.0  POLYGON ((-25655.000 6676005.000, -25655.000 6...     1
    2           18.1  POLYGON ((-25645.000 6676005.000, -25645.000 6...     1
    3           15.8  POLYGON ((-25635.000 6676005.000, -25635.000 6...     1
    4           11.6  POLYGON ((-25625.000 6676005.000, -25625.000 6...     1
    ...          ...                                                ...   ...
    25096       13.4  POLYGON ((-24935.000 6674005.000, -24935.000 6...     1
    25097        9.4  POLYGON ((-24925.000 6674005.000, -24925.000 6...     1
    25098        5.3  POLYGON ((-24915.000 6674005.000, -24915.000 6...     1
    25099        2.3  POLYGON ((-24905.000 6674005.000, -24905.000 6...     1
    25100        0.1  POLYGON ((-24895.000 6674005.000, -24895.000 6...     1

    The image can also be clipped by a mask while loading.

    >>> small_circle = gdf.centroid.buffer(50)
    >>> raster = sg.Raster.from_path(path).clip(small_circle, crop=True)
    Raster(shape=(1, 11, 11), res=10, crs=ETRS89 / UTM zone 33N (N-E), path=https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/dtm_10.tif)

    Construct raster from GeoDataFrame.
    The arrays are put on top of each other in a 3 dimensional array.

    >>> r2 = r.from_gdf(gdf, columns=["elevation", "elevation_x2"], res=20)
    >>> r2
    Raster(shape=(2, 100, 100), res=20, crs=ETRS89 / UTM zone 33N (N-E), path=None)

    Calculate zonal statistics for each polygon in 'gdf'.

    >>> zonal = r.zonal(gdf, aggfunc=["sum", np.mean])
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
        *,
        path: str | None = None,
        array: np.ndarray | None = None,
        nodata: int | float | None = None,
        indexes: int | list[int] | None = None,
        name: str | None = None,
        meta: dict | None = None,
        dapla: bool = False,
        **kwargs,
    ):
        if path is not None and array is not None:
            raise ValueError("Cannot supply both 'path' and 'array'.")
        if path is None and array is None:
            raise ValueError("Must supply either 'path' or 'array'.")

        if not hasattr(self, "dapla"):
            self.dapla = dapla

        self.array = array
        self.path = path
        self.nodata = nodata
        self.indexes = indexes

        if path and not name:
            self.name = Path(path).stem
        else:
            self.name = name

        if path is not None:
            self._add_meta()
        else:
            if not isinstance(array, np.ndarray):
                raise TypeError

            self.array = array

            self._meta = {} if meta is None else meta

            if not isinstance(self._meta, (dict, pd.Series, rasterio.profiles.Profile)):
                raise TypeError

            self._meta = dict(self._meta)

            if "crs" not in self._meta and "crs" not in kwargs:
                raise TypeError("Must specify crs when constructing raster from array.")

            self._crs = self._meta.pop("crs", kwargs.pop("crs", None))
            self.transform = self._meta.pop("transform", kwargs.pop("transform", None))

            # bounds is a property, but can be used to create Affine transform
            bounds = self._meta.pop("bounds", kwargs.pop("bounds", None))

            if bounds and not self.transform:
                bounds: tuple = self._bounds_as_tuple(bounds)
                self.transform = rasterio.transform.from_bounds(
                    *bounds, self.width, self.height
                )
            if not self.transform:
                raise TypeError(
                    "Must specify either bounds or transform when constructing raster from array."
                )

            self.indexes = self._add_indexes_from_array(indexes)

        self._raster_has_changed = False
        self._hash = uuid.uuid4()

    @classmethod
    def from_path(
        cls,
        path: str,
        *,
        indexes: int | list[int] | None = None,
        nodata: float | int | None = None,
        dapla: bool = False,
        name: str | None = None,
    ):
        """Construct Raster from file path.

        Args:
            path: Path to a raster image file.
            indexes: Band indexes to read. Defaults to None, meaning all.
            nodata: The value to give to missing values. Defaults to the
                nodata value of the raster, if it has one.
            dapla: Whether to read in Dapla.
            name: Optional name to give the raster.

        Returns:
            A Raster instance.
        """
        return cls(path=path, indexes=indexes, nodata=nodata, dapla=dapla, name=name)

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        *,
        crs=None,
        transform: Affine | None = None,
        bounds: tuple | Geometry | None = None,
        meta: dict | None = None,
        name: str | None = None,
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
            meta: dictionary with at least the keys 'crs' and 'transform'/'bounds'.
                These can be fetched from an existing Raster in the meta attribute.
            name: Optional name to give the raster.
            **kwargs: Additional keyword arguments passed to the Raster initialiser.

        Returns:
            A Raster instance.
        """
        if array is None:
            raise TypeError("Must specify array.")

        return cls(
            array=array,
            crs=crs,
            transform=transform,
            bounds=bounds,
            meta=meta,
            name=name,
            **kwargs,
        )

    @classmethod
    def from_gdf(
        cls,
        gdf: GeoDataFrame,
        columns: str | list[str],
        res: int,
        **kwargs,
    ):
        """Construct Raster from a GeoDataFrame.

        Args:
            gdf: The GeoDataFrame.
            column: The column to be used as values for the array.
        """
        if not isinstance(gdf, GeoDataFrame):
            gdf = to_gdf(gdf)

        crs = gdf.crs or kwargs.get("crs")

        if crs is None:
            raise TypeError("Must specify crs if the object doesn't have crs.")

        if isinstance(res, numbers.Number):
            resx, resy = res, res
        elif not hasattr(res, "__iter__"):
            raise TypeError
        elif len(res) == 2:
            resx, resy = res
        else:
            raise TypeError

        minx, miny, maxx, maxy = gdf.total_bounds
        diffx = maxx - minx
        diffy = maxy - miny

        shape = int(diffx / resx), int(diffy / resy)
        transform = cls.get_transform_from_bounds(gdf, shape)

        if "meta" not in kwargs:
            meta = {}
        elif not isinstance(kwargs["meta"], dict):
            raise TypeError("'meta' must be dict.")
        else:
            meta = kwargs.pop("meta")

        meta = {
            "transform": transform,
            "crs": gdf.crs,
        } | meta

        rasterize_kwargs = [
            "fill",
            "out",
            "all_touched",
            "merge_alg",
            "default_value",
            "dtype",
        ]

        for kwarg in kwargs:
            if kwarg not in rasterize_kwargs:
                meta[kwarg] = kwargs.pop(kwarg)

        if isinstance(columns, str):
            array = features.rasterize(
                cls._to_geojson_geom_val(gdf, columns),
                out_shape=shape,
                transform=transform,
                **kwargs,
            )
            name = kwargs.get("name", columns)
        elif hasattr(columns, "__iter__"):
            array = []
            for col in columns:
                arr = features.rasterize(
                    cls._to_geojson_geom_val(gdf, col),
                    out_shape=shape,
                    transform=transform,
                    **kwargs,
                )
                array.append(arr)
            array = np.array(array)
            name = kwargs.get("name", None)

        return cls.from_array(array=array, meta=meta, name=name)

    def open(self):
        pass

    def __enter__(self):
        return self.open

    def __exit__(self, *args):
        if self.src is not None:
            self.src.close()

    @classmethod
    def get_transform_from_bounds(cls, obj, shape: tuple[float, ...]) -> Affine:
        minx, miny, maxx, maxy = cls._bounds_as_tuple(obj)
        width, height = shape
        return rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    def load(self, **kwargs):
        """Load the entire image as an np.array.

        The array is stored in the 'array' attribute
        of the Raster.

        Args:
            **kwargs: Keyword arguments passed to the rasterio read
                method.
        """
        if self._raster_has_changed or self._array_has_changed():
            raise RasterHasChangedError("load")

        kwargs = self._pop_from_dict(kwargs)

        self._from_path(mask=None, **kwargs)

        return self

    def clip(self, mask, **kwargs):
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
        if self._raster_has_changed or self._array_has_changed():
            raise RasterHasChangedError("clip")

        kwargs = self._pop_from_dict(kwargs)

        self._from_path(mask=mask, **kwargs)
        return self

    def zonal(
        self,
        polygons: GeoDataFrame,
        aggfunc: str | Callable | list[Callable | str],
        raster_calc_func: Callable | None = None,
        dropna: bool = True,
    ) -> GeoDataFrame:
        """Calculate zonal statistics in polygons.

        Args:
            polygons: A GeoDataFrame of polygon geometries.
            aggfunc: Function(s) of which to aggregate the values
                within each polygon.
            raster_calc_func: Optional calculation of the raster
                array before calculating the zonal statistics.
            dropna: If True (default), polygons with all missing
                values will be removed.

        Returns:
            A GeoDataFrame with the aggregate functions
        """
        if self._raster_has_changed or self._array_has_changed():
            raise RasterHasChangedError("zonal")
        if not pyproj.CRS(self.crs).equals(pyproj.CRS(polygons.crs)):
            raise ValueError("crs mismatch.")

        if isinstance(aggfunc, (str, Callable)):
            aggfunc = [aggfunc]

        def _get_func_name(f):
            if callable(f):
                return f.__name__
            return str(f).replace("np.", "").replace("numpy.", "")

        new_cols = [_get_func_name(f) for f in aggfunc]

        def _to_numpy_func(text):
            f = getattr(np, text, None)
            if f is not None:
                return f
            f = getattr(np.ndarray, text, None)
            if f is not None:
                return f
            raise ValueError(
                "aggfunc must be functions or " "strings of numpy functions or methods."
            )

        aggfunc = [f if callable(f) else _to_numpy_func(f) for f in aggfunc]

        aggregated = {}

        for i in polygons.index:
            poly = polygons.loc[[i]]

            box1 = shapely.box(*self.total_bounds)
            box2 = shapely.box(*poly.total_bounds)

            if not box1.intersects(box2) or box1.touches(box2):
                aggregated[i] = [pd.NA for _ in range(len(aggfunc))]
                continue

            if self.dapla:
                import dapla as dp

                fs = dp.FileClient.get_gcs_file_system()
                with fs.open(self.path, mode="rb") as file:
                    with rasterio.open(file) as src:
                        aggs = self._zonal_one_poly(
                            src, poly, aggfunc, raster_calc_func, i
                        )
            else:
                with rasterio.open(self.path) as src:
                    aggs = self._zonal_one_poly(src, poly, aggfunc, raster_calc_func, i)

            aggregated = aggregated | aggs

        out = gpd.GeoDataFrame(
            pd.DataFrame(
                polygons.index.map(aggregated).tolist(),
                columns=new_cols,
            ).astype("Float64"),
            geometry=polygons.geometry.values,
            crs=polygons.crs,
        )

        index_mapper = {i: idx for i, idx in enumerate(polygons.index)}
        out.index = out.index.map(index_mapper)

        if dropna:
            return out.loc[~out.isna().all(axis=1)]
        else:
            return out

    def _zonal_one_poly(self, src, poly, aggfunc, raster_calc_func, i):
        array, _ = rast_mask(
            dataset=src,
            shapes=self._to_geojson(poly),
            crop=True,
            filled=False,
        )

        flat_array = array.filled(np.nan).flatten()
        no_nans = flat_array[~np.isnan(flat_array)]

        if raster_calc_func:
            array_shape = no_nans.shape
            no_nans = raster_calc_func(no_nans)
            if array_shape != no_nans.shape:
                raise ValueError(
                    "raster_calc_func cannot change the shape of the array."
                )

        aggs: dict[float | int, list[float]] = {i: [f(no_nans) for f in aggfunc]}

        return aggs

    def write(self, path: str, window=None, colormap=None, **kwargs):
        """Write the raster as a single file.

        Multiband arrays will result in a multiband image file.

        Args:
            path: File path to write to.
            window: Optional window to clip the image to.
        """
        if self.array is None:
            raise AttributeError("The image hasn't been loaded yet.")
        kwargs = kwargs | {
            "driver": kwargs.get("driver", "GTiff"),
            "dtype": str(self.array.dtype),
            "crs": self.crs,
            "transform": self.transform,
            "nodata": self.nodata,
            "count": self.count,
            "height": self.height,
            "width": self.width,
            "compress": kwargs.get("compress", "LZW"),
        }

        if self.dapla:
            import dapla as dp

            fs = dp.FileClient.get_gcs_file_system()
            with fs.open(path, mode="rb") as file:
                args = (str(file), "w")
                with rasterio.open(*args, **kwargs) as dst:
                    self._write(dst, colormap, window)
        else:
            args = (str(path), "w")
            with rasterio.open(*args, **kwargs) as dst:
                self._write(dst, colormap, window)

    @staticmethod
    def _array_to_geojson(array: np.ndarray, transform: Affine):
        return [
            (value, shape(geom))
            for geom, value in features.shapes(array, transform=transform)
        ]

    def to_gdf(self, column: str | list[str] | None = None, mask=None) -> GeoDataFrame:
        """Create a GeoDataFrame from the raster.

        For multiband rasters, the bands are in separate rows with a "band" column
        value corresponding to the band indexes of the raster.

        Args:
            column: Name of resulting column(s) that holds the raster values.
                Can be a single string or an iterable with the same length as
                the number of raster bands.
            mask: Optional mask to clip the images by.

        Returns:
            A GeoDataFrame with a geometry column, a 'band' column and a
            one or more value columns.
        """
        if self.array is None:
            if mask is not None:
                self.clip(mask=mask)
            else:
                self.load()

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
            gdf["band"] = i + 1
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

        self._crs = crs
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
            self.load()

        self.array, self.transform = reproject(
            source=self.array,
            src_crs=self.crs,
            src_transform=self.transform,
            dst_crs=pyproj.CRS(crs),
            **kwargs,
        )

        self._crs = pyproj.CRS(crs)
        self._warped_crs = pyproj.CRS(crs)

        return self

    def plot(self, mask=None) -> None:
        """Plot the images. One image per band."""
        if len(self.shape) == 3:
            for arr in self.array:
                self._plot_2d(arr)

    @staticmethod
    def _plot_2d(array, mask=None) -> None:
        ax = plt.axes()
        ax.imshow(array)
        ax.axis("off")
        plt.show()
        plt.close()

    def _add_indexes_from_array(self, indexes):
        if indexes is not None:
            return indexes
        elif indexes is None and len(self.array.shape) == 3:
            return [x + 1 for x in range(len(self.array))]
        elif indexes is None and len(self.array.shape) == 2:
            return 1
        else:
            raise ValueError

    def _add_meta(self):
        if self.dapla:
            import dapla as dp

            fs = dp.FileClient.get_gcs_file_system()
            with fs.open(self.path, mode="rb") as file:
                with rasterio.open(file) as src:
                    self._add_meta_from_src(src)
        else:
            with rasterio.open(self.path) as src:
                self._add_meta_from_src(src)

    def _add_meta_from_src(self, src):
        self._shape = src.shape

        self.transform = src.transform

        self._height = src.height
        self._width = src.width
        self._bounds = src.bounds

        self._meta = dict(src.profile) | dict(src.meta)
        self.profile = src.profile

        if self.indexes is None:
            self.indexes = src.indexes

        if self.nodata is None:
            self.nodata = src.nodata

        self._crs = pyproj.CRS(src.crs)

        try:
            self.colormap = {}
            for i in src.indexes:
                self.colormap[i] = src.colormap(i)
        except ValueError:
            pass

    def _array_has_changed(self):
        if not hasattr(self, "array") or self.array is None:
            return False

        _stats = (
            self.array.shape,
            np.min(self.array),
            np.max(self.array),
            np.mean(self.array),
            np.std(self.array),
            np.sum(self.array),
        )

        if not hasattr(self, "_stats"):
            self._stats = _stats
            return False

        # if any(s1 != s2 for s1, s2 in zip(_stats, self._stats, strict=True)):
        if _stats == self._stats:
            return False

        return True

    @staticmethod
    def _to_2d_array_list(array: np.ndarray) -> list[np.ndarray]:
        if len(array.shape) == 2:
            return [array]
        elif len(array.shape) == 3:
            return [array for array in array]
        else:
            raise ValueError

    def _from_path(self, mask, **kwargs):
        if mask is not None:
            if not isinstance(mask, GeoDataFrame):
                mask = self._return_gdf(mask)
            self.array, self.transform = self._read_with_mask(mask=mask, **kwargs)
        else:
            self.array = self._read_tif(**kwargs)

    def _read_tif(self, **kwargs) -> np.ndarray:
        if self.dapla:
            import dapla as dp

            fs = dp.FileClient.get_gcs_file_system()
            with fs.open(self.path, mode="rb") as file:
                with rasterio.open(file) as src:
                    return src.read(
                        indexes=self.indexes,
                        **kwargs,
                    )
        else:
            with rasterio.open(self.path) as src:
                return src.read(indexes=self.indexes, **kwargs)

    def _read_with_mask(self, mask, **kwargs):
        if isinstance(mask, (GeoDataFrame, GeoSeries)):
            mask = self._to_geojson(mask)

        if self.dapla:
            import dapla as dp

            fs = dp.FileClient.get_gcs_file_system()
            with fs.open(self.path, mode="rb") as file:
                with rasterio.open(file) as src:
                    array, transform = rast_mask(
                        dataset=src,
                        shapes=mask,
                        indexes=self.indexes,
                        nodata=self.nodata,
                        **kwargs,
                    )
        else:
            with rasterio.open(self.path) as src:
                array, transform = rast_mask(
                    dataset=src,
                    shapes=mask,
                    nodata=self.nodata,
                    indexes=self.indexes,
                    **kwargs,
                )

        return array, transform

    def _write(self, dst, colormap, window):
        if colormap is None and hasattr(self, "colormap"):
            colormap = self.colormap

        if colormap is not None:
            for i in range(len(self.indexes_as_tuple())):
                dst.write_colormap(i + 1, {int(k): v for k, v in colormap.items()})

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

    def _max(self):
        if self.array is None:
            return np.nan
        return np.max(self.array)

    def _min(self):
        if self.array is None:
            return np.nan
        return np.min(self.array)

    def _mean(self):
        if self.array is None:
            return np.nan
        return np.mean(self.array)

    @property
    def meta(self):
        if not hasattr(self, "_meta"):
            self._meta = {}
        return self._meta | {
            "crs": self.crs,
            "res": self.res,
            "height": self.height,
            "width": self.width,
            "shape": self.shape,
            "bounds": self.bounds,
        }

    def get_coords(self) -> np.ndarray:
        xs = np.arange(self.bounds[0], self.bounds[2], self.res[0])
        ys = np.arange(self.bounds[1], self.bounds[3], self.res[1])

    @property
    def crs(self):
        if hasattr(self, "_warped_crs"):
            return self._warped_crs
        return self._crs

    @property
    def res(self) -> tuple[float, float]:
        diffx = self.bounds[2] - self.bounds[0]
        diffy = self.bounds[3] - self.bounds[1]
        resx = diffx / self.width
        resy = diffy / self.height
        return resx, resy

    @property
    def unary_union(self) -> shapely.geometry.Polygon:
        return shapely.box(*self.bounds)

    @property
    def height(self):
        if self.array is None:
            return self._height
        i = 2 if len(self.array.shape) == 3 else 1
        return self.array.shape[i]

    @property
    def width(self):
        if self.array is None:
            return self._width
        i = 1 if len(self.array.shape) == 3 else 0
        return self.array.shape[i]

    @property
    def count(self):
        if self.array is not None:
            if len(self.array.shape) == 3:
                return self.array.shape[0]
            if len(self.array.shape) == 2:
                return 1
        if not hasattr(self.indexes, "__iter__"):
            return 1
        return len(self.indexes)

    @property
    def shape(self):
        """Shape that is consistent with the array, whether it is loaded or not."""
        if self.array is not None:
            return self.array.shape
        if self.indexes is None or hasattr(self.indexes, "__iter__"):
            return self.count, self.width, self.height
        return self.width, self.height

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return rasterio.transform.array_bounds(self.height, self.width, self.transform)

    @property
    def total_bounds(self) -> tuple[float, float, float, float]:
        return self.bounds

    def indexes_as_tuple(self) -> tuple[int, ...]:
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

    def _pop_from_dict(self, the_dict) -> dict:
        for key in self.__dict__.keys():
            if key in the_dict:
                if hasattr(self, key) and isinstance(getattr(self, key), property):
                    continue
                self[key] = the_dict.pop(key)
        return the_dict

    def _return_gdf(self, obj) -> GeoDataFrame:
        if isinstance(obj, str) and not is_wkt(obj):
            return self._read_tif(obj)
        elif isinstance(obj, Raster):
            return obj.to_gdf()
        elif is_bbox_like(obj):
            bounds = self._bounds_as_tuple(obj)
            return to_gdf(shapely.box(*bounds), crs=self.crs)
        else:
            return to_gdf(obj, crs=self.crs)

    @staticmethod
    def _bounds_as_tuple(obj) -> tuple[float, float, float, float]:
        """Try to return 4-length tuple of bounds."""
        if (
            hasattr(obj, "__iter__")
            and len(obj) == 4
            and all(isinstance(x, numbers.Number) for x in obj)
        ):
            return obj
        if isinstance(obj, (GeoDataFrame, GeoSeries)):
            return obj.total_bounds
        if isinstance(obj, Geometry):
            return obj.bounds
        if is_dict_like(obj) and all(
            x in obj for x in ["minx", "miny", "maxx", "maxy"]
        ):
            try:
                minx = np.min(obj["minx"])
                miny = np.min(obj["miny"])
                maxx = np.max(obj["maxx"])
                maxy = np.max(obj["maxy"])
            except TypeError:
                minx = np.min(obj.minx)
                miny = np.min(obj.miny)
                maxx = np.max(obj.maxx)
                maxy = np.max(obj.maxy)
            return minx, miny, maxx, maxy
        if is_dict_like(obj) and all(
            x in obj for x in ["xmin", "ymin", "xmax", "ymax"]
        ):
            try:
                xmin = np.min(obj["xmin"])
                ymin = np.min(obj["ymin"])
                xmax = np.max(obj["xmax"])
                ymax = np.max(obj["ymax"])
            except TypeError:
                xmin = np.min(obj.xmin)
                ymin = np.min(obj.ymin)
                xmax = np.max(obj.xmax)
                ymax = np.max(obj.ymax)
            return xmin, ymin, xmax, ymax
        raise TypeError

    def __hash__(self):
        return hash(self._hash)

    def __eq__(self, other):
        if isinstance(other, Raster):
            return self.array == other.array
        return NotImplemented

    def __repr__(self) -> str:
        """The print representation."""
        shape = self.shape
        shp = ", ".join([str(x) for x in shape])
        res = int(self.res[0])
        crs = str(self._crs_to_string(self.crs))
        path = str(self.path)
        return f"{self.__class__.__name__}(shape=({shp}), res={res}, crs={crs}, path={path})"

    def __setattr__(self, __name: str, __value) -> None:
        return super().__setattr__(__name, __value)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __mul__(self, scalar):
        if self.array is None:
            raise ValueError
        self.array = self.array * scalar
        self._raster_has_changed = True
        return self

    def __add__(self, scalar):
        if self.array is None:
            raise ValueError
        self.array = self.array + scalar
        self._raster_has_changed = True
        return self

    def __sub__(self, scalar):
        if self.array is None:
            raise ValueError
        self.array = self.array - scalar
        self._raster_has_changed = True
        return self

    def __truediv__(self, scalar):
        if self.array is None:
            raise ValueError
        self.array = self.array / scalar
        self._raster_has_changed = True
        return self

    def __floordiv__(self, scalar):
        if self.array is None:
            raise ValueError
        self.array = self.array // scalar
        self._raster_has_changed = True
        return self

    def __pow__(self, exponent):
        if self.array is None:
            raise ValueError
        self.array = self.array**exponent
        self._raster_has_changed = True
        return self

    def _return_self_or_copy(self, array, copy: bool):
        if not copy:
            self.array = array
            return self
        else:
            copy = self.copy()
            copy.array = array
            return copy

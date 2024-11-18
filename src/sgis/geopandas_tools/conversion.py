import numbers
import re
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sized
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from numpy.typing import NDArray
from pandas.api.types import is_array_like
from pandas.api.types import is_dict_like
from pandas.api.types import is_list_like
from pyproj import CRS
from shapely import Geometry
from shapely import box
from shapely import wkb
from shapely import wkt
from shapely.errors import GEOSException
from shapely.geometry import Point
from shapely.geometry import shape
from shapely.ops import unary_union

try:
    import rasterio
    from rasterio import features
except ImportError:
    pass

try:
    from affine import Affine
except ImportError:

    class Affine:
        """Placeholder."""


try:
    from torchgeo.datasets.geo import RasterDataset
except ImportError:

    class RasterDataset:  # type: ignore
        """Placeholder."""


def crs_to_string(crs: Any) -> str:
    """Extract the string of a CRS-like object."""
    if crs is None:
        return "None"
    crs = pyproj.CRS(crs)
    crs_str = str(crs.to_json_dict()["name"])
    pattern = r"\d{4,5}"
    match = re.search(pattern, crs_str)
    return match.group() if match else crs_str


def to_geoseries(obj: Any, crs: Any | None = None) -> GeoSeries:
    """Convert an object to GeoSeries."""
    if crs is None:
        try:
            crs = obj.crs
        except AttributeError:
            pass

    try:
        if hasattr(obj.index, "values"):
            # pandas objects
            index = obj.index
        else:
            # list
            index = None
    except AttributeError:
        index = None

    try:
        # this works for geodataframe, geoseries and DataFrame with geometry column
        obj = obj.geometry.values
    except AttributeError:
        try:
            # if pandas series
            obj = obj.values
        except AttributeError:
            # geoseries will raise an Exception for non-geometry objects,
            # so we can safely pass here
            pass

    return GeoSeries(obj, index=index, crs=crs)


def to_shapely(obj: Any) -> Geometry:
    """Convert a geometry object or bounding box to a shapely Geometry."""
    if isinstance(obj, Geometry):
        return obj
    if not hasattr(obj, "__iter__"):
        raise TypeError(type(obj))
    try:
        return shapely.union_all(obj.geometry.values)
    except AttributeError:
        pass
    try:
        return Point(*obj)
    except TypeError:
        pass
    try:
        return box(*to_bbox(obj))
    except TypeError:
        pass
    try:
        return shapely.wkt.loads(obj)
    except TypeError:
        pass
    try:
        return shapely.wkb.loads(obj)
    except TypeError:
        pass
    raise TypeError(type(obj), obj)


def to_bbox(
    obj: GeoDataFrame | GeoSeries | Geometry | Collection | Mapping,
) -> tuple[float, float, float, float]:
    """Returns 4-length tuple of bounds if possible, else raises ValueError.

    Args:
        obj: Object to be converted to bounding box. Can be geopandas or shapely
            objects, iterables of exactly four numbers or dictionary like/class
            with a the keys/attributes "minx", "miny", "maxx", "maxy" or
            "xmin", "ymin", "xmax", "ymax".
    """
    if isinstance(obj, (GeoDataFrame, GeoSeries)):
        bounds = tuple(obj.total_bounds)
        assert isinstance(bounds, tuple)
        return bounds
    try:
        bounds = tuple(obj.bounds)
        assert isinstance(bounds, tuple)
        return bounds
    except Exception:
        pass

    try:
        minx = float(np.min(obj["minx"]))  # type: ignore [index]
        miny = float(np.min(obj["miny"]))  # type: ignore [index]
        maxx = float(np.max(obj["maxx"]))  # type: ignore [index]
        maxy = float(np.max(obj["maxy"]))  # type: ignore [index]
        return minx, miny, maxx, maxy
    except Exception:
        pass
    try:
        minx = float(np.min(obj.minx))  # type: ignore [union-attr]
        miny = float(np.min(obj.miny))  # type: ignore [union-attr]
        maxx = float(np.max(obj.maxx))  # type: ignore [union-attr]
        maxy = float(np.max(obj.maxy))  # type: ignore [union-attr]
        return minx, miny, maxx, maxy
    except Exception:
        pass

    try:
        minx = float(np.min(obj["west_longitude"]))  # type: ignore [index]
        miny = float(np.min(obj["south_latitude"]))  # type: ignore [index]
        maxx = float(np.max(obj["east_longitude"]))  # type: ignore [index]
        maxy = float(np.max(obj["north_latitude"]))  # type: ignore [index]
        return minx, miny, maxx, maxy
    except Exception:
        pass

    if hasattr(obj, "geometry"):
        try:
            return tuple(GeoSeries(obj["geometry"]).total_bounds)  # type: ignore [index]
        except Exception:
            return tuple(GeoSeries(obj.geometry).total_bounds)

    if (
        hasattr(obj, "__len__")
        and len(obj) == 4
        and all(isinstance(x, numbers.Number) for x in obj)
    ):
        return tuple(obj)

    try:
        of_length = f" of length {len(obj)}"
    except TypeError:
        of_length = ""
    raise TypeError(f"Cannot convert type {obj.__class__.__name__}{of_length} to bbox")


def from_4326(lon: float, lat: float, crs=25833) -> tuple[float, float]:
    """Get utm 33 N coordinates from lonlat (4326)."""
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", f"EPSG:{crs}", always_xy=True
    )
    return transformer.transform(lon, lat)


def to_4326(lon: float, lat: float, crs=25833) -> tuple[float, float]:
    """Get degree coordinates  33 N coordinates from lonlat (4326)."""
    transformer = pyproj.Transformer.from_crs(
        f"EPSG:{crs}", "EPSG:4326", always_xy=True
    )
    return transformer.transform(lon, lat)


def coordinate_array(
    gdf: GeoDataFrame | GeoSeries,
    strict: bool = False,
    include_z: bool = False,
) -> NDArray[np.float64]:
    """Creates a 2d ndarray of coordinates from point geometries.

    Args:
        gdf: GeoDataFrame or GeoSeries of point geometries.
        strict: If False (default), geometries without coordinates
            are given the value None.
        include_z: Whether to include z-coordinates. Defaults to False.

    Returns:
        np.ndarray of np.ndarrays of coordinates.

    Examples:
    ---------
    >>> import sgis as sg
    >>> points = sg.to_gdf(
    ...     [
    ...         (0, 1),
    ...         (1, 0),
    ...         (1, 1),
    ...         (0, 0),
    ...         (0.5, 0.5),
    ...         (0.5, 0.25),
    ...         (0.25, 0.25),
    ...         (0.75, 0.75),
    ...         (0.25, 0.75),
    ...         (0.75, 0.25),
    ...     ]
    ... )
    >>> points
                    geometry
    0  POINT (0.59376 0.92577)
    1  POINT (0.34075 0.91650)
    2  POINT (0.74841 0.10627)
    3  POINT (0.00966 0.87868)
    4  POINT (0.38046 0.87879)
    >>> sg.coordinate_array(points)
    array([[0.59376221, 0.92577159],
        [0.34074678, 0.91650446],
        [0.74840912, 0.10626954],
        [0.00965935, 0.87867915],
        [0.38045827, 0.87878816]])
    >>> sg.coordinate_array(points.geometry)
    array([[0.59376221, 0.92577159],
        [0.34074678, 0.91650446],
        [0.74840912, 0.10626954],
        [0.00965935, 0.87867915],
        [0.38045827, 0.87878816]])
    """
    try:
        return shapely.get_coordinates(gdf.geometry.values, include_z=include_z)
    except AttributeError:
        return shapely.get_coordinates(gdf, include_z=include_z)


def to_gdf(
    obj: (
        Geometry
        | str
        | bytes
        | list
        | tuple
        | dict
        | GeoSeries
        | pd.Series
        | pd.DataFrame
        | Iterator
    ),
    crs: str | tuple[str] | None = None,
    geometry: str | tuple[str] | int | None = None,
    **kwargs,
) -> GeoDataFrame:
    """Converts geometry-like objects to a GeoDataFrame.

    Constructs a GeoDataFrame from any geometry-like object
    (coordinates, wkt, wkb, dict, string) or any interable of such objects.

    NOTE: The function is meant for convenience in testing and exploring,
    not for production code since it introduces unnecessary overhead
    and potential errors.

    If obj is a DataFrame or dictionary, geometries can be in one column/key or 2-3
    if coordiantes are in x and x (and z) columns. The column/key "geometry" is used
    by default if it exists. The index and other columns/keys are preserved.

    Args:
        obj: the object to be converted to a GeoDataFrame.
        crs: if None (default), it uses the crs of the GeoSeries if GeoSeries
            is the input type. Otherwise, no crs is used.
        geometry: Name of column(s) or key(s) in obj with geometry-like values.
            If not specified, the key/column 'geometry' will be used if it
            exists. If multiple columns, can be given as e.g. "xyz" or ["x", "y"].
        **kwargs: additional keyword arguments taken by the GeoDataFrame constructor.

    Returns:
        A GeoDataFrame with one column, the geometry column.

    Examples:
    ---------
    >>> import sgis as sg
    >>> coords = (10, 60)
    >>> sg.to_gdf(coords, crs=4326)
                        geometry
    0  POINT (10.00000 60.00000)

    From wkt.

    >>> wkt = "POINT (10 60)"
    >>> sg.to_gdf(wkt, crs=4326)
                        geometry
    0  POINT (10.00000 60.00000)

    From DataFrame with x, y (optionally z) coordinate columns. Index and
    columns are preserved.

    >>> df = pd.DataFrame({"x": [10, 11], "y": [60, 59]}, index=[1,3])
        x   y
    1  10  60
    3  11  59
    >>> gdf = sg.to_gdf(df, geometry=["x", "y"], crs=4326)
    >>> gdf
        x   y                   geometry
    1  10  60  POINT (10.00000 60.00000)
    3  11  59  POINT (11.00000 59.00000)

    For DataFrame/dict with a geometry-like column named "geometry". If the column has
    another name, it must be set with the geometry parameter.

    >>> df = pd.DataFrame({"col": [1, 2], "geometry": ["point (10 60)", (11, 59)]})
    >>> df
       col       geometry
    0    1  point (10 60)
    1    2       (11, 59)
    >>> gdf = sg.to_gdf(df, crs=4326)
    >>> gdf
       col                   geometry
    0    1  POINT (10.00000 60.00000)
    1    2  POINT (11.00000 59.00000)

    From Series.

    >>> series = Series({1: (10, 60), 3: (11, 59)})
    >>> sg.to_gdf(series)
                        geometry
    1  POINT (10.00000 60.00000)
    3  POINT (11.00000 59.00000)

    Multiple coordinates will be converted to points, unless a line or polygon geometry
    is constructed beforehand.

    >>> coordslist = [(10, 60), (11, 59)]
    >>> sg.to_gdf(coordslist, crs=4326)
                        geometry
    0  POINT (10.00000 60.00000)
    1  POINT (11.00000 59.00000)

    >>> from shapely.geometry import LineString
    >>> sg.to_gdf(LineString(coordslist), crs=4326)
                                                geometry
    0  LINESTRING (10.00000 60.00000, 11.00000 59.00000)

    From 2 or 3 dimensional array.

    >>> arr = np.random.randint(100, size=(5, 3))
    >>> sg.to_gdf(arr)
                             geometry
    0  POINT Z (82.000 88.000 82.000)
    1  POINT Z (70.000 92.000 20.000)
    2   POINT Z (91.000 34.000 3.000)
    3   POINT Z (1.000 50.000 77.000)
    4  POINT Z (58.000 49.000 46.000)
    """
    if isinstance(obj, GeoDataFrame):
        if not crs:
            return obj
        if not obj.crs:
            return obj.set_crs(crs)
        return obj.to_crs(crs)

    if obj is None:
        raise TypeError("Cannot convert NoneType to GeoDataFrame.")

    if isinstance(obj, GeoSeries):
        geom_col = geometry or "geometry"
        return _geoseries_to_gdf(obj, geom_col, crs, **kwargs)

    if crs is None:
        try:
            crs = obj.crs  # type: ignore
        except AttributeError:
            try:
                matches = re.search(r"SRID=(\d+);", obj)  # type: ignore
            except TypeError:
                try:
                    matches = re.search(r"SRID=(\d+);", obj[0])  # type: ignore
                except Exception:
                    pass
            try:
                crs = CRS(
                    int(
                        "".join(
                            x
                            for x in matches.group(0)  # type:ignore
                            if x.isnumeric()
                        )
                    )
                )
            except Exception:
                pass

    if isinstance(obj, RasterDataset):
        # read the entire dataset
        obj = obj[obj.bounds]
        crs = obj["crs"]
        array = np.array(obj["image"])
        transform = get_transform_from_bounds(obj["bbox"], shape=array.shape)
        return gpd.GeoDataFrame(
            pd.DataFrame(
                _array_to_geojson(array, transform),
                columns=["value", "geometry"],
            ),
            geometry="geometry",
            crs=crs,
        )

    if is_array_like(geometry) and len(geometry) == len(obj):  # type: ignore
        geometry = GeoSeries(
            _make_one_shapely_geom(g) for g in geometry if g is not None  # type: ignore
        )
        return GeoDataFrame(obj, geometry=geometry, crs=crs, **kwargs)

    geom_col: str = _find_geometry_column(obj, geometry)  # type: ignore[no-redef]
    index = kwargs.pop("index", None)

    # get done with iterators that would get consumed by 'all' later
    if isinstance(obj, Iterator) and not isinstance(obj, Sized):
        obj = list(obj)
        # obj = GeoSeries(
        #     (_make_one_shapely_geom(g) for g in obj if g is not None), index=index
        # )
        # return GeoDataFrame({geom_col: obj}, geometry=geom_col, crs=crs, **kwargs)

    if hasattr(obj, "__len__") and not len(obj):
        return GeoDataFrame({"geometry": []}, crs=crs)

    crs = crs or get_crs_from_dict(obj)

    if not is_dict_like(obj):
        if is_bbox_like(obj):
            obj = GeoSeries(shapely.box(*obj), index=index)
            return GeoDataFrame({geom_col: obj}, geometry=geom_col, crs=crs, **kwargs)
        elif all(hasattr(obj, attr) for attr in ["minx", "miny", "maxx", "maxy"]):
            obj = GeoSeries(shapely.box(*to_bbox(obj)), index=index)
            return GeoDataFrame({geom_col: obj}, geometry=geom_col, crs=crs, **kwargs)
        if is_nested_geojson(obj):
            # crs = crs or get_crs_from_dict(obj)
            obj = pd.concat(
                (GeoSeries(_from_json(g)) for g in obj if g is not None),  # type: ignore
                ignore_index=True,
            )
            if index is not None:
                obj.index = index
            return GeoDataFrame({geom_col: obj}, geometry=geom_col, crs=crs, **kwargs)
        else:
            if isinstance(obj, str):
                if (
                    obj.replace(" ", "")
                    .replace(",", "")
                    .replace(".", "")
                    .replace("(", "")
                    .replace(")", "")
                    .isnumeric()
                ):
                    # string of coordinates
                    obj = eval(obj)
                    obj = GeoSeries([_make_one_shapely_geom(obj)], index=index)
                else:
                    # wkt
                    obj = GeoSeries([_make_one_shapely_geom(obj)], index=index)
            else:
                # list etc.
                obj = GeoSeries(_make_shapely_geoms(obj), index=index)
            return GeoDataFrame(
                {geom_col: obj}, geometry=geom_col, index=index, crs=crs, **kwargs
            )

    # now we have dict, Series or DataFrame

    obj = obj.copy()  # type: ignore [union-attr]

    # preserve Series/DataFrame index
    index = obj.index if hasattr(obj, "index") and index is None else index

    if geom_col in obj.keys():
        if isinstance(obj, pd.DataFrame):
            notna = obj[geom_col].notna()
            obj.loc[notna, geom_col] = list(
                _make_shapely_geoms(obj.loc[notna, geom_col])
            )
            obj[geom_col] = GeoSeries(obj[geom_col])
            return GeoDataFrame(obj, geometry=geom_col, crs=crs, **kwargs)
        if isinstance(obj[geom_col], Geometry):
            return GeoDataFrame(
                dict(obj), geometry=geom_col, crs=crs, index=[0], **kwargs
            )
        if not hasattr(obj[geom_col], "__iter__") or len(obj[geom_col]) == 1:
            obj[geom_col] = _make_shapely_geoms(obj[geom_col])
            return GeoDataFrame(
                dict(obj), geometry=geom_col, crs=crs, index=index, **kwargs
            )
        obj[geom_col] = GeoSeries(_make_shapely_geoms(obj[geom_col]), index=index)
        return GeoDataFrame(dict(obj), geometry=geom_col, crs=crs, **kwargs)

    if geometry and all(g in obj for g in geometry):  # type: ignore [union-attr]
        obj[geom_col] = _geoseries_from_xyz(obj, geometry, index=index)
        return GeoDataFrame(obj, geometry=geom_col, crs=crs, **kwargs)

    if len(obj.keys()) == 1:
        key = next(iter(obj.keys()))
        if isinstance(obj, dict):
            geoseries = GeoSeries(
                _make_shapely_geoms(next(iter(obj.values()))), index=index
            )
        elif isinstance(obj, pd.Series):
            geoseries = GeoSeries(_make_shapely_geoms(obj), index=index)
        else:
            geoseries = GeoSeries(_make_shapely_geoms(obj.iloc[:, 0]), index=index)
        return GeoDataFrame({key: geoseries}, geometry=key, crs=crs, **kwargs)

    if geometry and geom_col not in obj or isinstance(obj, pd.DataFrame):
        raise ValueError("Cannot find geometry column(s)", geometry)

    # geojson, __geo_interface__
    if (
        isinstance(obj, dict)
        and sum(key in obj for key in ["type", "coordinates", "features"]) >= 2
    ):
        if "geometry" in obj:
            geometry = "geometry"

        # crs = crs or get_crs_from_dict(obj)
        obj = GeoSeries(_from_json(obj), index=index)
        return GeoDataFrame({geom_col: obj}, geometry=geom_col, crs=crs, **kwargs)

    try:
        geoseries = _series_like_to_geoseries(obj, index=index)
    except ValueError:
        geoseries = _series_like_to_geoseries(obj.dropna(), index=obj.dropna().index)
    return GeoDataFrame(geometry=geoseries, crs=crs, **kwargs)


def _array_to_geojson(
    array: np.ndarray, transform: Affine
) -> list[tuple[dict, Geometry]]:
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


def _make_shapely_geoms(obj: Any) -> Geometry | Any:
    if _is_one_geometry(obj):
        return _make_one_shapely_geom(obj)
    if isinstance(obj, dict) and "coordinates" in obj:
        return _from_json(obj)
    return (_make_one_shapely_geom(g) for g in obj)


def is_bbox_like(obj: Any) -> bool:
    """Returns True if the object is an iterable of 4 numbers.

    Args:
        obj: Any object.
    """
    if (
        hasattr(obj, "__len__")
        and len(obj) == 4
        and hasattr(obj, "__iter__")
        and all(isinstance(x, numbers.Number) for x in obj)
    ):
        return True
    return False


def is_nested_geojson(obj: Any) -> bool:
    """Returns True if the object is an iterable of all dicts.

    Args:
        obj: Any object.
    """
    if hasattr(obj, "__iter__") and all(isinstance(g, dict) for g in obj):
        return True
    return False


def get_crs_from_dict(obj: Any) -> CRS | None | Any:
    """Try to extract the 'crs' attribute of the object or an object in the object."""
    if (
        not hasattr(obj, "__iter__")
        or not is_dict_like(obj)
        and not is_dict_like(obj[0])
    ):
        return None

    if not is_dict_like(obj) and is_dict_like(obj[0]):
        crss = list({get_crs_from_dict(g) for g in obj})
        return crss[0] if len(crss) == 1 else None

    if "properties" in obj:
        return get_crs_from_dict(obj["properties"])

    if "crs" in obj:
        obj = obj["crs"]
        while is_dict_like(obj):
            if "properties" in obj:
                obj = obj["properties"]
            elif "name" in obj:
                obj = obj["name"]
            else:
                return None
        return obj

    return None


def _from_json(obj: dict | list[dict]) -> Geometry:
    if not isinstance(obj, dict) and isinstance(obj[0], dict):
        return [_from_json(g) for g in obj]
    if "geometry" in obj:
        return _from_json(obj["geometry"])
    if "features" in obj:
        return _from_json(obj["features"])
    coords = obj["coordinates"]
    constructor: Callable = eval("shapely.geometry." + obj.get("type", Point))
    try:
        return constructor(coords)
    except TypeError:
        while len(coords) == 1:
            coords = coords[0]
        return constructor(coords)


def _series_like_to_geoseries(obj: Iterable, index: Iterable) -> GeoSeries:
    if index is None:
        index = obj.keys()
    if isinstance(obj, dict):
        return GeoSeries(_make_shapely_geoms(list(obj.values())), index=index)
    else:
        return GeoSeries(_make_shapely_geoms(obj.values), index=index)


def _geoseries_to_gdf(
    obj: GeoSeries, geometry: str | GeoSeries | Iterable, crs: CRS | Any, **kwargs
) -> GeoDataFrame:
    if not crs:
        crs = obj.crs
    else:
        obj = obj.to_crs(crs) if obj.crs else obj.set_crs(crs)

    return GeoDataFrame({geometry: obj}, geometry=geometry, crs=crs, **kwargs)


def _find_geometry_column(obj: Any, geometry: GeoSeries | Iterable | None) -> str:
    if geometry is None:
        return "geometry"

    # dict key
    if not is_list_like(geometry) and geometry in obj:
        return geometry

    # nested dict key
    if len(geometry) == 1 and geometry[0] in obj:
        return geometry[0]

    if len(geometry) in {2, 3}:
        return "geometry"

    raise ValueError(
        "geometry should be a geometry column or x, y (z) coordinate columns."
    )


def _geoseries_from_xyz(
    obj: Any,
    geometry: Iterable[float, float] | Iterable[float, float, float],
    index: Iterable | None,
) -> GeoSeries:
    """Make geoseries from the geometry column or columns (x y (z))."""
    if len(geometry) == 2:
        x, y = geometry
        z = None

    elif len(geometry) == 3:
        x, y, z = geometry
        z = obj[z]

    else:
        raise ValueError(
            "geometry should be a geometry column or x, y (z) coordinate columns."
        )

    return gpd.GeoSeries.from_xy(x=obj[x], y=obj[y], z=z, index=index)


def _is_one_geometry(obj: Any) -> bool:
    if (
        isinstance(obj, (str, bytes, Geometry))  # type: ignore [unreachable]
        or all(isinstance(i, numbers.Number) for i in obj)
        or not hasattr(obj, "__iter__")
    ):
        return True
    return False  # type: ignore [unreachable]


def _make_one_shapely_geom(obj: Any) -> Geometry:
    """Create shapely geometry from wkt, wkb or coordinate tuple.

    Works recursively if the object is a nested iterable.
    """
    if isinstance(obj, str):
        try:
            return wkt.loads(obj)
        except GEOSException:
            if obj.startswith("geography"):
                matches = re.search(r"SRID=(\d+);", obj)
                srid = matches.group(0)
                _, _wkt = obj.split(srid)
                return wkt.loads(_wkt)

    if isinstance(obj, bytes):
        return wkb.loads(obj)

    if isinstance(obj, Geometry):
        return obj

    if not hasattr(obj, "__iter__"):
        raise ValueError(
            f"Couldn't create shapely geometry from {obj} of type {type(obj)}"
        )

    if isinstance(obj, GeoSeries):
        raise TypeError(
            "to_gdf doesn't accept iterable of GeoSeries. Instead use: "
            "pd.concat(to_gdf(obj) for obj in geoseries_iterable)"
        )

    if not any(isinstance(g, numbers.Number) for g in obj):
        # we're likely dealing with a nested iterable, so let's
        # recursively dig down to the coords/wkt/wkb
        if len(obj) == 2 or len(obj) == 3:
            try:
                obj = [float(g) for g in obj]
                return Point(obj)
            except Exception:
                pass
        return unary_union([_make_one_shapely_geom(g) for g in obj])

    elif len(obj) == 2 or len(obj) == 3:
        return Point(obj)

    elif len(obj) == 4:
        return box(*obj)
    else:
        raise ValueError(
            "If 'geom' is an iterable, each item should consist of "
            "wkt, wkb or (x, y (z) or bbox). Got ",
            obj,
        )

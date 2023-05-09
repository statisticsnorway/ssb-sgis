import numbers
from collections.abc import Iterator, Sized

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from pandas.api.types import is_array_like, is_dict_like, is_list_like
from shapely import Geometry, wkb, wkt
from shapely.geometry import Point
from shapely.ops import unary_union


def to_gdf(
    geom: Geometry
    | str
    | bytes
    | list
    | tuple
    | dict
    | GeoSeries
    | pd.Series
    | pd.DataFrame
    | Iterator,
    crs: str | tuple[str] | None = None,
    geometry: str | tuple[str] | int | None = None,
    **kwargs,
) -> GeoDataFrame:
    """Converts geometry-like objects to a GeoDataFrame.

    Constructs a GeoDataFrame from any geometry-like object (coordinates, wkt, wkb),
    or any interable of such objects.

    If geom is a DataFrame or dictionary, geometries can be in one column/key or 2-3
    if coordiantes are in x and x (and z) columns. The column/key "geometry" is used
    by default if it exists. The index and other columns/keys are preserved.

    Args:
        geom: the object to be converted to a GeoDataFrame.
        crs: if None (default), it uses the crs of the GeoSeries if GeoSeries
            is the input type. Otherwise, no crs is used.
        geometry: Name of column(s) or key(s) in geom with geometry-like values.
            If not specified, the key/column 'geometry' will be used if it
            exists. If multiple columns, can be given as e.g. "xyz" or ["x", "y"].
        **kwargs: additional keyword arguments taken by the GeoDataFrame constructor.

    Returns:
        A GeoDataFrame with one column, the geometry column.

    Raises:
        TypeError: If geom is a GeoDataFrame.

    Examples
    --------
    >>> from sgis import to_gdf
    >>> coords = (10, 60)
    >>> to_gdf(coords, crs=4326)
                        geometry
    0  POINT (10.00000 60.00000)

    From wkt.

    >>> wkt = "POINT (10 60)"
    >>> to_gdf(wkt, crs=4326)
                        geometry
    0  POINT (10.00000 60.00000)

    From DataFrame with x, y (optionally z) coordinate columns. Index and
    columns are preserved.

    >>> df = pd.DataFrame({"x": [10, 11], "y": [60, 59]}, index=[1,3])
        x   y
    1  10  60
    3  11  59
    >>> gdf = to_gdf(df, geometry=["x", "y"], crs=4326)
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
    >>> gdf = to_gdf(df, crs=4326)
    >>> gdf
       col                   geometry
    0    1  POINT (10.00000 60.00000)
    1    2  POINT (11.00000 59.00000)

    From Series.

    >>> series = Series({1: (10, 60), 3: (11, 59)})
    >>> to_gdf(series)
                        geometry
    1  POINT (10.00000 60.00000)
    3  POINT (11.00000 59.00000)

    Multiple coordinates will be converted to points, unless a line or polygon geometry
    is constructed beforehand.

    >>> coordslist = [(10, 60), (11, 59)]
    >>> to_gdf(coordslist, crs=4326)
                        geometry
    0  POINT (10.00000 60.00000)
    1  POINT (11.00000 59.00000)

    >>> from shapely.geometry import LineString
    >>> to_gdf(LineString(coordslist), crs=4326)
                                                geometry
    0  LINESTRING (10.00000 60.00000, 11.00000 59.00000)

    From 2 or 3 dimensional array.

    >>> arr = np.random.randint(100, size=(5, 3))
    >>> to_gdf(arr)
                             geometry
    0  POINT Z (82.000 88.000 82.000)
    1  POINT Z (70.000 92.000 20.000)
    2   POINT Z (91.000 34.000 3.000)
    3   POINT Z (1.000 50.000 77.000)
    4  POINT Z (58.000 49.000 46.000)
    """
    if isinstance(geom, GeoDataFrame):
        raise TypeError("'to_gdf' doesn't accept GeoDataFrames as input type.")

    if isinstance(geom, GeoSeries):
        geom_col = "geometry" if not geometry else geometry
        return _geoseries_to_gdf(geom, geom_col, crs, **kwargs)

    geom_col = _find_geometry_column(geom, geometry)
    index = kwargs.get("index", None)

    if is_array_like(geom_col):
        geometry = GeoSeries((_make_one_shapely_geom(g) for g in geometry), index=index)
        return GeoDataFrame(geom, geometry=geometry, crs=crs, **kwargs)

    # get done with the iterators that get consumed by 'all' statements
    if isinstance(geom, Iterator) and not isinstance(geom, Sized):
        geom = GeoSeries((_make_one_shapely_geom(g) for g in geom), index=index)
        return GeoDataFrame({geom_col: geom}, geometry=geom_col, crs=crs, **kwargs)

    if not is_dict_like(geom):
        geom = GeoSeries(_make_shapely_geoms(geom), index=index)
        return GeoDataFrame({geom_col: geom}, geometry=geom_col, crs=crs, **kwargs)

    # now we have dict, Series or DataFrame

    geom = geom.copy()

    # preserve Series/DataFrame index, overrides kwargs for now
    index = geom.index if hasattr(geom, "index") else kwargs.get("index", None)

    if geom_col in geom.keys():
        geom[geom_col] = GeoSeries(_make_shapely_geoms(geom[geom_col]), index=index)
        return GeoDataFrame(geom, geometry=geom_col, crs=crs, **kwargs)

    if geometry and all(g in geom for g in geometry):
        geom[geom_col] = _geoseries_from_xyz(geom, geometry, index=index)
        return GeoDataFrame(geom, geometry=geom_col, crs=crs, **kwargs)

    if len(geom.keys()) == 1:
        key = list(geom.keys())[0]
        if isinstance(geom, dict):
            geoseries = GeoSeries(
                _make_shapely_geoms(list(geom.values())[0]), index=index
            )
        else:
            geoseries = GeoSeries(_make_shapely_geoms(geom.iloc[:, 0]), index=index)
        return GeoDataFrame({key: geoseries}, geometry=key, crs=crs, **kwargs)

    if geometry and geom_col not in geom or isinstance(geom, pd.DataFrame):
        raise ValueError("Cannot find geometry column(s)", geometry)

    geoseries = _series_like_to_geoseries(geom, index=index)
    return GeoDataFrame(geometry=geoseries, crs=crs, **kwargs)


def _series_like_to_geoseries(geom, index):
    if index is None:
        index = geom.keys()
    if isinstance(geom, dict):
        return GeoSeries(_make_shapely_geoms(list(geom.values())), index=index)
    else:
        return GeoSeries(_make_shapely_geoms(geom.values), index=index)


def _geoseries_to_gdf(geom: GeoSeries, geometry, crs, **kwargs) -> GeoDataFrame:
    if not crs:
        crs = geom.crs
    else:
        geom = geom.to_crs(crs) if geom.crs else geom.set_crs(crs)

    return GeoDataFrame({geometry: geom}, geometry=geometry, crs=crs, **kwargs)


def _find_geometry_column(geom, geometry) -> str:
    if geometry is None:
        return "geometry"

    if not is_list_like(geometry) and geometry in geom:
        return geometry

    if len(geometry) == 1 and geometry[0] in geom:
        return geometry[0]

    if len(geometry) == 2 or len(geometry) == 3:
        return "geometry"

    if is_array_like(geometry) and len(geometry) == len(geom):
        return geometry

    raise ValueError(
        "geometry should be a geometry column or x, y (z) coordinate columns."
    )


def _geoseries_from_xyz(geom, geometry, index) -> GeoSeries:
    """Make geoseries from the geometry column or columns (x y (z))."""

    if len(geometry) == 2:
        x, y = geometry
        z = None

    elif len(geometry) == 3:
        x, y, z = geometry
        z = geom[z]

    else:
        raise ValueError(
            "geometry should be a geometry column or x, y (z) coordinate columns."
        )

    return gpd.GeoSeries.from_xy(x=geom[x], y=geom[y], z=z, index=index)


def _is_one_geometry(geom) -> bool:
    if (
        isinstance(geom, (str, bytes, Geometry))
        or all(isinstance(i, numbers.Number) for i in geom)
        or not hasattr(geom, "__iter__")
    ):
        return True
    return False


def _make_shapely_geoms(geom):
    if _is_one_geometry(geom):
        return _make_one_shapely_geom(geom)
    return (_make_one_shapely_geom(g) for g in geom)


def _make_one_shapely_geom(geom):
    """Create shapely geometry from wkt, wkb or coordinate tuple.

    Works recursively if the object is a nested iterable.
    """
    if isinstance(geom, str):
        return wkt.loads(geom)

    if isinstance(geom, bytes):
        return wkb.loads(geom)

    if isinstance(geom, Geometry):
        return geom

    if not hasattr(geom, "__iter__"):
        raise ValueError(
            f"Couldn't create shapely geometry from {geom} of type {type(geom)}"
        )

    if isinstance(geom, GeoSeries):
        raise TypeError(
            "to_gdf doesn't accept iterable of GeoSeries. Instead use: "
            "pd.concat(to_gdf(geom) for geom in geoseries_iterable)"
        )

    if not any(isinstance(g, numbers.Number) for g in geom):
        # we're likely dealing with a nested iterable, so let's
        # recursively dig down to the coords/wkt/wkb
        if len(geom) == 2 or len(geom) == 3:
            try:
                geom = [float(g) for g in geom]
                return Point(geom)
            except Exception:
                pass
        return unary_union([_make_one_shapely_geom(g) for g in geom])

    elif len(geom) == 2 or len(geom) == 3:
        return Point(geom)

    else:
        raise ValueError(
            "If 'geom' is an iterable, each item should consist of "
            "wkt, wkb or 2/3 coordinates (x, y, z). Got ",
            geom,
        )

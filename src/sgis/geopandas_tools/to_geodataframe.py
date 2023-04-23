import numbers
import warnings
from collections.abc import Iterator, Sized

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryDtype
from pandas.api.types import is_dict_like
from shapely import (
    Geometry,
    force_2d,
    get_exterior_ring,
    get_interior_ring,
    get_num_interior_rings,
    get_parts,
    wkb,
    wkt,
)
from shapely.geometry import LineString, Point
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
    geometry: str | None = None,
    **kwargs,
) -> GeoDataFrame:
    """Converts geometry-like objects to a GeoDataFrame.

    Constructs a GeoDataFrame from any geometry-like object, or an interable of such.
    Accepted types are string (wkt), byte (wkb), coordinate tuples, shapely geometries,
    GeoSeries, Series/DataFrame. The index/keys will be preserved if the input type is
    Series, DataFrame or dictionary.

    Args:
        geom: the object to be converted to a GeoDataFrame
        crs: if None (default), it uses the crs of the GeoSeries if GeoSeries
            is the input type. Otherwise, no crs is used.
        geometry: name of column(s) containing the geometry-like values. Can be
            ['x', 'y', 'z'] if coordinate columns, or e.g. 'geometry' if one column.
            The resulting geometry column will always be named 'geometry'.
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

    For DataFrame/dict with a geometry-like column, the geometry column can be
    speficied with the geometry parameter, which is set to "geometry" by default.

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

    From Series or Series-like dictionary.

    >>> d = {1: (10, 60), 3: (11, 59)}
    >>> to_gdf(d)
                        geometry
    1  POINT (10.00000 60.00000)
    3  POINT (11.00000 59.00000)

    >>> from pandas import Series
    >>> to_gdf(Series(d))
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
        return _geoseries_to_gdf(geom, crs, **kwargs)

    # now get done with the iterators that get consumed by 'all' statements
    if isinstance(geom, Iterator) and not isinstance(geom, Sized):
        geom = GeoSeries(_make_shapely_geom(x) for x in geom)
        return GeoDataFrame({"geometry": geom}, geometry="geometry", crs=crs, **kwargs)

    # dataframes and dicts with geometry/xyz key(s)
    geometry = "geometry" if not geometry else geometry
    if _is_df_like(geom, geometry):
        geom = geom.copy()
        geom = _make_geomcol_df_like(geom, geometry, index=kwargs.get("index"))
        if "geometry" in geom:
            return GeoDataFrame(geom, geometry="geometry", crs=crs, **kwargs)
        elif isinstance(geom, pd.DataFrame):
            raise ValueError(f"Cannot find 'geometry' column {geometry!r}")

    if is_dict_like(geom):
        if isinstance(geom, dict):
            geom = pd.Series(geom)
        if "index" not in kwargs:
            index_ = geom.index
        else:
            index_ = None
        geom = GeoSeries(_make_shapely_geom(x) for x in geom)
        gdf = GeoDataFrame({"geometry": geom}, geometry="geometry", crs=crs, **kwargs)
        if index_ is not None:
            gdf.index = index_
        return gdf

    # single geometry objects like wkt, wkb, shapely geometry or iterable of numbers
    if _is_one_geometry(geom):
        geom = GeoSeries(_make_shapely_geom(geom))
        return GeoDataFrame({"geometry": geom}, geometry="geometry", crs=crs, **kwargs)

        # single geometry objects like wkt, wkb, shapely geometry or iterable of numbers
    if hasattr(geom, "__iter__"):
        geom = GeoSeries(_make_shapely_geom(x) for x in geom)
        return GeoDataFrame({"geometry": geom}, geometry="geometry", crs=crs, **kwargs)

    raise TypeError(f"Got unexpected type {type(geom)}")


def _geoseries_to_gdf(geom: GeoSeries, crs, **kwargs) -> GeoDataFrame:
    if not crs:
        crs = geom.crs
    else:
        geom = geom.to_crs(crs) if geom.crs else geom.set_crs(crs)

    return GeoDataFrame({"geometry": geom}, geometry="geometry", crs=crs, **kwargs)


def _is_one_geometry(geom) -> bool:
    if (
        isinstance(geom, (str, bytes, Geometry))
        or all(isinstance(i, numbers.Number) for i in geom)
        or not hasattr(geom, "__iter__")
    ):
        return True
    return False


def _is_df_like(geom, geometry: str | None) -> bool:
    """Check if object is DataFrame-like and not Series-like.

    Series-like as in 1 dimension of geometry-like objects.

    Dictionaries can be both.
    """
    if not is_dict_like(geom):
        return False

    if isinstance(geom, pd.DataFrame):
        return True

    if isinstance(geom, pd.Series):
        return False

    # geometry is a hashable key
    if len(geometry) == 1 and geometry[0] in geom:
        print("hhhh000")
        return True

    # geometry is x, y (z)
    if len(geometry) == 2 or len(geometry) == 3 and all(g in geom for g in geometry):
        return True

    # geometry is non-hashable key
    if geometry in geom and not any(
        isinstance(g, numbers.Number) for g in geom[geometry]
    ):
        print("hhhh444", geom)
        assert isinstance(geometry, str)
        return True

    # 1 key and more than 1 geometry pair
    if len(geom.keys()) == 1 and not any(
        isinstance(g, numbers.Number) for g in list(geom.values())[0]
    ):
        print("hhhh555")
        return True

    print("hhhh666")

    return False


def _make_geomcol_df_like(
    geom: pd.DataFrame | dict,
    geometry: str | tuple[str],
    index: list | tuple | None,
) -> pd.DataFrame | dict:
    """Create GeoSeries column in DataFrame or dictionary that has a DataFrame structure
    rather than a Series structure.

    Use the same logic for DataFrame and dict, as long as the speficied 'geometry'
    value or values (if x, y, z coordinate columns) are keys in the dictionary. If
    not, nothing happens here, and the dict goes on to be treated as a Series where
    keys are index and values the geometries.
    """

    if not index and hasattr(geom, "index"):
        index = geom.index
    elif index and hasattr(geom, "index"):
        geom.index = index

    if len(geometry) == 1 and geometry[0] in geom:
        geometry = geometry[0]

    if (
        isinstance(geometry, (tuple, list))
        or isinstance(geometry, str)
        and all(g in geom for g in geometry)
    ):
        if len(geometry) == 2:
            x, y = geometry
            z = None
        elif len(geometry) == 3:
            x, y, z = geometry
            z = geom[z]
        else:
            raise ValueError(
                "geometry should be one geometry-like column or 2-3 x, y, z columns."
            )
        geom["geometry"] = gpd.GeoSeries.from_xy(x=geom[x], y=geom[y], z=z)
    elif geometry in geom:
        if not hasattr(geom[geometry], "__iter__"):
            geom[geometry] = [geom[geometry]]
        geom["geometry"] = GeoSeries(
            map(_make_shapely_geom, geom[geometry]), index=index
        )

    elif isinstance(geom, pd.DataFrame) and len(geom.columns) == 1:
        geometry = geom.columns[0]
        geom["geometry"] = GeoSeries(
            map(_make_shapely_geom, geom[geometry]), index=index
        )

    elif isinstance(geom, dict) and len(geom.keys()) == 1:
        if not any(isinstance(g, numbers.Number) for g in list(geom.values())[0]):
            geometry = next(iter(geom))
            geom["geometry"] = GeoSeries(
                map(_make_shapely_geom, geom[geometry]), index=index
            )
        else:
            geometry = list(geom.values())[0]
            geom["geometry"] = _make_shapely_geom(geometry)

    return geom


def _make_shapely_geom(geom):
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
        return unary_union([_make_shapely_geom(g) for g in geom])
    elif len(geom) == 2 or len(geom) == 3:
        return Point(geom)
    else:
        raise ValueError(
            "If 'geom' is an iterable, each item should consist of "
            "wkt, wkb or 2/3 coordinates (x, y, z)."
        )

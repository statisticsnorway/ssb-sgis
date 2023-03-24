"""Functions for polygon geometries."""

import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
from shapely import (
    Geometry,
    area,
    get_exterior_ring,
    get_interior_ring,
    get_num_interior_rings,
    get_parts,
    polygons,
)
from shapely.ops import unary_union

from ..helpers import unit_is_meters


def close_small_holes(
    gdf: GeoDataFrame | GeoSeries | Geometry,
    *,
    max_km2: int | float | None = None,
    max_m2: int | float | None = None,
    copy: bool = True,
) -> GeoDataFrame | GeoSeries | Geometry:
    """Closes holes in polygons if the area is less than the given maximum.

    It takes a GeoDataFrame, GeoSeries or shapely geometry of polygons object and
    fills the holes that are smaller than the specified area given in units of
    either square meters ('max_m2') or square kilometers ('max_km2').

    Args:
        gdf: GeoDataFrame, GeoSeries or shapely Geometry.
        max_km2: The maximum area in square kilometers.
        max_m2: The maximum area in square meters.
        copy: if True (default), the input GeoDataFrame or GeoSeries is copied.
            Defaults to True.

    Returns:
        A GeoDataFrame, GeoSeries or shapely Geometry with closed holes in the geometry
        column.

    Raises:
        ValueError: If the coordinate reference system of the GeoDataFrame is not in
            meter units.
        ValueError: If both 'max_m2' and 'max_km2' is given.

    Examples
    --------

    Let's create a circle with a hole in it.

    >>> from sgis import close_holes, buff
    >>> point = to_gdf([260000, 6650000], crs=25833)
    >>> point
                            geometry
    0  POINT (260000.000 6650000.000)
    >>> circle = buff(point, 1000)
    >>> small_circle = buff(point, 500)
    >>> circle_with_hole = circle.overlay(small_circle, how="difference")
    >>> circle_with_hole.area
    0    2.355807e+06
    dtype: float64

    Now we close the hole.

    >>> holes_closed = close_holes(circle_with_hole, max_km2=1)
    >>> holes_closed.area
    0    3.141076e+06
    dtype: float64

    The hole will not be closed if it is larger in square kilometers than 'max_km2'.

    >>> holes_closed = close_holes(
    ...     circle_with_hole,
    ...     max_km2=0.1
    ... )
    >>> holes_closed.area
    0    2.355807e+06
    dtype: float64
    """
    if not unit_is_meters(gdf):
        raise ValueError("The 'crs' unit has to be 'metre'.")

    if max_m2 and max_km2:
        raise ValueError("Can only specify one of 'max_km2' and 'max_m2'")

    if copy:
        gdf = gdf.copy()

    if isinstance(gdf, GeoDataFrame):
        gdf["geometry"] = gdf.geometry.map(
            lambda x: _close_small_holes_poly(x, max_km2=max_km2, max_m2=max_m2)
        )

    elif isinstance(gdf, gpd.GeoSeries):
        gdf = gdf.map(
            lambda x: _close_small_holes_poly(x, max_km2=max_km2, max_m2=max_m2)
        )
        gdf = gpd.GeoSeries(gdf)

    else:
        gdf = _close_small_holes_poly(gdf, max_km2=max_km2, max_m2=max_m2)

    return gdf


def close_holes(
    gdf: GeoDataFrame | GeoSeries | Geometry,
    *,
    copy: bool = True,
) -> GeoDataFrame | GeoSeries | Geometry:
    """Closes all holes in polygons.

    It takes a GeoDataFrame, GeoSeries or shapely geometry of polygons object and
    returns the outer circle.

    Args:
        gdf: GeoDataFrame, GeoSeries or shapely Geometry.
        copy: if True (default), the input GeoDataFrame or GeoSeries is copied.
            Defaults to True.

    Returns:
        A GeoDataFrame, GeoSeries or shapely Geometry with closed holes in the geometry
        column.

    Examples
    --------

    Let's create a circle with a hole in it.

    >>> from sgis import close_holes, buff
    >>> point = to_gdf([260000, 6650000], crs=25833)
    >>> point
                            geometry
    0  POINT (260000.000 6650000.000)
    >>> circle = buff(point, 1000)
    >>> small_circle = buff(point, 500)
    >>> circle_with_hole = circle.overlay(small_circle, how="difference")
    >>> circle_with_hole.area
    0    2.355807e+06
    dtype: float64

    Close the hole.

    >>> holes_closed = close_holes(circle_with_hole)
    >>> holes_closed.area
    0    3.141076e+06
    dtype: float64
    """
    if copy:
        gdf = gdf.copy()

    def close_holes_func(poly):
        return polygons(get_exterior_ring(get_parts(poly)))

    if isinstance(gdf, GeoDataFrame):
        gdf["geometry"] = close_holes_func(gdf.geometry)
        return gdf

    elif isinstance(gdf, gpd.GeoSeries):
        gdf = close_holes_func(gdf)
        return gdf

    else:
        gdf = close_holes_func(gdf)

    return gdf


def _close_small_holes_poly(poly, max_km2, max_m2):
    """Closes cmall holes within one shapely geometry of polygons."""

    if max_km2:
        max_m2 = max_km2 * 1_000_000

    # start with a list containing the polygon,
    # then append all holes smaller than 'max_km2' to the list.
    holes_closed = [poly]
    singlepart = get_parts(poly)
    for part in singlepart:
        n_interior_rings = get_num_interior_rings(part)

        if not (n_interior_rings):
            continue

        for n in range(n_interior_rings):
            hole = polygons(get_interior_ring(part, n))

            if area(hole) < max_m2:
                holes_closed.append(hole)

    return unary_union(holes_closed)

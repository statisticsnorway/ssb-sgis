"""Functions for polygon geometries."""

import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from shapely import (
    area,
    get_exterior_ring,
    get_interior_ring,
    get_num_interior_rings,
    get_parts,
    polygons,
)
from shapely.ops import unary_union


def close_small_holes(
    gdf: GeoDataFrame | GeoSeries,
    max_area: int | float,
    *,
    copy: bool = True,
) -> GeoDataFrame | GeoSeries:
    """Closes holes in polygons if the area is less than the given maximum.

    It takes a GeoDataFrame or GeoSeries of polygons and
    fills the holes that are smaller than the specified area given in units of
    either square meters ('max_m2') or square kilometers ('max_km2').

    Args:
        gdf: GeoDataFrame or GeoSeries of polygons.
        max_area: The maximum area in the unit of the GeoDataFrame's crs.
        copy: if True (default), the input GeoDataFrame or GeoSeries is copied.
            Defaults to True.

    Returns:
        A GeoDataFrame or GeoSeries of polygons with closed holes in the geometry
        column.

    Raises:
        ValueError: If the coordinate reference system of the GeoDataFrame is not in
            meter units.
        ValueError: If both 'max_m2' and 'max_km2' is given.

    Examples
    --------

    Let's create a circle with a hole in it.

    >>> from sgis import close_small_holes, buff, to_gdf
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

    Close holes smaller than 1 square kilometer (1 million square meters).

    >>> holes_closed = close_small_holes(circle_with_hole, max_area=1_000_000)
    >>> holes_closed.area
    0    3.141076e+06
    dtype: float64

    The hole will not be closed if it is larger.

    >>> holes_closed = close_small_holes(circle_with_hole, max_area=1_000)
    >>> holes_closed.area
    0    2.355807e+06
    dtype: float64
    """
    if copy:
        gdf = gdf.copy()

    if isinstance(gdf, GeoDataFrame):
        gdf["geometry"] = gdf.geometry.map(
            lambda x: _close_small_holes_poly(x, max_area)
        )
        return gdf

    elif isinstance(gdf, gpd.GeoSeries):
        return gdf.map(lambda x: _close_small_holes_poly(x, max_area))

    else:
        raise ValueError(
            f"'gdf' should be of type GeoDataFrame or GeoSeries. Got {type(gdf)}"
        )


def close_all_holes(
    gdf: GeoDataFrame | GeoSeries,
    *,
    copy: bool = True,
) -> GeoDataFrame | GeoSeries:
    """Closes all holes in polygons.

    It takes a GeoDataFrame or GeoSeries of polygons and
    returns the outer circle.

    Args:
        gdf: GeoDataFrame or GeoSeries of polygons.
        copy: if True (default), the input GeoDataFrame or GeoSeries is copied.
            Defaults to True.

    Returns:
        A GeoDataFrame or GeoSeries of polygons with closed holes in the geometry
        column.

    Examples
    --------
    Let's create a circle with a hole in it.

    >>> from sgis import close_all_holes, buff, to_gdf
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

    >>> holes_closed = close_all_holes(circle_with_hole)
    >>> holes_closed.area
    0    3.141076e+06
    dtype: float64
    """
    if copy:
        gdf = gdf.copy()

    def close_all_holes_func(poly):
        return unary_union(polygons(get_exterior_ring(get_parts(poly))))

    close_all_holes_func = np.vectorize(close_all_holes_func)

    if isinstance(gdf, GeoDataFrame):
        gdf["geometry"] = close_all_holes_func(gdf.geometry)
        return gdf

    elif isinstance(gdf, gpd.GeoSeries):
        return close_all_holes_func(gdf)

    else:
        raise ValueError(
            f"'gdf' should be of type GeoDataFrame or GeoSeries. Got {type(gdf)}"
        )


def _close_small_holes_poly(poly, max_area):
    """Closes cmall holes within one shapely geometry of polygons."""

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

            if area(hole) < max_area:
                holes_closed.append(hole)

    return unary_union(holes_closed)

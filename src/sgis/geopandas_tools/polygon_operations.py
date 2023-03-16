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


def close_holes(
    polygons: GeoDataFrame | GeoSeries | Geometry,
    max_km2: int | float | None = None,
    copy: bool = True,
) -> GeoDataFrame | GeoSeries | Geometry:
    """Closes holes in polygons, either all holes or holes smaller than 'max_km2'.

    It takes a GeoDataFrame, GeoSeries or shapely geometry of polygons object and
    returns the outer circle. Closes only holes smaller than 'max_km2' if speficied.
    km2 as in square kilometers.

    Args:
        polygons: GeoDataFrame, GeoSeries or shapely Geometry.
        max_km2: if None (default), all holes are closed.
            Otherwise, closes holes with an area below the specified number in
            square kilometers if the crs unit is in meters.
        copy: if True (default), the input GeoDataFrame or GeoSeries is copied.
            Defaults to True.

    Returns:
        A GeoDataFrame, GeoSeries or shapely Geometry with closed holes in the geometry
        column.

    Examples
    --------

    Let's create a circle with a hole in it.

    >>> from gis_utils import close_holes, buff
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

    >>> holes_closed = close_holes(circle_with_hole)
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
    if copy:
        polygons = polygons.copy()

    if isinstance(polygons, GeoDataFrame):
        polygons["geometry"] = polygons.geometry.map(
            lambda x: _close_holes_poly(x, max_km2)
        )

    elif isinstance(polygons, gpd.GeoSeries):
        polygons = polygons.map(lambda x: _close_holes_poly(x, max_km2))
        polygons = gpd.GeoSeries(polygons)

    else:
        polygons = _close_holes_poly(polygons, max_km2)

    return polygons


def _close_holes_poly(poly, max_km2=None):
    """Closes holes within one shapely geometry of polygons."""
    # dissolve the exterior ring(s)
    if max_km2 is None:
        holes_closed = polygons(get_exterior_ring(get_parts(poly)))
        return unary_union(holes_closed)

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

            if area(hole) / 1_000_000 < max_km2:
                holes_closed.append(hole)

    return unary_union(holes_closed)

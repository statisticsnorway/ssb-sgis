"""Functions for polygon geometries."""

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
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

from .general import _push_geom_col
from .neighbors import get_neighbor_indices
from .overlay import clean_overlay


def get_polygon_clusters(
    *gdfs: GeoDataFrame | GeoSeries, cluster_col: str = "cluster", explode: bool = True
) -> GeoDataFrame | tuple[GeoDataFrame]:
    """Find which polygons overlap without dissolving.

    Devides polygons into clusters in a fast and precice manner by using spatial join
    and networkx to find the connected components, i.e. overlapping geometries.
    If multiple GeoDataFrames are given, the clusters will be based on all
    combined.

    This can be used instead of dissolve+explode, or before dissolving by the cluster
    column. This has been tested to be a lot faster if there are many
    non-overlapping polygons, but somewhat slower than dissolve+explode if most
    polygons overlap.

    Args:
        gdfs: One or more GeoDataFrames of polygons.
        cluster_col: Name of the resulting cluster column.
        explode: Whether to explode the geometries to singlepart before the spatial
            join. Defaults to True. Index will be preserved.

    Returns:
        One or more GeoDataFrames (same amount as was given) with a new cluster column.

    Examples
    --------

    Create polygon geometries where row 0, 1 and 2 overlap, 3 and 4 overlap
    and 6 is on its own.

    >>> import sgis as sg
    >>> gdf = sg.to_gdf([(0, 0), (1, 1), (0, 1), (4, 4), (4, 3), (7, 7)])
    >>> buffered = sg.buff(gdf, 1)
    >>> gdf
                                                geometry
    0  POLYGON ((1.00000 0.00000, 0.99951 -0.03141, 0...
    1  POLYGON ((2.00000 1.00000, 1.99951 0.96859, 1....
    2  POLYGON ((1.00000 1.00000, 0.99951 0.96859, 0....
    3  POLYGON ((5.00000 4.00000, 4.99951 3.96859, 4....
    4  POLYGON ((5.00000 3.00000, 4.99951 2.96859, 4....
    5  POLYGON ((8.00000 7.00000, 7.99951 6.96859, 7....

    This will add a cluster column to the GeoDataFrame:

    >>> gdf = sg.get_polygon_clusters(gdf, cluster_col="cluster")
    >>> gdf
       cluster                                           geometry
    0        0  POLYGON ((1.00000 0.00000, 0.99951 -0.03141, 0...
    1        0  POLYGON ((2.00000 1.00000, 1.99951 0.96859, 1....
    2        0  POLYGON ((1.00000 1.00000, 0.99951 0.96859, 0....
    3        1  POLYGON ((5.00000 4.00000, 4.99951 3.96859, 4....
    4        1  POLYGON ((5.00000 3.00000, 4.99951 2.96859, 4....
    5        2  POLYGON ((8.00000 7.00000, 7.99951 6.96859, 7....

    If multiple GeoDataFrames are given, all are returned with common
    cluster values.

    >>> gdf2 = sg.to_gdf([(0, 0), (7, 7)])
    >>> gdf, gdf2 = sg.get_polygon_clusters(gdf, gdf2, cluster_col="cluster")
    >>> gdf2
    cluster                 geometry
    0        0  POINT (0.00000 0.00000)
    1        2  POINT (7.00000 7.00000)
    >>> gdf
       cluster                                           geometry
    0        0  POLYGON ((1.00000 0.00000, 0.99951 -0.03141, 0...
    1        0  POLYGON ((2.00000 1.00000, 1.99951 0.96859, 1....
    2        0  POLYGON ((1.00000 1.00000, 0.99951 0.96859, 0....
    3        1  POLYGON ((5.00000 4.00000, 4.99951 3.96859, 4....
    4        1  POLYGON ((5.00000 3.00000, 4.99951 2.96859, 4....
    5        2  POLYGON ((8.00000 7.00000, 7.99951 6.96859, 7....

    Dissolving 'by' the cluster column will make the dissolve much
    faster if there are a lot of non-overlapping polygons.

    >>> dissolved = gdf.dissolve(by="cluster", as_index=False)
    >>> dissolved
       cluster                                           geometry
    0        0  POLYGON ((0.99951 -0.03141, 0.99803 -0.06279, ...
    1        1  POLYGON ((4.99951 2.96859, 4.99803 2.93721, 4....
    2        2  POLYGON ((8.00000 7.00000, 7.99951 6.96859, 7....

    Which is equivelen to this in straigt geopandas:

    >>> dissolved2 = gdf.dissolve().explode(ignore_index=True).assign(cluster=lambda x: x.index)
    >>> dissolved2
       cluster                                           geometry
    0        0  POLYGON ((0.99803 -0.06279, 0.99556 -0.09411, ...
    1        1  POLYGON ((4.99803 2.93721, 4.99556 2.90589, 4....
    2        2  POLYGON ((7.99556 6.90589, 7.99211 6.87467, 7....

    Note that the order of the coordinates is different, and there is
    some deviations in the rounding on microscopic levels.

    >>> dissolved.area.sum()
    15.016909720698278
    >>> dissolved2.area.sum()
    15.016909720698285
    """
    if isinstance(gdfs[-1], str):
        *gdfs, cluster_col = gdfs

    concated = pd.DataFrame()
    for i, gdf in enumerate(gdfs):
        if isinstance(gdf, GeoSeries):
            gdf = gdf.to_frame()

        if not isinstance(gdf, GeoDataFrame):
            raise TypeError("'gdfs' should be one or more GeoDataFrames or GeoSeries.")

        if explode:
            gdf = gdf.explode(index_parts=False)

        gdf["orig_idx___"] = gdf.index
        gdf["_i___"] = i

        concated = pd.concat([concated, gdf], ignore_index=True)

    neighbors = get_neighbor_indices(concated, concated)

    edges = [(source, target) for source, target in neighbors.items()]

    graph = nx.Graph()
    graph.add_edges_from(edges)

    component_mapper = {
        j: i
        for i, component in enumerate(nx.connected_components(graph))
        for j in component
    }

    concated[cluster_col] = concated.index.map(component_mapper)

    concated.index = concated["orig_idx___"].values

    concated = _push_geom_col(concated)

    _i___ = concated["_i___"].unique()

    if len(_i___) == 1:
        return concated.drop(["_i___", "orig_idx___"], axis=1)

    unconcated = ()
    for i in _i___:
        gdf = concated[concated["_i___"] == i].drop(["_i___", "orig_idx___"], axis=1)
        unconcated = unconcated + (gdf,)

    return unconcated


def get_overlapping_polygons(
    gdf: GeoDataFrame | GeoSeries, ignore_index=False
) -> GeoDataFrame | GeoSeries:
    """Find the areas that overlap.

    Does an intersection with itself and keeps only the duplicated geometries. The
    index of 'gdf' is preserved.

    Args:
        gdf: GeoDataFrame of polygons.
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Defaults to False.

    Returns:
        A GeoDataFrame of the overlapping polygons.
    """
    if not gdf.index.is_unique:
        raise ValueError(
            "Index must be unique in order to correctly find "
            "overlapping polygon indices."
        )

    gdf = gdf.assign(overlap=gdf.index)

    intersected = clean_overlay(gdf, gdf[["geometry"]], how="intersection")

    points_joined = intersected.representative_point().to_frame().sjoin(intersected)

    duplicated_points = points_joined.loc[points_joined.index.duplicated()]

    duplicated_geoms = intersected.loc[intersected.index.isin(duplicated_points.index)]
    duplicated_geoms.index = duplicated_geoms["overlap"].values

    if ignore_index:
        duplicated_geoms = duplicated_geoms.reset_index(drop=True)

    return duplicated_geoms.drop("overlap", axis=1)


def get_overlapping_polygon_indices(gdf: GeoDataFrame | GeoSeries) -> pd.Index:
    """Get the index of the rows that contain overlapping geometries.

    Args:
        gdf: GeoDataFrame of polygons.

    Returns:
        A pandas Index with the overlapping polygon indices.
    """
    if not gdf.index.is_unique:
        raise ValueError(
            "Index must be unique in order to correctly find "
            "overlapping polygon indices."
        )

    gdf = gdf.assign(overlap=gdf.index)

    intersected = clean_overlay(gdf, gdf[["geometry"]], how="intersection")

    intersected = intersected.set_index("overlap")

    points_joined = intersected.representative_point().to_frame().sjoin(intersected)

    duplicated_points = points_joined.loc[points_joined.index.duplicated()]

    return duplicated_points.index.unique()


def get_overlapping_polygon_product(gdf: GeoDataFrame | GeoSeries) -> pd.Index:
    if not gdf.index.is_unique:
        raise ValueError("Index must be unique to find overlapping polygon indices.")

    gdf = gdf.assign(overlap=gdf.index)

    intersected = clean_overlay(gdf, gdf[["geometry"]], how="intersection")

    intersected = intersected.set_index("overlap")

    points_joined = intersected.representative_point().to_frame().sjoin(intersected)

    duplicated_points = points_joined.loc[points_joined.index.duplicated()]

    unique = (
        duplicated_points.reset_index()
        .groupby(["overlap", "index_right"])
        .size()
        .reset_index()
    )

    unique = unique[unique.overlap != unique.index_right]
    series = unique.set_index("index_right").overlap
    series.index.name = None

    return series


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

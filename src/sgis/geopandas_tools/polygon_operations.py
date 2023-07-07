"""Functions for polygon geometries."""
import warnings

import networkx as nx
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

from .general import _push_geom_col, to_lines
from .neighbors import get_neighbor_indices
from .overlay import clean_overlay


def get_centroid_ids(gdf: GeoDataFrame, groupby: str) -> pd.Series:
    centerpoints = gdf.assign(geometry=lambda x: x.centroid)

    grouped_centerpoints = centerpoints.dissolve(by=groupby).assign(
        geometry=lambda x: x.centroid
    )
    xs = grouped_centerpoints.geometry.x
    ys = grouped_centerpoints.geometry.y

    grouped_centerpoints["wkt"] = [f"{int(x)}_{int(y)}" for x, y in zip(xs, ys)]

    return gdf[groupby].map(grouped_centerpoints["wkt"])


def get_polygon_clusters(
    *gdfs: GeoDataFrame | GeoSeries,
    cluster_col: str = "cluster",
    allow_multipart: bool = False,
    wkt_col: bool = False,
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
        allow_multipart: Whether to allow mutipart geometries in the gdfs.
            Defaults to False to avoid confusing results.
        wkt_col: Whether to return the cluster column values as a string with x and y
            coordinates. Convinient to always get unique ids.
            Defaults to False because of speed.

    Returns:
        One or more GeoDataFrames (same amount as was given) with a new cluster column.

    Examples
    --------

    Create geometries with three clusters of overlapping polygons.

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

    Add a cluster column to the GeoDataFrame:

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
    """
    if isinstance(gdfs[-1], str):
        *gdfs, cluster_col = gdfs

    concated = []
    orig_indices = ()

    # take a copy only if there are gdfs with the same id
    # To not get any overwriting in the for loop
    if sum(df1 is df2 for df1 in gdfs for df2 in gdfs) > len(gdfs):
        new_gdfs = ()
        for gdf in gdfs:
            new_gdfs = new_gdfs + (gdf.copy(),)
        gdfs = new_gdfs

    for i, gdf in enumerate(gdfs):
        if isinstance(gdf, GeoSeries):
            gdf = gdf.to_frame()

        if not isinstance(gdf, GeoDataFrame):
            raise TypeError("'gdfs' should be GeoDataFrames or GeoSeries.")

        if not allow_multipart and len(gdf) != len(gdf.explode(index_parts=False)):
            raise ValueError(
                "All geometries should be exploded to singlepart "
                "in order to get correct polygon clusters. "
                "To allow multipart geometries, set allow_multipart=True"
            )

        orig_indices = orig_indices + (gdf.index,)

        gdf["i__"] = i

        concated.append(gdf)

    concated = pd.concat(concated, ignore_index=True)

    if not len(concated):
        return concated.drop("i__", axis=1).assign(**{cluster_col: []})

    neighbors = get_neighbor_indices(concated, concated)

    edges = [(source, target) for source, target in neighbors.items()]

    graph = nx.Graph()
    graph.add_edges_from(edges)

    component_mapper = {
        j: i
        for i, component in enumerate(nx.connected_components(graph))
        for j in component
    }

    concated[cluster_col] = component_mapper

    if wkt_col:
        concated[cluster_col] = get_centroid_ids(concated, groupby=cluster_col)

    concated = _push_geom_col(concated)

    n_gdfs = concated["i__"].unique()

    if len(n_gdfs) == 1:
        concated.index = orig_indices[0]
        return concated.drop(["i__"], axis=1)

    unconcated = ()
    for i in n_gdfs:
        gdf = concated[concated["i__"] == i]
        gdf.index = orig_indices[i]
        gdf = gdf.drop(["i__"], axis=1)
        unconcated = unconcated + (gdf,)

    return unconcated


def eliminate_by_longest(
    gdf: GeoDataFrame,
    to_eliminate: GeoDataFrame,
    *,
    remove_isolated: bool = False,
    ignore_index: bool = False,
    aggfunc: str | dict | list = "first",
    **kwargs,
) -> GeoDataFrame:
    """Dissolves selected polygons with the longest bordering neighbor polygon.

    Eliminates selected geometries by dissolving them with the neighboring
    polygon with the longest shared border. The index and column values of the
    large polygons will be kept, unless else is specified.

    Args:
        gdf: GeoDataFrame with polygon geometries.
        to_eliminate: The geometries to be eliminated by 'gdf'.
        remove_isolated: If False (default), polygons in 'to_eliminate' that share
            no border with any polygon in 'gdf' will be kept. If True, the isolated
            polygons will be removed.
        ignore_index: If False (default), the resulting GeoDataFrame will keep the
            index of the large polygons. If True, the resulting axis will be labeled
            0, 1, …, n - 1.
        aggfunc: Aggregation function(s) to use when dissolving. Defaults to 'first',
            meaning the column values of the large polygons are kept.
        kwargs: Keyword arguments passed to the dissolve method.

    Returns:
        The GeoDataFrame with the small polygons dissolved into the large polygons.
    """

    # remove polygons in gdf that are present in to_eliminate
    gdf = gdf.loc[~gdf.geometry.astype(str).isin(to_eliminate.geometry.astype(str))]

    if not ignore_index:
        idx_mapper = {i: idx for i, idx in enumerate(gdf.index)}
        idx_name = gdf.index.name

    # resetting in case not unique index
    gdf = gdf.reset_index(drop=True)

    gdf = gdf.assign(poly_idx=lambda x: x.index)
    to_eliminate = to_eliminate.assign(eliminate_idx=lambda x: range(len(x)))

    # convert to lines to get the border lines
    lines = to_lines(
        gdf[["poly_idx", "geometry"]], to_eliminate[["eliminate_idx", "geometry"]]
    )
    lines = lines[lines["eliminate_idx"].notna()]
    lines["length__"] = lines.length

    longest_border = lines.sort_values("length__", ascending=False).drop_duplicates(
        "eliminate_idx"
    )

    to_poly_idx = longest_border.set_index("eliminate_idx")["poly_idx"]
    to_eliminate["dissolve_idx"] = to_eliminate["eliminate_idx"].map(to_poly_idx)

    gdf["dissolve_idx"] = gdf["poly_idx"]

    kwargs.pop("as_index", None)
    eliminated = pd.concat([gdf, to_eliminate]).dissolve(
        "dissolve_idx", aggfunc=aggfunc, **kwargs
    )

    if ignore_index:
        return eliminated.reset_index(drop=True)
    else:
        eliminated.index = eliminated.index.map(idx_mapper)
        eliminated.index.name = idx_name

    if not remove_isolated:
        isolated = to_eliminate.loc[to_eliminate["dissolve_idx"].isna()]
        eliminated = pd.concat([eliminated, isolated])

    return eliminated.drop(
        ["dissolve_idx", "length__", "eliminate_idx", "poly_idx"],
        axis=1,
        errors="ignore",
    )


def eliminate_by_largest(
    gdf: GeoDataFrame,
    to_eliminate: GeoDataFrame,
    remove_isolated: bool = False,
    ignore_index: bool = False,
    aggfunc: str | dict | list = "first",
    **kwargs,
) -> GeoDataFrame:
    """Dissolves selected polygons with the largest neighbor polygon.

    Eliminates selected geometries by dissolving them with the neighboring
    polygon with the largest area. The index and column values of the
    large polygons will be kept, unless else is specified.

    Args:
        gdf: GeoDataFrame with polygon geometries.
        to_eliminate: The geometries to be eliminated by 'gdf'.
        remove_isolated: If False (default), polygons in 'to_eliminate' that share
            no border with any polygon in 'gdf' will be kept. If True, the isolated
            polygons will be removed.
        ignore_index: If False (default), the resulting GeoDataFrame will keep the
            index of the large polygons. If True, the resulting axis will be labeled
            0, 1, …, n - 1.
        aggfunc: Aggregation function(s) to use when dissolving. Defaults to 'first',
            meaning the column values of the large polygons are kept.
        kwargs: Keyword arguments passed to the dissolve method.

    Returns:
        The GeoDataFrame with the selected polygons dissolved into the polygons of
        'gdf'.
    """
    return _eliminate_by_area(
        gdf,
        to_eliminate=to_eliminate,
        remove_isolated=remove_isolated,
        ignore_index=ignore_index,
        sort_ascending=False,
        aggfunc=aggfunc,
        **kwargs,
    )


def eliminate_by_smallest(
    gdf: GeoDataFrame,
    to_eliminate: GeoDataFrame,
    remove_isolated: bool = False,
    ignore_index: bool = False,
    aggfunc: str | dict | list = "first",
    **kwargs,
) -> GeoDataFrame:
    return _eliminate_by_area(
        gdf,
        to_eliminate=to_eliminate,
        remove_isolated=remove_isolated,
        ignore_index=ignore_index,
        sort_ascending=True,
        aggfunc=aggfunc,
        **kwargs,
    )


def _eliminate_by_area(
    gdf: GeoDataFrame,
    to_eliminate: GeoDataFrame,
    remove_isolated: bool,
    sort_ascending: bool,
    ignore_index: bool = False,
    aggfunc="first",
    **kwargs,
) -> GeoDataFrame:
    # remove polygons in gdf that are present in to_eliminate
    gdf = gdf.loc[~gdf.geometry.astype(str).isin(to_eliminate.geometry.astype(str))]

    if not ignore_index:
        idx_mapper = {i: idx for i, idx in enumerate(gdf.index)}
        idx_name = gdf.index.name

    gdf = gdf.reset_index(drop=True)

    gdf["area__"] = gdf.area

    joined = to_eliminate.sjoin(
        gdf[["area__", "geometry"]], predicate="touches", how="left"
    ).sort_values("area__", ascending=sort_ascending)

    largest = joined[~joined.index.duplicated()]

    gdf = gdf.assign(index_right=lambda x: x.index)

    kwargs.pop("as_index", None)
    eliminated = (
        pd.concat([gdf, largest])
        .dissolve("index_right", aggfunc=aggfunc, **kwargs)
        .drop(["area__"], axis=1, errors="ignore")
    )

    if ignore_index:
        return eliminated.reset_index(drop=True)

    eliminated.index = eliminated.index.map(idx_mapper)
    eliminated.index.name = idx_name

    if not remove_isolated:
        isolated = joined.loc[joined["index_right"].isna()]
        eliminated = pd.concat([eliminated, isolated])

    return eliminated


def get_overlapping_polygons(
    gdf: GeoDataFrame | GeoSeries, ignore_index: bool = False
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
    if not any(gdf.geom_type.isin(["Polygon", "MultiPolygon"])):
        raise ValueError("'gdf' has no polygons.")

    elif not all(gdf.geom_type.isin(["Polygon", "MultiPolygon"])):
        warnings.warn("'gdf' has mixed geometries. Non-polygons will be removed.")

    if not ignore_index:
        idx_mapper = {i: idx for i, idx in enumerate(gdf.index)}
        idx_name = gdf.index.name

    gdf = gdf.reset_index(drop=True).assign(overlap=gdf.index)

    intersected = clean_overlay(
        gdf, gdf[["geometry"]], how="intersection", geom_type="polygon"
    )

    points_joined = intersected.representative_point().to_frame().sjoin(intersected)

    duplicated_points = points_joined.loc[points_joined.index.duplicated()]

    duplicated_geoms = intersected.loc[intersected.index.isin(duplicated_points.index)]
    duplicated_geoms.index = duplicated_geoms["overlap"].values

    if ignore_index:
        duplicated_geoms = duplicated_geoms.reset_index(drop=True)
    else:
        duplicated_geoms.index = duplicated_geoms.index.map(idx_mapper)
        duplicated_geoms.index.name = idx_name

    return duplicated_geoms.drop("overlap", axis=1)


def get_overlapping_polygon_indices(gdf: GeoDataFrame | GeoSeries) -> pd.Index:
    if not gdf.index.is_unique:
        raise ValueError(
            "Index must be unique in order to correctly find "
            "overlapping polygon indices."
        )

    idx_mapper = {i: idx for i, idx in enumerate(gdf.index)}
    idx_name = gdf.index.name

    gdf = gdf.reset_index(drop=True).assign(overlap=gdf.index)

    intersected = clean_overlay(
        gdf, gdf[["geometry"]], how="intersection", geom_type="polygon"
    )

    intersected = intersected.set_index("overlap")

    points_joined = intersected.representative_point().to_frame().sjoin(intersected)

    duplicated_points = points_joined.loc[points_joined.index.duplicated()]

    duplicated_points.index = duplicated_points.index.map(idx_mapper)
    duplicated_points.index.name = idx_name

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


def close_all_holes(
    gdf: GeoDataFrame | GeoSeries,
    *,
    without_islands: bool = True,
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
    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):
        raise ValueError(
            f"'gdf' should be of type GeoDataFrame or GeoSeries. Got {type(gdf)}"
        )

    if copy:
        gdf = gdf.copy()

    if without_islands:
        all_geoms = gdf.unary_union
        if isinstance(gdf, GeoDataFrame):
            gdf["geometry"] = gdf.geometry.map(
                lambda x: _close_all_holes_no_islands(x, all_geoms)
            )
            return gdf
        else:
            return gdf.map(lambda x: _close_all_holes_no_islands(x, all_geoms))
    else:
        if isinstance(gdf, GeoDataFrame):
            gdf["geometry"] = gdf.geometry.map(_close_all_holes)
            return gdf
        else:
            return gdf.map(_close_all_holes)


def close_small_holes(
    gdf: GeoDataFrame | GeoSeries,
    max_area: int | float,
    *,
    without_islands: bool = True,
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
    if not isinstance(gdf, (GeoSeries, GeoDataFrame)):
        raise ValueError(
            f"'gdf' should be of type GeoDataFrame or GeoSeries. Got {type(gdf)}"
        )

    if copy:
        gdf = gdf.copy()

    if without_islands:
        all_geoms = gdf.unary_union

        if isinstance(gdf, GeoDataFrame):
            gdf["geometry"] = gdf.geometry.map(
                lambda x: _close_small_holes_no_islands(x, max_area, all_geoms)
            )
            return gdf
        else:
            return gdf.map(
                lambda x: _close_small_holes_no_islands(x, max_area, all_geoms)
            )
    else:
        if isinstance(gdf, GeoDataFrame):
            gdf["geometry"] = gdf.geometry.map(
                lambda x: _close_small_holes(x, max_area)
            )
            return gdf
        else:
            return gdf.map(lambda x: _close_small_holes(x, max_area))


def _close_small_holes(poly, max_area):
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

            print(area(hole))

            if area(hole) < max_area:
                holes_closed.append(hole)

    return unary_union(holes_closed)


def _close_small_holes_no_islands(poly, max_area, all_geoms):
    """Closes small holes within one shapely geometry of polygons."""

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
            no_islands = unary_union(hole.difference(all_geoms))
            if area(no_islands) < max_area:
                holes_closed.append(no_islands)

    return unary_union(holes_closed)


def _close_all_holes(poly):
    return unary_union(polygons(get_exterior_ring(get_parts(poly))))


def _close_all_holes_no_islands(poly, all_geoms):
    """Closes all holes within one shapely geometry of polygons."""

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
            no_islands = unary_union(hole.difference(all_geoms))
            holes_closed.append(no_islands)

    return unary_union(holes_closed)

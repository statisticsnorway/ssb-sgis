"""Functions for polygon geometries."""
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

from .general import _push_geom_col, clean_geoms, get_grouped_centroids, to_lines
from .geometry_types import make_all_singlepart
from .neighbors import get_neighbor_indices


def get_polygon_clusters(
    *gdfs: GeoDataFrame | GeoSeries,
    cluster_col: str = "cluster",
    allow_multipart: bool = False,
    as_string: bool = False,
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
        as_string: Whether to return the cluster column values as a string with x and y
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
        gdfs = [gdf.copy() for gdf in gdfs]

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

    if as_string:
        concated[cluster_col] = get_grouped_centroids(concated, groupby=cluster_col)

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
    aggfunc: str | dict | list | None = None,
    **kwargs,
) -> GeoDataFrame:
    """Dissolves selected polygons with the longest bordering neighbor polygon.

    Eliminates selected geometries by dissolving them with the neighboring
    polygon with the longest shared border. The index and column values of the
    large polygons will be kept, unless else is specified.

    Note that this might be a lot slower than eliminate_by_largest.

    Args:
        gdf: GeoDataFrame with polygon geometries.
        to_eliminate: The geometries to be eliminated by 'gdf'.
        remove_isolated: If False (default), polygons in 'to_eliminate' that share
            no border with any polygon in 'gdf' will be kept. If True, the isolated
            polygons will be removed.
        ignore_index: If False (default), the resulting GeoDataFrame will keep the
            index of the large polygons. If True, the resulting axis will be labeled
            0, 1, …, n - 1.
        aggfunc: Aggregation function(s) to use when dissolving/eliminating.
            Defaults to None, meaning the values of 'gdf' is used. Otherwise,
            aggfunc will be passed to pandas groupby.agg. note: The geometries of
            'gdf' are sorted first, but if 'gdf' has missing values, the resulting
            polygons might get values from the polygons to be eliminated
            (if aggfunc="first").
        kwargs: Keyword arguments passed to the dissolve method.

    Returns:
        The GeoDataFrame with the small polygons dissolved into the large polygons.
    """
    crs = gdf.crs

    common_geometries = gdf.geometry.astype(str).isin(to_eliminate.geometry.astype(str))
    if any(common_geometries):
        gdf = gdf.loc[~common_geometries]

    if not ignore_index:
        idx_mapper = dict(enumerate(gdf.index))
        idx_name = gdf.index.name

    gdf = gdf.reset_index(drop=True)

    gdf["poly_idx"] = gdf.index
    to_eliminate = to_eliminate.assign(eliminate_idx=lambda x: range(len(x)))

    # convert to lines to get the borders
    lines_gdf = to_lines(gdf[["poly_idx", "geometry"]], copy=False)
    lines_eliminate = to_lines(to_eliminate[["eliminate_idx", "geometry"]], copy=False)

    borders = lines_gdf.overlay(lines_eliminate).loc[
        lambda x: x["eliminate_idx"].notna()
    ]

    borders["length__"] = borders.length

    # as DataFrame because GeoDataFrame constructor is expensive
    borders = pd.DataFrame(borders)
    gdf = pd.DataFrame(gdf)

    longest_border = borders.sort_values("length__", ascending=False).drop_duplicates(
        "eliminate_idx"
    )

    to_poly_idx = longest_border.set_index("eliminate_idx")["poly_idx"]
    to_eliminate["_dissolve_idx"] = to_eliminate["eliminate_idx"].map(to_poly_idx)

    gdf = gdf.rename(columns={"poly_idx": "_dissolve_idx"}, errors="raise")

    actually_eliminate = to_eliminate.loc[lambda x: x["_dissolve_idx"].notna()]

    eliminated = _eliminate(gdf, actually_eliminate, aggfunc, crs, **kwargs)

    if ignore_index:
        eliminated = eliminated.reset_index(drop=True)
    else:
        eliminated.index = eliminated.index.map(idx_mapper)
        eliminated.index.name = idx_name

    if not remove_isolated:
        isolated = to_eliminate.loc[to_eliminate["_dissolve_idx"].isna()]
        if len(isolated):
            eliminated = pd.concat([eliminated, isolated])

    eliminated = eliminated.drop(
        ["_dissolve_idx", "length__", "eliminate_idx", "poly_idx"],
        axis=1,
        errors="ignore",
    )

    return GeoDataFrame(eliminated, geometry="geometry", crs=crs).pipe(clean_geoms)


def eliminate_by_largest(
    gdf: GeoDataFrame,
    to_eliminate: GeoDataFrame,
    *,
    remove_isolated: bool = False,
    ignore_index: bool = False,
    aggfunc: str | dict | list | None = None,
    predicate: str = "intersects",
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
        aggfunc: Aggregation function(s) to use when dissolving/eliminating.
            Defaults to None, meaning the values of 'gdf' is used. Otherwise,
            aggfunc will be passed to pandas groupby.agg. note: The geometries of
            'gdf' are sorted first, but if 'gdf' has missing values, the resulting
            polygons might get values from the polygons to be eliminated
            (if aggfunc="first").
        predicate: Binary predicate passed to sjoin. Defaults to "intersects".
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
        predicate=predicate,
        **kwargs,
    )


def eliminate_by_smallest(
    gdf: GeoDataFrame,
    to_eliminate: GeoDataFrame,
    *,
    remove_isolated: bool = False,
    ignore_index: bool = False,
    aggfunc: str | dict | list | None = None,
    predicate: str = "intersects",
    **kwargs,
) -> GeoDataFrame:
    return _eliminate_by_area(
        gdf,
        to_eliminate=to_eliminate,
        remove_isolated=remove_isolated,
        ignore_index=ignore_index,
        sort_ascending=True,
        aggfunc=aggfunc,
        predicate=predicate,
        **kwargs,
    )


def _eliminate_by_area(
    gdf: GeoDataFrame,
    to_eliminate: GeoDataFrame,
    remove_isolated: bool,
    sort_ascending: bool,
    ignore_index: bool = False,
    aggfunc: str | dict | list | None = None,
    predicate="intersects",
    **kwargs,
) -> GeoDataFrame:
    crs = gdf.crs

    common_geometries = gdf.geometry.astype(str).isin(to_eliminate.geometry.astype(str))
    if any(common_geometries):
        gdf = gdf.loc[~common_geometries]

    if not ignore_index:
        idx_mapper = dict(enumerate(gdf.index))
        idx_name = gdf.index.name

    gdf = make_all_singlepart(gdf).reset_index(drop=True)
    to_eliminate = make_all_singlepart(to_eliminate).reset_index(drop=True)

    gdf["_area"] = gdf.area
    gdf["_dissolve_idx"] = gdf.index

    joined = to_eliminate.sjoin(
        gdf[["_area", "_dissolve_idx", "geometry"]], predicate=predicate, how="left"
    )

    # as DataFrames because GeoDataFrame constructor is expensive
    joined = (
        pd.DataFrame(joined)
        .drop(columns="index_right")
        .sort_values("_area", ascending=sort_ascending)
        .loc[lambda x: ~x.index.duplicated(keep="first")]
    )

    gdf = pd.DataFrame(gdf)

    notna = joined.loc[lambda x: x["_dissolve_idx"].notna()]
    eliminated = _eliminate(gdf, notna, aggfunc, crs, **kwargs)

    if ignore_index:
        eliminated = eliminated.reset_index(drop=True)
    else:
        eliminated.index = eliminated.index.map(idx_mapper)
        eliminated.index.name = idx_name

    if not remove_isolated:
        isolated = joined.loc[joined["_dissolve_idx"].isna()]
        if len(isolated):
            eliminated = pd.concat([eliminated, isolated])

    return GeoDataFrame(eliminated, geometry="geometry", crs=crs).pipe(clean_geoms)


def _eliminate(gdf, to_eliminate, aggfunc, crs, **kwargs):
    if not len(to_eliminate):
        return gdf

    in_to_eliminate = gdf["_dissolve_idx"].isin(to_eliminate["_dissolve_idx"])
    to_dissolve = gdf.loc[in_to_eliminate]
    not_to_dissolve = gdf.loc[~in_to_eliminate].set_index("_dissolve_idx")

    if aggfunc is None:
        concatted = pd.concat(
            [to_dissolve, to_eliminate[["_dissolve_idx", "geometry"]]]
        )
        aggfunc = "first"
    else:
        concatted = pd.concat([to_dissolve, to_eliminate])

    only_one = concatted.loc[
        lambda x: (x.groupby("_dissolve_idx").transform("size") == 1)
        & (x["_dissolve_idx"].notna())
    ].set_index("_dissolve_idx")
    assert len(only_one) == 0

    more_than_one = concatted.loc[
        lambda x: x.groupby("_dissolve_idx").transform("size") > 1
    ]

    if not len(more_than_one):
        return only_one

    kwargs.pop("as_index", None)
    eliminated = (
        # more_than_one.dissolve("_dissolve_idx", aggfunc=aggfunc, **kwargs)
        more_than_one.drop(columns="geometry")
        .groupby("_dissolve_idx", **kwargs)
        .agg(aggfunc)
        .drop(["_area"], axis=1, errors="ignore")
    )
    eliminated["geometry"] = more_than_one.groupby("_dissolve_idx")["geometry"].agg(
        lambda x: unary_union(x.values)
    )

    # setting crs on geometryarray to avoid warning in concat
    not_to_dissolve.geometry.values.crs = crs
    eliminated.geometry.values.crs = crs
    only_one.geometry.values.crs = crs

    to_concat = [not_to_dissolve, eliminated, only_one]

    assert all(df.index.name == "_dissolve_idx" for df in to_concat)

    return pd.concat(to_concat).sort_index()


def close_all_holes(
    gdf: GeoDataFrame | GeoSeries,
    *,
    ignore_islands: bool = False,
    copy: bool = True,
) -> GeoDataFrame | GeoSeries:
    """Closes all holes in polygons.

    It takes a GeoDataFrame or GeoSeries of polygons and
    returns the outer circle.

    Args:
        gdf: GeoDataFrame or GeoSeries of polygons.
        copy: if True (default), the input GeoDataFrame or GeoSeries is copied.
            Defaults to True.
        ignore_islands: If False (default), polygons inside the holes (islands)
            will be erased from the output geometries. If True, the entire
            holes will be closed and the islands kept, meaning there might be
            duplicate surfaces in the resulting geometries.

    Returns:
        A GeoDataFrame or GeoSeries of polygons with closed holes in the geometry
        column.

    Examples
    --------
    Let's create a circle with a hole in it.

    >>> point = sg.to_gdf([260000, 6650000], crs=25833)
    >>> point
                            geometry
    0  POINT (260000.000 6650000.000)
    >>> circle = sg.buff(point, 1000)
    >>> small_circle = sg.buff(point, 500)
    >>> circle_with_hole = circle.overlay(small_circle, how="difference")
    >>> circle_with_hole.area
    0    2.355807e+06
    dtype: float64

    Close the hole.

    >>> holes_closed = sg.close_all_holes(circle_with_hole)
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

    if ignore_islands:
        if isinstance(gdf, GeoDataFrame):
            gdf["geometry"] = gdf.geometry.map(_close_all_holes)
            return gdf
        else:
            return gdf.map(_close_all_holes)

    all_geoms = gdf.unary_union
    if isinstance(gdf, GeoDataFrame):
        gdf["geometry"] = gdf.geometry.map(
            lambda x: _close_all_holes_no_islands(x, all_geoms)
        )
        return gdf
    else:
        return gdf.map(lambda x: _close_all_holes_no_islands(x, all_geoms))


def close_small_holes(
    gdf: GeoDataFrame | GeoSeries,
    max_area: int | float,
    *,
    ignore_islands: bool = False,
    copy: bool = True,
) -> GeoDataFrame | GeoSeries:
    """Closes holes in polygons if the area is less than the given maximum.

    It takes a GeoDataFrame or GeoSeries of polygons and
    fills the holes that are smaller than the specified area given in units of
    either square meters ('max_m2') or square kilometers ('max_km2').

    Args:
        gdf: GeoDataFrame or GeoSeries of polygons.
        max_area: The maximum area in the unit of the GeoDataFrame's crs.
        ignore_islands: If False (default), polygons inside the holes (islands)
            will be erased from the "hole" geometries before the area is calculated.
            If True, the entire polygon interiors will be considered, meaning there
            might be duplicate surfaces in the resulting geometries.
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

    >>> point = sg.to_gdf([260000, 6650000], crs=25833)
    >>> point
                            geometry
    0  POINT (260000.000 6650000.000)
    >>> circle = sg.buff(point, 1000)
    >>> small_circle = sg.buff(point, 500)
    >>> circle_with_hole = circle.overlay(small_circle, how="difference")
    >>> circle_with_hole.area
    0    2.355807e+06
    dtype: float64

    Close holes smaller than 1 square kilometer (1 million square meters).

    >>> holes_closed = sg.close_small_holes(circle_with_hole, max_area=1_000_000)
    >>> holes_closed.area
    0    3.141076e+06
    dtype: float64

    The hole will not be closed if it is larger.

    >>> holes_closed = sg.close_small_holes(circle_with_hole, max_area=1_000)
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

    if not ignore_islands:
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

"""Functions for polygon geometries."""

import networkx as nx
import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame, GeoSeries
from shapely import (
    STRtree,
    area,
    box,
    buffer,
    difference,
    get_exterior_ring,
    get_interior_ring,
    get_num_interior_rings,
    get_parts,
    is_empty,
    make_valid,
    polygons,
    unary_union,
)
from shapely.errors import GEOSException

from .duplicates import get_intersections
from .general import _push_geom_col, clean_geoms, get_grouped_centroids, to_lines
from .geometry_types import get_geom_type, make_all_singlepart, to_single_geom_type
from .neighbors import get_neighbor_indices
from .overlay import clean_overlay
from .polygons_as_rings import PolygonsAsRings
from .sfilter import sfilter, sfilter_inverse


def get_polygon_clusters(
    *gdfs: GeoDataFrame | GeoSeries,
    cluster_col: str = "cluster",
    allow_multipart: bool = False,
    predicate: str | None = "intersects",
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

        gdf = gdf.assign(i__=i)

        concated.append(gdf)

    concated = pd.concat(concated, ignore_index=True)

    if not len(concated):
        return concated.drop("i__", axis=1).assign(**{cluster_col: []})

    concated[cluster_col] = get_cluster_mapper(concated, predicate)

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


def get_cluster_mapper(gdf, predicate="intersects"):
    if not gdf.index.is_unique:
        raise ValueError("Index must be unique")
    neighbors = get_neighbor_indices(gdf, gdf, predicate=predicate)

    edges = [(source, target) for source, target in neighbors.items()]

    graph = nx.Graph()
    graph.add_edges_from(edges)

    return {
        j: i
        for i, component in enumerate(nx.connected_components(graph))
        for j in component
    }


def eliminate_by_longest(
    gdf: GeoDataFrame | list[GeoDataFrame],
    to_eliminate: GeoDataFrame,
    *,
    remove_isolated: bool = False,
    fix_double: bool = True,
    ignore_index: bool = False,
    aggfunc: str | dict | list | None = None,
    grid_size=None,
    **kwargs,
) -> GeoDataFrame | tuple[GeoDataFrame]:
    """Dissolves selected polygons with the longest bordering neighbor polygon.

    Eliminates selected geometries by dissolving them with the neighboring
    polygon with the longest shared border. The index and column values of the
    large polygons will be kept, unless else is specified.

    Note that this might be a lot slower than eliminate_by_largest.

    Args:
        gdf: GeoDataFrame with polygon geometries, or a list of GeoDataFrames.
        to_eliminate: The geometries to be eliminated by 'gdf'.
        remove_isolated: If False (default), polygons in 'to_eliminate' that share
            no border with any polygon in 'gdf' will be kept. If True, the isolated
            polygons will be removed.
        fix_double: If True, geometries to be eliminated will be erased by overlapping
            geometries to not get double surfaces if the geometries in 'to_eliminate'
            overlaps with multiple geometries in 'gdf'.
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
        The GeoDataFrame (gdf) with the geometries of 'to_eliminate' dissolved in.
        If multiple GeoDataFrame are passed as 'gdf', they are returned as a tuple.

    Examples
    --------

    Create two polygons with a sliver in between:

    >>> sliver = sg.to_gdf(Polygon([(0, 0), (0.1, 1), (0, 2), (-0.1, 1)]))
    >>> small_poly = sg.to_gdf(
    ...     Polygon([(0, 0), (-0.1, 1), (0, 2), (-1, 2), (-2, 2), (-1, 1)])
    ... )
    >>> large_poly = sg.to_gdf(
    ...     Polygon([(0, 0), (0.1, 1), (1, 2), (2, 2), (3, 2), (3, 0)])
    ... )

    Using multiple GeoDataFrame as input, the sliver is eliminated into the small
    polygon (because it has the longest border with sliver).

    >>> small_poly_eliminated, large_poly_eliminated = sg.eliminate_by_longest(
    ...     [small_poly, large_poly], sliver
    ... )

    With only one input GeoDataFrame:

    >>> polys = pd.concat([small_poly, large_poly])
    >>> eliminated = sg.eliminate_by_longest(polys, sliver)


    """
    if isinstance(gdf, (list, tuple)):
        # concat, then break up the dataframes in the end
        was_multiple_gdfs = True
        original_cols = [df.columns for df in gdf]
        gdf = pd.concat(df.assign(**{"_df_idx": i}) for i, df in enumerate(gdf))
    else:
        was_multiple_gdfs = False

    crs = gdf.crs
    geom_type = get_geom_type(gdf)

    if not ignore_index:
        idx_mapper = dict(enumerate(gdf.index))
        idx_name = gdf.index.name

    gdf = gdf.reset_index(drop=True)

    gdf["_dissolve_idx"] = gdf.index
    to_eliminate = to_eliminate.assign(_eliminate_idx=lambda x: range(len(x)))

    # convert to lines to get the borders
    lines_eliminate = to_lines(to_eliminate[["_eliminate_idx", "geometry"]])

    borders = (
        gdf[["_dissolve_idx", "geometry"]]
        .overlay(lines_eliminate, keep_geom_type=False)
        .loc[lambda x: x["_eliminate_idx"].notna()]
    )

    borders["_length"] = borders.length

    # as DataFrame because GeoDataFrame constructor is expensive
    borders = pd.DataFrame(borders)

    longest_border = borders.sort_values("_length", ascending=False).drop_duplicates(
        "_eliminate_idx"
    )

    to_dissolve_idx = longest_border.set_index("_eliminate_idx")["_dissolve_idx"]
    to_eliminate["_dissolve_idx"] = to_eliminate["_eliminate_idx"].map(to_dissolve_idx)

    actually_eliminate = to_eliminate.loc[to_eliminate["_dissolve_idx"].notna()]

    isolated = to_eliminate.loc[to_eliminate["_dissolve_idx"].isna()]
    containing_eliminators = (
        pd.DataFrame(
            isolated.drop(columns="_dissolve_idx").sjoin(
                gdf[["_dissolve_idx", "geometry"]], predicate="contains"
            )
        )
        .drop(columns="index_right")
        .drop_duplicates("_eliminate_idx")
    )

    eliminated = _eliminate(
        pd.DataFrame(gdf),
        pd.concat([actually_eliminate, containing_eliminators]),
        aggfunc,
        crs,
        fix_double,
        grid_size=grid_size,
        **kwargs,
    )

    if not ignore_index:
        eliminated.index = eliminated.index.map(idx_mapper)
        eliminated.index.name = idx_name

    if not remove_isolated and len(isolated):
        eliminated = pd.concat([eliminated, isolated])

    eliminated = eliminated.drop(
        ["_dissolve_idx", "_length", "_eliminate_idx", "_dissolve_idx"],
        axis=1,
        errors="ignore",
    )

    out = GeoDataFrame(eliminated, geometry="geometry", crs=crs).pipe(clean_geoms)

    if geom_type != "mixed":
        out = to_single_geom_type(out, geom_type)

    out = out.reset_index(drop=True) if ignore_index else out

    if not was_multiple_gdfs:
        return out

    gdfs = ()
    for i, cols in enumerate(original_cols):
        df = out.loc[out["_df_idx"] == i, cols]
        gdfs += (df,)

    return gdfs


def eliminate_by_largest(
    gdf: GeoDataFrame | list[GeoDataFrame],
    to_eliminate: GeoDataFrame,
    *,
    max_distance: int | float | None = None,
    remove_isolated: bool = False,
    fix_double: bool = False,
    ignore_index: bool = False,
    aggfunc: str | dict | list | None = None,
    predicate: str = "intersects",
    grid_size=None,
    **kwargs,
) -> GeoDataFrame | tuple[GeoDataFrame]:
    """Dissolves selected polygons with the largest neighbor polygon.

    Eliminates selected geometries by dissolving them with the neighboring
    polygon with the largest area. The index and column values of the
    large polygons will be kept, unless else is specified.

    Args:
        gdf: GeoDataFrame with polygon geometries, or a list of GeoDataFrames.
        to_eliminate: The geometries to be eliminated by 'gdf'.
        remove_isolated: If False (default), polygons in 'to_eliminate' that share
            no border with any polygon in 'gdf' will be kept. If True, the isolated
            polygons will be removed.
        fix_double: If True, geometries to be eliminated will be erased by overlapping
            geometries to not get double surfaces if the geometries in 'to_eliminate'
            overlaps with multiple geometries in 'gdf'.
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
        The GeoDataFrame (gdf) with the geometries of 'to_eliminate' dissolved in.
        If multiple GeoDataFrame are passed as 'gdf', they are returned as a tuple.

    Examples
    --------

    Create two polygons with a sliver in between:

    >>> sliver = sg.to_gdf(Polygon([(0, 0), (0.1, 1), (0, 2), (-0.1, 1)]))
    >>> small_poly = sg.to_gdf(
    ...     Polygon([(0, 0), (-0.1, 1), (0, 2), (-1, 2), (-2, 2), (-1, 1)])
    ... )
    >>> large_poly = sg.to_gdf(
    ...     Polygon([(0, 0), (0.1, 1), (1, 2), (2, 2), (3, 2), (3, 0)])
    ... )

    Using multiple GeoDataFrame as input, the sliver is eliminated into
    the large polygon.

    >>> small_poly_eliminated, large_poly_eliminated = sg.eliminate_by_largest(
    ...     [small_poly, large_poly], sliver
    ... )

    With only one input GeoDataFrame:

    >>> polys = pd.concat([small_poly, large_poly])
    >>> eliminated = sg.eliminate_by_largest(polys, sliver)

    """
    return _eliminate_by_area(
        gdf,
        to_eliminate=to_eliminate,
        remove_isolated=remove_isolated,
        max_distance=max_distance,
        ignore_index=ignore_index,
        sort_ascending=False,
        aggfunc=aggfunc,
        predicate=predicate,
        fix_double=fix_double,
        grid_size=grid_size,
        **kwargs,
    )


def eliminate_by_smallest(
    gdf: GeoDataFrame | list[GeoDataFrame],
    to_eliminate: GeoDataFrame,
    *,
    max_distance: int | float | None = None,
    remove_isolated: bool = False,
    ignore_index: bool = False,
    aggfunc: str | dict | list | None = None,
    predicate: str = "intersects",
    fix_double: bool = False,
    grid_size=None,
    **kwargs,
) -> GeoDataFrame | tuple[GeoDataFrame]:
    return _eliminate_by_area(
        gdf,
        to_eliminate=to_eliminate,
        remove_isolated=remove_isolated,
        max_distance=max_distance,
        ignore_index=ignore_index,
        sort_ascending=True,
        aggfunc=aggfunc,
        predicate=predicate,
        fix_double=fix_double,
        grid_size=grid_size,
        **kwargs,
    )


def _eliminate_by_area(
    gdf: GeoDataFrame,
    to_eliminate: GeoDataFrame,
    remove_isolated: bool,
    max_distance: int | float | None,
    sort_ascending: bool,
    ignore_index: bool = False,
    aggfunc: str | dict | list | None = None,
    predicate="intersects",
    fix_double: bool = False,
    grid_size=None,
    **kwargs,
) -> GeoDataFrame:
    if isinstance(gdf, (list, tuple)):
        was_multiple_gdfs = True
        original_cols = [df.columns for df in gdf]
        gdf = pd.concat(df.assign(**{"_df_idx": i}) for i, df in enumerate(gdf))
    else:
        was_multiple_gdfs = False

    crs = gdf.crs
    geom_type = get_geom_type(gdf)

    if not ignore_index:
        idx_mapper = dict(enumerate(gdf.index))
        idx_name = gdf.index.name

    gdf = make_all_singlepart(gdf).reset_index(drop=True)
    to_eliminate = make_all_singlepart(to_eliminate).reset_index(drop=True)

    gdf["_area"] = gdf.area
    gdf["_dissolve_idx"] = gdf.index

    if max_distance:
        to_join = gdf[["_area", "_dissolve_idx", "geometry"]]
        to_join.geometry = to_join.buffer(max_distance)
        joined = to_eliminate.sjoin(to_join, predicate=predicate, how="left")
    else:
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

    eliminated = _eliminate(
        gdf, notna, aggfunc, crs, fix_double=fix_double, grid_size=grid_size, **kwargs
    )

    if not ignore_index:
        eliminated.index = eliminated.index.map(idx_mapper)
        eliminated.index.name = idx_name

    if not remove_isolated:
        isolated = joined.loc[joined["_dissolve_idx"].isna()]
        if len(isolated):
            eliminated = pd.concat([eliminated, isolated])

    eliminated = eliminated.drop(
        ["_dissolve_idx", "_area", "_eliminate_idx", "_dissolve_idx"],
        axis=1,
        errors="ignore",
    )

    out = GeoDataFrame(eliminated, geometry="geometry", crs=crs).pipe(clean_geoms)

    if geom_type != "mixed":
        out = to_single_geom_type(out, geom_type)

    out = out.reset_index(drop=True) if ignore_index else out

    if not was_multiple_gdfs:
        return out

    gdfs = ()
    for i, cols in enumerate(original_cols):
        df = out.loc[out["_df_idx"] == i, cols]
        gdfs += (df,)

    return gdfs


def _eliminate(gdf, to_eliminate, aggfunc, crs, fix_double, grid_size, **kwargs):
    if not len(to_eliminate):
        return gdf

    in_to_eliminate = gdf["_dissolve_idx"].isin(to_eliminate["_dissolve_idx"])
    to_dissolve = gdf.loc[in_to_eliminate]
    not_to_dissolve = gdf.loc[~in_to_eliminate].set_index("_dissolve_idx")

    to_eliminate["_to_eliminate"] = 1

    if aggfunc is None:
        concatted = pd.concat(
            [to_dissolve, to_eliminate[["_to_eliminate", "_dissolve_idx", "geometry"]]]
        )
        aggfunc = "first"
    else:
        concatted = pd.concat([to_dissolve, to_eliminate])

    one_hit = concatted.loc[
        lambda x: (x.groupby("_dissolve_idx").transform("size") == 1)
        & (x["_dissolve_idx"].notna())
    ].set_index("_dissolve_idx")

    assert len(one_hit) == 0

    many_hits = concatted.loc[
        lambda x: x.groupby("_dissolve_idx").transform("size") > 1
    ]

    if not len(many_hits):
        return one_hit

    # aggregate all columns except geometry
    kwargs.pop("as_index", None)
    eliminated = (
        many_hits.drop(columns="geometry")
        .groupby("_dissolve_idx", **kwargs)
        .agg(aggfunc)
        .drop(["_area"], axis=1, errors="ignore")
    )

    # aggregate geometry
    if fix_double:
        assert eliminated.index.is_unique

        many_hits = many_hits.set_index("_dissolve_idx")
        many_hits["_row_idx"] = range(len(many_hits))

        # TODO kan dette fikses trygt med .duplicated og ~x.duplicated?
        eliminators: pd.Series = many_hits.loc[
            many_hits["_to_eliminate"] != 1, "geometry"
        ]
        to_be_eliminated = many_hits.loc[many_hits["_to_eliminate"] == 1]

        if 0:
            tree = STRtree(eliminators.values)
            left, right = tree.query(
                to_be_eliminated.geometry.values, predicate="intersects"
            )
            pairs = pd.Series(right, index=left).to_frame("right")
            pairs["_dissolve_idx"] = pairs.index.map(
                dict(enumerate(to_be_eliminated.index))
            )

            soon_erased = to_be_eliminated.iloc[pairs.index]
            intersecting = eliminators.iloc[pairs["right"]]

            intersecting.index = soon_erased.index
            soon_erased = soon_erased.geometry.groupby(level=0).agg(unary_union)
            intersecting = intersecting.groupby(level=0).agg(unary_union)

            soon_erased.loc[:] = difference(
                soon_erased.values,
                intersecting.values,
            )
            intersecting.loc[:] = difference(
                intersecting.values,
                soon_erased.values,
            )

            eliminated["geometry"] = (
                pd.concat([intersecting, soon_erased])
                .groupby(level=0)
                .agg(lambda x: make_valid(unary_union(x.dropna().values)))
            )
            from ..maps.maps import explore, explore_locals

            explore_locals()

        # all_geoms: pd.Series = gdf.set_index("_dissolve_idx").geometry
        all_geoms: pd.Series = gdf.geometry

        tree = STRtree(all_geoms.values)
        left, right = tree.query(
            to_be_eliminated.geometry.values, predicate="intersects"
        )
        pairs = pd.Series(right, index=left).to_frame("right")
        pairs["_dissolve_idx"] = pairs.index.map(
            dict(enumerate(to_be_eliminated.index))
        )

        # pairs = pairs.loc[lambda x: x["right"] != x["_dissolve_idx"]]

        soon_erased = to_be_eliminated.iloc[pairs.index]
        intersecting = all_geoms.iloc[pairs["right"]]

        shoud_not_erase = soon_erased.index != intersecting.index
        soon_erased = soon_erased[shoud_not_erase]
        intersecting = intersecting[shoud_not_erase]

        missing = to_be_eliminated.loc[
            # (~to_be_eliminated.index.isin(soon_erased.index))
            # |
            (~to_be_eliminated["_row_idx"].isin(soon_erased["_row_idx"])),
            # | (~to_be_eliminated["_row_idx"].isin(soon_erased.index)),
            "geometry",
        ]

        if 0:
            from ..geopandas_tools.conversion import to_gdf
            from ..maps.maps import explore, explore_locals

            display(pairs)
            display(soon_erased.index.unique())
            display(soon_erased._row_idx.unique())
            display(to_be_eliminated.index.unique())
            display(to_be_eliminated._row_idx.unique())
            display(missing.index.unique())

            display(soon_erased)
            display(to_be_eliminated)
            display(missing)

            explore(
                to_gdf(soon_erased, 25833), intersecting=to_gdf(intersecting, 25833)
            )
            for j, ((i, g), (i2, g2)) in enumerate(
                zip(intersecting.items(), soon_erased.geometry.items())
            ):
                explore(
                    to_gdf(g, 25833).assign(ii=i, j=j),
                    g2=to_gdf(g2, 25833).assign(ii=i2, j=j),
                )

        if 0:
            explore(to_gdf(to_be_eliminated.iloc[[16]]))
            explore(to_gdf(to_be_eliminated.iloc[[15]]))
            explore(to_gdf(to_be_eliminated.iloc[[0]]))
            print("hei")
            explore(to_gdf(soon_erased.loc[soon_erased.index == 16]))
            explore(to_gdf(soon_erased.loc[soon_erased.index == 36]))

            explore(to_gdf(soon_erased.loc[soon_erased._row_idx == 16]))
            explore(to_gdf(soon_erased.loc[soon_erased._row_idx == 36]))

        # allign and aggregate by dissolve index to not get duplicates in difference
        intersecting.index = soon_erased.index
        soon_erased = soon_erased.geometry.groupby(level=0).agg(
            lambda x: unary_union(x, grid_size=grid_size)
        )
        intersecting = intersecting.groupby(level=0).agg(
            lambda x: unary_union(x, grid_size=grid_size)
        )

        # from ..maps.maps import explore_locals
        # explore_locals()

        soon_erased.loc[:] = difference(
            soon_erased.values,
            intersecting.values,
        )

        eliminated["geometry"] = (
            pd.concat([eliminators, soon_erased, missing])
            .groupby(level=0)
            .agg(
                lambda x: make_valid(
                    unary_union(x.dropna().values, grid_size=grid_size)
                )
            )
        )

    else:
        eliminated["geometry"] = many_hits.groupby("_dissolve_idx")["geometry"].agg(
            lambda x: make_valid(unary_union(x.values, grid_size=grid_size))
        )

    # setting crs on the GeometryArrays to avoid warning in concat
    not_to_dissolve.geometry.values.crs = crs
    eliminated.geometry.values.crs = crs
    one_hit.geometry.values.crs = crs

    to_concat = [not_to_dissolve, eliminated, one_hit]

    assert all(df.index.name == "_dissolve_idx" for df in to_concat)

    return pd.concat(to_concat).sort_index().drop(columns="_to_eliminate")


def close_thin_holes(gdf: GeoDataFrame, tolerance: int | float) -> GeoDataFrame:
    gdf = make_all_singlepart(gdf)
    holes = get_holes(gdf)
    inside_holes = sfilter(gdf, holes, predicate="within").unary_union

    def to_none_if_thin(geoms):
        try:
            buffered_in = buffer(
                difference(polygons(geoms), inside_holes), -(tolerance / 2)
            )
            return np.where(is_empty(buffered_in), None, geoms)
        except ValueError as e:
            if not len(geoms):
                return geoms
            raise e

    if not (gdf.geom_type == "Polygon").all():
        raise ValueError(gdf.geom_type.value_counts())

    return PolygonsAsRings(gdf).apply_numpy_func_to_interiors(to_none_if_thin).to_gdf()


def return_correct_geometry_object(in_obj, out_obj):
    if isinstance(in_obj, GeoDataFrame):
        in_obj.geometry = out_obj
        return in_obj
    elif isinstance(in_obj, GeoSeries):
        return GeoSeries(out_obj, crs=in_obj.crs)
    else:
        return out_obj


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
            Note that ignoring islands is a lot faster.

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

    if not len(gdf):
        return gdf

    if copy:
        gdf = gdf.copy()

    gdf = make_all_singlepart(gdf)

    if ignore_islands:
        geoms = gdf.geometry if isinstance(gdf, GeoDataFrame) else gdf
        holes_closed = make_valid(polygons(get_exterior_ring(geoms)))
        if isinstance(gdf, GeoDataFrame):
            gdf.geometry = holes_closed
            return gdf
        elif isinstance(gdf, GeoSeries):
            return GeoSeries(holes_closed, crs=gdf.crs)
        else:
            return holes_closed

    all_geoms = make_valid(gdf.unary_union)
    if isinstance(gdf, GeoDataFrame):
        gdf.geometry = gdf.geometry.map(
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
            Note that ignoring islands is a lot faster.
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

    if not len(gdf):
        return gdf

    if copy:
        gdf = gdf.copy()

    gdf = make_all_singlepart(gdf)

    if not ignore_islands:
        all_geoms = make_valid(gdf.unary_union)

        if isinstance(gdf, GeoDataFrame):
            gdf.geometry = gdf.geometry.map(
                lambda x: _close_small_holes_no_islands(x, max_area, all_geoms)
            )
            return gdf
        else:
            return gdf.map(
                lambda x: _close_small_holes_no_islands(x, max_area, all_geoms)
            )
    else:
        geoms = (
            gdf.geometry.to_numpy() if isinstance(gdf, GeoDataFrame) else gdf.to_numpy()
        )
        exteriors = get_exterior_ring(geoms)
        assert len(exteriors) == len(geoms)

        max_rings = max(get_num_interior_rings(geoms))

        if not max_rings:
            return gdf

        # looping through max for all geoms since arrays must be equal length
        interiors = np.array(
            [[get_interior_ring(geom, i) for i in range(max_rings)] for geom in geoms]
        )
        assert interiors.shape == (len(geoms), max_rings), interiors.shape

        areas = area(polygons(interiors))
        interiors[(areas < max_area) | np.isnan(areas)] = None

        results = polygons(exteriors, interiors)

        if isinstance(gdf, GeoDataFrame):
            gdf.geometry = results
            return gdf
        else:
            return GeoSeries(results, crs=gdf.crs)


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
            try:
                no_islands = unary_union(hole.difference(all_geoms))
            except GEOSException:
                no_islands = make_valid(unary_union(hole.difference(all_geoms)))

            if area(no_islands) < max_area:
                holes_closed.append(no_islands)

    return make_valid(unary_union(holes_closed))


def _close_all_holes_no_islands(poly, all_geoms):
    """Closes all holes within one shapely geometry of polygons."""

    # start with a list containing the polygon,
    # then append all holes smaller than 'max_km2' to the list.
    holes_closed = [poly]
    singlepart = get_parts(poly)
    for part in singlepart:
        n_interior_rings = get_num_interior_rings(part)

        for n in range(n_interior_rings):
            hole = polygons(get_interior_ring(part, n))
            try:
                no_islands = unary_union(hole.difference(all_geoms))
            except GEOSException:
                no_islands = make_valid(unary_union(hole.difference(all_geoms)))

            holes_closed.append(no_islands)

    return make_valid(unary_union(holes_closed))


def get_gaps(
    gdf: GeoDataFrame,
    include_interiors: bool = False,
    grid_size: float | int | None = None,
) -> GeoDataFrame:
    """Get the gaps between polygons.

    Args:
        gdf: GeoDataFrame of polygons.
        include_interiors: If False (default), the holes inside individual polygons
            will not be included as gaps.

    Note:
        See get_holes to find holes inside singlepart polygons.

    Returns:
        GeoDataFrame of polygons with only a geometry column.
    """
    if not len(gdf):
        return GeoDataFrame({"geometry": []}, crs=gdf.crs)

    if not include_interiors:
        gdf = close_all_holes(gdf)

    bbox = GeoDataFrame(
        {"geometry": [box(*tuple(gdf.total_bounds)).buffer(1)]}, crs=gdf.crs
    )

    bbox_diff = make_all_singlepart(
        clean_overlay(
            bbox, gdf, how="difference", geom_type="polygon", grid_size=grid_size
        )
    )

    # remove the outer "gap", i.e. the surrounding area
    bbox_ring = get_exterior_ring(bbox.geometry.values)
    without_outer_ring = sfilter_inverse(bbox_diff, bbox_ring)
    return without_outer_ring.reset_index(drop=True)


def get_holes(gdf: GeoDataFrame, as_polygons=True) -> GeoDataFrame:
    """Get the holes inside polygons.

    Args:
        gdf: GeoDataFrame of polygons.
        as_polygons: If True (default), the holes will be returned as polygons.
            If False, they will be returned as LinearRings.

    Note:
        See get_gaps to find holes/gaps between undissolved polygons.

    Returns:
        GeoDataFrame of polygons or linearrings with only a geometry column.
    """
    if not len(gdf):
        return GeoDataFrame({"geometry": []}, index=gdf.index, crs=gdf.crs)

    def as_linearring(x):
        return x

    astype = polygons if as_polygons else as_linearring

    geoms = make_all_singlepart(gdf.geometry).to_numpy()

    rings = [
        GeoSeries(astype(get_interior_ring(geoms, i)), crs=gdf.crs)
        for i in range(max(get_num_interior_rings(geoms)))
    ]

    return (
        GeoDataFrame({"geometry": (pd.concat(rings).pipe(clean_geoms).sort_index())})
        if rings
        else GeoDataFrame({"geometry": []}, crs=gdf.crs)
    )

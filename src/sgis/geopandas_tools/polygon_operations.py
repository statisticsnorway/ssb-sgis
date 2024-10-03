"""Functions for polygon geometries."""

from collections.abc import Callable

import networkx as nx
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from shapely import STRtree
from shapely import area
from shapely import box
from shapely import buffer
from shapely import difference
from shapely import extract_unique_points
from shapely import get_exterior_ring
from shapely import get_interior_ring
from shapely import get_num_geometries
from shapely import get_num_interior_rings
from shapely import get_parts
from shapely import is_empty
from shapely import make_valid
from shapely import polygons
from shapely import union_all
from shapely.errors import GEOSException
from shapely.geometry import LinearRing
from shapely.ops import SplitOp

from ..debug_config import _DEBUG_CONFIG
from ..debug_config import _try_debug_print
from ..maps.maps import explore_locals
from .conversion import to_gdf
from .conversion import to_geoseries
from .duplicates import _get_intersecting_geometries
from .general import _grouped_unary_union
from .general import _parallel_unary_union
from .general import _parallel_unary_union_geoseries
from .general import _push_geom_col
from .general import _unary_union_for_notna
from .general import clean_geoms
from .general import extend_lines
from .general import get_grouped_centroids
from .general import get_line_segments
from .general import to_lines
from .geometry_types import get_geom_type
from .geometry_types import make_all_singlepart
from .geometry_types import to_single_geom_type
from .neighbors import get_neighbor_indices
from .overlay import _try_difference
from .overlay import clean_overlay
from .polygons_as_rings import PolygonsAsRings
from .sfilter import sfilter
from .sfilter import sfilter_inverse

PRECISION = 1e-3
_BUFFER = False


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
        predicate: Spatial predicate. Defaults to "intersects".
        as_string: Whether to return the cluster column values as a string with x and y
            coordinates. Convinient to always get unique ids.
            Defaults to False because of speed.

    Returns:
        One or more GeoDataFrames (same amount as was given) with a new cluster column.

    Examples:
    ---------
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


def get_cluster_mapper(
    gdf: GeoDataFrame | GeoSeries, predicate: str = "intersects"
) -> dict[int, int]:
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
    gdf: GeoDataFrame | tuple[GeoDataFrame],
    to_eliminate: GeoDataFrame,
    *,
    fix_double: bool = True,
    ignore_index: bool = False,
    aggfunc: str | dict | list | None = None,
    grid_size=None,
    n_jobs: int = 1,
    **kwargs,
) -> tuple[GeoDataFrame]:
    """Dissolves selected polygons with the longest bordering neighbor polygon.

    Eliminates selected geometries by dissolving them with the neighboring
    polygon with the longest shared border. The index and column values of the
    large polygons will be kept, unless else is specified.

    Note that this might be a lot slower than eliminate_by_largest.

    Args:
        gdf: GeoDataFrame with polygon geometries, or a list of GeoDataFrames.
        to_eliminate: The geometries to be eliminated by 'gdf'.
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
        grid_size: Rounding of the coordinates. Defaults to None.
        n_jobs: Number of threads to use. Defaults to 1.
        **kwargs: Keyword arguments passed to the dissolve method.

    Returns:
        A tuple of the GeoDataFrame with the geometries of 'to_eliminate'
        dissolved in and a GeoDataFrame with the potentionally isolated
        polygons that could not be eliminated. If multiple GeoDataFrame
        are passed as 'gdf', the returned tuple will contain each frame
        plus the isolated polygons as the last item.

    Examples:
    ---------
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

    >>> small_poly_eliminated, large_poly_eliminated, isolated = sg.eliminate_by_longest(
    ...     [small_poly, large_poly], sliver
    ... )

    With only one input GeoDataFrame:

    >>> polys = pd.concat([small_poly, large_poly])
    >>> eliminated, isolated = sg.eliminate_by_longest(polys, sliver)
    """
    _recurse = kwargs.pop("_recurse", False)

    if not len(to_eliminate) or not len(gdf):
        if isinstance(gdf, (list, tuple)):
            return (*gdf, to_eliminate)
        return gdf, to_eliminate

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

    # TODO: is it ok to singlepart here?
    gdf = make_all_singlepart(gdf, ignore_index=True).pipe(
        to_single_geom_type, "polygon"
    )

    if _BUFFER:
        gdf.geometry = gdf.buffer(
            PRECISION,
            resolution=1,
            join_style=2,
        )

        to_eliminate.geometry = to_eliminate.buffer(
            PRECISION,
            resolution=1,
            join_style=2,
        )

    # more_than_one = get_num_geometries(to_eliminate.geometry.values) > 1
    # filt = more_than_one, to_eliminate._geometry_column_name
    # to_eliminate.loc[*filt] = to_eliminate.loc[*filt].apply(_unary_union_for_notna)

    gdf["_dissolve_idx"] = gdf.index
    to_eliminate = to_eliminate.assign(_eliminate_idx=lambda x: range(len(x)))

    # convert to lines to get the borders
    lines_eliminate = to_lines(to_eliminate[["_eliminate_idx", "geometry"]])

    borders = clean_overlay(
        gdf[["_dissolve_idx", "geometry"]],
        lines_eliminate,
        keep_geom_type=False,
        grid_size=grid_size,
        n_jobs=n_jobs,
    ).loc[lambda x: x["_eliminate_idx"].notna()]

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
    isolated = isolated.drop(
        ["_dissolve_idx", "_length", "_eliminate_idx"],
        axis=1,
        errors="ignore",
    )

    eliminated = _eliminate(
        pd.DataFrame(gdf),
        pd.concat([actually_eliminate, containing_eliminators]),
        aggfunc,
        crs,
        fix_double,
        grid_size=grid_size,
        n_jobs=n_jobs,
        **kwargs,
    )

    if not ignore_index:
        eliminated.index = eliminated.index.map(idx_mapper)
        eliminated.index.name = idx_name

    eliminated = eliminated.drop(
        ["_dissolve_idx", "_length", "_eliminate_idx"],
        axis=1,
        errors="ignore",
    )

    out = GeoDataFrame(eliminated, geometry="geometry", crs=crs).pipe(clean_geoms)

    if _BUFFER:
        out.geometry = out.buffer(
            -PRECISION,
            resolution=1,
            join_style=2,
        )
        isolated.geometry = isolated.buffer(
            -PRECISION,
            resolution=1,
            join_style=2,
        )

    if geom_type != "mixed":
        out = to_single_geom_type(out, geom_type)

    out = out.reset_index(drop=True) if ignore_index else out

    _try_debug_print("inni eliminate_by_longest")
    explore_locals(center=_DEBUG_CONFIG["center"])

    if not _recurse and len(isolated):
        if 0:
            isolated.geometry = isolated.buffer(
                -PRECISION,
                resolution=1,
                join_style=2,
            )
        out, isolated = _recursively_eliminate_new_neighbors(
            out,
            isolated,
            func=eliminate_by_longest,
            fix_double=fix_double,
            ignore_index=ignore_index,
            aggfunc=aggfunc,
            grid_size=grid_size,
            n_jobs=n_jobs,
        )

    _try_debug_print("inni eliminate_by_longest 2")
    explore_locals(center=_DEBUG_CONFIG["center"])

    # assert (
    #     out[["ARTYPE", "ARTRESLAG", "ARSKOGBON", "ARGRUNNF", "kilde"]]
    #     .notna()
    #     .all()
    #     .all()
    # ), out[["ARTYPE", "ARTRESLAG", "ARSKOGBON", "ARGRUNNF", "kilde"]].sort_values(
    #     ["ARTYPE", "ARTRESLAG", "ARSKOGBON", "ARGRUNNF", "kilde"]
    # )

    if not was_multiple_gdfs:
        return out, isolated

    gdfs = ()
    for i, cols in enumerate(original_cols):
        df = out.loc[out["_df_idx"] == i, cols]
        gdfs += (df,)

    return (*gdfs, isolated)


def _recursively_eliminate_new_neighbors(
    df: GeoDataFrame,
    isolated: GeoDataFrame,
    func: Callable,
    **kwargs,
):
    len_now = len(isolated)
    while len(isolated):
        _try_debug_print(f"recurse len({len(isolated)})")
        df, isolated = func(
            df,
            isolated,
            _recurse=True,
            **kwargs,
        )

        if len_now == len(isolated):
            break
        len_now = len(isolated)

    return df, isolated


def eliminate_by_largest(
    gdf: GeoDataFrame | list[GeoDataFrame],
    to_eliminate: GeoDataFrame,
    *,
    max_distance: int | float | None = None,
    fix_double: bool = True,
    ignore_index: bool = False,
    aggfunc: str | dict | list | None = None,
    predicate: str = "intersects",
    grid_size=None,
    n_jobs: int = 1,
    **kwargs,
) -> tuple[GeoDataFrame]:
    """Dissolves selected polygons with the largest neighbor polygon.

    Eliminates selected geometries by dissolving them with the neighboring
    polygon with the largest area. The index and column values of the
    large polygons will be kept, unless else is specified.

    Args:
        gdf: GeoDataFrame with polygon geometries, or a list of GeoDataFrames.
        to_eliminate: The geometries to be eliminated by 'gdf'.
        max_distance: Max distance to search for neighbors. Defaults to None, meaning
            0.
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
        grid_size: Rounding of the coordinates. Defaults to None.
        n_jobs: Number of threads to use. Defaults to 1.
        **kwargs: Keyword arguments passed to the dissolve method.

    Returns:
        A tuple of the GeoDataFrame with the geometries of 'to_eliminate'
        dissolved in and a GeoDataFrame with the potentionally isolated
        polygons that could not be eliminated. If multiple GeoDataFrame
        are passed as 'gdf', the returned tuple will contain each frame
        plus the isolated polygons as the last item.

    Examples:
    ---------
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

    >>> small_poly_eliminated, large_poly_eliminated, isolated = sg.eliminate_by_largest(
    ...     [small_poly, large_poly], sliver
    ... )

    With only one input GeoDataFrame:

    >>> polys = pd.concat([small_poly, large_poly])
    >>> eliminated, isolated = sg.eliminate_by_largest(polys, sliver)
    """
    return _eliminate_by_area(
        gdf,
        to_eliminate=to_eliminate,
        max_distance=max_distance,
        ignore_index=ignore_index,
        sort_ascending=False,
        aggfunc=aggfunc,
        predicate=predicate,
        fix_double=fix_double,
        grid_size=grid_size,
        n_jobs=n_jobs,
        **kwargs,
    )


def eliminate_by_smallest(
    gdf: GeoDataFrame | list[GeoDataFrame],
    to_eliminate: GeoDataFrame,
    *,
    max_distance: int | float | None = None,
    ignore_index: bool = False,
    aggfunc: str | dict | list | None = None,
    predicate: str = "intersects",
    fix_double: bool = True,
    grid_size=None,
    n_jobs: int = 1,
    **kwargs,
) -> tuple[GeoDataFrame]:
    return _eliminate_by_area(
        gdf,
        to_eliminate=to_eliminate,
        max_distance=max_distance,
        ignore_index=ignore_index,
        sort_ascending=True,
        aggfunc=aggfunc,
        predicate=predicate,
        fix_double=fix_double,
        grid_size=grid_size,
        n_jobs=n_jobs,
        **kwargs,
    )


def _eliminate_by_area(
    gdf: GeoDataFrame,
    to_eliminate: GeoDataFrame,
    max_distance: int | float | None,
    sort_ascending: bool,
    ignore_index: bool = False,
    aggfunc: str | dict | list | None = None,
    predicate="intersects",
    fix_double: bool = True,
    grid_size=None,
    n_jobs: int = 1,
    **kwargs,
) -> GeoDataFrame:
    _recurse = kwargs.pop("_recurse", False)

    if not len(to_eliminate) or not len(gdf):
        return gdf, to_eliminate

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
        idx_mapper_to_eliminate = dict(enumerate(to_eliminate.index))
        idx_name_to_eliminate = to_eliminate.index.name

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
        gdf,
        notna,
        aggfunc,
        crs,
        fix_double=fix_double,
        grid_size=grid_size,
        n_jobs=n_jobs,
        **kwargs,
    )

    eliminated = eliminated.drop(
        ["_dissolve_idx", "_area", "_eliminate_idx", "_dissolve_idx"],
        axis=1,
        errors="ignore",
    )

    out = GeoDataFrame(
        eliminated,
        geometry="geometry",
        crs=crs,
    ).pipe(clean_geoms)

    isolated = (
        GeoDataFrame(
            joined.loc[joined["_dissolve_idx"].isna()], geometry="geometry", crs=crs
        )
        .drop(
            ["_dissolve_idx", "_area", "_eliminate_idx", "_dissolve_idx"],
            axis=1,
            errors="ignore",
        )
        .pipe(clean_geoms)
    )

    if not ignore_index:
        out.index = out.index.map(idx_mapper)
        out.index.name = idx_name
        isolated.index = isolated.index.map(idx_mapper_to_eliminate)
        isolated.index.name = idx_name_to_eliminate

    if geom_type != "mixed":
        out = to_single_geom_type(out, geom_type)

    out = out.reset_index(drop=True) if ignore_index else out

    if not _recurse and len(isolated):
        out, isolated = _recursively_eliminate_new_neighbors(
            out,
            isolated,
            func=_eliminate_by_area,
            max_distance=max_distance,
            sort_ascending=sort_ascending,
            fix_double=fix_double,
            predicate=predicate,
            ignore_index=ignore_index,
            aggfunc=aggfunc,
            grid_size=grid_size,
            n_jobs=n_jobs,
        )

    if not was_multiple_gdfs:
        return out, isolated

    for k, v in locals().items():
        try:
            print(k, v.columns)
        except Exception:
            pass

    gdfs = ()
    for i, cols in enumerate(original_cols):
        df = out.loc[out["_df_idx"] == i, cols]
        gdfs += (df,)

    return (*gdfs, isolated)


def _eliminate(
    gdf, to_eliminate, aggfunc, crs, fix_double, grid_size, n_jobs, **kwargs
):
    if not len(to_eliminate):
        return gdf

    gdf["_range_idx_elim"] = range(len(gdf))

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

        # all_geoms: pd.Series = gdf.set_index("_dissolve_idx").geometry
        all_geoms: pd.Series = gdf.geometry

        # more_than_one = get_num_geometries(all_geoms.values) > 1
        # all_geoms.loc[more_than_one] = all_geoms.loc[more_than_one].apply(
        #     _unary_union_for_notna
        # )

        # more_than_one = get_num_geometries(to_be_eliminated.values) > 1
        # to_be_eliminated.loc[more_than_one, "geometry"] = to_be_eliminated.loc[
        #     more_than_one, "geometry"
        # ].apply(_unary_union_for_notna)

        # create DataFrame of intersection pairs
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

        # allign and aggregate by dissolve index to not get duplicates in difference
        intersecting.index = soon_erased.index

        soon_erased = _grouped_unary_union(soon_erased, level=0, grid_size=grid_size)
        intersecting = _grouped_unary_union(intersecting, level=0, grid_size=grid_size)

        assert soon_erased.index.equals(soon_erased.index)

        # soon_erased = soon_erased.geometry.groupby(level=0).agg(
        #     lambda x: unary_union(x, grid_size=grid_size)
        # )
        # intersecting = intersecting.groupby(level=0).agg(
        #     lambda x: unary_union(x, grid_size=grid_size)
        # )

        # explore_locals(center=_DEBUG_CONFIG["center"])

        soon_erased.loc[:] = _try_difference(
            soon_erased.to_numpy(),
            intersecting.to_numpy(),
            grid_size=grid_size,
            n_jobs=n_jobs,
            geom_type="polygon",
        )

        missing = _grouped_unary_union(missing, level=0, grid_size=grid_size)

        missing = make_all_singlepart(missing).loc[lambda x: x.area > 0]

        soon_erased = make_all_singlepart(soon_erased).loc[lambda x: x.area > 0]

        if 0:
            tree = STRtree(soon_erased.values)
            left, right = tree.query(missing.values, predicate="intersects")
            explore_locals(
                missing2=to_gdf(missing.to_numpy()[left], 25833),
                soon_erased2=to_gdf(soon_erased.to_numpy()[right], 25833),
                center=_DEBUG_CONFIG["center"],
            )
            missing = pd.Series(
                difference(
                    missing.to_numpy()[left],
                    soon_erased.to_numpy()[right],
                    grid_size=grid_size,
                ),
                index=left,
            ).loc[lambda x: (x.notna()) & (~is_empty(x))]

        soon_eliminated = pd.concat([eliminators, soon_erased, missing])
        more_than_one = get_num_geometries(soon_eliminated.values) > 1

        soon_eliminated.loc[more_than_one] = soon_eliminated.loc[more_than_one].apply(
            _unary_union_for_notna
        )

        if n_jobs > 1:
            eliminated["geometry"] = GeoSeries(
                _parallel_unary_union_geoseries(
                    soon_eliminated,
                    level=0,
                    grid_size=grid_size,
                    n_jobs=n_jobs,
                ),
                index=eliminated.index,
            )
        else:
            eliminated["geometry"] = _grouped_unary_union(soon_eliminated, level=0)
            # eliminated["geometry"] = soon_eliminated.groupby(level=0).agg(
            #     lambda x: make_valid(unary_union(x))
            # )

    else:
        if n_jobs > 1:
            eliminated["geometry"] = _parallel_unary_union(
                many_hits, by="_dissolve_idx", grid_size=grid_size, n_jobs=n_jobs
            )
        else:
            eliminated["geometry"] = _grouped_unary_union(many_hits, by="_dissolve_idx")

    # setting crs on the GeometryArrays to avoid warning in concat
    not_to_dissolve.geometry.values.crs = crs
    try:
        eliminated.geometry.values.crs = crs
    except AttributeError:
        pass
    one_hit.geometry.values.crs = crs

    to_concat = [not_to_dissolve, eliminated, one_hit]

    assert all(df.index.name == "_dissolve_idx" for df in to_concat)

    out = pd.concat(to_concat).sort_index()

    duplicated_geoms = _get_intersecting_geometries(
        GeoDataFrame(
            {
                "geometry": out.geometry.values,
                "_range_idx_elim_dups": out["_range_idx_elim"].values,
            },
        ),
        geom_type="polygon",
        keep_geom_type=True,
        n_jobs=n_jobs,
        predicate="intersects",
    ).pipe(clean_geoms)
    duplicated_geoms.geometry = duplicated_geoms.buffer(-PRECISION)
    duplicated_geoms = duplicated_geoms.pipe(clean_geoms)

    if len(duplicated_geoms):
        hits_in_original_df = duplicated_geoms.sjoin(
            GeoDataFrame(
                {
                    "geometry": gdf.geometry.values,
                    "_range_idx_elim": gdf["_range_idx_elim"].values,
                },
            ),
            how="inner",
        )

        should_be_erased = hits_in_original_df.loc[
            lambda x: x["_range_idx_elim"] != x["_range_idx_elim_dups"]
        ]

        should_be_erased_idx = list(
            sorted(should_be_erased["_range_idx_elim_dups"].unique())
        )
        should_erase = (
            should_be_erased.groupby("_range_idx_elim_dups")["geometry"]
            .agg(lambda x: make_valid(union_all(x)))
            .sort_index()
        )

        # aligining out with "should_erase" before rowwise difference
        out = out.sort_values("_range_idx_elim")
        assert out["_range_idx_elim"].is_unique
        to_be_erased_idx = out["_range_idx_elim"].isin(should_be_erased_idx)

        out.loc[to_be_erased_idx, "geometry"] = make_valid(
            difference(
                out.loc[
                    to_be_erased_idx,
                    "geometry",
                ].values,
                should_erase.values,
            )
        )

        from ..maps.maps import explore

        # display(hits_in_original_df)
        # display(should_be_erased.assign(area=lambda x: x.area))

        explore(
            gdf=to_gdf(gdf, 25833),
            out=to_gdf(out, 25833),
            should_be_erased=to_gdf(should_be_erased, 25833),
            duplicated_geoms=duplicated_geoms.set_crs(25833),
            eli=GeoDataFrame(
                {
                    "geometry": out.geometry.values,
                    "_range_idx_elim": out["_range_idx_elim"].values,
                },
                crs=25833,
            ),
            center=_DEBUG_CONFIG["center"],
        )

    _try_debug_print("inni _eliminate")
    _try_debug_print(duplicated_geoms)
    explore_locals(center=_DEBUG_CONFIG["center"])

    return out.drop(columns=["_to_eliminate", "_range_idx_elim"])


def clean_dissexp(df: GeoDataFrame, dissolve_func: Callable, **kwargs) -> GeoDataFrame:
    """Experimental."""
    original_points = GeoDataFrame(
        {"geometry": get_parts(extract_unique_points(df.geometry.values))}
    )[lambda x: ~x.geometry.duplicated()]

    dissolved = df.copy()

    try:
        dissolved.geometry = dissolved.buffer(PRECISION, resolution=1, join_style=2)
    except AttributeError as e:
        if isinstance(dissolved, GeoSeries):
            dissolved.loc[:] = dissolved.buffer(PRECISION, resolution=1, join_style=2)
        else:
            raise e

    dissolved = dissolve_func(dissolved, **kwargs)

    try:
        dissolved.geometry = dissolved.buffer(-PRECISION, resolution=1, join_style=2)
    except AttributeError as e:
        if isinstance(dissolved, GeoSeries):
            dissolved.loc[:] = dissolved.buffer(-PRECISION, resolution=1, join_style=2)
        else:
            raise e

    dissolved = dissolved.loc[lambda x: ~x.geometry.is_empty]
    dissolved = dissolved.explode(ignore_index=True)

    original_points = sfilter_inverse(original_points, dissolved.buffer(-PRECISION))

    snapped = (
        PolygonsAsRings(
            dissolved.geometry,
        )
        .apply_numpy_func(
            _snap_points_back,
            kwargs={"snap_to": original_points, "tolerance": PRECISION},
        )
        .to_numpy()
    )

    try:
        dissolved.geometry = snapped
    except AttributeError as e:
        if isinstance(dissolved, GeoSeries):
            dissolved.loc[:] = snapped
        else:
            raise e

    return dissolved


def _snap_points_back(rings, snap_to, tolerance):
    points = GeoDataFrame({"geometry": extract_unique_points(rings)})
    points = points.explode(index_parts=True)

    snap_to["geom_right"] = snap_to.geometry
    nearest = points.sjoin_nearest(snap_to, max_distance=tolerance)
    points.loc[nearest.index, points.geometry.name] = nearest["geom_right"]

    new_rings = points.groupby(level=0)[points.geometry.name].agg(LinearRing)
    return new_rings


def close_thin_holes(gdf: GeoDataFrame, tolerance: int | float) -> GeoDataFrame:
    gdf = make_all_singlepart(gdf)
    holes = get_holes(gdf)
    inside_holes = union_all(sfilter(gdf, holes, predicate="within").geometry.values)

    def to_none_if_thin(geoms):
        if not len(geoms):
            return geoms
        try:
            polys = polygons(geoms)
        except GEOSException:
            polys = make_valid(polygons(make_valid(geoms)))
        if inside_holes is not None:
            polys = difference(polys, inside_holes)
        buffered_in = buffer(polys, -(tolerance / 2))
        return np.where(is_empty(buffered_in), None, geoms)

    if not (gdf.geom_type == "Polygon").all():
        raise ValueError(gdf.geom_type.value_counts())

    return PolygonsAsRings(gdf).apply_numpy_func_to_interiors(to_none_if_thin).to_gdf()


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

    Examples:
    ---------
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

    all_geoms = make_valid(union_all(gdf.geometry.values))
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

    Examples:
    ---------
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
        all_geoms = make_valid(union_all(gdf.geometry.values))

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
                no_islands = union_all(hole.difference(all_geoms))
            except GEOSException:
                no_islands = make_valid(union_all(hole.difference(all_geoms)))

            if area(no_islands) < max_area:
                holes_closed.append(no_islands)

    return make_valid(union_all(holes_closed))


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
                no_islands = union_all(hole.difference(all_geoms))
            except GEOSException:
                no_islands = make_valid(union_all(hole.difference(all_geoms)))

            holes_closed.append(no_islands)

    return make_valid(union_all(holes_closed))


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
        grid_size: Rounding of the coordinates.

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


def get_holes(gdf: GeoDataFrame, as_polygons: bool = True) -> GeoDataFrame:
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


def split_polygons_by_lines(polygons: GeoSeries, lines: GeoSeries) -> GeoSeries:
    idx_mapper = dict(enumerate(polygons.index))
    idx_name = polygons.index.name
    polygons = polygons.copy()
    polygons.index = range(len(polygons))

    # use pandas to explode faster (from list instead of GeoSeries.explode)
    splitted = pd.Series(polygons.geometry.to_numpy())
    lines = to_geoseries(lines)
    lines.index = range(len(lines))

    # find intersection pairs to split relevant polygon for each line
    tree = STRtree(splitted.values)
    left, right = tree.query(lines.values, predicate="intersects")
    pairs = pd.Series(right, index=left)

    lines = lines.loc[lambda x: x.index.isin(pairs.index)]

    for i, line in lines.items():
        intersecting = pairs.loc[[i]].values
        try:
            splitted.loc[intersecting] = splitted.loc[intersecting].apply(
                lambda poly: SplitOp._split_polygon_with_line(poly, line) or poly
            )
        except TypeError:
            # if we got multipolygon
            splitted = splitted.apply(get_parts).explode()
            splitted.loc[intersecting] = splitted.loc[intersecting].apply(
                lambda poly: SplitOp._split_polygon_with_line(poly, line) or poly
            )
        splitted = splitted.explode()

    if isinstance(polygons, GeoDataFrame):
        polygons = polygons.loc[splitted.index]
        polygons.geometry = splitted
        polygons.index = polygons.index.map(idx_mapper)
        polygons.index.name = idx_name
        return polygons
    else:
        splitted.index = splitted.index.map(idx_mapper)
        splitted.index.name = idx_name
        return splitted


def split_by_neighbors(
    df: GeoDataFrame,
    split_by: GeoDataFrame,
    tolerance: int | float,
    grid_size: float | int | None = None,
) -> GeoDataFrame:
    if not len(df):
        return df

    df = make_all_singlepart(df)

    split_by = split_by.copy()

    intersecting_lines = (
        clean_overlay(
            to_lines(split_by),
            df.buffer(tolerance).to_frame("geometry"),
            how="intersection",
            grid_size=grid_size,
        )
        .pipe(get_line_segments)
        .reset_index(drop=True)
    )

    endpoints = intersecting_lines.boundary.explode(index_parts=False)

    lines = extend_lines(
        endpoints.loc[lambda x: ~x.index.duplicated(keep="first")].values,
        endpoints.loc[lambda x: ~x.index.duplicated(keep="last")].values,
        distance=tolerance * 3,
    )

    return split_polygons_by_lines(df, lines)

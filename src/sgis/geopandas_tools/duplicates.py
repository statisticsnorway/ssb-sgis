from collections.abc import Iterable

import networkx as nx
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from shapely import STRtree
from shapely import difference
from shapely import simplify

from ..conf import _get_instance
from ..conf import config
from .general import _determine_geom_type_args
from .general import _push_geom_col
from .general import clean_geoms
from .geometry_types import get_geom_type
from .geometry_types import make_all_singlepart
from .geometry_types import to_single_geom_type
from .overlay import clean_overlay
from .runners import OverlayRunner
from .runners import RTreeQueryRunner
from .runners import UnionRunner
from .sfilter import sfilter_inverse

PRECISION = 1e-3


def update_geometries(
    gdf: GeoDataFrame,
    geom_type: str | None = None,
    keep_geom_type: bool | None = None,
    grid_size: int | None = None,
    predicate: str | None = "intersects",
    n_jobs: int = 1,
    union_runner: UnionRunner | None = None,
    rtree_runner: RTreeQueryRunner | None = None,
    overlay_runner: OverlayRunner | None = None,
) -> GeoDataFrame:
    """Puts geometries on top of each other rowwise.

    IMPORTANT: Since this operation is done rowwise, meaning the top rows
    are put on top of bottommore rows, it is important to first sort the
    GeoDataFrame approriately. See examples below.

    Args:
        gdf: The GeoDataFrame to be updated.
        keep_geom_type: If True, return only geometries of original type in case
            of intersection resulting in multiple geometry types or
            GeometryCollections. If False, return all resulting geometries
            (potentially mixed types).
        geom_type: Optionally specify what geometry type to keep.,
            if there are mixed geometry types. Must be either "polygon",
            "line" or "point".
        grid_size: Precision grid size to round the geometries. Will use the highest
            precision of the inputs by default.
        predicate: Spatial predicate for the spatial tree.
        n_jobs: Number of workers.
        union_runner: Optionally debug/manipulate the spatial union operations.
            See the 'runners' module for example implementations.
        rtree_runner: Optionally debug/manipulate the spatial indexing operations.
            See the 'runners' module for example implementations.
        overlay_runner: Optionally debug/manipulate the spatial overlay operations.
            See the 'runners' module for example implementations.

    Example:
    --------
    Create some overlapping circles and update the geometries based on area.
    Sorting the data small to large might be the appropriate choice if you want to keep all details
    in the coverage and the data has no attributes that should be given priority over another.

    >>> import sgis as sg
    >>> coords = [(0, 0), (0, 1), (1, 1), (1, 0)]
    >>> buffers = [0.9, 1.3, 0.7, 1.1]
    >>> circles = sg.to_gdf(coords)
    >>> circles["geometry"] = circles["geometry"].buffer(buffers)
    >>> updated_smallest_first = sg.update_geometries(sg.sort_small_first(circles))
    >>> updated_largest_first = sg.update_geometries(sg.sort_large_first(circles))

    Compare the results in an interactive map::

        sg.explore(circles, updated_smallest_first, updated_largest_first, tiles=["dark"])

    If you want to prioritize geometries based on attributes,

    >>> circles["hva"] = ["skog", "elv", "bro", "tunnel"]

    Create two circles and get the overlap.

    >>> import sgis as sg
    >>> circles = sg.to_gdf([(0, 0), (1, 1)]).pipe(sg.buff, 1)
    >>> duplicates = sg.get_intersections(circles)
    >>> duplicates
                                                geometry
    0  POLYGON ((0.05234 0.99863, 0.10453 0.99452, 0....
    1  POLYGON ((0.94766 0.00137, 0.89547 0.00548, 0....

    The polygons are identical except for the order of the coordinates.

    >>> poly1, poly2 = duplicates.geometry
    >>> poly1.equals(poly2)
    True

    'update_geometries' gives different results based on the order
    of the GeoDataFrame.

    >>> sg.update_geometries(duplicates)
                                                geometry
    0  POLYGON ((0.05234 0.99863, 0.10453 0.99452, 0....

    >>> dups_rev = duplicates.iloc[::-1]
    >>> sg.update_geometries(dups_rev)
                                                geometry
    1  POLYGON ((0.94766 0.00137, 0.89547 0.00548, 0....

    It might be appropriate to put the largest polygons on top
    and sort all NaNs to the bottom.

    >>> updated = (
    ...     sg.sort_large_first(duplicates)
    ...     .pipe(sg.sort_nans_last)
    ...     .pipe(sg.update_geometries)
    ... )
    >>> updated
                                                geometry
    0  POLYGON ((0.05234 0.99863, 0.10453 0.99452, 0....

    """
    if len(gdf) <= 1:
        return gdf

    if rtree_runner is None:
        rtree_runner = _get_instance(config, "rtree_runner", n_jobs=n_jobs)
    if union_runner is None:
        union_runner = _get_instance(config, "union_runner", n_jobs=n_jobs)
    if overlay_runner is None:
        overlay_runner = _get_instance(config, "overlay_runner", n_jobs=n_jobs)

    if geom_type == "polygon" or get_geom_type(gdf) == "polygon":
        gdf.geometry = gdf.buffer(0)

    copied = make_all_singlepart(clean_geoms(gdf))

    copied, geom_type, keep_geom_type = _determine_geom_type_args(
        copied, geom_type, keep_geom_type
    )

    geom_col = copied._geometry_column_name
    index_mapper = dict(enumerate(copied.index))
    copied = copied.reset_index(drop=True)

    left, right = rtree_runner.run(
        copied.geometry.values, copied.geometry.values, predicate=predicate
    )
    indices = pd.Series(right, index=left).loc[lambda x: x.index > x.values]

    # select geometries from 'right', index from 'left', dissolve by 'left'
    erasers = pd.Series(copied.geometry.loc[indices.values].values, index=indices.index)
    only_one = erasers.groupby(level=0).transform("size") == 1
    one_hit = erasers[only_one]
    many_hits = union_runner.run(erasers[~only_one], level=0, grid_size=grid_size)
    erasers = pd.concat([one_hit, many_hits]).sort_index()

    # match up the aggregated erasers by index
    arr1 = copied.geometry.loc[erasers.index].to_numpy()
    arr2 = erasers.to_numpy()
    erased = overlay_runner.run(
        difference, arr1, arr2, grid_size=grid_size, geom_type=geom_type
    )

    erased = GeoSeries(erased, index=erasers.index)
    copied.loc[erased.index, geom_col] = erased
    copied = copied.loc[~copied.is_empty]
    copied.index = copied.index.map(index_mapper)
    copied = make_all_singlepart(copied)

    # TODO check why polygons dissappear in rare cases. For now, just add back the missing
    dissapeared = sfilter_inverse(
        gdf, copied.buffer(-PRECISION), rtree_runner=rtree_runner
    )
    copied = pd.concat([copied, dissapeared])

    # TODO fix dupliates again with dissolve?
    # dups = get_intersections(copied, geom_type="polygon")
    # dups["_cluster"] = get_cluster_mapper(dups.geometry.values)
    # no_dups = dissexp(dups, by="_cluster").drop(columns="_cluster")
    # copied = clean_overlay(copied, no_dups, how="update", geom_type="polygon")

    if keep_geom_type:
        copied = to_single_geom_type(copied, geom_type)

    return copied


def get_intersections(
    gdf: GeoDataFrame,
    geom_type: str | None = None,
    keep_geom_type: bool | None = None,
    predicate: str | None = "intersects",
    grid_size: float | None = None,
    **kwargs,
) -> GeoDataFrame:
    """Find geometries that intersect in a GeoDataFrame.

    Does an intersection with itself and keeps only the geometries that appear
    more than once.

    Note that the returned GeoDataFrame in most cases contain two rows per
    intersection pair. It might also contain more than two overlapping polygons
    if there were multiple overlapping. These can be removed with
    update_geometries. See example below.

    Args:
        gdf: GeoDataFrame of polygons.
        geom_type: Optionally specify which geometry type to keep.
            Either "polygon", "line" or "point".
        keep_geom_type: Whether to keep the original geometry type.
            If mixed geometry types and keep_geom_type=True,
            an exception is raised.
        grid_size: Precision grid size to round the geometries. Will use the highest
            precision of the inputs by default.
        n_jobs: Number of threads.
        predicate: Spatial predicate for the spatial tree.
        **kwargs: Keyword arguments passed to clean_overlay.

    Returns:
        A GeoDataFrame of the overlapping polygons.

    Examples:
    ---------
    Create three partially overlapping polygons.

    >>> import sgis as sg
    >>> circles = sg.to_gdf([(0, 0), (1, 0), (2, 0)]).pipe(sg.buff, 1.2)
    >>> circles.area
    0    4.521827
    1    4.521827
    2    4.521827
    dtype: float64

    Get the duplicates.

    >>> duplicates = sg.get_intersections(circles)
    >>> duplicates["area"] = duplicates.area
    >>> duplicates
                                                geometry      area
    0  POLYGON ((1.19836 -0.0628, 1.19343 -0.12543, 1...  2.193761
    0  POLYGON ((1.19836 -0.0628, 1.19343 -0.12543, 1...  0.359361
    1  POLYGON ((0.45521 -1.06921, 0.4 -1.03923, 0.34...  2.193761
    1  POLYGON ((2.19836 -0.0628, 2.19343 -0.12543, 2...  2.193761
    2  POLYGON ((0.9936 -0.65357, 0.96077 -0.6, 0.930...  0.359361
    2  POLYGON ((1.45521 -1.06921, 1.4 -1.03923, 1.34...  2.193761

    We get two rows for each intersection pair.

    To get no overlapping geometries without , we can put geometries
    on top of each other rowwise.

    >>> updated = sg.update_geometries(duplicates)
    >>> updated["area"] = updated.area
    >>> updated
                                                geometry      area
    0  POLYGON ((1.19836 -0.0628, 1.19343 -0.12543, 1...  2.193761
    1  POLYGON ((2.19343 -0.12543, 2.18523 -0.18772, ...  1.834401

    It might be appropriate to sort the dataframe by columns.
    Or put large polygons first and NaN values last.

    >>> updated = (
    ...     sg.sort_large_first(duplicates)
    ...     .pipe(sg.sort_nans_last)
    ...     .pipe(sg.update_geometries)
    ... )
    >>> updated
                                                geometry      area
    1  POLYGON ((2.19836 -0.0628, 2.19343 -0.12543, 2...  2.193761
    0  POLYGON ((0.97082 -0.70534, 0.93258 -0.75518, ...  2.193761

    """
    if isinstance(gdf, GeoSeries):
        gdf = GeoDataFrame({"geometry": gdf}, crs=gdf.crs)
        was_geoseries = True
    else:
        was_geoseries = False

    gdf, geom_type, keep_geom_type = _determine_geom_type_args(
        gdf, geom_type, keep_geom_type
    )

    idx_name = gdf.index.name
    gdf = gdf.assign(orig_idx=gdf.index).reset_index(drop=True)

    duplicated_geoms = _get_intersecting_geometries(
        gdf,
        geom_type,
        keep_geom_type,
        grid_size=grid_size,
        predicate=predicate,
        **kwargs,
    ).pipe(clean_geoms)

    duplicated_geoms.index = duplicated_geoms["orig_idx"].values
    duplicated_geoms.index.name = idx_name

    if was_geoseries:
        return duplicated_geoms.geometry

    return duplicated_geoms.drop(columns="orig_idx")


def _get_intersecting_geometries(
    gdf: GeoDataFrame,
    geom_type: str | None,
    keep_geom_type: bool,
    grid_size: float | None = None,
    predicate: str | None = None,
    **kwargs,
) -> GeoDataFrame:
    right = gdf[[gdf._geometry_column_name]]
    right["idx_right"] = right.index

    left = (
        gdf.copy()
        if not any("index_" in str(col) for col in gdf)
        else gdf.loc[:, lambda x: x.columns.difference({"index_right", "index_left"})]
    )
    left["idx_left"] = left.index

    def are_not_identical(df):
        return df["idx_left"] != df["idx_right"]

    if geom_type or get_geom_type(gdf) != "mixed":
        intersected = clean_overlay(
            left,
            right,
            how="intersection",
            predicate=predicate,
            grid_size=grid_size,
            geom_type=geom_type,
            keep_geom_type=keep_geom_type,
            **kwargs,
        ).loc[are_not_identical]
    else:
        if keep_geom_type:
            raise ValueError(
                "Cannot set keep_geom_type=True when the geom_type is mixed."
            )
        gdf = make_all_singlepart(gdf)
        intersected = []
        for geom_type in ["polygon", "line", "point"]:
            if not len(to_single_geom_type(gdf, geom_type)):
                continue
            intersected += [
                clean_overlay(
                    left,
                    right,
                    how="intersection",
                    grid_size=grid_size,
                    predicate=predicate,
                    geom_type=geom_type,
                    **kwargs,
                )
            ]
        intersected = pd.concat(intersected, ignore_index=True).loc[are_not_identical]

    # make sure it's correct by sjoining a point inside the polygons
    points_joined = (
        # large and very detailed geometries can dissappear with small negative buffer
        simplify(intersected.geometry, 1e-3)
        .buffer(-1e-3)
        # intersected.geometry
        .representative_point()
        .to_frame()
        .sjoin(intersected)
    )

    duplicated_points = points_joined.loc[points_joined.index.duplicated(keep=False)]

    out = intersected.loc[intersected.index.isin(duplicated_points.index)].drop(
        columns=["idx_left", "idx_right"]
    )

    # return out

    # some polygons within polygons are not counted in the intersection
    within = (
        gdf.assign(_range_idx_inters_left=lambda x: range(len(x)))
        .sjoin(
            GeoDataFrame(
                {
                    # "geometry": polygons(
                    #     get_exterior_ring(gdf.geometry.values)
                    # ),  # buffer(1e-6).values)),
                    "geometry": gdf.geometry.values,
                    # "geometry": buffer(simplify(gdf.geometry.values, 1e-3), 1e-6),
                    "_range_idx_inters_right": range(len(gdf)),
                },
                crs=gdf.crs,
            ),
            how="inner",
            predicate="within",
        )
        .loc[lambda x: x["_range_idx_inters_left"] != x["_range_idx_inters_right"]]
        .drop(
            columns=["index_right", "_range_idx_inters_left", "_range_idx_inters_right"]
        )
        .pipe(sfilter_inverse, out.buffer(-PRECISION))
    )

    return pd.concat([out, within])


def _drop_duplicate_geometries(gdf: GeoDataFrame, **kwargs) -> GeoDataFrame:
    """Drop geometries that are considered equal.

    Args:
        gdf: GeoDataFrame.
        **kwargs: Keyword arguments passed to pandas.DataFrame.drop_duplicates

    Returns:
        The GeoDataFrame with duplicate geometries dropped.
    """
    return (
        _get_duplicate_geometry_groups(gdf, group_col="dup_idx_")
        .drop_duplicates("dup_idx_", **kwargs)
        .drop(columns="dup_idx_")
    )


def _get_duplicate_geometry_groups(
    gdf: GeoDataFrame, group_col: str | Iterable[str] = "duplicate_index"
):
    idx_mapper = dict(enumerate(gdf.index))
    idx_name = gdf.index.name

    gdf = gdf.reset_index(drop=True)

    tree = STRtree(gdf.geometry.values)
    left, right = tree.query(gdf.geometry.values, predicate="within")

    edges = list(zip(left, right, strict=False))

    graph = nx.Graph()
    graph.add_edges_from(edges)

    component_mapper = {
        j: i
        for i, component in enumerate(nx.connected_components(graph))
        for j in component
    }

    gdf[group_col] = component_mapper

    gdf = _push_geom_col(gdf)

    gdf.index = gdf.index.map(idx_mapper)
    gdf.index.name = idx_name

    return gdf

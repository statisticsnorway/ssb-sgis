from collections.abc import Iterable

import networkx as nx
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from shapely import STRtree
from shapely import difference
from shapely import make_valid
from shapely import simplify
from shapely.errors import GEOSException

from .general import _determine_geom_type_args
from .general import _grouped_unary_union
from .general import _parallel_unary_union_geoseries
from .general import _push_geom_col
from .general import clean_geoms
from .geometry_types import get_geom_type
from .geometry_types import make_all_singlepart
from .geometry_types import to_single_geom_type
from .overlay import _run_overlay_dask
from .overlay import clean_overlay
from .overlay import make_valid_and_keep_geom_type
from .sfilter import sfilter_inverse

PRECISION = 1e-3


def update_geometries(
    gdf: GeoDataFrame,
    geom_type: str | None = None,
    keep_geom_type: bool | None = None,
    grid_size: int | None = None,
    n_jobs: int = 1,
    predicate: str | None = "intersects",
) -> GeoDataFrame:
    """Puts geometries on top of each other rowwise.

    Since this operation is done rowwise, it's important to
    first sort the GeoDataFrame approriately. See example below.

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
        n_jobs: Number of threads.
        predicate: Spatial predicate for the spatial tree.

    Example:
    --------
    Create two circles and get the overlap.

    >>> import sgis as sg
    >>> circles = sg.to_gdf([(0, 0), (1, 1)]).pipe(sg.buff, 1)
    >>> duplicates = sg.get_intersections(circles)
    >>> duplicates
       idx                                           geometry
    0    1  POLYGON ((0.03141 0.99951, 0.06279 0.99803, 0....
    1    2  POLYGON ((1.00000 0.00000, 0.96859 0.00049, 0....

    The polygons are identical except for the order of the coordinates.

    >>> poly1, poly2 = duplicates.geometry
    >>> poly1.equals(poly2)
    True

    'update_geometries' gives different results based on the order
    of the GeoDataFrame.

    >>> sg.update_geometries(duplicates)
        idx                                           geometry
    0    1  POLYGON ((0.03141 0.99951, 0.06279 0.99803, 0....

    >>> dups_rev = duplicates.iloc[::-1]
    >>> sg.update_geometries(dups_rev)
        idx                                           geometry
    1    2  POLYGON ((1.00000 0.00000, 0.96859 0.00049, 0....

    It might be appropriate to put the largest polygons on top
    and sort all NaNs to the bottom.

    >>> updated = (
    ...     sg.sort_large_first(duplicates)
    ...     .pipe(sg.sort_nans_last)
    ...     .pipe(sg.update_geometries)
    >>> updated
        idx                                           geometry
    0    1  POLYGON ((0.03141 0.99951, 0.06279 0.99803, 0....

    """
    if len(gdf) <= 1:
        return gdf

    copied = make_all_singlepart(clean_geoms(gdf))

    copied, geom_type, keep_geom_type = _determine_geom_type_args(
        copied, geom_type, keep_geom_type
    )

    geom_col = copied._geometry_column_name
    index_mapper = {i: idx for i, idx in enumerate(copied.index)}
    copied = copied.reset_index(drop=True)

    tree = STRtree(copied.geometry.values)
    left, right = tree.query(copied.geometry.values, predicate=predicate)
    indices = pd.Series(right, index=left).loc[lambda x: x.index > x.values]

    # select geometries from 'right', index from 'left', dissolve by 'left'
    erasers = pd.Series(copied.geometry.loc[indices.values].values, index=indices.index)
    if n_jobs > 1:
        erasers = _parallel_unary_union_geoseries(
            erasers,
            level=0,
            n_jobs=n_jobs,
            grid_size=grid_size,
        )
        erasers = pd.Series(erasers, index=indices.index.unique())
    else:
        only_one = erasers.groupby(level=0).transform("size") == 1
        one_hit = erasers[only_one]
        many_hits = _grouped_unary_union(
            erasers[~only_one], level=0, grid_size=grid_size
        )
        erasers = pd.concat([one_hit, many_hits]).sort_index()

    # match up the aggregated erasers by index
    if n_jobs > 1:
        arr1 = copied.geometry.loc[erasers.index].to_numpy()
        arr2 = erasers.to_numpy()
        try:
            erased = _run_overlay_dask(
                arr1, arr2, func=difference, n_jobs=n_jobs, grid_size=grid_size
            )
        except GEOSException:
            arr1 = make_valid_and_keep_geom_type(
                arr1, geom_type=geom_type, n_jobs=n_jobs
            )
            arr2 = make_valid_and_keep_geom_type(
                arr2, geom_type=geom_type, n_jobs=n_jobs
            )
            erased = _run_overlay_dask(
                arr1, arr2, func=difference, n_jobs=n_jobs, grid_size=grid_size
            )
        erased = GeoSeries(erased, index=erasers.index)
    else:
        erased = make_valid(
            difference(
                copied.geometry.loc[erasers.index],
                erasers,
                grid_size=grid_size,
            )
        )

    copied.loc[erased.index, geom_col] = erased

    copied = copied.loc[~copied.is_empty]

    copied.index = copied.index.map(index_mapper)

    # TODO check why polygons dissappear in rare cases. For now, just add back the missing
    dissapeared = sfilter_inverse(gdf, copied.buffer(-PRECISION))
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
    n_jobs: int = 1,
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
        n_jobs: Number of threads.
        predicate: Spatial predicate for the spatial tree.

    Returns:
        A GeoDataFrame of the overlapping polygons.

    Examples:
    ---------
    Create three partially overlapping polygons.

    >>> import sgis as sg
    >>> circles = sg.to_gdf([(0, 0), (1, 0), (2, 0)]).pipe(sg.buff, 1.2)
    >>> circles.area
    0    4.523149
    1    4.523149
    2    4.523149
    dtype: float64

    Get the duplicates.

    >>> duplicates = sg.get_intersections(circles)
    >>> duplicates["area"] = duplicates.area
    >>> duplicates
                                                geometry      area
    0  POLYGON ((1.19941 -0.03769, 1.19763 -0.07535, ...  2.194730
    0  POLYGON ((1.19941 -0.03769, 1.19763 -0.07535, ...  0.359846
    1  POLYGON ((0.48906 -1.08579, 0.45521 -1.06921, ...  2.194730
    1  POLYGON ((2.19941 -0.03769, 2.19763 -0.07535, ...  2.194730
    2  POLYGON ((0.98681 -0.64299, 0.96711 -0.61085, ...  0.359846
    2  POLYGON ((1.48906 -1.08579, 1.45521 -1.06921, ...  2.194730

    We get two rows for each intersection pair.

    To get no overlapping geometries without , we can put geometries
    on top of each other rowwise.

    >>> updated = sg.update_geometries(duplicates)
    >>> updated["area"] = updated.area
    >>> updated
           area                                           geometry
    0  2.194730  POLYGON ((1.19941 -0.03769, 1.19763 -0.07535, ...
    1  1.834884  POLYGON ((2.19763 -0.07535, 2.19467 -0.11293, ...

    It might be appropriate to sort the dataframe by columns.
    Or put large polygons first and NaN values last.

    >>> updated = (
    ...     sg.sort_large_first(duplicates)
    ...     .pipe(sg.sort_nans_last)
    ...     .pipe(sg.update_geometries)
    ... )
    >>> updated
          area                                           geometry
    0  2.19473  POLYGON ((1.19941 -0.03769, 1.19763 -0.07535, ...
    1  2.19473  POLYGON ((2.19763 -0.07535, 2.19467 -0.11293, ...

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
        n_jobs=n_jobs,
        predicate=predicate,
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
    n_jobs: int,
    predicate: str | None,
) -> GeoDataFrame:
    right = gdf[[gdf._geometry_column_name]]
    right["idx_right"] = right.index

    left = (
        gdf
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
            geom_type=geom_type,
            keep_geom_type=keep_geom_type,
            n_jobs=n_jobs,
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
                    predicate=predicate,
                    geom_type=geom_type,
                    n_jobs=n_jobs,
                )
            ]
        intersected = pd.concat(intersected, ignore_index=True).loc[are_not_identical]

    # make sure it's correct by sjoining a point inside the polygons
    points_joined = (
        # large and very detailed geometries can dissappear with small negative buffer
        simplify(intersected.geometry, 1e-3)
        .buffer(-1e-3)
        .representative_point()
        .to_frame()
        .sjoin(intersected)
    )

    duplicated_points = points_joined.loc[points_joined.index.duplicated(keep=False)]

    out = intersected.loc[intersected.index.isin(duplicated_points.index)].drop(
        columns=["idx_left", "idx_right"]
    )

    # some polygons within polygons are not counted in the
    within = (
        gdf.assign(_range_idx_inters_left=lambda x: range(len(x)))
        .sjoin(
            GeoDataFrame(
                {
                    "geometry": gdf.buffer(1e-6).values,
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

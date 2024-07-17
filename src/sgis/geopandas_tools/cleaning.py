# %%
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from numpy.typing import NDArray
from shapely import Geometry
from shapely import STRtree
from shapely import extract_unique_points
from shapely import polygons
from shapely.errors import GEOSException
from shapely.geometry import LinearRing
from shapely.geometry import LineString
from shapely.geometry import Point

from ..debug_config import _DEBUG_CONFIG
from ..maps.maps import explore
from ..maps.maps import explore_locals
from ..networkanalysis.cutting_lines import split_lines_by_nearest_point
from .buffer_dissolve_explode import buff, dissexp
from .buffer_dissolve_explode import dissexp_by_cluster
from .conversion import to_gdf
from .conversion import to_geoseries
from .duplicates import get_intersections
from .duplicates import update_geometries
from .general import clean_geoms
from .general import make_lines_between_points
from .general import to_lines
from .geometry_types import make_all_singlepart
from .geometry_types import to_single_geom_type
from .overlay import clean_overlay
from .polygon_operations import close_all_holes
from .polygon_operations import eliminate_by_longest
from .polygon_operations import get_cluster_mapper
from .polygon_operations import get_gaps
from .polygon_operations import split_by_neighbors
from .polygons_as_rings import PolygonsAsRings
from .sfilter import sfilter
from .sfilter import sfilter_inverse
from .sfilter import sfilter_split

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


# def explore(*args, **kwargs) -> None:
#     pass


# def print(*args) -> None:
#     pass


# def explore_locals(*args, **kwargs) -> None:
#     pass


PRECISION = 1e-3
BUFFER_RES = 50


def coverage_clean(
    gdf: GeoDataFrame,
    tolerance: int | float,
    mask: GeoDataFrame | GeoSeries | Geometry | None = None,
    **kwargs,
) -> GeoDataFrame:
    """Fix thin gaps, holes, slivers and double surfaces.

    The operations might raise GEOSExceptions, so it might be nessecary to set
    the 'grid_sizes' argument, it might also be a good idea to run coverage_clean
    twice to fill gaps resulting from these GEOSExceptions.

    Rules:
    - Holes (interiors) thinner than the tolerance are closed.
    - Gaps between polygons are filled if thinner than the tolerance.
    - Sliver polygons thinner than the tolerance are eliminated
    into the neighbor polygon with the longest shared border.
    - Double surfaces thinner than the tolerance are eliminated.
    If duplicate_action is "fix", thicker double surfaces will
    be updated.
    - Line and point geometries are removed with no warning.
    - MultiPolygons and GeometryCollections are exploded to Polygons.
    - Index is reset.

    Args:
        gdf: GeoDataFrame to be cleaned.
        tolerance: distance (usually meters) used as the minimum thickness
            for polygons to be eliminated. Any gap, hole, sliver or double
            surface that are empty after a negative buffer of tolerance / 2
            are eliminated into the neighbor with the longest shared border.
        duplicate_action: Either "fix", "error" or "ignore".
            If "fix" (default), double surfaces thicker than the
            tolerance will be updated from top to bottom (function update_geometries)
            and then dissolved into the neighbor polygon with the longest shared border.
            If "error", an Exception is raised if there are any double surfaces thicker
            than the tolerance. If "ignore", double surfaces are kept as is.
        grid_sizes: One or more grid_sizes used in overlay and dissolve operations that
            might raise a GEOSException. Defaults to (None,), meaning no grid_sizes.
        n_jobs: Number of threads.
        mask: Mask to clip gdf to.

    Returns:
        A GeoDataFrame with cleaned polygons.
    """
    gdf_original = gdf.copy()

    # more_than_one = (gdf.count_geometries() > 1).values
    # gdf.loc[more_than_one, gdf._geometry_column_name] = gdf.loc[
    #     more_than_one, gdf._geometry_column_name
    # ].apply(_unary_union_for_notna)

    if mask is None:
        mask: GeoDataFrame = close_all_holes(
            dissexp_by_cluster(gdf[["geometry"]])
        ).pipe(make_all_singlepart)
    else:
        try:
            mask: GeoDataFrame = mask[["geometry"]].pipe(make_all_singlepart)
        except Exception:
            mask: GeoDataFrame = (
                to_geoseries(mask).to_frame("geometry").pipe(make_all_singlepart)
            )

    gdf = snap_polygons(gdf, tolerance, mask=mask)

    explore(
        gdf,
        gdf_original,
        msk=mask,
        points=gdf.extract_unique_points().explode().to_frame(),
        center=_DEBUG_CONFIG["center"],
    )

    if 0:
        gdf.geometry = (
            gdf.buffer(
                -PRECISION,
                resolution=1,
                join_style=2,
            )
            .buffer(
                PRECISION * 2,
                resolution=1,
                join_style=2,
            )
            .buffer(
                -PRECISION,
                resolution=1,
                join_style=2,
            )
        )
        gdf = clean_geoms(gdf)
        print("etter buffer")
        explore(
            gdf,
            gdf_original,
            msk=mask,
            points=gdf.extract_unique_points().explode().to_frame(),
            center=_DEBUG_CONFIG["center"],
        )

    gdf["_gdf_range_idx"] = range(len(gdf))

    missing_from_gdf = clean_overlay(
        gdf_original, gdf, how="difference", geom_type="polygon"
    )
    is_thin = missing_from_gdf.buffer(-tolerance / 2).is_empty
    thin_missing_from_gdf, thick_missing_from_gdf = (
        missing_from_gdf[is_thin],
        missing_from_gdf[~is_thin],
    )

    is_thin = gdf.buffer(-tolerance / 2).is_empty
    thin, gdf = gdf[is_thin], gdf[~is_thin]
    to_eliminate = pd.concat([thin_missing_from_gdf, thin], ignore_index=True)

    print("split_by_neighbors gdf thin")
    gdf, isolated = split_and_eliminate_by_longest(gdf, to_eliminate, tolerance)

    gdf = _eliminate_not_really_isolated(gdf, isolated)

    print("etter eliminate 1 (med gdf)")
    explore(
        gdf_original,
        gdf=gdf,
        isolated=isolated,
        to_eliminate=to_eliminate,
        dissexped=dissexp(gdf, by="_gdf_range_idx", dropna=False, as_index=False),
        points=gdf.extract_unique_points().explode().to_frame(),
        center=_DEBUG_CONFIG["center"],
    )

    gdf = dissexp(gdf, by="_gdf_range_idx", dropna=False, as_index=False)

    # we don't want the thick polygons from the mask
    thin_missing_from_mask = clean_overlay(
        mask, gdf, how="difference", geom_type="polygon"
    ).loc[lambda x: x.buffer(-tolerance / 2).is_empty]

    print("split_by_neighbors 2 mask")
    gdf, isolated = split_and_eliminate_by_longest(
        gdf, thin_missing_from_mask, tolerance
    )
    gdf = _eliminate_not_really_isolated(gdf, isolated)

    # if 0:
    #     thin_missing_from_mask.geometry = thin_missing_from_mask.buffer(
    #         -PRECISION,
    #         resolution=1,
    #         join_style=2,
    #     ).buffer(
    #         PRECISION,
    #         resolution=1,
    #         join_style=2,
    #     )
    # print("split_by_neighbors 2")

    # single_neighbored, multi_neighbored = (
    #     _separate_single_neighbored_from_multi_neighoured_geometries(
    #         thin_missing_from_mask, gdf
    #     )
    # )

    # multi_neighbored = split_by_neighbors(multi_neighbored, gdf, tolerance=tolerance)
    # thin_missing_from_mask = pd.concat([multi_neighbored, single_neighbored])
    # thin_missing_from_mask["_was_to_eliminate"] = 1
    # print("eliminate_by_longest again with mask")
    # gdf_between = gdf.copy()
    # gdf, isolated = eliminate_by_longest(
    #     gdf, thin_missing_from_mask, return_isolated=True
    # )
    # gdf = gdf.explode(ignore_index=True)
    # assert not len(isolated), (explore(isolated, gdf, browser=True), isolated)

    print("etter eliminate 2 (av mask)")
    explore(
        gdf=gdf,
        thin_missing_from_mask=thin_missing_from_mask,
        thin_missing_from_gdf=thin_missing_from_gdf,
        thick_missing_from_gdf=thick_missing_from_gdf,
        thin=thin,
        to_eliminate=to_eliminate,
        center=_DEBUG_CONFIG["center"],
    )

    gdf = clean_overlay(gdf, mask, how="intersection", geom_type="polygon")

    if 0:
        mmm3 = sfilter(gdf, to_gdf([5.37027276, 59.00997572], 4326).to_crs(25833))
        explore(mmm3)
        was_to_eliminate = gdf["_was_to_eliminate"] == 1
        still_not_eliminated = gdf[was_to_eliminate]

        still_not_eliminated.geometry = still_not_eliminated.buffer(0.01)

        still_not_eliminated = clean_overlay(
            still_not_eliminated,
            mask,
            how="intersection",
            geom_type="polygon",
        )
        explore(
            gdf,
            still_not_eliminated,
            msk=mask,
            center=_DEBUG_CONFIG["center"],
            # center=_DEBUG_CONFIG["center"],
        )

        gdf = (
            eliminate_by_longest(
                gdf[~was_to_eliminate], still_not_eliminated, remove_isolated=True
            ).explode(ignore_index=True)
            # .pipe(clean_overlay, mask, geom_type="polygon")
            # .pipe(update_geometries, geom_type="polygon")
        ).drop(columns="_was_to_eliminate")

        explore(
            gdf,
            msk=mask,
            center=_DEBUG_CONFIG["center"],
            # center=_DEBUG_CONFIG["center"],
        )

        mm4 = sfilter(
            gdf, to_gdf([5.37027276, 59.00997572], 4326).to_crs(25833).buffer(1)
        )
        explore(mm4)
    # else:
    #     gdf = gdf.drop(columns="_was_to_eliminate")

    if 0:
        gdf.geometry = gdf.buffer(
            PRECISION,
            resolution=1,
            join_style=2,
        ).buffer(
            -PRECISION,
            resolution=1,
            join_style=2,
        )

        explore(
            gdf,
            msk=mask,
            # center=_DEBUG_CONFIG["center"],
        )

    # return pd.concat([gdf, thick_missing_from_gdf], ignore_index=True)

    gdf = pd.concat([gdf, thick_missing_from_gdf], ignore_index=True)

    dissappeared = sfilter_inverse(gdf_original, gdf.buffer(-PRECISION)).loc[
        lambda x: ~x.buffer(-PRECISION).is_empty
    ]

    return pd.concat([gdf, dissappeared], ignore_index=True).drop(
        columns="_gdf_range_idx"
    )


def snap_polygons(
    gdf: GeoDataFrame,
    tolerance: int | float,
    mask: GeoDataFrame | GeoSeries | Geometry | None = None,
    **kwargs,
) -> GeoDataFrame:
    if not len(gdf):
        return gdf.copy()

    gdf_orig = gdf.copy()

    crs = gdf.crs

    gdf = (
        clean_geoms(gdf)
        .pipe(make_all_singlepart, ignore_index=True)
        .pipe(to_single_geom_type, "polygon")
    )

    gdf.crs = None

    # gdf = close_thin_holes(gdf, tolerance)

    if mask is None:
        mask: GeoDataFrame = (
            close_all_holes(dissexp_by_cluster(gdf))
            # .dissolve()
            .pipe(make_all_singlepart)
        )
    else:
        try:
            mask: GeoDataFrame = mask[["geometry"]].pipe(make_all_singlepart)
        except Exception:
            mask: GeoDataFrame = (
                to_geoseries(mask).to_frame("geometry").pipe(make_all_singlepart)
            )
        mask.crs = None

    # donuts_without_spikes = (
    #     gdf.geometry.buffer(tolerance / 2, resolution=1, join_style=2)
    #     .buffer(-tolerance, resolution=1, join_style=2)
    #     .buffer(tolerance / 2, resolution=1, join_style=2)
    #     .pipe(to_lines)
    #     .buffer(tolerance)
    # )

    gdf.geometry = (
        PolygonsAsRings(gdf.geometry.values)
        .apply_numpy_func(
            _snap_linearrings,
            kwargs=dict(
                tolerance=tolerance,
                mask=mask,
                # donuts_without_spikes=donuts_without_spikes,
            ),
        )
        .to_numpy()
    )

    gdf = (
        to_single_geom_type(make_all_singlepart(clean_geoms(gdf)), "polygon")
        .reset_index(drop=True)
        .set_crs(crs)
    )

    explore(
        gdf_orig,
        gdf,
        dups=get_intersections(gdf, geom_type="polygon"),
        msk=mask,
        gaps=get_gaps(gdf),
        updated=update_geometries(gdf, geom_type="polygon"),
        # browser=True,
    )

    gdf = update_geometries(gdf, geom_type="polygon")

    return gdf  # .pipe(clean_clip, mask, geom_type="polygon")


def _snap_linearrings(
    geoms: NDArray[LinearRing],
    tolerance: int | float,
    mask: GeoDataFrame | None = None,
    gaps=None,
    # donuts_without_spikes=None,
):
    if not len(geoms):
        return geoms

    idx_start = len(mask)

    gdf = GeoDataFrame(
        {
            "geometry": geoms,
            "_geom_idx": np.arange(idx_start, len(geoms) + idx_start),
        }
    )

    as_polygons = GeoDataFrame(
        {
            "geometry": polygons(geoms),
            "_geom_idx": np.arange(idx_start, len(geoms) + idx_start),
        }
    )

    is_thin = as_polygons.buffer(-tolerance / 2).is_empty

    # as_polygons.geometry = as_polygons.geometry.buffer(-PRECISION)

    polygon_mapper: GeoSeries = as_polygons.set_index("_geom_idx")["geometry"]

    thin = is_thin[lambda x: x == True]
    thin.loc[:] = None
    thin.index = thin.index.map(gdf["_geom_idx"])

    as_polygons["_is_thin"] = is_thin
    as_polys = as_polygons.copy()

    gdf = gdf.loc[is_thin == False]
    as_polygons = as_polygons.loc[is_thin == False]

    points: GeoDataFrame = (
        gdf.assign(geometry=lambda x: extract_unique_points(x.geometry.values)).explode(
            ignore_index=True
        )
        # .pipe(sfilter, donuts_without_spikes)
    )

    not_snapped = (
        points.sort_index()
        .set_index("_geom_idx")
        .loc[lambda x: x.groupby(level=0).size() > 2]
        .groupby(level=0)["geometry"]
        .agg(LinearRing)
    )

    mask_nodes = GeoDataFrame(
        {
            "geometry": extract_unique_points(mask.geometry),
            "_geom_idx": range(len(mask)),
        }
    ).explode(ignore_index=True)

    points["_is_snapped"] = False

    points0 = points.copy()

    if 0:
        points = _snap(points, mask_nodes, tolerance, as_polygons)

    points1 = points.copy()
    not_snapped1 = (
        points.sort_index()
        .set_index("_geom_idx")
        .pipe(_remove_legit_spikes)
        .loc[lambda x: x.groupby(level=0).size() > 2]
        .groupby(level=0)["geometry"]
        .agg(LinearRing)
    )

    mask = to_lines(mask).pipe(
        split_lines_by_nearest_point, points, max_distance=tolerance
    )

    mask_nodes = GeoDataFrame(
        {
            "geometry": extract_unique_points(mask.geometry),
            "_geom_idx": range(len(mask)),
        }
    ).explode(ignore_index=True)

    try:
        print("inni snap_polygons 0 ")
        explore(
            as_rings=to_gdf(
                polygons(
                    points.sort_index()
                    .set_index("_geom_idx")
                    .pipe(_remove_legit_spikes)
                    .loc[lambda x: x.groupby(level=0).size() > 2]
                    .groupby(level=0)["geometry"]
                    .agg(LinearRing)
                ),
                25833,
            ),
            as_polys=to_gdf(as_polys, 25833),
            center=_DEBUG_CONFIG["center"],
        )
    except GEOSException:
        pass

    if 0:
        points = _snap(points, mask_nodes, tolerance, as_polygons)

    points2 = points.copy()
    not_snapped2 = (
        points.sort_index()
        .set_index("_geom_idx")
        .loc[lambda x: x.groupby(level=0).size() > 2]
        .groupby(level=0)["geometry"]
        .agg(LinearRing)
    )

    try:
        print("inni snap_polygons 1 ")
        explore(
            as_rings=to_gdf(
                polygons(
                    points.sort_index()
                    .set_index("_geom_idx")
                    .pipe(_remove_legit_spikes)
                    .loc[lambda x: x.groupby(level=0).size() > 2]
                    .groupby(level=0)["geometry"]
                    .agg(LinearRing)
                ),
                25833,
            ),
            as_polys=to_gdf(as_polys, 25833),
            center=_DEBUG_CONFIG["center"],
        )
    except GEOSException:
        pass

    snapped, anchors = _snap_to_anchors(
        points,
        tolerance,
        anchors=mask_nodes,
        idx_start=idx_start,
        polygon_mapper=polygon_mapper,
        geoms=geoms,
        as_polygons=as_polygons,
        # polygon_mapper=gdf.set_index("_geom_idx")["geometry"],
    )  # anchors)

    as_rings = (
        snapped.sort_index()
        .set_index("_geom_idx")
        .pipe(_remove_legit_spikes)
        .loc[lambda x: x.groupby(level=0).size() > 2]
        .groupby(level=0)["geometry"]
        .agg(LinearRing)
    )

    try:
        print("inni snap_polygons 2 ")
        explore(
            points=to_gdf((points), 25833),
            points2=to_gdf((points2), 25833),
            points1=to_gdf((points1), 25833),
            points0=to_gdf((points0), 25833),
            snapped=to_gdf((snapped), 25833),
            # relevant_mask_nodes=to_gdf((relevant_mask_nodes), 25833),
            anchors=to_gdf((anchors), 25833),
            # lines_between_mask_and_node=to_gdf((lines_between_mask_and_node), 25833),
            not_snapped=to_gdf(polygons(not_snapped), 25833),
            not_snapped1=to_gdf(polygons(not_snapped1), 25833),
            not_snapped2=to_gdf(polygons(not_snapped2), 25833),
            as_rings=to_gdf(polygons(as_rings), 25833),
            as_rings_p=to_gdf(extract_unique_points(as_rings), 25833).explode(),
            as_polys=to_gdf(as_polys, 25833),
            msk=to_gdf(mask, 25833),
            mask_nodes=to_gdf(mask_nodes, 25833).explode(),
            center=_DEBUG_CONFIG["center"],
            # mask=to_gdf([5.37439002, 59.01144682], 4326).to_crs(25833).buffer(20),
        )
    except GEOSException:
        pass

    for idx in []:  # [1, 2, 3]:
        print(idx)
        explore(
            snapped=to_gdf(snapped[lambda x: x._geom_idx == idx].sort_index(), 25833),
            points=to_gdf((points[lambda x: x._geom_idx == idx]), 25833),
            points2=to_gdf((points2[lambda x: x._geom_idx == idx]), 25833),
            points1=to_gdf((points1[lambda x: x._geom_idx == idx]), 25833),
            points0=to_gdf((points0[lambda x: x._geom_idx == idx]), 25833),
            # relevant_mask_nodes=to_gdf((relevant_mask_nodes), 25833),
            anchors=to_gdf((anchors), 25833),
            # lines_between_mask_and_node=to_gdf((lines_between_mask_and_node), 25833),
            # not_snapped=to_gdf(polygons(not_snapped), 25833),
            as_rings=to_gdf(polygons(as_rings[lambda x: x.index == idx]), 25833),
            # as_rings=to_gdf(polygons(as_rings[lambda x: x.index == idx]), 25833),
            as_rings_p=to_gdf(
                extract_unique_points(as_rings[lambda x: x.index == idx]), 25833
            ).explode(),
            # geoms=to_gdf(polygons(geoms[idx]), 25833),
            center=_DEBUG_CONFIG["center"],
        )

    missing = gdf.set_index("_geom_idx")["geometry"].loc[
        lambda x: (~x.index.isin(as_rings.index.union(thin.index)))
        & (x.index >= idx_start)
    ]
    missing.loc[:] = None

    return pd.concat([as_rings, missing, thin]).sort_index()


def _snap_to_anchors(
    points: GeoDataFrame,
    tolerance: int | float,
    anchors: GeoDataFrame | None = None,
    custom_func: Callable | None = None,
    polygon_mapper: GeoSeries = None,
    idx_start=0,
    geoms=None,
    as_polygons=None,
) -> GeoDataFrame:
    if not len(points):
        try:
            return points, anchors[["geometry"]]
        except TypeError:
            return points, points[["geometry"]]

    assert points.index.is_unique

    tree = STRtree(points.loc[lambda x: x["_is_snapped"] == False, "geometry"].values)
    left, right = tree.query(
        points.loc[lambda x: x["_is_snapped"] == False, "geometry"].values,
        # points.geometry.values,
        predicate="dwithin",
        distance=tolerance,
    )
    indices = pd.Series(right, index=left, name="_right_idx")

    idx_mapper = dict(
        enumerate(points.loc[lambda x: x["_is_snapped"] == False, "_geom_idx"])
    )
    geom_idx_left = indices.index.map(idx_mapper)
    geom_idx_right = indices.map(idx_mapper)

    left_on_top = indices.loc[geom_idx_left < geom_idx_right].sort_index()

    # keep only indices from left if they have not already appeared in right
    # these shouldn't be anchors, but instead be snapped
    new_indices = []
    values = []
    right_indices = set()
    for left, right in left_on_top.items():
        if left not in right_indices:
            new_indices.append(left)
            values.append(right)
            right_indices.add(right)

    snap_indices = pd.Series(values, index=new_indices)

    if custom_func:
        snap_indices = custom_func(snap_indices)

    # new_anchors = points.loc[
    #     points.index.isin(snap_indices.index), ["geometry", "_geom_idx"]
    # ]

    only_neighbor_with_self = indices.loc[indices.groupby(level=0).size() == 1]
    # isolated_points = points.loc[
    #     points.index.isin(only_neighbor_with_self.index), ["geometry", "_geom_idx"]
    # ]

    neither_isolated_nor_by_anchors = points.loc[
        ~points.index.isin(
            snap_indices.index.union(only_neighbor_with_self.index).union(
                pd.Index(snap_indices.values)
            )
        ),
        ["geometry", "_geom_idx"],
    ]
    neither_isolated_nor_by_anchors["_cluster"] = get_cluster_mapper(
        neither_isolated_nor_by_anchors.buffer(tolerance / 2)
    )
    neither_isolated_nor_by_anchors = neither_isolated_nor_by_anchors.drop_duplicates(
        "_cluster"
    )

    new_anchors = points.loc[
        (points.index.isin(snap_indices.index.union(only_neighbor_with_self.index))),
        ["geometry", "_geom_idx"],
    ]
    new_anchors = pd.concat([new_anchors, neither_isolated_nor_by_anchors])

    # explore(
    #     points=to_gdf(points[lambda x: x._geom_idx == 1], 25833),
    #     new_anchors=to_gdf(new_anchors[lambda x: x._geom_idx == 1], 25833),
    #     snap_indices=to_gdf(
    #         points.loc[
    #             points.index.isin(snap_indices.index), ["geometry", "_geom_idx"]
    #         ][lambda x: x._geom_idx == 1],
    #         25833,
    #     ),
    #     isolated_points=to_gdf(
    #         points.loc[
    #             points.index.isin(only_neighbor_with_self.index),
    #             ["geometry", "_geom_idx"],
    #         ][lambda x: x._geom_idx == 1],
    #         25833,
    #     ),
    #     neither_isolated_nor_by_anchors=to_gdf(
    #         neither_isolated_nor_by_anchors[lambda x: x._geom_idx == 1],
    #         25833,
    #     ),
    # )

    if 1:
        new_anchors["_cluster"] = get_cluster_mapper(new_anchors.buffer(PRECISION))

        assert new_anchors["_geom_idx"].notna().all()

        no_longer_anchors: pd.Index = new_anchors.loc[
            lambda x: (x["_cluster"].duplicated())  # & (x["_geom_idx"] >= idx_start)
        ].index
        new_anchors = new_anchors.loc[lambda x: ~x.index.isin(no_longer_anchors)]

    explore(
        points=to_gdf(points, 25833),
        new_anchors=to_gdf(new_anchors, 25833),
        anchors=to_gdf(anchors, 25833),
        snap_indices=to_gdf(
            points.loc[
                points.index.isin(snap_indices.index), ["geometry", "_geom_idx"]
            ],
            25833,
        ),
        isolated_points=to_gdf(
            points.loc[
                points.index.isin(only_neighbor_with_self.index),
                ["geometry", "_geom_idx"],
            ],
            25833,
        ),
        neither_isolated_nor_by_anchors=to_gdf(
            neither_isolated_nor_by_anchors,
            25833,
        ),
        center=_DEBUG_CONFIG["center"],
    )

    if 0:
        new_anchors.geometry = shapely.set_precision(new_anchors.geometry, PRECISION)

    if anchors is not None:
        # anchors["_geometry"] = anchors.geometry
        # new_anchors["_geometry"] = new_anchors.geometry
        # anchors.geometry = anchors.buffer(tolerance / 2)
        # new_anchors.geometry = new_anchors.buffer(tolerance / 2)
        # anchors, new_anchors = get_polygon_clusters(
        #     anchors, new_anchors, cluster_col="_cluster"
        # )
        # new_anchors = new_anchors.loc[
        #     ~new_anchors["_cluster"].isin(anchors["_cluster"])
        # ]
        new_anchors["what"] = "new"
        anchors["what"] = "old"
        anchors = pd.concat([anchors, new_anchors], ignore_index=True).loc[
            lambda x: ~x.geometry.duplicated()
        ]
        anchors["_cluster"] = get_cluster_mapper(anchors.buffer(tolerance / 2))
        old_clusters = anchors.loc[lambda x: x["what"] == "old", "_cluster"]
        anchors = anchors.loc[
            lambda x: ~((x["what"] == "new") & (x["_cluster"].isin(old_clusters)))
        ]
        # anchors.geometry = anchors["_geometry"]
    else:
        anchors = new_anchors
        anchors["_was_anchor"] = 0

    anchors["_right_geom"] = anchors.geometry

    points = _add_midpoints_to_segments(
        points.sort_index(), anchors, tolerance, as_polygons
    )

    explore(
        points=to_gdf(points, 25833),
        new_anchors=to_gdf(new_anchors, 25833),
        anchors=to_gdf(anchors, 25833),
        snap_indices=to_gdf(
            points.loc[
                points.index.isin(snap_indices.index), ["geometry", "_geom_idx"]
            ],
            25833,
        ),
        isolated_points=to_gdf(
            points.loc[
                points.index.isin(only_neighbor_with_self.index),
                ["geometry", "_geom_idx"],
            ],
            25833,
        ),
        neither_isolated_nor_by_anchors=to_gdf(
            neither_isolated_nor_by_anchors,
            25833,
        ),
        center=_DEBUG_CONFIG["center"],
    )

    # explore(
    #     points=to_gdf(points[lambda x: x._geom_idx == 1], 25833),
    #     new_anchors=to_gdf(new_anchors[lambda x: x._geom_idx == 1], 25833),
    #     snap_indices=to_gdf(
    #         points.loc[
    #             points.index.isin(snap_indices.index), ["geometry", "_geom_idx"]
    #         ][lambda x: x._geom_idx == 1],
    #         25833,
    #     ),
    #     isolated_points=to_gdf(
    #         points.loc[
    #             points.index.isin(only_neighbor_with_self.index),
    #             ["geometry", "_geom_idx"],
    #         ][lambda x: x._geom_idx == 1],
    #         25833,
    #     ),
    #     neither_isolated_nor_by_anchors=to_gdf(
    #         neither_isolated_nor_by_anchors[lambda x: x._geom_idx == 1],
    #         25833,
    #     ),
    # )

    # should_be_snapped = (points.index.isin(snap_indices.values)) | (
    #     points.index.isin(no_longer_anchors)
    # )
    # if anchors is not None:
    #     should_be_snapped |= points.index.isin(
    #         sfilter(points, anchors.buffer(tolerance)).index
    #     )

    to_be_snapped = points.loc[
        lambda x: x["_is_snapped"] == False
    ]  # .loc[should_be_snapped].rename(
    #     columns={"_geom_idx": "_geom_idx_left"}, errors="raise"
    # )

    snapped = (
        to_be_snapped.sjoin_nearest(anchors, max_distance=tolerance)
        .sort_values("index_right")
        .loc[lambda x: ~x.index.duplicated()]
    )
    explore(
        points=to_gdf(points, 25833),
        snapped=to_gdf(snapped, 25833),
        to_be_snapped=to_gdf(to_be_snapped, 25833),
        new_anchors=to_gdf(new_anchors, 25833),
        anchors=to_gdf(anchors, 25833),
        snap_indices=to_gdf(
            points.loc[
                points.index.isin(snap_indices.index), ["geometry", "_geom_idx"]
            ],
            25833,
        ),
        isolated_points=to_gdf(
            points.loc[
                points.index.isin(only_neighbor_with_self.index),
                ["geometry", "_geom_idx"],
            ],
            25833,
        ),
        neither_isolated_nor_by_anchors=to_gdf(
            neither_isolated_nor_by_anchors,
            25833,
        ),
        center=_DEBUG_CONFIG["center"],
    )

    if 0:
        snapped.geometry = shapely.shortest_line(
            snapped.geometry.values, snapped["_right_geom"].values
        )
        snapped = sfilter_inverse(snapped, as_polygons)  # .loc[lambda x: ~x.intersects(
        #     as_.loc[lambda y: ~y.index.duplicated()].loc[midpoints.index].geometry
        # )]

    points.loc[snapped.index, "geometry"] = snapped["_right_geom"]

    explore(
        points=to_gdf(points, 25833),
        snapped=to_gdf(snapped, 25833),
        new_anchors=to_gdf(new_anchors, 25833),
        to_be_snapped=to_gdf(to_be_snapped, 25833),
        anchors=to_gdf(anchors, 25833),
        snap_indices=to_gdf(
            points.loc[
                points.index.isin(snap_indices.index), ["geometry", "_geom_idx"]
            ],
            25833,
        ),
        isolated_points=to_gdf(
            points.loc[
                points.index.isin(only_neighbor_with_self.index),
                ["geometry", "_geom_idx"],
            ],
            25833,
        ),
        neither_isolated_nor_by_anchors=to_gdf(
            neither_isolated_nor_by_anchors,
            25833,
        ),
        center=_DEBUG_CONFIG["center"],
    )

    return points, anchors[["geometry"]]


def _snap(
    points: GeoDataFrame,
    anchors: GeoDataFrame,
    tolerance: int | float,
    as_polygons: GeoDataFrame,
) -> GeoDataFrame:
    if not len(points):
        return points

    assert points.index.is_unique

    anchors["_right_geom"] = anchors.geometry

    points = _add_midpoints_to_segments(
        points.sort_index(), anchors, tolerance, as_polygons
    )

    # explore(
    #     points=to_gdf(points[lambda x: x._geom_idx == 1], 25833),
    #     new_anchors=to_gdf(new_anchors[lambda x: x._geom_idx == 1], 25833),
    #     snap_indices=to_gdf(
    #         points.loc[
    #             points.index.isin(snap_indices.index), ["geometry", "_geom_idx"]
    #         ][lambda x: x._geom_idx == 1],
    #         25833,
    #     ),
    #     isolated_points=to_gdf(
    #         points.loc[
    #             points.index.isin(only_neighbor_with_self.index),
    #             ["geometry", "_geom_idx"],
    #         ][lambda x: x._geom_idx == 1],
    #         25833,
    #     ),
    #     neither_isolated_nor_by_anchors=to_gdf(
    #         neither_isolated_nor_by_anchors[lambda x: x._geom_idx == 1],
    #         25833,
    #     ),
    # )

    # to_be_snapped = points.loc[should_be_snapped].rename(
    #     columns={"_geom_idx": "_geom_idx_left"}, errors="raise"
    # )

    snapped = (
        points.loc[lambda x: x["_is_snapped"] == False]
        .sjoin_nearest(anchors, max_distance=tolerance, distance_col="_dist")
        .loc[lambda x: x["_dist"] > 0]
        .sort_values("_dist")
        # .sort_values(["index_right"])
        # ["_right_geom"]
        .loc[lambda x: ~x.index.duplicated()]
    )
    snapped.geometry = shapely.shortest_line(
        snapped.geometry.values, snapped["_right_geom"].values
    )

    explore(
        points=to_gdf(points, 25833),
        snapped=to_gdf(snapped, 25833),
        anchors=to_gdf(anchors, 25833),
        center=_DEBUG_CONFIG["center"],
    )
    # snapped = sfilter_inverse(
    #     snapped, as_polygons
    # )  # snapped.loc[lambda x: ~x.intersects(
    #     as_.loc[lambda y: ~y.index.duplicated()].loc[midpoints.index].geometry
    # )]

    points.loc[snapped.index, "geometry"] = snapped["_right_geom"]
    points.loc[snapped.index, "_is_snapped"] = True

    return points


def points_to_line_segments(points: GeoDataFrame) -> GeoDataFrame:
    points = points.copy()
    points["next"] = points.groupby(level=0)["geometry"].shift(-1)

    first_points = points.loc[lambda x: ~x.index.duplicated(), "geometry"]
    is_last_point = points["next"].isna()

    points.loc[is_last_point, "next"] = first_points
    assert points["next"].notna().all()

    points["geometry"] = make_lines_between_points(
        points["geometry"].values, points["next"].values
    )

    return GeoDataFrame(
        points.drop(columns=["next"]), geometry="geometry", crs=points.crs
    )


def _add_midpoints_to_segments(points, relevant_mask_nodes, tolerance, as_polygons):
    assert points.index.is_unique
    relevant_mask_nodes["_right_geom"] = relevant_mask_nodes.geometry

    relevant_mask_nodes = buff(relevant_mask_nodes, tolerance, resolution=10)

    segments = points_to_line_segments(points.set_index("_geom_idx"))
    segments["_geom_idx_left"] = segments.index
    segments["_seg_length"] = segments.length
    segments.index = points.index

    joined = (
        segments.sjoin(relevant_mask_nodes).loc[
            lambda x: x["_geom_idx_left"] != x["_geom_idx"]
        ]
        # .loc[lambda x: ~x.index.duplicated()]
    )
    boundaries_groupby = joined.boundary.explode(index_parts=False).groupby(level=0)

    lines_between_mask_and_edge = (
        GeoSeries(
            shapely.shortest_line(joined.geometry.values, joined["_right_geom"].values),
            index=joined.index,
        ).loc[lambda x: x.length > 0]
        # .pipe(sfilter_inverse, as_polygons)
    )
    # snapped_too_far = (
    #     GeoDataFrame(
    #         {
    #             "geometry": lines_between_mask_and_edge,
    #             "_geom_idx_left": joined["_geom_idx"].values,
    #         },
    #         index=range(len(joined)),
    #     )
    #     .sjoin(buff(as_polygons, -PRECISION, resolution=10))
    #     .loc[lambda x: x["_geom_idx_left"] != x["_geom_idx"]]
    # )
    # not_too_far = pd.Index(range(len(joined))).difference(snapped_too_far.index)
    # # joined = joined.iloc[not_too_far]
    # lines_between_mask_and_edge = lines_between_mask_and_edge[not_too_far]

    # midpoints = shapely.get_point(lines_between_mask_and_edge, 0)
    midpoints = GeoSeries(
        shapely.get_point(lines_between_mask_and_edge, 0),
        index=lines_between_mask_and_edge.index,
    )

    midpoints = midpoints.loc[
        lambda x: ~x.intersects(
            joined.loc[lambda y: ~y.index.duplicated()].loc[midpoints.index].geometry
        )
    ]

    with_new_midpoints = (
        pd.concat(
            [
                GeoSeries(boundaries_groupby.nth(0)),
                midpoints,
                GeoSeries(boundaries_groupby.nth(-1)),
            ]
        )
        # .loc[lambda x: ~x.geometry.duplicated()]
        # .loc[lambda x: x.groupby(level=0).size() > 1]
    )

    explore_locals(
        midpoints=to_gdf(midpoints, 25833),
        center=_DEBUG_CONFIG["center"],
    )
    # explore_locals(
    #     to_gdf(segments, 25833),
    #     mask=to_gdf([5.36995682, 59.00939296], 4326).to_crs(25833).buffer(2.5),
    # )

    # already_sorted = (
    #     with_new_midpoints.loc[lambda x: x.groupby(level=0).size() <= 3]
    #     .groupby(level=0)
    #     .agg(lambda x: LineString(x.values))
    # )
    # print(already_sorted)

    # avoid groupby.agg(LineString) for indices with two or three points
    has_two_points = with_new_midpoints.loc[lambda x: x.groupby(level=0).size() == 2]
    has_two_points = GeoSeries(
        make_lines_between_points(
            has_two_points.groupby(level=0).nth(0).values,
            has_two_points.groupby(level=0).nth(1).values,
        ),
        index=has_two_points.index.unique(),
    )

    has_three_points = with_new_midpoints.loc[lambda x: x.groupby(level=0).size() == 3]
    has_three_points = GeoSeries(
        make_lines_between_points(
            has_three_points.groupby(level=0).nth(0).values,
            has_three_points.groupby(level=0).nth(1).values,
            has_three_points.groupby(level=0).nth(2).values,
        ),
        index=has_three_points.index.unique(),
    )

    with_new_midpoints = (
        with_new_midpoints.loc[lambda x: x.groupby(level=0).size() > 3]
        .groupby(level=0)
        .agg(lambda x: _sorted_unary_union(x.values))
    )
    with_new_midpoints_orig = with_new_midpoints.copy()

    with_new_midpoints = pd.concat(
        [with_new_midpoints, has_two_points, has_three_points]
    )

    # explore(
    #     segments=to_gdf(segments[lambda x: x._geom_idx == 1], 25833),
    #     # seg_points=to_gdf(segments, 25833)
    #     # .assign(geometry=lambda x: shapely.extract_unique_points(x.geometry))
    #     # .explode(),
    #     # relevant_mask_nodes=to_gdf(relevant_mask_nodes, 25833),
    #     midpoints=to_gdf(midpoints, 25833),
    #     lines_between_mask_and_edge=to_gdf(
    #         lines_between_mask_and_edge, 25833
    #     ).pipe(buff, 0.05),
    #     with_new_midpoints=to_gdf(with_new_midpoints, 25833),
    #     geoms=to_gdf(geoms, 25833),
    # )

    segs_before = to_gdf(segments, 25833)
    segments.loc[with_new_midpoints.index, "geometry"] = with_new_midpoints
    explore_locals(
        has_two_points=to_gdf(has_two_points, 25833),
        has_three_points=to_gdf(has_three_points, 25833),
        with_new_midpoints_orig=to_gdf(with_new_midpoints_orig, 25833),
        segs_before=segs_before,
        segs=to_gdf(segments, 25833),
        center=_DEBUG_CONFIG["center"],
    )
    # explore(
    #     segments=to_gdf(segments[lambda x: x._geom_idx == 1], 25833),
    #     segment_points=to_gdf(
    #         segments[lambda x: x._geom_idx == 1].assign(
    #             geometry=lambda x: extract_unique_points(x.geometry)
    #         ),
    #         25833,
    #     ).explode(),
    #     # segments_all=to_gdf(segments, 25833),
    #     # seg_points=to_gdf(segments, 25833)
    #     # .assign(geometry=lambda x: shapely.extract_unique_points(x.geometry))
    #     # .explode(),
    #     # relevant_mask_nodes=to_gdf(relevant_mask_nodes, 25833),
    #     midpoints=to_gdf(midpoints, 25833),
    #     lines_between_mask_and_edge=to_gdf(
    #         lines_between_mask_and_edge, 25833
    #     ).pipe(buff, 0.05),
    #     with_new_midpoints=to_gdf(with_new_midpoints, 25833),
    #     geoms=to_gdf(geoms, 25833),
    # )

    segments.geometry = extract_unique_points(segments.geometry)
    return (
        segments.rename(columns={"_geom_idx_left": "_geom_idx"}, errors="raise")
        .sort_index()
        .explode(ignore_index=True)
    )

    relevant_mask_nodes = relevant_mask_nodes.loc[
        lambda x: ~x.index.isin(joined["index_right"])
    ]

    return points


def _sorted_unary_union(df: NDArray[Point]) -> LineString:
    first = df[0]
    last = df[-1]
    try:
        mid = GeoSeries(df[1:-1])
    except IndexError:
        return LineString([first, last])

    distances_to_first = mid.distance(first).sort_values()
    mid_sorted = mid.loc[distances_to_first.index].values
    # if last.intersects(
    #     to_gdf([5.37420256, 59.0113640], 4326).to_crs(25833).buffer(5).union_all()
    # ):
    #     explore(
    #         first=to_gdf(first, 25833),
    #         last=to_gdf(last, 25833),
    #         mid_sorted=to_gdf(mid_sorted, 25833),
    #         center=_DEBUG_CONFIG["center"],
    #     )

    return LineString([first, *mid_sorted, last])


def _separate_single_neighbored_from_multi_neighoured_geometries(
    gdf: GeoDataFrame, neighbors: GeoDataFrame
) -> tuple[GeoDataFrame, GeoDataFrame]:
    """Split GeoDataFrame in two: those with 0 or 1 neighbors and those with 2 or more."""
    tree = STRtree(neighbors.geometry.values)
    left, right = tree.query(gdf.geometry.values, predicate="intersects")
    pairs = pd.Series(right, index=left)
    has_more_than_one_neighbor = (
        pairs.groupby(level=0).size().loc[lambda x: x > 1].index
    )

    more_than_one_neighbor = gdf.iloc[has_more_than_one_neighbor]
    one_or_zero_neighbors = gdf.iloc[
        pd.Index(range(len(gdf))).difference(has_more_than_one_neighbor)
    ]

    return one_or_zero_neighbors, more_than_one_neighbor


def split_and_eliminate_by_longest(
    gdf: GeoDataFrame | tuple[GeoDataFrame],
    to_eliminate: GeoDataFrame,
    tolerance: float | int,
    ignore_index: bool = False,
    **kwargs,
) -> tuple[GeoDataFrame]:
    if isinstance(gdf, (list, tuple)):
        # concat, then break up the dataframes in the end
        was_multiple_gdfs = True
        original_cols = [df.columns for df in gdf]
        gdf = pd.concat(df.assign(**{"_df_idx": i}) for i, df in enumerate(gdf))
    else:
        was_multiple_gdfs = False

    if 0:
        to_eliminate.geometry = to_eliminate.buffer(
            -PRECISION,
            resolution=1,
            join_style=2,
        ).buffer(
            PRECISION,
            resolution=1,
            join_style=2,
        )
        to_eliminate = to_eliminate.loc[lambda x: ~x.is_empty]

    # now to split polygons to be eliminated to avoid weird shapes
    # split only the polygons with multiple neighbors
    single_neighbored, multi_neighbored = (
        _separate_single_neighbored_from_multi_neighoured_geometries(to_eliminate, gdf)
    )
    multi_neighbored = split_by_neighbors(multi_neighbored, gdf, tolerance=tolerance)
    to_eliminate = pd.concat([multi_neighbored, single_neighbored])
    gdf, isolated = eliminate_by_longest(
        gdf, to_eliminate, return_isolated=True, ignore_index=ignore_index, **kwargs
    )

    if not was_multiple_gdfs:
        return gdf, isolated

    gdfs = ()
    for i, cols in enumerate(original_cols):
        df = gdf.loc[gdf["_df_idx"] == i, cols]
        gdfs += (df,)
    gdfs += (isolated,)

    return gdfs


def _eliminate_not_really_isolated(gdf, isolated):
    gdf = make_all_singlepart(gdf)
    not_really_isolated, really_isolated = sfilter_split(
        isolated, gdf.buffer(PRECISION)
    )
    not_really_isolated.geometry = not_really_isolated.buffer(PRECISION * 10)
    gdf, still_isolated = eliminate_by_longest(
        gdf,
        not_really_isolated,
        return_isolated=True,
    )
    assert not len(still_isolated), still_isolated
    gdf = make_all_singlepart(gdf)

    return pd.concat([gdf, really_isolated])


def _remove_legit_spikes(df):
    """Remove points where the next and previous points are the same.

    The lines these points make are as spiky as they come,
    hence the term "legit spikes".
    """
    df["next"] = df.groupby(level=0)["geometry"].shift(-1)
    df["prev"] = df.groupby(level=0)["geometry"].shift(1)

    first_points = df.loc[lambda x: ~x.index.duplicated(keep="first"), "geometry"]
    is_last_point = df["next"].isna()
    df.loc[is_last_point, "next"] = first_points

    last_points = df.loc[lambda x: ~x.index.duplicated(keep="last"), "geometry"]
    is_first_point = df["prev"].isna()
    df.loc[is_first_point, "prev"] = last_points

    assert df["next"].notna().all()
    assert df["prev"].notna().all()

    # print("_remove_legit_spikes")
    # print(df)
    # print(df.loc[lambda x: x["next"] != x["prev"]])
    # explore(
    #     df.set_crs(25833),
    #     df.loc[lambda x: x["next"] != x["prev"]],
    #     df.loc[lambda x: x["next"] == x["prev"]],
    # )

    return df.loc[lambda x: x["next"] != x["prev"]]

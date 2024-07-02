# %%

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

src = str(Path(__file__).parent).replace("tests", "") + "src"


sys.path.insert(0, src)

import warnings
from collections.abc import Callable

import numpy as np
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from numpy.typing import NDArray
from shapely import Geometry
from shapely import STRtree
from shapely import extract_unique_points
from shapely import polygons
from shapely.geometry import LinearRing
from shapely.geometry import LineString
from shapely.geometry import Point

import sgis as sg
from sgis import *
from sgis.geopandas_tools.cleaning import *

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def explore(*args, **kwargs):
    pass


def explore_locals(*args, **kwargs):
    pass


PRECISION = 1e-3
BUFFER_RES = 50


def coverage_clean(
    gdf: GeoDataFrame,
    tolerance: int | float,
    mask: GeoDataFrame | GeoSeries | Geometry | None = None,
    **kwargs,
) -> GeoDataFrame:
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
    gdf = snap_polygons(gdf, tolerance, mask=mask)
    # gdf = split_and_eliminate_thin(gdf, tolerance, mask=mask)
    # gdf = snap_polygons(gdf, tolerance, mask=mask)

    gdf = clean_overlay(gdf, mask, how="intersection", geom_type="polygon")
    missing = clean_overlay(mask, gdf, how="difference", geom_type="polygon").loc[
        lambda x: x.buffer(-tolerance / 2).is_empty
    ]
    is_thin = gdf.buffer(-tolerance).is_empty
    thin = gdf[is_thin]
    gdf = gdf[~is_thin]
    print(missing.columns)
    print(thin.columns)
    print("concat")
    print(pd.concat([missing, thin]).columns)
    to_eliminate = buff(pd.concat([missing, thin]), PRECISION).pipe(
        clean_overlay, mask, how="intersection", geom_type="polygon"
    )
    return eliminate_by_largest(gdf, to_eliminate).explode(ignore_index=True)

    return split_and_eliminate_thin(gdf, tolerance, mask=mask)


def split_and_eliminate_thin(
    gdf: GeoDataFrame,
    tolerance: int | float,
    mask: GeoDataFrame | GeoSeries | Geometry | None = None,
    join_style: int | str = 1,
    resolution: int = 1,
) -> GeoDataFrame:
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

    gdf_split = gdf.copy()
    gdf_split.geometry = gdf_split.buffer(
        -tolerance / 2, join_style=join_style, resolution=resolution
    ).buffer(tolerance / 2, join_style=join_style, resolution=resolution)
    gdf = gdf_split.explode(ignore_index=True)
    gdf = gdf.pipe(clean_overlay, mask, how="intersection", geom_type="polygon")
    missing = clean_overlay(mask, gdf, how="difference", geom_type="polygon").loc[
        lambda x: x.buffer(-tolerance / 2).is_empty
    ]
    missing = buff(missing, PRECISION).pipe(
        clean_overlay, mask, how="intersection", geom_type="polygon"
    )
    return eliminate_by_largest(gdf, missing).explode(ignore_index=True)


def snap_polygons(
    gdf: GeoDataFrame,
    tolerance: int | float,
    mask: GeoDataFrame | GeoSeries | Geometry | None = None,
    snap_to_nodes: bool = True,
    **kwargs,
) -> GeoDataFrame:
    if not len(gdf):
        return gdf

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

    donuts_without_spikes = (
        gdf.geometry.buffer(tolerance / 2, resolution=1, join_style=2)
        .buffer(-tolerance, resolution=1, join_style=2)
        .buffer(tolerance / 2, resolution=1, join_style=2)
        .pipe(to_lines)
        .buffer(tolerance)
    )

    gdf.geometry = (
        PolygonsAsRings(gdf.geometry.values)
        .apply_numpy_func(
            _snap_linearrings,
            kwargs=dict(
                tolerance=tolerance,
                mask=mask,
                snap_to_nodes=snap_to_nodes,
                donuts_without_spikes=donuts_without_spikes,
            ),
        )
        .to_numpy()
    )

    gdf = (
        to_single_geom_type(make_all_singlepart(clean_geoms(gdf)), "polygon")
        .reset_index(drop=True)
        .set_crs(crs)
    )

    gdf = update_geometries(gdf, geom_type="polygon")

    return gdf  # .pipe(clean_clip, mask, geom_type="polygon")


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

    tree = STRtree(points.loc[lambda x: x["_is_snapped"] != True, "geometry"].values)
    left, right = tree.query(
        points.loc[lambda x: x["_is_snapped"] != True, "geometry"].values,
        # points.geometry.values,
        predicate="dwithin",
        distance=tolerance,
    )
    indices = pd.Series(right, index=left, name="_right_idx")

    idx_mapper = dict(
        enumerate(points.loc[lambda x: x["_is_snapped"] != True, "_geom_idx"])
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
        center=(6550872, -29405, 10),
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

    if 1:
        new_anchors["_cluster"] = get_cluster_mapper(new_anchors.buffer(PRECISION))

        assert new_anchors["_geom_idx"].notna().all()

        no_longer_anchors: pd.Index = new_anchors.loc[
            lambda x: (x["_cluster"].duplicated())  # & (x["_geom_idx"] >= idx_start)
        ].index
        new_anchors = new_anchors.loc[lambda x: ~x.index.isin(no_longer_anchors)]

    # new_anchors.geometry = shapely.set_precision(new_anchors.geometry, PRECISION)

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
        lambda x: x["_is_snapped"] != True
    ]  # .loc[should_be_snapped].rename(
    #     columns={"_geom_idx": "_geom_idx_left"}, errors="raise"
    # )

    snapped = (
        to_be_snapped.sjoin_nearest(anchors, max_distance=tolerance)
        .sort_values("index_right")
        .loc[lambda x: ~x.index.duplicated()]
    )
    snapped.geometry = shapely.shortest_line(
        snapped.geometry.values, snapped["_right_geom"].values
    )
    # TODO sfilter_inverse here?
    # snapped = sfilter_inverse(
    #     snapped, as_polygons
    # )  # snapped.loc[lambda x: ~x.intersects(

    #     as_.loc[lambda y: ~y.index.duplicated()].loc[midpoints.index].geometry
    # )]

    points.loc[snapped.index, "geometry"] = snapped["_right_geom"]

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
        points.loc[lambda x: x["_is_snapped"] != True]
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
    # snapped = sfilter_inverse(
    #     snapped, as_polygons
    # )  # snapped.loc[lambda x: ~x.intersects(
    #     as_.loc[lambda y: ~y.index.duplicated()].loc[midpoints.index].geometry
    # )]

    points.loc[snapped.index, "geometry"] = snapped["_right_geom"]
    points.loc[snapped.index, "_is_snapped"] = True

    return points


def _snap_linearrings(
    geoms: NDArray[LinearRing],
    tolerance: int | float,
    mask: GeoDataFrame | None = None,
    snap_to_nodes: bool = True,
    gaps=None,
    donuts_without_spikes=None,
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

    as_polygons.geometry = as_polygons.geometry.buffer(-PRECISION)

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

    points = _snap(points, mask_nodes, tolerance, as_polygons)

    points1 = points.copy()
    not_snapped1 = (
        points.sort_index()
        .set_index("_geom_idx")
        # .pipe(_remove_legit_spikes)
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

    points = _snap(points, mask_nodes, tolerance, as_polygons)

    points2 = points.copy()
    not_snapped2 = (
        points.sort_index()
        .set_index("_geom_idx")
        .loc[lambda x: x.groupby(level=0).size() > 2]
        .groupby(level=0)["geometry"]
        .agg(LinearRing)
    )
    if snap_to_nodes:
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
    else:
        snapped = points

    assert (snapped["_geom_idx"] >= idx_start).all()

    as_rings = (
        snapped.sort_index()
        .set_index("_geom_idx")
        # .pipe(_remove_legit_spikes)
        .loc[lambda x: x.groupby(level=0).size() > 2]
        .groupby(level=0)["geometry"]
        .agg(LinearRing)
    )

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
        center=(6550872, -29405, 10),
        # mask=to_gdf([5.37439002, 59.01144682], 4326).to_crs(25833).buffer(20),
    )

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
            center=(6550872, -29405, 10),
        )

    missing = gdf.set_index("_geom_idx")["geometry"].loc[
        lambda x: (~x.index.isin(as_rings.index.union(thin.index)))
        & (x.index >= idx_start)
    ]
    missing.loc[:] = None

    return pd.concat([as_rings, missing, thin]).sort_index()


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

    # sss

    explore_locals(
        midpoints=to_gdf(midpoints, 25833),
        center=(6550872, -29405, 10),
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
        center=(6550872, -29405, 10),
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
    #     to_gdf([5.37420256, 59.0113640], 4326).to_crs(25833).buffer(5).unary_union
    # ):
    #     explore(
    #         first=to_gdf(first, 25833),
    #         last=to_gdf(last, 25833),
    #         mid_sorted=to_gdf(mid_sorted, 25833),
    #         center=(6550872, -29405, 10),
    #     )

    return LineString([first, *mid_sorted, last])


def test_clean_dissappearing_polygon():
    AREA_SHOULD_BE = 104

    with open(Path(__file__).parent / "testdata/dissolve_error.txt") as f:
        df = sg.to_gdf(f.readlines(), 25833)

    dissappears = sg.to_gdf([5.95201, 62.41451], 4326).to_crs(25833).buffer(100)
    df_problem_area = sg.sfilter(df, dissappears.buffer(0.1))

    assert len(df_problem_area) == 3

    assert (area := int(df_problem_area.area.sum())) == AREA_SHOULD_BE, area

    cleaned = sg.coverage_clean(df, 0.1, duplicate_action="fix")

    cleaned_problem_area = sg.sfilter(cleaned, dissappears.buffer(0.1))

    sg.explore(cleaned, cleaned_problem_area, dissappears, df_problem_area)
    assert (area := int(cleaned_problem_area.area.sum())) == AREA_SHOULD_BE, area

    cleaned_dissolved_problem_area = sg.sfilter(
        sg.dissexp(cleaned), dissappears.buffer(0.1)
    )

    # cleaned_dissolved_problem_area.to_parquet(
    #     "c:/users/ort/downloads/cleaned_dissolved_problem_area.parquet"
    # )

    assert len(cleaned_dissolved_problem_area) == 1, (
        sg.explore(
            cleaned_dissolved_problem_area.assign(
                col=lambda x: [str(i) for i in range(len(x))]
            ),
            "col",
        ),
        cleaned_dissolved_problem_area,
    )

    assert (
        area := int(cleaned_dissolved_problem_area.area.sum())
    ) == AREA_SHOULD_BE, area


def test_clean_1144():
    df = gpd.read_parquet(
        Path(__file__).parent / "testdata" / "snap_problem_area_1144.parquet"
    )

    bbox = sg.to_gdf(shapely.minimum_rotated_rectangle(df.unary_union), df.crs)

    kommune_utenhav = gpd.read_parquet(
        Path(__file__).parent / "testdata" / "kommune_utenhav_1144_2023.parquet"
    )

    kommune_utenhav = sg.clean_clip(kommune_utenhav, bbox, geom_type="polygon")

    assert sg.get_intersections(df).dissolve().area.sum() == 0
    assert int(df.area.sum()) == 154240, df.area.sum()

    cols = [
        "ARGRUNNF",
        "ARJORDBR",
        "ARKARTSTD",
        "ARSKOGBON",
        "ARTRESLAG",
        "ARTYPE",
        "ARVEGET",
        "ASTSSB",
        "df_idx",
        "geometry",
        "kilde",
    ]

    df["df_idx"] = range(len(df))

    for tolerance in [2, 1, 5]:
        print("\ntolerance")
        print(tolerance)
        # cleaned = sg.coverage_clean(df, tolerance, pre_dissolve_func=_buff).pipe(
        #     sg.clean_clip, df, geom_type="polygon"

        # )

        # allow near-thin polygons to dissappear. this happens because snapping makes them thin
        # before eliminate
        thick_df_indices = df.loc[
            lambda x: ~x.buffer(-tolerance / 2.2).is_empty, "df_idx"
        ]

        cleaned = sg.coverage_clean(
            df, tolerance, mask=kommune_utenhav
        )  # .pipe(sg.coverage_clean, tolerance)

        # cleaned = sg.coverage_clean(
        #     sg.sort_large_first(df), tolerance, mask=kommune_utenhav
        # ).pipe(sg.snap_polygons, 0.1, mask=kommune_utenhav)

        # allow edge cases
        cleaned_clipped = sg.clean_clip(cleaned, bbox.buffer(-tolerance * 1.1))

        gaps = sg.get_gaps(cleaned_clipped)
        double = sg.get_intersections(cleaned_clipped)
        missing = get_missing(
            sg.clean_clip(df, bbox.buffer(-tolerance * 1.1)), cleaned_clipped
        )

        sg.explore(cleaned, gaps, double, missing, df, kommune_utenhav)

        print(
            "cleaned",
            "gaps",
            gaps.area.sum(),
            "dup",
            double.area.sum(),
            "missing",
            missing.area.sum(),
        )
        assert gaps.area.sum() <= 1e-2, gaps.area.sum()
        assert double.area.sum() <= 1e-2, double.area.sum()
        assert missing.area.sum() <= 1e-2, missing.area.sum()

        assert thick_df_indices.isin(cleaned_clipped["df_idx"]).all(), sg.explore(
            df,
            cleaned,
            missing_polygons=df[
                (df["df_idx"].isin(thick_df_indices))
                & (~df["df_idx"].isin(cleaned_clipped["df_idx"]))
            ],
        )


def get_missing(df, other):
    return (
        sg.clean_overlay(df, other, how="difference", geom_type="polygon")
        .pipe(sg.buff, -0.0001)
        .pipe(sg.clean_overlay, other, how="difference", geom_type="polygon")
    )


def test_clean():

    df = gpd.read_parquet(Path(__file__).parent / "testdata" / "polygon_snap.parquet")

    bbox = sg.to_gdf(shapely.minimum_rotated_rectangle(df.unary_union), df.crs)

    kommune_utenhav = gpd.read_parquet(
        Path(__file__).parent / "testdata" / "kommune_utenhav_5435_2023.parquet"
    )
    kommune_utenhav = sg.clean_clip(kommune_utenhav, bbox, geom_type="polygon")

    holes = sg.to_gdf(
        [
            "POINT (905200 7878700)",
            "POINT (905250 7878780)",
            "POINT (905242.961 7878773.758)",
        ],
        25833,
    ).pipe(sg.buff, 3)

    df = sg.clean_overlay(df, holes, how="difference")

    mask = sg.close_all_holes(sg.dissexp_by_cluster(df)).dissolve()

    for tolerance in [5, 10]:
        print("tolerance:", tolerance)

        snapped = sg.coverage_clean(df, tolerance)
        assert sg.get_geom_type(snapped) == "polygon", sg.get_geom_type(snapped)

        double = sg.get_intersections(snapped).loc[lambda x: ~x.buffer(-1e-9).is_empty]
        gaps = sg.get_gaps(snapped).loc[lambda x: ~x.buffer(-1e-9).is_empty]
        missing = get_missing(df, snapped)

        print(double.area.sum(), missing.area.sum(), gaps.area.sum())

        sg.explore(
            df=df.to_crs(25833),
            snapped=snapped.to_crs(25833),
            double=double.to_crs(25833),
            missing=missing,
            gaps=gaps.to_crs(25833),
        )

        assert (a := max(list(double.area) + [0])) < 1e-4, a
        assert (a := max(list(missing.area) + [0])) < 1e-4, a
        assert (a := max(list(gaps.area) + [0])) < 1e-4, a

    sg.explore(
        snapped1=sg.coverage_clean(df, 1),
        snapped3=sg.coverage_clean(df, 3),
        snapped5=sg.coverage_clean(df, 5),
        df=df,
    )


def not_test_spikes():
    from shapely.geometry import Polygon

    factor = 10000

    sliver = sg.to_gdf(
        Polygon(
            [
                (0, 0),
                (0.1 * factor, 1 * factor),
                (0, 2 * factor),
                (-0.1 * factor, 1 * factor),
            ]
        )
    ).assign(what="sliver", num=1)
    poly_with_spike = sg.to_gdf(
        Polygon(
            [
                (0 * factor, 0 * factor),
                (-0.1 * factor, 1 * factor),
                (0 * factor, 2 * factor),
                (-0.99 * factor, 2.001 * factor),
                (-0.99 * factor, 1.5 * factor),
                (-1.01 * factor, 1.5 * factor),
                (-1.01 * factor, 2.001 * factor),
                (-1.51 * factor, 2.001 * factor),
                (-1.51 * factor, 1.7 * factor),
                (-1.52 * factor, 2.001 * factor),
                (-2 * factor, 2.001 * factor),
                (-1 * factor, 1 * factor),
            ],
            holes=[
                (
                    [
                        (-0.5 * factor, 1.25 * factor),
                        (-0.5 * factor, 1.65 * factor),
                        (-0.49 * factor, 1.65 * factor),
                        (-0.49 * factor, 1.25 * factor),
                    ]
                ),
            ],
        )
    ).assign(what="small", num=2)
    poly_filling_the_spike = sg.to_gdf(
        Polygon(
            [
                (0, 2.001 * factor),
                (-0.99 * factor, 2 * factor),
                (-0.99 * factor, 1.5 * factor),
                (-1.01 * factor, 1.5 * factor),
                (-1.01 * factor, 2 * factor),
                (-2 * factor, 2 * factor),
                (-2 * factor, 6 * factor),
                (0, 6 * factor),
                (0, 2.001 * factor),
            ],
        )
    ).assign(what="large", num=2)

    df = pd.concat([sliver, poly_with_spike, poly_filling_the_spike])
    holes = sg.buff(
        sg.to_gdf([(-0.84 * factor, 3 * factor), (-0.84 * factor, 4.4 * factor)]),
        [0.4 * factor, 0.3 * factor],
    )
    df = sg.clean_overlay(df, holes, how="update")
    df.crs = 25833

    tolerance = 0.09 * factor

    snapped = sg.coverage_clean(df, tolerance)
    # spikes_removed = sg.remove_spikes(df, tolerance)
    gaps = sg.get_gaps(snapped, True)
    print(gaps)
    # spikes_fixed = sg.split_spiky_polygons(df, tolerance)
    # fixed_and_cleaned = sg.coverage_clean(
    #     spikes_fixed, tolerance  # , pre_dissolve_func=_buff
    # )  # .pipe(sg.remove_spikes, tolerance / 100)

    if __name__ == "__main__":
        sg.explore(
            # fixed_and_cleaned=fixed_and_cleaned,
            snapped=snapped,
            gaps=gaps,
            # spikes_fixed=spikes_fixed,
            # df=df,
        )

    def is_close_enough(num1, num2):
        if num1 >= num2 - 1e-3 and num1 <= num2 + 1e-3:
            return True
        return False

    return
    area_should_be = [
        725264293.6535025,
        20000000.0,
        190000000.0,
        48285369.993336275,
        26450336.353161283,
    ]
    print(list(fixed_and_cleaned.area))
    for area1, area2 in zip(
        sorted(fixed_and_cleaned.area),
        sorted(area_should_be),
        strict=False,
    ):
        assert is_close_enough(area1, area2), (area1, area2)

    length_should_be = [
        163423.91054766334,
        40199.502484483564,
        68384.02248970368,
        24882.8908851665,
        18541.01966249684,
    ]

    print(list(fixed_and_cleaned.length))
    for length1, length2 in zip(
        sorted(fixed_and_cleaned.length),
        sorted(length_should_be),
        strict=False,
    ):
        assert is_close_enough(length1, length2), (length1, length2)


def main():
    test_clean()
    test_clean_dissappearing_polygon()
    not_test_spikes()
    test_clean_1144()


if __name__ == "__main__":
    sg.coverage_clean = coverage_clean

    # cProfile.run("main()", sort="cumtime")

    main()

# %%

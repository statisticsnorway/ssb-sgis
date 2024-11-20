# %%
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from numpy.typing import NDArray
from shapely import Geometry
from shapely import STRtree
from shapely import extract_unique_points
from shapely import get_coordinates
from shapely import linearrings
from shapely import polygons
from shapely.errors import GEOSException
from shapely.geometry import LinearRing
from shapely.geometry import LineString
from shapely.geometry import Point

try:
    import numba
except ImportError:

    class numba:
        """Placeholder."""

        @staticmethod
        def njit(func) -> Callable:
            """Placeholder that does nothing."""

            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper


from ..debug_config import _DEBUG_CONFIG
from ..maps.maps import explore
from .conversion import to_gdf
from .conversion import to_geoseries
from .duplicates import update_geometries
from .general import clean_geoms
from .geometry_types import make_all_singlepart
from .geometry_types import to_single_geom_type
from .overlay import clean_overlay
from .polygon_operations import eliminate_by_longest
from .polygon_operations import split_by_neighbors
from .polygons_as_rings import PolygonsAsRings
from .sfilter import sfilter
from .sfilter import sfilter_inverse

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


PRECISION = 1e-3
BUFFER_RES = 50


# def explore(*args, **kwargs):
#     pass


# def explore_locals(*args, **kwargs):
#     pass


# def no_njit(func):
#     def wrapper(*args, **kwargs):
#         result = func(*args, **kwargs)
#         return result

#     return wrapper


# numba.njit = no_njit


def coverage_clean(
    gdf: GeoDataFrame,
    tolerance: int | float,
    mask: GeoDataFrame | GeoSeries | Geometry | None = None,
    snap_to_anchors: bool = True,
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
        mask: Mask to clip gdf to.
        snap_to_anchors: If True (default), snaps to anchor nodes in gdf. If False,
            only snaps to mask nodes (mask cannot be None in this case).
        **kwargs: Temporary backwards compatibility to avoid TypeErrors.

    Returns:
        A GeoDataFrame with cleaned polygons.
    """
    if not len(gdf):
        return gdf

    gdf_original = gdf.copy()

    # more_than_one = get_num_geometries(gdf.geometry.values) > 1
    # gdf.loc[more_than_one, gdf._geometry_column_name] = gdf.loc[
    #     more_than_one, gdf._geometry_column_name
    # ].apply(_unary_union_for_notna)

    if mask is not None:
        try:
            mask: GeoDataFrame = mask[["geometry"]].pipe(make_all_singlepart)
        except Exception:
            mask: GeoDataFrame = (
                to_geoseries(mask).to_frame("geometry").pipe(make_all_singlepart)
            )

        # mask: GeoDataFrame = close_all_holes(
        #     dissexp_by_cluster(gdf[["geometry"]])
        # ).pipe(make_all_singlepart)
        # mask = GeoDataFrame(
        #     {
        #         "geometry": [
        #             mask.union_all()
        #             .buffer(
        #                 PRECISION,
        #                 resolution=1,
        #                 join_style=2,
        #             )
        #             .buffer(
        #                 -PRECISION,
        #                 resolution=1,
        #                 join_style=2,
        #             )
        #         ]
        #     },
        #     crs=gdf.crs,
        # ).pipe(make_all_singlepart)
        # # gaps = shapely.union_all(get_gaps(mask).geometry.values)
        # # mask = shapely.get_parts(extract_unique_points(mask.geometry.values))
        # # not_by_gaps = shapely.distance(mask, gaps) > PRECISION
        # # mask = GeoDataFrame({"geometry": mask[not_by_gaps]})

    gdf = snap_polygons(gdf, tolerance, mask=mask, snap_to_anchors=snap_to_anchors)

    if mask is not None:
        missing_from_mask = clean_overlay(
            mask, gdf, how="difference", geom_type="polygon"
        ).loc[lambda x: x.buffer(-tolerance + PRECISION).is_empty]
        gdf, _ = eliminate_by_longest(gdf, missing_from_mask)

    missing_from_gdf = sfilter_inverse(gdf_original, gdf.buffer(-PRECISION)).loc[
        lambda x: (~x.buffer(-PRECISION).is_empty)
    ]
    return pd.concat([gdf, missing_from_gdf], ignore_index=True).pipe(
        update_geometries, geom_type="polygon"
    )


def snap_polygons(
    gdf: GeoDataFrame,
    tolerance: int | float,
    mask: GeoDataFrame | GeoSeries | Geometry | None = None,
    snap_to_anchors: bool = True,
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

    gdf = gdf[lambda x: ~x.buffer(-tolerance / 2 - PRECISION).is_empty]
    # gdf = gdf[lambda x: ~x.buffer(-tolerance / 3).is_empty]

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
                snap_to_anchors=snap_to_anchors,
            ),
        )
        .to_numpy()
    )

    gdf = (
        to_single_geom_type(make_all_singlepart(clean_geoms(gdf)), "polygon")
        .reset_index(drop=True)
        .set_crs(crs)
    )

    missing = clean_overlay(gdf_orig, gdf, how="difference").loc[
        lambda x: ~x.buffer(-tolerance / 2).is_empty
    ]

    if mask is None:
        mask = GeoDataFrame({"geometry": []})
    explore(
        gdf,
        # gdf_orig,
        # thin,
        mask,
        missing,
        mask_p=to_gdf(mask.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        gdf_p=to_gdf(gdf.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        center=(5.36765872, 59.01199837, 1),
    )
    explore(
        gdf,
        gdf_orig,
        # thin,
        mask,
        missing,
        mask_p=to_gdf(mask.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        gdf_p=to_gdf(gdf.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        center=(5.36820681, 59.01182298, 2),
    )
    explore(
        gdf,
        gdf_orig,
        # thin,
        mask,
        missing,
        mask_p=to_gdf(mask.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        gdf_p=to_gdf(gdf.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        center=(5.37327042, 59.01099359, 5),
    )
    explore(
        gdf,
        gdf_orig,
        # thin,
        mask,
        missing,
        mask_p=to_gdf(mask.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        gdf_p=to_gdf(gdf.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        center=(5.36853688, 59.01169013, 5),
    )
    explore(
        gdf,
        # gdf_orig,
        missing,
        mask,
        mask_p=to_gdf(mask.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        gdf_p=to_gdf(gdf.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        center=(5.37142966, 59.009799, 0.01),
        max_zoom=40,
    )
    explore(
        gdf,
        # gdf_orig,
        missing,
        mask,
        mask_p=to_gdf(mask.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        gdf_p=to_gdf(gdf.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        center=(5.36866312, 59.00842846, 0.01),
        max_zoom=40,
    )

    explore(
        gdf,
        # gdf_orig,
        missing,
        mask,
        mask_p=to_gdf(mask.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        gdf_p=to_gdf(gdf.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        center=(5.37707146, 59.01065274, 0.4),
        max_zoom=40,
    )

    explore(
        gdf,
        # gdf_orig,
        missing,
        mask,
        mask_p=to_gdf(mask.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        gdf_p=to_gdf(gdf.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        center=(-52074.0241, 6580847.4464, 0.1),
        max_zoom=40,
    )

    explore(
        gdf,
        # gdf_orig,
        missing,
        mask,
        mask_p=to_gdf(mask.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        gdf_p=to_gdf(gdf.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        center=(5.38389153, 59.00548223, 1),
        max_zoom=40,
    )

    # explore(
    #     gdf_orig,
    #     gdf,
    #     dups=get_intersections(gdf, geom_type="polygon"),
    #     msk=mask,
    #     gaps=get_gaps(gdf),
    #     updated=update_geometries(gdf, geom_type="polygon"),
    #     # browser=False,
    # )

    # gdf = update_geometries(gdf, geom_type="polygon")

    return gdf  # .pipe(clean_clip, mask, geom_type="polygon")


# @numba.njit
def _snap_to_anchors(
    geoms,
    indices: NDArray[np.int32],
    anchors,
    anchor_indices,
    mask,
    mask_indices,
    was_midpoint,
    was_midpoint_mask,
    tolerance: int | float,
) -> tuple[NDArray, NDArray, NDArray]:

    coords, all_distances = _snap_to_anchors_inner(
        geoms,
        indices,
        anchors,
        anchor_indices,
        mask,
        mask_indices,
        was_midpoint,
        was_midpoint_mask,
        tolerance,
    )

    not_inf = coords[:, 0] != np.inf
    all_distances = all_distances[not_inf]
    indices = indices[not_inf]
    coords = coords[not_inf]

    is_snapped = np.full(len(coords), False)

    n_coords = len(coords)

    range_indices = np.arange(len(coords))

    range_index = -1
    for index in np.unique(indices):
        cond = indices == index
        these_coords = coords[cond]

        # explore(ll=to_gdf(LineString(shapely.points(these_coords)), 25833))

        # assert np.array_equal(these_coords[0], these_coords[-1]), these_coords

        these_range_indices = range_indices[cond]
        these_distances = all_distances[cond]
        for i in range(len(these_coords)):
            range_index += 1
            if is_snapped[range_index]:
                print(i, "000")
                continue
            # distances = all_distances[range_index]
            distances = these_distances[i]
            # distances = these_distances[:, i]
            min_dist = np.min(distances)
            if min_dist > tolerance:  # or min_dist == 0:
                print(i, "111", min_dist)
                continue

            is_snapped_now = False

            for j in np.argsort(distances):
                if distances[j] > tolerance:  # TODO or distances[j] == 0:
                    break

                if was_midpoint_mask[j]:
                    continue

                anchor = anchors[j]
                ring = these_coords.copy()
                ring[i] = anchor

                # snap the nexts points to same anchor if neighboring points have same anchor
                # in order to properly check if the ring will be simple after snapping
                indices_with_same_anchor = [range_index]
                # these_coords = coords[indices==index]

                pos_counter = 0
                # has_same_anchor_pos = True
                # has_same_anchor_neg = True
                while (
                    pos_counter + i < len(these_distances) - 1
                ):  # has_same_anchor_pos or has_same_anchor_neg:
                    pos_counter += 1

                    # if indices[i + pos_counter] != index:
                    #     break
                    # next_distances = all_distances[range_index + pos_counter]
                    next_distances = these_distances[i + pos_counter]
                    has_same_anchor_pos = False
                    for j2 in np.argsort(next_distances):
                        if was_midpoint_mask[j2]:
                            continue
                        if next_distances[j2] > tolerance:
                            break

                        has_same_anchor_pos = j2 == j
                        # print(
                        #     "pos c",
                        #     i,
                        #     j,
                        #     j2,
                        #     pos_counter,
                        #     has_same_anchor_pos,
                        #     distances[j],
                        #     next_distances[j2],
                        # )
                        break
                    if has_same_anchor_pos:
                        ring[i + pos_counter] = anchor
                        indices_with_same_anchor.append(range_index + pos_counter)
                    else:
                        break

                # for j4 in np.arange(
                #     indices_with_same_anchor[0], indices_with_same_anchor[-1]
                # ):
                #     ring[j4 - range_index + i] = anchor
                #     indices_with_same_anchor.append(j4)

                if i == 0:
                    # snap points at the end of the line if same anchor
                    neg_counter = 0
                    # has_same_anchor_neg = True
                    while True:  # has_same_anchor_pos or has_same_anchor_neg:
                        neg_counter -= 1

                        # if indices[i + pos_counter] != index:
                        #     break
                        this_range_index = these_range_indices[neg_counter]
                        # next_distances = all_distances[this_range_index]
                        next_distances = these_distances[neg_counter]
                        has_same_anchor_neg = False
                        for j3 in np.argsort(next_distances):
                            if was_midpoint_mask[j3]:
                                continue
                            if next_distances[j3] > tolerance:
                                break

                            has_same_anchor_neg = j3 == j
                            # print(
                            #     "neg c",
                            #     i,
                            #     j,
                            #     j3,
                            #     pos_counter,
                            #     # has_same_anchor,
                            #     distances[j],
                            #     next_distances[j3],
                            # )
                            break
                        if has_same_anchor_neg:
                            ring[neg_counter] = anchor
                            indices_with_same_anchor.append(this_range_index)
                        else:
                            break

                    # for j5 in np.arange(0, indices_with_same_anchor[-1]):
                    #     ring[j5 - range_index + i] = anchor
                    #     indices_with_same_anchor.append(j5)

                indices_with_same_anchor = np.unique(indices_with_same_anchor)

                line_is_simple: bool = LineString(ring).is_simple

                # if i in [67, 68, 69, 173, 174, 175, 176, 177]:  # or
                if Point(these_coords[i]).intersects(
                    to_gdf([12.08375303, 67.50052183], 4326)
                    .to_crs(25833)
                    .buffer(10)
                    .union_all()
                ):
                    # for xxx, yyy in locals().items():
                    #     if len(str(yyy)) > 50:
                    #         continue
                    #     print(xxx)
                    #     print(yyy)

                    # print("prev:", was_midpoint_mask[j - 1])
                    # print(distances[np.argsort(distances)])
                    # print(anchors[np.argsort(distances)])
                    # print(ring)
                    explore(
                        out_coords=to_gdf(
                            shapely.linestrings(coords, indices=indices), 25833
                        ),
                        llll=to_gdf(LineString(ring), 25833),
                        # this=to_gdf(this),
                        # next_=to_gdf(next_),
                        # line=to_gdf(LineString(np.array([this, next_])), 25833),
                        geom=to_gdf(these_coords[i], 25833),
                        prev=to_gdf(these_coords[i - 1], 25833),
                        nxt=to_gdf(these_coords[i + 1], 25833),
                        nxt2=to_gdf(these_coords[i + 2], 25833),
                        anchor=to_gdf(anchor, 25833),
                        # browser=True,
                    )

                # print(
                #     "line_is_simple", line_is_simple, range_index, i, index, j
                # )  # , j2, j3, x)

                if not line_is_simple:
                    #     for j4 in range(len(ring)):
                    #         this_p = ring[j4]
                    #         for j5 in range(len(ring)):
                    #             that_p = ring[j5]
                    #             dist_ = np.sqrt(
                    #                 (this_p[0] - that_p[0]) ** 2
                    #                 + (this_p[1] - that_p[1]) ** 2
                    #             )
                    #             if dist_ > 0 and dist_ < 1e-5:
                    #                 print(this_p)
                    #                 print(that_p)
                    #                 ring[j5] = this_p

                    print(LineString(ring).wkt)
                    # explore(
                    #     out_coords=to_gdf(
                    #         shapely.linestrings(coords, indices=indices), 25833
                    #     ),
                    #     llll=to_gdf(LineString(ring), 25833),
                    #     # this=to_gdf(this),
                    #     # next_=to_gdf(next_),
                    #     # line=to_gdf(LineString(np.array([this, next_])), 25833),
                    #     geom=to_gdf(these_coords[i], 25833),
                    #     prev=to_gdf(these_coords[i - 1], 25833),
                    #     nxt=to_gdf(these_coords[i + 1], 25833),
                    #     nxt2=to_gdf(these_coords[i + 2], 25833),
                    #     anchor=to_gdf(anchor, 25833),
                    #     # browser=True,
                    # )

                line_is_simple: bool = LineString(ring).is_simple

                if line_is_simple:
                    # coords[i] = anchors[j]
                    # is_snapped_to[j] = True
                    # is_snapped[i] = True
                    # explore(
                    #     out_coords=to_gdf(
                    #         shapely.linestrings(coords, indices=indices), 25833
                    #     ),
                    #     llll=to_gdf(LineString(ring), 25833),
                    #     # this=to_gdf(this),
                    #     # next_=to_gdf(next_),
                    #     # line=to_gdf(LineString(np.array([this, next_])), 25833),
                    #     anc=to_gdf(anchors[j]),
                    #     geom=to_gdf(coords[i], 25833),
                    #     these=to_gdf(coords[i : i + n_points_with_same_anchor ], 25833),
                    #     prev=to_gdf(coords[i - 1], 25833),
                    #     prev2=to_gdf(coords[i - 2], 25833),
                    #     nxt=to_gdf(coords[i + n_points_with_same_anchor + 1], 25833),
                    #     nxt2=to_gdf(coords[i + n_points_with_same_anchor + 2], 25833),
                    #     nxt3=to_gdf(coords[i + n_points_with_same_anchor + 3], 25833),
                    # )
                    # print(coords[i : i + n_points_with_same_anchor + 1])
                    for (
                        x
                    ) in indices_with_same_anchor:  # range(n_points_with_same_anchor):
                        # print(range_index, i, index, j, j2, j3, x)
                        coords[x] = anchor  # s[j]
                        is_snapped[x] = True
                        # coords[i + x] = anchors[j]
                        # is_snapped[i + x] = True
                    # print(coords[i : i + n_points_with_same_anchor + 1])

                    is_snapped_now = True
                    break
                # else:

            if not is_snapped_now:
                coords[range_index] = anchors[np.argmin(distances)]
                # is_snapped_to[np.argmin(distances)] = True

            if 0 and index == 0:  # i > 30 and i < 40:
                print(i)
                explore(
                    out_coords=to_gdf(
                        shapely.linestrings(coords, indices=indices), 25833
                    ),
                    llll=to_gdf(LineString(ring), 25833),
                    pppp=to_gdf(shapely.points(ring), 25833).assign(
                        wkt=lambda x: [g.wkt for g in x.geometry]
                    ),
                    # this=to_gdf(this),
                    # next_=to_gdf(next_),
                    # line=to_gdf(LineString(np.array([this, next_])), 25833),
                    anc=to_gdf(anchors[j]).assign(
                        wkt=lambda x: [g.wkt for g in x.geometry]
                    ),
                    geom=to_gdf(these_coords[i], 25833).assign(
                        wkt=lambda x: [g.wkt for g in x.geometry]
                    ),
                    # these=to_gdf(
                    #     these_coords[i : i + n_points_with_same_anchor], 25833
                    # ).assign(wkt=lambda x: [g.wkt for g in x.geometry]),
                    prev=to_gdf(these_coords[i - 1], 25833).assign(
                        wkt=lambda x: [g.wkt for g in x.geometry]
                    ),
                    prev2=to_gdf(these_coords[i - 2], 25833).assign(
                        wkt=lambda x: [g.wkt for g in x.geometry]
                    ),
                    nxt=to_gdf(these_coords[i + 1], 25833).assign(
                        wkt=lambda x: [g.wkt for g in x.geometry]
                    ),
                    nxt2=to_gdf(these_coords[i + 2], 25833).assign(
                        wkt=lambda x: [g.wkt for g in x.geometry]
                    ),
                    nxt3=to_gdf(these_coords[i + 3], 25833).assign(
                        wkt=lambda x: [g.wkt for g in x.geometry]
                    ),
                    # browser=True,
                    # nxt_n=to_gdf(
                    #     coords[i + n_points_with_same_anchor + 1], 25833
                    # ).assign(wkt=lambda x: [g.wkt for g in x.geometry]),
                    # nxt_n2=to_gdf(
                    #     coords[i + n_points_with_same_anchor + 2], 25833
                    # ).assign(wkt=lambda x: [g.wkt for g in x.geometry]),
                    # nxt_n3=to_gdf(
                    #     coords[i + n_points_with_same_anchor + 3], 25833
                    # ).assign(wkt=lambda x: [g.wkt for g in x.geometry]),
                )
            # if (
            #     indices[i] == 48
            # ):  # and int(out_coords[i][0]) == 375502 and int(out_coords[i][1]) == 7490104:
            #     print(geom, out_coords[i], out_coords[-3:])
            #     xxx += 1
            #     if xxx > 100 and i >= 2106:
            #         print(locals())
            #         explore(
            #             geom=to_gdf(geom, 25833),
            #             out=to_gdf(out_coords[i], 25833),
            #             anc=to_gdf(shapely.points(anchors), 25833),
            #             llll=to_gdf(
            #                 shapely.geometry.LineString(
            #                     np.array(out_coords)[indices[: len(out_coords)] == 48]
            #                 ),
            #                 25833,
            #             ),
            #         )

    return coords, indices


@numba.njit
def _snap_to_anchors_inner(
    geoms,
    indices: NDArray[np.int32],
    anchors,
    anchor_indices,
    mask,
    mask_indices,
    was_midpoint,
    was_midpoint_mask,
    tolerance: int | float,
) -> tuple[NDArray, NDArray, NDArray]:
    # def orientation(p, q, r):
    #     # Calculate orientation of the triplet (p, q, r).
    #     # 0 -> collinear, 1 -> clockwise, 2 -> counterclockwise
    #     val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    #     if val == 0:
    #         return 0
    #     return 1 if val > 0 else 2

    # def on_segment(p, q, r):
    #     # Check if point q lies on line segment pr
    #     if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[
    #         1
    #     ] <= max(p[1], r[1]):
    #         return True
    #     return False

    # def check_intersection(line1, line2):
    #     """
    #     Check if two line segments intersect.

    #     Parameters:
    #     line1 : np.array : 2x2 array with endpoints of the first line segment [[x1, y1], [x2, y2]]
    #     line2 : np.array : 2x2 array with endpoints of the second line segment [[x3, y3], [x4, y4]]

    #     Returns:
    #     bool : True if the lines intersect, False otherwise.
    #     """

    #     p1, q1 = line1
    #     p2, q2 = line2

    #     # Find the four orientations needed for the general and special cases
    #     o1 = orientation(p1, q1, p2)
    #     o2 = orientation(p1, q1, q2)
    #     o3 = orientation(p2, q2, p1)
    #     o4 = orientation(p2, q2, q1)

    #     # General case
    #     if o1 != o2 and o3 != o4:
    #         return True

    #     # Special cases
    #     # p1, q1, p2 are collinear and p2 lies on segment p1q1
    #     if o1 == 0 and on_segment(p1, p2, q1):
    #         return True

    #     # p1, q1, q2 are collinear and q2 lies on segment p1q1
    #     if o2 == 0 and on_segment(p1, q2, q1):
    #         return True

    #     # p2, q2, p1 are collinear and p1 lies on segment p2q2
    #     if o3 == 0 and on_segment(p2, p1, q2):
    #         return True

    #     # p2, q2, q1 are collinear and q1 lies on segment p2q2
    #     if o4 == 0 and on_segment(p2, q1, q2):
    #         return True

    #     return False

    out_coords = geoms.copy()
    # is_snapped = np.full(len(geoms), False)

    n_anchors = len(anchors)
    mask_n_minus_1 = len(mask) - 1
    is_snapped_to = np.full(len(anchors), False)
    out_distances = np.full((len(geoms), n_anchors), tolerance * 3)

    for i in range(len(geoms)):
        # if is_snapped[i]:
        #     continue
        geom = geoms[i]
        index = indices[i]
        # if i == 0 or index != indices[i - 1]:
        #     i_for_this_index = 0
        # else:
        #     i_for_this_index += 1

        is_snapped = False
        for j in range(len(mask)):
            mask_index = mask_indices[j]

            is_last = j == mask_n_minus_1 or mask_index != mask_indices[j + 1]
            if is_last:
                continue

            mask_point0 = mask[j]

            # if (
            #     not mask_is_snapped_to[j]
            #     and np.sqrt(
            #         (geom[0] - mask_point0[0]) ** 2 + (geom[1] - mask_point0[1]) ** 2
            #     )
            #     <= tolerance
            # ):
            #     out_coords[i] = mask_point0
            #     mask_is_snapped_to[j] = True
            #     is_snapped = True
            #     break

            mask_point1 = mask[j + 1]

            segment_vector = mask_point1 - mask_point0
            point_vector = geom - mask_point0
            segment_length_squared = np.dot(segment_vector, segment_vector)
            if segment_length_squared == 0:
                closest_point = mask_point0
            else:
                factor = np.dot(point_vector, segment_vector) / segment_length_squared
                factor = max(0, min(1, factor))
                closest_point = mask_point0 + factor * segment_vector

            if np.linalg.norm(geom - closest_point) == 0 and was_midpoint[i]:
                out_coords[i] = np.array([np.inf, np.inf])
                is_snapped = True
                break

        if is_snapped:
            continue

        distances = np.full(n_anchors, tolerance * 3)
        for j2 in range(n_anchors):
            anchor = anchors[j2]

            # if anchor_indices[j] == index:
            #     continue

            dist = np.sqrt((geom[0] - anchor[0]) ** 2 + (geom[1] - anchor[1]) ** 2)
            distances[j2] = dist
            out_distances[i, j2] = dist
            if dist == 0 and not was_midpoint_mask[j2]:
                break

    return out_coords, out_distances


@numba.njit
def _build_anchors(
    geoms: NDArray[np.float64],
    indices: NDArray[np.int32],
    mask_coords: NDArray[np.float64],
    mask_indices: NDArray[np.int32],
    was_midpoint_mask: NDArray[bool],
    tolerance: int | float,
):
    anchors = list(mask_coords)
    anchor_indices = list(mask_indices)
    is_anchor_arr = np.full(len(geoms), False)
    was_midpoint_mask = list(was_midpoint_mask)
    for i in np.arange(len(geoms)):
        geom = geoms[i]
        index = indices[i]
        # distances = []
        # for j, anchor in zip(anchor_indices, anchors):

        is_anchor = True
        for j in range(len(anchors)):
            # if indices[i] != indices[j]:
            # if i != j  and indices[i] != indices[j]:
            anchor = anchors[j]
            dist = np.sqrt((geom[0] - anchor[0]) ** 2 + (geom[1] - anchor[1]) ** 2)
            if dist <= tolerance:
                is_anchor = False
                break
            # distances.append(dist)
        # distances = np.array(distances)
        is_anchor_arr[i] = is_anchor
        if is_anchor:  # not len(distances) or np.min(distances) > tolerance:
            anchors.append(geom)
            anchor_indices.append(index)
            was_midpoint_mask.append(True)
    return anchors, anchor_indices, is_anchor_arr, was_midpoint_mask


@numba.njit
def _add_last_points_to_end(
    coords: NDArray[np.float64],
    indices: NDArray[np.int32],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int32],
]:
    out_coords, out_indices = [coords[0]], [indices[0]]
    last_coords = []
    prev = coords[0]
    first_coords = prev
    n_minus_1 = len(coords) - 1
    for i in np.arange(1, len(coords)):
        idx = indices[i]
        xy = coords[i]
        distance_to_prev: float = np.sqrt(
            (xy[0] - prev[0]) ** 2 + (xy[1] - prev[1]) ** 2
        )
        if idx != indices[i - 1]:
            first_coords = xy
            out_coords.append(xy)
            out_indices.append(idx)
        elif not distance_to_prev:
            if i == n_minus_1 or idx != indices[i + 1]:
                last_coords.append(xy)
            prev = xy
            continue
        elif i == n_minus_1 or idx != indices[i + 1]:
            out_coords.append(xy)
            out_coords.append(first_coords)
            out_indices.append(idx)
            out_indices.append(idx)
            last_coords.append(xy)
        else:
            out_coords.append(xy)
            out_indices.append(idx)

        prev = xy

    return (out_coords, out_indices)


@numba.njit
def _add_last_points_to_end_with_third_arr(
    coords: NDArray[np.float64],
    indices: NDArray[np.int32],
    third_arr: NDArray[Any],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[Any],
]:
    out_coords, out_indices, out_third_arr = [coords[0]], [indices[0]], [third_arr[0]]
    last_coords = []
    prev = coords[0]
    first_coords = prev
    n_minus_1 = len(coords) - 1
    for i in np.arange(1, len(coords)):
        idx = indices[i]
        xy = coords[i]
        distance_to_prev: float = np.sqrt(
            (xy[0] - prev[0]) ** 2 + (xy[1] - prev[1]) ** 2
        )
        if idx != indices[i - 1]:
            first_coords = xy
            out_coords.append(xy)
            out_indices.append(idx)
            out_third_arr.append(third_arr[i])
        elif not distance_to_prev:
            if i == n_minus_1 or idx != indices[i + 1]:
                last_coords.append(xy)
            prev = xy
            continue
        elif i == n_minus_1 or idx != indices[i + 1]:
            out_coords.append(xy)
            out_coords.append(first_coords)
            out_indices.append(idx)
            out_indices.append(idx)
            last_coords.append(xy)
            out_third_arr.append(third_arr[i])
            out_third_arr.append(third_arr[i])
        else:
            out_coords.append(xy)
            out_indices.append(idx)
            out_third_arr.append(third_arr[i])

        prev = xy

    return (out_coords, out_indices, out_third_arr)


@numba.njit
def _remove_duplicate_points(
    coords: NDArray[np.float64],
    indices: NDArray[np.int32],
    third_arr: NDArray[Any],
):
    out_coords, out_indices, out_third_arr = [coords[0]], [indices[0]], [third_arr[0]]
    prev = coords[0]
    for i in np.arange(1, len(coords)):
        idx = indices[i]
        xy = coords[i]
        distance_to_prev: float = np.sqrt(
            (xy[0] - prev[0]) ** 2 + (xy[1] - prev[1]) ** 2
        )
        if not distance_to_prev and idx == indices[i - 1]:
            prev = xy
            continue

        if idx != indices[i - 1]:
            out_coords.append(xy)
            out_indices.append(idx)
            out_third_arr.append(third_arr[i])
            prev = xy
            continue

        out_coords.append(xy)
        out_indices.append(idx)
        out_third_arr.append(third_arr[i])
        prev = xy

    return out_coords, out_indices, out_third_arr


def _snap_linearrings(
    geoms: NDArray[LinearRing],
    tolerance: int | float,
    mask: GeoDataFrame | None,
    snap_to_anchors: bool = True,
):
    if not len(geoms):
        return geoms

    points = GeoDataFrame(
        {
            "geometry": extract_unique_points(geoms),
            "_geom_idx": np.arange(len(geoms)),
        }
    ).explode(ignore_index=True)
    coords = get_coordinates(points.geometry.values)
    indices = points["_geom_idx"].values

    if mask is not None:
        mask_coords, mask_indices = get_coordinates(
            mask.geometry.values, return_index=True
        )
        is_anchor = np.full(len(mask_coords), False)

        mask_coords, mask_indices, is_anchor = _remove_duplicate_points(
            mask_coords, mask_indices, is_anchor
        )
        mask_coords, mask_indices = _add_last_points_to_end(mask_coords, mask_indices)
        mask_coords = np.array(mask_coords)
        mask_indices = np.array(mask_indices)

        is_anchor = np.full(len(mask_coords), False)
        mask_coords, mask_indices, is_anchor = _remove_duplicate_points(
            mask_coords, mask_indices, is_anchor
        )
        mask_coords = np.array(mask_coords)
        mask_indices = np.array(mask_indices)

        original_mask_buffered = shapely.buffer(
            shapely.linearrings(mask_coords, indices=mask_indices),
            tolerance * 1.1,
        )
        mask_coords, mask_indices, was_midpoint_mask, _ = (
            _add_midpoints_to_segments_numba(
                mask_coords,
                mask_indices,
                get_coordinates(
                    sfilter(
                        points.geometry.drop_duplicates(),
                        original_mask_buffered,
                    )
                ),
                tolerance * 1.1,
            )
        )

        mask_coords = np.array(mask_coords)
        mask_indices = np.array(mask_indices)
        mask_indices = (mask_indices + 1) * -1

    is_anchor = np.full(len(coords), False)
    coords, indices, is_anchor = _remove_duplicate_points(coords, indices, is_anchor)

    coords, indices = _add_last_points_to_end(coords, indices)
    coords = np.array(coords)
    indices = np.array(indices)

    is_anchor = np.full(len(coords), False)
    coords, indices, is_anchor = _remove_duplicate_points(coords, indices, is_anchor)
    coords = np.array(coords)
    indices = np.array(indices)

    # if 0:
    #     coords, indices, was_midpoint, _ = _add_midpoints_to_segments_numba(
    #         coords,
    #         indices,
    #         mask_coords,
    #         tolerance * 1.1,  # + PRECISION * 100,
    #     )

    #     was_midpoint = np.array(was_midpoint)

    #     coords, is_snapped_to = _snap_to_anchors(
    #         coords,
    #         indices,
    #         mask_coords,
    #         mask_indices,
    #         mask_coords,
    #         mask_indices,
    #         was_midpoint,
    #         was_midpoint_mask,
    #         tolerance + PRECISION * 20,
    #     )
    #     indices = np.array(indices)
    #     coords = np.array(coords)

    #     indices = indices[coords[:, 0] != np.inf]
    #     coords = coords[coords[:, 0] != np.inf]

    if snap_to_anchors:
        if mask is None:
            mask_coords = [coords[0]]
            mask_indices = [indices[0]]
            was_midpoint_mask = [False]
        anchors, anchor_indices, is_anchor, was_midpoint_anchors = _build_anchors(
            coords,
            indices,
            mask_coords,
            mask_indices,
            was_midpoint_mask,
            tolerance + PRECISION,  # * 100
        )
        anchors = np.array(anchors)
        anchor_indices = np.array(anchor_indices)

        # anchors = np.round(anchors, 3)

    else:
        anchors, anchor_indices, was_midpoint_anchors = (
            mask_coords,
            mask_indices,
            was_midpoint_mask,
        )

    coords, indices, was_midpoint, _ = _add_midpoints_to_segments_numba(
        coords,
        indices,
        anchors,
        tolerance * 1.1,
    )

    was_midpoint = np.array(was_midpoint)

    coords_up_here000 = (
        pd.Series(_coords_to_rings(np.array(coords), np.array(indices), geoms))
        .loc[lambda x: x.notna()]
        .values
    )
    coords_up_here000 = to_gdf(polygons(coords_up_here000), 25833)

    coords, indices, was_midpoint = _add_last_points_to_end_with_third_arr(
        coords, indices, was_midpoint
    )

    coords, indices, was_midpoint = _remove_duplicate_points(
        coords, indices, was_midpoint
    )

    coords = np.array(coords)
    indices = np.array(indices)
    was_midpoint = np.array(was_midpoint)

    coords_up_here = (
        pd.Series(_coords_to_rings(coords, indices, geoms))
        .loc[lambda x: x.notna()]
        .values
    )
    coords_up_here = to_gdf(polygons(coords_up_here), 25833)

    explore(
        coords=to_gdf(shapely.points(coords), 25833).assign(
            idx=indices, wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        anchors=to_gdf(shapely.points(anchors), 25833).assign(
            idx=anchor_indices, wkt=lambda x: [g.wkt for g in x.geometry]
        ),  # , straight_distances=straight_distances, distances_to_lines=distances_to_lines),
        coords_up_here000=coords_up_here000,
        coords_up_here=coords_up_here,
        geoms=to_gdf(polygons(geoms), 25833),
        msk=to_gdf(shapely.points(mask_coords), 25833).assign(
            was_midpoint_mask=was_midpoint_mask
        ),
        # center=_DEBUG_CONFIG["center"],
    )

    coords, indices = _snap_to_anchors(
        coords,
        indices,
        anchors,
        anchor_indices,
        mask_coords,
        mask_indices,
        was_midpoint,
        was_midpoint_anchors,
        tolerance + PRECISION * 100,
    )
    indices = np.array(indices)
    coords = np.array(coords)
    indices = indices[coords[:, 0] != np.inf]
    coords = coords[coords[:, 0] != np.inf]

    # coords_up_here111 = (
    #     pd.Series(_coords_to_rings(coords, indices, geoms))
    #     .loc[lambda x: x.notna()]
    #     .values
    # )
    # coords_up_here111 = to_gdf(polygons(coords_up_here111), 25833)

    # if 0:
    #     # coords = get_coordinates(points.geometry.values)
    #     # indices = points["_geom_idx"].values

    #     is_anchor = np.full(len(coords), False)
    #     coords, indices, is_anchor = _remove_duplicate_points(
    #         coords, indices, is_anchor
    #     )
    #     coords, indices = _add_last_points_to_end(coords, indices)
    #     coords = np.array(coords)
    #     indices = np.array(indices)
    #     is_anchor = np.full(len(coords), False)
    #     coords, indices, is_anchor = _remove_duplicate_points(
    #         coords, indices, is_anchor
    #     )
    #     coords = np.array(coords)
    #     indices = np.array(indices)

    # display(pd.DataFrame(coords, index=indices, columns=[*"xy"]))

    # if 0:
    #     mask_coords, mask_indices, , dist_to_closest_geom = (
    #         _add_midpoints_to_segments_numba(
    #             mask_coords,
    #             mask_indices,
    #             # coords,
    #             get_coordinates(
    #                 sfilter(
    #                     GeoSeries(shapely.points(coords)).drop_duplicates(),
    #                     original_mask_buffered,
    #                 )
    #             ),
    #             tolerance * 1.1,
    #         )
    #     )

    #     mask_coords = np.array(mask_coords)
    #     mask_indices = np.array(mask_indices)

    #     anchors, anchor_indices, is_anchor = _build_anchors(
    #         coords,
    #         indices,
    #         mask_coords,
    #         mask_indices,
    #         # is_anchor,
    #         tolerance + PRECISION,  # * 100
    #     )
    #     anchors = np.array(anchors)
    #     anchor_indices = np.array(anchor_indices)

    #     coords, indices, was_midpoint, _ = _add_midpoints_to_segments_numba(
    #         coords,
    #         indices,
    #         anchors,
    #         tolerance * 1.1,  # + PRECISION * 100,
    #         # GeoDataFrame({"geometry": shapely.points(coords), "_geom_idx": indices}),
    #         # GeoDataFrame({"geometry": shapely.points(anchors)}),
    #         # tolerance,  # + PRECISION * 100,
    #         # None,
    #     )
    #     print(len(coords), len(anchors), len(was_midpoint))

    #     indices = np.array(indices)
    #     coords = np.array(coords)

    #     was_midpoint = np.array(was_midpoint)

    #     coords, is_snapped_to = _snap_to_anchors(
    #         coords,
    #         indices,
    #         anchors,
    #         anchor_indices,
    #         mask_coords,
    #         mask_indices,
    #         was_midpoint,
    #         was_midpoint_anchors,
    #         tolerance + PRECISION * 20,
    #     )
    #     indices = np.array(indices)
    #     coords = np.array(coords)
    #     indices = indices[coords[:, 0] != np.inf]
    #     coords = coords[coords[:, 0] != np.inf]

    # coords = np.array(coords)

    # indices = np.array(indices)

    coords_down_here = (
        pd.Series(_coords_to_rings(coords, indices, geoms))
        .loc[lambda x: x.notna()]
        .values
    )
    lines_down_here = to_gdf(shapely.buffer(coords_down_here, 0.1), 25833)
    coords_down_here = to_gdf(polygons(coords_down_here), 25833)

    try:
        explore(
            coords=to_gdf(shapely.points(coords), 25833).assign(
                idx=indices, wkt=lambda x: [g.wkt for g in x.geometry]
            ),
            anchors=to_gdf(shapely.points(anchors), 25833).assign(
                idx=anchor_indices, wkt=lambda x: [g.wkt for g in x.geometry]
            ),  # , straight_distances=straight_distances, distances_to_lines=distances_to_lines),
            coords_up_here000=coords_up_here000,
            coords_up_here=coords_up_here,
            coords_down_here=coords_down_here,
            lines_down_here=lines_down_here,
            geoms=to_gdf(polygons(geoms), 25833),
            msk=to_gdf(shapely.points(mask_coords), 25833).assign(
                was_midpoint_mask=was_midpoint_mask
            ),
        )

        explore(
            coords=to_gdf(shapely.points(coords), 25833).assign(
                idx=indices, wkt=lambda x: [g.wkt for g in x.geometry]
            ),
            anchors=to_gdf(shapely.points(anchors), 25833).assign(
                idx=anchor_indices, wkt=lambda x: [g.wkt for g in x.geometry]
            ),  # , straight_distances=straight_distances, distances_to_lines=distances_to_lines),
            coords_up_here000=coords_up_here000,
            coords_up_here=coords_up_here,
            coords_down_here=coords_down_here,
            lines_down_here=lines_down_here,
            geoms=to_gdf(polygons(geoms), 25833),
            msk=to_gdf(shapely.points(mask_coords), 25833).assign(
                was_midpoint_mask=was_midpoint_mask
            ),
            center=(5.37707159, 59.01065276, 1),
        )
        explore(
            coords=to_gdf(shapely.points(coords), 25833).assign(
                idx=indices, wkt=lambda x: [g.wkt for g in x.geometry]
            ),
            anchors=to_gdf(shapely.points(anchors), 25833).assign(
                idx=anchor_indices, wkt=lambda x: [g.wkt for g in x.geometry]
            ),  # , straight_distances=straight_distances, distances_to_lines=distances_to_lines),
            coords_up_here000=coords_up_here000,
            coords_up_here=coords_up_here,
            coords_down_here=coords_down_here,
            lines_down_here=lines_down_here,
            geoms=to_gdf(polygons(geoms), 25833),
            msk=to_gdf(shapely.points(mask_coords), 25833).assign(
                was_midpoint_mask=was_midpoint_mask
            ),
            center=(5.37419946, 59.01138812, 15),
        )

        explore(
            coords=to_gdf(shapely.points(coords), 25833).assign(
                idx=indices, wkt=lambda x: [g.wkt for g in x.geometry]
            ),
            anchors=to_gdf(shapely.points(anchors), 25833).assign(
                idx=anchor_indices, wkt=lambda x: [g.wkt for g in x.geometry]
            ),  # , straight_distances=straight_distances, distances_to_lines=distances_to_lines),
            coords_up_here000=coords_up_here000,
            coords_up_here=coords_up_here,
            lines_down_here=lines_down_here,
            coords_down_here=coords_down_here,
            geoms=to_gdf(polygons(geoms), 25833),
            msk=to_gdf(shapely.points(mask_coords), 25833).assign(
                was_midpoint_mask=was_midpoint_mask
            ),
            center=(5.38389153, 59.00548223, 1),
        )
        explore(
            coords=to_gdf(shapely.points(coords), 25833).assign(
                idx=indices, wkt=lambda x: [g.wkt for g in x.geometry]
            ),
            anchors=to_gdf(shapely.points(anchors), 25833).assign(
                idx=anchor_indices, wkt=lambda x: [g.wkt for g in x.geometry]
            ),  # , straight_distances=straight_distances, distances_to_lines=distances_to_lines),
            coords_up_here000=coords_up_here000,
            coords_up_here=coords_up_here,
            coords_down_here=coords_down_here,
            lines_down_here=lines_down_here,
            geoms=to_gdf(polygons(geoms), 25833),
            msk=to_gdf(shapely.points(mask_coords), 25833).assign(
                was_midpoint_mask=was_midpoint_mask
            ),
            center=_DEBUG_CONFIG["center"],
        )

    except GEOSException as e:
        print(e)

    return _coords_to_rings(coords, indices, geoms)


def _coords_to_rings(
    coords: NDArray[np.float64],
    indices: NDArray[np.int32],
    original_geoms: NDArray[LinearRing],
) -> NDArray[LinearRing]:
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]}, index=indices).loc[
        lambda x: x.groupby(level=0).size() > 2
    ]
    to_int_idx = {idx: i for i, idx in enumerate(df.index.unique())}
    rings = pd.Series(
        linearrings(df.values, indices=df.index.map(to_int_idx)),
        index=df.index.unique(),
    )

    missing = pd.Series(
        index=pd.Index(range(len(original_geoms))).difference(rings.index)
    )

    return pd.concat([rings, missing]).sort_index().values


@numba.njit
def _add_midpoints_to_segments_numba(
    geoms: NDArray[np.float64],
    indices: NDArray[np.int32],
    anchors: NDArray[np.float64],
    tolerance: int | float,
):
    n_minus_1 = len(geoms) - 1
    out_coords = []
    out_indices = []
    was_midpoint = []
    out_distances = []
    for i in range(len(geoms)):
        index = indices[i]

        is_last = i == n_minus_1 or index != indices[i + 1]
        if is_last:
            continue

        geom0 = geoms[i]
        geom1 = geoms[i + 1]

        closest_points = np.full((len(anchors) + 2, 2), np.inf)
        these_out_distances = np.full(len(anchors) + 2, np.inf)
        closest_points[-1] = geom1
        closest_points[-2] = geom0
        these_out_distances[-1] = 0
        these_out_distances[-2] = 0

        segment_vector = geom1 - geom0
        segment_length_squared = np.dot(segment_vector, segment_vector)
        for j in range(len(anchors)):
            anchor = anchors[j]

            if segment_length_squared == 0:
                closest_point = geom0
            else:
                point_vector = anchor - geom0
                factor = np.dot(point_vector, segment_vector) / segment_length_squared
                factor = max(0, min(1, factor))
                if factor < 1e-6:
                    closest_point = geom0
                elif factor > 1 - 1e-6:
                    closest_point = geom1
                else:
                    closest_point = geom0 + factor * segment_vector

            dist = np.linalg.norm(anchor - closest_point)
            if dist <= tolerance and dist > PRECISION:
                closest_points[j] = closest_point
                these_out_distances[j] = dist

            # if (
            #     closest_point[0] == 905049.3317999999
            # ):  # and int(closest_point[1]) == 7877676:
            #     print()
            #     for xxx in closest_point:
            #         print(xxx)
            #     for xxx in geom0:
            #         print(xxx)
            #     for xxx in geom1:
            #         print(xxx)
            #     for xxx, yyy in locals().items():
            #         print(xxx, yyy)
            #     ssss

        not_inf = closest_points[:, 0] != np.inf
        arr = closest_points[not_inf]
        these_out_distances = these_out_distances[not_inf]

        # sort by first and second column
        # could have used np.lexsort, but it's not numba compatible
        arr = arr[np.argsort(arr[:, 0])]
        any_unsorted = True
        while any_unsorted:
            any_unsorted = False
            for i in range(len(arr) - 1):
                if arr[i, 0] < arr[i + 1, 0]:
                    continue
                if arr[i, 1] > arr[i + 1, 1]:
                    copied = arr[i].copy()
                    arr[i] = arr[i + 1]
                    arr[i + 1] = copied

                    copied = these_out_distances[i]
                    these_out_distances[i] = these_out_distances[i + 1]
                    these_out_distances[i + 1] = copied

                    any_unsorted = True

        with_midpoints = []
        these_out_distances2 = []
        first_is_added = False
        last_is_added = False
        is_reverse = False
        for y in range(len(arr)):
            point = arr[y]
            if (
                not first_is_added
                and np.sqrt((geom0[0] - point[0]) ** 2 + (geom0[1] - point[1]) ** 2)
                == 0
            ):
                first_is_added = True
                with_midpoints.append(point)
                these_out_distances2.append(these_out_distances[y])
                if last_is_added:
                    is_reverse = True
                    break
                else:
                    continue
            elif (
                not last_is_added
                and np.sqrt((geom1[0] - point[0]) ** 2 + (geom1[1] - point[1]) ** 2)
                == 0
            ):
                last_is_added = True
                with_midpoints.append(point)
                these_out_distances2.append(these_out_distances[y])
                if not first_is_added:
                    is_reverse = True
                    continue
                else:
                    with_midpoints.append(point)
                    break
            if first_is_added or last_is_added:
                with_midpoints.append(point)
                these_out_distances2.append(these_out_distances[y])

            # these_out_distances2.append(these_out_distances[y])
            # these_anchors2.append(these_anchors[y])

        # with_midpoints = np.array(with_midpoints)

        if is_reverse:
            with_midpoints = with_midpoints[::-1]
            these_out_distances2 = these_out_distances2[::-1]
            # these_anchors2 = these_anchors2[::-1]

        # print(index, is_reverse, arr)
        # print(with_midpoints)
        # print(to_gdf(LineString([geom0, geom1]), 25833))
        # print(to_gdf(shapely.points(closest_points)))
        # explore(
        #     to_gdf(shapely.points(with_midpoints)).assign(
        #         idx=lambda x: range(len(x))
        #     ),
        #     "idx",
        # )
        # explore(
        #     l=to_gdf(LineString([geom0, geom1]), 25833),
        #     # anchors=to_gdf(shapely.points(anchors)),
        #     # anchors_in_dist=to_gdf(shapely.points(these_anchors)),
        #     # closest_points=to_gdf(shapely.points(closest_points)),
        #     with_midpoints=to_gdf(shapely.points(with_midpoints)),
        #     anchors=to_gdf(shapely.points(anchors)),
        #     arr=to_gdf(shapely.points(arr)),
        #     # center=(-0.07034028, 1.80337784, 0.4),
        # )

        with_midpoints_no_dups = []
        these_out_distances_no_dups = []

        for y2 in range(len(with_midpoints)):
            point = with_midpoints[y2]
            should_be_added = True
            for z in range(len(with_midpoints_no_dups)):
                out_point = with_midpoints_no_dups[z]
                if (
                    np.sqrt(
                        (point[0] - out_point[0]) ** 2 + (out_point[1] - point[1]) ** 2
                    )
                    == 0
                ):
                    should_be_added = False
                    break
            if should_be_added:
                with_midpoints_no_dups.append(point)
                these_out_distances_no_dups.append(these_out_distances2[y2])

        n_minus_1_midpoints = len(with_midpoints_no_dups) - 1
        for y3 in range(len(with_midpoints_no_dups)):
            point = with_midpoints_no_dups[y3]
            should_be_added = True

            for z2 in np.arange(len(out_coords))[::-1]:
                if out_indices[z2] != index:
                    continue
                out_point = out_coords[z2]

                if (
                    np.sqrt(
                        (point[0] - out_point[0]) ** 2 + (out_point[1] - point[1]) ** 2
                    )
                    == 0
                ):
                    should_be_added = False
                    break

            if not should_be_added:
                continue

            out_coords.append(point)
            out_indices.append(index)
            out_distances.append(these_out_distances_no_dups[y3])
            if y3 == 0 or y3 == n_minus_1_midpoints:
                was_midpoint.append(False)
            else:
                was_midpoint.append(True)

    return (
        out_coords,
        out_indices,
        was_midpoint,
        out_distances,
    )


def _separate_single_neighbored_from_multi_neighoured_geometries(
    gdf: GeoDataFrame, neighbors: GeoDataFrame
) -> tuple[GeoDataFrame, GeoDataFrame]:
    """Split GeoDataFrame in two: those with 0 or 1 neighbors and those with 2 or more.

    Because single-neighbored polygons does not need splitting.
    """
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
        gdf, to_eliminate, ignore_index=ignore_index, **kwargs
    )

    if not was_multiple_gdfs:
        return gdf, isolated

    gdfs = ()
    for i, cols in enumerate(original_cols):
        df = gdf.loc[gdf["_df_idx"] == i, cols]
        gdfs += (df,)
    gdfs += (isolated,)

    return gdfs

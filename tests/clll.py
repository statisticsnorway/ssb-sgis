# %%
import random
import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numba
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

src = str(Path(__file__).parent).replace("tests", "") + "src"
sys.path.insert(0, src)

import sgis as sg
from sgis.debug_config import _DEBUG_CONFIG
from sgis.debug_config import _try_debug_print
from sgis.geopandas_tools.buffer_dissolve_explode import dissexp
from sgis.geopandas_tools.buffer_dissolve_explode import dissexp_by_cluster
from sgis.geopandas_tools.conversion import to_gdf
from sgis.geopandas_tools.conversion import to_geoseries
from sgis.geopandas_tools.duplicates import update_geometries
from sgis.geopandas_tools.general import clean_geoms
from sgis.geopandas_tools.geometry_types import make_all_singlepart
from sgis.geopandas_tools.geometry_types import to_single_geom_type
from sgis.geopandas_tools.overlay import clean_overlay
from sgis.geopandas_tools.polygon_operations import close_all_holes
from sgis.geopandas_tools.polygon_operations import eliminate_by_longest
from sgis.geopandas_tools.polygon_operations import get_gaps
from sgis.geopandas_tools.polygon_operations import split_by_neighbors
from sgis.geopandas_tools.polygons_as_rings import PolygonsAsRings
from sgis.geopandas_tools.sfilter import sfilter
from sgis.geopandas_tools.sfilter import sfilter_inverse
from sgis.geopandas_tools.sfilter import sfilter_split
from sgis.maps.maps import explore
from sgis.maps.maps import explore_locals

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


PRECISION = 1e-3
BUFFER_RES = 50


# def xxx(func):
#     def wrapper(*args, **kwargs):
#         result = func(*args, **kwargs)
#         return result

#     return wrapper


# numba.njit = xxx


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
        mask: Mask to clip gdf to.
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

    if mask is None:
        mask: GeoDataFrame = close_all_holes(
            dissexp_by_cluster(gdf[["geometry"]])
        ).pipe(make_all_singlepart)
        mask = GeoDataFrame(
            {
                "geometry": [
                    mask.union_all()
                    .buffer(
                        PRECISION,
                        resolution=1,
                        join_style=2,
                    )
                    .buffer(
                        -PRECISION,
                        resolution=1,
                        join_style=2,
                    )
                ]
            },
            crs=gdf.crs,
        ).pipe(make_all_singlepart)
        # gaps = shapely.union_all(get_gaps(mask).geometry.values)
        # mask = shapely.get_parts(extract_unique_points(mask.geometry.values))
        # not_by_gaps = shapely.distance(mask, gaps) > PRECISION
        # mask = GeoDataFrame({"geometry": mask[not_by_gaps]})
    else:
        try:
            mask: GeoDataFrame = mask[["geometry"]].pipe(make_all_singlepart)
        except Exception:
            mask: GeoDataFrame = (
                to_geoseries(mask).to_frame("geometry").pipe(make_all_singlepart)
            )

    gdf = snap_polygons(gdf, tolerance, mask=mask)

    missing_from_gdf = sfilter_inverse(gdf_original, gdf.buffer(-PRECISION)).loc[
        lambda x: (~x.buffer(-PRECISION).is_empty)
    ]
    return pd.concat([gdf, missing_from_gdf], ignore_index=True)

    # missing_from_gdf = sfilter_inverse(gdf_original, gdf).loc[
    missing_from_gdf = clean_overlay(
        gdf_original, gdf, how="difference", geom_type="polygon"
    ).loc[
        lambda x: (
            ~x.buffer(-PRECISION).is_empty
        )  # & (~x.buffer(-tolerance / 2).is_empty)
    ]

    return pd.concat([gdf, missing_from_gdf], ignore_index=True).pipe(
        update_geometries, geom_type="polygon"
    )

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

    # errors can occur, so keeping only polygons within gdf after negative buffer
    # resetting index to be able to do iloc, in case of non-unique index
    thick_missing_from_gdf = thick_missing_from_gdf.iloc[
        lambda x: sfilter_inverse(
            x.buffer(-tolerance / 2).reset_index(drop=True), gdf  # , predicate="within"
        ).index
    ]

    # missing_grid_size_01 = clean_overlay(
    #     gdf_original, gdf, how="difference", geom_type="polygon", grid_size=0.1
    # )

    # missing_grid_size_001 = clean_overlay(
    #     gdf_original, gdf, how="difference", geom_type="polygon", grid_size=0.01
    # )

    # missing_grid_size_0001 = clean_overlay(
    #     gdf_original, gdf, how="difference", geom_type="polygon", grid_size=0.001
    # )

    is_thin = gdf.buffer(-tolerance / 2).is_empty
    thin, gdf = gdf[is_thin], gdf[~is_thin]
    gaps = get_gaps(gdf, include_interiors=True).loc[
        lambda x: x.buffer(-tolerance / 2).is_empty
    ]
    to_eliminate = pd.concat([thin_missing_from_gdf, thin, gaps], ignore_index=True)
    assert gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all()

    gdf, isolated = split_and_eliminate_by_longest(gdf, to_eliminate, tolerance)
    # assert gdf.ARTYPE.notna().all(), gdf[gdf.ARTYPE.isna()]
    assert gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all()

    gdf = pd.concat(_eliminate_not_really_isolated(gdf, isolated), ignore_index=True)
    # assert gdf.ARTYPE.notna().all(), gdf[gdf.ARTYPE.isna()]

    _try_debug_print("etter eliminate 1 (med gdf)")
    explore_locals(
        gdf=gdf,
        isolated=isolated,
        to_eliminate=to_eliminate,
        # dissexped=dissexp(gdf, by="_gdf_range_idx", dropna=False, as_index=False),
        points=gdf.extract_unique_points().explode().to_frame(),
        center=(5.76394723, 58.82643877, 1),
    )
    explore(
        gdf_original,
        gdf=gdf,
        isolated=isolated,
        to_eliminate=to_eliminate,
        # dissexped=dissexp(gdf, by="_gdf_range_idx", dropna=False, as_index=False),
        points=gdf.extract_unique_points().explode().to_frame(),
        center=_DEBUG_CONFIG["center"],
    )

    assert gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all()

    return gdf.drop(columns="_gdf_range_idx", errors="ignore")

    gdf = dissexp(gdf, by="_gdf_range_idx", dropna=False, as_index=False)

    # assert gdf.ARTYPE.notna().all(), gdf[gdf.ARTYPE.isna()]

    # we don't want the thick polygons from the mask
    thin_missing_from_mask = clean_overlay(
        mask, gdf, how="difference", geom_type="polygon"
    ).loc[lambda x: x.buffer(-tolerance / 2).is_empty]

    _try_debug_print("split_by_neighbors 2 mask")

    assert gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all()

    gdf, isolated = split_and_eliminate_by_longest(
        gdf, thin_missing_from_mask, tolerance
    )
    # assert gdf.ARTYPE.notna().all(), gdf[gdf.ARTYPE.isna()]
    gdf, _ = _eliminate_not_really_isolated(gdf, isolated)
    # assert gdf.ARTYPE.notna().all(), gdf[gdf.ARTYPE.isna()]

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
    # _try_debug_print("split_by_neighbors 2")

    # single_neighbored, multi_neighbored = (
    #     _separate_single_neighbored_from_multi_neighoured_geometries(
    #         thin_missing_from_mask, gdf
    #     )
    # )

    # multi_neighbored = split_by_neighbors(multi_neighbored, gdf, tolerance=tolerance)
    # thin_missing_from_mask = pd.concat([multi_neighbored, single_neighbored])
    # thin_missing_from_mask["_was_to_eliminate"] = 1
    # _try_debug_print("eliminate_by_longest again with mask")
    # gdf_between = gdf.copy()
    # gdf, isolated = eliminate_by_longest(
    #     gdf, thin_missing_from_mask
    # )
    # gdf = gdf.explode(ignore_index=True)
    # assert not len(isolated), (explore(isolated, gdf, browser=True), isolated)

    _try_debug_print("etter eliminate 2 (av mask)")
    explore(
        gdf=gdf,
        thin_missing_from_mask=thin_missing_from_mask,
        thin_missing_from_gdf=thin_missing_from_gdf,
        thick_missing_from_gdf=thick_missing_from_gdf,
        # thick_missing_from_gdf2=thick_missing_from_gdf2,
        # missing_grid_size_01=missing_grid_size_01,
        # missing_grid_size_001=missing_grid_size_001,
        # missing_grid_size_0001=missing_grid_size_0001,
        thin=thin,
        to_eliminate=to_eliminate,
        center=_DEBUG_CONFIG["center"],
    )

    gdf = clean_overlay(gdf, mask, how="intersection", geom_type="polygon")
    # assert gdf.ARTYPE.notna().all(), gdf[gdf.ARTYPE.isna()]

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
            eliminate_by_longest(gdf[~was_to_eliminate], still_not_eliminated).explode(
                ignore_index=True
            )
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

    # return pd.concat([gdf, thick_missing_from_gdf], ignore_index=True)

    # TODO this can create duble surfaces because of errors
    gdf = pd.concat([gdf, thick_missing_from_gdf], ignore_index=True)
    # assert gdf.ARTYPE.notna().all(), gdf[gdf.ARTYPE.isna()]

    dissappeared = sfilter_inverse(gdf_original, gdf.buffer(-PRECISION)).loc[
        lambda x: ~x.buffer(-PRECISION).is_empty
    ]
    # assert dissappeared.ARTYPE.notna().all()

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

    # if mask is None:
    #     mask: GeoDataFrame = (
    #         close_all_holes(dissexp_by_cluster(gdf))
    #         # .dissolve()
    #         .pipe(make_all_singlepart)
    #     )
    # else:
    #     try:
    #         mask: GeoDataFrame = mask[["geometry"]].pipe(make_all_singlepart)
    #     except Exception:
    #         mask: GeoDataFrame = (
    #             to_geoseries(mask).to_frame("geometry").pipe(make_all_singlepart)
    #         )
    #     mask.crs = None

    # thin = GeoDataFrame()  # gdf[lambda x: x.buffer(-tolerance / 2).is_empty]
    # thin = gdf[lambda x: x.buffer(-tolerance / 2 - PRECISION).is_empty]
    gdf = gdf[lambda x: ~x.buffer(-tolerance / 2 - PRECISION).is_empty]

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

    missing = get_missing(gdf_orig, gdf).loc[
        lambda x: ~x.buffer(-tolerance / 2).is_empty
    ]

    sg.explore(
        gdf,
        # gdf_orig,
        # thin,
        mask,
        missing,
        mask_p=sg.to_gdf(mask.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        gdf_p=sg.to_gdf(gdf.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        center=(5.36765872, 59.01199837, 1),
    )
    sg.explore(
        gdf,
        gdf_orig,
        # thin,
        mask,
        missing,
        mask_p=sg.to_gdf(mask.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        gdf_p=sg.to_gdf(gdf.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        center=(5.36820681, 59.01182298, 2),
    )
    sg.explore(
        gdf,
        gdf_orig,
        # thin,
        mask,
        missing,
        mask_p=sg.to_gdf(mask.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        gdf_p=sg.to_gdf(gdf.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        center=(5.37327042, 59.01099359, 5),
    )
    sg.explore(
        gdf,
        gdf_orig,
        # thin,
        mask,
        missing,
        mask_p=sg.to_gdf(mask.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        gdf_p=sg.to_gdf(gdf.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        center=(5.36853688, 59.01169013, 5),
    )
    sg.explore(
        gdf,
        # gdf_orig,
        missing,
        mask,
        mask_p=sg.to_gdf(mask.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        gdf_p=sg.to_gdf(gdf.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        center=(5.37142966, 59.009799, 0.01),
        max_zoom=40,
    )
    sg.explore(
        gdf,
        # gdf_orig,
        missing,
        mask,
        mask_p=sg.to_gdf(mask.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        gdf_p=sg.to_gdf(gdf.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        center=(5.36866312, 59.00842846, 0.01),
        max_zoom=40,
    )

    sg.explore(
        gdf,
        # gdf_orig,
        missing,
        mask,
        mask_p=sg.to_gdf(mask.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        gdf_p=sg.to_gdf(gdf.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        center=(5.37707146, 59.01065274, 0.4),
        max_zoom=40,
    )

    sg.explore(
        gdf,
        # gdf_orig,
        missing,
        mask,
        mask_p=sg.to_gdf(mask.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        gdf_p=sg.to_gdf(gdf.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        center=(-52074.0241, 6580847.4464, 0.1),
        max_zoom=40,
    )

    sg.explore(
        gdf,
        # gdf_orig,
        missing,
        mask,
        mask_p=sg.to_gdf(mask.extract_unique_points().explode()).assign(
            wkt=lambda x: [g.wkt for g in x.geometry]
        ),
        gdf_p=sg.to_gdf(gdf.extract_unique_points().explode()).assign(
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
    #     # browser=True,
    # )

    # gdf = update_geometries(gdf, geom_type="polygon")

    return gdf  # .pipe(clean_clip, mask, geom_type="polygon")


@numba.njit
def _snap_to_anchors222(
    geoms,
    indices,
    anchors,
    anchor_indices,
    mask,
    mask_indices,
    was_midpoint,
    was_midpoint_mask,
    tolerance: int | float,
) -> tuple[NDArray, NDArray, NDArray]:
    out_coords = geoms.copy()
    # is_snapped = np.full(len(geoms), False)

    n_anchors = len(anchors)
    mask_n_minus_1 = len(mask) - 1
    is_snapped_to = np.full(len(anchors), False)
    mask_is_snapped_to = np.full(len(mask), False)

    for i in range(len(geoms)):
        # if is_snapped[i]:
        #     continue
        geom = geoms[i]
        # index = indices[i]

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
            if dist == 0 and not was_midpoint_mask[j2]:
                break

        if np.min(distances) > tolerance:
            continue

        # if anchor_indices[np.argmin(distances)] == index:
        #     continue

        is_snapped_now = False
        for j3 in np.argsort(distances):
            if distances[j3] > tolerance:
                break
            if not was_midpoint_mask[j3]:
                out_coords[i] = anchors[j3]
                is_snapped_to[j3] = True
                is_snapped_now = True
                break

        if not is_snapped_now:
            out_coords[i] = anchors[np.argmin(distances)]
            is_snapped_to[np.argmin(distances)] = True

    return out_coords, is_snapped_to


@numba.njit
def _build_anchors(
    geoms, indices, mask_coords, mask_indices, was_midpoint_mask, tolerance: int | float
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
            if 1:  # indices[i] != indices[j]:
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
            was_midpoint_mask.append(False)
    return anchors, anchor_indices, is_anchor_arr, was_midpoint_mask


@numba.njit
def _add_last_points_to_end(coords, indices):
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
def _add_last_points_to_end_with_third_arr(coords, indices, third_arr):
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
def _remove_duplicate_points(coords, indices, is_anchor):
    out_coords, out_indices, out_is_anchor = [coords[0]], [indices[0]], [is_anchor[0]]
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
            out_is_anchor.append(is_anchor[i])
            continue

        out_coords.append(xy)
        out_indices.append(idx)
        out_is_anchor.append(is_anchor[i])
        prev = xy

    return out_coords, out_indices, out_is_anchor


def _snap_linearrings(
    geoms: NDArray[LinearRing],
    tolerance: int | float,
    mask: GeoDataFrame | None = None,
    gaps=None,
    # donuts_without_spikes=None,
):
    mmm = shapely.union_all(
        sg.to_gdf("POINT (-51784.678297 6581037.308089)", 25833).buffer(0.001)
    )

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

    mask_coords, mask_indices = get_coordinates(mask.geometry.values, return_index=True)
    is_anchor = np.full(len(mask_coords), False)
    mask_coords, mask_indices, is_anchor = _remove_duplicate_points(
        mask_coords, mask_indices, is_anchor
    )
    mask_coords, mask_indices = _add_last_points_to_end(mask_coords, mask_indices)
    mask_coords = np.array(mask_coords)
    mask_indices = np.array(mask_indices)
    is_anchor = np.array(is_anchor)

    mask_coords, mask_indices, is_anchor = _remove_duplicate_points(
        mask_coords, mask_indices, is_anchor
    )
    mask_coords = np.array(mask_coords)
    mask_indices = np.array(mask_indices)

    original_mask_buffered = shapely.buffer(
        shapely.linearrings(mask_coords, indices=mask_indices),
        tolerance * 1.1,
    )
    if 1:
        mask_coords, mask_indices, was_midpoint_mask, dist_to_closest_geom = (
            _add_midpoints_to_segments_numba(
                mask_coords,
                mask_indices,
                # coords,
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

        # dist_to_closest_geom = np.array(dist_to_closest_geom)
        # mask_range_indices_sorted_by_distance = np.argsort(dist_to_closest_geom)[::-1]
        # mask_coords = mask_coords[mask_range_indices_sorted_by_distance]
    else:
        mask_indices = (mask_indices + 1) * -1

    # coords, indices = get_coordinates(geoms, return_index=True)
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

    #     coords, is_snapped_to = _snap_to_anchors222(
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

    anchors, anchor_indices, is_anchor, was_midpoint_anchors = _build_anchors(
        coords,
        indices,
        mask_coords,
        mask_indices,
        was_midpoint_mask,
        # is_anchor,
        tolerance + PRECISION,  # * 100
    )
    anchors = np.array(anchors)
    anchor_indices = np.array(anchor_indices)

    # is_anchor = np.array(is_anchor)

    # anchors = np.concatenate([mask_coords, np.array(anchors)])
    # anchor_indices = np.concatenate([mask_indices, np.array(anchor_indices)])

    print(len(coords), len(anchors), coords.dtype, anchors.dtype, mask_coords.dtype)
    # coords = coords.astype(float)
    # anchors = anchors.astype(float)

    coords, indices, was_midpoint, _ = _add_midpoints_to_segments_numba(
        coords,
        indices,
        anchors,
        tolerance * 1.1,  # + PRECISION * 100,
        # GeoDataFrame({"geometry": shapely.points(coords), "_geom_idx": indices}),
        # GeoDataFrame({"geometry": shapely.points(anchors)}),
        # tolerance,  # + PRECISION * 100,
        # None,
    )
    print(len(coords), len(anchors), len(was_midpoint))

    was_midpoint = np.array(was_midpoint)

    # print(points)
    # coords = get_coordinates(points.geometry.values)
    # indices = points["_geom_idx"].values

    coords_up_here000 = (
        pd.Series(_coords_to_rings(np.array(coords), np.array(indices), geoms))
        .loc[lambda x: x.notna()]
        .values
    )
    coords_up_here000 = sg.to_gdf(polygons(coords_up_here000), 25833)

    # for i in [0, 1]:
    #     print("haaa", i)
    #     sg.explore(
    #         coords=sg.to_gdf(shapely.points(coords), 25833)
    #         .assign(
    #             idx=indices,
    #             was_midpoint=[str(x) for x in was_midpoint],
    #             wkt=lambda x: [g.wkt for g in x.geometry],
    #         )
    #         .loc[lambda x: x.idx == i],
    #         column="was_midpoint",
    #     )

    print("hei0", np.sum(was_midpoint), len(was_midpoint), len(coords))
    coords, indices, was_midpoint = _add_last_points_to_end_with_third_arr(
        coords, indices, was_midpoint
    )
    print("hei1", np.sum(was_midpoint), len(was_midpoint), len(coords))

    # display(pd.DataFrame(coords, columns=["x", "y"], index=indices))

    coords, indices, was_midpoint = _remove_duplicate_points(
        coords, indices, was_midpoint
    )
    print("hei2", np.sum(was_midpoint), len(was_midpoint), len(coords))

    coords = np.array(coords)
    indices = np.array(indices)
    was_midpoint = np.array(was_midpoint)

    coords_up_here = (
        pd.Series(_coords_to_rings(coords, indices, geoms))
        .loc[lambda x: x.notna()]
        .values
    )
    coords_up_here = sg.to_gdf(polygons(coords_up_here), 25833)

    # sg.explore(
    #     coords=sg.to_gdf(shapely.points(coords), 25833).assign(
    #         idx=indices, wkt=lambda x: [g.wkt for g in x.geometry]
    #     ),
    #     anchors=sg.to_gdf(shapely.points(anchors), 25833).assign(
    #         idx=anchor_indices, wkt=lambda x: [g.wkt for g in x.geometry]
    #     ),  # , straight_distances=straight_distances, distances_to_lines=distances_to_lines),
    #     coords_up_here000=coords_up_here000,
    #     coords_up_here=coords_up_here,
    #     geoms=sg.to_gdf(polygons(geoms), 25833),
    #     msk=sg.to_gdf(shapely.points(mask_coords), 25833).assign(
    #         was_midpoint_mask=was_midpoint_mask
    #     ),
    # )

    coords, is_snapped_to = _snap_to_anchors222(
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
    # coords_up_here111 = sg.to_gdf(polygons(coords_up_here111), 25833)

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

    #     coords, is_snapped_to = _snap_to_anchors222(
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

    # sg.explore(
    #     sg.to_gdf(shapely.points(anchors), 25833).assign(
    #         is_snapped_to=[str(x) for x in is_snapped_to], idx=anchor_indices
    #     ),
    #     "is_snapped_to",
    # )

    # for xy in [
    #     (5.36765774, 59.0119981, 0.1),
    #     (5.37707147, 59.01065279, 0.1),
    #     (5.36780998, 59.01202003, 30),
    # ]:
    #     sg.explore(
    #         sg.to_gdf(shapely.points(anchors), 25833).assign(
    #             is_snapped_to=[str(x) for x in is_snapped_to], idx=anchor_indices
    #         ),
    #         "is_snapped_to",
    #         center=xy,
    #     )

    # coords = np.array(coords)

    # indices = np.array(indices)

    coords_down_here = (
        pd.Series(_coords_to_rings(coords, indices, geoms))
        .loc[lambda x: x.notna()]
        .values
    )
    coords_down_here = sg.to_gdf(polygons(coords_down_here), 25833)
    try:
        sg.explore(
            coords=sg.to_gdf(shapely.points(coords), 25833).assign(
                idx=indices, wkt=lambda x: [g.wkt for g in x.geometry]
            ),
            anchors=sg.to_gdf(shapely.points(anchors), 25833).assign(
                idx=anchor_indices, wkt=lambda x: [g.wkt for g in x.geometry]
            ),  # , straight_distances=straight_distances, distances_to_lines=distances_to_lines),
            coords_up_here000=coords_up_here000,
            coords_up_here=coords_up_here,
            coords_down_here=coords_down_here,
            geoms=sg.to_gdf(polygons(geoms), 25833),
            msk=sg.to_gdf(shapely.points(mask_coords), 25833).assign(
                was_midpoint_mask=was_midpoint_mask
            ),
        )

        sg.explore(
            coords=sg.to_gdf(shapely.points(coords), 25833).assign(
                idx=indices, wkt=lambda x: [g.wkt for g in x.geometry]
            ),
            anchors=sg.to_gdf(shapely.points(anchors), 25833).assign(
                idx=anchor_indices, wkt=lambda x: [g.wkt for g in x.geometry]
            ),  # , straight_distances=straight_distances, distances_to_lines=distances_to_lines),
            coords_up_here000=coords_up_here000,
            coords_up_here=coords_up_here,
            coords_down_here=coords_down_here,
            geoms=sg.to_gdf(polygons(geoms), 25833),
            msk=sg.to_gdf(shapely.points(mask_coords), 25833).assign(
                was_midpoint_mask=was_midpoint_mask
            ),
            center=(5.37707159, 59.01065276, 1),
        )
        sg.explore(
            coords=sg.to_gdf(shapely.points(coords), 25833).assign(
                idx=indices, wkt=lambda x: [g.wkt for g in x.geometry]
            ),
            anchors=sg.to_gdf(shapely.points(anchors), 25833).assign(
                idx=anchor_indices, wkt=lambda x: [g.wkt for g in x.geometry]
            ),  # , straight_distances=straight_distances, distances_to_lines=distances_to_lines),
            coords_up_here000=coords_up_here000,
            coords_up_here=coords_up_here,
            coords_down_here=coords_down_here,
            geoms=sg.to_gdf(polygons(geoms), 25833),
            msk=sg.to_gdf(shapely.points(mask_coords), 25833).assign(
                was_midpoint_mask=was_midpoint_mask
            ),
            center=(5.37419946, 59.01138812, 15),
        )

        sg.explore(
            coords=sg.to_gdf(shapely.points(coords), 25833).assign(
                idx=indices, wkt=lambda x: [g.wkt for g in x.geometry]
            ),
            anchors=sg.to_gdf(shapely.points(anchors), 25833).assign(
                idx=anchor_indices, wkt=lambda x: [g.wkt for g in x.geometry]
            ),  # , straight_distances=straight_distances, distances_to_lines=distances_to_lines),
            coords_up_here000=coords_up_here000,
            coords_up_here=coords_up_here,
            coords_down_here=coords_down_here,
            geoms=sg.to_gdf(polygons(geoms), 25833),
            msk=sg.to_gdf(shapely.points(mask_coords), 25833).assign(
                was_midpoint_mask=was_midpoint_mask
            ),
            center=(5.38389153, 59.00548223, 1),
        )

    except GEOSException as e:
        print(e)

    return _coords_to_rings(coords, indices, geoms)


def _coords_to_rings(coords, indices, original_geoms):
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
    geoms,
    indices,
    anchors,
    # anchor_indices,
    tolerance,
):
    n_minus_1 = len(geoms) - 1
    out_coords = []
    out_indices = []
    was_midpoint = []
    out_distances = []
    # nearest_anchors = []
    for i in range(len(geoms)):
        index = indices[i]

        is_last = i == n_minus_1 or index != indices[i + 1]
        if is_last:
            # out_coords.append(geoms[i])
            # out_indices.append(index)
            continue

        geom0 = geoms[i]
        geom1 = geoms[i + 1]

        # these_out_distances = []
        # these_anchors = []
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

            # # no need to add midpoints if the anchor is within distance of the endpoints
            # is_close_enough_to_an_endpoint = (
            #     np.sqrt((geom0[0] - anchor[0]) ** 2 + (geom0[1] - anchor[1]) ** 2)
            #     <= tolerance
            #     pr np.sqrt((geom1[0] - anchor[0]) ** 2 + (geom1[1] - anchor[1]) ** 2)
            #     <= tolerance
            # )
            # if 0:#is_close_enough_to_an_endpoint:
            #     continue

            if segment_length_squared == 0:
                closest_point = geom0
            else:
                point_vector = anchor - geom0
                factor = np.dot(point_vector, segment_vector) / segment_length_squared
                factor = max(0, min(1, factor))
                closest_point = geom0 + factor * segment_vector

            dist = np.linalg.norm(anchor - closest_point)
            if dist <= tolerance:
                # closest_points.append(closest_point)  # anchor
                closest_points[j] = closest_point
                these_out_distances[j] = dist
                # these_anchors.append(anchor)

        not_inf = closest_points[:, 0] != np.inf
        arr = closest_points[not_inf]
        these_out_distances = these_out_distances[not_inf]
        # arr = np.array([(x, y) for x, y in closest_points if x != np.inf])

        # sort by first and second column
        # could have used np.lexsort, but it's not numba compatible
        if 0:
            these_out_distances = arr[np.lexsort((arr[:, 1], arr[:, 0]))]
            arr = arr[np.lexsort((arr[:, 1], arr[:, 0]))]
        else:
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
        # these_anchors2 = []
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
        # print(sg.to_gdf(LineString([geom0, geom1]), 25833))
        # print(sg.to_gdf(shapely.points(closest_points)))
        # sg.explore(
        #     sg.to_gdf(shapely.points(with_midpoints)).assign(
        #         idx=lambda x: range(len(x))
        #     ),
        #     "idx",
        # )
        # sg.explore(
        #     l=sg.to_gdf(LineString([geom0, geom1]), 25833),
        #     # anchors=sg.to_gdf(shapely.points(anchors)),
        #     # anchors_in_dist=sg.to_gdf(shapely.points(these_anchors)),
        #     # closest_points=sg.to_gdf(shapely.points(closest_points)),
        #     with_midpoints=sg.to_gdf(shapely.points(with_midpoints)),
        #     anchors=sg.to_gdf(shapely.points(anchors)),
        #     arr=sg.to_gdf(shapely.points(arr)),
        #     # center=(-0.07034028, 1.80337784, 0.4),
        # )

        # print("TODO needed if dups are fixed beforehand?")
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


def _eliminate_not_really_isolated(gdf, isolated):
    # TODO
    return gdf, isolated
    gdf = make_all_singlepart(gdf)
    not_really_isolated, really_isolated = sfilter_split(
        isolated, gdf.buffer(PRECISION)
    )
    not_really_isolated.geometry = not_really_isolated.buffer(PRECISION * 10)
    gdf, still_isolated = eliminate_by_longest(
        gdf,
        not_really_isolated,
    )
    assert not len(still_isolated), still_isolated
    gdf = make_all_singlepart(gdf)

    return gdf, really_isolated


sg.coverage_clean = coverage_clean


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


def not_test_clean_complicated_roads():
    for tolerance in [
        0.5,
        0.4,
        0.3,
    ]:
        print(tolerance)

        _test_clean_complicated_roads_base(
            Path(__file__).parent / "testdata/roads_difficult_to_dissolve4.txt",
            "POLYGON ((-32050 6557614, -32050 6556914, -32750 6556914, -32750 6557614, -32050 6557614))",
            tolerance=tolerance,
        )

        _test_clean_complicated_roads_base(
            Path(__file__).parent / "testdata/roads_difficult_to_dissolve3.txt",
            "POLYGON ((28120 6945720, 28120 6945020, 27420 6945020, 27420 6945720, 28120 6945720))",
            tolerance=tolerance,
        )

        _test_clean_complicated_roads_base(
            Path(__file__).parent / "testdata/roads_difficult_to_dissolve2.txt",
            "POLYGON ((270257 6654842, 270257 6654142, 269557 6654142, 269557 6654842, 270257 6654842))",
            tolerance=tolerance,
        )

        _test_clean_complicated_roads_base(
            Path(__file__).parent / "testdata/roads_difficult_to_dissolve.txt",
            "POLYGON ((-49922 6630166, -49922 6629466, -50622 6629466, -50622 6630166, -49922 6630166))",
            tolerance=tolerance,
        )


def test_clean_dissexp():
    from geopandas import GeoDataFrame
    from shapely import extract_unique_points
    from shapely import get_parts

    df = sg.to_gdf(
        [
            "POLYGON ((373693.16000000015 7321024.640000001, 373690.5999999996 7321023.460000001, 373688.5499999998 7321022.210000001, 373686.01999999955 7321021.34, 373685.04000000004 7321020.43, 373684.76999999955 7321019.190000001, 373681.96999999974 7321015.460000001, 373680.11000000034 7321012.82, 373677.33999999985 7321010.59, 373673.21999999974 7321003.699999999, 373671.70999999996 7321002.870000001, 373667.29000000004 7321001.620000001, 373677.5 7321015, 373695 7321030, 373700.8520873802 7321030, 373695.46999999974 7321027.460000001, 373694.63999999966 7321026.039999999, 373693.16000000015 7321024.640000001))",
            "POLYGON ((373700.4003424102 7321029.786805352, 373700.8520873802 7321030, 373700.85208738025 7321030, 373700.4003424102 7321029.786805352))",
        ],
        25833,
    )

    original_points = GeoDataFrame(
        {"geometry": get_parts(extract_unique_points(df.geometry.values))}
    )[lambda x: ~x.geometry.duplicated()]

    cleaned = sg.clean_dissexp(df, dissolve_func=sg.dissexp, by=None)

    print(cleaned)
    sg.explore(cleaned)

    cleaned.geometry = extract_unique_points(cleaned.geometry.values)
    assert cleaned.index.is_unique
    cleaned = cleaned.explode(index_parts=True)

    still_in, gone = sg.sfilter_split(cleaned, original_points.buffer(1e-10))
    sg.explore(still_in, gone, df, cleaned)


def _test_clean_complicated_roads_base(path, mask, tolerance):
    print(path)

    with open(path) as f:
        df = sg.to_gdf(f.readlines(), 25833)

    mask = sg.to_gdf(mask, 25833)
    df["df_index"] = range(len(df))

    cleaned = sg.coverage_clean(df, tolerance, mask=mask)

    gaps = sg.get_gaps(cleaned)
    double = sg.get_intersections(cleaned)
    # missing = get_missing(sg.clean_clip(df, bbox.buffer(-tolerance * 1.1)), cleaned)
    missing = get_missing(df, cleaned)
    sg.explore(df, cleaned, gaps, missing, double)

    print(
        f"tolerance {tolerance}",
        "gaps",
        gaps.area.sum(),
        "dup",
        double.area.sum(),
        "missing",
        missing.area.sum(),
    )

    # check that the geometries still have same column values by ensuring that the range index is the same
    intersected = sg.clean_overlay(df, cleaned, how="intersection", geom_type="polygon")
    area_same_index = intersected[
        lambda x: x["df_index_1"] == x["df_index_2"]
    ].area.sum()
    area_same_index_ratio = area_same_index / intersected.area.sum()
    assert area_same_index_ratio > 0.999, area_same_index_ratio

    assert (
        gaps.area.sum() == 0
    ), f"path: {Path(path).stem}, tolerance {tolerance}, gaps: {gaps.area.sum()}"
    assert (
        missing.area.sum() == 0
    ), f"path: {Path(path).stem}, tolerance {tolerance}, missing: {missing.area.sum()}"
    assert (
        double.area.sum() == 0
    ), f"path: {Path(path).stem}, tolerance {tolerance}, double: {double.area.sum()}"


def test_clean_1144():
    df = gpd.read_parquet(
        Path(__file__).parent / "testdata" / "snap_problem_area_1144.parquet"
    )

    bbox = sg.to_gdf(
        shapely.minimum_rotated_rectangle(shapely.union_all(df.geometry.values)), df.crs
    )

    kommune_utenhav = gpd.read_parquet(
        Path(__file__).parent / "testdata" / "kommune_utenhav_1144_2023.parquet"
    )

    # kommune_utenhav = sg.buff(
    #     kommune_utenhav,
    #     0.001,
    #     resolution=1,
    #     join_style=2,
    # )
    kommune_utenhav = sg.clean_clip(
        kommune_utenhav,
        bbox,
        geom_type="polygon",
    )
    kommune_utenhav_points = (
        kommune_utenhav.extract_unique_points()
        .to_frame("geometry")
        .explode()
        .assign(wkt=lambda x: x.geometry.to_wkt())
    )

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
        "df_index",
        "geometry",
        "kilde",
    ]

    df["df_index"] = range(len(df))

    for tolerance in [
        0.51,
        0.5,
        0.91,
        0.57,
        5,
        1,
        2,
        0.75,
        1.5,
        2.25,
        *[round(random.random() + 0.5, 2) for _ in range(10)],
        *[round(x, 2) for x in np.arange(0.4, 1, 0.01)],
    ]:
        print("\ntolerance")
        print(tolerance)
        # cleaned = sg.coverage_clean(df, tolerance, pre_dissolve_func=_buff).pipe(
        #     sg.clean_clip, df, geom_type="polygon"

        # )

        # allow near-thin polygons to dissappear. this happens because snapping makes them thin
        # before eliminate

        thick_df_indices = df.loc[
            lambda x: ~x.buffer(-tolerance / 1.3).is_empty, "df_index"
        ]

        cleaned = sg.coverage_clean(df, tolerance, mask=kommune_utenhav).pipe(
            lambda x: x  # sg.coverage_clean, tolerance, mask=kommune_utenhav
        )

        # allow edge cases
        cleaned_clipped = sg.clean_clip(cleaned, bbox.buffer(-tolerance * 1.1))

        gaps = sg.get_gaps(cleaned_clipped)

        double = sg.get_intersections(cleaned_clipped)
        missing = get_missing(
            # kommune_utenhav, cleaned
            sg.clean_clip(kommune_utenhav, bbox.buffer(-tolerance * 1.1)),
            cleaned,
            # sg.clean_clip(df, bbox.buffer(-tolerance * 1.1)), cleaned_clipped
        )

        cleaned_points = (
            cleaned.extract_unique_points()
            .to_frame("geometry")
            .explode()
            .assign(wkt=lambda x: x.geometry.to_wkt())
        )
        df_points = (
            df.extract_unique_points()
            .to_frame("geometry")
            .explode()
            .assign(wkt=lambda x: x.geometry.to_wkt())
        )

        sg.explore(
            cleaned,
            gaps,
            double,
            missing,
            df,
            kommune_utenhav,
            center=sg.debug_config._DEBUG_CONFIG["center"],
        )

        sg.explore(
            cleaned,
            gaps,
            double,
            missing,
            df,
            kommune_utenhav,
            center=(-52074.0241, 6580847.4464, 0.1),
            max_zoom=40,
        )
        sg.explore(
            cleaned,
            gaps,
            double,
            missing,
            df,
            kommune_utenhav,
            center=(5.38389153, 59.00548223, 1),
            max_zoom=40,
        )

        sg.explore(
            cleaned,
            gaps,
            double,
            missing,
            df,
            kommune_utenhav,
            cleaned_points,
            kommune_utenhav_points,
            df_points,
            gaps_buff=sg.buff(gaps, np.log(gaps.area.values + 2) ** 2),
            missing_buff=sg.buff(missing, np.log(missing.area.values + 2) ** 2),
            double_buff=sg.buff(double, np.log(double.area.values + 2) ** 2),
        )

        print(
            f"tolerance {tolerance}",
            "gaps",
            gaps.area.sum(),
            "dup",
            double.area.sum(),
            "missing",
            missing.area.sum(),
        )
        assert (
            gaps.area.sum() <= 1e-6
        ), f"tolerance {tolerance}, gaps: {gaps.area.sum()}"
        assert (
            double.area.sum() <= 1e-6
        ), f"tolerance {tolerance}, double: {double.area.sum()}"
        assert (
            missing.area.sum() <= 1e-6
        ), f"tolerance {tolerance}, missing: {missing.area.sum()}"

        assert thick_df_indices.isin(cleaned_clipped["df_index"]).all(), sg.explore(
            df,
            cleaned,
            missing_polygons=df[
                (df["df_index"].isin(thick_df_indices))
                & (~df["df_index"].isin(cleaned_clipped["df_index"]))
            ],
        )

        intersected = sg.clean_overlay(
            df, cleaned, how="intersection", geom_type="polygon"
        )
        area_same_index = intersected[
            lambda x: x["df_index_1"] == x["df_index_2"]
        ].area.sum()
        area_same_index_ratio = area_same_index / intersected.area.sum()
        assert area_same_index_ratio > 0.998, area_same_index_ratio

        notna_df = df.notna().all()
        cols_notna = list(notna_df[lambda x: x == True].index)
        notna_df_relevant_cols = df[cols_notna].notna().all()
        notna_cleaned = cleaned[cols_notna].notna().all()
        assert notna_cleaned.equals(notna_df_relevant_cols), (
            notna_cleaned,
            notna_df_relevant_cols,
            cleaned[cols_notna].sort_values(by=cols_notna),
        )

        assert list(sorted(cleaned.columns)) == list(sorted(cols)), cleaned.columns


def get_missing(df, other):
    return (
        sg.clean_overlay(df, other, how="difference", geom_type="polygon")
        # .pipe(sg.buff, -0.0001)
        # .pipe(sg.clean_overlay, other, how="difference", geom_type="polygon")
        .pipe(sg.sfilter_inverse, other.buffer(-0.001))
        .pipe(sg.sfilter_inverse, other.buffer(-0.002))
        .pipe(sg.sfilter_inverse, other.buffer(-0.003))
        .pipe(sg.sfilter_inverse, other.buffer(-0.004))
        .pipe(
            sg.buff,
            -0.001,
            resolution=1,
            join_style=2,
        )
        .pipe(
            sg.buff,
            0.001,
            resolution=1,
            join_style=2,
        )
        .pipe(sg.clean_geoms)
    )


def test_clean():

    df = gpd.read_parquet(Path(__file__).parent / "testdata" / "polygon_snap.parquet")

    bbox = sg.to_gdf(
        shapely.minimum_rotated_rectangle(shapely.union_all(df.geometry.values)), df.crs
    )

    kommune_utenhav = gpd.read_parquet(
        Path(__file__).parent / "testdata" / "kommune_utenhav_5435_2023.parquet"
    )
    kommune_utenhav = sg.clean_clip(kommune_utenhav, bbox, geom_type="polygon")

    holes = sg.to_gdf(
        [
            "POINT (905200 7878700)",
            # "POINT (905250 7878780)",
            "POINT (905275 7878800)",
            "POINT (905242.961 7878773.758)",
        ],
        25833,
    ).pipe(sg.buff, 3)

    df = sg.clean_overlay(df, holes, how="difference")
    df["df_index"] = range(len(df))

    mask = sg.close_all_holes(sg.dissexp_by_cluster(df)).dissolve()

    for tolerance in [10, 9, 8, 7, 6, 5]:  # 5, 7, 6, 8, 9]:
        print("tolerance:", tolerance)

        cleaned = sg.coverage_clean(df, tolerance)
        assert sg.get_geom_type(cleaned) == "polygon", sg.get_geom_type(cleaned)

        double = sg.get_intersections(cleaned).loc[lambda x: ~x.buffer(-1e-9).is_empty]
        gaps = sg.get_gaps(cleaned).loc[lambda x: ~x.buffer(-1e-9).is_empty]
        missing = get_missing(df, cleaned)

        print(
            f"tolerance: {tolerance}, double: {double.area.sum()}, "
            f"missing: {missing.area.sum()}, gaps: {gaps.area.sum()}"
        )

        sg.explore(
            df=df.to_crs(25833),
            cleaned=cleaned.to_crs(25833),
            double=double.to_crs(25833),
            missing=missing,
            gaps=gaps.to_crs(25833),
            points=sg.to_gdf(extract_unique_points(cleaned.geometry).explode()),
        )

        assert (
            a := max(list(double.area) + [0])
        ) < 1e-5, f"tolerance: {tolerance}, double: {a}"
        assert (
            a := max(list(missing.area) + [0])
        ) < 1e-5, f"tolerance: {tolerance}, missing: {a}"
        assert (
            a := max(list(gaps.area) + [0])
        ) < 1e-5, f"tolerance: {tolerance}, gaps: {a}"

        notna_cleaned = cleaned[df.columns].notna().all()
        notna_df = df.notna().all()
        assert notna_cleaned.equals(notna_df), (notna_cleaned, notna_df)

        intersected = sg.clean_overlay(
            df, cleaned, how="intersection", geom_type="polygon"
        )
        area_same_index = intersected[
            lambda x: x["df_index_1"] == x["df_index_2"]
        ].area.sum()
        area_same_index_ratio = area_same_index / intersected.area.sum()
        assert area_same_index_ratio > 0.996, area_same_index_ratio

    sg.explore(
        cleaned1=sg.coverage_clean(df, 1),
        cleaned3=sg.coverage_clean(df, 3),
        cleaned5=sg.coverage_clean(df, 5),
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

    cleaned = sg.coverage_clean(df, tolerance)
    gaps = sg.get_gaps(cleaned, True)

    if __name__ == "__main__":
        sg.explore(
            cleaned=cleaned,
            gaps=gaps,
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
    print(list(cleaned.area))
    for area1, area2 in zip(
        sorted(cleaned.area),
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

    print(list(cleaned.length))

    for length1, length2 in zip(
        sorted(cleaned.length),
        sorted(length_should_be),
        strict=False,
    ):
        assert is_close_enough(length1, length2), (length1, length2)


def test_snappping(_test=False):
    from shapely.geometry import Polygon

    if _test:
        loop = np.arange(0.2, 1, 0.01)
    else:
        loop = np.arange(0.25, 0.5, 0.05)

    a = sg.to_gdf(
        Polygon(
            [
                (0, -2),
                (0, 0),
                (-0.02, 1),
                (-0.03, 1.2),
                (-0.04, 1.4),
                (-0.02, 1.6),
                (-0.1, 1.8),
                (0, 2),
                (0, 4),
                (-4, 4),
                (-4, -2),
                (0, -2),
            ]
        )
    )
    thin = sg.to_gdf(
        Polygon(
            [
                (-3, -4),
                (0, -2),
                (0, 0),
                (0.01, 1),
                (0, 4),
                (0.1, 4),
                (0.1, -2),
                (3, -4),
            ]
        )
    )
    thick = sg.to_gdf(
        Polygon(
            [
                (0.1, -2),
                (0.1001, 0),
                (0.1002, 1),
                (0.1000001, 4),
                (4, 4),
                (4, -2),
                (0.1, -2),
            ]
        )
    )

    b = sg.to_gdf(
        Polygon(
            [
                (0, -2),
                (0, 0),
                (0.1, 1),
                (0, 2),
                (0, 4),
                (4, 4),
                (4, -2),
                (0, -2),
            ]
        )
    )
    c = sg.to_gdf(
        Polygon(
            [
                (0, -2),
                # (0, 0),
                # (0, 2),
                (0, 4),
                (4, 4),
                (4, -2),
                (0, -2),
            ]
        )
    )
    for i, df in {
        "dfm1": pd.concat([a, thick, thin]),
        "df0": pd.concat([a, c]),
        "df1": pd.concat([a, b]),
        "df2": pd.concat(
            [
                a,
                c.assign(
                    geometry=lambda x: [
                        Polygon(get_coordinates(g)[::-1]) for g in x.geometry
                    ]
                ),
            ]
        ),
        "df3": pd.concat(
            [
                a,
                b.assign(
                    geometry=lambda x: [
                        Polygon(get_coordinates(g)[::-1]) for g in x.geometry
                    ]
                ),
            ]
        ),
        "df4": pd.concat(
            [
                a.assign(
                    geometry=lambda x: [
                        Polygon(get_coordinates(g)[::-1]) for g in x.geometry
                    ]
                ),
                c,
            ]
        ),
        "df5": pd.concat(
            [
                a.assign(
                    geometry=lambda x: [
                        Polygon(get_coordinates(g)[::-1]) for g in x.geometry
                    ]
                ),
                b,
            ]
        ),
    }.items():
        # if i != "df5":
        #     continue

        print(i)
        df["idx"] = [str(x) for x in range(len(df))]
        p = (
            sg.to_gdf(extract_unique_points(df.geometry.values))
            .explode()
            .assign(wkt=lambda x: [g.wkt for g in x.geometry])
        )

        for tolerance in loop:
            # if i != "df1" or tolerance != 0.25:
            #     continue

            print(tolerance)
            cleaned = sg.coverage_clean(df, tolerance=tolerance)
            # cleaned = sg.coverage_clean(cleaned, tolerance=tolerance)
            gaps = sg.get_gaps(cleaned)
            double = sg.get_intersections(cleaned)
            missing = get_missing(df, cleaned)
            sg.explore(cleaned, gaps, double, missing, p)
            cleaned = pd.concat([df.assign(idx="df"), cleaned])
            # sg.explore(cleaned, column="idx")

            assert (
                gaps.area.sum() == 0
            ), f"tolerance {tolerance} {i}, gaps: {gaps.area.sum()}"
            assert (
                double.area.sum() == 0
            ), f"tolerance {tolerance} {i}, double: {double.area.sum()}"

            if i != "dfm1":
                assert (
                    missing.area.sum() == 0
                ), f"tolerance {tolerance} {i}, missing: {missing.area.sum()}"


def main():
    test_clean()
    test_snappping(_test=False)
    test_clean_dissappearing_polygon()
    not_test_clean_complicated_roads()
    test_clean_1144()
    not_test_spikes()
    test_clean_dissexp()


if __name__ == "__main__":

    # df = cprofile_df("main()")
    # print(df.iloc[:50])
    # print(df.iloc[50:100])

    main()

    import cProfile

    cProfile.run("main()", sort="cumtime")
    _time = time.perf_counter()
    print("seconds passed:", time.perf_counter() - _time)


# %%

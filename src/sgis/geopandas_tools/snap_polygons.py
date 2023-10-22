import functools
import itertools
import warnings
from typing import Callable, Iterable

import geopandas as gpd
import igraph
import networkx as nx
import numba
import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryArray
from IPython.display import display
from numpy import ndarray
from numpy.typing import NDArray
from pandas import Index
from pandas.core.groupby import SeriesGroupBy
from shapely import (
    Geometry,
    box,
    buffer,
    centroid,
    difference,
    distance,
    extract_unique_points,
    get_coordinates,
    get_exterior_ring,
    get_interior_ring,
    get_num_interior_rings,
    get_parts,
    get_rings,
    intersection,
    intersects,
    is_empty,
    is_ring,
    length,
    line_merge,
    linearrings,
    linestrings,
    make_valid,
    points,
    polygons,
    segmentize,
    snap,
    unary_union,
)
from shapely.geometry import (
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
)
from shapely.ops import nearest_points

from ..maps.maps import explore, explore_locals, qtm
from ..networkanalysis.closing_network_holes import close_network_holes, get_angle
from ..networkanalysis.cutting_lines import split_lines_by_nearest_point
from .bounds import get_total_bounds
from .buffer_dissolve_explode import buff, buffdissexp_by_cluster, dissexp_by_cluster
from .conversion import coordinate_array, to_gdf
from .duplicates import get_intersections
from .general import clean_geoms
from .general import sort_large_first as sort_large_first_func
from .general import to_lines
from .geometry_types import get_geom_type, make_all_singlepart, to_single_geom_type
from .neighbors import get_all_distances, k_nearest_neighbors
from .overlay import clean_overlay
from .polygon_operations import close_small_holes, close_thin_holes, get_gaps, get_holes
from .polygons_as_rings import PolygonsAsRings
from .polygons_to_lines import get_rough_centerlines, traveling_salesman_problem
from .sfilter import sfilter, sfilter_inverse, sfilter_split


warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

mask = to_gdf(
    [
        "POINT (905200 7878700)",
        "POINT (905250 7878780)",
    ],
    25833,
).pipe(buff, 30)

mask = to_gdf("POINT (905139.722 7878785.909)", crs=25833).buffer(111)
# mask = to_gdf("POINT (905043 7878849)", crs=25833).buffer(65)
mask = to_gdf("POINT (905097 7878848)", crs=25833).buffer(35)
mask = to_gdf("POINT (905070 7878815)", crs=25833).buffer(35)
# mask = to_gdf("POINT (905098.5 7878848.9)", crs=25833).buffer(3)

# mask = to_gdf("POINT (905276.7 7878549)", crs=25833).buffer(17)
# mask = to_gdf((905271.5, 7878555), crs=25833).buffer(10)

# mask = to_gdf("POINT (905295 7878563)", crs=25833).buffer(50)

# mask = to_gdf([5.95201, 62.41451], 4326).to_crs(25833).buffer(100)

# mask = to_gdf([905100.59664904, 7878744.08293462], 25833).buffer(30)
mask = to_gdf([5.38801, 59.00896], 4326).to_crs(25833).buffer(50)


PRECISION = 1e-4

# TODO # sørg for at polygonene er nokså enkle så centerlinen blir god
# bufre inn og del opp
# kanskje bare del opp uansett?


def coverage_clean(
    gdf: GeoDataFrame,
    tolerance: int | float,
    duplicate_action: str = "fix",
    max_segment_length: int = 5,
) -> GeoDataFrame:
    """

    Rules:
    - Internal holes thinner than the tolerance are closed.
    - Polygons thinner than the tolerance are removed and filled
        by the surrounding polygons.
    - Line and point geometries are removed.
    - MultiPolygons are exploded to Polygons.
    - Index is reset.

    """
    NotImplemented

    _snap_checks(gdf, tolerance)

    crs = gdf.crs

    gdf_orig = gdf.copy()

    gdf = close_thin_holes(gdf, tolerance)

    gdf, slivers = split_out_slivers(gdf, tolerance)

    gaps = get_gaps(gdf)
    double = get_intersections(gdf)
    double["_double_idx"] = range(len(double))

    thin_gaps_and_double = pd.concat([gaps, double]).loc[
        lambda x: x.buffer(-tolerance / 2).is_empty
    ]

    all_are_thin = double["_double_idx"].isin(thin_gaps_and_double["_double_idx"]).all()

    if not all_are_thin and duplicate_action == "fix":
        gdf = _fix_double(gdf, double, thin_gaps_and_double)
    elif not all_are_thin and duplicate_action == "error":
        raise ValueError("Large double surfaces.")

    gaps = (
        pd.concat([thin_gaps_and_double, slivers])
        .pipe(buffdissexp_by_cluster, tolerance, resolution=20)
        .pipe(buffdissexp_by_cluster, -tolerance, resolution=20)
    )

    i = 0
    while len(gaps):
        centerlines: GeoSeries = get_rough_centerlines(
            gaps, max_segment_length
        ).geometry.explode(ignore_index=True)

        """explore(
            centerlines=to_gdf(centerlines, 25833),
            gaps=to_gdf(gaps.geometry, 25833),
            gdf_orig=gdf_orig,
        )"""

        gdf = _snap_last_part(gdf, centerlines, tolerance, max_segment_length, crs)

        gaps = get_gaps(gdf)
        double = get_intersections(gdf)

        gaps = (
            pd.concat([gaps, double])
            .loc[lambda x: x.buffer(-tolerance / 2).is_empty]
            .pipe(buffdissexp_by_cluster, 0.1, resolution=20)
            .pipe(buffdissexp_by_cluster, -0.1, resolution=20)
            .pipe(clean_geoms)
        )
        i += 1

        print("\ngaps", gaps.length.sum(), gaps.area.sum(), "\n\n")

        if i > 1:
            explore(gdf=gdf.clip(gaps.buffer(100)), gaps=gaps)
            sss

        if i == 5:
            raise ValueError("Still thin gaps or double surfaces.")

    return gdf


def snap_polygons(
    gdf: GeoDataFrame,
    snap_to: GeoDataFrame,
    tolerance: float,
    max_segment_length: int | None = 5,  # None,
) -> GeoDataFrame:
    NotImplemented
    _snap_checks(gdf, tolerance)

    crs = gdf.crs

    line_types = ["LineString", "MultiLineString", "LinearRing"]

    if not snap_to.geom_type.isin(line_types).all():
        snap_to = to_lines(snap_to[["geometry"]])
    else:
        snap_to = snap_to[["geometry"]]

    gap_lines: GeoSeries = clean_overlay(
        snap_to,
        gdf.buffer(tolerance).to_frame(),
        how="intersection",
        keep_geom_type=False,
    ).geometry

    return _snap_last_part(gdf, gap_lines, tolerance, max_segment_length, crs)


def _snap_last_part(gdf, snap_to, tolerance, max_segment_length, crs):
    if not len(snap_to):
        return gdf

    intersect_gaps, dont_intersect = sfilter_split(gdf, snap_to.buffer(tolerance))

    if max_segment_length is not None:
        # TODO dropp dette
        has_length = snap_to.length > 0
        snap_to.loc[has_length] = segmentize(
            line_merge(snap_to.loc[has_length]), max_segment_length
        )

        """intersect_gaps.geometry = segmentize(
            line_merge(intersect_gaps.geometry), max_segment_length
        )"""

    intersect_gaps = _snap_polygons_to_lines(
        intersect_gaps, snap_to.unary_union, tolerance
    )

    return _return_snapped_gdf([intersect_gaps, dont_intersect], crs=crs)


def _fix_double(gdf, double, thin_double):
    large = double.loc[~double["_double_idx"].isin(thin_double["_double_idx"])].pipe(
        dissexp_by_cluster
    )
    return clean_overlay(gdf, large, how="update")


def _snap_polygons_to_lines(gdf, lines, tolerance):
    explore(
        lines=to_gdf(lines, 25833),
        gdf=buff(gdf.clip(lines.buffer(10)), -1),
    )
    # explore(gap_lines=(lines), lines=to_gdf(lines, 25833), gdf=gdf)

    if lines.is_empty:
        return lines

    geoms_negbuff = buffer(gdf.geometry.values, -PRECISION * 100)
    # geoms_negbuff = buffer(gdf.geometry.values, -tolerance / 2)

    gdf.geometry = (
        PolygonsAsRings(gdf.geometry)
        .apply_numpy_func(
            _snap_linearring,
            args=(
                lines,
                # unary_union(lines.geometry.values),
                geoms_negbuff,
                tolerance * 1.0001,
            ),
        )
        .to_numpy()
    )

    # exploding to remove lines from geometrycollections. Should give same number of rows
    return make_all_singlepart(gdf, ignore_index=True).loc[lambda x: x.area > 0]


def _snap_checks(gdf, tolerance):
    if not len(gdf) or not tolerance:
        return gdf
    if not gdf.index.is_unique:
        raise ValueError("Index must be unique")
    if get_geom_type(gdf) != "polygon":
        raise ValueError("Must be polygons.")
    if tolerance < PRECISION:
        raise ValueError(
            f"'tolerance' must be larger than {PRECISION} to avoid "
            "problems with floating point precision."
        )


def _return_snapped_gdf(dfs, crs):
    cols_to_keep = lambda x: x.columns.difference(
        {
            "index_right",
            "_gap_idx",
            "_ring_index",
            "_on_outskirts",
        }
    )
    return make_all_singlepart(
        GeoDataFrame(
            pd.concat(
                dfs,
                ignore_index=True,
            ),
            crs=crs,
        )
    ).loc[lambda x: x.area > 0, cols_to_keep]


def split_out_slivers(
    gdf: GeoDataFrame | GeoSeries, tolerance: float | int
) -> tuple[GeoDataFrame, GeoDataFrame] | tuple[GeoSeries, GeoSeries]:
    is_sliver = gdf.buffer(-tolerance / 2).is_empty
    slivers = gdf.loc[is_sliver]
    gdf = gdf.loc[~is_sliver]
    return gdf, slivers


def identify_outskirts(gaps, gdf):
    assert get_geom_type(gaps) == "line"

    # converting to rings and removing the parts intersecting thick polygons
    # the remaining lines can be snapped to
    on_outskirts = clean_overlay(gaps, buff(gdf, PRECISION), how="difference")

    # these thin polygons are in the middle of the area and can be treated as gaps
    gaps["_on_outskirts"] = np.where(
        gaps["_gap_idx"].isin(on_outskirts["_gap_idx"]), 1, 0
    )

    return gaps


def join_lines_with_snap_to(
    lines: GeoDataFrame,
    snap_to: MultiLineString,
    tolerance: int | float,
) -> GeoDataFrame:
    # intersection(lines, snap_to.buffer(tolerance)

    points: NDArray[Point] = get_parts(extract_unique_points(snap_to))
    points_df = GeoDataFrame({"geometry": points}, index=points)
    joined = buff(lines, tolerance).sjoin(points_df, how="left")
    joined.geometry = lines.geometry

    notna = joined["index_right"].notna()

    ring_points = nearest_points(
        joined.loc[notna, "index_right"].values, joined.loc[notna, "geometry"].values
    )[1]

    explore(
        ring_points=to_gdf(ring_points, 25833),
        joined=joined.set_crs(25833),
        points_df=points_df.set_crs(25833),
        snap_to=to_gdf(snap_to, 25833),
    )

    joined.loc[notna, "ring_point"] = ring_points

    return joined


def make_lines_between_points(
    arr1: NDArray[Point], arr2: NDArray[Point]
) -> NDArray[LineString]:
    if arr1.shape != arr2.shape:
        raise ValueError(
            f"Arrays must have equal shape. Got {arr1.shape} and {arr2.shape}"
        )
    coords: pd.DataFrame = pd.concat(
        [
            pd.DataFrame(get_coordinates(arr1), columns=["x", "y"]),
            pd.DataFrame(get_coordinates(arr2), columns=["x", "y"]),
        ]
    ).sort_index()

    return linestrings(coords.values, indices=coords.index)


def snap_to_nearest(
    points: NDArray[Point],
    snap_to: MultiLineString | MultiPoint,
    geoms: GeoSeries,
    tolerance: int | float,
) -> NDArray[Point | None]:
    nearest = nearest_points(points, unary_union(snap_to))[1]
    distance_to_nearest = distance(points, nearest)

    as_lines = make_lines_between_points(points, nearest)
    intersect_geoms: list[NDArray[bool]] = [
        intersects(as_lines, geom) for geom in geoms
    ]
    intersect_geoms: NDArray[bool] = np.any(np.array(intersect_geoms), axis=0)

    return np.where(
        (distance_to_nearest <= tolerance),  # & (~intersect_geoms)),
        nearest,
        None,  # points,  # None,  # ,
    )


def sorted_unary_union(df: pd.DataFrame) -> MultiPoint:
    assert len(df["endpoints"].unique()) <= 1, df["endpoints"].unique()
    assert len(df["geometry"].unique()) <= 1, df["geometry"].unique()

    endpoints: NDArray[float] = get_coordinates(df["endpoints"].iloc[0])
    between: NDArray[float] = get_coordinates(df["ring_point"].dropna().values)

    coords: NDArray[float] = np.concatenate([endpoints, between])
    sorted_coords: NDArray[float] = coords[np.argsort(coords[:, -1])]

    # droping points outside the line (returned from sjoin because of buffer)
    is_between_endpoints: NDArray[bool] = (
        sorted_coords[:, 0] >= np.min(endpoints[:, 0])
    ) & (sorted_coords[:, 0] <= np.max(endpoints[:, 0]))

    sorted_coords: NDArray[float] = sorted_coords[is_between_endpoints]

    return LineString(sorted_coords)


def get_line_segments(lines) -> GeoDataFrame:
    assert lines.index.is_unique
    if isinstance(lines, GeoDataFrame):
        multipoints = lines.assign(
            **{
                lines._geometry_column_name: extract_unique_points(
                    lines.geometry.values
                )
            }
        )
        return multipoints_to_line_segments(multipoints.geometry)

    multipoints = GeoSeries(extract_unique_points(lines.values), index=lines.index)

    return multipoints_to_line_segments(multipoints)


def multipoints_to_line_segments(
    multipoints: GeoSeries, to_next: bool = True
) -> GeoDataFrame:
    if not len(multipoints):
        return GeoDataFrame({"geometry": multipoints})

    try:
        crs = multipoints.crs
    except AttributeError:
        crs = None

    try:
        point_df = multipoints.explode(index_parts=False)
    except AttributeError:
        points, indices = get_parts(multipoints, return_index=True)
        point_df = pd.DataFrame({"geometry": GeometryArray(points)}, index=indices)

    if to_next:
        shift = -1
        filt = lambda x: ~x.index.duplicated(keep="first")
    else:
        shift = 1
        filt = lambda x: ~x.index.duplicated(keep="last")

    point_df["next"] = point_df.groupby(level=0)["geometry"].shift(shift)

    first_points = point_df.loc[filt, "geometry"]
    is_last_point = point_df["next"].isna()

    point_df.loc[is_last_point, "next"] = first_points
    assert point_df["next"].notna().all()

    point_df["geometry"] = [
        LineString([x1, x2]) for x1, x2 in zip(point_df["geometry"], point_df["next"])
    ]
    return GeoDataFrame(point_df.drop(columns=["next"]), geometry="geometry", crs=crs)


def _snap_linearring(
    rings: NDArray[LinearRing],
    snap_to: MultiLineString,
    geoms: GeoSeries,
    tolerance: int | float,
) -> pd.Series:
    assert len(rings.shape) == 1, "ring array should be 1 dimensional"

    multipoints: NDArray[MultiPoint] = extract_unique_points(rings)

    if not len(multipoints):
        return pd.Series()

    line_segments: GeoDataFrame = multipoints_to_line_segments(multipoints)
    line_segments["_ring_index"] = line_segments.index

    snap_points = GeoDataFrame(
        {"geometry": extract_unique_points(get_parts(line_merge(snap_to)))}
    ).explode(ignore_index=True)

    ring_points = GeoDataFrame({"geometry": multipoints}).explode(index_parts=False)
    ring_points["line_to_next"] = multipoints_to_line_segments(
        multipoints, to_next=True
    ).geometry.values
    ring_points["line_to_prev"] = multipoints_to_line_segments(
        multipoints, to_next=False
    ).geometry.values

    ring_points["next"] = ring_points.groupby(level=0)["geometry"].shift(-1)
    ring_points["prev"] = ring_points.groupby(level=0)["geometry"].shift(1)

    ring_points["_ring_point_index"] = range(len(ring_points))
    ring_points["_ring_index"] = ring_points.index
    ring_points = ring_points.reset_index(drop=True)

    snap_df = GeoDataFrame({"geometry": get_parts(line_merge(snap_to))})
    ring_df = GeoDataFrame({"geometry": rings, "_ring_index": range(len(rings))})

    joined = ring_points.sjoin(buff(snap_df, tolerance, resolution=10))
    # joined.index = joined["index_right"]

    intersected = clean_overlay(
        snap_df, buff(ring_df, tolerance, resolution=10)
    ).explode(ignore_index=True)
    intersected["_intersect_index"] = intersected.index

    erased = clean_overlay(
        ring_df, buff(snap_df, tolerance, resolution=10), how="difference"
    )

    erased = (
        erased.groupby("_ring_index", as_index=False)
        .geometry.agg(lambda x: unary_union(line_merge(unary_union(x))))
        .explode(ignore_index=True)
    )

    points = extract_unique_points(erased.geometry).explode(index_parts=False)
    erased["first"] = points.groupby(level=0).nth(0)
    erased["second"] = points.groupby(level=0).nth(1)
    erased["last"] = points.groupby(level=0).nth(-1)
    erased["second_last"] = points.groupby(level=0).nth(-2)

    def extend_lines(arr1, arr2, distance):
        if len(arr1) != len(arr2):
            raise ValueError
        if not len(arr1):
            return GeometryArray([LineString()])

        arr1, arr2 = arr2, arr1  # TODO fix

        coords1 = coordinate_array(arr1)
        coords2 = coordinate_array(arr2)

        dx = coords2[:, 0] - coords1[:, 0]
        dy = coords2[:, 1] - coords1[:, 1]
        len_xy = np.sqrt((dx**2.0) + (dy**2.0))
        x = coords1[:, 0] + (coords1[:, 0] - coords2[:, 0]) / len_xy * distance
        y = coords1[:, 1] + (coords1[:, 1] - coords2[:, 1]) / len_xy * distance

        new_points = shapely.points(x, y)

        print("arr1.shape")
        print(arr1)
        print(arr2)
        print(new_points)
        print(x)
        print(y)
        print(len_xy)
        print(dx)
        print(dy)
        print("ffff")

        return make_lines_between_points(arr2, new_points)

    def extend_lines_to(lines, extend_to):
        assert lines.index.is_unique
        assert (lines.geom_type == "LineString").all()
        geom_col = lines._geometry_column_name
        extension_length = shapely.box(*get_total_bounds(lines, extend_to)).length
        extend_to_union = extend_to.unary_union

        lines = lines.copy()
        points = extract_unique_points(lines.geometry).explode(index_parts=False)

        for i, j, out_col in zip(
            (0, 1), (-1, -2), ("extension_line0", "extension_line1")
        ):
            lines["first"] = points.groupby(level=0).nth(i)
            lines["second"] = points.groupby(level=0).nth(j)

            filt = (~lines["first"].intersects(extend_to_union)) & (
                ~lines["second"].intersects(extend_to_union)
            )
            if i == -1:
                filt &= points.groupby(level=0).nth(i).size() > 2

            lines.loc[filt, "long_extended_line"] = extend_lines(
                lines.loc[filt, "second"],
                lines.loc[filt, "first"],
                extension_length,
            )

            lines.loc[filt, "intersections"] = intersection(
                lines.loc[filt, "long_extended_line"].values, extend_to_union
            )

            filt &= ~shapely.is_empty(lines["intersections"])

            lines.loc[filt, "extend_to_point"] = nearest_points(
                lines.loc[filt, "intersections"],
                lines.loc[filt, geom_col],
            )[0]

            lines.loc[filt, out_col] = make_lines_between_points(
                lines.loc[filt, "first"].values,
                lines.loc[filt, "extend_to_point"].values,
            )

            lines = lines.drop(
                columns=[
                    "first",
                    "second",
                    "extend_to_point",
                    "long_extended_line",
                    "intersections",
                ]
            )

        lines.loc[filt, geom_col] = (
            lines.loc[filt]
            .groupby(level=0)
            .agg(
                lambda x: unary_union(
                    [x["extension_line0"], x["extension_line1"], x.geometry]
                )
            )
        )

        return lines.drop(
            columns=[
                "first",
                "second",
                "extension_line0",
                "extension_line1",
                "extend_to_point",
                "long_extended_line",
                "intersections",
            ]
        )

    display(erased)

    extended = extend_lines_to(erased, intersected)
    extended["_idx"] = range(len(extended))
    self_intersections = get_intersections(extended)
    too_long = erased.loc[erased["_idx"].isin(self_intersections["_idx"])]
    extended = extended.loc[~extended["_idx"].isin(self_intersections["_idx"])]
    extended = pd.concat([extended, extend_lines_to(too_long, self_intersections)])

    explore(
        rings=to_gdf(rings, 25833),
        ring_points=to_gdf(ring_points.geometry, 25833),
        first=to_gdf(erased["first"], 25833),
        second=to_gdf(erased.second, 25833),
        second_last=to_gdf(erased.second_last, 25833),
        last=to_gdf(erased["last"], 25833),
        g=to_gdf(erased.geometry, 25833),
        intersected=to_gdf(intersected.geometry, 25833),
        long_extended_line=to_gdf(erased.long_extended_line, 25833),
        extended_last_line=to_gdf(erased.extended_last_line, 25833),
    )

    erased["long_extended_line"] = extend_lines(
        erased["second"], erased["first"], tolerance
    )
    erased["extended_last_line"] = extend_lines(
        erased["second_last"], erased["last"], tolerance
    )
    display(erased)

    explore(
        rings=to_gdf(rings, 25833),
        ring_points=to_gdf(ring_points.geometry, 25833),
        first=to_gdf(erased["first"], 25833),
        second=to_gdf(erased.second, 25833),
        second_last=to_gdf(erased.second_last, 25833),
        last=to_gdf(erased["last"], 25833),
        g=to_gdf(erased.geometry, 25833),
        intersected=to_gdf(intersected.geometry, 25833),
        long_extended_line=to_gdf(erased.long_extended_line, 25833),
        extended_last_line=to_gdf(erased.extended_last_line, 25833),
    )

    sss

    endpoints = erased.geometry.boundary

    complete_circles = erased.loc[endpoints.geometry.is_empty]
    erased = erased.loc[~endpoints.geometry.is_empty]

    endpoints = endpoints.explode(index_parts=False)

    erased["startpoint"] = endpoints.groupby(level=0).first()
    erased["endpoint"] = endpoints.groupby(level=0).last()

    """for i in erased.index:
        explore(
            startpoint=to_gdf(erased.iloc[[i]].startpoint, 25833),
            endpoint=to_gdf(erased.iloc[[i]].endpoint, 25833),
            line=to_gdf(erased.iloc[[i]].geometry, 25833),
        )"""

    explore(
        ring_points=to_gdf(ring_points.geometry, 25833),
        startpoint=to_gdf(erased.startpoint, 25833),
        endpoint=to_gdf(erased.endpoint, 25833),
        g=to_gdf(erased.geometry, 25833),
        intersected=to_gdf(intersected.geometry, 25833),
    )

    ring_points_by_erased = ring_points.unary_union.intersection(
        erased.geometry.buffer(PRECISION).unary_union
    )
    """
    erased["nearest_to_start"] = erased.groupby(level=0)["startpoint"].agg(
        lambda x: nearest_points(x["startpoint"], all_ring_points.intersection(x.buffer(PRECISION)))[
            1
        ]
    )
    erased["nearest_to_end"] = erased.groupby(level=0)["endpoint"].agg(
        lambda x: nearest_points(x, all_ring_points.intersection(x.buffer(PRECISION)))[
            1
        ]
    )"""

    next_lines = sfilter(line_segments, erased.buffer(PRECISION))

    snapped = (
        pd.concat([next_lines, intersected, erased])
        .groupby("_ring_index")
        .geometry.agg(lambda x: unary_union(line_merge(unary_union(x))))
    )

    missing = sfilter_inverse(ring_points, snapped.buffer(PRECISION))

    explore(
        missing,
        next_lines.reset_index(),
        intersected=to_gdf(intersected.geometry, 25833).reset_index(),
        snapped=to_gdf(snapped, 25833).reset_index(),
        erased=to_gdf(erased.geometry, 25833).reset_index(),
    )
    sss

    erased["nearest_to_start"] = snap(
        erased["startpoint"], ring_points_by_erased, tolerance=PRECISION
    )
    erased["nearest_to_end"] = snap(
        erased["endpoint"], ring_points_by_erased, tolerance=PRECISION
    )

    erased["nearest_to_start"] = np.where(
        erased["nearest_to_start"] == erased["startpoint"],
        pd.NA,
        erased["nearest_to_start"],
    )
    erased["nearest_to_end"] = np.where(
        erased["nearest_to_end"] == erased["endpoint"], pd.NA, erased["nearest_to_end"]
    )

    no_points_at_line = erased["nearest_to_start"].isna()
    display(erased)
    display(no_points_at_line)

    all_ring_points = ring_points.unary_union
    erased.loc[no_points_at_line, "nearest_to_start"] = nearest_points(
        erased.loc[no_points_at_line, "startpoint"], all_ring_points
    )[1]
    erased.loc[no_points_at_line, "nearest_to_end"] = nearest_points(
        erased.loc[no_points_at_line, "endpoint"], all_ring_points
    )[1]

    nextlinemapper = {
        p: line for p, line in zip(ring_points.geometry, ring_points["line_to_next"])
    }

    prevlinemapper = {
        p: line for p, line in zip(ring_points.geometry, ring_points["line_to_prev"])
    }

    erased["next"] = GeoSeries(erased["nearest_to_end"]).map(nextlinemapper)
    erased["prev"] = GeoSeries(erased["nearest_to_start"]).map(prevlinemapper)
    erased["next2"] = GeoSeries(erased["nearest_to_start"]).map(nextlinemapper)
    erased["prev2"] = GeoSeries(erased["nearest_to_end"]).map(prevlinemapper)

    display(erased)

    explore(
        s_e_point=to_gdf(
            pd.concat([erased.startpoint, erased.endpoint]), 25833
        ).reset_index(),
        next_=to_gdf(erased.next, 25833).reset_index(),
        prev=to_gdf(erased.prev, 25833).reset_index(),
        prev2=to_gdf(erased.prev2, 25833).reset_index(),
        next2=to_gdf(erased.next2, 25833).reset_index(),
        nearest_to_end=to_gdf(erased.nearest_to_end, 25833).reset_index(),
        nearest_to_start=to_gdf(erased.nearest_to_start, 25833).reset_index(),
        intersected=to_gdf(intersected.geometry, 25833).reset_index(),
        erased=to_gdf(erased.geometry, 25833).reset_index(),
    )

    sss

    nextmapper = {
        p: next_p
        for p, next_p in zip(
            ring_points.geometry, ring_points.groupby("_ring_index").shift(-1).geometry
        )
    }
    prevmapper = {
        p: prev_p
        for p, prev_p in zip(
            ring_points.geometry, ring_points.groupby("_ring_index").shift(1).geometry
        )
    }

    erased["next"] = GeoSeries(erased["nearest_to_end"].dropna()).map(nextmapper)
    erased["prev"] = GeoSeries(erased["nearest_to_start"].dropna()).map(prevmapper)

    display(erased)

    erased.loc[erased["next"].notna(), "line_to_next"] = make_lines_between_points(
        erased.loc[erased["next"].notna(), "endpoint"],
        erased.loc[erased["next"].notna(), "next"],
    )
    erased.loc[erased["prev"].notna(), "line_to_prev"] = make_lines_between_points(
        erased.loc[erased["prev"].notna(), "startpoint"],
        erased.loc[erased["prev"].notna(), "prev"],
    )
    display(erased)

    assert erased.index.is_unique

    explore(
        line_to_next=to_gdf(erased.line_to_next, 25833),
        line_to_prev=to_gdf(erased.line_to_prev, 25833),
        startpoint=to_gdf(erased.startpoint, 25833),
        endpoint=to_gdf(erased.endpoint, 25833),
        next_=to_gdf(erased.next, 25833),
        prev=to_gdf(erased.prev, 25833),
        nearest_to_end=to_gdf(erased.nearest_to_end, 25833),
        nearest_to_start=to_gdf(erased.nearest_to_start, 25833),
        g=to_gdf(erased.geometry, 25833),
        intersected=to_gdf(intersected.geometry, 25833),
    )

    not_snapped = (
        pd.concat([erased["line_to_prev"], erased["line_to_next"], erased.geometry])
        .dropna()
        .groupby(level=0)
        .agg(lambda x: unary_union(line_merge(unary_union(x))))
        .dropna()
    )

    """not_snapped = sfilter_inverse(
        ring_points.loc[lambda x: ~x.index.isin(joined.index)],
        snap_to.buffer(tolerance),
    )

    not_snapped = clean_overlay(
        ring_df,
        GeoDataFrame({"geometry": [snap_to.buffer(tolerance)]}),
        how="difference",
    )"""

    explore(to_gdf(not_snapped, 25833), intersected)

    snapped = pd.concat([])

    display(snapped)
    sss

    next_ = not_snapped.copy()
    next_.index = next_.index + 1
    next_ = next_.loc[lambda x: ~x.index.duplicated()]
    next_.geometry = ring_points.geometry

    prev = not_snapped.copy()
    prev.index = prev.index - 1
    prev = prev.loc[lambda x: ~x.index.duplicated()]
    prev.geometry = ring_points.geometry

    explore(next_, prev, not_snapped, joined)

    concatted = pd.concat([not_snapped, next_, prev])
    # distances = get_all_distances(concatted.geometry.reset_index(drop=True), concatted.geometry.reset_index(drop=True))

    def points_to_line(df):
        sorted_points = traveling_salesman_problem(df)
        try:
            return LineString(sorted_points)
        except Exception:
            return sorted_points

    as_lines = concatted.groupby(level=0)["geometry"].agg(points_to_line)

    explore(next_, prev, joined, not_snapped, as_lines, concatted)

    sss

    joined["endpoints"] = joined["index_right"].map(ring_points.geometry.boundary)

    intersected_line_mapper = dict(
        zip(intersected["_intersect_index"], intersected.geometry)
    )

    intersected.geometry = intersected.geometry.boundary
    intersected = intersected.explode(ignore_index=True)

    snapped = pd.concat(
        [
            intersected,
            ring_points.loc[
                lambda x: ~x["_ring_point_index"].isin(joined["_ring_point_index"])
            ],
        ]
    )

    snapped = (
        snapped.groupby("_ring_index", as_index=False)["geometry"].apply(
            traveling_salesman_problem
        )
        # .explode()
        # .groupby("_ring_index")
        .agg(LineString)
    )

    explore(
        snapped=snapped.set_crs(25833).assign(idx=lambda x: x._ring_index.astype(str)),
        column="idx",
    )

    explore(
        joined.set_crs(25833),
        ring_points.loc[
            lambda x: ~x["_ring_point_index"].isin(joined["_ring_point_index"])
        ].set_crs(25833),
        snapped=snapped.set_crs(25833).assign(idx=lambda x: x.index.astype(str)),
    )

    # joined = joined.groupby(level=0).apply(sorted_unary_union)

    explore(
        snapped=to_gdf(snapped, 25833).assign(idx=lambda x: x.index.astype(str)),
        column="idx",
    )
    sss

    print("\n\nsnapped")
    print(snapped)

    def sort_points(df):
        return traveling_salesman_problem(df, return_to_start=False)

    snapped = snapped.groupby(level=0)["geometry"].agg(sort_points).explode()
    print(snapped)

    to_int_index = {
        ring_idx: i for i, ring_idx in enumerate(sorted(set(snapped.index)))
    }
    int_indices = snapped.index.map(to_int_index)
    as_lines = pd.Series(
        linearrings(
            get_coordinates(snapped.values),
            indices=int_indices.values,
        ),
        index=snapped.index.unique(),
    )

    print(snapped)
    print(as_lines)

    explore(
        snap_to=to_gdf(snap_to, 25833),
        rings=to_gdf(rings, 25833),
        snapped=to_gdf(extract_unique_points(snapped), 25833),
    )

    for idx in reversed(snapped.index.unique()):
        explore(
            # rings=to_gdf(rings, 25833),
            # snap_to=to_gdf(snap_to, 25833),
            snapped=to_gdf(
                extract_unique_points(snapped.loc[snapped.index == idx]), 25833
            )
            .reset_index(drop=True)
            .reset_index(),
            # column="index",
            # k=20,
            as_lines=to_gdf(as_lines.loc[idx], 25833),
        )

    explore(
        as_lines=to_gdf(as_lines, 25833),
        snap_to=to_gdf(snap_to, 25833),
        snapped=to_gdf(snapped, 25833),
        rings=to_gdf(rings, 25833),
    )

    no_values = pd.Series(
        {i: None for i in range(len(rings)) if i not in as_lines.index}
    )

    return pd.concat([as_lines, no_values]).sort_index()

    snap_points = GeoDataFrame(
        {"geometry": extract_unique_points(get_parts(snap_to))}
    ).explode(ignore_index=True)

    ring_points = GeoDataFrame({"geometry": extract_unique_points(rings)}).explode(
        index_parts=False
    )

    snap_df = GeoDataFrame({"geometry": get_parts(line_merge(snap_to))})
    ring_df = GeoDataFrame({"geometry": rings, "_ring_index": range(len(rings))})

    # ring_points.index = pd.MultiIndex.from_arrays([ring_points.index, range(len(ring_points))])
    ring_points["_range_idx"] = range(len(ring_points))

    intersected = clean_overlay(
        snap_df, buff(ring_df, tolerance, resolution=10)
    ).explode(ignore_index=True)

    erased = clean_overlay(
        ring_df, buff(snap_df, tolerance, resolution=10), how="difference"
    ).explode(ignore_index=True)

    def connect_lines(gdf):
        return close_network_holes(gdf, max_distance=tolerance * 2, max_angle=180)

    snapped = (
        pd.concat([erased, intersected])
        .groupby("_ring_index", as_index=False)
        .apply(connect_lines)
    )
    print(snapped)

    for idx in ring_df._ring_index.unique():
        explore(
            ring_df=ring_df.loc[ring_df._ring_index == idx].set_crs(25833),
            intersected=intersected.loc[intersected._ring_index == idx].set_crs(25833),
            erased=erased.loc[erased._ring_index == idx].set_crs(25833),
            snapped=snapped.loc[snapped._ring_index == idx].set_crs(25833),
            # within_tolerance=within_tolerance.loc[within_tolerance._ring_index==idx].set_crs(25833),
            # not_within=not_within.loc[not_within._ring_index==idx].set_crs(25833),
        )
    sss
    points = extract_unique_points(erased.geometry)
    erased["startpoint"] = erased.groupby(level=0).first()
    erased["endpoint"] = erased.groupby(level=0).last()
    erased = erased.explode(index_parts=False)
    within_tolerance, not_within = sfilter_split(
        ring_points, snap_to.buffer(tolerance, resolution=10)
    )
    explore(
        ring_points=ring_points.set_crs(25833),
        within_tolerance=within_tolerance.set_crs(25833),
        not_within=not_within.set_crs(25833),
        intersected=intersected.set_crs(25833),
        erased=erased.set_crs(25833),
        snap_to=to_gdf(snap_to, 25833),
        rings=to_gdf(rings, 25833),
    )

    def groups_from_consecutive_values(
        series: pd.Series | SeriesGroupBy, column: str
    ) -> NDArray[int]:
        """return (
            (series.apply(lambda x: (x.diff().fillna(1) != 1).cumsum()) + 1)
            .cummax()
            .values
        )"""

        values = series.apply(  # df.groupby(level=0)[column]
            lambda x: (x.diff().fillna(1) != 1).cumsum() + x.index
        )
        print(series.apply(lambda x: (x.diff())))  # df.groupby(level=0)[column]

        print(values)

        out = [0]
        max_value = 0
        # prev = 0
        for value, prev in zip(values[1:], values):
            print(value, prev, max_value, out[-1])
            if value == prev:
                out.append(max_value)
                continue

            # max_value = max(value, max_value)
            if value < max_value:
                max_value += 1
                out.append(max_value)
                continue

                value = max_value + 1
                max_value = value
                out.append(value)
                continue
            # if value > max_value:
            max_value = value
            out.append(value)

        print(out)
        return out

        print(values.value_counts())
        print((values >= values.cummax()).value_counts())

        print(values)
        values = np.where(values >= values.cummax(), values, values + 1)
        print(values)

        sss
        return values <= np.roll(values, 1).cumsum()
        print(values)
        print(values.cumsum())

        return ((values != values.shift()).cumsum() - 1).values

    # column indicating where the chain/line is broken
    not_within["_line_idx"] = groups_from_consecutive_values(
        not_within.groupby(level=0)["_range_idx"], "_range_idx"
    )

    not_within["_line_idx_not_increasing"] = (
        not_within.groupby(level=0)["_range_idx"]
        .apply(lambda x: (x.diff().fillna(1) != 1).cumsum() + x.index)
        .values
    )

    print(not_within)

    explore(not_within=not_within.set_crs(25833), column="_line_idx")

    not_within.groupby("_line_idx")["geometry"].agg(lambda x: print(x) or x)
    not_within = not_within.groupby("_line_idx")["geometry"].agg(LineString)

    explore(
        within_tolerance=within_tolerance.set_crs(25833),
        not_within=to_gdf(not_within, 25833),
        intersected=intersected.set_crs(25833),
    )
    sss
    joined = snap_points.sjoin(buff(ring_points, tolerance, resolution=10))
    joined.index = joined["index_right"]

    snapped = pd.concat(
        [
            joined,
            ring_points.loc[
                lambda x: ~x["_ring_point_index"].isin(joined["_ring_point_index"])
            ],
        ]
    )

    def sort_points(df):
        return traveling_salesman_problem(df, return_to_start=False)

    snapped = snapped.groupby(level=0)["geometry"].agg(sort_points).explode()
    snapped = snapped[~snapped.isin(extract_unique_points(intersected.geometry))]
    print(snapped)

    explore(
        snap_df=snap_df.set_crs(25833),
        ring_df=ring_df.set_crs(25833),
        intersected=intersected.set_crs(25833),
        snapped=to_gdf(snapped, 25833),
    )

    ring_points["_ring_point_index"] = range(len(ring_points))
    ring_df["_ring_index"] = range(len(ring_df))
    intersected["_intersect_index"] = range(len(intersected))

    ring_lines = sfilter_inverse(ring_points, snap_to.buffer(tolerance))

    def sjoin_with_snap_to(gdf):
        return (
            gdf.sjoin_nearest(
                intersected.loc[intersected["_ring_index"] == gdf.index[0]]
            )
            .loc[lambda x: ~x.index.duplicated()]
            .sort_index()
            .pipe(lambda x: print(x) or x)
        )

    ring_lines = (
        ring_lines.groupby(level=0)
        .apply(sjoin_with_snap_to)
        .groupby("_intersect_index", as_index=False)["geometry"]
        .agg(LineString)
    )

    """snapped_lines = (
        snap_points.sjoin(buff(intersected, tolerance, resolution=10))
        .groupby("_intersect_index")["geometry"]
        .agg(LineString)
    )"""

    print(intersected)
    print(ring_lines)

    explore(
        snap_to=to_gdf(snap_to, 25833),
        rings=to_gdf(rings, 25833),
        ring_lines=ring_lines,
        intersected=intersected,
    )

    snapped = (
        pd.concat(
            [
                intersected,
                ring_lines,
            ]
        )
        .groupby("_ring_index")["geometry"]
        .agg(line_merge_by_force)
    )

    print("\n\nsnapped")
    print(snapped)

    to_int_index = {
        ring_idx: i for i, ring_idx in enumerate(sorted(set(snapped.index)))
    }
    int_indices = snapped.index.map(to_int_index)
    as_lines = pd.Series(
        linearrings(
            get_coordinates(snapped.values),
            indices=int_indices.values,
        ),
        index=snapped.index.unique(),
    )

    print(snapped)
    print(as_lines)

    no_values = pd.Series(
        {i: None for i in range(len(rings)) if i not in as_lines.index}
    )

    # return pd.concat([as_lines, no_values]).sort_index()

    snap_df = GeoDataFrame({"geometry": get_parts(line_merge(snap_to))})
    ring_df = GeoDataFrame({"geometry": rings, "_ring_index": range(len(rings))})
    intersected = clean_overlay(snap_df, buff(ring_df, tolerance, resolution=10))
    erased = clean_overlay(
        ring_df,
        buff(snap_df, tolerance, resolution=10, cap_style="square"),
        how="difference",
    )
    explore(
        snap_df=snap_df.set_crs(25833),
        ring_df=ring_df.set_crs(25833),
        intersected=intersected.set_crs(25833),
        erased=erased.set_crs(25833),
    )
    snapped = (
        pd.concat(
            [
                intersected,
                erased,
            ]
        )
        .pipe(
            lambda x: explore(
                x.set_crs(25833).assign(_ring_index=x._ring_index.astype(str)),
                "_ring_index",
            )
            or x
        )
        .groupby("_ring_index")["geometry"]
        # .agg(lambda x: unary_union(line_merge(unary_union(x))))
        .agg(line_merge_by_force)
    )

    print(snapped)
    sss

    points = extract_unique_points(intersected.geometry)
    intersected["startpoint"] = points.groupby(level=0).first()
    intersected["endpoint"] = points.groupby(level=0).last()
    intersected = intersected.explode(index_parts=False)

    points = extract_unique_points(erased.geometry)
    erased["startpoint"] = points.groupby(level=0).first()
    erased["endpoint"] = points.groupby(level=0).last()
    erased = erased.explode(index_parts=False)

    def sort_points(df):
        return traveling_salesman_problem(df, return_to_start=False)

    intersected = intersected.groupby(level=0)["geometry"].agg(sort_points).explode()
    print(snapped)

    ss

    multipoints: NDArray[MultiPoint] = extract_unique_points(rings)

    if not len(multipoints):
        return pd.Series()

    line_segments: GeoDataFrame = multipoints_to_line_segments(multipoints)

    # to integer index
    line_segments.index.name = "_ring_index"
    line_segments = line_segments.reset_index()

    snap_df: GeoDataFrame = join_lines_with_snap_to(
        lines=line_segments,
        snap_to=snap_to,
        tolerance=tolerance,
    )

    snap_df["endpoints"] = snap_df.geometry.boundary

    agged = snap_df.groupby(level=0).apply(sorted_unary_union)
    snap_df = snap_df.loc[lambda x: ~x.index.duplicated()]
    snap_df.geometry = agged

    snap_df = snap_df.groupby("_ring_index", as_index=False)["geometry"].agg(
        unary_union
    )
    snap_df.geometry = line_merge(snap_df.geometry)

    is_not_merged = snap_df.geom_type == "MultiLineString"

    snap_df.loc[is_not_merged, "geometry"] = snap_df.loc[
        is_not_merged, "geometry"
    ].apply(line_merge_by_force)

    assert (
        snap_df.geom_type.isin(["LineString", "LinearRing"])
    ).all(), snap_df.geom_type

    snap_df.geometry = extract_unique_points(snap_df.geometry.values)
    snap_df = snap_df.explode(ignore_index=True)

    if 0:
        snap_df.loc[:, "snapped"] = snap_to_nearest(
            snap_df["geometry"].values, extract_unique_points(snap_to), geoms, tolerance
        )

        more_snap_points = nearest_points(snap_df["geometry"].values, snap_to)[1]
        distances = distance(snap_df["geometry"].values, more_snap_points)
        more_snap_points = more_snap_points[distances < tolerance]

        not_snapped = snap_df["snapped"].isna()
        if not_snapped.any():
            snap_df.loc[not_snapped, "snapped"] = snap_to_nearest(
                snap_df.loc[not_snapped, "geometry"],
                unary_union(more_snap_points),
                geoms,
                tolerance,
            )

        snap_df["geometry"] = np.where(
            snap_df["snapped"].notna(), snap_df["snapped"], snap_df["geometry"]
        )
    else:
        snap_df.index = pd.MultiIndex.from_arrays(
            [snap_df["_ring_index"].values, range(len(snap_df))]
        )

        snap_points_df = GeoDataFrame(
            {"geometry": extract_unique_points(get_parts(line_merge(snap_to)))}
        ).explode(index_parts=False)

        snap_points_df.index = snap_points_df.geometry

        joined = snap_df.sjoin(
            buff(snap_points_df, tolerance, resolution=10, copy=True)
        )
        joined["distance"] = distance(
            joined.geometry.values, joined["index_right"].values
        )

        """# remove conections to points that cross the geometries
        joined.geometry = make_lines_between_points(
            joined.geometry.values, joined["index_right"].values
        )
        joined = sfilter_inverse(joined, geoms)
        qtm(joined2=joined.clip(mask))"""

        joined["wkt"] = joined["index_right"].astype(str)

        unique: pd.Series = (
            joined.sort_index()
            .drop_duplicates(["wkt", "_ring_index"])
            .sort_values("distance")
            .loc[lambda x: ~x.index.duplicated(), "index_right"]
        )

        missing = snap_df.loc[lambda x: ~x.index.isin(unique.index)]
        missing["geometry"] = snap_to_nearest(
            missing["geometry"].values, snap_to, geoms, tolerance
        )
        snapped = pd.concat([unique, missing["geometry"].dropna()])

        snapped.index = snapped.index.droplevel(0)
        snap_df.index = snap_df.index.droplevel(0)

        snap_df["geometry"] = np.where(
            snap_df.index.isin(snapped.index),
            snap_df.index.map(snapped),
            snap_df["geometry"],
        )

        explore(
            snap_points_df,
            snap_df=to_gdf(snap_df.geometry, 25833),
            missing=to_gdf(missing["geometry"].dropna(), 25833),
            joined=to_gdf(joined["index_right"], 25833),
            unique=to_gdf(unique, 25833),
            rings=to_gdf(rings, 25833),
            snap_to=to_gdf(snap_to, 25833),
        )

    explore(
        snap_df=to_gdf(snap_df.geometry, 25833),
        snap_to=to_gdf(snap_to, 25833),
        line_segments=to_gdf(line_segments.geometry, 25833),
        geoms=to_gdf(geoms, 25833),
    )

    assert snap_df["geometry"].notna().all(), snap_df[snap_df["geometry"].isna()]

    # remove lines with only two points. They cannot be converted to polygons.
    is_ring = snap_df.groupby("_ring_index").transform("size") > 2

    not_rings = snap_df.loc[~is_ring].loc[lambda x: ~x.index.duplicated()]
    snap_df = snap_df.loc[is_ring]

    if 1:

        def sort_points(df):
            return traveling_salesman_problem(df, return_to_start=False)

        snap_df = (
            snap_df.groupby("_ring_index", as_index=False)["geometry"]
            .agg(sort_points)
            .explode("geometry")
        )
        print(snap_df)

        """for idx in snap_df["_ring_index"].unique():
            explore(snap_df=snap_df.loc[snap_df._ring_index==idx].set_crs(25833).reset_index(), column="index")
        sss"""
    else:
        snap_df["wkt"] = snap_df.geometry.to_wkt()
        snap_df = snap_df.drop_duplicates(["wkt", "_ring_index"])

    to_int_index = {
        ring_idx: i for i, ring_idx in enumerate(sorted(set(snap_df["_ring_index"])))
    }
    int_indices = snap_df["_ring_index"].map(to_int_index)
    for i in snap_df["_ring_index"].unique():
        print(snap_df.loc[snap_df._ring_index == i, "geometry"])
    as_lines = pd.Series(
        linearrings(
            get_coordinates(snap_df["geometry"].values),
            indices=int_indices.values,
        ),
        index=snap_df["_ring_index"].unique(),
    )
    not_rings = pd.Series(
        [None] * len(not_rings),
        index=not_rings["_ring_index"].values,
    )

    print(snap_df)
    print(as_lines)

    for idx in snap_df["_ring_index"].unique():
        explore(
            rings=to_gdf(rings, 25833),
            snap_to=to_gdf(snap_to, 25833),
            snap_df=to_gdf(snap_df.loc[snap_df._ring_index == idx].geometry, 25833),
            as_lines=to_gdf(as_lines.loc[idx], 25833),
        )

    explore(
        as_lines=to_gdf(as_lines, 25833),
        snap_to=to_gdf(snap_to, 25833),
        snap_df=to_gdf(snap_df.geometry, 25833),
        rings=to_gdf(rings, 25833),
        more_snap_points=to_gdf(more_snap_points, 25833),
        snapped=to_gdf(snap_df.loc[:, "snapped"].dropna(), 25833),
    )

    as_lines = pd.concat([as_lines, not_rings]).sort_index()

    no_values = pd.Series(
        {i: None for i in range(len(rings)) if i not in as_lines.index}
    )

    return pd.concat([as_lines, no_values]).sort_index()


def __snap_linearring(
    rings: NDArray[LinearRing],
    snap_to: MultiLineString,
    # all_gaps: Geometry,
    geoms: GeoSeries,
    tolerance: int | float,
) -> pd.Series:
    assert len(rings.shape) == 1, "ring array should be 1 dimensional"

    multipoints: NDArray[MultiPoint] = extract_unique_points(rings)

    if not len(multipoints):
        return pd.Series()

    line_segments: GeoDataFrame = multipoints_to_line_segments(multipoints)

    # to integer index
    line_segments.index.name = "_ring_index"
    line_segments = line_segments.reset_index()

    snap_df: GeoDataFrame = join_lines_with_snap_to(
        lines=line_segments,
        snap_to=snap_to,
        tolerance=tolerance,
    )
    snap_df["endpoints"] = snap_df.geometry.boundary

    agged = snap_df.groupby(level=0).apply(sorted_unary_union)
    snap_df = snap_df.loc[lambda x: ~x.index.duplicated()]
    snap_df.geometry = agged

    snap_df = snap_df.groupby("_ring_index", as_index=False)["geometry"].agg(
        unary_union
    )
    snap_df.geometry = line_merge(snap_df.geometry)

    is_not_merged = snap_df.geom_type == "MultiLineString"

    line_merge_func = functools.partial(line_merge_by_force, max_segment_length=5)
    snap_df.loc[is_not_merged, "geometry"] = snap_df.loc[
        is_not_merged, "geometry"
    ].apply(line_merge_func)

    assert (
        snap_df.geom_type.isin(["LineString", "LinearRing"])
    ).all(), snap_df.geom_type

    snap_df.geometry = extract_unique_points(snap_df.geometry.values)
    snap_df = snap_df.explode(ignore_index=True)

    explore(
        rings=to_gdf(rings, 25833).assign(ring_index=range(len(rings))),
        column="ring_index",
    )

    if 1:
        explore(snap_df.assign(rim1=lambda x: x._ring_index.astype(str)), "rim1")

        snap_points_df = GeoDataFrame(
            {"geometry": extract_unique_points(get_parts(line_merge(snap_to)))}
        ).explode(index_parts=False)

        snap_points_df["wkt"] = snap_points_df.geometry.to_wkt()
        snap_points_df["snap_to_index"] = range(len(snap_points_df))

        snapped = (
            snap_points_df.sjoin(buff(snap_df, tolerance, resolution=20))
            .sort_values(["index_right", "snap_to_index"])
            .drop_duplicates(["_ring_index", "wkt"])
            .rename(columns={"index_right": "point_index"})
        )

        not_snapped = sfilter_inverse(
            snap_df.loc[lambda x: ~x.index.isin(snapped["point_index"])],
            snap_to.buffer(tolerance),
        )
        not_snapped = clean_overlay(
            GeoDataFrame({"geometry": rings, "_ring_index": range(len(rings))}),
            GeoDataFrame({"geometry": [snap_to.buffer(tolerance)]}),
            how="difference",
            keep_geom_type=False,
        )
        not_snapped.geometry = extract_unique_points(not_snapped.geometry)
        not_snapped = not_snapped.explode(index_parts=False)
        explore(not_snapped)

        not_snapped["point_index"] = not_snapped.index
        not_snapped["snap_to_index"] = (
            not_snapped.groupby("_ring_index")
            .apply(
                lambda x: x.reset_index(drop=True)
                .sjoin_nearest(snap_points_df)["snap_to_index"]
                .sort_index()
            )
            .values
        )

        snap_df = pd.concat([not_snapped, snapped]).sort_values(
            ["point_index", "snap_to_index"]
        )

        explore(snap_df.assign(ri0=lambda x: x._ring_index.astype(str)), "ri0")

        sorted_points = snap_df.groupby(["_ring_index"])["geometry"].agg(LineString)

        explore(
            sorted_points.reset_index().assign(ri1=lambda x: x._ring_index.astype(str)),
            "ri1",
        )

        sorted_points2 = snap_df.groupby(["_ring_index"])["geometry"].agg(
            lambda x: LineString(traveling_salesman_problem(x))
        )
        sorted_points2 = GeoDataFrame({"geometry": sorted_points2})
        explore(
            sorted_points2.reset_index().assign(
                ri2=lambda x: x._ring_index.astype(str)
            ),
            "ri2",
        )
        sss

        if 0:  # for ring_idx in snapped["_ring_index"].unique():
            display(
                snapped.loc[snapped._ring_index == ring_idx]
                .reset_index()[["index", "geometry"]]
                .dropna(),
            )
            explore(
                snapped.loc[snapped._ring_index == ring_idx]
                .reset_index()[["index", "geometry"]]
                .dropna(),
                "index",
            )

            explore(
                snapped.loc[snapped._ring_index == ring_idx][
                    ["index_right", "geometry"]
                ].dropna(),
                "index_right",
            )

            explore(
                snapped.loc[snapped._ring_index == ring_idx][
                    ["snap_to_index", "geometry"]
                ].dropna(),
                "snap_to_index",
            )
            # snapped.index = snapped["_ring_index"]  # ["index_right"]

            explore(
                snapped.loc[snapped._ring_index == ring_idx].assign(
                    idx=lambda x: range(len(x))
                ),
                "idx",
            )

        @numba.njit
        def groups_from_consecutive_values(values):
            i = 0
            prev = values[0]
            indices = [i]
            for value in values[1:]:
                if value == prev + 1:
                    indices.append(i)
                else:
                    i += 1
                    indices.append(i)
                prev = value
            return indices

        # column indicating where the chain/line is broken
        snapped["_line_idx"] = (
            snapped.groupby("_ring_index")["snap_to_index"]
            .agg(lambda x: groups_from_consecutive_values(list(x)))
            .explode()
            .values
        )

        display("snapped")
        display(snapped)
        display("snap_df")
        display(snap_df)

        explore(snapped.assign(ri=lambda x: x._ring_index.astype(str)), "ri")
        sss
        #      lambda x: LineString(x) if len(x) > 1 else x
        # )
        explore(
            snapped.reset_index(),
            "_ring_index",
        )

        """not_within["_line_idx_not_increasing"] = (
            not_within.groupby(level=0)["_range_idx"]
            .apply(lambda x: (x.diff().fillna(1) != 1).cumsum() + x.index)
            .values
        )"""

        print(snapped.loc[snapped.index == 0])
        explore(
            i0=snapped.loc[snapped.index == 0].assign(idx=lambda x: range(len(x))),
            column="idx",
        )
        explore(
            i0=snapped.loc[snapped._ring_index == 0].assign(
                idx=lambda x: range(len(x))
            ),
            column="idx",
        )
        explore(
            snapped=snapped.assign(_ring_index=lambda x: x._ring_index.astype(str)),
            column="_ring_index",
        )

        explore(
            snap_df=snap_df,
            snapped=snapped,
            not_snapped=not_snapped,  # .loc[lambda x: ~x.index.isin(snapped.index)],
        )
        """
        sss

        snap_df.loc[:, "snapped"] = snap_to_nearest(
            snap_df["geometry"].values, extract_unique_points(snap_to), geoms, tolerance
        )

        more_snap_points = nearest_points(snap_df["geometry"].values, snap_to)[1]
        distances = distance(snap_df["geometry"].values, more_snap_points)
        more_snap_points = more_snap_points[distances < tolerance]

        not_snapped = snap_df["snapped"].isna()
        if not_snapped.any():
            snap_df.loc[not_snapped, "snapped"] = snap_to_nearest(
                snap_df.loc[not_snapped, "geometry"],
                unary_union(more_snap_points),
                geoms,
                tolerance,
            )

        explore(
            snap_df=to_gdf(snap_df["geometry"].dropna(), 25833),
            snapped=to_gdf(snap_df["snapped"].dropna(), 25833),
        )

        snapped_ = to_gdf(snap_df["snapped"].dropna(), 25833)
        snap_df_ = to_gdf(snap_df["geometry"].dropna(), 25833)

        snap_df["geometry"] = np.where(
            snap_df["snapped"].notna(), snap_df["snapped"], snap_df["geometry"]
        )

        explore(
            snapped_,
            snap_df_,
            snapped_now=to_gdf(
                snap_df["geometry"],
                25833,
            ),
        )"""

        """distance_to_snap_to = distance(snap_df["geometry"].values, snap_to)
        snap_df = snap_df.loc[
            (distance_to_snap_to <= PRECISION) | (distance_to_snap_to >= tolerance)
        ]"""

    else:
        snap_df.index = pd.MultiIndex.from_arrays(
            [snap_df["_ring_index"].values, range(len(snap_df))]
        )

        snap_points_df = GeoDataFrame(
            {"geometry": extract_unique_points(get_parts(line_merge(snap_to)))}
        ).explode(index_parts=False)

        snap_points_df.index = snap_points_df.geometry

        joined = snap_df.sjoin(
            buff(snap_points_df, tolerance, resolution=10, copy=True)
        )
        joined["distance"] = distance(
            joined.geometry.values, joined["index_right"].values
        )

        """# remove conections to points that cross the geometries
        joined.geometry = make_lines_between_points(
            joined.geometry.values, joined["index_right"].values
        )
        joined = sfilter_inverse(joined, geoms)
        qtm(joined2=joined.clip(mask))"""

        joined["wkt"] = joined["index_right"].astype(str)

        unique: pd.Series = (
            joined.sort_index()
            .drop_duplicates(["wkt", "_ring_index"])
            .sort_values("distance")
            .loc[lambda x: ~x.index.duplicated(), "index_right"]
        )

        missing = snap_df.loc[lambda x: ~x.index.isin(unique.index)]
        missing["geometry"] = snap_to_nearest(
            missing["geometry"].values, snap_to, geoms, tolerance
        )
        snapped = pd.concat([unique, missing["geometry"].dropna()])

        snapped.index = snapped.index.droplevel(0)
        snap_df.index = snap_df.index.droplevel(0)

        snap_df["geometry"] = np.where(
            snap_df.index.isin(snapped.index),
            snap_df.index.map(snapped),
            snap_df["geometry"],
        )

        explore(
            snap_points_df,
            snap_df=to_gdf(snap_df.geometry, 25833),
            missing=to_gdf(missing["geometry"].dropna(), 25833),
            joined=to_gdf(joined["index_right"], 25833),
            unique=to_gdf(unique, 25833),
            rings=to_gdf(rings, 25833),
            snap_to=to_gdf(snap_to, 25833),
        )

    explore(
        snap_df=to_gdf(snap_df.geometry, 25833),
        snap_to=to_gdf(snap_to, 25833),
        line_segments=to_gdf(line_segments.geometry, 25833),
        geoms=to_gdf(geoms, 25833),
    )

    assert snap_df["geometry"].notna().all(), snap_df[snap_df["geometry"].isna()]

    # remove lines with only two points. They cannot be converted to polygons.
    is_ring = snap_df.groupby("_ring_index").transform("size") > 2

    not_rings = snap_df.loc[~is_ring].loc[lambda x: ~x.index.duplicated()]
    snap_df = snap_df.loc[is_ring]

    if 0:

        def sort_points(df):
            return traveling_salesman_problem(df, return_to_start=False)

        snap_df = (
            snap_df.groupby("_ring_index", as_index=False)["geometry"]
            .agg(sort_points)
            .explode("geometry")
        )
        print(snap_df)

        """for idx in snap_df["_ring_index"].unique():
            explore(snap_df=snap_df.loc[snap_df._ring_index==idx].set_crs(25833).reset_index(), column="index")
        sss"""
    else:
        snap_df["wkt"] = snap_df.geometry.to_wkt()
        snap_df = snap_df.drop_duplicates(["wkt", "_ring_index"])

    to_int_index = {
        ring_idx: i for i, ring_idx in enumerate(sorted(set(snap_df["_ring_index"])))
    }
    int_indices = snap_df["_ring_index"].map(to_int_index)
    for i in snap_df["_ring_index"].unique():
        print(snap_df.loc[snap_df._ring_index == i, "geometry"])
    as_lines = pd.Series(
        linearrings(
            get_coordinates(snap_df["geometry"].values),
            indices=int_indices.values,
        ),
        index=snap_df["_ring_index"].unique(),
    )
    not_rings = pd.Series(
        [None] * len(not_rings),
        index=not_rings["_ring_index"].values,
    )

    print(snap_df)
    print(as_lines)

    for idx in snap_df["_ring_index"].unique():
        explore(
            rings=to_gdf(rings, 25833),
            snap_to=to_gdf(snap_to, 25833),
            snap_df=to_gdf(snap_df.loc[snap_df._ring_index == idx].geometry, 25833),
            as_lines=to_gdf(as_lines.loc[idx], 25833),
        )

    explore(
        as_lines=to_gdf(as_lines, 25833),
        snap_to=to_gdf(snap_to, 25833),
        snap_df=to_gdf(snap_df.geometry, 25833),
        rings=to_gdf(rings, 25833),
        more_snap_points=to_gdf(more_snap_points, 25833),
        snapped=to_gdf(snap_df.loc[:, "snapped"].dropna(), 25833),
    )
    sssss

    as_lines = pd.concat([as_lines, not_rings]).sort_index()

    no_values = pd.Series(
        {i: None for i in range(len(rings)) if i not in as_lines.index}
    )

    return pd.concat([as_lines, no_values]).sort_index()


def line_merge_by_force(
    line: MultiLineString | LineString, max_segment_length: int | float
) -> LineString:
    """converts a (multi)linestring to a linestring if possible."""

    if isinstance(line, LineString):
        return line

    line = line_merge(unary_union(line))

    if isinstance(line, LineString):
        return line

    if not isinstance(line, MultiLineString):
        raise TypeError(
            f"Line should be of type MultiLineString or LineString. Got {type(line)}"
        )

    length_before = line.length

    lines = GeoDataFrame({"geometry": get_parts(line)})

    rings = lines[lines.is_ring]
    not_rings = lines[~lines.is_ring]

    one_large_ring = (len(rings) == 1) and (
        rings.length.sum() * (1 + PRECISION)
    ) > lines.length.sum()

    if one_large_ring:
        return _split_line_by_line_points(rings, not_rings)

    if rings.length.sum() < PRECISION and len(not_rings) == 1:
        return not_rings
    elif len(rings):
        display(rings)
        display(not_rings)
        display(rings.length)
        display(not_rings.length)
        if rings.length.sum() < lines.length.sum() * 0.02:
            rings = get_rough_centerlines(rings, max_segment_length)
        else:
            for ring in rings.geometry:
                qtm(ring=to_gdf(ring))
            qtm(rings=(rings), not_rings=(not_rings))
            raise ValueError(rings.length)

    not_rings = pd.concat([not_rings, rings[~rings.is_ring]])
    rings = rings[rings.is_ring]

    if rings.length.sum() > PRECISION * 10:
        for i in rings.geometry:
            print(i)
        # rings.geometry = rings_to_straight_lines(rings.geometry)
        for i in rings.geometry:
            print(i)
        print(rings.is_ring)
        qtm(rings, not_rings)
        qtm(rings)
        raise ValueError(rings.length)

    if not (not_rings.length > PRECISION).any():
        print(not_rings)
        qtm(not_rings=(not_rings))
        notannylong

    lines = close_network_holes(
        not_rings[not_rings.length > PRECISION],
        max_distance=PRECISION * 100,
        max_angle=180,
    )
    line = line_merge(unary_union(lines.geometry.values))

    if isinstance(line, LineString):
        assert line.length >= length_before - PRECISION * 100, (
            line.length - length_before
        )
        return line

    lines = GeoDataFrame({"geometry": get_parts(line)})

    largest_idx: int = lines.length.idxmax()
    largest = lines.loc[[largest_idx]]

    not_largest = lines.loc[lines.index != largest_idx]
    if not_largest.length.sum() > PRECISION * 100:
        qtm(largest, not_largest)
        raise ValueError(not_largest.length.sum())

    return _split_line_by_line_points(largest, not_largest)


def _split_line_by_line_points(
    lines: GeoDataFrame, split_by: GeoDataFrame
) -> LineString:
    split_by.geometry = extract_unique_points(split_by.geometry)
    split_by = split_by.explode(ignore_index=True)

    length_before = lines.length.sum()

    splitted = split_lines_by_nearest_point(lines, split_by, max_distance=PRECISION * 2)

    line = line_merge(unary_union(splitted.geometry.values))

    if not isinstance(line, (LineString, LinearRing)):
        raise ValueError("Couldn't merge lines", line)

    assert line.length >= (length_before - PRECISION * 20), line.length - length_before

    return line

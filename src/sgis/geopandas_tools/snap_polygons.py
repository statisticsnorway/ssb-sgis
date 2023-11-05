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
from ..networkanalysis.closing_network_holes import (
    _close_holes_all_lines,
    close_network_holes,
    close_network_holes_to,
    get_angle,
)
from ..networkanalysis.cutting_lines import (
    change_line_endpoint,
    split_lines_by_nearest_point,
)
from ..networkanalysis.nodes import make_edge_wkt_cols, make_node_ids
from ..networkanalysis.traveling_salesman import traveling_salesman_problem
from .bounds import get_total_bounds
from .buffer_dissolve_explode import buff, buffdissexp_by_cluster, dissexp_by_cluster
from .centerlines import (
    get_line_segments,
    get_rough_centerlines,
    multipoints_to_line_segments,
)
from .conversion import coordinate_array, to_gdf
from .duplicates import get_intersections
from .general import clean_geoms
from .general import sort_large_first as sort_large_first_func
from .general import sort_long_first, to_lines
from .geometry_types import get_geom_type, make_all_singlepart, to_single_geom_type
from .neighbors import get_all_distances, k_nearest_neighbors
from .overlay import clean_overlay
from .polygon_operations import close_small_holes, close_thin_holes, get_gaps, get_holes
from .polygons_as_rings import PolygonsAsRings
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
mask = to_gdf("POINT (905180 7878815)", crs=25833).buffer(545)
# mask = to_gdf("POINT (905098.5 7878848.9)", crs=25833).buffer(3)

# mask = to_gdf("POINT (905276.7 7878549)", crs=25833).buffer(17)
# mask = to_gdf((905271.5, 7878555), crs=25833).buffer(10)

# mask = to_gdf("POINT (905295 7878563)", crs=25833).buffer(50)

# mask = to_gdf([5.95201, 62.41451], 4326).to_crs(25833).buffer(100)

# mask = to_gdf([905100.59664904, 7878744.08293462], 25833).buffer(30)
# mask = to_gdf([5.38801, 59.00896], 4326).to_crs(25833).buffer(50)


PRECISION = 1e-4

# TODO # sørg for at polygonene er nokså enkle så centerlinen blir god
# bufre inn og del opp
# kanskje bare del opp uansett?


def print(*a, **k):
    pass


def explore(*a, **k):
    pass


def qtm(*a, **k):
    pass


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

    _snap_checks(gdf, tolerance)

    crs = gdf.crs

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

        if i == 5:
            print("Still thin gaps or double surfaces.")
            return gdf
            raise ValueError("Still thin gaps or double surfaces.")

    return gdf


def snap_polygons(
    gdf: GeoDataFrame,
    snap_to: GeoDataFrame,
    tolerance: float,
    max_segment_length: int | None = 5,  # None,
) -> GeoDataFrame:
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
    qtm(
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

    qtm(
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


def extend_lines_to(
    lines,
    extend_to,
    max_distance: int | float | None = None,
    to_vertice: bool = True,
    precision: float | int = PRECISION,
):
    assert lines.index.is_unique
    assert (lines.geom_type == "LineString").all()

    if not len(extend_to):
        return lines

    geom_col = lines._geometry_column_name
    if not max_distance:
        max_distance = shapely.box(*get_total_bounds(lines, extend_to)).length

    extend_to_union = extend_to.unary_union

    # if to_vertice:
    #     extend_to_union = extract_unique_points(extend_to_union)

    lines = lines.copy()
    points = extract_unique_points(lines.geometry).explode(index_parts=False)
    unique = points.groupby(level=0).unique()

    for i, j in zip((0, -1), (1, -2)):
        lines["first"] = unique.str[i]
        lines["second"] = unique.str[j]

        filt = (~intersects(lines["first"], extend_to_union)) & (
            ~(intersects(lines["second"], extend_to_union))
        )
        if i == -1:
            filt &= points.groupby(level=0).size() > 2

        lines.loc[filt, "long_extended_line"] = extend_lines(
            lines.loc[filt, "second"],
            lines.loc[filt, "first"],
            max_distance,
        )
        lines.loc[filt, "intersections"] = (
            GeoSeries(lines.loc[filt, "long_extended_line"])
            .buffer(precision)
            .intersection(extend_to_union)
        )

        filt &= (~lines["intersections"].is_empty) & (lines["intersections"].notna())

        lines.loc[filt, "extend_to_point"] = nearest_points(
            lines.loc[filt, "intersections"],
            lines.loc[filt, geom_col],
        )[0]

        if to_vertice:
            pass

        lines.loc[filt, lines._geometry_column_name] = change_line_endpoint(
            lines.loc[filt],
            lines.loc[filt].index,
            pointmapper=lines.loc[filt, "extend_to_point"],
            change_what=i,
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

    return lines


def extend_lines(arr1, arr2, distance):
    if len(arr1) != len(arr2):
        raise ValueError
    if not len(arr1):
        return arr1

    arr1, arr2 = arr2, arr1  # TODO fix

    coords1 = coordinate_array(arr1)
    coords2 = coordinate_array(arr2)

    dx = coords2[:, 0] - coords1[:, 0]
    dy = coords2[:, 1] - coords1[:, 1]
    len_xy = np.sqrt((dx**2.0) + (dy**2.0))
    x = coords1[:, 0] + (coords1[:, 0] - coords2[:, 0]) / len_xy * distance
    y = coords1[:, 1] + (coords1[:, 1] - coords2[:, 1]) / len_xy * distance

    new_points = np.array([None for _ in range(len(arr1))])
    new_points[~np.isnan(x)] = shapely.points(x[~np.isnan(x)], y[~np.isnan(x)])

    new_points[~np.isnan(x)] = make_lines_between_points(
        arr2[~np.isnan(x)], new_points[~np.isnan(x)]
    )
    return new_points


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

    line_segments: GeoDataFrame = multipoints_to_line_segments(multipoints).to_frame(
        "geometry"
    )
    line_segments["_ring_index"] = line_segments.index

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

    snap_df = close_network_holes(snap_df, max_distance=tolerance * 2, max_angle=90)

    intersected = clean_overlay(
        snap_df, buff(ring_df, tolerance, resolution=10)
    ).explode(ignore_index=True)
    intersected["_intersect_index"] = intersected.index

    erased = clean_overlay(
        ring_df, buff(snap_df, tolerance, resolution=10), how="difference"
    )

    erased = (
        erased.groupby("_ring_index")
        .geometry.agg(lambda x: unary_union(line_merge(unary_union(x))))
        .explode(index_parts=False)  # ignore_index=True)
    )
    erased_endpoints = erased.boundary.explode(index_parts=False)

    intersected_endoints: MultiPoint = GeoSeries(
        unary_union(line_merge(unary_union(intersected.geometry)))
    ).boundary.unary_union
    nearest = nearest_points(erased_endpoints.values, intersected_endoints)[
        1
    ]  # TODO or snap_to

    lines_between = GeoSeries(
        make_lines_between_points(erased_endpoints.values, nearest),
        index=erased_endpoints.index,
    )

    erased = pd.concat([erased, lines_between])

    def close_holes_and_merge(df) -> LineString:
        merged = make_valid(line_merge(unary_union(df.geometry.dropna())))
        if isinstance(merged, LineString):
            return merged
        merged = GeoSeries([merged]).explode(ignore_index=True)

        line_segments = sfilter(
            get_line_segments(merged), merged.buffer(PRECISION), predicate="within"
        ).to_frame("geometry")
        line_segments.index = range(len(line_segments))

        endpoints = line_segments.boundary.explode(index_parts=False).to_wkt()
        line_segments["source"] = endpoints.groupby(level=0).nth(0)
        line_segments["target"] = endpoints.groupby(level=0).nth(-1)

        no_dups = pd.DataFrame(
            np.sort(line_segments[["source", "target"]].values, axis=1),
            columns=["source", "target"],
        ).drop_duplicates()

        no_dups["geometry"] = no_dups.index.map(line_segments.geometry)

        all_endpoints = pd.concat([no_dups["source"], no_dups["target"]])

        appear_only_once = all_endpoints.value_counts().loc[lambda x: x == 1].index

        merged_again = GeoSeries(
            get_parts(make_valid(line_merge(unary_union(no_dups.geometry))))
        ).to_frame("geometry")
        endpoints = merged_again.boundary.explode(index_parts=False).to_wkt()
        merged_again["source"] = endpoints.groupby(level=0).nth(0)
        merged_again["target"] = endpoints.groupby(level=0).nth(-1)

        relevant_lines = merged_again.loc[
            lambda x: (~x["source"].isin(appear_only_once))
            & (~x["target"].isin(appear_only_once))
        ]

        return make_valid(line_merge(unary_union(relevant_lines.geometry)))

    all_together = GeoSeries(
        pd.concat([intersected.set_index("_ring_index").geometry, erased.geometry])
        .groupby(level=0)
        .agg(close_holes_and_merge)
    )  # .explode(index_parts=False)

    all_together = all_together.reset_index()
    all_together.index.name = "_ring_index"

    snapped = all_together.geometry.explode(index_parts=False).loc[lambda x: x.is_ring]
    coords, indices = get_coordinates(snapped, return_index=True)
    as_lines = pd.Series(linearrings(coords, indices=indices), index=snapped.index)
    if not as_lines.index.is_unique:
        as_lines = as_lines.groupby(level=0).agg(unary_union)

    print(all_together)
    print(all_together.is_ring)
    for i in all_together._ring_index.unique():
        explore(all_together[all_together._ring_index == i])

    explore(
        erased=to_gdf(erased, 25833),
        intersected=to_gdf(intersected, 25833),
        all_together=to_gdf(all_together, 25833),
        as_lines=to_gdf(as_lines, 25833),
        snap_to=to_gdf(snap_to, 25833),
        snapped=to_gdf(snapped, 25833),
        rings=to_gdf(rings, 25833),
    )

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

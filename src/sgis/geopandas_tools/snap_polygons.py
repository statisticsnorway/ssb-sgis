import functools
import itertools
import warnings
from typing import Callable, Iterable

import geopandas as gpd
import igraph
import networkx as nx
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryArray
from IPython.display import display
from numpy import ndarray
from numpy.typing import NDArray
from pandas import Index
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
from .buffer_dissolve_explode import buff, dissexp_by_cluster
from .conversion import coordinate_array, to_gdf
from .duplicates import get_intersections
from .general import clean_geoms
from .general import sort_large_first as sort_large_first_func
from .general import to_lines
from .geometry_types import get_geom_type, make_all_singlepart, to_single_geom_type
from .neighbors import get_all_distances, k_nearest_neighbors
from .overlay import clean_overlay
from .polygon_operations import close_small_holes, close_thin_holes, get_gaps, get_holes
from .polygons_to_lines import PolygonsAsRings, get_cheap_centerlines
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
# mask = to_gdf("POINT (905098.5 7878848.9)", crs=25833).buffer(3)

# mask = to_gdf("POINT (905276.7 7878549)", crs=25833).buffer(17)
# mask = to_gdf((905271.5, 7878555), crs=25833).buffer(10)

# mask = to_gdf("POINT (905295 7878563)", crs=25833).buffer(50)

# mask = to_gdf([5.95201, 62.41451], 4326).to_crs(25833).buffer(100)

# mask = to_gdf([905100.59664904, 7878744.08293462], 25833).buffer(30)


PRECISION = 1e-4

# TODO # sørg for at polygonene er nokså enkle så centerlinen blir god
# bufre inn og del opp
# kanskje bare del opp uansett?


def coverage_clean(
    gdf: GeoDataFrame,
    tolerance: int | float,
    duplicate_action: str = "error",
    max_segment_length: int | None = None,
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
    snap_checks(gdf, tolerance)

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

    if not all_are_thin and duplicate_action == "error":
        large = double.loc[
            ~double["_double_idx"].isin(thin_gaps_and_double["_double_idx"])
        ]
        large["area"] = large.area
        raise ValueError("Large double surfaces.", large[["area", "geometry"]])
    elif not all_are_thin and duplicate_action == "fix":
        large = double.loc[
            ~double["_double_idx"].isin(thin_gaps_and_double["_double_idx"])
        ].pipe(dissexp_by_cluster)
        gdf = clean_overlay(gdf, large, how="update")

    gaps = pd.concat([thin_gaps_and_double, slivers]).pipe(dissexp_by_cluster)

    centerlines = get_cheap_centerlines(gaps, max_segment_length=max_segment_length)

    if not len(centerlines):
        return gdf

    intersect_gaps, dont_intersect = sfilter_split(gdf, centerlines.buffer(tolerance))

    centerlines.geometry = segmentize(centerlines.geometry, 10)

    intersect_gaps = _snap_polygons_to_lines(intersect_gaps, centerlines, tolerance)

    return _return_snapped_gdf([intersect_gaps, dont_intersect], crs=crs)


def snap_polygons(
    gdf: GeoDataFrame,
    snap_to: GeoDataFrame,
    tolerance: float,
    sort_large_first: bool = True,
    max_segment_length: int | None = None,
) -> GeoDataFrame:
    snap_checks(gdf, tolerance)

    crs = gdf.crs

    line_types = ["LineString", "MultiLineString", "LinearRing"]

    if not snap_to.geom_type.isin(line_types).all():
        snap_to = to_lines(snap_to[["geometry"]])
    else:
        snap_to = snap_to[["geometry"]]

    gap_lines = clean_overlay(
        snap_to,
        gdf.buffer(tolerance).to_frame(),
        how="intersection",
        keep_geom_type=False,
    ).loc[lambda x: x.geom_type.isin(line_types)]

    if not len(gap_lines):
        return gdf

    intersect_gaps, dont_intersect = sfilter_split(gdf, gap_lines.buffer(tolerance))

    # intersect_gaps.geometry = segmentize(intersect_gaps.geometry, 1)
    gap_lines.geometry = segmentize(gap_lines.geometry, 10)

    intersect_gaps = _snap_polygons_to_lines(intersect_gaps, gap_lines, tolerance)

    return _return_snapped_gdf([intersect_gaps, dont_intersect], crs=crs)


def _snap_polygons_to_lines(gdf, lines, tolerance):
    snap_to: MultiLineString = lines.unary_union

    qtm(
        gap_lines=(lines),
        snap_to=to_gdf(snap_to, 25833),
        gdf=buff(gdf.clip(lines.buffer(10)), -1),
    )
    # explore(gap_lines=(lines), snap_to=to_gdf(snap_to, 25833), gdf=gdf)

    if snap_to.is_empty:
        return lines

    geoms_negbuff = buffer(gdf.geometry.values, -PRECISION * 100)

    gdf.geometry = (
        PolygonsAsRings(gdf.geometry)
        .apply_numpy_func(
            _snap_linearring,
            args=(
                snap_to,
                unary_union(lines.geometry.values),
                geoms_negbuff,
                tolerance * 1.0001,
            ),
        )
        .to_numpy()
    )

    # exploding to remove lines from geometrycollections. Should give same number of rows
    return make_all_singlepart(gdf, ignore_index=True).loc[lambda x: x.area > 0]


def get_angle_between_indexed_points(point_df: GeoDataFrame):
    """ "Get angle difference between the two lines"""

    point_df["next"] = point_df.groupby(level=0)["geometry"].shift(-1)

    notna = point_df["next"].notna()

    this = coordinate_array(point_df.loc[notna, "geometry"].values)
    next_ = coordinate_array(point_df.loc[notna, "next"].values)

    point_df.loc[notna, "angle"] = get_angle(this, next_)
    point_df["prev_angle"] = point_df.groupby(level=0)["angle"].shift(1)

    point_df["angle_diff"] = np.abs(
        np.abs(point_df["angle"]) - np.abs(point_df["prev_angle"])
    )

    return point_df


def _get_angle_between_indexed_points(point_df: GeoSeries):
    """ "Get angle difference between the two lines"""

    next_ = point_df.groupby(level=0).shift(-1)

    notna = next_.notna()

    this = coordinate_array(point_df.loc[notna].values)
    next_ = coordinate_array(next_.loc[notna].values)

    angle = get_angle(this, next_)
    point_df["prev_angle"] = point_df.groupby(level=0)["angle"].shift(1)

    point_df["angle_diff"] = np.abs(
        np.abs(point_df["angle"]) - np.abs(point_df["prev_angle"])
    )

    return point_df


def get_two_closest_bounds_points(geoms: GeoSeries) -> GeoSeries:
    envelopes = geoms.envelope
    envelopes.loc[:] = get_rings(envelopes)
    corner_points = extract_unique_points(envelopes).explode(index_parts=False)

    envelopes = envelopes.loc[corner_points.index]

    nearest_ring_point = nearest_points(corner_points, geoms.loc[corner_points.index])[
        1
    ]
    distance_to_corner = distance(nearest_ring_point, corner_points)

    is_two_closest: NDArray[np.bool] = (
        distance_to_corner.groupby(level=0)
        .apply(lambda x: x <= x.nsmallest(2).iloc[-1])
        .values
    )

    return nearest_ring_point.iloc[is_two_closest]


def rings_to_straight_lines(rings: GeoSeries) -> GeoSeries:
    assert (
        rings.geom_type.isin(["LineString", "LinearRing"])
    ).all(), rings.geom_type.value_counts()

    two_closest = get_two_closest_bounds_points(rings)

    rings.loc[:] = two_closest.groupby(level=0).agg(LineString)

    return rings


def split_rings_at_boundary(thin):
    to_gdf(thin, 25833).to_parquet(
        r"C:\Users\ort\git\ssb-sgis\tests\testdata\thin.parquet"
    )
    assert (
        thin.geom_type.isin(["LineString", "LinearRing"])
    ).all(), thin.geom_type.value_counts()

    envelopes = thin.envelope
    envelopes.loc[:] = get_rings(envelopes)
    corner_points = extract_unique_points(envelopes).explode(index_parts=False)

    envelopes = envelopes.loc[corner_points.index]

    nearest_ring_point = nearest_points(corner_points, thin.loc[corner_points.index])[1]
    distance_to_corner = distance(nearest_ring_point, corner_points)

    is_two_closest: NDArray[np.bool] = (
        distance_to_corner.groupby(level=0)
        .apply(lambda x: x <= x.nsmallest(2).iloc[-1])
        .values
    )

    two_nearest_ring_points = (
        nearest_ring_point.iloc[is_two_closest].groupby(level=0).agg(unary_union)
    )

    """qtm(
        thin=to_gdf(thin, 25833).clip(mask),
        nearest_ring_point=to_gdf(nearest_ring_point, 25833).clip(mask),
        envelopes=to_gdf(envelopes, 25833).clip(mask),
    )

    qtm(
        thin=to_gdf(thin, 25833).clip(mask),
        nearest_ring_point=to_gdf(nearest_ring_point, 25833).clip(mask),
    )
    qtm(
        thin=to_gdf(thin, 25833).clip(mask),
        two_nearest_ring_points=to_gdf(two_nearest_ring_points, 25833).clip(mask),
    )
    qtm(
        thin=to_gdf(thin, 25833).clip(mask),
        two_nearest_ring_points=to_gdf(two_nearest_ring_points, 25833).clip(mask),
        thin_diff=thin.difference(two_nearest_ring_points.buffer(0.1))
        .clip(mask)
        .to_frame(),
    )"""

    return thin.difference(two_nearest_ring_points.buffer(PRECISION))


def to_none_if_thin(rings: GeoSeries, inside_holes, tolerance):
    # buffered_in = rings.difference(inside_holes).buffer(-tolerance / 2)
    # return np.where(buffered_in.is_empty, None, rings)
    buffered_in = buffer(
        difference(polygons(rings), inside_holes),
        -(tolerance / 2),
    )
    return np.where(is_empty(buffered_in), None, rings)

    cond = (~np.isnan(rings)) & (~is_empty(rings))
    rings[cond] = buffer(
        difference(polygons(rings[cond]), inside_holes),
        -(tolerance / 2),
    )
    return np.where(is_empty(rings), None, rings)


def snap_checks(gdf, tolerance):
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


def sjoin_gap_lines(gap_lines, gdf):
    # gap_lines_joined = pd.DataFrame(
    gap_lines_joined = (
        buff(gap_lines, PRECISION).sjoin(gdf[["geometry"]], how="left")
    ).sort_values("index_right")

    gap_lines_joined.geometry = gap_lines.geometry

    intersect_gaps = gdf.loc[lambda x: x.index.isin(gap_lines_joined["index_right"])]
    dont_intersect = gdf.loc[lambda x: ~x.index.isin(gap_lines_joined["index_right"])]

    gap_lines_joined = gap_lines_joined.drop_duplicates(
        "_gap_idx"
    )  # .loc[lambda x: ~x.index.duplicated()]

    return gap_lines_joined, intersect_gaps, dont_intersect


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
    # TODO: dissexp on the gaps?

    assert get_geom_type(gaps) == "line"

    # converting to rings and removing the parts intersecting thick polygons
    # the remaining lines can be snapped to
    on_outskirts = clean_overlay(gaps, buff(gdf, PRECISION), how="difference")

    # these thin polygons are in the middle of the area and can be treated as gaps
    gaps["_on_outskirts"] = np.where(
        gaps["_gap_idx"].isin(on_outskirts["_gap_idx"]), 1, 0
    )

    return gaps


def segmentize_triangles(geoms: GeoSeries):
    if not len(geoms):
        return geoms

    def is_triangle(x):
        n_points = extract_unique_points(x).apply(lambda p: len(get_parts(p)))
        return n_points == 3

    triangles = geoms.loc[is_triangle]

    if not len(triangles):
        return geoms

    def get_max_segment_length(geoms):
        return np.max(distance(geoms[:-1], geoms[:1]))

    max_segment_length = (
        extract_unique_points(triangles)
        .explode(index_parts=False)
        .groupby(level=0)
        .agg(get_max_segment_length)
    )
    triangles = GeoSeries(
        segmentize(
            triangles.to_numpy(),
            max_segment_length=max_segment_length.to_numpy() / 2,
        ),
        index=triangles.index,
        crs=geoms.crs,
    )

    return pd.concat([triangles, geoms[lambda x: ~is_triangle(x)]])


def join_lines_with_snap_to(
    lines: GeoDataFrame,
    snap_to: MultiLineString,
    tolerance: int | float,
) -> GeoDataFrame:
    points: NDArray[Point] = get_parts(extract_unique_points(snap_to))
    points_df = GeoDataFrame({"geometry": points}, index=points)
    joined = buff(lines, tolerance).sjoin(points_df, how="left")
    joined.geometry = lines.geometry

    notna = joined["index_right"].notna()

    ring_points = nearest_points(
        joined.loc[notna, "index_right"].values, joined.loc[notna, "geometry"].values
    )[1]

    joined.loc[notna, "ring_point"] = ring_points

    return joined


def make_lines_between_points(
    arr1: NDArray[Point], arr2: NDArray[Point]
) -> NDArray[LineString]:
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have equal shape.")
    coords: pd.DataFrame = pd.concat(
        [
            pd.DataFrame(get_coordinates(arr1), columns=["x", "y"]),
            pd.DataFrame(get_coordinates(arr2), columns=["x", "y"]),
        ]
    ).sort_index()

    try:
        return linestrings(coords.values, indices=coords.index)
    except Exception as e:
        print(arr1)
        print(arr2)
        print(coords[coords.notna()])
        print(coords[coords.isna()])
        raise e.__class__(e, coords)


def map_to_nearest(
    points: NDArray[Point],
    snap_to: MultiLineString,
    geoms: GeoSeries,
    tolerance: int | float,
) -> NDArray[Point]:
    nearest_vertice = nearest_points(points, extract_unique_points(snap_to))[1]
    distance_to_nearest_vertice = distance(points, nearest_vertice)

    as_lines = make_lines_between_points(points, nearest_vertice)
    intersect_geoms: list[NDArray[np.bool]] = [
        intersects(as_lines, geom) for geom in geoms
    ]
    intersect_geoms: NDArray[np.bool] = np.any(np.array(intersect_geoms), axis=0)

    nearest = nearest_points(points, snap_to)[1]
    distance_to_nearest = distance(points, nearest)

    as_lines = make_lines_between_points(points, nearest)
    intersect_geoms2: list[NDArray[np.bool]] = [
        intersects(as_lines, geom) for geom in geoms
    ]
    intersect_geoms2: NDArray[np.bool] = np.any(np.array(intersect_geoms2), axis=0)

    snap_to_vertice: NDArray[np.bool] = (
        (distance_to_nearest_vertice <= tolerance)  # * 2.0001)
        & (distance_to_nearest_vertice > 0)
        & (~intersect_geoms)
    )

    snap_to_point: NDArray[np.bool] = (
        (distance_to_nearest_vertice > tolerance)  # * 2.0001)
        & (distance_to_nearest <= tolerance)  # * 2.0001)
        & (distance_to_nearest > 0)
        & (~intersect_geoms2)
    )

    conditions: list[NDArray[np.bool]] = [
        snap_to_vertice,
        snap_to_point,
    ]

    # explore_locals(explore=False, mask=mask)

    choices: list[NDArray[Point]] = [nearest_vertice, nearest]

    return np.select(conditions, choices, default=points)


def snap_to_nearest(
    points: NDArray[Point],
    snap_to: NDArray[Point],
    geoms: GeoSeries,
    tolerance: int | float,
) -> NDArray[Point]:
    nearest = nearest_points(points, unary_union(snap_to))[1]
    distance_to_nearest = distance(points, nearest)

    as_lines = make_lines_between_points(points, nearest)
    intersect_geoms: list[NDArray[np.bool]] = [
        intersects(as_lines, geom) for geom in geoms
    ]
    intersect_geoms: NDArray[np.bool] = np.any(np.array(intersect_geoms), axis=0)

    return np.where(
        (
            (distance_to_nearest <= tolerance)
            # & (distance_to_nearest > 0)
            & (~intersect_geoms)
        ),
        nearest,
        points,  # None,  # ,
    )


def sorted_unary_union(df: pd.DataFrame) -> MultiPoint:
    assert len(df["endpoints"].unique()) <= 1, df["endpoints"].unique()
    assert len(df["geometry"].unique()) <= 1, df["geometry"].unique()

    endpoints: NDArray[np.float] = get_coordinates(df["endpoints"].iloc[0])
    between: NDArray[np.float] = get_coordinates(df["ring_point"].dropna().values)

    coords: NDArray[np.float] = np.concatenate([endpoints, between])
    sorted_coords: NDArray[np.float] = coords[np.argsort(coords[:, -1])]

    # droping points outside the line (returned from sjoin because of buffer)
    is_between_endpoints: NDArray[np.bool] = (
        sorted_coords[:, 0] >= np.min(endpoints[:, 0])
    ) & (sorted_coords[:, 0] <= np.max(endpoints[:, 0]))

    sorted_coords: NDArray[np.float] = sorted_coords[is_between_endpoints]

    return LineString(sorted_coords)


def get_line_segments(lines) -> GeoDataFrame:
    if isinstance(lines, GeoDataFrame):
        multipoints = lines.assign(
            **{
                lines._geometry_column_name: extract_unique_points(
                    lines.geometry.values
                )
            }
        )
        return multipoints_to_line_segments(multipoints)

    multipoints = extract_unique_points(lines)

    return multipoints_to_line_segments(multipoints)


def multipoints_to_line_segments(multipoints: GeoSeries) -> GeoDataFrame:
    if not len(multipoints):
        return multipoints

    try:
        crs = multipoints.crs
    except AttributeError:
        crs = None

    try:
        point_df = multipoints.explode(index_parts=False)
    except AttributeError:
        points, indices = get_parts(multipoints, return_index=True)
        point_df = pd.DataFrame({"geometry": GeometryArray(points)}, index=indices)

    point_df["next"] = point_df.groupby(level=0)["geometry"].shift(-1)

    first_points = point_df.loc[lambda x: ~x.index.duplicated(), "geometry"]
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
    all_gaps: Geometry,
    geoms: GeoSeries,
    tolerance: int | float,
) -> pd.Series:
    assert len(rings.shape) == 1, "ring array should be 1 dimensional"

    multipoints: NDArray[MultiPoint] = extract_unique_points(rings)

    line_segments: GeoDataFrame = multipoints_to_line_segments(multipoints)

    line_segments.index.name = "_ring_index"
    line_segments = line_segments.reset_index()

    snap_df: GeoDataFrame = join_lines_with_snap_to(
        lines=line_segments,
        snap_to=all_gaps,
        tolerance=tolerance,
    )

    display("snap_df joined with points")
    display(snap_df)

    # TODO as coord tuples?
    snap_df["endpoints"] = snap_df.geometry.boundary

    print("len snap_df")
    print(len(snap_df))

    agged = snap_df.groupby(level=0).apply(sorted_unary_union)
    snap_df = snap_df.loc[lambda x: ~x.index.duplicated()]
    snap_df.geometry = agged

    print("len snap_df 2")
    print(len(snap_df))

    qtm(
        snap_df=to_gdf(snap_df.geometry, 25833).clip(mask.buffer(0.5)),
        geoms=to_gdf(geoms, 25833).clip(mask.buffer(0.5)),
        snap_to=to_gdf(snap_to, 25833).clip(mask.buffer(0.5)),
        title="snap_df",
    )

    to_gdf(extract_unique_points(snap_to)).to_parquet(
        r"C:\Users\ort\git\ssb-sgis\tests\testdata\snap_to_points.parquet"
    )
    to_gdf(snap_df.geometry).to_parquet(
        r"C:\Users\ort\git\ssb-sgis\tests\testdata\snap_df.parquet"
    )
    to_gdf(geoms).to_parquet(r"C:\Users\ort\git\ssb-sgis\tests\testdata\geoms.parquet")
    to_gdf(snap_df.loc[snap_df.ring_point.notna(), "ring_point"]).to_parquet(
        r"C:\Users\ort\git\ssb-sgis\tests\testdata\ring_points.parquet"
    )

    # snap_df = sfilter_inverse(snap_df, geoms)

    print("len snap_df 3")
    print(len(snap_df))

    qtm(
        snap_df=to_gdf(snap_df.geometry, 25833).clip(mask.buffer(0.5)),
        geoms=to_gdf(geoms, 25833).clip(mask.buffer(0.5)),
        snap_to=to_gdf(snap_to, 25833).clip(mask.buffer(0.5)),
        title="snap_df2 etter sfilter_inverse",
    )

    # snap_df = snap_df.dissolve(by="_ring_index", as_index=False)
    snap_df = snap_df.groupby("_ring_index", as_index=False)["geometry"].agg(
        unary_union
    )
    snap_df.geometry = line_merge(snap_df.geometry)

    display("disssss")
    display(snap_df.geometry)

    is_not_merged = snap_df.geom_type == "MultiLineString"

    qtm(circ=snap_df.clip(to_gdf((905270.000, 7878560.000), crs=25833).buffer(30)))

    print("is not merged")
    print(snap_df.loc[is_not_merged, "geometry"])
    snap_df.loc[is_not_merged, "geometry"] = snap_df.loc[
        is_not_merged, "geometry"
    ].apply(line_merge_by_force)

    assert (
        snap_df.geom_type.isin(["LineString", "LinearRing"])
    ).all(), snap_df.geom_type
    """assert (snap_df.is_ring).all(), (
        explore(to_gdf(snap_df[~snap_df.is_ring].geometry, 25833)),
        snap_df.is_ring,
    )"""

    snap_df.geometry = extract_unique_points(snap_df.geometry.values)
    snap_df = snap_df.explode(ignore_index=True)

    to_gdf(snap_df.geometry).to_parquet(
        r"C:\Users\ort\git\ssb-sgis\tests\testdata\snap_df3.parquet"
    )

    try:
        qtm(
            snap_df=to_gdf(
                linearrings(
                    get_coordinates(snap_df.geometry.values),
                    indices=snap_df.ring_index.values,
                ),
                25833,
            ).clip(mask.buffer(0.5)),
            geoms=to_gdf(geoms, 25833).clip(mask.buffer(0.5)),
            title="snap_df3",
        )
    except Exception:
        pass

    qtm(
        # gaps=to_gdf(gaps).clip(mask),
        rings=to_gdf(rings).clip(mask),
        snap_to=to_gdf(snap_to).clip(mask),
        # ring_points=to_gdf(ring_points["ring_point"]).clip(mask),
        # ring_point=to_gdf(snap_df["ring_point"]).clip(mask),
        snap_df=to_gdf(snap_df["geometry"]).clip(mask),
        title="hernaaa",
    )

    display("\nnærmer oss")
    display(snap_df)

    if 0:
        snap_df.index = pd.MultiIndex.from_arrays(
            [snap_df["_ring_index"].values, range(len(snap_df))]
        )

        snap_points_df = (
            GeoDataFrame(
                {"geometry": extract_unique_points(get_parts(line_merge(snap_to)))}
            ).explode(index_parts=False)
            # .pipe(remove_points_on_straight_lines)
        )

        snap_points_df.index = snap_points_df.geometry

        """snap_points_df

        snap_points_df = (
            GeoDataFrame({"geometry": snap_points_arr}, index=snap_points_arr)
            .pipe(remove_points_on_straight_lines_from_ring)
            .pipe(buff, tolerance, resolution=10)
        )"""

        joined = snap_df.sjoin(
            buff(snap_points_df, tolerance, resolution=10, copy=True)
        )  # temp true
        joined["distance"] = distance(
            joined.geometry.values, joined["index_right"].values
        )

        qtm(joined=joined.clip(mask))

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

        explore(
            snap_points_df,
            snap_df=to_gdf(snap_df.geometry, 25833),
            missing=to_gdf(missing["geometry"].dropna(), 25833),
            joined=to_gdf(joined["index_right"], 25833),
            unique=to_gdf(unique, 25833),
            rings=to_gdf(rings, 25833),
            snap_to=to_gdf(snap_to, 25833),
        )

        unique_qgis = (
            joined.sort_index()
            # .sort_values("angle_diff", ascending=False)
            .drop_duplicates(["wkt", "_ring_index"]).sort_values("distance")
            # .loc[lambda x: ~x.index.duplicated()]
            .drop(columns="geometry")
        )
        for col in unique_qgis.columns.difference({"index_right"}):
            unique_qgis[col] = unique_qgis[col].astype(str)
        GeoDataFrame(unique_qgis, geometry="index_right", crs=25833).to_parquet(
            r"C:\Users\ort\git\ssb-sgis\tests\testdata\unique.parquet"
        )
        if 0:
            for col in joined.columns.difference({"geometry"}):
                joined[col] = joined[col].astype(str)

            (joined.set_crs(25833)).to_parquet(
                r"C:\Users\ort\git\ssb-sgis\tests\testdata\joined.parquet"
            )

            snap_points_df["wkt"] = [x.wkt for x in snap_points_df.index]
            snap_points_df.geometry = list(snap_points_df.index)
            snap_points_df = snap_points_df.reset_index()

            for col in snap_points_df.columns.difference({"geometry"}):
                snap_points_df[col] = snap_points_df[col].astype(str)

            (make_all_singlepart(snap_points_df.set_crs(25833))).to_parquet(
                r"C:\Users\ort\git\ssb-sgis\tests\testdata\snap_points_df.parquet"
            )
        snapped.index = snapped.index.droplevel(0)
        snap_df.index = snap_df.index.droplevel(0)

        snap_df["geometry"] = np.where(
            snap_df.index.isin(snapped.index),
            snap_df.index.map(snapped),
            snap_df["geometry"],
        )

    snap_df.loc[:, "geometry"] = snap_to_nearest(
        snap_df["geometry"].values, extract_unique_points(snap_to), geoms, tolerance
    )

    assert snap_df["geometry"].notna().all(), snap_df[snap_df["geometry"].isna()]

    qtm(
        rings=to_gdf(rings, 25833).clip(mask),
        # ring_points=to_gdf(snap_df["ring_point"], 25833).clip(mask),
        snap_to=to_gdf(snap_to, 25833).clip(mask),
        snap_df=to_gdf(snap_df["geometry"], 25833).clip(mask),
        title="denne",
    )

    # remove lines with only two points. They cannot be converted to polygons.
    is_ring = snap_df.groupby("_ring_index").transform("size") > 2

    not_rings = snap_df.loc[~is_ring].loc[lambda x: ~x.index.duplicated()]
    snap_df = snap_df.loc[is_ring]

    to_int_index = {
        ring_idx: i for i, ring_idx in enumerate(sorted(set(snap_df["_ring_index"])))
    }
    int_indices = snap_df["_ring_index"].map(to_int_index)
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

    print("\nas_lines")
    display(as_lines)

    to_gdf(as_lines).to_parquet(
        r"C:\Users\ort\git\ssb-sgis\tests\testdata\as_lines.parquet"
    )

    if len(not_rings):
        qtm(
            not_rings=to_gdf(not_rings).clip(mask).buffer(3),
            snap_to=to_gdf(snap_to, 25833).clip(mask),
            snap_df=to_gdf(snap_df["geometry"], 25833).clip(mask),
            title="not_rings",
        )

    as_lines = pd.concat([as_lines, not_rings]).sort_index()

    """
    for line in as_lines:
        qtm(
            line=to_gdf(line).clip(mask).buffer(1).to_frame(),
            snap_to=to_gdf(snap_to, 25833).clip(mask),
            snap_df=to_gdf(snap_df["geometry"], 25833).clip(mask),
            title="line",
        )
    """
    try:
        qtm(
            snap_to=to_gdf(snap_to, 25833).clip(mask),
            snap_df=to_gdf(snap_df["geometry"], 25833).clip(mask),
            as_lines=to_gdf(as_lines, 25833).clip(mask),
            title="helt nederst",
        )
        qtm(
            geoms=to_gdf(geoms, 25833).clip(mask.buffer(11)),
            snap_to=to_gdf(snap_to, 25833).clip(mask.buffer(11)),
            points=to_gdf(points, 25833)
            .assign(geometry=lambda x: extract_unique_points(x))
            .clip(mask.buffer(11)),
            snap_df=to_gdf(snap_df["geometry"], 25833).clip(mask.buffer(11)),
            as_lines=to_gdf(as_lines, 25833).clip(mask.buffer(11)),
            title="helt nederst buff",
        )
    except Exception:
        pass

    no_values = pd.Series(
        {i: None for i in range(len(rings)) if i not in as_lines.index}
    )

    # TODO TEMP
    assert len(pd.concat([as_lines, no_values]).sort_index()) == len(rings), (
        len(pd.concat([as_lines, no_values]).sort_index()),
        len(rings),
    )

    return pd.concat([as_lines, no_values]).sort_index()


def get_shortest_line_between_points(points: MultiPoint):
    points = GeoSeries([points]).explode(index_parts=False)
    points.index = points.values

    distances = get_all_distances(points, points)
    edges = [
        (source, target, weight)
        for source, target, weight in zip(
            distances.index, distances["neighbor_index"], distances["distance"]
        )
        # if not (source == points.index.iloc[0] and target == points.index.iloc[-1])
    ]

    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)

    shortest_path = nx.approximation.traveling_salesman_problem(
        graph, nodes=list(points.index)
    )
    return LineString(shortest_path)


def line_merge_by_force(line: MultiLineString | LineString) -> LineString:
    """converts a (multi)linestring to a linestring if possible."""

    print("\nhei line_merge_by_force")
    print(line)
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

    """if rings.length.sum() > lines.length.sum() * 0.01:
        rings = get_cheap_centerlines(rings)
        qtm(rings, not_rings)
        qtm(rings)
        raise ValueError(rings.length)"""
    if rings.length.sum() < PRECISION and len(not_rings) == 1:
        return not_rings
    elif len(rings):
        if rings.length.sum() < lines.length.sum() * 0.02:
            rings = get_cheap_centerlines(rings)
        else:
            for ring in rings.geometry:
                qtm(ring)
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

    qtm(
        lin=to_gdf(line),
        lines111=(lines),
        not_rings=(not_rings),
        long_not_rings=(not_rings[not_rings.length > PRECISION]),
        alpha=0.5,
        title="by_force1",
    )

    """lines.geometry = extract_unique_points(lines.geometry)

    lines.geometry = lines.geometry.apply(get_shortest_line_between_points)
    qtm(line=to_gdf(line), lines=(lines), title="by_force nx")"""

    # rings = lines[lines.is_ring]
    print(lines.length)
    print(rings.length)
    qtm(
        lines222=lines[~lines.is_ring],
        rings222=lines[lines.is_ring].clip(lines[~lines.is_ring].buffer(1)),
    )
    qtm(lines333=lines[~lines.is_ring])

    # lines = lines.loc[lambda x: (~x.index.isin(rings.index)) & (x.length > PRECISION)]

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

    print(line)
    if isinstance(line, LineString):
        assert line.length >= length_before - PRECISION * 100, (
            line.length - length_before
        )
        return line

    lines = GeoDataFrame({"geometry": get_parts(line)})

    print(lines)

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

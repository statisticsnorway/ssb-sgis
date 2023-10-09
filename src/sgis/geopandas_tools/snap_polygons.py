import itertools
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
    intersection,
    intersects,
    is_empty,
    length,
    line_merge,
    linearrings,
    linestrings,
    make_valid,
    points,
    polygons,
    segmentize,
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
from ..networkanalysis.closing_network_holes import close_network_holes
from ..networkanalysis.cutting_lines import split_lines_by_nearest_point
from .buffer_dissolve_explode import buff, dissexp_by_cluster
from .conversion import coordinate_array, to_gdf
from .duplicates import get_intersections
from .general import sort_large_first, to_lines
from .geometry_types import get_geom_type, make_all_singlepart
from .neighbors import get_all_distances, k_nearest_neighbors
from .overlay import clean_overlay
from .polygon_operations import close_small_holes, get_gaps, get_holes
from .sfilter import sfilter_inverse


mask = to_gdf(
    [
        "POINT (905200 7878700)",
        "POINT (905250 7878780)",
    ],
    25833,
).pipe(buff, 30)

mask = to_gdf("POINT (905139.722 7878785.909)", crs=25833).buffer(110)
mask = to_gdf("POINT (905043 7878849)", crs=25833).buffer(65)
mask = to_gdf("POINT (905097 7878848)", crs=25833).buffer(35)
# mask = to_gdf("POINT (905098.5 7878848.9)", crs=25833).buffer(3)

mask = to_gdf("POINT (905225 7878532)", crs=25833).buffer(100)
mask = to_gdf("POINT (905265 7878553)", crs=25833).buffer(50)

# mask = to_gdf([5.95201, 62.41451], 4326).to_crs(25833).buffer(100)


PRECISION = 1e-4


def coverage_clean(gdf: GeoDataFrame, tolerance: int | float) -> GeoDataFrame:
    return close_small_holes(gdf, tolerance).pipe(snap_polygons, tolerance)


def snap_polygons(
    gdf: GeoDataFrame,
    tolerance: float,
    snap_to: GeoDataFrame | None = None,
    snap_to_largest: bool = True,
) -> GeoDataFrame:
    if not len(gdf) or not tolerance:
        return gdf
    if not gdf.index.is_unique:
        raise ValueError("Index must be unique")
    if get_geom_type(gdf) != "polygon":
        raise ValueError("Must be polygons.")

    crs = gdf.crs

    gdf = make_all_singlepart(gdf)

    if snap_to is None:
        gap_lines, gdf = _get_gaps_and_double(gdf, tolerance)
    else:
        gap_lines = to_lines(snap_to[["geometry"]])
        gap_lines = gap_lines.explode(ignore_index=True)
        gap_lines["_gap_idx"] = range(len(gap_lines))
        gap_lines["_on_outskirts"] = 0

    if not len(gap_lines):
        return gdf

    if snap_to_largest:
        gdf = sort_large_first(gdf)

    gdf = gdf.reset_index(drop=True)
    gdf["_gdf_idx"] = range(len(gdf))

    gap_lines_joined = (
        pd.DataFrame(buff(gap_lines, PRECISION).sjoin(gdf, how="left"))
        .sort_values("index_right")
        .drop_duplicates("_gap_idx")
    )

    gap_lines_joined.geometry = gap_lines.geometry

    gdf = pd.DataFrame(gdf)

    intersect_gaps = gdf.loc[lambda x: x.index.isin(gap_lines_joined["index_right"])]
    do_not_intersect = gdf.loc[lambda x: ~x.index.isin(gap_lines_joined["index_right"])]

    display(intersect_gaps)

    qtm(
        to_gdf(extract_unique_points(intersect_gaps.geometry.values)).clip(
            to_gdf("POINT (905139.722 7878785.909)", crs=25833).buffer(20)
        ),
        title="foer segmentize_polygons_by_points",
    )

    """if 0:
        intersect_gaps = make_all_singlepart(intersect_gaps, ignore_index=True)
        gap_points = GeoDataFrame(
            {
                "geometry": get_parts(
                    extract_unique_points(gap_lines_joined.geometry.values)
                )
            },
            crs=gdf.crs,
        )

        def func(gdf, points, tolerance):
            gdf = split_lines_by_nearest_point(gdf, points, tolerance)
            return gdf.dissolve(by="_gdf_idx", as_index=False)

        intersect_gaps: GeoDataFrame = (
            PolygonRings(intersect_gaps)
            .apply_gdf_func(
                func=func,
                args=(gap_points, tolerance),
            )
            .to_gdf()
        )

        intersect_gaps = make_all_singlepart(intersect_gaps, ignore_index=True)"""

    """intersect_gaps.geometry = segmentize_polygons_by_points(
        intersect_gaps.geometry, gap_lines_joined.geometry, tolerance
    )
    """
    display(intersect_gaps)

    qtm(
        # to_gdf(intersect_gaps.geometry).clip(
        #    to_gdf("POINT (905139.722 7878785.909)", crs=25833).buffer(20)
        # ),
        to_gdf(extract_unique_points(intersect_gaps.geometry.values)).clip(
            to_gdf("POINT (905139.722 7878785.909)", crs=25833).buffer(20)
        ),
        title="etter segmentize_polygons_by_points",
    )

    snap_to: MultiLineString = get_snap_to_lines(gap_lines_joined, intersect_gaps)

    qtm(
        to_gdf(snap_to).clip(
            to_gdf("POINT (905139.722 7878785.909)", crs=25833).buffer(110)
        ),
        title="etter clip",
    )

    geoms_negbuff = buffer(intersect_gaps.geometry.values, -PRECISION * 100)

    intersect_gaps.geometry = (
        PolygonRings(intersect_gaps.geometry, crs=crs)
        .apply_numpy_func(
            _snap_linearring,
            args=(
                snap_to,
                unary_union(gap_lines_joined.geometry.values),
                geoms_negbuff,
                tolerance * 2.0001,
            ),
        )
        .to_numpy()
    )

    qtm(
        to_gdf(intersect_gaps.geometry.values).clip(mask),
        gaps1=get_gaps(to_gdf(intersect_gaps.geometry.values).clip(mask))
        .buffer(1)
        .to_frame(),
        gaps=get_gaps(to_gdf(intersect_gaps.geometry.values).clip(mask)),
        title="etter snap",
    )
    """intersect_gaps.geometry = _snap(
        intersect_gaps.geometry,
        snap_to,
        unary_union(gap_lines_joined.geometry.values),
        tolerance * 2.0001,
    )"""

    cols_to_keep = lambda x: x.columns.difference(
        {
            "index_right",
            "_gap_idx",
            "_gdf_idx",
            "_n_gaps",
            "_ring_index",
            "_on_outskirts",
        }
    )
    return GeoDataFrame(
        pd.concat(
            [
                intersect_gaps,
                do_not_intersect,
            ],
            ignore_index=True,
        ).loc[:, cols_to_keep],
        crs=crs,
    )


def _get_all_holes(
    gdf: GeoDataFrame, tolerance: float | int
) -> tuple[GeoDataFrame, GeoDataFrame]:
    """Trying two options for finding gaps."""

    holes = dissexp_by_cluster(gdf).pipe(get_holes)

    # bbox slightly larger than the bounds to be sure the outer surface is
    # one large polygon after difference
    bbox = GeoDataFrame(
        {"geometry": [box(*tuple(gdf.total_bounds)).buffer(tolerance * 2)]}, crs=gdf.crs
    )

    bbox_gaps = clean_overlay(bbox, gdf, how="difference", geom_type="polygon")

    return holes, bbox_gaps


def _get_gaps_and_double(gdf: GeoDataFrame, tolerance: float | int) -> GeoDataFrame:
    geom_col = gdf._geometry_column_name

    gdf = gdf[[geom_col]]

    gaps_and_holes = get_gaps(gdf, include_interiors=True)

    double = get_intersections(gdf)

    # slivers to be eliminated
    is_sliver = gdf.buffer(-tolerance).is_empty
    sliver_polygons = gdf.loc[is_sliver]
    gdf = gdf.loc[~is_sliver]

    sliver_polygons["_idx"] = range(len(sliver_polygons))

    # converting to rings and removing the parts intersecting thick polygons
    # the remaining lines can be snapped to
    sliver_on_outskirts = clean_overlay(
        to_lines(sliver_polygons), buff(gdf, tolerance), how="difference"
    )

    # these thin polygons are in the middle of the area and can be treated as gaps
    slivers_in_between = sliver_polygons.loc[
        lambda x: ~x["_idx"].isin(sliver_on_outskirts["_idx"])
    ]

    thin_gaps = (
        pd.concat([gaps_and_holes, double, slivers_in_between])
        .loc[lambda x: x.buffer(-tolerance).is_empty]
        .pipe(dissexp_by_cluster)
        .pipe(to_lines)
    )

    thin_gaps.loc[thin_gaps.area > 1, geom_col] = segmentize_triangles(
        thin_gaps.loc[thin_gaps.area > 1, geom_col]
    )

    sliver_on_outskirts["_on_outskirts"] = 1
    thin_gaps["_on_outskirts"] = 0

    snap_to = pd.concat([thin_gaps, sliver_on_outskirts], ignore_index=True)
    snap_to["_gap_idx"] = range(len(snap_to))

    return snap_to, gdf


def get_snap_to_lines(
    gaps: pd.DataFrame, intersect_gaps: pd.DataFrame
) -> MultiLineString:
    on_outskirts: NDArray = gaps.loc[
        lambda x: x["_on_outskirts"] == 1, "geometry"
    ].values
    gaps = gaps.loc[lambda x: x["_on_outskirts"] != 1]

    # convert lines shorter than PRECISION to points
    not_connected = gaps[gaps["index_right"].isna()].geometry
    if len(not_connected):
        assert (length_max := length(not_connected).max()) < PRECISION, length_max
        not_connected = centroid(not_connected.values)

        gaps = gaps[gaps["index_right"].notna()]

    indices = gaps["index_right"].values
    intersect_gaps = intersect_gaps.loc[indices]

    assert (gaps["index_right"] == intersect_gaps.index).all()

    intersected: NDArray[LineString] = intersection(
        gaps.geometry.values, buffer(intersect_gaps.geometry.values, PRECISION)
    )

    return unary_union(np.concatenate([intersected, on_outskirts, not_connected]))


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


class PolygonRings:
    def __init__(self, gdf: GeoDataFrame, crs=None):
        if not isinstance(gdf, (pd.DataFrame, pd.Series, ndarray, GeometryArray)):
            raise TypeError(type(gdf))

        self.integer_index = list(range(len(gdf)))
        self.index_mapper = {i: idx for i, idx in enumerate(gdf.index)}
        self.gdf = gdf.reset_index(drop=True)

        if crs is not None:
            self.crs = crs
        elif hasattr(gdf, "crs"):
            self.crs = crs
        else:
            self.crs = None

        self.polygons: GeoSeries = (
            gdf.geometry if isinstance(gdf, (GeoDataFrame, pd.DataFrame)) else gdf
        )

        exterior = pd.Series(
            get_exterior_ring(self.polygons.values),
            index=self.exterior_index,
        )

        self.max_rings: int = np.max(get_num_interior_rings(self.polygons.values))

        if not self.max_rings:
            self.rings = exterior
            return

        # series same length as number of potential inner rings
        interiors = pd.Series(
            (
                [
                    [get_interior_ring(geom, i) for i in range(self.max_rings)]
                    for geom in self.polygons
                ]
            ),
        ).explode()

        interiors.index = self.interiors_index

        interiors = interiors.dropna()

        self.rings = pd.concat([exterior, interiors])

    def apply_numpy_func(
        self, func: Callable, args: tuple | None = None, kwargs: dict | None = None
    ):
        kwargs = kwargs or {}
        self.rings = pd.Series(
            np.array(func(self.rings.values, *args, **kwargs)),
            index=self.rings.index,
        )
        return self

    def apply_geoseries_func(
        self, func: Callable, args: tuple | None = None, kwargs: dict | None = None
    ):
        kwargs = kwargs or {}
        self.rings = pd.Series(
            np.array(
                func(
                    GeoSeries(self.rings, crs=self.crs, index=self.rings.index),
                    *args,
                    **kwargs,
                )
            ),
            index=self.rings.index,
        )

        return self

    def apply_gdf_func(
        self, func: Callable, args: tuple | None = None, kwargs: dict | None = None
    ):
        kwargs = kwargs or {}

        gdf = GeoDataFrame(
            {"geometry": self.rings.values},
            crs=self.crs,
            index=self.rings.index.get_level_values(1),
        ).join(self.gdf.drop(columns="geometry"))

        assert len(gdf) == len(self.rings)

        gdf.index = self.rings.index

        self.rings = pd.Series(
            func(
                gdf,
                *args,
                **kwargs,
            ).geometry.values,
            index=self.rings.index,
        )

        return self

    def to_gdf(self) -> GeoDataFrame:
        self.gdf.geometry = self.to_numpy()
        return self.gdf

    @property
    def interiors_index(self):
        """A three-leveled MultiIndex.

        Used to separate interior and exterior and sort the interior in
        the 'to_numpy' method.

        level 0: all 1s, indicating "is interior".
        level 1: gdf index repeated *self.max_rings* times.
        level 2: interior number index. 0 * len(gdf), 1 * len(gdf), 2 * len(gdf)...
        """
        if not self.max_rings:
            return pd.MultiIndex()
        len_gdf = len(self.gdf)
        n_potential_interiors = len_gdf * self.max_rings
        gdf_index = sorted(list(self.gdf.index) * self.max_rings)
        interior_number_index = np.tile(np.arange(self.max_rings), len_gdf)
        is_interior = np.repeat(1, n_potential_interiors)

        return pd.MultiIndex.from_arrays(
            [is_interior, gdf_index, interior_number_index]
        )

    @property
    def exterior_index(self):
        """A three-leveled MultiIndex.

        Used to separate interior and exterior in the 'to_numpy' method.
        Only leve 1 is used for the exterior.

        level 0: all 0s, indicating "not interior".
        level 1: gdf index.
        level 2: All 0s.
        """
        is_interior = np.repeat(0, len(self.gdf))
        return pd.MultiIndex.from_arrays([is_interior, self.gdf.index, is_interior])

    def to_numpy(self) -> NDArray[Polygon]:
        exterior = (
            self.rings.loc[lambda x: x.index.get_level_values(0) == 0]
            .sort_index()
            .values
        )
        assert exterior.shape == (len(self.gdf),)

        nonempty_interiors = self.rings.loc[lambda x: x.index.get_level_values(0) == 1]

        if not len(nonempty_interiors):
            return make_valid(polygons(exterior))

        empty_interiors = pd.Series(
            [None for _ in range(len(self.gdf) * self.max_rings)],
            index=self.interiors_index,
        ).loc[lambda x: ~x.index.isin(nonempty_interiors.index)]

        interiors = (
            pd.concat([nonempty_interiors, empty_interiors])
            .sort_index()
            # make each ring level a column with same length and order as gdf
            .unstack(level=2)
            .sort_index()
            .values
        )
        assert interiors.shape == (len(self.gdf), self.max_rings), interiors.shape

        return make_valid(polygons(exterior, interiors))


def _snap(
    geoms: GeoSeries, snap_to: Geometry, all_gaps: Geometry, tolerance
) -> NDArray[Polygon]:
    geoms_negbuff = buffer(geoms, -PRECISION * 100)

    return (
        PolygonRings(geoms)
        .apply_numpy_func(
            _snap_linearring, args=(snap_to, all_gaps, geoms_negbuff, tolerance)
        )
        .to_numpy()
    )


def get_more_line_points(
    line: LinearRing | MultiLineString,
    snap_to: MultiLineString,
    multipoints: NDArray[MultiPoint],
    tolerance: float | int,
) -> pd.DataFrame:
    """Make points along the line between the vertices.

    Can be more than one between two vertices.

    Returns:
        Series with index of line vertices ('point' column of df) and
            aggregated values (multipoints) of vertices in 'snap_to'.
    """
    snap_vertices, ring_points = nearest_points(
        get_parts(extract_unique_points(snap_to)), unary_union(line)
    )
    ring_vertices = nearest_points(ring_points, unary_union(multipoints))[1]
    distance_to_snap_to = distance(snap_vertices, ring_points)
    distance_to_ring_vertice = distance(ring_vertices, ring_points)

    return pd.DataFrame(
        {
            "ring_point": ring_points,
            "snap_vertice": snap_vertices,
        },
    ).loc[
        lambda x: (  # (x["distance_to_snap_to"] <= tolerance)
            x["distance_to_ring_vertice"] > 0
        )
        & (x["ring_point"].notna()),
        ["ring_point"],
    ]


def join_lines_with_snap_to(
    lines: GeoDataFrame,
    snap_to: MultiLineString,
    tolerance: int | float,
) -> GeoDataFrame:
    points: NDArray[Point] = get_parts(extract_unique_points(snap_to))
    points_df = GeoDataFrame({"geometry": points}, index=points)
    joined = (
        # clean_overlay(
        buff(lines, tolerance)  # lines.length)  # ,
        # geoms.to_frame(),
        # how="difference",
        # geom_type="polygon",
        # )
        # .pipe(clean_overlay, geoms)
        .sjoin(points_df, how="left")
    )
    joined.geometry = lines.geometry

    notna = joined["index_right"].notna()

    ring_points = nearest_points(
        joined.loc[notna, "index_right"].values, joined.loc[notna, "geometry"].values
    )[1]

    joined.loc[notna, "ring_point"] = ring_points

    # joined["ring_point"] = sfilter_inverse(GeoSeries(joined["ring_point"]), geoms)

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

    return linestrings(coords.values, indices=coords.index)


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
        point_df = pd.DataFrame({"point": GeometryArray(points)}, index=indices)

    point_df["next"] = point_df.groupby(level=0)["point"].shift(-1)

    first_points = point_df.loc[lambda x: ~x.index.duplicated(), "point"]
    is_last_point = point_df["next"].isna()

    point_df.loc[is_last_point, "next"] = first_points
    assert point_df["next"].notna().all()

    point_df["geometry"] = [
        LineString([x1, x2]) for x1, x2 in zip(point_df["point"], point_df["next"])
    ]
    return GeoDataFrame(
        point_df.drop(columns=["point", "next"]), geometry="geometry", crs=crs
    )


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

    """
    diffs = difference(rings, snap_to.buffer(PRECISION))
    ring_points: pd.DataFrame = pd.concat(
        get_more_line_points(diff, snap_to, multipoints, tolerance) for diff in diffs
    )

    more_ring_points: pd.DataFrame = pd.concat(
        get_more_line_points(ring, all_gaps, multipoints, tolerance) for ring in rings
    )

    even_more_ring_points: pd.DataFrame = pd.concat(
        get_more_line_points(diff, all_gaps, multipoints, tolerance) for diff in diffs
    )

    ring_points = pd.concat([ring_points, more_ring_points, even_more_ring_points])

    try:
        qtm(
            rings=to_gdf(rings).clip(mask.buffer(55)),
            snap_to=to_gdf(snap_to).clip(mask.buffer(55)),
            ring_points=to_gdf(ring_points["ring_point"]).clip(mask.buffer(55)),
            # snap_df=to_gdf(snap_df.point).clip(mask.buffer(55)),
            title="toppen",
        )
    except Exception:
        pass

    display(snap_df)

    snap_df = snap_df.sjoin(
        GeoDataFrame(
            ring_points.assign(ring_point2=lambda x: x["ring_point"]),
            geometry="ring_point2",
        ).pipe(buff, PRECISION),
        how="left",
    )
    # snap_df: line segments med kolonne ring_point
    """
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
        title="snap_df",
    )

    # snap_df = sfilter_inverse(snap_df, geoms)

    print("len snap_df 3")
    print(len(snap_df))

    qtm(
        snap_df=to_gdf(snap_df.geometry, 25833).clip(mask.buffer(0.5)),
        geoms=to_gdf(geoms, 25833).clip(mask.buffer(0.5)),
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

    snap_df.loc[is_not_merged, "geometry"] = snap_df.loc[
        is_not_merged, "geometry"
    ].apply(line_merge_by_force)

    assert (
        snap_df.geom_type.isin(["LineString", "LinearRing"])
    ).all(), snap_df.geom_type
    # assert (snap_df.is_ring).all(), snap_df.is_ring

    snap_df.geometry = extract_unique_points(snap_df.geometry.values)
    snap_df = snap_df.explode(index_parts=False)

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

    try:
        qtm(
            # gaps=to_gdf(gaps).clip(mask),
            rings=to_gdf(rings).clip(mask),
            snap_to=to_gdf(snap_to).clip(mask),
            # ring_points=to_gdf(ring_points["ring_point"]).clip(mask),
            snap_df=to_gdf(snap_df["geometry"]).clip(mask),
            title="hernaaa",
        )
    except Exception:
        pass

    '''

    is_close = snap_df["ring_point"].notna()
    to_insert = snap_df.loc[is_close]
    not_to_insert = snap_df.loc[~is_close]

    to_insert["next_point"] = (
        to_insert["point"].groupby(level=0).shift(1).fillna(Point())
    )
    to_insert["prev_point"] = (
        to_insert["point"].groupby(level=0).shift(-1).fillna(Point())
    )

    closest_to_next = (to_insert["prev_point"].isna()) | (
        distance(to_insert["point"], to_insert["next_point"])
        < distance(to_insert["point"], to_insert["prev_point"])
    )
    """to_insert["point"] = np.where(
        closest_to_next,
        (to_insert["point"], to_insert["ring_point"]),
        (to_insert["ring_point"], to_insert["point"]),
    )"""

    to_insert["point"] = [
        (vertice, point) if is_closer else (point, vertice)
        for vertice, point, is_closer in zip(
            to_insert["point"], to_insert["ring_point"], closest_to_next
        )
    ]
    snap_df = (
        pd.concat([to_insert, not_to_insert]).sort_values("range_idx").explode("point")
    )
    '''

    # snap_df["point"] = snap_df["point"].replace(snapmapper)

    # snap_df = GeoSeries(snap_df).explode(index_parts=False)
    # snap_df = GeoDataFrame(snap_df, geometry="line").explode(index_parts=False)
    display("\nnærmer oss")
    display(snap_df)

    qtm(
        # p=to_gdf(snap_df["point"], 25833).clip(mask),
        snap_df=to_gdf(snap_df["geometry"], 25833).clip(mask),
        linepoints=to_gdf(
            extract_unique_points(snap_df["geometry"].values), 25833
        ).clip(mask),
        title="geometry",
    )
    snap_df["geometry"] = map_to_nearest(
        snap_df["geometry"].values, snap_to, geoms, tolerance
    )

    try:
        qtm(
            rings=to_gdf(rings, 25833).clip(mask),
            ring_points=to_gdf(snap_df["ring_point"], 25833).clip(mask),
            snap_to=to_gdf(snap_to, 25833).clip(mask),
            snap_df=to_gdf(snap_df["geometry"], 25833).clip(mask),
            title="denne",
        )
    except Exception:
        pass

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

    display(as_lines)

    if len(not_rings):
        qtm(
            not_rings=to_gdf(not_rings).clip(mask).buffer(3),
            snap_to=to_gdf(snap_to, 25833).clip(mask),
            snap_df=to_gdf(snap_df["geometry"], 25833).clip(mask),
            title="not_rings",
        )

    as_lines = pd.concat([as_lines, not_rings]).sort_index()

    for line in as_lines:
        qtm(
            line=to_gdf(line).clip(mask).buffer(1).to_frame(),
            snap_to=to_gdf(snap_to, 25833).clip(mask),
            snap_df=to_gdf(snap_df["geometry"], 25833).clip(mask),
            title="line",
        )

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

    return pd.concat([as_lines, no_values]).sort_index()


def line_merge_by_force(line: MultiLineString | LineString) -> LineString:
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

    if rings.length.sum() > PRECISION * 10:
        raise ValueError(rings.length)

    qtm(
        lin=to_gdf(line),
        lines111=(lines),
        not_rings=(not_rings),
        long_not_rings=(not_rings[not_rings.length > PRECISION]),
        alpha=0.5,
    )

    # rings = lines[lines.is_ring]
    print(lines.length)
    print(rings.length)
    qtm(
        lines222=lines[~lines.is_ring],
        rings222=lines[lines.is_ring].clip(lines[~lines.is_ring].buffer(1)),
    )
    qtm(lines333=lines[~lines.is_ring])

    # lines = lines.loc[lambda x: (~x.index.isin(rings.index)) & (x.length > PRECISION)]

    lines = close_network_holes(
        not_rings[not_rings.length > PRECISION],
        max_distance=PRECISION * 10,
        max_angle=180,
    )
    line = line_merge(unary_union(lines.geometry.values))
    if isinstance(line, LineString):
        assert line.length >= length_before - PRECISION, line.length - length_before
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


"""
def sort_points_to_line(geoms):
    geoms.index = geoms.geometry
    distances = get_all_distances(geoms, geoms)
    edges = [
        (source, target, weight)
        for source, target, weight in zip(
            distances.index, distances["neighbor_index"], distances["distance"]
        )
    ]
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)
    tree = nx.minimum_spanning_tree(graph)
    return list(itertools.chain(*tree.edges()))
"""

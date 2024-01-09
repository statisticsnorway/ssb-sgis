# %%

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import extract_unique_points


src = str(Path(__file__).parent).strip("tests") + "src"


import sys


sys.path.insert(0, src)


import shapely
from geopandas import *
from geopandas.array import GeometryArray
from numpy.typing import NDArray
from shapely import *
from shapely.ops import *

import sgis as sg
from sgis import *
from sgis.geopandas_tools.cleaning import _remove_spikes


PRECISION = 1e-4
BUFFER_RES = 50


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
) -> NDArray[Point]:
    nearest = nearest_points(points, unary_union(snap_to))[1]
    distance_to_nearest = distance(points, nearest)

    as_lines = make_lines_between_points(points, nearest)
    intersect_geoms: list[NDArray[bool]] = [
        intersects(as_lines, geom) for geom in geoms
    ]
    intersect_geoms: NDArray[bool] = np.any(np.array(intersect_geoms), axis=0)

    return np.where(
        ((distance_to_nearest <= tolerance) & (~intersect_geoms)),
        nearest,
        None,  # points,  # None,  # ,
    )


def multipoints_to_line_segments(multipoints: GeoSeries) -> GeoDataFrame:
    if not len(multipoints):
        return GeoDataFrame({"geometry": multipoints}, index=multipoints.index)

    try:
        crs = multipoints.crs
    except AttributeError:
        crs = None

    try:
        point_df = multipoints.explode(index_parts=False)
    except AttributeError:
        points, indices = get_parts(multipoints, return_index=True)
        if hasattr(multipoints, "index") and isinstance(
            multipoints.index, pd.MultiIndex
        ):
            indices = pd.MultiIndex.from_arrays(indices, names=multipoints.index.names)

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

    joined.loc[notna, "ring_point"] = ring_points

    return joined


def sorted_unary_union(df: pd.DataFrame) -> MultiPoint:
    assert len(df["endpoints"].unique()) <= 1, df["endpoints"].unique()
    # assert len(df["point"].unique()) <= 1, df["point"].unique()
    assert len(df["geometry"].unique()) <= 1, df["geometry"].unique()

    endpoints: ndarray = get_coordinates(df["endpoints"].iloc[0])
    between: ndarray = get_coordinates(df["ring_point"].dropna().values)

    coords = np.concatenate([endpoints, between])
    sorted_coords = coords[np.argsort(coords[:, -1])]

    # droping points outside the line (returned from sjoin because of buffer)
    is_between_endpoints = (sorted_coords[:, 0] >= np.min(endpoints[:, 0])) & (
        sorted_coords[:, 0] <= np.max(endpoints[:, 0])
    )
    sorted_coords = sorted_coords[is_between_endpoints]

    return LineString(sorted_coords)


def _snap_linearring(
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

    # snap_df = snap_df.dissolve(by="_ring_index", as_index=False)
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

    """snap_df.loc[:, "geometry"] = snap_to_nearest(
        snap_df["geometry"].values, extract_unique_points(snap_to), geoms, tolerance
    )"""

    assert snap_df["geometry"].notna().all(), snap_df[snap_df["geometry"].isna()]

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

    as_lines = pd.concat([as_lines, not_rings]).sort_index()

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


def _remove_spikes(
    geoms: NDArray[LinearRing], tolerance: int | float
) -> NDArray[LinearRing]:
    geoms = to_geoseries(geoms).dropna().reset_index(drop=True)
    buffered = (
        geoms.buffer(PRECISION, resolution=BUFFER_RES)
        .apply(lambda x: Polygon(x.exterior))
        .buffer(-PRECISION * 2, resolution=BUFFER_RES)
        .buffer(PRECISION, resolution=BUFFER_RES)
        .apply(lambda x: x.exterior)
        .buffer(PRECISION, resolution=BUFFER_RES)
    )
    missing = ~buffered.index.isin(geoms.index)
    buffered.loc[missing] = geoms.loc[missing]

    points = extract_unique_points(geoms).explode(index_parts=False)
    points_without_spikes = sfilter(points, buffered)
    print(points_without_spikes.index.values)
    for i in points_without_spikes.index.unique():
        explore(
            sg.to_gdf(points_without_spikes[points_without_spikes.index == i], 25833)
        )
    sssssa
    return linearrings(
        get_coordinates(points_without_spikes),
        indices=points_without_spikes.index.values,
    )


def _snap(df, snap_to, tolerance):
    df = df.reset_index()
    df["_ring_index"] = range(len(df))
    intersecting = clean_overlay(
        to_lines(snap_to),
        buff(df, tolerance),
        how="intersection",
    )
    assert intersecting.geometry.notna().all()

    erased = clean_overlay(
        df,
        buff(snap_to, tolerance),
        how="difference",
    )
    all_together = (
        GeoSeries(
            pd.concat([erased, intersecting])
            # intersecting.groupby("_ring_index")
            ["geometry"]
            .agg(lambda x: _remove_spikes(x, tolerance))
            .explode()
        )
        .pipe(clean_geoms)
        .groupby("_ring_index")
    ).agg(unary_union)

    sg.explore(intersecting, df, snap_to, all_together=all_together)

    for i in all_together.geometry:
        sg.explore(to_gdf(i, 25833))
    return all_together


def snap_polygons(
    gdf: GeoDataFrame,
    snap_to: GeoDataFrame,
    tolerance: int | float,
) -> GeoDataFrame:
    gdf = make_all_singlepart(clean_geoms(gdf))
    gdf.geometry = (
        PolygonsAsRings(gdf.geometry)
        # .apply_numpy_func(_snap, args=(snap_to.unary_union, gdf.geometry, tolerance))
        .apply_geoseries_func(_snap, args=(snap_to, tolerance)).to_numpy()
    )
    return gdf


# sg.snap_polygons = snap_polygons


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

    assert len(cleaned_dissolved_problem_area) == 1, cleaned_dissolved_problem_area

    assert (
        area := int(cleaned_dissolved_problem_area.area.sum())
    ) == AREA_SHOULD_BE, area


def test_clean_1144():
    df = gpd.read_parquet(
        Path(__file__).parent / "testdata" / "snap_problem_area_1144.parquet"
    )

    for tolerance in [1, 0.1, 0.01, 0.001]:
        print(tolerance)

        cleaned = sg.coverage_clean(df, tolerance)

        gaps = sg.get_gaps(cleaned)
        double = sg.get_intersections(cleaned)

        sg.explore(
            df,  # =cleaned.assign(wkt=lambda x: x.geometry.to_wkt()),
            cleaned,  # =cleaned.assign(wkt=lambda x: x.geometry.to_wkt()),
            double,  # =double.assign(wkt=lambda x: x.to_wkt()),
            gaps,  # =gaps.assign(wkt=lambda x: x.to_wkt()),
        )

        cols = [
            "ARGRUNNF",
            "ARJORDBR",
            "ARKARTSTD",
            "ARSKOGBON",
            "ARTRESLAG",
            "ARTYPE",
            "ARVEGET",
            "ASTSSB",
            "geometry",
            "kilde",
        ]

        assert list(sorted(cleaned.columns)) == cols, list(sorted(cleaned.columns))

        assert double.area.sum() < 1e-3, double.area.sum()
        assert gaps.area.sum() < 1e-2, (
            gaps.area.sum(),
            gaps.area.max(),
            gaps.area,
            gaps,
        )
        assert int(cleaned.area.sum()) == 154240, cleaned.area.sum()
        assert int(df.area.sum()) == 154240, df.area.sum()

        assert sg.get_geom_type(cleaned) == "polygon", sg.get_geom_type(cleaned)

        cleaned2 = sg.coverage_clean(cleaned, tolerance=tolerance)

        gaps = sg.get_gaps(cleaned2)
        double = sg.get_intersections(cleaned2)

        assert list(sorted(cleaned2.columns)) == cols, cleaned2.columns

        assert double.area.sum() < 1e-6, double.area.sum()
        assert gaps.area.sum() < 1e-2, gaps.area.sum()
        assert int(cleaned2.area.sum()) == 154240, cleaned2.area.sum()

        assert sg.get_geom_type(cleaned2) == "polygon", sg.get_geom_type(cleaned2)

        # snapped = sg.snap_polygons(df, cleaned, tolerance)


def test_clean():
    mask = sg.to_gdf("POINT (905139.722 7878785.909)", crs=25833).buffer(330)

    df = gpd.read_parquet(Path(__file__).parent / "testdata" / "polygon_snap.parquet")

    holes = sg.to_gdf(
        [
            "POINT (905200 7878700)",
            "POINT (905250 7878780)",
            "POINT (905242.961 7878773.758)",
        ],
        25833,
    ).pipe(sg.buff, 3)

    df = sg.clean_overlay(df, holes, how="difference")

    # sg.qtm(df.clip(mask), alpha=0.5)
    # sg.qtm(df=df.clip(mask).pipe(sg.buff, -0.5), alpha=0.5)

    tolerance = 5

    cleaned = sg.coverage_clean(df, tolerance)

    return
    snapped = sg.snap_polygons(df.iloc[:3], cleaned, tolerance)

    sg.explore(snapped, cleaned, df=df.iloc[:3])

    sg.qtm(
        cleaned=cleaned.clip(mask).pipe(sg.buff, -0.5),
    )

    gaps = sg.get_gaps(cleaned)
    double = sg.get_intersections(cleaned)

    sg.qtm(
        cleaned=cleaned.clip(mask),
        double=double.clip(mask),
        gaps=gaps.clip(mask),
        alpha=0.5,
    )

    sg.explore(
        cleaned=cleaned,  # .assign(wkt=lambda x: x.to_wkt()),
        double=double,  # .assign(wkt=lambda x: x.to_wkt()),
        gaps=gaps,  # .assign(wkt=lambda x: x.to_wkt()),
    )

    assert list(sorted(cleaned.columns)) == ["geometry"], cleaned.columns

    assert double.area.sum() < 1e-6, double.area.sum()
    assert gaps.area.sum() < 1e-2, (gaps.area.sum(), gaps.area.max())
    assert int(cleaned.area.sum()) == 431076, cleaned.area.sum()

    assert sg.get_geom_type(cleaned) == "polygon", sg.get_geom_type(cleaned)

    for g in cleaned.explode().explode().geometry:
        sg.qtm(cleaned=cleaned.clip(g.buffer(1)), g=sg.to_gdf(g), alpha=0.5)

    cleaned2 = sg.coverage_clean(cleaned, tolerance=tolerance)

    gaps = sg.get_gaps(cleaned2)
    double = sg.get_intersections(cleaned2)

    assert list(sorted(cleaned2.columns)) == ["geometry"], cleaned2.columns

    assert double.area.sum() < 1e-6, double.area.sum()
    assert gaps.area.sum() < 1e-2, (gaps.area.sum(), gaps.area.max())
    assert cleaned2.area.sum() > df.area.sum()
    assert int(cleaned2.area.sum()) == 431076, cleaned2.area.sum()

    assert sg.get_geom_type(cleaned2) == "polygon", sg.get_geom_type(cleaned2)

    mask = sg.to_gdf("POINT (905139.722 7878785.909)", crs=25833).buffer(110)

    sg.qtm(cleaned2=cleaned2.clip(mask), alpha=0.5)
    sg.explore(
        cleaned=cleaned.clip(mask),
        cleaned2=cleaned2.clip(mask),
        # alpha=0.5,
        # column="idx",
    )

    # add a double surface
    df = pd.concat([df, sg.buff(df.iloc[[0]], 10)], ignore_index=True)

    cleaned3 = sg.coverage_clean(df, tolerance=tolerance)

    gaps = sg.get_gaps(cleaned3)
    double = sg.get_intersections(cleaned3)

    if __name__ == "__main__":
        sg.explore(
            gaps=gaps,  # .clip(mask),
            double=double,  # .clip(mask),
            gdf=df,  # .clip(mask),
            cleaned3=cleaned3,  # .clip(mask),
            p=sg.to_gdf(extract_unique_points(cleaned3.geometry)).pipe(sg.buff, 1),
            # alpha=0.5,
            # column="idx",
        )

    assert list(sorted(cleaned3.columns)) == ["geometry"], cleaned3.columns

    assert double.area.sum() < 1e-6, double.area.sum()
    assert gaps.area.sum() < 1e-2, gaps.area.sum()
    assert int(cleaned3.area.sum()) == 441769, cleaned3.area.sum()

    assert sg.get_geom_type(cleaned3) == "polygon", sg.get_geom_type(cleaned3)

    mask = sg.to_gdf("POINT (905139.722 7878785.909)", crs=25833).buffer(110)

    if __name__ == "__main__":
        sg.qtm(cleaned3=cleaned3.clip(mask), alpha=0.5)
        sg.explore(
            cleaned=cleaned.clip(mask),
            cleaned2=cleaned2.clip(mask),
            cleaned3=cleaned3.clip(mask),
            # alpha=0.5,
            # column="idx",
        )


def test_spikes():
    from shapely.geometry import Point, Polygon

    sliver = sg.to_gdf(Polygon([(0, 0), (0.1, 1), (0, 2), (-0.1, 1)])).assign(
        what="sliver", num=1
    )
    poly_with_spike = sg.to_gdf(
        Polygon(
            [
                (0, 0),
                (-0.1, 1),
                (0, 2),
                (-0.99, 2),
                (-0.99, 1.5),
                (-1.01, 1.5),
                (-1.01, 2),
                (-1.51, 2),
                (-1.51, 1.7),
                (-1.52, 2),
                (-2, 2),
                (-1, 1),
            ],
            holes=[
                (
                    [
                        (-0.5, 1.25),
                        (-0.5, 1.65),
                        (-0.49, 1.65),
                        (-0.49, 1.25),
                    ]
                ),
            ],
        )
    ).assign(what="small", num=2)
    poly_filling_the_spike = sg.to_gdf(
        Polygon(
            [
                (0, 2),
                (-0.99, 2),
                (-0.99, 1.5),
                (-1.01, 1.5),
                (-1.01, 2),
                (-2, 2),
                (-2, 6),
                (0, 6),
                (0, 2),
            ],
        )
    ).assign(what="small", num=2)

    df = pd.concat([sliver, poly_with_spike, poly_filling_the_spike])
    holes = sg.buff(sg.to_gdf([(-0.84, 3), (-0.84, 4.4)]), [0.4, 0.3])
    df = sg.clean_overlay(df, holes, how="update")

    tolerance = 0.09

    cleaned = sg.coverage_clean(df, tolerance)
    if __name__ == "__main__":
        sg.explore(
            cleaned=cleaned,
            df=df,
        )

    assert (area := sorted([round(x, 3) for x in cleaned.area])) == sorted(
        [
            7.225,
            1.89,
            0.503,
            0.283,
            0.2,
        ]
    ), area
    assert (length := sorted([round(x, 3) for x in cleaned.length])) == sorted(
        [
            17.398,
            7.838,
            2.513,
            1.885,
            4.02,
        ]
    ), length

    spikes_fixed = sg.split_spiky_polygons(df, tolerance)
    fixed_and_cleaned = sg.coverage_clean(spikes_fixed, tolerance)

    if __name__ == "__main__":
        sg.explore(
            fixed_and_cleaned=fixed_and_cleaned,
            spikes_fixed=spikes_fixed,
            cleaned=cleaned,
            df=df,
        )

    assert (area := sorted([round(x, 3) for x in fixed_and_cleaned.area])) == sorted(
        [
            7.215,
            1.9,
            0.503,
            0.283,
            0.2,
        ]
    ), area
    assert (
        length := sorted([round(x, 3) for x in fixed_and_cleaned.length])
    ) == sorted(
        [
            16.398,
            6.838,
            2.513,
            1.885,
            4.02,
        ]
    ), length


def main():
    test_spikes()
    test_clean()
    test_clean_1144()
    test_clean_dissappearing_polygon()


if __name__ == "__main__":
    import cProfile

    # cProfile.run("main()", sort="cumtime")

    main()

# %%
()

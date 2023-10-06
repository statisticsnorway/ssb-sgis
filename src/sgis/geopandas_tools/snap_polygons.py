import itertools

import geopandas as gpd
import igraph
import networkx as nx
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryArray
from IPython.display import display
from numpy import ndarray
from pandas import Index
from shapely import (
    Geometry,
    box,
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
    line_merge,
    linearrings,
    linestrings,
    make_valid,
    points,
    polygons,
    segmentize,
    unary_union,
)
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point
from shapely.ops import nearest_points

from ..maps.maps import explore, explore_locals, qtm
from ..networkanalysis.cutting_lines import split_lines_by_nearest_point
from .buffer_dissolve_explode import buff, dissexp_by_cluster
from .conversion import coordinate_array, to_gdf
from .duplicates import get_intersections
from .geometry_types import get_geom_type, make_all_singlepart
from .neighbors import get_all_distances, k_nearest_neighbors
from .overlay import clean_overlay
from .polygon_operations import close_small_holes, get_holes
from .sfilter import sfilter_inverse


mask = to_gdf(
    [
        "POINT (905200 7878700)",
        "POINT (905250 7878780)",
    ],
    25833,
).pipe(buff, 30)

"""
mask = to_gdf("POINT (905139.722 7878785.909)", crs=25833).buffer(110)
mask = to_gdf("POINT (905043 7878849)", crs=25833).buffer(60)
mask = to_gdf(
)
"""

PRECISION = 1e-5


def coverage_clean(gdf: GeoDataFrame, tolerance: int | float) -> GeoDataFrame:
    return (
        gdf.loc[lambda x: ~x.buffer(-tolerance).is_empty]
        .pipe(close_small_holes, tolerance)
        .pipe(snap_polygons, tolerance)
    )


def _get_gaps_and_double(gdf: GeoDataFrame, tolerance: float | int) -> GeoDataFrame:
    geom_col = gdf._geometry_column_name

    gdf = gdf[[geom_col]]

    gaps = dissexp_by_cluster(gdf).pipe(get_holes)

    # bbox slightly larger than the bounds to be sure the outer surface is
    # one large polygon after difference
    bbox = GeoDataFrame(
        {"geometry": [box(*tuple(gdf.total_bounds)).buffer(tolerance * 2)]}, crs=gdf.crs
    )

    bbox_gaps = clean_overlay(gdf, bbox, how="difference", geom_type="polygon")

    double = get_intersections(gdf)

    thin_gaps = (
        pd.concat([gaps, bbox_gaps, double])
        .loc[lambda x: x.buffer(-tolerance).is_empty]
        .pipe(dissexp_by_cluster)
    )

    thin_gaps.loc[thin_gaps.area > tolerance, geom_col] = segmentize_triangles(
        thin_gaps.loc[thin_gaps.area > tolerance, geom_col]
    )

    thin_gaps["_gap_idx"] = range(len(thin_gaps))

    thin_gaps["n_vertices"] = extract_unique_points(thin_gaps.geometry).apply(
        lambda x: len(get_parts(x))
    )
    return thin_gaps


def snap_polygons(
    gdf: GeoDataFrame, tolerance: float, snap_to: GeoDataFrame | None = None
) -> GeoDataFrame:
    if not len(gdf) or not tolerance:
        return gdf
    if not gdf.index.is_unique:
        raise ValueError("Index must be unique")
    if get_geom_type(gdf) != "polygon":
        raise ValueError("Must be polygons.")

    gdf = make_all_singlepart(gdf)

    if snap_to is None:
        thin_gaps = _get_gaps_and_double(gdf, tolerance)
    else:
        thin_gaps = snap_to[["geometry"]]
        thin_gaps["_gap_idx"] = range(
            len(thin_gaps)
        )  # .assign(_gap_idx=range(len(snap_to)))

    if not len(thin_gaps):
        return gdf

    print("\ndup", get_intersections(gdf).area.sum())

    intersect_gaps = gdf.sjoin(buff(thin_gaps, PRECISION))
    do_not_intersect = gdf.loc[lambda x: ~x.index.isin(intersect_gaps.index)]

    intersect_gaps["n_gaps"] = intersect_gaps.groupby(level=0).size()

    intersect_gaps = intersect_gaps.sort_values("n_gaps", ascending=False)

    snapped: list[GeoDataFrame] = [do_not_intersect]
    n_gaps: int = len(intersect_gaps["_gap_idx"].unique())
    visited_indices: set[int] = set()
    visited_gaps: set[int] = set()

    for idx in intersect_gaps.index.unique():
        print("idx", idx)
        geom = intersect_gaps.loc[lambda x: (x.index == idx)]
        gaps = thin_gaps.loc[lambda x: x["_gap_idx"].isin(geom["_gap_idx"])]
        new_gaps: set[int] = set(gaps["_gap_idx"]).difference(visited_gaps)

        if not new_gaps:
            continue

        gaps = gaps.loc[lambda x: x["_gap_idx"].isin(new_gaps)]

        filt = (
            lambda x: (x["_gap_idx"].isin(new_gaps))
            & (x.index != idx)
            & (~x.index.duplicated())
        )
        geoms = intersect_gaps.loc[filt]
        # snap_to: Geometry = intersection(geom.geometry.iloc[0], gaps.unary_union)
        snap_to: Geometry = intersection(gaps.unary_union, geom.geometry.iloc[0])

        qtm(
            geoms=geoms.clip(mask),
            geom=geom.clip(mask),
            snap_to=to_gdf(snap_to).clip(mask),
            alpha=0.5,
            title="foer",
        )

        geoms.geometry = _snap(geoms.geometry, snap_to, gaps.geometry, tolerance)

        intersect_gaps.loc[filt, "geometry"] = geoms.geometry

        snapped += [geoms]

        visited_gaps |= set(new_gaps)  # .difference(not_filled)
        visited_indices |= set(geoms.index)  # .union({idx}))

        dup = get_intersections(geoms).area.sum()
        if dup > 0.001:
            print("\ndup22", dup)

        qtm(
            geoms=geoms.clip(mask),
            geom=geom.clip(mask),
            snap_to=to_gdf(snap_to).clip(mask),
            alpha=0.5,
            title="etter",
        )

        if len(visited_gaps) == n_gaps:
            snapped += [geom.iloc[[0]]]
            visited_indices |= {idx}
            break

    keep_cols = lambda x: x.columns.difference(
        {"index_right", "_gap_idx", "n_gaps", "n_vertices"}
    )
    snapped = pd.concat(snapped).loc[
        lambda x: ~x.index.duplicated(keep="last"), keep_cols
    ]
    remaining = intersect_gaps.loc[
        lambda x: (~x.index.isin(visited_indices)) & (~x.index.duplicated()), keep_cols
    ]

    """snapped = pd.concat(
        [
            snapped,
            remaining,
        ]
    )
    double = get_intersections(snapped)"""

    """qtm(snapped=snapped.clip(mask), remaining=remaining.clip(mask), alpha=0.5)
    mask2 = to_gdf("POINT (905139.722 7878785.909)", crs=25833).buffer(110)
    qtm(
        snapped=snapped.clip(mask2),
        remaining=remaining.clip(mask2),
        alpha=0.5,
    )

    qtm(
        out=pd.concat(
            [
                remaining,
                snapped,
            ]
        ).clip(mask.buffer(100)),
        alpha=0.5,
    )"""

    return pd.concat(
        [
            remaining,
            snapped,
        ]
    ).pipe(lambda x: print("\ndup33", get_intersections(x).area.sum()) or x)

    return clean_overlay(
        remaining,
        snapped,
        how="update",
        geom_type="polygon",
    ).pipe(lambda x: print("\ndup33", get_intersections(x).area.sum()) or x)


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


def _snap(geoms: GeoSeries, snap_to: Geometry, gaps, tolerance):
    geoms_negbuff = geoms.buffer(-PRECISION)
    exteriors = get_exterior_ring(geoms.values)
    exteriors_snapped = _snap_linearring(
        exteriors, snap_to, gaps, geoms_negbuff, tolerance
    )

    # looping through max for all geoms since arrays must be equal length
    max_rings = np.max(get_num_interior_rings(geoms))
    interiors = np.array(
        [[get_interior_ring(geom, i) for i in range(max_rings)] for geom in geoms]
    )

    # print(interiors.shape)

    if not interiors.shape[0]:  # len(interiors):
        interiors_snapped = None
    elif interiors.shape[1]:  # len(interiors):
        # print("\n\nhihihihihihi")
        """print(interiors.shape)
        print(
            "\n\nshsassddpp",
            ([rings for rings in interiors]),
        )
        print(
            "\n\nshsassddpp",
            ([rings for rings in interiors.T]),
        )
        print(
            "\n\nshpppp",
            (
                [
                    _snap_linearring(rings, snap_to, gaps, geoms, tolerance)
                    for rings in interiors.T
                ]
            ),
        )"""
        interiors_snapped = np.array(
            [
                _snap_linearring(rings, snap_to, gaps, geoms_negbuff, tolerance)
                for rings in interiors.T
            ]
        ).T
        # print("\ninteriors")
        # print(interiors.shape)
        # print(interiors_snapped.shape)
        """assert interiors_snapped.shape == interiors.shape, (
            interiors_snapped.shape,
            interiors.shape,
        )"""

    else:
        interiors_snapped = None

    """
    try:
        qtm(
            ext=to_gdf(polygons(exteriors_snapped), 25833).clip(mask),
            inters=to_gdf(polygons(interiors_snapped), 25833).clip(mask),
        )
        qtm(
            valid=to_gdf(
                make_valid(polygons(exteriors_snapped, interiors_snapped))[0], 25833
            ),
            ext=to_gdf(make_valid(polygons(exteriors_snapped))[0], 25833),
            inter=to_gdf(make_valid(polygons(interiors_snapped))[0], 25833),
            geoms=to_gdf(geoms.iloc[0], 25833),
            alpha=0.5,
        )
        print("\n\nheidududu")
        print(interiors_snapped)
        # print((polygons(exteriors_snapped, interiors_snapped)[0].wkt))
        # print(make_valid(polygons(exteriors_snapped, interiors_snapped))[0].wkt)
        print(get_num_interior_rings(polygons(exteriors_snapped, interiors_snapped)))
        print(
            get_num_interior_rings(
                make_valid(polygons(exteriors_snapped, interiors_snapped))
            )
        )
        print(polygons(exteriors_snapped, interiors_snapped))
        print(make_valid(polygons(exteriors_snapped, interiors_snapped)))
        print(make_valid(polygons(exteriors_snapped)))

        qtm(
            valid=to_gdf(
                make_valid(polygons(exteriors_snapped, interiors_snapped)), 25833
            ).clip(mask)
        )
    except Exception:
        pass
    """
    return make_valid(polygons(exteriors_snapped, interiors_snapped))


def _snap_linearring2(
    rings: ndarray,
    snap_to: Geometry,
    gaps: GeoSeries,
    tolerance: int | float,
):
    from shapely.wkt import loads

    assert len(rings.shape) == 1, rings.shape
    multipoints = extract_unique_points(rings)

    # get closest point on ring, can be between two vertices
    snap_vertice, ring_points = nearest_points(
        get_parts(extract_unique_points(snap_to)), unary_union(rings)
    )
    ring_vertices = nearest_points(ring_points, unary_union(multipoints))[1]

    distance_to_rings = distance(snap_vertice, ring_points)

    """_, indices = k_nearest_neighbors(
        get_coordinates(ring_points), get_coordinates(multipoints), k=2
    )"""

    # snapmapper: dict[Point, list[Point]] = {}

    points, indices = get_parts(multipoints, return_index=True)

    snap_df = pd.DataFrame(
        {
            "ring_vertices": ring_vertices,
            "snap_vertice": snap_vertice,
            "ring_point": ring_points,
            "distance_to_rings": distance_to_rings,
        }
    ).loc[
        lambda x: (x["distance_to_rings"] < tolerance) & (x["ring_vertices"].notna()),
        ["ring_vertices", "snap_vertice", "ring_point"],
    ]

    snap_df["ring_index"] = snap_df["ring_vertices"].map(
        {p: i for p, i in zip(points, indices)}
    )

    display(snap_df)

    snapmapper = snap_df.set_index("ring_vertices")["snap_vertice"]

    for ring_p, snap_p, dist in zip(ring_vertices, snap_vertice, distance_to_rings):
        if (dist > tolerance) or not ring_p:
            continue

        snapmapper[ring_p] = snapmapper.get(ring_p, [snap_p, ring_p]) + [snap_p]

    """snapmapper = {
    : point_list for, point_list in snapmapper.items() if len(point_list) > 1
    }"""

    snapped_df = pd.DataFrame(
        {"point": [snapmapper.get(x, [x]) for x in points]}, index=indices
    ).explode("point")

    nearest = nearest_points(snapped_df["point"], snap_to)[1]
    distance_to_nearest = distance(snapped_df["point"], nearest)

    distance_to_gap: list[ndarray] = [
        distance(snapped_df["point"], gap) for gap in gaps
    ]
    distance_to_gap: ndarray = np.min(np.array(distance_to_gap), axis=0)

    snapped_df0 = to_gdf(snapped_df["point"], 25833)

    snapped_df["point"] = np.where(
        (distance_to_nearest < tolerance)
        & (distance_to_gap <= PRECISION)  # TODO 0?
        & (distance_to_nearest > 0),
        nearest,
        snapped_df["point"],
    )

    """try:
        qtm(
            rings=to_gdf(rings, 25833).clip(mask),
            snap_to=to_gdf(snap_to, 25833).clip(mask),
            snapped_df0=snapped_df0.clip(mask),
            snapped_df=to_gdf(snapped_df["point"], 25833).clip(mask),
            points=to_gdf(points, 25833)
            .assign(geometry=lambda x: extract_unique_points(x))
            .clip(mask),
        )
        qtm(
            rings=to_gdf(rings, 25833).clip(mask),
            snap_to=to_gdf(snap_to, 25833).clip(mask),
            snap_vertice=to_gdf(snap_vertice, 25833)
            .assign(geometry=lambda x: extract_unique_points(x))
            .clip(mask),
            ring_points=to_gdf(ring_points, 25833)
            .assign(geometry=lambda x: extract_unique_points(x))
            .clip(mask),
            ring_vertices=to_gdf(ring_vertices, 25833).clip(mask),
        )
        qtm(
            rings=to_gdf(rings, 25833).clip(mask),
            snap_to=to_gdf(snap_to, 25833).clip(mask),
            snap_vertice=to_gdf(snap_vertice, 25833)
            .assign(geometry=lambda x: extract_unique_points(x))
            .clip(mask),
            ring_points=to_gdf(ring_points, 25833)
            .assign(geometry=lambda x: extract_unique_points(x))
            .clip(mask),
            ring_vertices=to_gdf(ring_vertices, 25833).clip(mask),
            # snapped_df=to_gdf(snapped_df["point"], 25833).clip(mask),
            snapmapper_keys=to_gdf(
                pd.Series([(x) for x in snapmapper.keys()]), 25833
            ).clip(mask),
            snapmapper_vals=to_gdf(
                pd.Series([x for x in snapmapper.values()]).explode(), 25833
            ).clip(mask),
        )
    except ValueError:
        pass"""

    to_int_index = {idx: i for i, idx in enumerate(sorted(set(snapped_df.index)))}
    snapped_df["_int_idx"] = snapped_df.index.map(to_int_index)

    as_lines = pd.Series(
        linearrings(
            get_coordinates(snapped_df["point"]), indices=snapped_df["_int_idx"]
        ),
        index=snapped_df.index.unique(),
    )

    """for i in range(10):
        try:
            qtm(
                # rings=to_gdf(rings, 25833).clip(mask),
                snap_to=to_gdf(snap_to, 25833).clip(mask),
                snap_vertice=to_gdf(snap_vertice, 25833).clip(mask),
                ring_points=to_gdf(ring_points, 25833).clip(mask),
                ring_vertices=to_gdf(ring_vertices, 25833).clip(mask),
                points=to_gdf(points, 25833)
                .assign(geometry=lambda x: extract_unique_points(x))
                .clip(mask),
                as_lines=to_gdf(as_lines, 25833).clip(mask),
                **{f"p{i}": to_gdf(ring_vertices, 25833).clip(mask).iloc[[i]]},
            )
        except Exception:
            pass
    """

    """try:
        qtm(
            rings=to_gdf(rings, 25833).clip(mask),
            snap_to=to_gdf(snap_to, 25833).clip(mask),
            snap_vertice=to_gdf(snap_vertice, 25833).clip(mask),
            ring_points=to_gdf(ring_points, 25833).clip(mask),
            ring_vertices=to_gdf(ring_vertices, 25833).clip(mask),
            points=to_gdf(points, 25833)
            .assign(geometry=lambda x: extract_unique_points(x))
            .clip(mask),
            as_lines=to_gdf(as_lines, 25833).clip(mask),
            snapmapper_keys=to_gdf(
                pd.Series([(x) for x in snapmapper.keys()]), 25833
            ).clip(mask),
            snapmapper_vals=to_gdf(
                pd.Series([x for x in snapmapper.values()]).explode(), 25833
            ).clip(mask),
        )
    except Exception:
        pass"""

    no_values = pd.Series(
        {i: None for i in range(len(rings)) if i not in as_lines.index}
    )

    """try:
        qtm(
            lines=to_gdf(polygons(as_lines), 25833).clip(mask),
            l=pd.concat([as_lines, no_values]).sort_index().clip(mask),
        )
    except ValueError:
        pass"""

    # explore_locals(explore=False, mask=mask)

    return pd.concat([as_lines, no_values]).sort_index()

    coords = list(get_coordinates(snapped)) + list(no_values)
    indices = list(snapped.index) + list(no_values.index)

    print(snapped)
    print(coords)
    print(indices)
    return linearrings(coords, indices=indices)


def _snap_linearring0(
    rings: ndarray,
    snap_to: Geometry,
    gaps: GeoSeries,
    tolerance: int | float,
):
    # print("rings")
    # print(rings.shape)
    # print(rings)
    assert len(rings.shape) == 1, rings.shape
    multipoints = extract_unique_points(rings)
    points, indices = get_parts(multipoints, return_index=True)

    nearest = nearest_points(points, snap_to)[1]
    distance_to_nearest = distance(points, nearest)

    distance_to_gap: list[ndarray] = [distance(points, gap) for gap in gaps]
    distance_to_gap: ndarray = np.min(np.array(distance_to_gap), axis=0)

    snapped = np.where(
        (distance_to_nearest < tolerance)
        & (distance_to_gap <= PRECISION)  # TODO 0?
        & (distance_to_nearest > 0),
        nearest,
        points,
    )

    snap_vertice, ring_points = nearest_points(
        get_parts(extract_unique_points(snap_to)), unary_union(rings)
    )
    ring_vertices = nearest_points(ring_points, unary_union(multipoints))[1]

    try:
        qtm(
            rings=to_gdf(rings, 25833).clip(mask),
            snap_to=to_gdf(snap_to, 25833).clip(mask),
            points=to_gdf(points, 25833)
            .assign(geometry=lambda x: extract_unique_points(x))
            .clip(mask),
            snapped=to_gdf(snapped, 25833).clip(mask),
        )
        qtm(
            rings=to_gdf(rings, 25833).clip(mask),
            snap_to=to_gdf(snap_to, 25833).clip(mask),
            snapped=to_gdf(snapped, 25833).clip(mask),
            snap_vertice=to_gdf(snap_vertice, 25833)
            .assign(geometry=lambda x: extract_unique_points(x))
            .clip(mask),
            ring_vertices=to_gdf(ring_vertices, 25833).clip(mask),
        )
    except ValueError:
        pass

    distance_to_rings = distance(snap_vertice, ring_points)
    # distance_to_rings[distance_to_rings > tolerance] = None

    snapmapper: dict[str, list[Point]] = {}

    for n, ring_p, snap_p, dist in zip(
        nearest, ring_vertices, snap_vertice, distance_to_rings
    ):
        # if (dist > tolerance) or (dist == 0) or not ring_p:
        if (dist > tolerance) or not ring_p:
            # if not dist or not ring_p:
            continue
        snapmapper[ring_p.wkt] = snapmapper.get(ring_p.wkt, []) + [snap_p]
        snapmapper[n.wkt] = snapmapper.get(ring_p.wkt, []) + [snap_p]

    print(snapmapper)
    print("p1", to_gdf(ring_vertices, 25833).clip(mask).iloc[[1]])
    snapmapper = {
        wkt: point_list for wkt, point_list in snapmapper.items() if len(point_list) > 1
    }
    print(snapmapper)
    snapped_df = pd.DataFrame(
        {"point": [snapmapper.get(x.wkt, [x]) for x in snapped]}, index=indices
    ).explode("point")

    to_int_index = {idx: i for i, idx in enumerate(sorted(set(snapped_df.index)))}
    snapped_df["_int_idx"] = snapped_df.index.map(to_int_index)
    # print(snapped_df)

    as_lines = pd.Series(
        linearrings(
            get_coordinates(snapped_df["point"]), indices=snapped_df["_int_idx"]
        ),
        index=snapped_df.index.unique(),
    )

    for i in range(10):
        try:
            qtm(
                # rings=to_gdf(rings, 25833).clip(mask),
                snap_to=to_gdf(snap_to, 25833).clip(mask),
                snap_vertice=to_gdf(snap_vertice, 25833).clip(mask),
                ring_points=to_gdf(ring_points, 25833).clip(mask),
                ring_vertices=to_gdf(ring_vertices, 25833).clip(mask),
                points=to_gdf(points, 25833)
                .assign(geometry=lambda x: extract_unique_points(x))
                .clip(mask),
                as_lines=to_gdf(as_lines, 25833).clip(mask),
                **{f"p{i}": to_gdf(ring_vertices, 25833).clip(mask).iloc[[i]]},
            )
        except Exception:
            pass
    from shapely.wkt import loads

    try:
        qtm(
            # rings=to_gdf(rings, 25833).clip(mask),
            snap_to=to_gdf(snap_to, 25833).clip(mask),
            snap_vertice=to_gdf(snap_vertice, 25833).clip(mask),
            ring_points=to_gdf(ring_points, 25833).clip(mask),
            ring_vertices=to_gdf(ring_vertices, 25833).clip(mask),
            points=to_gdf(points, 25833)
            .assign(geometry=lambda x: extract_unique_points(x))
            .clip(mask),
            as_lines=to_gdf(as_lines, 25833).clip(mask),
            snapmapper_keys=to_gdf(
                pd.Series([(x) for x in snapmapper.keys()]), 25833
            ).clip(mask),
            snapmapper_vals=to_gdf(
                pd.Series([x for x in snapmapper.values()]).explode(), 25833
            ).clip(mask),
        )
    except Exception:
        pass
    sss

    no_values = pd.Series(
        {i: None for i in range(len(rings)) if i not in as_lines.index}
    )

    """try:
        qtm(
            lines=to_gdf(polygons(as_lines), 25833).clip(mask),
            l=pd.concat([as_lines, no_values]).sort_index().clip(mask),
        )
    except ValueError:
        pass"""

    # explore_locals(explore=False, mask=mask)

    return pd.concat([as_lines, no_values]).sort_index()

    coords = list(get_coordinates(snapped)) + list(no_values)
    indices = list(snapped.index) + list(no_values.index)

    print(snapped)
    print(coords)
    print(indices)
    return linearrings(coords, indices=indices)


def _snap_linearring0(
    rings: ndarray,
    snap_to: Geometry,
    gaps: GeoSeries,
    tolerance: int | float,
):
    from shapely.wkt import loads

    assert len(rings.shape) == 1, rings.shape
    multipoints = extract_unique_points(rings)

    # get closest point on ring, can be between two vertices
    snap_vertice, ring_points = nearest_points(
        get_parts(extract_unique_points(snap_to)), unary_union(rings)
    )
    ring_vertices = nearest_points(ring_points, unary_union(multipoints))[1]

    distance_to_rings = distance(snap_vertice, ring_points)

    """_, indices = k_nearest_neighbors(
        get_coordinates(ring_points), get_coordinates(multipoints), k=2
    )"""

    # snapmapper: dict[Point, list[Point]] = {}

    points, indices = get_parts(multipoints, return_index=True)

    snap_df = pd.DataFrame(
        {
            "ring_vertices": ring_vertices,
            "snap_vertice": snap_vertice,
            "ring_point": ring_points,
            "distance_to_rings": distance_to_rings,
        }
    ).loc[
        lambda x: (x["distance_to_rings"] < tolerance) & (x["ring_vertices"].notna()),
        ["ring_vertices", "snap_vertice", "ring_point"],
    ]

    snap_df["ring_index"] = snap_df["ring_vertices"].map(
        {p: i for p, i in zip(points, indices)}
    )

    display(snap_df)

    snapmapper = snap_df.set_index("ring_vertices")["snap_vertice"]

    for ring_p, snap_p, dist in zip(ring_vertices, snap_vertice, distance_to_rings):
        if (dist > tolerance) or not ring_p:
            continue

        snapmapper[ring_p] = snapmapper.get(ring_p, [snap_p, ring_p]) + [snap_p]

    """snapmapper = {
    : point_list for, point_list in snapmapper.items() if len(point_list) > 1
    }"""

    snapped_df = pd.DataFrame(
        {"point": [snapmapper.get(x, [x]) for x in points]}, index=indices
    ).explode("point")

    nearest = nearest_points(snapped_df["point"], snap_to)[1]
    distance_to_nearest = distance(snapped_df["point"], nearest)

    distance_to_gap: list[ndarray] = [
        distance(snapped_df["point"], gap) for gap in gaps
    ]
    distance_to_gap: ndarray = np.min(np.array(distance_to_gap), axis=0)

    snapped_df0 = to_gdf(snapped_df["point"], 25833)

    snapped_df["point"] = np.where(
        (distance_to_nearest < tolerance)
        & (distance_to_gap <= PRECISION)  # TODO 0?
        & (distance_to_nearest > 0),
        nearest,
        snapped_df["point"],
    )

    try:
        qtm(
            rings=to_gdf(rings, 25833).clip(mask),
            snap_to=to_gdf(snap_to, 25833).clip(mask),
            snapped_df0=snapped_df0.clip(mask),
            snapped_df=to_gdf(snapped_df["point"], 25833).clip(mask),
            points=to_gdf(points, 25833)
            .assign(geometry=lambda x: extract_unique_points(x))
            .clip(mask),
        )
        qtm(
            rings=to_gdf(rings, 25833).clip(mask),
            snap_to=to_gdf(snap_to, 25833).clip(mask),
            snap_vertice=to_gdf(snap_vertice, 25833)
            .assign(geometry=lambda x: extract_unique_points(x))
            .clip(mask),
            ring_points=to_gdf(ring_points, 25833)
            .assign(geometry=lambda x: extract_unique_points(x))
            .clip(mask),
            ring_vertices=to_gdf(ring_vertices, 25833).clip(mask),
        )
        qtm(
            rings=to_gdf(rings, 25833).clip(mask),
            snap_to=to_gdf(snap_to, 25833).clip(mask),
            snap_vertice=to_gdf(snap_vertice, 25833)
            .assign(geometry=lambda x: extract_unique_points(x))
            .clip(mask),
            ring_points=to_gdf(ring_points, 25833)
            .assign(geometry=lambda x: extract_unique_points(x))
            .clip(mask),
            ring_vertices=to_gdf(ring_vertices, 25833).clip(mask),
            # snapped_df=to_gdf(snapped_df["point"], 25833).clip(mask),
            snapmapper_keys=to_gdf(
                pd.Series([(x) for x in snapmapper.keys()]), 25833
            ).clip(mask),
            snapmapper_vals=to_gdf(
                pd.Series([x for x in snapmapper.values()]).explode(), 25833
            ).clip(mask),
        )
    except ValueError:
        pass

    to_int_index = {idx: i for i, idx in enumerate(sorted(set(snapped_df.index)))}
    snapped_df["_int_idx"] = snapped_df.index.map(to_int_index)

    as_lines = pd.Series(
        linearrings(
            get_coordinates(snapped_df["point"]), indices=snapped_df["_int_idx"]
        ),
        index=snapped_df.index.unique(),
    )

    """for i in range(10):
        try:
            qtm(
                # rings=to_gdf(rings, 25833).clip(mask),
                snap_to=to_gdf(snap_to, 25833).clip(mask),
                snap_vertice=to_gdf(snap_vertice, 25833).clip(mask),
                ring_points=to_gdf(ring_points, 25833).clip(mask),
                ring_vertices=to_gdf(ring_vertices, 25833).clip(mask),
                points=to_gdf(points, 25833)
                .assign(geometry=lambda x: extract_unique_points(x))
                .clip(mask),
                as_lines=to_gdf(as_lines, 25833).clip(mask),
                **{f"p{i}": to_gdf(ring_vertices, 25833).clip(mask).iloc[[i]]},
            )
        except Exception:
            pass
    """

    try:
        qtm(
            rings=to_gdf(rings, 25833).clip(mask),
            snap_to=to_gdf(snap_to, 25833).clip(mask),
            snap_vertice=to_gdf(snap_vertice, 25833).clip(mask),
            ring_points=to_gdf(ring_points, 25833).clip(mask),
            ring_vertices=to_gdf(ring_vertices, 25833).clip(mask),
            points=to_gdf(points, 25833)
            .assign(geometry=lambda x: extract_unique_points(x))
            .clip(mask),
            as_lines=to_gdf(as_lines, 25833).clip(mask),
            snapmapper_keys=to_gdf(
                pd.Series([(x) for x in snapmapper.keys()]), 25833
            ).clip(mask),
            snapmapper_vals=to_gdf(
                pd.Series([x for x in snapmapper.values()]).explode(), 25833
            ).clip(mask),
        )
    except Exception:
        pass

    no_values = pd.Series(
        {i: None for i in range(len(rings)) if i not in as_lines.index}
    )

    """try:
        qtm(
            lines=to_gdf(polygons(as_lines), 25833).clip(mask),
            l=pd.concat([as_lines, no_values]).sort_index().clip(mask),
        )
    except ValueError:
        pass"""

    # explore_locals(explore=False, mask=mask)

    return pd.concat([as_lines, no_values]).sort_index()

    coords = list(get_coordinates(snapped)) + list(no_values)
    indices = list(snapped.index) + list(no_values.index)

    print(snapped)
    print(coords)
    print(indices)
    return linearrings(coords, indices=indices)


def make_snapmapper(
    rings: ndarray, snap_to: Geometry, multipoints: ndarray, tolerance: float | int
) -> pd.DataFrame:
    """Make points along the rings between the vertices.

    Can be more than one between two vertices.

    Returns:
        Series with index of ring vertices ('point' column of df) and
            aggregated values (multipoints) of vertices in 'snap_to'.
    """
    snap_vertices, ring_points = nearest_points(
        get_parts(extract_unique_points(snap_to)), unary_union(rings)
    )
    ring_vertices = nearest_points(ring_points, unary_union(multipoints))[1]
    distance_to_snap_to = distance(snap_vertices, ring_points)
    distance_to_ring_vertice = distance(ring_vertices, ring_points)

    return pd.DataFrame(
        {
            "ring_point": ring_points,
            "snap_vertice": snap_vertices,
            "distance_to_snap_to": distance_to_snap_to,
            "distance_to_ring_vertice": distance_to_ring_vertice,
        },
        index=GeometryArray(ring_vertices),
    ).loc[
        lambda x: (x["distance_to_snap_to"] <= tolerance)
        & (x["distance_to_ring_vertice"] > 0)
        & (x["ring_point"].notna()),
        ["ring_point"],
    ]

    add = pd.DataFrame(
        {"ring_point": ring_vertices}, index=GeometryArray(ring_vertices)
    ).loc[lambda x: x.index.isin(snapmapper.index)]

    return pd.concat([snapmapper, add])

    def sorted_unary_union(points: pd.Series) -> MultiPoint | Point:
        return unary_union(MultiPoint(np.sort(get_coordinates(points.values), axis=0)))

    return pd.concat([snapmapper, add]).groupby(level=0).agg(sorted_unary_union)


def make_lines_between_points(arr1: ndarray, arr2: ndarray) -> ndarray:
    print(arr1)
    print(arr2)
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have equal shape.")
    coords: pd.DataFrame = pd.concat(
        [
            pd.DataFrame(get_coordinates(arr1), columns=["x", "y"]),
            pd.DataFrame(get_coordinates(arr2), columns=["x", "y"]),
        ]
    ).sort_index()

    # coords = get_coordinates(np.concatenate([arr1, arr2]))
    # indices = np.concatenate([np.arange(0, len(arr1)), np.arange(0, len(arr2))])
    return linestrings(coords.values, indices=coords.index)


def map_to_nearest(points, snap_to, gaps, geoms, tolerance):
    distance_to_gap: list[ndarray] = [distance(points, gap) for gap in gaps]
    distance_to_gap: ndarray = np.min(np.array(distance_to_gap), axis=0)

    nearest_vertice = nearest_points(points, extract_unique_points(snap_to))[1]
    distance_to_nearest_vertice = distance(points, nearest_vertice)

    as_lines = make_lines_between_points(points, nearest_vertice)
    intersect_geoms: list[ndarray] = [intersects(as_lines, geom) for geom in geoms]
    intersect_geoms: ndarray = np.any(np.array(intersect_geoms), axis=0)

    nearest = nearest_points(points, snap_to)[1]
    distance_to_nearest = distance(points, nearest)

    as_lines = make_lines_between_points(points, nearest)
    intersect_geoms2: list[ndarray] = [intersects(as_lines, geom) for geom in geoms]
    intersect_geoms2: ndarray = np.any(np.array(intersect_geoms2), axis=0)

    snap_to_vertice = (
        (distance_to_nearest_vertice <= tolerance)
        & (distance_to_gap <= PRECISION)  # TODO 0?
        & (distance_to_nearest_vertice > 0)
        & (~intersect_geoms)
    )

    snap_to_point = (
        (distance_to_nearest_vertice > tolerance)
        & (distance_to_nearest <= tolerance)
        & (distance_to_gap <= PRECISION)  # TODO 0?
        & (distance_to_nearest > 0)
        & (~intersect_geoms2)
    )

    conditions = [
        snap_to_vertice,
        snap_to_point,
    ]

    choices = [nearest_vertice, nearest]
    return np.select(conditions, choices, default=points)

    snapped_to_vertice = np.where(
        (distance_to_nearest_vertice <= tolerance)
        & (distance_to_gap <= PRECISION)  # TODO 0?
        & (distance_to_nearest_vertice > 0),
        nearest_vertice,
        points,
    )

    return np.where(
        (distance_to_nearest <= tolerance)
        & (distance_to_nearest_vertice > tolerance)
        & (distance_to_gap <= PRECISION)  # TODO 0?
        & (distance_to_nearest > 0),
        nearest,
        snapped_to_vertice,
    )


def sorted_unary_union2_(df: pd.DataFrame) -> MultiPoint | Point:
    print("heiheiheih df")
    display(df)
    # these are identical for all rows
    this: Point = df["this"].iloc[0]
    next_: Point = df["next"].iloc[0]
    prev: Point = df["prev"].iloc[0]

    # TODO prev = np.nan # pd.NA # None

    prev_to_this = LineString([prev, this])
    this_to_next = LineString([this, next_])

    sorted_ring_points = points(
        np.sort(get_coordinates(df["ring_point"].values), axis=0)
    )

    qtm(
        sorted_ring_points=to_gdf(sorted_ring_points, 25833),
        this=to_gdf(this, 25833),
        prev=to_gdf(prev, 25833),
        next_=to_gdf(next_, 25833),
        prev_to_this=to_gdf(prev_to_this, 25833),
        this_to_next=to_gdf(this_to_next, 25833),
        title=prev.wkt[-10:]
        + "\n"
        + next_.wkt[-10:]
        + "\n"
        + str(len(sorted_ring_points)),
    )
    between_prev_and_this = intersection(sorted_ring_points, prev_to_this)
    between_prev_and_this = between_prev_and_this[~is_empty(between_prev_and_this)]

    between_this_and_next = intersection(sorted_ring_points, this_to_next)
    between_this_and_next = between_this_and_next[~is_empty(between_this_and_next)]

    print("sorted_ring_points")
    print(sorted_ring_points)
    print("between_prev_and_this")
    print(between_prev_and_this)
    print("between_this_and_next")
    print(between_this_and_next)

    return MultiPoint(list(between_prev_and_this) + list(between_this_and_next))

    sorted_ring_points = np.sort(get_coordinates(df["ring_point"].values), axis=0)

    option1 = [prev] + list(sorted_ring_points) + [next_]
    option2 = [next_] + list(sorted_ring_points) + [prev]

    """qtm(
        sorted_ring_points=to_gdf(sorted_ring_points, 25833)
        .clip(mask)
        .assign(idx=lambda x: range(len(x))),
        column="idx",
    )"""
    """qtm(
        **{
            f"o2_{len(sorted_ring_points)}": to_gdf(LineString(option2), 25833).clip(
                mask
            ),
            f"o1_{len(sorted_ring_points)}": to_gdf(LineString(option1), 25833).clip(
                mask
            ),
        },
        sorted_ring_points=to_gdf(sorted_ring_points, 25833).clip(mask),
        this=to_gdf(this, 25833).clip(mask),
        next_=to_gdf(next_, 25833).clip(mask),
        prev=to_gdf(prev, 25833).clip(mask),
        title="prev_" + prev.wkt + "\n" + "next_" + next_.wkt,
    )"""

    if LineString(option1).length > LineString(option2).length:
        qtm(
            **{
                f"o1_{len(sorted_ring_points)}": to_gdf(LineString(option1), 25833),
            },
            sorted_ring_points=to_gdf(sorted_ring_points, 25833),
            this=to_gdf(this, 25833),
            next_=to_gdf(next_, 25833),
            prev=to_gdf(prev, 25833),
            title=prev.wkt[:10] + "\n" + next_.wkt[:10],
        )
        return MultiPoint(option1)
    else:
        qtm(
            **{
                f"o2_{len(sorted_ring_points)}": to_gdf(LineString(option2), 25833),
            },
            sorted_ring_points=to_gdf(sorted_ring_points, 25833),
            this=to_gdf(this, 25833),
            next_=to_gdf(next_, 25833),
            prev=to_gdf(prev, 25833),
            title=prev.wkt[:10] + "\n" + next_.wkt[:10],
        )
        return MultiPoint(option2)

    print("this")
    print(this)

    print("sorted_ring_points")
    print(sorted_ring_points)

    where_is_this = np.where(sorted_ring_points == this)[0][0]
    this_is_first = where_is_this == 0
    this_is_last = where_is_this == len(sorted_ring_points)
    print("where_is_this")
    print(where_is_this, this_is_first, this_is_last)

    # hvis prev er dritlangt unna, har det ingenting å si hva som er nærmest
    # print("next_")
    # print(next_)
    # print("prev")
    # print(prev)

    # this will make a (1,) shaped array to (2,)
    first_and_last = sorted_ring_points[np.array([0, -1])]
    # print("first_and_last")
    # print(first_and_last)

    dist_to_prev = distance(first_and_last, prev)
    dist_to_next = distance(first_and_last, next_)
    dist_to_this = distance(first_and_last, this)

    first_is_closer_to_prev = dist_to_prev[0] < dist_to_prev[1]
    first_is_closer_to_next = dist_to_next[0] < dist_to_next[1]

    # print("first_is_closer_to_prev")
    # print(first_is_closer_to_prev)
    # print("first_is_closer_to_next")
    # print(first_is_closer_to_next)

    if first_is_closer_to_prev and not first_is_closer_to_next:
        print("first_is_closer_to_prev")
        return MultiPoint([prev] + list(sorted_ring_points) + [next_])
    if first_is_closer_to_next and not first_is_closer_to_prev:
        print("first_is_closer_to_next")
        return MultiPoint([next_] + list(sorted_ring_points) + [prev])
    else:
        print("neither....")
        return MultiPoint(list(sorted_ring_points))  # TODO

    if (
        (np.min(dist_to_next) > np.min(dist_to_this))
        and (  # closer to this
            np.min(dist_to_next) > np.min(dist_to_prev)
        )  # closer to prev
        # so furthest to next, but what if next is very long...
        and (dist_to_prev[0] > dist_to_next[0])  # closer to prev...
        and (dist_to_prev[0] < dist_to_this[0])  # closer to this than prev
    ):
        return MultiPoint([prev] + list(sorted_ring_points) + [this])

    if (
        (dist_to_prev[0] > dist_to_prev[1])  # first is closer to next than last
        and (dist_to_prev[0] > dist_to_next[0])  # closer to prev...
        and (dist_to_prev[0] < dist_to_this[0])  # closer to this than prev
    ):
        return MultiPoint([prev] + list(sorted_ring_points) + [this])

    if (dist_to_prev[0] < dist_to_prev[1]) and (dist_to_prev[0] < dist_to_next[0]):
        "første punkt er nærmere prev enn next og nærmere prev enn siste punkt"


def sorted_unary_union(points: pd.Series) -> MultiPoint | Point | None:
    values: ndarray = points.dropna().values
    if not len(values):
        return None
    if len(values) == 1:
        return values[0]  # Point

    return MultiPoint(np.sort(get_coordinates(values), axis=0))


def sorted_unary_union(df: pd.DataFrame) -> MultiPoint:
    assert len(df["endpoints"].unique()) <= 1, df["endpoints"].unique()
    assert len(df["point"].unique()) <= 1, df["point"].unique()
    assert len(df["geometry"].unique()) <= 1, df["geometry"].unique()

    """endpoints =
    if isinstance(endpoints, MultiPoint):
        assert isinstance(endpoints[0], Point), endpoints
        assert isinstance(endpoints[1], Point), endpoints
    # elif isinstance(endpoints, Point):
    #     endpoints = np.array([endpoints])
    else:
        raise ValueError(endpoints)"""
    """try:
        endpoints: list[Point] = list(df["endpoints"].iloc[0].geoms)
    except AttributeError:
        endpoints: list[Point] = [df["endpoints"].iloc[0]]
    """

    endpoints: ndarray = get_coordinates(df["endpoints"].iloc[0])
    between: ndarray = get_coordinates(df["ring_point"].dropna().values)

    coords = np.concatenate([endpoints, between])
    sorted_coords = coords[np.argsort(coords[:, -1])]

    # droping points outside the line (returned from sjoin because of buffer)
    is_between_endpoints = (sorted_coords[:, 0] >= np.min(endpoints[:, 0])) & (
        sorted_coords[:, 0] <= np.max(endpoints[:, 0])
    )
    sorted_coords = sorted_coords[is_between_endpoints]

    """assert (
        Point(sorted_coords[0]) == df["point"].iloc[0]
        or Point(sorted_coords[-1]) == df["point"].iloc[0]
    ), (sorted_coords, df["point"].iloc[0])"""

    """qtm(
        endpoints=to_gdf(endpoints, 25833).clip(mask.buffer(25)),
        between=to_gdf(between, 25833).clip(mask.buffer(25)),
        sorted_coords=to_gdf(sorted_coords, 25833).clip(mask.buffer(25)),
        line=to_gdf(LineString(sorted_coords), 25833).clip(mask.buffer(25)),
    )"""

    return LineString(sorted_coords)
    return MultiPoint(sorted_coords)

    return MultiPoint(np.sort(get_coordinates(endpoints + between), axis=0))


def _snap_linearring(
    rings: ndarray,
    snap_to: Geometry,
    gaps: GeoSeries,
    geoms: GeoSeries,
    tolerance: int | float,
) -> pd.Series:
    assert len(rings.shape) == 1, "ring array should be 1 dimensional"

    multipoints = extract_unique_points(rings)
    points, indices = get_parts(multipoints, return_index=True)

    if not len(points):
        return pd.Series()

    snap_df = pd.DataFrame({"point": GeometryArray(points)}, index=indices)

    snap_df["next"] = snap_df.groupby(level=0)["point"].shift(-1)

    first_points = snap_df.loc[lambda x: ~x.index.duplicated(), "point"]
    is_last_point = snap_df["next"].isna()

    try:
        qtm(
            rings=to_gdf(rings).clip(mask.buffer(55)),
            first=to_gdf(first_points).clip(mask.buffer(55)),
            last=to_gdf(snap_df.loc[is_last_point, "point"]).clip(mask.buffer(55)),
        )
    except Exception:
        pass
    snap_df.loc[is_last_point, "next"] = first_points
    assert snap_df["next"].notna().all()

    snap_df.index.name = "ring_index"
    snap_df = snap_df.reset_index()

    ring_points: pd.DataFrame = make_snapmapper(rings, snap_to, multipoints, tolerance)

    not_last = snap_df["next"].notna()
    lines = snap_df.loc[not_last]
    # not_lines = snap_df.loc[~not_last]
    # lines = snap_df.copy()  # TODO temp

    """lines.loc[not_last, "geometry"] = [
        LineString([x1, x2])
        for x1, x2 in zip(
            lines.loc[not_last, "point"],
            lines.loc[not_last, "next"],
        )
    ]"""
    lines["geometry"] = [
        LineString([x1, x2]) for x1, x2 in zip(lines["point"], lines["next"])
    ]
    lines = GeoDataFrame(lines, geometry="geometry")
    display(lines)
    """
    lines = lines.sjoin_nearest(
        GeoDataFrame(
            ring_points.assign(ring_point2=lambda x: x["ring_point"]),
            geometry="ring_point2",
        ),
        max_distance=PRECISION,
    )  # .loc[lambda x: x["index_right"] == x["point"]]
    """
    lines = lines.sjoin(
        GeoDataFrame(
            ring_points.assign(ring_point2=lambda x: x["ring_point"]),
            geometry="ring_point2",
        ).pipe(buff, PRECISION),
        how="left",
    )
    display("lines joined with points")
    display(lines)
    display(lines[lines.ring_point.notna()])

    qtm(lines=lines.clip(mask.buffer(24)), title="nesten øverst")

    # TODO as coord tuples?
    lines["endpoints"] = lines.geometry.boundary  # np.where(
    #     lines["geometry"].notna(),
    #     lines["geometry"].boundary,
    #     lines["point"],
    # )

    """
    for i in lines.sample(50).index:
        qtm(
            ring_point=to_gdf(lines.loc[[i]].ring_point),
            index_right=to_gdf(lines.loc[[i]].index_right),
            geometry=to_gdf(lines.loc[[i]].geometry),
            point=to_gdf(lines.loc[[i]].point),
            next=to_gdf(lines.loc[[i]].next),
            endpoints=to_gdf(lines.loc[[i]].endpoints),
        )
    sss"""

    # lines["ring_index"] = lines.index

    print("len snap_df")
    print(len(snap_df))
    print(len(lines))

    # if len(lines):
    agged = lines.groupby(level=0).apply(sorted_unary_union)
    lines = lines.loc[lambda x: ~x.index.duplicated()]
    lines.geometry = agged

    qtm(
        lines=to_gdf(lines.geometry, 25833).clip(mask.buffer(0.5)),
        geoms=to_gdf(geoms, 25833).clip(mask.buffer(0.5)),
        title="lines",
    )

    lines = sfilter_inverse(lines, geoms)

    qtm(
        lines=to_gdf(lines.geometry, 25833).clip(mask.buffer(0.5)),
        geoms=to_gdf(geoms, 25833).clip(mask.buffer(0.5)),
        title="lines2",
    )

    display("disssss")

    # lines = lines.dissolve(by="ring_index", as_index=False)
    lines = lines.groupby("ring_index", as_index=False)["geometry"].agg(unary_union)
    lines.geometry = line_merge(lines.geometry)

    not_merged = lines.geom_type == "MultiLineString"
    lines.loc[not_merged, "geometry"] = lines.loc[not_merged, "geometry"].apply(
        line_merge_by_force
    )

    assert (lines.geom_type == "LineString").all()

    display(lines.geometry)
    """for i, g in lines.geometry.items():
        g = to_gdf(g, 25833)
        # if i == 2:
        #    explore(g.explode(index_parts=False).assign(idx=lambda x: [str(y) for y in range(len(x))]), "idx", max_zoom=60)
        for g2 in g.explode(index_parts=False).geometry:
            if i == 2:
                print(g2.wkt)
            qtm(
                g=g.clip(to_gdf(g2, 25833).buffer(0.1)),
                g2=to_gdf(g2, 25833),
                title=str(i),
                alpha=0.7,
            )"""

    as_points = extract_unique_points(lines.geometry.values)
    qtm(
        lines212=(lines),
    )
    qtm(
        pointsw232=to_gdf(as_points, 25833),
    )
    siste, indices = get_parts(
        extract_unique_points(lines.geometry.values), return_index=True
    )
    siste = to_gdf(
        linearrings(get_coordinates(siste), indices=indices),
        25833,
    )
    print(as_points)
    print(siste)
    """for (i, g), p in zip(lines.clip(mask).geometry.items(), as_points):
        print(i, g)
        print(p)
        qtm(
            g=to_gdf(g, 25833).clip(mask.buffer(20)),
            p1=to_gdf(extract_unique_points(g), 25833).clip(mask.buffer(20)),
            p=to_gdf(p, 25833).clip(mask.buffer(20)),
            pointsw232=to_gdf(as_points, 25833).clip(mask.buffer(20)),
            lsss=(lines).clip(mask.buffer(20)),
            siste=siste,
            title=str(i),
        )
    """

    lines.geometry = extract_unique_points(lines.geometry.values)
    lines = lines.explode(index_parts=False)
    display(lines)

    try:
        qtm(
            lines=to_gdf(
                linearrings(
                    get_coordinates(lines.geometry.values),
                    indices=lines.ring_index.values,
                ),
                25833,
            ).clip(mask.buffer(0.5)),
            geoms=to_gdf(geoms, 25833).clip(mask.buffer(0.5)),
            title="lines3",
        )
    except Exception:
        pass

    """
    lines["grouper"] = [
        f"{idx}_{p.wkt}" for idx, p in zip(lines["ring_index"], lines["point"])
    ]
    sorted_points: pd.Series = (
        lines.groupby("grouper").apply(sorted_unary_union)
        # .droplevel(1)
        # .to_frame(name="point")
        # .reset_index(level=1)
    )
    assert len(sorted_points) == len(lines)

    print("sorted_points")
    print("sorted_points")
    print("sorted_points")
    print(sorted_points)
    assert isinstance(sorted_points, pd.Series), type(sorted_points)
    lines["point"] = lines["grouper"].map(sorted_points)
    print(len(lines))
    display((lines))
    """

    print("\n\nheihei")

    """snap_df = (
        pd.concat([lines, snap_df]).loc[lambda x: ~x.index.duplicated()].sort_index()
    )
    display(snap_df)"""
    snap_df = lines
    display(snap_df)

    """
    joined = pd.concat(
        [
            joined,
            joined[["point"]].rename(columns={"point": "ring_point"})[["ring_point"]],
        ]
    )
    display(joined)

    for i in range(50):
        v = joined.iloc[[i]].point.iloc[0]
        qtm(
            line=to_gdf(joined.loc[joined.point == v, "geometry"], 25833),
            ring_point=to_gdf(joined.loc[joined.point == v, "ring_point"], 25833),
            point=to_gdf(joined.loc[joined.point == v, "point"], 25833),
        )
    sssssss
    """
    """joined["point"] = GeometryArray(
        np.where(
            joined["ring_point"].notna(), joined["ring_point"], joined["point"]
        )
    )"""
    """
    display(joined)

    joined["ring_index"] = joined.index

    snap_df: pd.DataFrame = (
        joined.groupby(["ring_index", "point"])["ring_point"]
        .agg(sorted_unary_union)
        .reset_index(level=1)
    )
    snap_df["point"] = np.where(
        snap_df["ring_point"].notna(), snap_df["ring_point"], snap_df["point"]
    )
    display(snap_df)
    """
    """
    ssss

    add = pd.DataFrame(
        {
            "this": snap_df["point"].values,
            "prev": snap_df["prev"].values,
            "next": snap_df["prev"].values,
        },
        index=GeometryArray(snap_df["point"].values),
    ).loc[lambda x: (x.index.isin(ring_points.index)) & (~x.index.duplicated())]

    add = snap_df.loc[
        lambda x: (x["point"].isin(ring_points.index)), ["point", "prev", "next"]
    ].merge(ring_points, left_on="point", right_index=True)
    display(add)

    add["ring_index"] = add.index

    snapmapper: pd.Series = add.groupby(["ring_index", "point"]).apply(sorted_unary_union)

    ssss
    display("snap_df")
    display(snap_df)
    display("add")
    display(add)
    display("ring_points")
    display(ring_points)
    print("joined")
    display(add.join(ring_points, how="inner"))
    ssss

    snapmapper: pd.Series = (
        add.join(ring_points, how="inner").groupby(level=0).apply(sorted_unary_union)
    )"""

    try:
        qtm(
            gaps=to_gdf(gaps).clip(mask),
            rings=to_gdf(rings).clip(mask),
            snap_to=to_gdf(snap_to).clip(mask),
            points=to_gdf(points).clip(mask),
            ring_points=to_gdf(ring_points["ring_point"]).clip(mask),
            snap_df=to_gdf(snap_df["geometry"]).clip(mask),
            # xxx=to_gdf(snapmapper).clip(mask),
            title="hernaaa",
        )
        """for i in snapmapper.clip(mask).index.unique():
            print(i)
            qtm(
                snap_to=to_gdf(snap_to).clip(mask),
                **{i.wkt: to_gdf(snapmapper.loc[snapmapper.index == i]).clip(mask)},
            )"""
    except Exception:
        pass

    # snap_df["snap_vertice"] = snap_df["point"].map(snapmapper["snap_vertice"])
    # snap_df["distance_to_ring"] = snap_df["point"].map(snapmapper["distance_to_ring"])

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
        lines=to_gdf(lines["geometry"], 25833).clip(mask),
        linepoints=to_gdf(extract_unique_points(lines["geometry"].values), 25833).clip(
            mask
        ),
        title="geometry",
    )
    snap_df["geometry"] = map_to_nearest(
        snap_df["geometry"], snap_to, gaps, geoms, tolerance
    )

    # remove lines crossing the geometries
    # snap_df["next"] = snap_df.groupby(level=0)["geometry"].shift(-1)
    # snap_df = sfilter_inverse(snap_df, geoms)

    # snap_df = snap_df.drop_duplicates(["ring_index", "point"])

    """snap_df = pd.Series(
        map_to_nearest(snap_df, snap_to, gaps, geoms, tolerance), index=snap_df.index
    )"""
    """
    display(snap_df)

    for idx in snap_df.index.unique():
        qtm(
            snap=to_gdf(snap_df.loc[snap_df.index == idx, "point"], 25833),
            cp222=to_gdf(snap_df_copy.loc[snap_df_copy.index == idx, "point"], 25833),
        )

    display(snap_df)
    """
    """
    ssss
    snap_df = pd.DataFrame(
        {
            "ring_vertices": ring_vertices,
            "snap_vertice": snap_vertice,
            "ring_point": ring_points,
            "distance_to_rings": distance_to_rings,
        }
    ).loc[
        lambda x: (x["distance_to_rings"] < tolerance) & (x["ring_vertices"].notna()),
        ["ring_vertices", "snap_vertice", "ring_point"],
    ]

    snapped = np.where(
        (distance_to_nearest < tolerance)
        & (distance_to_gap <= PRECISION)  # TODO 0?
        & (distance_to_nearest > 0),
        nearest,
        points,
    )
    """

    try:
        qtm(
            rings=to_gdf(rings, 25833).clip(mask),
            ring_points=to_gdf(ring_points, 25833).clip(mask),
            snap_to=to_gdf(snap_to, 25833).clip(mask),
            snap_df=to_gdf(snap_df["geometry"], 25833).clip(mask),
            title="denne",
        )
    except Exception:
        pass

    if "nx" == 1:
        snap_df = pd.DataFrame(
            {
                "point": snap_df.groupby("ring_index")["point"].apply(
                    sort_points_to_line
                ),
                "ring_index": snap_df["ring_index"].unique(),
            },
            index=snap_df["ring_index"].unique(),
        ).explode("point")

    """as_lines = pd.Series(
        linearrings(
            # get_coordinates(snap_df.values),
            get_coordinates(snap_df["point"].values),
            # indices=snap_df["_int_idx"].values,
            indices=snap_df["ring_index"].map(to_int_index).values,
        ),
        index=snap_df["ring_index"].unique(),
    )"""

    if 1:
        to_int_index = {
            idx: i for i, idx in enumerate(sorted(set(snap_df["ring_index"])))
        }
        int_indices = snap_df["ring_index"].map(to_int_index)
        as_lines = pd.Series(
            linearrings(
                # get_coordinates(snap_df.values),
                get_coordinates(snap_df["geometry"].values),
                # indices=snap_df["_int_idx"].values,
                indices=int_indices.values,
            ),
            index=int_indices.unique(),
        )

    """qtm(
        rings=to_gdf(rings, 25833).clip(mask),
        snap_to=to_gdf(snap_to, 25833).clip(mask),
        ring_points=to_gdf(ring_points, 25833).clip(mask),
        as_lines=to_gdf(as_lines, 25833).clip(mask),
        title="denne",
    )"""

    """
    as_lines = pd.Series(
        linestrings(
            get_coordinates(snap_df["point"].values),
            indices=snap_df["_int_idx"].values,
        ),
        index=snap_df.index.unique(),
    )

    qtm(
        points=to_gdf(snap_df["point"], 25833)
        .clip(mask)
        .assign(idx=lambda x: range(len(x)) ** x.index),
        column="idx",
    )
    qtm(
        as_lines=to_gdf(as_lines, 25833).clip(mask),
        points=to_gdf(snap_df["point"], 25833).clip(mask),
    )

    as_lines = line_merge(as_lines)
    qtm(as_lines=to_gdf(as_lines, 25833).clip(mask))
    as_lines = pd.Series(
        linearrings(get_coordinates(as_lines.values)), index=as_lines.index
    )
    qtm(as_lines=to_gdf(as_lines, 25833).clip(mask))
    """
    """
    for i in range(10):
        try:
            qtm(
                # rings=to_gdf(rings, 25833).clip(mask),
                snap_to=to_gdf(snap_to, 25833).clip(mask),
                snap_vertice=to_gdf(snap_vertice, 25833).clip(mask),
                ring_points=to_gdf(ring_points, 25833).clip(mask),
                ring_vertices=to_gdf(ring_vertices, 25833).clip(mask),
                points=to_gdf(points, 25833)
                .assign(geometry=lambda x: extract_unique_points(x))
                .clip(mask),
                as_lines=to_gdf(as_lines, 25833).clip(mask),
                **{f"p{i}": to_gdf(ring_vertices, 25833).clip(mask).iloc[[i]]},
            )
        except Exception:
            pass
    from shapely.wkt import loads
    """

    try:
        qtm(
            # rings=to_gdf(rings, 25833).clip(mask),
            geoms=to_gdf(geoms, 25833).clip(mask),
            snap_to=to_gdf(snap_to, 25833).clip(mask),
            # snap_vertice=to_gdf(snap_vertices, 25833).clip(mask),
            # ring_points=to_gdf(ring_points, 25833).clip(mask),
            # ring_vertices=to_gdf(ring_vertices, 25833).clip(mask),
            points=to_gdf(points, 25833)
            .assign(geometry=lambda x: extract_unique_points(x))
            .clip(mask),
            snap_df=to_gdf(snap_df["geometry"], 25833).clip(mask),
            as_lines=to_gdf(as_lines, 25833).clip(mask),
            title="helt nederst",
        )
        qtm(
            # rings=to_gdf(rings, 25833).clip(mask),
            geoms=to_gdf(geoms, 25833).clip(mask.buffer(11)),
            snap_to=to_gdf(snap_to, 25833).clip(mask.buffer(11)),
            # snap_vertice=to_gdf(snap_vertices, 25833).clip(mask.buffer(11)),
            # ring_points=to_gdf(ring_points, 25833).clip(mask.buffer(11)),
            # ring_vertices=to_gdf(ring_vertices, 25833).clip(mask.buffer(11)),
            points=to_gdf(points, 25833)
            .assign(geometry=lambda x: extract_unique_points(x))
            .clip(mask.buffer(11)),
            snap_df=to_gdf(snap_df["geometry"], 25833).clip(mask.buffer(11)),
            as_lines=to_gdf(as_lines, 25833).clip(mask.buffer(11)),
            title="helt nederst buff",
        )
    except Exception:
        pass

    l = to_gdf(geoms, 25833).clip(mask.buffer(23))
    for g in l.geometry:
        try:
            qtm(
                rings=to_gdf(rings, 25833).clip(mask.buffer(23)).pipe(buff, 0.5),
                snap_df=to_gdf(snap_df["geometry"], 25833).clip(mask.buffer(23)),
                lines=to_gdf(lines.geometry, 25833).clip(mask.buffer(23)),
                as_lines=to_gdf(as_lines, 25833).clip(mask.buffer(23)),
                g=to_gdf(g, 25833).clip(mask.buffer(23)),
            )
        except Exception:
            pass

    no_values = pd.Series(
        {i: None for i in range(len(rings)) if i not in as_lines.index}
    )

    """try:
        qtm(
            lines=to_gdf(polygons(as_lines), 25833).clip(mask),
            l=pd.concat([as_lines, no_values]).sort_index().clip(mask),
        )
    except ValueError:
        pass"""

    # explore_locals(explore=False, mask=mask)

    return pd.concat([as_lines, no_values]).sort_index()

    coords = list(get_coordinates(snapped)) + list(no_values)
    indices = list(snapped.index) + list(no_values.index)

    print(snapped)
    print(coords)
    print(indices)
    return linearrings(coords, indices=indices)


# def line_merge_by_force(line: MultiLineString | LineString) -> LineString:
def line_merge_by_force(lines: GeoSeries) -> GeoSeries:
    """converts a GeoSeries of (multi)linestrings to linestrings."""

    lines.geometry = line_merge(lines.geometry)

    not_merged = lines.geometry.geom_type == "MultiLineString"

    if not not_merged.any():
        return lines

    """
    lines.loc[not_merged, "geometry"] = lines.loc[not_merged, "geometry"].apply(
        line_merge_by_force
    )

    lines.geometry = line_merge(unary_union(lines.geometry))

    if isinstance(line, LineString):
        return line

    line = line_merge(unary_union(line))

    if isinstance(line, LineString):
        return line
    """

    parts = lines.loc[not_merged, "geometry"].explode(
        index_parts=False, ignore_index=False
    )

    largest = lines.loc[parts.length.groupby(level=0).idxmax().values]
    not_largest = lines.loc[~lines.index.isin(largest.index)]
    not_largest = extract_unique_points(not_largest)

    lines = GeoDataFrame({"geometry": get_parts(line)})
    if not len(lines):
        return LineString()
    largest_idx: int = lines.length.idxmax()
    largest = lines.loc[[largest_idx]]
    not_largest = lines.loc[lines.index != largest_idx]
    not_largest.geometry = extract_unique_points(not_largest.geometry)
    not_largest = not_largest.explode(ignore_index=True)

    splitted = split_lines_by_nearest_point(
        largest, not_largest, max_distance=PRECISION
    )

    # close_network_holes?? TODO

    line = line_merge(splitted.unary_union)

    if not isinstance(line, LineString):
        raise ValueError("Couldn't merge lines", line)

    return line


def line_merge_by_force(line: MultiLineString | LineString) -> LineString:
    """converts a (multi)linestring to a linestring if possible."""

    if isinstance(line, LineString):
        return line

    line = line_merge(unary_union(line))

    if isinstance(line, LineString):
        return line

    lines = GeoDataFrame({"geometry": get_parts(line)})
    if not len(lines):
        return LineString()
    largest_idx: int = lines.length.idxmax()
    largest = lines.loc[[largest_idx]]
    not_largest = lines.loc[lines.index != largest_idx]
    not_largest.geometry = extract_unique_points(not_largest.geometry)
    not_largest = not_largest.explode(ignore_index=True)

    splitted = split_lines_by_nearest_point(
        largest, not_largest, max_distance=PRECISION
    )

    # close_network_holes?? TODO

    line = line_merge(splitted.unary_union)

    if not isinstance(line, LineString):
        raise ValueError("Couldn't merge lines", line)

    return line


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

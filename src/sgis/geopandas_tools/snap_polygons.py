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
    STRtree,
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
    snap,
    unary_union,
)
from shapely.geometry import LinearRing, LineString, MultiLineString, MultiPoint, Point
from shapely.ops import nearest_points

from ..maps.maps import explore, explore_locals, qtm
from ..networkanalysis.cutting_lines import split_lines_by_nearest_point
from .buffer_dissolve_explode import buff, dissexp_by_cluster
from .conversion import coordinate_array, to_gdf
from .duplicates import get_intersections
from .general import _determine_geom_type_args, clean_geoms, sort_large_first, to_lines
from .geometry_types import get_geom_type, make_all_singlepart, to_single_geom_type
from .neighbors import get_all_distances, k_nearest_neighbors
from .overlay import clean_overlay
from .polygon_operations import close_small_holes, get_holes
from .polygons_as_rings import PolygonsAsRings
from .sfilter import sfilter_inverse


PRECISION = 1e-4


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
        if isinstance(multipoints.index, pd.MultiIndex):
            indices = pd.MultiIndex.from_arrays(indices, names=multipoints.index.names)

        point_df = pd.DataFrame({"geometry": GeometryArray(points)}, index=indices)

    try:
        point_df = point_df.to_frame("geometry")
    except AttributeError:
        pass

    point_df["next"] = point_df.groupby(level=0)["geometry"].shift(-1)

    first_points = point_df.loc[lambda x: ~x.index.duplicated(), "geometry"]
    is_last_point = point_df["next"].isna()

    point_df.loc[is_last_point, "next"] = first_points
    assert point_df["next"].notna().all()

    point_df["geometry"] = [
        LineString([x1, x2]) for x1, x2 in zip(point_df["geometry"], point_df["next"])
    ]
    return GeoDataFrame(point_df.drop(columns=["next"]), geometry="geometry", crs=crs)


def build_snap_index(gdf, tolerance):
    if len(gdf) <= 1:
        return gdf

    geom_type = get_geom_type(gdf)

    gdf = make_all_singlepart(clean_geoms(gdf), ignore_index=True)

    gdf.geometry = (
        PolygonsAsRings(gdf.geometry.values)
        .apply_numpy_func(
            _snap,
            kwargs=dict(
                tolerance=tolerance,
            ),
        )
        .to_numpy()
    )

    gdf = make_all_singlepart(gdf)

    return to_single_geom_type(gdf, geom_type) if geom_type != "mixed" else gdf


def _build_snap_index(gdf, tolerance) -> pd.DataFrame:
    tree = STRtree(gdf.geometry.values)
    left, right = tree.query(
        gdf.geometry.values, predicate="dwithin", distance=tolerance
    )
    indices = pd.Series(right, index=left, name="_int_idx").to_frame()
    indices["_geom_idx_left"] = indices.index.map(dict(enumerate(gdf.index)))
    indices["_geom_idx_right"] = indices["_int_idx"].map(dict(enumerate(gdf.index)))

    not_from_same_geom = indices["_geom_idx_left"] != indices["_geom_idx_right"]
    return indices[not_from_same_geom]


def _snap(geoms, tolerance):
    gdf = GeoDataFrame({"geometry": geoms, "_geom_idx": range(len(geoms))})

    points = gdf.assign(
        geometry=lambda x: extract_unique_points(x.geometry.values)
    ).explode(ignore_index=True)

    snap_vertice_index = _build_snap_index(points, tolerance)
    points_in_index = points.loc[snap_vertice_index.index]

    segments = get_line_segments(gdf)
    segments.index = range(len(segments))
    endpoints = segments.boundary
    sources = endpoints.groupby(level=0).nth(0)
    targets = endpoints.groupby(level=0).nth(-1)

    missing_segments = segments.loc[
        (~sources.isin(points_in_index)) & (~targets.isin(points_in_index))
    ]
    explore(
        segments,
        missing_segments,
        points_in_index,
    )

    # missing = gdf.loc[]
    snap_point_index = _build_snap_index(segments, tolerance)
    print(snap_vertice_index)
    print(snap_point_index)

    # indices["_distance"] = distance(
    #     segments.loc[indices.index, "geometry"].values,
    #     segments.loc[indices["_int_idx"].values, "geometry"].values,
    # )
    indices["_distance"] = 1  # temp

    indices["_nearest_point"] = nearest_points(
        segments.loc[indices.index, "geometry"].values,
        segments.loc[indices["_int_idx"].values, "geometry"].values,
    )[1]

    snap_points = indices.sort_values(["_geom_idx_right", "_distance"]).drop_duplicates(
        "_geom_idx_left"
    )

    explore(
        segments,
        points0=to_gdf(
            segments.loc[snap_points.index.values, "geometry"].values, 25833
        ),
        points1=to_gdf(segments.loc[indices.index.values, "geometry"].values, 25833),
        points2=to_gdf(
            segments.loc[snap_points["_int_idx"].values, "geometry"].values, 25833
        ),
        points3=to_gdf(
            segments.loc[indices["_int_idx"].values, "geometry"].values, 25833
        ),
    )
    print(indices)
    print(indices)
    print(snap_points)

    gdf.loc[snap_points.index, "geometry"] = gdf.loc[
        snap_points["_int_idx"].values, "geometry"
    ].values

    return gdf.groupby("_geom_idx")["geometry"].agg(LinearRing)

    gdf.loc[snap_points.index, geom_col] = gdf.loc[
        snap_points["_int_idx"].values, geom_col
    ].values

    return gdf


def snap_qgis(gdf):
    subjPointFlags = []
    for i, poly in enumerate(gdf.geometry):
        subjPointFlags.append([])
        for j, ring in enumerate(poly._rings):
            ring_points = extract_unique_points(ring)
            nVerts = len(ring_points)
            for point in ring_points:
                if j > 0 and j < nVerts:
                    subjPointFlags[i][j].append(point)


def snap_polygons(
    gdf: GeoDataFrame,
    snap_to: GeoDataFrame,
    tolerance: float,
) -> GeoDataFrame:
    _snap_checks(gdf, tolerance)

    geom_type = get_geom_type(gdf)

    line_types = ["LineString", "MultiLineString", "LinearRing"]

    if not snap_to.geom_type.isin(line_types).all():
        snap_to = to_lines(snap_to[["geometry"]])
    else:
        snap_to = snap_to[["geometry"]]

    snap_to: GeoSeries = clean_overlay(
        snap_to,
        gdf.buffer(tolerance).to_frame(),
        how="intersection",
        keep_geom_type=False,
    )  # .geometry

    gdf.geometry = (
        PolygonsAsRings(gdf.geometry.values)
        .apply_numpy_func(
            _snap_linearrings,
            kwargs=dict(
                snap_to=snap_to,
                tolerance=tolerance,
            ),
        )
        .to_numpy()
    )

    gdf = make_all_singlepart(gdf)

    return to_single_geom_type(gdf, geom_type) if geom_type != "mixed" else gdf


def _snap_linearrings(rings, snap_to, tolerance):
    assert len(rings.shape) == 1, "ring array should be 1 dimensional"

    ring_points = GeoDataFrame({"geometry": rings})
    isna = ring_points[lambda x: x["geometry"].isna()]
    ring_points = ring_points[lambda x: x["geometry"].notna()]

    if not len(ring_points):
        return np.array([])

    ring_points.geometry = extract_unique_points(ring_points.geometry)
    ring_points = ring_points.explode(index_parts=False)

    ring_points["_ring_index"] = ring_points.index
    ring_points["_point_index"] = range(len(ring_points))
    ring_points.index = ring_points.geometry.values

    more_snap_to_points = GeoDataFrame(
        {
            "geometry": nearest_points(
                ring_points.geometry.values, snap_to.unary_union
            )[1]
        }
    )

    snap_to.geometry = extract_unique_points(snap_to.geometry)
    snap_to = snap_to.explode(ignore_index=True)
    snap_to = pd.concat([snap_to, more_snap_to_points])

    joined = snap_to.sjoin(buff(ring_points, tolerance, resolution=10))
    joined["_distance"] = distance(joined.geometry.values, joined["index_right"].values)

    joined["_distance"] = distance(joined.geometry.values, joined["index_right"].values)

    unique = (
        joined.sort_values(["_point_index", "_distance"])
        .drop_duplicates("index_right")
        .drop_duplicates("_point_index")
    )

    missing_snap_to = snap_to.loc[lambda x: ~x.index.isin(unique.index)]
    missing_ring_points = ring_points.loc[
        lambda x: ~x["_point_index"].isin(unique["_point_index"])
    ]

    missing = missing_ring_points.sjoin_nearest(missing_snap_to, max_distance=tolerance)

    explore(
        to_gdf([r for r in rings if r is not None], 25833),
        ring_points,
        joined,
        more_snap_to_points,
        snap_to,
        mask=to_gdf([5.36750884, 59.00830496], 4326).to_crs(25833).buffer(10),
        max_zoom=50,
    )

    snapped = (
        pd.concat([unique, missing, ring_points])
        .drop_duplicates("_point_index")
        .sort_values("_point_index")
        .groupby("_ring_index", dropna=False)["geometry"]
        .agg(LinearRing)
    )

    explore(
        to_gdf(rings, 25833),
        snapped,
        ring_points,
        joined,
        snap_to,
        mask=to_gdf([5.36750884, 59.00830496], 4326).to_crs(25833).buffer(10),
        max_zoom=50,
    )

    return pd.concat([snapped, isna["geometry"]]).sort_index().values

    snap_df: GeoDataFrame = join_lines_with_snap_to(
        lines=line_segments,
        snap_to=snap_to,
        tolerance=tolerance,
    )

    snap_df["endpoints"] = snap_df.geometry.boundary

    agged = snap_df.groupby(level=0).apply(sorted_unary_union)
    snap_df = snap_df.loc[lambda x: ~x.index.duplicated()]
    snap_df.geometry = agged

    return rings


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

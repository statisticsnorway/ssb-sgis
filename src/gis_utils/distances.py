import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame
from shapely import force_2d
from shapely.geometry import LineString, Point
from sklearn.neighbors import NearestNeighbors

from .buffer_dissolve_explode import buff
from .geopandas_utils import gdf_concat, snap_to
from .network_functions import make_edge_coords_cols


def get_k_nearest_neighbors(
    gdf: GeoDataFrame,
    neighbors: GeoDataFrame,
    k: int,
    id_cols: str | list[str, str] | tuple[str, str] | None = None,
    min_dist: int = 0.0000001,
    max_dist: int | None = None,
    strict: bool = False,
) -> DataFrame:
    """
    It takes a GeoDataFrame of points, a GeoDataFrame of neighbors, and a number of
    neighbors to find, and returns a DataFrame of the k nearest neighbors for each point
    in the GeoDataFrame.

    Args:
      gdf: a GeoDataFrame of points
      neighbors: a GeoDataFrame of points
      k (int): number of neighbors to find
      id_cols: one or two column names (strings)
      min_dist (int): The minimum distance between the two points. Defaults to 0.0000001
        so that identical points aren't considered neighbors.
      max_dist: if specified, distances larger than this number will be removed.
      strict (bool): If True, will raise an error if k is greater than the number of
        points in to_array. If False, will return all distances if there is less than
        k points in to_array. Defaults to False.

    Returns:
      A DataFrame with the following columns:
    """

    if gdf.crs != neighbors.crs:
        raise ValueError("crs mismatch:", gdf.crs, "and", neighbors.crs)

    if id_cols:
        id_col1, id_col2 = return_two_id_cols(id_cols)
        id_dict_gdf = {i: col for i, col in zip(range(len(gdf)), gdf[id_col1])}
        id_dict_neighbors = {
            i: col for i, col in zip(range(len(neighbors)), neighbors[id_col2])
        }
    else:
        id_col1, id_col2 = "gdf_idx", "neighbors_idx"

    gdf_array = coordinate_array(gdf)
    neighbors_array = coordinate_array(neighbors)

    dists, neighbor_indices = k_nearest_neighbors(gdf_array, neighbors_array, k, strict)

    edges = get_edges(gdf, neighbor_indices)

    if max_dist:
        condition = (dists <= max_dist) & (dists >= min_dist)
    else:
        condition = dists >= min_dist

    edges = edges[condition]
    if len(edges.shape) == 3:
        edges = edges[0]

    dists = dists[condition]

    if id_col1 == id_col2:
        id_col2 = id_col2 + "2"

    df = DataFrame(edges, columns=[id_col1, id_col2])

    df = df.assign(
        dist=dists,
        dist_min=lambda df: df.groupby(id_col1)["dist"].transform("min"),
        k=lambda df: df.groupby(id_col1)["dist"].transform("rank"),
    )

    if id_cols:
        df[id_col1] = df[id_col1].map(id_dict_gdf)
        df[id_col2] = df[id_col2].map(id_dict_neighbors)

    return df


def split_lines_at_closest_point(
    lines: GeoDataFrame,
    points: GeoDataFrame,
    max_dist: int | None = None,
) -> DataFrame:
    BUFFDIST = 0.000001

    if points.crs != lines.crs:
        raise ValueError("crs mismatch:", points.crs, "and", lines.crs)

    lines.geometry = force_2d(lines.geometry)

    points_snapped = snap_to(points, lines, max_dist=max_dist, to_node=False)

    points_snapped["point_coords"] = [
        (geom.x, geom.y) for geom in points_snapped.geometry
    ]

    lines["temp_lineidx"] = lines.index
    points_snapped = points_snapped.sjoin_nearest(lines)

    line_indices = points_snapped.set_index("temp_lineidx").index
    relevant_lines = lines.loc[line_indices]
    lines = lines.loc[~lines.index.isin(line_indices)]

    # splitting geometry doesn't work, so doing a buffer and difference instead
    splitted = relevant_lines.overlay(
        buff(points_snapped, BUFFDIST), how="difference"
    ).explode(ignore_index=True)

    splitted["splitidx"] = splitted.index

    splitted = make_edge_coords_cols(splitted)

    splitted_source = GeoDataFrame(
        {
            "splitidx": splitted.splitidx,
            "geometry": GeoSeries(
                [Point(geom) for geom in splitted["source_coords"]], crs=lines.crs
            ),
        }
    )
    splitted_target = GeoDataFrame(
        {
            "splitidx": splitted.splitidx,
            "geometry": GeoSeries(
                [Point(geom) for geom in splitted["target_coords"]], crs=lines.crs
            ),
        }
    )

    dists_source = get_k_nearest_neighbors(
        splitted_source,
        points_snapped,
        k=1,
        max_dist=BUFFDIST * 2,
        id_cols=("splitidx", "point_coords"),
    )
    dists_target = get_k_nearest_neighbors(
        splitted_target,
        points_snapped,
        k=1,
        max_dist=BUFFDIST * 2,
        id_cols=("splitidx", "point_coords"),
    )

    splitdict_source = {
        idx: coords
        for idx, coords in zip(dists_source.splitidx, dists_source.point_coords)
    }
    splitdict_target = {
        idx: coords
        for idx, coords in zip(dists_target.splitidx, dists_target.point_coords)
    }

    # change the first point of each line that has a source by the point
    for idx in dists_source.splitidx:
        line = splitted.loc[idx, "geometry"]
        coordslist = list(line.coords)
        coordslist[0] = splitdict_source[idx]
        splitted.loc[splitted.splitidx == idx, "geometry"] = LineString(coordslist)

    # change the last point of each line that has a target by the point
    for idx in dists_target.splitidx:
        line = splitted.loc[idx, "geometry"]
        coordslist = list(line.coords)
        coordslist[-1] = splitdict_target[idx]
        splitted.loc[splitted.splitidx == idx, "geometry"] = LineString(coordslist)

    splitted["splitted"] = 1

    lines = gdf_concat([lines, splitted]).drop(["temp_lineidx"], axis=1)

    return lines


def coordinate_array(gdf: GeoDataFrame) -> np.ndarray[np.ndarray[float]]:
    """Takes a GeoDataFrame of point geometries and turns it into a 2d ndarray
    of coordinates.
    """
    return np.array([(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)])


def k_nearest_neighbors(
    from_array: np.ndarray[np.ndarray[float]],
    to_array: np.ndarray[np.ndarray[float]],
    k: int,
    strict: bool = False,
) -> tuple[np.ndarray[float]]:
    """
    Given a set of points, find the k nearest neighbors of each point in another
    set of points.

    Args:
      from_array: The array of points (coordinate tuples) you want to find the nearest
        neighbors for.
      to_array: The array of points that we want to find the nearest neighbors of.
      k: the number of nearest neighbors to find.
      strict: If True, will raise an error if k is greater than the number of points in
        to_array. If False, will return all distances if there is less than k points in
        to_array. Defaults to False

    Returns:
      The distances and indices of the nearest neighbors.
    """

    if not strict:
        k = k if len(to_array) >= k else len(to_array)

    nbr = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(to_array)
    dists, indices = nbr.kneighbors(from_array)
    return dists, indices


def get_edges(gdf: GeoDataFrame, indices: np.ndarray[float]) -> np.ndarray[tuple[int]]:
    """Takes a GeoDataFrame and a list of indices, and returns a list of edges.

    Args:
      gdf (GeoDataFrame): GeoDataFrame
      indices (np.ndarray[float]): a numpy array of the indices of the nearest neighbors
        for each point in the GeoDataFrame.

    Returns:
      A numpy array of edge tuples (from-to indices).
    """
    return np.array(
        [[(i, neighbor) for neighbor in indices[i]] for i in range(len(gdf))]
    )


def return_two_id_cols(id_cols: str | list[str, str] | tuple[str, str]) -> tuple[str]:
    """
    Make sure the id_cols are a 2 length tuple.> If the input is a string, return
    a tuple of two strings. If the input is a list or tuple of two
    strings, return the list or tuple. Otherwise, raise a ValueError

    Args:
      id_cols: one or two id columns (strings)

    Returns:
      A tuple of two strings.
    """

    if isinstance(id_cols, (tuple, list)) and len(id_cols) == 2:
        return id_cols
    elif isinstance(id_cols, str):
        return id_cols, id_cols
    if isinstance(id_cols, (tuple, list)) and len(id_cols) == 1:
        return id_cols[0], id_cols[0]
    else:
        raise ValueError

"""Get neighbors and K-nearest neighbors."""
import warnings

import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame
from sklearn.neighbors import NearestNeighbors

from ..helpers import return_two_vals
from .general import coordinate_array


def get_neighbor_indices(
    gdf: GeoDataFrame | GeoSeries,
    neighbors: GeoDataFrame | GeoSeries,
    max_dist: int = 0,
    predicate: str = "intersects",
) -> list:
    """Returns a list of the indices of a GeoDataFrame's neigbours.

    Finds all the geometries in 'neighbors' that intersect with 'gdf' and returns a
    list of the indices of the neighbors.

    Args:
        gdf: GeoDataFrame or GeoSeries
        neighbors: GeoDataFrame or GeoSeries
        max_dist: The maximum distance between the two geometries. Defaults to 0.
        predicate: Spatial predicate to use. Defaults to "intersects", meaning the
            geometry itself and geometries within will be considered neighbors if they
            are part of the 'neighbors' GeoDataFrame.

    Returns:
        A list of the indices of the intersecting neighbor indices.

    Raises:
        ValueError: If gdf and neighbors do not have the same coordinate reference
            system.

    Examples
    --------
    >>> from sgis import get_neighbor_indices, to_gdf
    >>> points = to_gdf([(0, 0), (0.5, 0.5), (2, 2)])
    >>> points
                    geometry
    0  POINT (0.00000 0.00000)
    1  POINT (0.50000 0.50000)
    2  POINT (2.00000 2.00000)

    >>> p1 = points.iloc[[0]]
    >>> get_neighbor_indices(p1, points)
    [0]
    >>> get_neighbor_indices(p1, points, max_dist=1)
    [0, 1]
    >>> get_neighbor_indices(p1, points, max_dist=3)
    [0, 1, 2]
    """
    return _get_neighborlist(
        gdf=gdf,
        neighbors=neighbors,
        id_col="index",
        max_dist=max_dist,
        predicate=predicate,
    )


def get_neighbor_ids(
    gdf: GeoDataFrame | GeoSeries,
    neighbors: GeoDataFrame | GeoSeries,
    id_col: str,
    max_dist: int = 0,
    predicate: str = "intersects",
) -> list:
    """Returns a list of the column values of a GeoDataFrame's neigbours.

    Finds all the geometries in 'neighbors' that intersect with 'gdf' and returns a
    list of the 'id_col' column values of the neighbors.

    Args:
        gdf: GeoDataFrame or GeoSeries
        neighbors: GeoDataFrame or GeoSeries
        id_col: The column in the GeoDataFrame to use as identifier for the
            neighbors.
        max_dist: The maximum distance between the two geometries. Defaults to 0.
        predicate: Spatial predicate to use. Defaults to "intersects", meaning the
            geometry itself and geometries within will be considered neighbors if they
            are part of the 'neighbors' GeoDataFrame.

    Returns:
        A list of values from the 'id_col' column in the 'neighbors' GeoDataFrame.

    Raises:
        ValueError: If gdf and neighbors do not have the same coordinate reference
            system.

    Examples
    --------
    >>> from sgis import get_neighbor_ids, to_gdf
    >>> points = to_gdf([(0, 0), (0.5, 0.5), (2, 2)])
    >>> points["id_col"] = [*"abc"]
    >>> points
                    geometry id_col
    0  POINT (0.00000 0.00000)      a
    1  POINT (0.50000 0.50000)      b
    2  POINT (2.00000 2.00000)      c

    >>> p1 = points.iloc[[0]]
    >>> get_neighbor_ids(p1, points, id_col="id_col")
    ['a']
    >>> get_neighbor_ids(p1, points, max_dist=1, id_col="id_col")
    ['a', 'b']
    >>> get_neighbor_ids(p1, points, max_dist=3, id_col="id_col")
    ['a', 'b', 'c']
    """
    return _get_neighborlist(
        gdf=gdf,
        neighbors=neighbors,
        id_col=id_col,
        max_dist=max_dist,
        predicate=predicate,
    )


def _get_neighborlist(
    gdf: GeoDataFrame | GeoSeries,
    neighbors: GeoDataFrame | GeoSeries,
    id_col: str = "index",
    max_dist: int = 0,
    predicate: str = "intersects",
) -> list[str]:
    """Returns list of indices or values of the 'id_col'."""

    if gdf.crs != neighbors.crs:
        raise ValueError(f"'crs' mismatch. Got {gdf.crs} and {neighbors.crs}")

    # if index, use the column returned by geopandas.sjoin
    if id_col == "index":
        id_col = "index_right"

    # buffer and keep only geometry column
    if max_dist:
        if gdf.crs == 4326:
            warnings.warn(
                "'gdf' has latlon crs, meaning the 'max_dist' paramter "
                "will not be in meters, but degrees."
            )
        gdf = gdf.buffer(max_dist).to_frame()
    else:
        gdf = gdf.geometry.to_frame()

    joined = gdf.sjoin(neighbors, how="inner", predicate=predicate)

    return [x for x in joined[id_col].unique()]


def get_all_distances(
    gdf: GeoDataFrame,
    neighbors: GeoDataFrame,
    id_cols: str | tuple[str, str] = ("gdf_idx", "neighbour_idx"),
) -> DataFrame:
    """Get distance and id from 'gdf' to all points in 'neighbors'.

    Uses the K-nearest neighbors algorithm method from sklearn.neighbors to find the
    # distance from each point in 'gdf' to each point in 'neighbors'. Identical points
    are considered neighbors.

    Args:
        gdf: a GeoDataFrame of points
        neighbors: a GeoDataFrame of points
        id_cols: column(s) to use as identifiers. Either a string if one column or a
            tuple/list for 'gdf' and 'neighbors' respectfully. Defaults to "gdf_idx"
            and "neighbour_idx".

    Returns:
        A DataFrame with id columns and the distance from gdf to neighbour. Also
        includes columns for the minumum distance for each point in 'gdf' and
        the rank.

    Raises:
        ValueError: If the coordinate reference system of 'gdf' and 'neighbors' are
            not the same.

    Examples
    --------
    >>> from sgis import get_all_distances, random_points
    >>> points = random_points(10)
    >>> neighbors = random_points(10)
    >>> get_all_distances(points, neighbors)
        gdf_idx  neighbour_idx      dist  dist_min     k
    0         0              9  0.329634  0.329634   1
    1         0              6  0.457103  0.329634   2
    2         0              0  0.763695  0.329634   3
    3         0              8  0.788817  0.329634   4
    4         0              5  0.800746  0.329634   5
    ..      ...            ...       ...       ...   ...
    95        9              6  0.509254  0.106837   6
    96        9              8  0.535454  0.106837   7
    97        9              9  0.663042  0.106837   8
    98        9              1  0.749489  0.106837   9
    99        9              7  0.841321  0.106837  10

    [100 rows x 5 columns]

    Setting id columns.

    >>> points["from_id"] = [*"abcdefghij"]
    >>> neighbors["to_id"] = neighbors.index
    >>> get_all_distances(points, neighbors, id_cols=("from_id", "to_id"))
       from_id  to_id      dist  dist_min     k
    0        a      9  0.273176  0.273176   1
    1        a      6  0.426856  0.273176   2
    2        a      8  0.685729  0.273176   3
    3        a      5  0.729792  0.273176   4
    4        a      0  0.736613  0.273176   5
    ..     ...    ...       ...       ...   ...
    95       j      8  0.398965  0.034901   6
    96       j      6  0.419967  0.034901   7
    97       j      9  0.564396  0.034901   8
    98       j      1  0.620475  0.034901   9
    99       j      7  0.723542  0.034901  10

    [100 rows x 5 columns]
    """
    return get_k_nearest_neighbors(
        gdf=gdf,
        neighbors=neighbors,
        k=len(neighbors),
        id_cols=id_cols,
    )


def get_k_nearest_neighbors(
    gdf: GeoDataFrame,
    neighbors: GeoDataFrame,
    k: int | None = None,
    id_cols: str | tuple[str, str] = ("gdf_idx", "neighbour_idx"),
    strict: bool = False,
) -> DataFrame:
    """Finds the k nearest neighbors for a GeoDataFrame of points.

    Uses the K-nearest neighbors algorithm method from sklearn.neighbors to find the
    given number of neighbors for each point in 'gdf'. Identical points are considered
    neighbors.

    Args:
        gdf: a GeoDataFrame of points
        neighbors: a GeoDataFrame of points
        k: number of neighbors to find. If not specified, all neighbors will be
            returned.
        id_cols: column(s) to use as identifiers. Either a string if one column or a
            tuple/list for 'gdf' and 'neighbors' respectfully. Defaults to "gdf_idx"
            and "neighbour_idx".
        strict: If False (the default), no exception is raised if k is larger than the
            number of points in 'neighbors'. If True, 'k' must be less than or equal
            to the number of points in 'neighbors'.

    Returns:
        A DataFrame with id columns and the distance from gdf to neighbour. Also
        includes columns for the minumum distance for each point in 'gdf' and
        the rank.

    Raises:
        ValueError: If the coordinate reference system of 'gdf' and 'neighbors' are
            not the same.

    Examples
    --------

    >>> from sgis import get_k_nearest_neighbors, random_points
    >>> points = random_points(10)
    >>> neighbors = random_points(10)
    >>> get_k_nearest_neighbors(points, neighbors, k=10)
        gdf_idx  neighbour_idx      dist  dist_min   k
    0         0              9  0.329634  0.329634   1
    1         0              6  0.457103  0.329634   2
    2         0              0  0.763695  0.329634   3
    3         0              8  0.788817  0.329634   4
    4         0              5  0.800746  0.329634   5
    ..      ...            ...       ...       ...  ..
    95        9              6  0.509254  0.106837   6
    96        9              8  0.535454  0.106837   7
    97        9              9  0.663042  0.106837   8
    98        9              1  0.749489  0.106837   9
    99        9              7  0.841321  0.106837  10

    [100 rows x 5 columns]

    Setting id columns.

    >>> points["from_id"] = [*"abcdefghij"]
    >>> neighbors["to_id"] = neighbors.index
    >>> get_k_nearest_neighbors(
    ...      points,
    ...      neighbors,
    ...      k=10,
    ...      id_cols=("from_id", "to_id")
    ... )
       from_id  to_id      dist  dist_min   k
    0        a      9  0.273176  0.273176   1
    1        a      6  0.426856  0.273176   2
    2        a      8  0.685729  0.273176   3
    3        a      5  0.729792  0.273176   4
    4        a      0  0.736613  0.273176   5
    ..     ...    ...       ...       ...  ..
    95       j      8  0.398965  0.034901   6
    96       j      6  0.419967  0.034901   7
    97       j      9  0.564396  0.034901   8
    98       j      1  0.620475  0.034901   9
    99       j      7  0.723542  0.034901  10

    [100 rows x 5 columns]
    """
    if gdf.crs != neighbors.crs:
        raise ValueError("crs mismatch:", gdf.crs, "and", neighbors.crs)

    id_col1, id_col2 = return_two_vals(id_cols)

    gdf, id_col1 = _add_id_col_if_missing(gdf, id_col1, default="gdf_idx")
    neighbors, id_col2 = _add_id_col_if_missing(
        neighbors, id_col2, default="neighbour_idx"
    )

    id_dict_gdf = {i: col for i, col in zip(range(len(gdf)), gdf[id_col1], strict=True)}
    id_dict_neighbors = {
        i: col for i, col in zip(range(len(neighbors)), neighbors[id_col2], strict=True)
    }

    if id_col1 == id_col2:
        id_col2 = id_col2 + "_right"

    gdf_array = coordinate_array(gdf)
    neighbors_array = coordinate_array(neighbors)

    dists, neighbor_indices = k_nearest_neighbors(gdf_array, neighbors_array, k, strict)

    edges = _get_edges(gdf, neighbor_indices)

    edges = edges[dists >= 0]
    dists = dists[dists >= 0]

    df = DataFrame(edges, columns=[id_col1, id_col2])

    df = df.assign(
        distance=dists,
        distance_min=lambda df: df.groupby(id_col1)["distance"].transform("min"),
        k=lambda df: df.groupby(id_col1)["distance"].transform("rank").astype(int),
    )

    df[id_col1] = df[id_col1].map(id_dict_gdf)
    df[id_col2] = df[id_col2].map(id_dict_neighbors)

    return df


def _add_id_col_if_missing(
    gdf: GeoDataFrame, id_col: str | None, default: str
) -> tuple[GeoDataFrame, str]:
    if (not id_col) or (id_col and id_col not in gdf.columns):
        id_col = default
        gdf[id_col] = range(len(gdf))
    return gdf, id_col


def k_nearest_neighbors(
    from_array: np.ndarray[np.ndarray[float]],
    to_array: np.ndarray[np.ndarray[float]],
    k: int | None = None,
    strict: bool = False,
) -> tuple[np.ndarray[float], np.ndarray[int]]:
    if not k:
        k = len(to_array)

    if not strict:
        k = k if len(to_array) >= k else len(to_array)

    nbr = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(to_array)
    dists, indices = nbr.kneighbors(from_array)
    return dists, indices


def _get_edges(gdf: GeoDataFrame, indices: np.ndarray[int]) -> np.ndarray[tuple[int]]:
    """Takes a GeoDataFrame and array of indices, and returns a 2d array of edges.

    Args:
        gdf (GeoDataFrame): GeoDataFrame
        indices (np.ndarray[float]): a numpy array of the indices of the nearest
            neighbors for each point in the GeoDataFrame.

    Returns:
      A 2d numpy array of edges (from-to indices).
    """
    return np.array(
        [[(i, neighbor) for neighbor in indices[i]] for i in range(len(gdf))]
    )

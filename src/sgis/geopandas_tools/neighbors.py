"""Get neighbors and K-nearest neighbors."""
import warnings

import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame
from sklearn.neighbors import NearestNeighbors

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
        A list of the indices of the intersecting neighbors.

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


def get_all_distances(gdf: GeoDataFrame, neighbors: GeoDataFrame) -> DataFrame:
    """Get distances from 'gdf' to all points in 'neighbors'.

    Find the distance from each point in 'gdf' to each point in 'neighbors'. Preserves
    the index of 'gdf' and adds column 'neighbor_index' with the indices of the
    neighbors.

    Args:
        gdf: a GeoDataFrame of points
        neighbors: a GeoDataFrame of points

    Returns:
        DataFrame with distances and index values from the 'gdf'. Also includes the
        column 'neighbor_index'.

    Raises:
        ValueError: If the coordinate reference system of 'gdf' and 'neighbors' are
            not the same.

    Examples
    --------
    >>> from sgis import get_all_distances, random_points
    >>> points = random_points(100)
    >>> neighbors = random_points(100)

    >>> distances = get_all_distances(points, neighbors)
    >>> distances
        distance  neighbor_index
    0   0.028806              41
    0   0.083476              26
    0   0.083965              68
    0   0.103845              94
    0   0.115894              42
    ..       ...             ...
    99  1.137072              66
    99  1.159835              19
    99  1.161581              17
    99  1.177528              34
    99  1.211463              45

    [10000 rows x 2 columns]

    Use set_index to get values from other columns.

    >>> neighbors["custom_id"] = np.random.choice([*"abcde"], len(neighbors))
    >>> distances = get_all_distances(points, neighbors.set_index("custom_id"))
        distance custom_id
    0   0.028806         c
    0   0.083476         b
    0   0.083965         d
    0   0.103845         e
    0   0.115894         c
    ..       ...       ...
    99  1.137072         d
    99  1.159835         a
    99  1.161581         a
    99  1.177528         b
    99  1.211463         c

    [10000 rows x 2 columns]

    Since the index from 'gdf' is preserved, we can join the results with the 'points'.

    >>> joined = points.join(distances)
    >>> joined["k"] = joined.groupby(level=0)["distance"].transform("rank")
    >>> joined
                       geometry  distance custom_id      k
    0   POINT (0.36938 0.47401)  0.028806         c    1.0
    0   POINT (0.36938 0.47401)  0.083476         b    2.0
    0   POINT (0.36938 0.47401)  0.083965         d    3.0
    0   POINT (0.36938 0.47401)  0.103845         e    4.0
    0   POINT (0.36938 0.47401)  0.115894         c    5.0
    ..                      ...       ...       ...    ...
    99  POINT (0.14842 0.94335)  1.137072         d   96.0
    99  POINT (0.14842 0.94335)  1.159835         a   97.0
    99  POINT (0.14842 0.94335)  1.161581         a   98.0
    99  POINT (0.14842 0.94335)  1.177528         b   99.0
    99  POINT (0.14842 0.94335)  1.211463         c  100.0

    Or assign aggregated values onto the points.

    >>> points["mean_distance"] = distances.groupby(level=0)["distance"].mean()
    >>> points["min_distance"] = distances.groupby(level=0)["distance"].min()
    >>> points
                    geometry  mean_distance  min_distance
    0   POINT (0.36938 0.47401)       0.406185      0.028806
    1   POINT (0.63229 0.69861)       0.445811      0.074979
    2   POINT (0.69216 0.93944)       0.583675      0.027223
    3   POINT (0.79615 0.31667)       0.496825      0.086139
    4   POINT (0.28328 0.31433)       0.460716      0.024028
    ..                      ...            ...           ...
    95  POINT (0.59569 0.57141)       0.408475      0.052947
    96  POINT (0.13525 0.90606)       0.621634      0.041611
    97  POINT (0.65454 0.22109)       0.480939      0.055018
    98  POINT (0.34857 0.14396)       0.522410      0.104077
    99  POINT (0.14842 0.94335)       0.637836      0.022742

    [100 rows x 3 columns]
    """
    return get_k_nearest_neighbors(
        gdf=gdf,
        neighbors=neighbors,
        k=len(neighbors),
    )


def get_k_nearest_neighbors(
    gdf: GeoDataFrame,
    neighbors: GeoDataFrame,
    k: int,
    *,
    strict: bool = False,
) -> DataFrame:
    """Finds the k nearest neighbors for a GeoDataFrame of points.

    Uses the K-nearest neighbors algorithm method from scikit-learn to find the given
    number of neighbors for each point in 'gdf'. Identical points are considered
    neighbors. Preserves the index of 'gdf' and adds 'neighbor_index' with the indices
    of the neighbors.

    Args:
        gdf: a GeoDataFrame of points
        neighbors: a GeoDataFrame of points
        k: number of neighbors to find.
        strict: If False (default), no exception is raised if k is larger than the
            number of points in 'neighbors'. If True, 'k' must be less than or equal
            to the number of points in 'neighbors'.

    Returns:
        A DataFrame with the distance from gdf to the k nearest neighbours. The
        index follows the index of 'gdf' and a column 'neighbor_index' is added
        as identifier for the 'neighbors'.

    Raises:
        ValueError: If the coordinate reference system of 'gdf' and 'neighbors' are
            not the same.

    Examples
    --------
    Make some random points.

    >>> from sgis import get_k_nearest_neighbors, random_points
    >>> points = random_points(100)
    >>> neighbors = random_points(100)

    Get 10 nearest neighbors.

    >>> distances = get_k_nearest_neighbors(points, neighbors, k=10)
    >>> distances
        distance  neighbor_index
    0   0.069727               9
    0   0.121001              88
    0   0.141688              45
    0   0.142749              67
    0   0.199803              31
    ..       ...             ...
    99  0.124003              81
    99  0.129462              20
    99  0.174019              36
    99  0.176593              80
    99  0.185566              79

    [1000 rows x 2 columns]

    Use set_index to use another column as identifier for the neighbors.

    >>> neighbors["custom_id"] = [letter for letter in [*"abcde"] for _ in range(20)]
    >>> distances = get_k_nearest_neighbors(points, neighbors.set_index("custom_id"), k=10)
    >>> distances
        distance   custom_id
    0   0.069727           a
    0   0.121001           e
    0   0.141688           c
    0   0.142749           d
    0   0.199803           b
    ..       ...         ...
    99  0.124003           e
    99  0.129462           b
    99  0.174019           b
    99  0.176593           e
    99  0.185566           d

    [1000 rows x 2 columns]

    The index from 'points' is preserved. Use join to get the distance and neighbor ids
    onto the 'points' GeoDataFrame.

    >>> joined = points.join(distances)
    >>> joined["k"] = joined.groupby(level=0)["distance"].transform("rank")
    >>> joined
                       geometry  distance   custom_id     k
    0   POINT (0.02201 0.24950)  0.069727           a   1.0
    0   POINT (0.02201 0.24950)  0.121001           e   2.0
    0   POINT (0.02201 0.24950)  0.141688           c   3.0
    0   POINT (0.02201 0.24950)  0.142749           d   4.0
    0   POINT (0.02201 0.24950)  0.199803           b   5.0
    ..                      ...       ...         ...   ...
    99  POINT (0.33255 0.50495)  0.124003           e   6.0
    99  POINT (0.33255 0.50495)  0.129462           b   7.0
    99  POINT (0.33255 0.50495)  0.174019           b   8.0
    99  POINT (0.33255 0.50495)  0.176593           e   9.0
    99  POINT (0.33255 0.50495)  0.185566           d  10.0

    Or assign aggregated values onto the points.

    >>> points["mean_distance"] = distances.groupby(level=0)["distance"].mean()
    >>> points["min_distance"] = distances.groupby(level=0)["distance"].min()
    >>> points
                    geometry  mean_distance  min_distance
    0   POINT (0.02201 0.24950)       0.187598      0.069727
    1   POINT (0.38886 0.12449)       0.132704      0.066233
    2   POINT (0.09747 0.06234)       0.227391      0.021921
    3   POINT (0.35139 0.45285)       0.124564      0.061903
    4   POINT (0.60701 0.38296)       0.122539      0.021324
    ..                      ...            ...           ...
    95  POINT (0.54114 0.03624)       0.175222      0.063944
    96  POINT (0.45601 0.51177)       0.110830      0.057889
    97  POINT (0.67200 0.56723)       0.134046      0.100790
    98  POINT (0.38345 0.17332)       0.126508      0.029528
    99  POINT (0.33255 0.50495)       0.122239      0.048511

    [100 rows x 3 columns]
    """
    if gdf.crs != neighbors.crs:
        raise ValueError("crs mismatch:", gdf.crs, "and", neighbors.crs)

    # using the range index
    idx_dict_gdf = {i: col for i, col in zip(range(len(gdf)), gdf.index, strict=True)}
    id_dict_neighbors = {
        i: col for i, col in zip(range(len(neighbors)), neighbors.index, strict=True)
    }

    gdf_array = coordinate_array(gdf)
    neighbors_array = coordinate_array(neighbors)

    dists, neighbor_indices = k_nearest_neighbors(gdf_array, neighbors_array, k, strict)

    edges = _get_edges(gdf, neighbor_indices)

    edges = edges[dists >= 0]
    dists = dists[dists >= 0]

    df = DataFrame(edges, columns=["tmp__idx__", "neighbor_index"])

    df["distance"] = dists

    df["tmp__idx__"] = df["tmp__idx__"].map(idx_dict_gdf)
    df["neighbor_index"] = df["neighbor_index"].map(id_dict_neighbors)

    df = df.set_index("tmp__idx__")

    df.index.name = gdf.index.name

    return df


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

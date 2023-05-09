"""Get neighbors and K-nearest neighbors.

The functions rely on the pandas index as identifiers for the geometries and their
neighbors. This makes it easy to join or aggregate the results onto the input
GeoDataFrames.

The results of all functions will be identical with GeoDataFrame and GeoSeries as input
types.
"""
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame, Series
from sklearn.neighbors import NearestNeighbors

from .general import coordinate_array
from .geometry_types import get_geom_type


def get_neighbor_indices(
    gdf: GeoDataFrame | GeoSeries,
    neighbors: GeoDataFrame | GeoSeries,
    max_distance: int = 0,
    predicate: str = "intersects",
) -> Series:
    """Creates a pandas Series with the index of 'gdf' and values of 'neighbors'.

    Finds all the geometries in 'neighbors' that intersect with 'gdf' and returns a
    Series where the values are the 'neighbors' indices and the index is the indices of
    'gdf'. Use set_index inside the function call to get values from a column instead of
    the current index.

    Args:
        gdf: GeoDataFrame or GeoSeries.
        neighbors: GeoDataFrame or GeoSeries.
        max_distance: The maximum distance between the geometries. Defaults to 0.
        predicate: Spatial predicate to use in sjoin. Defaults to "intersects", meaning
            the geometry itself and geometries within will be considered neighbors if
            they are part of the 'neighbors' GeoDataFrame.

    Returns:
        A pandas Series with values of the intersecting 'neighbors' indices.
        The Series' index will follow the index of 'gdf'.

    Raises:
        ValueError: If gdf and neighbors do not have the same coordinate reference
            system.

    Examples
    --------
    >>> from sgis import get_neighbor_indices, to_gdf
    >>> points = to_gdf([(0, 0), (0.5, 0.5)])
    >>> points
                    geometry
    0  POINT (0.00000 0.00000)
    1  POINT (0.50000 0.50000)

    With the default max_distance (0), the points return the index of themselves.

    >>> neighbor_indices = get_neighbor_indices(points, points)
    >>> neighbor_indices
    0    0
    1    1
    Name: neighbor_index, dtype: int64

    With max_distance=1, each point find themselves and the neighbor.

    >>> neighbor_indices = get_neighbor_indices(points, points, max_distance=1)
    >>> neighbor_indices
    0    0
    1    0
    0    1
    1    1
    Name: neighbor_index, dtype: int64

    Using a column instead of the index.

    >>> points["text"] = [*"ab"]
    >>> neighbor_indices = get_neighbor_indices(points, points.set_index("text"), max_distance=1)
    >>> neighbor_indices
    0    a
    1    a
    0    b
    1    b
    Name: neighbor_index, dtype: object

    The returned Series will always keep the index of 'gdf' and have values of the
    'neighbors' index.

    >>> neighbor_indices.index
    Int64Index([0, 1, 0, 1], dtype='int64')

    >>> neighbor_indices.values
    ['a' 'a' 'b' 'b']

    """

    if gdf.crs != neighbors.crs:
        raise ValueError(f"'crs' mismatch. Got {gdf.crs} and {neighbors.crs}")

    # buffer and keep only geometry column
    if max_distance and predicate != "nearest":
        gdf = gdf.buffer(max_distance).to_frame()
    else:
        gdf = gdf.geometry.to_frame()

    if predicate == "nearest":
        max_distance = None if max_distance == 0 else max_distance
        joined = gdf.sjoin_nearest(
            neighbors, how="inner", max_distance=max_distance
        ).rename(columns={"index_right": "neighbor_index"}, errors="raise")
    else:
        joined = gdf.sjoin(neighbors, how="inner", predicate=predicate).rename(
            columns={"index_right": "neighbor_index"}, errors="raise"
        )

    return joined["neighbor_index"]


def get_all_distances(
    gdf: GeoDataFrame | GeoSeries, neighbors: GeoDataFrame | GeoSeries
) -> DataFrame:
    """Get distances from 'gdf' to all points in 'neighbors'.

    Find the distance from each point in 'gdf' to each point in 'neighbors'. Also
    returns the indices of 'neighbors' (as the column 'neighbor_index') and 'gdf'
    (as the index).

    Args:
        gdf: a GeoDataFrame of points.
        neighbors: a GeoDataFrame of points.

    Returns:
        DataFrame with the columns 'neighbor_index' and 'distance'. The index follows
        the index of 'gdf'.

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
        neighbor_index  distance
    0               70  0.050578
    0               24  0.070267
    0               91  0.088510
    0               72  0.095352
    0               40  0.103720
    ..             ...       ...
    99              27  0.713055
    99              60  0.718162
    99              63  0.719675
    99              62  0.719747
    99              90  0.761324

    [10000 rows x 2 columns]

    Use set_index to get values from other columns.

    >>> neighbors["custom_id"] = np.random.choice([*"abcde"], len(neighbors))
    >>> distances = get_all_distances(points, neighbors.set_index("custom_id"))
       neighbor_index  distance
    0               d  0.050578
    0               b  0.070267
    0               e  0.088510
    0               d  0.095352
    0               c  0.103720
    ..            ...       ...
    99              b  0.713055
    99              d  0.718162
    99              d  0.719675
    99              d  0.719747
    99              e  0.761324

    [10000 rows x 2 columns]

    Since the index from 'gdf' is preserved, we can join the results with the 'points'.

    >>> joined = points.join(distances)
    >>> joined
                       geometry neighbor_index  distance
    0   POINT (0.59809 0.34636)              d  0.050578
    0   POINT (0.59809 0.34636)              b  0.070267
    0   POINT (0.59809 0.34636)              e  0.088510
    0   POINT (0.59809 0.34636)              d  0.095352
    0   POINT (0.59809 0.34636)              c  0.103720
    ..                      ...            ...       ...
    99  POINT (0.35305 0.47445)              b  0.713055
    99  POINT (0.35305 0.47445)              d  0.718162
    99  POINT (0.35305 0.47445)              d  0.719675
    99  POINT (0.35305 0.47445)              d  0.719747
    99  POINT (0.35305 0.47445)              e  0.761324

    [10000 rows x 3 columns]

    Or assign aggregated values onto the points.

    >>> points["mean_distance"] = distances.groupby(level=0)["distance"].mean()
    >>> points["min_distance"] = distances.groupby(level=0)["distance"].min()
    >>> points
                       geometry  mean_distance  min_distance
    0   POINT (0.59809 0.34636)       0.417128      0.050578
    1   POINT (0.25444 0.02876)       0.673966      0.016781
    2   POINT (0.22475 0.08637)       0.643514      0.030049
    3   POINT (0.14814 0.23037)       0.593224      0.025758
    4   POINT (0.69298 0.81931)       0.434355      0.051575
    ..                      ...            ...           ...
    95  POINT (0.62453 0.26793)       0.460177      0.031749
    96  POINT (0.11882 0.26615)       0.592930      0.044010
    97  POINT (0.03998 0.77527)       0.592031      0.090983
    98  POINT (0.46047 0.79056)       0.400134      0.016012
    99  POINT (0.35305 0.47445)       0.397660      0.052134

    [100 rows x 3 columns]
    """
    return get_k_nearest_neighbors(
        gdf=gdf,
        neighbors=neighbors,
        k=len(neighbors),
    )


def get_k_nearest_neighbors(
    gdf: GeoDataFrame | GeoSeries,
    neighbors: GeoDataFrame | GeoSeries,
    k: int,
    *,
    strict: bool = False,
) -> DataFrame:
    """Finds the k nearest neighbors for a GeoDataFrame of points.

    Uses the K-nearest neighbors algorithm method from scikit-learn to find the given
    number of neighbors for each point in 'gdf'. Identical points are considered
    neighbors. Preserves the index of 'gdf' and adds the column 'neighbor_index'
    with the indices of the neighbors, as well as a column 'distance'.

    Args:
        gdf: a GeoDataFrame of points
        neighbors: a GeoDataFrame of points
        k: number of neighbors to find.
        strict: If False (default), no exception is raised if k is larger than the
            number of points in 'neighbors'. If True, 'k' must be less than or equal
            to the number of points in 'neighbors'.

    Returns:
        DataFrame with the columns 'neighbor_index' and 'distance'. The index follows
        the index of 'gdf'.

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
        neighbor_index  distance
    0               84  0.049168
    0               59  0.053592
    0               14  0.091812
    0               40  0.118403
    0               77  0.129565
    ..             ...       ...
    99              86  0.153771
    99              92  0.157481
    99              70  0.177368
    99              65  0.184087
    99              26  0.202216

    [1000 rows x 2 columns]

    Use set_index to use another column as identifier for the neighbors.

    >>> neighbors["custom_id"] = [letter for letter in [*"abcde"] for _ in range(20)]
    >>> distances = get_k_nearest_neighbors(points, neighbors.set_index("custom_id"), k=10)
    >>> distances
       neighbor_index  distance
    0               e  0.049168
    0               c  0.053592
    0               a  0.091812
    0               c  0.118403
    0               d  0.129565
    ..            ...       ...
    99              e  0.153771
    99              e  0.157481
    99              d  0.177368
    99              d  0.184087
    99              b  0.202216

    [1000 rows x 2 columns]

    The index from 'points' is preserved. Use join to get the distance and neighbor
    index onto the 'points' GeoDataFrame.

    >>> joined = points.join(distances)
    >>> joined["k"] = joined.groupby(level=0)["distance"].transform("rank")
    >>> joined
                       geometry neighbor_index  distance     k
    0   POINT (0.89067 0.75346)              e  0.049168   1.0
    0   POINT (0.89067 0.75346)              c  0.053592   2.0
    0   POINT (0.89067 0.75346)              a  0.091812   3.0
    0   POINT (0.89067 0.75346)              c  0.118403   4.0
    0   POINT (0.89067 0.75346)              d  0.129565   5.0
    ..                      ...            ...       ...   ...
    99  POINT (0.65910 0.16714)              e  0.153771   6.0
    99  POINT (0.65910 0.16714)              e  0.157481   7.0
    99  POINT (0.65910 0.16714)              d  0.177368   8.0
    99  POINT (0.65910 0.16714)              d  0.184087   9.0
    99  POINT (0.65910 0.16714)              b  0.202216  10.0

    [1000 rows x 4 columns]

    Or assign aggregated values directly onto the points.

    >>> points["mean_distance"] = distances.groupby(level=0)["distance"].mean()
    >>> points["min_distance"] = distances.groupby(level=0)["distance"].min()
    >>> points
                       geometry  mean_distance  min_distance
    0   POINT (0.89067 0.75346)       0.132193      0.049168
    1   POINT (0.41308 0.09462)       0.116610      0.009072
    2   POINT (0.13458 0.44248)       0.100539      0.059576
    3   POINT (0.32670 0.44102)       0.117730      0.056133
    4   POINT (0.82184 0.41231)       0.106685      0.013174
    ..                      ...            ...           ...
    95  POINT (0.29706 0.27520)       0.137398      0.079672
    96  POINT (0.42416 0.26956)       0.160817      0.074759
    97  POINT (0.98337 0.54492)       0.164551      0.070798
    98  POINT (0.42458 0.77459)       0.127562      0.027662
    99  POINT (0.65910 0.16714)       0.143257      0.058453

    [100 rows x 3 columns]
    """
    if gdf.crs != neighbors.crs:
        raise ValueError("crs mismatch:", gdf.crs, "and", neighbors.crs)

    if get_geom_type(gdf) != "point" or get_geom_type(neighbors) != "point":
        raise ValueError("Geometries must be points.")

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


def _get_edges(
    gdf: GeoDataFrame | GeoSeries, indices: np.ndarray[int]
) -> np.ndarray[tuple[int]]:
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

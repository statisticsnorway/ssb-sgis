"""Get neighbors and K-nearest neighbors.

The functions rely on the pandas index as identifiers for the geometries and their
neighbors. This makes it easy to join or aggregate the results onto the input
GeoDataFrames.

The results of all functions will be identical with GeoDataFrame and GeoSeries as input
types.
"""

import numpy as np
import shapely
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from pandas import DataFrame
from pandas import Index
from pandas import MultiIndex
from pandas import Series

from ..conf import _get_instance
from ..conf import config
from .conversion import coordinate_array
from .geometry_types import get_geom_type
from .runners import RTreeQueryRunner


def get_neighbor_indices(
    gdf: GeoDataFrame | GeoSeries,
    neighbors: GeoDataFrame | GeoSeries,
    max_distance: int = 0,
    predicate: str = "intersects",
    rtree_runner: RTreeQueryRunner | None = None,
    n_jobs: int | None = None,
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
        rtree_runner: Optionally debug/manipulate the spatial indexing operations.
            See the 'runners' module for example implementations.
        n_jobs: Number of workers.

    Returns:
        A pandas Series with values of the intersecting 'neighbors' indices.
        The Series' index will follow the index of 'gdf'.

    Raises:
        ValueError: If gdf and neighbors do not have the same coordinate reference
            system.

    Examples:
    ---------
    >>> from sgis import get_neighbor_indices, to_gdf
    >>> points = to_gdf([(0, 0), (0.5, 0.5)])
    >>> points
              geometry
    0      POINT (0 0)
    1  POINT (0.5 0.5)

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
    0    1
    1    0
    1    1
    Name: neighbor_index, dtype: int64

    Using a column instead of the index.

    >>> points["text"] = [*"ab"]
    >>> neighbor_indices = get_neighbor_indices(points, points.set_index("text"), max_distance=1)
    >>> neighbor_indices
    0    a
    0    b
    1    a
    1    b
    Name: neighbor_index, dtype: str

    The returned Series will always keep the index of 'gdf' and have values of the
    'neighbors' index.

    >>> neighbor_indices.index
    Index([0, 0, 1, 1], dtype='int64')

    >>> neighbor_indices.values
    <ArrowStringArray>
    ['a', 'b', 'a', 'b']
    Length: 4, dtype: str

    """
    if isinstance(gdf.index, MultiIndex) or isinstance(neighbors.index, MultiIndex):
        raise ValueError("get_neighbor_indices not implemented for pandas.MultiIndex")
    if gdf.crs != neighbors.crs:
        raise ValueError(f"'crs' mismatch. Got {gdf.crs} and {neighbors.crs}")

    index_name = gdf.index.name

    if rtree_runner is None:
        rtree_runner = _get_instance(config, "rtree_runner", n_jobs=n_jobs)

    if isinstance(neighbors, GeoSeries):
        neighbors = neighbors.to_frame()
    else:
        neighbors = neighbors[[neighbors._geometry_column_name]]

    # buffer and keep only geometry column
    if max_distance and predicate != "nearest":
        gdf = gdf.buffer(max_distance).to_frame("geometry")
    else:
        gdf = gdf.geometry.to_frame("geometry")

    neighbors.index.name = None
    gdf.index.name = None

    if predicate == "nearest":
        max_distance = None if max_distance == 0 else max_distance
        left, right = rtree_runner.run(
            gdf.geometry.values,
            neighbors.geometry.values,
            method="query_nearest",
            max_distance=max_distance,
        )
    else:
        left, right = rtree_runner.run(
            gdf.geometry.values, neighbors.geometry.values, predicate=predicate
        )
    index_mapper1 = {i: x for i, x in enumerate(gdf.index)}
    left = np.array([index_mapper1[i] for i in left])
    if index_name:
        left = Index(left, name=index_name)
    index_mapper2 = {i: x for i, x in enumerate(neighbors.index)}
    right = np.array([index_mapper2[i] for i in right])
    return Series(right, index=left, name="neighbor_index")


def get_neighbor_dfs(
    df: GeoDataFrame | DataFrame,
    neighbor_mapper: Series,
) -> list[GeoDataFrame | DataFrame]:
    return [
        df[df.index.isin(neighbor_mapper[i])] for i in neighbor_mapper.index.unique()
    ]


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

    Examples:
    ---------
    >>> from sgis import get_all_distances, random_points
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> points = random_points(100)
    >>> neighbors = random_points(100)

    >>> distances = get_all_distances(points, neighbors)
    >>> print(distances.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
        neighbor_index  distance
    0                8  0.049319
    0               69  0.064355
    0               88  0.070973
    0               26  0.138782
    0               35  0.159142
    ..             ...       ...
    99              70  1.008302
    99              51  1.017274
    99              10  1.041393
    99              86  1.048825
    99              98  1.054635
    <BLANKLINE>
    [10000 rows x 2 columns]

    Use set_index to get values from other columns.

    >>> neighbors["custom_id"] = np.random.choice([*"abcde"], len(neighbors))
    >>> distances = get_all_distances(points, neighbors.set_index("custom_id"))
    >>> print(distances.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
       neighbor_index  distance
    0               a  0.049319
    0               d  0.064355
    0               c  0.070973
    0               b  0.138782
    0               d  0.159142
    ..            ...       ...
    99              e  1.008302
    99              b  1.017274
    99              e  1.041393
    99              b  1.048825
    99              e  1.054635
    <BLANKLINE>
    [10000 rows x 2 columns]

    Since the index from 'gdf' is preserved, we can join the results with the 'points'.

    >>> joined = points.join(distances)
    >>> print(joined.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
                       geometry neighbor_index  distance
    0   POINT (0.54881 0.67782)              a  0.049319
    0   POINT (0.54881 0.67782)              d  0.064355
    0   POINT (0.54881 0.67782)              c  0.070973
    0   POINT (0.54881 0.67782)              b  0.138782
    0   POINT (0.54881 0.67782)              d  0.159142
    ..                      ...            ...       ...
    99   POINT (0.0047 0.43442)              e  1.008302
    99   POINT (0.0047 0.43442)              b  1.017274
    99   POINT (0.0047 0.43442)              e  1.041393
    99   POINT (0.0047 0.43442)              b  1.048825
    99   POINT (0.0047 0.43442)              e  1.054635
    <BLANKLINE>
    [10000 rows x 3 columns]

    Or assign aggregated values onto the points.

    >>> points["mean_distance"] = distances.groupby(level=0)["distance"].mean()
    >>> points["min_distance"] = distances.groupby(level=0)["distance"].min()
    >>> print(points.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
                       geometry  mean_distance  min_distance
    0   POINT (0.54881 0.67782)       0.433036      0.049319
    1   POINT (0.71519 0.27001)       0.454673      0.049724
    2   POINT (0.60276 0.73519)       0.462998      0.066877
    3   POINT (0.54488 0.96219)       0.594874      0.056894
    4   POINT (0.42365 0.24875)       0.439790      0.050622
    ..                      ...            ...           ...
    95  POINT (0.18319 0.49046)       0.483288      0.080965
    96  POINT (0.58651 0.22741)       0.443947      0.045408
    97  POINT (0.02011 0.25436)       0.624337      0.052075
    98  POINT (0.82894 0.05803)       0.607370      0.082215
    99   POINT (0.0047 0.43442)       0.600952      0.033732
    <BLANKLINE>
    [100 rows x 3 columns]
    """
    return get_k_nearest_neighbors(
        gdf=gdf,
        neighbors=neighbors,
        k=len(neighbors),
    )


def sjoin_within_distance(
    gdf: GeoDataFrame | GeoSeries,
    neighbors: GeoDataFrame | GeoSeries,
    distance: int | float,
    distance_col: str = "distance",
    **kwargs,
) -> GeoDataFrame:
    """Sjoin with a buffer on the right GeoDataFrame and adds a distance column."""
    new_neighbor_cols = {"__left_range_idx": range(len(neighbors))}
    if distance:
        new_neighbor_cols[neighbors._geometry_column_name] = lambda x: x.buffer(
            distance
        )

    # using assign to get a copy
    neighbors = neighbors.assign(**new_neighbor_cols)

    out = gdf.sjoin(neighbors, **kwargs)

    out[distance_col] = shapely.distance(
        out.geometry.values, neighbors.geometry.iloc[out["__left_range_idx"]].values
    )

    return out.drop(columns="__left_range_idx")


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

    Examples:
    ---------
    Make some random points.

    >>> from sgis import get_k_nearest_neighbors, random_points
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> points = random_points(100)
    >>> neighbors = random_points(100)

    Get 10 nearest neighbors.

    >>> distances = get_k_nearest_neighbors(points, neighbors, k=10)
    >>> print(distances.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
        neighbor_index  distance
    0                8  0.049319
    0               69  0.064355
    0               88  0.070973
    0               26  0.138782
    0               35  0.159142
    ..             ...       ...
    99              48  0.200319
    99               5  0.211638
    99              47  0.219490
    99              14  0.238166
    99              24  0.249852
    <BLANKLINE>
    [1000 rows x 2 columns]

    Use set_index to use another column as identifier for the neighbors.

    >>> neighbors["custom_id"] = [letter for letter in [*"abcde"] for _ in range(20)]
    >>> distances = get_k_nearest_neighbors(points, neighbors.set_index("custom_id"), k=10)
    >>> print(distances.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
       neighbor_index  distance
    0               a  0.049319
    0               d  0.064355
    0               e  0.070973
    0               b  0.138782
    0               b  0.159142
    ..            ...       ...
    99              c  0.200319
    99              a  0.211638
    99              c  0.219490
    99              a  0.238166
    99              b  0.249852
    <BLANKLINE>
    [1000 rows x 2 columns]

    The index from 'points' is preserved. Use join to get the distance and neighbor
    index onto the 'points' GeoDataFrame.

    >>> joined = points.join(distances)
    >>> joined["k"] = joined.groupby(level=0)["distance"].transform("rank")
    >>> print(joined.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
                       geometry neighbor_index  distance     k
    0   POINT (0.54881 0.67782)              a  0.049319   1.0
    0   POINT (0.54881 0.67782)              d  0.064355   2.0
    0   POINT (0.54881 0.67782)              e  0.070973   3.0
    0   POINT (0.54881 0.67782)              b  0.138782   4.0
    0   POINT (0.54881 0.67782)              b  0.159142   5.0
    ..                      ...            ...       ...   ...
    99   POINT (0.0047 0.43442)              c  0.200319   6.0
    99   POINT (0.0047 0.43442)              a  0.211638   7.0
    99   POINT (0.0047 0.43442)              c  0.219490   8.0
    99   POINT (0.0047 0.43442)              a  0.238166   9.0
    99   POINT (0.0047 0.43442)              b  0.249852  10.0
    <BLANKLINE>
    [1000 rows x 4 columns]

    Or assign aggregated values directly onto the points.

    >>> points["mean_distance"] = distances.groupby(level=0)["distance"].mean()
    >>> points["min_distance"] = distances.groupby(level=0)["distance"].min()
    >>> print(points.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
                       geometry  mean_distance  min_distance
    0   POINT (0.54881 0.67782)       0.140596      0.049319
    1   POINT (0.71519 0.27001)       0.119401      0.049724
    2   POINT (0.60276 0.73519)       0.136178      0.066877
    3   POINT (0.54488 0.96219)       0.180756      0.056894
    4   POINT (0.42365 0.24875)       0.142112      0.050622
    ..                      ...            ...           ...
    95  POINT (0.18319 0.49046)       0.156066      0.080965
    96  POINT (0.58651 0.22741)       0.121569      0.045408
    97  POINT (0.02011 0.25436)       0.165561      0.052075
    98  POINT (0.82894 0.05803)       0.150797      0.082215
    99   POINT (0.0047 0.43442)       0.165959      0.033732
    <BLANKLINE>
    [100 rows x 3 columns]
    """
    if not len(gdf) or not len(neighbors):
        return DataFrame(columns=["neighbor_index", "distance"])

    if gdf.crs != neighbors.crs:
        raise ValueError("crs mismatch:", gdf.crs, "and", neighbors.crs)

    if get_geom_type(gdf) != "point" or get_geom_type(neighbors) != "point":
        raise ValueError("Geometries must be points")

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
    """Finds nearest neighbors for an array of coordinates to another array of coordinates.

    Args:
        from_array: Numpy array of coordinates.
        to_array: Numpy array of coordinates.
        k: Number of neighbors to find.
        strict: If True (not default), an exception is raised
            if k is larger than the length of 'to_array'.

    Returns a tuple of arrays, one with distances and one with indices
        of the neighbors.

    """
    from sklearn.neighbors import NearestNeighbors

    if not len(to_array) or not len(from_array):
        return np.array([]), np.array([])

    if k is None:
        k = len(to_array)

    if not strict:
        k = k if len(to_array) >= k else len(to_array)

    nbr = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(to_array)
    distances, indices = nbr.kneighbors(from_array)

    return distances, indices


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
    row_indices = np.arange(len(indices)).reshape(-1, 1)
    return np.stack((np.broadcast_to(row_indices, indices.shape), indices), axis=-1)

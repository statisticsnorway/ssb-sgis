"""Get neighbors and K-nearest neighbors."""
import warnings

import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame
from sklearn.neighbors import NearestNeighbors

from ..helpers import return_two_vals
from .general import coordinate_array


def get_neighbors(
    gdf: GeoDataFrame | GeoSeries,
    neighbors: GeoDataFrame | GeoSeries,
    id_col: str = "index",
    max_dist: int = 0,
) -> list[str]:
    """Returns a list of a GeoDataFrame's neigbours.

    Finds all the geometries in 'neighbors' that intersect with 'gdf'. If
    max_dist is specified, neighbors

    Args:
        gdf: GeoDataFrame or GeoSeries
        neighbors: GeoDataFrame or GeoSeries
        id_col: Optionally a column in the GeoDataFrame to use as identifier for the
            neighbors. Defaults to the index of the GeoDataFrame.
        max_dist: The maximum distance between the two geometries. Defaults to 0.

    Returns:
        A list of unique values from the id_col column in the joined dataframe.

    Raises:
        ValueError: If gdf and neighbors do not have the same coordinate reference
            system.

    Examples
    --------
    >>> from gis_utils import get_neighbors, to_gdf
    >>> points = to_gdf([(0, 0), (0.5, 0.5), (2, 2)])
    >>> points
                    geometry
    0  POINT (0.00000 0.00000)
    1  POINT (0.50000 0.50000)
    2  POINT (2.00000 2.00000)
    >>> p1 = points.iloc[[0]]
    >>> get_neighbors(p1, points)
    [0]
    >>> get_neighbors(p1, points, max_dist=1)
    [0, 1]
    >>> get_neighbors(p1, points, max_dist=3)
    [0, 1, 2]

    The pandas index is used by default, but an id column can be specified.

    >>> points["id_col"] = [*"abc"]
    >>> points
                    geometry id_col
    0  POINT (0.00000 0.00000)      a
    1  POINT (0.50000 0.50000)      b
    2  POINT (2.00000 2.00000)      c
    >>> get_neighbors(p1, points, id_col="id_col")
    ['a']
    >>> get_neighbors(p1, points, max_dist=1, id_col="id_col")
    ['a', 'b']
    >>> get_neighbors(p1, points, max_dist=3, id_col="id_col")
    ['a', 'b', 'c']
    """

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

    joined = gdf.sjoin(neighbors, how="inner")

    return [x for x in joined[id_col].unique()]


def get_k_nearest_neighbors(
    gdf: GeoDataFrame,
    neighbors: GeoDataFrame,
    k: int,
    id_cols: str | tuple[str, str] = ("gdf_idx", "neighbour_idx"),
    max_dist: int | float | None = None,
    min_dist: float | int = 0,
    strict: bool = False,
) -> DataFrame:
    """Finds the k nearest neighbors for a GeoDataFrame of points.

    Uses the K-nearest neighbors algorithm method from sklearn.neighbors to find the
    given number of neighbors for each point in 'gdf'.

    Args:
        gdf: a GeoDataFrame of points
        neighbors: a GeoDataFrame of points
        k: number of neighbors to find
        id_cols: column(s) to use as identifiers. Either a string if one column or a
            tuple/list for 'gdf' and 'neighbors' respectfully. Defaults to "gdf_idx"
            and "neighbour_idx".
        max_dist: if specified, rows with greater distances than max_dist will be
            removed.
        min_dist: The minimum distance for points to be considered neighbors. Defaults to
            0, meaning identical points aren't considered neighbors.
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

    >>> from gis_utils import get_k_nearest_neighbors, random_points
    >>> points = random_points(10)
    >>> neighbors = random_points(10)
    >>> get_k_nearest_neighbors(points, neighbors, k=10)
        gdf_idx  neighbour_idx      dist  dist_min     k
    0         0              9  0.329634  0.329634   1.0
    1         0              6  0.457103  0.329634   2.0
    2         0              0  0.763695  0.329634   3.0
    3         0              8  0.788817  0.329634   4.0
    4         0              5  0.800746  0.329634   5.0
    ..      ...            ...       ...       ...   ...
    95        9              6  0.509254  0.106837   6.0
    96        9              8  0.535454  0.106837   7.0
    97        9              9  0.663042  0.106837   8.0
    98        9              1  0.749489  0.106837   9.0
    99        9              7  0.841321  0.106837  10.0

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
       from_id  to_id      dist  dist_min     k
    0        a      9  0.273176  0.273176   1.0
    1        a      6  0.426856  0.273176   2.0
    2        a      8  0.685729  0.273176   3.0
    3        a      5  0.729792  0.273176   4.0
    4        a      0  0.736613  0.273176   5.0
    ..     ...    ...       ...       ...   ...
    95       j      8  0.398965  0.034901   6.0
    96       j      6  0.419967  0.034901   7.0
    97       j      9  0.564396  0.034901   8.0
    98       j      1  0.620475  0.034901   9.0
    99       j      7  0.723542  0.034901  10.0

    [100 rows x 5 columns]

    Setting maximum and minimum distance.

    >>> get_k_nearest_neighbors(
    ...      points,
    ...      neighbors,
    ...      k=10,
    ...      max_dist=0.5,
    ...      min_dist=0.4,
    ... )
       gdf_idx  neighbour_idx      dist  dist_min    k
    0        0              6  0.457103  0.457103  1.0
    1        1              8  0.464321  0.464321  1.0
    2        4              6  0.427395  0.427395  1.0
    3        5              4  0.407684  0.407684  1.0
    4        5              3  0.482697  0.407684  2.0
    5        6              4  0.435376  0.435376  1.0
    6        6              5  0.461744  0.435376  2.0
    7        8              9  0.400464  0.400464  1.0

    Note that the 'k' value is calculated based on the remaining distances after
    filtering out distances below 'min_dist'.
    """
    if gdf.crs != neighbors.crs:
        raise ValueError("crs mismatch:", gdf.crs, "and", neighbors.crs)

    id_col1, id_col2 = return_two_vals(id_cols)
    if not id_col1:
        id_col1 = "gdf_idx"
        gdf[id_col1] = range(len(gdf))
    if not id_col2:
        id_col2 = "neighbour_idx", "gdf_idx"
        neighbors[id_col2] = range(len(neighbors))

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

    if max_dist is not None:
        condition = (dists <= max_dist) & (dists > min_dist)
    else:
        condition = dists > min_dist

    edges = edges[condition]
    if len(edges.shape) == 3:
        edges = edges[0]

    dists = dists[condition]

    df = DataFrame(edges, columns=[id_col1, id_col2])

    df = df.assign(
        dist=dists,
        dist_min=lambda df: df.groupby(id_col1)["dist"].transform("min"),
        k=lambda df: df.groupby(id_col1)["dist"].transform("rank"),
    )

    df[id_col1] = df[id_col1].map(id_dict_gdf)
    df[id_col2] = df[id_col2].map(id_dict_neighbors)

    return df


def k_nearest_neighbors(
    from_array: np.ndarray[np.ndarray[float]],
    to_array: np.ndarray[np.ndarray[float]],
    k: int,
    strict: bool = False,
) -> tuple[np.ndarray[float], np.ndarray[int]]:
    """Finds the k nearest neighbors for arrays of points.

    Uses the K-nearest neighbors method from sklearn.neighbors to find the
    given number of neighbors in 'to_array' for each point in 'from_array'.

    Args:
        from_array: an np.ndarray of coordinates
        to_array: an np.ndarray of coordinates
        k: number of neighbors to find
        strict: If True, will raise an error if 'k' is greater than the number
            of points in 'to_array'. If False, will return all distances if there
            is less than k points in 'to_array'. Defaults to False.

    Returns:
        The distances and indices of the nearest neighbors. Both distances and
        neighbors are np.ndarrays.
    """
    if not strict:
        k = k if len(to_array) >= k else len(to_array)

    nbr = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(to_array)
    dists, indices = nbr.kneighbors(from_array)
    return dists, indices


def _get_edges(gdf: GeoDataFrame, indices: np.ndarray[int]) -> np.ndarray[tuple[int]]:
    """Takes a GeoDataFrame and a list of indices, and returns a list of edges.

    Args:
        gdf (GeoDataFrame): GeoDataFrame
        indices (np.ndarray[float]): a numpy array of the indices of the nearest
            neighbors for each point in the GeoDataFrame.

    Returns:
      A numpy array of edge tuples (from-to indices).
    """
    return np.array(
        [[(i, neighbor) for neighbor in indices[i]] for i in range(len(gdf))]
    )


def get_neighbours(
    gdf: GeoDataFrame | GeoSeries,
    neighbours: GeoDataFrame | GeoSeries,
    id_col: str = "index",
    max_dist: int = 0,
) -> list[str]:
    """American alias for get_neighbors."""
    return get_neighbors(gdf, neighbours, id_col, max_dist)


def get_k_nearest_neighbours(
    gdf: GeoDataFrame,
    neighbours: GeoDataFrame,
    k: int,
    id_cols: str | tuple[str, str] | None = None,
    min_dist: int | float = 0,
    max_dist: int | None = None,
    strict: bool = False,
) -> DataFrame:
    """American alias of get_k_nearest_neighbors."""
    return get_k_nearest_neighbors(
        gdf=gdf,
        neighbors=neighbours,
        k=k,
        id_cols=id_cols,
        min_dist=min_dist,
        max_dist=max_dist,
        strict=strict,
    )


def k_nearest_neighbours(
    from_array: np.ndarray[np.ndarray[float]],
    to_array: np.ndarray[np.ndarray[float]],
    k: int,
    strict: bool = False,
) -> tuple[np.ndarray[float], np.ndarray[int]]:
    """American alias of k_nearest_neighbors."""
    return k_nearest_neighbors(
        from_array=from_array,
        to_array=to_array,
        k=k,
        strict=strict,
    )

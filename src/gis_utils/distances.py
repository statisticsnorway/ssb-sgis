import numpy as np
from geopandas import GeoDataFrame
from pandas import DataFrame
from sklearn.neighbors import NearestNeighbors


def get_k_nearest_neighbours(
    gdf: GeoDataFrame,
    neighbours: GeoDataFrame,
    k: int,
    id_cols: str | list[str, str] | tuple[str, str] | None = None,
    min_dist: int = 0.0000001,
    max_dist: int | None = None,
    strict: bool = False,
) -> DataFrame:
    """Finds the k nearest neighbours for a GeoDataFrame of points

    Uses the K-nearest neighbors algorithm method from sklearn.neighbors to find the
    given number of neighbours for each point in 'gdf'.

    Args:
        gdf: a GeoDataFrame of points
        neighbours: a GeoDataFrame of points
        k: number of neighbours to find
        id_cols: column(s) to use as identifiers. Either a string if one column or a
            tuple/list for 'gdf' and 'neighbours' respectfully.
        min_dist: The minimum distance between the two points. Defaults to
            0.0000001, so that identical points aren't considered neighbours.
        max_dist: if specified, distances larger than this number will be removed.
        strict: If False (the default), no exception is raised if k is larger than the
            number of points in 'neighbours'. If True, 'k' must be less than or equal
            to the number of points in 'neighbours'.

    Returns:
        A DataFrame with id columns and the distance from gdf to neighbour. Also
        includes columns for the minumum distance for each point in 'gdf' and
        the rank.
    """

    if gdf.crs != neighbours.crs:
        raise ValueError("crs mismatch:", gdf.crs, "and", neighbours.crs)

    if id_cols:
        id_col1, id_col2 = _return_two_id_cols(id_cols)
        id_dict_gdf = {i: col for i, col in zip(range(len(gdf)), gdf[id_col1])}
        id_dict_neighbours = {
            i: col for i, col in zip(range(len(neighbours)), neighbours[id_col2])
        }
    else:
        id_col1, id_col2 = "gdf_idx", "neighbours_idx"

    gdf_array = coordinate_array(gdf)
    neighbours_array = coordinate_array(neighbours)

    dists, neighbor_indices = k_nearest_neighbours(
        gdf_array, neighbours_array, k, strict
    )

    edges = _get_edges(gdf, neighbor_indices)

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
        df[id_col2] = df[id_col2].map(id_dict_neighbours)

    return df


def get_k_nearest_neighbors(
    gdf: GeoDataFrame,
    neighbors: GeoDataFrame,
    k: int,
    id_cols: str | list[str, str] | tuple[str, str] | None = None,
    min_dist: int = 0.0000001,
    max_dist: int | None = None,
    strict: bool = False,
) -> DataFrame:
    """American alias of get_k_nearest_neighbours."""
    return get_k_nearest_neighbours(
        gdf=gdf,
        neighbours=neighbors,
        k=k,
        id_cols=id_cols,
        min_dist=min_dist,
        max_dist=max_dist,
        strict=strict,
    )


def coordinate_array(gdf: GeoDataFrame) -> np.ndarray[np.ndarray[float]]:
    """Takes a GeoDataFrame of point geometries and turns it into a 2d ndarray
    of coordinates.

    Args:
        gdf: GeoDataFrame of point geometries

    Returns:
        np.ndarray of np.ndarrays of coordinates
    """
    return np.array([(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)])


def k_nearest_neighbours(
    from_array: np.ndarray[np.ndarray[float]],
    to_array: np.ndarray[np.ndarray[float]],
    k: int,
    strict: bool = False,
) -> tuple[np.ndarray[float]]:
    """Finds the k nearest neighbours for an array of points

    Uses the K-nearest neighbors algorithm method from sklearn.neighbors to find the
    given number of neighbours for each point in 'from_array'.

    Args:
        gdf: an np.ndarray of coordinates
        neighbours: an np.ndarray of coordinates
        k: number of neighbours to find
        strict: If True, will raise an error if 'k' is greater than the number
            of points in 'to_array'. If False, will return all distances if there
            is less than k points in 'to_array'. Defaults to False.

    Returns:
        The distances and indices of the nearest neighbors. Both distances and neighbours
        are np.ndarrays.
    """

    if not strict:
        k = k if len(to_array) >= k else len(to_array)

    nbr = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(to_array)
    dists, indices = nbr.kneighbors(from_array)
    return dists, indices


def k_nearest_neighbors(
    from_array: np.ndarray[np.ndarray[float]],
    to_array: np.ndarray[np.ndarray[float]],
    k: int,
    strict: bool = False,
) -> tuple[np.ndarray[float]]:
    """American alias of k_nearest_neighbours."""
    return k_nearest_neighbours(
        from_array=from_array,
        to_array=to_array,
        k=k,
        strict=strict,
    )


def _get_edges(gdf: GeoDataFrame, indices: np.ndarray[float]) -> np.ndarray[tuple[int]]:
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


def _return_two_id_cols(id_cols: str | list[str, str] | tuple[str, str]) -> tuple[str]:
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

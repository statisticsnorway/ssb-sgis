import numpy as np
from sklearn.neighbors import NearestNeighbors
from geopandas import GeoDataFrame
from pandas import DataFrame


def coordinate_array(gdf: GeoDataFrame) -> np.ndarray[np.ndarray[float]]:
    """Takes a GeoDataFrame of point geometries and turns it into a 2d ndarray 
    of coordinates.
    """
    return np.array(
        [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
    )


def k_nearest_neigbors(
    from_array: np.ndarray[np.ndarray[float]], 
    to_array: np.ndarray[np.ndarray[float]],
    k: int,
    strict: bool = False,
    ) -> tuple[np.ndarray[float]]:
    """Given a set of points, find the k nearest neighbors of each point in another set of points.
    
    Args:
      from_array (np.ndarray[np.ndarray[float]]): The array of points (coordinate tuples) you want to find the nearest
    neighbors for.
      to_array (np.ndarray[np.ndarray[float]]): The array of points that we want to find the nearest
    neighbors of.
      k (int): the number of nearest neighbors to find.
      strict (bool): If True, will raise an error if k is greater than the number of points in to_array.
    If False, will return all distances if there is less than k points in to_array.
    Defaults to False
    
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
      indices (np.ndarray[float]): a numpy array of the indices of the nearest neighbors for each point
    in the GeoDataFrame.
    
    Returns:
      A numpy array of edge tuples (from-to indices).
    """
    return np.array(
        [
            [(i, neighbor) for neighbor in indices[i]]
            for i in range(len(gdf))
        ]
    )


def get_k_nearest_neighbors(
    gdf: GeoDataFrame,
    neighbors: GeoDataFrame, 
    k: int, 
    id_cols: str | list[str, str] | tuple[str, str] | None = None,
    min_dist: int = 0.0001,
    max_dist: int | None = None,
    strict: bool = False,
    ) -> DataFrame:
    """ 
    It takes a GeoDataFrame of points, a GeoDataFrame of neighbors, and a number of neighbors to find,
    and returns a DataFrame of the k nearest neighbors for each point in the GeoDataFrame
    
    Args:
      gdf: a GeoDataFrame of points
      neighbors: a GeoDataFrame of points
      k (int): number of neighbors to find
      id_cols: one or two column names (strings) 
      min_dist (int): The minimum distance between the two points. Defaults to 0.0001 so that identical points 
    arent considered neighbors. 
      max_dist: if specified, distances larger than this number will be removed.
      strict (bool): If True, will raise an error if k is greater than the number of points in to_array.
    If False, will return all distances if there is less than k points in to_array.
    Defaults to False
    
    Returns:
      A DataFrame with the following columns:
    """
    if id_cols:
        id_col1, id_col2 = return_two_id_cols(id_cols)
        id_dict_gdf = {i: col for i, col in zip(range(len(gdf)), gdf[id_col1])}
        id_dict_neighbors = {i: col for i, col in zip(range(len(neighbors)), neighbors[id_col2])}
    else:
        id_col1, id_col2 = "gdf_idx", "neighbors_idx"
    
    gdf_array = coordinate_array(gdf)
    neighbors_array = coordinate_array(neighbors)

    dists, neighbor_indices = k_nearest_neigbors(gdf_array, neighbors_array, k, strict)

    edges = get_edges(gdf, neighbor_indices)
    
    if max_dist:
        condition = (dists <= max_dist) & (dists > min_dist)
    else:
        condition = (dists > min_dist)

    edges = edges[condition]
    if len(edges.shape) == 3:
        edges = edges[0]

    dists = dists[condition]
    
    if id_col1 == id_col2:
        id_col2 = id_col2 + "2"

    df = DataFrame(edges, columns=[id_col1, id_col2])

    df = (df.assign(
            dist=dists,
            dist_min=lambda df: df.groupby(id_col1)["dist"].transform("min")
        )
    )

    if id_cols:
        df[id_col1] = df[id_col1].map(id_dict_gdf)
        df[id_col2] = df[id_col2].map(id_dict_neighbors)

    return df


def return_two_id_cols(
    id_cols: str | list[str, str] | tuple[str, str]
    ) -> tuple[str]:
    """
    Make sure the id_cols are a 2 length tuple.> If the input is a string, return a tuple of two strings. If the input is a list or tuple of two
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


def main():
    import geopandas as gpd
    p = gpd.read_parquet(r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\tilfeldige_adresser_10000.parquet")
    p["temp_idx"] = p.index

#    r = gpd.read_parquet(r"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/vegdata/veger_landet_2022.parquet")
    r = gpd.read_parquet(r"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/vegdata/veger_oslo_og_naboer_2022.parquet")
    r = r.to_crs(p.crs)
    r["road_idx"] = r.index
    r["geometry"] = r.centroid
    
    print(
        get_k_nearest_neighbors(p, r, id_cols=("temp_idx", "road_idx"), k=100)#, max_dist=500)
        )

    print(
        p.sjoin_nearest(r, distance_col="dist")
        )


if __name__ == "__main__":
    import cProfile
    cProfile.run("main()", sort="cumtime")    
    #main()
    
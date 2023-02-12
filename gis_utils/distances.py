import numpy as np
from sklearn.neighbors import NearestNeighbors
from geopandas import GeoDataFrame
from pandas import DataFrame


def coordinate_array(gdf):
    return np.array(
        [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
    )


def k_nearest_neigbors(
    from_array: np.ndarray[np.ndarray[float]], 
    to_array: np.ndarray[np.ndarray[float]],
    k: int,
    ) -> tuple[np.ndarray[float]]:

    nbr = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(from_array)
    dists, indices = nbr.kneighbors(to_array)
    return dists, indices


def make_id_dicts(gdf, neighbors, id_cols):
    if isinstance(id_cols, (tuple, list)) and len(id_cols) == 2:
        id_col1, id_col2 = id_cols
    elif isinstance(id_cols, str):
        id_col1, id_col2 = id_cols, id_cols
    else:
        raise ValueError
    id_dict_gdf = {i: col for i, col in zip(range(len(gdf)), gdf[id_col1])}
    id_dict_neighbors = {i: col for i, col in zip(range(len(neighbors)), neighbors[id_col2])}

    return id_dict_gdf, id_dict_neighbors


def get_k_nearest_neighbors(
    gdf: GeoDataFrame, 
    neighbors: GeoDataFrame, 
    k: int, 
    id_cols: str | list[str, str] | tuple[str, str] | None = None,
    min_dist: int = 0.00000001,
    max_dist: int = 1_000_000_000,
    dist_factor: int | None = None,
    ) -> DataFrame:

    if id_cols:
        id_dict_gdf, id_dict_neighbors = make_id_dicts(gdf, neighbors, id_cols)
    
    gdf_array = coordinate_array(gdf)
    neighbors_array = coordinate_array(neighbors)

    dists, indices = k_nearest_neigbors(gdf_array, neighbors_array, k)

    edges = get_edges(gdf, indices)

    dists = get_dists(gdf, dists)
    
    edges = edges[(dists <= max_dist) & (dists > min_dist)]
    if len(edges.shape) == 3:
        edges = edges[0]
        
    dists = dists[(dists <= max_dist) & (dists > min_dist)]

    df = DataFrame(edges, columns=["gdf_idx", "neighbor_idx"])
    df = (df.assign(
            dist=dists,
            dist_min=lambda df: df.groupby("gdf_idx")["dist"].transform("min")
        )
    )

    if dist_factor:
        dist_factor_mult = (1 + dist_factor / 100)
        df = df.loc[df.dist <= df.dist_min * dist_factor_mult + dist_factor]

    if id_cols:
        df["gdf_idx"] = df["gdf_idx"].map(id_dict_gdf)
        df["neighbor_idx"] = df["neighbor_idx"].map(id_dict_neighbors)

    return df


def get_edges(gdf, indices):
    return np.array(
        [
            [(i, neighbor) for neighbor in indices[i]]
            for i in range(len(gdf))
        ]
    )


def get_dists(gdf, dists):
    
    return np.array(
        [
            [dist for dist in dists[i]]
            for i in range(len(gdf))
        ]
    )


if __name__ == "__main__":
    import geopandas as gpd
    p = gpd.read_parquet(r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\tilfeldige_adresser_1000.parquet")

    print(
        get_k_nearest_neighbors(p, p, k=100, max_dist=500, dist_factor=0)
        )
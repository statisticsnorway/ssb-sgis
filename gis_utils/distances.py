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
    ) -> tuple[np.ndarray[float]]:

    nbr = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(to_array)
    dists, indices = nbr.kneighbors(from_array)
    return dists, indices
   

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


def get_k_nearest_neighbors(
    gdf: GeoDataFrame, 
    neighbors: GeoDataFrame, 
    k: int, 
    id_cols: str | list[str, str] | tuple[str, str] | None = None,
    min_dist: int = 0.0001,
    max_dist: int | None = None,
    ) -> DataFrame:

    if id_cols:
        id_col1, id_col2 = return_two_id_cols(id_cols)
        id_dict_gdf = {i: col for i, col in zip(range(len(gdf)), gdf[id_col1])}
        id_dict_neighbors = {i: col for i, col in zip(range(len(neighbors)), neighbors[id_col2])}
    else:
        id_col1, id_col2 = "gdf_idx", "neighbors_idx"
    
    gdf_array = coordinate_array(gdf)
    neighbors_array = coordinate_array(neighbors)

    dists, neighbor_indices = k_nearest_neigbors(gdf_array, neighbors_array, k)

    edges = get_edges(gdf, neighbor_indices)

    dists = get_dists(gdf, dists)
    
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

    if isinstance(id_cols, (tuple, list)) and len(id_cols) == 2:
        return id_cols
    elif isinstance(id_cols, str):
        return id_cols, id_cols
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
    
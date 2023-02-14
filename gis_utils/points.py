from geopandas import GeoDataFrame
from pandas import DataFrame
import numpy as np
from .distances import get_k_nearest_neighbors


class Points:
    def __init__(
        self, 
        points: GeoDataFrame,
        crs: str | int,
        temp_idx_start: int,
        id_col: str | list[str, str] | tuple[str, str] | None = None,
    ) -> None:

        self.points = points.to_crs(crs).reset_index(drop=True)
                
        self.id_col = id_col
        self.temp_idx_start = temp_idx_start

    def make_temp_idx(self):

        self.points["temp_idx"] = np.arange(
            start=self.temp_idx_start, 
            stop=self.temp_idx_start+len(self.points)
            )
        self.points["temp_idx"] = self.points["temp_idx"].astype(str)

        if self.id_col:
            self.id_dict = {temp_idx: idx for temp_idx, idx in zip(self.points.temp_idx, self.points[self.id_col])}

    def check_id_col(
        self,
        index: int,
        ) -> None:

        if not self.id_col:
            return

        if isinstance(self.id_col, str):
            pass
        elif isinstance(self.id_col, (list, tuple)) and len(self.id_col) == 2:
            self.id_col = self.id_col[index]
        else:
            raise ValueError("'id_col' should be a string or a list/tuple with two strings.")

        if not self.id_col in self.points.columns:
            raise KeyError(f"'startpoints' has no attribute '{self.id_col}'")

    def n_missing(
        self, 
        results: GeoDataFrame | DataFrame,
        col: str,
        ) -> None:

        self.points["n_missing"] = self.points["temp_idx"].map(
            len(results[col].unique()) - results.dropna().groupby(col).count().iloc[:, 0]
        )
    
    def distance_to_nodes(self, nodes: GeoDataFrame, search_tolerance, search_factor) -> DataFrame:

        df = get_k_nearest_neighbors(
            gdf=self.points, 
            neighbors=nodes, 
            id_cols=("temp_idx", "node_id"),
            k=50, 
            max_dist=search_tolerance, 
            )
        
        search_factor_mult = (1 + search_factor / 100)
        df = df.loc[df.dist <= df.dist_min * search_factor_mult + search_factor]

        return df


class StartPoints(Points):
    def __init__(
        self, 
        points: GeoDataFrame,
        **kwargs,
    ) -> None:

        super().__init__(points, **kwargs)

        self.check_id_col(index=0)
        self.make_temp_idx()
    
    def distance_to_nodes(self, nodes, search_tolerance, search_factor) -> None:
        df = super().distance_to_nodes(nodes, search_tolerance, search_factor)
        
        self.edges = [(temp_idx, node_id) for temp_idx, node_id in zip(df.temp_idx, df.node_id)]
        self.dists = list(df.dist)


class EndPoints(Points):
    def __init__(
        self, 
        points: GeoDataFrame,
        **kwargs,
    ) -> None:

        super().__init__(points, **kwargs)

        self.check_id_col(index=1)
        self.make_temp_idx()
    
    def distance_to_nodes(self, nodes, search_tolerance, search_factor) -> None:
        """Same as for the startpoints, but opposite edge direction"""
        df = super().distance_to_nodes(nodes, search_tolerance, search_factor)
        
        self.edges = [(node_id, temp_idx) for temp_idx, node_id in zip(df.temp_idx, df.node_id)]
        self.dists = list(df.dist)

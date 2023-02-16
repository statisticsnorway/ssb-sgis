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

    def make_temp_idx(self) -> None:
        """Make a temporary id column that is not present in the node ids of the network.
        The original ids are stored in a dict and mapped to the results after the network analysis. 
        This method has to be run after get_id_col, because this determines the id column differently for start- and endpoints. """

        self.points["temp_idx"] = np.arange(
            start=self.temp_idx_start, 
            stop=self.temp_idx_start+len(self.points)
            )
        self.points["temp_idx"] = self.points["temp_idx"].astype(str)

        if self.id_col:
            self.id_dict = {temp_idx: idx for temp_idx, idx in zip(self.points.temp_idx, self.points[self.id_col])}

    def get_id_col(
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

    def get_n_missing(
        self, 
        results: GeoDataFrame | DataFrame,
        col: str,
        ) -> None:

        self.points["n_missing"] = self.points["temp_idx"].map(
            len(results[col].unique()) - results.dropna().groupby(col).count().iloc[:, 0]
        )
    
    def distance_to_nodes(self, nodes: GeoDataFrame, search_tolerance: int, search_factor: int) -> DataFrame:
        """Creates a DataFrame with distances and indices of the 50 closest nodes for each point,
        then keeps only the rows with a distance less than the search_tolerance and the search_factor. 
        """

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

        self.get_id_col(index=0)
        self.make_temp_idx()
    
    def distance_to_nodes(self, nodes: GeoDataFrame, search_tolerance: int, search_factor: int) -> None:
        """First runs the super method, which returns a DataFrame with distances and indices of the startpoints and the nodes,
        then an edgelist (of tuples) and a distance list is created. """
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

        self.get_id_col(index=1)
        self.make_temp_idx()
    
    def distance_to_nodes(self, nodes: GeoDataFrame, search_tolerance: int, search_factor: int) -> None:
        """Same as the method in the StartPoints class, but with opposite edge direction (node_id, point_id). """
        df = super().distance_to_nodes(nodes, search_tolerance, search_factor)
        
        self.edges = [(node_id, temp_idx) for temp_idx, node_id in zip(df.temp_idx, df.node_id)]
        self.dists = list(df.dist)

import numpy as np
from geopandas import GeoDataFrame
from pandas import DataFrame

from .distances import get_k_nearest_neighbors, split_lines_at_closest_point
from .network_functions import make_node_ids
from .networkanalysisrules import NetworkAnalysisRules


class Points:
    def __init__(
        self,
        points: GeoDataFrame,
        temp_idx_start: int,
        id_col: str | list[str, str] | tuple[str, str] | None = None,
    ) -> None:
        self.gdf = points.reset_index(drop=True)

        self.id_col = id_col
        self.temp_idx_start = temp_idx_start

    def _get_id_col(
        self,
        index: int,
    ) -> None:
        if not self.id_col:
            return

        if isinstance(self.id_col, (list, tuple)) and len(self.id_col) == 2:
            self.id_col = self.id_col[index]

        elif not isinstance(self.id_col, str):
            raise ValueError(
                "'id_col' should be a string or a list/tuple with two strings."
            )

        if self.id_col not in self.gdf.columns:
            raise KeyError(f"'startpoints' has no attribute '{self.id_col}'")

    def _make_temp_idx(self) -> None:
        """Make a temporary id column that is not present in the node ids of the network.
        The original ids are stored in a dict and mapped to the results after the network analysis.
        This method has to be run after _get_id_col, because this determines the id column differently for start- and endpoints.
        """

        self.gdf["temp_idx"] = np.arange(
            start=self.temp_idx_start, stop=self.temp_idx_start + len(self.gdf)
        )
        self.gdf["temp_idx"] = self.gdf["temp_idx"].astype(str)

        if self.id_col:
            self.id_dict = {
                temp_idx: idx
                for temp_idx, idx in zip(self.gdf.temp_idx, self.gdf[self.id_col])
            }

    def _get_n_missing(
        self,
        results: GeoDataFrame | DataFrame,
        col: str,
    ) -> None:
        """
        Get number of missing values for each point after a network analysis.

        Args:
            results: resulting (Geo)DataFrame of od_cost_matrix, shortest_path or service_area analysis.
            col: id column of the results. Either 'origin' or 'destination'.

        Returns:
            None, but gives the points.gdf a column n_missing.

        """
        self.gdf["n_missing"] = self.gdf["temp_idx"].map(
            results.groupby(col).count().iloc[:, 0]
            - results.dropna().groupby(col).count().iloc[:, 0]
        )

    def _distance_to_nodes(
        self, nodes: GeoDataFrame, search_tolerance: int, search_factor: int
    ) -> DataFrame:
        """Creates a DataFrame with distances and indices of the 50 closest nodes for each point,
        then keeps only the rows with a distance less than the search_tolerance and the search_factor.
        """

        df = get_k_nearest_neighbors(
            gdf=self.gdf,
            neighbors=nodes,
            id_cols=("temp_idx", "node_id"),
            k=50,
            max_dist=search_tolerance,
        )

        search_factor_mult = 1 + search_factor / 100
        df = df.loc[df.dist <= df.dist_min * search_factor_mult + search_factor]

        return df

    @staticmethod
    def _to_weight(dists, rules):
        """
        Gjør om meter to minutter for lenkene mellom punktene og nabonodene.
        og ganger luftlinjeavstanden med 1.5 siden det alltid er svinger i Norge.
        Gjør ellers ingenting.
        """

        if (
            not rules.weight_to_nodes_dist
            and not rules.weight_to_nodes_kmh
            and not rules.weight_to_nodes_mph
        ):
            return [0 for _ in dists]

        if (
            bool(rules.weight_to_nodes_dist)
            + bool(rules.weight_to_nodes_kmh)
            + bool(rules.weight_to_nodes_mph)
            > 1
        ):
            raise ValueError(
                "Can only specify one of 'weight_to_nodes_dist', 'weight_to_nodes_kmh' and 'weight_to_nodes_mph'"
            )

        if rules.weight_to_nodes_dist and not rules.weight == "meters":
            raise ValueError(
                "Can only specify 'weight_to_nodes_dist' when the 'weight' is meters"
            )

        if rules.weight_to_nodes_kmh:
            return [x / (16.666667 * rules.weight_to_nodes_kmh) for x in dists]

        if rules.weight_to_nodes_mph:
            return [x / (26.8224 * rules.weight_to_nodes_mph) for x in dists]

        return dists

    def _make_edges(self, df, from_col, to_col):
        return [(f, t) for f, t in zip(df[from_col], df[to_col])]

    def _get_edges_and_weights(
        self,
        nodes: GeoDataFrame,
        rules: NetworkAnalysisRules,
        from_col: str,
        to_col: str,
    ):
        df = get_k_nearest_neighbors(
            gdf=self.gdf,
            neighbors=nodes,
            id_cols=("temp_idx", "node_id"),
            k=50,
            max_dist=rules.search_tolerance,
        )

        search_factor_mult = 1 + rules.search_factor / 100
        df = df.loc[df.dist <= df.dist_min * search_factor_mult + rules.search_factor]
        #        df = self._distance_to_nodes(nodes, rules.search_tolerance, rules.search_factor)

        edges = self._make_edges(df, from_col=from_col, to_col=to_col)

        weighs = self._to_weight(dists=list(df.dist), rules=rules)

        return edges, weighs


class StartPoints(Points):
    def __init__(
        self,
        points: GeoDataFrame,
        **kwargs,
    ) -> None:
        super().__init__(points, **kwargs)

        self._get_id_col(index=0)
        self._make_temp_idx()

    def _get_edges_and_weights(self, nodes: GeoDataFrame, rules: NetworkAnalysisRules):
        return super()._get_edges_and_weights(
            nodes, rules, from_col="temp_idx", to_col="node_id"
        )


class EndPoints(Points):
    def __init__(
        self,
        points: GeoDataFrame,
        **kwargs,
    ) -> None:
        super().__init__(points, **kwargs)

        self._get_id_col(index=1)
        self._make_temp_idx()

    def _get_edges_and_weights(self, nodes: GeoDataFrame, rules: NetworkAnalysisRules):
        return super()._get_edges_and_weights(
            nodes, rules, from_col="node_id", to_col="temp_idx"
        )

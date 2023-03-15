import numpy as np
from geopandas import GeoDataFrame
from pandas import DataFrame

from ..geopandas_tools.neighbors import get_k_nearest_neighbors
from ..helpers import return_two_vals
from .networkanalysisrules import NetworkAnalysisRules


"""
These are internal classes used in the NetworkAnalysis class. The classes used in
NetworkAnalysis are Origins and Destinations, which are subclasses of Points. The
Origins and Destinations classes are the same, except that the order of the id_cols
parameter in the NetworkAnalysis methods is different (0 for Origins, 1 for
Destinations) and the order of the edge ids (the edges from the Origins go from origin
to node and the Destination edges go from node to Destination).
"""


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

        id_cols = return_two_vals(self.id_col)

        self.id_col = id_cols[index]
        """
        if isinstance(self.id_col, (list, tuple)) and len(self.id_col) == 2:
            self.id_col = self.id_col[index]

        elif not isinstance(self.id_col, str):
            raise ValueError(
                "'id_col' should be a string or a list/tuple with two strings."
            )
        """
        if self.id_col not in self.gdf.columns:
            raise KeyError(
                f"{self.__class__.__name__!r} has no attribute {self.id_col!r}"
            )

    def _make_temp_idx(self) -> None:
        """Make a temporary id column thad don't overlap with the node ids.

        The original ids are stored in a dict and mapped back to the results in the
        end. This method has to be run after _get_id_col, because this determines the
        id column differently for origins and destinations.
        """

        self.gdf["temp_idx"] = np.arange(
            start=self.temp_idx_start, stop=self.temp_idx_start + len(self.gdf)
        )
        self.gdf["temp_idx"] = self.gdf["temp_idx"].astype(str)

        if self.id_col:
            self.id_dict = {
                temp_idx: idx
                for temp_idx, idx in zip(
                    self.gdf.temp_idx, self.gdf[self.id_col], strict=True
                )
            }

    def _get_n_missing(
        self,
        results: GeoDataFrame | DataFrame,
        col: str,
    ) -> None:
        """
        Get number of missing values for each point after a network analysis.

        Args:
            results: (Geo)DataFrame resulting from od_cost_matrix, get_route,
                get_k_routes, get_route_frequencies or service_area.
            col: id column of the results. Either 'origin' or 'destination'.
        """
        self.gdf["missing"] = self.gdf["temp_idx"].map(
            results.groupby(col).count().iloc[:, 0]
            - results.dropna().groupby(col).count().iloc[:, 0]
        )

    @staticmethod
    def _dist_to_weight(dists, rules):
        """Meters to minutes based on 'weight_to_nodes_' attribute of the rules."""
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
                "Can only specify one of 'weight_to_nodes_dist', 'weight_to_nodes_kmh'"
                " and 'weight_to_nodes_mph'"
            )

        if rules.weight_to_nodes_dist and rules.weight != "meters":
            raise ValueError(
                "Can only specify 'weight_to_nodes_dist' when the 'weight' is meters"
            )

        if rules.weight_to_nodes_kmh:
            return [x / (16.666667 * rules.weight_to_nodes_kmh) for x in dists]

        if rules.weight_to_nodes_mph:
            return [x / (26.8224 * rules.weight_to_nodes_mph) for x in dists]

        return dists

    def _make_edges(self, df, from_col, to_col):
        return [(f, t) for f, t in zip(df[from_col], df[to_col], strict=True)]

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

        edges = self._make_edges(df, from_col=from_col, to_col=to_col)

        weighs = self._dist_to_weight(dists=list(df.dist), rules=rules)

        return edges, weighs


class Origins(Points):
    def __init__(
        self,
        points: GeoDataFrame,
        temp_idx_start: int,
        **kwargs,
    ) -> None:
        super().__init__(points, temp_idx_start, **kwargs)

        self._get_id_col(index=0)
        self._make_temp_idx()

    def _get_edges_and_weights(
        self,
        nodes: GeoDataFrame,
        rules: NetworkAnalysisRules,
        from_col="temp_idx",
        to_col="node_id",
    ):
        return super()._get_edges_and_weights(nodes, rules, from_col, to_col)


class Destinations(Points):
    def __init__(
        self,
        points: GeoDataFrame,
        temp_idx_start: int,
        **kwargs,
    ) -> None:
        super().__init__(points, temp_idx_start, **kwargs)

        self._get_id_col(index=1)
        self._make_temp_idx()

    def _get_edges_and_weights(
        self,
        nodes: GeoDataFrame,
        rules: NetworkAnalysisRules,
        from_col="node_id",
        to_col="temp_idx",
    ):
        return super()._get_edges_and_weights(nodes, rules, from_col, to_col)

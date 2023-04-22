import numpy as np
from geopandas import GeoDataFrame

from ..geopandas_tools.neighbors import get_k_nearest_neighbors
from .networkanalysisrules import NetworkAnalysisRules


"""
These are internal classes used in the NetworkAnalysis class. The classes used in
NetworkAnalysis are Origins and Destinations, which are subclasses of Points. The
Origins and Destinations classes are the same, except that the edges are in
opposite directions.
"""


class Points:
    def __init__(
        self,
        points: GeoDataFrame,
    ) -> None:
        self.gdf = points.copy()

    def _make_temp_idx(self, start: int) -> None:
        """Make a temporary id column thad don't overlap with the node ids.

        The original indices are stored in a dict and mapped back to the results in the
        end.
        """

        self.gdf["temp_idx"] = np.arange(start=start, stop=start + len(self.gdf))
        self.gdf["temp_idx"] = self.gdf["temp_idx"].astype(str)

        self.idx_dict = {
            temp_idx: idx
            for temp_idx, idx in zip(self.gdf.temp_idx, self.gdf.index, strict=True)
        }

    @staticmethod
    def _convert_distance_to_weight(distances, rules):
        """Meters to minutes based on 'weight_to_nodes_' attribute of the rules."""
        if not rules.nodedist_multiplier and not rules.nodedist_kmh:
            return [0 for _ in distances]

        if rules.nodedist_multiplier and rules.nodedist_kmh:
            raise ValueError(
                "Can only specify one of 'nodedist_multiplier' and 'nodedist_kmh'"
            )

        if rules.nodedist_multiplier:
            if rules.weight != "meters":
                raise ValueError(
                    "Can only specify 'nodedist_multiplier' when the 'weight' is meters"
                )
            return [x * rules.nodedist_multiplier for x in distances]

        if rules.nodedist_kmh and rules.weight != "minutes":
            raise ValueError(
                "Can only specify 'nodedist_kmh' when the 'weight' is minutes"
            )

        return [x / (16.666667 * rules.nodedist_kmh) for x in distances]

    def _make_edges(self, df, from_col, to_col):
        return [(f, t) for f, t in zip(df[from_col], df[to_col], strict=True)]

    def _get_edges_and_weights(
        self,
        nodes: GeoDataFrame,
        rules: NetworkAnalysisRules,
        k: int,
        from_col: str,
        to_col: str,
    ):
        distances = get_k_nearest_neighbors(
            gdf=self.gdf.set_index("temp_idx"),
            neighbors=nodes.set_index("node_id"),
            k=k,
        )

        distances["distance_min"] = distances.groupby(level=0)["distance"].min()

        distances = distances.reset_index()

        search_factor_multiplier = 1 + rules.search_factor / 100
        distances = distances.loc[
            lambda df: (df.distance <= rules.search_tolerance)
            & (
                df.distance
                <= df.distance_min * search_factor_multiplier + rules.search_factor
            )
        ]

        edges = self._make_edges(distances, from_col=from_col, to_col=to_col)

        weighs = self._convert_distance_to_weight(
            distances=list(distances.distance), rules=rules
        )

        return edges, weighs


class Origins(Points):
    def __init__(
        self,
        points: GeoDataFrame,
    ) -> None:
        super().__init__(points)

    def _get_edges_and_weights(
        self,
        nodes: GeoDataFrame,
        rules: NetworkAnalysisRules,
        k: int,
        from_col="temp_idx",
        to_col="neighbor_index",
    ):
        """Note: opposite order as in Destinations."""
        return super()._get_edges_and_weights(nodes, rules, k, from_col, to_col)


class Destinations(Points):
    def __init__(
        self,
        points: GeoDataFrame,
    ) -> None:
        super().__init__(points)

    def _get_edges_and_weights(
        self,
        nodes: GeoDataFrame,
        rules: NetworkAnalysisRules,
        k: int,
        from_col="neighbor_index",
        to_col="temp_idx",
    ):
        """Note: opposite order as in Origins."""
        return super()._get_edges_and_weights(nodes, rules, k, from_col, to_col)

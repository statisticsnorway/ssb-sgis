from datetime import datetime
from typing import Tuple

import igraph
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from igraph import Graph
from pandas import DataFrame

from .directednetwork import DirectedNetwork
from .distances import split_lines_at_closest_point
from .geopandas_utils import gdf_concat, push_geom_col
from .network import Network
from .network_functions import make_node_ids
from .networkanalysisrules import NetworkAnalysisRules
from .od_cost_matrix import od_cost_matrix
from .points import EndPoints, StartPoints
from .service_area import service_area
from .shortest_path import shortest_path


class NetworkAnalysis:
    """Class that holds the actual network analysis methods.

    Args:
        network: either the base Network class or a subclass, chiefly the DirectedNetwork class.
            The network should be customized beforehand, but can also be accessed through
            the 'network' attribute of this class.
        weight: e.i. 'minutes' or 'meters'. Or custom numeric column.
        search_tolerance: meters.
        search_factor: .
        weight_to_nodes: .

    Example:

    roads = gpd.GeoDataFrame(filepath_roads)
    points = gpd.GeoDataFrame(filepath_points)

    # the data should have crs with meters as units, e.g. UTM:
    roads = roads.to_crs(25833)
    points = points.to_crs(25833)

    nw = (
        DirectedNetwork(roads)
        .make_directed_network_osm()
        .remove_isolated()
        )

    nwa = NetworkAnalysis(nw, weight="minutes")

    od = nwa.od_cost_matrix(p, p)

    """

    def __init__(
        self,
        network: Network | DirectedNetwork,
        rules: NetworkAnalysisRules,
    ):
        self.network = network
        self.rules = rules

        if not isinstance(rules, NetworkAnalysisRules):
            raise ValueError(
                f"'rules' should be of type NetworkAnalysisRules. Got {type(rules)}"
            )

        if not isinstance(network, (Network, DirectedNetwork)):
            raise ValueError(
                f"'network' should of type DirectedNetwork or Network. Got {type(network)}"
            )

        self.network.gdf = self.rules._validate_weight(
            self.network.gdf, raise_error=False
        )

        if isinstance(self.network, DirectedNetwork):
            self.network._warn_if_not_directed()

        self._update_point_wkts()
        self.rules._update_rules()

    def od_cost_matrix(
        self,
        startpoints: GeoDataFrame,
        endpoints: GeoDataFrame,
        id_col: str | Tuple[str, str] | None = None,
        lines: bool = False,
        **kwargs,
    ) -> DataFrame | GeoDataFrame:
        self._prepare_network_analysis(startpoints, endpoints, id_col)

        results = od_cost_matrix(
            graph=self.graph,
            startpoints=self.startpoints.gdf,
            endpoints=self.endpoints.gdf,
            weight=self.rules.weight,
            lines=lines,
            **kwargs,
        )

        self.startpoints._get_n_missing(results, "origin")
        self.endpoints._get_n_missing(results, "destination")

        if id_col:
            results["origin"] = results["origin"].map(self.startpoints.id_dict)
            results["destination"] = results["destination"].map(self.endpoints.id_dict)

        if lines:
            results = push_geom_col(results)

        self._runlog("od_cost_matrix", results, **kwargs)

        return results

    def shortest_path(
        self,
        startpoints: GeoDataFrame,
        endpoints: GeoDataFrame,
        id_col: str | Tuple[str, str] | None = None,
        summarise: bool = False,
        **kwargs,
    ) -> GeoDataFrame:
        self._prepare_network_analysis(startpoints, endpoints, id_col)

        results = shortest_path(
            graph=self.graph,
            startpoints=self.startpoints.gdf,
            endpoints=self.endpoints.gdf,
            weight=self.rules.weight,
            roads=self.network.gdf,
            summarise=summarise,
            **kwargs,
        )

        if not summarise:
            self.startpoints._get_n_missing(results, "origin")
            self.endpoints._get_n_missing(results, "destination")

        if id_col and not summarise:
            results["origin"] = results["origin"].map(self.startpoints.id_dict)
            results["destination"] = results["destination"].map(self.endpoints.id_dict)

        results = push_geom_col(results)

        self._runlog("shortest_path", results, **kwargs)

        return results

    def service_area(
        self, startpoints: GeoDataFrame, id_col: str | None = None, **kwargs
    ) -> GeoDataFrame:
        self._prepare_network_analysis(startpoints, id_col=id_col)

        results = service_area(
            self.graph,
            self.startpoints.gdf,
            self.rules.weight,
            self.network.gdf,
            **kwargs,
        )

        if id_col:
            results[id_col] = results["origin"].map(self.startpoints.id_dict)
            results = results.drop("origin", axis=1)

        results = push_geom_col(results)

        self._runlog("service_area", results, **kwargs)

        return results

    def __repr__(self) -> str:
        # remove 'weight_to_nodes_' arguments in the repr of the NetworkAnalysisRules instance
        rules = self.rules.__repr__()
        for txt in ["weight_to_nodes_", "dist", "kmh", "mph", "=None", "=False"]:
            rules = rules.replace(txt, "")
        for txt in [", )"] * 5:
            rules = rules.replace(txt, ")")
        rules = rules.strip(")")

        # add a 'weight_to_nodes_' argument if used,
        # else inform that there are more with some '...'
        if self.rules.weight_to_nodes_dist:
            x = f", weight_to_nodes_dist={self.rules.weight_to_nodes_dist}"
        elif self.rules.weight_to_nodes_kmh:
            x = f", weight_to_nodes_dist={self.rules.weight_to_nodes_kmh}"
        elif self.rules.weight_to_nodes_mph:
            x = f", weight_to_nodes_dist={self.rules.weight_to_nodes_mph}"
        else:
            x = ", ..."

        return (
            f"{self.__class__.__name__}("
            f"network={self.network.__repr__()}, "
            f"rules={rules}{x}))"
        )

    def _log_df_template(self, fun: str) -> DataFrame:
        """
        The 'isolated_removed' column does not account for
        preperation done before initialising the (Directed)Network class.
        """

        if not hasattr(self, "log"):
            self.log = DataFrame()

        df = DataFrame(
            {
                "endtime": pd.to_datetime(datetime.now()).floor("S").to_pydatetime(),
                "function": fun,
                "percent_missing": np.nan,
                "cost_mean": np.nan,
                "n_startpoints": np.nan,
                "n_endpoints": np.nan,
                "isolated_removed": self.network._isolated_removed,
                "percent_directional": self.network._percent_directional,
            },
            index=[0],
        )

        for key, value in self.rules.__dict__.items():
            if key.startswith("_") or key.endswith("_"):
                continue
            df = pd.concat([df, pd.DataFrame({key: [value]})], axis=1)

        return df

    def _runlog(self, fun: str, results: DataFrame | GeoDataFrame, **kwargs) -> None:
        df = self._log_df_template(fun)

        df["n_startpoints"] = len(self.startpoints.gdf)

        if self.rules.weight in results.columns:
            df["percent_missing"] = results[self.rules.weight].isna().mean() * 100
            df["cost_mean"] = results[self.rules.weight].mean()

        if fun != "service_area":
            df["n_endpoints"] = len(self.endpoints.gdf)
        else:
            df["percent_missing"] = results["geometry"].isna().mean() * 100

        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                value = list(value)
            if isinstance(value, (list, tuple)):
                value = [str(x) for x in value]
                value = ", ".join(value)
            df[key] = value

        self.log = pd.concat([self.log, df], ignore_index=True)

    def _prepare_network_analysis(
        self, startpoints, endpoints=None, id_col: str | None = None
    ) -> None:
        """Prepares the weight column, node ids and start- and endpoints.
        Also updates the graph if it is not yet created and no parts of the analysis has changed.
        this method is run inside od_cost_matrix, shortest_path and service_area.
        """

        self.network.gdf = self.rules._validate_weight(
            self.network.gdf, raise_error=True
        )

        self.startpoints = StartPoints(
            startpoints,
            id_col=id_col,
            temp_idx_start=max(self.network.nodes.node_id.astype(int)) + 1,
        )

        if endpoints is not None:
            self.endpoints = EndPoints(
                endpoints,
                id_col=id_col,
                temp_idx_start=max(self.startpoints.gdf.temp_idx.astype(int)) + 1,
            )

        else:
            self.endpoints = None

        if not (self._graph_is_up_to_date() and self.network._nodes_are_up_to_date()):
            self.network._update_nodes_if()

            edges, weights = self._get_edges_and_weights()

            self.graph = self._make_graph(
                edges=edges, weights=weights, directed=self.network.directed
            )

            self._add_missing_vertices()

        self._update_point_wkts()
        self.rules._update_rules()

    def _get_edges_and_weights(self) -> Tuple[list[Tuple[str, ...]], list[float]]:
        """Creates lists of edges and weights which will be used to make the graph.
        Edges and weights between startpoints and nodes and nodes and endpoints are also added.
        """

        if self.rules.split_lines:
            if self.endpoints:
                points = gdf_concat([self.startpoints.gdf, self.endpoints.gdf])
            else:
                points = self.startpoints.gdf

            points = points.drop_duplicates("geometry")

            self.network.gdf["meters"] = self.network.gdf.length

            lines = split_lines_at_closest_point(
                lines=self.network.gdf,
                points=points,
                max_dist=self.rules.search_tolerance,
            )

            # adjust the weight to splitted length
            lines.loc[lines["splitted"] == 1, self.rules.weight] = lines[
                self.rules.weight
            ] * (lines.length / lines["meters"])

            self.network.gdf = lines
            self.network.make_node_ids()

        edges = [
            (str(source), str(target))
            for source, target in zip(
                self.network.gdf["source"], self.network.gdf["target"]
            )
        ]

        weights = list(self.network.gdf[self.rules.weight])

        edges_start, weights_start = self.startpoints._get_edges_and_weights(
            nodes=self.network.nodes, rules=self.rules
        )
        edges = edges + edges_start
        weights = weights + weights_start

        if self.endpoints is None:
            return edges, weights

        edges_end, weights_end = self.endpoints._get_edges_and_weights(
            nodes=self.network.nodes, rules=self.rules
        )

        edges = edges + edges_end
        weights = weights + weights_end

        return edges, weights

    def _add_missing_vertices(self):
        """Adds the points that had no nodes within the search_tolerance
        to the graph. To prevent error when running the distance calculation.
        """
        # TODO: either check if any() beforehand, or add fictional edges before making the graph,
        # to make things faster (this method took 64.660 out of 500 seconds)
        self.graph.add_vertices(
            [
                idx
                for idx in self.startpoints.gdf["temp_idx"]
                if idx not in self.graph.vs["name"]
            ]
        )
        if self.endpoints is not None:
            self.graph.add_vertices(
                [
                    idx
                    for idx in self.endpoints.gdf["temp_idx"]
                    if idx not in self.graph.vs["name"]
                ]
            )

    @staticmethod
    def _make_graph(
        edges: list[Tuple[str, ...]] | np.ndarray[Tuple[str, ...]],
        weights: list[float] | np.ndarray[float],
        directed: bool,
    ) -> Graph:
        """Creates an igraph Graph from a list of edges and weights."""

        assert len(edges) == len(weights)

        graph = igraph.Graph.TupleList(edges, directed=directed)

        graph.es["weight"] = weights

        assert min(graph.es["weight"]) >= 0

        return graph

    def _graph_is_up_to_date(self) -> bool:
        """Returns False if the rules of the graphmaking has changed,
        or if the points have changed"""

        if not hasattr(self, "graph") or not hasattr(self, "wkts"):
            return False

        if self.rules._rules_have_changed():
            return False

        if self._points_have_changed(self.startpoints.gdf, what="start"):
            return False

        if self.endpoints is None:
            return True

        if self._points_have_changed(self.endpoints.gdf, what="end"):
            return False

        return True

    def _points_have_changed(self, points: GeoDataFrame, what: str) -> bool:
        """This method is best stored in the NetworkAnalysis class,
        since the point classes are initialised each time an analysis is run."""
        if self.wkts[what] != [geom.wkt for geom in points.geometry]:
            return True

        if not all(x in self.graph.vs["name"] for x in list(points.temp_idx.values)):
            return True

        return False

    def _update_point_wkts(self) -> None:
        """Creates a dict of wkt lists. This method is run after the graph is created.
        If the wkts haven't updated since the last run, the graph doesn't have to be remade.
        """
        self.wkts = {}

        self.wkts["network"] = [geom.wkt for geom in self.network.gdf.geometry]

        if not hasattr(self, "startpoints"):
            return

        self.wkts["start"] = [geom.wkt for geom in self.startpoints.gdf.geometry]

        if self.endpoints is not None:
            self.wkts["end"] = [geom.wkt for geom in self.endpoints.gdf.geometry]

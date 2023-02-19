from typing import Tuple

import igraph
import numpy as np
from geopandas import GeoDataFrame
from igraph import Graph
from pandas import DataFrame

from .directednetwork import DirectedNetwork
from .geopandas_utils import push_geom_col
from .network import Network
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
        cost: e.i. 'minutes' or 'meters'. Or custom numeric column.
        search_tolerance: meters.
        search_factor: .
        cost_to_nodes: .

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

    nwa = NetworkAnalysis(nw, cost="minutes")

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

        self.network.gdf = self.rules.validate_cost(self.network.gdf, raise_error=False)

        self.update_point_wkts()
        self.rules.update_rules()

    def od_cost_matrix(
        self,
        startpoints: GeoDataFrame,
        endpoints: GeoDataFrame,
        id_col: str | Tuple[str, str] | None = None,
        lines: bool = False,
        **kwargs,
    ) -> DataFrame | GeoDataFrame:
        self.prepare_network_analysis(startpoints, endpoints, id_col)

        results = od_cost_matrix(
            graph=self.graph,
            startpoints=self.startpoints.gdf,
            endpoints=self.endpoints.gdf,
            cost=self.rules.cost,
            lines=lines,
            **kwargs,
        )

        self.startpoints.get_n_missing(results, "origin")
        self.endpoints.get_n_missing(results, "destination")

        if id_col:
            results["origin"] = results["origin"].map(self.startpoints.id_dict)
            results["destination"] = results["destination"].map(self.endpoints.id_dict)

        if lines:
            results = push_geom_col(results)

        return results

    def shortest_path(
        self,
        startpoints: GeoDataFrame,
        endpoints: GeoDataFrame,
        id_col: str | Tuple[str, str] | None = None,
        summarise: bool = False,
        **kwargs,
    ) -> GeoDataFrame:
        self.prepare_network_analysis(startpoints, endpoints, id_col)

        results = shortest_path(
            graph=self.graph,
            startpoints=self.startpoints.gdf,
            endpoints=self.endpoints.gdf,
            cost=self.rules.cost,
            roads=self.network.gdf,
            summarise=summarise,
            **kwargs,
        )

        if not summarise:
            self.startpoints.get_n_missing(results, "origin")
            self.endpoints.get_n_missing(results, "destination")

        if id_col and not summarise:
            results["origin"] = results["origin"].map(self.startpoints.id_dict)
            results["destination"] = results["destination"].map(self.endpoints.id_dict)

        results = push_geom_col(results)

        return results

    def service_area(
        self, startpoints: GeoDataFrame, id_col: str | None = None, **kwargs
    ) -> GeoDataFrame:
        self.prepare_network_analysis(startpoints, id_col=id_col)

        results = service_area(
            self.graph,
            self.startpoints.gdf,
            self.rules.cost,
            self.network.gdf,
            **kwargs,
        )

        if id_col:
            results[id_col] = results["origin"].map(self.startpoints.id_dict)
            results = results.drop("origin", axis=1)

        results = push_geom_col(results)

        return results

    def prepare_network_analysis(
        self, startpoints, endpoints=None, id_col: str | None = None
    ) -> None:
        """Prepares the cost column, node ids and start- and endpoints.
        Also updates the graph if it is not yet created and no parts of the analysis has changed.
        this method is run inside od_cost_matrix, shortest_path and service_area.
        """

        self.network.gdf = self.rules.validate_cost(self.network.gdf, raise_error=True)

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

        self.network.update_nodes_if()

        if not self.graph_is_up_to_date():
            edges, costs = self.get_edges_and_costs()

            self.graph = self.make_graph(
                edges=edges, costs=costs, directed=self.network.directed
            )

            self.add_missing_vertices()

        self.update_point_wkts()
        self.rules.update_rules()

    def get_edges_and_costs(self) -> Tuple[list[Tuple[str, ...]], list[float]]:
        """Creates lists of edges and costs which will be used to make the graph.
        Edges and costs between startpoints and nodes and nodes and endpoints are also added.
        """

        edges = [
            (str(source), str(target))
            for source, target in zip(
                self.network.gdf["source"], self.network.gdf["target"]
            )
        ]

        costs = list(self.network.gdf[self.rules.cost])

        edges_start, costs_start = self.startpoints.get_edges_and_costs(
            self.network.nodes, self.rules
        )
        edges = edges + edges_start
        costs = costs + costs_start

        if self.endpoints is None:
            return edges, costs

        edges_end, costs_end = self.endpoints.get_edges_and_costs(
            self.network.nodes, self.rules
        )
        edges = edges + edges_end
        costs = costs + costs_end

        return edges, costs

    def add_missing_vertices(self):
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
    def make_graph(
        edges: list[Tuple[str, ...]] | np.ndarray[Tuple[str, ...]],
        costs: list[float] | np.ndarray[float],
        directed: bool,
    ) -> Graph:
        assert len(edges) == len(costs)

        graph = igraph.Graph.TupleList(edges, directed=directed)

        graph.es["weight"] = costs

        assert min(graph.es["weight"]) > 0

        return graph

    def graph_is_up_to_date(self) -> bool:
        """Returns False if the rules of the graphmaking has changed,"""

        if not hasattr(self, "graph") or not hasattr(self, "wkts"):
            return False

        if self.rules.rules_have_changed():
            return False

        if self.points_have_changed(self.startpoints.gdf, what="start"):
            return False

        if self.endpoints is None:
            return True

        if self.points_have_changed(self.endpoints.gdf, what="end"):
            return False

        return True

    def points_have_changed(self, points: GeoDataFrame, what: str) -> bool:
        if self.wkts[what] != [geom.wkt for geom in points.geometry]:
            return True

        if not all(x in self.graph.vs["name"] for x in list(points.temp_idx.values)):
            return True

        return False

    def update_point_wkts(self):
        if not hasattr(self, "endpoints"):
            return

        self.wkts = {}
        self.wkts["start"] = [geom.wkt for geom in self.startpoints.gdf.geometry]

        if self.endpoints is not None:
            self.wkts["end"] = [geom.wkt for geom in self.endpoints.gdf.geometry]

    def __repr__(self) -> str:
        return f"""
{self.__class__.__name__}(cost={self.rules.cost}, search_tolerance={self.rules.search_tolerance}, search_factor={self.rules.search_factor}, cost_to_nodes={self.rules.cost_to_nodes})
"""

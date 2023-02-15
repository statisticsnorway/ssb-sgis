# NetworkAnalysis is a class that holds the actual network analysis methods.
import numpy as np
import warnings
from igraph import Graph
from geopandas import GeoDataFrame
from pandas import DataFrame, RangeIndex
from .make_graph import make_graph

from .od_cost_matrix import od_cost_matrix
from .shortest_path import shortest_path
from .service_area import service_area
from .points import StartPoints, EndPoints

from .network import Network
from .directednetwork import DirectedNetwork
from .makegraph import MakeGraph


from dataclasses import dataclass

@dataclass
class Rules:
    cost: str
    directed: bool
    search_tolerance: int = 250
    search_factor: int = 10
    cost_to_nodes: int = 5


class NetworkAnalysis(Rules):
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
        cost: str,
        **kwargs,
#        cost: str,
 #       search_tolerance: int = 1000,
  #      search_factor: int = 10,
   #     cost_to_nodes: int = 5,
    ):

        if isinstance(network, DirectedNetwork):
            directed = True
        elif isinstance(network, Network):
            directed = False
        else:
            raise ValueError(f"'network' should be either a DirectedNetwork or Network. Got {type(network)}")
        
        self.network = network

        super().__init__(cost, directed, **kwargs)
        self.makegraph = MakeGraph(cost=cost, directed=directed, **kwargs)
        """
        self.cost = cost
        self.search_tolerance = search_tolerance
        self.search_factor = search_factor
        self.cost_to_nodes = cost_to_nodes

        # attributes to check whether the rules have changed and the graph has to be remade
        self._search_tolerance = search_tolerance
        self._search_factor = search_factor
        self._cost_to_nodes = cost_to_nodes
        """

        self.validate_cost(self.network.gdf, raise_error=False)

#        self.makegraph = MakeGraph(self, **kwargs)

        self.update_unders()

    def od_cost_matrix(
        self, 
        startpoints: GeoDataFrame, 
        endpoints: GeoDataFrame,
        id_col: str | list[str] | tuple[str] | None = None,
        **kwargs
        ) -> DataFrame | GeoDataFrame:

        self.prepare_network_analysis(
            startpoints, endpoints, id_col
        )

        results = od_cost_matrix(self, self.startpoints.points, self.endpoints.points, **kwargs)

        self.startpoints.get_n_missing(results, "origin")
        self.endpoints.get_n_missing(results, "destination")

        if id_col:
            results["origin"] = results["origin"].map(
                self.startpoints.id_dict
            )
            results["destination"] = results["destination"].map(
                self.endpoints.id_dict
            )

        return results

    def shortest_path(
        self, 
        startpoints: GeoDataFrame, 
        endpoints: GeoDataFrame,
        id_col: str | list[str] | tuple[str] | None = None,
        summarise: bool = False,
        **kwargs,
        ) -> GeoDataFrame:

        self.prepare_network_analysis(
            startpoints, endpoints, id_col
        )

        results = shortest_path(self, self.startpoints.points, self.endpoints.points, summarise=summarise, **kwargs)

        if not summarise:
            self.startpoints.get_n_missing(results, "origin")
            self.endpoints.get_n_missing(results, "destination")

        if id_col and not summarise:
            results["origin"] = results["origin"].map(
                self.startpoints.id_dict
            )
            results["destination"] = results["destination"].map(
                self.endpoints.id_dict
            )

        return results

    def service_area(
        self, 
        startpoints: GeoDataFrame, 
        id_col: str | list[str] | tuple[str] | None = None,
        **kwargs
        ) -> GeoDataFrame:

        self.prepare_network_analysis(
            startpoints, id_col=id_col
        )

        results = service_area(self, self.startpoints.points, **kwargs)

        return results

    def prepare_network_analysis(self, startpoints, endpoints=None, id_col: str | None = None) -> None:

        self.validate_cost(self.network.gdf, raise_error=True)

        self.startpoints = StartPoints(
            startpoints, 
            id_col=id_col, 
            crs=self.network.gdf.crs,
            temp_idx_start=max(self.network.nodes.node_id.astype(int)) + 1
            )

        if endpoints is not None:
            self.endpoints = EndPoints(
                endpoints, 
                id_col=id_col, 
                crs=self.network.gdf.crs,
                temp_idx_start=max(self.startpoints.points.temp_idx.astype(int)) + 1
                )
        
        else:
            self.endpoints = None

        self.network.update_nodes_if()

        if not self.graph_is_up_to_date(startpoints, endpoints):
#        if not self.makegraph.graph_is_up_to_date():
            self.startpoints.distance_to_nodes(self.network.nodes, self.search_tolerance, self.search_factor)
            if endpoints is not None:
                self.endpoints.distance_to_nodes(self.network.nodes, self.search_tolerance, self.search_factor)
            """
            self.makegraph.graph = self.makegraph.make_graph()
            self.graph = self.makegraph.make_graph(self.network.gdf)
            self.graph = self.makegraph.add_to_graph(self.graph, self.startpoints.edges, self.startpoints.dists)
            if endpoints is not None:
                self.graph = self.makegraph.add_to_graph(self.graph, self.endpoints.edges, self.endpoints.dists)
            self.graph.add_vertices([idx for idx in self.startpoints.points.temp_idx if idx not in self.graph.vs["name"]])
            if self.endpoints is not None:
                self.graph.add_vertices([idx for idx in self.endpoints.points.temp_idx if idx not in self.graph.vs["name"]])
                """
            self.graph = self.make_graph(self.network.gdf)
        
        self.update_unders()

    def make_graph(
        self,
        ) -> Graph:
        return make_graph(self)

    def make_graph(self, gdf: GeoDataFrame) -> Graph:
        """Lager igraph.Graph som inkluderer edges to/from start-/sluttpunktene.
        """

        edges = [
            (str(source), str(target))
            for source, target in zip(gdf["source"], gdf["target"])
        ]

        costs = list(gdf[self.cost])

        edges = edges + self.startpoints.edges
        costs = costs + self.calculate_costs(self.startpoints.dists)

        if self.endpoints is not None:
            edges = edges + self.endpoints.edges
            costs = costs + self.calculate_costs(self.endpoints.dists)

        graph = Graph.TupleList(edges, directed=self.directed)
        assert len(graph.get_edgelist()) == len(costs)

        graph.es["weight"] = costs
        assert min(graph.es["weight"]) > 0

        graph.add_vertices([idx for idx in self.startpoints.points.temp_idx if idx not in graph.vs["name"]])
        if self.endpoints is not None:
            graph.add_vertices([idx for idx in self.endpoints.points.temp_idx if idx not in graph.vs["name"]])
            
        return graph

    def calculate_costs(self, distances: list[float]):
        """
        Gjør om meter to minutter for lenkene mellom punktene og nabonodene.
        og ganger luftlinjeavstanden med 1.5 siden det alltid er svinger i Norge.
        Gjør ellers ingentinnw.
        """

        if self.cost_to_nodes == 0:
            return [0 for _ in distances]

        elif "meter" in self.cost:
            return [x * 1.5 for x in distances]

        elif "min" in self.cost:
            return [(x * 1.5) / (16.666667 * self.cost_to_nodes) for x in distances]

        else:
            return distances

    def graph_is_up_to_date(self, startpoints: GeoDataFrame, endpoints: GeoDataFrame) -> bool:
        """Returns False if the rules of the graphmaking has changed, """

        if not hasattr(self, "graph") or not hasattr(self, "start_wkts"):
            return False
        
        # check if the rules for making the graph have changed
        if self.search_factor != self._search_factor:
            return False
        if self.search_tolerance != self._search_tolerance:
            return False
        if self.cost_to_nodes != self._cost_to_nodes:
            return False
        
        if self.start_wkts != [
            geom.wkt for geom in startpoints.geometry
        ]:
            return False

        if self.endpoints is not None:
            if self.end_wkts != [
                geom.wkt for geom in endpoints.geometry
            ]:
                return False

        if not all(
            x in self.graph.vs["name"] for x in list(self.startpoints.points.temp_idx.values)
            ):
            return False

        if self.endpoints:
            if not all(
            x in self.graph.vs["name"] for x in self.endpoints.points.temp_idx
            ):
                return False

        return True
    
    def update_unders(self):
        self._search_tolerance = self.search_tolerance
        self._search_factor = self.search_factor
        self._cost_to_nodes = self.cost_to_nodes

        if hasattr(self, "startpoints"):
            self.start_wkts = [geom.wkt for geom in self.startpoints.points.geometry]
        if hasattr(self, "endpoints"):
            if self.endpoints is not None:
                self.end_wkts = [geom.wkt for geom in self.endpoints.points.geometry]

    def validate_cost(self, gdf, raise_error: bool = True) -> None:

        if self.cost in gdf.columns:

            self.warn_if_nans(gdf, self.cost)
            self.warn_if_negative_values(gdf, self.cost)

            try:
                gdf[self.cost] = gdf[self.cost].astype(float)    
            except ValueError as e:
                raise ValueError(f"The 'cost' column must be numeric. Got characters that couldn't be interpreted as numbers.")

            if "min" in self.cost:
                self.cost = "minutes"
                    
        if "meter" in self.cost or "metre" in self.cost:

            if gdf.crs == 4326:
                raise ValueError("'roads' cannot have crs 4326 (latlon) when cost is 'meters'.")

            self.cost = "meters"
            gdf[self.cost] = gdf.length
            return gdf

        if self.cost == "minutes" and "minutes" not in gdf.columns:
            if raise_error:
                raise KeyError(f"Cannot find 'cost' column for minutes. \nTry running one of the 'make_directed_network_' methods, or set 'cost' to 'meters'.")

    @staticmethod
    def warn_if_nans(gdf, cost):

        if all(gdf[cost].isna()):
            raise ValueError("All values in the 'cost' column are NaN.")

        nans = sum(gdf[cost].isna())
        if nans:
            if nans > len(gdf) * 0.05:
                warnings.warn(f"Warning: {nans} rows have missing values in the 'cost' column. Removing these rows.")
            gdf = gdf.loc[gdf[cost].notna()]
        
    @staticmethod
    def warn_if_negative_values(gdf, cost):
        negative = sum(gdf[cost] < 0)
        if negative:
            if negative > len(gdf) * 0.05:
                warnings.warn(f"Warning: {negative} rows have a 'cost' less than 0. Removing these rows.")
            gdf = gdf.loc[gdf[cost] >= 0]
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cost={self.cost}, search_tolerance={self.search_tolerance}, search_factor={self.search_factor}, cost_to_nodes={self.cost_to_nodes})"

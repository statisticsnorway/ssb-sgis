import numpy as np
import warnings
from igraph import Graph
from geopandas import GeoDataFrame
from pandas import DataFrame, RangeIndex
from copy import copy, deepcopy
from .make_graph import make_graph

from .od_cost_matrix import od_cost_matrix
from .shortest_path import shortest_path
from .service_area import service_area
from .points import StartPoints, EndPoints

from .network import Network
from .directednetwork import DirectedNetwork


class Rules:
    def __init__(
        self,
        cost: int,
        search_tolerance: int = 1000,
        search_factor: int = 10,
        cost_to_nodes: int = 5,
    ):
        self.cost = cost
        self.search_tolerance = search_tolerance
        self.search_factor = search_factor
        self.cost_to_nodes = cost_to_nodes


class NetworkAnalysis:
    """Class that holds the actual network analysis methods. 
    
    Args:
        network: either the base Network class or a subclass, chiefly the DirectedNetwork class.
            The network should be customized beforehand, but can also be accessed through 
            the 'network' attribute of this class. 
        cost: e.i. 'minutes' or 'meters'. Or custom numeric column.
        search_tolerance: meters.

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
        search_tolerance: int = 1000,
        search_factor: int = 10,
        cost_to_nodes: int = 5,
    ):

        if isinstance(network, DirectedNetwork):
            self.directed = True
        elif isinstance(network, Network):
            self.directed = False
        else:
            raise ValueError(f"'network' should be either a DirectedNetwork or Network. Got {type(network)}")
        
        self.network = network
        self.cost = cost
        self.search_tolerance = search_tolerance
        self.search_factor = search_factor
        self.cost_to_nodes = cost_to_nodes

        # attributes to check whether the rules have changed and the graph has to be remade
        self._search_tolerance = search_tolerance
        self._search_factor = search_factor
        self._cost_to_nodes = cost_to_nodes

        self.validate_cost(raise_error=False)

    def make_graph(
        self,
        ) -> Graph:
        return make_graph(self)

    def od_cost_matrix(
        self, 
        startpoints: GeoDataFrame, 
        endpoints: GeoDataFrame,
        id_col: str | list[str, str] | tuple[str, str] | None = None,
        **kwargs
        ) -> DataFrame | GeoDataFrame:

        self.prepare_network_analysis(
            startpoints, endpoints, id_col
        )

        results = od_cost_matrix(self, self.startpoints.points, self.endpoints.points, **kwargs)

        self.startpoints.n_missing(results, "origin")
        self.endpoints.n_missing(results, "destination")

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
        id_col: str | list[str, str] | tuple[str, str] | None = None,
        summarise: bool = False,
        **kwargs,
        ) -> GeoDataFrame:

        self.prepare_network_analysis(
            startpoints, endpoints, id_col
        )

        results = shortest_path(self, self.startpoints.points, self.endpoints.points, summarise=summarise, **kwargs)

        self.startpoints.n_missing(results, "origin")
        self.endpoints.n_missing(results, "destination")

        if id_col:
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
        id_col: str | list[str, str] | tuple[str, str] | None = None,
        **kwargs
        ) -> GeoDataFrame:

        self.prepare_network_analysis(
            startpoints, id_col=id_col
        )

        results = service_area(self, self.startpoints.points, **kwargs)

        self.startpoints.n_missing(results, self.cost)

        if id_col:
            results[id_col] = results[id_col].map(
                self.startpoints.id_dict
            )

        return results

    def prepare_network_analysis(self, startpoints, endpoints=None, id_col: str | None = None) -> None:

        self.validate_cost(raise_error=True)

        self.startpoints = StartPoints(
            startpoints, 
            crs=self.network.gdf.crs,
            id_col=id_col, 
            )

        if endpoints is not None:
            self.endpoints = EndPoints(endpoints, id_col=id_col, crs=self.network.gdf.crs)
        else:
            del self.endpoints

        self.make_temp_idx()

        self.network.update_nodes_if()

        if self.graph_is_up_to_date(startpoints, endpoints):
            self.update_unders()
            return
        
        self.update_unders()
        self.graph = self.make_graph()

    def make_temp_idx(self):
        if hasattr(self.network, "nodes"):
            self.startpoints.make_temp_idx(
                start=max(self.network.nodes.node_id.astype(int)) + 1
            )
            if hasattr(self, "endpoints"):
                self.endpoints.make_temp_idx(
                    start=max(self.startpoints.points.temp_idx.astype(int)) + 1
                )

    def graph_is_up_to_date(self, startpoints, endpoints):
        
        if not hasattr(self, "graph"):
            return False
        
        if self.search_factor != self._search_factor:
            return False
        if self.search_tolerance != self._search_tolerance:
            return False
        if self.cost_to_nodes != self._cost_to_nodes:
            return False
        
        if self.startpoints.wkt != [
            geom.wkt for geom in startpoints.geometry
        ]:
            return False

        if hasattr(self, "endpoints"):
            if self.endpoints.wkt != [
                geom.wkt for geom in endpoints.geometry
            ]:
                return False

        if not all(
            x in self.graph.vs["name"] for x in list(self.startpoints.points.temp_idx.values)
            ):
            return False

        if hasattr(self, "endpoints"):
            if not all(
            x in self.graph.vs["name"] for x in self.endpoints.points.temp_idx
            ):
                return False
        print("Truetruetrue")
        return True
    
    def update_unders(self):
        self._search_tolerance = self.search_tolerance
        self._search_factor = self.search_factor
        self._cost_to_nodes = self.cost_to_nodes

    def validate_cost(self, raise_error: bool = True) -> None:
        
        gdf = self.network.gdf
        cost = self.cost

        if cost in self.network.gdf.columns:

            if all(self.network.gdf[cost].isna()):
                raise ValueError("All values in the 'cost' column are NaN.")

            if (n := sum(self.network.gdf[cost].isna())):
                if n > len(self.network.gdf) * 0.05:
                    warnings.warn(f"Warning: {n} rows have missing values in the 'cost' column. Removing NaNs.")
                self.network.gdf = self.network.gdf.loc[self.network.gdf[cost].notna()]
            
            if (n := sum(self.network.gdf[cost] < 0)):
                if n > len(self.network.gdf) * 0.05:
                    warnings.warn(f"Warning: {n} rows have a 'cost' less than 0. Removing these rows.")
                self.network.gdf = self.network.gdf.loc[self.network.gdf[cost] > 0]

            try:
                self.network.gdf[cost] = self.network.gdf[cost].astype(float)    
            except ValueError as e:
                raise ValueError(f"There are alphabetical characters in the 'cost' column: {str(e)}")

            if "min" in cost:
                cost = "minutes"
                    
        if "meter" in cost or "metre" in cost:

            if self.network.gdf.crs == 4326:
                raise ValueError("'roads' cannot have crs 4326 (latlon) when cost is 'meters'.")

            cost = "meters"

            self.network.gdf[cost] = self.network.gdf.length
            
            return

        if cost == "minutes" and "minutes" not in self.network.gdf.columns:
            if raise_error:
                raise KeyError(f"Cannot find 'cost' column for minutes. Try running one of the 'make_directed_network_' methods, or set 'cost' to 'meters'.")

    def copy(self):
        return copy(self)

    def deepcopy(self):
        return deepcopy(self)
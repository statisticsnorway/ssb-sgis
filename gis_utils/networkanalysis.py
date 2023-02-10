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


# TODO: vurdere å gjøre denne mindre, 
# dumt at parent-en skal være avhengig av Network
# noe composition:
# MakeGraph, Rules, NetworkAnalysis -> alle som composition inni Network?
# hva har jeg å tjene? gjør det mulig å bruke Network uten analyse, men da må man initiere analyse manuelt.

# RunAnalysis(od, sp, sa, temp_ids, map_ids)
# Rules -> ABC, protocol...
# MakeGraph - self.graph = MakeGraph(self), graphisuptodate, - hva med update_graph_info, graph_is_up_to_date???
# ValidateGraph - update_graph_info, graph_is_up_to_date


class Rules:
    def __init__(
        self,
        search_tolerance: int = 1000,
        search_factor: int = 10,
        cost_to_nodes: int = 5,
    ):
        self.search_tolerance = search_tolerance
        self.search_factor = search_factor
        self.cost_to_nodes = cost_to_nodes

class NetworkAnalysis:
    def __init__(
        self,
        cost: str,
        search_tolerance: int = 1000,
        search_factor: int = 10,
        cost_to_nodes: int = 5,
    ):
        self.cost = cost
        self.search_tolerance = search_tolerance
        self.search_factor = search_factor
        self.cost_to_nodes = cost_to_nodes

    def prepare_network_analysis(self, startpoints, endpoints=None, id_col: str | None = None) -> None:

        self.validate_cost(raise_error=True)

        self.startpoints = StartPoints(startpoints, id_col=id_col, crs=self.network.crs)

        if endpoints is not None:
            self.endpoints = EndPoints(endpoints, id_col=id_col, crs=self.network.crs)
        else:
            del self.endpoints

        if self.graph_is_up_to_date(startpoints, endpoints):
            return

        self.make_node_ids()

        self._network_len = len(self.network)

        self.graph = self.make_graph()
        
    def graph_is_up_to_date(self, startpoints, endpoints):
        
        if not hasattr(self, "graph"):
            return False
        
        if self.startpoints.wkt != [
            geom.wkt for geom in startpoints.geometry
        ]:
            print("hei")
            return False

        if hasattr(self, "endpoints"):
            if self.endpoints.wkt != [
                geom.wkt for geom in endpoints.geometry
            ]:
                print("hei2")
                return False

        if not all(
            x in self.graph.vs["name"] for x in self.startpoints.temp_idx
            ):
            return False

        if hasattr(self, "endpoints"):
            if not all(
            x in self.graph.vs["name"] for x in self.endpoints.temp_idx
            ):
                return False
                                
        idx = self.network.index
        if not isinstance(idx, RangeIndex):
            return False
        if idx.start != 0:
            return False
        if idx.stop != len(self.network):
            return False
        print("hei4")

        if not self._network_len == len(self.network):
            return False

        return True

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

        self.startpoints.n_missing(results)
        self.endpoints.n_missing(results)

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

        self.startpoints.n_missing(results)
        self.endpoints.n_missing(results)

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

        self.startpoints.n_missing(results)

        if id_col:
            results[id_col] = results[id_col].map(
                self.startpoints.id_dict
            )

        return results

    def validate_cost(self, raise_error: bool = True) -> None:

        if self.cost in self.network.columns:

            if all(self.network[self.cost].isna()):
                raise ValueError("All values in the 'cost' column are NaN.")

            if (n := sum(self.network[self.cost].isna())):
                if n > len(self.network) * 0.05:
                    warnings.warn(f"Warning: {n} rows have missing values in the 'cost' column. Removing NaNs.")
                self.network = self.network.loc[self.network[self.cost].notna()]
            
            if (n := sum(self.network[self.cost] < 0)):
                if n > len(self.network) * 0.05:
                    warnings.warn(f"Warning: {n} rows have a 'cost' less than 0. Removing these rows.")
                self.network = self.network.loc[self.network[self.cost] > 0]

            try:
                self.network[self.cost] = self.network[self.cost].astype(float)    
            except ValueError as e:
                raise ValueError(f"There are alphabetical characters in the 'cost' column: {str(e)}")

            if "min" in self.cost:
                self.cost = "minutes"
                    
        if "meter" in self.cost or "metre" in self.cost:

            if self.network.crs == 4326:
                raise ValueError("'roads' cannot have crs 4326 (latlon) when cost is 'meters'.")

            self.cost = "meters"

            self.network[self.cost] = self.network.length
            
            return

        if self.cost == "minutes" and "minutes" not in self.network.columns:
            if raise_error:
                raise KeyError(f"Cannot find 'cost' column for minutes. \nTry running one of the 'make_directed_network_' methods, or set 'cost' to 'meters'.")

    def copy(self):
        return copy(self)

    def deepcopy(self):
        return deepcopy(self)
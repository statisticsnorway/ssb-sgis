
import warnings
from shapely import line_merge
from shapely.constructive import reverse
from igraph import Graph
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame, RangeIndex
from abc import ABC
from copy import copy, deepcopy

"""
from .core import (
    clean_geoms,
)
from .network_functions import (
    make_node_ids,
    close_network_holes,
    find_isolated_networks,
    ZeroRoadsError,
)
from .od_cost_matrix import od_cost_matrix
"""


class NetworkAnalysis(ABC):
    def __init__(
        self,
        search_tolerance: int = 1000,
        search_factor: int = 10,
        cost_to_nodes: int = 5,
    ):
        self.search_tolerance = search_tolerance
        self.search_factor = search_factor
        self.cost_to_nodes = cost_to_nodes

    def prepare_network_analysis(self, startpoints, endpoints=None, id_col: str | None = None) -> None:

        self.validate_cost(raise_error=True)

        self.validate_points(startpoints, endpoints, id_col)

        startpoints, endpoints = self.prepare_points(startpoints, endpoints)
        
        if isinstance(id_col, str):
            id_col = id_col, id_col

        if self.graph_is_up_to_date(startpoints, endpoints):
            if endpoints is None:
                return startpoints, id_col
            return startpoints, endpoints, id_col
                    
        self.make_node_ids()

        startpoints, endpoints = self.prepare_points(startpoints, endpoints)

        self.graph = self.make_graph(startpoints, endpoints)
        
        self.update_graph_info(startpoints, endpoints)
        
        if endpoints is None:
            return startpoints, id_col
        
        return startpoints, endpoints, id_col
    
    def graph_is_up_to_date(self, startpoints, endpoints):

        if not hasattr(self, "graph"):
            return False
    
        if any(
            True if x not in self.graph.vs["name"] else False for x in startpoints.temp_idx
            ):
            return False

        if endpoints is not None:
            if any(
                True if x not in self.graph.vs["name"] else False for x in endpoints.temp_idx
                ):
                return False
            if self._temp_idx_end_max == max(endpoints.temp_idx):
                return True
                        
        if self._network_len == len(self.network):
            return True
        
        if self._temp_idx_start_max == max(startpoints.temp_idx):
            return True
            
        idx = self.network.index
        if not isinstance(idx, RangeIndex):
            return False
        if idx.start != 0:
            return False
        if idx.stop != len(self.network):
            return False
        
        return True

    def update_graph_info(self, startpoints, endpoints):

        self._network_len = len(self.network)

        self._temp_idx_start_max = max(startpoints.temp_idx)

        if endpoints is not None:
            self._temp_idx_end_max = max(endpoints.temp_idx)

    def make_graph(
        self,
        startpoints: GeoDataFrame,
        endpoints: GeoDataFrame | None = None,
        ) -> Graph:

        return make_graph(self, startpoints, endpoints)

    def od_cost_matrix(
        self, 
        startpoints: GeoDataFrame, 
        endpoints: GeoDataFrame,
        id_col: str | list | tuple = None, 
        **kwargs
        ) -> DataFrame | GeoDataFrame:

        startpoints, endpoints, id_cols = self.prepare_network_analysis(
            startpoints, endpoints, id_col
        )

        res = od_cost_matrix(self, startpoints, endpoints, **kwargs)

        if id_col:
            res["origin"] = self.map_ids(
                res["origin"], startpoints, id_cols[0],
            )
            res["destination"] = self.map_ids(
                res["destination"], endpoints, id_cols[1],
            )

        return res

    def shortest_path(
        self, 
        startpoints: GeoDataFrame, 
        endpoints: GeoDataFrame,
        id_col: str | list | tuple = None, 
        **kwargs
        ) -> GeoDataFrame:

        startpoints, endpoints, id_cols = self.prepare_network_analysis(
            startpoints, endpoints, id_col
        )

        res = shortest_path(self, startpoints, endpoints, **kwargs)

        res["from"] = self.map_ids(
            res["from"], startpoints[id_cols[0]]
        ) 
        res["to"] = self.map_ids(
            res["to"], endpoints[id_cols[1]]
        )

        return res

    def service_area(
        self, 
        startpoints: GeoDataFrame, 
        id_col: str | list | tuple = None, 
        **kwargs
        ) -> GeoDataFrame:

        startpoints, id_cols = self.prepare_network_analysis(
            startpoints, id_col
        )

        res = service_area(self, startpoints, **kwargs)

        res["from"] = self.map_ids(
            res["from"], startpoints[id_cols[0]]
        )

        return res

    def make_temp_ids(self, points, plus=0):
        """
        Lager id-kolonne som brukes som node-id-er i igraph.Graph().
        Fordi start- og sluttpunktene må ha node-id-er som ikke finnes i networket.
        """
        start = max(self.nodes.node_id.astype(int)) + 1 + plus
        stop = start + len(points)
        return [str(idx) for idx in np.arange(start, stop)]

    def map_ids(self, col, points, id_col):
        """From temp to original ids."""

        id_dict = {
            temp_idx: idx
            for temp_idx, idx in zip(points["temp_idx"], points[id_col])
        }

        return col.map(id_dict)
    
    def prepare_points(self, startpoints, endpoints):
        startpoints = startpoints.to_crs(self.network.crs)
        startpoints["temp_idx"] = self.make_temp_ids(startpoints)

        if endpoints is not None:
            endpoints = endpoints.to_crs(self.network.crs)
            endpoints["temp_idx"] = self.make_temp_ids(endpoints, plus=len(startpoints))
        
        return startpoints, endpoints

    @staticmethod
    def validate_points(
        startpoints: GeoDataFrame,
        endpoints: GeoDataFrame | None = None,
        id_col: str | list | tuple = None,
        ) -> None: 

        if isinstance(id_col, str):
            if not id_col in startpoints.columns:
                raise KeyError(f"'startpoints' has no attribute '{id_col}'")
            if endpoints is not None:
                if not id_col in endpoints.columns:
                    raise KeyError(f"'endpoints' has no attribute '{id_col}'")
        elif isinstance(id_col, (list, tuple)):
            if not id_col[0] in startpoints.columns:
                raise KeyError(f"'startpoints' has no attribute '{id_col[0]}'")
            if endpoints is None:
                warnings.warn(f"'id_col' is of type {type(id_col)} even though there are no endpoints")
            else:
                if not id_col[1] in endpoints.columns:
                    raise KeyError(f"'endpoints' has no attribute '{id_col[1]}'")
    
    def validate_cost(self, raise_error: bool = True) -> None:

        if self.cost in self.network.columns:

            if all(self.network[self.cost].isna()):
                raise ValueError("All values in the 'cost' column are NaN.")

            if (n := sum(self.network[self.cost].isna())):
                warnings.warn(f"Warning: {n} rows have missing values in the 'cost' column. Removing NaNs.")
                self.network = self.network.loc[self.network[self.cost].notna()]
            
            if (n := sum(self.network[self.cost] < 0)):
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
            
            return

        if self.cost == "minutes" and "minutes" not in self.network.columns:
            if raise_error:
                raise KeyError(f"Cannot find 'cost' column for minutes.")
            else:
                warnings.warn("Warning: Cannot find 'cost' column for minutes. Try running one of the 'make_directed_network_' methods, or set 'cost' to 'meters'.")

    def copy(self):
        return copy(self)

    def deepcopy(self):
        return deepcopy(self)
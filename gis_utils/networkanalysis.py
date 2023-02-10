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

# TODO: vurdere å gjøre denne mindre, dumt at denne skal være avhengig av Network
# eller composition
# MakeGraph, Rules, Points 

# RunAnalysis(od, sp, sa, temp_ids, map_ids) - hva da med graph_is_up_to_date???
# Points - self.startpoints = Points(startpoints, id_col)
# Rules -> ABC, protocol...
# MakeGraph - self.graph = MakeGraph(self) - hva med update_graph_info, graph_is_up_to_date???
# ValidateGraph - update_graph_info, graph_is_up_to_date


"""
points = Points()
points.start = startpoints
points.end = endpoints
points.validate_points()
points.prepare_points()
"""
class Points:
    def __init__(
        self, 
        points: GeoDataFrame,
        id_col: str,
    ) -> None:
#        self.points = points
 #       self.len = len(points)
        self.temp_idx_max = 0
        self.id_dict = {temp_idx: idx for temp_idx, idx in zip(points.temp_idx, points[id_col])}

    def n_missing(self):
        """ inn i network? """
        self.points["n"] = self.points["idx"].map(
            res.groupby("origin")[self.cost].count()
        )
    def prepare_points(self):
        self.start = self.start.to_crs(self.network.crs)
        self.start["temp_idx"] = self.make_temp_ids(self.start)

        if self.end is not None:
            self.end = self.end.to_crs(self.network.crs)
            self.end["temp_idx"] = self.make_temp_ids(self.end, plus=len(self.start))
        
    def validate_points(self,
        id_col: str | list | tuple = None,
        ) -> None: 

        if isinstance(id_col, str):
            if not id_col in self.start.columns:
                raise KeyError(f"'startpoints' has no attribute '{id_col}'")
            if self.end is not None:
                if not id_col in self.end.columns:
                    raise KeyError(f"'endpoints' has no attribute '{id_col}'")
        elif isinstance(id_col, (list, tuple)):
            if not id_col[0] in self.star.columns:
                raise KeyError(f"'startpoints' has no attribute '{id_col[0]}'")
            if self.end is None:
                warnings.warn(f"'id_col' is of type {type(id_col)} even though there are no endpoints")
            else:
                if not id_col[1] in self.end.columns:
                    raise KeyError(f"'endpoints' has no attribute '{id_col[1]}'")


class EndPoints(Points):
    def __init__(
        self, 
        points: GeoDataFrame,
        id_col: str,
    ) -> None:

        super().__init__(points, id_col)


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

        self.validate_points(startpoints, endpoints, id_col)

        startpoints, endpoints = self.prepare_points(startpoints, endpoints)
        
        if isinstance(id_col, str):
            id_col = id_col, id_col

        if self.graph_is_up_to_date(startpoints, endpoints):
            if endpoints is None:
                return startpoints, id_col
            return startpoints, endpoints, id_col
                    
        self.startpoints = startpoints
        if endpoints is not None:
            self.endpoints = endpoints

        self.make_node_ids()

        startpoints, endpoints = self.prepare_points(startpoints, endpoints)

        self.graph = self.make_graph(startpoints, endpoints)
        self.network["idx"] = self.network.index
        
        self.update_graph_info(startpoints, endpoints)
        
        if endpoints is None:
            return startpoints, id_col
        
        return startpoints, endpoints, id_col

    def graph_is_up_to_date(self, startpoints, endpoints):

        if not hasattr(self, "graph"):
            return False

        if not startpoints.wkt.equals(self.startpoints.wkt):
            return False

        if endpoints is not None:
            if not endpoints.wkt.equals(self.endpoints.wkt):
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
        summarise: bool = False,
        **kwargs,
        ) -> GeoDataFrame:

        startpoints, endpoints, id_cols = self.prepare_network_analysis(
            startpoints, endpoints, id_col
        )

        res = shortest_path(self, startpoints, endpoints, summarise, **kwargs)

        if id_col and not summarise:
            res["origin"] = self.map_ids(
                res["origin"], startpoints, id_cols[0],
            )
            res["destination"] = self.map_ids(
                res["destination"], endpoints, id_cols[1],
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

        if id_col:
            res["origin"] = self.map_ids(
                res["origin"], startpoints, id_cols[0],
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
        """ Point class??? """
        startpoints = startpoints.to_crs(self.network.crs)
        startpoints["temp_idx"] = self.make_temp_ids(startpoints)
        startpoints["wkt"] = [x.wkt for x in startpoints.geometry]

        if endpoints is not None:
            endpoints = endpoints.to_crs(self.network.crs)
            endpoints["temp_idx"] = self.make_temp_ids(endpoints, plus=len(startpoints))    
            endpoints["wkt"] = [x.wkt for x in endpoints.geometry]

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
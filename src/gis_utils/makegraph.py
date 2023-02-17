import warnings
from copy import copy, deepcopy

import igraph
import numpy as np
from geopandas import GeoDataFrame
from igraph import Graph
from pandas import DataFrame, RangeIndex
from sklearn.neighbors import NearestNeighbors

from .directednetwork import DirectedNetwork
from .exceptions import NoPointsWithinSearchTolerance
from .make_graph import make_graph
from .network import Network
from .od_cost_matrix import od_cost_matrix
from .points import EndPoints, StartPoints
from .service_area import service_area
from .shortest_path import shortest_path


# lage grafen på forhånd???

# TODO: vurdere å gjøre denne mindre,
# dumt at parent-en skal være avhengig av Network
# noe composition:
# MakeGraph, Rules, NetworkAnalysis -> alle som composition inni Network?
# hva har jeg å tjene? gjør det mulig å bruke Network uten analyse, men da må man
# initiere analyse manuelt.

# RunAnalysis(od, sp, sa, temp_ids, map_ids)
# Rules -> ABC, protocol...
# MakeGraph - self.graph = MakeGraph(self), graphisuptodate, - hva med
# update_graph_info, graph_is_up_to_date???
# ValidateGraph - update_graph_info, graph_is_up_to_date

# TODO: oppdatere noder når nettverket endres, ikke når det accesses.
# så lage temp_idx i init av Points, så sjekke om oppdatert graf.


# Network -> DirectedNetwork
# NetworkAnalysis(network = Network | DirectedNetwork)
# Ma

'''
class MakeGraph:
    def __init__(
        self,
        nw: Network | DirectedNetwork,
        cost,
        search_tolerance,
        search_factor,
        cost_to_nodes,
    ):
        pass
    

    def make_graph(self, nwa, 
    gdf: GeoDataFrame, startpoints: StartPoints, endpoints: EndPoints | None, cost_to_nodes/speed_kmh, ) -> Graph:
        """Lager igraph.Graph som inkluderer edges to/from start-/sluttpunktene.
        """

        edges = [
            (str(source), str(target))
            for source, target in zip(nwa.network.gdf["source"], nwa.network.gdf["target"])
        ]

        costs = list(nwa.network.gdf[self.cost])

        edges = edges + nwa.startpoints.edges
        costs = costs + self.calculate_costs(nwa.startpoints.dists)

        if nwa.endpoints is not None:
            edges = edges + nwa.endpoints.edges
            costs = costs + self.calculate_costs(nwa.endpoints.dists)

        graph = igraph.Graph.TupleList(edges, directed=nwa.directed)
        assert len(graph.get_edgelist()) == len(costs)

        graph.es["weight"] = costs
        assert min(graph.es["weight"]) > 0

        return graph

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

        if self.endpoints:
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

        return True

    def calculate_costs(self, distances):
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

    def validate_cost(self, gdf, raise_error: bool = True) -> None:

        if self.cost in gdf.columns:

            self.warn_if_nans_or_negative()

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

    def warn_if_nans_or_negative(self):
        """Give a warning if there are NaNs or negative values in the gdf."""

        if all(gdf[self.cost].isna()):
            raise ValueError("All values in the 'cost' column are NaN.")

        nans = sum(gdf[self.cost].isna())
        if nans:
            if nans > len(gdf) * 0.05:
                warnings.warn(f"Warning: {nans} rows have missing values in the 'cost' column. Removing these rows.")
            gdf = gdf.loc[gdf[self.cost].notna()]
        
        negative = sum(gdf[self.cost] < 0)
        if negative:
            if negative > len(gdf) * 0.05:
                warnings.warn(f"Warning: {negative} rows have a 'cost' less than 0. Removing these rows.")
            gdf = gdf.loc[gdf[self.cost] >= 0]

"""
cost, cost_to_nodes, startpoints, endpoints, search_factor, search_tolerance, 
network.gdf
"""












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

import igraph
from .exceptions import NoPointsWithinSearchTolerance

import numpy as np
import igraph
from igraph import Graph
from sklearn.neighbors import NearestNeighbors

from geopandas import GeoDataFrame

from .exceptions import NoPointsWithinSearchTolerance
from .directednetwork import DirectedNetwork


# lage grafen på forhånd???

# TODO: vurdere å gjøre denne mindre, 
# dumt at parent-en skal være avhengig av Network
# noe composition:
# MakeGraph, Rules, NetworkAnalysis -> alle som composition inni Network?
# hva har jeg å tjene? gjør det mulig å bruke Network uten analyse, men da må man initiere analyse manuelt.

# RunAnalysis(od, sp, sa, temp_ids, map_ids)
# Rules -> ABC, protocol...
# MakeGraph - self.graph = MakeGraph(self), graphisuptodate, - hva med update_graph_info, graph_is_up_to_date???
# ValidateGraph - update_graph_info, graph_is_up_to_date

# TODO: oppdatere noder når nettverket endres, ikke når det accesses.
# så lage temp_idx i init av Points, så sjekke om oppdatert graf.


# Network -> DirectedNetwork
# NetworkAnalysis(network = Network | DirectedNetwork)
# Ma
'''
from dataclasses import dataclass


@dataclass
class Rules:
    cost: str
    directed: bool
    search_tolerance: int = 250
    search_factor: int = 10
    cost_to_nodes: int = 5


class MakeGraph(Rules):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def make_graph(
        self,
        gdf: GeoDataFrame,
        #     startpoint_edges: np.ndarray[tuple],
        #      startpoint_dists: np.ndarray[float],
        #       endpoint_edges: np.ndarray[tuple] | None = None,
        #        endpoint_dists: np.ndarray[float] | None = None,
        #        startpoints: StartPoints,
        #       endpoints: EndPoints | None
    ) -> Graph:
        edges = [
            (str(source), str(target))
            for source, target in zip(gdf["source"], gdf["target"])
        ]

        costs = list(gdf[self.cost])

        #        edges = edges + startpoints.edges
        #       costs = costs + self.calculate_costs(startpoints.dists)
        #
        #       if endpoints is not None:
        #          edges = edges + endpoints.edges
        #         costs = costs + self.calculate_costs(endpoints.dists)

        graph = igraph.Graph.TupleList(edges, directed=self.directed)
        assert len(graph.get_edgelist()) == len(costs)

        graph.es["weight"] = costs
        assert min(graph.es["weight"]) > 0

        return graph

    @staticmethod
    def add_to_graph(graph: Graph, edges: np.ndarray[tuple], costs: np.ndarray[float]):
        graph.add_edges(edges, attributes=costs)
        return graph

    #        graph.

    def graph_is_up_to_date(self, startpoints, endpoints):
        if not hasattr(self, "graph"):
            return False

        if self.search_factor != self._search_factor:
            return False
        if self.search_tolerance != self._search_tolerance:
            return False
        if self.cost_to_nodes != self._cost_to_nodes:
            return False

        if self.startpoints.wkt != [geom.wkt for geom in startpoints.geometry]:
            return False

        if self.endpoints:
            if self.endpoints.wkt != [geom.wkt for geom in endpoints.geometry]:
                return False

        if not all(
            x in self.graph.vs["name"]
            for x in list(self.startpoints.points.temp_idx.values)
        ):
            return False

        if hasattr(self, "endpoints"):
            if not all(
                x in self.graph.vs["name"] for x in self.endpoints.points.temp_idx
            ):
                return False

        return True

    def calculate_costs(self, distances):
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

    def validate_cost(self, gdf, raise_error: bool = True) -> None:
        if self.cost in gdf.columns:
            self.warn_if_nans_or_negative()

            try:
                gdf[self.cost] = gdf[self.cost].astype(float)
            except ValueError as e:
                raise ValueError(
                    f"The 'cost' column must be numeric. Got characters that "
                    f"couldn't be interpreted as numbers."
                )

            if "min" in self.cost:
                self.cost = "minutes"

        if "meter" in self.cost or "metre" in self.cost:
            if gdf.crs == 4326:
                raise ValueError(
                    "'roads' cannot have crs 4326 (latlon) when cost is 'meters'."
                )

            self.cost = "meters"
            gdf[self.cost] = gdf.length
            return gdf

        if self.cost == "minutes" and "minutes" not in gdf.columns:
            if raise_error:
                raise KeyError(
                    f"Cannot find 'cost' column for minutes. \nTry running one of the "
                    f"'make_directed_network_' methods, or set 'cost' to 'meters'."
                )

    def warn_if_nans_or_negative(self):
        """Give a warning if there are NaNs or negative values in the gdf."""

        if all(gdf[self.cost].isna()):
            raise ValueError("All values in the 'cost' column are NaN.")

        nans = sum(gdf[self.cost].isna())
        if nans:
            if nans > len(gdf) * 0.05:
                warnings.warn(
                    f"Warning: {nans} rows have missing values in the 'cost' column. "
                    f"Removing these rows."
                )
            gdf = gdf.loc[gdf[self.cost].notna()]

        negative = sum(gdf[self.cost] < 0)
        if negative:
            if negative > len(gdf) * 0.05:
                warnings.warn(
                    f"Warning: {negative} rows have a 'cost' less than 0. "
                    f"Removing these rows."
                )
            gdf = gdf.loc[gdf[self.cost] >= 0]


"""
cost, cost_to_nodes, startpoints, endpoints, search_factor, search_tolerance, 
network.gdf
"""


def make_graph(
    gdf: GeoDataFrame,
    startpoints: StartPoints,
    endpoints: EndPoints | None,
    cost,
    cost_to_nodes,
    directed,
) -> Graph:
    """Lager igraph.Graph som inkluderer edges to/from start-/sluttpunktene."""

    edges = [
        (str(source), str(target))
        for source, target in zip(gdf["source"], gdf["target"])
    ]

    costs = list(gdf[cost])

    edges = edges + startpoints.edges
    costs = costs + calculate_costs(startpoints.dists, cost, cost_to_nodes)

    if endpoints is not None:
        edges = edges + endpoints.edges
        costs = costs + calculate_costs(endpoints.dists, cost, cost_to_nodes)

    graph = igraph.Graph.TupleList(edges, directed=directed)
    assert len(graph.get_edgelist()) == len(costs)

    graph.es["weight"] = costs
    assert min(graph.es["weight"]) > 0

    return graph


def calculate_costs(distances, cost, cost_to_nodes):
    """
    Gjør om meter to minutter for lenkene mellom punktene og nabonodene.
    og ganger luftlinjeavstanden med 1.5 siden det alltid er svinger i Norge.
    Gjør ellers ingentinnw.
    """

    if cost_to_nodes == 0:
        return [0 for _ in distances]

    elif "meter" in cost:
        return [x * 1.5 for x in distances]

    elif "min" in cost:
        return [(x * 1.5) / (16.666667 * cost_to_nodes) for x in distances]

    else:
        return distances


def validate_cost(gdf, cost, raise_error: bool = True) -> None:
    if cost in gdf.columns:
        warn_if_nans_or_negative(gdf, cost)

        try:
            gdf[cost] = gdf[cost].astype(float)
        except ValueError as e:
            raise ValueError(
                f"The 'cost' column must be numeric. Got characters that couldn't "
                f"be interpreted as numbers."
            )

        if "min" in cost:
            cost = "minutes"

    if "meter" in cost or "metre" in cost:
        if gdf.crs == 4326:
            raise ValueError(
                "'roads' cannot have crs 4326 (latlon) when cost is 'meters'."
            )

        cost = "meters"
        gdf[cost] = gdf.length
        return gdf

    if cost == "minutes" and "minutes" not in gdf.columns:
        if raise_error:
            raise KeyError(
                f"Cannot find 'cost' column for minutes. \nTry running one of the "
                f"'make_directed_network_' methods, or set 'cost' to 'meters'."
            )


def warn_if_nans_or_negative(gdf, cost):
    """Give a warning if there are NaNs or negative values in the gdf."""

    if all(gdf[cost].isna()):
        raise ValueError("All values in the 'cost' column are NaN.")

    nans = sum(gdf[cost].isna())
    if nans:
        if nans > len(gdf) * 0.05:
            warnings.warn(
                f"Warning: {nans} rows have missing values in the 'cost' column. "
                f"Removing these rows."
            )
        gdf = gdf.loc[gdf[cost].notna()]

    negative = sum(gdf[cost] < 0)
    if negative:
        if negative > len(gdf) * 0.05:
            warnings.warn(
                f"Warning: {negative} rows have a 'cost' less than 0. "
                f"Removing these rows."
            )
        gdf = gdf.loc[gdf[cost] >= 0]


"""
cost, cost_to_nodes, startpoints, endpoints, search_factor, search_tolerance, 
network.gdf
"""

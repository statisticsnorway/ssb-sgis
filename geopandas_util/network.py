
import warnings
from shapely import line_merge
from shapely.constructive import reverse
from igraph import Graph
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame, RangeIndex
from abc import ABC
from copy import copy, deepcopy

from .networkanalysis import NetworkAnalysis
from .network_functions import ZeroRoadsError, close_network_holes, find_isolated_networks, make_node_ids

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


class Network(NetworkAnalysis):
    """Parent network class containing methods common for directed and undirected networks. """

    def __init__(
        self,
        roads: GeoDataFrame,
        cost: str,
        **kwargs,
        ):

        if not isinstance(roads, GeoDataFrame):
            raise TypeError(f"'roads' should be GeoDataFrame, got {type(roads)}")
        
        if not len(roads):
            raise ZeroRoadsError

        super().__init__(**kwargs)

        self.network = roads
        self.network["idx_orig"] = roads.index

        self.cost = cost

        self.make_node_ids()

    def prepare_network(self) -> None:
        """Make the necessary adjustments to the road network before network analysis can start. """
        
        if not self.network._geometry_column_name == "geometry":
            self.network = self.network.rename_geometry('geometry')
        
        self.network = (
            self.network
            .pipe(clean_geoms)
            .assign(geometry=lambda x: line_merge(x.geometry))
        )

        if not len(self.network):
            raise ZeroRoadsError

        n = len(self.network)
        self.network = self.network.explode(ignore_index=True)
        if len(self.network) < n:
            if n-len(self.network)==1:
                print(
                    f"1 multi-geometry was split into single part geometries. Minute column(s) will be wrong for these rows."
                    )
            else:
                print(
                    f"{n-len(self.network)} multi-geometries were split into single part geometries. Minute column(s) will be wrong for these rows."
                )
            
        self.network["idx"] = self.network.index

    def make_node_ids(self) -> None:
        self.network, self._nodes = make_node_ids(self.network)

    def close_network_holes(self, max_dist, deadends_only=False, min_dist=0, hole_col = "hole") -> None:
        self.network = close_network_holes(self.network, max_dist, min_dist, deadends_only, min_dist, hole_col)
        return self

    def find_isolated(self, max_length: int, remove=False) -> None:
        if "isolated" in self.network.columns:
            warning.warn("Warning: there is already a column named 'isolated' in your network. Try running .remove_isolated(max_length) to remove the isolated networks.")
        self.network = find_isolated_networks(self.network, max_length)
        if remove:
            self.network = self.network.loc[self.network.isolated == 0]

        return self

    def remove_isolated(self, max_length: int = None) -> None:
        if not "isolated" in self.network.columns:
            if not max_length:
                raise ValueError("'max_length' has to be specified when there is no column 'isolated' in the network.")
            self.network = find_isolated_networks(self.network, max_length)
        self.network = self.network.loc[self.network.isolated == 0]

        return self

    def cut_lines(self) -> None:
        pass
    
    @property
    def nodes(self):
        return self._nodes


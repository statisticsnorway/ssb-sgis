import warnings
from shapely import line_merge
from geopandas import GeoDataFrame
import numpy as np

from .networkanalysis import NetworkAnalysis

from .network_functions import (
    ZeroRoadsError,
    close_network_holes, 
    make_node_ids,
    find_isolated_components,
    cut_lines,
)

from .geopandas_utils import clean_geoms


class Network(NetworkAnalysis):

    def __init__(
        self,
        roads: GeoDataFrame,
        cost: str = "meters",
        directed: bool = False,
        **kwargs,
        ):

        super().__init__(cost=cost, **kwargs)

        if not isinstance(roads, GeoDataFrame):
            raise TypeError(f"'roads' should be GeoDataFrame, got {type(roads)}")
        
        if not len(roads):
            raise ZeroRoadsError

        self.network = roads
        self.network["idx_orig"] = roads.index

        self.directed = directed

        self.prepare_network()

        self.validate_cost(raise_error=False)

        self.make_node_ids()

    def prepare_network(self) -> None:
        """Make the necessary adjustments to the road network before network analysis can start.
        Rename geometry column to 'geometry', 
        merge Linestrings rowwise.
        keep only (Multi)LineStrings, then split MultiLineStrings into LineStrings.
        Remove LinearRings, split into singlepart LineStrings""" 
        
        if not self.network._geometry_column_name == "geometry":
            self.network = self.network.rename_geometry('geometry')
        
        self.network = (
            clean_geoms(self.network, single_geom_type=True)
            .assign(geometry=lambda x: line_merge(x.geometry))
        )

        if not len(self.network):
            raise ZeroRoadsError

        # make sure there are only singlepart LineStrings in the network

        n = len(self.network)
        self.network = self.network.loc[self.network.geom_type != "LinearRing"]
        if (diff := n-len(self.network)):
            if diff == 1:
                print(f"{diff} LinearRing was removed from the network.")
            else:
                print(f"{diff} LinearRings were removed from the network.")

        n = len(self.network)
        self.network = self.network.explode(ignore_index=True)
        if (diff := n-len(self.network)):
            if diff == 1:
                print(
                    f"1 multi-geometry was split into single part geometries. Minute column(s) will be wrong for these rows."
                    )
            else:
                print(
                    f"{diff} multi-geometries were split into single part geometries. Minute column(s) will be wrong for these rows."
                )

        self.network["idx"] = self.network.index

    def make_node_ids(self) -> None:
        self.network, self._nodes = make_node_ids(self.network)

    def close_network_holes(self, max_dist, min_dist=0, deadends_only=False, hole_col = "hole") -> None:
        self.network = close_network_holes(self.network, max_dist, min_dist, deadends_only, hole_col)
        return self

    def find_isolated(self, remove=False) -> None:
        if "isolated" in self.network.columns:
            warnings.warn("There is already a column named 'isolated' in your network. Run .remove_isolated() to remove the isolated networks.")
        self.network = find_isolated_components(self.network)
        if remove:
            self.network = self.network.loc[self.network.isolated == 0]

        return self

    def remove_isolated(self) -> None:
        if not "isolated" in self.network.columns:
            self.network = find_isolated_components(self.network)
        
        self.network = self.network.loc[self.network.isolated == 0]

        return self

    def cut_lines(self, max_length) -> None:
        self.network = cut_lines(self, max_length)
    
    @property
    def nodes(self):
        return self._nodes

    def __repr__(self) -> str:
        return f"""
Network class instance with {len(self.network)} rows.
- cost: {self.cost}
- search_tolerance: {self.search_tolerance} meters
- search_factor: {self.search_factor} %
- cost_to_nodes: {self.cost_to_nodes} km/h
"""
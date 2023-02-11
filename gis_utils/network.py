import warnings
from shapely import line_merge
from geopandas import GeoDataFrame
from pandas import RangeIndex
import numpy as np

from .exceptions import ZeroRowsError

from .network_functions import (
    close_network_holes, 
    make_node_ids,
    get_largest_component,
    get_component_size,
    cut_lines,
)

from .geopandas_utils import clean_geoms


class Network:

    def __init__(
        self,
        gdf: GeoDataFrame,
        merge_lines: bool = True,
        ):

        if not isinstance(gdf, GeoDataFrame):
            raise TypeError(f"'lines' should be GeoDataFrame, got {type(gdf)}")
        
        if not len(gdf):
            raise ZeroRowsError

        self._gdf = self.prepare_network(gdf, merge_lines)

        self.make_node_ids()

    @staticmethod
    def prepare_network(
        gdf: GeoDataFrame, 
        merge_lines: bool = True
        ) -> GeoDataFrame:
        """Make sure there are only singlepart LineStrings in the network.
        This is needed when making node-ids based on the lines' endpoints, because MultiLineStrings have more than two endpoints, and LinearRings have zero.
        Rename geometry column to 'geometry', 
        merge Linestrings rowwise.
        keep only (Multi)LineStrings, then split MultiLineStrings into LineStrings.
        Remove LinearRings, split into singlepart LineStrings

        Args: 
            gdf: GeoDataFrame with (multi)line geometries. MultiLineStrings will be merged, then split if not possible to merge.
            merge_lines (bool): merge MultiLineStrings into LineStrings rowwise. No rows will be dissolved. 
                If false, the network might get more and shorter lines, making network analysis more accurate, but possibly slower. 
                Might also make minute column wrong.
        
        """ 
        
        gdf["idx_orig"] = gdf.index

        if not gdf._geometry_column_name == "geometry":
            gdf = gdf.rename_geometry('geometry')
        
        gdf = clean_geoms(gdf, single_geom_type=True)

        if not len(gdf):
            raise ZeroRowsError

        if merge_lines:
            gdf.geometry = line_merge(gdf.geometry)

        rows_now = len(gdf)
        gdf = gdf.loc[gdf.geom_type != "LinearRing"]

        if (diff := rows_now - len(gdf)):
            if diff == 1:
                print(f"{diff} LinearRing was removed from the network.")
            else:
                print(f"{diff} LinearRings were removed from the network.")

        rows_now = len(gdf)
        gdf = gdf.explode(ignore_index=True)

        if (diff := rows_now - len(gdf)):
            if diff == 1:
                print(
                    f"1 multi-geometry was split into single part geometries. Minute column(s) will be wrong for these rows."
                    )
            else:
                print(
                    f"{diff} multi-geometries were split into single part geometries. Minute column(s) will be wrong for these rows."
                )
        
        return gdf

    def make_node_ids(self) -> None:
        self._gdf, self._nodes = make_node_ids(self._gdf)

    def close_network_holes(self, max_dist, min_dist=0, deadends_only=False, hole_col = "hole"):
        self.gdf = close_network_holes(self.gdf, max_dist, min_dist, deadends_only, hole_col)
        return self

    def get_largest_component(self, remove: bool = False):
        if "connected" in self._gdf.columns:
            warnings.warn("There is already a column 'connected' in the network. Run .remove_connected() if you want to remove the connected networks.")
        self._gdf = get_largest_component(self._gdf)
        if remove:
            self._gdf = self._gdf.loc[self._gdf.connected == 1]
            self.make_node_ids()

        return self

    def get_component_size(self):
        self._gdf = get_component_size(self._gdf)
        return self

    def remove_isolated(self):
        if not "connected" in self._gdf.columns:
            self._gdf = get_largest_component(self._gdf)
        
        self._gdf = self._gdf.loc[self._gdf.connected == 1]

        return self

    def cut_lines(self, max_length: int):
        self.gdf = cut_lines(self.gdf, max_length)
        return self

    def nodes_are_up_to_date(self):

        if not all(
            self._gdf.source.isin(self._nodes.node_id)
            ) or not all(
                self._gdf.target.isin(self._nodes.node_id)
                ):
            return False
        
        return True

    @property
    def nodes(self):
        """Nodes cannot be altered directly because it has to follow the numeric index. """
        return self._nodes

    @property
    def gdf(self):
        return self._gdf

    @gdf.setter
    def gdf(self, new_gdf):
        """Updates the node ids and source-target columns 
        if the gdf has been altered in a meaningful way,
        meaningful meaning that the rows have changed.
        """
        self._gdf = new_gdf
        if not self.nodes_are_up_to_date():
            self.make_node_ids()
        return self._gdf

    def __repr__(self) -> str:
        return f"Network class instance with {len(self._gdf)} rows and {len(self._gdf.columns)} columns."


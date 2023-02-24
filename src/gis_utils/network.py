import warnings
from copy import copy, deepcopy

import numpy as np
from geopandas import GeoDataFrame
from pandas import DataFrame
from shapely import line_merge

from .exceptions import ZeroRowsError
from .geopandas_utils import clean_geoms, push_geom_col
from .network_functions import (
    close_network_holes,
    cut_lines,
    get_component_size,
    get_largest_component,
    make_node_ids,
)


class Network:
    """
    The Network class is a wrapper around a GeoDataFrame with (Multi)LineStrings.
    It makes sure there are only singlepart LineStrings in the network, and that the nodes are up to date with the
    lines. It also contains methods for optimizing the network before the network analysis.

    Args:
        gdf: a GeoDataFrame of line geometries.
        merge_lines (bool): if True (default), multilinestrings within the same row
          will be merged if they overlap. if False, multilines will be split into
          separate rows of singlepart lines.

    """

    def __init__(
        self,
        gdf: GeoDataFrame,
        merge_lines: bool = True,
        allow_degree_units: bool = False,
    ):
        # the 'directed' attribute will be overridden when initialising the DirectedNetwork class
        self.directed = False

        if not isinstance(gdf, GeoDataFrame):
            raise TypeError(f"'lines' should be GeoDataFrame, got {type(gdf)}")

        if not len(gdf):
            raise ZeroRowsError

        if not len(gdf):
            raise ZeroRowsError

        if not allow_degree_units and gdf.crs.axis_info[0].unit_name == "degree":
            raise ValueError(
                "The crs cannot have degrees as unit. Change to a projected crs with e.g. 'metre' as unit."
                "If you really want to use an unprojected crs, set 'allow_degree_units' to True."
            )

        self.gdf = self._prepare_network(gdf, merge_lines)

        self.make_node_ids()

        # attributes for the log
        if "connected" in self.gdf.columns:
            if all(self.gdf["connected"] == 1):
                self._isolated_removed = True
        else:
            self._isolated_removed = False

        self._percent_directional = self._check_percent_directional()

    def make_node_ids(self) -> None:
        self.gdf, self._nodes = make_node_ids(self.gdf)

    def close_network_holes(
        self, max_dist, min_dist=0, deadends_only=False, hole_col="hole"
    ):
        self._update_nodes_if()
        self.gdf = close_network_holes(
            self.gdf, max_dist, min_dist, deadends_only, hole_col
        )
        return self

    def get_largest_component(self, remove: bool = False):
        self._update_nodes_if()
        if "connected" in self.gdf.columns:
            warnings.warn(
                "There is already a column 'connected' in the network. Run "
                ".remove_isolated() if you want to remove the isolated networks."
            )
        self.gdf = get_largest_component(self.gdf)
        if remove:
            self.gdf = self.gdf.loc[self.gdf.connected == 1]
            self.make_node_ids()
            self._isolated_removed = True

        return self

    def get_component_size(self):
        self._update_nodes_if()
        self.gdf = get_component_size(self.gdf)
        return self

    def remove_isolated(self):
        if not self._nodes_are_up_to_date():
            self.make_node_ids()
            self.gdf = get_largest_component(self.gdf)
        elif not "connected" in self.gdf.columns:
            self.gdf = get_largest_component(self.gdf)

        self.gdf = self.gdf.loc[self.gdf.connected == 1]

        self._isolated_removed = True

        return self

    def cut_lines(self, max_length: int, ignore_index=True):
        self.gdf = cut_lines(self.gdf, max_length=max_length, ignore_index=ignore_index)
        return self

    @staticmethod
    def _prepare_network(gdf: GeoDataFrame, merge_lines: bool = True) -> GeoDataFrame:
        """Make sure there are only singlepart LineStrings in the network.
        This is needed when making node-ids based on the lines' endpoints, because
        MultiLineStrings have more than two endpoints, and LinearRings have zero.
        Rename geometry column to 'geometry',
        merge Linestrings rowwise.
        keep only (Multi)LineStrings, then split MultiLineStrings into LineStrings.
        Remove LinearRings, split into singlepart LineStrings

        Args:
            gdf: GeoDataFrame with (multi)line geometries. MultiLineStrings will be
              merged, then split if not possible to merge.
            merge_lines (bool): merge MultiLineStrings into LineStrings rowwise. No
              rows will be dissolved. If false, the network might get more and shorter
              lines, making network analysis more accurate, but possibly slower.
              Might also make minute column wrong.

        """

        gdf["idx_orig"] = gdf.index

        if not gdf._geometry_column_name == "geometry":
            gdf = gdf.rename_geometry("geometry")

        gdf = clean_geoms(gdf, single_geom_type=True)

        if not len(gdf):
            raise ZeroRowsError

        if merge_lines:
            gdf.geometry = line_merge(gdf.geometry)

        rows_now = len(gdf)
        gdf = gdf.loc[gdf.geom_type != "LinearRing"]

        if diff := rows_now - len(gdf):
            if diff == 1:
                print(f"{diff} LinearRing was removed from the network.")
            else:
                print(f"{diff} LinearRings were removed from the network.")

        rows_now = len(gdf)
        gdf = gdf.explode(ignore_index=True)

        if diff := len(gdf) - rows_now:
            if diff == 1:
                print(
                    f"1 multi-geometry was split into single part geometries. "
                    f"Minute column(s) will be wrong for these rows."
                )
            else:
                print(
                    f"{diff} multi-geometries were split into single part geometries. "
                    f"Minute column(s) will be wrong for these rows."
                )

        return gdf

    def _check_percent_directional(self) -> int:
        """Road data often have to be duplicated and flipped to make it directed."""
        no_dups = DataFrame(
            np.sort(self.gdf[["source", "target"]].values, axis=1),
            columns=[["source", "target"]],
        ).drop_duplicates()

        return int((len(self.gdf) - len(no_dups)) / len(self.gdf) * 100)

    def _nodes_are_up_to_date(self) -> bool:
        """
        Returns False if there are any source or target values not in the node-ids,
        or any superfluous node-ids (meaning rows have been removed from the lines gdf).

        """

        new_or_missing = (~self.gdf.source.isin(self._nodes.node_id)) | (
            ~self.gdf.target.isin(self._nodes.node_id)
        )

        if any(new_or_missing):
            return False

        removed = ~(
            (self._nodes.node_id.isin(self.gdf.source))
            | (self._nodes.node_id.isin(self.gdf.target))
        )

        if any(removed):
            return False

        return True

    def _update_nodes_if(self):
        if not self._nodes_are_up_to_date():
            self.make_node_ids()

    @property
    def nodes(self):
        """Nodes cannot be altered directly because it has to follow the numeric
        index."""
        return self._nodes

    def __repr__(self) -> str:
        cl = self.__class__.__name__
        km = int(sum(self.gdf.length) / 1000)
        return f"{cl}({km} km)"

    def __iter__(self):
        return iter(self.__dict__.values())

    def copy(self):
        return copy(self)

    def deepcopy(self):
        return deepcopy(self)

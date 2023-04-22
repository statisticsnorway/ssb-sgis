"""Network class with methods for manipulating line geometries.

The module includes functions for cutting and splitting lines, filling holes in the
network, finding and removing isolated network islands and creating unique node ids.
"""

import warnings
from copy import copy, deepcopy

import numpy as np
from geopandas import GeoDataFrame
from pandas import DataFrame
from shapely import line_merge

from ..exceptions import ZeroLinesError
from ..geopandas_tools.general import clean_geoms
from ..geopandas_tools.geometry_types import to_single_geom_type
from .nodes import make_node_ids


class Network:
    """Class used in NetworkAnalysis."""

    def __init__(self, gdf: GeoDataFrame):
        """The lines are fixed, welded together rowwise and exploded. Creates node-ids.

        Raises:
            TypeError: If 'gdf' is not of type GeoDataFrame.
            ZeroLinesError: If 'gdf' has zero rows.
        """
        if not isinstance(gdf, GeoDataFrame):
            raise TypeError(f"'lines' should be GeoDataFrame, got {type(gdf)}")

        if not len(gdf):
            raise ZeroLinesError

        self.gdf = self._prepare_network(gdf)

        self._make_node_ids()

        self._percent_bidirectional = self._check_percent_bidirectional()

    def _make_node_ids(self) -> None:
        """Gives the lines node ids and return lines (edges) and nodes.

        Takes the first and last point of each line and creates a GeoDataFrame of
        nodes (points) with a column 'node_id'. The node ids are then assigned to the
        input GeoDataFrame of lines as the columns 'source' and 'target'.

        Note:
            The lines must be singlepart linestrings.
        """
        self.gdf, self._nodes = make_node_ids(self.gdf)

    @staticmethod
    def _prepare_network(gdf: GeoDataFrame) -> GeoDataFrame:
        """Make sure there are only singlepart LineStrings in the network.

        This is needed when making node-ids based on the lines' endpoints, because
        MultiLineStrings have more than two endpoints, and LinearRings have zero.
        Rename geometry column to 'geometry',

        Args:
            gdf: GeoDataFrame with (multi)line geometries. MultiLineStrings will be
                merged, then exploded if a merge was not possible.

        Returns:
            A GeoDataFrame of line geometries.

        Raises:
            ZeroLinesError: If the GeoDataFrame has 0 rows.
        """
        gdf["idx_orig"] = gdf.index

        if gdf._geometry_column_name != "geometry":
            gdf = gdf.rename_geometry("geometry")

        gdf = clean_geoms(gdf)
        gdf = to_single_geom_type(gdf, geom_type="lines")

        if not len(gdf):
            raise ZeroLinesError

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
                    "1 multi-geometry was split into single part geometries. "
                    "Minute column(s) will be wrong for these rows."
                )
            else:
                print(
                    f"{diff} multi-geometries were split into single part geometries. "
                    "Minute column(s) will be wrong for these rows."
                )

        return gdf

    def _check_percent_bidirectional(self) -> int:
        """Road data often have to be duplicated and flipped to make it directed.

        Here we check how.
        """
        self.gdf["meters"] = self.gdf.length.round(10).astype(str)
        no_dups = DataFrame(
            np.sort(self.gdf[["source", "target", "meters"]].values, axis=1),
            columns=[["source", "target", "meters"]],
        ).drop_duplicates()

        # back to numeric
        self.gdf["meters"] = self.gdf.length

        percent_bidirectional = len(self.gdf) / len(no_dups) * 100 - 100

        return int(round(percent_bidirectional, 0))

    def _nodes_are_up_to_date(self) -> bool:
        """Check if nodes need to be updated.

        Returns False if there are any source or target values not in the node-ids,
        or any superfluous node-ids (meaning rows have been removed from the lines
        gdf).
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

    def get_edges(self) -> list[tuple[str, str]]:
        return [
            (str(source), str(target))
            for source, target in zip(
                self.gdf["source"], self.gdf["target"], strict=True
            )
        ]

    @staticmethod
    def _create_edge_ids(
        edges: list[tuple[str, str]], weights: list[float]
    ) -> list[str]:
        """Edge identifiers represented with source and target ids and the weight."""
        return [f"{s}_{t}_{w}" for (s, t), w in zip(edges, weights, strict=True)]

    def _update_nodes_if(self):
        if not self._nodes_are_up_to_date():
            self._make_node_ids()

    @property
    def nodes(self):
        """GeoDataFrame with the network nodes (line endpoints).

        Upon instantiation of the class, a GeoDataFrame of points is created from the
        unique endpoints of the lines. The node ids are then used to make the 'source'
        and 'target' columns of the line gdf. The nodes are remade every time the
        geometries of the line gdf changes, since the 'source' and 'target' columns
        must be up to date with the actual line geometries when making the graph.
        """
        return self._nodes

    def _warn_if_undirected(self):
        """Road data often have to be duplicated and flipped to make it directed."""
        if self.percent_bidirectional > 5:
            return

        mess = (
            "Your network is likely not directed. "
            f"Only {self.percent_bidirectional:.1f} percent of the lines go both ways."
        )
        if "oneway" in [col.lower() for col in self.gdf.columns]:
            mess = mess + (
                " Try setting direction_col='oneway' in the 'make_directed_network' "
                "method"
            )
        else:
            mess = mess + "Try running 'make_directed_network'"

        warnings.warn(mess, stacklevel=2)

    @property
    def percent_bidirectional(self):
        """The percentage of lines that appear in both directions."""
        return self._percent_bidirectional

    def copy(self):
        """Returns a shallow copy of the class instance."""
        return copy(self)

    def deepcopy(self):
        """Returns a deep copy of the class instance."""
        return deepcopy(self)

    def __repr__(self) -> str:
        """The print representation."""
        cl = self.__class__.__name__
        km = int(sum(self.gdf.length) / 1000)
        return f"{cl}({km} km, percent_bidirectional={self._percent_bidirectional})"

    def __iter__(self):
        """So the attributes can be iterated through."""
        return iter(self.__dict__.items())

    def __len__(self):
        return len(self.gdf)

""""""

import warnings
from copy import copy, deepcopy

import numpy as np
from geopandas import GeoDataFrame
from pandas import DataFrame
from shapely import line_merge

from .exceptions import ZeroLinesError
from .geopandas_utils import clean_geoms
from .network_functions import (
    close_network_holes,
    cut_lines,
    get_component_size,
    get_largest_component,
    make_node_ids,
)


# put this a better place
def _edge_ids(
    gdf: GeoDataFrame | list[tuple[int, int]], weight: str | list[float]
) -> list[str]:
    """Quite messy way to deal with different input types."""
    if isinstance(gdf, GeoDataFrame):
        return _edge_id_template(
            zip(gdf["source"], gdf["target"], strict=True),
            weight_arr=gdf[weight],
        )
    if isinstance(gdf, list):
        return _edge_id_template(gdf, weight_arr=weight)


def _edge_id_template(*source_target_arrs, weight_arr):
    """Edge identifiers represented with source and target ids and the weight."""
    return [
        f"{s}_{t}_{w}"
        for (s, t), w in zip(*source_target_arrs, weight_arr, strict=True)
    ]


class Network:
    """Prepares a GeoDataFrame of lines for network analysis.

    The class can be used as the 'network' parameter in the NetworkAnalysis class for
    undirected network analysis. For directed network analysis, use the DirectedNetwork
    class.

    The geometries are made valid and into singlepart LineStrings, and given 'source'
    and 'target' ids based on the first and last point of the lines. The Network
    instance can immediately be used in the NetworkAnalysis class. But the Network
    class also contains methods for optimizing the network further. The most
    important, is the remove_isolated method. It will remove network islands, meaning
    higher success rate in the network analyses. The network islands can be found and
    inspected with the get_largest_component method, or get_component_size to find the
    actual length of each network.

    If the network has a lot of unconnected parts, that are supposed to be
    connected, network holes can be filled with close_network_holes method.
    Often, this should be run before removing the isolated network components.

    Long lines can be cut into equal length pieces with the cut_lines method.
    This is mostly relevant for service_area analysis, since shorter lines
    will give more accurate results.

    All methods return self, and can therefore be chained together. However,
    all methods also overwrite the instance to save memory. Take a copy with the copy
    or deepcopy methods before using another method if you want to save the previous
    instance.

    The 'source' and 'target' ids are also stored as points in the 'nodes' attribute,
    which is always kept up to date with the lines and the actual geometries. The ids
    therefore changes whenever the lines change, so they cannot be used as fixed
    identifiers.

    Attributes:
        gdf: the GeoDataFrame of lines

    See also:
        DirectedNetwork: subclass of Network for directed network analysis

    Examples
    --------

    >>> roads = gpd.read_parquet(filepath_roads)
    >>> nw = Network(roads)
    >>> nw
    Network(3851 km, undirected)

    Check for isolated network islands.

    >>> nw = nw.get_largest_component()
    >>> nw.gdf.connected.value_counts()
    1.0    85638
    0.0     7757
    Name: connected, dtype: int64

    Remove the network islands. The get_largest_component method is not needed
    beforehand.

    >>> nw = nw.remove_isolated()
    >>> nw.gdf.connected.value_counts()
    1.0    85638
    Name: connected, dtype: int64

    Filling small gaps/holes in the network.

    >>> len(nw.gdf)
    85638
    >>> nw = nw.close_network_holes(max_dist=1.5)
    >>> len(nw.gdf)
    86929

    Cutting long lines into pieces. This is only relevant for service area analysis and
    similar analyses.

    >>> nw.gdf.length.max()
    5213.749177803526
    >>> nw = nw.cut_lines(100)
    >>> nw.gdf.length.max()
    100.00000000046512

    """

    def __init__(
        self,
        gdf: GeoDataFrame,
        *,
        merge_lines: bool = True,
        allow_degree_units: bool = False,
    ):
        """The lines are fixed, welded together rowwise and exploded. Creates node-ids.

        Args:
            gdf: a GeoDataFrame of line geometries.
            merge_lines: if True (default), multilinestrings within the same row will
                be merged if they overlap. if False, multilines will be split into
            separate rows of singlepart lines.
            allow_degree_units: If False (the default), it will raise an exception if
                the coordinate reference system of 'gdf' is in degree units, i.e.
                unprojected (4326). If set to True, all crs are allowed, but it might
                raise exceptions and give erronous results.

        Raises:
            TypeError: If 'gdf' is not of type GeoDataFrame
            ZeroLinesError: If 'gdf' has zero rows
            ValueError: If the coordinate reference system is in degree units
        """

        # for the base Network class, the graph will be undirected in network analysis
        self._as_directed = False

        if not isinstance(gdf, GeoDataFrame):
            raise TypeError(f"'lines' should be GeoDataFrame, got {type(gdf)}")

        if not len(gdf):
            raise ZeroLinesError

        if not allow_degree_units and gdf.crs.axis_info[0].unit_name == "degree":
            raise ValueError(
                "The crs cannot have degrees as unit. Change to a projected crs with "
                "e.g. 'metre' as unit. If you really want to use an unprojected crs, "
                "set 'allow_degree_units' to True."
            )

        self.gdf = self._prepare_network(gdf, merge_lines)

        self._make_node_ids()

        # attributes for the log
        if "connected" in self.gdf.columns:
            if all(self.gdf["connected"] == 1):
                self._isolated_removed = True
        else:
            self._isolated_removed = False

        self._percent_bidirectional = self._check_percent_bidirectional()

    def remove_isolated(self):
        """Removes lines not connected to the largest network component.

        It creates a graph and finds the edges that are part of the largest
        connected component of the network. Then removes all lines not part of
        the largest component. Updates node ids if neccessary.

        Note:
            If the nodes are updated and the column already has a column named
            'connected', the network will be filtered based on this column,
            and no graph will be built.

        Returns:
            Self

        See also:
            get_largest_component: to find the isolated lines without removing them
            get_component_size: to get the exact number of lines in the component

        Examples
        --------
        >>> len(nw.gdf)
        93395
        >>> nw = nw.remove_isolated()
        >>> len(nw.gdf)
        85638

        """
        if not self._nodes_are_up_to_date():
            self._make_node_ids()
            self.gdf = get_largest_component(self.gdf)
        elif "connected" not in self.gdf.columns:
            self.gdf = get_largest_component(self.gdf)

        self.gdf = self.gdf.loc[self.gdf.connected == 1]

        self._isolated_removed = True

        return self

    def get_largest_component(self):
        """Create column 'connected' in the network.gdf, where '1' means connected.

        It takes the lines of the network, creates a graph, finds the largest component,
        and maps this as the value '1' in the column 'connected' and gives the rest the
        value 0.

        Returns:
            self

        See also:
            remove_isolated: to remove the isolated lines from the network
            get_component_size: to get the exact number of lines in the component

        Examples
        --------
        >>> nw = Network(roads)
        >>> nw = nw.get_largest_component()
        >>> nw.gdf.connected.value_counts()
        1.0    85638
        0.0     7757
        Name: connected, dtype: int64
        """
        self._update_nodes_if()
        if "connected" in self.gdf.columns:
            warnings.warn(
                "There is already a column 'connected' in the network. Run "
                ".remove_isolated() if you want to remove the isolated networks."
                "And write nw.gdf.connected... if you want to access the column"
            )
        self.gdf = get_largest_component(self.gdf)

        return self

    def get_component_size(self):
        """Create column 'connected' in the network.gdf, where '1' means connected.

        that the line is part of the largest
        component of the network.

        It takes the lines of the network, creates a graph, finds the largest component,
        and maps this as the value '1' in the column 'connected'.

        Returns:
            self

        See also:
            get_largest_component: to find the isolated lines without removing them
            get_component_size: to get the exact number of lines in the component

        Examples
        --------
        >>> nw = Network(roads)
        >>> nw = nw.get_component_size()
        >>> nw.gdf.component_size.value_counts().head()
        79180    85638
        2         1601
        4          688
        6          406
        3          346
        Name: component_size, dtype: int64
        """
        self._update_nodes_if()
        self.gdf = get_component_size(self.gdf)
        return self

    def close_network_holes(
        self, max_dist, min_dist=0, deadends_only=False, hole_col="hole"
    ):
        """Fills holes in the network lines shorter than the max_dist.

        It fills holes in the network by finding the nearest neighbors of each node,
        then connecting the nodes that are within the max_dist of each other. The
        minimum distance is set to 0, but can be changed with the min_dist parameter.

        Args:
            max_dist: The maximum distance between two nodes to be considered a hole.
            min_dist: minimum distance between nodes to be considered a hole. Defaults
                to 0
            deadends_only: If True, only holes between two deadends will be filled.
                If False (the default), deadends might be connected to any node of the
                network.
            hole_col: Holes will get the value 1 in a column named 'hole' by default,
                or what is specified as hole_col. If set to None or False, no column
                will be added.

        Returns:
            The input GeoDataFrame with new lines added

        Examples
        --------

        Let's compare the number of isolated/unconnected lines before and after closing
        holes.

        >>> nw = gs.Network(roads)
        >>> nw = nw.get_largest_component()
        >>> nw.gdf.connected.value_counts()
        1.0    85638
        0.0     7757
        Name: connected, dtype: int64
        >>> nw = nw.close_network_holes(max_dist=1.1)
        >>> nw = nw.get_largest_component()
        >>> nw.gdf.connected.value_counts()
        1.0    100315
        0.0       180
        Name: connected, dtype: int64

        """
        self._update_nodes_if()
        self.gdf = close_network_holes(
            self.gdf, max_dist, min_dist, deadends_only, hole_col
        )
        return self

    def cut_lines(
        self, max_length: int, adjust_weight_col: str | None = None, ignore_index=True
    ):
        """Cuts lines into pieces no longer than 'max_length'.

        Args:
            max_length: The maximum length of the line segments.
            adjust_weight_col: If you have a column in your GeoDataFrame that you want
                to adjust based on the length of the new lines, you can pass the name
                of that column here. For example, if you have a column called
                "minutes", the minute value will be halved if the line is halved.
            ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
                Defaults to True

        Returns:
            Self

        Examples
        --------
        >>> nw = gs.Network(roads)
        >>> nw.gdf.length.describe().round(1)
        count    93395.0
        mean        41.2
        std         78.5
        min          0.2
        25%         14.0
        50%         27.7
        75%         47.5
        max       5213.7
        dtype: float64
        >>> nw = nw.cut_lines(max_length=100)
        >>> nw.gdf.length.describe().round(1)
        count    126304.0
        mean         30.5
        std          30.1
        min           0.0
        25%           5.7
        50%          22.5
        75%          44.7
        max         100.0
        dtype: float64
        """
        if adjust_weight_col:
            if adjust_weight_col not in self.gdf.columns:
                raise KeyError(f"'gdf' has no column {adjust_weight_col}")
            self.gdf["original_length"] = self.gdf.length
        self.gdf = cut_lines(self.gdf, max_length=max_length, ignore_index=ignore_index)

        if adjust_weight_col:
            self.gdf[adjust_weight_col] = self.gdf[adjust_weight_col] * (
                self.gdf.length / self.gdf[adjust_weight_col]
            )

        return self

    def _make_node_ids(self) -> None:
        """Gives the lines node ids and return lines (edges) and nodes.

        Takes the first and last point of each line and creates a GeoDataFrame of
        nodes (points) with a column 'node_id'. The node ids are then assigned to the
        input GeoDataFrame of lines as the columns 'source' and 'target'.

        Note:
            The lines must be singlepart linestrings
        """
        self.gdf, self._nodes = make_node_ids(self.gdf)

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

        if gdf._geometry_column_name != "geometry":
            gdf = gdf.rename_geometry("geometry")

        gdf = clean_geoms(gdf, geom_type="lines")

        if not len(gdf):
            raise ZeroLinesError

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

        gdf["meters"] = gdf.length

        return gdf

    def _check_percent_bidirectional(self) -> int:
        """Road data often have to be duplicated and flipped to make it directed.
        Here we check how"""
        self.gdf["meters"] = self.gdf["meters"].astype(str)
        no_dups = DataFrame(
            np.sort(self.gdf[["source", "target", "meters"]].values, axis=1),
            columns=[["source", "target", "meters"]],
        ).drop_duplicates()
        self.gdf["meters"] = self.gdf.length

        percent_bidirectional = len(self.gdf) / len(no_dups) * 100 - 100

        return int(round(percent_bidirectional, 0))

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

    @property
    def as_directed(self):
        """This attribute decides whether the graph should be made directed or not.
        This depends on what network class is used. 'as_directed' is False for the
        base Network class and True for the DirectedNetwork subclass.
        """
        return self._as_directed

    @property
    def percent_bidirectional(self):
        """The percentage of lines that appear in both directions."""
        return self._percent_bidirectional

    def __repr__(self) -> str:
        cl = self.__class__.__name__
        km = int(sum(self.gdf.length) / 1000)
        return f"{cl}({km} km, undirected)"

    def __iter__(self):
        return iter(self.__dict__.values())

    def copy(self):
        return copy(self)

    def deepcopy(self):
        return deepcopy(self)

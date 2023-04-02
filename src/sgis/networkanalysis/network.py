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
from ..geopandas_tools.line_operations import (
    close_network_holes,
    close_network_holes_to_deadends,
    cut_lines,
    get_component_size,
    get_largest_component,
    make_node_ids,
)


class Network:
    """Prepares a GeoDataFrame of lines for network analysis.

    The class can be used as the 'network' parameter in the NetworkAnalysis class for
    undirected network analysis. For directed network analysis, use the DirectedNetwork
    class.

    The Network class also contains methods for optimizing the network. The
    remove_isolated method will remove isolated network islands. This often
    dramatically increases the success rate of the network analyses. The network
    islands can also be inspected with the get_largest_component or
    get_component_size methods.

    If the network contains holes/gaps that are not supposed to be there, the
    close_network_holes method can be used. This will fill any network holes
    shorter than a given distance. Often, this should be run before removing the
    isolated network components.

    Long lines can be cut into equal length pieces with the cut_lines method.
    This is mainly relevant for service_area analysis, since shorter lines
    will give more accurate results here.

    All methods can be chained together like in pandas. However, all methods also
    overwrite the prevous object to save memory. Take a copy with the 'copy' or
    'deepcopy' methods before using another method if you want to save the previous
    instance.

    The 'source' and 'target' ids are also stored as points in the 'nodes' attribute,
    which is always kept up to date with the lines and the actual geometries. The ids
    therefore changes whenever the lines change, so they cannot be used as fixed
    identifiers.

    Args:
        gdf: a GeoDataFrame of line geometries.
        merge_lines: if True (default), multilinestrings within the same row will
            be merged if they overlap. if False, multilines will be split into
            separate rows of singlepart lines.
        allow_degree_units: If False (default), it will raise an exception if
            the coordinate reference system of 'gdf' is in degree units, i.e.
            unprojected (4326). If set to True, all crs are allowed, but it might
            raise exceptions and give erronous results.

    Attributes:
        gdf: the GeoDataFrame of lines with source and target ids.
        nodes: GeoDataFrame of points with unique ids. The ids are dynamic, and changes
            every time the lines change. The nodes cannot be changed directly.

    See also
    --------
    DirectedNetwork: subclass of Network for directed network analysis.

    Examples
    --------
    Read example data.

    >>> from sgis import read_parquet_url
    >>> roads = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")
    >>> nw = Network(roads)
    >>> nw
    Network(3851 km, undirected)

    Check for isolated network islands.

    >>> nw = nw.get_largest_component()
    >>> nw.gdf.connected.value_counts()
    1.0    85638
    0.0     7757
    Name: connected, dtype: int64

    Remove the network islands.

    >>> nw = nw.remove_isolated()
    >>> nw.gdf.connected.value_counts()
    1.0    85638
    Name: connected, dtype: int64

    Filling small gaps/holes in the network.

    >>> len(nw.gdf)
    85638
    >>> nw = nw.close_network_holes(max_distance=1.5, fillna=0)
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
        allow_degree_units: bool = False,
    ):
        """The lines are fixed, welded together rowwise and exploded. Creates node-ids.

        Raises:
            TypeError: If 'gdf' is not of type GeoDataFrame.
            ZeroLinesError: If 'gdf' has zero rows.
            ValueError: If the coordinate reference system is in degree units and
                allow_degree_units is False.
        """
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

        self.gdf = self._prepare_network(gdf)

        self._make_node_ids()

        # attributes for the log
        if "connected" in self.gdf.columns:
            if all(self.gdf["connected"] == 1):
                self._isolated_removed = True
        else:
            self._isolated_removed = False

        self._percent_bidirectional = self._check_percent_bidirectional()

        # for the base Network class, the graph will be undirected in network analysis
        self._as_directed = False

    def remove_isolated(self):
        """Removes lines not connected to the largest network component.

        It creates a graph and finds the edges that are part of the largest
        connected component of the network. Then removes all lines not part of
        the largest component. Updates node ids if neccessary.

        Returns:
            Self

        See also:
            get_largest_component: to find the isolated lines without removing them
            get_component_size: to get the exact number of lines in the component

        Examples
        --------
        Compare the number of rows.

        >>> from sgis import Network, read_parquet_url
        >>> roads = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")
        >>> nw = Network(roads)
        >>> len(nw.gdf)
        93395
        >>> nw = nw.remove_isolated()
        >>> len(nw.gdf)
        85638

        Removing isolated can have a large inpact on the number of missing values in
        network analyses.

        >>> nwa = NetworkAnalysis(
        ...     network=Network(roads),
        ...     rules=NetworkAnalysisRules(weight="meters")
        ... )
        >>> od = nwa.od_cost_matrix(points, points)
        >>> nwa.network = nwa.network.remove_isolated()
        >>> od = nwa.od_cost_matrix(points, points)
        >>> nwa.log[['isolated_removed', 'percent_missing']]
        isolated_removed  percent_missing
        0             False          23.2466
        1              True           0.3994
        """
        if not self._nodes_are_up_to_date():
            self._make_node_ids()

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
                "And write nw.gdf.connected... if you want to access the column",
                stacklevel=2,
            )
        self.gdf = get_largest_component(self.gdf)

        return self

    def get_component_size(self):
        """Finds the size of each component in the network.

        Creates the column "component_size" in the 'gdf' of the network, which
        indicates the number of lines in the network component the line is a part of.

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
        self,
        max_distance: int | float,
        max_angle: int,
        fillna: float | int | None,
        hole_col: str = "hole",
    ):
        """Fills network gaps with straigt lines.

        Fills holes in the network by connecting deadends with the the closest node
        that is within the 'max_distance' distance and have an angle less 'max_angle'.

        Args:
            max_distance: The maximum distance for the holes to be filled.
            max_angle: Absolute number between 0 and 180 that represents the maximum
                difference in angle between the new line and the prior, i.e. the line
                at which the deadend ends. A value of 0 means the new lines must
                have the exact same angle as the prior line, and 180 means the new
                lines can go in any direction.
            fillna: Numeric value to assign to all NaN values of the holes. This is
                chiefly the 'weight' column that is to be used in network analysis.
            hole_col: Holes will get the value 1 in a column named 'hole' by default,
                or what is specified as hole_col. If set to None or False, no column
                will be added.

        Returns:
            The input GeoDataFrame with new lines added.

        Note:
            The 'fillna' parameter has no default value, but can be set to None if NaN
            values should be preserved. This will, however, mean that the filled holes
            will be opened again in network analysis.

        Examples
        --------
        Check for number of isolated lines before filling gaps.

        >>> nw = Network(roads)
        >>> nw = nw.get_largest_component()
        >>> nw.gdf.connected.value_counts()
        1.0    85638
        0.0     7757
        Name: connected, dtype: int64

        Fill gaps shorter than 1.1 meters and an angle deviation of no more than
        30 degrees.

        >>> nw = nw.close_network_holes(max_distance=1.1, max_angle=30, fillna=0)
        >>> nw = nw.get_largest_component()
        >>> nw.gdf.connected.value_counts()
        1.0    100315
        0.0       191
        Name: connected, dtype: int64

        Fill gaps with all angles allowed.

        >>> nw = nw.close_network_holes(max_distance=1.1, max_angle=180, fillna=0)
        >>> nw = nw.get_largest_component()
        >>> nw.gdf.connected.value_counts()
        1.0    100315
        0.0       180
        Name: connected, dtype: int64

        It's not always wise to fill gaps. In the case of this data, these small gaps
        are intentional. They are road blocks where most cars aren't allowed to pass.
        Fill the holes only if it makes the travel times/routes more realistic.
        """
        self.gdf = close_network_holes(
            self.gdf,
            max_distance,
            hole_col=hole_col,
            max_angle=max_angle,
        )

        if fillna is not None:
            self.gdf.loc[self.gdf[hole_col] == 1] = self.gdf.loc[
                self.gdf[hole_col] == 1
            ].fillna(fillna)

        self._hole_col = hole_col

        return self

    def close_network_holes_to_deadends(
        self,
        max_distance: int | float,
        fillna: int | float | None,
        hole_col: str = "hole",
    ):
        """Fills holes between two deadends if the distance is less than the max_distance.

        It fills gaps in the network by finding the nearest neighbors of each node,
        then creating straigt lines between the nodes that are within the 'max_distance'
        of each other.

        Args:
            max_distance: The maximum distance between two nodes to be considered a hole.
            fillna: The value to give the closed holes in all columns with NaN. This
                has to be set in order for the closed holes to be included in network
                analyses. If set to 0, the lines will get a weight of 0.
            hole_col: Holes will get the value 1 in a column named 'hole' by default,
                or what is specified as hole_col. If set to None or False, no column
                will be added.

        Returns:
            The input GeoDataFrame with new lines added.

        Note:
            The 'fillna' parameter has no default value, but can be set to None if NaN
            values should be preserved. This will, however, mean that the filled holes
            will be opened again in network analysis.

        Examples
        --------
        Check for number of isolated lines before filling gaps.

        >>> nw = Network(roads)
        >>> nw = nw.get_largest_component()
        >>> nw.gdf.connected.value_counts()
        1.0    85638
        0.0     7757
        Name: connected, dtype: int64

        Fill gaps shorter than 1.1 meters and assign 0 to all columns.

        >>> nw = nw.close_network_holes(max_distance=1.1, fillna=0)
        >>> nw = nw.get_largest_component()
        >>> nw.gdf.connected.value_counts()
        1.0    100315
        0.0       180
        Name: connected, dtype: int64

        It's not always wise to fill gaps. In the case of this data, these small gaps
        are intentional. They are road blocks where most cars aren't allowed to pass.
        Fill the holes only if it makes the travel times/routes more realistic.
        """
        self.gdf = close_network_holes_to_deadends(
            self.gdf,
            max_distance,
            hole_col=hole_col,
        )

        if fillna is not None:
            self.gdf.loc[self.gdf[hole_col] == 1] = self.gdf.loc[
                self.gdf[hole_col] == 1
            ].fillna(fillna)

        self._hole_col = hole_col

        return self

    def cut_lines(
        self,
        max_length: int | float,
        adjust_weight_col: str | None = None,
        ignore_index: bool = False,
    ):
        """Cuts lines into pieces no longer than 'max_length'.

        Will give more accurate results in the service_area method, but not in the
        precice_service_area method. Shorter lines also mean more holes might be filled
        in close_network_holes. Other than that, cutting lines has no impact on network
        analysis except making it slower.

        Args:
            max_length: The maximum length of the line segments.
            adjust_weight_col: Name of a numeric column in the GeoDataFrame to
                be adjusted based on the length of the new lines. For example, if the
                column is "minutes", the minute value will be halved if the line is
                halved in length.
            ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
                Defaults to False.

        Returns:
            Self

        Raises:
            KeyError: If the specified 'adjust_weight_col' column is not present in the
                GeoDataFrame of the network.

        Note:
            This method is time consuming for large networks and low 'max_length'.

        Examples
        --------
        >>> nw = Network(roads)
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

        gdf["meters"] = gdf.length

        return gdf

    def _check_percent_bidirectional(self) -> int:
        """Road data often have to be duplicated and flipped to make it directed.

        Here we check how.
        """
        self.gdf["meters"] = self.gdf["meters"].astype(str)
        no_dups = DataFrame(
            np.sort(self.gdf[["source", "target", "meters"]].values, axis=1),
            columns=[["source", "target", "meters"]],
        ).drop_duplicates()
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
        return f"{cl}({km} km, undirected)"

    def __iter__(self):
        """So the attributes can be iterated through."""
        return iter(self.__dict__.items())

    def __len__(self):
        return len(self.gdf)


# TODO: put these a better place:


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

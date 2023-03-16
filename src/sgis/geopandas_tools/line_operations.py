"""Functions for line geometries.

The module includes functions for cutting and splitting lines, cutting lines into 
pieces, filling holes in a network of lines, finding isolated network islands and
creating unique node ids.

The functions are also methods of the Network class, where some checks and
preperation is done before each method is run, making sure the lines are correct.
"""

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame
from shapely import force_2d, shortest_line
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from .buffer_dissolve_explode import buff
from .neighbors import get_k_nearest_neighbors, k_nearest_neighbors
from .general import gdf_concat, to_gdf, _push_geom_col, coordinate_array
from .point_operations import (
    snap_to,
)


def get_largest_component(lines: GeoDataFrame) -> GeoDataFrame:
    """Finds the largest network component.

    It takes a GeoDataFrame of lines, creates a graph, finds the largest component,
    and maps this as the value '1' in the column 'connected'.

    Args:
        lines: A GeoDataFrame of lines.

    Returns:
        A GeoDataFrame with a new column "connected".

    Note:
        If the lines have the columns 'source' and 'target', these will be used as
        node ids. If these columns are incorrect, run 'make_node_ids' first.

    Examples
    --------
    >>> from sgis import read_parquet_url, get_largest_component
    >>> roads = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")

    >>> roads = get_largest_component(roads)
    >>> roads.connected.value_counts()
    1.0    85638
    0.0     7757
    Name: connected, dtype: int64
    """
    if "source" not in lines.columns or "target" not in lines.columns:
        lines, _ = make_node_ids(lines)

    edges = [
        (str(source), str(target))
        for source, target in zip(lines["source"], lines["target"], strict=True)
    ]

    graph = nx.Graph()
    graph.add_edges_from(edges)

    largest_component = max(nx.connected_components(graph), key=len)

    largest_component_dict = {node_id: 1 for node_id in largest_component}

    lines["connected"] = lines.source.map(largest_component_dict).fillna(0)

    return lines


def get_component_size(lines: GeoDataFrame) -> GeoDataFrame:
    """Finds the size of each component in the network.

    Creates the column "component_size", which indicates the size of the network
    component the line is a part of.

    Args:
        lines: GeoDataFrame

    Returns:
        A GeoDataFrame with a new column "component_size".

    Note:
        If the lines have the columns 'source' and 'target', these will be used as
        node ids. If these columns are incorrect, run 'make_node_ids' first.

    Examples
    --------
    >>> from sgis import read_parquet_url, get_component_size
    >>> roads = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")

    >>> roads = get_component_size(roads)
    >>> roads.component_size.value_counts().head()
    79180    85638
    2         1601
    4          688
    6          406
    3          346
    Name: component_size, dtype: int64
    """
    if "source" not in lines.columns or "target" not in lines.columns:
        lines, _ = make_node_ids(lines)

    edges = [
        (str(source), str(target))
        for source, target in zip(lines["source"], lines["target"], strict=True)
    ]

    graph = nx.Graph()
    graph.add_edges_from(edges)
    components = [list(x) for x in nx.connected_components(graph)]

    componentsdict = {
        idx: len(component) for component in components for idx in component
    }

    lines["component_size"] = lines.source.map(componentsdict)

    return lines


def split_lines_at_closest_point(
    lines: GeoDataFrame,
    points: GeoDataFrame,
    max_dist: int | None = None,
) -> DataFrame:
    """Split lines where nearest to a set of points.

    Snaps points to lines and splits the lines in two at the snap point. The splitting
    is done pointwise, meaning each point splits one line in two. The line will not be
    split if the point is closest to the endpoint of the line.

    Args:
        lines: GeoDataFrame of lines that will be split.
        points: GeoDataFrame of points to split the lines with.
        max_dist: the maximum distance between the point and the line.
            Points further away than max_dist will not split any lines.
            Defaults to None.

    Returns:
        A GeoDataFrame with the same columns as the input lines, but with the lines
        split at the closest point to the points.

    Raises:
        ValueError: If the crs of the input data differs.

    Examples
    --------
    >>> from sgis import read_parquet_url, split_lines_at_closest_point
    >>> roads = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")
    >>> points = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet")
    >>> rows = len(roads)
    >>> rows
    93395

    Splitting lines for points closer than 10 meters from the lines.

    >>> roads = split_lines_at_closest_point(roads, points, max_dist=10)
    >>> print("number of lines that were split:", len(roads) - rows)
    number of lines that were split: 380

    Splitting lines by all points.

    >>> roads = split_lines_at_closest_point(roads, points)
    >>> print("number of lines that were split:", len(roads) - rows)
    number of lines that were split: 848

    Not all lines were split. That is because some points were closest to an endpoint
    of a line.
    """
    BUFFDIST = 0.000001

    if points.crs != lines.crs:
        raise ValueError("crs mismatch:", points.crs, "and", lines.crs)

    lines["temp_idx_"] = lines.index

    # move the points to the nearest exact point of the line (to_node=False)
    # and get the line index ('id_col')
    snapped = snap_to(
        points,
        lines,
        max_dist=max_dist,
        to_node=False,
        id_col="temp_idx_",
    )

    condition = lines["temp_idx_"].isin(snapped["temp_idx_"])
    relevant_lines = lines.loc[condition]
    the_other_lines = lines.loc[~condition]

    # need consistent coordinate dimensions later
    # (doing it down here to not overwrite the original data)
    relevant_lines.geometry = force_2d(relevant_lines.geometry)
    snapped.geometry = force_2d(snapped.geometry)

    # split the lines with buffer + difference, since shaply.split usually doesn't work
    splitted = relevant_lines.overlay(
        buff(snapped, BUFFDIST), how="difference"
    ).explode(ignore_index=True)

    # the endpoints of the new lines are now sligtly off. To get the exact snapped
    # point coordinates, using get_k_nearest_neighbors. This will map the sligtly
    # off line endpoints with the point the line was split by.

    # columns that will be used as id_cols in get_k_nearest_neighbors
    splitted["splitidx"] = splitted.index
    snapped["point_coords"] = [(geom.x, geom.y) for geom in snapped.geometry]

    # get the endpoints of the lines as columns
    splitted = make_edge_coords_cols(splitted)

    # create geodataframes with the source and target points as geometries
    splitted_source = to_gdf(
        {
            "splitidx": splitted["splitidx"],
            "geometry": splitted["source_coords"],
        },
        crs=lines.crs,
    )
    splitted_target = to_gdf(
        {
            "splitidx": splitted["splitidx"],
            "geometry": splitted["target_coords"],
        },
        crs=lines.crs,
    )

    # find the nearest snapped point for each source and target of the lines
    # low 'max_dist' makes sure we only get either source or target of the split lines
    dists_source = get_k_nearest_neighbors(
        splitted_source,
        snapped,
        k=1,
        max_dist=BUFFDIST * 2,
        id_cols=("splitidx", "point_coords"),
    )
    dists_target = get_k_nearest_neighbors(
        splitted_target,
        snapped,
        k=1,
        max_dist=BUFFDIST * 2,
        id_cols=("splitidx", "point_coords"),
    )

    # use the id columns from k-neighbours to map line id with snapped point
    splitdict_source: pd.Series = dists_source.set_index("splitidx")["point_coords"]
    splitdict_target: pd.Series = dists_target.set_index("splitidx")["point_coords"]

    # now, we can finally replace the source/target coordinate with the coordinates of
    # the snapped points.

    # loop for each line where the source is the endpoint that was split
    # change the first point of the line to the point it was split by
    for idx in dists_source.splitidx:
        line = splitted.loc[idx, "geometry"]
        coordslist = list(line.coords)
        coordslist[0] = splitdict_source[idx]
        splitted.loc[idx, "geometry"] = LineString(coordslist)

    # same for the lines where the target was split, but change the last point of the
    # line
    for idx in dists_target.splitidx:
        line = splitted.loc[idx, "geometry"]
        coordslist = list(line.coords)
        coordslist[-1] = splitdict_target[idx]
        splitted.loc[idx, "geometry"] = LineString(coordslist)

    splitted["splitted"] = 1

    lines = gdf_concat([the_other_lines, splitted]).drop(
        ["temp_idx_", "splitidx", "source_coords", "target_coords"], axis=1
    )

    return lines


def cut_lines(gdf: GeoDataFrame, max_length: int, ignore_index=True) -> GeoDataFrame:
    """Cuts lines of a GeoDataFrame into pieces of a given length.

    Args:
        gdf: GeoDataFrame.
        max_length: The maximum length of the lines in the output GeoDataFrame.
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Defaults to True.

    Returns:
        A GeoDataFrame with lines cut to the maximum distance.

    Note:
        This method is time consuming for large networks and low 'max_length'.

    Examples
    --------
    >>> from sgis import read_parquet_url, cut_lines
    >>> roads = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")
    >>> roads.length.describe().round(1)
    count    93395.0
    mean        41.2
    std         78.5
    min          0.2
    25%         14.0
    50%         27.7
    75%         47.5
    max       5213.7
    dtype: float64
    >>> roads = cut_lines(roads, max_length=100)
    >>> roads.length.describe().round(1)
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
    gdf["geometry"] = force_2d(gdf.geometry)

    gdf = gdf.explode(ignore_index=ignore_index)

    long_lines = gdf.loc[gdf.length > max_length]

    if not len(long_lines):
        return gdf

    def cut(line, distance):
        """From the shapely docs, but added unary_union in the returns."""
        if distance <= 0.0 or distance >= line.length:
            return line
        coords = list(line.coords)
        for i, p in enumerate(coords):
            prd = line.project(Point(p))
            if prd == distance:
                return unary_union(
                    [LineString(coords[: i + 1]), LineString(coords[i:])]
                )
            if prd > distance:
                cp = line.interpolate(distance)
                return unary_union(
                    [
                        LineString(coords[:i] + [(cp.x, cp.y)]),
                        LineString([(cp.x, cp.y)] + coords[i:]),
                    ]
                )

    cut_vectorised = np.vectorize(cut)

    for x in [10, 5, 1]:
        max_ = max(long_lines.length)
        while max_ > max_length * x + 1:
            max_ = max(long_lines.length)

            long_lines["geometry"] = cut_vectorised(long_lines.geometry, max_length)

            long_lines = long_lines.explode(ignore_index=ignore_index)

            if max_ == max(long_lines.length):
                break

    long_lines = long_lines.explode(ignore_index=ignore_index)

    short_lines = gdf.loc[gdf.length <= max_length]

    return pd.concat([short_lines, long_lines], ignore_index=ignore_index)


def close_network_holes(
    lines: GeoDataFrame,
    max_dist: int,
    min_dist: int = 0,
    deadends_only: bool = False,
    hole_col: str | None = "hole",
    k: int = 25,
    length_factor: int = 25,
):
    """Fills gaps shorter than 'max_dist' in a GeoDataFrame of lines.

    Fills holes in the network by finding the nearest neighbors of each node, and
    connecting the nodes that are within a certain distance from each other.

    Args:
        lines: GeoDataFrame with lines
        max_dist: The maximum distance between two nodes to be considered a hole.
        min_dist: minimum distance between nodes to be considered a hole. Defaults to 0
        deadends_only: If True, only lines that connect dead ends will be created. If
            False (the default), deadends might be connected to nodes that are not
            deadends.
        hole_col: If you want to keep track of which lines were added, you can add a
            column with a value of 1. Defaults to 'hole'
        k: number of nearest neighbors to consider. Defaults to 25.
        length_factor: The percentage longer the new lines have to be compared to the
            distance from the other end of the deadend line relative to the line's
            length. Or said (a bit) simpler: higher length_factor means the new lines
            will have an angle more similar to the deadend line it originates from.

    Returns:
        The input GeoDataFrame with new lines added.

    Examples
    --------
    Read road data with small gaps.

    >>> import sgis as sg
    >>> roads = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")

    Roads need to be singlepart linestrings for this to work.

    >>> from shapely import line_merge
    >>> roads.geometry = line_merge(roads)

    Check for number of isolated lines now.

    >>> roads = sg.get_largest_component(roads)
    >>> roads.connected.value_counts()
    1.0    85638
    0.0     7757
    Name: connected, dtype: int64

    Fill gaps shorter than 1.1 meters.

    >>> roads = sg.close_network_holes(roads, max_dist=1.1)
    >>> roads = sg.get_largest_component(roads)
    >>> roads.connected.value_counts()
    1.0    100315
    0.0       180
    Name: connected, dtype: int64

    It's not always wise to fill gaps. In the case of this data, these small gaps are
    intentional. They are road blocks where most cars aren't allowed to pass. Fill the
    holes only if it makes the travel times/routes more realistic.
    """
    lines, nodes = make_node_ids(lines)

    if deadends_only:
        new_lines = _find_holes_deadends(nodes, max_dist, min_dist)
    else:
        new_lines = _find_holes_all_lines(
            lines,
            nodes,
            max_dist,
            min_dist,
            length_factor=length_factor,
            k=k,
        )

    if not len(new_lines):
        return lines

    new_lines = make_edge_wkt_cols(new_lines)

    wkt_id_dict = {
        wkt: id for wkt, id in zip(nodes["wkt"], nodes["node_id"], strict=True)
    }
    new_lines["source"] = new_lines["source_wkt"].map(wkt_id_dict)
    new_lines["target"] = new_lines["target_wkt"].map(wkt_id_dict)

    if hole_col:
        new_lines[hole_col] = 1
        lines[hole_col] = 0

    return gdf_concat([lines, new_lines])


def make_node_ids(
    lines: GeoDataFrame,
    wkt: bool = True,
) -> tuple[GeoDataFrame, GeoDataFrame]:
    """Gives the lines unique node ids and returns lines (edges) and nodes.

    Takes the first and last point of each line and creates a GeoDataFrame of
    nodes (points) with a column 'node_id'. The node ids are then assigned to the
    input GeoDataFrame of lines as the columns 'source' and 'target'.

    Args:
        lines: GeoDataFrame with line geometries
        wkt: If True (the default), the resulting nodes will include the column 'wkt',
            containing the well-known text representation of the geometry. If False, it
            will include the column 'coords', a tuple with x and y geometries.

    Returns:
        A tuple of two GeoDataFrames, one with the lines and one with the nodes.

    Note:
        The lines must be singlepart linestrings.
    """
    if wkt:
        lines = make_edge_wkt_cols(lines)
        geomcol1, geomcol2, geomcol_final = "source_wkt", "target_wkt", "wkt"
    else:
        lines = make_edge_coords_cols(lines)
        geomcol1, geomcol2, geomcol_final = "source_coords", "target_coords", "coords"

    # remove identical lines in opposite directions
    lines["meters_"] = lines.length.astype(str)

    sources = lines[[geomcol1, geomcol2, "meters_"]].rename(
        columns={geomcol1: geomcol_final, geomcol2: "temp"}
    )
    targets = lines[[geomcol1, geomcol2, "meters_"]].rename(
        columns={geomcol2: geomcol_final, geomcol1: "temp"}
    )

    nodes = (
        pd.concat([sources, targets], axis=0, ignore_index=True)
        .drop_duplicates([geomcol_final, "temp", "meters_"])
        .drop("meters_", axis=1)
    )

    lines = lines.drop("meters_", axis=1)

    nodes["n"] = nodes.assign(n=1).groupby(geomcol_final)["n"].transform("sum")

    nodes = nodes.drop_duplicates(subset=[geomcol_final]).reset_index(drop=True)

    nodes["node_id"] = nodes.index
    nodes["node_id"] = nodes["node_id"].astype(str)

    id_dict = {
        geom: node_id
        for geom, node_id in zip(nodes[geomcol_final], nodes["node_id"], strict=True)
    }
    lines["source"] = lines[geomcol1].map(id_dict)
    lines["target"] = lines[geomcol2].map(id_dict)

    n_dict = {geom: n for geom, n in zip(nodes[geomcol_final], nodes["n"], strict=True)}
    lines["n_source"] = lines[geomcol1].map(n_dict)
    lines["n_target"] = lines[geomcol2].map(n_dict)

    if wkt:
        nodes["geometry"] = gpd.GeoSeries.from_wkt(nodes[geomcol_final], crs=lines.crs)
    else:
        nodes["geometry"] = GeoSeries(
            [Point(geom) for geom in nodes[geomcol_final]], crs=lines.crs
        )
    nodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs=lines.crs)
    nodes = nodes.reset_index(drop=True)

    lines = _push_geom_col(lines)

    return lines, nodes


def make_edge_coords_cols(lines: GeoDataFrame) -> GeoDataFrame:
    """Get the wkt of the first and last points of lines as columns.

    It takes a GeoDataFrame of LineStrings and returns a GeoDataFrame with two new
    columns, source_coords and target_coords, which are the x and y coordinates of the
    first and last points of the LineStrings in a tuple. The lines all have to be

    Args:
        lines (GeoDataFrame): the GeoDataFrame with the lines

    Returns:
        A GeoDataFrame with new columns 'source_coords' and 'target_coords'
    """
    lines, endpoints = _prepare_make_edge_cols(lines)

    coords = [(geom.x, geom.y) for geom in endpoints.geometry]
    lines["source_coords"], lines["target_coords"] = (
        coords[0::2],
        coords[1::2],
    )

    return lines


def make_edge_wkt_cols(lines: GeoDataFrame) -> GeoDataFrame:
    """Get coordinate tuples of the first and last points of lines as columns.

    It takes a GeoDataFrame of LineStrings and returns a GeoDataFrame with two new
    columns, source_wkt and target_wkt, which are the WKT representations of the first
    and last points of the LineStrings

    Args:
        lines (GeoDataFrame): the GeoDataFrame with the lines

    Returns:
        A GeoDataFrame with new columns 'source_wkt' and 'target_wkt'
    """
    lines, endpoints = _prepare_make_edge_cols(lines)

    wkt_geom = [
        f"POINT ({x} {y})" for x, y in zip(endpoints.x, endpoints.y, strict=True)
    ]
    lines["source_wkt"], lines["target_wkt"] = (
        wkt_geom[0::2],
        wkt_geom[1::2],
    )

    return lines


def _find_holes_all_lines(
    lines: GeoDataFrame,
    nodes: GeoDataFrame,
    max_dist: int,
    min_dist: int = 0,
    k: int = 25,
    length_factor: int = 25,
):
    """Creates lines between deadends and closest node in forward-going direction.

    Creates a straight line between deadends and the closest node in a
    forward-going direction, if the distance is between the max_dist and min_dist.

    Args:
        lines: the lines you want to find holes in
        nodes: a GeoDataFrame of nodes
        max_dist: The maximum distance between the dead end and the node it should be
            connected to.
        min_dist: The minimum distance between the dead end and the node. Defaults to 0
        k: number of nearest neighbors to consider. Defaults to 25
        length_factor: The percentage longer the new lines have to be compared to the
            distance from the other end of the deadend line relative to the line's
            length. Or said (a bit) simpler: higher length_factor means the new lines
            will have an angle more similar to the deadend line it originates from.

    Returns:
        A GeoDataFrame with the shortest line between the two points.
    """
    # wkt: well-known text, e.g. "POINT (60 10)"

    crs = nodes.crs

    # remove duplicates of lines going both directions
    lines["sorted"] = [
        "_".join(sorted([s, t]))
        for s, t in zip(lines["source"], lines["target"], strict=True)
    ]

    no_dups = lines.drop_duplicates("sorted")

    # make new node ids without bidirectional lines
    no_dups, nodes = make_node_ids(no_dups)

    # deadends are the target endpoints of the lines appearing once
    deadends_target = no_dups.loc[no_dups.n_target == 1].rename(
        columns={"target_wkt": "wkt", "source_wkt": "wkt_other_end"}
    )
    deadends_source = no_dups.loc[no_dups.n_source == 1].rename(
        columns={"source_wkt": "wkt", "target_wkt": "wkt_other_end"}
    )

    deadends = pd.concat([deadends_source, deadends_target], ignore_index=True)

    if len(deadends) <= 1:
        return []

    deadends_lengths = deadends.length

    deadends_other_end = deadends.copy()
    deadends_other_end["geometry"] = gpd.GeoSeries.from_wkt(
        deadends_other_end["wkt_other_end"], crs=crs
    )

    deadends["geometry"] = gpd.GeoSeries.from_wkt(deadends["wkt"], crs=crs)

    deadends_array = coordinate_array(deadends)

    nodes_array = coordinate_array(nodes)

    all_dists, all_indices = k_nearest_neighbors(deadends_array, nodes_array, k=k)

    # now to find the lines that go in the right direction. Collecting the startpoints
    # and endpoints of the new lines in lists, looping through the k neighbour points
    new_sources: list[str] = []
    new_targets: list[str] = []
    for i in np.arange(1, k):
        # to break out of the loop if no new_targets that meet the condition are found
        len_now = len(new_sources)

        # selecting the arrays for the k neighbour
        indices = all_indices[:, i]
        dists = all_dists[:, i]

        # get the distance from the other end of the deadends to the k neighbours
        dists_source = deadends_other_end.distance(nodes.loc[indices], align=False)

        # select the distances between min_dist and max_dist that are also shorter than
        # the distance from the other side of the line, meaning the new line will go
        # forward. All arrays have the same shape, and can be easily filtered by index
        # The node wkts have to be extracted after indexing 'indices'
        condition = (
            (dists < max_dist)
            & (dists > min_dist)
            & (dists < dists_source - deadends_lengths * length_factor / 100)
        )
        from_wkt = deadends.loc[condition, "wkt"]
        to_idx = indices[condition]
        to_wkt = nodes.loc[to_idx, "wkt"]

        # now add the wkts to the lists of new sources and targets. If the source
        # is already added, the new wks will not be added again
        new_targets = new_targets + [
            t for f, t in zip(from_wkt, to_wkt, strict=True) if f not in new_sources
        ]
        new_sources = new_sources + [
            f for f, _ in zip(from_wkt, to_wkt, strict=True) if f not in new_sources
        ]

        # break out of the loop when no new new_targets meet the condition
        if len_now == len(new_sources):
            break

    # make GeoDataFrame with straight lines
    new_sources = gpd.GeoSeries.from_wkt(new_sources, crs=crs)
    new_targets = gpd.GeoSeries.from_wkt(new_targets, crs=crs)
    new_lines = shortest_line(new_sources, new_targets)
    new_lines = gpd.GeoDataFrame({"geometry": new_lines}, geometry="geometry", crs=crs)

    if not len(new_lines):
        return new_lines

    new_lines = make_edge_wkt_cols(new_lines)

    return new_lines


def _find_holes_deadends(
    nodes: GeoDataFrame, max_dist: float | int, min_dist: float | int = 0
):
    """Creates lines between two deadends if between max_dist and min_dist.

    It takes a GeoDataFrame of nodes, chooses the deadends, and creates a straight line
    between the closest deadends if the distance is between the specifies max_dist and
    min_dist.

    Args:
        nodes: the nodes of the network
        max_dist: The maximum distance between two nodes to be connected.
        min_dist: minimum distance between nodes to be considered a hole. Defaults to 0

    Returns:
        A GeoDataFrame with the new lines.
    """
    crs = nodes.crs

    # deadends are nodes that appear only once
    deadends = nodes[nodes["n"] == 1]

    # have to reset index to be able to integrate with numpy/scikit-learn
    deadends = deadends.reset_index(drop=True)

    if len(deadends) <= 1:
        return []

    deadends_array = coordinate_array(deadends)

    dists, indices = k_nearest_neighbors(deadends_array, deadends_array, k=2)

    # choose the second column (the closest neighbour)
    dists = dists[:, 1]
    indices = indices[:, 1]

    # get the geometry of the distances between min_dist and max_dist
    # the from geometries can be taken directly from the deadends index,
    # since 'dists' has the same index. 'to_geom' must be selected through the index
    # of the neighbours ('indices')
    condition = (dists < max_dist) & (dists > min_dist)
    from_geom = deadends.loc[condition, "geometry"].reset_index(drop=True)
    to_idx = indices[condition]
    to_geom = deadends.loc[to_idx, "geometry"].reset_index(drop=True)

    # GeoDataFrame with straight lines
    new_lines = shortest_line(from_geom, to_geom)
    new_lines = gpd.GeoDataFrame({"geometry": new_lines}, geometry="geometry", crs=crs)

    if not len(new_lines):
        return new_lines

    return new_lines


def _prepare_make_edge_cols(
    lines: GeoDataFrame,
) -> tuple[GeoDataFrame, GeoDataFrame]:
    lines = lines.loc[lines.geom_type != "LinearRing"]

    if not all(lines.geom_type == "LineString"):
        multilinestring_error_message = (
            "MultiLineStrings have more than two endpoints. "
            "Try shapely.line_merge and/or explode() to get LineStrings. "
            "Or use the Network class methods, where the lines are prepared correctly."
        )
        if any(lines.geom_type == "MultiLinestring"):
            raise ValueError(multilinestring_error_message)
        else:
            raise ValueError(
                "You have mixed geometries. Only lines are accepted. "
                "Try using: to_single_geom_type(gdf, 'lines')."
            )

    # some LinearRings are coded as LineStrings and need to be removed manually
    boundary = lines.geometry.boundary
    circles = boundary.loc[boundary.is_empty]
    lines = lines[~lines.index.isin(circles.index)]

    endpoints = lines.geometry.boundary.explode(ignore_index=True)

    if len(endpoints) / len(lines) != 2:
        raise ValueError(
            "The lines should have only two endpoints each. "
            "Try splitting multilinestrings with explode."
        )

    return lines, endpoints


def _roundabouts_to_intersections(roads, query="ROADTYPE=='Rundkjøring'"):
    from shapely.geometry import LineString
    from shapely.ops import nearest_points

    from .buffer_dissolve_explode import buffdissexp

    roundabouts = roads.query(query)
    not_roundabouts = roads.loc[~roads.index.isin(roundabouts.index)]

    roundabouts = buffdissexp(roundabouts[["geometry"]], 1)
    roundabouts["roundidx"] = roundabouts.index

    border_to = roundabouts.overlay(not_roundabouts, how="intersection")

    # for hver rundkjøring: lag linjer mellom rundkjøringens mitdpunkt og hver linje
    # som grenser til rundkjøringen
    as_intersections = []
    for idx in roundabouts.roundidx:
        this = roundabouts.loc[roundabouts.roundidx == idx]
        border_to_this = border_to.loc[border_to.roundidx == idx].drop(
            "roundidx", axis=1
        )

        midpoint = this.unary_union.centroid

        # straight lines to the midpoint
        for i, line in enumerate(border_to_this.geometry):
            closest_point = nearest_points(midpoint, line)[1]
            border_to_this.geometry.iloc[i] = LineString([closest_point, midpoint])

        as_intersections.append(border_to_this)

    as_intersections = gdf_concat(as_intersections, crs=roads.crs)
    out = gdf_concat([not_roundabouts, as_intersections], crs=roads.crs)

    return out

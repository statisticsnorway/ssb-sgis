"""Functions for line geometries.

The module includes functions for cutting and splitting lines, cutting lines into
pieces, filling holes in a network of lines, finding isolated network islands and
creating unique node ids.

The functions are also methods of the Network class, where some checks and
preperation is done before each method is run, making sure the lines are correct.
"""
import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame, Series
from shapely import force_2d, shortest_line
from shapely.geometry import LineString, Point

from .buffer_dissolve_explode import buff
from .general import _push_geom_col, coordinate_array, to_gdf
from .neighbors import get_k_nearest_neighbors, k_nearest_neighbors
from .point_operations import snap_all, snap_within_distance


def get_largest_component(lines: GeoDataFrame) -> GeoDataFrame:
    """Finds the largest network component.

    It takes a GeoDataFrame of lines and finds the lines that are
    part of the largest connected network component. These lines are given the value
    1 in the added column 'connected', while isolated network islands get the value
    0.

    Uses the connected_components function from the networkx package.

    Args:
        lines: A GeoDataFrame of lines.

    Returns:
        The GeoDataFrame with a new column "connected".

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

    Takes a GeoDataFrame of linea and creates the column "component_size", which
    indicates the size of the network component the line is a part of.

    Args:
        lines: a GeoDataFrame of lines.

    Returns:
        A GeoDataFrame with a new column "component_size".

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


def split_lines_by_nearest_point(
    lines: GeoDataFrame,
    points: GeoDataFrame,
    max_distance: int | None = None,
) -> DataFrame:
    """Split lines that are closest to s point.

    Snaps points to nearest lines and splits the lines in two at the snap point.
    The splitting is done pointwise, meaning each point splits one line in two.
    The line will not be split if the point is closest to the endpoint of the line.

    This function is used in NetworkAnalysis if split_lines is set to True.

    Args:
        lines: GeoDataFrame of lines that will be split.
        points: GeoDataFrame of points to split the lines with.
        max_distance: the maximum distance between the point and the line. Points further
            away than max_distance will not split any lines. Defaults to None.

    Returns:
        A GeoDataFrame with the same columns as the input lines, but with the lines
        split at the closest point to the points.

    Raises:
        ValueError: If the crs of the input data differs.

    Examples
    --------
    >>> from sgis import read_parquet_url, split_lines_by_nearest_point
    >>> roads = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")
    >>> points = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet")
    >>> rows = len(roads)
    >>> rows
    93395

    Splitting lines for points closer than 10 meters from the lines.

    >>> roads = split_lines_by_nearest_point(roads, points, max_distance=10)
    >>> print("number of lines that were split:", len(roads) - rows)
    number of lines that were split: 380

    Splitting lines by all points.

    >>> roads = split_lines_by_nearest_point(roads, points)
    >>> print("number of lines that were split:", len(roads) - rows)
    number of lines that were split: 848

    Not all lines were split. That is because some points were closest to an endpoint
    of a line.
    """
    BUFFDIST = 0.000001

    if points.crs != lines.crs:
        raise ValueError("crs mismatch:", points.crs, "and", lines.crs)

    lines["temp_idx_"] = lines.index

    # move the points to the nearest exact point of the line
    if max_distance:
        snapped = snap_within_distance(points, lines, max_distance=max_distance)
    else:
        snapped = snap_all(points, lines)

    # find the lines that were snapped to (or are very close)
    snapped_buff = buff(snapped, BUFFDIST)
    intersect = lines.intersects(snapped_buff.unary_union)
    relevant_lines = lines.loc[intersect]
    the_other_lines = lines.loc[~intersect]

    # need consistent coordinate dimensions later
    # (doing it down here to not overwrite the original data)
    relevant_lines.geometry = force_2d(relevant_lines.geometry)
    snapped.geometry = force_2d(snapped.geometry)

    # split the lines with buffer + difference, since shaply.split usually doesn't work
    splitted = relevant_lines.overlay(snapped_buff, how="difference").explode(
        ignore_index=True
    )

    # the endpoints of the new lines are now sligtly off. To get the exact snapped
    # point coordinates, using get_k_nearest_neighbors. This will map the sligtly
    # off line endpoints with the point the line was split by.

    snapped["point_coords"] = [(geom.x, geom.y) for geom in snapped.geometry]

    # get the endpoints of the lines as columns
    splitted = make_edge_coords_cols(splitted)

    splitted_source = to_gdf(splitted["source_coords"], crs=lines.crs)
    splitted_target = to_gdf(splitted["target_coords"], crs=lines.crs)

    # find the nearest snapped point for each source and target of the lines
    snapped = snapped.set_index("point_coords")
    dists_source = get_k_nearest_neighbors(splitted_source, snapped, k=1)
    dists_target = get_k_nearest_neighbors(splitted_target, snapped, k=1)

    dists_source = dists_source.loc[dists_source.distance <= BUFFDIST * 2]
    dists_target = dists_target.loc[dists_target.distance <= BUFFDIST * 2]

    pointmapper_source: pd.Series = dists_source["neighbor_index"]
    pointmapper_target: pd.Series = dists_target["neighbor_index"]

    # now, we can finally replace the source/target coordinate with the coordinates of
    # the snapped points.

    # loop for each line where the source is the endpoint that was split
    # change the first point of the line to the point it was split by
    for idx in dists_source.index:
        line = splitted.loc[idx, "geometry"]
        coordslist = list(line.coords)
        coordslist[0] = pointmapper_source[idx]
        splitted.loc[idx, "geometry"] = LineString(coordslist)

    # same for the lines where the target was split, but change the last point of the
    # line
    for idx in dists_target.index:
        line = splitted.loc[idx, "geometry"]
        coordslist = list(line.coords)
        coordslist[-1] = pointmapper_target[idx]
        splitted.loc[idx, "geometry"] = LineString(coordslist)

    splitted["splitted"] = 1

    lines = pd.concat([the_other_lines, splitted], ignore_index=True).drop(
        ["temp_idx_", "source_coords", "target_coords"], axis=1
    )

    return lines


def cut_lines(gdf: GeoDataFrame, max_length: int, ignore_index=False) -> GeoDataFrame:
    """Cuts lines of a GeoDataFrame into pieces of a given length.

    Args:
        gdf: GeoDataFrame.
        max_length: The maximum length of the lines in the output GeoDataFrame.
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Defaults to False.

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

    gdf = gdf.explode(ignore_index=ignore_index, index_parts=False)

    long_lines = gdf.loc[gdf.length > max_length]

    if not len(long_lines):
        return gdf

    for x in [10, 5, 1]:
        max_ = max(long_lines.length)
        while max_ > max_length * x + 1:
            max_ = max(long_lines.length)

            long_lines = cut_lines_once(long_lines, max_length)

            if max_ == max(long_lines.length):
                break

    long_lines = long_lines.explode(ignore_index=ignore_index, index_parts=False)

    short_lines = gdf.loc[gdf.length <= max_length]

    return pd.concat([short_lines, long_lines], ignore_index=ignore_index)


def cut_lines_once(
    lines: GeoDataFrame,
    distances: int | float | str | Series,
    ignore_index: bool = False,
) -> GeoDataFrame:
    """Cuts lines of a GeoDataFrame in two at the given distance or distances.

    Takes a GeoDataFrame of lines and cuts each line in two. If distances is a number,
    all lines will be cut at the same length.

    Args:
        gdf: GeoDataFrame.
        distances: The distance from the start of the lines to cut at. Either a number,
            the name of a column or array-like of same length as the line GeoDataFrame.
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Defaults to False.

    Examples
    --------
    >>> from sgis import cut_lines_once, to_gdf
    >>> import pandas as pd
    >>> from shapely.geometry import LineString
    >>> gdf = to_gdf(LineString([(0, 0), (1, 1), (2, 2)]))
    >>> gdf = pd.concat([gdf, gdf, gdf])
    >>> gdf
                                                geometry
    0  LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...
    0  LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...
    0  LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...

    Cut all lines at 1 unit from the start of the lines.

    >>> cut_lines_once(gdf, 1)
                                                geometry
    0      LINESTRING (0.00000 0.00000, 0.70711 0.70711)
    1  LINESTRING (0.70711 0.70711, 1.00000 1.00000, ...
    2      LINESTRING (0.00000 0.00000, 0.70711 0.70711)
    3  LINESTRING (0.70711 0.70711, 1.00000 1.00000, ...
    4      LINESTRING (0.00000 0.00000, 0.70711 0.70711)
    5  LINESTRING (0.70711 0.70711, 1.00000 1.00000, ...

    Cut distance as column.

    >>> gdf["dist"] = [1, 2, 3]
    >>> cut_lines_once(gdf, "dist")
                                                geometry  dist
    0      LINESTRING (0.00000 0.00000, 0.70711 0.70711)     1
    1  LINESTRING (0.70711 0.70711, 1.00000 1.00000, ...     1
    2  LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...     2
    3      LINESTRING (1.41421 1.41421, 2.00000 2.00000)     2
    4  LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...     3

    Cut distance as list (same result as above).

    >>> cut_lines_once(gdf, [1, 2, 3])
                                                geometry  dist
    0      LINESTRING (0.00000 0.00000, 0.70711 0.70711)     1
    1  LINESTRING (0.70711 0.70711, 1.00000 1.00000, ...     1
    2  LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...     2
    3      LINESTRING (1.41421 1.41421, 2.00000 2.00000)     2
    4  LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...     3
    """

    def _cut(line: LineString, distance: int | float) -> list[LineString]:
        """From the shapely docs"""
        if distance <= 0.0 or distance >= line.length:
            return line
        coords = list(line.coords)
        for i, p in enumerate(coords):
            prd = line.project(Point(p))
            if prd == distance:
                return [LineString(coords[: i + 1]), LineString(coords[i:])]
            if prd > distance:
                cp = line.interpolate(distance)
                return [
                    LineString(coords[:i] + [(cp.x, cp.y)]),
                    LineString([(cp.x, cp.y)] + coords[i:]),
                ]

    crs = lines.crs
    geom_col = lines._geometry_column_name

    lines = lines.copy()

    # cutting lines will give lists of linestrings in the geometry column. Ignoring
    # the warning it triggers
    warnings.filterwarnings(
        "ignore", message="Geometry column does not contain geometry."
    )

    if isinstance(distances, str):
        lines[geom_col] = np.vectorize(_cut)(lines[geom_col], lines[distances])
    else:
        lines[geom_col] = np.vectorize(_cut)(lines[geom_col], distances)

    # explode will give pandas df if not gdf is constructed
    lines = GeoDataFrame(
        lines.explode(ignore_index=ignore_index, index_parts=False),
        geometry=geom_col,
        crs=crs,
    )

    return lines


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
        wkt: If True (default), the resulting nodes will include the column 'wkt',
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
    try:
        lines, endpoints = _prepare_make_edge_cols_simple(lines)
    except ValueError:
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
    try:
        lines, endpoints = _prepare_make_edge_cols_simple(lines)
    except ValueError:
        lines, endpoints = _prepare_make_edge_cols(lines)

    wkt_geom = [
        f"POINT ({x} {y})" for x, y in zip(endpoints.x, endpoints.y, strict=True)
    ]
    lines["source_wkt"], lines["target_wkt"] = (
        wkt_geom[0::2],
        wkt_geom[1::2],
    )

    return lines


def close_network_holes(
    lines: GeoDataFrame,
    max_distance: int | float,
    max_angle: int,
    hole_col: str | None = "hole",
):
    """Fills network gaps with straigt lines.

    Fills holes in the network by connecting deadends with the nodes that are
    within the 'max_distance' distance.

    Args:
        lines: GeoDataFrame with lines.
        max_distance: The maximum distance for the holes to be filled.
        max_angle: Absolute number between 0 and 180 that represents the maximum
            difference in angle between the new line and the prior, i.e. the line
            at which the deadend terminates. A value of 0 means the new lines must
            have the exact same angle as the prior line, and 180 means the new
            lines can go in any direction.
        hole_col: If you want to keep track of which lines were added, you can add a
            column with a value of 1. Defaults to 'hole'

    Returns:
        The input GeoDataFrame with new lines added.

    Examples
    --------
    Read road data with small gaps.

    >>> import sgis as sg
    >>> roads = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")

    Roads need to be singlepart linestrings for this to work.

    >>> from shapely import line_merge
    >>> roads.geometry = line_merge(roads.geometry)

    Fill gaps shorter than 1.1 meters.

    >>> filled = sg.close_network_holes(roads, max_distance=1.1, max_angle=180)
    >>> filled.hole.value_counts()
    Name: connected, dtype: int64
    0    93395
    1     7102
    Name: hole, dtype: int64

    Compare the number of isolated lines before and after.

    >>> roads = sg.get_largest_component(roads)
    >>> roads.connected.value_counts()
    1.0    85638
    0.0     7757
    Name: connected, dtype: int64

    >>> filled = sg.get_largest_component(filled)
    >>> filled.connected.value_counts()
    1.0    100315
    0.0       180
    Name: connected, dtype: int64

    Fill only gaps with an angle deviation between 0 and 30 compared to the prior line.

    >>> filled = sg.close_network_holes(roads, max_distance=1.1, max_angle=30)
    >>> filled.hole.value_counts()
    0    93395
    1     7092
    Name: hole, dtype: int64

    It's not always wise to fill gaps. In the case of this data, these small gaps are
    intentional. They are road blocks where most cars aren't allowed to pass. Fill the
    holes only if it makes the travel times/routes more realistic.
    """
    lines, nodes = make_node_ids(lines)

    new_lines = _find_holes_all_lines(
        lines,
        nodes,
        max_distance,
        max_angle=max_angle,
    )

    if not len(new_lines):
        lines[hole_col] = (
            0 if hole_col not in lines.columns else lines[hole_col].fillna(0)
        )
        return lines

    new_lines = make_edge_wkt_cols(new_lines)

    wkt_id_dict = {
        wkt: id for wkt, id in zip(nodes["wkt"], nodes["node_id"], strict=True)
    }
    new_lines["source"] = new_lines["source_wkt"].map(wkt_id_dict)
    new_lines["target"] = new_lines["target_wkt"].map(wkt_id_dict)

    if hole_col:
        new_lines[hole_col] = 1
        lines[hole_col] = (
            0 if hole_col not in lines.columns else lines[hole_col].fillna(0)
        )

    return pd.concat([lines, new_lines], ignore_index=True)


def close_network_holes_to_deadends(
    lines: GeoDataFrame,
    max_distance: int | float,
    hole_col: str | None = "hole",
):
    """Fills gaps between two deadends if the distance is less than 'max_distance'.

    Fills holes between deadends in the network with straight lines if the distance is
    less than 'max_distance'.

    Args:
        lines: GeoDataFrame with lines
        max_distance: The maximum distance between two nodes to be considered a hole.
        hole_col: If you want to keep track of which lines were added, you can add a
            column with a value of 1. Defaults to 'hole'

    Returns:
        The input GeoDataFrame with new lines added.

    Examples
    --------
    Read road data with small gaps.

    >>> import sgis as sg
    >>> roads = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")

    Roads need to be singlepart linestrings for this to work.

    >>> from shapely import line_merge
    >>> roads.geometry = line_merge(roads.geometry)

    Check for number of isolated lines now.

    >>> roads = sg.get_largest_component(roads)
    >>> roads.connected.value_counts()
    1.0    85638
    0.0     7757
    Name: connected, dtype: int64

    Fill gaps shorter than 1.1 meters.

    >>> filled = sg.close_network_holes_to_deadends(roads, max_distance=1.1)
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

    new_lines = _find_holes_deadends(nodes, max_distance)

    if not len(new_lines):
        lines[hole_col] = (
            0 if hole_col not in lines.columns else lines[hole_col].fillna(0)
        )
        return lines

    new_lines = make_edge_wkt_cols(new_lines)

    wkt_id_dict = {
        wkt: id for wkt, id in zip(nodes["wkt"], nodes["node_id"], strict=True)
    }
    new_lines["source"] = new_lines["source_wkt"].map(wkt_id_dict)
    new_lines["target"] = new_lines["target_wkt"].map(wkt_id_dict)

    if hole_col:
        new_lines[hole_col] = 1
        lines[hole_col] = (
            0 if hole_col not in lines.columns else lines[hole_col].fillna(0)
        )

    return pd.concat([lines, new_lines], ignore_index=True)


def _find_holes_all_lines(
    lines: GeoDataFrame,
    nodes: GeoDataFrame,
    max_distance: int,
    max_angle: int = 90,
):
    """Creates lines between deadends and closest node.

    Creates lines if distance is less than max_distance and angle less than max_angle.

    wkt: well-known text, e.g. "POINT (60 10)"
    """
    k = 50 if len(nodes) >= 50 else len(nodes)
    crs = nodes.crs

    # remove duplicates of lines going both directions
    lines["sorted"] = [
        "_".join(sorted([s, t]))
        for s, t in zip(lines["source"], lines["target"], strict=True)
    ]

    no_dups = lines.drop_duplicates("sorted")

    no_dups, nodes = make_node_ids(no_dups)

    # make point gdf for the deadends and the other endpoint of the deadend lines
    deadends_target = no_dups.loc[no_dups.n_target == 1].rename(
        columns={"target_wkt": "wkt", "source_wkt": "wkt_other_end"}
    )
    deadends_source = no_dups.loc[no_dups.n_source == 1].rename(
        columns={"source_wkt": "wkt", "target_wkt": "wkt_other_end"}
    )
    deadends = pd.concat([deadends_source, deadends_target], ignore_index=True)

    if len(deadends) <= 1:
        return []

    deadends_other_end = deadends.copy()
    deadends_other_end["geometry"] = gpd.GeoSeries.from_wkt(
        deadends_other_end["wkt_other_end"], crs=crs
    )

    deadends["geometry"] = gpd.GeoSeries.from_wkt(deadends["wkt"], crs=crs)

    deadends_array = coordinate_array(deadends)
    nodes_array = coordinate_array(nodes)

    all_dists, all_indices = k_nearest_neighbors(deadends_array, nodes_array, k=k)

    deadends_other_end_array = coordinate_array(deadends_other_end)

    def get_angle(array_a, array_b):
        dx = array_b[:, 0] - array_a[:, 0]
        dy = array_b[:, 1] - array_a[:, 1]

        angles_rad = np.arctan2(dx, dy)
        angles_degrees = np.degrees(angles_rad)
        return angles_degrees

    # now to find the lines that have the correct angle and distance
    # and endpoints of the new lines in lists, looping through the k neighbour points
    new_sources: list[str] = []
    new_targets: list[str] = []
    for i in np.arange(1, k):
        # to break out of the loop if no new_targets that meet the condition are found
        len_now = len(new_sources)

        # selecting the arrays for the current k neighbour
        indices = all_indices[:, i]
        dists = all_dists[:, i]

        these_nodes_array = coordinate_array(nodes.loc[indices])

        if np.all(deadends_other_end_array == these_nodes_array):
            continue

        angles_deadend_to_node = get_angle(deadends_array, these_nodes_array)

        angles_deadend_to_deadend_other_end = get_angle(
            deadends_other_end_array, deadends_array
        )

        angles_difference = np.abs(
            np.abs(angles_deadend_to_deadend_other_end) - np.abs(angles_deadend_to_node)
        )

        angles_difference[
            np.all(deadends_other_end_array == these_nodes_array, axis=1)
        ] = np.nan

        condition = (dists <= max_distance) & (angles_difference <= max_angle)

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


def _find_holes_deadends(nodes: GeoDataFrame, max_distance: float | int):
    """Creates lines between two deadends if between max_distance and min_dist.

    It takes a GeoDataFrame of nodes, chooses the deadends, and creates a straight line
    between the closest deadends if the distance is no greater than 'max_distance'.

    Args:
        nodes: the nodes of the network
        max_distance: The maximum distance between two nodes to be connected.

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

    # get the geometry of the distances no greater than max_distance
    # the from geometries can be taken directly from the deadends index,
    # since 'dists' has the same index. 'to_geom' must be selected through the index
    # of the neighbours ('indices')
    condition = dists < max_distance
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

    geom_col = lines._geometry_column_name

    # some LinearRings are coded as LineStrings and need to be removed manually
    boundary = lines[geom_col].boundary
    circles = boundary.loc[boundary.is_empty]
    lines = lines[~lines.index.isin(circles.index)]

    endpoints = lines[geom_col].boundary.explode(ignore_index=True)

    if len(endpoints) / len(lines) != 2:
        raise ValueError(
            "The lines should have only two endpoints each. "
            "Try splitting multilinestrings with explode."
        )

    return lines, endpoints


def _prepare_make_edge_cols_simple(
    lines: GeoDataFrame,
) -> tuple[GeoDataFrame, GeoDataFrame]:
    """Faster version of _prepare_make_edge_cols."""

    endpoints = lines[lines._geometry_column_name].boundary.explode(ignore_index=True)

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

    as_intersections = GeoDataFrame(
        pd.concat(as_intersections, ignore_index=True), crs=roads.crs
    )
    out = GeoDataFrame(
        pd.concat([not_roundabouts, as_intersections], ignore_index=True), crs=roads.crs
    )

    return out

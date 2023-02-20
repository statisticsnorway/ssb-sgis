from typing import Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from shapely import force_2d, shortest_line
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from sklearn.neighbors import NearestNeighbors

from .geopandas_utils import gdf_concat, push_geom_col


def make_node_ids(
    lines: GeoDataFrame, ignore_index: bool = False
) -> Tuple[GeoDataFrame, GeoDataFrame]:
    """
    Create an integer index for the unique endpoints (nodes) of the lines (edges),
    then map this index to the 'source' and 'target' columns of the 'lines' GeoDataFrame.
    Returns both the lines and the nodes.

    Args:
      lines (GeoDataFrame): GeoDataFrame with line geometries
      ignore_index (bool): If True, the index of the lines GeoDataFrame will be ignored.
        Defaults to False.

    Returns:
      A tuple of two GeoDataFrames, one with the lines and one with the nodes.
    """

    lines = _make_edge_wkt_cols(lines, ignore_index)

    sources = lines[["source_wkt"]].rename(columns={"source_wkt": "wkt"})
    targets = lines[["target_wkt"]].rename(columns={"target_wkt": "wkt"})

    nodes = pd.concat([sources, targets], axis=0, ignore_index=True)

    nodes["n"] = nodes.assign(n=1).groupby("wkt")["n"].transform("sum")

    nodes = nodes.drop_duplicates(subset=["wkt"]).reset_index(drop=True)

    nodes["node_id"] = nodes.index
    nodes["node_id"] = nodes["node_id"].astype(str)

    id_dict = {wkt: node_id for wkt, node_id in zip(nodes["wkt"], nodes["node_id"])}
    lines["source"] = lines["source_wkt"].map(id_dict)
    lines["target"] = lines["target_wkt"].map(id_dict)

    n_dict = {wkt: n for wkt, n in zip(nodes["wkt"], nodes["n"])}
    lines["n_source"] = lines["source_wkt"].map(n_dict)
    lines["n_target"] = lines["target_wkt"].map(n_dict)

    nodes["geometry"] = gpd.GeoSeries.from_wkt(nodes.wkt, crs=lines.crs)
    nodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs=lines.crs)
    nodes = nodes.reset_index(drop=True)

    lines = push_geom_col(lines)

    return lines, nodes


def _make_edge_wkt_cols(lines: GeoDataFrame, ignore_index: bool = True) -> GeoDataFrame:
    """
    It takes a GeoDataFrame of LineStrings and returns a GeoDataFrame with two new
    columns, source_wkt and target_wkt, which are the WKT representations of the first
    and last points of the LineStrings

    Args:
      lines (GeoDataFrame): the GeoDataFrame with the lines
      ignore_index (bool): True by default to avoid futurewarning. But will change to
        False to be consistent with pandas. Defaults to True.

    Returns:
      A GeoDataFrame with the columns 'source_wkt' and 'target_wkt'
    """

    lines = lines.loc[lines.geom_type != "LinearRing"]

    if not all(lines.geom_type == "LineString"):
        if all(lines.geom_type.isin(["LineString", "MultiLinestring"])):
            raise ValueError(
                "MultiLineStrings have more than two endpoints. "
                "Try explode() to get LineStrings."
            )
        else:
            raise ValueError(
                "You have mixed geometry types. Only singlepart LineStrings are "
                "allowed in _make_edge_wkt_cols."
            )

    boundary = lines.geometry.boundary
    circles = boundary.loc[boundary.is_empty]
    lines = lines[~lines.index.isin(circles.index)]

    endpoints = lines.geometry.boundary.explode(
        ignore_index=ignore_index, index_parts=False
    )  # to silence warning

    if len(endpoints) / len(lines) != 2:
        raise ValueError(
            "The lines should have only two endpoints each. "
            "Try splitting multilinestrings with explode."
        )

    wkt_geom = [f"POINT ({x} {y})" for x, y in zip(endpoints.x, endpoints.y)]
    lines["source_wkt"], lines["target_wkt"] = (
        wkt_geom[0::2],
        wkt_geom[1::2],
    )

    return lines


def get_largest_component(lines: GeoDataFrame) -> GeoDataFrame:
    """
    We create a graph from the lines, find the largest connected component,
    and assign this to the value 1 in a new column 'connected'

    Args:
      lines (GeoDataFrame): GeoDataFrame

    Returns:
      A GeoDataFrame with a new boolean column "connected"
    """

    if "source" not in lines.columns or "target" not in lines.columns:
        lines, nodes = make_node_ids(lines)

    edges = [
        (str(source), str(target))
        for source, target in zip(lines["source"], lines["target"])
    ]

    G = nx.Graph()
    G.add_edges_from(edges)

    largest_component = max(nx.connected_components(G), key=len)

    largest_component_dict = {node_id: 1 for node_id in largest_component}

    lines["connected"] = lines.source.map(largest_component_dict).fillna(0)

    return lines


def get_component_size(lines: GeoDataFrame) -> GeoDataFrame:
    """
    It takes a GeoDataFrame of lines, and returns a GeoDataFrame of lines with a new column called
    "component_size" that indicates the size of the component that each line is a part of

    Args:
      lines (GeoDataFrame): GeoDataFrame

    Returns:
      A GeoDataFrame with the size of the component that each line is in.
    """

    if not "source" in lines.columns or not "target" in lines.columns:
        lines, nodes = make_node_ids(lines)

    edges = [
        (str(source), str(target))
        for source, target in zip(lines["source"], lines["target"])
    ]

    G = nx.Graph()
    G.add_edges_from(edges)
    components = [list(x) for x in nx.connected_components(G)]

    componentsdict = {
        idx: len(component) for component in components for idx in component
    }

    lines["component_size"] = lines.source.map(componentsdict)

    return lines


def close_network_holes(
    lines: GeoDataFrame,
    max_dist: int,
    min_dist: int = 0,
    deadends_only: bool = False,
    hole_col: str | None = "hole",
):
    """
    It fills holes in the network by finding the nearest neighbors of each node, and then connecting the
    nodes that are within a certain distance of each other

    Args:
      lines (GeoDataFrame): GeoDataFrame with lines
      max_dist (int): The maximum distance between two nodes to be considered a hole.
      min_dist (int): minimum distance between nodes to be considered a hole. Defaults to 0
      deadends_only (bool): If True, only lines that connect dead ends will be created. If False (the default),
      deadends might be connected to nodes that are not deadends.
      hole_col (str | None): If you want to keep track of which lines were added, you can add a column
        with a value of 1. Defaults to 'hole'

    Returns:
      A GeoDataFrame with the same columns as the input, but with new lines added.
    """

    lines, nodes = make_node_ids(lines)

    if deadends_only:
        new_lines = find_holes_deadends(nodes, max_dist, min_dist)
    else:
        new_lines = find_holes_all_lines(lines, nodes, max_dist, min_dist)

    if not len(new_lines):
        return lines

    wkt_id_dict = {wkt: id for wkt, id in zip(nodes["wkt"], nodes["node_id"])}
    new_lines["source"] = new_lines["source_wkt"].map(wkt_id_dict)
    new_lines["target"] = new_lines["target_wkt"].map(wkt_id_dict)

    if any(new_lines.source.isna()) or any(new_lines.target.isna()):
        raise ValueError("Missing source/target ids.")

    if hole_col:
        new_lines[hole_col] = 1

    return gdf_concat([lines, new_lines])


def find_holes_all_lines(lines, nodes, max_dist, min_dist=0, k=10):
    """
    creates a straight line between deadends and the closest node in a
    forward-going direction, if the distance is between the max_dist and min_dist.

    Args:
      lines: the lines you want to find holes in
      nodes: a GeoDataFrame of nodes
      max_dist: The maximum distance between the dead end and the node it should be connected to.
      min_dist: The minimum distance between the dead end and the node. Defaults to 0
      k: number of nearest neighbors to consider. Defaults to 10

    Returns:
      A GeoDataFrame with the shortest line between the two points.
    """
    crs = nodes.crs

    # velger ut nodene som kun finnes i én lenke. Altså blindveier i en networksanalyse.
    deadends_source = lines.loc[lines.n_source == 1].rename(
        columns={"source_wkt": "wkt", "target_wkt": "wkt_other_end"}
    )
    deadends_target = lines.loc[lines.n_target == 1].rename(
        columns={"source_wkt": "wkt_other_end", "target_wkt": "wkt"}
    )

    deadends = pd.concat([deadends_source, deadends_target], ignore_index=True)

    if len(deadends) <= 1:
        deadends["minutes"] = -1
        return deadends

    deadends_lengths = deadends.length

    deadends_other_end = deadends.copy()

    deadends["geometry"] = gpd.GeoSeries.from_wkt(deadends["wkt"], crs=crs)

    deadends_other_end["geometry"] = gpd.GeoSeries.from_wkt(
        deadends_other_end["wkt_other_end"], crs=crs
    )

    deadends_array = np.array(
        [(x, y) for x, y in zip(deadends.geometry.x, deadends.geometry.y)]
    )

    nodes_array = np.array([(x, y) for x, y in zip(nodes.geometry.x, nodes.geometry.y)])

    # finn nærmeste naboer
    k = k if len(deadends) >= k else len(deadends)
    nbr = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(nodes_array)
    all_dists, all_indices = nbr.kneighbors(deadends_array)

    fra = []
    til = []
    for i in np.arange(1, k):
        len_naa = len(fra)

        indices = all_indices[:, i]
        dists = all_dists[:, i]

        dists_other_end = deadends_other_end.distance(nodes.loc[indices], align=False)

        # get the deadend wkt and the node wkt if the distance is
        # between max_dist and min_dist and the distance is (considerably) shorter
        # than the distance from the other end of the line.
        fratil = [
            (geom, nodes.loc[idx, "wkt"])
            for geom, idx, dist, dist_andre, length in zip(
                deadends["wkt"],
                indices,
                dists,
                dists_other_end,
                deadends_lengths,
                strict=True,
            )
            if dist < max_dist and dist > min_dist and dist < dist_andre - length * 0.25
        ]

        til = til + [t for f, t in fratil if f not in fra]
        fra = fra + [f for f, _ in fratil if f not in fra]

        if len_naa == len(fra):
            break

    # make GeoDataFrame with straight lines
    fra = gpd.GeoSeries.from_wkt(fra, crs=crs)
    til = gpd.GeoSeries.from_wkt(til, crs=crs)
    new_lines = shortest_line(fra, til)
    new_lines = gpd.GeoDataFrame({"geometry": new_lines}, geometry="geometry", crs=crs)

    if not len(new_lines):
        return new_lines

    new_lines = _make_edge_wkt_cols(new_lines)

    return new_lines


def find_holes_deadends(nodes, max_dist, min_dist=0):
    """
    It takes a GeoDataFrame of nodes, chooses the deadends,
    and creates a straight line between the closest deadends
    if the distance is between the specifies max_dist and min_dist.

    Args:
      nodes: the nodes of the network
      max_dist: The maximum distance between two nodes to be connected.
      min_dist: minimum distance between nodes to be considered a hole. Defaults to 0

    Returns:
      A GeoDataFrame with the new lines.
    """

    crs = nodes.crs

    # velger ut nodene som kun finnes i én lenke. Altså blindveier i en networksanalyse.
    deadends = nodes[nodes["n"] == 1]

    # viktig å nullstille index siden sklearn kneighbors gir oss en numpy.array
    # med indekser
    deadends = deadends.reset_index(drop=True)

    if len(deadends) <= 1:
        deadends["minutter"] = -1
        return deadends

    # koordinater i tuple som numpy array
    deadends_array = np.array(
        [(x, y) for x, y in zip(deadends.geometry.x, deadends.geometry.y)]
    )

    # finn nærmeste to naboer
    nbr = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(deadends_array)
    dists, indices = nbr.kneighbors(deadends_array)

    # velg ut nest nærmeste (nærmeste er fra og til samme punkt)
    dists = dists[:, 1]
    indices = indices[:, 1]

    """
    Nå har vi 1d-numpy arrays av lik lengde som blindvegene.
    'indices' inneholder numpy-indeksen for vegen som er nærmest, altså endepunktene for de
    nye lenkene. For å konvertere dette fra numpy til geopandas, trengs geometri og
    node-id.
    """

    # fra-geometrien kan hentes direkte siden avstandene og blindvegene har samme
    # rekkefølge
    fra = np.array(
        [
            geom.wkt
            for dist, geom in zip(dists, deadends.geometry)
            if dist < max_dist and dist > min_dist
        ]
    )

    # til-geometrien må hentes via index-en
    til = np.array(
        [
            deadends.loc[idx, "wkt"]
            for dist, idx in zip(dists, indices)
            if dist < max_dist and dist > min_dist
        ]
    )

    # lag GeoDataFrame med rette linjer
    fra = gpd.GeoSeries.from_wkt(fra, crs=crs)
    til = gpd.GeoSeries.from_wkt(til, crs=crs)
    new_lines = shortest_line(fra, til)
    new_lines = gpd.GeoDataFrame({"geometry": new_lines}, geometry="geometry", crs=crs)

    if not len(new_lines):
        return new_lines

    new_lines = _make_edge_wkt_cols(new_lines)

    return new_lines


def cut_lines(gdf: GeoDataFrame, max_length: int, ignore_index=False) -> GeoDataFrame:
    """
    It cuts lines of a GeoDataFrame into pieces of a given length

    Args:
      gdf (GeoDataFrame): GeoDataFrame
      max_length (int): The maximum length of the lines in the output GeoDataFrame.
      ignore_index: If True, the resulting GeoDataFrame will have a simple RangeIndex. If False, it will
    retain the index of the original GeoDataFrame. Defaults to False

    Returns:
      A GeoDataFrame with lines cut to the maximum distance.

    """

    gdf["geometry"] = force_2d(gdf.geometry)

    gdf = gdf.explode(ignore_index=ignore_index)

    over_max_length = gdf.loc[gdf.length > max_length]

    if not len(over_max_length):
        return gdf

    def cut(line, distance):
        """from the shapely docs"""
        if distance <= 0.0 or distance >= line.length:
            return line
        coords = list(line.coords)
        for i, p in enumerate(coords):
            pd = line.project(Point(p))
            if pd == distance:
                return unary_union(
                    [LineString(coords[: i + 1]), LineString(coords[i:])]
                )
            if pd > distance:
                cp = line.interpolate(distance)
                return unary_union(
                    [
                        LineString(coords[:i] + [(cp.x, cp.y)]),
                        LineString([(cp.x, cp.y)] + coords[i:]),
                    ]
                )

    cut_vektorisert = np.vectorize(cut)

    for x in [10, 5, 1]:
        _max = max(over_max_length.length)
        print(_max)
        while _max > max_length * x + 1:
            _max = max(over_max_length.length)

            over_max_length["geometry"] = cut_vektorisert(
                over_max_length.geometry, max_length
            )

            over_max_length = over_max_length.explode(ignore_index=ignore_index)

            if _max == max(over_max_length.length):
                break

    over_max_length = over_max_length.explode(ignore_index=ignore_index)

    under_max_length = gdf.loc[gdf.length <= max_length]

    return pd.concat([under_max_length, over_max_length], ignore_index=ignore_index)

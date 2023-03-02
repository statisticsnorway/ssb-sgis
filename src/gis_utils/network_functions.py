import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame
from shapely import force_2d, shortest_line
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from sklearn.neighbors import NearestNeighbors

from .buffer_dissolve_explode import buff
from .distances import coordinate_array, get_k_nearest_neighbors, k_nearest_neighbours
from .geopandas_utils import gdf_concat, push_geom_col, snap_to


def get_largest_component(lines: GeoDataFrame) -> GeoDataFrame:
    """Create column 'connected', where '1' means that the line is part of the largest
    component of the network.

    It takes a GeoDataFrame of lines, creates a graph, finds the largest component,
    and maps this as the value '1' in the column 'connected'.

    Args:
      lines: A GeoDataFrame of lines

    Returns:
      A GeoDataFrame with a new column "connected"
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
    """Creates the column "component_size", which indicates the size of the network
    component the line is a part of.

    Args:
        lines: GeoDataFrame

    Returns:
        A GeoDataFrame with a new column "component_size"
    """

    if "source" not in lines.columns or "target" not in lines.columns:
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


def split_lines_at_closest_point(
    lines: GeoDataFrame,
    points: GeoDataFrame,
    max_dist: int | None = None,
) -> DataFrame:
    """Snaps points to lines and splits the lines in two at the snap point

    Args:
        lines: GeoDataFrame of lines that will be split
        points: GeoDataFrame of points to split the lines with
        max_dist: the maximum distance between the point and the line.
            Points further away than max_dist will not split any lines.
            Defaults to None.

    Returns:
        A GeoDataFrame with the same columns as the input lines, but with the lines split at the closest
        point to the points.

    Raises:
        ValueError if the crs of the input data differs.

    """

    BUFFDIST = 0.000001

    if points.crs != lines.crs:
        raise ValueError("crs mismatch:", points.crs, "and", lines.crs)

    lines["temp_idx_"] = lines.index

    # move the points to the nearest exact point of the line (to_node=False)
    # and get the line index ('snap_to_id')
    snapped = snap_to(
        points,
        lines,
        max_dist=max_dist,
        to_node=False,
        snap_to_id="temp_idx_",
    )

    condition = lines["temp_idx_"].isin(snapped["temp_idx_"])
    relevant_lines = lines.loc[condition]
    the_other_lines = lines.loc[~condition]

    # we need consistent coordinate dimensions later
    # doing it down here to not overwrite the original data
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
    splitted_source = GeoDataFrame(
        {
            "splitidx": splitted["splitidx"].reset_index(drop=True),
            "geometry": GeoSeries(
                [Point(geom) for geom in splitted["source_coords"]], crs=lines.crs
            ),
        }
    )
    splitted_target = GeoDataFrame(
        {
            "splitidx": splitted["splitidx"].reset_index(drop=True),
            "geometry": GeoSeries(
                [Point(geom) for geom in splitted["target_coords"]], crs=lines.crs
            ),
        }
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
    splitdict_source = {
        idx: coords
        for idx, coords in zip(dists_source["splitidx"], dists_source["point_coords"])
    }
    splitdict_target = {
        idx: coords
        for idx, coords in zip(dists_target["splitidx"], dists_target["point_coords"])
    }

    # now, we can finally replace the source/target coordinate with the coordinates of
    # the snapped points.

    # loop for each line where the source is the endpoint that was split
    # change the first point of the line to the point it was split by
    for idx in dists_source.splitidx:
        line = splitted.loc[idx, "geometry"]
        coordslist = list(line.coords)
        coordslist[0] = splitdict_source[idx]
        splitted.loc[splitted.splitidx == idx, "geometry"] = LineString(coordslist)

    # same for the lines where the target was split, but change the last point of the
    # line
    for idx in dists_target.splitidx:
        line = splitted.loc[idx, "geometry"]
        coordslist = list(line.coords)
        coordslist[-1] = splitdict_target[idx]
        splitted.loc[splitted.splitidx == idx, "geometry"] = LineString(coordslist)

    splitted["splitted"] = 1

    lines = gdf_concat([the_other_lines, splitted]).drop(
        ["temp_idx_", "splitidx"], axis=1
    )

    return lines


def cut_lines(gdf: GeoDataFrame, max_length: int, ignore_index=True) -> GeoDataFrame:
    """
    It cuts lines of a GeoDataFrame into pieces of a given length

    Args:
      gdf (GeoDataFrame): GeoDataFrame
      max_length (int): The maximum length of the lines in the output GeoDataFrame.
      ignore_index: If True, the resulting GeoDataFrame will have a simple RangeIndex.
        If False, you will get a
      MultiIndex. Defaults to True

    Returns:
      A GeoDataFrame with lines cut to the maximum distance.

    """

    gdf["geometry"] = force_2d(gdf.geometry)

    gdf = gdf.explode(ignore_index=ignore_index)

    long_lines = gdf.loc[gdf.length > max_length]

    if not len(long_lines):
        return gdf

    def cut(line, distance):
        """from the shapely docs, but returns unary_union."""
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
):
    """Fills gaps shorter than 'max_dist' in a GeoDataFrame of lines

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

    Returns:
        The input GeoDataFrame with new lines added
    """

    lines, nodes = make_node_ids(lines)

    if deadends_only:
        new_lines = _find_holes_deadends(nodes, max_dist, min_dist)
    else:
        new_lines = _find_holes_all_lines(lines, nodes, max_dist, min_dist)

    if not len(new_lines):
        return lines

    wkt_id_dict = {wkt: id for wkt, id in zip(nodes["wkt"], nodes["node_id"])}
    new_lines["source"] = new_lines["source_wkt"].map(wkt_id_dict)
    new_lines["target"] = new_lines["target_wkt"].map(wkt_id_dict)

    if any(new_lines.source.isna()) or any(new_lines.target.isna()):
        raise ValueError("Missing source/target ids.")

    if hole_col:
        new_lines[hole_col] = 1
        lines[hole_col] = 0

    return gdf_concat([lines, new_lines])


def _find_holes_all_lines(lines, nodes, max_dist, min_dist=0, k=10, length_factor=0.25):
    """
    creates a straight line between deadends and the closest node in a
    forward-going direction, if the distance is between the max_dist and min_dist.

    Args:
        lines: the lines you want to find holes in
        nodes: a GeoDataFrame of nodes
        max_dist: The maximum distance between the dead end and the node it should be
            connected to.
        min_dist: The minimum distance between the dead end and the node. Defaults to 0
        k: number of nearest neighbors to consider. Defaults to 10
        length_factor:

    Returns:
        A GeoDataFrame with the shortest line between the two points.
    """

    # wkt: well-known text, e.g. "POINT (60 10)"

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

    deadends_array = coordinate_array(deadends)

    nodes_array = coordinate_array(nodes)

    all_dists, all_indices = k_nearest_neighbours(deadends_array, nodes_array, k=k)

    # now to find the lines that go in the right direction. Collecting the startpoints
    # and endpoints of the new lines in lists, looping through the k neighbour points
    startpoints = []
    endpoints = []
    for i in np.arange(1, k):
        # to break out of the loop if no endpoints that meet the condition are found
        len_now = len(startpoints)

        # selecting the arrays for the k neighbour
        indices = all_indices[:, i]
        dists = all_dists[:, i]

        # get the distance from the other end of the deadends to the k neighbours
        dists_other_end = deadends_other_end.distance(nodes.loc[indices], align=False)

        # select the distances between min_dist and max_dist that are also shorter than
        # the distance from the other side of the line, meaning the new line will go
        # forward. All arrays have the same shape, and can be easily filtered by index
        # The node wkts have to be extracted after indexing 'indices'
        condition = (
            (dists < max_dist)
            & (dists > min_dist)
            & (dists < dists_other_end - deadends_lengths * length_factor)
        )
        from_wkt = deadends.loc[condition, "wkt"]
        to_idx = indices[condition]
        to_wkt = nodes.loc[to_idx, "wkt"]

        # now add the wkts to the lists of startpoints and endpoints. If the startpoint
        # is already added, the new wks will not be added again
        endpoints = endpoints + [
            t for f, t in zip(from_wkt, to_wkt) if f not in startpoints
        ]
        startpoints = startpoints + [
            f for f, _ in zip(from_wkt, to_wkt) if f not in startpoints
        ]

        # break out of the loop when no new endpoints meet the condition
        if len_now == len(startpoints):
            break

    # make GeoDataFrame with straight lines
    startpoints = gpd.GeoSeries.from_wkt(startpoints, crs=crs)
    endpoints = gpd.GeoSeries.from_wkt(endpoints, crs=crs)
    new_lines = shortest_line(startpoints, endpoints)
    new_lines = gpd.GeoDataFrame({"geometry": new_lines}, geometry="geometry", crs=crs)

    if not len(new_lines):
        return new_lines

    new_lines = make_edge_wkt_cols(new_lines)

    return new_lines


def _find_holes_deadends(nodes, max_dist, min_dist=0):
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

    condition = (dists < max_dist) & (dists > min_dist)
    from_geom = deadends.loc[condition, "geometry"]
    to_idx = indices[condition]
    to_geom = deadends.loc[to_idx, "geometry"]

    # lag GeoDataFrame med rette linjer
    new_lines = shortest_line(from_geom, to_geom)
    new_lines = gpd.GeoDataFrame({"geometry": new_lines}, geometry="geometry", crs=crs)

    if not len(new_lines):
        return new_lines

    new_lines = make_edge_wkt_cols(new_lines)

    return new_lines


def make_node_ids(
    lines: GeoDataFrame,
    wkt: bool = True,
) -> tuple[GeoDataFrame, GeoDataFrame]:
    """Create node_ids and assign them to the lines' columns 'source' and 'target'

    Creates an index for the unique endpoints (nodes) of the lines (edges) of a
    GeoDataFrame, then maps this index to the 'source' and 'target' columns of the
    'lines' GeoDataFrame. Returns both the lines and the nodes.

    Args:
      lines (GeoDataFrame): GeoDataFrame with line geometries

    Returns:
      A tuple of two GeoDataFrames, one with the lines and one with the nodes.
    """

    if wkt:
        lines = make_edge_wkt_cols(lines)
        source_geom_col, target_geom_col = "source_wkt", "target_wkt"
    else:
        lines = make_edge_coords_cols(lines)
        source_geom_col, target_geom_col = "source_coords", "target_coords"

    sources = lines[[source_geom_col]].rename(columns={source_geom_col: "wkt"})
    targets = lines[[target_geom_col]].rename(columns={target_geom_col: "wkt"})

    nodes = pd.concat([sources, targets], axis=0, ignore_index=True)

    nodes["n"] = nodes.assign(n=1).groupby("wkt")["n"].transform("sum")

    nodes = nodes.drop_duplicates(subset=["wkt"]).reset_index(drop=True)

    nodes["node_id"] = nodes.index
    nodes["node_id"] = nodes["node_id"].astype(str)

    id_dict = {wkt: node_id for wkt, node_id in zip(nodes["wkt"], nodes["node_id"])}
    lines["source"] = lines[source_geom_col].map(id_dict)
    lines["target"] = lines[target_geom_col].map(id_dict)

    n_dict = {wkt: n for wkt, n in zip(nodes["wkt"], nodes["n"])}
    lines["n_source"] = lines[source_geom_col].map(n_dict)
    lines["n_target"] = lines[target_geom_col].map(n_dict)

    nodes["geometry"] = gpd.GeoSeries.from_wkt(nodes.wkt, crs=lines.crs)
    nodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs=lines.crs)
    nodes = nodes.reset_index(drop=True)

    lines = push_geom_col(lines)

    return lines, nodes


def make_edge_coords_cols(lines: GeoDataFrame) -> GeoDataFrame:
    """Get the wkt of the first and last points of lines as columns

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
    """Get coordinate tuples of the first and last points of lines as columns

    It takes a GeoDataFrame of LineStrings and returns a GeoDataFrame with two new
    columns, source_wkt and target_wkt, which are the WKT representations of the first
    and last points of the LineStrings

    Args:
      lines (GeoDataFrame): the GeoDataFrame with the lines

    Returns:
      A GeoDataFrame with new columns 'source_wkt' and 'target_wkt'
    """

    lines, endpoints = _prepare_make_edge_cols(lines)

    wkt_geom = [f"POINT ({x} {y})" for x, y in zip(endpoints.x, endpoints.y)]
    lines["source_wkt"], lines["target_wkt"] = (
        wkt_geom[0::2],
        wkt_geom[1::2],
    )

    return lines


def _prepare_make_edge_cols(
    lines: GeoDataFrame,
) -> tuple[GeoDataFrame, GeoDataFrame]:
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
                "allowed in make_edge_wkt_cols."
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

    # for hver rundkjøring: lag linjer mellom rundkjøringens mitdpunkt og hver linje som grenser til rundkjøringen
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

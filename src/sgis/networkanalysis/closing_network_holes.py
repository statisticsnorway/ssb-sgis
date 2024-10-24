"""Functions for filling gaps in the network with straight lines."""

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from pandas import DataFrame

from ..geopandas_tools.conversion import coordinate_array
from ..geopandas_tools.general import get_line_segments
from ..geopandas_tools.neighbors import k_nearest_neighbors
from ..geopandas_tools.sfilter import sfilter
from .nodes import make_edge_wkt_cols
from .nodes import make_node_ids


def get_k_nearest_points_for_deadends(
    lines: GeoDataFrame, k: int, max_distance: int
) -> GeoDataFrame:

    assert lines.index.is_unique
    lines = lines.assign(_range_idx_left=range(len(lines)))
    points = (
        lines.assign(
            geometry=lambda x: x.extract_unique_points().values,
            _range_idx_right=range(len(lines)),
        )
        .explode(index_parts=False)
        .sort_index()
    )

    points_grouper = points.groupby("_range_idx_right")["geometry"]
    nodes = pd.concat(
        [
            points_grouper.nth(0),
            points_grouper.nth(-1),
        ]
    )
    nodes.index.name = "_range_idx_right"
    nodes = nodes.reset_index()

    def has_no_duplicates(nodes):
        counts = nodes.geometry.value_counts()
        return nodes.geometry.isin(counts[counts == 1].index)

    deadends = nodes[has_no_duplicates].reset_index(drop=True)

    deadends_buffered = deadends.assign(geometry=lambda x: x.buffer(max_distance))

    segs_by_deadends = (
        sfilter(lines, deadends_buffered)
        .pipe(get_line_segments)
        .sjoin(deadends_buffered)
        .loc[lambda x: x["_range_idx_left"] != x["_range_idx_right"]]
    )

    lines_between = shapely.shortest_line(
        segs_by_deadends.geometry.values,
        deadends.loc[segs_by_deadends["index_right"].values].geometry.values,
    )
    segs_by_deadends.geometry.loc[:] = shapely.get_point(lines_between, 0)

    length_mapper = dict(enumerate(shapely.length(lines_between)))
    sorted_lengths = dict(
        reversed(sorted(length_mapper.items(), key=lambda item: item[1]))
    )
    nearest_first = segs_by_deadends.iloc[list(sorted_lengths)]

    k_nearest_per_deadend = nearest_first.geometry.groupby(level=0).apply(
        lambda x: x.head(k)
    )

    return GeoDataFrame({"geometry": k_nearest_per_deadend.values}, crs=lines.crs)


def close_network_holes(
    gdf: GeoDataFrame,
    max_distance: int | float,
    max_angle: int,
    hole_col: str | None = "hole",
) -> GeoDataFrame:
    """Fills network gaps with straigt lines.

    Fills holes in the network by connecting deadends with the nodes that are
    within the 'max_distance' distance.

    Args:
        gdf: GeoDataFrame with lines.
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

    Note:
        The holes will have missing values in the weight column used in
        NetworkAnalysis. These values must be filled before analysis.

    Examples:
    ---------
    Read road data with small gaps.

    >>> import sgis as sg
    >>> roads = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")

    Roads need to be singlepart linestrings for this to work.

    >>> from shapely import line_merge
    >>> roads.geometry = line_merge(roads.geometry)

    Fill gaps shorter than 1.1 meters.

    >>> filled = sg.close_network_holes(roads, max_distance=1.1, max_angle=180)
    >>> filled.hole.value_counts()
    hole
    0    93395
    1     7102
    Name: count, dtype: int64

    Compare the number of isolated lines before and after.

    >>> roads = sg.get_connected_components(roads)
    >>> roads.connected.value_counts()
    1.0    85638
    0.0     7757
    Name: connected, dtype: int64

    >>> filled = sg.get_connected_components(filled)
    >>> filled.connected.value_counts()
    1.0    100315
    0.0       180
    Name: connected, dtype: int64

    Fill only gaps with an angle deviation between 0 and 30 compared to the prior line.

    >>> filled = sg.close_network_holes(roads, max_distance=1.1, max_angle=30)
    >>> filled.hole.value_counts()
    hole
    0    93395
    1     7092
    Name: count, dtype: int64

    It's not always wise to fill gaps. In the case of this data, these small gaps are
    intentional. They are road blocks where most cars aren't allowed to pass. Fill the
    holes only if it makes the travel times/routes more realistic.
    """
    lines, nodes = make_node_ids(gdf)

    # remove duplicates of lines going both directions
    lines["_sorted"] = [
        "_".join(sorted([source, target])) + str(round(length, 4))
        for source, target, length in zip(
            lines["source"], lines["target"], lines.length, strict=True
        )
    ]

    lines = lines.drop_duplicates("_sorted").drop(columns="_sorted")

    # new_lines, angles = _close_holes_all_lines(
    new_lines: GeoSeries = _close_holes_all_lines(
        lines,
        nodes,
        max_distance,
        max_angle,
        idx_start=1,
    )

    new_lines = gpd.GeoDataFrame(
        {"geometry": new_lines}, geometry="geometry", crs=gdf.crs
    )

    if not len(new_lines):
        lines[hole_col] = (
            0 if hole_col not in lines.columns else lines[hole_col].fillna(0)
        )
        return lines

    new_lines = make_edge_wkt_cols(new_lines)

    wkt_id_dict = dict(zip(nodes["wkt"], nodes["node_id"], strict=True))
    new_lines["source"] = new_lines["source_wkt"].map(wkt_id_dict)
    new_lines["target"] = new_lines["target_wkt"].map(wkt_id_dict)

    if hole_col:
        new_lines[hole_col] = 1
        lines[hole_col] = (
            0 if hole_col not in lines.columns else lines[hole_col].fillna(0)
        )

    return pd.concat([lines, new_lines], ignore_index=True)


def close_network_holes_to_deadends(
    gdf: GeoDataFrame,
    max_distance: int | float,
    hole_col: str | None = "hole",
) -> GeoDataFrame:
    """Fills gaps between two deadends if the distance is less than 'max_distance'.

    Fills holes between deadends in the network with straight lines if the distance is
    less than 'max_distance'.

    Args:
        gdf: GeoDataFrame with lines
        max_distance: The maximum distance between two nodes to be considered a hole.
        hole_col: If you want to keep track of which lines were added, you can add a
            column with a value of 1. Defaults to 'hole'

    Returns:
        The input GeoDataFrame with new lines added.

    Examples:
    ---------
    Read road data with small gaps.

    >>> import sgis as sg
    >>> roads = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")

    Roads need to be singlepart linestrings for this to work.

    >>> from shapely import line_merge
    >>> roads.geometry = line_merge(roads.geometry)

    Check for number of isolated lines now.

    >>> roads = sg.get_connected_components(roads)
    >>> roads.connected.value_counts()
    1.0    85638
    0.0     7757
    Name: connected, dtype: int64

    Fill gaps shorter than 1.1 meters.

    >>> filled = sg.close_network_holes_to_deadends(roads, max_distance=1.1)
    >>> roads = sg.get_connected_components(roads)
    >>> roads.connected.value_counts()
    1.0    100315
    0.0       180
    Name: connected, dtype: int64

    It's not always wise to fill gaps. In the case of this data, these small gaps are
    intentional. They are road blocks where most cars aren't allowed to pass. Fill the
    holes only if it makes the travel times/routes more realistic.
    """
    gdf, nodes = make_node_ids(gdf)

    new_lines = _find_holes_deadends(nodes, max_distance)

    if not len(new_lines):
        gdf[hole_col] = 0 if hole_col not in gdf.columns else gdf[hole_col].fillna(0)
        return gdf

    new_lines = make_edge_wkt_cols(new_lines)

    wkt_id_dict = {
        wkt: id_ for wkt, id_ in zip(nodes["wkt"], nodes["node_id"], strict=True)
    }
    new_lines["source"] = new_lines["source_wkt"].map(wkt_id_dict)
    new_lines["target"] = new_lines["target_wkt"].map(wkt_id_dict)

    if hole_col:
        new_lines[hole_col] = 1
        gdf[hole_col] = 0 if hole_col not in gdf.columns else gdf[hole_col].fillna(0)

    return pd.concat([gdf, new_lines], ignore_index=True)


def _close_holes_all_lines(
    lines: GeoDataFrame,
    nodes: GeoDataFrame,
    max_distance: int | None,
    max_angle: int | None,
    idx_start: int,
) -> GeoSeries:
    k = min(len(nodes), 50)

    # make points for the deadends and the other endpoint of the deadend lines
    deadends_target = lines.loc[lines["n_target"] == 1].rename(
        columns={"target_wkt": "wkt", "source_wkt": "wkt_other_end"}
    )
    deadends_source = lines.loc[lines["n_source"] == 1].rename(
        columns={"source_wkt": "wkt", "target_wkt": "wkt_other_end"}
    )
    deadends = pd.concat([deadends_source, deadends_target], ignore_index=True)

    if len(deadends) <= 1:
        return GeoSeries()

    deadends_other_end = deadends.copy()
    deadends_other_end["geometry"] = gpd.GeoSeries.from_wkt(
        deadends_other_end["wkt_other_end"]
    )

    deadends["geometry"] = gpd.GeoSeries.from_wkt(deadends["wkt"])

    deadends_array = coordinate_array(deadends)
    nodes_array = coordinate_array(nodes)

    all_dists, all_indices = k_nearest_neighbors(deadends_array, nodes_array, k=k)

    deadends_other_end_array = coordinate_array(deadends_other_end)

    # now to find the lines that have the correct angle and distance
    # and endpoints of the new lines in lists, looping through the k neighbour points
    new_sources: list[str] = []
    new_targets: list[str] = []
    # all_angles = []
    for i in np.arange(idx_start, k):
        # to break out of the loop if no new_targets that meet the condition are found
        len_now = len(new_sources)

        # selecting the arrays for the current k neighbour
        indices = all_indices[:, i]
        dists = all_dists[:, i]

        these_nodes_array = coordinate_array(nodes.iloc[indices])

        if np.all(deadends_other_end_array == these_nodes_array):
            continue

        angles_deadend_to_node = get_angle(deadends_array, these_nodes_array)

        angles_deadend_to_deadend_other_end = get_angle(
            deadends_other_end_array, deadends_array
        )

        def get_angle_difference(angle1, angle2):
            return np.abs((angle1 - angle2 + 180) % 360 - 180)

        angles_difference = get_angle_difference(
            angles_deadend_to_deadend_other_end, angles_deadend_to_node
        )

        angles_difference[
            np.all(deadends_other_end_array == these_nodes_array, axis=1)
        ] = np.nan

        condition = (dists <= max_distance) & (angles_difference <= max_angle)

        from_wkt = deadends.loc[condition, "wkt"]
        to_idx = indices[condition]
        to_wkt = nodes.iloc[to_idx]["wkt"]

        # all_angles = all_angles + [
        #     diff
        #     for f, diff in zip(from_wkt, angles_difference[condition], strict=True)
        #     if f not in new_sources
        # ]

        # now add the wkts to the lists of new sources and targets. If the source
        # is already added, the new wks will not be added again
        new_targets = new_targets + [
            t for f, t in zip(from_wkt, to_wkt, strict=True) if f not in new_sources
        ]
        new_sources = new_sources + [f for f in from_wkt if f not in new_sources]

        # break out of the loop when no new new_targets meet the condition
        if len_now == len(new_sources):
            break

    # make GeoSeries with straight lines
    new_sources = gpd.GeoSeries.from_wkt(new_sources, crs=lines.crs)
    new_targets = gpd.GeoSeries.from_wkt(new_targets, crs=lines.crs)
    return shapely.shortest_line(new_sources, new_targets)  # , all_angles


def get_angle(array_a: np.ndarray, array_b: np.ndarray) -> np.ndarray:
    dx = array_b[:, 0] - array_a[:, 0]
    dy = array_b[:, 1] - array_a[:, 1]

    angles_rad = np.arctan2(dx, dy)
    angles_degrees = np.degrees(angles_rad)
    return angles_degrees


def _find_holes_deadends(
    nodes: GeoDataFrame, max_distance: float | int
) -> GeoDataFrame | DataFrame:
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
        return pd.DataFrame()

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
    new_lines = shapely.shortest_line(from_geom, to_geom)
    new_lines = gpd.GeoDataFrame({"geometry": new_lines}, geometry="geometry", crs=crs)

    return new_lines

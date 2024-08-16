import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from igraph import Graph
from shapely import force_2d
from shapely import reverse
from shapely import unary_union
from shapely import union_all
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from shapely.ops import nearest_points
from shapely.wkt import loads

from ..geopandas_tools.sfilter import sfilter
from .cutting_lines import cut_lines_once
from .nodes import make_edge_wkt_cols


def _service_area(
    graph: Graph,
    origins: GeoDataFrame,
    weight: str,
    breaks: np.ndarray,
    lines: GeoDataFrame,
    nodes: GeoDataFrame,
    directed: bool,
    precice: bool,
) -> GeoDataFrame:
    # make sure the nodes are alligned with the vertices in the graph
    # the weight/distances can then be assigned directly onto the nodes
    node_df = pd.DataFrame(index=np.array(graph.vs["name"]))
    nodes = nodes.set_index("node_id").drop("geometry", axis=1)
    nodes = node_df.join(nodes)

    # double edge df with source/target as index
    edge_df = lines.loc[:, ["source", "target", "src_tgt_wt", "geometry"]]
    edge_df = pd.concat(
        [
            edge_df.set_index("source", drop=False),
            edge_df.set_index("target", drop=False),
        ]
    )

    # distance from all origins to all vertices/nodes in the graph
    all_distances: list[list[str]] = graph.distances(
        weights="weight", source=origins["temp_idx"], mode="out"
    )

    if not directed:
        nodes_union: MultiPoint = unary_union(loads(nodes["wkt"].dropna().values))

    # loop through every origin and every break
    service_areas: list[GeoDataFrame] = []
    for i, idx in enumerate(origins["temp_idx"]):
        # assign distances to the nodes and join the column to the edges
        nodes[weight] = all_distances[i]
        distance_df = edge_df.join(nodes)

        for break_ in breaks:
            nodes_within_break = nodes.loc[nodes[weight] <= break_]

            whole_edge_is_within = (edge_df.source.isin(nodes_within_break.index)) & (
                edge_df.target.isin(nodes_within_break.index)
            )
            edges_within = edge_df.loc[whole_edge_is_within]

            if not precice:
                if not len(edges_within):
                    edges_within = GeoDataFrame(
                        {"geometry": [None]}, geometry="geometry", crs=lines.crs
                    )
                edges_within["origin"] = idx
                edges_within[weight] = break_
                service_areas.append(edges_within)
                continue

            # only part of the line is within the break
            partly_within = _part_of_edge_within(
                distance_df, nodes_within_break, directed
            )

            if not len(edges_within) and not len(partly_within):
                nan_gdf = GeoDataFrame(
                    {"origin": [idx], weight: [break_], "geometry": [None]},
                    geometry="geometry",
                    crs=lines.crs,
                )
                service_areas.append(nan_gdf)
                continue

            if not len(partly_within):
                edges_within["origin"] = idx
                edges_within[weight] = break_
                service_areas.append(edges_within)
                continue

            partly_within["remaining_distance"] = break_ - partly_within[weight]

            split_lines = _split_lines(partly_within, directed)

            split_lines = make_edge_wkt_cols(split_lines)

            # select the cutted lines shorter than the cut distance that intersects
            # with the nodes
            within = split_lines.loc[
                lambda df: (df.length <= df["remaining_distance"] * 1.01)
                & (df["source_wkt"].isin(nodes.wkt) | df["target_wkt"].isin(nodes.wkt))
            ]

            # keep only lines intersecting the edges completely within or the snapped origin
            if not directed:
                if len(edges_within):
                    within = sfilter(within, edges_within.buffer(0.01))
                else:
                    snapped_origin: Point = nearest_points(
                        nodes_union,
                        union_all(
                            origins.loc[
                                origins["temp_idx"] == idx, "geometry"
                            ].geometry.values
                        ),
                    )[0]

                    within = sfilter(within, snapped_origin.buffer(0.01))

            edges_within = pd.concat([edges_within, within], ignore_index=True)
            edges_within["origin"] = idx
            edges_within[weight] = break_

            service_areas.append(edges_within)

    return pd.concat(
        service_areas,
        ignore_index=True,
    )


def _part_of_edge_within(distance_df, nodes_within_break, directed):
    if directed:
        return distance_df.loc[(distance_df.source.isin(nodes_within_break.index))]
    else:
        return distance_df.loc[
            (distance_df.source.isin(nodes_within_break.index))
            | (distance_df.target.isin(nodes_within_break.index))
        ]


def _split_lines(partly_within, directed):
    partly_within.geometry = force_2d(partly_within.geometry)

    if not directed:
        rev = partly_within.copy()
        rev.geometry = reverse(partly_within.geometry)
        partly_within = pd.concat([partly_within, rev])

    lines_cut = cut_lines_once(partly_within, "remaining_distance")

    return lines_cut

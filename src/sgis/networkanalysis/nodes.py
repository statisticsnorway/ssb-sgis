"""Create nodes (points) from a GeoDataFrame of lines.

The functions are used inside inside other functions, and aren't needed
to use explicitly.
"""

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from shapely.geometry import Point

from ..geopandas_tools.general import _push_geom_col
from ..geopandas_tools.general import make_edge_coords_cols
from ..geopandas_tools.general import make_edge_wkt_cols
from ..geopandas_tools.geometry_types import make_all_singlepart


def make_node_ids(
    gdf: GeoDataFrame,
    wkt: bool = True,
) -> tuple[GeoDataFrame, GeoDataFrame]:
    """Gives the lines unique node ids and returns lines (edges) and nodes.

    Takes the first and last point of each line and creates a GeoDataFrame of
    nodes (points) with a column 'node_id'. The node ids are then assigned to the
    input GeoDataFrame of lines as the columns 'source' and 'target'.

    Args:
        gdf: GeoDataFrame with line geometries
        wkt: If True (default), the resulting nodes will include the column 'wkt',
            containing the well-known text representation of the geometry. If False, it
            will include the column 'coords', a tuple with x and y geometries.

    Returns:
        A tuple of two GeoDataFrames, one with the lines and one with the nodes.

    Note:
        The lines must be singlepart linestrings.
    """
    gdf = make_all_singlepart(gdf, ignore_index=True)

    if wkt:
        gdf = make_edge_wkt_cols(gdf)
        geomcol1, geomcol2, geomcol_final = "source_wkt", "target_wkt", "wkt"
    else:
        gdf = make_edge_coords_cols(gdf)
        geomcol1, geomcol2, geomcol_final = "source_coords", "target_coords", "coords"

    # remove identical lines in opposite directions
    gdf["meters_"] = gdf.length.astype(str)

    sources = gdf[[geomcol1, geomcol2, "meters_"]].rename(
        columns={geomcol1: geomcol_final, geomcol2: "temp"}
    )
    targets = gdf[[geomcol1, geomcol2, "meters_"]].rename(
        columns={geomcol2: geomcol_final, geomcol1: "temp"}
    )

    nodes = (
        pd.concat([sources, targets], axis=0, ignore_index=True)
        .drop_duplicates([geomcol_final, "temp", "meters_"])
        .drop(["meters_", "temp"], axis=1)
    )

    gdf = gdf.drop("meters_", axis=1)

    nodes["n"] = nodes.assign(n=1).groupby(geomcol_final)["n"].transform("sum")

    nodes = nodes.drop_duplicates(subset=[geomcol_final]).reset_index(drop=True)

    nodes["node_id"] = nodes.index
    nodes["node_id"] = nodes["node_id"].astype(str)

    id_dict = {
        geom: node_id
        for geom, node_id in zip(nodes[geomcol_final], nodes["node_id"], strict=True)
    }
    gdf["source"] = gdf[geomcol1].map(id_dict)
    gdf["target"] = gdf[geomcol2].map(id_dict)

    n_dict = {geom: n for geom, n in zip(nodes[geomcol_final], nodes["n"], strict=True)}
    gdf["n_source"] = gdf[geomcol1].map(n_dict)
    gdf["n_target"] = gdf[geomcol2].map(n_dict)

    if wkt:
        nodes["geometry"] = gpd.GeoSeries.from_wkt(nodes[geomcol_final], crs=gdf.crs)
    else:
        nodes["geometry"] = GeoSeries(
            [Point(geom) for geom in nodes[geomcol_final]], crs=gdf.crs
        )
    nodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs=gdf.crs)
    nodes = nodes.reset_index(drop=True)

    gdf = _push_geom_col(gdf)

    return gdf, nodes

"""Create nodes (points) from a GeoDataFrame of lines.

The functions are used inside inside other functions, and aren't needed
to use explicitly.
"""

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point

from ..geopandas_tools.general import _push_geom_col


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

    gdf = gdf.explode(index_parts=False)

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


def make_edge_coords_cols(gdf: GeoDataFrame) -> GeoDataFrame:
    """Get the wkt of the first and last points of lines as columns.

    It takes a GeoDataFrame of LineStrings and returns a GeoDataFrame with two new
    columns, source_coords and target_coords, which are the x and y coordinates of the
    first and last points of the LineStrings in a tuple. The lines all have to be

    Args:
        gdf (GeoDataFrame): the GeoDataFrame with the lines

    Returns:
        A GeoDataFrame with new columns 'source_coords' and 'target_coords'
    """
    try:
        gdf, endpoints = _prepare_make_edge_cols_simple(gdf)
    except ValueError:
        gdf, endpoints = _prepare_make_edge_cols(gdf)

    coords = [(geom.x, geom.y) for geom in endpoints.geometry]
    gdf["source_coords"], gdf["target_coords"] = (
        coords[0::2],
        coords[1::2],
    )

    return gdf


def make_edge_wkt_cols(gdf: GeoDataFrame) -> GeoDataFrame:
    """Get coordinate tuples of the first and last points of lines as columns.

    It takes a GeoDataFrame of LineStrings and returns a GeoDataFrame with two new
    columns, source_wkt and target_wkt, which are the WKT representations of the first
    and last points of the LineStrings

    Args:
        gdf (GeoDataFrame): the GeoDataFrame with the lines

    Returns:
        A GeoDataFrame with new columns 'source_wkt' and 'target_wkt'
    """
    try:
        gdf, endpoints = _prepare_make_edge_cols_simple(gdf)
    except ValueError:
        gdf, endpoints = _prepare_make_edge_cols(gdf)

    wkt_geom = [
        f"POINT ({x} {y})" for x, y in zip(endpoints.x, endpoints.y, strict=True)
    ]
    gdf["source_wkt"], gdf["target_wkt"] = (
        wkt_geom[0::2],
        wkt_geom[1::2],
    )

    return gdf


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

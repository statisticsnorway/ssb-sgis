"""Functions for Finding network components in a GeoDataFrame of lines."""

import networkx as nx
from geopandas import GeoDataFrame

from .nodes import make_node_ids


def get_connected_components(gdf: GeoDataFrame) -> GeoDataFrame:
    """Finds the largest network component.

    It takes a GeoDataFrame of lines and finds the lines that are
    part of the largest connected network component. These lines are given the value
    1 in the added column 'connected', while isolated network islands get the value
    0.

    Uses the connected_components function from the networkx package.

    Args:
        gdf: A GeoDataFrame of lines.

    Returns:
        The GeoDataFrame with a new column "connected".

    Examples
    --------
    >>> from sgis import read_parquet_url, get_connected_components
    >>> roads = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")

    >>> roads = get_connected_components(roads)
    >>> roads.connected.value_counts()
    1.0    85638
    0.0     7757
    Name: connected, dtype: int64

    Removing the isolated network islands.

    >>> connected_roads = get_connected_components(roads).query("connected == 1")
    >>> roads.connected.value_counts()
    1.0    85638
    Name: connected, dtype: int64
    """
    gdf, _ = make_node_ids(gdf)

    edges = [
        (str(source), str(target))
        for source, target in zip(gdf["source"], gdf["target"], strict=True)
    ]

    graph = nx.Graph()
    graph.add_edges_from(edges)

    largest_component = max(nx.connected_components(graph), key=len)

    largest_component_dict = {node_id: 1 for node_id in largest_component}

    gdf["connected"] = gdf.source.map(largest_component_dict).fillna(0)

    gdf = gdf.drop(
        ["source_wkt", "target_wkt", "source", "target", "n_source", "n_target"], axis=1
    )

    return gdf


def get_component_size(gdf: GeoDataFrame) -> GeoDataFrame:
    """Finds the size of each component in the network.

    Takes a GeoDataFrame of linea and creates the column "component_size", which
    indicates the size of the network component the line is a part of.

    Args:
        gdf: a GeoDataFrame of lines.

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
    gdf, _ = make_node_ids(gdf)

    edges = [
        (str(source), str(target))
        for source, target in zip(gdf["source"], gdf["target"], strict=True)
    ]

    graph = nx.Graph()
    graph.add_edges_from(edges)
    components = [list(x) for x in nx.connected_components(graph)]

    componentsdict = {
        idx: len(component) for component in components for idx in component
    }

    gdf["component_size"] = gdf.source.map(componentsdict)

    gdf = gdf.drop(
        ["source_wkt", "target_wkt", "source", "target", "n_source", "n_target"], axis=1
    )

    return gdf

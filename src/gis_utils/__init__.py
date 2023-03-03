# flake8: noqa: F401
from geopandas import GeoDataFrame, GeoSeries

from .buffer_dissolve_explode import buff, buffdiss, buffdissexp, diss, dissexp, exp
from .directednetwork import DirectedNetwork
from .distances import coordinate_array, get_k_nearest_neighbors, k_nearest_neighbors
from .geopandas_utils import (
    clean_clip,
    clean_geoms,
    close_holes,
    find_neighbors,
    find_neighbours,
    gdf_concat,
    push_geom_col,
    random_points,
    sjoin,
    snap_to,
    to_gdf,
    to_multipoint,
    to_single_geom_type,
)
from .maps import clipmap, concat_explore, qtm, samplemap
from .network import Network
from .network_functions import (
    close_network_holes,
    cut_lines,
    get_component_size,
    get_largest_component,
    make_edge_coords_cols,
    make_edge_wkt_cols,
    make_node_ids,
    split_lines_at_closest_point,
)
from .networkanalysis import NetworkAnalysis
from .networkanalysisrules import NetworkAnalysisRules
from .overlay import clean_shapely_overlay, overlay, overlay_update


try:
    from .dapla import exists, read_geopandas, write_geopandas
except ImportError:
    pass


def testgdf(cols: str | None = None) -> GeoDataFrame:
    """GeoDataFrame with 9 rows consisting of points, a line and a polygon.

    Args:
        cols: What columns to include. Defaults to None, meaning all. These are
        'txtcol', 'numcol' and 'geometry'.

    Returns:
        GeoDataFrame
    """

    if isinstance(cols, str):
        cols = [cols]

    if cols:
        if not all(col in ["txtcol", "numcol", "geometry"] for col in cols):
            raise ValueError(
                "Wrong 'cols' value. Should be any of 'txtcol', 'numcol', "
                f"'geometry'. Got {', '.join(cols)}"
            )

    xs = [
        10.7497196,
        10.7484624,
        10.7480624,
        10.7384624,
        10.7374624,
        10.7324624,
        10.7284624,
    ]
    ys = [
        59.9281407,
        59.9275268,
        59.9272268,
        59.9175268,
        59.9165268,
        59.9365268,
        59.9075268,
    ]
    points = [f"POINT ({x} {y})" for x, y in zip(xs, ys)]

    line = [
        "LINESTRING ("
        "10.7284623 59.9075267, "
        "10.7184623 59.9175267, "
        "10.7114623 59.9135267, "
        "10.7143623 59.8975267, "
        "10.7384623 59.900000, "
        "10.720000 59.9075200)"
    ]

    polygon = [
        "POLYGON (("
        "10.74 59.92, 10.735 59.915, "
        "10.73 59.91, 10.725 59.905, "
        "10.72 59.9, 10.72 59.91, "
        "10.72 59.91, 10.74 59.92))"
    ]

    geoms = points + line + polygon

    gdf = GeoDataFrame(
        {"geometry": GeoSeries.from_wkt(geoms)}, geometry="geometry", crs=4326
    ).to_crs(25833)

    gdf2 = to_gdf(points + line + polygon, crs=4326).to_crs(25833)

    assert gdf.equals(
        gdf2
    ), "to_gdf does not give same results as manual gdf constructor"

    gdf["numcol"] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    gdf["txtcol"] = [*"aaaabbbcc"]

    assert len(gdf) == 9, "wrong number of rows"

    x = round(gdf.dissolve().centroid.x.iloc[0], 5)
    y = round(gdf.dissolve().centroid.y.iloc[0], 5)
    assert (
        f"POINT ({x} {y})" == "POINT (261106.48628 6649101.81219)"
    ), "wrong centerpoints. Have the testdata changed?"

    if cols:
        return gdf[cols]
    else:
        return gdf

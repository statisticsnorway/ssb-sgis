config = {
    "n_jobs": 1,
}


import sgis.raster.indices as indices

from .geopandas_tools.bounds import Gridlooper
from .geopandas_tools.bounds import bounds_to_points
from .geopandas_tools.bounds import bounds_to_polygon
from .geopandas_tools.bounds import get_total_bounds
from .geopandas_tools.bounds import gridloop
from .geopandas_tools.bounds import make_grid
from .geopandas_tools.bounds import make_grid_from_bbox
from .geopandas_tools.bounds import make_ssb_grid
from .geopandas_tools.buffer_dissolve_explode import buff
from .geopandas_tools.buffer_dissolve_explode import buffdiss
from .geopandas_tools.buffer_dissolve_explode import buffdissexp
from .geopandas_tools.buffer_dissolve_explode import buffdissexp_by_cluster
from .geopandas_tools.buffer_dissolve_explode import diss
from .geopandas_tools.buffer_dissolve_explode import diss_by_cluster
from .geopandas_tools.buffer_dissolve_explode import dissexp
from .geopandas_tools.buffer_dissolve_explode import dissexp_by_cluster
from .geopandas_tools.centerlines import get_rough_centerlines
from .geopandas_tools.cleaning import coverage_clean
from .geopandas_tools.cleaning import split_and_eliminate_by_longest

# from .geopandas_tools.cleaning import split_by_neighbors
from .geopandas_tools.conversion import coordinate_array
from .geopandas_tools.conversion import from_4326
from .geopandas_tools.conversion import to_4326
from .geopandas_tools.conversion import to_bbox
from .geopandas_tools.conversion import to_gdf
from .geopandas_tools.conversion import to_geoseries
from .geopandas_tools.conversion import to_shapely
from .geopandas_tools.duplicates import get_intersections  # drop_duplicate_geometries,
from .geopandas_tools.duplicates import update_geometries  # drop_duplicate_geometries,
from .geopandas_tools.general import _rename_geometry_if
from .geopandas_tools.general import clean_clip
from .geopandas_tools.general import clean_geoms
from .geopandas_tools.general import drop_inactive_geometry_columns
from .geopandas_tools.general import get_common_crs
from .geopandas_tools.general import get_grouped_centroids
from .geopandas_tools.general import get_line_segments
from .geopandas_tools.general import make_lines_between_points
from .geopandas_tools.general import points_in_bounds
from .geopandas_tools.general import random_points
from .geopandas_tools.general import random_points_in_polygons
from .geopandas_tools.general import sort_large_first
from .geopandas_tools.general import sort_long_first
from .geopandas_tools.general import sort_short_first
from .geopandas_tools.general import sort_small_first
from .geopandas_tools.general import split_out_circles
from .geopandas_tools.general import to_lines
from .geopandas_tools.geocoding import address_to_coords
from .geopandas_tools.geocoding import address_to_gdf
from .geopandas_tools.geometry_types import get_geom_type
from .geopandas_tools.geometry_types import is_single_geom_type
from .geopandas_tools.geometry_types import make_all_singlepart
from .geopandas_tools.geometry_types import to_single_geom_type
from .geopandas_tools.neighbors import get_all_distances
from .geopandas_tools.neighbors import get_k_nearest_neighbors
from .geopandas_tools.neighbors import get_neighbor_dfs
from .geopandas_tools.neighbors import get_neighbor_indices
from .geopandas_tools.neighbors import k_nearest_neighbors
from .geopandas_tools.neighbors import sjoin_within_distance
from .geopandas_tools.overlay import clean_overlay
from .geopandas_tools.point_operations import snap_all
from .geopandas_tools.point_operations import snap_within_distance
from .geopandas_tools.polygon_operations import clean_dissexp
from .geopandas_tools.polygon_operations import close_all_holes
from .geopandas_tools.polygon_operations import close_small_holes
from .geopandas_tools.polygon_operations import close_thin_holes
from .geopandas_tools.polygon_operations import eliminate_by_largest
from .geopandas_tools.polygon_operations import eliminate_by_longest
from .geopandas_tools.polygon_operations import eliminate_by_smallest
from .geopandas_tools.polygon_operations import get_cluster_mapper
from .geopandas_tools.polygon_operations import get_gaps
from .geopandas_tools.polygon_operations import get_holes
from .geopandas_tools.polygon_operations import get_polygon_clusters
from .geopandas_tools.polygon_operations import split_polygons_by_lines
from .geopandas_tools.polygons_as_rings import PolygonsAsRings
from .geopandas_tools.sfilter import sfilter
from .geopandas_tools.sfilter import sfilter_inverse
from .geopandas_tools.sfilter import sfilter_split
from .helpers import get_object_name
from .helpers import sort_nans_last
from .io.opener import opener
from .io.read_parquet import read_parquet_url
from .maps.examine import Examine
from .maps.explore import Explore
from .maps.httpserver import run_html_server
from .maps.legend import Legend
from .maps.maps import clipmap
from .maps.maps import explore
from .maps.maps import explore_locals
from .maps.maps import qtm
from .maps.maps import samplemap
from .maps.thematicmap import ThematicMap
from .maps.tilesources import kartverket as kartverket_tiles
from .maps.tilesources import xyz as xyztiles
from .networkanalysis.closing_network_holes import close_network_holes
from .networkanalysis.closing_network_holes import close_network_holes_to_deadends
from .networkanalysis.closing_network_holes import get_k_nearest_points_for_deadends
from .networkanalysis.cutting_lines import cut_lines
from .networkanalysis.cutting_lines import cut_lines_once
from .networkanalysis.cutting_lines import split_lines_by_nearest_point
from .networkanalysis.directednetwork import make_directed_network
from .networkanalysis.directednetwork import make_directed_network_norway
from .networkanalysis.finding_isolated_networks import get_component_size
from .networkanalysis.finding_isolated_networks import get_connected_components
from .networkanalysis.network import Network
from .networkanalysis.networkanalysis import NetworkAnalysis
from .networkanalysis.networkanalysisrules import NetworkAnalysisRules
from .networkanalysis.nodes import make_edge_coords_cols
from .networkanalysis.nodes import make_edge_wkt_cols
from .networkanalysis.nodes import make_node_ids
from .networkanalysis.traveling_salesman import traveling_salesman_problem
from .parallel.parallel import Parallel
from .parallel.parallel import parallel_overlay
from .parallel.parallel import parallel_overlay_rowwise
from .parallel.parallel import parallel_sjoin
from .raster.image_collection import Band
from .raster.image_collection import Image
from .raster.image_collection import ImageCollection
from .raster.image_collection import NDVIBand
from .raster.image_collection import Sentinel2Band
from .raster.image_collection import Sentinel2CloudlessBand
from .raster.image_collection import Sentinel2CloudlessCollection
from .raster.image_collection import Sentinel2CloudlessImage
from .raster.image_collection import Sentinel2Collection
from .raster.image_collection import Sentinel2Image
from .raster.image_collection import concat_image_collections

try:
    from .io.dapla_functions import check_files
    from .io.dapla_functions import get_bounds_series
    from .io.dapla_functions import read_geopandas
    from .io.dapla_functions import write_geopandas
except ImportError:
    pass

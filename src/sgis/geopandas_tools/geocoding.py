try:
    import geocoder
except ImportError:
    pass
from geopandas import GeoDataFrame

from .conversion import to_gdf


def address_to_gdf(address: str, crs=4326) -> GeoDataFrame:
    """Takes an address and returns a point GeoDataFrame."""
    g = geocoder.osm(address).json
    coords = g["lng"], g["lat"]
    return to_gdf(coords, crs=4326).to_crs(crs)


def address_to_coords(address: str, crs=4326) -> tuple[float, float]:
    """Takes an address and returns a tuple of xy coordinates."""
    g = geocoder.osm(address).json
    coords = g["lng"], g["lat"]
    point = to_gdf(coords, crs=4326).to_crs(crs)
    x, y = point.geometry.iloc[0].x, point.geometry.iloc[0].y
    return x, y

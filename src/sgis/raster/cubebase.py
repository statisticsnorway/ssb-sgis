from collections.abc import Callable
from pathlib import Path

from geopandas import GeoDataFrame

from .raster import Raster


def _from_gdf_func(gdf: GeoDataFrame, **kwargs) -> Raster:
    return Raster.from_gdf(gdf, **kwargs)


def _raster_from_path(path: str, **kwargs) -> Raster:
    return Raster.from_path(path, **kwargs)


def _method_as_func(self: Raster, method: str, **kwargs) -> Callable:
    return getattr(self, method)(**kwargs)


def _write_func(raster: Raster, folder: str, **kwargs):
    path = str(Path(folder) / Path(raster.name).stem) + ".tif"
    raster.write(path, **kwargs)
    raster.path = path
    return raster

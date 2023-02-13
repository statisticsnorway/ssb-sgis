from shapely import Geometry
from geopandas import GeoDataFrame, GeoSeries
import matplotlib.pyplot as plt
from .geopandas_utils import gdf_concat


def qtm(gdf, kolonne=None, *, scheme="Quantiles", title=None, size=12, fontsize=16, legend=True, **kwargs) -> None:
    """ Quick, thematic map (name stolen from R's tmap package). """
    fig, ax = plt.subplots(1, figsize=(size, size))
    ax.set_axis_off()
    ax.set_title(title, fontsize = fontsize)
    gdf.plot(kolonne, scheme=scheme, legend=legend, ax=ax, **kwargs)


def concat_explore(*gdfs: GeoDataFrame, cmap=None, **kwargs) -> None:
    for i, gdf in enumerate(gdfs):
        gdf["nr"] = i

    if not cmap:
        cmap = "viridis" if len(gdfs) < 6 else "rainbow"
        
    display(gdf_concat(gdfs).explore("nr", cmap=cmap, **kwargs))


def samplemap(
    gdf: GeoDataFrame, 
    explore: bool = True,
    size: int = 1000,
    **kwargs
    ) -> None:

    random_point = gdf.sample(1).assign(geometry=lambda x: x.centroid)
    
    clipped = gdf.clip(random_point.buffer(size))

    if explore:
        display(clipped.explore(**kwargs))
    else:
        qtm(clipped, **kwargs)


def clipmap(
    gdf: GeoDataFrame, 
    mask: GeoDataFrame | GeoSeries | Geometry,
    explore: bool = True,
    **kwargs
    ) -> None:

    clipped = gdf.clip(mask.to_crs(gdf.crs))

    if explore:
        display(clipped.explore(**kwargs))
    else:
        qtm(clipped, **kwargs)

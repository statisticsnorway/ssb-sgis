import matplotlib.pyplot as plt
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from matplotlib.colors import LinearSegmentedColormap
from shapely import Geometry

from .geopandas_utils import gdf_concat


def qtm(
    gdf,
    column=None,
    *,
    scheme="Quantiles",
    title=None,
    size=12,
    fontsize=16,
    legend=True,
    **kwargs,
) -> None:
    """Quick, thematic map (name stolen from R's tmap package)."""
    fig, ax = plt.subplots(1, figsize=(size, size))
    ax.set_axis_off()
    ax.set_title(title, fontsize=fontsize)
    gdf.plot(column, scheme=scheme, legend=legend, ax=ax, **kwargs)


def concat_explore(*gdfs: GeoDataFrame, cmap=None, **kwargs) -> None:
    for i, gdf in enumerate(gdfs):
        gdf["nr"] = i

    if not cmap:
        cmap = "viridis" if len(gdfs) < 6 else "rainbow"

    display(gdf_concat(gdfs).explore("nr", cmap=cmap, **kwargs))


def samplemap(
    gdf: GeoDataFrame, explore: bool = True, size: int = 1000, **kwargs
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
    *args,
    **kwargs,
) -> None:
    clipped = gdf.clip(mask.to_crs(gdf.crs))

    if explore:
        display(clipped.explore(*args, **kwargs))
    else:
        qtm(clipped, *args, **kwargs)


def chop_cmap(cmap: LinearSegmentedColormap, frac: float) -> LinearSegmentedColormap:
    """Chops off the beginning `frac` fraction of a colormap.
    https://stackoverflow.com/questions/7574748/setting-range-of-a-colormap-in-matplotlib
    """
    cmap = plt.get_cmap(cmap)
    cmap_as_array = cmap(np.arange(256))
    cmap_as_array = cmap_as_array[int(frac * len(cmap_as_array)) :]
    return LinearSegmentedColormap.from_list(cmap.name + f"_frac{frac}", cmap_as_array)

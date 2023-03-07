import matplotlib.pyplot as plt
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from matplotlib.colors import LinearSegmentedColormap
from pandas.api.types import is_numeric_dtype
from shapely import Geometry

from .explore import Explore
from .geopandas_utils import gdf_concat


def explore(
    *gdfs,
    labels: tuple[str] | None = None,
    popup: bool = True,
    **kwargs,
):
    """Interactive map of one or more GeoDataFrames with different colors for each gdf

    It takes all the GeoDataFrames specified and displays them together in an
    interactive map (the explore method), with a different color for each GeoDataFrame.
    The legend values is numbered in the order of the GeoDataFrames, if not column is
    specified.

    Uses the 'viridis' cmap if less than 6 GeoDataFrames, and a rainbow palette otherwise.

    Args:
        *gdfs: one or more GeoDataFrames separated by a comma in the function call,
            with no keyword. If the last arg specified is a string, it will be used
            used as the 'column' parameter if this is not specified.
         **kwargs: Keyword arguments to pass to geopandas.GeoDataFrame.explore

    Returns:
        Displays the interactive map, but returns nothing.
    """
    m = Explore(*gdfs, labels=labels, popup=popup, **kwargs)
    m.explore()


def samplemap(
    *gdfs,
    size: int = 500,
    labels: list[str] | None = None,
    popup: bool = True,
    **kwargs,
):
    """Takes a random sample of a GeoDataFrame and plots all data within a 1 km radius.

    The radius to plot can be changed with the 'size' parameter. By default, tries to
    display interactive map, but falls back to static if not in Jupyter. Can be
    changed to static by setting 'explore' to False. This will run the function 'qtm'.

    Args:
        gdf: the GeoDataFrame to plot
        column: The column to color the map by
        size: the radius to buffer the sample point by before clipping with the data
        explore: If True (the default), it tries to display an interactive map.
            If it raises a NameError because 'display' is not defined, it tries a
            static plot. If False, uses the 'qtm' function to show a static map
        **kwargs: keyword arguments taken by the geopandas' explore method or
            the 'qtm' method if this library.

    Returns:
        Displays the map, but returns nothing.
    """
    try:
        display
    except NameError:
        for gdf in gdfs:
            if not isinstance(gdf, GeoDataFrame):
                random_point = gdf.sample(1).assign(geometry=lambda x: x.centroid)
                clipped = gdf.clip(random_point.buffer(size))
                qtm(clipped, **kwargs)
        return

    m = Explore(*gdfs, labels=labels, popup=popup, **kwargs)
    m.samplemap(size)


def clipmap(
    gdf,
    mask,
    labels: list[str] | None = None,
    popup: bool = True,
    **kwargs,
):
    clipped = gdf.clip(mask)

    try:
        display
    except NameError:
        qtm(clipped, **kwargs)
        return

    m = Explore(clipped, labels=labels, popup=popup, **kwargs)
    m.explore()


def qtm(
    gdf: GeoDataFrame,
    column: str | None = None,
    *,
    title: str | None = None,
    scheme: str = "quantiles",
    legend: bool = True,
    black: bool = True,
    size: int = 10,
    fontsize: int = 15,
    **kwargs,
) -> None:
    """Quick, thematic map (name stolen from the tmap package in R).

    Like geopandas' plot method, with some different default parameter values:
    - includes legend by default
    - no axis labels
    - a bit larger
    - quantiles scheme as default if numeric column
    - can include a title with the title parameter
    - black background color to go with the default 'viridis' cmap, and to make
    geometries more visible

    Args:
        gdf: The GeoDataFrame to plot
        column: The column to color the map by
        Title: string to use as title
        scheme: how to group the column values. Defaults to 'quantiles' if numeric
            column
        legend: whether to include a legend explaining the colors and their values
        black: if True (the default), the background color will be black and the title
            white. If False, it will be the other way around.
        size: size of the plot. Defaults to 10
        fontsize: size of the title.
        **kwargs: additional keyword arguments taken by the geopandas plot method.

    Returns:
        displays the map, but nothing is returned.

    """
    if black:
        facecolor = "#0f0f0f"
        title_color = "#f7f7f7"
    else:
        facecolor = "#f7f7f7"
        title_color = "#0f0f0f"

    if column:
        if not is_numeric_dtype(gdf[column]):
            scheme = None
    fig, ax = plt.subplots(1, figsize=(size, size))
    fig.patch.set_facecolor(facecolor)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=fontsize, color=title_color)
    gdf.plot(column, scheme=scheme, legend=legend, ax=ax, **kwargs)


def _chop_cmap(cmap: LinearSegmentedColormap, frac: float) -> LinearSegmentedColormap:
    """Removes the given share of a cmap
    https://stackoverflow.com/questions/7574748/setting-range-of-a-colormap-in-matplotlib
    """
    cmap = plt.get_cmap(cmap)
    cmap_as_array = cmap(np.arange(256))
    cmap_as_array = cmap_as_array[int(frac * len(cmap_as_array)) :]
    return LinearSegmentedColormap.from_list(cmap.name + f"_frac{frac}", cmap_as_array)

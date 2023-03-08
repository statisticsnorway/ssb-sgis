"""Functions for interactive and static mapping of multiple GeoDataFrames.

This module builds on the geopandas explore and plot methods. The explore function
displays one of more GeoDataFrames together in an interactive map with layers that can
be toggled on and off. The samplemap function does the same, but shows only a random
sample area of the data. Size can be specified. The clipmap has the same functionality,
but clips the geometries to a mask extent.

In addition, there is the qtm function, which shows a static map with some additional
functionality compared to the geopandas plot method.

The three interactive map functions build on the Explore class, which has the same
functionality, but the parameters are stored in the class, making it easier and faster
to display many maps.
"""
import matplotlib.pyplot as plt
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from matplotlib.colors import LinearSegmentedColormap
from pandas.api.types import is_numeric_dtype
from shapely import Geometry

from .explore import Explore, _separate_args
from .geopandas_utils import gdf_concat


def explore(
    *gdfs: GeoDataFrame,
    column: str | None = None,
    labels: tuple[str] | None = None,
    popup: bool = True,
    max_zoom: int = 30,
    **kwargs,
) -> None:
    """Interactive map of GeoDataFrames with layers that can be toggles on/off.

    It takes all the given GeoDataFrames and displays them together in an
    interactive map with a common legend. The layers can be toggled on and off.

    If 'column' is not specified, each GeoDataFrame is given a unique color. The
    default colormap is a custom, strongly colored palette. If a numerical column
    is given, the 'viridis' palette is used.

    If the GeoDataFrames have name attributes (hint: write 'gdf.name = "gdf"'), these
    will be used as labels. Alternatively, labels can be specified as a tuple of
    strings of same length as the number of GeoDataFrames. Otherwise, the labels will
    be 0, 1, …, n - 1.

    Args:
        *gdfs: one or more GeoDataFrames. Separated by a comma in the function call,
            with no keyword.
        column: The column to color the geometries by. Defaults to None, which means
            each GeoDataFrame will get a unique color.
        labels: Names that will be shown in the toggle on/off menu. Defaults to None,
            meaning the GeoDataFrames will be labeled 0, 1, …, n - 1, unless the
            GeoDataFrames have a 'name' attribute. In this case, the name will be used
            as label.
        popup: If True (default), clicking on a geometry will create a popup box with
            column names and values for the given geometry. The box stays until
            clicking elsewhere. If False (the geopandas default), the box will only
            show when hovering over the geometry.
        max_zoom: The maximum allowed level of zoom. Higher number means more zoom
            allowed. Defaults to 30, which is higher than the geopandas default.
        **kwargs: Keyword arguments to pass to geopandas.GeoDataFrame.explore, for
            instance 'cmap' to change the colors, 'scheme' to change how the data
            is grouped. This defaults to 'quantiles' for numeric data.

    Returns:
        Displays the interactive map, but returns nothing.

    See also:
        samplemap: same functionality, but shows only a random area of a given size.
        clipmap: same functionality, but shows only the areas clipped by a given mask.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from gis_utils import roadpath, pointpath
    >>> points = gpd.read_parquet(pointpath)
    >>> roads = gpd.read_parquet(roadpath)

    Simple explore of two GeoDataFrames.

    >>> from gis_utils import explore
    >>> explore(roads, points)

    With column and labels.

    >>> roads["meters"] = roads.length
    >>> points["meters"] = points.length
    >>> explore(roads, points, column="meters", labels=("roads", "points"))

    Alternatively, with names as labels.

    >>> roads.name = "roads"
    >>> points.name = "points"
    >>> explore(roads, points, column="meters")
    """
    kwargs: dict = kwargs | {"popup": popup, "column": column, "max_zoom": max_zoom}

    m = Explore(*gdfs, labels=labels, **kwargs)
    m.explore()


def samplemap(
    *gdfs: GeoDataFrame,
    size: int = 500,
    column: str | None = None,
    labels: tuple[str] | None = None,
    popup: bool = True,
    max_zoom: int = 30,
    explore: bool = True,
    **kwargs,
) -> None:
    """Shows an interactive map of a random area of GeoDataFrames.

    It takes all the GeoDataFrames specified, takes a random sample point, and shows
    all geometries within a given radius of this point. Displays an interactive map
     with a common legend. The layers can be toggled on and off.

    The radius to plot can be changed with the 'size' parameter.

    By default, tries to display interactive map, but falls back to static if not in
    Jupyter. Can be changed to static by setting 'explore' to False. This will run the
    function 'qtm'.

    For more info about the labeling and coloring of the map, see the explore function.

    Args:
        *gdfs: one or more GeoDataFrames. Separated by a comma in the function call,
            with no keyword.
        size: the radius to buffer the sample point by before clipping with the data
        column: The column to color the geometries by. Defaults to None, which means
            each GeoDataFrame will get a unique color.
        labels: Names that will be shown in the toggle on/off menu. Defaults to None,
            meaning the GeoDataFrames will be labeled 0, 1, …, n - 1, unless the
            GeoDataFrames have a 'name' attribute. In this case, the name will be used
            as label.
        popup: If True (default), clicking on a geometry will create a popup box with
            column names and values for the given geometry. The box stays until
            clicking elsewhere. If False (the geopandas default), the box will only
            show when hovering over the geometry.
        max_zoom: The maximum allowed level of zoom. Higher number means more zoom
            allowed. Defaults to 30, which is higher than the geopandas default.
        explore: If True (the default), an interactive map will be displayed. If False,
            or not in Jupyter, a static plot will be shown.
        **kwargs: Keyword arguments to pass to geopandas.GeoDataFrame.explore, for
            instance 'cmap' to change the colors, 'scheme' to change how the data
            is grouped. This defaults to 'quantiles' for numeric data.

    Returns:
        Displays the interactive map, but returns nothing.

    See also:
        explore: same functionality, but shows the entire area of the geometries.
        clipmap: same functionality, but shows only the areas clipped by a given mask.
    """
    kwargs: dict = kwargs | {"popup": popup, "column": column, "max_zoom": max_zoom}

    try:
        display
    except NameError:
        explore = False

    if not size and isinstance(gdfs[-1], (float, int)):
        *gdfs, size = gdfs

    gdfs, labels, kwargs = _separate_args(gdfs, labels, kwargs)

    if explore:
        m = Explore(*gdfs, labels=labels, **kwargs)
        m.samplemap(size)
    else:
        for gdf in gdfs:
            if not isinstance(gdf, GeoDataFrame):
                random_point = (
                    gdf.sample(1).assign(geometry=lambda x: x.centroid).buffer(size)
                )
                clipped_ = ()
                for i, gdf in enumerate(gdfs):
                    clipped_ = gdf.clip(random_point)
                    clipped_["label"] = str(i)
                    clipped = clipped + (clipped_,)
                if "column" not in kwargs:
                    kwargs["column"] = "label"
                qtm(
                    gdf_concat(clipped),
                    **{key: value for key, value in kwargs.items() if key != "popup"},
                )


def clipmap(
    *gdfs: GeoDataFrame,
    mask: GeoDataFrame | GeoSeries | Geometry | None = None,
    column: str | None = None,
    labels: tuple[str] | None = None,
    popup: bool = True,
    explore: bool = True,
    max_zoom: int = 30,
    **kwargs,
) -> None:
    """Shows an interactive map of a of GeoDataFrames clipped to the mask extent.

    It takes all the GeoDataFrames specified, clips them to the extent of the mask,
    and displays the resulting geometries in an interactive map with a common legends.
    The layers can be toggled on and off.

    For more info about the labeling and coloring of the map, see the explore function.

    Args:
        *gdfs: one or more GeoDataFrames. Separated by a comma in the function call,
            with no keyword.
        mask: the geometry to clip the data by.
        column: The column to color the geometries by. Defaults to None, which means
            each GeoDataFrame will get a unique color.
        labels: Names that will be shown in the toggle on/off menu. Defaults to None,
            meaning the GeoDataFrames will be labeled 0, 1, …, n - 1, unless the
            GeoDataFrames have a 'name' attribute. In this case, the name will be used
            as label.
        popup: If True (default), clicking on a geometry will create a popup box with
            column names and values for the given geometry. The box stays until
            clicking elsewhere. If False (the geopandas default), the box will only
            show when hovering over the geometry.
        max_zoom: The maximum allowed level of zoom. Higher number means more zoom
            allowed. Defaults to 30, which is higher than the geopandas default.
        explore: If True (the default), an interactive map will be displayed. If False,
            or not in Jupyter, a static plot will be shown.
        **kwargs: Keyword arguments to pass to geopandas.GeoDataFrame.explore, for
            instance 'cmap' to change the colors, 'scheme' to change how the data
            is grouped. This defaults to 'quantiles' for numeric data.

    Returns:
        Displays the interactive map, but returns nothing.

    See also:
        explore: same functionality, but shows the entire area of the geometries.
        samplemap: same functionality, but shows only a random area of a given size.
    """
    kwargs: dict = kwargs | {"popup": popup, "column": column, "max_zoom": max_zoom}

    try:
        display
    except NameError:
        explore = False

    clipped = ()

    gdfs, labels, kwargs = _separate_args(gdfs, labels, kwargs)

    if mask:
        for gdf in gdfs:
            clipped_ = gdf.clip(mask)
            clipped = clipped + (clipped_,)

    else:
        for gdf in gdfs[:-1]:
            clipped_ = gdf.clip(gdfs[-1])
            clipped = clipped + (clipped_,)

    if explore:
        m = Explore(*clipped, labels=labels, **kwargs)
        m.explore()
    else:
        qtm(
            gdf_concat(clipped),
            **{key: value for key, value in kwargs.items() if key != "popup"},
        )


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

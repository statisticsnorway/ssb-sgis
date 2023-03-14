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
from geopandas import GeoDataFrame, GeoSeries
from pandas.api.types import is_numeric_dtype
from shapely import Geometry

from .exceptions import NotInJupyterError
from .explore import Explore, _separate_args
from .geopandas_utils import gdf_concat
from .helpers import make_namedict


def _check_if_jupyter_is_needed(explore, show_in_browser):
    if explore and not show_in_browser:
        try:
            display
        except NameError as e:
            raise NotInJupyterError(
                "Cannot display interactive map. Try setting "
                "'show_in_browser' to True, or 'explore' to False."
            ) from e


def explore(
    *gdfs: GeoDataFrame,
    column: str | None = None,
    labels: tuple[str] | None = None,
    popup: bool = True,
    max_zoom: int = 30,
    show_in_browser: bool = False,
    **kwargs,
) -> None:
    """Interactive map of GeoDataFrames with layers that can be toggles on/off.

    It takes all the given GeoDataFrames and displays them together in an
    interactive map with a common legend. The layers can be toggled on and off.

    If 'column' is not specified, each GeoDataFrame is given a unique color. The
    default colormap is a custom, strongly colored palette. If a numerical column
    is given, the 'viridis' palette is used.

    Note:
        The maximum zoom level only works on the OpenStreetMap background map.

    Args:
        *gdfs: one or more GeoDataFrames. Separated by a comma in the function call,
            with no keyword.
        column: The column to color the geometries by. Defaults to None, which means
            each GeoDataFrame will get a unique color.
        labels: By default, the GeoDataFrames will be labeled by their object names.
            Alternatively, labels can be specified as a tuple of strings the same
            length as the number of gdfs.
        popup: If True (default), clicking on a geometry will create a popup box with
            column names and values for the given geometry. The box stays until
            clicking elsewhere. If False (the geopandas default), the box will only
            show when hovering over the geometry.
        max_zoom: The maximum allowed level of zoom. Higher number means more zoom
            allowed. Defaults to 30, which is higher than the geopandas default.
        show_in_browser: If False (the default), the maps will be shown in Jupyter.
            If True the maps will be opened in a browser folder.
        **kwargs: Keyword arguments to pass to geopandas.GeoDataFrame.explore, for
            instance 'cmap' to change the colors, 'scheme' to change how the data
            is grouped. This defaults to 'quantiles' for numeric data.

    See also:
        samplemap: same functionality, but shows only a random area of a given size.
        clipmap: same functionality, but shows only the areas clipped by a given mask.

    Examples
    --------
    >>> from sgis import read_parquet_url
    >>> roads = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")
    >>> points = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet")

    Simple explore of two GeoDataFrames.

    >>> from sgis import explore
    >>> explore(roads, points)

    With additional arguments.

    >>> roads["meters"] = roads.length
    >>> points["meters"] = points.length
    >>> explore(roads, points, column="meters", cmap="plasma", max_zoom=60)
    """
    kwargs: dict = kwargs | {"popup": popup, "column": column, "max_zoom": max_zoom}

    m = Explore(*gdfs, labels=labels, show_in_browser=show_in_browser, **kwargs)
    m.explore()


def samplemap(
    *gdfs: GeoDataFrame,
    size: int = 500,
    column: str | None = None,
    labels: tuple[str] | None = None,
    popup: bool = True,
    max_zoom: int = 30,
    explore: bool = True,
    show_in_browser: bool = False,
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

    Note:
        The maximum zoom level only works on the OpenStreetMap background map.

    Args:
        *gdfs: one or more GeoDataFrames. Separated by a comma in the function call,
            with no keyword.
        size: the radius to buffer the sample point by before clipping with the data
        column: The column to color the geometries by. Defaults to None, which means
            each GeoDataFrame will get a unique color.
        labels: By default, the GeoDataFrames will be labeled by their object names.
            Alternatively, labels can be specified as a tuple of strings the same
            length as the number of gdfs.
        popup: If True (default), clicking on a geometry will create a popup box with
            column names and values for the given geometry. The box stays until
            clicking elsewhere. If False (the geopandas default), the box will only
            show when hovering over the geometry.
        max_zoom: The maximum allowed level of zoom. Higher number means more zoom
            allowed. Defaults to 30, which is higher than the geopandas default.
        explore: If True (the default), an interactive map will be displayed. If False,
            or not in Jupyter, a static plot will be shown.
        show_in_browser: If False (the default), the maps will be shown in Jupyter.
            If True the maps will be opened in a browser folder.
        **kwargs: Keyword arguments to pass to geopandas.GeoDataFrame.explore, for
            instance 'cmap' to change the colors, 'scheme' to change how the data
            is grouped. This defaults to 'quantiles' for numeric data.

    See also:
        explore: same functionality, but shows the entire area of the geometries.
        clipmap: same functionality, but shows only the areas clipped by a given mask.
    """
    kwargs: dict = kwargs | {"popup": popup, "column": column, "max_zoom": max_zoom}

    _check_if_jupyter_is_needed(explore, show_in_browser)

    if not size and isinstance(gdfs[-1], (float, int)):
        *gdfs, size = gdfs

    gdfs, column, kwargs = _separate_args(gdfs, column, kwargs)

    if explore:
        m = Explore(*gdfs, labels=labels, show_in_browser=show_in_browser, **kwargs)
        m.samplemap(size)
    else:
        for gdf in gdfs:
            if not isinstance(gdf, GeoDataFrame):
                random_point = (
                    gdf.sample(1).assign(geometry=lambda x: x.centroid).buffer(size)
                )
                namedict = make_namedict(gdfs)

                clipped: tuple[GeoDataFrame] = ()
                for i, gdf in enumerate(gdfs):
                    clipped_ = gdf.clip(random_point)
                    clipped_["label"] = namedict[i]
                    clipped = clipped + (clipped_,)
                if "column" not in kwargs:
                    kwargs["column"] = "label"
                qtm(
                    gdf_concat(clipped),
                    **{
                        key: value
                        for key, value in kwargs.items()
                        if key not in ["popup", "max_zoom"]
                    },
                )


def clipmap(
    *gdfs: GeoDataFrame,
    mask: GeoDataFrame | GeoSeries | Geometry | None = None,
    column: str | None = None,
    labels: tuple[str] | None = None,
    popup: bool = True,
    explore: bool = True,
    max_zoom: int = 30,
    show_in_browser: bool = False,
    **kwargs,
) -> None:
    """Shows an interactive map of a of GeoDataFrames clipped to the mask extent.

    It takes all the GeoDataFrames specified, clips them to the extent of the mask,
    and displays the resulting geometries in an interactive map with a common legends.
    The layers can be toggled on and off.

    For more info about the labeling and coloring of the map, see the explore function.

    Note:
        The maximum zoom level only works on the OpenStreetMap background map.

    Args:
        *gdfs: one or more GeoDataFrames. Separated by a comma in the function call,
            with no keyword.
        mask: the geometry to clip the data by.
        column: The column to color the geometries by. Defaults to None, which means
            each GeoDataFrame will get a unique color.
        labels: By default, the GeoDataFrames will be labeled by their object names.
            Alternatively, labels can be specified as a tuple of strings the same
            length as the number of gdfs.
        popup: If True (default), clicking on a geometry will create a popup box with
            column names and values for the given geometry. The box stays until
            clicking elsewhere. If False (the geopandas default), the box will only
            show when hovering over the geometry.
        max_zoom: The maximum allowed level of zoom. Higher number means more zoom
            allowed. Defaults to 30, which is higher than the geopandas default.
        explore: If True (the default), an interactive map will be displayed. If False,
            or not in Jupyter, a static plot will be shown.
        show_in_browser: If False (the default), the maps will be shown in Jupyter.
            If True the maps will be opened in a browser folder.
        **kwargs: Keyword arguments to pass to geopandas.GeoDataFrame.explore, for
            instance 'cmap' to change the colors, 'scheme' to change how the data
            is grouped. This defaults to 'quantiles' for numeric data.

    See also:
        explore: same functionality, but shows the entire area of the geometries.
        samplemap: same functionality, but shows only a random area of a given size.
    """
    kwargs: dict = kwargs | {"popup": popup, "column": column, "max_zoom": max_zoom}

    _check_if_jupyter_is_needed(explore, show_in_browser)

    clipped: tuple[GeoDataFrame] = ()

    gdfs, column, kwargs = _separate_args(gdfs, column, kwargs)

    # creating a dict of object names here, since the names disappear after clip
    if not labels:
        namedict = make_namedict(gdfs)
        kwargs["namedict"] = namedict

    if mask is not None:
        for gdf in gdfs:
            clipped_ = gdf.clip(mask)
            clipped = clipped + (clipped_,)

    else:
        for gdf in gdfs[:-1]:
            clipped_ = gdf.clip(gdfs[-1])
            clipped = clipped + (clipped_,)

    if explore:
        m = Explore(*clipped, labels=labels, show_in_browser=show_in_browser, **kwargs)
        m.explore()
    else:
        qtm(
            gdf_concat(clipped),
            **{
                key: value
                for key, value in kwargs.items()
                if key not in ["popup", "max_zoom"]
            },
        )


def qtm(
    gdf: GeoDataFrame,
    column: str | None = None,
    *,
    title: str | None = None,
    scheme: str | None = "quantiles",
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
        gdf: The GeoDataFrame to plot.
        column: The column to color the map by.
        title: Text to use as the map's heading.
        scheme: how to group the column values. Defaults to 'quantiles' if numeric
            column
        legend: whether to include a legend explaining the colors and their values
        black: if True (the default), the background color will be black and the title
            white. If False, it will be the other way around.
        size: size of the plot. Defaults to 10
        fontsize: size of the title.
        **kwargs: additional keyword arguments taken by the geopandas plot method.
    """
    if black:
        facecolor, title_color = "#0f0f0f", "#f7f7f7"
    else:
        facecolor, title_color = "#f7f7f7", "#0f0f0f"

    if column and not is_numeric_dtype(gdf[column]):
        scheme = None
    fig, ax = plt.subplots(1, figsize=(size, size))
    fig.patch.set_facecolor(facecolor)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=fontsize, color=title_color)
    gdf.plot(column, scheme=scheme, legend=legend, ax=ax, **kwargs)

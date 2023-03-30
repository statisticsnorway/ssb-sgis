"""Interactive and static mapping of multiple GeoDataFrames.

The main function is 'explore', which displays one of more GeoDataFrames together in an
interactive map with layers that can be toggled on and off. The 'samplemap' and
'clipmap' functions do the same, but displays a random and chosen area respectfully.

The 'qtm' function shows a static map of one or more GeoDataFrames.
"""
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame, GeoSeries
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pandas.api.types import is_numeric_dtype
from shapely import Geometry

from .exceptions import NotInJupyterError
from .explore import Explore, _separate_args
from .helpers import make_namedict, return_two_vals


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
        *gdfs: one or more GeoDataFrames.
        column: The column to color the geometries by. Defaults to None, which means
            each GeoDataFrame will get a unique color.
        labels: By default, the GeoDataFrames will be labeled by their object names.
            Alternatively, labels can be specified as a tuple of strings with the same
            length as the number of gdfs.
        max_zoom: The maximum allowed level of zoom. Higher number means more zoom
            allowed. Defaults to 30, which is higher than the geopandas default.
        show_in_browser: If False (default), the maps will be shown in Jupyter.
            If True the maps will be opened in a browser folder.
        **kwargs: Keyword arguments to pass to geopandas.GeoDataFrame.explore, for
            instance 'cmap' to change the colors, 'scheme' to change how the data
            is grouped. This defaults to 'fisherjenks' for numeric data.

    See also
    --------
    samplemap: same functionality, but shows only a random area of a given size.
    clipmap: same functionality, but shows only the areas clipped by a given mask.

    Examples
    --------
    >>> from sgis import read_parquet_url, explore
    >>> roads = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_eidskog_2022.parquet")
    >>> points = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_eidskog.parquet")

    Simple explore of two GeoDataFrames.

    >>> explore(roads, points)

    With additional arguments.

    >>> roads["meters"] = roads.length
    >>> points["meters"] = points.length
    >>> explore(roads, points, column="meters", cmap="plasma", max_zoom=60)
    """
    kwargs: dict = kwargs | {"column": column, "max_zoom": max_zoom}

    m = Explore(*gdfs, labels=labels, show_in_browser=show_in_browser, **kwargs)
    m.explore()


def samplemap(
    *gdfs: GeoDataFrame,
    column: str | None = None,
    size: int = 1500,
    sample_from_first: bool = True,
    labels: tuple[str] | None = None,
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
        *gdfs: one or more GeoDataFrames.
        column: The column to color the geometries by. Defaults to None, which means
            each GeoDataFrame will get a unique color.
        size: the radius to buffer the sample point by before clipping with the data.
            Defaults to 1500 (meters).
        sample_from_first: If True (default), the sample point is taken form the
            first specified GeoDataFrame. If False, all GeoDataFrames are considered.
        labels: By default, the GeoDataFrames will be labeled by their object names.
            Alternatively, labels can be specified as a tuple of strings the same
            length as the number of gdfs.
        max_zoom: The maximum allowed level of zoom. Higher number means more zoom
            allowed. Defaults to 30, which is higher than the geopandas default.
        explore: If True (default), an interactive map will be displayed. If False,
            or not in Jupyter, a static plot will be shown.
        show_in_browser: If False (default), the maps will be shown in Jupyter.
            If True the maps will be opened in a browser folder.
        **kwargs: Keyword arguments to pass to geopandas.GeoDataFrame.explore, for
            instance 'cmap' to change the colors, 'scheme' to change how the data
            is grouped. This defaults to 'fisherjenks' for numeric data.

    See also
    --------
    explore: same functionality, but shows the entire area of the geometries.
    clipmap: same functionality, but shows only the areas clipped by a given mask.

    Examples
    --------
    >>> from sgis import read_parquet_url, samplemap
    >>> roads = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_eidskog_2022.parquet")
    >>> points = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_eidskog.parquet")

    With default sample size. To get a new sample area, simply re-run the line.

    >>> samplemap(roads, points)

    Sample area with a radius of 5 kilometers.

    >>> samplemap(roads, points, size=5_000, column="meters")

    """
    kwargs: dict = kwargs | {"column": column, "max_zoom": max_zoom}

    _check_if_jupyter_is_needed(explore, show_in_browser)

    if not size and isinstance(gdfs[-1], (float, int)):
        *gdfs, size = gdfs

    gdfs, column, kwargs = _separate_args(gdfs, column, kwargs)

    m = Explore(*gdfs, labels=labels, show_in_browser=show_in_browser, **kwargs)

    if explore:
        m.samplemap(size, sample_from_first=sample_from_first)
    else:
        if sample_from_first:
            random_point = m.gdfs[0].sample(1).centroid.buffer(size)
        else:
            random_point = m.gdf.sample(1).centroid.buffer(size)
        m.gdf = m.gdf.clip(random_point)
        qtm(
            m.gdf,
            **{
                key: value
                for key, value in m.kwargs.items()
                if key not in ["popup", "max_zoom"]
            },
        )


def clipmap(
    *gdfs: GeoDataFrame,
    mask: GeoDataFrame | GeoSeries | Geometry,
    column: str | None = None,
    labels: tuple[str] | None = None,
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
        *gdfs: one or more GeoDataFrames.
        mask: the geometry to clip the data by.
        column: The column to color the geometries by. Defaults to None, which means
            each GeoDataFrame will get a unique color.
        labels: By default, the GeoDataFrames will be labeled by their object names.
            Alternatively, labels can be specified as a tuple of strings the same
            length as the number of gdfs.
        max_zoom: The maximum allowed level of zoom. Higher number means more zoom
            allowed. Defaults to 30, which is higher than the geopandas default.
        explore: If True (default), an interactive map will be displayed. If False,
            or not in Jupyter, a static plot will be shown.
        show_in_browser: If False (default), the maps will be shown in Jupyter.
            If True the maps will be opened in a browser folder.
        **kwargs: Keyword arguments to pass to geopandas.GeoDataFrame.explore, for
            instance 'cmap' to change the colors, 'scheme' to change how the data
            is grouped. This defaults to 'fisherjenks' for numeric data.

    See also
    --------
    explore: same functionality, but shows the entire area of the geometries.
    samplemap: same functionality, but shows only a random area of a given size.
    """

    kwargs: dict = kwargs | {"column": column, "max_zoom": max_zoom}

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

    m = Explore(*clipped, labels=labels, show_in_browser=show_in_browser, **kwargs)

    if explore:
        m.explore()
    else:
        qtm(
            m.gdf,
            **{
                key: value
                for key, value in m.kwargs.items()
                if key not in ["popup", "max_zoom"]
            },
        )


def qtm(
    *gdfs: GeoDataFrame,
    column: str | None = None,
    bg_gdf: GeoDataFrame | None = None,
    title: str | None = None,
    scheme: str | None = "fisherjenks",
    legend: bool = True,
    black: bool = True,
    size: int = 10,
    legend_title: str | None = None,
    bbox_to_anchor: tuple[float | int, float | int] = (1, 0.2),
    **kwargs,
) -> tuple[Figure, Axes]:
    """Quick, thematic map of one or more GeoDataFrames.

    Shows one or more GeoDataFrames in the same plot, with a common color scheme if
    column is specified, or with unique colors for each GeoDataFrame if not. The
    function simplifies the manual construction of a basic matplotlib plot. It also
    returns the matplotlib figure and axis, so the plot can be changed afterwards.

    Disclaimer: the 'qtm' name is taken from the tmap package in R.

    Args:
        *gdfs: One or more GeoDataFrames to plot.
        column: The column to color the map by. Defaults to None, meaning each
            GeoDataFrame is given a unique color.
        title: Text to use as the map's heading.
        scheme: How to group the column values. Defaults to 'fisherjenks' if numeric
            column.
        legend: Whether to include a legend explaining the colors and their values.
        black: If True (default), the background color will be black and the title
            white. If False, it will be the opposite.
        size: Size of the plot. Defaults to 10.
        title_fontsize: Size of the title.
        **kwargs: Additional keyword arguments passed to the geopandas plot method.

    Returns:
        The matplotlib figure and axis.
    """

    if black:
        facecolor, title_color, bg_gdf_color = "#0f0f0f", "#fefefe", "#383834"
    else:
        facecolor, title_color, bg_gdf_color = "#fefefe", "#0f0f0f", "#d1d1cd"

    # run the gdfs through the __init__ of the Explore class
    # this gives a custom categorical cmap and labels to be used in the legend
    kwargs["column"] = column
    gdfs, column, kwargs = _separate_args(gdfs, column, kwargs)
    m = Explore(*gdfs, **kwargs)

    # the 'column' in the kwargs has to be updated to the custom categorical column (if
    # column was not specified).
    kwargs["column"] = m.kwargs.pop("column")

    if m._is_categorical and any(
        kwarg in kwargs for kwarg in ("cmap", "color", "column")
    ):
        kwargs["color"] = m.gdf["color"]

    if column and not is_numeric_dtype(m.gdf[column]):
        scheme = None

    size_x, size_y = return_two_vals(size)
    fig, ax = _get_matplotlib_figure_and_axix(figsize=(size_x, size_y))

    fig.patch.set_facecolor(facecolor)
    ax.set_axis_off()

    # manually add legend if categorical, since geopandas.plot removes it otherwise.
    if legend and m._is_categorical:
        kwargs.pop("column")
        kwargs["color"] = m.gdf.color
        patches, categories = [], []
        for category, color in m._categories_colors_dict.items():
            categories.append(category)
            patches.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker="o",
                    alpha=kwargs.get("alpha", 1),
                    markersize=10,
                    markerfacecolor=color,
                    markeredgewidth=0,
                )
            )
        ax.legend(patches, categories, fontsize=size_x)

    # if single color
    elif legend and "color" in kwargs:
        m.gdf["color"] = kwargs["color"]
        kwargs.pop("column")
        categories = [kwargs["color"] if isinstance(kwargs["color"], str) else "color"]
        patches = [
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                alpha=kwargs.get("alpha", 1),
                markersize=10,
                markerfacecolor=kwargs["color"],
                markeredgewidth=0,
            )
        ]
        ax.legend(patches, categories, fontsize=size_x)

    if not m._is_categorical:
        # making unique bins manually so that all equal values get same color
        unique_bins = m._create_bins(m.gdf, column, scheme)
        kwargs["classification_kwds"] = {"bins": unique_bins}
        if len(unique_bins) < kwargs.get("k", 5):
            kwargs["k"] = len(unique_bins)

    if title:
        ax.set_title(title, fontsize=size_x * 2, color=title_color)

    if bg_gdf is not None:
        minx, miny, maxx, maxy = m.gdf.total_bounds
        diffx = maxx - minx
        diffy = maxy - miny
        ax.set_xlim([minx - diffx * 0.03, maxx + diffx * 0.03])
        ax.set_ylim([miny - diffy * 0.03, maxy + diffy * 0.03])
        bg_gdf.plot(ax=ax, color=bg_gdf_color)

    if "color" in kwargs:
        kwargs.pop("column", None)

    m.gdf.plot(scheme=scheme, legend=legend, ax=ax, **kwargs)

    if not legend:
        return fig, ax

    the_legend = plt.gca().get_legend()

    for text in the_legend.get_texts():
        text.set_fontsize(size_x)

    the_legend.set_bbox_to_anchor(bbox_to_anchor)

    legend_title = column if not legend_title else legend_title
    the_legend.set_title(legend_title, prop={"size": size_x * 1.25})

    plt.show()

    return fig, ax


def _get_matplotlib_figure_and_axix(figsize):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    return fig, ax

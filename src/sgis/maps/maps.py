"""Interactive and static mapping of multiple GeoDataFrames.

The main function is 'explore', which displays one of more GeoDataFrames together in an
interactive map with layers that can be toggled on and off. The 'samplemap' and
'clipmap' functions do the same, but displays a random and chosen area respectfully.

The 'qtm' function shows a static map of one or more GeoDataFrames.
"""
from geopandas import GeoDataFrame, GeoSeries
from shapely import Geometry

from ..exceptions import NotInJupyterError
from ..geopandas_tools.general import random_points_in_polygons
from ..geopandas_tools.geometry_types import get_geom_type
from ..helpers import make_namedict
from .explore import Explore
from .map import Map
from .thematicmap import ThematicMap


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
    """Interactive map of GeoDataFrames with layers that can be toggled on/off.

    It takes all the given GeoDataFrames and displays them together in an
    interactive map with a common legend. If 'column' is not specified, each
    GeoDataFrame is given a unique color.

    If the column is of type string and only one GeoDataFrame is given, the unique
    values will be split into separate GeoDataFrames so that each value can be toggled
    on/off.

    The coloring can be changed with the 'cmap' parameter. The default colormap is a
    custom, strongly colored palette. If a numerical colum is given, the 'viridis'
    palette is used.

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
            is grouped. This defaults to 'fisherjenkssampled' for numeric data.

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
    m = Explore(
        *gdfs,
        column=column,
        labels=labels,
        show_in_browser=show_in_browser,
        max_zoom=max_zoom,
        **kwargs,
    )
    m.explore()


def samplemap(
    *gdfs: GeoDataFrame,
    column: str | None = None,
    size: int = 1000,
    sample_from_first: bool = True,
    labels: tuple[str] | None = None,
    max_zoom: int = 30,
    explore: bool = True,
    show_in_browser: bool = False,
    **kwargs,
) -> None:
    """Shows an interactive map of a random area of GeoDataFrames.

    It takes all the GeoDataFrames specified, takes a random sample point from the
    first, and shows all geometries within a given radius of this point. Otherwise
    works like the explore function.

    Note:
        The maximum zoom level only works on the OpenStreetMap background map.

    Args:
        *gdfs: one or more GeoDataFrames.
        column: The column to color the geometries by. Defaults to None, which means
            each GeoDataFrame will get a unique color.
        size: the radius to buffer the sample point by before clipping with the data.
            Defaults to 1000 (meters).
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
            is grouped. This defaults to 'fisherjenkssampled' for numeric data.

    See also
    --------
    explore: Same functionality, but shows the entire area of the geometries.
    clipmap: Same functionality, but shows only the areas clipped by a given mask.

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
    _check_if_jupyter_is_needed(explore, show_in_browser)

    if not size and isinstance(gdfs[-1], (float, int)):
        *gdfs, size = gdfs

    if explore:
        m = Explore(
            *gdfs,
            column=column,
            labels=labels,
            show_in_browser=show_in_browser,
            max_zoom=max_zoom,
            **kwargs,
        )
        m.samplemap(size, sample_from_first=sample_from_first)
    else:
        m = Map(
            *gdfs,
            column=column,
            labels=labels,
            **kwargs,
        )

        if sample_from_first:
            sample = m._gdfs[0].sample(1)
        else:
            sample = m._gdf.sample(1)

        # convert lines to polygons
        if get_geom_type(sample) == "line":
            sample["geometry"] = sample.buffer(1)

        if get_geom_type(sample) == "polygon":
            random_point = random_points_in_polygons(sample, 1)

        # if point or mixed geometries
        else:
            random_point = sample.centroid.buffer

        m._gdf = m._gdf.clip(random_point.buffer(size))

        qtm(m._gdf, column=m.column, cmap=m._cmap, k=m.k)


def clipmap(
    *gdfs: GeoDataFrame,
    column: str | None = None,
    mask: GeoDataFrame | GeoSeries | Geometry,
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
            is grouped. This defaults to 'fisherjenkssampled' for numeric data.

    See also
    --------
    explore: same functionality, but shows the entire area of the geometries.
    samplemap: same functionality, but shows only a random area of a given size.
    """

    _check_if_jupyter_is_needed(explore, show_in_browser)

    gdfs, column = Explore._separate_args(gdfs, column)

    # storing object names in dict here, since the names disappear after clip
    if not labels:
        namedict = make_namedict(gdfs)
        kwargs["namedict"] = namedict

    clipped: tuple[GeoDataFrame] = ()

    if mask is not None:
        for gdf in gdfs:
            clipped_ = gdf.clip(mask)
            clipped = clipped + (clipped_,)

    else:
        for gdf in gdfs[:-1]:
            clipped_ = gdf.clip(gdfs[-1])
            clipped = clipped + (clipped_,)

    if not any(len(gdf) for gdf in clipped):
        raise ValueError("None of the GeoDataFrames are within the mask extent.")

    if explore:
        m = Explore(
            *clipped,
            column=column,
            labels=labels,
            show_in_browser=show_in_browser,
            max_zoom=max_zoom,
            **kwargs,
        )
        m.explore()
    else:
        m = Map(
            *gdfs,
            column=column,
            labels=labels,
            **kwargs,
        )
        qtm(m._gdf, column=m.column, cmap=m._cmap, k=m.k)


def qtm(
    *gdfs: GeoDataFrame,
    column: str | None = None,
    title: str | None = None,
    black: bool = True,
    size: int = 10,
    legend: bool = True,
    cmap: str | None = None,
    k: int = 5,
    **kwargs,
) -> None:
    """Quick, thematic map of one or more GeoDataFrames.

    Shows one or more GeoDataFrames in the same plot, with a common color scheme if
    column is specified, otherwise with unique colors for each GeoDataFrame.

    The 'qtm' name is taken from the tmap package in R.

    Args:
        *gdfs: One or more GeoDataFrames to plot.
        column: The column to color the map by. Defaults to None, meaning each
            GeoDataFrame is given a unique color.
        title: Text to use as the map's heading.
        black: If True (default), the background color will be black and the title
            white. If False, it will be the opposite. The colormap will also be
            'viridis' when black, and 'RdPu' when white.
        size: Size of the plot. Defaults to 10.
        title_fontsize: Size of the title.
        cmap: Color palette of the map. See:
            https://matplotlib.org/stable/tutorials/colors/colormaps.html
        k: Number of color groups.
        **kwargs: Additional keyword arguments taken by the geopandas plot method.

    See also:
        ThematicMap: Class with more options for customising the plot.
    """

    m = ThematicMap(*gdfs, column=column, size=size, black=black)

    m.title = title

    if k and len(m._unique_values) >= k:
        m.k = k

    if cmap:
        m.cmap = cmap

    if not legend:
        m.legend = None

    m.plot(**kwargs)

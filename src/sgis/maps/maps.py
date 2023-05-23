"""Interactive and static mapping of multiple GeoDataFrames.

The main function is 'explore', which displays one of more GeoDataFrames together in an
interactive map with layers that can be toggled on and off. The 'samplemap' and
'clipmap' functions do the same, but displays a random and chosen area respectfully.

The 'qtm' function shows a simple static map of one or more GeoDataFrames.
"""
from numbers import Number

from geopandas import GeoDataFrame, GeoSeries
from shapely import Geometry

from ..geopandas_tools.general import clean_clip, random_points_in_polygons, to_gdf
from ..geopandas_tools.geometry_types import get_geom_type
from ..helpers import make_namedict
from .explore import Explore
from .map import Map
from .thematicmap import ThematicMap


def _get_mask(kwargs: dict, crs) -> tuple[GeoDataFrame | None, dict]:
    masks = {
        "bygdoy": (10.6976899, 59.9081695),
        "kongsvinger": (12.0035242, 60.1875279),
        "stavanger": (5.6960601, 58.8946196),
    }

    if "size" in kwargs and kwargs["size"] is not None:
        size = kwargs["size"]
    else:
        size = 1000

    for key, value in kwargs.items():
        if key.lower() in masks:
            mask = masks[key]
            kwargs.pop(key)
            if isinstance(value, Number) and value > 1:
                size = value
            the_mask = to_gdf([mask], crs=4326).to_crs(crs).buffer(size)
            return the_mask, kwargs

    return None, kwargs


def explore(
    *gdfs: GeoDataFrame,
    column: str | None = None,
    labels: tuple[str] | None = None,
    max_zoom: int = 30,
    browser: bool = False,
    smooth_factor: int | float = 1.5,
    center: tuple[float, float] | None = None,
    size: int | None = None,
    **kwargs,
) -> None:
    """Interactive map of GeoDataFrames with layers that can be toggled on/off.

    It takes all the given GeoDataFrames and displays them together in an
    interactive map with a common legend. If 'column' is not specified, each
    GeoDataFrame is given a unique color.

    If the column is of type string and only one GeoDataFrame is given, the unique
    values will be split into separate GeoDataFrames so that each value can be toggled
    on/off.

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
        browser: If False (default), the maps will be shown in Jupyter.
            If True the maps will be opened in a browser folder.
        smooth_factor: How much to simplify the geometries. 1 is the minimum,
            5 is quite a lot of simplification.
        center: Optional coordinate pair (x, y) to use as centerpoint for the map.
            The geometries will then be clipped by a buffered circle around this point.
            If 'size' is not given, 1000 will be used as the buffer distance.
        size: The buffer distance. Only applies when center is specified. Defaults to
            1000 if center is given.
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
    mask, kwargs = _get_mask(kwargs | {"size": size}, crs=gdfs[0].crs)

    kwargs.pop("size", None)

    if mask is not None:
        return clipmap(
            *gdfs,
            column=column,
            mask=mask,
            labels=labels,
            browser=browser,
            max_zoom=max_zoom,
            **kwargs,
        )

    if center is not None:
        size = size if size else 1000
        if not isinstance(center, GeoDataFrame):
            mask = to_gdf(center, crs=gdfs[0].crs).buffer(size)
        elif get_geom_type(center) == "point":
            mask = center.buffer(size)

        return clipmap(
            *gdfs,
            column=column,
            mask=mask,
            labels=labels,
            browser=browser,
            max_zoom=max_zoom,
            **kwargs,
        )

    m = Explore(
        *gdfs,
        column=column,
        labels=labels,
        browser=browser,
        max_zoom=max_zoom,
        smooth_factor=smooth_factor,
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
    smooth_factor: int = 1.5,
    explore: bool = True,
    browser: bool = False,
    **kwargs,
) -> None:
    """Shows an interactive map of a random area of GeoDataFrames.

    It takes all the GeoDataFrames specified, takes a random sample point from the
    first, and shows all geometries within a given radius of this point. Otherwise
    works like the explore function.

    To re-use the sample area, use the line that is printed in this function,
    containing the size and centerpoint. This line can be copypasted directly
    into the explore or clipmap functions.

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
        smooth_factor: How much to simplify the geometries. 1 is the minimum,
            5 is quite a lot of simplification.
        explore: If True (default), an interactive map will be displayed. If False,
            or not in Jupyter, a static plot will be shown.
        browser: If False (default), the maps will be shown in Jupyter.
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

    if not size and isinstance(gdfs[-1], (float, int)):
        *gdfs, size = gdfs

    mask, kwargs = _get_mask(kwargs | {"size": size}, crs=gdfs[0].crs)
    kwargs.pop("size")

    if mask is not None:
        gdfs, column = Explore._separate_args(gdfs, column)
        gdfs, kwargs = _prepare_clipmap(
            *gdfs,
            mask=mask,
            labels=labels,
            **kwargs,
        )

    if explore:
        m = Explore(
            *gdfs,
            column=column,
            labels=labels,
            browser=browser,
            max_zoom=max_zoom,
            smooth_factor=smooth_factor,
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
            random_point = sample.centroid

        center = (random_point.geometry.iloc[0].x, random_point.geometry.iloc[0].y)
        print(f"center={center}, size={size}")

        m._gdf = clean_clip(m._gdf, random_point.buffer(size))

        qtm(m._gdf, column=m.column, cmap=m._cmap, k=m.k)


def _prepare_clipmap(*gdfs, mask, labels, **kwargs):
    if mask is None:
        mask, kwargs = _get_mask(kwargs, crs=gdfs[0].crs)
        if mask is None and len(gdfs) > 1:
            *gdfs, mask = gdfs
        elif mask is None:
            raise ValueError("Must speficy mask.")

    # storing object names in dict here, since the names disappear after clip
    if not labels:
        namedict = make_namedict(gdfs)
        kwargs["namedict"] = namedict

    clipped: tuple[GeoDataFrame] = ()

    if mask is not None:
        for gdf in gdfs:
            clipped_ = clean_clip(gdf, mask)
            clipped = clipped + (clipped_,)

    else:
        for gdf in gdfs[:-1]:
            clipped_ = clean_clip(gdf, gdfs[-1])
            clipped = clipped + (clipped_,)

    if not any(len(gdf) for gdf in clipped):
        raise ValueError("None of the GeoDataFrames are within the mask extent.")

    return clipped, kwargs


def clipmap(
    *gdfs: GeoDataFrame,
    column: str | None = None,
    mask: GeoDataFrame | GeoSeries | Geometry = None,
    labels: tuple[str] | None = None,
    explore: bool = True,
    max_zoom: int = 30,
    smooth_factor: int | float = 1.5,
    browser: bool = False,
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
        smooth_factor: How much to simplify the geometries. 1 is the minimum,
            5 is quite a lot of simplification.
        explore: If True (default), an interactive map will be displayed. If False,
            or not in Jupyter, a static plot will be shown.
        browser: If False (default), the maps will be shown in Jupyter.
            If True the maps will be opened in a browser folder.
        **kwargs: Keyword arguments to pass to geopandas.GeoDataFrame.explore, for
            instance 'cmap' to change the colors, 'scheme' to change how the data
            is grouped. This defaults to 'fisherjenkssampled' for numeric data.

    See also
    --------
    explore: same functionality, but shows the entire area of the geometries.
    samplemap: same functionality, but shows only a random area of a given size.
    """

    gdfs, column = Explore._separate_args(gdfs, column)

    clipped, kwargs = _prepare_clipmap(
        *gdfs,
        mask=mask,
        labels=labels,
        **kwargs,
    )

    center = kwargs.pop("center", None)
    size = kwargs.pop("size", None)

    if explore:
        m = Explore(
            *clipped,
            column=column,
            labels=labels,
            browser=browser,
            max_zoom=max_zoom,
            smooth_factor=smooth_factor,
            **kwargs,
        )
        m.explore(center=center, size=size)
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
        m.change_cmap(
            cmap, start=kwargs.pop("cmap_start", 0), stop=kwargs.pop("cmap_stop", 256)
        )

    if not legend:
        m.legend = None

    m.plot(**kwargs)

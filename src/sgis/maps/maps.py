"""Interactive and static mapping of multiple GeoDataFrames.

The main function is 'explore', which displays one of more GeoDataFrames together in an
interactive map with layers that can be toggled on and off. The 'samplemap' and
'clipmap' functions do the same, but displays a random and chosen area respectfully.

The 'qtm' function shows a simple static map of one or more GeoDataFrames.
"""

import inspect
import warnings
from numbers import Number
from typing import Any

from geopandas import GeoDataFrame, GeoSeries
from shapely import Geometry

from ..geopandas_tools.conversion import to_gdf as to_gdf_func
from ..geopandas_tools.general import clean_geoms, get_common_crs, is_wkt
from ..geopandas_tools.geocoding import address_to_gdf
from ..geopandas_tools.geometry_types import get_geom_type
from .explore import Explore
from .map import Map
from .thematicmap import ThematicMap


def _get_location_mask(kwargs: dict, gdfs) -> tuple[GeoDataFrame | None, dict]:
    try:
        crs = get_common_crs(gdfs)
    except IndexError:
        try:
            crs = [x for x in kwargs.values() if hasattr(x, "crs")][0].crs
        except IndexError:
            crs = None

    masks = {
        "bygdoy": (10.6976899, 59.9081695),
        "akersveien": (10.7476367, 59.9222191),
        "kongsvinger": (12.0035242, 60.1875279),
        "stavanger": (5.6960601, 58.8946196),
        "volda": (6.0705987, 62.146643),
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
            the_mask = to_gdf_func([mask], crs=4326).to_crs(crs).buffer(size)
            return the_mask, kwargs

    return None, kwargs


def explore(
    *gdfs: GeoDataFrame | dict[str, GeoDataFrame],
    column: str | None = None,
    center: Any | None = None,
    labels: tuple[str] | None = None,
    max_zoom: int = 40,
    browser: bool = False,
    smooth_factor: int | float = 1.5,
    size: int | None = None,
    **kwargs,
) -> None:
    """Interactive map of GeoDataFrames with layers that can be toggled on/off.

    It takes all the given GeoDataFrames and displays them together in an
    interactive map with a common legend. If 'column' is not specified, each
    GeoDataFrame is given a unique color.

    The 'center' parameter can be used to show only parts of large datasets.
    'center' can be a string of a city or address, a coordinate tuple or a
    geometry-like object.

    Args:
        *gdfs: one or more GeoDataFrames.
        column: The column to color the geometries by. Defaults to None, which means
            each GeoDataFrame will get a unique color.
        center: Either an address string to be geocoded or a geometry like object
            (coordinate pair (x, y), GeoDataFrame, bbox, etc.). If the geometry is a
            point (or line), it will be buffered by 1000 (can be changed with the)
            size parameter. If a polygon is given, it will not be buffered.
        labels: By default, the GeoDataFrames will be labeled by their object names.
            Alternatively, labels can be specified as a tuple of strings with the same
            length as the number of gdfs.
        max_zoom: The maximum allowed level of zoom. Higher number means more zoom
            allowed. Defaults to 30, which is higher than the geopandas default.
        browser: If False (default), the maps will be shown in Jupyter.
            If True the maps will be opened in a browser folder.
        smooth_factor: How much to simplify the geometries. 1 is the minimum,
            5 is quite a lot of simplification.
        size: The buffer distance. Only used when center is given. It then defaults to
            1000.
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

    gdfs, column, kwargs = Map._separate_args(gdfs, column, kwargs)

    loc_mask, kwargs = _get_location_mask(kwargs | {"size": size}, gdfs)

    kwargs.pop("size", None)

    mask = kwargs.pop("mask", loc_mask)

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
        size = size or 1000
        if isinstance(center, str) and not is_wkt(center):
            mask = address_to_gdf(center, crs=gdfs[0].crs).buffer(size)
        elif isinstance(center, GeoDataFrame):
            mask = center
        else:
            try:
                mask = to_gdf_func(center, crs=gdfs[0].crs)
            except IndexError:
                df = [x for x in kwargs.values() if hasattr(x, "crs")][0]
                mask = to_gdf_func(center, crs=df.crs)

        if get_geom_type(mask) in ["point", "line"]:
            mask = mask.buffer(size)

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

    if m.gdfs is None:
        return

    if not kwargs.pop("explore", True):
        return qtm(m._gdf, column=m.column, cmap=m._cmap, k=m.k)

    m.explore()


def samplemap(
    *gdfs: GeoDataFrame,
    column: str | None = None,
    size: int = 1000,
    sample_from_first: bool = True,
    labels: tuple[str] | None = None,
    max_zoom: int = 40,
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

    if gdfs and isinstance(gdfs[-1], (float, int)):
        *gdfs, size = gdfs

    gdfs, column, kwargs = Map._separate_args(gdfs, column, kwargs)

    mask, kwargs = _get_location_mask(kwargs | {"size": size}, gdfs)
    kwargs.pop("size")

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
        if m.gdfs is None:
            return
        if mask is not None:
            m._gdfs = [gdf.clip(mask) for gdf in m._gdfs]
            m._gdf = m._gdf.clip(mask)
            m._nan_idx = m._gdf[m._column].isna()
            m._get_unique_values()

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
            random_point = sample.sample_points(size=1)

        # if point or mixed geometries
        else:
            random_point = sample.centroid

        center = (random_point.geometry.iloc[0].x, random_point.geometry.iloc[0].y)
        print(f"center={center}, size={size}")

        m._gdf = m._gdf.clip(random_point.buffer(size))

        qtm(m._gdf, column=m.column, cmap=m._cmap, k=m.k)


def clipmap(
    *gdfs: GeoDataFrame,
    column: str | None = None,
    mask: GeoDataFrame | GeoSeries | Geometry = None,
    labels: tuple[str] | None = None,
    explore: bool = True,
    max_zoom: int = 40,
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

    gdfs, column, kwargs = Map._separate_args(gdfs, column, kwargs)

    if mask is None and len(gdfs) > 1:
        mask = gdfs[-1]
        gdfs = gdfs[:-1]

    center = kwargs.pop("center", None)
    size = kwargs.pop("size", None)

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
        if m.gdfs is None:
            return

        m._gdfs = [gdf.clip(mask) for gdf in m._gdfs]
        m._gdf = m._gdf.clip(mask)
        m._nan_idx = m._gdf[m._column].isna()
        m._get_unique_values()
        m.explore(center=center, size=size)
    else:
        m = Map(
            *gdfs,
            column=column,
            labels=labels,
            **kwargs,
        )
        if m.gdfs is None:
            return

        m._gdfs = [gdf.clip(mask) for gdf in m._gdfs]
        m._gdf = m._gdf.clip(mask)
        m._nan_idx = m._gdf[m._column].isna()
        m._get_unique_values()

        qtm(m._gdf, column=m.column, cmap=m._cmap, k=m.k)


def explore_locals(*gdfs, to_gdf: bool = True, **kwargs):
    """Viser kart (explore) over alle lokale GeoDataFrames.

    Lokalt betyr enten inni funksjonen/metoden du er i, eller i notebooken
    du jobber i.

    Args:
        *gdfs: Ekstra GeoDataFrames du vil legge til
        **kwargs: keyword arguments som sendes til sg.explore.
    """
    frame = inspect.currentframe()

    while True:
        local_gdfs = {}
        for name, value in frame.f_locals.items():
            if isinstance(value, GeoDataFrame):
                local_gdfs[name] = value
                continue
            if not to_gdf:
                continue
            try:
                if hasattr(value, "__len__") and not len(value):
                    continue
            except TypeError:
                continue

            try:
                gdf = clean_geoms(to_gdf_func(value))
                if len(gdf):
                    local_gdfs[name] = gdf
            except Exception:
                pass

        if local_gdfs:
            break

        frame = frame.f_back

        if not frame:
            break

    mask = kwargs.pop("mask", None)
    if mask is not None:
        local_gdfs = {name: gdf.clip(mask) for name, gdf in local_gdfs.items()}

    explore(*gdfs, **local_gdfs, **kwargs)


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
    gdfs, column, kwargs = Map._separate_args(gdfs, column, kwargs)

    new_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, GeoDataFrame):
            value.name = key
            gdfs += (value,)
        else:
            new_kwargs[key] = value

            # self.labels.append(key)
            # self.show.append(last_show)

    m = ThematicMap(*gdfs, column=column, size=size, black=black)

    if m._gdfs is None:
        return

    m.title = title

    if k and len(m._unique_values) >= k:
        m.k = k

    if cmap:
        m.change_cmap(
            cmap, start=kwargs.pop("cmap_start", 0), stop=kwargs.pop("cmap_stop", 256)
        )

    if not legend:
        m.legend = None

    m.plot(**new_kwargs)

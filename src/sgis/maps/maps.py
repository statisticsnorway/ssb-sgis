"""Interactive and static mapping of multiple GeoDataFrames.

The main function is 'explore', which displays one of more GeoDataFrames together in an
interactive map with layers that can be toggled on and off. The 'samplemap' and
'clipmap' functions do the same, but displays a random and chosen area respectfully.

The 'qtm' function shows a simple static map of one or more GeoDataFrames.
"""

import inspect
import random
from numbers import Number
from typing import Any

import pyproj
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from shapely import Geometry
from shapely import box
from shapely.geometry import Polygon

from ..debug_config import _NoExplore
from ..geopandas_tools.bounds import get_total_bounds
from ..geopandas_tools.conversion import to_bbox
from ..geopandas_tools.conversion import to_gdf
from ..geopandas_tools.conversion import to_shapely
from ..geopandas_tools.general import clean_geoms
from ..geopandas_tools.general import get_common_crs
from ..geopandas_tools.general import is_wkt
from ..geopandas_tools.geocoding import address_to_gdf
from ..geopandas_tools.geometry_types import get_geom_type
from .explore import Explore
from .map import Map
from .thematicmap import ThematicMap

try:
    from torchgeo.datasets.geo import RasterDataset
except ImportError:

    class RasterDataset:
        """Placeholder."""


def _get_location_mask(kwargs: dict, gdfs) -> tuple[GeoDataFrame | None, dict]:
    try:
        crs = get_common_crs(gdfs)
    except (IndexError, pyproj.exceptions.CRSError):
        for x in kwargs.values():
            try:
                crs = pyproj.CRS(x.crs) if hasattr(x, "crs") else pyproj.CRS(x["crs"])
                break
            except Exception:
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
            the_mask = to_gdf([mask], crs=4326).to_crs(crs).buffer(size)
            return the_mask, kwargs

    return None, kwargs


def explore(
    *gdfs: GeoDataFrame | dict[str, GeoDataFrame],
    column: str | None = None,
    center: Any | None = None,
    max_zoom: int = 40,
    browser: bool = False,
    smooth_factor: int | float = 1.5,
    size: int | None = None,
    max_images: int = 10,
    images_to_gdf: bool = False,
    **kwargs,
) -> Explore:
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
        center: Geometry-like object to center the map on. If a three-length tuple
            is given, the first two should be x and y coordinates and the third
            should be a number of meters to buffer the centerpoint by.
        max_zoom: The maximum allowed level of zoom. Higher number means more zoom
            allowed. Defaults to 30, which is higher than the geopandas default.
        browser: If False (default), the maps will be shown in Jupyter.
            If True the maps will be opened in a browser folder.
        smooth_factor: How much to simplify the geometries. 1 is the minimum,
            5 is quite a lot of simplification.
        size: The buffer distance. Only used when center is given. It then defaults to
            1000.
        max_images: Maximum number of images (Image, ImageCollection, Band) to show per
            map. Defaults to 10.
        images_to_gdf: If True (not default), images (Image, ImageCollection, Band)
            will be converted to GeoDataFrame and added to the map.
        **kwargs: Keyword arguments to pass to geopandas.GeoDataFrame.explore, for
            instance 'cmap' to change the colors, 'scheme' to change how the data
            is grouped. This defaults to 'fisherjenkssampled' for numeric data.

    See Also:
    ---------
    samplemap: same functionality, but shows only a random area of a given size.
    clipmap: same functionality, but shows only the areas clipped by a given mask.

    Examples:
    ---------
    >>> import sgis as sg
    >>> roads = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")
    >>> points = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet")

    Explore the area 500 meter around a given point. Coordinates are in UTM 33 format (25833).

    >>> sg.explore(roads, points, center=(262274.6528, 6650143.176, 500))

    Same as above, but with coordinates given as WGS84, same as the coordinates displayed in the corner of the map.

    >>> sg.explore(roads, points, center_4326=(10.7463, 59.92, 500))

    With additional arguments.

    >>> roads["meters"] = roads.length
    >>> points["meters"] = points.length
    >>> sg.explore(roads, points, column="meters", cmap="plasma", max_zoom=60, center_4326=(10.7463, 59.92, 500))
    """
    if isinstance(center, _NoExplore):
        return

    gdfs, column, kwargs = Map._separate_args(gdfs, column, kwargs)

    loc_mask, kwargs = _get_location_mask(kwargs | {"size": size}, gdfs)

    kwargs.pop("size", None)

    mask = kwargs.pop("mask", loc_mask)

    if mask is not None:
        return clipmap(
            *gdfs,
            column=column,
            mask=mask,
            browser=browser,
            max_zoom=max_zoom,
            **kwargs,
        )

    try:
        to_crs = gdfs[0].crs
    except (IndexError, AttributeError):
        try:
            to_crs = next(x for x in kwargs.values() if hasattr(x, "crs")).crs
        except (IndexError, StopIteration):
            to_crs = None

    if "crs" in kwargs:
        from_crs = kwargs.pop("crs")
    else:
        from_crs = to_crs

    if center is not None:
        size = size or 1000
        if isinstance(center, str) and not is_wkt(center):
            mask = address_to_gdf(center, crs=from_crs)
        elif isinstance(center, (GeoDataFrame, GeoSeries)):
            mask = center
        else:
            if isinstance(center, (tuple, list)) and len(center) == 3:
                *center, size = center
            mask = to_gdf(center, crs=from_crs)

        bounds: Polygon = box(*get_total_bounds(*gdfs, *list(kwargs.values())))

        any_intersections: bool = mask.intersects(bounds).any()
        if not any_intersections and to_crs is None:
            mask = to_gdf(Polygon(), to_crs)
        elif not any_intersections:
            bounds4326 = to_gdf(bounds, to_crs).to_crs(25833).geometry.iloc[0]
            mask4326 = mask.set_crs(4326, allow_override=True).to_crs(25833)

            if (mask4326.distance(bounds4326) > size).all():
                # try flipping coordinates
                x, y = next(iter(mask.geometry.iloc[0].coords))
                mask4326 = to_gdf([y, x], 4326).to_crs(25833)

            if (mask4326.distance(bounds4326) > size).all():
                mask = to_gdf(Polygon(), to_crs)
            else:
                mask = mask4326.to_crs(to_crs)

        # else:
        #     mask_flipped = mask

        # # coords = mask.get_coordinates()
        # if (
        #     (mask_flipped.distance(bounds) > size).all()
        #     # and coords["x"].max() < 180
        #     # and coords["y"].max() < 180
        #     # and coords["x"].min() > -180
        #     # and coords["y"].min() > -180
        # ):
        #     try:
        #         bounds4326 = to_gdf(bounds, to_crs).to_crs(4326).geometry.iloc[0]
        #     except ValueError:
        #         bounds4326 = to_gdf(bounds, to_crs).set_crs(4326).geometry.iloc[0]

        #     mask4326 = mask.set_crs(4326, allow_override=True)

        #     if (mask4326.distance(bounds4326) > size).all():
        #         # try flipping coordinates
        #         x, y = list(mask4326.geometry.iloc[0].coords)[0]
        #         mask4326 = to_gdf([y, x], 4326)

        #     mask = mask4326

        #     # if mask4326.intersects(bounds4326).any():
        #     #     mask = mask4326
        #     # else:
        #     #     try:
        #     #         mask = mask.to_crs(to_crs)
        #     #     except ValueError:
        #     #         pass
        # else:
        #     mask = mask_flipped

        # try:
        #     mask = mask.to_crs(to_crs)
        # except ValueError:
        #     pass

        if get_geom_type(mask) in ["point", "line"]:
            mask = mask.buffer(size)

        return clipmap(
            *gdfs,
            column=column,
            mask=mask,
            browser=browser,
            max_zoom=max_zoom,
            **kwargs,
        )

    m = Explore(
        *gdfs,
        column=column,
        browser=browser,
        max_zoom=max_zoom,
        smooth_factor=smooth_factor,
        max_images=max_images,
        **kwargs,
    )

    if m.gdfs is None and not len(m.rasters):
        return m

    if not kwargs.pop("explore", True):
        return qtm(m._gdf, column=m.column, cmap=m._cmap, k=m.k)

    m.explore()

    return m


def samplemap(
    *gdfs: GeoDataFrame,
    column: str | None = None,
    size: int = 1000,
    sample_from_first: bool = True,
    max_zoom: int = 40,
    smooth_factor: int = 1.5,
    explore: bool = True,
    browser: bool = False,
    max_images: int = 10,
    **kwargs,
) -> Explore:
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
        max_zoom: The maximum allowed level of zoom. Higher number means more zoom
            allowed. Defaults to 30, which is higher than the geopandas default.
        smooth_factor: How much to simplify the geometries. 1 is the minimum,
            5 is quite a lot of simplification.
        explore: If True (default), an interactive map will be displayed. If False,
            or not in Jupyter, a static plot will be shown.
        browser: If False (default), the maps will be shown in Jupyter.
            If True the maps will be opened in a browser folder.
        max_images: Maximum number of images (Image, ImageCollection, Band) to show per
            map. Defaults to 10.
        **kwargs: Keyword arguments to pass to geopandas.GeoDataFrame.explore, for
            instance 'cmap' to change the colors, 'scheme' to change how the data
            is grouped. This defaults to 'fisherjenkssampled' for numeric data.

    See Also:
    ---------
    explore: Same functionality, but shows the entire area of the geometries.
    clipmap: Same functionality, but shows only the areas clipped by a given mask.

    Examples:
    ---------
    >>> from sgis import read_parquet_url, samplemap
    >>> roads = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_eidskog_2022.parquet")
    >>> points = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_eidskog.parquet")

    With default sample size. To get a new sample area, simply re-run the line.

    >>> samplemap(roads, points)

    Sample area with a radius of 5 kilometers.

    >>> samplemap(roads, points, size=5_000, column="meters")

    """
    if isinstance(kwargs.get("center", None), _NoExplore):
        return

    if gdfs and len(gdfs) > 1 and isinstance(gdfs[-1], (float, int)):
        *gdfs, size = gdfs

    gdfs, column, kwargs = Map._separate_args(gdfs, column, kwargs)

    loc_mask, kwargs = _get_location_mask(kwargs | {"size": size}, gdfs)
    kwargs.pop("size")
    mask = kwargs.pop("mask", loc_mask)

    i = 0 if sample_from_first else random.choice(list(range(len(gdfs))))
    try:
        sample = gdfs[i]
    except IndexError:
        sample = list(kwargs.values())[i]

    if mask is None:
        try:
            sample = sample.geometry.loc[lambda x: ~x.is_empty].sample(1)
        except Exception:
            try:
                sample = sample.sample(1)
            except Exception:
                pass
        try:
            sample = to_gdf(to_shapely(sample)).explode(ignore_index=True)
        except Exception:
            sample = to_gdf(to_shapely(to_bbox(sample))).explode(ignore_index=True)
    else:
        try:
            sample = to_gdf(to_shapely(sample)).explode(ignore_index=True)
        except Exception:
            sample = to_gdf(to_shapely(to_bbox(sample))).explode(ignore_index=True)

        sample = sample.clip(mask).explode(ignore_index=True).sample(1)

    random_point = sample.sample_points(size=1)

    try:
        center = (random_point.geometry.iloc[0].x, random_point.geometry.iloc[0].y)
    except AttributeError as e:
        raise AttributeError(e, random_point.geometry.iloc[0]) from e

    print(f"center={center}, size={size}")

    mask = random_point.buffer(size)

    return clipmap(
        *gdfs,
        column=column,
        mask=mask,
        browser=browser,
        max_zoom=max_zoom,
        explore=explore,
        smooth_factor=smooth_factor,
        max_images=max_images,
        **kwargs,
    )


def clipmap(
    *gdfs: GeoDataFrame,
    column: str | None = None,
    mask: GeoDataFrame | GeoSeries | Geometry = None,
    explore: bool = True,
    max_zoom: int = 40,
    smooth_factor: int | float = 1.5,
    browser: bool = False,
    max_images: int = 10,
    **kwargs,
) -> Explore | Map:
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
        max_zoom: The maximum allowed level of zoom. Higher number means more zoom
            allowed. Defaults to 30, which is higher than the geopandas default.
        smooth_factor: How much to simplify the geometries. 1 is the minimum,
            5 is quite a lot of simplification.
        explore: If True (default), an interactive map will be displayed. If False,
            or not in Jupyter, a static plot will be shown.
        browser: If False (default), the maps will be shown in Jupyter.
            If True the maps will be opened in a browser folder.
        max_images: Maximum number of images (Image, ImageCollection, Band) to show per
            map. Defaults to 10.
        **kwargs: Keyword arguments to pass to geopandas.GeoDataFrame.explore, for
            instance 'cmap' to change the colors, 'scheme' to change how the data
            is grouped. This defaults to 'fisherjenkssampled' for numeric data.

    See Also:
    ---------
    explore: same functionality, but shows the entire area of the geometries.
    samplemap: same functionality, but shows only a random area of a given size.
    """
    if isinstance(kwargs.get("center", None), _NoExplore):
        return

    gdfs, column, kwargs = Map._separate_args(gdfs, column, kwargs)
    if mask is None and len(gdfs) > 1:
        mask = gdfs[-1]
        gdfs = gdfs[:-1]

    if not isinstance(mask, (GeoDataFrame | GeoSeries | Geometry | tuple)):
        try:
            mask = to_gdf(mask)
        except Exception:
            mask = to_shapely(to_bbox(mask))

    center = kwargs.pop("center", None)
    size = kwargs.pop("size", None)

    if explore:
        m = Explore(
            *gdfs,
            column=column,
            browser=browser,
            max_zoom=max_zoom,
            smooth_factor=smooth_factor,
            max_images=max_images,
            **kwargs,
        )
        m.mask = mask

        if m.gdfs is None and not len(m.rasters):
            return m

        m._gdfs = [gdf.clip(mask) for gdf in m._gdfs]
        m._gdf = m._gdf.clip(mask)
        m._nan_idx = m._gdf[m._column].isna()
        m._get_unique_values()
        m.explore(center=center, size=size)
        return m
    else:
        m = Map(
            *gdfs,
            column=column,
            **kwargs,
        )
        if m.gdfs is None:
            return m

        m._gdfs = [gdf.clip(mask) for gdf in m._gdfs]
        m._gdf = m._gdf.clip(mask)
        m._nan_idx = m._gdf[m._column].isna()
        m._get_unique_values()

        qtm(m._gdf, column=m.column, cmap=m._cmap, k=m.k)

        return m


def explore_locals(
    *gdfs: GeoDataFrame, convert: bool = True, crs: Any | None = None, **kwargs
) -> None:
    """Displays all local variables with geometries (GeoDataFrame etc.).

    Local means inside a function or file/notebook.

    Args:
        *gdfs: Additional GeoDataFrames.
        convert: If True (default), non-GeoDataFrames will be converted
            to GeoDataFrames if possible.
        crs: Optional crs if no objects have any crs.
        **kwargs: keyword arguments passed to sg.explore.
    """
    if isinstance(kwargs.get("center", None), _NoExplore):
        return

    def as_dict(obj):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        elif isinstance(obj, dict):
            return obj
        raise TypeError

    frame = inspect.currentframe().f_back

    allowed_types = (GeoDataFrame, GeoSeries, Geometry, RasterDataset)

    local_gdfs = {}
    while True:
        for name, value in frame.f_locals.items():
            if isinstance(value, GeoDataFrame) and len(value):
                local_gdfs[name] = value
                continue
            if not convert:
                continue

            try:
                gdf = clean_geoms(to_gdf(value, crs=crs))
                if len(gdf):
                    local_gdfs[name] = gdf
                continue
            except Exception:
                pass

            if isinstance(value, dict) or hasattr(value, "__dict__"):
                # add dicts or classes with GeoDataFrames to kwargs
                for key, val in as_dict(value).items():
                    if isinstance(val, allowed_types):
                        gdf = clean_geoms(to_gdf(val, crs=crs))
                        if len(gdf):
                            local_gdfs[key] = gdf

                    elif isinstance(val, dict) or hasattr(val, "__dict__"):
                        try:
                            for k, v in val.items():
                                if isinstance(v, allowed_types):
                                    gdf = clean_geoms(to_gdf(v, crs=crs))
                                    if len(gdf):
                                        local_gdfs[k] = gdf
                        except Exception:
                            # no need to raise here
                            pass

        if local_gdfs:
            break

        frame = frame.f_back

        if not frame:
            break

    return explore(*gdfs, **(local_gdfs | kwargs))


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
        legend: Whether to add legend. Defaults to True.
        cmap: Color palette of the map. See:
            https://matplotlib.org/stable/tutorials/colors/colormaps.html
        k: Number of color groups.
        **kwargs: Additional keyword arguments taken by the geopandas plot method.

    See Also:
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

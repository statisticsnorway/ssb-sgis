"""Interactive map of one or more GeoDataFrames with layers that can be toggles on/off.

This module holds the Explore class, which is the basis for the explore, samplemap and
clipmap functions from the 'maps' module.
"""

import os
import random
import re
import warnings
from collections.abc import Iterable
from numbers import Number
from pathlib import Path
from statistics import mean
from typing import Any
from typing import ClassVar

import branca as bc
import folium
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import xyzservices
from folium import plugins
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from IPython.display import display
from jinja2 import Template
from pandas.api.types import is_datetime64_any_dtype
from shapely import Geometry
from shapely import box
from shapely import union_all
from shapely.geometry import LineString

from ..geopandas_tools.bounds import get_total_bounds
from ..geopandas_tools.conversion import to_bbox
from ..geopandas_tools.conversion import to_gdf
from ..geopandas_tools.conversion import to_shapely
from ..geopandas_tools.general import clean_geoms
from ..geopandas_tools.general import make_all_singlepart
from ..geopandas_tools.geometry_types import get_geom_type
from ..geopandas_tools.geometry_types import to_single_geom_type

try:
    from ..raster.image_collection import Band
    from ..raster.image_collection import Image
    from ..raster.image_collection import ImageCollection
except ImportError:

    class Band:
        """Placeholder."""

    class Image:
        """Placeholder."""

    class ImageCollection:
        """Placeholder."""


from .httpserver import run_html_server
from .map import Map
from .map import _determine_best_name
from .tilesources import kartverket
from .tilesources import xyz

try:
    from torchgeo.datasets.geo import RasterDataset
except ImportError:

    class RasterDataset:
        """Placeholder."""


# the geopandas._explore raises a deprication warning. Ignoring for now.
warnings.filterwarnings(
    action="ignore", category=matplotlib.MatplotlibDeprecationWarning
)
pd.options.mode.chained_assignment = None


# gray for NaNs
NAN_COLOR = "#969696"


# cols to not show when hovering over geometries (tooltip)
COLS_TO_DROP = ["color", "col_as_int", "geometry"]


# from geopandas
_MAP_KWARGS = [
    "location",
    "prefer_canvas",
    "no_touch",
    "disable_3d",
    "png_enabled",
    "zoom_control",
    "crs",
    "zoom_start",
    "left",
    "top",
    "position",
    "min_zoom",
    "max_zoom",
    "min_lat",
    "max_lat",
    "min_lon",
    "max_lon",
    "max_bounds",
]


class MeasureControlFix(plugins.MeasureControl):
    """Monkey-patch to fix a bug in the lenght measurement control.

    Kudos to abewartech (https://github.com/ljagis/leaflet-measure/issues/171).
    """

    _template = Template(
        """
{% macro script(this, kwargs) %}
    L.Control.Measure.include({ _setCaptureMarkerIcon: function () { this._captureMarker.options.autoPanOnFocus = false; this._captureMarker.setIcon( L.divIcon({ iconSize: this._map.getSize().multiplyBy(2), }), ); }, });
    var {{ this.get_name() }} = new L.Control.Measure(
        {{ this.options|tojson }});
    {{this._parent.get_name()}}.addControl({{this.get_name()}});

{% endmacro %}
    """
    )

    def __init__(
        self, active_color: str = "red", completed_color: str = "red", **kwargs
    ) -> None:
        """Run super __init__ after the new _template class attribute is made."""
        super().__init__(
            active_color=active_color, completed_color=completed_color, **kwargs
        )


def to_tile(tile: str | xyzservices.TileProvider, max_zoom: int) -> folium.TileLayer:
    common_bgmaps = {
        "openstreetmap": folium.TileLayer(
            "OpenStreetMap", min_zoom=0, max_zoom=max_zoom
        ),
        "grunnkart": kartverket.topo,
        "gråtone": kartverket.topogråtone,
        "norge_i_bilder": kartverket.norge_i_bilder,
        "dark": xyz.CartoDB.DarkMatter,
        "voyager": xyz.CartoDB.Voyager,
        "strava": xyz.Strava.All,
    }
    try:
        name = tile["name"]
    except TypeError:
        name = tile

    if not isinstance(tile, str):
        try:
            return folium.TileLayer(tile, name=name, max_zoom=max_zoom)
        except TypeError:
            return folium.TileLayer(tile, max_zoom=max_zoom)

    try:
        provider = common_bgmaps[tile.lower()]
    except KeyError:
        provider = xyzservices.providers.query_name(tile)

    if isinstance(provider, folium.TileLayer):
        return provider

    if isinstance(provider, xyzservices.TileProvider):
        attr = provider.html_attribution
        provider = provider.build_url(scale_factor="{r}")
    else:
        try:
            attr = provider["attr"]
        except (AttributeError, TypeError):
            attr = None

    return folium.TileLayer(provider, name=name, attr=attr, max_zoom=max_zoom)


def _single_band_to_arr(band, mask, name, raster_data_dict):
    if band.has_array and mask is None:
        arr = band.values
    elif band.has_array:
        arr = band.clip(mask).values
    else:
        arr = band.load(indexes=1, bounds=mask).values
    bounds: tuple = (
        _any_to_bbox_crs4326(mask, band.crs)
        if mask is not None
        else union_all(
            gpd.GeoSeries(box(*band.bounds), crs=band.crs).to_crs(4326).geometry.values
        ).bounds
    )
    try:
        raster_data_dict["cmap"] = band.get_cmap(arr)
    except Exception:
        try:
            raster_data_dict["cmap"] = plt.get_cmap(band.cmap)
        except Exception:
            raster_data_dict["cmap"] = band.cmap or "Grays"
    raster_data_dict["arr"] = arr
    raster_data_dict["bounds"] = bounds
    raster_data_dict["label"] = name


def _any_to_bbox_crs4326(obj, crs):
    return to_bbox(to_gdf(obj, crs).to_crs(4326))


class Explore(Map):
    """Class for displaying and saving html maps of multiple GeoDataFrames."""

    # class attribute that can be overridden locally
    tiles: ClassVar[tuple[str, ...]] = (
        "grunnkart",
        "norge_i_bilder",
        "dark",
        "OpenStreetMap",
    )

    def __init__(
        self,
        *gdfs,
        mask=None,
        column: str | None = None,
        popup: bool = True,
        max_zoom: int = 40,
        smooth_factor: float = 1.1,
        browser: bool = False,
        prefer_canvas: bool = True,
        measure_control: bool = True,
        geocoder: bool = False,
        out_path: str | None = None,
        show: bool | Iterable[bool] | None = None,
        text: str | None = None,
        decimals: int = 6,
        max_images: int = 10,
        **kwargs,
    ) -> None:
        """Initialiser.

        Args:
            *gdfs: One or more GeoDataFrames.
            mask: Optional mask to clip to.
            column: Optional column to color the data by.
            popup: Whether to make the data popups clickable.
            max_zoom: Max levels of zoom.
            smooth_factor: Float of 1 or higher, 1 meaning no smoothing
                of lines.
            browser: Whether to open the map in a browser tab.
            prefer_canvas: Option.
            measure_control: Whether to include measurement box.
            geocoder: Whether to include search bar for addresses.
            out_path: Optional file path to an html file. The map will then
                be saved instead of displayed.
            show: Whether to show or hide the data upon creating the map.
                If False, the data can be toggled on later. 'show' can also be
                a sequence of boolean values the same length as the number of
                GeoDataFrames.
            max_images: Maximum number of images (Image, ImageCollection, Band) to show per
                map. Defaults to 15.
            text: Optional text for a text box in the map.
            decimals: Number of decimals in the coordinates.
            **kwargs: Additional keyword arguments. Can also be geometry-like objects
                where the key is the label.
        """
        self.popup = popup
        self.max_zoom = max_zoom
        self.smooth_factor = smooth_factor
        self.prefer_canvas = prefer_canvas
        self.measure_control = measure_control
        self.geocoder = geocoder
        self.out_path = out_path
        self.mask = mask
        self.text = text
        self.decimals = decimals
        self.max_images = max_images
        self.legend = None

        self.browser = browser
        if not self.browser and "show_in_browser" in kwargs:
            self.browser = kwargs.pop("show_in_browser")
        if not self.browser and "in_browser" in kwargs:
            self.browser = kwargs.pop("in_browser")

        if show is None:
            self._show_was_none = True
            show = True
        else:
            self._show_was_none = False

        new_gdfs = {}
        self.rasters = {}
        for i, gdf in enumerate(gdfs):
            name = _determine_best_name(gdf, column, i)

            if name in new_gdfs or name in self.rasters:
                name += str(i)

            if isinstance(gdf, (ImageCollection | Image | Band)):
                self.rasters[name] = gdf.copy()
                continue
            try:
                new_gdfs[name] = to_gdf(gdf)
            except Exception:
                continue

        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (ImageCollection | Image | Band)):
                self.rasters[key] = value
            else:
                new_kwargs[key] = value

        super().__init__(column=column, show=show, **(new_kwargs | new_gdfs))

        if self.gdfs is None:
            return

        # stringify or remove columns not renerable by leaflet (list, geometry etc.)
        new_gdfs, show_new = [], []
        for gdf, show in zip(self.gdfs, self.show, strict=True):
            try:
                gdf = gdf.reset_index()
            except Exception:
                pass
            for col in gdf.columns:
                if is_datetime64_any_dtype(gdf[col]):
                    try:
                        gdf[col] = [str(x) for x in gdf[col].dt.round("d")]
                    except Exception:
                        gdf = gdf.drop(col, axis=1)
                    continue

                if not len(gdf.loc[gdf[col].notna()]):
                    continue
                if not isinstance(
                    gdf.loc[gdf[col].notna(), col].iloc[0], (Number, str, Geometry)
                ) or (
                    col != gdf._geometry_column_name
                    and isinstance(gdf.loc[gdf[col].notna(), col].iloc[0], (Geometry))
                ):
                    try:
                        gdf[col] = gdf[col].astype(str).fillna(pd.NA)
                    except Exception:
                        gdf = gdf.drop(col, axis=1)

            try:
                gdf.index = gdf.index.astype(str)
            except Exception:
                pass
            new_gdfs.append(to_gdf(gdf))
            show_new.append(show)
        self._gdfs = new_gdfs
        if self._gdfs:
            self._gdf = pd.concat(new_gdfs, ignore_index=True)
        else:
            self._gdf = GeoDataFrame({"geometry": [], self._column: []})
        self.show = show_new

        # if self._show_was_none and len(self._gdfs) > 6:
        #     self.show = [False] * len(self._gdfs)

        if self._is_categorical:
            if len(self.gdfs) == 1:
                self._split_categories()
        else:
            if not self._cmap:
                self._cmap = "viridis"
            self.cmap_start = self.kwargs.pop("cmap_start", 0)
            self.cmap_stop = self.kwargs.pop("cmap_stop", 256)

        # if self._gdf.crs is None:
        #     self.kwargs["crs"] = "Simple"

        self.original_crs = self.gdf.crs

    def __repr__(self) -> str:
        """Representation."""
        return f"{self.__class__.__name__}({len(self)})"

    def __len__(self) -> int:
        """Number of gdfs that have rows plus number of raster images."""
        try:
            rasters = self.raster_data
        except AttributeError:
            rasters = self.rasters
        return len([gdf for gdf in self._gdfs if len(gdf)]) + len(rasters)

    def __bool__(self) -> bool:
        """True if any gdfs have rows or there are any raster images."""
        return bool(len(self))

    def explore(
        self,
        column: str | None = None,
        center: Any | None = None,
        size: int | None = None,
        mask: Any | None = None,
        **kwargs,
    ) -> None:
        """Explore all the data."""
        self.mask = mask if mask is not None else self.mask
        if (
            self._gdfs
            and not any(len(gdf) for gdf in self._gdfs)
            and not len(self.rasters)
        ):
            warnings.warn("None of the GeoDataFrames have rows.", stacklevel=1)
            return
        if column:
            self._column = column
            self._update_column()
            kwargs.pop("column", None)

        if self.mask is not None:
            return self.clipmap(mask=self.mask, column=self._column, **kwargs)

        if center is None:
            self.to_show = self._gdfs
            self._explore(**kwargs)
            return

        size = size if size else 1000

        centerpoint = (
            to_gdf(center, crs=self.crs)
            if not isinstance(center, GeoDataFrame)
            else center
        )

        gdfs: tuple[GeoDataFrame] = ()
        for gdf in self._gdfs:
            keep_geom_type = False if get_geom_type(gdf) == "mixed" else True
            gdf = gdf.clip(centerpoint.buffer(size), keep_geom_type=keep_geom_type)
            gdfs = gdfs + (gdf,)
        self._gdfs = gdfs
        self._gdf = pd.concat(gdfs, ignore_index=True)

        self._get_unique_values()

        self._explore(**kwargs)

    def samplemap(
        self,
        size: int,
        sample: Any,
        column: str | None = None,
        **kwargs,
    ) -> None:
        """Explore a sample of the data."""
        if column:
            self._column = column
            self._update_column()
            kwargs.pop("column", None)

        try:
            sample = sample.sample(1)
        except Exception:
            pass

        try:
            sample = to_gdf(to_shapely(sample)).pipe(make_all_singlepart)
        except Exception:
            sample = to_gdf(to_shapely(to_bbox(sample))).pipe(make_all_singlepart)

        random_point = sample.sample_points(size=1)

        self.center = (random_point.geometry.iloc[0].x, random_point.geometry.iloc[0].y)
        print(f"center={self.center}, size={size}")

        mask = random_point.buffer(size)

        return self.clipmap(mask, column, **kwargs)

    def clipmap(
        self,
        mask: Any,
        column: str | None = None,
        **kwargs,
    ) -> None:
        """Explore the data within a mask extent."""
        self.mask = mask
        if column:
            self._column = column
            self._update_column()
            kwargs.pop("column", None)

        gdfs: tuple[GeoDataFrame] = ()
        for gdf in self._gdfs:
            gdf = gdf.clip(self.mask)
            collections = gdf.loc[gdf.geom_type == "GeometryCollection"]
            if len(collections):
                collections = make_all_singlepart(collections)
                gdf = pd.concat([gdf, collections], ignore_index=False)
            gdfs = gdfs + (gdf,)
        self._gdfs = gdfs
        if self._gdfs:
            self._gdf = pd.concat(self._gdfs, ignore_index=True)
        else:
            self._gdf = GeoDataFrame({"geometry": [], self._column: []})

        self._explore(**kwargs)

    def _load_rasters_as_images(self):
        self.raster_data = []
        n_added_images = 0
        self._show_rasters = True
        for name, value in self.rasters.items():
            data, n_added_images = self._image_collection_to_background_map(
                value,
                self.mask,
                name,
                max_images=self.max_images,
                n_added_images=n_added_images,
            )
            self.raster_data += data

    def _rasters_to_background_maps(self):
        for raster_data_dict in self.raster_data:
            try:
                arr = raster_data_dict["arr"]
            except KeyError:
                continue
            if (arr.shape) == 1:
                continue
            if hasattr(arr, "mask"):
                arr = arr.data
            if "bool" in str(arr.dtype):
                arr = np.where(arr, 1, 0)
            try:
                arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
            except Exception as e:
                warnings.warn(str(e), stacklevel=1)

            label = raster_data_dict["label"]
            bounds = raster_data_dict["bounds"]
            if raster_data_dict["cmap"] is not None:
                kwargs = {"colormap": raster_data_dict["cmap"]}
            else:
                kwargs = {}
            minx, miny, maxx, maxy = bounds
            image_overlay = folium.raster_layers.ImageOverlay(
                arr,
                bounds=[[miny, minx], [maxy, maxx]],
                show=self._show_rasters,
                **kwargs,
            )
            image_overlay.layer_name = Path(label).stem
            image_overlay.add_to(self.map)

    def save(self, path: str) -> None:
        """Save the map to local disk as an html document."""
        with open(path, "w") as f:
            f.write(self.map._repr_html_())

    def _explore(self, **kwargs) -> None:
        self.kwargs = self.kwargs | kwargs

        if self._show_was_none and len([gdf for gdf in self._gdfs if len(gdf)]) > 6:
            self.show = [False] * len(self._gdfs)

        if self._is_categorical:
            self._create_categorical_map()
        else:
            self._create_continous_map()

        if self.out_path:
            with open(
                os.getcwd() + "/" + self.out_path.strip(".html") + ".html", "w"
            ) as f:
                f.write(self.map._repr_html_())
        elif self.browser:
            run_html_server(self.map._repr_html_())
        else:
            display(self.map)

    def _split_categories(self) -> None:
        new_gdfs, new_labels, new_shows = [], [], []
        for cat in self._unique_values:
            gdf = self.gdf.loc[self.gdf[self.column] == cat]
            new_gdfs.append(gdf)
            new_labels.append(cat)
            new_shows.append(self.show[0])
        self._gdfs = new_gdfs
        self._gdf = pd.concat(new_gdfs, ignore_index=True)
        self.labels = new_labels
        self.show = new_shows

    def _to_single_geom_type(self, gdf: GeoDataFrame) -> GeoDataFrame:
        gdf = clean_geoms(gdf)

        if get_geom_type(gdf) != "mixed":
            return gdf

        geom_types = gdf.geom_type.str.lower()
        mess = "Leaflet cannot render mixed geometry types well. "

        if geom_types.str.contains("collection").any():
            mess += "Exploding geometry collections. "
            gdf = make_all_singlepart(gdf)
            geom_types = gdf.geom_type.str.lower()

        if geom_types.str.contains("polygon").any():
            mess += "Keeping only polygons."
            gdf = to_single_geom_type(gdf, geom_type="polygon")

        elif geom_types.str.contains("lin").any():
            mess += "Keeping only lines."
            gdf = to_single_geom_type(gdf, geom_type="line")

        assert get_geom_type(gdf) != "mixed", gdf.geom_type.value_counts()

        warnings.warn(mess, stacklevel=1)

        return gdf

    def _update_column(self) -> None:
        if not self._gdfs:
            return
        self._is_categorical = self._check_if_categorical()
        self._fillna_if_col_is_missing()
        self._gdf = pd.concat(self._gdfs, ignore_index=True)

    def _get_bounds(
        self, gdf: GeoDataFrame
    ) -> tuple[float, float, float, float] | None:
        if not len(gdf) or all(x is None for x in gdf.total_bounds):
            try:
                return get_total_bounds([x["bounds"] for x in self.raster_data])
            except Exception:
                return None

        return gdf.total_bounds

    def _create_categorical_map(self) -> None:
        self._make_categories_colors_dict()
        if self._gdf is not None and len(self._gdf):
            self._fix_nans()
            gdf = self._prepare_gdf_for_map(self._gdf)
        else:
            gdf = GeoDataFrame({"geometry": [], self._column: []})

        self._load_rasters_as_images()

        bounds = self._get_bounds(gdf)

        if bounds is None:
            self.map = None
            return
        self.map = self._make_folium_map(
            bounds=bounds,
            max_zoom=self.max_zoom,
            popup=self.popup,
            prefer_canvas=self.prefer_canvas,
            **self.kwargs,
        )

        for gdf, label, show in zip(self._gdfs, self.labels, self.show, strict=True):
            if not len(gdf):
                continue

            gdf = self._to_single_geom_type(make_all_singlepart(gdf))

            if not len(gdf):
                continue

            gdf.geometry = shapely.force_2d(gdf.geometry.values)
            gdf = self._prepare_gdf_for_map(gdf)

            gjs = self._make_geojson(
                gdf,
                show=show,
                color=gdf["color"],
                tooltip=self._tooltip_cols(gdf),
                popup=self.popup,
                **{
                    key: value
                    for key, value in self.kwargs.items()
                    if key not in ["title", "tiles"]
                },
            )
            gjs.layer_name = label

            gjs.add_to(self.map)

        self._rasters_to_background_maps()

        _categorical_legend(
            self.map,
            self._column,
            self._categories_colors_dict.keys(),
            self._categories_colors_dict.values(),
        )

        self.map.add_child(folium.LayerControl())

    def _add_tiles(
        self, mapobj: folium.Map, tiles: list[str, xyzservices.TileProvider]
    ) -> None:
        for tile in tiles:
            to_tile(tile, max_zoom=self.max_zoom).add_to(mapobj)

    def _create_continous_map(self):
        self._prepare_continous_map()
        if self.scheme:
            classified = self._classify_from_bins(self._gdf, bins=self.bins)
            classified_sequential = self._push_classification(classified)
            n_colors = len(np.unique(classified_sequential)) - any(self._nan_idx)
            unique_colors = self._get_continous_colors(n=n_colors)

        self._load_rasters_as_images()

        gdf = self._prepare_gdf_for_map(self._gdf)
        bounds = self._get_bounds(gdf)
        if bounds is None:
            self.map = None
            return
        self.map = self._make_folium_map(
            bounds=bounds,
            max_zoom=self.max_zoom,
            popup=self.popup,
            prefer_canvas=self.prefer_canvas,
            **self.kwargs,
        )

        colorbar = bc.colormap.StepColormap(
            unique_colors,
            vmin=self._gdf[self._column].min(),
            vmax=self._gdf[self._column].max(),
            caption=self._column,
            index=self.bins,
        )

        for gdf, label, show in zip(self._gdfs, self.labels, self.show, strict=True):
            if not len(gdf):
                continue

            gdf = self._to_single_geom_type(make_all_singlepart(gdf))

            if not len(gdf):
                continue

            gdf.geometry = shapely.force_2d(gdf.geometry.values)
            gdf = self._prepare_gdf_for_map(gdf)

            classified = self._classify_from_bins(gdf, bins=self.bins)
            try:
                colorarray = unique_colors[classified]
            except IndexError:
                classified[classified > 0] = classified[classified > 0] - 1
                colorarray = unique_colors[classified]

            gjs = self._make_geojson(
                gdf,
                show=show,
                color=colorarray,
                tooltip=self._tooltip_cols(gdf),
                popup=self.popup,
                prefer_canvas=self.prefer_canvas,
                **{
                    key: value
                    for key, value in self.kwargs.items()
                    if key not in ["title", "tiles"]
                },
            )

            gjs.layer_name = label

            gjs.add_to(self.map)

        self._rasters_to_background_maps()

        self.map.add_child(colorbar)
        self.map.add_child(folium.LayerControl())

    def _tooltip_cols(self, gdf: GeoDataFrame) -> list:
        if "tooltip" in self.kwargs:
            tooltip = self.kwargs.pop("tooltip")
            return tooltip
        return [col for col in gdf.columns if col not in COLS_TO_DROP]

    @staticmethod
    def _prepare_gdf_for_map(gdf: GeoDataFrame) -> GeoDataFrame:
        if isinstance(gdf, GeoSeries):
            gdf = gdf.to_frame("geometry")

        # convert LinearRing to LineString
        rings_mask = gdf.geom_type == "LinearRing"
        if rings_mask.any():
            gdf.geometry[rings_mask] = gdf.geometry[rings_mask].apply(
                lambda g: LineString(g)
            )

        if gdf.crs is not None and not gdf.crs.equals(4326):
            gdf = gdf.to_crs(4326)

        return gdf

    def _make_folium_map(
        self,
        bounds: tuple[float, float, float, float],
        attr: Any = None,
        tiles: Any = None,
        width: str = "100%",
        height: str = "100%",
        control_scale: bool = True,
        map_kwds: dict | None = None,
        **kwargs,
    ):
        if not map_kwds:
            map_kwds = {}

        if tiles is None:
            tiles = self.tiles

        # create folium.Map object
        # Get bounds to specify location and map extent
        location = kwargs.pop("location", None)
        if location is None:
            x = mean([bounds[0], bounds[2]])
            y = mean([bounds[1], bounds[3]])
            location = (y, x)
            if "zoom_start" in kwargs.keys():
                fit = False
            else:
                fit = True
        else:
            fit = False

        # get a subset of kwargs to be passed to folium.Map
        for i in _MAP_KWARGS:
            if i in map_kwds:
                raise ValueError(
                    f"'{i}' cannot be specified in 'map_kwds'. "
                    f"Use the '{i}={map_kwds[i]}' argument instead."
                )
        map_kwds = {
            **map_kwds,
            **{i: kwargs[i] for i in kwargs.keys() if i in _MAP_KWARGS},
        }
        map_kwds["min_zoom"] = 0
        map_kwds["max_zoom"] = kwargs.get("max_zoom", self.max_zoom)

        if isinstance(tiles, (list, tuple)):
            default_tile, *more_tiles = tiles
        else:
            default_tile, more_tiles = tiles, []

        default_tile = to_tile(default_tile, max_zoom=self.max_zoom)

        if isinstance(default_tile, xyzservices.TileProvider):
            attr = attr if attr else default_tile.html_attribution
            default_tile = default_tile.build_url(scale_factor="{r}")

        m = folium.Map(
            location=location,
            control_scale=control_scale,
            tiles=default_tile,
            attr=attr,
            width=width,
            height=height,
            **map_kwds,
        )

        self._add_tiles(m, more_tiles)

        if self.measure_control:
            MeasureControlFix(
                primary_length_unit="m",
                secondary_length_unit="kilometers",
                primary_area_unit="sqmeters",
                secondary_area_unit="sqkilometers",
                position="bottomleft",
                capture_z_index=False,
                thousands_sep="",
                units={
                    "m": {"factor": 1, "display": "m", "decimals": self.decimals},
                },
            ).add_to(m)

        plugins.Fullscreen(
            position="topleft",
            title="Expand me",
            title_cancel="Exit me",
            force_separate_button=True,
        ).add_to(m)

        plugins.MousePosition(
            position="bottomleft",
            separator=", ",
            empty_string="NaN",
            lng_first=True,
            num_digits=8,
        ).add_to(m)

        if self.geocoder:
            plugins.Geocoder(position="topright").add_to(m)

        # fit bounds to get a proper zoom level
        if fit and "zoom_start" not in kwargs:
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        if self.text:
            style = bc.element.MacroElement()
            style._template = bc.element.Template(get_textbox(self.text))
            m.get_root().add_child(style)
            # folium.LayerControl(collapsed=False).add_to(m)

        return m

    def _make_geojson(
        self,
        df: GeoDataFrame,
        show: bool,
        color: str | None = None,
        tooltip: bool = True,
        popup: bool = False,
        highlight: bool = True,
        marker_type: str | None = None,
        marker_kwds: dict | None = None,
        style_kwds: dict | None = None,
        highlight_kwds: dict | None = None,
        tooltip_kwds: dict | None = None,
        popup_kwds: dict | None = None,
        map_kwds: dict | None = None,
        **kwargs,
    ) -> folium.GeoJson:

        marker_kwds = marker_kwds or {}
        style_kwds = style_kwds or {}
        highlight_kwds = highlight_kwds or {}
        tooltip_kwds = tooltip_kwds or {}
        popup_kwds = popup_kwds or {}
        map_kwds = map_kwds or {}

        gdf = df.copy()

        # convert LinearRing to LineString
        rings_mask = gdf.geom_type == "LinearRing"
        if rings_mask.any():
            gdf.geometry[rings_mask] = gdf.geometry[rings_mask].apply(
                lambda g: LineString(g)
            )

        if gdf.crs is None:
            pass
            # kwargs["crs"] = "Simple"
        elif not gdf.crs.equals(4326):
            gdf = gdf.to_crs(4326)

        # get a subset of kwargs to be passed to folium.Map
        for i in _MAP_KWARGS:
            if i in map_kwds:
                raise ValueError(
                    f"'{i}' cannot be specified in 'map_kwds'. "
                    f"Use the '{i}={map_kwds[i]}' argument instead."
                )
        map_kwds = {
            **map_kwds,
            **{i: kwargs[i] for i in kwargs.keys() if i in _MAP_KWARGS},
        }

        for map_kwd in _MAP_KWARGS:
            kwargs.pop(map_kwd, None)

        # set default style
        if "fillOpacity" not in style_kwds:
            style_kwds["fillOpacity"] = 0.5
        if "weight" not in style_kwds:
            style_kwds["weight"] = 2
        if "style_function" in style_kwds:
            style_kwds_function = style_kwds["style_function"]
            if not callable(style_kwds_function):
                raise ValueError("'style_function' has to be a callable")
            style_kwds.pop("style_function")
        else:

            def _no_style(x):
                return {}

            style_kwds_function = _no_style

        gdf["__folium_color"] = color

        stroke_color = style_kwds.pop("color", None)
        if not stroke_color:

            def _style_column(x):
                base_style = {
                    "fillColor": x["properties"]["__folium_color"],
                    "color": x["properties"]["__folium_color"],
                    **style_kwds,
                }
                return {
                    **base_style,
                    **style_kwds_function(x),
                }

            style_function = _style_column

        if highlight:
            if "fillOpacity" not in highlight_kwds:
                highlight_kwds["fillOpacity"] = 0.75

            def _style_highlight(x):
                return {**highlight_kwds}

            highlight_function = _style_highlight
        else:
            highlight_function = None

        # define default for points
        if marker_type is None:
            marker_type = "circle_marker"

        marker = marker_type
        if isinstance(marker_type, str):
            if marker_type == "marker":
                marker = folium.Marker(**marker_kwds)
            elif marker_type == "circle":
                marker = folium.Circle(**marker_kwds)
            elif marker_type == "circle_marker":
                marker_kwds["radius"] = marker_kwds.get("radius", 2)
                marker_kwds["fill"] = marker_kwds.get("fill", True)
                marker = folium.CircleMarker(**marker_kwds)
            else:
                raise ValueError(
                    "Only 'marker', 'circle', and 'circle_marker' are "
                    "supported as marker values"
                )

        # prepare tooltip and popup
        if isinstance(gdf, GeoDataFrame):
            # add named index to the tooltip
            if gdf.index.name is not None:
                gdf = gdf.reset_index()
            # specify fields to show in the tooltip
            tooltip = _tooltip_popup("tooltip", tooltip, gdf, **tooltip_kwds)
            popup = _tooltip_popup("popup", popup, gdf, **popup_kwds)
        else:
            tooltip = None
            popup = None

        gdf_as_json = gdf.__geo_interface__

        return folium.GeoJson(
            gdf_as_json,
            tooltip=tooltip,
            popup=popup,
            marker=marker,
            style_function=style_function,
            highlight_function=highlight_function,
            smooth_factor=self.smooth_factor,
            show=show,
            **kwargs,
        )

    def _image_collection_to_background_map(
        self,
        image_collection: ImageCollection | Image | Band,
        mask: Any | None,
        name: str,
        max_images: int,
        n_added_images: int,
        rbg_bands: list[str] = (("B04", "B02", "B03"), ("B4", "B2", "B3")),
    ) -> tuple[list[dict], int]:
        out = []

        if all(isinstance(x, str) for x in rbg_bands):
            rbg_bands = (rbg_bands,)

        if isinstance(image_collection, ImageCollection):
            images = image_collection.images
            name = None
        elif isinstance(image_collection, Image):
            img = image_collection
            if not _intersects_if_not_none_or_empty(
                mask, img.bounds
            ):  # is not None and not to_shapely(mask).intersects(
                #     to_shapely(img.bounds)
                # ):
                return out, n_added_images

            if len(img) == 1:
                band = next(iter(img))
                raster_data_dict = {}
                out.append(raster_data_dict)
                name = _determine_label(band, name, out, n_added_images)
                _single_band_to_arr(band, mask, name, raster_data_dict)
                n_added_images += 1
                return out, n_added_images
            elif len(img) < 3:
                raster_data_dict = {}
                out.append(raster_data_dict)
                for band in img:
                    name = _determine_label(band, None, out, n_added_images)
                    _single_band_to_arr(band, mask, name, raster_data_dict)
                    n_added_images += 1
                return out, n_added_images
            else:
                images = [image_collection]

        elif isinstance(image_collection, Band):
            band = image_collection

            if not _intersects_if_not_none_or_empty(
                mask, band.bounds
            ):  # mask is not None and not to_shapely(mask).intersects(
                #     to_shapely(band.bounds)
                # ):
                return out, n_added_images

            raster_data_dict = {}
            out.append(raster_data_dict)
            _single_band_to_arr(band, mask, name, raster_data_dict)
            return out, n_added_images

        else:
            raise TypeError(type(image_collection))

        if max(len(out), len(images)) + n_added_images > max_images:
            warnings.warn(
                f"Showing only a sample of {max_images}. Set 'max_images.", stacklevel=1
            )
            self._show_rasters = False
            random.shuffle(images)

            images = images[: (max_images - n_added_images)]
            images = (
                list(sorted([img for img in images if img.date is not None]))
                + sorted(
                    [
                        img
                        for img in images
                        if img.date is None and img.path is not None
                    ],
                    key=lambda x: x.path,
                )
                + [img for img in images if img.date is None and img.path is None]
            )

        for image in images:

            if not _intersects_if_not_none_or_empty(
                mask, image.bounds
            ):  # mask is not None and not to_shapely(mask).intersects(
                #     to_shapely(image.bounds)
                # ):
                continue

            raster_data_dict = {}
            out.append(raster_data_dict)

            if len(image) < 3:
                for band in image:
                    name = _determine_label(band, None, out, n_added_images)
                    _single_band_to_arr(band, mask, name, raster_data_dict)
                    n_added_images += 1
                continue

            def load(band_id: str) -> Band:
                band = image[band_id]
                if band.has_array and mask is not None:
                    band = band.clip(mask, copy=True)
                elif not band.has_array:
                    band = band.load(indexes=1, bounds=mask)
                return band

            for red, blue, green in rbg_bands:
                try:
                    red_band = load(red)
                except KeyError:
                    continue
                try:
                    blue_band = load(blue)
                except KeyError:
                    continue
                try:
                    green_band = load(green)
                except KeyError:
                    continue
                break

            crs = red_band.crs
            bounds = to_bbox(to_gdf(red_band.bounds, crs).to_crs(4326))

            red_band = red_band.values
            blue_band = blue_band.values
            green_band = green_band.values

            if (
                red_band.shape[0] == 0
                or blue_band.shape[0] == 0
                or green_band.shape[0] == 0
            ):
                continue

            # to 3d array in shape (x, y, 3)
            rbg_image = np.stack([red_band, blue_band, green_band], axis=2)

            raster_data_dict["arr"] = rbg_image
            raster_data_dict["bounds"] = bounds
            raster_data_dict["cmap"] = None
            raster_data_dict["label"] = _determine_label(
                image, name, out, n_added_images
            )

            n_added_images += 1

        return out, n_added_images


def _tooltip_popup(
    type_: str, fields: Any, gdf: GeoDataFrame, **kwargs
) -> folium.GeoJsonTooltip | folium.GeoJsonPopup:
    """Get tooltip or popup."""
    # specify fields to show in the tooltip
    if fields is False or fields is None or fields == 0:
        return None
    else:
        if fields is True:
            fields = gdf.columns.drop(gdf.geometry.name).to_list()
        elif isinstance(fields, int):
            fields = gdf.columns.drop(gdf.geometry.name).to_list()[:fields]
        elif isinstance(fields, str):
            fields = [fields]

    for field in ["__plottable_column", "__folium_color"]:
        if field in fields:
            fields.remove(field)

    # Cast fields to str
    fields = list(map(str, fields))
    if type_ == "tooltip":
        return folium.GeoJsonTooltip(fields, **kwargs)
    elif type_ == "popup":
        return folium.GeoJsonPopup(fields, **kwargs)


def _intersects_if_not_none_or_empty(obj: Any, other: Any) -> bool:
    if obj is None:
        return True
    obj = to_shapely(obj)
    if obj is None or obj.is_empty:
        return True
    return obj.intersects(to_shapely(other))


def _determine_label(
    obj: Image | Band | ImageCollection, obj_name: str | None, out: list[dict], i: int
) -> str:
    # Prefer the object's name
    if obj_name:
        # Avoid the generic label e.g. Image(1)
        does_not_have_generic_name = (
            re.sub("(\d+)", "", obj_name) != f"{obj.__class__.__name__}()"
        )
        if does_not_have_generic_name:
            return obj_name
    try:
        # Images/Bands/Collections constructed from arrays have no path stems
        if obj.name:
            name = obj.name
        else:
            name = str(obj)[:23]
    except (AttributeError, ValueError):
        name = str(obj)[:23]

    if name in [x["label"] for x in out if "label" in x]:
        name += f"_{i}"

    return name


def _categorical_legend(
    m: folium.Map, title: str, categories: list[str], colors: list[str]
) -> None:
    """Add categorical legend to a map.

    The implementation is using the code originally written by Michel Metran
    (@michelmetran) and released on GitHub
    (https://github.com/michelmetran/package_folium) under MIT license.

    Copyright (c) 2020 Michel Metran

    Parameters
    ----------
    m : folium.Map
        Existing map instance on which to draw the plot
    title : str
        title of the legend (e.g. column name)
    categories : list-like
        list of categories
    colors : list-like
        list of colors (in the same order as categories)
    """
    # Header to Add
    head = """
    {% macro header(this, kwargs) %}
    <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
    <script>$( function() {
        $( ".maplegend" ).draggable({
            start: function (event, ui) {
                $(this).css({
                    right: "auto",
                    top: "auto",
                    bottom: "auto"
                });
            }
        });
    });
    </script>
    <style type='text/css'>
      .maplegend {
        position: absolute;
        z-index:9999;
        background-color: rgba(255, 255, 255, .8);
        border-radius: 5px;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
        padding: 10px;
        font: 12px/14px Arial, Helvetica, sans-serif;
        right: 10px;
        bottom: 20px;
      }
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 0px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        list-style: none;
        margin-left: 0;
        line-height: 16px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 14px;
        width: 14px;
        margin-right: 5px;
        margin-left: 0;
        border: 0px solid #ccc;
        }
      .maplegend .legend-source {
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    {% endmacro %}
    """
    import branca as bc

    # Add CSS (on Header)
    macro = bc.element.MacroElement()
    macro._template = bc.element.Template(head)
    m.get_root().add_child(macro)

    body = f"""
    <div id='maplegend {title}' class='maplegend'>
        <div class='legend-title'>{title}</div>
        <div class='legend-scale'>
            <ul class='legend-labels'>"""

    # Loop Categories
    for label, color in zip(categories, colors, strict=True):
        body += f"""
                <li><span style='background:{color}'></span>{label}</li>"""

    body += """
            </ul>
        </div>
    </div>
    """

    # Add Body
    body = bc.element.Element(body, "legend")
    m.get_root().html.add_child(body)


def get_textbox(text: str) -> str:
    return f"""
{{% macro html(this, kwargs) %}}
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Textbox Project</title>
    <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.1/css/all.min.css" integrity="sha512-MV7K8+y+gLIBoVD59lQIYicR65iaqukzvf/nwasF0nqhPay5w/9lJmVM2hMDcnK1OnMGCdVK+iQrJ7lzPJQd1w==" crossorigin="anonymous" referrerpolicy="no-referrer"/>
    <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
    <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>

    <script>
      $( function() {{
        $( "#textbox" ).draggable({{
          start: function (event, ui) {{
            $(this).css({{
              right: "auto",
              top: "auto",
              bottom: "auto"
            }});
          }}
        }});
      }});
    </script>
  </head>

  <body>
    <div id="textbox" class="textbox">
      <div class="textbox-content">
        <p>{text}</p>
      </div>
    </div>

</body>
</html>

<style type='text/css'>
  .textbox {{
    position: absolute;
    z-index:9999;
    border-radius:4px;
    background: white;
    padding: 1px;
    font-size:11px;
    left: 20px;
    top: 50%;
    color: black;
  }}
</style>
{{% endmacro %}}
"""

"""Interactive map of one or more GeoDataFrames with layers that can be toggles on/off.

This module holds the Explore class, which is the basis for the explore, samplemap and
clipmap functions from the 'maps' module.
"""
import warnings
from statistics import mean

import branca as bc
import folium
import geopandas
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from mapclassify import classify
from shapely import Geometry
from shapely.geometry import LineString

from .geopandas_tools.general import clean_geoms, gdf_concat
from .helpers import get_name


# the geopandas._explore
warnings.filterwarnings(
    action="ignore", category=matplotlib.MatplotlibDeprecationWarning
)
pd.options.mode.chained_assignment = None


# custom default colors for non-numeric data, because the geopandas default has very
# similar colors. The palette is like the "Set2" cmap from matplotlib, but with more
# colors. If more than 14 categories, the geopandas default cmap is used.
_CATEGORICAL_CMAP = {
    0: "#4576ff",
    1: "#ff4545",
    2: "#59d45f",
    3: "#b51d8b",
    4: "#fc8626",
    5: "#f2dc4e",
    6: "#ff8cc9",
    7: "#6bf2eb",
    8: "#916209",
    9: "#008d94",
    10: "#8a030a",
    11: "#9c65db",
    12: "#228000",
    13: "#80ff00",
}

# gray for NaNs
NAN_COLOR = "#969696"


# cols to not show when hovering over geometries (tooltip)
COLS_TO_DROP = ["color", "geometry"]


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


def _all_are_geom(gdfs: tuple) -> bool:
    """Returns True if all elements in the tuple are geopandas/shapely geometries."""
    return all(isinstance(gdf, (GeoDataFrame, GeoSeries, Geometry)) for gdf in gdfs)


def _separate_args(
    args: tuple,
    column: str | None,
    kwargs: dict,
) -> tuple[tuple[GeoDataFrame], str, dict]:
    """Separate GeoDataFrames from string (column)."""
    if _all_are_geom(args):
        return args, column, kwargs

    gdfs: tuple[GeoDataFrame] = ()
    for arg in args:
        if isinstance(arg, str):
            if column is None:
                column = arg
            else:
                raise ValueError(
                    "Can specify at most one string as a positional argument."
                )
        elif isinstance(arg, (GeoDataFrame, GeoSeries, Geometry)):
            gdfs = gdfs + (arg,)

    if column and not kwargs["column"]:
        kwargs["column"] = column

    return gdfs, column, kwargs


class Explore:
    """Interactive map of GeoDataFrames with layers that can be toggles on/off.

    It takes all the given GeoDataFrames and displays them together in an
    interactive map with a common legend. The layers can be toggled on and off.

    If 'column' is not specified, each GeoDataFrame is given a unique color. The
    default colormap is a custom, strongly colored palette. If a numerical column
    is given, the 'viridis' palette is the default.

    Note:
        The maximum zoom level only works on the OpenStreetMap background map.
    """

    def __init__(
        self,
        *gdfs: GeoDataFrame,
        column: str | None = None,
        labels: tuple[str] | None = None,
        popup: bool = True,
        max_zoom: int = 30,
        show_in_browser: bool = False,
        **kwargs,
    ) -> None:
        """Takes the GeoDataFrames and mapping rules and prepares the mapmaking.

        The maps are displayed with the methods explore, samplemap and clipmap.

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
            popup: If True (default), clicking on a geometry will create a popup box
                with column names and values for the given geometry. The box stays
                until clicking elsewhere. If False (the geopandas default), the box
                will only show when hovering over the geometry.
            max_zoom: The maximum allowed level of zoom. Higher number means more zoom
                allowed. Defaults to 30, which is higher than the geopandas default.
            show_in_browser: If False (the default), the maps will be shown in Jupyter.
                If True the maps will be opened in a browser folder.
            **kwargs: Keyword arguments to pass to geopandas.GeoDataFrame.explore, for
                instance 'cmap' to change the colors, 'scheme' to change how the data
                is grouped. This defaults to 'quantiles' for numeric data.
        """
        self.show_in_browser = show_in_browser
        all_kwargs: dict = kwargs | {
            "popup": popup,
            "column": column,
            "max_zoom": max_zoom,
        }

        gdfs, column, all_kwargs = _separate_args(gdfs, column, all_kwargs)

        if "namedict" in all_kwargs:
            for i, gdf in enumerate(gdfs):
                gdf.name = all_kwargs["namedict"][i]
            all_kwargs.pop("namedict")

        self.gdfs: tuple[GeoDataFrame] = gdfs
        self.labels = labels
        self.kwargs = all_kwargs

        if not self.labels:
            self._create_labels()

        if not self.kwargs["column"]:
            for gdf, label in zip(self.gdfs, self.labels, strict=True):
                gdf["label"] = label
            self.kwargs["column"] = "label"

        self._is_categorical = self._check_if_categorical()
        self._fill_missings()

        self.gdf = gdf_concat(self.gdfs)

        if "title" not in self.kwargs:
            self.kwargs["title"] = self.kwargs["column"]

        if "categories" not in self.kwargs:
            self._choose_cmap()

        if self._is_categorical:
            self._get_categorical_colors()

        self.to_show: tuple[GeoDataFrame] = self.gdfs

    def explore(self, column: str | None = None, **kwargs) -> None:
        """Interactive map of the GeoDataFrames with layers that can be toggles on/off.

        It displays all the GeoDataFrames and displays them together in an interactive
        map with a common legend. The layers can be toggled on and off.

        Args:
            column: The column to color the geometries by. Defaults to the column
                that was specified last.
            **kwargs: Keyword arguments to pass to geopandas.GeoDataFrame.explore, for
                instance 'cmap' to change the colors, 'scheme' to change how the data
                is grouped. This defaults to 'quantiles' for numeric data.

        See also:
            samplemap: same functionality, but shows only a random area of a given size.
            clipmap: same functionality, but shows only the areas clipped by a given
            mask.

        Examples
        --------
        >>> from sgis import read_parquet_url
        >>> roads = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")
        >>> points = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet")

        Simple explore of two GeoDataFrames.

        >>> from sgis import Explore
        >>> ex = Explore(roads, points)
        >>> ex.explore()

        With column.

        >>> roads["meters"] = roads.length
        >>> points["meters"] = points.length
        >>> ex = Explore(roads, points, column="meters")
        >>> ex.samplemap()
        """
        if column:
            kwargs["column"] = column
        self.to_show = self.gdfs
        self._explore(**kwargs)

    def samplemap(self, size: int = 1000, column: str | None = None, **kwargs) -> None:
        """Shows an interactive map of a random area of the GeoDataFrames.

        It takes a random sample point of the GeoDataFrames, and shows all geometries
        within a given radius of this point. Displays an interactive map with a common
        legend. The layers can be toggled on and off.

        The radius to plot can be changed with the 'size' parameter.

        For more info about the labeling and coloring of the map, see the explore
        method.

        Args:
            size: the radius to buffer the sample point by before clipping with the data
            column: The column to color the geometries by. Defaults to the column
                that was specified last.
            **kwargs: Keyword arguments to pass to geopandas.GeoDataFrame.explore, for
                instance 'cmap' to change the colors, 'scheme' to change how the data
                is grouped. This defaults to 'quantiles' for numeric data.

        See also:
            explore: same functionality, but shows the entire area of the geometries.
            clipmap: same functionality, but shows only the areas clipped by a given
            mask.
        """
        if column:
            kwargs["column"] = column
        self.previous_sample_count = 0
        self.to_show = self.gdfs
        random_point = self.gdf.sample(1).assign(geometry=lambda x: x.centroid)
        to_show: tuple[GeoDataFrame] = ()
        for gdf in self.to_show:
            gdf = gdf.clip(random_point.buffer(size))
            to_show = to_show + (gdf,)
        self.to_show = to_show
        self._explore(**kwargs)

    def clipmap(
        self,
        mask,
        column: str | None = None,
        **kwargs,
    ) -> None:
        """Shows an interactive map of a of the GeoDataFrames clipped by the mask.

        It clips all the GeoDataFrames in the Explore instance to the mask extent,
        and displays the resulting geometries in an interactive map with a common
        legends. The layers can be toggled on and off.

        For more info about the labeling and coloring of the map, see the explore
        method.

        Args:
            mask: the geometry to clip the data by.
            column: The column to color the geometries by. Defaults to the column
                that was specified last.
            **kwargs: Keyword arguments to pass to geopandas.GeoDataFrame.explore, for
                instance 'cmap' to change the colors, 'scheme' to change how the data
                is grouped. This defaults to 'quantiles' for numeric data.

        See also:
            explore: same functionality, but shows the entire area of the geometries.
            samplemap: same functionality, but shows only a random area of a given size.
        """
        if column:
            kwargs["column"] = column
        to_show: tuple[GeoDataFrame] = ()
        for gdf in self.gdfs:
            gdf = gdf.clip(mask)
            to_show = to_show + (gdf,)
        self.to_show = to_show
        self._explore(**kwargs)

    def _explore(self, **kwargs):
        self.kwargs = self.kwargs | kwargs
        self._is_categorical = self._check_if_categorical()
        if self._is_categorical:
            self._create_categorical_map()
        else:
            self._create_continous_map()
        if self.show_in_browser:
            self.map.show_in_browser()
        else:
            display(self.map)

    def _create_labels(self) -> None:
        self.labels: list[str] = []
        for i, gdf in enumerate(self.gdfs):
            if hasattr(gdf, "name"):
                name = gdf.name
            else:
                name = get_name(gdf)
                if not name:
                    name = str(i)
            self.labels.append(name)
            gdf["label"] = name

    def _fill_missings(self) -> None:
        for gdf in self.gdfs:
            if self.kwargs["column"] in gdf.columns:
                continue
            if self._is_categorical:
                gdf[self.kwargs["column"]] = "missing"
            else:
                gdf[self.kwargs["column"]] = np.nan

    def _check_if_categorical(self) -> bool:
        if not self.kwargs["column"]:
            return True
        n_categorical = 0
        for gdf in self.gdfs:
            if self.kwargs["column"] not in gdf:
                continue
            if not pd.api.types.is_numeric_dtype(gdf[self.kwargs["column"]]):
                n_categorical += 1
        if n_categorical:
            return True
        return False

    def _choose_cmap(self) -> None:
        if "cmap" not in self.kwargs:
            if self._is_categorical:
                self.kwargs["cmap"] = "tab20"
            else:
                self.kwargs["cmap"] = "viridis"
        if "scheme" not in self.kwargs:
            self.kwargs["scheme"] = "quantiles"

    def _get_categorical_colors(self) -> None:
        cat_col = self.kwargs["column"]
        self._unique_categories = sorted(
            list(self.gdf.loc[self.gdf[cat_col] != "missing", cat_col].unique())
        )
        if len(self._unique_categories) <= len(_CATEGORICAL_CMAP):
            self.kwargs["cmap"] = None
            self._categories_colors_dict = {
                category: _CATEGORICAL_CMAP[i]
                for i, category in enumerate(self._unique_categories)
            }
        else:
            cmap = matplotlib.colormaps.get_cmap(self.kwargs["cmap"])

            self._categories_colors_dict = {
                category: colors.to_hex(cmap(int(i)))
                for i, category in enumerate(self._unique_categories)
            }

        if any(self.gdf[self.kwargs["column"]].isna()) or any(
            self.gdf[self.kwargs["column"]] == "missing"
        ):
            self._categories_colors_dict["missing"] = NAN_COLOR

        for gdf in self.gdfs:
            gdf["color"] = gdf[self.kwargs["column"]].map(self._categories_colors_dict)

        self.gdf["color"] = self.gdf[self.kwargs["column"]].map(
            self._categories_colors_dict
        )

    def _create_categorical_map(self):
        gdfs = gdf_concat(self.to_show)

        self.map = self._explore_return(gdfs, return_="empty_map", **self.kwargs)

        for gdf, label in zip(self.to_show, self.labels, strict=True):
            if not len(gdf):
                continue
            f = folium.FeatureGroup(name=label)
            gjs = self._explore_return(
                gdf,
                color=gdf["color"],
                return_="geojson",
                tooltip=self._tooltip_cols(gdf),
                **{
                    key: value
                    for key, value in self.kwargs.items()
                    if key not in ["title", "cmap"]
                },
            )
            f.add_child(gjs)
            self.map.add_child(f)
        _categorical_legend(
            self.map,
            self.kwargs["title"],
            self._categories_colors_dict.keys(),
            self._categories_colors_dict.values(),
        )
        folium.TileLayer("stamentoner").add_to(self.map)
        folium.TileLayer("cartodbdark_matter").add_to(self.map)
        self.map.add_child(folium.LayerControl())

    def _create_continous_map(self):
        gdfs = gdf_concat(self.to_show)

        bins = self._create_bins(gdfs, self.kwargs["column"], self.kwargs["scheme"])
        self.kwargs["classification_kwds"] = {"bins": bins}
        self.map, colorbar = self._explore_return(
            gdfs, return_="empty_map_and_colorbar", **self.kwargs
        )

        for gdf, label in zip(self.to_show, self.labels, strict=True):
            if not len(gdf):
                continue
            f = folium.FeatureGroup(name=label)
            gjs = self._explore_return(
                gdf,
                tooltip=self._tooltip_cols(gdf),
                return_="geojson",
                **{key: value for key, value in self.kwargs.items() if key != "title"},
            )
            f.add_child(gjs)
            self.map.add_child(f)
        self.map.add_child(colorbar)
        folium.TileLayer("stamentoner").add_to(self.map)
        folium.TileLayer("cartodbdark_matter").add_to(self.map)
        self.map.add_child(folium.LayerControl())

    def _tooltip_cols(self, gdf: GeoDataFrame) -> list:
        if "tooltip" in self.kwargs:
            tooltip = self.kwargs["tooltip"]
            self.kwargs.pop("tooltip")
            print(tooltip)
            print([col for col in gdf.columns if col not in COLS_TO_DROP])
            return tooltip
        return [col for col in gdf.columns if col not in COLS_TO_DROP]

    def _create_bins(self, gdf, column, scheme):
        binning = classify(
            np.asarray(gdf.loc[gdf[column].notna(), column]), scheme=scheme
        )
        return list(binning.bins)

    def _explore_return(
        self,
        df,
        return_: str,
        column=None,
        cmap=None,
        color=None,
        m=None,
        tiles="OpenStreetMap",
        attr=None,
        tooltip=True,
        popup=False,
        highlight=True,
        legend=True,
        scheme=None,
        k=5,
        vmin=None,
        vmax=None,
        width="100%",
        height="100%",
        categories=None,
        classification_kwds=None,
        control_scale=True,
        marker_type=None,
        marker_kwds={},
        style_kwds={},
        highlight_kwds={},
        missing_kwds={},
        tooltip_kwds={},
        popup_kwds={},
        legend_kwds={},
        map_kwds={},
        **kwargs,
    ):
        """Contains the nessecary parts of the geopandas _explore function.

        Also has a return_ parameter that controls what is returned. This should be
        replaced by separate functions, and irrelevant parameters should be removed.
        """
        # xyservices is an optional dependency
        try:
            import xyzservices

            has_xyzservices = True
        except ImportError:
            has_xyzservices = False

        gdf = df.copy()

        # convert LinearRing to LineString
        rings_mask = df.geom_type == "LinearRing"
        if rings_mask.any():
            gdf.geometry[rings_mask] = gdf.geometry[rings_mask].apply(
                lambda g: LineString(g)
            )

        if gdf.crs is None:
            kwargs["crs"] = "Simple"
            tiles = None
        elif not gdf.crs.equals(4326):
            gdf = gdf.to_crs(4326)

        # create folium.Map object
        if m is None:
            # Get bounds to specify location and map extent
            bounds = gdf.total_bounds
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

            if has_xyzservices:
                # match provider name string to xyzservices.TileProvider
                if isinstance(tiles, str):
                    try:
                        tiles = xyzservices.providers.query_name(tiles)
                    except ValueError:
                        pass

                if isinstance(tiles, xyzservices.TileProvider):
                    attr = attr if attr else tiles.html_attribution
                    map_kwds["min_zoom"] = tiles.get("min_zoom", 0)
                    map_kwds["max_zoom"] = tiles.get("max_zoom", 18)
                    tiles = tiles.build_url(scale_factor="{r}")

            m = folium.Map(
                location=location,
                control_scale=control_scale,
                tiles=tiles,
                attr=attr,
                width=width,
                height=height,
                **map_kwds,
            )

            # fit bounds to get a proper zoom level
            if fit:
                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        for map_kwd in _MAP_KWARGS:
            kwargs.pop(map_kwd, None)

        if pd.api.types.is_list_like(column):
            if len(column) != gdf.shape[0]:
                raise ValueError(
                    "The GeoDataFrame and given column have different number of rows."
                )
            else:
                column_name = "__plottable_column"
                gdf[column_name] = column
                column = column_name
        elif pd.api.types.is_categorical_dtype(gdf[column]):
            if categories is not None:
                raise ValueError(
                    "Cannot specify 'categories' when column has categorical dtype"
                )

        nan_idx = pd.isna(gdf[column])

        if not self._is_categorical:
            vmin = gdf[column].min() if vmin is None else vmin
            vmax = gdf[column].max() if vmax is None else vmax

            if len(gdf[column][~nan_idx]):
                bins = classification_kwds["bins"]

                binning = classify(
                    np.asarray(gdf[column][~nan_idx]), "UserDefined", bins=bins
                )

                color = np.apply_along_axis(
                    colors.to_hex,
                    1,
                    cm.get_cmap(cmap, k)(binning.yb),  # ! changed 256 to k
                )

            else:
                color = NAN_COLOR

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

        # specify color
        if color is not None:
            if (
                isinstance(color, str)
                and isinstance(gdf, geopandas.GeoDataFrame)
                and color in gdf.columns
            ):  # use existing column

                def _style_color(x):
                    base_style = {
                        "fillColor": x["properties"][color],
                        **style_kwds,
                    }
                    return {
                        **base_style,
                        **style_kwds_function(x),
                    }

                style_function = _style_color
            else:  # assign new column
                if isinstance(gdf, GeoSeries):
                    gdf = GeoDataFrame(geometry=gdf)

                if nan_idx is not None and nan_idx.any():
                    nan_color = missing_kwds.pop("color", NAN_COLOR)

                    gdf["__folium_color"] = nan_color
                    gdf.loc[~nan_idx, "__folium_color"] = color
                else:
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
                else:

                    def _style_stroke(x):
                        base_style = {
                            "fillColor": x["properties"]["__folium_color"],
                            "color": stroke_color,
                            **style_kwds,
                        }
                        return {
                            **base_style,
                            **style_kwds_function(x),
                        }

                    style_function = _style_stroke
        else:  # use folium default

            def _style_default(x):
                return {**style_kwds, **style_kwds_function(x)}

            style_function = _style_default

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

        # remove additional geometries
        if isinstance(gdf, GeoDataFrame):
            non_active_geoms = [
                name
                for name, val in (gdf.dtypes == "geometry").items()
                if val and name != gdf.geometry.name
            ]
            gdf = gdf.drop(columns=non_active_geoms)

        gdf = clean_geoms(gdf)

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

        if "geojson" in return_:
            # add dataframe to map
            gjs = folium.GeoJson(
                gdf.__geo_interface__,
                tooltip=tooltip,
                popup=popup,
                marker=marker,
                style_function=style_function,
                highlight_function=highlight_function,
                **kwargs,
            )
            return gjs

        if legend and not self._is_categorical:
            # NOTE: overlaps will be resolved in branca #88
            caption = column if column != "__plottable_column" else ""
            caption = legend_kwds.pop("caption", caption)
            if column is not None:
                cbar = legend_kwds.pop("colorbar", True)
                colormap_kwds = {}
                if "max_labels" in legend_kwds:
                    colormap_kwds["max_labels"] = legend_kwds.pop("max_labels")
                if scheme:
                    cb_colors = np.apply_along_axis(
                        colors.to_hex, 1, cm.get_cmap(cmap, binning.k)(range(binning.k))
                    )
                    if cbar:
                        if legend_kwds.pop("scale", True):
                            index = [vmin] + binning.bins.tolist()
                        else:
                            index = None
                        colorbar = bc.colormap.StepColormap(
                            cb_colors,
                            vmin=vmin,
                            vmax=vmax,
                            caption=caption,
                            index=index,
                            **colormap_kwds,
                        )
                    else:
                        fmt = legend_kwds.pop("fmt", "{:.2f}")
                        if "labels" in legend_kwds:
                            categories = legend_kwds["labels"]
                        else:
                            categories = binning.get_legend_classes(fmt)
                            show_interval = legend_kwds.pop("interval", False)
                            if not show_interval:
                                categories = [c[1:-1] for c in categories]

                        if nan_idx.any() and nan_color:
                            categories.append(missing_kwds.pop("label", "NaN"))
                            cb_colors = np.append(cb_colors, nan_color)
                        # _categorical_legend(m, caption, categories, cb_colors)

                else:
                    if isinstance(cmap, bc.colormap.ColorMap):
                        colorbar = cmap
                    else:
                        mp_cmap = cm.get_cmap(cmap)
                        cb_colors = np.apply_along_axis(
                            colors.to_hex, 1, mp_cmap(range(mp_cmap.N))
                        )

                        # linear legend
                        if mp_cmap.N > 20:
                            colorbar = bc.colormap.LinearColormap(
                                cb_colors,
                                vmin=vmin,
                                vmax=vmax,
                                caption=caption,
                                **colormap_kwds,
                            )

                        # steps
                        else:
                            colorbar = bc.colormap.StepColormap(
                                cb_colors,
                                vmin=vmin,
                                vmax=vmax,
                                caption=caption,
                                **colormap_kwds,
                            )

        if return_ == "empty_map":
            return m

        if return_ == "empty_map_and_colorbar":
            return m, colorbar

        if return_ == "map":
            m.add_child(gjs)
            if legend and cbar:
                if nan_idx.any() and nan_color:
                    _categorical_legend(
                        m, "", [missing_kwds.pop("label", "NaN")], [nan_color]
                    )
                m.add_child(colorbar)
            return m


def _tooltip_popup(type, fields, gdf, **kwds):
    """get tooltip or popup"""
    import folium

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
    if type == "tooltip":
        return folium.GeoJsonTooltip(fields, **kwds)
    elif type == "popup":
        return folium.GeoJsonPopup(fields, **kwds)


def _categorical_legend(m, title, categories, colors):
    """
    Add categorical legend to a map

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

"""Interactive map of one or more GeoDataFrames with layers that can be toggles on/off.

This module holds the Explore class, which is the basis for the explore, samplemap and
clipmap functions from the 'maps' module.
"""
import warnings
from statistics import mean

import branca as bc
import folium
import matplotlib
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import LineString

from ..geopandas_tools.general import clean_geoms, random_points_in_polygons
from ..geopandas_tools.geometry_types import get_geom_type
from .map import Map


# the geopandas._explore raises a deprication warning. Ignoring for now.
warnings.filterwarnings(
    action="ignore", category=matplotlib.MatplotlibDeprecationWarning
)
pd.options.mode.chained_assignment = None


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


class Explore(Map):
    def __init__(
        self,
        *gdfs,
        column: str | None = None,
        popup: bool = True,
        max_zoom: int = 30,
        show_in_browser: bool = False,
        **kwargs,
    ):
        super().__init__(*gdfs, column=column, **kwargs)
        self.popup = popup
        self.max_zoom = max_zoom
        self.show_in_browser = show_in_browser

        if not self._is_categorical:
            if not self._cmap:
                self._cmap = "viridis"
            self.cmap_start = kwargs.pop("cmap_start", 0)
            self.cmap_stop = kwargs.pop("cmap_stop", 256)

    def explore(self, column: str | None = None, **kwargs) -> None:
        """Interactive map of the GeoDataFrames with layers that can be toggles on/off.

        It displays all the GeoDataFrames and displays them together in an interactive
        map with a common legend. The layers can be toggled on and off.

        Args:
            column: The column to color the geometries by. Defaults to the column
                that was specified last.
            **kwargs: Keyword arguments to pass to geopandas.GeoDataFrame.explore, for
                instance 'cmap' to change the colors, 'scheme' to change how the data
                is grouped. This defaults to 'fisherjenks' for numeric data.

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
            self._column = column
            self._update_column()
            kwargs.pop("column", None)
        self.to_show = self._gdfs
        self._explore(**kwargs)

    def samplemap(
        self,
        size: int = 1000,
        column: str | None = None,
        sample_from_first: bool = True,
        **kwargs,
    ) -> None:
        """Shows an interactive map of a random area of the GeoDataFrames.

        It takes a random sample point of the GeoDataFrames, and shows all geometries
        within a given radius of this point. Displays an interactive map with a common
        legend. The layers can be toggled on and off.

        The radius to plot can be changed with the 'size' parameter.

        For more info about the labeling and coloring of the map, see the explore
        method.

        Args:
            size: the radius to buffer the sample point by before clipping with the
                data.
            column: The column to color the geometries by. Defaults to the column
                that was specified last.
            sample_from_first: If True (default), the sample point is taken from
                the first specified GeoDataFrame. If False, all GeoDataFrames are
                considered.
            **kwargs: Keyword arguments to pass to geopandas.GeoDataFrame.explore, for
                instance 'cmap' to change the colors, 'scheme' to change how the data
                is grouped. This defaults to 'fisherjenks' for numeric data.

        See also:
            explore: same functionality, but shows the entire area of the geometries.
            clipmap: same functionality, but shows only the areas clipped by a given
            mask.
        """
        if column:
            self._column = column
            self._update_column()
            kwargs.pop("column", None)

        self.previous_sample_count = 0

        if sample_from_first:
            sample = self._gdfs[0].sample(1)
        else:
            sample = self._gdf.sample(1)

        # convert lines to polygons
        if get_geom_type(sample) == "line":
            sample["geometry"] = sample.buffer(1)

        if get_geom_type(sample) == "polygon":
            random_point = random_points_in_polygons(sample, 1)

        # if point or mixed geometries
        else:
            random_point = sample.centroid

        gdfs: tuple[GeoDataFrame] = ()
        for gdf in self._gdfs:
            gdf = gdf.clip(random_point.buffer(size))
            gdfs = gdfs + (gdf,)
        self._gdfs = gdfs
        self._gdf = pd.concat(gdfs, ignore_index=True)
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
                is grouped. This defaults to 'fisherjenks' for numeric data.

        See also:
            explore: same functionality, but shows the entire area of the geometries.
            samplemap: same functionality, but shows only a random area of a given size.
        """
        if column:
            self._column = column
            self._update_column()
            kwargs.pop("column", None)

        gdfs: tuple[GeoDataFrame] = ()
        for gdf in self._gdfs:
            gdf = gdf.clip(mask)
            gdfs = gdfs + (gdf,)
        self._gdfs = gdfs
        self._gdf = pd.concat(gdfs, ignore_index=True)
        self._explore(**kwargs)

    def _update_column(self):
        self._is_categorical = self._check_if_categorical()
        self._fill_missings()
        self._gdf = pd.concat(self._gdfs, ignore_index=True)

    def _explore(self, **kwargs):
        self.kwargs = self.kwargs | kwargs

        if self._is_categorical:
            self._create_categorical_map()
        else:
            self._create_continous_map()

        if self.show_in_browser:
            self.map.show_in_browser()
        else:
            display(self.map)

    def _create_categorical_map(self):
        self._get_categorical_colors()

        self.map = self._explore_return(self._gdf, return_="empty_map", **self.kwargs)

        for gdf, label in zip(self._gdfs, self.labels, strict=True):
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
                    if key not in ["title"]
                },
            )
            f.add_child(gjs)
            self.map.add_child(f)
        _categorical_legend(
            self.map,
            self._column,
            self._categories_colors_dict.keys(),
            self._categories_colors_dict.values(),
        )
        folium.TileLayer("stamentoner").add_to(self.map)
        folium.TileLayer("cartodbdark_matter").add_to(self.map)
        self.map.add_child(folium.LayerControl())

    def _create_continous_map(self):
        self._prepare_continous_map()
        self.colorlist = self._get_continous_colors()
        self.colors = self._classify_from_bins(self._gdf)

        self.map = self._explore_return(
            self._gdf,
            return_="empty_map",
            **self.kwargs,
        )

        colorbar = bc.colormap.StepColormap(
            self.colorlist,
            vmin=self._gdf[self._column].min(),
            vmax=self._gdf[self._column].max(),
            caption=self._column,
            index=self.bins,
            #  **colormap_kwds,
        )

        for gdf, label in zip(self._gdfs, self.labels, strict=True):
            if not len(gdf):
                continue
            f = folium.FeatureGroup(name=label)

            self.colors = self._classify_from_bins(gdf)

            gjs = self._explore_return(
                gdf,
                tooltip=self._tooltip_cols(gdf),
                color=self.colors,
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
            tooltip = self.kwargs.pop("tooltip")
            return tooltip
        return [col for col in gdf.columns if col not in COLS_TO_DROP]

    def _explore_return(
        self,
        df,
        return_: str,
        column=None,
        color=None,
        attr=None,
        tiles="OpenStreetMap",
        tooltip=True,
        popup=False,
        highlight=True,
        width="100%",
        height="100%",
        control_scale=True,
        marker_type=None,
        marker_kwds={},
        style_kwds={},
        highlight_kwds={},
        tooltip_kwds={},
        popup_kwds={},
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
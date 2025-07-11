import uuid
import itertools
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import dash
import numpy as np
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import pandas as pd
import shapely
from daplapath import LocalFileSystem
from dash import Dash
from dash import Input
from dash import Output
from dash import State
from dash import ctx
from dash import callback
from dash import dcc
from dash import html
from dash import dash_table
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from jenkspy import jenks_breaks
import matplotlib


import sgis as sg

PORT: int = 8055
BASE_DIR = "/buckets/produkt"
BASE_DIR = "/buckets/produkt/strandsone/klargjorte-data/2024/strandsone_kode_p2024_v1.parquet/komm_nr=0301"
BASE_DIR = "/buckets/delt-kart/analyse_data/klargjorte-data/2025"
NAN_COLOR = "#969696"
NAN_LABEL = "Missing"

BASE_LAYERS = [
    dl.BaseLayer(
        dl.TileLayer("OpenStreetMap"),
        name="OpenStreetMap",
        checked=True,
    ),
    dl.BaseLayer(
        dl.TileLayer(
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
            attribution='&copy; <a href="https://carto.com/">CARTO</a>',
        ),
        name="CartoDB Dark Matter",
        checked=False,
    ),
    dl.BaseLayer(
        dl.TileLayer(
            url="https://opencache.statkart.no/gatekeeper/gk/gk.open_nib_web_mercator_wmts_v2?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=Nibcache_web_mercator_v2&STYLE=default&FORMAT=image/jpgpng&tileMatrixSet=default028mm&tileMatrix={z}&tileRow={y}&tileCol={x}",
            attribution="© Geovekst",
        ),
        name="Norge i bilder",
        checked=False,
    ),
]

# BASE_DIR = "c:/users/ort"


def nested_bounds_to_bounds(
    bounds: list[list[float]],
) -> tuple[float, float, float, float]:
    if bounds is None:
        bounds = default_bounds
    mins, maxs = bounds
    miny, minx = mins
    maxy, maxx = maxs
    return minx, miny, maxx, maxy


def read_files(exp, paths):
    read_func = partial(sg.read_geopandas, file_system=exp.file_system)
    with ThreadPoolExecutor() as executor:
        more_data = list(executor.map(read_func, paths))
    for path, df in zip(paths, more_data, strict=True):
        exp.data[path] = df.to_crs(4326).assign(
            _uuid=lambda x: [str(uuid.uuid4()) for _ in range(len(x))]
        )


def bounds_to_nested_bounds(
    bounds: tuple[float, float, float, float],
) -> list[list[float]]:
    minx, miny, maxx, maxy = bounds
    return [[miny, minx], [maxy, maxx]]


def dict_to_geopandas(dict_):
    geometry = dict_.pop("geometry")
    crs = dict_.pop("crs")
    geometry = GeoSeries.from_wkt(geometry, crs=crs)
    return GeoDataFrame(dict_, geometry=geometry, crs=crs)


def random_color():
    r, g, b = np.random.choice(range(256), size=3)
    return f"#{r:02x}{g:02x}{b:02x}"


if __name__ == "__main__":

    norge_i_bilder = False

    class ExploreApp:

        def __init__(
            self,
            base_dir: str = BASE_DIR,
            port=8055,
            paths: list[str] | None = None,
            column: str | None = None,
            # bounds=(
            #     9.858855440173372,
            #     59.62124229424823,
            #     11.563109590563998,
            #     60.207757877310925,
            # ),
            center=(59.91740845, 10.71394444),
            zoom: int = 10,
            nan_color: str = NAN_COLOR,
            nan_label: str = NAN_LABEL,
            file_system=LocalFileSystem(),
        ):
            self.base_dir = base_dir
            self.port = port
            self.center = center
            self.zoom = zoom
            self.column = column
            self.file_system = file_system
            self.nan_color = nan_color
            self.nan_label = nan_label
            self.file_system = file_system
            self.currently_in_bounds: set[str] = set()
            self.bounds_series = GeoSeries()
            self.data: dict[str, GeoDataFrame] = {}
            self.paths: list[str] = []

            if not paths:
                return

            for path in paths:
                self.append(path)

            read_files(self, paths)

        def append(self, path):
            print("append", path)
            self.paths.append(path)
            suffix = Path(path).suffix
            if suffix:
                paths = list(self.file_system.glob(str(Path(path) / f"**/*{suffix}")))
                if not paths:
                    paths = [path]
            else:
                paths = [path]
            self.bounds_series = pd.concat(
                [
                    self.bounds_series,
                    sg.get_bounds_series(paths, file_system=self.file_system).to_crs(
                        4326
                    ),
                ]
            )

        def __str__(self) -> str:
            def to_string(x):
                if isinstance(x, str):
                    return f"'{x}'"
                return x

            txt = ", ".join(
                [
                    f"{k}={to_string(v)}"
                    for k, v in self.__dict__.items()
                    if k
                    not in [
                        "paths",
                        "currently_in_bounds",
                        "bounds_series",
                        "data",
                    ]
                ]
            )
            return f"{self.__class__.__name__}({txt})"

    exp = ExploreApp(
        paths=[
            "/buckets/delt-kart/analyse_data/klargjorte-data/2025/FKB_arealbruk_flate_p2025_v1.parquet"
        ],
        column="objtype",
        zoom=15,
        center=(59.91740845, 10.71394444),
        file_system=LocalFileSystem(),
    )
    print(exp)

    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        external_stylesheets=[dbc.themes.SOLAR],
        requests_pathname_prefix=f"/proxy/{PORT}/",
        serve_locally=True,
        assets_folder="assets",
    )
    # default_center = [59.91740845, 10.71394444]
    default_bounds = [
        [59.62124229424823, 9.858855440173372],
        [60.207757877310925, 11.563109590563998],
    ]

    app.layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dl.Map(
                                center=exp.center,
                                zoom=exp.zoom,
                                # bounds=(
                                #     bounds_to_nested_bounds(exp.bounds)
                                #     if len(exp.bounds) == 4
                                #     else exp.bounds
                                # ),
                                children=[
                                    dl.LayersControl(BASE_LAYERS, id="lc"),
                                    dl.ScaleControl(position="bottomleft"),
                                    dl.MeasureControl(
                                        position="bottomright",
                                        primaryLengthUnit="meters",
                                    ),
                                ],
                                id="map",
                                style={"width": "100%", "height": "90vh"},
                            ),
                        ],
                        width=8,
                    ),
                    dbc.Col(
                        [
                            dbc.Row(
                                html.Button(
                                    "Split",
                                    style={
                                        "fillColor": "white",
                                        "color": "black",
                                    },
                                ),
                                id="splitter",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            [
                                                dcc.Dropdown(
                                                    id="column-dropdown",
                                                    placeholder="Select column to color by",
                                                    style={
                                                        "font-size": 22,
                                                        "width": "100%",
                                                        "overflow": "visible",
                                                    },
                                                    maxHeight=600,
                                                    clearable=True,
                                                ),
                                            ],
                                        ),
                                        width=10,
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            id="force-categorical",
                                        ),
                                        width=2,
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id="k",
                                            options=[
                                                {"label": f"k={i}", "value": i}
                                                for i in [3, 4, 5, 6, 7, 8, 9]
                                            ],
                                            value=5,
                                            style={
                                                "font-size": 22,
                                                "width": "100%",
                                                "overflow": "visible",
                                            },
                                            maxHeight=300,
                                            clearable=False,
                                        )
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            dcc.Dropdown(
                                                id="cmap-placeholder",
                                                options=[
                                                    {
                                                        "label": f"cmap={name}",
                                                        "value": name,
                                                    }
                                                    for name in [
                                                        "viridis",
                                                        "plasma",
                                                        "inferno",
                                                        "magma",
                                                        "Greens",
                                                    ]
                                                    + [
                                                        name
                                                        for name, cmap in matplotlib.colormaps.items()
                                                        if "linear" in str(cmap).lower()
                                                    ]
                                                ],
                                                value="viridis",
                                                # style={
                                                #     "font-size": 22,
                                                #     "width": "100%",
                                                #     # "overflow": "scroll",
                                                # },
                                                maxHeight=200,
                                                clearable=False,
                                            ),
                                            # dbc.Input(value="viridis", id="cmap"),
                                            # id="cmap-placeholder",
                                        )
                                    ),
                                ]
                            ),
                            dbc.Row(id="column-value-colors"),
                            dbc.Row(
                                html.Div(id="remove-buttons"),
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            [
                                                html.Button(
                                                    "Export",
                                                    id="export",
                                                    style={
                                                        "color": "blue",
                                                        # "border": "none",
                                                        # "background": "none",
                                                        # "cursor": "pointer",
                                                    },
                                                ),
                                                html.Div(
                                                    id="export-view",
                                                    style={"display": "none"},
                                                ),
                                            ]
                                        )
                                    ),
                                ]
                            ),
                        ],
                    ),
                ],
                style={
                    "height": "100vh",
                    "overflow": "scroll",
                },
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Button(
                            "❌ Clear table",
                            id="clear-table",
                            style={
                                "color": "red",
                                "border": "none",
                                "background": "none",
                                "cursor": "pointer",
                            },
                        ),
                        width=1,
                    ),
                    dbc.Col(
                        html.Div(id="feature-table-container"),
                        style={"width": "100%", "height": "100vh"},
                        width=11,
                    ),
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H2("File Browser"),
                                html.Button("⬆️ Go Up", id="up-button"),
                                dcc.Store(id="current-path", data=BASE_DIR),
                                dcc.Input(
                                    BASE_DIR,
                                    id="path-display",
                                    style={
                                        "width": "70%",
                                    },
                                ),
                                html.Div(
                                    id="file-list",
                                    style={
                                        "font-size": 12,
                                        "width": "100%",
                                        "height": "70vh",
                                        "overflow": "scroll",
                                    },
                                ),
                            ]
                        ),
                        # width=4,
                    ),
                ]
            ),
            html.Div(
                id="custom-popup",
                children=[
                    html.Div(
                        [
                            html.Button(
                                "×",
                                id="close-popup",
                                n_clicks=0,
                                style={
                                    "float": "right",
                                    "fontSize": "20px",
                                    "border": "none",
                                    "background": "white",
                                    "cursor": "pointer",
                                },
                            ),
                            html.H4("Copy map"),
                        ],
                        id="popup-header",
                        style={
                            "cursor": "move",
                            "padding": "10px",
                            "background": "#0E2A30",
                            "borderBottom": "1px solid #ccc",
                        },
                    ),
                    html.Div(id="pixel-check-map", style={"padding": "10px"}),
                ],
                style={
                    "display": "none",
                    "position": "fixed",
                    "bottom": "10px",
                    "right": "10px",
                    "zIndex": 9999,
                    "background": "#0E2A30",
                    "border": "2px solid #888",
                    "boxShadow": "2px 2px 10px rgba(0,0,0,0.2)",
                    "borderRadius": "8px",
                    "height": "40vh",  # Let height grow with content
                    "width": "125vh",  # Let height grow with content
                    "overflow": "visible",  # Scroll if content is too big
                },
            ),
            html.Button(id="dummy-button", style={"display": "none"}),
            dcc.Store(id="js_init_store", data=False),
            dcc.Store(id="js_init_store2", data=False),
            html.Div(id="currently-in-bounds", style={"display": "none"}),
            html.Div(id="currently-in-bounds2", style={"display": "none"}),
            html.Div(id="new-file-added", style={"display": "none"}),
            html.Div(id="file-removed", style={"display": "none"}),
            html.Div(id="column-value-color-dict", style={"display": "none"}),
            html.Div(id="bins", style={"display": "none"}),
            html.Div(False, id="is-numeric", style={"display": "none"}),
            # html.Div(False, id="cmap-has-been-set", style={"display": "none"}),
            dcc.Store(id="clicked-features", data=[]),
        ],
        fluid=True,
    )


def get_name(path):
    return Path(path).stem


def list_dir(path):
    print("list dir")
    items = os.listdir(path)
    items.sort()
    return html.Ul(
        [
            html.Li(
                [
                    (
                        html.Button(
                            "Load",
                            id={
                                "type": "load-parquet",
                                "index": os.path.join(path, item),
                            },
                            n_clicks=0,
                            style={"marginLeft": "10px"},
                        )
                        if not os.path.isdir(os.path.join(path, item))
                        or (
                            item.endswith(".parquet")
                            and os.path.isdir(os.path.join(path, item))
                        )
                        else None
                    ),
                    html.A(
                        (
                            f"[DIR] {item}"
                            if os.path.isdir(os.path.join(path, item))
                            else item
                        ),
                        href=("#" if os.path.isdir(os.path.join(path, item)) else None),
                        id={"type": "file-item", "index": os.path.join(path, item)},
                        n_clicks=0,
                    ),
                ]
            )
            for item in items
        ]
    )


@callback(
    Output("export-view", "children"),
    Output("export-view", "style"),
    Input("export", "n_clicks"),
    Input("file-removed", "children"),
    State("map", "bounds"),
    State("map", "zoom"),
)
def export(n_clicks, remove, bounds, zoom):
    triggered = dash.callback_context.triggered_id
    if triggered == "file-removed":
        return None, None

    bounds = nested_bounds_to_bounds(bounds)

    def to_string(x):
        if isinstance(x, str):
            return f"'{x}'"
        return x

    data = {
        k: v
        for k, v in exp.__dict__.items()
        if k
        not in [
            "paths",
            "currently_in_bounds",
            "bounds_series",
            "data",
        ]
    } | {"zoom": zoom, "bounds": bounds}
    txt = ", ".join([f"{k}={to_string(v)}" for k, v in data.items()])
    return f"{exp.__class__.__name__}({txt})", None


@callback(
    # Output("path-display", "value"),
    Output("file-list", "children"),
    Input("current-path", "data"),
)
def update_file_list(path):
    print("update_file_list")
    return list_dir(path)


@callback(
    Output("current-path", "data"),
    Output("path-display", "value"),
    Input({"type": "file-item", "index": dash.ALL}, "n_clicks"),
    Input("up-button", "n_clicks"),
    Input("path-display", "value"),
    State({"type": "file-item", "index": dash.ALL}, "id"),
    State("current-path", "data"),
    prevent_initial_call=True,
)
def handle_click(load_parquet, up_button_clicks, path, ids, current_path):
    triggered = dash.callback_context.triggered_id
    print("handle_click")
    if triggered == "path-display":
        return path, path
    if triggered == "up-button":
        current_path = str(Path(current_path).parent)
        return current_path, current_path
    elif not any(load_parquet) or not triggered:
        return dash.no_update, dash.no_update
    selected_path = triggered["index"]
    return selected_path, selected_path


# @callback(
#     Output("custom-popup", "style"),
#     Input("save_and_check_pixels", "n_clicks"),
#     Input("close-popup", "n_clicks"),
#     State("custom-popup", "style"),
# )
# def toggle_popup(open_clicks, close_clicks, style):
#     print("toggle_popup")
#     ctx = dash.callback_context
#     if not style:
#         style = {}
#     style = style.copy()
#     if ctx.triggered_id == "save_and_check_pixels":
#         style["display"] = "block"
#     elif ctx.triggered_id == "close-popup":
#         style["display"] = "none"
#     return style


@callback(
    Output("remove-buttons", "children"),
    Input("new-file-added", "n_clicks"),
    Input("file-removed", "children"),
)
def render_items(new_file_added, file_removed):
    return [
        html.Div(
            [
                html.Button(
                    "❌",
                    id={"type": "delete-btn", "index": f"{i} -- {path}"},
                    n_clicks=0,
                    style={
                        "color": "red",
                        "border": "none",
                        "background": "none",
                        "cursor": "pointer",
                    },
                ),
                html.Span(get_name(path), style={"marginRight": "10px"}),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "5px"},
        )
        for i, path in enumerate(exp.paths)
    ]


@callback(
    Output("file-removed", "children"),
    Input({"type": "delete-btn", "index": dash.ALL}, "n_clicks"),
    # State({"type": "delete-btn", "index": dash.ALL}, "index"),
    # State("items-store", "data"),
    prevent_initial_call=True,
)
def delete_item(n_clicks_list):
    triggered_id = ctx.triggered_id
    print("delete_item")
    if triggered_id and triggered_id["type"] == "delete-btn":
        print(n_clicks_list)  # [0]
        # print(index)  # [None]
        i, path_to_remove = triggered_id["index"].split("--")
        i = int(i.strip())
        path_to_remove = path_to_remove.strip()
        print(i, path_to_remove)
        n_clicks = n_clicks_list[i]
        if not n_clicks:
            return dash.no_update
        # n_clicks = n_clicks_list[triggered_id["index"]]
        print(n_clicks)
        # path_to_remove = triggered_id["index"]
        print(path_to_remove)
        exp.paths.pop(exp.paths.index(path_to_remove))
        for path in list(exp.data):
            if path_to_remove in path:
                del exp.data[path]
        exp.bounds_series = exp.bounds_series[
            lambda x: ~x.index.str.contains(path_to_remove)
        ]
    return 1


@callback(
    Output("new-file-added", "n_clicks"),
    # Output("remove-buttons", "children"),
    Input({"type": "load-parquet", "index": dash.ALL}, "n_clicks"),
    # Input("remove-buttons", "children"),
    State({"type": "file-item", "index": dash.ALL}, "id"),
    prevent_initial_call=True,
)
def append_path(load_parquet, ids):
    triggered = dash.callback_context.triggered_id
    print("append_path")
    print(triggered)
    print(load_parquet)
    if not any(load_parquet):
        return dash.no_update  # , dash.no_update
    if triggered:
        selected_path = triggered["index"]
        exp.append(selected_path)
        # print(remove_buttons)
        checklist = dcc.Checklist(
            # list(remove_buttons.options) + [get_name(selected_path)],
            # list(remove_buttons.value) + [get_name(selected_path)],
            [get_name(path) for path in exp.paths],
            [get_name(path) for path in exp.paths],
        )
    else:
        checklist = dash.no_update

    return 1
    return (1, checklist)


@callback(
    Output("currently-in-bounds", "children"),
    Input("map", "bounds"),
    Input("new-file-added", "n_clicks"),
    Input("file-removed", "children"),
    # prevent_initial_call=True,
)
def get_files_in_bounds(bounds, file_added, file_removed):
    print("get_files_in_bounds", bounds)
    box = shapely.box(*nested_bounds_to_bounds(bounds))
    files_in_bounds = sg.sfilter(exp.bounds_series, box)
    currently_in_bounds = set(files_in_bounds.index)
    missing = list({path for path in files_in_bounds.index if path not in exp.data})
    if missing:
        read_files(exp, missing)
    return list(currently_in_bounds)


@callback(
    Output({"type": "geojson", "filename": dash.ALL}, "checked"),
    Input({"type": "geojson", "filename": dash.ALL}, "checked"),
    prevent_initial_call=True,
)
def uncheck(is_checked):
    print("is_checked")
    print(is_checked)
    stopp
    return is_checked


@callback(
    Output("column-dropdown", "options"),
    Input("currently-in-bounds", "children"),
    Input("file-removed", "children"),
    prevent_initial_call=True,
)
def update_column_dropdown(currently_in_bounds, file_removed):
    columns = set(
        itertools.chain.from_iterable(
            set(
                exp.data[path].columns.difference(
                    {exp.data[path].geometry.name, "_uuid"}
                )
            )
            for path in currently_in_bounds
        )
    )
    return [{"label": col, "value": col} for i, col in enumerate(sorted(columns))]


@callback(
    Output("column-value-color-dict", "children"),
    Output("bins", "children"),
    Output("is-numeric", "children"),
    Output("force-categorical", "children"),
    Output("currently-in-bounds2", "children"),
    Input("column-dropdown", "value"),
    Input("cmap-placeholder", "value"),
    Input("k", "value"),
    Input("force-categorical", "n_clicks"),
    Input("currently-in-bounds", "children"),
    State("map", "bounds"),
    prevent_initial_call=True,
)
def get_column_value_color_dict(
    column,
    cmap: str,
    k: int,
    force_categorical_clicks: int,
    currently_in_bounds,
    bounds,
):
    if column is None or not any(column in df for df in exp.data.values()):
        return None, None, False, dash.no_update, currently_in_bounds

    box = shapely.box(*nested_bounds_to_bounds(bounds))
    values = pd.concat(
        [
            sg.sfilter(df[[column, df.geometry.name]], box)[column]
            for df in exp.data.values()
            if column in df
        ],
        ignore_index=True,
    ).dropna()

    if not pd.api.types.is_numeric_dtype(values):
        force_categorical_button = None
    elif (force_categorical_clicks or 0) % 2 == 0:
        force_categorical_button = html.Button(
            "Force categorical",
            n_clicks=force_categorical_clicks,
            style={
                "fillColor": "white",
                "color": "black",
            },
        )
    else:
        force_categorical_button = html.Button(
            "Force categorical",
            n_clicks=force_categorical_clicks,
            style={
                "fillColor": "black",
                "color": "white",
            },
        )
    is_numeric = (
        force_categorical_clicks or 0
    ) % 2 == 0 and pd.api.types.is_numeric_dtype(values)
    if is_numeric:
        series = pd.concat(
            [
                df[column]
                for path, df in exp.data.items()
                if any(x in path for x in exp.paths) and column in df
            ]
        ).dropna()
        bins = jenks_breaks(series, n_classes=k)
        cmap_ = matplotlib.colormaps.get_cmap(cmap)
        colors_ = [
            matplotlib.colors.to_hex(cmap_(int(i))) for i in np.linspace(0, 255, num=k)
        ]
        column_value_color_dict = (
            [(f"{round(min(series), 1)} - {bins[0]}", colors_[0])]
            + [
                (f"{start} - {stop}", colors_[i + 1])
                for i, (start, stop) in enumerate(itertools.pairwise(bins[1:-1]))
            ]
            + [(f"{bins[-1]} - {round(max(series), 1)}", colors_[-1])]
        )
    else:

        unique_values = values.unique()
        default_colors = list(sg.maps.map._CATEGORICAL_CMAP.values())
        colors = default_colors[: min(len(unique_values), len(default_colors))] + [
            random_color()
            for _ in range(
                len(unique_values) - len(sg.maps.map._CATEGORICAL_CMAP.values())
            )
        ]
        column_value_color_dict = list(
            zip(
                unique_values,
                colors,
                strict=True,
            )
        )
        bins = None

    return (
        column_value_color_dict,
        bins,
        is_numeric,
        force_categorical_button,
        currently_in_bounds,
    )


@callback(
    Output("column-value-colors", "children"),
    Input("column-value-color-dict", "children"),
    Input("file-removed", "children"),
    State("is-numeric", "children"),
    prevent_initial_call=True,
)
def update_column_dropdown(values_to_colors, file_removed, is_numeric):
    if values_to_colors is None:
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Input(
                                type="color",
                                id={
                                    "type": "colorpicker",
                                    "column_value": get_name(path),
                                },
                                value=color,
                                style={"width": 50, "height": 50},
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Label([get_name(path)]),
                            width="auto",
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "flex-start",
                        "alignItems": "center",
                        "marginBottom": "5px",
                    },
                )
                for path, color in zip(
                    exp.paths, sg.maps.map._CATEGORICAL_CMAP.values(), strict=False
                )
            ]
        )
    if is_numeric:
        values_to_colors = dict(values_to_colors)
    else:
        values_to_colors = dict(values_to_colors)
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Input(
                            type="color",
                            id={"type": "colorpicker", "column_value": value},
                            value=color,
                            style={"width": 50, "height": 50},
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Label([value]),
                        width="auto",
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "flex-start",
                    "alignItems": "center",
                    "marginBottom": "5px",
                },
            )
            for value, color in values_to_colors.items()
        ]
        + [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Input(
                            type="color",
                            id={"type": "colorpicker", "column_value": NAN_LABEL},
                            value=NAN_COLOR,
                            style={"width": 50, "height": 50},
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Label([NAN_LABEL]),
                        width="auto",
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "flex-start",
                    "alignItems": "center",
                    "marginBottom": "5px",
                },
            )
        ],
        style={
            "height": "50vh",
            "overflow": "scroll",
        },
    )


@callback(
    Output("lc", "children"),
    Input("currently-in-bounds2", "children"),
    Input({"type": "colorpicker", "column_value": dash.ALL}, "value"),
    Input("is-numeric", "children"),
    Input("file-removed", "children"),
    State("map", "bounds"),
    State({"type": "geojson", "filename": dash.ALL}, "checked"),
    State("column-dropdown", "value"),
    State("column-value-color-dict", "children"),
    State("bins", "children"),
    prevent_initial_call=True,
)
def add_data(
    currently_in_bounds,
    color_values,
    is_numeric,
    file_removed,
    bounds,
    is_checked,
    column,
    values_to_colors,
    bins,
):
    print("add_data")
    print(is_checked)
    triggered = dash.callback_context.triggered_id
    print(triggered)
    box = shapely.box(*nested_bounds_to_bounds(bounds))
    data = []
    choices = np.arange(len(bins)) if bins is not None else None

    if not is_numeric and values_to_colors is not None and color_values is not None:
        values_to_colors = {
            x[0]: color_ for x, color_ in zip(values_to_colors, color_values)
        }
    elif values_to_colors is not None and color_values is not None:
        values_to_colors = {i: x[1] for i, x in enumerate(values_to_colors)}

    for x, v in dict(locals()).items():
        print()
        print(x)
        print(v)

    print(exp)
    for i, path in enumerate(exp.paths):
        # for path, color in zip(
        #     exp.paths, sg.maps.map._CATEGORICAL_CMAP.values(), strict=False
        # ):
        if path in exp.data:
            df = exp.data[path]
        else:
            df = pd.concat([df for key, df in exp.data.items() if path in key])

        df = sg.sfilter(df, box)
        if column is not None and column in df and not is_numeric:
            print("\nhei")
            print(path)
            print(values_to_colors)
            df["_color"] = df[column].map(values_to_colors)
            print(df["_color"].sort_values())
        elif column is not None and column in df:
            conditions = [
                df[column] < bins[0],
                *[
                    (df[column] >= bins[i]) & (df[column] < bins[i + 1])
                    for i in np.arange(1, len(bins) - 1)
                ],
                df[column] >= bins[-1],
            ]
            print(len(conditions))
            print(values_to_colors)
            df["_color"] = [values_to_colors[x] for x in np.select(conditions, choices)]
            print(df[["_color"]].sort_values("_color"))

        if not any(path in x for x in currently_in_bounds):

            data.append(
                dl.Overlay(
                    dl.GeoJSON(id={"type": "geojson", "filename": path}),
                    name=get_name(path),
                    checked=True,
                )
            )
            continue
        if column and column not in df:
            data.append(
                dl.Overlay(
                    dl.GeoJSON(
                        data=df.__geo_interface__,
                        style={
                            "color": NAN_COLOR,
                            "fillColor": NAN_COLOR,
                            "weight": 2,
                            "fillOpacity": 0.7,
                        },
                        # onEachFeature=ns("popup"),
                        id={"type": "geojson", "filename": path},
                        hideout=dict(selected=[]),
                    ),
                    name=get_name(path),
                    checked=True,
                )
            )
        elif column:
            data.append(
                dl.Overlay(
                    dl.LayerGroup(
                        [
                            dl.GeoJSON(
                                data=(df[df["_color"] == color_]).__geo_interface__,
                                style={
                                    "color": color_,
                                    "fillColor": color_,
                                    "weight": 2,
                                    "fillOpacity": 0.7,
                                },
                                # onEachFeature=ns("popup"),
                                id={
                                    "type": "geojson",
                                    "filename": path + color_,
                                },
                                hideout=dict(selected=[]),
                            )
                            for color_ in df["_color"].unique()
                        ]
                        + [
                            dl.GeoJSON(
                                data=df[df[column].isna()].__geo_interface__,
                                style={
                                    "color": NAN_COLOR,
                                    "fillColor": NAN_COLOR,
                                    "weight": 2,
                                    "fillOpacity": 0.7,
                                },
                                id={
                                    "type": "geojson",
                                    "filename": path + "nan",
                                },
                                # onEachFeature=ns("popup"),
                                hideout=dict(selected=[]),
                            )
                        ]
                    ),
                    name=get_name(path),
                    checked=True,
                )
            )
        else:
            # no column
            color = color_values[i]
            data.append(
                dl.Overlay(
                    dl.GeoJSON(
                        data=df.__geo_interface__,
                        style={
                            "color": color,
                            "fillColor": color,
                            "weight": 2,
                            "fillOpacity": 0.7,
                        },
                        # onEachFeature=ns("popup"),
                        id={"type": "geojson", "filename": path},
                    ),
                    name=get_name(path),
                    checked=True,
                )
            )

    return BASE_LAYERS + data


@callback(
    # Output("feature-table-container", "children"),
    Output("clicked-features", "data"),
    Input("clear-table", "n_clicks"),
    Input({"type": "geojson", "filename": dash.ALL}, "n_clicks"),
    State({"type": "geojson", "filename": dash.ALL}, "clickData"),
    # State({"type": "geojson", "filename": dash.ALL}, "hideout"),
    State("clicked-features", "data"),
    prevent_initial_call=True,
)
def display_feature_attributes(clear_table, n_clicks, feature, hideout):
    print("display_feature_attributes")
    triggered = dash.callback_context.triggered_id
    if triggered == "clear-table":
        hideout = []
        return hideout
    if not feature or not any(feature):
        return dash.no_update
    triggered = dash.callback_context.triggered_id
    filename_id = triggered["filename"]
    path = next(iter(x for x in exp.paths if x in filename_id))
    index = exp.paths.index(path)
    feature = feature[index]
    print(locals())
    props = feature["properties"]
    if props["_uuid"] not in {x["_uuid"] for x in hideout}:
        hideout.append(props)
    return hideout


@callback(
    Output("feature-table-container", "children"),
    Input("clicked-features", "data"),
    State("column-dropdown", "options"),
)
def update_table(data, column_dropdown):
    if not data:
        return "No features clicked."
    all_columns = {x["label"] for x in column_dropdown}
    columns = [{"name": k, "id": k} for k in data[0].keys() if k in all_columns]
    return html.Div(
        [
            # html.Div(f"Table view on {path}"),
            dash_table.DataTable(
                columns=columns,
                data=data,
                style_header={
                    "backgroundColor": "#2f2f2f",
                    "color": "white",
                    "fontWeight": "bold",
                },
                style_data={
                    "backgroundColor": "#d3d3d3",
                    "color": "black",
                },
                style_table={"overflowX": "auto"},
            ),
        ]
    )


app.clientside_callback(
    """function(_, feature, hideout){
    let selected = hideout.selected;
    const name = feature.properties.name;
    if(selected.includes(name)){selected = selected.filter((item) => (item !== name))}
    else{selected.push(name);}
    return {selected: selected};
}""",
    Output("geojson", "hideout"),
    Input({"type": "geojson", "filename": dash.ALL}, "n_clicks"),
    State({"type": "geojson", "filename": dash.ALL}, "clickData"),
    State({"type": "geojson", "filename": dash.ALL}, "hideout"),
    prevent_initial_call=True,
)


# app.clientside_callback(
#     ClientsideFunction(namespace="clientside", function_name="make_draggable"),
#     Output(
#         "custom-popup", "className"
#     ),  # the attribute here will not be updated, it is just used as a dummy
#     [Input("custom-popup", "id")],
# )

# app.clientside_callback(
#     """
#     function(trigger) {
#         if (window.gridInteractionInitialized) {
#             return true;
#         }

#         const gridConfigs = [
#             {
#                 id: "grid_button_container",
#                 cellClass: "grid-cell",
#                 rowAttr: "data-row",
#                 colAttr: "data-col"
#             },
#             {
#                 id: "grid_button_container_end",
#                 cellClass: "grid-cell-end",
#                 rowAttr: "data-row",
#                 colAttr: "data-col"
#             }
#         ];

#         let isMouseDown = false;
#         let visited = new Set();

#         document.addEventListener("mousedown", () => {
#             isMouseDown = true;
#             visited.clear();
#         });

#         document.addEventListener("mouseup", () => {
#             isMouseDown = false;
#             visited.clear();
#         });

#         gridConfigs.forEach(cfg => {
#             const grid = document.getElementById(cfg.id);
#             if (!grid) return;

#             grid.addEventListener("mouseover", function(e) {
#                 if (!isMouseDown) return;

#                 const cell = e.target.closest(`.${cfg.cellClass}`);
#                 if (!cell) return;

#                 const row = cell.getAttribute(cfg.rowAttr);
#                 const col = cell.getAttribute(cfg.colAttr);
#                 const cellId = `${cfg.id}:${row},${col}`;

#                 const currentColor = getComputedStyle(cell).backgroundColor;
#                 if (currentColor === "rgb(14, 42, 48)") return;

#                 if (!visited.has(cellId)) {
#                     visited.add(cellId);
#                     cell.click();
#                 }
#             });
#         });

#         console.log("✅ Grid interaction initialized.");
#         window.gridInteractionInitialized = true;
#         return true;
#     }
#     """,
#     Output("js_init_store", "data"),
#     Input("grid_button_container", "children"),
#     Input("grid_button_container_end", "children"),
# )


if __name__ == "__main__":
    app.run(debug=True, port=PORT)

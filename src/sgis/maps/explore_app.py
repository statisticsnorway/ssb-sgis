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

# BASE_DIR = "c:/users/ort"


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
        def __init__(self, file_system):
            self.selected_paths: list[str] = []
            self.currently_in_bounds: set[str] = set()
            self.bounds = GeoSeries()
            self.data: dict[str, GeoDataFrame] = {}
            self.file_system = file_system

        def append(self, path):
            self.selected_paths.append(path)
            suffix = Path(path).suffix
            if suffix:
                paths = list(self.file_system.glob(str(Path(path) / f"**/*{suffix}")))
                if not paths:
                    paths = [path]
            else:
                paths = [path]
            self.bounds = pd.concat(
                [
                    self.bounds,
                    sg.get_bounds_series(paths, file_system=self.file_system).to_crs(
                        4326
                    ),
                ]
            )

    exp = ExploreApp(file_system=LocalFileSystem())

    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        external_stylesheets=[dbc.themes.SOLAR],
        requests_pathname_prefix=f"/proxy/{PORT}/",
        serve_locally=True,
        assets_folder="assets",
    )
    default_center = [59.91740845, 10.71394444]
    default_zoom = 10
    default_bounds = [
        [59.62124229424823, 9.858855440173372],
        [60.207757877310925, 11.563109590563998],
    ]

    app.layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="k",
                            options=[
                                {"label": i, "value": i} for i in [3, 4, 5, 6, 7, 8, 9]
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
                            dbc.Input(value="viridis", id="cmap"), id="cmap-placeholder"
                        )
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dl.Map(
                                center=default_center,
                                zoom=default_zoom,
                                children=[
                                    # dl.TileLayer(),  # Base map layer
                                    # dl.LayerGroup(
                                    #     id="layer"
                                    # ),  # Optional layer for future use
                                    # dl.GeoJSON(id="geojson"),  # Optional GeoJSON layer
                                    # dl.LocateControl(
                                    #     # options={
                                    #     #     "locateOptions": {"enableHighAccuracy": True}
                                    #     # }
                                    # ),
                                    dl.LayersControl(
                                        [
                                            dl.BaseLayer(
                                                dl.TileLayer("OpenStreetMap"),
                                                name="OpenStreetMap",
                                                checked=True,
                                            ),
                                            dl.BaseLayer(
                                                dl.TileLayer(
                                                    url="https://opencache.statkart.no/gatekeeper/gk/gk.open_nib_web_mercator_wmts_v2?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=Nibcache_web_mercator_v2&STYLE=default&FORMAT=image/jpgpng&tileMatrixSet=default028mm&tileMatrix={z}&tileRow={y}&tileCol={x}",
                                                    attribution="© Geovekst",
                                                ),
                                                name="Norge i bilder",
                                                checked=False,
                                            ),
                                        ],
                                        id="lc",
                                    ),
                                    dl.ScaleControl(position="bottomleft"),
                                    dl.MeasureControl(
                                        position="bottomright",
                                        primaryLengthUnit="meters",
                                    ),
                                    # dl.LayerGroup(
                                    #     dl.Marker(position=[60.39, 5.32]), id="marker-layer"
                                    # ),
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
                            dbc.Row(id="column-value-colors"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            [
                                                html.H2("File Browser"),
                                                html.Button("⬆️ Go Up", id="up-button"),
                                                dcc.Store(
                                                    id="current-path", data=BASE_DIR
                                                ),
                                                html.Div(id="path-display"),
                                                html.Div(id="file-list"),
                                                html.Div(
                                                    id="selected-path",
                                                    style={
                                                        "marginTop": "20px",
                                                        "fontWeight": "bold",
                                                    },
                                                ),
                                                html.Div(
                                                    id="selected-file",
                                                    style={"display": "none"},
                                                ),
                                            ]
                                        ),
                                        # width=4,
                                    ),
                                ]
                            ),
                        ],
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(id="feature-table-container"),
                    ),
                    dbc.Col(
                        html.Div(id="remove-buttons"),
                    ),
                ],
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
            html.Div(id="currently-in-bounds"),
            html.Div(id="currently-in-bounds2"),
            html.Div(id="new-file-added"),
            html.Div(id="file-removed"),
            html.Div(id="column-value-color-dict"),
            html.Div(id="bins"),
            html.Div(False, id="is-numeric"),
            html.Div(False, id="cmap-has-been-set"),
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
    Output("path-display", "children"),
    Output("file-list", "children"),
    Input("current-path", "data"),
)
def update_file_list(path):
    print("update_file_list")
    return f"Browsing: {path}", list_dir(path)


@callback(
    Output("current-path", "data"),
    Output("selected-path", "children"),
    Input({"type": "file-item", "index": dash.ALL}, "n_clicks"),
    # Input({"type": "load-parquet", "index": dash.ALL}, "n_clicks"),
    Input("up-button", "n_clicks"),
    State({"type": "file-item", "index": dash.ALL}, "id"),
    State("current-path", "data"),
    prevent_initial_call=True,
)
def handle_click(load_parquet, up_button_clicks, ids, current_path):
    triggered = dash.callback_context.triggered_id
    print("handle_click")
    print(triggered)
    print(load_parquet)
    if triggered == "up-button":
        current_path = str(Path(current_path).parent)
    elif not any(load_parquet):
        return dash.no_update, dash.no_update
    elif triggered:
        selected_path = triggered["index"]
        # return current_path, selected_path
        if os.path.isdir(selected_path):  # and triggered["type"] != "load-parquet":
            return selected_path, None
        else:
            return current_path, selected_path
    return current_path, None


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

# @callback(
#     Output("selected-file", "children"),
#     Input("selected-path", "children"),
#     # prevent_initial_call=True,
# )
# def check_if_is_file(file_path):
#     print("check_if_is_file")
#     if file_path is None or not os.path.getsize(file_path):
#         return dash.no_update
#     print("is file", file_path)
#     return file_path


@app.callback(
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
        for i, path in enumerate(exp.selected_paths)
    ]


@app.callback(
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
        exp.selected_paths.pop(exp.selected_paths.index(path_to_remove))
        for path in list(exp.data):
            if path_to_remove in path:
                del exp.data[path]
        exp.bounds = exp.bounds[lambda x: ~x.index.str.contains(path_to_remove)]
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
            [get_name(path) for path in exp.selected_paths],
            [get_name(path) for path in exp.selected_paths],
        )
    else:
        checklist = dash.no_update

    return 1
    return (1, checklist)


@callback(
    Output("currently-in-bounds", "children"),
    Input("map", "bounds"),
    Input("new-file-added", "n_clicks"),
    # prevent_initial_call=True,
)
def get_files_in_bounds(bounds, n_clicks):
    print("get_files_in_bounds", bounds)
    box = shapely.box(*nested_bounds_to_bounds(bounds))
    files_in_bounds = sg.sfilter(exp.bounds, box)
    currently_in_bounds = set(files_in_bounds.index)
    missing = list({path for path in files_in_bounds.index if path not in exp.data})
    if not missing:
        return list(currently_in_bounds)

    read_func = partial(sg.read_geopandas, file_system=exp.file_system)
    with ThreadPoolExecutor() as executor:
        more_data = list(executor.map(read_func, missing))
    # more_data_dict = {}
    for path, df in zip(missing, more_data, strict=True):
        exp.data[path] = df.to_crs(4326)
        continue
        if path in exp.selected_paths:
            more_data_dict[path] = df
            continue
        root = [x for x in exp.selected_paths if x in path]
        if len(root) == 1:
            path = next(iter(root))
        try:
            more_data_dict[path].append(df)
        except KeyError:
            more_data_dict[path] = [df]

    # for path, df in more_data_dict.items():
    #     if isinstance(df, list):
    #         exp.data[path] = pd.concat(df, ignore_index=True).to_crs(4326)
    #     else:
    #         exp.data[path] = df.to_crs(4326)
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
    prevent_initial_call=True,
)
def update_column_dropdown(currently_in_bounds):
    columns = set(
        itertools.chain.from_iterable(
            set(exp.data[path].columns.difference({exp.data[path].geometry.name}))
            for path in currently_in_bounds
        )
    )
    return [{"label": col, "value": col} for i, col in enumerate(sorted(columns))]


@callback(
    Output("column-value-color-dict", "children"),
    Output("bins", "children"),
    Output("is-numeric", "children"),
    Output("cmap-placeholder", "children"),
    Output("force-categorical", "children"),
    Output("currently-in-bounds2", "children"),
    Input("column-dropdown", "value"),
    Input("cmap", "value"),
    Input("k", "value"),
    Input("force-categorical", "n_clicks"),
    Input("currently-in-bounds", "children"),
    State("cmap-has-been-set", "children"),
    State("map", "bounds"),
    prevent_initial_call=True,
)
def get_column_value_color_dict(
    column,
    cmap: str,
    k: int,
    force_categorical_clicks: int,
    currently_in_bounds,
    cmap_has_been_set: bool,
    bounds,
):
    if column is None or not any(column in df for df in exp.data.values()):
        return None, None, False, dash.no_update, dash.no_update, currently_in_bounds

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
    if is_numeric and not cmap_has_been_set:
        cmap_component = dbc.Input(
            placeholder="Cmap",
            value="viridis",
            type="text",
            id="cmap",
            style={"font-size": 22},
        )
    elif not cmap_has_been_set:
        cmap_component = None
    else:
        cmap_component = dash.no_update

    if is_numeric:
        series = pd.concat(
            [
                df[column]
                for path, df in exp.data.items()
                if any(x in path for x in exp.selected_paths) and column in df
            ]
        )
        bins = jenks_breaks(series.dropna(), n_classes=k)
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
        # list(enumerate(colors_))
    else:

        unique_values = values.unique()
        default_colors = list(sg.maps.map._CATEGORICAL_CMAP.values())
        colors = default_colors[: min(len(unique_values), len(default_colors))] + [
            random_color()
            for _ in range(
                len(unique_values) - len(sg.maps.map._CATEGORICAL_CMAP.values())
            )
        ]
        print(len(unique_values))
        print(len(colors))
        print(len(sg.maps.map._CATEGORICAL_CMAP.values()))
        print(len(unique_values) - len(sg.maps.map._CATEGORICAL_CMAP.values()))
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
        cmap_component,
        force_categorical_button,
        currently_in_bounds,
    )


@callback(
    Output("cmap-has-been-set", "children"),
    Input("cmap-placeholder", "children"),
    prevent_initial_call=True,
)
def update_cmap_has_been_set(cmap):
    return True


@callback(
    Output("column-value-colors", "children"),
    Input("column-value-color-dict", "children"),
    State("is-numeric", "children"),
    prevent_initial_call=True,
)
def update_column_dropdown(values_to_colors, is_numeric):
    if values_to_colors is None:
        return None
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


def nested_bounds_to_bounds(
    bounds: list[list[float]],
) -> tuple[float, float, float, float]:
    if bounds is None:
        bounds = default_bounds
    mins, maxs = bounds
    miny, minx = mins
    maxy, maxx = maxs
    return minx, miny, maxx, maxy


@callback(
    Output("lc", "children"),
    Input("currently-in-bounds2", "children"),
    Input({"type": "colorpicker", "column_value": dash.ALL}, "value"),
    Input("is-numeric", "children"),
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
    for path, color in zip(
        exp.selected_paths, sg.maps.map._CATEGORICAL_CMAP.values(), strict=False
    ):
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
                    dl.GeoJSON(),
                    id={"type": "geojson", "filename": get_name(path)},
                    name=get_name(path),
                    checked=True,
                )
            )
            continue
        if column and column not in df:
            data.append(
                dl.Overlay(
                    dl.GeoJSON(
                        data=sg.sfilter(df, box).__geo_interface__,
                        style={
                            "color": NAN_COLOR,
                            "fillColor": NAN_COLOR,
                            "weight": 2,
                            "fillOpacity": 0.7,
                        },
                    ),
                    name=get_name(path),
                    id={"type": "geojson", "filename": get_name(path)},
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
                            )
                            for color_ in df["_color"].unique()
                        ]
                        + [
                            dl.GeoJSON(
                                data=sg.sfilter(
                                    df[df[column].isna()], box
                                ).__geo_interface__,
                                style={
                                    "color": NAN_COLOR,
                                    "fillColor": NAN_COLOR,
                                    "weight": 2,
                                    "fillOpacity": 0.7,
                                },
                            )
                        ]
                    ),
                    name=get_name(path),
                    id={"type": "geojson", "filename": get_name(path)},
                    checked=True,
                )
            )
        else:
            # no column
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
                    ),
                    name=get_name(path),
                    id={"type": "geojson", "filename": get_name(path)},
                    checked=True,
                )
            )

    return [
        dl.BaseLayer(
            dl.TileLayer("OpenStreetMap"),
            name="OpenStreetMap",
            checked=True,
        ),
        dl.BaseLayer(
            dl.TileLayer(
                url="https://opencache.statkart.no/gatekeeper/gk/gk.open_nib_web_mercator_wmts_v2?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=Nibcache_web_mercator_v2&STYLE=default&FORMAT=image/jpgpng&tileMatrixSet=default028mm&tileMatrix={z}&tileRow={y}&tileCol={x}",
                attribution="© Geovekst",
            ),
            name="Norge i bilder",
            checked=False,
        ),
    ] + data


# @callback(
#     Output("feature-table-container", "children"),
#     Input({"type": "geojson", "filename": dash.ALL}, "click_feature"),
#     prevent_initial_call=True,
# )
# def display_feature_table(feature):
#     if not feature:
#         return dash.no_update
#     print("feature")
#     print(feature)
#     props = feature.get("properties", {})
#     return dash_table.DataTable(
#         columns=[{"name": k, "id": k} for k in props],
#         data=[props],
#         style_table={"overflowX": "auto"},
#         style_cell={"textAlign": "left"},
#     )


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

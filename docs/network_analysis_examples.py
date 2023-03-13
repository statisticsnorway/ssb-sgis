# %%--
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# %%--

import os
import warnings

import numpy as np
import pandas as pd


os.chdir("../src")
import sgis as sg


os.chdir("..")

# ignore some warnings to make it cleaner
pd.options.mode.chained_assignment = None
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
# %%
import geopandas as gpd

r = gpd.read_file(
    r"C:\Users\ort\Downloads\vegnettRuteplan_FGDB_20230109.gdb",
    layer="ruttger_link_geom",
)
r
# %%
import sgis as sg

roads = sg.read_parquet_url(
    "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet"
)

nw = (
    sg.DirectedNetwork(roads)
    .remove_isolated()
    .make_directed_network(
        direction_col="oneway",
        direction_vals_bft=("B", "FT", "TF"),
        minute_cols=("drivetime_fw", "drivetime_bw"),
    )
)

rules = sg.NetworkAnalysisRules(weight="minutes")

nwa = sg.NetworkAnalysis(network=nw, rules=rules)

nwa
# %%

points = sg.read_parquet_url(
    "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/random_points.parquet"
)

# %% [markdown]
# # sgis
# sgis builds on the geopandas package, providing functions for making it easier to do advanced, high quality GIS in python.
#
# This includes vector operations like finding k nearest neighbours, splitting lines by points, snapping geometries
# and closing holes in polygons by area.
# vector operations
# A core feature is network analysis, with methods for preparing road data and getting travel times, routes, route frequencies and service
# areas.
#  geopandas for easy, high quality GIS in python in science and production of statistics.
#
# It builds on the geopandas package by integrating the GeoDataFrame with, and provides tools for network analysis
#
# for GIS in python in science and production of statistics.
# This includes classes for preparing road data
# The package is an extension to geopandas.
# It provides functions for exploring multiple
#
# ## Network analysis integrated with geopandas
#
# The package offers methods that makes it easy to customise and optimise road data and
# calculate travel times, routes, frequencies and service areas.
#
# All you need is a GeoDataFrame of roads or other line geometries.
#
# Here are some examples. More examples and info here: https://github.com/statisticsnorway/ssb-sgis/blob/main/network_analysis_demo_template.md
#
# Road data can be prepared for network analysis like this:
#
# %%
import sgis as sg

roads = sg.read_parquet_url(
    "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet"
)

nw = (
    sg.DirectedNetwork(roads)
    .remove_isolated()
    .make_directed_network(
        direction_col="oneway",
        direction_vals_bft=("B", "FT", "TF"),
        minute_cols=("drivetime_fw", "drivetime_bw"),
    )
)

rules = sg.NetworkAnalysisRules(weight="minutes")

nwa = sg.NetworkAnalysis(network=nw, rules=rules)

nwa
# %% [markdown]
# Road data for Norway can be downloaded here: https://kartkatalog.geonorge.no/metadata/nvdb-ruteplan-nettverksdatasett/8d0f9066-34f9-4423-be12-8e8523089313
#
# #### get_route_frequencies: get the number of times each line segment was visited

# %%
freq = nwa.get_route_frequencies(points.sample(75), points.sample(75))

sg.qtm(
    sg.buff(freq, 15),
    "n",
    scheme="naturalbreaks",
    cmap="plasma",
    title="Number of times each road was used.",
)

# %% [markdown]
# #### od_cost_matrix: fast many-to-many travel times/distances

# %%
od = nwa.od_cost_matrix(points.iloc[[0]], points, id_col="idx", lines=True)

print(od.head(3))

sg.qtm(od, "minutes", title="Travel time (minutes) from 1 to 1000 points.")
# %% [markdown]
# #### service_area: get the area that can be reached within one or more breaks

# %%
sa = nwa.service_area(
    points.iloc[[0]],
    breaks=np.arange(1, 11),
)

sg.qtm(sa, "minutes", k=10, title="Roads that can be reached within 1 to 10 minutes")
# %% [markdown]
# #### get_route and get_k_routes: get one or more route per origin-destination pair

# %%
routes = nwa.get_k_routes(
    points.iloc[[0]], points.iloc[[1]], k=5, drop_middle_percent=50
)

sg.qtm(sg.buff(routes, 15), "k", title="k=5 low-cost routes", legend=False)

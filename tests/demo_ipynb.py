# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# ## Network analysis with ssb-gis-utils

# %% [markdown]
# Network analysis with igraph, integrated with geopandas.
#
# The package supports three types of network analysis:
# - od_cost_matrix: fast many-to-many travel times/distances
# - get_route: returns the geometry of the lowest-cost paths.
# - service_area: returns the roads that can be reached within one or more breaks.

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


pd.options.mode.chained_assignment = None  # ignore SettingWithCopyWarning for now

import os


os.chdir("../src")

import gis_utils as gs


os.chdir("..")

# %% [markdown]
# Let's start by loading the data:

# %%
points = gpd.read_parquet("tests/testdata/random_points.parquet")
points

# %%
roads = gpd.read_parquet("tests/testdata/roads_oslo_2022.parquet")
roads = roads[["oneway", "drivetime_fw", "drivetime_bw", "geometry"]]
roads.head(3)

# %% [markdown]
# ## The Network

# %%
nw = gs.Network(roads)
nw

# %%
nw.gdf.head(3)

# %% [markdown]
# The network class includes methods for optimizing the road data. More about this further down in this notebook.

# %%
nw = nw.close_network_holes(1.5).remove_isolated().cut_lines(100)
nw

# %% [markdown]
# For directed network analysis, the DirectedNetwork class can be used. This inherits all methods from the Network class, and also includes methods for making a directed network.

# %%
nw = gs.DirectedNetwork(roads).remove_isolated()
nw

# %%
isinstance(nw, gs.Network)

# %% [markdown]
# The above warning suggests that the data might not be directed yet. This is correct. The roads going both ways, only appear once, and the roads going backwards, have to be flipped around.
#
# This can be done in the make_directed_network method.

# %%
nw2 = nw.copy()
nw2 = nw2.make_directed_network(
    direction_col="oneway",
    direction_vals_bft=("B", "FT", "TF"),
    speed_col=None,
    minute_cols=("drivetime_fw", "drivetime_bw"),
    flat_speed=None,
)
nw2

# %% [markdown]
# The roads now have almost twice as many rows, since most roads are bidirectional in this network.
#
# OpenStreetMap road data and Norwegian road network can be made directional with custom methods, where the default parameters should give the correct results:

# %%
# nw.make_directed_network_osm()

# %%
nw = nw.make_directed_network_norway()
nw

# %% [markdown]
# ## NetworkAnalysis
#
# The NetworkAnalysis class takes a network and some rules.
#
# This will set the rules to its default values:

# %%
rules = gs.NetworkAnalysisRules(weight="minutes")
rules

# %% [markdown]
# Now we have what we need to start the network analysis.

# %%
nwa = gs.NetworkAnalysis(network=nw, rules=rules)
nwa

# %% [markdown]
# od_cost_matrix calculates the traveltime from a set of origins to a set of destinations:

# %%
od = nwa.od_cost_matrix(points, points, id_col="idx")
od

# %% [markdown]
# Set 'lines' to True to get straight lines between origin and destination:

# %%
od = nwa.od_cost_matrix(points.sample(1), points, lines=True)

gs.qtm(
    od,
    "minutes",
    title="Travel time (minutes) from 1 to 1000 addresses.",
    k=7,
)

# %% [markdown]
# The get_route method can be used to get the actual paths:

# %%
sp = nwa.get_route(points.iloc[[0]], points.sample(100), id_col="idx")

gs.qtm(sp, "minutes", cmap=gs.chop_cmap("RdPu", 0.2), title="Travel times")

sp

# %% [markdown]
# Set 'summarise' to True to get the number of times each road segment was used. This is faster than not summarising, because no dissolve is done.

# %%
sp = nwa.get_route(points.sample(150), points.sample(150), summarise=True)

gs.qtm(
    sp,
    "n",
    scheme="naturalbreaks",
    k=7,
    cmap=gs.chop_cmap("RdPu", 0.2),
    title="Number of times each road was used.",
)

# %% [markdown]
# The service_area method finds the area that can be reached within one or more breaks.
#
# Here, we find the areas that can be reached within 5, 10 and 15 minutes for five random points:

# %%
sa = nwa.service_area(points.sample(5), breaks=(5, 10, 15), id_col="idx")
sa

# %%
sa = nwa.service_area(points.iloc[[0]], breaks=np.arange(1, 11)).sort_values(
    "minutes", ascending=False
)
gs.qtm(sa, "minutes", k=9, title="Area that can be reached within 1 to 10 minutes.")

# %% [markdown]
# Set 'dissolve' to False to get every road segment returned, one for each service area that uses the segment. If there are a lot of overlapping service areas, that are to be dissolved in the end, removing duplicates first will make things a whole lot faster.

# %%
sa = nwa.service_area(points.sample(250), breaks=5, dissolve=False)

print(len(sa))

sa = sa.drop_duplicates(["source", "target"])

print(len(sa))

gs.qtm(sa)

# %% [markdown]
# ### Customising the network

# %%
nw = gs.DirectedNetwork(roads)
nw

# %% [markdown]
# Networks often consist of one large, connected network and many small, isolated "network islands".
#
# origins and destinations located inside these isolated networks, will have a hard time finding their way out.
#
# The large, connected network component can be found with the method get_largest_component:

# %%
nw = nw.get_largest_component()

gs.clipmap(
    nw.gdf,
    points.iloc[[0]].buffer(1000),
    column="connected",
    title="Connected and isolated networks",
    scheme="equalinterval",
    cmap="bwr",
    explore=False,
)

# %% [markdown]
# Use the remove_isolated method to remove the unconnected roads:

# %%
nw = nw.remove_isolated()

gs.clipmap(
    nw.gdf,
    points.iloc[[0]].buffer(1000),
    explore=False,
)

nw

# %% [markdown]
# If your road data has small gaps between the segments, these can be populated with straight lines:

# %%
nw = nw.close_network_holes(max_dist=1.5)  # meters
nw

# %% [markdown]
# The network analysis is done from node to node. In a service area analysis, the results will be inaccurate for long lines, since the destination will either be reached or not within the breaks. This can be fixed by cutting all lines to a maximum distance.
#
# Note: cutting the lines can be time consuming for large networks and low maximum distances.

# %%
nw = nw.cut_lines(25)  # meters
nw

# %%

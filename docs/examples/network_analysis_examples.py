# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # sgis
# %%

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


src = str(Path().resolve().parent.parent) + "/src"
print(src)
sys.path.insert(0, src)

import sgis as sg


# ignore some warnings to make it cleaner
pd.options.mode.chained_assignment = None
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
import geopandas as gpd


points = sg.read_parquet_url(
    "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet"
)

### !!!
### COPY EVERYTHING BELOW INTO README.md and change the png paths.
### !!!

# %% [markdown]
# sgis builds on the geopandas package and provides functions that make it easier to do GIS in python.
# Features include network analysis, functions for exploring multiple GeoDataFrames in a layered interactive map,
# and vector operations like finding k-nearest neighbours, splitting lines by points, snapping and closing holes
# in polygons by size.
#
# ## Network analysis examples
# Preparing for network analysis:
#
# %%
import sgis as sg


roads = sg.read_parquet_url(
    "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet"
)

connected_roads = sg.get_connected_components(roads).query("connected == 1")

directed_roads = sg.make_directed_network(
    connected_roads,
    direction_col="oneway",
    direction_vals_bft=("B", "FT", "TF"),
    minute_cols=("drivetime_fw", "drivetime_bw"),
)

rules = sg.NetworkAnalysisRules(directed=True, weight="minutes")

nwa = sg.NetworkAnalysis(network=directed_roads, rules=rules)

nwa

# %% [markdown]
# Fast many-to-many travel times/distances.
# %%
od = nwa.od_cost_matrix(points, points)

print(od)

# %% [markdown]
# Get number of times each line segment was visited, with optional weighting.

# %%
origins = points.iloc[:100]
destinations = points.iloc[100:200]

# creating uniform weights of 10
od_pairs = pd.MultiIndex.from_product([origins.index, destinations.index])
weights = pd.DataFrame(index=od_pairs)
weights["weight"] = 10

frequencies = nwa.get_route_frequencies(origins, destinations, weight_df=weights)

# plot the results
m = sg.ThematicMap(sg.buff(frequencies, 15), column="frequency", black=True)
m.cmap = "plasma"
m.title = "Number of times each road was used,\nweighted * 10"
m.plot()
# %% [markdown]
# Get the area that can be reached within one or more breaks.

# %%
service_areas = nwa.service_area(
    points.iloc[[0]],
    breaks=np.arange(1, 11),
)

# plot the results
m = sg.ThematicMap(service_areas, column="minutes", black=True, size=10)
m.k = 10
m.title = "Roads that can be reached within 1 to 10 minutes"
m.plot()
# %% [markdown]
# Get one or more route per origin-destination pair.

# %%
routes = nwa.get_k_routes(
    points.iloc[[0]], points.iloc[[1]], k=4, drop_middle_percent=50
)

m = sg.ThematicMap(sg.buff(routes, 15), column="k", black=True)
m.title = "Four fastest routes from A to B"
m.legend.title = "Rank"
m.plot()

# %% [markdown]
# More network analysis examples can be found here: https://github.com/statisticsnorway/ssb-sgis/blob/main/docs/network_analysis_demo_template.md
#
# Road data for Norway can be downloaded here: https://kartkatalog.geonorge.no/metadata/nvdb-ruteplan-nettverksdatasett/8d0f9066-34f9-4423-be12-8e8523089313

# %%

# %%

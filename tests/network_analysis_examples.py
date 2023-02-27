# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
# ---

import os

# +
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd


os.chdir("../src")
import gis_utils as gs


os.chdir("..")

# ignore some warnings to make it cleaner
pd.options.mode.chained_assignment = None
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
# -

# ## Network analysis
#
# The package supports four types of network analysis, and methods for customising and optimising road or other line data.
#
# Analysis can start by initialising a NetworkAnalysis instance:

# +
from gis_utils import DirectedNetwork, NetworkAnalysis, NetworkAnalysisRules


roads = gpd.read_parquet("tests/testdata/roads_oslo_2022.parquet")

nw = (
    DirectedNetwork(roads)
    .remove_isolated()
    .make_directed_network(
        direction_col="oneway",
        direction_vals_bft=("B", "FT", "TF"),
        minute_cols=("drivetime_fw", "drivetime_bw"),
    )
)

rules = NetworkAnalysisRules(weight="minutes")

nwa = NetworkAnalysis(network=nw, rules=rules)

nwa
# -

points = gpd.read_parquet("tests/testdata/random_points.parquet")

# ### od_cost_matrix
# Fast many-to-many travel times/distances

# +
od = nwa.od_cost_matrix(points.iloc[[0]], points, lines=True)

gs.qtm(
    od,
    "minutes",
    title="Travel time (minutes) from 1 to 1000 points.",
)

od
# -

# ### get_route
#
# Get the actual paths:

# +
routes = nwa.get_route(points.iloc[[0]], points.sample(100), id_col="idx")

gs.qtm(
    gs.buff(routes, 15),
    "minutes",
    cmap="plasma",
    title="Travel times (minutes)",
)

routes
# -

# ### get_route_frequencies
# Get the number of times each line segment was used:

# +
freq = nwa.get_route_frequencies(points.sample(100), points.sample(100))

gs.qtm(
    gs.buff(freq, 15),
    "n",
    scheme="naturalbreaks",
    cmap="plasma",
    title="Number of times each road was used.",
)
# -

# ### Service area
# Get the area that can be reached within one or more breaks.

# +
sa = nwa.service_area(points.iloc[[0]], breaks=np.arange(1, 11), id_col="idx")
sa
# -

# By setting dissolve to False, we get the individual road segments, and can remove
# duplicate rows to not have overlapping service areas.

# +
sa = nwa.service_area(points.iloc[[0]], breaks=np.arange(1, 11), dissolve=False)

sa = sa.drop_duplicates(["source", "target"])

gs.qtm(
    sa,
    "minutes",
    k=10,
    title="Roads that can be reached within 1 to 10 minutes",
    legend=False,
)
# -

# Check the log:

nwa.log

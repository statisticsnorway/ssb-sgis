# %%--
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# %% [markdown]
# ## Network analysis in ssb-gis-utils
# ##### igraph integrated with geopandas

# The package supports three types of network analysis, and methods for customising and optimising your road data.
#
# The package supports four types of network analysis:
# - od_cost_matrix: fast many-to-many travel times/distances
# - get_route: returns the geometry of the lowest-cost paths.
# - get_route_frequencies: summarises how many times each road/line segment was used.
# - service_area: returns the roads/lines that can be reached within one or more breaks.

# %%
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd


os.chdir("../src")

import gis_utils as gs


os.chdir("..")
# %%
# ignore some warnings to make it cleaner
pd.options.mode.chained_assignment = None
# warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
# %% [markdown]
# At the core of the network analysis, is the NetworkAnalysis class.
#
# It takes a network and a set of rules for the analysis:

# The rules can be instantiated like this:
# %%

rules = gs.NetworkAnalysisRules(weight="minutes")
rules
# %% [markdown]
# To create the network, we need some road data:
# %%

roads = gpd.read_parquet("tests/testdata/roads_oslo_2022.parquet")
roads = roads[["oneway", "drivetime_fw", "drivetime_bw", "geometry"]]
roads.head(3)
# %% [markdown]
# The road data can be made into a Network instance like this:
# %%
nw = (
    gs.Network(roads)
    .close_network_holes(1.5, fillna=0)
    .remove_isolated()
    .cut_lines(100)
)

nw = gs.Network(roads)
nw
# %% [markdown]
# The Network is now ready for undirected network analysis. The network can also be optimises with methods stored in the Network class. More about this further down in this notebook.
# %%

nw = nw.close_network_holes(1.5, fillna=0).remove_isolated().cut_lines(250)
nw
# %% [markdown]
# For directed network analysis, the DirectedNetwork class can be used. This inherits all methods from the Network class, and also includes methods for making a directed network.
# %%

nw = gs.DirectedNetwork(roads).remove_isolated()
nw
# %% [markdown]
# We now have a DirectedNetwork instance. However, the network isn't actually directed yet, as indicated by the percent_bidirectional attribute above.
#
# The roads going both ways have to be duplicated and the geometry of the new lines have to be flipped. The roads going backwards also have to be flipped.
#
# This can be done with the make_directed_network method:
# %%

nw = nw.make_directed_network(
    direction_col="oneway",
    direction_vals_bft=("B", "FT", "TF"),
    speed_col=None,
    minute_cols=("drivetime_fw", "drivetime_bw"),
    flat_speed=None,
)
nw
# %% [markdown]
# The network has now almost doubled in length, since most roads are bidirectional in this network.
#
# Norwegian road data can be made directional with a custom method:
# %%

nw = gs.DirectedNetwork(roads).remove_isolated().make_directed_network_norway()
nw
# %% [markdown]
# ## NetworkAnalysis
#
# To start the network analysis, we put our network and our rules into the NetworkAnalysis class:
# %%

nwa = gs.NetworkAnalysis(network=nw, rules=rules)
nwa
# %% [markdown]
# We also need some points that will be our origins and destinations:
# %%

points = gpd.read_parquet("tests/testdata/random_points.parquet")
points
# %% [markdown]
# ### OD cost matrix

# od_cost_matrix calculates the traveltime from a set of origins to a set of destinations:
# %%

od = nwa.od_cost_matrix(origins=points, destinations=points, id_col="idx")
od
# %% [markdown]
# Set 'lines' to True to get straight lines between origin and destination:

# %%
od = nwa.od_cost_matrix(points.iloc[[0]], points, lines=True)

print(od.head(3))

gs.qtm(
    od,
    "minutes",
    title="Travel time (minutes) from 1 to 1000 addresses.",
)
# %% [markdown]
# Information about the analyses are stored in a DataFrame in the 'log' attribute:
# %%

nwa.log
# %% [markdown]
# ### Get route

# The get_route method can be used to get the actual lowest cost path:

# %%
routes = nwa.get_route(points.iloc[[0]], points.sample(100), id_col="idx")

gs.qtm(
    gs.buff(routes, 12),
    "minutes",
    cmap="plasma",
    title="Travel times (minutes)",
)

routes
# %% [markdown]
# ## Get route frequencies
#
# get_route_frequencies finds the number of times each road segment was used.

# %%
pointsample = points.sample(100)
freq = nwa.get_route_frequencies(pointsample, pointsample)

gs.qtm(
    gs.buff(freq, 15),
    "n",
    scheme="naturalbreaks",
    cmap="plasma",
    title="Number of times each road was used (weight='minutes')",
)
# %% [markdown]
# The results will be quite different when it is the shortest, rather than the fastest, route that is used:

# %%
nwa.rules.weight = "meters"

freq = nwa.get_route_frequencies(pointsample, pointsample)

gs.qtm(
    gs.buff(freq, 15),
    "n",
    scheme="naturalbreaks",
    cmap="plasma",
    title="Number of times each road was used (weight='meters')",
)
# %%
nwa.rules.weight = "minutes"
# %% [markdown]
# ### Service area

# The service_area method finds the area that can be reached within one or more breaks.
#
# Here, we find the areas that can be reached within 5, 10 and 15 minutes for five random points:
# %%

sa = nwa.service_area(points.sample(5), breaks=(5, 10, 15), id_col="idx")
sa

# %%
sa = nwa.service_area(points.iloc[[0]], breaks=np.arange(1, 11), id_col="idx")

gs.qtm(
    sa,
    "minutes",
    k=10,
    title="Roads that can be reached within 1 to 10 minutes",
)
sa
# %% [markdown]
# By default, only the lowest break is kept for overlapping areas from the same origin, meaning the area for minutes=10
# covers not the entire area, only the outermost ring:

# %%
gs.qtm(
    sa.query("minutes == 10"),
    color="yellow",
    title="Roads that can be reached within 10 minutes",
)
# %% [markdown]
# This behaviour can be changed by setting drop_duplicates to False.

# Duplicate lines from different origins will not be removed. To drop all duplicates, if many
# origins in close proximity, set 'dissolve' to False to get each individual road or line returned,
# and then drop rows afterwards:

# %%
sa = nwa.service_area(points.sample(100), breaks=5, dissolve=False)
print("rows before drop_duplicates:", len(sa))
sa = sa.drop_duplicates(["source", "target"])
print("rows after drop_duplicates:", len(sa))
# %% [markdown]
# Let's check the log:

# %%
nwa.log
# %% [markdown]
# ### Customising the network

# %%
nw = gs.Network(roads)
nw
# %% [markdown]
# If you want to manipulate the roads after instantiating the Network, you can access the GeoDataFrame in the 'gdf' attribute:
# %%
nw.gdf.head(3)
# %% [markdown]
# ### remove_isolated
#
# The above log file has a column called 'isolated_removed'. This is set to True because the method 'remove_isolated' was used before the analyses.
#
# Networks often consist of one large, connected network and many small, isolated "network islands".
#
# origins and destinations located inside these isolated networks, will have a hard time finding their way out.
#
# The large, connected network component can be found (not removed) with the method get_largest_component:

# %%
nw = nw.get_largest_component()

# the GeoDataFrame of the network is stored in the gdf attribute:
nw.gdf["connected_str"] = np.where(nw.gdf.connected == 1, "connected", "isolated")

gs.clipmap(
    nw.gdf,
    points.iloc[[0]].buffer(1000),
    column="connected_str",
    title="Connected and isolated networks",
    cmap="bwr",
    explore=False,
)
# %% [markdown]
# Use the remove_isolated method to remove the unconnected roads:
# %%

nwa = gs.NetworkAnalysis(network=nw, rules=gs.NetworkAnalysisRules(weight="meters"))
od = nwa.od_cost_matrix(points, points)
percent_missing = od[nwa.rules.weight].isna().mean() * 100
print(f"Before removing isolated: {percent_missing=:.2f}")

# %%
nwa.network = nwa.network.remove_isolated()

od = nwa.od_cost_matrix(points, points)
percent_missing = od[nwa.rules.weight].isna().mean() * 100
print(f"After removing isolated: {percent_missing=:.2f}")
# %% [markdown]
# If the road data has some gaps between the segments, these can be filled with straight lines:
# %%

nw = nw.close_network_holes(max_dist=1.5)  # meters
nw
# %% [markdown]
# The network analysis is done from node to node. In a service area analysis, the results will be inaccurate for long lines, since the destination will either be reached or not within the breaks. This can be fixed by cutting all lines to a maximum distance.
#
# Note: cutting the lines can take a lot of time for large networks and low cut distances.
# %%
nw = nw.cut_lines(100)  # meters
nw.gdf.length.max()
# %% [markdown]
# ## DirectedNetwork

# Using the DirectedNetwork instead of the Network class, doesn't do anything to the network initially.
#
# But if we use it directly in the NetworkAnalysis, we see that 0 percent of the lines go in both directions:
# %%

nw = gs.DirectedNetwork(roads)
rules = gs.NetworkAnalysisRules(weight="metres")
nwa = gs.NetworkAnalysis(nw, rules=rules)
# %% [markdown]
# To make this network bidirectional, roads going both ways have to be duplicated and flipped. Roads going the opposite way also need to be flipped.
#
# The key here is in the 'oneway' column:
# %%
nw.gdf.oneway.value_counts()
# %% [markdown]
# We use this to make the network bidirectional with the 'make_directed_network' method.
#
# If we want a minute column, we also have to specify how to calculate this. Here, I use the two minute columns in the data:
# %%
nw.gdf[["oneway", "drivetime_fw", "drivetime_bw"]].drop_duplicates(
    "oneway"
)  # dropping duplicates for illustration's sake
# %% [markdown]
# Specify the values of the direction column in a tuple/list with the order "both ways", "from/forward", "to/backward".

# %%
nw = nw.make_directed_network(
    direction_col="oneway",
    direction_vals_bft=("B", "FT", "TF"),
    minute_cols=("drivetime_fw", "drivetime_bw"),
)

nw.gdf["minutes"]
# %% [markdown]
# You can also calculate minutes from a speed limit column. But you might want to do some manual adjusting, since keeping the speed limit at all times is unrealistic in most cases.
#
# You can set a flat speed that will be used for the entire network. Decent if the travel mode is walking, bike, boat etc.
# %%
bike_nw = nw.make_directed_network(
    direction_col="oneway",
    direction_vals_bft=("B", "FT", "TF"),
    flat_speed=20,
)

bike_nw.gdf["minutes"]
# %% [markdown]
# ## The NetworkAnalysisRules
#
# #### weight
# The weight parameter has to be specified. The weight can be the name of any numeric column in network.gdf.
#
# Or, if the weight is 'meters' or 'metres', a meter column will be created. The coordinate reference system of the network has to be meters as well.
# %%
rules = gs.NetworkAnalysisRules(weight="metres")
gs.NetworkAnalysis(nw, rules=rules).network.gdf["metres"]
# %% [markdown]
# If you want other distance units, create the column beforehand.
# %%

nw.gdf = nw.gdf.to_crs(6576).assign(feet=lambda x: x.length)
rules = gs.NetworkAnalysisRules(weight="feet")
gs.NetworkAnalysis(nw, rules=rules).network.gdf.feet
# %% [markdown]
# A minute column can be created through the 'make_directed_network' or 'make_directed_network_norway' methods.

# %%
nw = (
    gs.DirectedNetwork(roads)
    .remove_isolated()
    .make_directed_network(
        direction_col="oneway",
        direction_vals_bft=("B", "FT", "TF"),
        minute_cols=("drivetime_fw", "drivetime_bw"),
    )
)

rules = gs.NetworkAnalysisRules(weight="minutes")

nwa = gs.NetworkAnalysis(network=nw, rules=rules)

nwa
# %% [markdown]
# ### split_lines
#
# By default, the origins and destinations are connected to the closest nodes of the network:
# %%

nwa.rules.split_lines
# %% [markdown]
# By setting 'split_lines' to True, the line closest to each point will be split in two where the point is closest to the line. The points can then start their travels in the middle of lines. This makes things more accurate, but it takes a little more time.
#
# The split lines stays with the network until it is re-instantiated.
#
# Splitting the lines will have a larger effect if the lines in the network are long, and/or if the distances to be calculated are short.
#
# In this road network, most lines are short enough that splitting the lines usually doesn't do much. The longest lines are all in the forest.
# %%
nwa.network.gdf.length.describe()
# %% [markdown]
# It has a minimal impact on the results. Here comes one example (get_route) and the average travel minutes (od_cost_matrix).

# %%
nwa.rules.split_lines = False

od = nwa.od_cost_matrix(points, points)
sp1 = nwa.get_route(points.iloc[[97]], points.iloc[[135]])
sp1["split_lines"] = "Not splitted"

nwa.rules.split_lines = True

od = nwa.od_cost_matrix(points, points)
sp2 = nwa.get_route(points.iloc[[97]], points.iloc[[135]])
sp2["split_lines"] = "Splitted"
# %% [markdown]
# In the get_route example, when the lines are split, the trip starts a bit further up in the bottom-right corner (when the search_factor is 0). The trip also ends in a roundtrip, since the line that is split is a oneway street. So you're allowed to go to the intersection where the blue line goes, but not to the point where the line is cut.
# %%

gs.qtm(gs.gdf_concat([sp1, sp2]), column="split_lines", cmap="bwr")
# %% [markdown]
# But these kinds of deviations doesn't have much of an impact on the results in total here, where the mean is about 15 minutes. For shorter trips, the difference will be relatively larger, of course.
# %%

nwa.log.loc[
    nwa.log.method == "od_cost_matrix",
    ["split_lines", "cost_mean", "cost_p25", "cost_median", "cost_p75", "cost_std"],
]
# %%
nwa.rules.split_lines = False
# %% [markdown]
# If the point is located in the middle of a very long line, it has to travel all the way to the end of the line and then, half the time, traverse the whole line.
#
# #### search_factor
# Since the closest node might be intraversable, the points can be connected to all nodes within a given search_factor.
# The default is 0, which means the origins and destinations are only connected to the closest node.
# Setting the search_factor to for instance 10, would mean that 10 meters and 10 percent is added to the closest distance to a node.
#
# So if the closest node is 1 meter away, the point will be connected to all nodes within 11.1 meters.
#
# If the closest node is 100 meters away, the point will be connected to all nodes within 120 meters.
#
# Let's check how the search_factor influences the number of missing values:

# %%
for search_factor in [0, 10, 50, 100]:
    nwa.rules.search_factor = search_factor
    od = nwa.od_cost_matrix(points, points)

nwa.rules.search_factor = 0  # back to default

nwa.log.iloc[-4:][["search_factor", "percent_missing"]]
# %% [markdown]
# The remaining missing points are far away from the network. It might not be desirable to get results for these points. But if it is, it can be done with the search_tolerance parameter.

# #### search_tolerance
# search_tolerance is the maximum distance a start- or destination can be from the network. If the closest node is above the search_tolerance, this point will not be eligable for the analysis.
#
# The default is:
# %%
rules.search_tolerance
# %% [markdown]
# The search_tolerance unit is meters if the units of the crs is meters, which it is in this case:
# %%
nw.gdf.crs.axis_info[0].unit_name
# %% [markdown]
# Let's check how the search_tolerance influences the number of missing values:

# %%
for search_tolerance in [100, 250, 500, 5_000]:
    nwa.rules.search_tolerance = search_tolerance
    od = nwa.od_cost_matrix(points, points)

nwa.log.iloc[-4:][["search_tolerance", "percent_missing"]]
# %% [markdown]
# The remaining 0.2 percent are two points trapped behind oneway streets going the wrong way. A high search_tolerance won't help here, since the points are only connected to the closest node and the nodes within the search_factor. So the fix here (if a fix is desirable), is a higher search_tolerance (see above), but this will give more inaccurate results for the rest of the points. So consider using strict rules at first, then loosen up for only the points that give you problems.

# %%
nwa.rules.search_factor = 100

od = nwa.od_cost_matrix(points, points)

nwa.log.iloc[[-1]][["search_tolerance", "percent_missing"]]
# %%
# back to default:
nwa.rules.search_tolerance = 250
nwa.rules.search_factor = 0
# %% [markdown]
# Note: one of the points that had all missing values at a search_tolerance of 500, is on an island without a car ferry (but a regular ferry). With a search_tolerance of 5000, trips from this point will originate at the mainland with 0 weight penalty. If you want to include trips like this, it might be a good idea to give a weight for the trip to the mainland. this can be done with one of the 'weight_to_nodes_' parameters.

# #### weight_to_nodes_
# The class has three 'weight_to_nodes_' parameters. This is about the cost between the origins and destinations and the network nodes. All three paramters are set to False or None by default, meaning the cost will be 0.
#
# This will produce inaccurate results for points that are far away from the network. Especially when the search_factor is high.
#
# Therefore, you can set one of the 'weight_to_nodes_' parameters. If the weight is 'meters' (i.e. the length unit of the crs), setting 'weight_to_nodes_dist' to True will make the weight equivelant to the straight-line distance:
# %%

gs.NetworkAnalysisRules(weight="meters", weight_to_nodes_dist=True)
# %% [markdown]
# If the weight is "minutes", you specify the speed in kilometers or miles per hour:
# %%

gs.NetworkAnalysisRules(weight="minutes", weight_to_nodes_kmh=5)
# %%

gs.NetworkAnalysisRules(weight="minutes", weight_to_nodes_mph=3)
# %% [markdown]
# Let's check how the speed to the nodes influences the average speed:

# %%
for weight_to_nodes_kmh in [5, 20, 50, 0]:
    nwa.rules.weight_to_nodes_kmh = weight_to_nodes_kmh
    od = nwa.od_cost_matrix(points, points)

nwa.log.iloc[-4:][["weight_to_nodes_kmh", "cost_mean"]]
# %%

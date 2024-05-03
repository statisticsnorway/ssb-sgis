# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---
# %% [markdown]
# # Network analysis demo
# %%
import warnings

import numpy as np
import pandas as pd

import sgis as sg

# ignore some (for this purpose) irrelevant warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings(action="ignore", category=FutureWarning)
# %% [markdown]
# The network analysis happens in the NetworkAnalysis class.
# It takes a GeoDataFrame of lines and a set of rules for the analysis.
#
# The default rules can be instantiated like this (weight and directed do not have default values):
# %%
rules = sg.NetworkAnalysisRules(weight="minutes", directed=True)
rules
# %% [markdown]
# Now we need some road data:
# %%
roads = sg.read_parquet_url(
    "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet"
)
roads = roads[["oneway", "drivetime_fw", "drivetime_bw", "geometry"]]
roads.head(3)
# %% [markdown]
# For directed network analysis, we have to duplicate roads going both directions and
# flip roads going backwards. This can be done with the make_directed_network method:
# %%
directed_roads = sg.make_directed_network(
    roads,
    direction_col="oneway",
    direction_vals_bft=("B", "FT", "TF"),
    speed_col_kmh=None,
    minute_cols=("drivetime_fw", "drivetime_bw"),
    flat_speed_kmh=None,
)
directed_roads
# %% [markdown]
# Norwegian road data can be made directional with a custom function:
# %%
directed_roads = sg.make_directed_network_norway(roads)
directed_roads

# %% [markdown]
# We should also remove isolated network islands. Roads behind road blocks etc.
# %%
connected_roads = sg.get_connected_components(directed_roads).query("connected == 1")
connected_roads
# %% [markdown]
# ## NetworkAnalysis
#
# To start the network analysis, we put our roads and our rules into the NetworkAnalysis class:
# %%
nwa = sg.NetworkAnalysis(network=connected_roads, rules=rules)
nwa
# %% [markdown]
# We also need some points that will be our origins and destinations:
# %%
points = sg.read_parquet_url(
    "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet"
)
points
# %%
origins = points.iloc[:100]
destinations = points.iloc[100:200]
# %% [markdown]
# ### OD cost matrix

# od_cost_matrix calculates the traveltime from a set of origins to a set of destinations:
# %%
od = nwa.od_cost_matrix(origins, destinations)
od
# %% [markdown]
# The results can be joined with the origins or destinations via the index:
# %%
joined = origins.join(od.set_index("origin"))
joined

# %% [markdown]
# Or aggregate values can be assigned directly onto the origins/destinations:
# %%
origins["minutes_min"] = od.groupby("origin")["minutes"].min()
origins["minutes_mean"] = od.groupby("origin")["minutes"].mean()
origins["n_missing"] = len(origins) - od.groupby("origin")["minutes"].count()
origins
# %%
origins = origins.drop(["minutes_min", "minutes_mean", "n_missing"], axis=1)
# %% [markdown]
# Set 'lines' to True to get a geometry column with straight lines between origin and
# destination:
# %%
od = nwa.od_cost_matrix(points.iloc[[0]], points, lines=True)

m = sg.ThematicMap(od, column="minutes", black=True)
m.title = "Travel time (minutes) from 1 to 1000 addresses."
m.scheme = "quantiles"
m.plot()

# %% [markdown]
# Information about the analyses are stored in a DataFrame in the 'log' attribute.
# %%
print(nwa.log)
# %% [markdown]
# ### Get route

# The get_route method can be used to get the actual lowest cost path:

# %%
routes = nwa.get_route(origins.iloc[[0]], destinations)

# plot the results
m = sg.ThematicMap(sg.buff(routes, 15), column="minutes", black=True)
m.cmap = "plasma"
m.title = "Travel times (minutes)"
m.plot()
# %% [markdown]
# ### Get route frequencies
#
# get_route_frequencies finds the number of times each road segment was used.

# %%
frequencies = nwa.get_route_frequencies(origins, destinations)

# plot the results
m = sg.ThematicMap(sg.buff(frequencies, 15), column="frequency", black=True)
m.cmap = "plasma"
m.title = "Number of times each road was used"
m.plot()
# %% [markdown]
# The results will be quite different when it is the shortest, rather than the fastest,
# route that is used:
# %%
nwa.rules.weight = "meters"

frequencies = nwa.get_route_frequencies(origins, destinations)

m = sg.ThematicMap(sg.buff(frequencies, 15), column="frequency", black=True)
m.cmap = "plasma"
m.title = "Number of times each road was used (weight='meters')"
m.plot()
# %%
nwa.rules.weight = "minutes"
# %% [markdown]
# The routes can also be weighted with a DataFrame containing the indices of all
# origins and destinations combinations and the weight for the trip between them.
#
# Creating uniform weights of 10 for illustration's sake.
# %%
# creating long DataFrame of all od combinations
od_pairs = pd.MultiIndex.from_product([origins.index, destinations.index])
weights = pd.DataFrame(index=od_pairs)
weights["weight"] = 10

frequencies = nwa.get_route_frequencies(origins, destinations, weight_df=weights)

m = sg.ThematicMap(sg.buff(frequencies, 15), column="frequency", black=True)
m.cmap = "plasma"
m.title = "Number of times each road was used"
m.plot()
# %% [markdown]
# If not all origin-destination combinations are in the weight_df,
# a 'default_weight' has to be set. Setting the default to 1 and creating
# a weight_df with one OD pair with a very high weight.
# %%
od_pair = pd.MultiIndex.from_product([[1], [101]])
od_pair = pd.DataFrame({"weight": [100_000]}, index=od_pair)

frequencies = nwa.get_route_frequencies(
    origins, destinations, weight_df=od_pair, default_weight=1
)

# plot the results
m = sg.ThematicMap(sg.buff(frequencies, 15), column="frequency", black=True)
m.cmap = "plasma"
m.title = "Number of times each road was used"
m.plot()

# %% [markdown]
# ### Service area

# The service_area method finds the area that can be reached within one or more breaks.
#
# Here, we find the areas that can be reached within 5, 10 and 15 minutes for five random points:
# %%
service_areas = nwa.service_area(origins.sample(5), breaks=(5, 10, 15))
service_areas
# %% [markdown]
# Here we get the areas that can be reached within 1 to 10 minutes from one point:
# %%
service_areas = nwa.service_area(points.iloc[[0]], breaks=np.arange(1, 11))

m = sg.ThematicMap(service_areas, column="minutes", black=True)
m.k = 10
m.title = "Roads that can be reached within 1 to 10 minutes"
m.plot()
# %% [markdown]
# By default, only the lowest break is kept for overlapping areas from the same origin, meaning the area for minutes=10
# covers not the entire area, only the outermost ring:

# %%
sg.qtm(
    service_areas.query("minutes == 10"),
    color="yellow",
    title="Roads that can be reached within 10 minutes",
)
# %% [markdown]
# Set 'dissolve' to False to get each individual road or line returned,
# and then drop rows afterwards:
# %%
service_areas = nwa.service_area(points.sample(100), breaks=5, dissolve=False)
print("rows before drop_duplicates:", len(service_areas))
service_areas = service_areas.drop_duplicates(["source", "target"])
print("rows after drop_duplicates:", len(service_areas))
# %% [markdown]
# Let's check the log.
# %%
print(nwa.log)
# %% [markdown]
# ## Customising the network
# %%
connected_and_not = sg.get_connected_components(directed_roads)
connected_and_not["connected_str"] = np.where(
    connected_and_not.connected == 1, "yes", "no"
)

to_plot = connected_and_not.clip(points.iloc[[0]].buffer(1000))
sg.qtm(to_plot, column="connected_str", title="Connected and isolated networks")
# %% [markdown]
# Use the get_connected_components method to get the largest network, then remove rows not part of this network:
# %%
connected_roads = sg.get_connected_components(roads).query("connected == 1")
# %% [markdown]
# If the road data has some gaps between the segments, these can be filled with straight lines (the weight column will then have NaN values):
# %%

connected_roads = sg.close_network_holes(
    connected_roads, max_distance=1.5, max_angle=90
)
connected_roads
# %% [markdown]
# ### split_lines
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

sg.qtm(sp1, sp2, column="split_lines", cmap="bwr")
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
# ### search_factor
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

# ### search_tolerance
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

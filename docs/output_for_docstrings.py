# %%
"""Prints outputs to paste into docstrings. Run this in the terminal:

poetry run python docs/output_for_docstrings.py

"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


src = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src)

import sgis as sg


def print_function_name(func):
    def wrapper(*args, **kwargs):
        print("\n\n\n", func.__name__, "\n\n\n")
        func(*args, **kwargs)

    return wrapper


@print_function_name
def legend_docstring():
    import sgis as sg

    points = sg.random_points(10)
    points["number"] = range(10)
    print(points)

    # Creating the ThematicMap instance.

    m = sg.ThematicMap(points, column="number")
    print(m.legend)
    print(type(m.legend))

    # Changing the attributes that apply to both numeric and categorical columns.
    m.legend.title = "Number"
    m.legend.title_fontsize = 11
    m.legend.fontsize = 9
    m.legend.markersize = 7.5
    m.legend.position = (0.8, 0.2)
    m.legend.kwargs["labelcolor"] = "red"
    m.plot()

    # Changing the additional attributes that only apply only to numeric columns.

    m = sg.ThematicMap(points, column="number")
    m.legend.label_sep = "to"
    m.legend.label_suffix = "num"
    m.legend.rounding = 2
    m.plot()

    # The final attribute, labels, should be changed along with the bins attribute
    # of the ThematicMap class, for numeric columns. The following bins will create a
    # plot with the color groups 0-2, 3-5, 6-7 and 8-9. The legend labels can then be
    # set accordingly.

    m = sg.ThematicMap(points, column="number")
    m.bins = [0, 2, 5, 7, 9]
    m.legend.labels = ["0 to 2 num", "3 to 5 num", "6 to 7 num", "8 to 9 num"]
    m.plot()

    m = sg.ThematicMap(points, column="number")
    m.bins = [2, 5, 7]
    m.legend.labels = ["0 to 2 num", "3 to 5 num", "6 to 7 num", "8 to 9 num"]
    m.plot()

    # For categorical columns, labels can be specified as a dictionary with the
    # original or new column values.

    points["group"] = np.random.choice([*"abc"], size=10)
    m = sg.ThematicMap(points, column="group")
    m.legend.labels = {"a": "A", "b": "B", "c": "C"}
    m.plot()

    points["group"] = np.random.choice([*"abc"], size=10)
    labels = {"a": "A", "b": "B", "c": "C"}
    points["label"] = points["group"].map(labels)
    m = sg.ThematicMap(points, column="label")
    print(type(m.legend))
    m.legend.title = "Label"
    m.plot()


@print_function_name
def get_neighbor_indices_docstring():
    from sgis import get_neighbor_indices, to_gdf

    points = to_gdf([(0, 0), (0.5, 0.5)])
    points["text"] = [*"ab"]
    print(get_neighbor_indices(points, points))
    print(get_neighbor_indices(points, points, max_distance=1))
    print(get_neighbor_indices(points, points.set_index("text"), max_distance=1))

    neighbor_indices = get_neighbor_indices(
        points, points.set_index("text"), max_distance=1
    )
    print(neighbor_indices.values)
    print(neighbor_indices.index)


get_neighbor_indices_docstring()


@print_function_name
def networkanalysis_doctring(nwa, points):
    od = nwa.od_cost_matrix(points, points)
    print(od)
    print("\n")

    routes = nwa.get_route(points.sample(10), points.sample(10))
    print(routes)
    print("\n")

    frequencies = nwa.get_route_frequencies(points.sample(25), points.sample(25))
    print(frequencies[["source", "target", "frequency", "geometry"]])
    print("\n")

    service_areas = nwa.service_area(
        points.iloc[:3],
        breaks=[5, 10, 15],
    )
    print(service_areas)
    print("\n")

    print(nwa.log)
    print("\n")


@print_function_name
def networkanalysisrules_docstring():
    import sgis as sg

    roads = sg.read_parquet_url(
        "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet"
    )
    points = sg.read_parquet_url(
        "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet"
    )

    nw = sg.DirectedNetwork(roads).remove_isolated().make_directed_network_norway()
    rules = sg.NetworkAnalysisRules(weight="minutes")
    nwa = sg.NetworkAnalysis(network=nw, rules=rules)
    print(nwa)

    od = nwa.od_cost_matrix(points, points)
    nwa.rules.split_lines = True
    od = nwa.od_cost_matrix(points, points)
    print(nwa.log[["split_lines", "percent_missing", "cost_mean"]])
    nwa.rules.split_lines = False

    for i in [100, 250, 500, 1000]:
        print(i)
        nwa.rules.search_tolerance = i
        od = nwa.od_cost_matrix(points, points)

    print(
        nwa.log.iloc[-4:][
            ["percent_missing", "cost_mean", "search_tolerance", "search_factor"]
        ]
    )

    nwa.rules.search_tolerance = 250
    for i in [0, 10, 35, 100]:
        nwa.rules.search_factor = i
        od = nwa.od_cost_matrix(points, points)

    print(
        nwa.log.iloc[-4:][
            ["percent_missing", "cost_mean", "search_tolerance", "search_factor"]
        ]
    )

    n_missing = od.groupby("origin").minutes.agg(lambda x: x.isna().sum())
    print(n_missing.nlargest(3))

    nwa.rules.search_tolerance = 5000
    for i in [3, 10, 50]:
        nwa.rules.nodedist_kmh = i
        od = nwa.od_cost_matrix(points, points)

    print(nwa.log.iloc[-3:][["nodedist_kmh", "cost_mean"]])

    rules = sg.NetworkAnalysisRules(
        weight="meters",
        search_tolerance=5000,
    )
    nwa = sg.NetworkAnalysis(network=nw, rules=rules)
    od = nwa.od_cost_matrix(points, points)
    nwa.rules.nodedist_multiplier = 1
    od = nwa.od_cost_matrix(points, points)

    print(nwa.log[["nodedist_multiplier", "cost_mean"]])


@print_function_name
def get_k_routes_docstring(nwa, points):
    p1, p2 = points.iloc[[0]], points.iloc[[1]]
    k_routes = nwa.get_k_routes(p1, p2, k=10, drop_middle_percent=1)
    print(k_routes)
    print("\n")

    k_routes = nwa.get_k_routes(p1, p2, k=10, drop_middle_percent=50)
    print(k_routes)
    print("\n")

    k_routes = nwa.get_k_routes(p1, p2, k=10, drop_middle_percent=100)
    print(k_routes)
    print("\n")


@print_function_name
def get_route_docstring(nwa, points):
    routes = nwa.get_route(points.iloc[[0]], points)
    print(routes)
    print("\n")


@print_function_name
def get_route_frequencies_docstring(nwa, points):
    origins = points.iloc[:25]
    destinations = points.iloc[25:50]
    frequencies = nwa.get_route_frequencies(origins, destinations)
    print(frequencies[["source", "target", "frequency", "geometry"]])

    od_pairs = pd.MultiIndex.from_product(
        [origins.index, destinations.index], names=["origin", "destination"]
    )
    weight_df = pd.DataFrame(index=od_pairs).reset_index()
    weight_df["weight"] = 10
    print(weight_df)

    frequencies = nwa.get_route_frequencies(origins, destinations, weight_df=weight_df)
    print(frequencies[["source", "target", "frequency", "geometry"]])

    weight_df = pd.DataFrame(index=od_pairs)
    weight_df["weight"] = 10
    print(weight_df)

    frequencies = nwa.get_route_frequencies(origins, destinations, weight_df=weight_df)
    print(frequencies[["source", "target", "frequency", "geometry"]])


@print_function_name
def service_area_docstring(nwa, points):
    service_areas = nwa.service_area(
        points.loc[:2],
        breaks=10,
    )
    print(service_areas)
    print("\n")

    service_areas = nwa.service_area(
        points.iloc[:2],
        breaks=[5, 10, 15],
    )
    print(service_areas)
    print("\n")


@print_function_name
def od_cost_matrix_docstring(nwa, points):
    origins = points.loc[:99, ["geometry"]]
    print(origins)
    print("\n")

    destinations = points.loc[100:199, ["geometry"]]
    print(destinations)
    print("\n")

    od = nwa.od_cost_matrix(origins, destinations)
    print(od)
    print("\n")

    joined = origins.join(od.set_index("origin"))
    print(joined)
    print("\n")

    print("less_than_10_min")
    less_than_10_min = od.loc[od.minutes < 10]
    joined = origins.join(less_than_10_min.set_index("origin"))
    print(joined)
    print("\n")

    print("three_fastest")
    three_fastest = od.loc[od.groupby("origin")["minutes"].rank() <= 3]
    joined = origins.join(three_fastest.set_index("origin"))
    print(joined)
    print("\n")

    print("aggregate onto the origins")
    origins["minutes_mean"] = od.groupby("origin")["minutes"].mean()
    print(origins)
    print("\n")

    print("use different column")
    origins["letter"] = np.random.choice([*"abc"], len(origins))
    od = nwa.od_cost_matrix(origins.set_index("letter"), destinations)
    print(od)
    print("\n")

    points_reversed = points.iloc[::-1].reset_index(drop=True)
    od = nwa.od_cost_matrix(points, points_reversed, rowwise=True)
    print(od)
    print("\n")


@print_function_name
def buffdiss_docstring(points):
    points = points[["geometry"]]
    points["group"] = np.random.choice([*"abd"], len(points))
    points["number"] = np.random.random(size=len(points))
    print(points)

    print(sg.buffdiss(points, 250))
    print("\n")
    print(sg.buffdiss(points, 250, by="group", aggfunc="sum"))
    print("\n")
    print(sg.buffdiss(points, 250, by="group", as_index=False))
    print("\n")

    aggcols = points.groupby("group").agg(
        numbers_sum=("number", "count"),
        numbers_mean=("number", "mean"),
        n=("number", "count"),
    )
    print(aggcols)
    print("\n")

    points_agg = (
        sg.buffdiss(points, 250, by="group")[["geometry"]].join(aggcols).reset_index()
    )
    print(points_agg)


@print_function_name
def buffdissexp_docstring(points):
    points = points[["geometry"]]
    points["group"] = np.random.choice([*"abd"], len(points))
    points["number"] = np.random.random(size=len(points))
    print(points)
    print("\n")

    print(sg.buffdissexp(points, 250))
    print("\n")
    print(sg.buffdissexp(points, 250, by="group"))
    print("\n")
    print(sg.buffdissexp(points, 250, by="group", as_index=False))
    print("\n")


@print_function_name
def get_k_neighbors_docstring():
    from sgis import get_k_nearest_neighbors, random_points

    points = random_points(100)
    neighbors = random_points(100)

    distances = get_k_nearest_neighbors(points, neighbors, k=10)
    print(distances)
    print("\n")

    neighbors["custom_id"] = [letter for letter in [*"abcde"] for _ in range(20)]
    distances = get_k_nearest_neighbors(points, neighbors.set_index("custom_id"), k=10)
    print(distances)
    print("\n")

    joined = points.join(distances)
    joined["k"] = joined.groupby(level=0)["distance"].transform("rank")
    print(joined)
    print("\n")

    points["mean_distance"] = distances.groupby(level=0)["distance"].mean()
    points["min_distance"] = distances.groupby(level=0)["distance"].min()
    print(points)
    print("\n")


@print_function_name
def get_all_distances_docstring():
    from sgis import get_all_distances, random_points

    points = random_points(100)
    neighbors = random_points(100)

    distances = get_all_distances(points, neighbors)
    print(distances)
    print("\n")

    neighbors["custom_id"] = [letter for letter in [*"abcde"] for _ in range(20)]
    distances = get_all_distances(points, neighbors.set_index("custom_id"))
    print(distances)
    print("\n")

    joined = points.join(distances)
    print(joined)
    print("\n")

    points["mean_distance"] = distances.groupby(level=0)["distance"].mean()
    points["min_distance"] = distances.groupby(level=0)["distance"].min()
    print(points)
    print("\n")


def get_neighbor_indices():
    from sgis import get_neighbor_indices, to_gdf

    points = to_gdf([(0, 0), (0.5, 0.5), (2, 2)])
    points

    p1 = points.iloc[[0]]
    print(
        get_neighbor_indices(p1, points),
        get_neighbor_indices(p1, points, max_distance=1),
        get_neighbor_indices(p1, points, max_distance=3),
    )

    points["text"] = [*"abd"]
    print(get_neighbor_indices(p1, points.set_index("text"), max_distance=3))


def make_docstring_output():
    get_neighbor_indices()

    get_k_neighbors_docstring()

    get_all_distances_docstring()

    points = sg.read_parquet_url(
        "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet"
    )

    buffdiss_docstring(points)
    buffdissexp_docstring(points)

    roads = sg.read_parquet_url(
        "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet"
    )
    roads = roads[["oneway", "drivetime_fw", "drivetime_bw", "geometry"]]
    nw = sg.DirectedNetwork(roads).remove_isolated().make_directed_network_norway()
    rules = sg.NetworkAnalysisRules(weight="minutes")

    from sgis import NetworkAnalysis

    directed_isolated_dropped = NetworkAnalysis(network=nw, rules=rules)

    networkanalysis_doctring(directed_isolated_dropped, points)

    networkanalysisrules_docstring()

    od_cost_matrix_docstring(directed_isolated_dropped, points)
    get_k_routes_docstring(directed_isolated_dropped, points)
    service_area_docstring(directed_isolated_dropped, points)
    get_route_frequencies_docstring(directed_isolated_dropped, points)
    get_route_docstring(directed_isolated_dropped, points)


if __name__ == "__main__":
    make_docstring_output()

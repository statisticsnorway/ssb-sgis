"""Prints outputs to paste into docstrings. Run this in the terminal:

poetry run python docs/output_for_docstrings.py

"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


src = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src)

import sgis as sg


def print_function_name(func):
    def wrapper(*args, **kwargs):
        print("\n", func.__name__)
        func(*args, **kwargs)

    return wrapper


@print_function_name
def networkanalysis_doctring(nwa, points):
    od = nwa.od_cost_matrix(points, points)
    print(od)
    print("\n")

    routes = nwa.get_route(points.sample(10), points.sample(10))
    print(routes)
    print("\n")

    frequencies = nwa.get_route_frequencies(points.sample(25), points.sample(25))
    print(frequencies)
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
    frequencies = nwa.get_route_frequencies(points.sample(25), points.sample(25))
    print(frequencies)
    print("\n")


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
        get_neighbor_indices(p1, points, max_dist=1),
        get_neighbor_indices(p1, points, max_dist=3),
    )

    points["text"] = [*"abd"]
    print(get_neighbor_indices(p1, points.set_index("text"), max_dist=3))


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
    roads = roads[["oneway", "drivetime_fw", "drivetime_bw", "roadid", "geometry"]]
    nw = sg.DirectedNetwork(roads).remove_isolated().make_directed_network_norway()
    rules = sg.NetworkAnalysisRules(weight="minutes")

    from sgis import NetworkAnalysis

    directed_isolated_dropped = NetworkAnalysis(network=nw, rules=rules)

    od_cost_matrix_docstring(directed_isolated_dropped.deepcopy(), points)
    get_k_routes_docstring(directed_isolated_dropped.deepcopy(), points)
    service_area_docstring(directed_isolated_dropped.deepcopy(), points)
    get_route_frequencies_docstring(directed_isolated_dropped.deepcopy(), points)
    get_route_docstring(directed_isolated_dropped.deepcopy(), points)

    networkanalysis_doctring(directed_isolated_dropped, points)


if __name__ == "__main__":
    make_docstring_output()

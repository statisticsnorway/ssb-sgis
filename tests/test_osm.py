# %%
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio


src = str(Path(__file__).parent).strip("tests") + "src"

import sys


sys.path.insert(0, src)

import sgis as sg


print(sys.executable)


def osm_api():
    "https://api.openstreetmap.org/"

    import xml.etree.ElementTree as ET

    import osmapi

    api = osmapi.OsmApi()

    bbox = (10.85, 59.85, 10.9, 59.9)
    data = api.Map(*bbox)

    print(data)
    # Parse the XML data using ElementTree
    root = ET.fromstring(data)

    # Extract the road data from the XML data
    roads = []
    for way in root.findall(".//way"):
        if "highway" in way.attrib and way.attrib["highway"] != "pedestrian":
            road = {}
            road["id"] = way.attrib["id"]
            road["name"] = way.find('.//tag[@k="name"]').attrib["v"]
            road["type"] = way.attrib["highway"]
            road["nodes"] = [nd.attrib["ref"] for nd in way.findall("nd")]
            roads.append(road)

    # Print the extracted road data
    print(roads)


def read_osm():
    from pyrosm import OSM, get_data

    # Initialize reader
    osm = OSM(get_data("test_pbf"))

    # Read nodes and edges of the 'driving' network
    nodes, edges = osm.get_network(nodes=True, network_type="driving")

    # Plot nodes and edges on a map
    ax = edges.plot(figsize=(6, 6), color="gray")
    ax = nodes.plot(ax=ax, color="red", markersize=2.5)


def not_test_osm():
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    #    warnings.filterwarnings(action="ignore", category=UserWarning)
    pd.options.mode.chained_assignment = None

    split_lines = True

    ### READ FILES

    osm_path = r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\osm_oslo.parquet"
    if Path(osm_path).exists():
        r = gpd.read_parquet(osm_path)
    else:
        osm_path_norway = (
            r"C:\Users\ort\Downloads\norway-latest-free.shp\gis_osm_roads_free_1.shp"
        )

        crs = pyogrio.read_info(osm_path_norway)["crs"]

        oslo = (
            gpd.read_file(
                r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\Basisdata_0000_Norge_25833_Kommuner_FGDB.gdb",
                layer="kommune",
            )
            .query("kommunenummer == '0301'")
            .to_crs(crs)
        )

        r = gpd.read_file(osm_path_norway, engine="pyogrio", bbox=oslo)
        r = r.to_crs(25833)

        r.to_parquet(
            r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\osm_oslo.parquet"
        )

    p = gpd.read_parquet(Path(__file__).parent / "testdata" / "points_oslo.parquet")
    p["idx"] = p.index
    p["idx2"] = p.index
    print(p.idx)

    ### MAKE THE ANALYSIS CLASS
    print(r.oneway.value_counts())
    print(r.maxspeed.value_counts())
    print(r.oneway.isna().sum())
    print(r.maxspeed.isna().sum())
    print(r.maxspeed.value_counts())

    r["maxspeed"] = np.where((r.maxspeed.isna()) | (r.maxspeed == 0), 40, r.maxspeed)
    print(r.maxspeed.value_counts())

    for t1 in [False, True]:
        for t2 in [False, True]:
            nw = (
                sg.get_connected_components(r)
                .query("connected == 1")
                .pipe(
                    sg.make_directed_network,
                    direction_col="oneway",
                    direction_vals_bft=("B", "F", "T"),
                    speed_col_kmh="maxspeed",
                    reverse_tofrom=t2,
                )
            )
            rules = sg.NetworkAnalysisRules(
                weight="meters", directed=t1, split_lines=split_lines
            )
            nwa = sg.NetworkAnalysis(nw, rules=rules, detailed_log=True)

            ### OD COST MATRIX

            for search_factor in [25]:
                print("hei")
                nwa.rules.search_factor = search_factor
                od = nwa.od_cost_matrix(p.iloc[[0]], p, lines=True)

                print(sg.qtm(od, nwa.rules.weight))
                print(nwa.log)

            freq = nwa.get_route_frequencies(p.sample(50), p.sample(50))

            sg.qtm(sg.buff(freq, 15), "frequency", cmap="plasma")


def main():
    not_test_osm()


if __name__ == "__main__":
    main()

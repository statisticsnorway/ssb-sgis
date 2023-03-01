# %%
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio


src = str(Path(__file__).parent).strip("tests") + "src"

import sys


sys.path.insert(0, src)

import gis_utils as gs


def osm_api():
    "https://api.openstreetmap.org/"


def not_test_osm():
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    #    warnings.filterwarnings(action="ignore", category=UserWarning)
    pd.options.mode.chained_assignment = None

    split_lines = True

    ### READ FILES

    osm_path = r"C:\Users\ort\Downloads\norway-latest-free.shp\gis_osm_roads_free_1.shp"

    crs = pyogrio.read_info(osm_path)["crs"]

    oslo = (
        gpd.read_file(
            r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\Basisdata_0000_Norge_25833_Kommuner_FGDB.gdb",
            layer="kommune",
        )
        .query("kommunenummer == '0301'")
        .to_crs(crs)
    )

    r = gpd.read_file(osm_path, engine="pyogrio", bbox=oslo)
    r = r.to_crs(25833)

    p = gpd.read_parquet(Path(__file__).parent / "testdata" / "random_points.parquet")
    p["idx"] = p.index
    p["idx2"] = p.index
    print(p.idx)

    ### MAKE THE ANALYSIS CLASS
    print(r.oneway.value_counts())
    print(r.maxspeed.value_counts())
    print(r.oneway.isna().sum())
    print(r.maxspeed.isna().sum())

    for t1 in [True, False]:
        for t2 in [True, False]:
            nw = (
                gs.DirectedNetwork(r)
                .remove_isolated()
                .make_directed_network(
                    direction_col="oneway",
                    direction_vals_bft=("B", "F", "T"),
                    speed_col="maxspeed",
                    default_speed=40,
                    reverse1=t1,
                    reverse2=t2,
                )
            )
            rules = gs.NetworkAnalysisRules(weight="meters", split_lines=split_lines)
            nwa = gs.NetworkAnalysis(nw, rules=rules)

            ### OD COST MATRIX

            freq = nwa.get_route_frequencies(p.sample(75), p.sample(75))

            gs.qtm(
                gs.buff(freq, 15),
                "n",
                scheme="naturalbreaks",
                cmap="plasma",
                title="Number of times each road was used.",
            )

            for search_factor in [25]:
                print("hei")
                nwa.rules.search_factor = search_factor
                od = nwa.od_cost_matrix(p, p)

                print(
                    nwa.log[
                        [
                            "search_tolerance",
                            "search_factor",
                            "percent_missing",
                            "cost_mean",
                        ]
                    ]
                )
            break
        break


def main():
    not_test_osm()


if __name__ == "__main__":
    main()

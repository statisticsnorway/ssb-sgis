# %%
import warnings
from pathlib import Path

import geopandas as gpd

import gis_utils as gs


def test_shortest_path():
    warnings.filterwarnings(action="ignore", category=UserWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    p = gpd.read_parquet(Path(__file__).parent / "testdata" / "random_points.parquet")
    p["idx"] = p.index
    p["idx2"] = p.index
    r = gpd.read_parquet(Path(__file__).parent / "testdata" / "roads_oslo_2022.parquet")

    nw = gs.DirectedNetwork(r).make_directed_network_norway().remove_isolated()

    nwa = gs.NetworkAnalysis(nw, cost="minutes")

    sp = nwa.shortest_path(p.iloc[[0]], p.sample(250), id_col="idx", summarise=True)

    sp = nwa.shortest_path(p.sample(50), p.sample(50), summarise=True)

    sp = nwa.shortest_path(
        p.sample(25),
        p.sample(25),
    )


def main():
    test_shortest_path()


if __name__ == "__main__":
    main()
# %%

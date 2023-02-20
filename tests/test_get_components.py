# %%
from pathlib import Path
from time import perf_counter

import geopandas as gpd

import gis_utils as gs


def not_test_get_components():
    p = gpd.read_parquet(Path(__file__).parent / "testdata" / "random_points.parquet")
    p["idx"] = p.index
    p["idx2"] = p.index

    r = gpd.read_parquet(Path(__file__).parent / "testdata" / "roads_oslo_2022.parquet")
    nw = gs.DirectedNetwork(r)

    nw = nw.get_component_size()
    gs.qtm(
        nw.gdf.loc[nw.gdf.component_size != max(nw.gdf.component_size)].sjoin(
            gs.buff(p, 1000)
        ),
        "component_size",
        scheme="quantiles",
        k=7,
    )

    _time = perf_counter()
    nw = nw.get_largest_component()
    print("n", sum(nw.gdf.connected == 0))
    print("time get_largest_component: ", perf_counter() - _time)

    gs.qtm(
        nw.gdf.sjoin(gs.buff(p, 1000)), "connected", cmap="bwr", scheme="equalinterval"
    )


def main():
    not_test_get_components()


if __name__ == "__main__":
    main()

# %%

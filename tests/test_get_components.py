# %%
from pathlib import Path
from time import perf_counter

import geopandas as gpd

import sgis as sg


def not_test_get_components(roads_oslo, points_oslo):
    p = points_oslo
    p["idx"] = p.index
    p["idx2"] = p.index

    r = roads_oslo
    nw = sg.DirectedNetwork(r)

    nw = nw.get_component_size()
    sg.qtm(
        nw.gdf.loc[nw.gdf.component_size != max(nw.gdf.component_size)].sjoin(
            sg.buff(p, 1000)
        ),
        "component_size",
        scheme="quantiles",
        k=7,
    )

    _time = perf_counter()
    nw = nw.get_largest_component()
    print("n", sum(nw.gdf.connected == 0))
    print("time get_largest_component: ", perf_counter() - _time)

    sg.qtm(
        nw.gdf.sjoin(sg.buff(p, 1000)), "connected", cmap="bwr", scheme="equalinterval"
    )


def main():
    from oslo import points_oslo, roads_oslo

    not_test_get_components(roads_oslo(), points_oslo())


if __name__ == "__main__":
    main()

# %%

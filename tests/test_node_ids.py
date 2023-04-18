# %%
import sys
from pathlib import Path

import geopandas as gpd


src = str(Path(__file__).parent).strip("tests") + "src"

sys.path.insert(0, src)

import sgis as sg


def test_node_ids(points_oslo, roads_oslo):
    p = points_oslo
    p = sg.clean_clip(p, p.geometry.iloc[0].buffer(500))
    p["idx"] = p.index
    p["idx2"] = p.index

    r = roads_oslo
    r = sg.clean_clip(r, p.geometry.iloc[0].buffer(600))

    r, nodes = sg.make_node_ids(r)
    print(nodes)
    r, nodes = sg.make_node_ids(r, wkt=False)
    print(nodes)


def main():
    from oslo import points_oslo, roads_oslo

    test_node_ids(points_oslo(), roads_oslo())


if __name__ == "__main__":
    main()

# %%

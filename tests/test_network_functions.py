# %%
import sys
from pathlib import Path

import geopandas as gpd


src = str(Path(__file__).parent).strip("tests") + "src"


sys.path.insert(0, src)
import sgis as sg


def test_network_functions(points_oslo, roads_oslo):
    p = points_oslo
    p = p.iloc[[0]]

    r = roads_oslo
    r = sg.clean_clip(r, p.buffer(1000)).explode(ignore_index=True)

    r2 = sg.get_largest_component(r)
    sg.qtm(r2, column="connected", scheme="equalinterval", title="connected")

    r2 = sg.close_network_holes(r2, 1.1).pipe(sg.cut_lines, 250)

    if (l := max(r2.length)) > 250 + 1:
        raise ValueError(f"cut_lines did not cut lines. max line length: {l}")

    r2 = r2.loc[r2.connected == 1]
    sg.qtm(r2, column="connected", title="after removing isolated")

    holes_closed = sg.close_network_holes(r, 10.1)
    print(holes_closed.hole.value_counts())
    sg.qtm(holes_closed, column="hole", title="holes")

    holes_closed = sg.close_network_holes(r, 10.1, deadends_only=True)
    print(holes_closed.hole.value_counts())
    sg.qtm(holes_closed, column="hole", title="holes, deadends_only")


def main():
    from oslo import points_oslo, roads_oslo

    test_network_functions(points_oslo(), roads_oslo())


if __name__ == "__main__":
    main()

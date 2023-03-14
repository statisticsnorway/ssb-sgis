# %%
import sys
from pathlib import Path

import geopandas as gpd


src = str(Path(__file__).parent).strip("tests") + "src"


sys.path.insert(0, src)
import sgis as sg


def test_network_methods(points_oslo, roads_oslo):
    points = points_oslo
    p = points.iloc[[0]]
    points = sg.clean_clip(points, p.buffer(700))

    r = roads_oslo
    r = sg.clean_clip(r, p.buffer(1000))

    nw1 = sg.Network(r).get_largest_component()
    sg.qtm(nw1.gdf, column="connected", scheme="equalinterval", title="connected")

    len_now = len(nw1.gdf)

    nw = nw1.copy().remove_isolated().cut_lines(250)

    # check that the copy method works
    assert len(nw1.gdf) == len_now

    if (l := max(nw.gdf.length)) > 250 + 1:
        raise ValueError(f"cut_lines did not cut lines. max line length: {l}")

    sg.qtm(nw.gdf, column="connected", title="after removing isolated")

    holes_closed = sg.Network(r).close_network_holes(10.1, fillna=0).gdf
    print(holes_closed.hole.value_counts())
    sg.qtm(holes_closed, column="hole", title="holes")

    holes_closed2 = (
        sg.Network(r).close_network_holes(10.1, fillna=0, deadends_only=True).gdf
    )
    print(holes_closed2.hole.value_counts())
    sg.qtm(holes_closed2, column="hole", title="holes, deadends_only")

    """
    holes = holes_closed.query("hole==1").assign(hole=1)[["hole", "geometry"]]
    holes2 = holes_closed2.query("hole==1").assign(hole2=2)[["hole2", "geometry"]]
    sg.concat_explore(holes.overlay(holes2, how="symmetric_difference").pipe(sg.buff, 1), r[["geometry"]])
    """

    nw = (
        sg.Network(r)
        .close_network_holes(1.1, fillna=0, deadends_only=False)
        .remove_isolated()
    )
    sg.qtm(nw.gdf)
    rules = sg.NetworkAnalysisRules(
        weight="meters",
    )

    nwa = sg.NetworkAnalysis(nw, rules=rules)
    print(nwa)
    x = nwa.get_route_frequencies(p, points.sample(10))
    sg.qtm(x, "n")


def main():
    from oslo import points_oslo, roads_oslo

    test_network_methods(points_oslo(), roads_oslo())


if __name__ == "__main__":
    main()

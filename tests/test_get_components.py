# %%
import sys
from pathlib import Path


src = str(Path(__file__).parent).strip("tests") + "src"

sys.path.insert(0, src)

import sgis as sg


def test_get_components(roads_oslo, points_oslo):
    p = points_oslo
    p["idx"] = p.index
    p["idx2"] = p.index

    r = roads_oslo
    r = sg.clean_clip(r, p.iloc[[0]].buffer(500))
    assert len(r) == 68 + 488, len(r)

    nw = sg.get_component_size(r)

    if __name__ == "__main__":
        sg.qtm(
            nw.loc[nw.component_size != max(nw.component_size)].sjoin(sg.buff(p, 1000)),
            "component_size",
            scheme="quantiles",
            k=7,
        )

    nw = sg.get_connected_components(r)

    assert sum(nw.connected == 0) == 68
    assert sum(nw.connected == 0) == 68
    print("n", sum(nw.connected == 0))
    print("n", sum(nw.connected == 1))

    if __name__ == "__main__":
        sg.qtm(
            nw.sjoin(sg.buff(p, 1000)),
            "connected",
            cmap="bwr",
        )


def main():
    from oslo import points_oslo, roads_oslo

    test_get_components(roads_oslo(), points_oslo())


if __name__ == "__main__":
    main()

# %%

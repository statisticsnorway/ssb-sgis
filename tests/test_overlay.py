# %%
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.wkt import loads


src = str(Path(__file__).parent).strip("tests") + "src"

sys.path.insert(0, src)

import sgis as sg


def test_overlay(points_oslo):
    warnings.filterwarnings(action="ignore", category=UserWarning)

    p = points_oslo
    p = p.iloc[:50]

    p500 = sg.buff(p, 500)
    p1000 = sg.buff(p, 1000)

    updated = sg.overlay_update(p500, p1000)
    if __name__ == "__main__":
        updated["area_"] = updated.area
        sg.qtm(updated, "area_")
    updated = sg.overlay_update(p1000, p500)
    if __name__ == "__main__":
        updated["area_"] = updated.area
        sg.qtm(updated, "area_")

    for how in [
        "intersection",
        "difference",
        "symmetric_difference",
        "union",
        "identity",
    ]:
        overlayed = (
            sg.clean_geoms(p500)
            .explode(ignore_index=True)
            .overlay(sg.clean_geoms(p1000).explode(ignore_index=True), how=how)
        )
        overlayed2 = sg.clean_shapely_overlay(p500, p1000, how=how)

        if len(overlayed) != len(overlayed2):
            raise ValueError(how, len(overlayed), len(overlayed2))

        # area is slightly different, but same area with 3 digits is good enough
        for i in [1, 2, 3]:
            if round(sum(overlayed.area), i) != round(sum(overlayed2.area), i):
                raise ValueError(
                    how,
                    i,
                    round(sum(overlayed.area), i),
                    round(sum(overlayed2.area), i),
                )

            sg.overlay(p500, p1000, how=how, geom_type="polygon")
            sg.clean_shapely_overlay(
                p500.sample(1),
                p1000.sample(1),
                how=how,
                geom_type=("polygon", "polygon"),
            )


def main():
    from oslo import points_oslo

    test_overlay(points_oslo())


if __name__ == "__main__":
    main()

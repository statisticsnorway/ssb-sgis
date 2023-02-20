import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.wkt import loads


src = str(Path(__file__).parent).strip("tests") + "src"

sys.path.append(src)

import gis_utils as gs


def test_overlay():
    p = gpd.read_parquet(Path(__file__).parent / "testdata" / "random_points.parquet")
    p = p.iloc[:100]

    p500 = gs.buff(p, 500)
    p1000 = gs.buff(p, 1000)

    for how in [
        "intersection",
        "difference",
        "symmetric_difference",
        "union",
        "identity",
    ]:
        overlayed = (
            gs.clean_geoms(p500)
            .explode(ignore_index=True)
            .overlay(gs.clean_geoms(p1000).explode(ignore_index=True), how=how)
        )
        overlayed2 = gs.clean_shapely_overlay(p500, p1000, how=how)

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


def main():
    test_overlay()


if __name__ == "__main__":
    main()

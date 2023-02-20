import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.wkt import loads


src = str(Path(__file__).parent).strip("tests") + "src"

sys.path.append(src)

import gis_utils as gs


def test_distances():
    p = gpd.read_parquet(Path(__file__).parent / "testdata" / "random_points.parquet")
    p["idx"] = p.index
    p["idx2"] = p.index

    df = gs.get_k_nearest_neighbors(
        gdf=p,
        neighbors=p,
        k=50,
        id_cols=("idx", "idx2"),
        max_dist=None,
    )

    assert len(df) == len(p) * 50 - len(p)

    df = gs.get_k_nearest_neighbors(
        gdf=p,
        neighbors=p,
        k=50,
        id_cols="idx",
        min_dist=0,
        max_dist=None,
    )
    assert len(df) == len(p) * 50

    df = gs.get_k_nearest_neighbors(
        gdf=p,
        neighbors=p,
        k=10_000,
        id_cols="idx",
        max_dist=None,
    )

    assert len(df) == len(p) * len(p) - len(p)

    try:
        df = gs.get_k_nearest_neighbors(
            gdf=p,
            neighbors=p,
            k=10_000,
            id_cols="idx",
            max_dist=None,
            strict=True,
        )
    except ValueError:
        pass

    df = gs.get_k_nearest_neighbors(
        gdf=p,
        neighbors=p,
        k=100,
        id_cols="idx",
        max_dist=250,
    )

    assert max(df.dist) <= 250


def main():
    test_distances()


if __name__ == "__main__":
    main()

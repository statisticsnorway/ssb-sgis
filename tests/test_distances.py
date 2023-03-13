import sys
from pathlib import Path

import geopandas as gpd


src = str(Path(__file__).parent).strip("tests") + "src"

sys.path.insert(0, src)

import sgis as sg


def test_distances(points_oslo, roads_oslo):
    p = points_oslo

    p["idx"] = p.index
    p["idx2"] = p.index

    df = sg.get_k_nearest_neighbors(
        gdf=p,
        neighbors=p,
        k=50,
        id_cols=("idx", "idx2"),
        max_dist=None,
    )

    assert len(df) == len(p) * 50 - len(p)

    df = sg.get_k_nearest_neighbors(
        gdf=p,
        neighbors=p,
        k=50,
        id_cols="idx",
        min_dist=-1,
        max_dist=None,
    )

    assert len(df) == len(p) * 50

    df = sg.get_k_nearest_neighbors(
        gdf=p,
        neighbors=p,
        k=10_000,
        id_cols="idx",
        max_dist=None,
    )

    assert len(df) == len(p) * len(p) - len(p)

    try:
        df = sg.get_k_nearest_neighbors(
            gdf=p,
            neighbors=p,
            k=10_000,
            id_cols="idx",
            max_dist=None,
            strict=True,
        )
    except ValueError:
        pass

    df = sg.get_k_nearest_neighbors(
        gdf=p,
        neighbors=p,
        k=100,
        id_cols="idx",
        max_dist=250,
    )

    assert max(df.dist) <= 250


def main():
    from oslo import points_oslo, roads_oslo

    test_distances(points_oslo(), roads_oslo())


if __name__ == "__main__":
    main()

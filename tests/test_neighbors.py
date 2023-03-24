import sys
from pathlib import Path

import numpy as np


src = str(Path(__file__).parent).strip("tests") + "src"

sys.path.insert(0, src)

import sgis as sg


def test_neighbors(points_oslo):
    p = points_oslo

    p["idx"] = p.index
    p["idx2"] = np.random.randint(10_000, 20_000, size=len(p))

    neighbor_index = sg.get_neighbor_indices(
        p.iloc[[0]],
        neighbors=p,
        max_dist=2000,
    )
    assert isinstance(neighbor_index, list)
    assert len(neighbor_index) == 101, len(neighbor_index)

    neighbor_ids = sg.get_neighbor_ids(
        p.iloc[[0]],
        neighbors=p,
        id_col="idx2",
        max_dist=2000,
    )
    assert isinstance(neighbor_ids, list)
    assert len(neighbor_ids), len(neighbor_ids)

    df = sg.get_all_distances(
        gdf=p,
        neighbors=p,
        id_cols=("idx", "idx2"),
    )

    print(df)

    assert len(df) == len(p) * len(p)

    df = sg.get_k_nearest_neighbors(
        gdf=p,
        neighbors=p,
        k=50,
        id_cols="idx",
    )
    print(df)

    assert len(df) == len(p) * 50

    df = sg.get_k_nearest_neighbors(
        gdf=p,
        neighbors=p,
        k=10_000,
        id_cols="idx",
    )
    print(df)

    assert len(df) == len(p) * len(p)

    try:
        df = sg.get_k_nearest_neighbors(
            gdf=p,
            neighbors=p,
            k=10_000,
            id_cols="idx",
            strict=True,
        )
        failed = False
    except ValueError:
        failed = True
    assert failed


def main():
    from oslo import points_oslo

    test_neighbors(points_oslo())


if __name__ == "__main__":
    main()

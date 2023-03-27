import sys
from pathlib import Path

import numpy as np


src = str(Path(__file__).parent).strip("tests") + "src"

sys.path.insert(0, src)

import sgis as sg


def test_k_neighbors(points_oslo):
    p = points_oslo

    p["idx"] = p.index
    p["idx2"] = np.random.randint(10_000, 20_000, size=len(p))

    df = sg.get_k_nearest_neighbors(
        gdf=p,
        neighbors=p,
        k=50,
    )
    print(df)
    assert len(df) == len(p) * 50
    assert max(df.index) == 999

    p2 = p.join(df)
    assert len(p2) == len(p) * 50
    p2["k"] = p2.groupby(level=0)["distance"].transform("rank")
    assert max(p2.k) == 50
    assert min(p2.k) == 1
    print(p2)

    p["mean_distance"] = df.groupby(level=0)["distance"].mean()
    p["min_distance"] = df.groupby(level=0)["distance"].min()
    print(p)

    df = sg.get_k_nearest_neighbors(
        gdf=p,
        neighbors=p.set_index("idx2"),
        k=50,
    )
    print(df)
    assert len(df) == len(p) * 50
    assert max(df.index) == 999

    df = sg.get_k_nearest_neighbors(
        gdf=p,
        neighbors=p.set_index("idx2"),
        k=50,
    )
    print(df)
    assert len(df) == len(p) * 50
    assert max(df.index) == 999

    df = sg.get_k_nearest_neighbors(
        gdf=p.assign(a="a").set_index("a"),
        neighbors=p.set_index("idx2"),
        k=50,
    )
    assert all(df.index == "a")
    assert len(df) == len(p) * 50

    # should preserve index name
    p = p.rename_axis("point_idx", axis=0)
    df = sg.get_k_nearest_neighbors(p, p, k=10)
    assert df.index.name == "point_idx"

    # too many points should be ok when strict is False
    df = sg.get_k_nearest_neighbors(
        gdf=p,
        neighbors=p,
        k=10_000,
    )

    try:
        df = sg.get_k_nearest_neighbors(
            gdf=p,
            neighbors=p,
            k=10_000,
            strict=True,
        )
        failed = False
    except ValueError:
        failed = True
    assert failed

    df = sg.get_all_distances(
        gdf=p,
        neighbors=p.set_index("idx2"),
    )

    print(df)

    assert len(df) == len(p) * len(p)


def test_neighbor_indices_ids(points_oslo):
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


def main():
    from oslo import points_oslo

    points_oslo = points_oslo()

    test_k_neighbors(points_oslo)
    test_neighbor_indices_ids(points_oslo)


if __name__ == "__main__":
    main()

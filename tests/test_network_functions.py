# %%
import sys
from pathlib import Path

import geopandas as gpd


src = str(Path(__file__).parent).strip("tests") + "src"


sys.path.insert(0, src)
import gis_utils as gs


def test_network_methods():
    p = gpd.read_parquet(Path(__file__).parent / "testdata" / "random_points.parquet")
    p = p.iloc[[0]]

    r = gpd.read_parquet(Path(__file__).parent / "testdata" / "roads_oslo_2022.parquet")
    r = gs.clean_clip(r, p.buffer(1000)).explode(ignore_index=True)

    r2 = gs.get_largest_component(r)
    gs.qtm(r2, column="connected", scheme="equalinterval", title="connected")

    r2 = gs.close_network_holes(r2, 1.1).pipe(gs.cut_lines, 250)

    if (l := max(r2.length)) > 250 + 1:
        raise ValueError(f"cut_lines did not cut lines. max line length: {l}")

    r2 = r2.loc[r2.connected == 1]
    gs.qtm(r2, column="connected", title="after removing isolated")

    holes_closed = gs.close_network_holes(r, 10.1)
    print(holes_closed.hole.value_counts())
    gs.qtm(holes_closed, column="hole", title="holes")

    holes_closed = gs.close_network_holes(r, 10.1, deadends_only=True)
    print(holes_closed.hole.value_counts())
    gs.qtm(holes_closed, column="hole", title="holes, deadends_only")


def main():
    test_network_methods()


if __name__ == "__main__":
    main()

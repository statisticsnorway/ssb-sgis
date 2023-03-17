# %%
import sys
from pathlib import Path
import timeit
import cProfile

import pandas as pd
import numpy as np

src = str(Path(__file__).parent).strip("tests") + "src"

sys.path.insert(0, src)

import sgis as sg


def not_test_pandas():
    n = 1_000_000

    df = sg.random_points(n)
    df["idx"] = df.index

    df["rev"] = list(reversed(list(range(len(df)))))

    df2 = df.drop("geometry", axis=1)

    df2["num"] = np.random.random(len(df2))

    df_multi = df.set_index(["idx", "rev"])
    df2_multi = df2.set_index(["idx", "rev"])

    def _map():
        res = df.index.map(df2.num)
        assert len(res) == n
        return res

    def _join():
        res = df.join(df2.num)
        assert len(res) == n
        return res

    def _merge():
        res = df.merge(df2, on="idx")
        assert len(res) == n
        return res

    def _concat():
        res = pd.concat([df, df2], axis=1)
        assert len(res) == n
        return res

    def _concat_multi():
        res = pd.concat([df_multi, df2_multi], axis=1)
        assert len(res) == n
        return res

    def _map_multi():
        res = df_multi.index.map(df2_multi.num)
        assert len(res) == n
        return res

    def _join_multi():
        res = df_multi.join(df2_multi.num)
        assert len(res) == n
        return res

    def _map_set_index1():
        res = df_multi.index.map(df2.set_index(["idx", "rev"]).num)
        assert len(res) == n
        return res

    def _join_set_index1():
        res = df_multi.join(df2.set_index(["idx", "rev"]).num)
        assert len(res) == n
        return res

    def _map_set_index2():
        res = df.set_index(["idx", "rev"]).index.map(df2.set_index(["idx", "rev"]).num)
        assert len(res) == n
        return res

    def _join_set_index2():
        res = df.set_index(["idx", "rev"]).join(df2.set_index(["idx", "rev"]).num)
        assert len(res) == n
        return res

    def _concat_set_index2():
        res = pd.concat(
            [df.set_index(["idx", "rev"]), df2.set_index(["idx", "rev"])], axis=1
        )
        assert len(res) == n
        return res

    print(
        _map(),
        _join(),
        _merge(),
        _concat(),
        _concat_multi(),
        _map_multi(),
        _join_multi(),
        _map_set_index1(),
        _join_set_index1(),
        _map_set_index2(),
        _join_set_index2(),
        _concat_set_index2(),
    )

    print("_map", timeit.timeit(lambda: _map(), number=20))
    print("_join", timeit.timeit(lambda: _join(), number=20))
    print("_merge", timeit.timeit(lambda: _merge(), number=20))
    print("_concat", timeit.timeit(lambda: _concat(), number=20))
    print("_concat_multi", timeit.timeit(lambda: _concat_multi(), number=20))
    print("_map_multi", timeit.timeit(lambda: _map_multi(), number=20))
    print("_join_multi", timeit.timeit(lambda: _join_multi(), number=20))
    print("_map_set_index1", timeit.timeit(lambda: _map_set_index1(), number=20))
    print("_join_set_index1", timeit.timeit(lambda: _join_set_index1(), number=20))
    print("_map_set_index2", timeit.timeit(lambda: _map_set_index2(), number=20))
    print("_join_set_index2", timeit.timeit(lambda: _join_set_index2(), number=20))
    print("_concat_set_index2", timeit.timeit(lambda: _concat_set_index2(), number=20))


def main():
    # not_test_pandas()
    cProfile.run("not_test_pandas()", sort="cumtime")


if __name__ == "__main__":
    main()

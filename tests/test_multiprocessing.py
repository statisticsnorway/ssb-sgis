import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest


src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


def not_test_pool():
    def func(x, **kwargs):
        return x * 2

    p = sg.MultiProcessingPool()
    p.append_func(func, x=1, y=2)
    assert len(p) == 1

    iterable = [1, 2, 3, 4, 5, 6]

    p.chunkwise(func, iterable, n=3)

    res = p.execute_singleprocess()
    print(res)
    assert len(res) == len(p)

    assert res[0] == 2, res[0]
    assert np.all(np.equal(res[1], np.array([2, 4]))), res[1]
    assert np.all(np.equal(res[2], np.array([6, 8]))), res[2]
    assert np.all(np.equal(res[3], np.array([10, 12]))), res[3]


def func(x, **kwargs):
    return x * 2


def not_test_map():
    iterable = [1, 2, 3, 4, 5, 6]

    res = sg.MultiProcessingMapper().map(func, iterable)
    print(res)


if __name__ == "__main__":
    not_test_pool()
    not_test_map()

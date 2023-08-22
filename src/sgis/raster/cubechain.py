import functools


sssossss


class CubeChain:
    def __init__(self):
        self.funcs = []
        self.iterables = []
        self.names = []
        self.types = []

        self.write_in_chain = False

    def append_array_func(self, func, **kwargs):
        self._append_func(func, **kwargs)
        self.iterables.append(None)
        self.types.append("array")

    def append_raster_func(self, func, **kwargs):
        self._append_func(func, **kwargs)
        self.iterables.append(None)
        self.types.append("raster")

    def append_cube_func(self, func, **kwargs):
        self._append_func(func, **kwargs)
        self.iterables.append(None)
        self.types.append("cube")

    def append_cube_iter(self, func, iterable, **kwargs):
        self._append_func(func, **kwargs)
        self.iterables.append(iterable)
        self.types.append("cube_iter")

    def append_raster_iter(self, func, iterable, **kwargs):
        self._append_func(func, **kwargs)
        self.iterables.append(iterable)
        self.types.append("raster_iter")

    def _append_func(self, func, **kwargs):
        assert callable(func)
        func_name = self.get_func_name(func)
        if "write" in func_name:
            if self.write_in_chain:
                raise ValueError("Cannot keep chain going after writing files.")
            else:
                self.write_in_chain = True

        if "to_gdf" in func_name:
            if self.to_gdf_in_chain:
                raise ValueError("Cannot keep chain going after to_gdf.")
            else:
                self.to_gdf_in_chain = True

        func = functools.partial(func, **kwargs)
        self.funcs.append(func)

    def get_func_name(self, func):
        try:
            return func.__name__
        except AttributeError:
            return str(func)

    def __bool__(self):
        if len(self.funcs):
            return True
        return False

    def __len__(self):
        return len(self.funcs)

    def __iter__(self):
        return iter(
            [
                (func, typ, iterable)
                for func, typ, iterable in zip(
                    self.funcs,
                    self.types,
                    self.iterables,
                    strict=True,
                )
            ]
        )


if __name__ == "__main__":

    def x():
        pass

    def xs():
        pass

    chain = CubeChain()
    if chain:
        print("hi")
    chain.append_array_func(x, {})
    chain.append_raster_func(x, {})

    if chain:
        print("hi hi")

    for i in chain:
        print(i)

    print(getattr(chain, "__len__")())


import multiprocessing

import numpy as np


arr = np.arange(0, 100000)
arr = np.arange(0, 100000)
arr = np.arange(0, 100000)
arr = np.arange(0, 100000)
arr = np.arange(0, 100000)
arr = np.arange(0, 100000)
arr = np.arange(0, 100000)
arr = np.arange(0, 100000)
arr = np.arange(0, 100000)
arr = np.arange(0, 100000)
arr = np.arange(0, 100000)
arr = np.arange(0, 100000)


def x1():
    return arr * arr


def x2():
    return arr * arr


def x3():
    return arr * arr


funcs = [x1, x2, x3]


if __name__ == "__main__":
    n = 10

    import time

    t = time.perf_counter()

    for _ in range(n):
        with multiprocessing.get_context("spawn").Pool(6) as pool:
            results = pool.apply_async(x1).get()
            results = pool.apply_async(x2).get()
            results = pool.apply_async(x3).get()

    print("t1", time.perf_counter() - t)
    t = time.perf_counter()

    for _ in range(n):
        with multiprocessing.get_context("spawn").Pool(6) as pool:
            results = pool.apply_async(x1).get()
        with multiprocessing.get_context("spawn").Pool(6) as pool:
            results = pool.apply_async(x2).get()
        with multiprocessing.get_context("spawn").Pool(6) as pool:
            results = pool.apply_async(x3).get()

    print("t2", time.perf_counter() - t)

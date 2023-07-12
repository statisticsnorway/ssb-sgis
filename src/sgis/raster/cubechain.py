class CubeChain:
    def __init__(self):
        self.funcs = []
        self.names = []
        self.types = []
        self.kwargs = []

    def append_method(self, method, **kwargs):
        assert isinstance(method, str)
        self.funcs.append(method)
        self.names.append(method)
        self.types.append("method")
        self.kwargs.append(kwargs)

    def append_array_func(self, func, **kwargs):
        assert callable(func)
        self.funcs.append(func)
        self.names.append(func.__name__)
        self.types.append("array")
        self.kwargs.append(kwargs)

    def append_raster_func(self, func, **kwargs):
        assert callable(func)
        self.funcs.append(func)
        self.names.append(func.__name__)
        self.types.append("raster")
        self.kwargs.append(kwargs)

    def __bool__(self):
        if len(self.funcs):
            return True
        return False

    def __len__(self):
        return len(self.funcs)

    def __iter__(self):
        return iter(
            [
                (func, name, typ, kwargs)
                for func, name, typ, kwargs in zip(
                    self.funcs, self.names, self.types, self.kwargs, strict=True
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
    chain.append_method("x", {})

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

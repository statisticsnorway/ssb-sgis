import functools
import inspect
import itertools
import multiprocessing
from collections.abc import Callable, Collection, Iterable
from pathlib import Path
from typing import Any


try:
    import dapla as dp
except ImportError:
    pass

import joblib
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame
from shapely.geometry import MultiPolygon, Polygon

from ..geopandas_tools.general import clean_clip, clean_geoms
from ..geopandas_tools.neighbors import get_neighbor_indices
from ..geopandas_tools.overlay import clean_overlay
from ..helpers import LocalFunctionError, dict_zip, dict_zip_union, in_jupyter


try:
    from ..io.dapla_functions import exists, read_geopandas, write_geopandas

    # from ..io.write_municipality_data import write_municipality_data
except ImportError:
    pass


try:
    from dapla import read_pandas, write_pandas
except ImportError:
    pass


def turn_args_into_kwargs(func: Callable, args: tuple, index_start: int):
    if not isinstance(args, tuple):
        raise TypeError("args should be a tuple (it should not be unpacked with *)")
    argnames = inspect.getfullargspec(func).args[index_start:]
    return {name: value for value, name in zip(args, argnames, strict=False)}


class Parallel:
    """Run functions in parallell.

    The main method is 'map', which runs a single function for
    each item of an iterable. If the items of the iterable also are iterables,
    starmap can be used.

    The class also provides functions for reading and writing files in parallell
    in dapla.

    Note that nothing gets printed during execution if running in a notebook.
    Tip for debugging: set processes=1 to run without parallelization.

    Note that when using the default backend 'multiprocessing', all code except for
    imports and functions should be guarded by 'if __name__ == "__main__"' to not cause
    an eternal loop. This is not the case if setting backend to 'loky'. See joblib's
    documentation: https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation

    Args:
        processes: Number of parallel processes. Set to 1 to run without
            parallelization.
        backend: Defaults to "multiprocessing". Other options are 'loky' and 'threading',
            offered through joblib's Parallel class.
        context: Start method for the processes. Defaults to 'spawn'
            to avoid frozen processes.
        maxtasksperchild: Number of tasks a worker process can complete before
            it will exit and be replaced with a fresh worker process, to enable
            unused resources to be freed. Defaults to 10 to
        **kwargs: Keyword arguments to be passed to either
            multiprocessing.Pool or joblib.Parallel, depending
            on the backend. Not to be confused with the kwargs passed to functions in
            the map and starmap methods.
    """

    def __init__(
        self,
        processes: int,
        backend: str = "multiprocessing",
        context: str = "spawn",
        maxtasksperchild: int = 10,
        **kwargs,
    ):
        self.processes = int(processes)
        self.maxtasksperchild = maxtasksperchild
        self.backend = backend
        self.context = context
        self.kwargs = kwargs
        self.funcs: list[functools.partial] = []
        self.results: list[Any] = []

    def map(
        self,
        func: Callable,
        iterable: Collection,
        args: tuple | None = None,
        kwargs: dict | None = None,
    ) -> list[Any]:
        """Run functions in parallel with items of an iterable as 0th arguemnt.

        Args:
            func: Function to be run.
            iterable: An iterable where each item will be passed to func as
                0th positional argument.
            Args: Positional arguments passed to 'func' starting from the 1st argument.
                The 0th argument will be reserved for the values of 'iterable'.
            kwargs: Keyword arguments passed to 'func'. Must be passed as a dict,
                not unpacked into separate keyword arguments.

        Returns:
            A list of the return values of the function, one for each item in
            'iterable'.

        Examples
        --------
        Multiply each list element by 2.

        >>> iterable = [1, 2, 3]
        >>> def x2(x):
        ...     return x * 2
        >>> p = sg.Parallel(4, backend="loky")
        >>> results = p.map(x2, iterable)
        >>> results
        [2, 4, 6]

        With args and kwargs.

        >>> iterable = [1, 2, 3]
        >>> def x2(x, plus, minus):
        ...     return x * 2 + plus - minus
        >>> p = sg.Parallel(4, backend="loky")
        ...
        >>> # these three are the same
        >>> results1 = p.map(x2, iterable, args=(2, 1))
        >>> results2 = p.map(x2, iterable, kwargs=dict(plus=2, minus=1))
        >>> results3 = p.map(x2, iterable, args=(2,), kwargs=dict(minus=1))
        >>> assert results1 == results2 == results3
        ...
        >>> results1
        [3, 5, 7]

        If in Jupyter the function should be defined in another module.
        And if using the multiprocessing backend, the code should be
        guarded by if __name__ == "__main__".

        >>> from .file import x2
        >>> if __name__ == "__main__":
        ...     p = sg.Parallel(4, backend="loky")
        ...     results = p.map(x2, iterable)
        ...     print(results)
        [2, 4, 6]
        """

        if args:
            # start at index 1, meaning the 0th argument (the iterable) is still available
            args_as_kwargs = turn_args_into_kwargs(func, args, index_start=1)
        else:
            args_as_kwargs = {}

        self.validate_execution(func)

        kwargs = self._validate_kwargs(kwargs) | args_as_kwargs

        func_with_kwargs = functools.partial(func, **kwargs)

        if self.processes == 1:
            return list(map(func_with_kwargs, iterable))

        iterable = list(iterable)

        # don't use unnecessary processes
        processes = min(self.processes, len(iterable))

        if not processes:
            return []

        if self.backend == "multiprocessing":
            with multiprocessing.get_context(self.context).Pool(
                processes, maxtasksperchild=self.maxtasksperchild, **self.kwargs
            ) as pool:
                return pool.map(func_with_kwargs, iterable)

        with joblib.Parallel(
            n_jobs=processes, backend=self.backend, **self.kwargs
        ) as parallel:
            return parallel(joblib.delayed(func)(item, **kwargs) for item in iterable)

    def starmap(
        self,
        func: Callable,
        iterable: Collection[Iterable[Any]],
        args: tuple | None = None,
        kwargs: dict | None = None,
    ) -> list[Any]:
        """Run functions in parallel where items of the iterable are unpacked.

        This requires the items of the iterable to be iterables as well. See
        https://docs.python.org/3/library/itertools.html#itertools.starmap

        Args:
            func: Function to be run.
            iterable: An iterable of iterables, where each item will be
                unpacked as positional argument to the function.
            Args: Positional arguments passed to 'func' starting at argument position
                n + 1, where n is the length of the iterables inside the iterable.
            kwargs: Keyword arguments passed to 'func'. Must be passed as a dict,
                not unpacked into separate keyword arguments.

        Returns:
            A list of the return values of the function, one for each item in
            'iterable'.

        Examples
        --------
        Multiply each list element by 2.

        >>> iterable = [(1, 2), (2, 3), (3, 4)]
        >>> def add(x, y):
        ...     return x + y
        >>> p = sg.Parallel(3, backend="loky")
        >>> results = p.starmap(add, iterable)
        >>> results
        [3, 5, 7]

        With args and kwargs. Since the iterables inside 'iterable' are of length 2,
        'args' will start at argument number three, e.i. 'c'.

        >>> iterable = [(1, 2), (2, 3), (3, 4)]
        >>> def add(a, b, c, *, d):
        ...     return a + b + c + d
        >>> p = sg.Parallel(3, backend="loky")
        >>> results = p.starmap(add, iterable, args=(1,), kwargs={"d": 0.1})
        >>> results
        [4.1, 6.1, 8.1]

        If in Jupyter the function should be defined in another module.
        And if using the multiprocessing backend, the code should be
        guarded by if __name__ == "__main__".

        >>> from .file import x2
        >>> if __name__ == "__main__":
        ...     p = sg.Parallel(4, backend="loky")
        ...     results = p.starmap(add, iterable)
        ...     print(results)
        [3, 5, 7]

        """
        if args:
            # starting the count at the length of the iterables inside the iterables
            iterable = list(iterable)
            args_as_kwargs = turn_args_into_kwargs(
                func, args, index_start=len(iterable[0])
            )
        else:
            args_as_kwargs = {}

        self.validate_execution(func)

        kwargs = self._validate_kwargs(kwargs) | args_as_kwargs

        func_with_kwargs = functools.partial(func, **kwargs)

        if self.processes == 1:
            return list(itertools.starmap(func_with_kwargs, iterable))

        iterable = list(iterable)

        # don't use unnecessary processes
        processes = min(self.processes, len(iterable))

        if not processes:
            return []

        if self.backend == "multiprocessing":
            with multiprocessing.get_context(self.context).Pool(
                processes, maxtasksperchild=self.maxtasksperchild, **self.kwargs
            ) as pool:
                return pool.starmap(func_with_kwargs, iterable)

        with joblib.Parallel(
            n_jobs=processes, backend=self.backend, **self.kwargs
        ) as parallel:
            return parallel(joblib.delayed(func)(*item, **kwargs) for item in iterable)

    def read_pandas(
        self,
        files: list[str],
        concat: bool = True,
        ignore_index: bool = True,
        strict: bool = True,
        **kwargs,
    ) -> DataFrame | list[DataFrame]:
        """Read tabular files from a list in parallel.

        Args:
            files: List of file paths.
            concat: Whether to concat the results to a DataFrame.
            ignore_index: Defaults to True.
            strict: If True (default), all files must exist.
            **kwargs: Keyword arguments passed to dapla.read_pandas.

        Returns:
            A DataFrame, or a list of DataFrames if concat is False.
        """
        if not strict:
            files = [file for file in files if exists(file)]

        res = self.map(dp.read_pandas, files, kwargs=kwargs)

        return pd.concat(res, ignore_index=ignore_index) if concat else res

    def read_geopandas(
        self,
        files: list[str],
        concat: bool = True,
        ignore_index: bool = True,
        strict: bool = True,
        **kwargs,
    ) -> GeoDataFrame | list[GeoDataFrame]:
        """Read geospatial files from a list in parallel.

        Args:
            files: List of file paths.
            concat: Whether to concat the results to a GeoDataFrame.
            ignore_index: Defaults to True.
            strict: If True (default), all files must exist.
            **kwargs: Keyword arguments passed to sgis.read_geopandas.

        Returns:
            A GeoDataFrame, or a list of GeoDataFrames if concat is False.
        """
        if not strict:
            files = [file for file in files if exists(file)]
        res = self.map(read_geopandas, files, kwargs=kwargs)

        return pd.concat(res, ignore_index=ignore_index) if concat else res

    def write_municipality_data(
        self,
        in_data: dict[str, str | GeoDataFrame],
        out_data: str | dict[str, str],
        municipalities: GeoDataFrame,
        with_neighbors: bool = False,
        funcdict: dict[str, Callable] | None = None,
        file_type: str = "parquet",
        muni_number_col: str = "KOMMUNENR",
        strict: bool = False,
        write_empty: bool = False,
        clip: bool = True,
        max_rows_per_chunk: int = 150_000,
        processes_in_clip: int = 1,
    ):
        """Split multiple datasets into municipalities and write as separate files.

        The files will be named as the municipality number.

        Args:
            in_data: Dictionary with dataset names as keys and file paths or
                (Geo)DataFrames as values.
            out_data: Either a single folder path or a dictionary with same keys as
                'in_data' and folder paths as values. If a single folder is given,
                the 'in_data' keys will be used as subfolders.
            year: Year of the municipality numbers.
            funcdict: Dictionary with the keys of 'in_data' and functions as values.
                The functions should take a GeoDataFrame as input and return a
                GeoDataFrame. The function will be excecuted before the right after
                the data is read.
            file_type: Defaults to parquet.
            muni_number_col: String column name with municipality
                number/identifier. Defaults to KOMMUNENR.
            strict: If False (default), the dictionaries 'out_data' and 'funcdict' does
                not have to have the same length as 'in_data'.
            write_empty: If False (default), municipalities with no data will be skipped.
                If True, an empty parquet file will be written.
        """
        shared_kwds = {
            "municipalities": municipalities,
            "file_type": file_type,
            "muni_number_col": muni_number_col,
            "write_empty": write_empty,
            "with_neighbors": with_neighbors,
            "clip": clip,
            "max_rows_per_chunk": max_rows_per_chunk,
            "processes_in_clip": processes_in_clip,
        }

        if isinstance(out_data, (str, Path)):
            out_data = {name: Path(out_data) / name for name in in_data}

        if funcdict is None:
            funcdict = {}

        zip_func = dict_zip if strict else dict_zip_union

        for _, data, folder, postfunc in zip_func(in_data, out_data, funcdict):
            if data is None:
                continue

            kwds = shared_kwds | {
                "data": data,
                "func": postfunc,
                "out_folder": folder,
            }
            partial_func = functools.partial(write_municipality_data, **kwds)
            self.funcs.append(partial_func)

        return self._execute()

    def chunkwise(
        self,
        func: Callable,
        df: GeoDataFrame,
        args: tuple | None = None,
        kwargs: dict | None = None,
        max_rows_per_chunk: int = 150_000,
        n_chunks: int = None,
        concat: bool = False,
    ) -> GeoDataFrame:
        if len(df) < max_rows_per_chunk:
            return func(df, *args, **kwargs)

        if n_chunks is None:
            n_chunks = len(df) // max_rows_per_chunk

        chunks = np.array_split(np.arange(len(df)), n_chunks)

        df_chunked: list[GeoDataFrame] = [df.iloc[chunk] for chunk in chunks]

        out = self.map(
            func,
            df_chunked,
            args=args,
            kwargs=kwargs,
        )
        if concat:
            return pd.concat(out, ignore_index=True)
        else:
            return out

    def validate_execution(self, func):
        """Multiprocessing doesn't work with local variables in interactive interpreter.

        Raising Exception to avoid confusion.
        """
        if (
            func.__module__ == "__main__"
            and self.context == "spawn"
            and self.backend == "multiprocessing"
            and in_jupyter()
        ):
            raise LocalFunctionError(func)

    @staticmethod
    def _validate_kwargs(kwargs) -> dict:
        """Make sure kwargs is a dict (not ** unpacked or None)"""
        if kwargs is None:
            kwargs = {}
        elif not isinstance(kwargs, dict):
            raise TypeError("kwargs must be a dict")
        return kwargs

    def _execute(self) -> list[Any]:
        [self.validate_execution(func) for func in self.funcs]

        if self.processes == 1:
            return [func() for func in self.funcs]

        # don't use unnecessary processes
        if self.processes > len(self.funcs):
            processes = len(self.funcs)
        else:
            processes = self.processes

        if not processes:
            return []

        if self.backend != "multiprocessing":
            with joblib.Parallel(
                n_jobs=processes, backend=self.backend, **self.kwargs
            ) as parallel:
                return parallel(joblib.delayed(func)() for func in self.funcs)

        with multiprocessing.get_context(self.context).Pool(
            processes, **self.kwargs
        ) as pool:
            results = [pool.apply_async(func) for func in self.funcs]
            return [result.get() for result in results]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(processes={self.processes}, "
            f"backend='{self.backend}', context='{self.context}')"
        )


def write_municipality_data(
    data: str | GeoDataFrame | DataFrame,
    out_folder: str,
    municipalities: GeoDataFrame,
    with_neighbors: bool = False,
    muni_number_col: str = "KOMMUNENR",
    file_type: str = "parquet",
    func: Callable | None = None,
    write_empty: bool = False,
    clip: bool = True,
    max_rows_per_chunk: int = 150_000,
    processes_in_clip: int = 1,
) -> None:
    write_func = (
        _write_neighbor_municipality_data
        if with_neighbors
        else _write_municipality_data
    )

    return write_func(
        data=data,
        out_folder=out_folder,
        municipalities=municipalities,
        muni_number_col=muni_number_col,
        file_type=file_type,
        func=func,
        write_empty=write_empty,
        clip=clip,
        max_rows_per_chunk=max_rows_per_chunk,
        processes_in_clip=processes_in_clip,
    )


def _validate_data(data: str | list[str]) -> str:
    if isinstance(data, (str, Path)):
        return data
    if hasattr(data, "__iter__") and len(data) == 1:
        return data[0]
    elif not isinstance(data, GeoDataFrame):
        raise TypeError("'data' Must be a file path or a GeoDataFrame. Got", type(data))


def _get_out_path(out_folder, muni, file_type):
    return str(Path(out_folder) / f"{muni}.{file_type.strip('.')}")


def _write_municipality_data(
    data: str | GeoDataFrame | DataFrame,
    out_folder: str,
    municipalities: GeoDataFrame,
    muni_number_col: str = "KOMMUNENR",
    file_type: str = "parquet",
    func: Callable | None = None,
    write_empty: bool = False,
    clip: bool = True,
    max_rows_per_chunk: int = 150_000,
    processes_in_clip: int = 1,
) -> None:
    data = _validate_data(data)

    if isinstance(data, (str, Path)):
        try:
            gdf = read_geopandas(str(data))
        except ValueError as e:
            try:
                gdf = read_pandas(str(data))
            except ValueError:
                raise e.__class__(e, data)
    elif isinstance(data, DataFrame):
        gdf = data
    else:
        raise TypeError(type(data))

    if func is not None:
        gdf = func(gdf)

    gdf = _fix_missing_muni_numbers(
        gdf,
        municipalities,
        muni_number_col,
        clip,
        max_rows_per_chunk,
        processes_in_clip=processes_in_clip,
    )

    for muni in municipalities[muni_number_col]:
        print(muni)
        out = _get_out_path(out_folder, muni, file_type)

        gdf_muni = gdf.loc[gdf[muni_number_col] == muni]

        if not len(gdf_muni):
            if write_empty:
                gdf_muni = gdf_muni.drop(columns="geometry", errors="ignore")
                gdf_muni["geometry"] = None
                write_pandas(gdf_muni, out)
            continue

        write_geopandas(gdf_muni, out)


def _write_neighbor_municipality_data(
    data: str | GeoDataFrame | DataFrame,
    out_folder: str,
    municipalities: GeoDataFrame,
    muni_number_col: str = "KOMMUNENR",
    file_type: str = "parquet",
    func: Callable | None = None,
    write_empty: bool = False,
    clip: bool = True,
    max_rows_per_chunk: int = 150_000,
    processes_in_clip: int = 1,
) -> None:
    data = _validate_data(data)

    if isinstance(data, (str, Path)):
        gdf = read_geopandas(str(data))

    if func is not None:
        gdf = func(gdf)

    gdf = _fix_missing_muni_numbers(
        gdf,
        municipalities,
        muni_number_col,
        clip,
        max_rows_per_chunk,
        processes_in_clip,
    )

    if municipalities.index.name != muni_number_col:
        municipalities = municipalities.set_index(muni_number_col)

    neighbor_munis = get_neighbor_indices(
        municipalities, municipalities, max_distance=1
    )

    for muni in municipalities.index:
        out = _get_out_path(out_folder, muni, file_type)

        muni_and_neighbors = neighbor_munis.loc[[muni]]
        gdf_neighbor = gdf.loc[gdf[muni_number_col].isin(muni_and_neighbors)]

        if not len(gdf_neighbor):
            if write_empty:
                gdf_neighbor["geometry"] = gdf_neighbor["geometry"].astype(str)
                write_pandas(gdf_neighbor, out)
            continue

        write_geopandas(gdf_neighbor, out)


def _fix_missing_muni_numbers(
    gdf,
    municipalities,
    muni_number_col,
    clip,
    max_rows_per_chunk,
    processes_in_clip,
):
    if muni_number_col in gdf and gdf[muni_number_col].notna().all():
        if municipalities is None:
            return gdf
        if diffs := set(gdf[muni_number_col].values).difference(
            set(municipalities[muni_number_col].values)
        ):
            raise ValueError(f"Different municipality numbers: {diffs}")

        return gdf

    if municipalities is None:
        if muni_number_col not in gdf:
            raise ValueError(
                f"Cannot find column {muni_number_col}. "
                "Specify another column or a municipality GeoDataFrame to clip "
                "the geometries by."
            )
        assert gdf[muni_number_col].isna().any()
        raise ValueError(
            f"Column {muni_number_col} has missing values. Make sure gdf has "
            "correct municipality number info or specify a municipality "
            "GeoDataFrame to clip the geometries by."
        )

    municipalities = municipalities[[muni_number_col, "geometry"]].to_crs(gdf.crs)

    if muni_number_col in gdf and gdf[muni_number_col].isna().any():
        notna = gdf[gdf[muni_number_col].notna()]

        isna = gdf[gdf[muni_number_col].isna()].drop(muni_number_col, axis=1)

        if not clip:
            notna_anymore = isna.sjoin(municipalities).drop(columns="index_right")
        else:
            notna_anymore = intersect_by_municipalities(
                isna,
                municipalities,
                muni_number_col,
                max_rows_per_chunk,
                processes_in_clip,
            )

        return pd.concat([notna, notna_anymore], ignore_index=True)

    if not clip:
        return gdf.sjoin(municipalities).drop(columns="index_right")
    else:
        return intersect_by_municipalities(
            gdf,
            municipalities,
            muni_number_col,
            max_rows_per_chunk,
            processes_in_clip,
        )


def intersect_by_municipalities(
    df: GeoDataFrame,
    municipalities: GeoDataFrame,
    muni_number_col: str,
    max_rows_per_chunk: int,
    processes_in_clip: int,
) -> GeoDataFrame:
    if len(df) < max_rows_per_chunk:
        return clean_overlay(df, municipalities)

    municipalities = municipalities.dissolve(by=muni_number_col, as_index=False)

    n_chunks = len(df) // max_rows_per_chunk
    chunks = np.array_split(np.arange(len(df)), n_chunks)

    x_mapper = dict(enumerate(df.centroid))
    sorted_xs = dict(reversed(sorted(x_mapper.items(), key=lambda item: item[1])))
    df = df.iloc[list(sorted_xs)]

    df_chunked: list[GeoDataFrame] = [df.iloc[chunk] for chunk in chunks]

    out = Parallel(processes_in_clip, backend="loky").map(
        _clean_intersection,
        df_chunked,
        args=(municipalities,),
    )
    return pd.concat(out, ignore_index=True)


def _clean_intersection(df1, df2):
    print(len(df1))
    return clean_overlay(df1, df2, how="intersection")


def chunkwise(
    func: Callable,
    df: GeoDataFrame,
    max_rows_per_chunk: int = 150_000,
    n_chunks: int = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    n_jobs: int = 1,
    backend: str = "loky",
) -> GeoDataFrame:
    if len(df) < max_rows_per_chunk:
        return func(df, *args, **kwargs)

    if n_chunks is None:
        n_chunks = len(df) // max_rows_per_chunk

    chunks = np.array_split(np.arange(len(df)), n_chunks)

    df_chunked: list[GeoDataFrame] = [df.iloc[chunk] for chunk in chunks]

    out = Parallel(n_jobs, backend=backend).map(
        func,
        df_chunked,
        args=args,
        kwargs=kwargs,
    )
    return pd.concat(out, ignore_index=True)

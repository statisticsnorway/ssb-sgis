import functools
import multiprocessing
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Sized

import numpy as np
from geopandas import GeoDataFrame

from ..helpers import dict_zip, dict_zip_intersection
from ..io.dapla import read_geopandas
from ..io.write_municipality_data import (
    write_municipality_data,
    write_neighbor_municipality_data,
)
from .base import MultiProcessingBase


class MultiProcessingPool(MultiProcessingBase):
    """Class for streamlining multiprocessing in Dapla.

    Functions can be added one by one as a single process to the pool with the
    append_func method. Or a function can be split into multiple processes with
    the method chunkwise.

    The functions are appended to a list, which is then run in parallel
    by running the execute function. To run all processes normally, the
    method execute_singleprocess can be used. Nice for debugging.

    Note that all code in a program using multiprocessing, except for imports,
    should be guarded by 'if __name__ == "__main__"'.

    Reading the same dataset inside functions run in parallel might return
    empty results with no errors raised.

    Args:
        dapla: Whether to read the file in or outside of Dapla.
        context: Start method for the processes. Defaults to 'spawn',
            which avoids freezing.
    """

    def __init__(self, context: str = "spawn", dapla: bool = True):
        self.dapla = dapla
        self.context = context
        self.funcs: list[functools.partial] = []
        self.results: list[Any] = []
        self._source: list[str] = []

    def execute(self) -> list[Any]:
        """Executes all processes in the pool at the same time.

        Returns:
            A list of results from the function calls same length
            as number of processes.
        """
        n = len(self.funcs)

        if not n:
            raise ValueError

        [self.validate_execution(func) for func in self.funcs]

        if n == 1:
            source = self._source[0]
            mes = (
                "Number of processes to be parallelized is 1. "
                "This will be equivelent to running it normally."
            )

            if source == "append":
                mes += (
                    "\nDid you mean to run the function on chunks of an "
                    "iterable with the chunkwise method?"
                )

            warnings.warn(mes)

        with multiprocessing.get_context(self.context).Pool(n) as pool:
            results = [pool.apply_async(func) for func in self.funcs]

            self.results = [result.get() for result in results]

        return self.results

    def execute_singleprocess(self) -> list[Any]:
        """Execute all functions one by one.

        Returns:
            A list of results from the function calls same length
            as number of processes.
        """
        n = len(self.funcs)

        if not n:
            raise ValueError

        self.results = [func() for func in self.funcs]

        return self.results

    def append_func(self, func: Callable, *args, **kwargs):
        """Appends a partial function with args and kwargs to the pool.

        Args:
            func: A function.
            *args: Positional arguments passed to the function.
            **kwargs: Keyword argumens passed to the function.

        Examples
        --------

        >>> def x2(num):
        ...     return num * 2
        >>> if __name__ == "__main__":
        ...     p = MultiProcessingPool()
        ...     p.append(x2, num=1)
        ...     p.append(x2, num=2)
        ...     p.append(x2, num=3)
        ...     print(p.execute())
        [2, 4, 6]

        Equivelent to doing:

        >>> partial_func = functools.partial(x2, num=1)
        >>> p.results.append(partial_func)
        """
        self.validate_execution(func)
        partial_func = functools.partial(func, *args, **kwargs)
        self.funcs.append(partial_func)
        self._source.append("append")
        return self

    def chunkwise(
        self,
        func: Callable,
        iterable: Iterable,
        n: int,
        chunk_kwarg_name: str | None = None,
        **kwargs,
    ):
        """Splits an interable in n chunks and appends n processes to the pool.

        Args:
            func: Function to be run chunkwise.
            iterable: Iterable to be divided into n roughly equal length chunks.
                The chunk will be used as first argument in the function call,
                unless chunk_kwarg_name is specified.
            n: Number of chunks to divide the iterable in.
            chunk_kwarg_name: Optional keyword argument that the chunks should be
                assigned to. Defaults to None, meaning the chunk will be used as
                the first positional argument of the function.
            **kwargs: Additional keyword arguments passed to the function.

        Examples
        --------
        >>> def x2(num):
        ...     return num * 2
        >>> l = [1, 2, 3]
        >>> if __name__ == "__main__":
        ...     p = MultiProcessingPool()
        ...     p.chunkwise(x2, l, n=3)
        ...     print(p.execute())
        [2, 4, 6]

        """
        self.validate_execution(func)

        if isinstance(iterable, (str, bytes)) or not hasattr(iterable, "__iter__"):
            raise TypeError

        if not isinstance(iterable, Sized):
            iterable = list(iterable)

        n = n if n <= len(iterable) else len(iterable)

        try:
            splitted = list(np.array_split(iterable, n))
        except Exception:

            def split(a, n):
                k, m = divmod(len(a), n)
                return [
                    a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
                ]

            splitted = split(iterable, n)

        if not hasattr(self, "chunks"):
            self.chunks = []

        for chunk in splitted:
            if chunk_kwarg_name:
                partial_func = functools.partial(
                    func, **{chunk_kwarg_name: chunk}, **kwargs
                )
            else:
                partial_func = functools.partial(func, chunk, **kwargs)
            self.funcs.append(partial_func)
            self.chunks.append(chunk)
            self._source.append("chunkwise")

        return self

    def write_municipality_data(
        self,
        in_data: dict[str, str | GeoDataFrame],
        out_data: str | Path | dict[str, str],
        year: int,
        funcdict: dict[str, Callable] | None = None,
        file_type: str = "parquet",
        muni_number_col: str = "KOMMUNENR",
        strict: bool = False,
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
                GeoDataFrame.
            file_type: Defaults to parquet.
            muni_number_col: Column name that holds the municipality number. Defaults
                to KOMMUNENR.
            strict: If False (default), the dictionaries 'out_data' and 'funcdict' does
                not have to have the same length as 'in_data'.

        """
        KARTDATA_ANALYSE = "ssb-prod-kart-data-delt/kartdata_analyse/klargjorte-data"

        muni = read_geopandas(
            f"{KARTDATA_ANALYSE}/{year}/ABAS_kommune_flate_p{year}_v1.parquet"
        )

        if muni_number_col != "KOMMUNENR":
            muni = muni.rename(columns={"KOMMUNENR": muni_number_col}, errors="raise")

        kwds_for_all = {
            "municipalities": muni,
            "muni_number_col": muni_number_col,
            "file_type": file_type,
        }

        if isinstance(out_data, (str, Path)):
            out_data = {name: Path(out_data) / name for name in in_data.keys()}

        if funcdict is None:
            funcdict = {}

        zip_func = dict_zip if strict else dict_zip_intersection

        for _, data, folder, postfunc in zip_func(in_data, out_data, funcdict):
            kwds = kwds_for_all | {
                "data": data,
                "postread_func": postfunc,
                "out_folder": folder,
            }
            partial_func = functools.partial(write_municipality_data, **kwds)
            self.funcs.append(partial_func)
            self._source.append("write_municipality_data")

        return self

    def write_neighbor_municipality_data(
        self,
        in_data: dict[str, str | GeoDataFrame],
        out_data: str | Path | dict[str, str],
        year: int,
        funcdict: dict[str, Callable] | None = None,
        file_type: str = "parquet",
        muni_number_col: str = "KOMMUNENR",
        strict: bool = False,
    ):
        """Split datasets into municipalities+neighbors and write as separate files.

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
                GeoDataFrame.
            file_type: Defaults to parquet.
            muni_number_col: Column name that holds the municipality number. Defaults
                to KOMMUNENR.
            strict: If False (default), the dictionaries 'out_data' and 'funcdict' does
                not have to have the same length as 'in_data'.
        """
        KARTDATA_ANALYSE = "ssb-prod-kart-data-delt/kartdata_analyse/klargjorte-data"

        muni = read_geopandas(
            f"{KARTDATA_ANALYSE}/{year}/ABAS_kommune_flate_p{year}_v1.parquet"
        )

        if muni_number_col != "KOMMUNENR":
            muni = muni.rename(columns={"KOMMUNENR": muni_number_col}, errors="raise")

        kwds_for_all = {
            "municipalities": muni,
            "muni_number_col": muni_number_col,
            "file_type": file_type,
        }

        if isinstance(out_data, (str, Path)):
            out_data = {name: Path(out_data) / name for name in in_data.keys()}

        if funcdict is None:
            funcdict = {}

        zip_func = dict_zip if strict else dict_zip_intersection

        for _, data, folder, postfunc in zip_func(in_data, out_data, funcdict):
            kwds = kwds_for_all | {
                "data": data,
                "postread_func": postfunc,
                "out_folder": folder,
            }
            partial_func = functools.partial(write_neighbor_municipality_data, **kwds)
            self.funcs.append(partial_func)
            self._source.append("write_neighbor_municipality_data")

        return self

    def __len__(self):
        return len(self.funcs)

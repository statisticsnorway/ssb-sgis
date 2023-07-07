import functools
import multiprocessing
from typing import Any, Callable

import dapla as dp
import pandas as pd
from geopandas import GeoDataFrame
from pandas import DataFrame

from ..io.dapla import exists, read_geopandas
from .base import MultiProcessingBase


class MultiProcessingMapper(MultiProcessingBase):
    """Run single function in parallel based on iterable."""

    def __init__(self, context: str = "spawn", processes=None, dapla: bool = True):
        self.dapla = dapla
        self.context = context
        self.processes = processes

    def map(self, func: Callable, iterable: list, **kwargs) -> list[Any]:
        """Run functions in parallel with items of an iterable as first arguemnt.

        Args:
            func: Function to be run.
            iterable: An iterable where each item will be passed to func as
                first positional argument.
            **kwargs: Keyword arguments passed to 'func'.

        Returns:
            A list of the return values of the function, one for each item in
            'iterable'.
        """
        self.validate_execution(func)
        func_with_kwargs = functools.partial(func, **kwargs)

        with multiprocessing.get_context(self.context).Pool(self.processes) as pool:
            return pool.map(func_with_kwargs, iterable)

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

        res = self.map(func=dp.read_pandas, iterable=files, **kwargs)

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
        res = self.map(func=read_geopandas, iterable=files, **kwargs)

        return pd.concat(res, ignore_index=ignore_index) if concat else res

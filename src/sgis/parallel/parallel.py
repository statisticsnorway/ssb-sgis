import functools
import multiprocessing
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Sized

import dapla as dp
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from joblib import Parallel, delayed
from pandas import DataFrame

from ..helpers import dict_zip, dict_zip_union
from ..io.dapla import exists, read_geopandas
from ..io.write_municipality_data import write_municipality_data
from .base import ParallelBase


"""The background for these functions is a common operation before publishing to the "statbank" at statistics Norway.
All combinations (including total-groups), over all categorical codes, in a set of columns, need to have their numbers aggregated.
This has some similar functionality to "proc means" in SAS.
"""


from itertools import combinations

import pandas as pd


def all_combos(
    df: pd.DataFrame, columns: list, *agg_args, **agg_kwargs
) -> pd.DataFrame:
    df2 = df.copy()
    # Hack, for å beholde tomme grupper + observed i groupbyen
    for col in columns:
        df2[col] = df2[col].astype("category")
    # Lager alle kombinasjoner av grupperingskolonnene
    tab = pd.DataFrame()
    for x in range(5):
        groups = combinations(columns, x)
        for group in groups:
            print(x, list(group))
            if group:
                df2 = (
                    df2.groupby(list(group), dropna=False, observed=False)
                    .agg(*agg_args, **agg_kwargs)
                    .reset_index()
                )
            else:
                df2 = pd.DataFrame(df2.agg(*agg_args, **agg_kwargs).T).T
            for col in columns:
                if col not in df2.columns:
                    df2[col] = pd.NA
            tab = pd.concat([tab, df2])
    for col in columns:
        tab[col] = tab[col].astype("string")
    return tab


def fill_na_dict(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    for col, fill_val in mapping.items():
        df[col] = df[col].fillna(fill_val)
    return df


from itertools import combinations

import pandas as pd


def all_combos_agg(
    df: pd.DataFrame,
    groupcols: list,
    fillna_dict: dict = None,
    keep_empty: bool = False,
    grand_total: bool = False,
    *aggargs,
    **aggkwargs,
) -> pd.DataFrame:
    """Generate all aggregation levels for a set of columns in a dataframe

    Parameters:
    -----------
        df: dataframe to aggregate.
        groupcols: List of columns to group by.
        fillna_dict: Dict
        aggcols: List of columns to aggregate.
        aggfunc: List of aggregation function(s), like sum, mean, min, count.


    Returns:
    --------
        dataframe with all the group-by columns, all the aggregation columns combined
        with the aggregation functions, a column called aggregation_level which
        separates the different aggregation levels, and a column called aggregation_ways which
        counts the number of group columns used for the aggregation.

    Advice:
    -------
        When you want the frequency, create a column with the value 1 for each row first and then use that as the aggcol.
        Make sure that you don't have any values in the group columns that are the same as your fillna value.

    Known problems:
    ---------------
        You should not use dataframes with multi-index columns as they cause trouble.

    Examples:
    data = {
            'alder': [20, 60, 33, 33, 20],
            'kommune': ['0301', '3001', '0301', '5401', '0301'],
            'kjonn': ['1', '2', '1', '2', '2'],
            'inntekt': [1000000, 120000, 220000, 550000, 50000],
            'formue': [25000, 50000, 33000, 44000, 90000]
        }
    pers = pd.DataFrame(data)

    agg1 = aggregate_all(pers, groupcols=['kjonn'], aggcols=['inntekt'])
    display(agg1)

    agg2 = aggregate_all(pers, groupcols=['kommune', 'kjonn', 'alder'], aggcols=['inntekt', 'formue'])
    display(agg2)

    agg3 = aggregate_all(pers, groupcols=['kommune', 'kjonn', 'alder'], aggcols=['inntekt'], fillna='T', aggfunc=['mean', 'std'])
    display(agg3)

    pers['antall'] = 1
    groupcols = pers.columns[0:2].tolist()
    aggcols = pers.columns[3:5].tolist()
    aggcols.extend(['antall'])
    agg4 = aggregate_all(pers, groupcols=groupcols, aggcols=aggcols, fillna='T')
    display(agg4)
    """
    dataframe = df.copy()

    # Hack using categoricals to keep all unobserved groups
    if keep_empty:
        dataframe[groupcols] = dataframe[groupcols].astype("category")

    # Generate all possible combinations of group columns
    combos = []
    for r in range(len(groupcols) + 1, 0, -1):
        combos += list(combinations(groupcols, r))

    # Create an empty DataFrame to store the results
    all_levels = pd.DataFrame()

    # Calculate aggregates for each combination
    for i, comb in enumerate(combos):
        # Calculate statistics using groupby
        if keep_empty:
            # Hack using categoricals to keep all unobserved groups
            result = dataframe.groupby(list(comb), as_index=False, observed=False)
        else:
            result = dataframe.groupby(list(comb), as_index=False)

        result = result.agg(*aggargs, **aggkwargs).reset_index()

        # Add a column to differentiate the combinations
        result["level"] = len(combos) - i

        # Add a column with number of group columns used in the aggregation
        result["ways"] = int(len(comb))

        # Concatenate the current result with the combined results
        all_levels = pd.concat([all_levels, result], ignore_index=True)

    print(all_levels)

    # Calculate the grand total
    if grand_total:
        gt = pd.DataFrame(dataframe[aggcols].agg(*aggargs, **aggkwargs)).reset_index()
        gt["x", "z"] = 1
        gt = (
            gt.pivot(index=[("x", "z")], columns="index", values=aggcols)
            .reset_index()
            .drop(columns=[("x", "z")])
        )
        #        gt = pd.DataFrame(dataframe.agg(*aggargs, **aggkwargs)).reset_index()
        #        gt = (pd.DataFrame(gt
        #                           .agg(*aggargs, **aggkwargs)
        #                           .T)
        #              .T)
        gt["level"] = 0
        gt["ways"] = 0

        # Append the grand total row to the combined results and sort by levels and groupcols
        all_levels = pd.concat([all_levels, gt], ignore_index=True).sort_values(
            ["level"] + groupcols
        )
    else:
        all_levels = all_levels.sort_values(["level"] + groupcols)

    # Fill missing group columns with value
    if fillna_dict:
        all_levels = fill_na_dict(all_levels, fillna_dict)
    # all_levels[groupcols] = all_levels[groupcols].fillna(fillna)

    # change columns with multi-index to normal index
    # all_levels.columns = np.where(all_levels.columns.get_level_values(1) == '',
    #                               all_levels.columns.get_level_values(0),
    #                               all_levels.columns.get_level_values(0) + '_' + all_levels.columns.get_level_values(1)
    #                              )

    # Sett datatype tilbake til det den hadde i utgangpunktet
    if keep_empty:
        for col in groupcols:
            all_levels[col] = all_levels[col].astype(dataframe[col].dtype)

    return all_levels


def fill_na_dict(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    for col, fill_val in mapping.items():
        df[col] = df[col].fillna(fill_val)
    return df


cc = ["innvgr", "alder4", "botid6", "utd_3gr", "heldel"]

rader = 100

df = pd.DataFrame(
    [[np.random.choice([None, 1, 3, 7]) for _ in range(5)] for _ in range(rader)],
    columns=cc,
)
df["ant_jobb"] = [np.random.randint(0, 10) for _ in range(rader)]

# all_combinations = pd.MultiIndex.from_product([df[col] for col in cc])
# joined = all_combinations.join(df.set_index(cc))
df["ant_jobb_sum"] = df.groupby(cc, dropna=False)["ant_jobb"].transform("sum")

display(df)
display(df["ant_jobb_sum"].describe())

dff = all_combos(df, cc, func="sum")
display(dff)
print(dff.ant_jobb_sum.describe())

dff = all_combos_agg(df, groupcols=cc, func="sum")
display(dff)
print(dff.ant_jobb_sum.describe())

import itertools


cc = ["innvgr", "alder4", "botid6", "utd_3gr", "heldel"]

grouped_dfs = []


# Calculate sum for each individual column

for col in cc:
    grouped_total = df.groupby([col])["ant_jobb"].sum().reset_index()

    grouped_dfs.append(grouped_total)


# Calculate sum for each combination of columns up to all five columns

for i in range(2, len(cc) + 1):
    for cols in itertools.combinations(cc, i):
        grouped_total = df.groupby(list(cols))["ant_jobb"].sum().reset_index()

        cols_reordered = [
            col for col in cc if col in cols
        ]  # Reorder columns for the current combination

        grouped_total = grouped_total.reindex(columns=cols_reordered)

        grouped_dfs.append(grouped_total)


# Concatenate all the grouped dataframes into one dataframe

test_df = pd.concat(grouped_dfs)

display(test_df)
print(test_df.ant_jobb.describe())


class ParallelMapper(ParallelBase):
    """Run functions in parallell.

    The main methods are 'map' and 'chunkwise'. map runs a single function for
    each item of an iterable, while chunkwise splits an iterable in roughly equal
    length parts and runs a function on each chunk.

    The class also provides functions for reading and writing files in parallell
    in dapla.

    Nothing gets printed during execution if running in a notebook. Tip:
    set processes=1 to run without parallelization when debugging.

    Note that when using the default backend 'multiprocessing', all code except for
    imports and functions should be guarded by 'if __name__ == "__main__"' to not cause
    an internal loop. This is not the case if setting backend to 'loky'. See joblib's
    documentation: https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation

    Args:
        processes: Number of parallel processes. Set to 1 to run without
            parallelization.
        backend: Defaults to "multiprocessing". Can be set to any
            backend supported by joblib's Parallel class
            (except for "multiprocessing").
        context: Start method for the processes. Defaults to 'spawn'
            to avoid frozen processes.
        **kwargs: Keyword arguments to be passed to either
            multiprocessing.Pool or joblib.Parallel, depending
            on the chosen backend.
    """

    def __init__(
        self,
        processes: int,
        backend: str = "multiprocessing",
        context: str = "spawn",
        **kwargs,
    ):
        self.processes = processes
        self.backend = backend
        self.context = context
        self.kwargs = kwargs
        self.funcs: list[functools.partial] = []
        self.results: list[Any] = []
        self._source: list[str] = []

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

        Examples
        --------

        iterable = [1, 2, 3, 4]

        """
        self.validate_execution(func)
        func_with_kwargs = functools.partial(func, **kwargs)

        if self.processes == 1:
            return list(map(func_with_kwargs, iterable))

        if self.backend == "multiprocessing":
            with multiprocessing.get_context(self.context).Pool(
                self.processes, **self.kwargs
            ) as pool:
                return pool.map(func_with_kwargs, iterable)
        else:
            with Parallel(
                n_jobs=self.processes, backend=self.backend, **self.kwargs
            ) as parallel:
                return parallel(delayed(func)(item, **kwargs) for item in iterable)

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

    def _execute(self) -> list[Any]:
        [self.validate_execution(func) for func in self.funcs]

        if self.processes == 1:
            return [func() for func in self.funcs]

        if self.backend != "multiprocessing":
            with Parallel(
                n_jobs=self.processes, backend=self.backend, **self.kwargs
            ) as parallel:
                return parallel(delayed(func)() for func in self.funcs)

        with multiprocessing.get_context(self.context).Pool(
            self.processes, **self.kwargs
        ) as pool:
            results = [pool.apply_async(func) for func in self.funcs]
            return [result.get() for result in results]

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
        ...     p = ParallelPool()
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

        return self._execute()

    def write_municipality_data(
        self,
        in_data: dict[str, str | GeoDataFrame],
        out_data: str | dict[str, str],
        with_neighbors: bool = False,
        municipalities: GeoDataFrame | None = None,
        funcdict: dict[str, Callable] | None = None,
        file_type: str = "parquet",
        muni_number_col: str = "KOMMUNENR",
        strict: bool = False,
        write_empty: bool = False,
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
            write_empty: If False (default), municipalities with no data will be skipped.
                If True, an empty parquet file will be written.

        """
        shared_kwds = {
            "municipalities": municipalities,
            "muni_number_col": muni_number_col,
            "file_type": file_type,
            "write_empty": write_empty,
            "with_neighbors": with_neighbors,
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
            self._source.append("write_municipality_data")

        return self._execute()
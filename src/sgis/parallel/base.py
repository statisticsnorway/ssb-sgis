from geopandas import GeoDataFrame
from pandas import DataFrame

from ..helpers import in_jupyter


class LocalFunctionError(ValueError):
    def __init__(self, func: str):
        self.func = func.__name__

    def __str__(self):
        return (
            f"{self.func}. "
            "In Jupyter, functions to be multiprocessed must \n"
            "be defined in and imported from another file when context='spawn'. \n"
            "Note that setting context='fork' might cause freezing processes.\n"
        )


class ParallelBase:
    def validate_execution(self, func):
        if (
            func.__module__ == "__main__"
            and self.context == "spawn"
            and self.backend == "multiprocessing"
            and in_jupyter()
        ):
            raise LocalFunctionError(func)

    def chunksort_df(df: DataFrame | GeoDataFrame, n: int, column: str):
        df = df.sort_values(column).reset_index(drop=True)

        indexes = list(df.index)
        splitted = []
        for i in range(n):
            splitted.append(indexes[i::n])

        indexes_sorted = [j for i in reversed(splitted) for j in i]

        return df.loc[indexes_sorted].reset_index(drop=True)

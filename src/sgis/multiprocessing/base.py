from geopandas import GeoDataFrame
from pandas import DataFrame

from ..helpers import in_jupyter


class MultiProcessingBase:
    def validate_execution(self, func):
        if func.__module__ == "__main__" and self.context == "spawn" and in_jupyter():
            raise ValueError(
                "in Jupyter, functions to be multiprocessed must "
                "be defined in and imported from another file when context='spawn'. "
                "Note that setting context='fork' might cause freezing processes."
            )

    def chunksort_df(df: DataFrame | GeoDataFrame, n: int, column: str):
        df = df.sort_values(column).reset_index(drop=True)

        indexes = list(df.index)
        splitted = []
        for i in range(n):
            splitted.append(indexes[i::n])

        indexes_sorted = [j for i in reversed(splitted) for j in i]

        return df.loc[indexes_sorted].reset_index(drop=True)

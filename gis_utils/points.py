from geopandas import GeoDataFrame
from pandas import DataFrame
import numpy as np


class Points:
    def __init__(
        self, 
        points: GeoDataFrame,
        crs: str | int,
        id_col: str | list[str, str] | tuple[str, str] | None = None,
    ) -> None:

        self.points = points.to_crs(crs)
        
        self.wkt = [geom.wkt for geom in points.geometry]
        
        self.id_col = id_col

    def make_id_dict(self):
        if self.id_col:
            self.id_dict = {temp_idx: idx for temp_idx, idx in zip(self.points.temp_idx, self.points[self.id_col])}

    def make_temp_idx(self, start: int):

        self.points["temp_idx"] = np.arange(start=start, stop=start+len(self.points), step=1)
        self.points["temp_idx"] = self.points["temp_idx"].astype(str)

        if self.id_col:
            self.id_dict = {temp_idx: idx for temp_idx, idx in zip(self.points.temp_idx, self.points[self.id_col])}

    def check_id_col(
        self,
        index: int,
        id_col: str | list[str, str] | tuple[str, str] | None = None,
        ) -> None:

        if not id_col:
            return

        if isinstance(id_col, str):
            pass
        elif isinstance(id_col, (list, tuple)) and len(id_col) == 2:
            id_col = id_col[index]
        else:
            raise ValueError("'id_col' should be a string or a list/tuple with two strings.")

        if not id_col in self.points.columns:
            raise KeyError(f"'startpoints' has no attribute '{id_col}'")

        self.id_col = id_col

    def n_missing(
        self, 
        results: GeoDataFrame | DataFrame,
        col: str,
        ) -> None:

        self.points["n_missing"] = self.points["temp_idx"].map(
            len(results[col].unique()) - results.groupby(col).count().iloc[:, 0]
        )


class StartPoints(Points):
    def __init__(
        self, 
        points: GeoDataFrame,
        crs: str | int,
        id_col: str | list[str, str] | tuple[str, str] | None = None,
    ) -> None:

        super().__init__(points, crs=crs, id_col=id_col)

        self.check_id_col(id_col=id_col, index=0)


class EndPoints(Points):
    def __init__(
        self, 
        points: GeoDataFrame,
        crs: str | int,
        id_col: str | list[str, str] | tuple[str, str] | None,
    ) -> None:

        super().__init__(points, crs=crs, id_col=id_col)

        self.check_id_col(id_col=id_col, index=0)
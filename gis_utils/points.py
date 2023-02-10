from geopandas import GeoDataFrame
from pandas import DataFrame
from uuid import uuid4


class Points:
    def __init__(
        self, 
        points: GeoDataFrame,
        crs: str | int,
        id_col: str | list[str, str] | tuple[str, str] | None = None,
    ) -> None:

        self.points = points.to_crs(crs)
        
        self.wkt = [geom.wkt for geom in points.geometry]
        
        self.points["temp_idx"] = [str(uuid4()) for _ in range(len(points))]

        self.id_col = id_col

    def make_id_dict(self):
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
        results: GeoDataFrame | DataFrame
        ) -> None:

        self.points["n_missing_ori"] = self.points["temp_idx"].map(
            len(results.origin.unique()) - results.groupby("origin")["destination"].count()
        )
        self.points["n_missing_des"] = self.points["temp_idx"].map(
            len(results.destination.unique()) - results.groupby("destination")["origin"].count()
        )
        self.points["n_missing"] = self.points[["n_missing_ori", "n_missing_des"]].sum(axis=1)


class StartPoints(Points):
    def __init__(
        self, 
        points: GeoDataFrame,
        crs: str | int,
        id_col: str | list[str, str] | tuple[str, str] | None = None,
    ) -> None:

        super().__init__(points, crs=crs, id_col=id_col)

        self.check_id_col(id_col=id_col, index=0)

        self.make_id_dict()


class EndPoints(Points):
    def __init__(
        self, 
        points: GeoDataFrame,
        crs: str | int,
        id_col: str | list[str, str] | tuple[str, str] | None,
    ) -> None:

        super().__init__(points, crs=crs, id_col=id_col)

        self.check_id_col(id_col=id_col, index=0)

        self.make_id_dict()
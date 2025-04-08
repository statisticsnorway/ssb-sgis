import functools
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from shapely import Geometry
from shapely import STRtree

try:
    import dask.array as da
except ImportError:
    pass

from .utils import _unary_union_for_notna


@dataclass
class FunctionRunner(ABC):
    n_jobs: int | None
    backend: str | None

    @abstractmethod
    def run(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> np.ndarray:
        pass


@dataclass
class DissolveRunner:
    n_jobs: int
    backend: str | None = None

    def run(
        self,
        df: GeoDataFrame | GeoSeries | pd.DataFrame | pd.Series,
        by: str | list[str] | None = None,
        grid_size: float | int | None = None,
        **kwargs,
    ) -> GeoSeries | GeoDataFrame:
        try:
            geom_col = df.geometry.name
        except AttributeError:
            try:
                geom_col = df.name
                if geom_col is None:
                    geom_col = "geometry"
            except AttributeError:
                geom_col = "geometry"
        try:
            crs = df.crs
        except AttributeError:
            crs = None

        try:
            groupby_obj = df.groupby(by, **kwargs)[geom_col]
        except KeyError:
            groupby_obj = df.groupby(by, **kwargs)

        unary_union_for_grid_size = functools.partial(
            _unary_union_for_notna, grid_size=grid_size
        )
        if self.n_jobs is None or self.n_jobs == 1:
            results = groupby_obj.agg(unary_union_for_grid_size)
        else:
            backend = self.backend or "loky"
            with joblib.Parallel(n_jobs=self.n_jobs, backend=backend) as parallel:
                results = parallel(
                    joblib.delayed(unary_union_for_grid_size)(geoms)
                    for _, geoms in groupby_obj
                )
        if kwargs.get("as_index", True):
            return GeoSeries(
                results,
                index=groupby_obj.size().index,
                name=geom_col,
                crs=crs,
            )
        else:
            return GeoDataFrame(results, geometry=geom_col, crs=crs)


def strtree_query(arr1, arr2, **kwargs):
    tree = STRtree(arr2)
    return tree.query(arr1, **kwargs)


@dataclass
class RTreeRunner:
    n_jobs: int
    backend: str = "loky"

    def query(
        self, arr1: np.ndarray, arr2: np.ndarray, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        if (
            self.n_jobs > 1
            and len(arr1) / self.n_jobs > 1000
            and len(arr1) / len(arr2) > 3
        ):
            chunks = np.array_split(np.arange(len(arr1)), self.n_jobs)
            with joblib.Parallel(self.n_jobs, backend=self.backend) as parallel:
                results = parallel(
                    joblib.delayed(strtree_query)(arr1[chunk], arr2, **kwargs)
                    for chunk in chunks
                )
            left = np.concatenate([x[0] for x in results])
            right = np.concatenate([x[1] for x in results])
            return left, right
        elif (
            self.n_jobs > 1
            and len(arr2) / self.n_jobs > 1000
            and len(arr2) / len(arr1) > 3
        ):
            chunks = np.array_split(np.arange(len(arr2)), self.n_jobs)
            with joblib.Parallel(self.n_jobs, backend=self.backend) as parallel:
                results = parallel(
                    joblib.delayed(strtree_query)(arr1, arr2[chunk], **kwargs)
                    for chunk in chunks
                )
            left = np.concatenate([x[0] for x in results])
            right = np.concatenate([x[1] for x in results])
            return left, right
        return strtree_query(arr1, arr2, **kwargs)


@dataclass
class OverlayRunner(FunctionRunner):
    n_jobs: None = None
    backend: None = None

    def __post_init__(self) -> None:
        if self.n_jobs is not None or self.backend is not None:
            raise ValueError(
                "Cannot set n_jobs or backend on OverlayRunner. Use the classes meant for parallelization, DaskOverlayRunner or JoblibOverlayRunner."
            )

    def run(
        self,
        func: Callable,
        arr1: np.ndarray,
        arr2: np.ndarray,
        **kwargs: int | float | None,
    ) -> np.ndarray:
        return func(arr1, arr2, **kwargs)


@dataclass
class DaskOverlayRunner(FunctionRunner):
    n_jobs: int
    backend: None = None

    def run(
        self,
        func: Callable,
        arr1: np.ndarray,
        arr2: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        if len(arr1) // self.n_jobs <= 1:
            try:
                return func(arr1, arr2, **kwargs)
            except TypeError as e:
                raise TypeError(
                    e, {type(x) for x in arr1}, {type(x) for x in arr2}
                ) from e
        arr1 = da.from_array(arr1, chunks=len(arr1) // self.n_jobs)
        arr2 = da.from_array(arr2, chunks=len(arr2) // self.n_jobs)
        res = arr1.map_blocks(func, arr2, **kwargs, dtype=float)
        return res.compute(
            scheduler="threads", optimize_graph=False, num_workers=self.n_jobs
        )


@dataclass
class JoblibOverlayRunner(FunctionRunner):
    n_jobs: int
    backend: str = "loky"

    def run(
        self,
        func: Callable,
        arr1: np.ndarray,
        arr2: np.ndarray,
        **kwargs,
    ) -> list[Geometry]:
        if len(arr1) // self.n_jobs <= 1:
            try:
                return func(arr1, arr2, **kwargs)
            except TypeError as e:
                raise TypeError(
                    e, {type(x) for x in arr1}, {type(x) for x in arr2}
                ) from e

        chunks = np.array_split(np.arange(len(arr1)), self.n_jobs)
        with joblib.Parallel(n_jobs=self.n_jobs, backend=self.backend) as parallel:
            return np.concatenate(
                parallel(
                    joblib.delayed(func)(arr1[chunk], arr2[chunk], **kwargs)
                    for chunk in chunks
                )
            )

        # with joblib.Parallel(n_jobs=self.n_jobs, backend=self.backend) as parallel:
        #     return parallel(
        #         joblib.delayed(func)(g1, g2, **kwargs)
        #         for g1, g2 in zip(arr1, arr2, strict=True)
        #     )

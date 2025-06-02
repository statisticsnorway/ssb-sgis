import functools
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from shapely import Geometry
from shapely import STRtree
from shapely import get_parts
from shapely import make_valid
from shapely import union_all
from shapely.errors import GEOSException

from .utils import _unary_union_for_notna
from .utils import make_valid_and_keep_geom_type


@dataclass
class AbstractRunner(ABC):
    """Blueprint for 'runner' classes.

    Subclasses must implement a 'run' method.

    Args:
        n_jobs: Number of workers.
        backend: Backend for the workers.
    """

    n_jobs: int
    backend: str | None = None

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Abstract run method."""

    def __str__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}(n_jobs={self.n_jobs}, backend='{self.backend}')"
        )


@dataclass
class UnionRunner(AbstractRunner):
    """Run shapely.union_all with pandas.groupby.

    Subclasses must implement a 'run' method that takes the arguments
    'df' (GeoDataFrame or GeoSeries), 'by' (optional column to group by), 'grid_size'
    (passed to shapely.union_all) and **kwargs passed to pandas.DataFrame.groupby.
    Defaults to None, meaning the default runner with number of workers set
    to 'n_jobs'.


    Args:
        n_jobs: Number of workers.
        backend: Backend for the workers.
    """

    n_jobs: int
    backend: str | None = None

    def run(
        self,
        df: GeoDataFrame | GeoSeries | pd.DataFrame | pd.Series,
        by: str | list[str] | None = None,
        grid_size: float | int | None = None,
        **kwargs,
    ) -> GeoSeries | GeoDataFrame:
        """Run groupby on geometries in parallel (if n_jobs > 1)."""
        # assume geometry column is 'geometry' if input is pandas.Series og pandas.DataFrame
        try:
            geom_col: str = df.geometry.name
        except AttributeError:
            try:
                geom_col: str | None = df.name
                if geom_col is None:
                    geom_col = "geometry"
            except AttributeError:
                geom_col = "geometry"
        try:
            crs = df.crs
        except AttributeError:
            crs = None

        unary_union_for_grid_size = functools.partial(
            _unary_union_for_notna, grid_size=grid_size
        )

        as_index = kwargs.pop("as_index", True)
        if by is None and "level" not in kwargs:
            by = np.zeros(len(df), dtype="int64")

        try:
            # (Geo)DataFrame
            groupby_obj = df.groupby(by, **kwargs)[geom_col]
        except KeyError:
            # (Geo)Series
            groupby_obj = df.groupby(by, **kwargs)

        if self.n_jobs is None or self.n_jobs == 1:
            results = groupby_obj.agg(unary_union_for_grid_size)
            index = results.index
        else:
            backend = self.backend or "loky"
            with joblib.Parallel(n_jobs=self.n_jobs, backend=backend) as parallel:
                results = parallel(
                    joblib.delayed(unary_union_for_grid_size)(geoms)
                    for _, geoms in groupby_obj
                )
            index = groupby_obj.size().index
        agged = GeoSeries(results, index=index, name=geom_col, crs=crs)
        if not as_index:
            return agged.reset_index()
        return agged


@dataclass
class GridSizeUnionRunner(UnionRunner):
    """Run shapely.union_all with pandas.groupby for different grid sizes until no GEOSException is raised.

    Subclasses must implement a 'run' method that takes the arguments
    'df' (GeoDataFrame or GeoSeries), 'by' (optional column to group by), 'grid_size'
    (passed to shapely.union_all) and **kwargs passed to pandas.DataFrame.groupby.
    Defaults to None, meaning the default runner with number of workers set
    to 'n_jobs'.


    Args:
        n_jobs: Number of workers.
        backend: Backend for the workers.
    """

    n_jobs: int
    backend: str | None = None
    grid_sizes: list[float | int] | None = None

    def __post_init__(self) -> None:
        """Check that grid_sizes is passed."""
        if self.grid_sizes is None:
            raise ValueError(
                f"must set 'grid_sizes' in the {self.__class__.__name__} initialiser."
            )

    def run(
        self,
        df: GeoDataFrame | GeoSeries | pd.DataFrame | pd.Series,
        by: str | list[str] | None = None,
        grid_size: int | float | None = None,
        **kwargs,
    ) -> GeoSeries | GeoDataFrame:
        """Run groupby on geometries in parallel (if n_jobs > 1) with grid_sizes."""
        try:
            return super().run(df, by=by, grid_size=grid_size, **kwargs)
        except GEOSException:
            pass
        for i, grid_size in enumerate(self.grid_sizes):
            try:
                return super().run(df, by=by, grid_size=grid_size, **kwargs)
            except GEOSException as e:
                if i == len(self.grid_sizes) - 1:
                    raise e


def _strtree_query(
    arr1: np.ndarray,
    arr2: np.ndarray,
    method: str,
    indices1: np.ndarray | None = None,
    indices2: np.ndarray | None = None,
    **kwargs,
):
    tree = STRtree(arr2)
    func = getattr(tree, method)
    left, right = func(arr1, **kwargs)
    if indices1 is not None:
        index_mapper1 = {i: x for i, x in enumerate(indices1)}
        left = np.array([index_mapper1[i] for i in left])
    if indices2 is not None:
        index_mapper2 = {i: x for i, x in enumerate(indices2)}
        right = np.array([index_mapper2[i] for i in right])
    return left, right


@dataclass
class RTreeQueryRunner(AbstractRunner):
    """Run shapely.STRTree chunkwise.

    Subclasses must implement a 'query' method that takes a numpy.ndarray
    of geometries as 0th and 1st argument and **kwargs passed to the query method,
    chiefly 'predicate' and 'distance'. The 'query' method should return a tuple
    of two arrays representing the spatial index pairs of the left and right input arrays.
    Defaults to None, meaning the default runner with number of workers set
    to 'n_jobs'.

    Args:
        n_jobs: Number of workers.
        backend: Backend for the workers.
    """

    n_jobs: int
    backend: str = "loky"

    def run(
        self, arr1: np.ndarray, arr2: np.ndarray, method: str = "query", **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run a spatial rtree query and return indices of hits from arr1 and arr2 in a tuple of two arrays."""
        if (
            (self.n_jobs or 1) > 1
            and len(arr1) / self.n_jobs > 10_000
            and len(arr1) / len(arr2)
        ):
            chunks = np.array_split(np.arange(len(arr1)), self.n_jobs)
            assert sum(len(x) for x in chunks) == len(arr1)
            with joblib.Parallel(self.n_jobs, backend=self.backend) as parallel:
                results = parallel(
                    joblib.delayed(_strtree_query)(
                        arr1[chunk],
                        arr2,
                        method=method,
                        indices1=chunk,
                        **kwargs,
                    )
                    for chunk in chunks
                )
            left = np.concatenate([x[0] for x in results])
            right = np.concatenate([x[1] for x in results])
            return left, right
        elif (
            (self.n_jobs or 1) > 1
            and len(arr2) / self.n_jobs > 10_000
            and len(arr2) / len(arr1)
        ):
            chunks = np.array_split(np.arange(len(arr2)), self.n_jobs)
            with joblib.Parallel(self.n_jobs, backend=self.backend) as parallel:
                results = parallel(
                    joblib.delayed(_strtree_query)(
                        arr1,
                        arr2[chunk],
                        method=method,
                        indices2=chunk,
                        **kwargs,
                    )
                    for chunk in chunks
                )
            left = np.concatenate([x[0] for x in results])
            right = np.concatenate([x[1] for x in results])
            return left, right

        return _strtree_query(arr1, arr2, method=method, **kwargs)


@dataclass
class OverlayRunner(AbstractRunner):
    """Run a vectorized shapely overlay operation on two equal-length numpy arrays.

    Subclasses must implement a 'run' method that takes an overlay function (shapely.intersection, shapely.difference etc.)
    as 0th argument and two numpy.ndarrays of same length as 1st and 2nd argument.
    The 'run' method should also take the argument 'grid_size' to be passed to the overlay function
    and the argument 'geom_type' which is used to keep only relevant geometries (polygon, line or point)
    in cases of GEOSExceptions caused by geometry type mismatch.
    Defaults to an instance of OverlayRunner, which is run sequencially (no n_jobs)
    because the vectorized shapely functions are usually faster than any attempt to parallelize.
    """

    n_jobs: None = None
    backend: None = None

    def run(
        self,
        func: Callable,
        arr1: np.ndarray,
        arr2: np.ndarray,
        grid_size: int | float | None,
        geom_type: str | None,
    ) -> np.ndarray:
        """Run the overlay operation (func) with fallback.

        First tries to run func, then, if GEOSException, geometries are made valid
        and only geometries with correct geom_type (point, line, polygon) are kept
        in GeometryCollections.
        """
        try:
            return func(arr1, arr2, grid_size=grid_size)
        except GEOSException:
            arr1 = make_valid_and_keep_geom_type(arr1, geom_type=geom_type)
            arr2 = make_valid_and_keep_geom_type(arr2, geom_type=geom_type)
            arr1 = arr1.loc[lambda x: x.index.isin(arr2.index)].to_numpy()
            arr2 = arr2.loc[lambda x: x.index.isin(arr1.index)].to_numpy()
            return func(arr1, arr2, grid_size=grid_size)


@dataclass
class GridSizeOverlayRunner(OverlayRunner):
    """Run a shapely overlay operation rowwise for different grid_sizes until success."""

    n_jobs: int
    backend: str | None
    grid_sizes: list[float | int] | None = None

    def __post_init__(self) -> None:
        """Check that grid_sizes is passed."""
        if self.grid_sizes is None:
            raise ValueError(
                f"must set 'grid_sizes' in the {self.__class__.__name__} initialiser."
            )

    def run(
        self,
        func: Callable,
        arr1: np.ndarray,
        arr2: np.ndarray,
        grid_size: int | float | None = None,
        geom_type: str | None = None,
    ) -> np.ndarray:
        """Run the overlay operation rowwise with fallback.

        The overlay operation (func) is looped for each row in arr1 and arr2
        as 0th and 1st argument to 'func' and 'grid_size' as keyword argument. If a GEOSException is thrown,
        geometries are made valid and GeometryCollections are forced to either
        (Multi)Point, (Multi)Polygon or (Multi)LineString, depending on the value in "geom_type".
        Then, if Another GEOSException is thrown, the overlay operation is looped for the grid_sizes given
        in the instance's 'grid_sizes' attribute.

        """
        kwargs = dict(
            grid_size=grid_size,
            geom_type=geom_type.lower() if geom_type is not None else None,
            grid_sizes=self.grid_sizes,
        )
        with joblib.Parallel(self.n_jobs, backend="threading") as parallel:
            return parallel(
                joblib.delayed(_run_overlay_rowwise)(func, g1, g2, **kwargs)
                for g1, g2 in zip(arr1, arr2, strict=True)
            )


def _fix_gemetry_fast(geom: Geometry, geom_type: str | None) -> Geometry:
    geom = make_valid(geom)
    if geom.geom_type == geom_type or geom_type is None:
        return geom
    return union_all([g for g in get_parts(geom) if geom_type in g.geom_type])


def _run_overlay_rowwise(
    func: Callable,
    geom1: Geometry,
    geom2: Geometry,
    grid_size: float | int | None,
    geom_type: str | None,
    grid_sizes: list[float | int],
) -> Geometry:
    try:
        return func(geom1, geom2, grid_size=grid_size)
    except GEOSException:
        pass
    geom1 = _fix_gemetry_fast(geom1, geom_type)
    geom2 = _fix_gemetry_fast(geom2, geom_type)
    try:
        return func(geom1, geom2)
    except GEOSException:
        pass
    for i, grid_size in enumerate(grid_sizes):
        try:
            return func(geom1, geom2, grid_size=grid_size)
        except GEOSException as e:
            if i == len(grid_sizes) - 1:
                raise e

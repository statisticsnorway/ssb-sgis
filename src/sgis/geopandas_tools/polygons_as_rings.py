from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from geopandas.array import GeometryArray
from numpy.typing import NDArray
from pyproj import CRS
from shapely import difference
from shapely import get_coordinates
from shapely import get_exterior_ring
from shapely import get_interior_ring
from shapely import get_num_interior_rings
from shapely import linearrings
from shapely import make_valid
from shapely import polygons
from shapely import unary_union
from shapely.geometry import LinearRing
from shapely.geometry import Polygon

from .conversion import to_gdf
from .conversion import to_geoseries


class PolygonsAsRings:
    """Convert polygons to linearrings, apply linestring functions, then convert back to polygons."""

    def __init__(
        self,
        polys: GeoDataFrame | GeoSeries | GeometryArray,
        crs: CRS | Any | None = None,
        allow_multipart: bool = False,
        gridsize: int | None = None,
    ) -> None:
        """Initialize the PolygonsAsRings object with polygons and optional CRS information.

        Args:
            polys: GeoDataFrame, GeoSeries, or GeometryArray containing polygon geometries.
            crs: Coordinate Reference System to be used, defaults to None.
            allow_multipart: Allow multipart polygons if True, defaults to False.
            gridsize: Size of the grid for any grid operations, defaults to None.
        """
        if not isinstance(polys, (pd.DataFrame, pd.Series, GeometryArray)):
            raise TypeError(type(polys))

        self.polyclass = polys.__class__

        if not isinstance(polys, pd.DataFrame):
            polys = to_gdf(polys, crs)

        if not allow_multipart and not (polys.geom_type == "Polygon").all():
            raise ValueError(
                "All geometries must be single-type Polygons. Set allow_multipart=True to allow MultiPolygons",
                polys.geom_type.value_counts(),
            )
        if not polys.geom_type.isin(["Polygon", "MultiPolygon"]).all():
            raise ValueError(
                f"All geometries must be Polygons. Got {polys.geom_type.value_counts()}"
            )

        self._index_mapper = dict(enumerate(polys.index))
        self.gdf = polys.reset_index(drop=True)

        if crs is not None:
            self.crs = crs
        elif hasattr(polys, "crs"):
            self.crs = polys.crs
        else:
            self.crs = None

        if not len(self.gdf):
            self.rings = pd.Series()
            return

        exterior = pd.Series(
            get_exterior_ring(self.gdf.geometry.values),
            index=self._exterior_index,
        )

        self.max_rings: int = np.max(get_num_interior_rings(self.gdf.geometry.values))

        if not self.max_rings:
            self.rings = exterior
            return

        # series same length as number of potential inner rings
        interiors = pd.Series(
            (
                [
                    [get_interior_ring(geom, i) for i in range(self.max_rings)]
                    for geom in self.gdf.geometry
                ]
            ),
        ).explode()

        interiors.index = self._interiors_index

        interiors = interiors.dropna()

        self.rings = pd.concat([exterior, interiors])

    def get_rings(self, agg: bool = False) -> GeoDataFrame | GeoSeries | np.ndarray:
        """Retrieve rings from the polygons, optionally aggregating them.

        Args:
            agg: If True, aggregate the rings into single geometries.

        Returns:
            The rings either aggregated or separated, in the type of
                the input polygons.
        """
        gdf = self.gdf.copy()
        rings = self.rings.copy()
        if not len(rings):
            return GeoSeries()
        if agg:
            gdf.geometry = rings.groupby(level=1).agg(unary_union)
        else:
            rings.index = rings.index.get_level_values(1)
            rings.name = "geometry"
            gdf = gdf.drop(columns="geometry").join(rings)

        if issubclass(self.polyclass, pd.DataFrame):
            return GeoDataFrame(gdf, crs=self.crs)
        if issubclass(self.polyclass, pd.Series):
            return GeoSeries(gdf.geometry)
        return self.polyclass(gdf.geometry.values)

    def apply_numpy_func_to_interiors(
        self,
        func: Callable,
        args: tuple | None = None,
        kwargs: dict | None = None,
    ) -> "PolygonsAsRings":
        """Apply a numpy function specifically to the interior rings of the polygons.

        Args:
            func: Numpy function to apply.
            args: Tuple of positional arguments for the function.
            kwargs: Dictionary of keyword arguments for the function.

        Returns:
            PolygonsAsRings: The instance itself after applying the function.
        """
        kwargs = kwargs or {}
        args = args or ()
        arr: NDArray[LinearRing] = self.rings.loc[self.is_interior].values
        index: pd.Index = self.rings.loc[self.is_interior].index
        results = pd.Series(
            np.array(func(arr, *args, **kwargs)),
            index=index,
        )
        self.rings.loc[self.is_interior] = results
        return self

    def apply_numpy_func(
        self,
        func: Callable,
        args: tuple | None = None,
        kwargs: dict | None = None,
    ) -> "PolygonsAsRings":
        """Apply a numpy function to all rings of the polygons.

        Args:
            func: Numpy function to apply.
            args: Tuple of positional arguments for the function.
            kwargs: Dictionary of keyword arguments for the function.

        Returns:
            PolygonsAsRings: The instance itself after applying the function.
        """
        kwargs = kwargs or {}
        args = args or ()

        results = np.array(func(self.rings.values, *args, **kwargs))

        if len(results) != len(self.rings):
            raise ValueError(
                f"Different length of results. Got {len(results)} and {len(self.rings)} original rings"
            )

        self.rings.loc[:] = results  # type: ignore [call-overload]

        return self

    def apply_geoseries_func(
        self,
        func: Callable,
        args: tuple | None = None,
        kwargs: dict | None = None,
    ) -> "PolygonsAsRings":
        """Apply a function that operates on a GeoSeries to the rings.

        Args:
            func: Function to apply that expects a GeoSeries.
            args: Tuple of positional arguments for the function.
            kwargs: Dictionary of keyword arguments for the function.

        Returns:
            PolygonsAsRings: The instance itself after applying the function.
        """
        kwargs = kwargs or {}
        args = args or ()

        self.rings.loc[:] = np.array(  # type: ignore [call-overload]
            func(
                GeoSeries(
                    self.rings.values,
                    crs=self.crs,
                    index=self.rings.index.get_level_values(1).map(self._index_mapper),
                ),
                *args,
                **kwargs,
            )
        )

        return self

    def apply_gdf_func(
        self,
        func: Callable,
        args: tuple | None = None,
        kwargs: dict | None = None,
    ) -> "PolygonsAsRings":
        """Apply a function that operates on a GeoDataFrame to the rings.

        Args:
            func: Function to apply that expects a GeoDataFrame.
            args: Tuple of positional arguments for the function.
            kwargs: Dictionary of keyword arguments for the function.

        Returns:
            PolygonsAsRings: The instance itself after applying the function.
        """
        kwargs = kwargs or {}
        args = args or ()

        gdf = GeoDataFrame(
            {"geometry": self.rings.values},
            crs=self.crs,
            index=self.rings.index.get_level_values(1).map(self._index_mapper),
        ).join(self.gdf.drop(columns="geometry"))

        assert len(gdf) == len(self.rings)

        gdf.index = self.rings.index

        self.rings.loc[:] = func(  # type: ignore [call-overload]
            gdf,
            *args,
            **kwargs,
        ).geometry.values

        return self

    @property
    def is_interior(self) -> bool:
        """Returns a boolean Series of whether the row is an interior ring."""
        return self.rings.index.get_level_values(0) == 1

    @property
    def is_exterior(self) -> bool:
        """Returns a boolean Series of whether the row is an exterior ring."""
        return self.rings.index.get_level_values(0) == 0

    @property
    def _interiors_index(self) -> pd.MultiIndex:
        """A three-leveled MultiIndex.

        Used to separate interior and exterior and sort the interior in
        the 'to_numpy' method.

        level 0: all 1s, indicating "is interior".
        level 1: gdf index repeated *self.max_rings* times.
        level 2: interior number index. 0 * len(gdf), 1 * len(gdf), 2 * len(gdf)...
        """
        if not self.max_rings:
            return pd.MultiIndex()
        len_gdf = len(self.gdf)
        n_potential_interiors = len_gdf * self.max_rings
        gdf_index = sorted(list(self.gdf.index) * self.max_rings)
        interior_number_index = np.tile(np.arange(self.max_rings), len_gdf)
        one_for_interior = np.repeat(1, n_potential_interiors)

        return pd.MultiIndex.from_arrays(
            [one_for_interior, gdf_index, interior_number_index]
        )

    @property
    def _exterior_index(self) -> pd.MultiIndex:
        """A three-leveled MultiIndex.

        Used to separate interior and exterior in the 'to_numpy' method.
        Only leve 1 is used for the exterior.

        level 0: all 0s, indicating "not interior".
        level 1: gdf index.
        level 2: All 0s.
        """
        zero_for_not_interior = np.repeat(0, len(self.gdf))
        return pd.MultiIndex.from_arrays(
            [zero_for_not_interior, self.gdf.index, zero_for_not_interior]
        )

    def to_gdf(self) -> GeoDataFrame:
        """Return the GeoDataFrame with polygons."""
        self.gdf.geometry = self.to_numpy()
        return self.gdf

    def to_geoseries(self) -> GeoSeries:
        """Return the GeoSeries with polygons."""
        self.gdf.geometry = self.to_numpy()
        return self.gdf.geometry

    def to_numpy(self) -> NDArray[Polygon]:
        """Return a numpy array of polygons."""
        if not len(self.rings):
            return np.array([])

        exterior = self.rings.loc[self.is_exterior].sort_index()
        assert exterior.shape == (len(self.gdf),)
        nonempty_exteriors = exterior.loc[lambda x: x.notna()]
        empty_exteriors = exterior.loc[lambda x: x.isna()]

        nonempty_interiors = self.rings.loc[self.is_interior]

        if not len(nonempty_interiors):
            nonempty_exteriors.loc[:] = make_valid(polygons(nonempty_exteriors.values))
            return pd.concat([empty_exteriors, nonempty_exteriors]).sort_index().values

        empty_interiors = pd.Series(
            [None for _ in range(len(self.gdf) * self.max_rings)],
            index=self._interiors_index,
        ).loc[lambda x: ~x.index.isin(nonempty_interiors.index)]

        interiors = (
            pd.concat([nonempty_interiors, empty_interiors])
            .sort_index()
            # make each ring level a column with same length and order as gdf
            .unstack(level=2)
            .sort_index()
        )
        assert interiors.shape == (len(self.gdf), self.max_rings), interiors.shape

        interiors = interiors.loc[
            interiors.index.get_level_values(1).isin(
                nonempty_exteriors.index.get_level_values(1)
            )
        ]
        assert interiors.index.get_level_values(1).equals(
            nonempty_exteriors.index.get_level_values(1)
        )

        # nan gives TypeError in shapely.polygons. None does not.
        for i, _ in enumerate(interiors.columns):
            interiors.loc[interiors.iloc[:, i].isna(), i] = None
        nonempty_exteriors.loc[nonempty_exteriors.isna()] = None

        # construct polygons with holes
        polys = make_valid(
            polygons(
                nonempty_exteriors.values,
                interiors.values,
            )
        )

        # interiors might have moved (e.g. snapped) so that they are not within the exterior
        # these interiors will not be holes, so we need to erase them manually
        interiors_as_polys = make_valid(polygons(interiors.values))
        # merge interior polygons into 1d array
        interiors_as_polys = np.array(
            [
                make_valid(unary_union(interiors_as_polys[i, :]))
                for i in range(interiors_as_polys.shape[0])
            ]
        )
        # erase rowwise
        nonempty_exteriors.loc[:] = make_valid(difference(polys, interiors_as_polys))
        return pd.concat([empty_exteriors, nonempty_exteriors]).sort_index().values


def get_linearring_series(geoms: GeoDataFrame | GeoSeries) -> pd.Series:
    """Convert geometries into a series of LinearRings.

    Args:
        geoms: GeoDataFrame or GeoSeries from which to extract LinearRings.

    Returns:
        pd.Series: A series containing LinearRings.
    """
    geoms = to_geoseries(geoms).explode(index_parts=False)
    coords, indices = get_coordinates(geoms, return_index=True)
    return pd.Series(linearrings(coords, indices=indices), index=geoms.index)


def _geoms_to_linearrings_fallback(
    exterior: pd.Series, interiors: pd.Series | None = None
) -> pd.Series:
    exterior.index = exterior.index.get_level_values(1)
    assert exterior.index.is_monotonic_increasing

    exterior = get_linearring_series(exterior)

    if interiors is None:
        return (
            pd.Series(
                make_valid(polygons(exterior.values)),
                index=exterior.index,
            )
            .groupby(level=0)
            .agg(lambda x: make_valid(unary_union(x)))
        )

    interiors.index = interiors.index.get_level_values(1)
    new_interiors = []
    for col in interiors:
        new_interiors.append(get_linearring_series(interiors[col]))

    all_none = [[None] * len(new_interiors)] * len(exterior)
    cols = list(interiors.columns)
    out_interiors = pd.DataFrame(
        all_none,
        columns=cols,
        index=exterior.index,
    )
    out_interiors[cols] = pd.concat(new_interiors, axis=1)
    for col in out_interiors:
        out_interiors.loc[out_interiors[col].isna(), col] = None

    return (
        pd.Series(
            make_valid(polygons(exterior.values, out_interiors.values)),
            index=exterior.index,
        )
        .groupby(level=0)
        .agg(lambda x: make_valid(unary_union(x)))
    )

import functools
import itertools
from typing import Callable, Iterable

import geopandas as gpd
import igraph
import networkx as nx
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryArray
from IPython.display import display
from networkx.algorithms import approximation as approx
from numpy import ndarray
from numpy.typing import NDArray
from pandas import Index
from shapely import (
    Geometry,
    STRtree,
    area,
    box,
    buffer,
    centroid,
    difference,
    distance,
    extract_unique_points,
    get_coordinates,
    get_exterior_ring,
    get_interior_ring,
    get_num_interior_rings,
    get_parts,
    get_rings,
    intersection,
    intersects,
    is_empty,
    is_ring,
    length,
    line_merge,
    linearrings,
    linestrings,
    make_valid,
    points,
    polygons,
    segmentize,
    simplify,
    unary_union,
    voronoi_polygons,
)
from shapely.errors import GEOSException
from shapely.geometry import (
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
)

from .conversion import to_gdf


class PolygonsAsRings:
    """Convert polygons to linearrings, apply linestring functions, then convert back to polygons."""

    def __init__(self, polys: GeoDataFrame | GeoSeries | GeometryArray, crs=None):
        if not isinstance(polys, (pd.DataFrame, pd.Series, GeometryArray)):
            raise TypeError(type(polys))

        self.polyclass = polys.__class__

        if not isinstance(polys, pd.DataFrame):
            polys = to_gdf(polys, crs)

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

    def get_rings(self, agg: bool = False):
        gdf = self.gdf.copy()
        rings = self.rings.copy()
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
        self, func: Callable, args: tuple | None = None, kwargs: dict | None = None
    ):
        """Run an array function on only the interior rings of the polygons."""
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
        self, func: Callable, args: tuple | None = None, kwargs: dict | None = None
    ):
        """Run a function that takes an array of lines/rings and returns an array of lines/rings."""
        kwargs = kwargs or {}
        args = args or ()

        self.rings.loc[:] = np.array(func(self.rings.values, *args, **kwargs))
        return self

    def apply_geoseries_func(
        self, func: Callable, args: tuple | None = None, kwargs: dict | None = None
    ):
        """Run a function that takes a GeoSeries and returns a GeoSeries."""
        kwargs = kwargs or {}
        args = args or ()

        self.rings.loc[:] = np.array(
            func(
                GeoSeries(self.rings, crs=self.crs, index=self.rings.index),
                *args,
                **kwargs,
            )
        )

        return self

    def apply_gdf_func(
        self, func: Callable, args: tuple | None = None, kwargs: dict | None = None
    ):
        """Run a function that takes a GeoDataFrame and returns a GeoDataFrame."""
        kwargs = kwargs or {}
        args = args or ()

        gdf = GeoDataFrame(
            {"geometry": self.rings.values},
            crs=self.crs,
            index=self.rings.index.get_level_values(1),
        ).join(self.gdf.drop(columns="geometry"))

        assert len(gdf) == len(self.rings)

        gdf.index = self.rings.index

        self.rings.loc[:] = func(
            gdf,
            *args,
            **kwargs,
        ).geometry.values

        return self

    @property
    def is_interior(self):
        return self.rings.index.get_level_values(0) == 1

    @property
    def is_exterior(self):
        return self.rings.index.get_level_values(0) == 0

    @property
    def _interiors_index(self):
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
    def _exterior_index(self):
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

    def to_numpy(self) -> NDArray[Polygon]:
        """Return a numpy array of polygons."""
        if not len(self.rings):
            return np.array([])

        exterior = self.rings.loc[self.is_exterior].sort_index().values
        assert exterior.shape == (len(self.gdf),)

        nonempty_interiors = self.rings.loc[self.is_interior]

        if not len(nonempty_interiors):
            return make_valid(polygons(exterior))

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
            .values
        )
        assert interiors.shape == (len(self.gdf), self.max_rings), interiors.shape

        return make_valid(polygons(exterior, interiors))

import warnings

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from shapely import Geometry

from ..conf import _get_instance
from ..conf import config
from .conversion import to_gdf
from .runners import RTreeQueryRunner

gdf_type_error_message = "'gdf' should be of type GeoDataFrame or GeoSeries."


def sfilter(
    gdf: GeoDataFrame | GeoSeries,
    other: GeoDataFrame | GeoSeries | Geometry,
    predicate: str = "intersects",
    distance: int | float | None = None,
    n_jobs: int | None = None,
    rtree_runner: RTreeQueryRunner | None = None,
) -> GeoDataFrame:
    """Filter a GeoDataFrame or GeoSeries by spatial predicate.

    Does an sjoin and returns the rows of 'gdf' that were returned
    without getting duplicates or columns from 'other'.
    Works with unique and non-unique index.

    Like 'select by location' in ArcGIS/QGIS, except that the
    selection is permanent.

    Args:
        gdf: The GeoDataFrame.
        other: The geometry object to filter 'gdf' by.
        predicate: Spatial predicate to use. Defaults to 'intersects'.
        distance: Max distance to allow if predicate=="dwithin".
        n_jobs: Number of workers.
        rtree_runner: Optionally debug/manipulate the spatial indexing operations.
            See the 'runners' module for example implementations.

    Returns:
        A copy of 'gdf' with only the rows matching the
        spatial predicate with 'other'.

    Examples:
    ---------
    >>> import sgis as sg
    >>> df1 = sg.to_gdf([(0, 0), (0, 1)])
    >>> df1
                      geometry
    0  POINT (0.00000 0.00000)
    1  POINT (0.00000 1.00000)
    >>> df2 = sg.to_gdf([(0, 0), (1, 2)])
    >>> df2
                      geometry
    0  POINT (0.00000 0.00000)
    1  POINT (1.00000 2.00000)

    Keep rows in df1 intersecting any geometry in df2.

    >>> sg.sfilter(df1, df2)
                      geometry
    0  POINT (0.00000 0.00000)

    Equivelent to sjoin-ing and selecting based on integer index
    (in case of non-unique index).

    >>> df1["idx"] = range(len(df1))
    >>> joined = df1.sjoin(df2)
    >>> df1.loc[df1["idx"].isin(joined["idx"])].drop(columns="idx")
                          geometry
    0  POINT (0.00000 0.00000)

    Also equivelent to using the intersects method, which
    is often a lot slower since df2 must be dissolved:

    >>> df1.loc[df1.intersects(df2.union_all())]
                      geometry
    0  POINT (0.00000 0.00000)

    """
    if not isinstance(gdf, (GeoDataFrame | GeoSeries)):
        raise TypeError(gdf_type_error_message)
    if not len(gdf):
        return gdf

    other = _sfilter_checks(other, crs=gdf.crs)

    indices = _get_sfilter_indices(
        gdf, other, predicate, distance, n_jobs, rtree_runner
    )

    return gdf.iloc[indices]


def sfilter_split(
    gdf: GeoDataFrame | GeoSeries,
    other: GeoDataFrame | GeoSeries | Geometry,
    predicate: str = "intersects",
    distance: int | float | None = None,
    n_jobs: int = 1,
    rtree_runner: RTreeQueryRunner | None = None,
) -> tuple[GeoDataFrame, GeoDataFrame]:
    """Split a GeoDataFrame or GeoSeries by spatial predicate.

    Like sfilter, but returns both the rows that do and do not match
    the spatial predicate as separate GeoDataFrames.

    Args:
        gdf: The GeoDataFrame.
        other: The geometry object to filter 'gdf' by.
        predicate: Spatial predicate to use. Defaults to 'intersects'.
        distance: Max distance to allow if predicate=="dwithin".
        n_jobs: Number of workers.
        rtree_runner: Optionally debug/manipulate the spatial indexing operations.
            See the 'runners' module for example implementations.

    Returns:
        A tuple of GeoDataFrames, one with the rows that match the spatial predicate
        and one with the rows that do not.

    Examples:
    ---------
    >>> import sgis as sg
    >>> df1 = sg.to_gdf([(0, 0), (0, 1)])
    >>> df1
                      geometry
    0  POINT (0.00000 0.00000)
    1  POINT (0.00000 1.00000)
    >>> df2 = sg.to_gdf([(0, 0), (1, 2)])
    >>> df2
                      geometry
    0  POINT (0.00000 0.00000)
    1  POINT (1.00000 2.00000)

    Split df1 into the rows that do and do not intersect df2.

    >>> intersecting, not_intersecting = sg.sfilter_split(df1, df2)
    >>> intersecting
                      geometry
    0  POINT (0.00000 0.00000)
    >>> not_intersecting
                      geometry
    1  POINT (0.00000 1.00000)

    Equivelent to sjoin-ing and selecting based on index (which requires the
    index to be unique).

    >>> df1 = df1.reset_index(drop=True)
    >>> joined = df1.sjoin(df2)
    >>> intersecting = df1.loc[df1.index.isin(joined.index)]
    >>> not_intersecting = df1.loc[~df1.index.isin(joined.index)]

    Also equivelent to using the intersects method, which
    is often slower since df2 must be dissolved:

    >>> filt = df1.intersects(df2.union_all())
    >>> intersecting = df1.loc[filt]
    >>> not_intersecting = df1.loc[~filt]

    """
    if not len(gdf):
        return gdf, gdf

    if not isinstance(gdf, (GeoDataFrame | GeoSeries)):
        raise TypeError(gdf_type_error_message)

    other = _sfilter_checks(other, crs=gdf.crs)

    indices = _get_sfilter_indices(
        gdf, other, predicate, distance, n_jobs, rtree_runner
    )

    return (
        gdf.iloc[indices],
        gdf.iloc[pd.Index(range(len(gdf))).difference(pd.Index(indices))],
    )


def sfilter_inverse(
    gdf: GeoDataFrame | GeoSeries,
    other: GeoDataFrame | GeoSeries | Geometry,
    predicate: str = "intersects",
    distance: int | float | None = None,
    n_jobs: int = 1,
    rtree_runner: RTreeQueryRunner | None = None,
) -> GeoDataFrame | GeoSeries:
    """Filter a GeoDataFrame or GeoSeries by inverse spatial predicate.

    Returns the rows that do not match the spatial predicate.

    Args:
        gdf: The GeoDataFrame or GeoSeries.
        other: The geometry object to filter 'gdf' by.
        predicate: Spatial predicate to use. Defaults to 'intersects'.
        distance: Max distance to allow if predicate=="dwithin".
        n_jobs: Number of workers.
        rtree_runner: Optionally debug/manipulate the spatial indexing operations.
            See the 'runners' module for example implementations.

    Returns:
        A copy of 'gdf' with only the rows that do not match the
        spatial predicate with 'other'.

    Examples:
    ---------
    >>> import sgis as sg
    >>> df1 = sg.to_gdf([(0, 0), (0, 1)])
    >>> df1
                      geometry
    0  POINT (0.00000 0.00000)
    1  POINT (0.00000 1.00000)
    >>> df2 = sg.to_gdf([(0, 0), (1, 2)])
    >>> df2
                      geometry
    0  POINT (0.00000 0.00000)
    1  POINT (1.00000 2.00000)

    Keep the rows in df1 that do not intersect a geometry in df2.

    >>> not_intersecting = sg.sfilter_inverse(df1, df2)
    >>> not_intersecting
                      geometry
    1  POINT (0.00000 1.00000)

    Equivelent to sjoin-ing and selecting based on index (which requires the
    index to be unique).

    >>> df1 = df1.reset_index(drop=True)
    >>> joined = df1.sjoin(df2)
    >>> not_intersecting = df1.loc[~df1.index.isin(joined.index)]

    Also equivelent to using the intersects method, which
    is often slower since df2 must be dissolved:

    >>> not_intersecting = df1.loc[~df1.intersects(df2.union_all())]

    """
    if not isinstance(gdf, (GeoDataFrame | GeoSeries)):
        raise TypeError(gdf_type_error_message)
    if not len(gdf):
        return gdf

    other = _sfilter_checks(other, crs=gdf.crs)
    indices = _get_sfilter_indices(
        gdf, other, predicate, distance, n_jobs, rtree_runner
    )
    return gdf.iloc[pd.Index(range(len(gdf))).difference(pd.Index(indices))]


def _sfilter_checks(other, crs):
    """Allow 'other' to be any geometry object."""
    if isinstance(other, GeoSeries):
        return other

    if not isinstance(other, GeoDataFrame):
        try:
            other = to_gdf(other)
        except TypeError as e:
            raise TypeError(
                f"Unexpected type of 'other' {other.__class__.__name__}"
            ) from e

        if crs is None:
            return other

        try:
            other = other.set_crs(crs)
        except ValueError as e:
            raise ValueError("crs mismatch", crs, other.crs) from e

    return other


def _get_sfilter_indices(
    left: GeoDataFrame | GeoSeries,
    right: GeoDataFrame | GeoSeries | Geometry,
    predicate: str,
    distance: int | float | None,
    n_jobs: int,
    rtree_runner: RTreeQueryRunner | None,
) -> np.ndarray:
    """Compute geometric comparisons and get matching indices.

    Taken from:
    geopandas.tools.sjoin._geom_predicate_query

    Parameters
    ----------
    left : GeoDataFrame
    right : GeoDataFrame
    predicate : string
        Binary predicate to query.
    n_jobs: Number of workers.
    rtree_runner: Optionally debug/manipulate the spatial indexing operations.
        See the 'runners' module for example implementations.

    Returns:
    -------
    DataFrame
        DataFrame with matching indices in
        columns named `_key_left` and `_key_right`.
    """
    original_predicate = predicate

    if rtree_runner is None:
        rtree_runner = _get_instance(config, "rtree_runner", n_jobs=n_jobs)

    with warnings.catch_warnings():
        # We don't need to show our own warning here
        # TODO remove this once the deprecation has been enforced
        warnings.filterwarnings(
            "ignore", "Generated spatial index is empty", FutureWarning
        )

        if predicate == "within":
            # within is implemented as the inverse of contains
            # contains is a faster predicate
            # see discussion at https://github.com/geopandas/geopandas/pull/1421
            predicate = "contains"
            arr1 = right.geometry.values
            arr2 = left.geometry.values
        else:
            # all other predicates are symmetric
            # keep them the same
            arr1 = left.geometry.values
            arr2 = right.geometry.values

    left, right = rtree_runner.run(arr1, arr2, predicate=predicate, distance=distance)

    if original_predicate == "within":
        return np.sort(np.unique(right))
    return np.sort(np.unique(left))

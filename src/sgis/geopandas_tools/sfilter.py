import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.tools.sjoin import _geom_predicate_query
from shapely import Geometry

from .conversion import to_gdf


gdf_type_error_message = "'gdf' should be of type GeoDataFrame."


def sfilter(
    gdf: GeoDataFrame,
    other: GeoDataFrame | GeoSeries | Geometry,
    predicate: str = "intersects",
) -> GeoDataFrame:
    """Filter a GeoDataFrame by spatial predicate.

    Does an sjoin and returns the rows of 'gdf' that were returned
    without getting duplicates or columns from 'other'.
    Works with unique and non-unique index.

    Like 'select by location' in ArcGIS/QGIS, except that the
    selection is permanent.

    Args:
        gdf: The GeoDataFrame.
        other: The geometry object to filter 'gdf' by.
        predicate: Spatial predicate to use. Defaults to 'intersects'.

    Returns:
        A copy of 'gdf' with only the rows matching the
        spatial predicate with 'other'.

    Examples
    --------

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

    >>> df1.loc[df1.intersects(df2.unary_union)]
                      geometry
    0  POINT (0.00000 0.00000)

    """
    if not isinstance(gdf, GeoDataFrame):
        raise TypeError(gdf_type_error_message)

    other = _sfilter_checks(other, crs=gdf.crs)

    indices = _get_sfilter_indices(gdf, other, predicate)

    return gdf.iloc[indices]


def sfilter_split(
    gdf: GeoDataFrame,
    other: GeoDataFrame | GeoSeries | Geometry,
    predicate: str = "intersects",
) -> tuple[GeoDataFrame, GeoDataFrame]:
    """Split a GeoDataFrame by spatial predicate.

    Like sfilter, but returns both the rows that do and do not match
    the spatial predicate as separate GeoDataFrames.

    Args:
        gdf: The GeoDataFrame.
        other: The geometry object to filter 'gdf' by.
        predicate: Spatial predicate to use. Defaults to 'intersects'.

    Returns:
        A tuple of GeoDataFrames, one with the rows that match the spatial predicate
        and one with the rows that do not.

    Examples
    --------

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

    >>> filt = df1.intersects(df2.unary_union)
    >>> intersecting = df1.loc[filt]
    >>> not_intersecting = df1.loc[~filt]

    """
    if not isinstance(gdf, GeoDataFrame):
        raise TypeError(gdf_type_error_message)

    other = _sfilter_checks(other, crs=gdf.crs)

    indices = _get_sfilter_indices(gdf, other, predicate)

    return (
        gdf.iloc[indices],
        gdf.iloc[pd.Index(range(len(gdf))).difference(indices)],
    )


def sfilter_inverse(
    gdf: GeoDataFrame,
    other: GeoDataFrame | GeoSeries | Geometry,
    predicate: str = "intersects",
) -> tuple[GeoDataFrame, GeoDataFrame]:
    """Filter a GeoDataFrame by inverse spatial predicate.

    Returns the rows that do not match the spatial predicate.

    Args:
        gdf: The GeoDataFrame.
        other: The geometry object to filter 'gdf' by.
        predicate: Spatial predicate to use. Defaults to 'intersects'.

    Returns:
        A copy of 'gdf' with only the rows that do not match the
        spatial predicate with 'other'.

    Examples
    --------

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

    >>> not_intersecting = df1.loc[~df1.intersects(df2.unary_union)]

    """
    if not isinstance(gdf, GeoDataFrame):
        raise TypeError(gdf_type_error_message)

    other = _sfilter_checks(other, crs=gdf.crs)

    indices = _get_sfilter_indices(gdf, other, predicate)

    return gdf.iloc[pd.Index(range(len(gdf))).difference(indices)]


def _sfilter_checks(other, crs):
    """Allow 'other' to be any geometry object."""
    if isinstance(other, GeoSeries):
        other = other.to_frame()

    elif not isinstance(other, GeoDataFrame):
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


def _get_sfilter_indices(gdf, other, predicate) -> np.ndarray:
    idx_df = _geom_predicate_query(gdf, other, predicate=predicate)
    return idx_df["_key_left"].unique()

import numpy as np
import pandas as pd
from geopandas import GeoSeries
from shapely import make_valid
from shapely import union_all

from .geometry_types import to_single_geom_type


def _unary_union_for_notna(geoms, **kwargs):
    try:
        return make_valid(union_all(geoms, **kwargs))
    except TypeError:
        return make_valid(union_all([geom for geom in geoms.dropna().values], **kwargs))


def make_valid_and_keep_geom_type(geoms: np.ndarray, geom_type: str) -> GeoSeries:
    """Make GeometryCollections into (Multi)Polygons, (Multi)LineStrings or (Multi)Points.

    Because GeometryCollections might appear after dissolving (union_all).
    And this makes shapely difference/intersection fail.

    Args:
        geoms: Array of geometries.
        geom_type: geometry type to be kept.
    """
    geoms = GeoSeries(geoms)
    geoms.index = range(len(geoms))
    geoms.loc[:] = make_valid(geoms.to_numpy())
    geoms_with_correct_type = geoms.explode(index_parts=False).pipe(
        to_single_geom_type, geom_type
    )
    only_one = geoms_with_correct_type.groupby(level=0).transform("size") == 1
    one_hit = geoms_with_correct_type[only_one]
    many_hits = geoms_with_correct_type[~only_one].groupby(level=0).agg(union_all)
    geoms_with_wrong_type = geoms.loc[~geoms.index.isin(geoms_with_correct_type.index)]
    return pd.concat([one_hit, many_hits, geoms_with_wrong_type]).sort_index()

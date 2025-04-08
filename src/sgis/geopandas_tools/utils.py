from shapely import make_valid
from shapely import union_all


def _unary_union_for_notna(geoms, **kwargs):
    try:
        return make_valid(union_all(geoms, **kwargs))
    except TypeError:
        return make_valid(union_all([geom for geom in geoms.dropna().values], **kwargs))

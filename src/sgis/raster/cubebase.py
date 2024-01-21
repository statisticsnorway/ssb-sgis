# from .merge import cube_merge


# def intersection_base(row: pd.Series, cube, **kwargs):
#     cube = cube.copy()
#     geom = row.pop("geometry")
#     cube = cube.clip(geom, **kwargs)
#     for key, value in row.items():
#         cube.df[key] = value
#     return cube


# def _cube_merge(cubebounds, **kwargs):
#     assert isinstance(cubebounds, dict)
#     return cube_merge(cube=cubebounds["cube"], bounds=cubebounds["bounds"], **kwargs)


def _method_as_func(self, method, **kwargs):
    return getattr(self, method)(**kwargs)


# def _astype_raster(raster, raster_type):
#     """Returns raster as another raster type."""
#     return raster_type(raster)


def _raster_from_path(path, raster_type, indexes, res, **kwargs):
    return raster_type.from_path(path, indexes=indexes, res=res, **kwargs)


def _from_gdf_func(gdf, raster_type, **kwargs):
    return raster_type.from_gdf(gdf, **kwargs)


# def _to_gdf_func(raster, **kwargs):
#     return raster.to_gdf(**kwargs)


def _write_func(raster, path, **kwargs):
    raster.write(path, **kwargs)
    return raster


# def _clip_func(raster, mask, **kwargs):
#     return raster.clip(mask, **kwargs)


# def _load_func(raster, **kwargs):
#     return raster.load(**kwargs)


def _to_crs_func(raster, **kwargs):
    return raster.to_crs(**kwargs)


def _set_crs_func(raster, **kwargs):
    return raster.set_crs(**kwargs)


# def _array_astype_func(array, dtype):
#     return array.astype(dtype)


def _add(raster, scalar):
    return raster + scalar


def _mul(raster, scalar):
    return raster * scalar


def _sub(raster, scalar):
    return raster - scalar


def _truediv(raster, scalar):
    return raster / scalar


def _floordiv(raster, scalar):
    return raster // scalar


def _pow(raster, scalar):
    return raster**scalar

def _method_as_func(self, method, **kwargs):
    return getattr(self, method)(**kwargs)


def _raster_from_path(path, raster_type, res, **kwargs):
    return raster_type.from_path(path, res=res, **kwargs)


def _from_gdf_func(gdf, raster_type, **kwargs):
    return raster_type.from_gdf(gdf, **kwargs)


def _write_func(raster, path, **kwargs):
    raster.write(path, **kwargs)
    return raster


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

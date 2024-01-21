"""Method-to-function to use as mapping function."""
from pathlib import Path


def _cube_merge(cubebounds, **kwargs):
    assert isinstance(cubebounds, dict)
    return cube_merge(cube=cubebounds["cube"], bounds=cubebounds["bounds"], **kwargs)


def _method_as_func(self, method, **kwargs):
    return getattr(self, method)(**kwargs)


def _astype_raster(raster, raster_type):
    """Returns raster as another raster type."""
    return raster_type(raster)


def _raster_from_path(path, raster_type, band_index, **kwargs):
    return raster_type.from_path(path, band_index=band_index, **kwargs)


def _from_gdf_func(gdf, raster_type, **kwargs):
    return raster_type.from_gdf(gdf, **kwargs)


def _to_gdf_func(raster, **kwargs):
    return raster.to_gdf(**kwargs)


def _write_func(raster, folder, **kwargs):
    path = str(Path(folder) / Path(raster.name).stem) + ".tif"
    raster.write(path, **kwargs)
    raster.path = path
    return raster


def _clip_func(raster, mask, **kwargs):
    return raster.clip(mask, **kwargs)


def _clip_func(raster, mask, **kwargs):
    return raster.clip(mask, **kwargs)


def _load_func(raster, **kwargs):
    return raster.load(**kwargs)


def _zonal_func(raster, **kwargs):
    return raster.zonal(**kwargs)


def _to_crs_func(raster, **kwargs):
    return raster.to_crs(**kwargs)


def _set_crs_func(raster, **kwargs):
    return raster.set_crs(**kwargs)


def _array_astype_func(array, dtype):
    return array.astype(dtype)


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


def _clip_base(cube, mask):
    if (
        hasattr(mask, "crs")
        and mask.crs
        and not pyproj.CRS(cube.crs).equals(pyproj.CRS(mask.crs))
    ):
        raise ValueError("crs mismatch.")

    # first remove rows not within mask
    cube._df = cube._df.loc[cube.boxes.intersects(to_shapely(mask))]

    return cube


def _write_base(cube, subfolder_col, root):
    if cube.df["name"].isna().any():
        raise ValueError("Cannot have missing values in 'name' column when writing.")

    if cube.df["name"].duplicated().any():
        raise ValueError("Cannot have duplicate names when writing files.")

    cube.validate_cube_df(cube.df)

    if subfolder_col:
        folders = [
            Path(root) / subfolder if subfolder else Path(root)
            for subfolder in cube.df[subfolder_col]
        ]
    else:
        folders = [Path(root) for _ in cube]

    rasters = list(cube)
    args = [item for item in zip(rasters, folders)]

    return cube, args

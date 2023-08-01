import numpy as np
import shapely
from pandas import DataFrame
from pandas.api.types import is_list_like
from rasterio import merge
from rioxarray.merge import merge_arrays

from ..geopandas_tools.bounds import get_total_bounds, to_bbox
from ..geopandas_tools.general import get_common_crs, to_shapely
from .explode import explode_cube_df


def merge_by_bounds(
    cube,
    by=None,
    bounds=None,
    res=None,
    aggfunc="first",
    **kwargs,
):
    cube._df["tile"] = cube.tile.values

    if isinstance(by, str):
        by = [by, "tile"]
    elif by is None:
        by = ["tile"]
    elif not is_list_like(by):
        raise TypeError("'by' should be string or list like.", by)
    else:
        by = by + ["tile"]

    for col in by:
        if col in cube.df:
            continue
        try:
            cube._df[col] = cube.raster_attribute(col).values
        except AttributeError:
            pass

    has_arrays = cube.arrays.notna().all()
    can_merge_arrays = res is None and bounds is None
    if has_arrays and can_merge_arrays:
        return merge_arrays_by_bounds(cube, by, aggfunc)

    return cube_merge(
        cube,
        bounds=bounds,
        by=by,
        res=res,
        aggfunc=aggfunc,
        _as_3d=True,
        **kwargs,
    )


def merge_arrays_by_bounds(cube, by, aggfunc):
    unique = cube.df[by].drop_duplicates(by).set_index(by)

    unique["raster"] = cube.df.groupby(by).apply(merge_within_bounds)

    remaining_cols = cube._df.columns.difference(by + ["raster"])
    unique[remaining_cols] = cube._df.groupby(by)[remaining_cols].agg(aggfunc)

    cube.df = unique.reset_index()

    cube._update_df()

    return cube


def merge_within_bounds(cube_df):
    res = {r.res for r in cube_df["raster"]}
    if len(res) != 1:
        raise ValueError("Resolution mismatch.")

    raster_types = {r.__class__ for r in cube_df["raster"]}
    raster_type = list(raster_types)[0]

    arrays = []
    for raster in cube_df["raster"]:
        for arr in raster.array_list():
            arrays += [arr]
    merged = np.array(arrays)

    crs = get_common_crs([r.crs for r in cube_df["raster"]])
    bounds = list({tuple(r.bounds) for r in cube_df["raster"]})
    assert len(bounds) == 1

    return raster_type.from_array(merged, crs=crs, bounds=bounds[0])


def cube_merge(
    cube,
    bounds=None,
    by: str | list[str] | None = None,
    res=None,
    aggfunc="first",
    dropna: bool = False,
    **kwargs,
):
    temp_cols = ["minx", "miny", "maxx", "maxy", "res"]

    if bounds is None:
        cube._df[["minx", "miny", "maxx", "maxy"]] = cube.bounds.values
    else:
        bounds = to_bbox(bounds)
        cube._df = cube._df[cube.boxes.intersects(to_shapely(bounds))]

    if not len(cube):
        return cube

    raster_type = cube.most_common_raster_type()

    if res is None:
        cube._df["res"] = cube.res

    if isinstance(by, str):
        by = [by]
    if by is not None and not is_list_like(by):
        raise TypeError("'by' should be string or list like.", by)

    kwargs = {
        "bounds": bounds,
        "res": res,
        "raster_type": raster_type,
        "crs": cube.crs,
        "nodata": cube.nodata.iloc[0],
        **kwargs,
    }

    if by is None:
        raster = _grouped_merge(cube._df, **kwargs)
        cube._df = cube._df.iloc[[0]].drop(columns=temp_cols, errors="ignore")
        cube._df["raster"] = raster
        return cube

    unique = cube._df[by + ["raster"]].drop_duplicates(by).set_index(by)

    if len(unique) == len(cube._df):
        # no merging is needed
        cube._df = cube.df.drop(columns=temp_cols, errors="ignore")
        if bounds is not None:
            return cube.clip(bounds, res=res)
        elif cube.arrays.isna().all():
            return cube.load(res=res)
        elif res is None:
            return cube
        # continue from here if arrays are loaded and res is specified

    unique["raster"] = cube.df.groupby(by, dropna=dropna).apply(
        lambda x: _grouped_merge(x, **kwargs)
    )

    remaining_cols = cube._df.columns.difference(by + ["raster"])
    unique[remaining_cols] = cube._df.groupby(by)[remaining_cols].agg(aggfunc)

    cube._df = unique.reset_index().drop(columns=temp_cols, errors="ignore")
    cube._update_df()

    return cube


def _grouped_merge(
    group: DataFrame, bounds, raster_type, res, crs, nodata, _as_3d=False, **kwargs
):
    if res is None:
        res = group["res"].min()
    if bounds is None:
        bounds = (
            group["minx"].min(),
            group["miny"].min(),
            group["maxx"].max(),
            group["maxy"].max(),
        )

    is_3d = max(len(r.shape) for r in group["raster"]) == 3

    exploded = explode_cube_df(group)
    arrays = []
    for idx in sorted(exploded["band_index"].unique()):
        rasters = exploded.loc[exploded["band_index"] == idx, "raster"]
        if all(r.array is not None for r in rasters):
            # skip if geometries only touch to avoid Invalid dataset dimensions : 0 x n
            total_bounds = get_total_bounds(r.bounds for r in rasters)
            if shapely.box(*bounds).touches(shapely.box(*total_bounds)):
                continue

            xarrays = [r.to_xarray().transpose("y", "x") for r in rasters]
            merged = merge_arrays(
                xarrays, bounds=bounds, res=res, nodata=nodata, **kwargs
            )
            array = merged.to_numpy()

        else:
            datasets = [r._load_warp_file() for r in rasters]

            array, _ = merge.merge(
                datasets, indexes=(idx,), bounds=bounds, res=res, **kwargs
            )
            assert len(array.shape) == 3, array.shape

        if len(array.shape) == 3:
            array = array[0]

        arrays.append(array)

    if not arrays:
        return raster_type()

    array = np.array(arrays)
    assert len(array.shape) == 3, array.shape
    # if not _as_3d and array.shape[0] == 1:
    if not is_3d and array.shape[0] == 1:
        array = array[0]

    return raster_type.from_array(
        array,
        bounds=bounds,
        crs=crs,
    )

import numpy as np
from pandas import NA, DataFrame
from pandas.api.types import is_list_like
from rasterio import merge

from ..geopandas_tools.bounds import to_bbox
from ..geopandas_tools.general import get_common_crs, to_shapely
from .raster import Raster, get_numpy_func


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

    if cube.arrays.isna().all():
        return cube_merge(
            cube,
            bounds=bounds,
            by=by,
            res=res,
            aggfunc=aggfunc,
            _as_3d=True,
            **kwargs,
        )

    if res is not None:
        raise ValueError
    if bounds is not None:
        raise ValueError

    unique = cube.df[by].drop_duplicates(by).set_index(by)

    unique["raster"] = cube.df.groupby(by).apply(merge_arrays)

    remaining_cols = cube._df.columns.difference(by + ["raster"])
    unique[remaining_cols] = cube._df.groupby(by)[remaining_cols].agg(aggfunc)

    cube.df = unique.reset_index()

    cube.update_df()

    return cube

    cube._df["name"] = [r.name for r in cube.df["raster"]]
    cube._df["band_name"] = [r.band_name for r in cube.df["raster"]]
    cube._df["band_index"] = [r.band_index for r in cube.df["raster"]]
    display(cube.df)

    return cube


def merge_arrays(cube_df):
    res = {r.res for r in cube_df["raster"]}
    if len(res) != 1:
        raise ValueError("Resolution mismatch.")

    arrays = []
    for raster in cube_df["raster"]:
        for arr in raster.array_list:
            arrays += [arr]
    merged = np.array(arrays)

    crs = get_common_crs([r.crs for r in cube_df["raster"]])
    bounds = list({tuple(r.bounds) for r in cube_df["raster"]})
    assert len(bounds) == 1

    return Raster.from_array(merged, crs=crs, bounds=bounds[0])


def cube_merge(
    cube,
    bounds=None,
    by: str | list[str] | None = None,
    res=None,
    aggfunc="first",
    copy: bool = True,
    **kwargs,
):
    if copy:
        cube = cube.copy()

    if bounds is None:
        cube._df[["minx", "miny", "maxx", "maxy"]] = cube.bounds.values
    else:
        bounds = to_bbox(bounds)
        cube._df = cube._df[cube.boxes.intersects(to_shapely(bounds))]

    raster_type = cube.most_common_raster_type()

    if res is None:
        cube._df["res"] = cube.res

    if isinstance(by, str):
        by = [by]
    if by is not None and not is_list_like(by):
        raise TypeError("'by' should be string or list like.", by)

    if by is None:
        raster = _grouped_merge(
            cube._df,
            bounds=bounds,
            res=res,
            raster_type=raster_type,
            crs=cube.crs,
            **kwargs,
        )
        cube._df = cube._df.iloc[[0]]
        cube._df["raster"] = raster
        return cube

    unique = cube._df[by + ["raster"]].drop_duplicates(by).set_index(by)

    if len(unique) == len(cube._df):
        if bounds is not None:
            return cube.clip(bounds, res=res)
        else:
            return cube.load(res=res)

    unique["raster"] = cube.df.groupby(by).apply(
        lambda x: _grouped_merge(
            x,
            bounds=bounds,
            res=res,
            raster_type=raster_type,
            crs=cube.crs,
            **kwargs,
        )
    )

    remaining_cols = cube._df.columns.difference(by + ["raster"])
    unique[remaining_cols] = cube._df.groupby(by)[remaining_cols].agg(aggfunc)

    cube._df = unique.reset_index().drop(
        ["minx", "miny", "maxx", "maxy", "res"], axis=1, errors="ignore"
    )
    # cube._df["path"] = NA
    # cube._df["name"] = [r.name for r in cube.df["raster"]]
    # [
    #   f"{int(minx)}_{int(miny)}_{name}"
    #  for minx, miny, name in zip(cube.minx, cube.miny, cube.band_name)
    # ]
    cube.update_df()

    return cube


def _grouped_merge(
    group: DataFrame, bounds, raster_type, res, crs, _as_3d=False, **kwargs
):
    # if res is None:
    #   res = group["res"].min()
    if bounds is None:
        bounds = (
            group["minx"].min(),
            group["miny"].min(),
            group["maxx"].max(),
            group["maxy"].max(),
        )
    exploded = group.explode(column="band_index")
    band_index = tuple(exploded["band_index"].sort_values().unique())
    arrays = []
    for idx in band_index:
        paths = exploded.loc[exploded["band_index"] == idx, "path"]

        array, transform = merge.merge(
            list(paths), indexes=(idx,), bounds=bounds, res=res, **kwargs
        )
        # merge doesn't allow single index (numpy error), so changing afterwards
        if len(array.shape) == 3:
            assert array.shape[0] == 1
            array = array[0]
        arrays.append(array)
    array = np.array(arrays)
    assert len(array.shape) == 3
    if not _as_3d and array.shape[0] == 1:
        array = array[0]

    return raster_type.from_array(
        array,
        transform=transform,
        crs=crs,
    )

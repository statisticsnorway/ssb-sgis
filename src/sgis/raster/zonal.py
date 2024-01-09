from collections.abc import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from ..helpers import get_non_numpy_func_name, get_numpy_func


def prepare_zonal(polygons: gpd.GeoDataFrame, aggfunc):
    polygons = polygons.reset_index(drop=True)[["geometry"]]

    if isinstance(aggfunc, str) or callable(aggfunc):
        aggfunc = [aggfunc]

    aggfunc = [f if callable(f) else get_numpy_func(f) for f in aggfunc]

    func_names = [get_non_numpy_func_name(f) for f in aggfunc]

    return polygons, aggfunc, func_names


def make_geometry_iterrows(gdf):
    """Because pandas iterrows returns non-geo Series"""
    return list(gdf.geometry.items())


def zonal_func(
    poly_iter: tuple[int, Polygon],
    cube,
    array_func: Callable,
    aggfunc,
    func_names,
    by_date,
) -> pd.DataFrame:
    cube = cube.copy()
    i, polygon = poly_iter
    if not by_date or cube.df["date"].isna().all():
        df = _clip_and_aggregate(
            cube, polygon, array_func, aggfunc, func_names, date=None, i=i
        )
        return df if by_date else df.drop(columns="date")

    out = []

    na_date = cube.query("date.isna()")
    df = _clip_and_aggregate(
        na_date, polygon, array_func, aggfunc, func_names, date=pd.NA, i=i
    )
    out.append(df)

    cube.df = cube.df[lambda x: x["date"].notna()]

    for dt in cube.df["date"].unique():
        cube_date = cube.query(f"date == {dt}")
        df = _clip_and_aggregate(
            cube_date, polygon, array_func, aggfunc, func_names, dt, i
        )
        out.append(df)

    return pd.concat(out)


def _no_overlap_df(func_names, i, date):
    df = pd.DataFrame(columns=func_names, index=[i])
    df["date"] = date
    df["_no_overlap"] = 1
    return df


def _clip_and_aggregate(cube, polygon, array_func, aggfunc, func_names, date, i):
    if not len(cube):
        return _no_overlap_df(func_names, i, date)
    clipped = cube.clipmerge(polygon)
    if not len(clipped) or clipped.arrays.isna().all():
        return _no_overlap_df(func_names, i, date)
    assert len(clipped) == 1
    array = clipped[0].array
    df = _aggregate(array, array_func, aggfunc, func_names, date, i)
    return df


def _aggregate(array, array_func, aggfunc, func_names, date, i):
    if array_func:
        array = array_func(array)
    # flat_array = array.astype(np.float64).flatten()
    flat_array = array.flatten()
    no_nans = flat_array[~np.isnan(flat_array)]
    data = {}
    for f, name in zip(aggfunc, func_names, strict=True):
        num = f(no_nans)
        data[name] = num
    df = pd.DataFrame(data, index=[i])
    df["date"] = date
    df["_no_overlap"] = 0
    return df


def zonal_post(
    aggregated: list[pd.DataFrame], polygons, idx_mapper, idx_name, dropna
) -> pd.DataFrame:
    out = gpd.GeoDataFrame(
        pd.concat(aggregated), geometry=polygons.geometry.values, crs=polygons.crs
    ).sort_index()

    out.index = out.index.map(idx_mapper)
    out.index.name = idx_name

    if dropna:
        out = out.loc[~out.drop(columns="geometry").isna().all(axis=1)]
        return out.loc[out["_no_overlap"] != 1].drop(columns="_no_overlap")
    else:
        return out.drop(columns="_no_overlap")

from pathlib import Path
from typing import Callable

import pandas as pd
from dapla import write_pandas
from geopandas import GeoDataFrame
from pandas import DataFrame

from ..geopandas_tools.general import clean_clip, clean_geoms
from ..geopandas_tools.neighbors import get_neighbor_indices
from .dapla import read_geopandas, write_geopandas


def write_municipality_data(
    data: str | GeoDataFrame | DataFrame,
    out_folder: str,
    with_neighbors: bool = False,
    municipalities: GeoDataFrame | None = None,
    muni_number_col: str = "KOMMUNENR",
    file_type: str = "parquet",
    func: Callable | None = None,
    write_empty: bool = False,
) -> None:
    write_func = (
        _write_neighbor_municipality_data
        if with_neighbors
        else _write_municipality_data
    )

    return write_func(
        data=data,
        out_folder=out_folder,
        municipalities=municipalities,
        muni_number_col=muni_number_col,
        file_type=file_type,
        func=func,
        write_empty=write_empty,
    )


'''def write_municipality_data(
    in_data: dict[str, str | GeoDataFrame],
    out_data: str | dict[str, str],
    *,
    municipalities: GeoDataFrame | None = None,
    n_jobs: int,
    with_neighbors: bool = False,
    funcdict: dict[str, Callable] | None = None,
    file_type: str = "parquet",
    muni_number_col: str = "KOMMUNENR",
    strict: bool = False,
    write_empty: bool = False,
):
    """Split one or more datasets into municipalities and write as separate files.

    Optionally with neighbor municipalities included.

    The files will be named as the municipality number.

    Args:
        in_data: Dictionary with dataset names as keys and file paths or
            (Geo)DataFrames as values.
        out_data: Either a single folder path or a dictionary with same keys as
            'in_data' and folder paths as values. If a single folder is given,
            the 'in_data' keys will be used as subfolders.
        municipalities: GeoDataFrame of municipality polygons.
        n_jobs: Number of parallel workers.
        with_neighbors: If True (not default), each municipality file will include
            data from all municipalities they share a border with.
        funcdict: Dictionary with the keys of 'in_data' and functions as values.
            The functions should take a GeoDataFrame as input and return a
            GeoDataFrame.
        file_type: Defaults to parquet.
        muni_number_col: Column name that holds the municipality number. Defaults
            to KOMMUNENR.
        strict: If False (default), the dictionaries 'out_data' and 'funcdict' does
            not have to have the same length as 'in_data'.
        write_empty: If False (default), municipalities with no data will be skipped.
            If True, an empty parquet file will be written.

    """

    shared_kwds = {
        "municipalities": municipalities,
        "muni_number_col": muni_number_col,
        "file_type": file_type,
        "write_empty": write_empty,
    }

    if isinstance(in_data, (str, Path)):
        in_data = {Path(in_data).stem: in_data}

    if not isinstance(in_data, dict):
        raise TypeError(
            "'in_data' should be a dict of names: paths or a single file path."
        )

    if isinstance(out_data, (str, Path)):
        out_data = {name: Path(out_data) / name for name in in_data}

    if funcdict is None:
        funcdict = {}

    zip_func = dict_zip if strict else dict_zip_union

    write_func = (
        _write_neighbor_municipality_data
        if with_neighbors
        else _write_municipality_data
    )

    funcs = []
    for _, data, folder, postfunc in zip_func(in_data, out_data, funcdict):
        if data is None:
            continue
        all_kwds = shared_kwds | {
            "data": data,
            "func": postfunc,
            "out_folder": folder,
        }
        partial_func = functools.partial(write_func, **all_kwds)
        funcs.append(partial_func)

    n_jobs = min(len(funcs), n_jobs)

    if n_jobs > 1:
        Parallel(n_jobs=n_jobs)(delayed(func)() for func in funcs)
    else:
        [func() for func in funcs]
'''


def _validate_data(data: str | list[str]) -> str:
    if isinstance(data, (str, Path)):
        return data
    if hasattr(data, "__iter__") and len(data) == 1:
        return data[0]
    elif not isinstance(data, GeoDataFrame):
        raise TypeError("'data' Must be a file path or a GeoDataFrame. Got", type(data))


def _get_out_path(out_folder, muni, file_type):
    return str(Path(out_folder) / f"{muni}.{file_type.strip('.')}")


def _write_municipality_data(
    data: str | GeoDataFrame | DataFrame,
    out_folder: str,
    municipalities: GeoDataFrame,
    muni_number_col: str = "KOMMUNENR",
    file_type: str = "parquet",
    func: Callable | None = None,
    write_empty: bool = False,
) -> None:
    data = _validate_data(data)

    if isinstance(data, (str, Path)):
        gdf = read_geopandas(str(data))

    gdf = _fix_missing_muni_numbers(gdf, municipalities, muni_number_col)

    for muni in municipalities[muni_number_col]:
        out = _get_out_path(out_folder, muni, file_type)

        gdf_muni = gdf.loc[gdf[muni_number_col] == muni]

        if not len(gdf_muni):
            if write_empty:
                gdf_muni = gdf_muni.drop(columns="geometry")
                gdf_muni["geometry"] = None
                write_pandas(gdf_muni, out)
            continue

        if func is not None:
            gdf_muni = func(gdf_muni)

        if not len(gdf_muni):
            if write_empty:
                gdf_muni = gdf_muni.drop(columns="geometry")
                gdf_muni["geometry"] = None
                write_pandas(gdf_muni, out)
            continue

        write_geopandas(gdf_muni, out)


def _write_neighbor_municipality_data(
    data: str | GeoDataFrame | DataFrame,
    out_folder: str,
    municipalities: GeoDataFrame,
    muni_number_col: str = "KOMMUNENR",
    file_type: str = "parquet",
    func: Callable | None = None,
    write_empty: bool = False,
) -> None:
    data = _validate_data(data)

    if isinstance(data, (str, Path)):
        gdf = read_geopandas(str(data))

    gdf = _fix_missing_muni_numbers(gdf, municipalities, muni_number_col)

    if municipalities.index.name != muni_number_col:
        municipalities = municipalities.set_index(muni_number_col)

    neighbor_munis = get_neighbor_indices(
        municipalities, municipalities, max_distance=1
    )

    for muni in municipalities.index:
        out = _get_out_path(out_folder, muni, file_type)

        muni_and_neighbors = neighbor_munis.loc[[muni]]
        gdf_neighbor = gdf.loc[gdf[muni_number_col].isin(muni_and_neighbors)]

        if not len(gdf_neighbor):
            if write_empty:
                gdf_neighbor["geometry"] = gdf_neighbor["geometry"].astype(str)
                write_pandas(gdf_neighbor, out)
            continue

        if func is not None:
            gdf_neighbor = func(gdf_neighbor)

        write_geopandas(gdf_neighbor, out)


def _fix_missing_muni_numbers(gdf, municipalities, muni_number_col):
    if muni_number_col in gdf and gdf[muni_number_col].notna().all():
        return gdf

    def _clean_overlay(df1, df2):
        return df1.pipe(clean_geoms).overlay(df2, how="intersection").pipe(clean_geoms)

    def _clean_clip(df1, df2, muni_number_col):
        """Looping clip for large datasets because it's faster and safer."""
        all_clipped = []
        for muni in df2[muni_number_col]:
            clipped = clean_clip(df1, df2[df2[muni_number_col] == muni])
            clipped[muni_number_col] = muni
            all_clipped.append(clipped)
        return pd.concat(all_clipped)

    municipalities = municipalities[[muni_number_col, "geometry"]].to_crs(gdf.crs)

    if muni_number_col in gdf and gdf[muni_number_col].isna().any():
        notna = gdf[gdf[muni_number_col].notna()]

        isna = gdf[gdf[muni_number_col].isna()].drop(muni_number_col, axis=1)

        if len(isna) < 10_000:
            notna_anymore = _clean_overlay(isna, municipalities)
        else:
            notna_anymore = _clean_clip(isna, municipalities, muni_number_col)

        return pd.concat([notna, notna_anymore])

    if len(gdf) < 10_000:
        return _clean_overlay(gdf, municipalities)
    else:
        return _clean_clip(gdf, municipalities, muni_number_col)

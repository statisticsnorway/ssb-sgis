from collections.abc import Callable
from pathlib import Path

import pandas as pd
from dapla import read_pandas, write_pandas
from geopandas import GeoDataFrame
from pandas import DataFrame

from ..geopandas_tools.general import clean_clip, clean_geoms
from ..geopandas_tools.neighbors import get_neighbor_indices
from .dapla_functions import read_geopandas, write_geopandas


def write_municipality_data(
    data: str | GeoDataFrame | DataFrame,
    out_folder: str,
    municipalities: GeoDataFrame,
    with_neighbors: bool = False,
    muni_number_col: str = "KOMMUNENR",
    file_type: str = "parquet",
    func: Callable | None = None,
    write_empty: bool = False,
    clip: bool = True,
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
        clip=clip,
    )


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
    clip: bool = True,
) -> None:
    data = _validate_data(data)

    if isinstance(data, (str, Path)):
        try:
            gdf = read_geopandas(str(data))
        except ValueError as e:
            try:
                gdf = read_pandas(str(data))
            except ValueError:
                raise e.__class__(e, data)
    elif isinstance(data, DataFrame):
        gdf = data
    else:
        raise TypeError(type(data))

    if func is not None:
        gdf = func(gdf)

    gdf = _fix_missing_muni_numbers(gdf, municipalities, muni_number_col, clip)

    for muni in municipalities[muni_number_col]:
        out = _get_out_path(out_folder, muni, file_type)

        gdf_muni = gdf.loc[gdf[muni_number_col] == muni]

        if not len(gdf_muni):
            if write_empty:
                gdf_muni = gdf_muni.drop(columns="geometry", errors="ignore")
                gdf_muni["geometry"] = None
                write_pandas(gdf_muni, out)
            continue

        if not len(gdf_muni):
            if write_empty:
                gdf_muni = gdf_muni.drop(columns="geometry", errors="ignore")
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
    clip: bool = True,
) -> None:
    data = _validate_data(data)

    if isinstance(data, (str, Path)):
        gdf = read_geopandas(str(data))

    if func is not None:
        gdf = func(gdf)

    gdf = _fix_missing_muni_numbers(gdf, municipalities, muni_number_col, clip)

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

        write_geopandas(gdf_neighbor, out)


def _fix_missing_muni_numbers(gdf, municipalities, muni_number_col, clip):
    if muni_number_col in gdf and gdf[muni_number_col].notna().all():
        return gdf

    if municipalities is None:
        if muni_number_col not in gdf:
            raise ValueError(
                f"Cannot find column {muni_number_col}. "
                "Specify another column or a municipality GeoDataFrame to clip "
                "the geometries by."
            )
        assert gdf[muni_number_col].isna().any()
        raise ValueError(
            f"Column {muni_number_col} has missing values. Make sure gdf has "
            "correct municipality number info or specify a municipality "
            "GeoDataFrame to clip the geometries by."
        )

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

        if not clip:
            notna_anymore = isna.sjoin(municipalities).drop(columns="index_right")
        elif len(isna) < 10_000:
            notna_anymore = _clean_overlay(isna, municipalities)
        else:
            notna_anymore = _clean_clip(isna, municipalities, muni_number_col)

        return pd.concat([notna, notna_anymore])

    if not clip:
        return gdf.sjoin(municipalities).drop(columns="index_right")
    if len(gdf) < 10_000:
        return _clean_overlay(gdf, municipalities)
    else:
        return _clean_clip(gdf, municipalities, muni_number_col)

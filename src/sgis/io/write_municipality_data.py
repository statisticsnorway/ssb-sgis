from pathlib import Path
from typing import Callable

import pandas as pd
from geopandas import GeoDataFrame
from pandas import DataFrame


def _validate_data(data: str | list[str]) -> str:
    if isinstance(data, (str, Path)):
        return data
    if hasattr(data, "__iter__") and len(data) == 1:
        return data[0]
    elif not isinstance(data, GeoDataFrame):
        raise TypeError("'data' Must be a file path or a GeoDataFrame. Got", type(data))


def _out_path(out_folder, muni, file_type):
    return Path(out_folder) / f"{muni}.{file_type.strip('.')}"


def write_municipality_data(
    data: str | GeoDataFrame | DataFrame,
    out_folder: str,
    municipalities: GeoDataFrame,
    muni_number_col: str = "KOMMUNENR",
    file_type: str = "parquet",
    func: Callable | None = None,
) -> None:
    data = _validate_data(data)

    if isinstance(data, (str, Path)):
        gdf = sg.read_geopandas(str(data))

    gdf = fix_missing_muni_numbers(gdf, municipalities, muni_number_col)

    for muni in municipalities[muni_number_col]:
        gdf_muni = gdf.loc[gdf[muni_number_col] == muni]

        if not len(gdf_muni):
            continue

        if func is not None:
            gdf_muni = func(gdf_muni)

        if not len(gdf_muni):
            continue

        out = _out_path(out_folder, muni, file_type)

        sg.write_geopandas(gdf_muni, out)


def write_neighbor_municipality_data(
    data: str | GeoDataFrame | DataFrame,
    out_folder: str,
    municipalities: GeoDataFrame,
    muni_number_col: str = "KOMMUNENR",
    file_type: str = "parquet",
    func: Callable | None = None,
) -> None:
    data = _validate_data(data)

    if isinstance(data, (str, Path)):
        gdf = sg.read_geopandas(str(data))

    gdf = fix_missing_muni_numbers(gdf, municipalities, muni_number_col)

    if municipalities.index.name != muni_number_col:
        municipalities = municipalities.set_index(muni_number_col)

    neighbor_munis = sg.get_neighbor_indices(
        municipalities, municipalities, max_distance=1
    )

    for muni in municipalities.index:
        muni_and_neighbors = neighbor_munis.loc[[muni]]
        gdf_neighbor = gdf.loc[gdf[muni_number_col].isin(muni_and_neighbors)]

        if not len(gdf_neighbor):
            continue

        if func is not None:
            gdf_neighbor = func(gdf_neighbor)

        out = _out_path(out_folder, muni, file_type)

        sg.write_geopandas(gdf_neighbor, out)


def fix_missing_muni_numbers(gdf, municipalities, muni_number_col):
    if muni_number_col in gdf and gdf[muni_number_col].notna().all():
        return gdf

    def _clean_overlay(df1, df2):
        return (
            df1.pipe(sg.clean_geoms)
            .overlay(df2, how="intersection")
            .pipe(sg.clean_geoms)
        )

    def _clean_clip(df1, df2, muni_number_col):
        """Looping clip for large datasets because it's faster and safer."""
        all_clipped = []
        for muni in df2[muni_number_col]:
            clipped = sg.clean_clip(df1, df2[df2[muni_number_col] == muni])
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

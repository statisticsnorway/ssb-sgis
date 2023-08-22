from pandas import DataFrame


def explode(cube, ignore_index: bool = False):
    cube._update_df()

    cube.df = explode_cube_df(cube.df, ignore_index=ignore_index)

    cube._update_df()

    return cube


def explode_cube_df(cube_df: DataFrame, ignore_index: bool = False) -> DataFrame:
    df = cube_df.explode(column="band_index", ignore_index=ignore_index)

    # Raster object is mutable, so dupicates after explode must be copied
    df["id"] = df["raster"].map(id)
    df["duplicate_id"] = df["id"].duplicated()

    df["__i"] = range(len(df))

    for i, band_idx, raster in zip(df["__i"], df["band_index"], df["raster"]):
        row = df[df["__i"] == i]

        assert len(row) == 1

        if row["duplicate_id"].all():
            raster = raster.copy()

        if len(raster.shape) == 3 and raster.array is not None:
            try:
                raster.array = raster.array[band_idx - 1]
            except IndexError:
                raster.array = raster.array[band_idx - 2]

        raster._band_index = band_idx

        df.loc[df["__i"] == i, "raster"] = raster

    return df.drop(["__i", "id", "duplicate_id"], axis=1)

def explode(cube, ignore_index: bool = False):
    df = cube.df.explode(column="band_index", ignore_index=ignore_index)

    # Raster object is mutable, so dupicates after explode must be copied
    df["id"] = df["raster"].map(id)
    df["duplicate_id"] = df["id"].duplicated()

    df["__i"] = range(len(df))
    filt = lambda x: x["__i"] == i

    for i, band_idx, raster in zip(df["__i"], df["band_index"], df["raster"]):
        row = df[filt]

        assert len(row) == 1

        if row["duplicate_id"].all():
            raster = raster.copy()

        if len(raster.shape) == 3 and raster.array is not None:
            raster.array = raster.array[band_idx - 1]

        raster._band_index = band_idx

        df.loc[filt, "raster"] = raster

    cube.df = df.drop(["__i", "id", "duplicate_id"], axis=1)

    cube.update_df()

    # cube._df["name"] = [r.name for r in cube]
    # cube._df["band_name"] = [r.band_name for r in cube]

    return cube

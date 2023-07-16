def explode(cube, ignore_index: bool = False):
    df = cube.df

    # Raster object is mutable, so dupicates after explode must be copied
    df["id"] = df["raster"].map(id)
    df["duplicate_id"] = df["id"].duplicated()

    df = df.explode(column="band_index", ignore_index=ignore_index)
    df["__i"] = range(len(df))
    filt = lambda x: x["__i"] == i

    for i, band_idx, raster in zip(df["__i"], df["band_index"], df["raster"]):
        row = df[filt]

        if row["duplicate_id"] is True:
            raster = raster.copy()

        if len(raster.shape) == 3 and raster.array is not None:
            raster.array = raster.array[band_idx - 1]

        raster._band_index = band_idx

        df.loc[filt, "raster"] = raster

    cube._df = df.drop(["__i", "id", "duplicate_id"], axis=1)

    return cube

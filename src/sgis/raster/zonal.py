def zonal(
    polygons: GeoDataFrame,
    aggfunc: str | Callable | list[Callable | str],
    raster_calc_func: Callable | None = None,
    dropna: bool = True,
) -> GeoDataFrame:
    if self._raster_has_changed or self._array_has_changed():
        raise RasterHasChangedError("zonal")
    if not pyproj.CRS(self.crs).equals(pyproj.CRS(polygons.crs)):
        raise ValueError("crs mismatch.")

    if isinstance(aggfunc, (str, Callable)):
        aggfunc = [aggfunc]

    def _get_func_name(f):
        if callable(f):
            return f.__name__
        return str(f).replace("np.", "").replace("numpy.", "")

    new_cols = [_get_func_name(f) for f in aggfunc]

    def _to_numpy_func(text):
        f = getattr(np, text, None)
        if f is not None:
            return f
        f = getattr(np.ndarray, text, None)
        if f is not None:
            return f
        raise ValueError(
            "aggfunc must be functions or " "strings of numpy functions or methods."
        )

    aggfunc = [f if callable(f) else _to_numpy_func(f) for f in aggfunc]

    aggregated = {}

    for i in polygons.index:
        poly = polygons.loc[[i]]

        box1 = shapely.box(*self.total_bounds)
        box2 = shapely.box(*poly.total_bounds)

        if not box1.intersects(box2) or box1.touches(box2):
            aggregated[i] = [pd.NA for _ in range(len(aggfunc))]
            continue

        if self.dapla:
            import dapla as dp

            fs = dp.FileClient.get_gcs_file_system()
            with fs.open(self.path, mode="rb") as file:
                with rasterio.open(file) as src:
                    aggs = self._zonal_one_poly(src, poly, aggfunc, raster_calc_func, i)
        else:
            with rasterio.open(self.path) as src:
                aggs = self._zonal_one_poly(src, poly, aggfunc, raster_calc_func, i)

        aggregated = aggregated | aggs

    out = pd.DataFrame(
        polygons.index.map(aggregated).tolist(),
        columns=new_cols,
        index=polygons.index,
    )

    if dropna:
        return out.loc[~out.isna().all(axis=1)]
    else:
        return out


def _zonal_one_poly(src, poly, aggfunc, raster_calc_func, i):
    array, _ = rast_mask(
        dataset=src,
        shapes=self._to_geojson(poly),
        crop=True,
        filled=False,
    )

    flat_array = array.filled(np.nan).flatten()
    no_nans = flat_array[~np.isnan(flat_array)]

    if raster_calc_func:
        array_shape = no_nans.shape
        no_nans = raster_calc_func(no_nans)
        if array_shape != no_nans.shape:
            raise ValueError("raster_calc_func cannot change the shape of the array.")

    aggs: dict[float | int, list[float]] = {i: [f(no_nans) for f in aggfunc]}

    return aggs

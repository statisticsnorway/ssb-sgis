import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.wkt import loads


def les_geopandas(sti: str, engine = "pyogrio", **qwargs) -> gpd.GeoDataFrame:
    try:
        from dapla import FileClient
        fs = FileClient.get_gcs_file_system()

        if "parquet" in sti:
            with fs.open(sti, mode='rb') as file:
                return gpd.read_parquet(file, **qwargs)
        else:
            with fs.open(sti, mode='rb') as file:
                return gpd.read_file(file, engine=engine, **qwargs)
    except Exception:
        if "parquet" in sti:
            return gpd.read_parquet(sti, **qwargs)
        else:
            return gpd.read_file(sti, engine=engine, **qwargs)


def skriv_geopandas(df: gpd.GeoDataFrame, gcs_path: str, schema=None, **kwargs) -> None:
    """ funker ikke for shp og gdb """
    from dapla import FileClient
    from pyarrow import parquet

    pd.io.parquet.BaseImpl.validate_dataframe(df)

    fs = FileClient.get_gcs_file_system()

    if ".parquet" in gcs_path:
        from geopandas.io.arrow import _encode_metadata, _geopandas_to_arrow
        with fs.open(gcs_path, mode="wb") as buffer:
            table = _geopandas_to_arrow(df, index=df.index, schema_version=None)
            parquet.write_table(table, buffer, compression="snappy", **kwargs)
        return

    if ".gpkg" in gcs_path:
        driver = 'GPKG'
    elif ".geojson" in gcs_path:
        driver = "GeoJSON"
    elif ".gml" in gcs_path:
        driver = "GML"
    elif ".shp" in gcs_path:
        driver = "ESRI Shapefile"
    else:
        driver = None

    with fs.open(gcs_path, 'wb') as file:
        df.to_file(file, driver=driver)

        
def fiks_geometrier(gdf, ignore_index=False):
    """ reparerer geometri, så fjerner invalide, tomme og NaN-geometrier. """
    
    if isinstance(gdf, gpd.GeoDataFrame):
        gdf["geometry"] = gdf.make_valid()
        gdf = gdf[gdf.geometry.is_valid]
        gdf = gdf[~gdf.geometry.is_empty]
        gdf = gdf.dropna(subset = ["geometry"])
        if ignore_index:
            gdf = gdf.reset_index(drop=True)
    elif isinstance(gdf, gpd.GeoSeries):
        gdf = gdf.make_valid()
        gdf = gdf[gdf.is_valid]
        gdf = gdf[~gdf.is_empty]
        gdf = gdf.dropna()
        if ignore_index:
            gdf = gdf.reset_index(drop=True)
    else:
        raise ValueError("Input må være GeoDataFrame eller GeoSeries")
    return gdf


def til_gdf(geom, crs=None, **qwargs) -> gpd.GeoDataFrame:
    """ 
    Konverterer til geodataframe fra geoseries, shapely-objekt, wkt, liste med shapely-objekter eller shapely-sekvenser 
    OBS: når man har shapely-objekter eller wkt, bør man velge crs. 
    """

    if not crs:
        if isinstance(geom, str):
            raise ValueError("Du må bestemme crs når input er string.")
        crs = geom.crs
        
    if isinstance(geom, str):
        from shapely.wkt import loads
        geom = loads(geom)
        gdf = gpd.GeoDataFrame({"geometry": gpd.GeoSeries(geom)}, crs=crs, **qwargs)
    else:
        gdf = gpd.GeoDataFrame({"geometry": gpd.GeoSeries(geom)}, crs=crs, **qwargs)
    
    return gdf


def gdf_concat(gdf_liste: list, crs=None, axis=0, ignore_index=True, geometry="geometry", **concat_qwargs) -> gpd.GeoDataFrame:
    """ 
    Samler liste med geodataframes til en lang geodataframe.
    Ignorerer index, endrer til felles crs. 
    """
    
    gdf_liste = [gdf for gdf in gdf_liste if len(gdf)]
    
    if not len(gdf_liste):
        raise ValueError("gdf_concat: alle gdf-ene har 0 rader")
    
    if not crs:
        crs = gdf_liste[0].crs
    
    try:
        gdf_liste = [gdf.to_crs(crs) for gdf in gdf_liste]
    except ValueError:
        print("OBS: ikke alle gdf-ene dine har crs. Hvis du nå samler latlon og utm, må du først bestemme crs med set_crs(), så gi dem samme crs med to_crs()")

    return gpd.GeoDataFrame(pd.concat(gdf_liste, axis=axis, ignore_index=ignore_index, **concat_qwargs), geometry=geometry, crs=crs)


# lager n tilfeldige punkter innenfor et gitt område (mask)
def tilfeldige_punkter(n, mask=None):
    import random
    if mask is None:
        x = np.array([random.random()*10**7 for _ in range(n*1000)])
        y = np.array([random.random()*10**8 for _ in range(n*1000)])
        punkter = til_gdf([loads(f"POINT ({x} {y})") for x, y in zip(x, y)], crs=25833)
        return punkter
    mask_kopi = mask.copy()
    mask_kopi = mask_kopi.to_crs(25833)
    out = gpd.GeoDataFrame({"geometry":[]}, geometry="geometry", crs=25833)
    while len(out) < n:
        x = np.array([random.random()*10**7 for _ in range(n*1000)])
        x = x[(x > mask_kopi.bounds.minx.iloc[0]) & (x < mask_kopi.bounds.maxx.iloc[0])]
        
        y = np.array([random.random()*10**8 for _ in range(n*1000)])
        y = y[(y > mask_kopi.bounds.miny.iloc[0]) & (y < mask_kopi.bounds.maxy.iloc[0])]
        
        punkter = til_gdf([loads(f"POINT ({x} {y})") for x, y in zip(x, y)], crs=25833)
        overlapper = punkter.clip(mask_kopi)
        out = gdf_concat([out, overlapper])
    out = out.sample(n).reset_index(drop=True).to_crs(mask.crs)
    out["idx"] = out.index
    return out
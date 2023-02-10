import geopandas as gpd
import pandas as pd
import numpy as np

import geopandas as gpd
import pandas as pd


def eksists(path: str) -> bool:
    try:
        from dapla import details
        details(path)
        return True
    except FileNotFoundError:
        return False
    except ModuleNotFoundError:
        from os.path import exists
        return exists(path)


def read_geopandas(sti: str, **kwargs) -> gpd.GeoDataFrame:
    from dapla import FileClient
    fs = FileClient.get_gcs_file_system()

    if "parquet" in sti:
        with fs.open(sti, mode='rb') as file:
            return gpd.read_parquet(file, **kwargs)
    else:
        with fs.open(sti, mode='rb') as file:
            return gpd.read_file(file, **kwargs)


def write_geopandas(df: gpd.GeoDataFrame, gcs_path: str, **kwargs) -> None:
    """ funker ikke for shp og gdb """
    from dapla import FileClient
    from pyarrow import parquet

    pd.io.parquet.BaseImpl.validate_dataframe(df)

    fs = FileClient.get_gcs_file_system()

    if ".parquet" in gcs_path:
        from geopandas.io.arrow import _geopandas_to_arrow
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


def samle_filer(filer: list, **kwargs) -> gpd.GeoDataFrame:
    return pd.concat((les_geopandas(fil, **kwargs) for fil in filer), axis=0, ignore_index=True)
    
    
def lag_mappe(sti):
    if not eksisterer(sti):
        from os import makedirs
        makedirs(sti)
        
        
def read_invalid(sti, layer=None, **kwargs):
    """ Hvis fx en punktgeometri er kodet som LineString, klarer ikke geopandas å lese den. 
    Leser da 1000 rader av gangen, gjentar for problemradene fram til suksess. """
    
    import pyogrio
    
    rader = pyogrio.read_info(sti, layer)["features"]
    crs = pyogrio.read_info(sti, layer)["crs"]
    
    data = pd.DataFrame()
    mangler = []
    n = 0
    for i in np.arange(rader, step=1000):
        try:
            df = gpd.read_file(sti, layer=layer, engine="pyogrio", 
                                      skip_features=i, 
                                      max_features=1000, **kwargs)
            
            data = gpd.GeoDataFrame(pd.concat([data, df], axis=0, ignore_index=True), geometry="geometry", crs=crs)

            del df
            
        except Exception as e:
            mangler.append(i)
    
    print(mangler)
    print("len", len(mangler))
    
    mangler2 = []
    for i in mangler:
        for j in np.arange(start=i, stop=i+1000, step=50):
            try:
                df = gpd.read_file(sti, layer=layer, engine="pyogrio", 
                                        skip_features=j, 
                                        max_features=50, **kwargs)
                data = gpd.GeoDataFrame(pd.concat([data, df], axis=0, ignore_index=True), geometry="geometry", crs=crs)
                del df
            except Exception as e:
                mangler2.append(j)
    
    del mangler
    
    print(mangler2)
    print("len", len(mangler2))
    
    for i in mangler2:
        for j in np.arange(start=i, stop=i+50, step=1):
            try:
                df = gpd.read_file(sti, layer=layer, engine="pyogrio", 
                                        skip_features=j, 
                                        max_features=1, **kwargs)
                data = gpd.GeoDataFrame(pd.concat([data, df], axis=0, ignore_index=True), geometry="geometry", crs=crs)
                del df
            except Exception as e:
                n += 1
                if n<=5:
                    print(e)
                elif n==6:
                    print("Slutter å printe feilmeldinger")
                else:
                    pass

    print(f"{n} rader ble ikke lest")
        
    return data


if __name__=="__main__":
    x = read_invalid(r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\ruteplanleggerSykkel_FGDB_20220601.gdb", layer="lenker_line",
                    columns=["startnode", "sluttnode", "trafikkretning", "sykkelforbud", "typeveg", "aadt", "fartsgrense"])

    x.to_parquet(r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\ruteplanleggerSykkel_2022.parquet")

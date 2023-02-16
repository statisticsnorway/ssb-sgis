#%%
import geopandas as gpd
import requests
import shutil
import os
from zipfile import ZipFile
#from dapla import FileClient


def eksisterer(sti: str) -> bool:
    """ returnerer True hvis filen eksisterer, False hvis ikke. """
    return os.path.exists(sti)


def les_geopandas(sti: str, engine = "pyogrio", **kwargs) -> gpd.GeoDataFrame:
    if "parquet" in sti:
        return gpd.read_parquet(sti, **kwargs)
    else:
        return gpd.read_file(sti, engine=engine, **kwargs)


def skriv_geopandas(df: gpd.GeoDataFrame, gcs_path: str, schema=None, **kwargs) -> None:
    from pyarrow import parquet

    if ".parquet" in gcs_path:
        df.to_parquet(gcs_path)
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

    df.to_file(gcs_path, driver=driver)
        
        
def geonorge_json(metadataUuid, 
                  crs="25833", 
                  filformat = "FGDB 10.0", # "GML 3.2.1"
                  areas = {"code": "0000", "type": "landsdekkende", "name": "Hele landet"},
                  **kwargs,
                 ):

    return {
    "orderLines": [
        {"metadataUuid": metadataUuid,
        "areas": areas,
        "formats": [{"name": filformat}],
        "projections": [{"code": str(crs)}]
        } | {key: value for key, value in kwargs.items()}
    ]}


def unzipp(zipfil, unzippet_fil):
#    fs = FileClient.get_gcs_file_system()
    with ZipFile(zipfil) as z:
        z.extractall(unzippet_fil)


def slett(sti):
    if os.path.isfile(sti):
        os.remove(sti)
    if os.path.isdir(sti):
        shutil.rmtree(sti)
        
  
def hent_fra_geonorge(metadataUuid, sti, filformat = "FGDB 10.0", parquet = True, **kwargs):

    geonorge = "https://nedlasting.geonorge.no/api/order"
    
    zipfil = sti+".zip"
  
    js = geonorge_json(metadataUuid, filformat=filformat, **kwargs)
    
    p = requests.post(geonorge, json=js)
    p = p.json()
    
#    download_url = p["files"][0]["downloadUrl"]
 #   filnavn = p["files"][0]["name"]

    for fil in p["files"]:
        download_url = fil["downloadUrl"]
        filnavn = fil["name"]
    
        r = requests.get(download_url)
        innhold = r.content

    #    fs = FileClient.get_gcs_file_system()

        with open(zipfil, 'wb') as file:
            file.write(innhold)
        
#        print(eksisterer(zipfil))
        
        out = f"{sti}/{filnavn.strip('.zip')}"
    
#    print(out)
    
 #   print(filnavn)
    
        unzipp(zipfil, out)
    
        if parquet and not filformat=="TIFF":
            parquetfil = f"{sti}/{filnavn.strip('.zip')}.parquet"
            skriv_geopandas(les_geopandas(out), parquetfil)
            slett(out)
            slett(sti)
        
        slett(zipfil)


if __name__=="__main__":
    
    n5000_uuid = "c777d53d-8916-4d9d-bae4-6d5140e0c569"
    sti = f"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/n5000"
    
    hent_fra_geonorge(n5000_uuid, sti)
    
    print(gpd.read_parquet(sti+".parquet"))


    dem_uuid = "dddbb667-1303-4ac5-8640-7ec04c0e3918"
    sti = f"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/dtm10"
    hent_fra_geonorge(dem_uuid, 
                      sti,
                      filformat="TIFF",
    )

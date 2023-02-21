# %%
import os
import shutil
from zipfile import ZipFile

import geopandas as gpd
import requests

from .dapla import exists, read_geopandas, write_geopandas


# from dapla import FileClient


def geonorge_json(
    metadataUuid,
    crs="25833",
    filformat="FGDB 10.0",  # "GML 3.2.1"
    areas={"code": "0000", "type": "landsdekkende", "name": "Hele landet"},
    **kwargs,
):
    return {
        "orderLines": [
            {
                "metadataUuid": metadataUuid,
                "areas": areas,
                "formats": [{"name": filformat}],
                "projections": [{"code": str(crs)}],
            }
            | {key: value for key, value in kwargs.items()}
        ]
    }


def unzipp(zipfil, unzippet_fil):
    #    fs = FileClient.get_gcs_file_system()
    with ZipFile(zipfil) as z:
        z.extractall(unzippet_fil)


def slett(sti):
    if os.path.isfile(sti):
        os.remove(sti)
    if os.path.isdir(sti):
        shutil.rmtree(sti)


def hent_fra_geonorge(metadataUuid, sti, filformat="FGDB 10.0", parquet=True, **kwargs):
    geonorge = "https://nedlasting.geonorge.no/api/order"

    zipfil = sti + ".zip"

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

        with open(zipfil, "wb") as file:
            file.write(innhold)

        #        print(eksisterer(zipfil))

        out = f"{sti}/{filnavn.strip('.zip')}"

        #    print(out)

        #   print(filnavn)

        unzipp(zipfil, out)

        if parquet and not filformat == "TIFF":
            parquetfil = f"{sti}/{filnavn.strip('.zip')}.parquet"
            write_geopandas(read_geopandas(out), parquetfil)
            slett(out)
            slett(sti)

        slett(zipfil)


if __name__ == "__main__":
    n5000_uuid = "c777d53d-8916-4d9d-bae4-6d5140e0c569"
    sti = f"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/n5000"

    hent_fra_geonorge(n5000_uuid, sti)

    print(gpd.read_parquet(sti + ".parquet"))

    dem_uuid = "dddbb667-1303-4ac5-8640-7ec04c0e3918"
    sti = f"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/dtm10"
    hent_fra_geonorge(
        dem_uuid,
        sti,
        filformat="TIFF",
    )

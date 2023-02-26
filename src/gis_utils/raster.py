import warnings
from json import loads
from time import perf_counter

import geopandas as gpd
import numpy as np
import pandas as pd
from dapla import FileClient
from rasterio import open as rast_open
from rasterio.features import shapes
from rasterio.mask import mask
from shapely import box
from shapely.geometry import shape


warnings.filterwarnings(action="ignore", category=FutureWarning)


def loop_rasters(
    rasterliste: list,
    kommuner: gpd.GeoDataFrame,
    utmappe,
    overskriv=False,
):
    """
    Beregner bratthetskode (1-4) for alle tif-filer i en mappe.
    Intersecter med kommuneflate, og lagrer som parquet i angitt utmappe.
    """

    tid = perf_counter()
    n = len(rasterliste)

    # looper for filene i mappa
    for i, raster in enumerate(rasterliste):
        print(
            f"ferdig med {i} av {n} raster-tiles etter {round(perf_counter()-tid, 1)} sekunder",
            end="\r",
        )

        # isolerer filnavnet uten .tif
        tilenavn = raster.strip(".tif").split("/")[-1]
        utsti = f"{utmappe}/gridkode_{tilenavn}.parquet"

        # går videre hvis overskriv=True og filen eksisterer
        if not overskriv and eksisterer(utsti):
            continue

        bratthet = beregn_bratthet(raster)

        if bratthet is None:
            continue

        bratthet = del_i_kommuner(bratthet, kommuner)

        bratthet["tile"] = tilenavn

        skriv_geopandas(bratthet, utsti)

    print(
        f"ferdig med {i+1} av {n} raster-tiles etter {round(perf_counter()-tid, 1)} sekunder",
        end="\r",
    )


def loop_rasters_forsok(
    rasterliste: list,
    maske: gpd.GeoDataFrame,
):
    """ """

    # looper for filene i mappa
    bratthet = pd.DataFrame()
    for i, raster in enumerate(rasterliste):
        print(f"ferdig med {i} av {len(rasterliste)} rastertiles", end="\r")

        res = beregn_bratthet(raster, maske=maske)

        if res is None:
            continue

        bratthet = gpd.GeoDataFrame(
            pd.concat([bratthet, res], ignore_index=True),
            geometry="geometry",
            crs=res.crs,
        )

    return bratthet


def beregn_bratthet(
    raster: str, maske: gpd.GeoDataFrame | None = None
) -> gpd.GeoDataFrame:
    """
    Beregner bratthetskode for én rasterfil
    """

    crs, cell_size_x, cell_size_y = finn_crs_og_cellsize_xy(raster)

    # lager en firkant som er identisk som raster-utsnittet
    if maske is None:
        maske = make_square_mask(raster, crs)

    # endrer til formatet som godtas av rasterio.features.mask
    maske = to_geojson(maske)

    # man må først gjennom sikkerhetsmuren til dapla, så lese filen
    fs = FileClient.get_gcs_file_system()

    with fs.open(raster, mode="rb") as file:
        with rast_open(file) as dataset:
            verdier, metadata = mask(dataset=dataset, shapes=maske)

            if not len(verdier[verdier > 0]):
                return

            # verdier inneholder raster-verdiene. Formatet er en tredimensjonal numpy array med shape (1, x, y),
            # som vil si at verdier[0] er en todimensjonal array/matrise med x- og y-verdier
            # sjekker at dette fortsatt stemmer
            assert (
                verdier.shape[0] == 1
            ), f"mask har tidligere returnert en 3d array med shape (1, x, y). Nå er shape-en {verdier.shape}."

            # np.gradient returnerer stigning i to retninger (x og y)
            stigning_x, stigning_y = np.gradient(verdier[0], cell_size_x, cell_size_y)

            # gjør om negativ stigning til positiv og plusser retningene sammen
            stigning = abs(stigning_x) + abs(stigning_y)

            # endrer til stigningsgrader med litt matematikk
            radians = np.arctan(stigning)
            grader = np.degrees(radians)

            assert np.max(grader) <= 90, "90 skal være maksimal stigningsgrader"

            # koder om fra grader til gridkode-kategorier (1,2,3,4)
            kategorier = [
                (grader < 3),
                (grader >= 3) & (grader < 10),
                (grader >= 10) & (grader < 25),
                (grader >= 25) & (grader <= 90),
            ]
            nye_verdier = [1, 2, 3, 4]

            verdier[0] = np.select(kategorier, nye_verdier)

            # fra raster til dataframe
            df = gpd.GeoDataFrame(
                pd.DataFrame(
                    [
                        (value, shape(geom))
                        for geom, value in shapes(verdier, transform=metadata)
                    ],
                    columns=["gridkode", "geometry"],
                ),
                geometry="geometry",
                crs=crs,
            )

            return df


def to_geojson(gdf):
    return [loads(gdf.to_json())["features"][0]["geometry"]]


def from_geojson(json):
    return json["coordinates"][0]


def make_square_mask(raster, crs):
    """
    lager firkanta polygon i området for raster-filen
    """

    fs = FileClient.get_gcs_file_system()

    with fs.open(raster, mode="rb") as file:
        with rast_open(file) as dataset:
            xmin, ymin, xmax, ymax = dataset.bounds
            maske = box(xmin, ymin, xmax, ymax)
            return gpd.GeoDataFrame({"geometry": [maske]}, geometry="geometry", crs=crs)


def finn_crs_og_cellsize_xy(raster):
    fs = FileClient.get_gcs_file_system()

    with fs.open(raster, mode="rb") as file:
        with rast_open(file) as dataset:
            cell_size_x, cell_size_y = dataset.res
            crs = dataset.crs

    return crs, cell_size_x, cell_size_y


if __name__ == "__main__":
    aar = 2022

    from dapla import FileClient

    raster = "ssb-prod-arealstrandsone-data-synk-opp/data2022/66m1_2_10m_z33.tif"

    print(beregn_bratthet(raster, grid_size=7500))

    print(x.gridkode.value_counts())

from pathlib import Path
from urllib.request import urlopen

import geopandas as gpd
from geopandas import GeoDataFrame


def read_parquet_url(url) -> GeoDataFrame:
    filename = url.split("/")[-1]

    save_as = Path.cwd().resolve() / filename

    if save_as.exists():
        return gpd.read_parquet(save_as)

    # Download from URL
    with urlopen(url) as file:
        content = file.read()

    # Download from URL
    with urlopen(url) as file:
        content = file.read()

    # Save to file
    with open(save_as, "wb") as download:
        download.write(content)

    return gpd.read_parquet(save_as)

import abc
import datetime
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import folium
import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from shapely import Geometry
from shapely import get_exterior_ring
from shapely import make_valid
from shapely import polygons
from shapely import set_precision
from shapely import simplify
from shapely.errors import GEOSException

from ..geopandas_tools.conversion import to_gdf
from ..geopandas_tools.conversion import to_shapely
from ..geopandas_tools.sfilter import sfilter
from ..raster.image_collection import Band

JSON_PATH = Path(__file__).parent / "norge_i_bilder.json"

JSON_YEARS = [str(year) for year in range(2006, datetime.datetime.now().year + 1)]

DEFAULT_YEARS: tuple[str] = tuple(
    str(year)
    for year in range(
        int(datetime.datetime.now().year) - 10,
        int(datetime.datetime.now().year) + 1,
    )
)


@dataclass
class WmsLoader(abc.ABC):
    """Abstract base class for wms loaders.

    Child classes must implement the method 'get_tiles',
    which should return a list of folium.WmsTileLayer.
    """

    @abc.abstractmethod
    def get_tiles(self, bbox: Any, max_zoom: int = 40) -> list[folium.WmsTileLayer]:
        """Get all tiles intersecting with a bbox."""

    @abc.abstractmethod
    def load_tiles(self) -> None:
        """Load all tiles into self.tiles.

        Not needed in sgis.explore.
        """
        pass


@dataclass
class NorgeIBilderWms(WmsLoader):
    """Loads Norge i bilder tiles as folium.WmsTiles."""

    years: Iterable[int | str] = DEFAULT_YEARS
    contains: str | Iterable[str] | None = None
    not_contains: str | Iterable[str] | None = None
    show: bool | Iterable[int] | int = False
    _use_json: bool = True

    def load_tiles(self, verbose: bool = False) -> None:
        """Load all Norge i bilder tiles into self.tiles."""
        url = "https://wms.geonorge.no/skwms1/wms.nib-prosjekter?SERVICE=WMS&REQUEST=GetCapabilities"

        name_pattern = r"<Name>(.*?)</Name>"
        bbox_pattern = (
            r"<EX_GeographicBoundingBox>.*?"
            r"<westBoundLongitude>(.*?)</westBoundLongitude>.*?"
            r"<eastBoundLongitude>(.*?)</eastBoundLongitude>.*?"
            r"<southBoundLatitude>(.*?)</southBoundLatitude>.*?"
            r"<northBoundLatitude>(.*?)</northBoundLatitude>.*?</EX_GeographicBoundingBox>"
        )

        all_tiles: list[dict] = []
        with urlopen(url) as file:
            xml_data: str = file.read().decode("utf-8")

            for text in xml_data.split('<Layer queryable="1">')[1:]:

                # Extract bounding box values
                bbox_match = re.search(bbox_pattern, text, re.DOTALL)
                if bbox_match:
                    minx, maxx, miny, maxy = (
                        float(bbox_match.group(i)) for i in [1, 2, 3, 4]
                    )
                    this_bbox = shapely.box(minx, miny, maxx, maxy)
                else:
                    this_bbox = None

                name_match = re.search(name_pattern, text, re.DOTALL)
                name = name_match.group(1) if name_match else None

                if (
                    not name
                    or not any(year in name for year in self.years)
                    or (
                        self.contains
                        and not any(re.search(x, name.lower()) for x in self.contains)
                    )
                    or (
                        self.not_contains
                        and any(re.search(x, name.lower()) for x in self.not_contains)
                    )
                ):
                    continue

                this_tile = {}
                this_tile["name"] = name
                this_tile["bbox"] = this_bbox
                year = name.split(" ")[-1]
                is_year_or_interval: bool = all(
                    part.isnumeric() and len(part) == 4 for part in year.split("-")
                )
                if is_year_or_interval:
                    this_tile["year"] = year
                else:
                    this_tile["year"] = "9999"

                all_tiles.append(this_tile)

        self.tiles = sorted(all_tiles, key=lambda x: (x["year"]))

        masks = self._get_norge_i_bilder_polygon_masks(verbose=verbose)
        for tile in self.tiles:
            mask = masks.get(tile["name"], None)
            tile["geometry"] = mask

    def _get_norge_i_bilder_polygon_masks(self, verbose: bool):
        from owslib.util import ServiceException
        from owslib.wms import WebMapService
        from PIL import Image

        relevant_names: dict[str, str] = {x["name"]: x["bbox"] for x in self.tiles}
        assert len(relevant_names), relevant_names

        url = "https://wms.geonorge.no/skwms1/wms.nib-mosaikk?SERVICE=WMS&REQUEST=GetCapabilities"
        wms = WebMapService(url, version="1.3.0")
        out = {}
        # ttiles = {wms[layer].title: [] for layer in list(wms.contents)}
        # for layer in list(wms.contents):
        #     if wms[layer].title not in relevant_names:
        #         continue
        #     ttiles[wms[layer].title].append(layer)
        # import pandas as pd

        # df = pd.Series(ttiles).to_frame("title")
        # df["n"] = df["title"].str.len()
        # df = df.sort_values("n")
        # for x in df["title"]:
        #     if len(x) == 1:
        #         continue
        #     bounds = {tuple(wms[layer].boundingBoxWGS84) for layer in x}
        #     if len(bounds) <= 1:
        #         continue
        #     print()
        #     for layer in x:
        #         print(layer)
        #         print(wms[layer].title)
        #         bbox = wms[layer].boundingBoxWGS84
        #         print(bbox)

        for layer in list(wms.contents):
            title = wms[layer].title
            if title not in relevant_names:
                continue
            bbox = wms[layer].boundingBoxWGS84
            bbox = tuple(to_gdf(bbox, crs=4326).to_crs(25832).total_bounds)

            existing_bbox = relevant_names[title]
            existing_bbox = to_gdf(existing_bbox, crs=4326).to_crs(25832).union_all()
            if not to_shapely(bbox).intersects(existing_bbox):
                continue
            diffx = bbox[2] - bbox[0]
            diffy = bbox[3] - bbox[1]
            width = int(diffx / 40)
            height = int(diffy / 40)
            if not bbox:
                continue
            try:
                img = wms.getmap(
                    layers=[layer],
                    styles=[""],  # Empty unless you know the style
                    srs="EPSG:25832",
                    bbox=bbox,
                    size=(width, height),
                    format="image/jpeg",
                    transparent=True,
                    bgcolor="#FFFFFF",
                )
            except (ServiceException, AttributeError) as e:
                if verbose:
                    print(type(e), e)
                continue

            arr = np.array(Image.open(BytesIO(img.read())))
            if not np.sum(arr):
                continue

            band = Band(
                np.where(np.any(arr != 0, axis=-1), 1, 0), bounds=bbox, crs=25832
            )
            polygon = band.to_geopandas()[lambda x: x["value"] == 1].geometry.values
            polygon = make_valid(polygons(get_exterior_ring(polygon)))
            polygon = make_valid(set_precision(polygon, 1))
            polygon = make_valid(simplify(polygon, 100))
            polygon = make_valid(set_precision(polygon, 1))
            polygon = GeoSeries(polygon, crs=25832).to_crs(4326)
            if verbose:
                print(f"Layer name: {layer}")
                print(f"Title: {wms[layer].title}")
                print(f"Bounding box: {wms[layer].boundingBoxWGS84}")
                print(f"polygon: {polygon}")
                print("-" * 40)

            for x in [0, 0.1, 0.001, 1]:
                try:
                    out[title] = make_valid(polygon.buffer(x).make_valid().union_all())
                except GEOSException:
                    pass
                break

        return out

    def get_tiles(self, mask: Any, max_zoom: int = 40) -> list[folium.WmsTileLayer]:
        """Get all Norge i bilder tiles intersecting with a mask (bbox or polygon)."""
        if self.tiles is None:
            self.load_tiles()

        if not isinstance(mask, (GeoSeries | GeoDataFrame | Geometry)):
            mask = to_shapely(mask)

        if isinstance(self.show, bool):
            show = self.show
        else:
            show = False

        relevant_tiles = self._filter_tiles(mask)
        tile_layers = {
            name: folium.WmsTileLayer(
                url="https://wms.geonorge.no/skwms1/wms.nib-prosjekter",
                name=name,
                layers=name,
                format="image/png",  # Tile format
                transparent=True,  # Allow transparency
                version="1.3.0",  # WMS version
                attr="&copy; <a href='https://www.geonorge.no/'>Geonorge</a>",
                show=show,
                max_zoom=max_zoom,
            )
            for name in relevant_tiles["name"]
        }

        if not len(tile_layers):
            return tile_layers

        if isinstance(self.show, int):
            tile = tile_layers[list(tile_layers)[self.show]]
            tile.show = True
        elif isinstance(self.show, Iterable):
            for i in self.show:
                tile = tile_layers[list(tile_layers)[i]]
                tile.show = True

        return tile_layers

    def _filter_tiles(self, mask):
        """Filter relevant dates with pandas and geopandas because fast."""
        df = pd.DataFrame(self.tiles)
        filt = (df["name"].notna()) & (df["year"].str.contains("|".join(self.years)))
        if self.contains:
            for x in self.contains:
                filt &= df["name"].str.contains(x)
        if self.not_contains:
            for x in self.not_contains:
                filt &= ~df["name"].str.contains(x)
        df = df[filt]
        geoms = np.where(df["geometry"].notna(), df["geometry"], df["bbox"])
        geoms = GeoSeries(geoms)
        assert geoms.index.is_unique
        return df.iloc[sfilter(geoms, mask).index]

    def __post_init__(self) -> None:
        """Fix typings."""
        if self.contains and isinstance(self.contains, str):
            self.contains = [self.contains]
        elif self.contains:
            self.contains = [x for x in self.contains]
        if self.not_contains and isinstance(self.not_contains, str):
            self.not_contains = [self.not_contains]
        elif self.not_contains:
            self.not_contains = [x for x in self.not_contains]

        self.years = [str(int(year)) for year in self.years]

        if self._use_json and all(year in JSON_YEARS for year in self.years):
            try:
                with open(JSON_PATH, encoding="utf-8") as file:
                    self.tiles = json.load(file)
            except FileNotFoundError:
                self.tiles = None
                return
            self.tiles = [
                {
                    key: (
                        value
                        if key not in ["bbox", "geometry"]
                        else shapely.wkt.loads(value)
                    )
                    for key, value in tile.items()
                }
                for tile in self.tiles
                if any(str(year) in tile["name"] for year in self.years)
            ]
        else:
            self.tiles = None

    def __repr__(self) -> str:
        """Print representation."""
        return f"{self.__class__.__name__}({len(self.tiles or [])})"

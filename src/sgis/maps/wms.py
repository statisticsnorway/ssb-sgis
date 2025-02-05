import abc
import datetime
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import folium
import shapely

from ..geopandas_tools.conversion import to_shapely

JSON_PATH = Path(__file__).parent / "norge_i_bilder.json"

JSON_YEARS = [str(year) for year in range(1999, 2025)]

DEFAULT_YEARS: tuple[str] = tuple(
    str(year)
    for year in range(
        int(datetime.datetime.now().year) - 8,
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

    def load_tiles(self) -> None:
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
                if year.isnumeric() and len(year) == 4:
                    this_tile["year"] = year
                else:
                    this_tile["year"] = "9999"
                all_tiles.append(this_tile)

        self.tiles = sorted(all_tiles, key=lambda x: x["year"])

    def get_tiles(self, bbox: Any, max_zoom: int = 40) -> list[folium.WmsTileLayer]:
        """Get all Norge i bilder tiles intersecting with a bbox."""
        if self.tiles is None:
            self.load_tiles()

        all_tiles = {}

        bbox = to_shapely(bbox)

        if isinstance(self.show, bool):
            show = self.show
        else:
            show = False

        for tile in self.tiles:
            if not tile["bbox"] or not tile["bbox"].intersects(bbox):
                continue

            name = tile["name"]

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

            all_tiles[name] = folium.WmsTileLayer(
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

        if isinstance(self.show, int):
            tile = all_tiles[list(all_tiles)[self.show]]
            tile.show = True
        elif isinstance(self.show, Iterable):
            for i in self.show:
                tile = all_tiles[list(all_tiles)[i]]
                tile.show = True

        return all_tiles

    def __post_init__(self) -> None:
        """Fix typings."""
        if self.contains and isinstance(self.contains, str):
            self.contains = [self.contains.lower()]
        elif self.contains:
            self.contains = [x.lower() for x in self.contains]

        if self.not_contains and isinstance(self.not_contains, str):
            self.not_contains = [self.not_contains.lower()]
        elif self.not_contains:
            self.not_contains = [x.lower() for x in self.not_contains]

        self.years = [str(int(year)) for year in self.years]

        if all(year in JSON_YEARS for year in self.years):
            try:
                with open(JSON_PATH, encoding="utf-8") as file:
                    self.tiles = json.load(file)
            except FileNotFoundError:
                self.tiles = None
                return
            self.tiles = [
                {
                    key: value if key != "bbox" else shapely.wkt.loads(value)
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

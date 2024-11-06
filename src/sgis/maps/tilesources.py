from xyzservices import Bunch
from xyzservices import TileProvider
from xyzservices import providers

kartverket = Bunch(
    topo=TileProvider(
        name="Topografisk norgeskart",
        url="https://cache.kartverket.no/v1/wmts/1.0.0/topo/default/webmercator/{z}/{y}/{x}.png",
        attribution="© Kartverket",
        html_attribution='&copy; <a href="https://kartverket.no">Kartverket</a>',
    ),
    topogråtone=TileProvider(
        name="Topografisk norgeskart gråtone",
        url="https://cache.kartverket.no/v1/wmts/1.0.0/topograatone/default/webmercator/{z}/{y}/{x}.png",
        attribution="© Kartverket",
        html_attribution='&copy; <a href="https://kartverket.no">Kartverket</a>',
    ),
    toporaster=TileProvider(
        name="Topografisk raster",
        url="https://cache.kartverket.no/v1/wmts/1.0.0/toporaster/default/webmercator/{z}/{y}/{x}.png",
        attribution="© Kartverket",
        html_attribution='&copy; <a href="https://kartverket.no">Kartverket</a>',
    ),
    sjøkart=TileProvider(
        name="Sjøkart",
        url="https://cache.kartverket.no/v1/wmts/1.0.0/sjokartraster/default/webmercator/{z}/{y}/{x}.png",
        attribution="© Kartverket",
        html_attribution='&copy; <a href="https://kartverket.no">Kartverket</a>',
    ),
    norge_i_bilder=TileProvider(
        name="Norge i bilder",
        url="https://opencache.statkart.no/gatekeeper/gk/gk.open_nib_web_mercator_wmts_v2?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=Nibcache_web_mercator_v2&STYLE=default&FORMAT=image/jpgpng&tileMatrixSet=default028mm&tileMatrix={z}&tileRow={y}&tileCol={x}",
        max_zoom=19,
        attribution="© Geovekst",
    ),
)

xyz = Bunch({"Kartverket": kartverket} | providers)

from xyzservices import TileProvider, Bunch, providers

kartverket = Bunch(
    norgeskart=TileProvider(
        name="Norgeskart",
        url="https://opencache.statkart.no/gatekeeper/gk/gk.open_gmaps?layers=norgeskart_bakgrunn&zoom={z}&x={x}&y={y}",
        attribution="© Kartverket",
        html_attribution='&copy; <a href="https://kartverket.no">Kartverket</a>',
    ),

    bakgrunnskart_forenklet=TileProvider(
        name="Norgeskart forenklet",
        url="https://opencache.statkart.no/gatekeeper/gk/gk.open_gmaps?layers=bakgrunnskart_forenklet&zoom={z}&x={x}&y={y}",
        attribution="© Kartverket",
        html_attribution='&copy; <a href="https://kartverket.no">Kartverket</a>',
    ),

    norges_grunnkart=TileProvider(
        name="Norges grunnkart",
        url="https://opencache.statkart.no/gatekeeper/gk/gk.open_gmaps?layers=norges_grunnkart&zoom={z}&x={x}&y={y}",
        attribution="© Kartverket",
        html_attribution='&copy; <a href="https://kartverket.no">Kartverket</a>',
    ),

    norges_grunnkart_gråtone=TileProvider(
        name="Norges grunnkart gråtone",
        url="https://opencache.statkart.no/gatekeeper/gk/gk.open_gmaps?layers=norges_grunnkart_graatone&zoom={z}&x={x}&y={y}",
        attribution="© Kartverket",
        html_attribution='&copy; <a href="https://kartverket.no">Kartverket</a>',
    ),

    n50=TileProvider(
        name="N5 til N50 kartdata",
        url="https://opencache.statkart.no/gatekeeper/gk/gk.open_gmaps?layers=kartdata3&zoom={z}&x={x}&y={y}",
        attribution="© Kartverket",
        html_attribution='&copy; <a href="https://kartverket.no">Kartverket</a>',
    ),

    topogråtone=TileProvider(
        name="Topografisk norgeskart gråtone",
        url="https://opencache.statkart.no/gatekeeper/gk/gk.open_gmaps?layers=topo4graatone&zoom={z}&x={x}&y={y}",
        attribution="© Kartverket",
        html_attribution='&copy; <a href="https://kartverket.no">Kartverket</a>',
    ),

    toporaster=TileProvider(
        name="Topografisk raster",
        url="https://opencache.statkart.no/gatekeeper/gk/gk.open_gmaps?layers=toporaster4&zoom={z}&x={x}&y={y}",
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

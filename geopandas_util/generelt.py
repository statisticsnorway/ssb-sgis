import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import Geometry
from geopandas import GeoDataFrame, GeoSeries


# evt fix_geoms()
def clean_geoms(
    gdf: GeoDataFrame | GeoSeries, 
    ignore_index=False,
    single_geom_type: bool = True,
    ) -> GeoDataFrame | GeoSeries:

    """Repairs geometries,
    removes geometries that are invalid, empty, NaN and None,
    keeps only rows with in the data.

    Args:
        gdf: GeoDataFrame or GeoSeries to be fixed or removed.
        ignore_index: whether the index should be reset and dropped. Defaults to False to be consistent with pandas.
        single_geomtype: if only the most common geometry type ((multi)point, (multi)line, (multi)poly) should be kept. 
            Defaults to False, eventhough this might raise an exception in overlay operations
            Please note thatin order for the function to not do anything unexpected.        

    Returns:
        GeoDataFrame or GeoSeries with fixed geometries and only the rows with valid, non-empty and not-NaN/-None geometries.

    """
    
    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):
        raise TypeError(
            f"'gdf' should be GeoDataFrame or GeoSeries, got {type(gdf)}"
        )
    
    fixed = gdf.make_valid()

    cleaned = fixed.loc[
        (fixed.is_valid) & 
        (~fixed.is_empty) &
        (fixed.notna())
    ]

    if single_geom_type:
        cleaned = to_single_geom_type(cleaned)
    
    if isinstance(gdf, gpd.GeoDataFrame):
        gdf = gdf.loc[cleaned.index]
        gdf._geometry_column_name = cleaned
    else:
        gdf = cleaned

    if ignore_index:
        gdf = gdf.reset_index(drop=True)

    return gdf
    
git config --global diff.ipynb.textconv "C:\Users\ort\AppData\Local\Programs\Python\Python311\python.exe -m nbstripout -t"


def to_single_geom_type(
    gdf: GeoDataFrame | GeoSeries, 
    ignore_index: bool = False
    ) -> GeoDataFrame | GeoSeries:
    """ 
    overlay godtar ikke blandede geometrityper i samme gdf.
    """
    
    polys = ["Polygon", "MultiPolygon"]
    lines = ["LineString", "MultiLineString", "LinearRing"]
    points = ["Point", "MultiPoint"]

    poly_check = len(gdf.loc[gdf.geom_type.isin(polys)])
    lines_check = len(gdf.loc[gdf.geom_type.isin(lines)])
    points_check = len(gdf.loc[gdf.geom_type.isin(points)])
    
    _max = max([poly_check, lines_check, points_check])

    if _max == len(geom):
        return geom

    if poly_check == _max:
        gdf = gdf.loc[gdf.geom_type.isin(polys)]
    elif lines_check == _max:
        gdf = gdf.loc[gdf.geom_type.isin(lines)]
    elif points_check == _max:
        gdf = gdf.loc[gdf.geom_type.isin(points)]
    else:
        raise ValueError("Mixed geometry types and equal amount of two or all the types.")
    
    if ignore_index:
        gdf = gdf.reset_index(drop=True)
    
    return gdf


#def clipclean(
def clipfix(
    gdf: GeoDataFrame | GeoSeries,
    clip_to: GeoDataFrame | GeoSeries | Geometry
    keep_geom_type: bool = True,
    **kwargs
    ) -> GeoDataFrame | GeoSeries:

    """Clip geometries to the mask extent, then cleans the geometries.
    geopandas.clip does a fast clipping, with no guarantee for valid outputs. Here, geometries are made valid, then invalid, empty, nan and None geometries are removed. 

    """
    
    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):
        raise TypeError(
            f"'gdf' should be GeoDataFrame or GeoSeries, got {type(gdf)}"
        )
    
    return gdf.clip(clip_to, keep_geom_type=keep_geom_type, **kwargs).pipe(clean_geoms)


def gdf_concat(gdfs: list | tuple, crs=None, axis=0, ignore_index=True, geometry="geometry", **concat_kwargs) -> gpd.GeoDataFrame:
    """ 
    Samler liste med geodataframes til en lang geodataframe.
    Ignorerer index, endrer til felles crs. 
    """
    
    gdfs = [gdf for gdf in gdfs if len(gdf)]
    
    if not len(gdfs):
        raise ValueError("gdf_concat: alle gdf-ene har 0 rader")
    
    if not crs:
        crs = gdfs[0].crs
    
    try:
        gdfs = [gdf.to_crs(crs) for gdf in gdfs]
    except ValueError:
        print("OBS: ikke alle gdf-ene dine har crs. Hvis du nå samler latlon og utm, må du først bestemme crs med set_crs(), så gi dem samme crs med to_crs()")

    return gpd.GeoDataFrame(pd.concat(gdfs, axis=axis, ignore_index=ignore_index, **concat_kwargs), geometry=geometry, crs=crs)


def to_gdf(geom, crs=None, **kwargs) -> gpd.GeoDataFrame:
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
        gdf = gpd.GeoDataFrame({"geometry": gpd.GeoSeries(geom)}, crs=crs, **kwargs)
    else:
        gdf = gpd.GeoDataFrame({"geometry": gpd.GeoSeries(geom)}, crs=crs, **kwargs)
    
    return gdf


def overlay_update(gdf1: GeoDataFrame, gdf2: GeoDataFrame, **kwargs) -> GeoDataFrame:
    """ En overlay-variant som ikke finnes i geopandas. """
    
    out = gdf1.overlay(gdf2, how ="difference", keep_geom_type=True, **kwargs)
    out = out.loc[:, ~out.columns.str.contains('index|level_')]
    out = gdf_concat([out, gdf2])
    return out


def try_overlay(
    gdf1: GeoDataFrame, 
    gdf2: GeoDataFrame, 
    presicion_col: bool = True, 
    max_rounding: int = 3, 
    single_geomtype: bool = True, 
    **kwargs
    ) -> GeoDataFrame:
    """ 
    Overlay, i hvert fall union, har gitt TopologyException: found non-noded intersection error from overlay.
    https://github.com/geopandas/geopandas/issues/1724
    En løsning er å avrunde koordinatene for å få valid polygon.
    Prøver først uten avrunding, så runder av til 10 koordinatdesimaler, så 9, 8, ..., og så gir opp på 0
    
    presisjonskolonne: om man skal inkludere en kolonne som angir hvilken avrunding som måtte til.
    max_avrunding: hvilken avrunding man stopper på. 0 betyr at man fortsetter fram til 0 desimaler.
    """

    try:
        gdf1 = fix_geoms(gdf1, single_geomtype=single_geomtype)
        gdf2 = fix_geoms(gdf2, single_geomtype=single_geomtype)
        return gdf1.overlay(gdf2, **kwargs)

    except Exception:
        from shapely.wkt import loads, dumps

        # loop through list from 10 to 'max_rounding'

        roundings = list(range(max_rounding, 11))
        roundings.reverse()

        for rounding in roundings:
            try:
                gdf1.geometry = [loads(dumps(gdf, rounding_precision=rounding)) for geom in gdf1.geometry]
                gdf2.geometry = [loads(dumps(gdf, rounding_precision=rounding)) for geom in gdf2.geometry]

                gdf1 = fix_geoms(gdf1, single_geomtype=single_geomtype)
                gdf2 = fix_geoms(gdf2, single_geomtype=single_geomtype)

                overlayet = gdf1.overlay(gdf2, **kwargs)

                if presicion_col:
                    overlayet["avrunding"] = rounding

                return overlayet

            except Exception:
                rounding -= 1

        # returnerer feilmeldingen hvis det fortsatt ikke funker
        gdf1.overlay(gdf2, **kwargs)


def try_diss(gdf, presicion_col=True, max_rounding = 5, **kwargs):
    """ 
    dissolve har gitt TopologyException: found non-noded intersection error from overlay.
    En løsning er å avrunde koordinatene for å få valid polygon.
    Prøver først uten avrunding, så runder av til 10 koordinatdesimaler, så 9, 8, ..., og så gir opp på 0
    
    presisjonskolonne: om man skal inkludere en kolonne som angir hvilken avrunding som måtte til.
    max_avrunding: hvilken avrunding man stopper på. 0 betyr at man fortsetter fram til 0 desimaler.
    """

    from .buffer_dissolve_explode import diss

    try:
        dissolvet = diss(gdf, **kwargs)
        if presicion_col:
            dissolvet["avrunding"] = np.nan
        return dissolvet

    except Exception:
        from shapely.wkt import loads, dumps

        # liste fra 10 til 0, eller max_avrunding til 0
        avrundinger = list(range(max_rounding, 11))
        avrundinger.reverse()

        for avrunding in avrundinger:
            try:
                gdf.geometry = [loads(dumps(gdf, rounding_precision=avrunding)) for geom in gdf.geometry]

                dissolvet = diss(gdf, **kwargs)

                if presicion_col:
                    dissolvet["avrunding"] = avrunding

                return dissolvet

            except Exception:
                avrunding -= 1

        # returnerer feilmeldingen hvis det fortsatt ikke funker
        diss(gdf, **kwargs)


def del_i_kommuner(gdf, kommunedata):
    
    kommunedata = kommunedata.loc[:, ["KOMMUNENR", "FYLKE", "geometry"]]
    
    gdf = gdf.loc[:, ~gdf.columns.str.contains("KOMM|FYLK|Shape_|SHAPE_|index|level_")]
    
    return (gdf
            .overlay(kommunedata, how="intersection")
            .drop("index_right", axis=1, errors="ignore")
            .pipe(fix_geoms)
           )
    

def min_sjoin(left_gdf, right_gdf, fjern_dupkol = True, **kwargs) -> gpd.GeoDataFrame:
    """ 
    som gpd.sjoin bare at kolonner i right_gdf som også er i left_gdf fjernes (fordi det snart vil gi feilmelding i geopandas)
    og kolonner som har med index å gjøre fjernes, fordi sjoin returnerer index_right som kolonnenavn, som gir feilmelding ved neste join. 
    """

    #fjern index-kolonner
    left_gdf = left_gdf.loc[:, ~left_gdf.columns.str.contains('index|level_')]
    right_gdf = right_gdf.loc[:, ~right_gdf.columns.str.contains('index|level_')]

    #fjern kolonner fra gdf2 som er i gdf1
    if fjern_dupkol:
        right_gdf.columns = [col if col not in left_gdf.columns or col=="geometry" else "skal_fjernes" for col in right_gdf.columns]
        right_gdf = right_gdf.loc[:, ~right_gdf.columns.str.contains('skal_fjernes')]

    joinet = left_gdf.sjoin(right_gdf, **kwargs).reset_index()

    return joinet.loc[:, ~joinet.columns.str.contains('index|level_')]


def kartlegg(gdf, kolonne=None, scheme="Quantiles", tittel=None, storrelse=15, fontsize=16, legend=True, alpha=0.7, **kwargs) -> None:
    """ Enkel, statisk kartlegging. """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, figsize=(storrelse, storrelse))
    ax.set_axis_off()
    ax.set_title(tittel, fontsize = fontsize)
    gdf.plot(kolonne, scheme=scheme, legend=legend, alpha=alpha, ax=ax, **kwargs)


def tilfeldige_punkter(n: int, mask=None) -> gpd.GeoDataFrame:
    """ lager n tilfeldige punkter innenfor et gitt område (mask). """
    from shapely.wkt import loads
    import random
    if mask is None:
        x = np.array([random.random()*10**7 for _ in range(n*1000)])
        y = np.array([random.random()*10**8 for _ in range(n*1000)])
        punkter = to_gdf([loads(f"POINT ({x} {y})") for x, y in zip(x, y)], crs=25833)
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

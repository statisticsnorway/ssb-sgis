import pandas as pd
import geopandas as gpd
from shapely import Geometry
from geopandas import GeoDataFrame, GeoSeries

"""
Functions that buffer, dissolve and/or explodes (multipart to singlepart) geodataframes.

Regler for alle funksjonene:
 - høyere buffer-oppløsning enn standard
 - reparerer geometrien etter buffer og dissolve, men ikke etter explode, siden reparering kan returnere multipart-geometrier
 - index ignoreres og resettes alltid og kolonner som har med index å gjøre fjernes fordi de kan gi unødvendige feilmeldinger
"""


def buff(
    geom: GeoDataFrame | GeoSeries | Geometry, 
    distance: int,
    resolution: int = 50, 
    copy: bool = True, 
    **kwargs
) -> GeoDataFrame | GeoSeries | Geometry:
    """
    buffer that returns same type of input and output, so returns GeoDataFrame when GeoDataFrame is input.
    buffers with higher resolution than the geopandas default.
    repairs geometry afterwards
    """

    if copy:
        geom = geom.copy()

    if isinstance(geom, gpd.GeoDataFrame):
        geom["geometry"] = geom.buffer(distance, resolution=resolution, **kwargs)
        geom["geometry"] = geom.make_valid()
    else:
        geom = geom.buffer(distance, resolution=resolution, **kwargs)
        geom = geom.make_valid()

    return geom


def diss(gdf: GeoDataFrame, by=None, aggfunc="sum", **kwargs) -> GeoDataFrame:
    """
    dissolve som ignorerer og resetter index
    med aggfunc='sum' som default fordi 'first' gir null mening
    hvis flere aggfuncs, gjøres kolonnene om til string, det vil si fra (kolonne, sum) til kolonne_sum
    """

    if isinstance(gdf, GeoSeries):
        return gpd.GeoSeries(gdf.unary_union)
    
    if isinstance(gdf, Geometry):
        return gdf.unary_union

    dissolvet = (gdf
                 .dissolve(by=by, aggfunc=aggfunc, **kwargs)
                 .reset_index()
    )

    dissolvet["geometry"] = dissolvet.make_valid()

    # gjør kolonner fra tuple til string
    dissolvet.columns = ["_".join(kolonne).strip("_") if isinstance(kolonne, tuple) else kolonne for kolonne in dissolvet.columns]

    return dissolvet.loc[:, ~dissolvet.columns.str.contains("index|level_")]


# denne funksjonen trengs egentlig ikke, bare la den til for å være konsistent med opplegget med buff, diss og exp
def exp(gdf, ignore_index=True, **kwargs) -> gpd.GeoDataFrame:
    """ 
    explode (til singlepart) som reparerer geometrien og ignorerer index som default (samme som i pandas). 
    reparerer før explode fordi reparering kan skape multigeometrier. 
    """
    gdf["geometry"] = gdf.make_valid()
    return gdf.explode(ignore_index=ignore_index, **kwargs)


def buffdissexp(gdf, distance, resolution=50, by=None, id=None, copy=True, **dissolve_kwargs) -> gpd.GeoDataFrame:
    """
    Bufrer og samler overlappende. Altså buffer, dissolve, explode (til singlepart).
    avstand: buffer-avstand
    resolution: buffer-oppløsning
    by: dissolve by
    id: navn på eventuell id-kolonne
    """

    gdf = (
        buff(gdf, distance, resolution=resolution)
        .pipe(diss, **dissolve_kwargs)
        .explode(ignore_index=True)
    )
    
    if copy:
        gdf = gdf.copy()

    if isinstance(gdf, gpd.GeoSeries):
        return (
            gpd.GeoSeries(gdf
                          .buffer(avstand, resolution=resolution)
                          .make_valid()
                          .unary_union)
            .make_valid()
            .explode(ignore_index=True)
        )

    gdf["geometry"] = gdf.buffer(avstand, resolution=resolution)
    gdf["geometry"] = gdf.make_valid()

    dissolvet = (gdf
                .dissolve(by=by, **dissolve_kwargs)
                .reset_index()
    )

    dissolvet["geometry"] = dissolvet.make_valid()

    # gjør kolonner fra tuple til string (hvis flere by-kolonner)
    dissolvet.columns = ["_".join(kolonne).strip("_") if isinstance(kolonne, tuple) else kolonne for kolonne in dissolvet.columns]

    singlepart = dissolvet.explode(ignore_index=True)

    if id:
        singlepart[id] = list(range(len(singlepart)))
    
    return singlepart.loc[:, ~singlepart.columns.str.contains("index|level_")]


def dissexp(gdf, by=None, id=None, **kwargs) -> gpd.GeoDataFrame:
    """
    Dissolve, explode (til singlepart). Altså dissolve overlappende.
    resolution: buffer-oppløsning
    by: dissolve by
    id: navn på eventuell id-kolonne
    """
    
    gdf["geometry"] = gdf.make_valid()
    
    dissolvet = (gdf
                .dissolve(by=by, **kwargs)
                .reset_index()
    )

    dissolvet["geometry"] = dissolvet.make_valid()

    # gjør kolonner fra tuple til string (hvis flere by-kolonner)
    dissolvet.columns = ["_".join(kolonne).strip("_") if isinstance(kolonne, tuple) else kolonne for kolonne in dissolvet.columns]

    singlepart = dissolvet.explode(ignore_index=True)

    if id:
        singlepart[id] = list(range(len(singlepart)))

    return singlepart.loc[:, ~singlepart.columns.str.contains("index|level_")]


def buffdiss(gdf, avstand, resolution=50, by=None, id=None, copy = True, **dissolve_kwargs) -> gpd.GeoDataFrame:
    """
    Buffer, dissolve.
    """
    
    if copy:
        gdf = gdf.copy()

    gdf["geometry"] = gdf.buffer(avstand, resolution=resolution)
    gdf["geometry"] = gdf.make_valid()
    
    dissolvet = (gdf
                .dissolve(by=by, **dissolve_kwargs)
                .reset_index()
    )

    dissolvet["geometry"] = dissolvet.make_valid()

    # gjør kolonner fra tuple til string (hvis flere by-kolonner)
    dissolvet.columns = ["_".join(kolonne).strip("_") if isinstance(kolonne, tuple) else kolonne for kolonne in dissolvet.columns]

    if id:
        dissolvet[id] = list(range(len(dissolvet)))
    
    return dissolvet.loc[:, ~dissolvet.columns.str.contains("index|level_")]


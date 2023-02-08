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


def close_holes(
    geom: GeoDataFrame | GeoSeries | Geometry, 
    max_km2: int | None = None, 
    copy: bool=True
) -> GeoDataFrame | GeoSeries | Geometry:
    """
    Closes holes in polygons. The operation is done row-wise if 'geom' is a GeoDataFrame or GeoSeries.

    max_km2: if None (default), all holes are closed. 
    Otherwise, closes holes with an area below the specified number in square kilometers.
    """

    if copy:
        geom = geom.copy()

    if isinstance(geom, gpd.GeoDataFrame):
        geom["geometry"] = geom.geometry.map(lambda x: _close_holes_geom(x, max_km2))

    elif isinstance(geom, gpd.GeoSeries):
        geom = geom.map(lambda x: _close_holes_geom(x, max_km2))
        geom = gpd.GeoSeries(geom)

    else:
        geom = _close_holes_geom(geom, max_km2)

    return geom


# flytte hvor?
from shapely import (
    polygons,
    get_exterior_ring,
    get_parts,
    get_num_interior_rings,
    get_interior_ring,
    area,
)
from shapely.ops import unary_union


def _close_holes_geom(geom, max_km2=None):
    """ closes holes within one shapely geometry. """

    # dissolve the exterior ring(s)
    if max_km2 is None:
        holes_closed = polygons(get_exterior_ring(get_parts(geom)))
        return unary_union(holes_closed)

    # start with list containing the geometry, then append all holes smaller than 'max_km2'.
    holes_closed = [geom]
    singlepart = get_parts(geom)
    for part in singlepart:
        antall_indre_ringer = get_num_interior_rings(part)
        if antall_indre_ringer > 0:
            for n in range(antall_indre_ringer):
                hull = polygons(get_interior_ring(part, n))
                if area(hull) / 1_000_000 < max_km2:
                    holes_closed.append(hull)
    return unary_union(holes_closed)
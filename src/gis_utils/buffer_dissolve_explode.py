import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
from shapely import Geometry, get_parts, make_valid
from shapely.ops import unary_union


"""
Functions that buffer, dissolve and/or explodes (multipart to singlepart) geodataframes.

Rules that apply to all functions:
 - higher buffer-oppløsning enn standard
 - reparerer geometrien etter buffer og dissolve, men ikke etter explode,
   siden reparering kan returnere multipart-geometrier
 - index ignoreres og resettes alltid og kolonner som har med index å gjøre fjernes
   fordi de kan gi unødvendige feilmeldinger
"""


def buff(
    gdf: GeoDataFrame | GeoSeries | Geometry,
    distance: int,
    resolution: int = 50,
    copy: bool = True,
    **kwargs,
) -> GeoDataFrame | GeoSeries | Geometry:
    """
    buffer that returns same type of input and output, so returns GeoDataFrame when
    GeoDataFrame is input.
    buffers with higher resolution than the geopandas default.
    repairs geometry afterwards
    """

    if copy and not isinstance(gdf, Geometry):
        gdf = gdf.copy()

    if isinstance(gdf, GeoDataFrame):
        gdf["geometry"] = gdf.buffer(distance, resolution=resolution, **kwargs)
        gdf["geometry"] = gdf.make_valid()
    elif isinstance(gdf, GeoSeries):
        gdf = gdf.buffer(distance, resolution=resolution, **kwargs)
        gdf = gdf.make_valid()
    elif isinstance(gdf, Geometry):
        gdf = gdf.buffer(distance, resolution=resolution, **kwargs)
        gdf = make_valid(gdf)
    else:
        raise TypeError(
            f"'gdf' should be GeoDataFrame, GeoSeries or shapely Geometry. "
            f"Got {type(gdf)}"
        )

    return gdf


def diss(
    gdf: GeoDataFrame | GeoSeries | Geometry,
    by=None,
    aggfunc="sum",
    reset_index=True,
    **kwargs,
) -> GeoDataFrame | GeoSeries | Geometry:
    """
    dissolve som ignorerer og resetter index
    med aggfunc='sum' som default fordi 'first' gir null mening
    hvis flere aggfuncs, gjøres kolonnene om til string, det vil si fra (kolonne, sum)
    til kolonne_sum
    """

    if isinstance(gdf, GeoSeries):
        return gpd.GeoSeries(gdf.unary_union)

    if isinstance(gdf, Geometry):
        return unary_union(gdf)

    if not isinstance(gdf, GeoDataFrame):
        raise TypeError(
            f"'gdf' should be GeoDataFrame, GeoSeries or shapely Geometry. "
            f"Got {type(gdf)}"
        )

    dissolved = gdf.dissolve(by=by, aggfunc=aggfunc, **kwargs)

    if reset_index:
        dissolved = dissolved.reset_index()

    dissolved["geometry"] = dissolved.make_valid()

    # gjør kolonner fra tuple til string
    dissolved.columns = [
        "_".join(kolonne).strip("_") if isinstance(kolonne, tuple) else kolonne
        for kolonne in dissolved.columns
    ]

    return dissolved.loc[:, ~dissolved.columns.str.contains("index|level_")]


def exp(
    gdf: GeoDataFrame | GeoSeries | Geometry,
    ignore_index=True,
    **kwargs,
) -> GeoDataFrame | GeoSeries | Geometry:
    """
    explode (til singlepart) som reparerer geometrien og ignorerer index som default
    (samme som i pandas).
    reparerer før explode fordi reparering kan skape multigeometrier.
    """
    if isinstance(gdf, GeoDataFrame):
        gdf["geometry"] = gdf.make_valid()
        return gdf.explode(ignore_index=ignore_index, **kwargs)

    elif isinstance(gdf, GeoSeries):
        gdf = gdf.make_valid()
        return gdf.explode(ignore_index=ignore_index, **kwargs)

    elif isinstance(gdf, Geometry):
        return get_parts(make_valid(gdf))

    else:
        raise TypeError(
            f"'gdf' should be GeoDataFrame, GeoSeries or shapely Geometry. "
            f"Got {type(gdf)}"
        )


def buffdissexp(
    gdf: GeoDataFrame | GeoSeries | Geometry,
    distance: int,
    resolution: int = 50,
    id=None,
    ignore_index=True,
    reset_index=True,
    copy=True,
    **dissolve_kwargs,
) -> GeoDataFrame | GeoSeries | Geometry:
    """
    Bufrer og samler overlappende. Altså buffer, dissolve, explode (til singlepart).
    distance: buffer-distance
    resolution: buffer-oppløsning
    by: dissolve by
    id: navn på eventuell id-kolonne
    """

    if isinstance(gdf, Geometry):
        return exp(diss(buff(gdf, distance, resolution=resolution)))

    gdf = (
        buff(gdf, distance, resolution=resolution, copy=copy)
        .pipe(diss, reset_index=reset_index, **dissolve_kwargs)
        .pipe(exp, ignore_index=ignore_index)
    )

    if id:
        gdf[id] = list(range(len(gdf)))

    return gdf


def dissexp(
    gdf: GeoDataFrame | GeoSeries | Geometry,
    id=None,
    reset_index=True,
    ignore_index=True,
    **kwargs,
) -> GeoDataFrame | GeoSeries | Geometry:
    """
    Dissolve, explode (til singlepart). Altså dissolve overlappende.
    resolution: buffer-oppløsning
    by: dissolve by
    id: navn på eventuell id-kolonne
    """

    gdf = diss(gdf, reset_index=reset_index, **kwargs).pipe(
        exp, ignore_index=ignore_index
    )

    if id:
        gdf[id] = list(range(len(gdf)))

    return gdf


def buffdiss(
    gdf,
    distance,
    resolution=50,
    id=None,
    reset_index=True,
    copy=True,
    **dissolve_kwargs,
) -> GeoDataFrame | GeoSeries | Geometry:
    """
    Buffer, dissolve.
    """

    gdf = buff(gdf, distance, resolution=resolution, copy=copy).pipe(
        diss, reset_index=reset_index, **dissolve_kwargs
    )

    if id:
        gdf[id] = list(range(len(gdf)))

    return gdf

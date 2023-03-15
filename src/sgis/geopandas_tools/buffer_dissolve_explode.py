"""Functions that buffer, dissolve and/or explodes geometries.

Rules that apply to all functions in the module:
 - higher buffer resolution (50) than the geopandas default (16).

 - Ignoring and reseting index by default and removes index-columns.

 - Input type is the same as the output type.

 - Fixes geometries after buffer and dissolve, but not after explode.
"""


import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
from shapely import Geometry, get_parts, make_valid
from shapely.ops import unary_union


def buff(
    gdf: GeoDataFrame | GeoSeries | Geometry,
    distance: int | float,
    resolution: int = 50,
    copy: bool = True,
    **kwargs,
) -> GeoDataFrame | GeoSeries | Geometry:
    """Buffers with high resolution and fixes geometries.

    It buffers a GeoDataFrame, GeoSeries, or Geometry, and returns the same type of
    object. Returns a copy of the object unless 'copy' is set to False. The default
    buffer resolution of 50 is higher than the geopandas default of 16 for the sake of
    accuaracy. Geometries are fixed after the buffer.

    Args:
        gdf: GeoDataFrame, GeoSeries or shapely Geometry
        distance: the distance (meters, degrees, depending on the crs) to buffer
            the geometry by.
        resolution: The number of segments used to approximate a quarter circle.
            Here defaults to 50, as opposed to the default 16 in geopandas.
        copy: if True (the default), the input geometry will not be buffered.
            Setting copy to False will save memory.
        **kwargs: Additional parameters to the GeoDataFrame buffer function.

    Returns:
         A GeoDataFrame with the buffered geometry.

    Raises:
        TypeError: If 'gdf' is not of type GeoDataFrame, GeoSeries or Geometry.
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
            "'gdf' should be GeoDataFrame, GeoSeries or shapely Geometry. "
            f"Got {type(gdf)}"
        )

    return gdf


def diss(
    gdf: GeoDataFrame | GeoSeries | Geometry,
    reset_index=True,
    **kwargs,
) -> GeoDataFrame | GeoSeries | Geometry:
    """Dissolves, fixes geometries, resets index tuple columns to string.

    It dissolves a GeoDataFrame, GeoSeries or Geometry, fixes the geometry,
    resets the index and makes columns from tuple to string if there are multiple
    aggfuncs (so the resulting columns will not be ('col', 'sum'), but 'col_sum').

    Args:
        gdf: the GeoDataFrame, GeoSeries or shapely Geometry that will be dissolved
        reset_index: If True, the 'by' columns become columns, not index, and the
            resulting axis will be labeled 0, 1, …, n - 1. Defaults to True.
        **kwargs: keyword arguments passed to geopandas' dissolve.

    Returns:
        A GeoDataFrame with the dissolved polygons.

    Raises:
        TypeError: If 'gdf' is not of type GeoDataFrame, GeoSeries or Geometry.
    """
    if isinstance(gdf, GeoSeries):
        return gpd.GeoSeries(gdf.unary_union)

    if isinstance(gdf, Geometry):
        return unary_union(gdf)

    if not isinstance(gdf, GeoDataFrame):
        raise TypeError(
            "'gdf' should be GeoDataFrame, GeoSeries or shapely Geometry. "
            f"Got {type(gdf)}"
        )

    dissolved = gdf.dissolve(**kwargs)

    if reset_index:
        dissolved = dissolved.reset_index()

    dissolved["geometry"] = dissolved.make_valid()

    # columns from tuple to string
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
    """Fixes geometries before exploding from multipart to singlepart geometries.

    Args:
        gdf: the GeoDataFrame, GeoSeries or shapely Geometry that will be exploded
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Defaults to True
        **kwargs: keyword arguments passed to geopandas' ecplode.

    Returns:
        A GeoDataFrame, GeoSeries or shapely Geometry with singlepart geometries.

    Raises:
        TypeError: If 'gdf' is not of type GeoDataFrame, GeoSeries or Geometry.
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
            "'gdf' should be GeoDataFrame, GeoSeries or shapely Geometry. "
            f"Got {type(gdf)}"
        )


def buffdissexp(
    gdf: GeoDataFrame | GeoSeries | Geometry,
    distance: int | float,
    resolution: int = 50,
    id: str | None = None,
    ignore_index: bool = True,
    reset_index: bool = True,
    copy: bool = True,
    **dissolve_kwargs,
) -> GeoDataFrame | GeoSeries | Geometry:
    """Buffers and dissolves overlapping geometries.

    It takes a GeoDataFrame, GeoSeries or shapely geometry, and buffer, dissolves and
    explodes (to singlepart). It uses the functions buff, diss and exp.

    Args:
        gdf: the GeoDataFrame, GeoSeries or shapely Geometry that will be
            buffered, dissolved and exploded
        distance: the distance (meters, degrees, depending on the crs) to buffer
            the geometry by
        resolution: The number of segments used to approximate a quarter circle.
            Here defaults to 50, as opposed to the default 16 in geopandas.
        copy: if True (the default), the input geometry will not be buffered.
            Setting copy to False will save memory.
        id: if not None (the default), an id column will be created
            from the integer index (from 0 and up).
        reset_index: If True, the index is reset to the default integer index
            after dissolve. Defaults to True
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Defaults to True
        **dissolve_kwargs: keyword arguments passed to geopandas' dissolve.

    Returns:
        A buffered GeoDataFrame, GeoSeries or shapely Geometry where overlapping
        geometries are dissolved.

    Raises:
        TypeError: If 'gdf' is not of type GeoDataFrame, GeoSeries or Geometry.
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
    id: str | None = None,
    reset_index: bool = True,
    ignore_index: bool = True,
    **dissolve_kwargs,
) -> GeoDataFrame | GeoSeries | Geometry:
    """Dissolves overlapping geometries. So dissolve and explode (to singlepart).

    Args:
        gdf: the GeoDataFrame, GeoSeries or shapely Geometry that will be
            dissolved and exploded
        id: if not None (the default), an id column will be created from
            the integer index (from 0 and up).
        reset_index: If True, the 'by' columns become columns, not index, and the
            resulting axis will be labeled 0, 1, …, n - 1. Defaults to True.
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Defaults to True
        **dissolve_kwargs: keyword arguments passed to geopandas' dissolve.

    Returns:
        A GeoDataFrame, GeoSeries or shapely Geometry where overlapping geometries are
        dissolved.

    Raises:
        TypeError: If 'gdf' is not of type GeoDataFrame, GeoSeries or Geometry.
    """
    gdf = diss(gdf, reset_index=reset_index, **dissolve_kwargs).pipe(
        exp, ignore_index=ignore_index
    )

    if id:
        gdf[id] = list(range(len(gdf)))

    return gdf


def buffdiss(
    gdf: GeoDataFrame | GeoSeries | Geometry,
    distance: int | float,
    resolution: int = 50,
    id: str | None = None,
    reset_index: bool = True,
    copy: bool = True,
    **dissolve_kwargs,
) -> GeoDataFrame | GeoSeries | Geometry:
    """Buffers and dissolves all geometries.

    Args:
        gdf: the GeoDataFrame, GeoSeries or shapely Geometry that will be
            buffered and dissolved
        distance: the distance (meters, degrees, depending on the crs) to buffer
            the geometry by.
        resolution: The number of segments used to approximate a quarter circle.
            Here defaults to 50, as opposed to the default 16 in geopandas.
        copy: if True (the default), the input geometry will not be buffered.
            Setting copy to False will save memory.
        id: if not None (the default), an id column will be created from
            the integer index (from 0 and up).
        reset_index: If True, the index is reset to the default integer index after
            dissolve. Defaults to True
        **dissolve_kwargs: keyword arguments passed to geopandas' dissolve.

    Returns:
          A buffered GeoDataFrame, GeoSeries or shapely Geometry where all geometries
          are dissolved.

    Raises:
        TypeError: If 'gdf' is not of type GeoDataFrame, GeoSeries or Geometry.
    """
    gdf = buff(gdf, distance, resolution=resolution, copy=copy).pipe(
        diss, reset_index=reset_index, **dissolve_kwargs
    )

    if id:
        gdf[id] = list(range(len(gdf)))

    return gdf

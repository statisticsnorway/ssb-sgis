"""Functions that buffer, dissolve and/or explodes geometries while fixing geometries.

Functions with the purpose of making the code cleaner when buffering, dissolving,
exploding and repairing geometries. The functions are identical to doing buffer,
dissolve and explode individually, except for the following:

- Geometries are always repaired after buffer and dissolve.

- The buffer resolution defaults to 50, while geopandas' default is 16.

- The buff function returns a GeoDataFrame, the geopandas method returns a GeoSeries.

- index_parts is set to False, which will be the default value in a future version of geopandas.
"""

from geopandas import GeoDataFrame


IGNORE_INDEX_ERROR_MESSAGE = (
    "Cannot set ignore_index. Set as_index=False to reset the index and keep "
    "the 'by' columns. Or use reset_index(drop=True) to remove the 'by' "
    "columns completely"
)


def buffdissexp(
    gdf: GeoDataFrame,
    distance,
    *,
    resolution=50,
    index_parts: bool = False,
    copy: bool = True,
    **dissolve_kwargs,
) -> GeoDataFrame:
    """Buffers and dissolves overlapping geometries.

    It takes a GeoDataFrame and buffer, fixes, dissolves, fixes and explodes geometries.

    Args:
        gdf: the GeoDataFrame that will be buffered, dissolved and exploded.
        distance: the distance (meters, degrees, depending on the crs) to buffer
            the geometry by
        resolution: The number of segments used to approximate a quarter circle.
            Here defaults to 50, as opposed to the default 16 in geopandas.
        index_parts: If False (default), the index after dissolve is respected. If
            True, an integer index level is added during explode.
        copy: Whether to copy the GeoDataFrame before buffering.
        **dissolve_kwargs: additional keyword arguments passed to geopandas' dissolve.
            The most important are:
                by: string or list of strings represeting the columns to dissolve by.
                aggfunc: function, string representing function or dictionary of column
                    name and function name/string.
                as_index: Whether the 'by' columns should be returned as index or
                    column. Defaults to True.

    Returns:
        A buffered GeoDataFrame where overlapping geometries are dissolved.

    Examples
    --------

    Buffer 100 meters and dissolve all overlapping geometries.

    Dissolve overlapping geometries with the same values in ''..

    Get 'by' columns as columns, not index.

    How to aggregate values can be specified with 'aggfunc'. Keep in mind that these
    values do not make sense after exploding the geometries.

    >>>
    """
    if "ignore_index" in dissolve_kwargs:
        raise ValueError(IGNORE_INDEX_ERROR_MESSAGE)

    geom_col = gdf._geometry_column_name

    buffered = buff(gdf, distance, resolution=resolution, copy=copy)

    dissolved = buffered.dissolve(**dissolve_kwargs)

    dissolved[geom_col] = dissolved.make_valid()

    return dissolved.explode(index_parts=index_parts)


def buffdiss(
    gdf: GeoDataFrame,
    distance,
    resolution=50,
    copy: bool = True,
    **dissolve_kwargs,
) -> GeoDataFrame:
    """Buffers and dissolves geometries.

    It takes a GeoDataFrame and buffer, fixes, dissolves and fixes geometries.

    Args:
        gdf: the GeoDataFrame that will be
            buffered and dissolved.
        distance: the distance (meters, degrees, depending on the crs) to buffer
            the geometry by
        resolution: The number of segments used to approximate a quarter circle.
            Here defaults to 50, as opposed to the default 16 in geopandas.
        copy: Whether to copy the GeoDataFrame before buffering.
        **dissolve_kwargs: additional keyword arguments passed to geopandas' dissolve.
            The most important are:
                by: string or list of strings represeting the columns to dissolve by.
                aggfunc: function, string representing function or dictionary of column
                    name and function name/string.
                as_index: Whether the 'by' columns should be returned as index or
                    column. Defaults to True.

    Returns:
        A buffered GeoDataFrame where geometries are dissolved.

    Examples
    --------
    Create some random points.
    >>> import sgis as sg
    >>> import numpy as np
    >>> points = sg.random_points(100)
    >>> points["group"] = np.random.choice([*"abd"], len(points))
    >>> points["number"] = np.random.random(size=len(points))
    >>> points
                    geometry group    number
    0   POINT (0.63331 0.85744)     a  0.279622
    1   POINT (0.98536 0.43368)     d  0.686433
    2   POINT (0.00099 0.55209)     d  0.943445
    3   POINT (0.18126 0.87312)     d  0.757552
    4   POINT (0.14582 0.73144)     b  0.259795
    ..                      ...   ...       ...
    95  POINT (0.19386 0.13392)     a  0.741990
    96  POINT (0.38129 0.43777)     d  0.296729
    97  POINT (0.86136 0.83022)     d  0.562658
    98  POINT (0.48929 0.08860)     d  0.937703
    99  POINT (0.57027 0.15667)     a  0.312668

    [100 rows x 3 columns]

    Buffer by 0.5.
    >>> sg.buffdiss(points, 0.5)
                                                geometry group    number
    0  POLYGON ((0.20196 -0.46016, 0.18635 -0.45843, ...     a  0.279622

    Buffer by group and summarise columns.

    >>> sg.buffdiss(points, 0.5, by="group", aggfunc="sum")
                                                    geometry     number
    group
    a      POLYGON ((0.86263 -0.33061, 0.85170 -0.34189, ...  15.143742
    b      POLYGON ((0.58850 -0.31011, 0.57565 -0.31914, ...  12.049528
    d      POLYGON ((0.24902 -0.46238, 0.23331 -0.46213, ...  22.079456

    To get the 'by' columns as columns, not index.

    >>> sg.buffdiss(points, 0.5, by="group", as_index=False)
      group                                           geometry    number
    0     a  POLYGON ((0.86263 -0.33061, 0.85170 -0.34189, ...  0.279622
    1     b  POLYGON ((0.58850 -0.31011, 0.57565 -0.31914, ...  0.259795
    2     d  POLYGON ((0.24902 -0.46238, 0.23331 -0.46213, ...  0.686433

    If doing different aggregate functions, it might be a good idea to specify
    each in groupby.agg, then join these columns with the dissolved geometries.

    >>> aggcols = points.groupby("group").agg(
    ...     numbers_sum=("number", "count"),
    ...     numbers_mean=("number", "mean"),
    ...     n=("number", "count"),
    ... )
    >>> points_agg = (
    ...     sg.buffdiss(points, 0.5, by="group")
    ...     [["geometry"]]
    ...     .join(aggcols)
    ...     .reset_index()
    ... )
    >>> points_agg
      group                                           geometry  numbers_sum  numbers_mean   n
    0     a  POLYGON ((0.86263 -0.33061, 0.85170 -0.34189, ...           32      0.473242  32
    1     b  POLYGON ((0.58850 -0.31011, 0.57565 -0.31914, ...           27      0.446279  27
    2     d  POLYGON ((0.24902 -0.46238, 0.23331 -0.46213, ...           41      0.538523  41

    """
    if "ignore_index" in dissolve_kwargs:
        raise ValueError(IGNORE_INDEX_ERROR_MESSAGE)

    geom_col = gdf._geometry_column_name

    buffered = buff(gdf, distance, resolution=resolution, copy=copy)

    dissolved = buffered.dissolve(**dissolve_kwargs)

    dissolved[geom_col] = dissolved.make_valid()

    return dissolved


def dissexp(
    gdf: GeoDataFrame,
    index_parts: bool = False,
    **dissolve_kwargs,
):
    """Dissolves overlapping geometries.

    It takes a GeoDataFrame and dissolves, fixes and explodes geometries.

    Args:
        gdf: the GeoDataFrame that will be buffered, dissolved and exploded.
        index_parts: If False (default), the index after dissolve is respected. If
            True, an integer index level is added during explode.
        **dissolve_kwargs: additional keyword arguments passed to geopandas' dissolve.
            The most important are:
                by: string or list of strings represeting the columns to dissolve by.
                aggfunc: function, string representing function or dictionary of column
                    name and function name/string.
                as_index: Whether the 'by' columns should be returned as index or
                    column. Defaults to True.

    Returns:
        A GeoDataFrame where overlapping geometries are dissolved.

    Examples
    --------
    """
    if "ignore_index" in dissolve_kwargs:
        raise ValueError(IGNORE_INDEX_ERROR_MESSAGE)

    geom_col = gdf._geometry_column_name

    dissolved = gdf.dissolve(**dissolve_kwargs)

    dissolved[geom_col] = dissolved.make_valid()

    return dissolved.explode(index_parts=index_parts)


def buffexp(
    gdf,
    distance,
    resolution=50,
    index_parts: bool = False,
    copy: bool = True,
    **buffer_kwargs,
):
    """Buffers and explodes geometries.

    It takes a GeoDataFrame and buffer, fixes and explodes geometries.

    Args:
        gdf: the GeoDataFrame that will be buffered, dissolved and exploded.
        distance: the distance (meters, degrees, depending on the crs) to buffer
            the geometry by
        resolution: The number of segments used to approximate a quarter circle.
            Here defaults to 50, as opposed to the default 16 in geopandas.
        index_parts: If False (default), the index after dissolve is respected. If
            True, an integer index level is added during explode.
        copy: Whether to copy the GeoDataFrame before buffering.
        **buffer_kwargs: additional keyword arguments passed to geopandas' buffer.

    Returns:
        A buffered GeoDataFrame where geometries are exploded.

    Examples
    --------

    >>>
    """
    if "ignore_index" in buffer_kwargs:
        raise ValueError(IGNORE_INDEX_ERROR_MESSAGE)

    return buff(
        gdf, distance, resolution=resolution, copy=copy, **buffer_kwargs
    ).explode(index_parts=index_parts)


def buff(
    gdf: GeoDataFrame,
    distance: int | float,
    resolution: int = 50,
    copy: bool = True,
    **buffer_kwargs,
) -> GeoDataFrame:
    """Buffers a GeoDataFrame with high resolution and returns a new GeoDataFrame.

    Args:
        gdf: the GeoDataFrame that will be buffered, dissolved and exploded.
        distance: the distance (meters, degrees, depending on the crs) to buffer
            the geometry by
        resolution: The number of segments used to approximate a quarter circle.
            Here defaults to 50, as opposed to the default 16 in geopandas.
        copy: Whether to copy the GeoDataFrame before buffering.
        **buffer_kwargs: additional keyword arguments passed to geopandas' buffer.

    Returns:
        A buffered GeoDataFrame.

    Examples
    --------
    """
    geom_col = gdf._geometry_column_name

    if copy:
        gdf = gdf.copy()

    gdf[geom_col] = gdf.buffer(
        distance, resolution=resolution, **buffer_kwargs
    ).make_valid()

    return gdf

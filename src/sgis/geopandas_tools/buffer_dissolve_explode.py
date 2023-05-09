"""Functions that buffer, dissolve and/or explodes geometries while fixing geometries.

The functions do the same as the geopandas buffer, dissolve and explode methods, except
for the following:

- Geometries are made valid after buffer and dissolve.

- The buffer resolution defaults to 50 (geopandas' default is 16).

- If 'by' is not specified, the index will be labeled 0, 1, …, n - 1 after exploded, instead of 0, 0, …, 0 as it will with the geopandas defaults.

- index_parts is set to False, which will be the default in a future version of geopandas.

- The buff function returns a GeoDataFrame, the geopandas method returns a GeoSeries.
"""

from geopandas import GeoDataFrame, GeoSeries


def _decide_ignore_index(kwargs: dict) -> tuple[dict, bool]:
    if "ignore_index" in kwargs:
        return kwargs, kwargs.pop("ignore_index")

    if not kwargs.get("by", None):
        return kwargs, True

    if kwargs.get("as_index", True):
        return kwargs, False

    return kwargs, True


def buffdissexp(
    gdf: GeoDataFrame,
    distance: int | float,
    *,
    resolution: int = 50,
    index_parts: bool = False,
    copy: bool = True,
    **dissolve_kwargs,
) -> GeoDataFrame:
    """Buffers and dissolves overlapping geometries.

    It takes a GeoDataFrame and buffer, fixes, dissolves, fixes and explodes geometries.
    If the 'by' parameter is not specified, the index will labeled 0, 1, …, n - 1,
    instead of 0, 0, …, 0. If 'by' is speficied, this will be the index.

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

    Returns:
        A buffered GeoDataFrame where overlapping geometries are dissolved.

    """
    dissolve_kwargs, ignore_index = _decide_ignore_index(dissolve_kwargs)

    dissolved = buffdiss(
        gdf,
        distance,
        resolution=resolution,
        copy=copy,
        **dissolve_kwargs,
    )

    return dissolved.explode(index_parts=index_parts, ignore_index=ignore_index)


def buffdiss(
    gdf: GeoDataFrame,
    distance: int | float,
    resolution: int = 50,
    copy: bool = True,
    **dissolve_kwargs,
) -> GeoDataFrame:
    """Buffers and dissolves geometries.

    It takes a GeoDataFrame and buffer, fixes, dissolves and fixes geometries.
    If the 'by' parameter is not specified, the index will labeled 0, 1, …, n - 1,
    instead of 0, 0, …, 0. If 'by' is speficied, this will be the index.

    Args:
        gdf: the GeoDataFrame that will be
            buffered and dissolved.
        distance: the distance (meters, degrees, depending on the crs) to buffer
            the geometry by
        resolution: The number of segments used to approximate a quarter circle.
            Here defaults to 50, as opposed to the default 16 in geopandas.
        copy: Whether to copy the GeoDataFrame before buffering.
        **dissolve_kwargs: additional keyword arguments passed to geopandas' dissolve.

    Returns:
        A buffered GeoDataFrame where geometries are dissolved.

    Examples
    --------
    Create some random points.

    >>> import sgis as sg
    >>> import numpy as np
    >>> points = sg.read_parquet_url(
    ...     "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet"
    ... )[["geometry"]]
    >>> points["group"] = np.random.choice([*"abd"], len(points))
    >>> points["number"] = np.random.random(size=len(points))
    >>> points
                               geometry group    number
    0    POINT (263122.700 6651184.900)     a  0.878158
    1    POINT (272456.100 6653369.500)     a  0.693311
    2    POINT (270082.300 6653032.700)     b  0.323960
    3    POINT (259804.800 6650339.700)     a  0.606745
    4    POINT (272876.200 6652889.100)     a  0.194360
    ..                              ...   ...       ...
    995  POINT (266801.700 6647844.500)     a  0.814424
    996  POINT (261274.000 6653593.400)     b  0.769479
    997  POINT (263542.900 6645427.000)     a  0.925991
    998  POINT (269226.700 6650628.000)     b  0.431972
    999  POINT (264570.300 6644239.500)     d  0.555239

    Buffer by 100 meters and dissolve.

    >>> sg.buffdiss(points, 100)
                                                geometry group    number
    0  MULTIPOLYGON (((256421.833 6649878.117, 256420...     d  0.580157

    Dissolve by 'group' and get sum of columns.

    >>> sg.buffdiss(points, 100, by="group", aggfunc="sum")
                                                    geometry      number
    group
    a      MULTIPOLYGON (((258866.258 6648220.031, 258865...  167.265619
    b      MULTIPOLYGON (((258404.858 6647830.931, 258404...  171.939169
    d      MULTIPOLYGON (((258180.258 6647935.731, 258179...  156.964300

    To get the 'by' columns as columns, not index.

    >>> sg.buffdiss(points, 100, by="group", as_index=False)
      group                                           geometry    number
    0     a  MULTIPOLYGON (((258866.258 6648220.031, 258865...  0.323948
    1     b  MULTIPOLYGON (((258404.858 6647830.931, 258404...  0.687635
    2     d  MULTIPOLYGON (((258180.258 6647935.731, 258179...  0.580157
    """
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

    Returns:
        A GeoDataFrame where overlapping geometries are dissolved.
    """
    geom_col = gdf._geometry_column_name

    dissolve_kwargs, ignore_index = _decide_ignore_index(dissolve_kwargs)

    dissolved = gdf.dissolve(**dissolve_kwargs)

    dissolved[geom_col] = dissolved.make_valid()

    return dissolved.explode(index_parts=index_parts, ignore_index=ignore_index)


def buff(
    gdf: GeoDataFrame | GeoSeries,
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
    """

    if isinstance(gdf, GeoSeries):
        return gdf.buffer(distance, resolution=resolution, **buffer_kwargs).make_valid()

    geom_col = gdf._geometry_column_name

    if copy:
        gdf = gdf.copy()

    gdf[geom_col] = gdf.buffer(
        distance, resolution=resolution, **buffer_kwargs
    ).make_valid()

    return gdf

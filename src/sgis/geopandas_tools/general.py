import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryDtype
from shapely import (
    Geometry,
    box,
    force_2d,
    get_exterior_ring,
    get_interior_ring,
    get_num_interior_rings,
    get_parts,
    wkt,
)
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .geometry_types import to_single_geom_type
from .to_geodataframe import to_gdf


def coordinate_array(
    gdf: GeoDataFrame | GeoSeries,
) -> np.ndarray[np.ndarray[float], np.ndarray[float]]:
    """Creates a 2d ndarray of coordinates from point geometries.

    Args:
        gdf: GeoDataFrame or GeoSeries of point geometries.

    Returns:
        np.ndarray of np.ndarrays of coordinates.

    Examples
    --------
    >>> from sgis import coordinate_array, random_points
    >>> points = random_points(5)
    >>> points
                    geometry
    0  POINT (0.59376 0.92577)
    1  POINT (0.34075 0.91650)
    2  POINT (0.74841 0.10627)
    3  POINT (0.00966 0.87868)
    4  POINT (0.38046 0.87879)
    >>> coordinate_array(points)
    array([[0.59376221, 0.92577159],
        [0.34074678, 0.91650446],
        [0.74840912, 0.10626954],
        [0.00965935, 0.87867915],
        [0.38045827, 0.87878816]])
    >>> coordinate_array(points.geometry)
    array([[0.59376221, 0.92577159],
        [0.34074678, 0.91650446],
        [0.74840912, 0.10626954],
        [0.00965935, 0.87867915],
        [0.38045827, 0.87878816]])
    """
    if isinstance(gdf, GeoDataFrame):
        return np.array([(geom.x, geom.y) for geom in gdf.geometry])
    else:
        return np.array([(geom.x, geom.y) for geom in gdf])


def _push_geom_col(gdf: GeoDataFrame) -> GeoDataFrame:
    """Makes the geometry column the rightmost column in the GeoDataFrame.

    Args:
        gdf: GeoDataFrame.

    Returns:
        The GeoDataFrame with the geometry column pushed all the way to the right.
    """
    geom_col = gdf._geometry_column_name
    return gdf.reindex(columns=[c for c in gdf.columns if c != geom_col] + [geom_col])


def drop_inactive_geometry_columns(gdf: GeoDataFrame) -> GeoDataFrame:
    for col in gdf.columns:
        if (
            isinstance(gdf[col].dtype, GeometryDtype)
            and col != gdf._geometry_column_name
        ):
            gdf = gdf.drop(col, axis=1)
    return gdf


def rename_geometry_if(gdf: GeoDataFrame) -> GeoDataFrame:
    geom_col = gdf._geometry_column_name
    if geom_col == "geometry" and geom_col in gdf.columns:
        return gdf
    elif geom_col in gdf.columns:
        return gdf.rename_geometry("geometry")

    geom_cols = list(
        {col for col in gdf.columns if isinstance(gdf[col].dtype, GeometryDtype)}
    )
    if len(geom_cols) == 1:
        gdf._geometry_column_name = geom_cols[0]
        return gdf.rename_geometry("geometry")

    raise ValueError(
        "There are multiple geometry columns and none are the active geometry"
    )


def make_grid_from_bbox(
    minx: int | float,
    miny: int | float,
    maxx: int | float,
    maxy: int | float,
    *_,
    gridsize: int | float,
    crs,
) -> GeoDataFrame:
    """Creates a polygon grid from a bounding box.

    Creates a GeoDataFrame of grid cells of a given size within the given
    maxumum and mimimum x and y values.

    Args:
        minx: Minumum x coordinate.
        miny: Minumum y coordinate.
        maxx: Maximum x coordinate.
        maxy: Maximum y coordinate.
        gridsize: Length of the grid walls.
        crs: Coordinate reference system.

    Returns:
        GeoDataFrame with grid geometries.
    """
    grid_cells1 = []
    grid_cells2 = []
    grid_cells3 = []
    grid_cells4 = []
    for x0 in np.arange(minx, maxx + gridsize, gridsize):
        for y0 in np.arange(miny, maxy + gridsize, gridsize):
            x1 = x0 - gridsize
            y1 = y0 + gridsize
            grid_cells1.append(x0)
            grid_cells2.append(y0)
            grid_cells3.append(x1)
            grid_cells4.append(y1)

    grid_cells = box(grid_cells1, grid_cells2, grid_cells3, grid_cells4)

    return gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs=crs)


def make_grid(gdf: GeoDataFrame, gridsize: int | float) -> GeoDataFrame:
    """Create a polygon grid around a GeoDataFrame.

    Creates a GeoDataFrame of grid cells of a given size within the bounds of
    a given GeoDataFrame.

    Args:
        gdf: A GeoDataFrame.
        gridsize: Length of the grid walls.

    Returns:
        GeoDataFrame with grid polygons.
    """
    minx, miny, maxx, maxy = gdf.total_bounds
    return make_grid_from_bbox(minx, miny, maxx, maxy, gridsize=gridsize, crs=gdf.crs)


def make_ssb_grid(gdf: GeoDataFrame, gridsize: int = 1000) -> GeoDataFrame:
    """Creates a polygon grid around a GeoDataFrame with an SSB id column.

    Creates a grid that follows the grids produced by Statistics Norway.
    The GeoDataFrame must have 25833 as crs (UTM 33 N).

    Courtesy https://gis.stackexchange.com/questions/269243/creating-polygon-grid-using-geopandas

    Args:
        gdf: A GeoDataFrame.
        gridsize: Size of the grid in meters.

    Returns:
        GeoDataFrame with grid geometries and a column 'SSBID'.

    Raises:
        ValueError: If the GeoDataFrame does not have 25833 as crs.
    """
    if gdf.crs != 25833:
        raise ValueError(
            "Geodataframe must have crs = 25833. Use df.set_crs(25833) to set "
            "projection or df.to_crs(25833) for transforming."
        )

    minx, miny, maxx, maxy = gdf.total_bounds

    # Adjust for SSB-grid
    minx = int(minx / int(gridsize)) * int(gridsize)
    miny = int(miny / int(gridsize)) * int(gridsize)

    cols = list(np.arange(minx, maxx + gridsize, gridsize))
    rows = list(np.arange(miny, maxy + gridsize, gridsize))

    polygons = []
    for x in cols[:-1]:
        for y in rows[:-1]:
            polygons.append(
                Polygon(
                    [
                        (x, y),
                        (x + gridsize, y),
                        (x + gridsize, y + gridsize),
                        (x, y + gridsize),
                    ]
                )
            )

    grid = gpd.GeoDataFrame({"geometry": polygons}, crs=25833)

    # Make SSB-id
    grid["ostc"] = (
        (np.floor((grid.geometry.centroid.x + 2000000) / gridsize) * gridsize).apply(
            int
        )
    ).apply(str)
    grid["nordc"] = (
        (np.floor((grid.geometry.centroid.y) / gridsize) * gridsize).apply(int)
    ).apply(str)
    grid["SSBID"] = grid["ostc"] + grid["nordc"]
    grid = grid.drop(columns=["ostc", "nordc"])
    return grid


def add_grid_id(
    gdf: GeoDataFrame, gridsize: int, out_column: str = "SSBID"
) -> GeoDataFrame:
    """Adds a grid ID column to a GeoDataFrame of points.

    The GeoDataFrame must have 25833 as crs (UTM 33 N).

    Args:
        gdf: A GeoDataFrame.
        gridsize: Size of the grid in meters.

    Returns:
        The input GeoDataFrame with a new grid id column.

    Raises:
        ValueError: If the GeoDataFrame does not have 25833 as crs.
    """
    if gdf.crs != 25833:
        raise ValueError(
            "Geodataframe must have crs = 25833. Use df.set_crs(25833) to set "
            "projection or df.to_crs(25833) for transforming."
        )
    midlrdf = gdf.copy()
    midlrdf["ostc"] = (
        (np.floor((midlrdf.geometry.x + 2000000) / gridsize) * gridsize).apply(int)
    ).apply(str)
    midlrdf["nordc"] = (
        (np.floor((midlrdf.geometry.y) / gridsize) * gridsize).apply(int)
    ).apply(str)
    midlrdf[out_column] = midlrdf["ostc"] + midlrdf["nordc"]
    midlrdf = midlrdf.drop(columns=["nordc", "ostc"])
    return midlrdf


def bounds_to_polygon(gdf: GeoDataFrame) -> GeoDataFrame:
    """Creates a box around the geometry in each row of a GeoDataFrame.

    Args:
        gdf: The GeoDataFrame.

    Returns:
        GeoDataFrame of box polygons with same length and index as 'gdf'.
    """
    bbox_each_row = [box(*arr) for arr in gdf.bounds.values]
    return to_gdf(bbox_each_row, index=gdf.index, crs=gdf.crs)


def bounds_to_points(gdf: GeoDataFrame) -> GeoDataFrame:
    """Creates a 4-noded multipoint around the geometry in each row of a GeoDataFrame.

    Args:
        gdf: The GeoDataFrame.

    Returns:
        GeoDataFrame of multipoints with same length and index as 'gdf'.
    """
    return bounds_to_polygon(gdf).pipe(to_multipoint)


def clean_geoms(
    gdf: GeoDataFrame | GeoSeries, ignore_index: bool = False
) -> GeoDataFrame | GeoSeries:
    """Fixes geometries and removes invalid, empty, NaN and None geometries.

    Args:
        gdf: GeoDataFrame or GeoSeries to be cleaned.
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Defaults to False

    Returns:
        GeoDataFrame or GeoSeries with fixed geometries and only the rows with valid,
        non-empty and not-NaN/-None geometries.

    Examples
    --------
    >>> from sgis import clean_geoms, to_gdf
    >>> import pandas as pd
    >>> from shapely import wkt
    >>> gdf = to_gdf([
    ...         "POINT (0 0)",
    ...         "LINESTRING (1 1, 2 2)",
    ...         "POLYGON ((3 3, 4 4, 3 4, 3 3))"
    ...         ])
    >>> gdf
                                                geometry
    0                            POINT (0.00000 0.00000)
    1      LINESTRING (1.00000 1.00000, 2.00000 2.00000)
    2  POLYGON ((3.00000 3.00000, 4.00000 4.00000, 3....

    Removing None and empty geometries.

    >>> missing = pd.DataFrame({"geometry": [None]})
    >>> empty = to_gdf(wkt.loads("POINT (0 0)").buffer(0))
    >>> gdf = pd.concat([gdf, missing, empty])
    >>> gdf
                                                geometry
    0                            POINT (0.00000 0.00000)
    1      LINESTRING (1.00000 1.00000, 2.00000 2.00000)
    2  POLYGON ((3.00000 3.00000, 4.00000 4.00000, 3....
    0                                               None
    0                                      POLYGON EMPTY
    >>> clean_geoms(gdf)
                                                geometry
    0                            POINT (0.00000 0.00000)
    1      LINESTRING (1.00000 1.00000, 2.00000 2.00000)
    2  POLYGON ((3.00000 3.00000, 4.00000 4.00000, 3....
    """
    warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)

    if isinstance(gdf, GeoDataFrame):
        geom_col = gdf._geometry_column_name
        gdf[geom_col] = gdf.make_valid()
        gdf = gdf.loc[
            (gdf[geom_col].is_valid)
            & (~gdf[geom_col].is_empty)
            & (gdf[geom_col].notna())
        ]
    elif isinstance(gdf, GeoSeries):
        gdf = gdf.make_valid()
        gdf = gdf.loc[(gdf.is_valid) & (~gdf.is_empty) & (gdf.notna())]
    else:
        raise TypeError(f"'gdf' should be GeoDataFrame or GeoSeries, got {type(gdf)}")

    if ignore_index:
        gdf = gdf.reset_index(drop=True)

    return gdf


def random_points(n: int, loc: float | int = 0.5) -> GeoDataFrame:
    """Creates a GeoDataFrame with n random points.

    Args:
        n: Number of points/rows to create.
        loc: Mean ('centre') of the distribution.

    Returns:
        A GeoDataFrame of points with n rows.

    Examples
    --------
    >>> from sgis import random_points
    >>> points = random_points(10_000)
    >>> points
                         geometry
    0     POINT (0.62044 0.22805)
    1     POINT (0.31885 0.38109)
    2     POINT (0.39632 0.61130)
    3     POINT (0.99401 0.35732)
    4     POINT (0.76403 0.73539)
    ...                       ...
    9995  POINT (0.90433 0.75080)
    9996  POINT (0.10959 0.59785)
    9997  POINT (0.00330 0.79168)
    9998  POINT (0.90926 0.96215)
    9999  POINT (0.01386 0.22935)
    [10000 rows x 1 columns]

    Values with a mean of 100.

    >>> points = random_points(10_000, loc=100)
    >>> points
                         geometry
    0      POINT (50.442 199.729)
    1       POINT (26.450 83.367)
    2     POINT (111.054 147.610)
    3      POINT (93.141 141.456)
    4       POINT (94.101 24.837)
    ...                       ...
    9995   POINT (174.344 91.772)
    9996    POINT (95.375 11.391)
    9997    POINT (45.694 60.843)
    9998   POINT (73.261 101.881)
    9999  POINT (134.503 168.155)
    [10000 rows x 1 columns]
    """
    if isinstance(n, (str, float)):
        n = int(n)

    x = np.random.rand(n) * float(loc) * 2
    y = np.random.rand(n) * float(loc) * 2

    return GeoDataFrame(
        (Point(x, y) for x, y in zip(x, y, strict=True)), columns=["geometry"]
    )


def random_points_in_polygons(
    gdf: GeoDataFrame | GeoSeries, n: int, ignore_index=False
) -> GeoDataFrame:
    """Creates n random points inside each polygon of a GeoDataFrame.

    Args:
        gdf: GeoDataFrame to use as mask for the points.
        n: Number of points to create per polygon in 'gdf'.
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Defaults to False, meaning the points will have the index of the polygon
            it is within.

    Returns:
        A GeoDataFrame of points with 'n' rows per row in 'gdf'. It uses the index
        values of 'gdf'.

    Examples
    --------
    First create and buffer 100 random points.

    >>> import sgis as sg
    >>> gdf = sg.random_points(100)
    >>> polygons = sg.buff(gdf, 1)
    >>> polygons
                                                 geometry
    0   POLYGON ((1.49436 0.36088, 1.49387 0.32947, 1....
    1   POLYGON ((1.38427 0.21069, 1.38378 0.17928, 1....
    2   POLYGON ((1.78894 0.94134, 1.78845 0.90992, 1....
    3   POLYGON ((1.47174 0.81259, 1.47125 0.78118, 1....
    4   POLYGON ((1.13941 0.20821, 1.13892 0.17680, 1....
    ..                                                ...
    95  POLYGON ((1.13462 0.18908, 1.13412 0.15767, 1....
    96  POLYGON ((1.96391 0.43191, 1.96342 0.40050, 1....
    97  POLYGON ((1.30569 0.46956, 1.30520 0.43815, 1....
    98  POLYGON ((1.18172 0.10944, 1.18122 0.07803, 1....
    99  POLYGON ((1.06156 0.99893, 1.06107 0.96752, 1....
    [100 rows x 1 columns]

    >>> points = sg.random_points_in_polygons(polygons, 3)
    >>> points
                        geometry
    0   POINT (0.74944 -0.41658)
    0    POINT (1.27490 0.54076)
    0    POINT (0.22523 0.49323)
    1   POINT (0.25302 -0.34825)
    1    POINT (0.21124 0.89223)
    ..                       ...
    98  POINT (-0.39865 0.87135)
    98   POINT (0.03573 0.50788)
    99  POINT (-0.79089 0.57835)
    99   POINT (0.39838 1.50881)
    99   POINT (0.98383 0.77298)
    [300 rows x 1 columns]
    """
    if not isinstance(gdf, GeoDataFrame):
        gdf = to_gdf(gdf)

    if not all(gdf.geom_type.isin(["Polygon", "MultiPolygon"])):
        raise ValueError("Geometry types must be polygon.")

    if gdf.index.is_unique:
        gdf["temp_idx____"] = gdf.index
    else:
        gdf["temp_idx____"] = range(len(gdf))

    all_points = pd.DataFrame()

    for _ in range(n):
        bounds = gdf.bounds
        temp_idx____ = gdf["temp_idx____"].values
        overlapping = pd.DataFrame()
        overlapping_indices = ()

        while len(bounds):
            xs = np.random.uniform(bounds.minx, bounds.maxx)
            ys = np.random.uniform(bounds.miny, bounds.maxy)

            points_df = pd.DataFrame({"x": xs, "y": ys}, index=temp_idx____)

            points = to_gdf(points_df, geometry=["x", "y"], crs=gdf.crs).drop(
                ["x", "y"], axis=1
            )

            overlapping = points.sjoin(gdf[["temp_idx____", "geometry"]], how="inner")

            overlapping = overlapping.loc[overlapping.index == overlapping.temp_idx____]

            all_points = pd.concat([all_points, overlapping], ignore_index=ignore_index)

            overlapping_indices = overlapping_indices + tuple(overlapping.index.values)

            gdf__ = gdf.loc[~gdf["temp_idx____"].isin(overlapping_indices)]
            temp_idx____ = gdf__["temp_idx____"].values
            bounds = gdf__.bounds

    all_points = all_points.sort_index()

    all_points = all_points.loc[
        :, ~all_points.columns.str.contains("temp_idx____|index_right")
    ]

    if gdf.index.is_unique:
        gdf = gdf.drop("temp_idx____", axis=1)
        return all_points

    original_index = {
        temp_idx: idx for temp_idx, idx in zip(gdf.temp_idx____, gdf.index)
    }

    all_points.index = all_points.index.map(original_index)
    all_points.index.name = None

    gdf = gdf.drop("temp_idx____", axis=1)

    return all_points


def points_in_bounds(gdf: GeoDataFrame | GeoSeries, n2: int):
    minx, miny, maxx, maxy = gdf.total_bounds
    xs = np.linspace(minx, maxx, num=n2)
    ys = np.linspace(miny, maxy, num=n2)
    x_coords, y_coords = np.meshgrid(xs, ys, indexing="ij")
    coords = np.concatenate((x_coords.reshape(-1, 1), y_coords.reshape(-1, 1)), axis=1)
    return to_gdf(coords, crs=gdf.crs)


def to_lines(*gdfs: GeoDataFrame, copy: bool = True) -> GeoDataFrame:
    """Makes lines out of one or more GeoDataFrames and splits them at intersections.

    The GeoDataFrames' geometries are converted to LineStrings, then unioned together
    and made to singlepart. The lines are split at the intersections. Mimics
    'feature to line' in ArcGIS.

    Args:
        *gdfs: one or more GeoDataFrames.
        copy: whether to take a copy of the incoming GeoDataFrames. Defaults to True.

    Returns:
        A GeoDataFrame with singlepart line geometries and columns of all input
            GeoDataFrames.

    Note:
        The index is preserved if only one GeoDataFrame is given, but otherwise
        ignored. This is because the union overlay used if multiple GeoDataFrames
        always ignores the index.

    Examples
    --------
    Convert single polygon to linestring.

    >>> import sgis as sg
    >>> from shapely.geometry import Polygon
    >>> poly1 = sg.to_gdf(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]))
    >>> poly1["poly1"] = 1
    >>> line = sg.to_lines(poly1)
    >>> line
                                                geometry  poly1
    0  LINESTRING (0.00000 0.00000, 0.00000 1.00000, ...      1

    Convert two overlapping polygons to linestrings.

    >>> poly2 = sg.to_gdf(Polygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)]))
    >>> poly2["poly2"] = 1
    >>> lines = sg.to_lines(poly1, poly2)
    >>> lines
    poly1  poly2                                           geometry
    0    1.0    NaN  LINESTRING (0.00000 0.00000, 0.00000 1.00000, ...
    1    1.0    NaN  LINESTRING (0.50000 1.00000, 1.00000 1.00000, ...
    2    1.0    NaN  LINESTRING (1.00000 0.50000, 1.00000 0.00000, ...
    3    NaN    1.0      LINESTRING (0.50000 0.50000, 0.50000 1.00000)
    4    NaN    1.0  LINESTRING (0.50000 1.00000, 0.50000 1.50000, ...
    5    NaN    1.0      LINESTRING (1.00000 0.50000, 0.50000 0.50000)

    Plot before and after.

    >>> sg.qtm(poly1, poly2)
    >>> lines["l"] = lines.length
    >>> sg.qtm(lines, "l")
    """

    if any(any(gdf.geom_type.isin(["Point", "MultiPoint"])) for gdf in gdfs):
        raise ValueError("Cannot convert points to lines.")

    def _shapely_geometry_to_lines(geom):
        if geom.area == 0:
            return geom

        singlepart = get_parts(geom)
        lines = []
        for part in singlepart:
            exterior_ring = get_exterior_ring(part)
            lines.append(exterior_ring)

            n_interior_rings = get_num_interior_rings(part)
            if not (n_interior_rings):
                continue

            interior_rings = [
                LineString(get_interior_ring(part, n)) for n in range(n_interior_rings)
            ]

            lines = lines + interior_rings

        return unary_union(lines)

    lines = []
    for gdf in gdfs:
        if copy:
            gdf = gdf.copy()

        gdf[gdf._geometry_column_name] = gdf[gdf._geometry_column_name].map(
            _shapely_geometry_to_lines
        )

        gdf = to_single_geom_type(gdf, "line")

        lines.append(gdf)

    if len(lines) == 1:
        return lines[0]

    unioned = lines[0].overlay(lines[1], how="union", keep_geom_type=True)

    if len(lines) > 2:
        for line_gdf in lines[2:]:
            unioned = unioned.overlay(line_gdf, how="union", keep_geom_type=True)

    return unioned.explode(ignore_index=True)


def to_multipoint(
    gdf: GeoDataFrame | GeoSeries, copy: bool = True
) -> GeoDataFrame | GeoSeries:
    """Creates multipoint geometries from GeoDataFrame or GeoSeries.

    Takes a GeoDataFrame or GeoSeries and turns it into a MultiPoint.

    Args:
        gdf: The geometry to be converted to MultiPoint.
        copy: If True, the geometry will be copied. Defaults to True.

    Returns:
        A GeoDataFrame or GeoSeries with MultiPoint geometries. If the input type
        if GeoDataFrame, the other columns will be preserved.

    Examples
    --------
    Let's create a GeoDataFrame with a point, a line and a polygon.

    >>> from sgis import to_multipoint, to_gdf
    >>> from shapely.geometry import LineString, Polygon
    >>> gdf = to_gdf([
    ...     (0, 0),
    ...     LineString([(1, 1), (2, 2)]),
    ...     Polygon([(3, 3), (4, 4), (3, 4), (3, 3)])
    ...     ])
    >>> gdf
                                                geometry
    0                            POINT (0.00000 0.00000)
    1      LINESTRING (1.00000 1.00000, 2.00000 2.00000)
    2  POLYGON ((3.00000 3.00000, 4.00000 4.00000, 3....

    >>> to_multipoint(gdf)
                                                geometry
    0                            POINT (0.00000 0.00000)
    1      MULTIPOINT (1.00000 1.00000, 2.00000 2.00000)
    2  MULTIPOINT (3.00000 3.00000, 3.00000 4.00000, ...
    """
    if copy:
        gdf = gdf.copy()

    if gdf.is_empty.any():
        raise ValueError("Cannot create multipoints from empty geometry.")

    def _to_multipoint(gdf):
        koordinater = "".join(
            [x for x in gdf.wkt if x.isdigit() or x.isspace() or x == "." or x == ","]
        ).strip()

        alle_punkter = [
            wkt.loads(f"POINT ({punkt.strip()})") for punkt in koordinater.split(",")
        ]

        return unary_union(alle_punkter)

    if isinstance(gdf, GeoDataFrame):
        gdf[gdf._geometry_column_name] = (
            gdf[gdf._geometry_column_name]
            .pipe(force_2d)
            .apply(lambda x: _to_multipoint(x))
        )

    elif isinstance(gdf, gpd.GeoSeries):
        gdf = force_2d(gdf)
        gdf = gdf.apply(lambda x: _to_multipoint(x))

    else:
        gdf = to_gdf(gdf)
        gdf["geometry"] = (
            gdf["geometry"].pipe(force_2d).apply(lambda x: _to_multipoint(x))
        )

    return gdf


def clean_clip(
    gdf: GeoDataFrame | GeoSeries,
    mask: GeoDataFrame | GeoSeries | Geometry,
    **kwargs,
) -> GeoDataFrame | GeoSeries:
    """Clips geometries to the mask extent and cleans the geometries.

    Geopandas.clip does a fast and durty clipping, with no guarantee for valid outputs.
    Here, the clipped geometries are made valid, and then empty, NaN and invalid
    geometries are removed.

    Args:
        gdf: GeoDataFrame or GeoSeries to be clipped
        mask: the geometry to clip gdf
        **kwargs: Additional keyword arguments passed to GeoDataFrame.clip

    Returns:
        The cleanly clipped GeoDataFrame.

    Raises:
        TypeError: If gdf is not of type GeoDataFrame or GeoSeries.
    """
    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):
        raise TypeError(f"'gdf' should be GeoDataFrame or GeoSeries, got {type(gdf)}")

    try:
        return gdf.clip(mask, **kwargs).pipe(clean_geoms)
    except Exception:
        gdf = clean_geoms(gdf)
        mask = clean_geoms(mask)
        return gdf.clip(mask, **kwargs).pipe(clean_geoms)

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from shapely import box, extract_unique_points
from shapely.geometry import Polygon

from .to_geodataframe import to_gdf


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
    xs0, ys0, xs1, ys1 = [], [], [], []
    for x0 in np.arange(minx, maxx + gridsize, gridsize):
        for y0 in np.arange(miny, maxy + gridsize, gridsize):
            x1 = x0 - gridsize
            y1 = y0 + gridsize
            xs0.append(x0)
            ys0.append(y0)
            xs1.append(x1)
            ys1.append(y1)

    grid_cells = box(xs0, ys0, xs1, ys1)

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

    minx = int(minx) if minx > 0 else int(minx - 1)
    miny = int(miny) if miny > 0 else int(miny - 1)

    return make_grid_from_bbox(minx, miny, maxx, maxy, gridsize=gridsize, crs=gdf.crs)


def make_ssb_grid(
    gdf: GeoDataFrame, gridsize: int = 1000, add: int | float = 1
) -> GeoDataFrame:
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
    if not gdf.crs.equals(25833):
        raise ValueError(
            "Geodataframe must have crs = 25833. Use df.set_crs(25833) to set "
            "projection or df.to_crs(25833) for transforming."
        )

    minx, miny, maxx, maxy = gdf.total_bounds

    minx = minx - add * gridsize
    miny = miny - add * gridsize
    maxx = maxx + add * gridsize
    maxy = maxy + add * gridsize

    # Adjust for SSB-grid
    if minx > 0:
        minx = int(minx / int(gridsize)) * int(gridsize)
    else:
        minx = int((minx - gridsize) / int(gridsize)) * int(gridsize)

    if minx > 0:
        miny = int(miny / int(gridsize)) * int(gridsize)
    else:
        miny = int((miny - gridsize) / int(gridsize)) * int(gridsize)

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
    return grid[["SSBID", "geometry"]]


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
    gdf = bounds_to_polygon(gdf)
    gdf["geometry"] = extract_unique_points(gdf)
    return gdf


def points_in_bounds(gdf: GeoDataFrame | GeoSeries, n2: int):
    minx, miny, maxx, maxy = gdf.total_bounds
    xs = np.linspace(minx, maxx, num=n2)
    ys = np.linspace(miny, maxy, num=n2)
    x_coords, y_coords = np.meshgrid(xs, ys, indexing="ij")
    coords = np.concatenate((x_coords.reshape(-1, 1), y_coords.reshape(-1, 1)), axis=1)
    return to_gdf(coords, crs=gdf.crs)

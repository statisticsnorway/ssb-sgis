import numbers
from typing import Any, Callable, Iterable

import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from pandas.api.types import is_dict_like
from shapely import Geometry, box, extract_unique_points
from shapely.geometry import Polygon

from .general import clean_clip, is_bbox_like
from .to_geodataframe import to_gdf


def gridloop(
    func: Callable,
    mask: GeoDataFrame | GeoSeries | Geometry,
    gridsize: int,
    gridbuffer: int = 0,
    clip: bool = True,
    keep_geom_type: bool = True,
    verbose: bool = False,
    **kwargs,
) -> list[Any]:
    """Runs a function in a loop cellwise based on a grid.

    Creates grid from a mask, and runs the function for each cell
    with all GeoDataFrame keyword arguments clipped to the cell
    extent.

    Args:
        func: Function to run cellwise.
        mask: Geometry object to create a grid around.
        gridsize: Size of the grid cells in units of the crs (meters, degrees).
        gridbuffer: Units to buffer each gridcell by. For edge cases.
            Defaults to 0.
        clip: If True (default) geometries are clipped by the grid cells.
            If False, all geometries that intersect will be selected in each iteration.
        verbose: Whether to print progress. Defaults to False.
        keep_geom_type: Whether to keep only the input geometry types after clipping.
            Defaults to True.
        **kwargs: Keyword arguments passed to the function (func). Arguments that are
            of type GeoDataFrame or GeoSeries will be clipped by the mask in each
            iteration.

    Returns:
        List of results with the same length as number of grid cells.

    """
    if not isinstance(mask, GeoDataFrame):
        mask = to_gdf(mask)

    if not len(mask):
        raise ValueError("'mask' has no rows.")

    grid = make_grid(mask, gridsize=gridsize)
    grid = grid.loc[lambda df: df.index.isin(df.sjoin(mask).index)]

    if verbose:
        n = len(grid)

    results = []
    for i, cell in enumerate(grid.geometry.buffer(gridbuffer)):
        cell_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (gpd.GeoDataFrame, gpd.GeoSeries)):
                if clip:
                    value = clean_clip(value, cell, keep_geom_type=keep_geom_type)
                else:
                    value = value.loc[value.intersects(cell)]
            elif isinstance(value, Geometry):
                value = value.intersection(cell).make_valid()

            cell_kwargs[key] = value

        cell_res = func(**cell_kwargs)

        results.append(cell_res)

        if verbose:
            print(f"Done with {i+1} of {n} grid cells", end="\r")

    return results


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


def make_grid(
    obj: GeoDataFrame | GeoSeries | Geometry | tuple,
    gridsize: int | float,
    *,
    crs=None,
    clip_to_bounds: bool = False,
) -> GeoDataFrame:
    """Create a polygon grid around geometries.

    Creates a GeoDataFrame of grid cells of a given size around the bounds of
    a given GeoDataFrame. The corners are rounded to the nearest integer.

    Args:
        obj: GeoDataFrame, GeoSeries, shapely geometry or bounding box
            (an iterable with four values (minx, miny, maxx, maxy)).
        gridsize: Length of the grid cell walls.
        crs: Coordinate reference system if 'obj' is not GeoDataFrame or GeoSeries.
        clip_to_bounds: Whether to clip the grid to the total bounds of the geometries.
            Defaults to False.

    Returns:
        GeoDataFrame with grid polygons.

    """
    if isinstance(obj, (GeoDataFrame, GeoSeries)):
        crs = obj.crs or crs
    elif not crs:
        raise ValueError(
            "'crs' cannot be None when 'obj' is not GeoDataFrame/GeoSeries."
        )

    minx, miny, maxx, maxy = to_bbox(obj)

    minx = int(minx) if minx > 0 else int(minx - 1)
    miny = int(miny) if miny > 0 else int(miny - 1)

    grid = make_grid_from_bbox(minx, miny, maxx, maxy, gridsize=gridsize, crs=crs)

    if clip_to_bounds:
        grid = grid.clip(to_bbox(obj))

    return grid


def make_ssb_grid(
    gdf: GeoDataFrame | GeoSeries, gridsize: int = 1000, add: int | float = 1
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
    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):
        raise TypeError("gdf must be GeoDataFrame og GeoSeries.")

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

    # Make SSB id
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
    """Adds an SSB grid ID column to a GeoDataFrame of points.

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


def bounds_to_polygon(gdf: GeoDataFrame, copy: bool = True) -> GeoDataFrame:
    """Creates a box around the geometry in each row of a GeoDataFrame.

    Args:
        gdf: The GeoDataFrame.
        copy: Defaults to True.

    Returns:
        GeoDataFrame of box polygons with length and index of 'gdf'.

    Examples
    --------

    >>> gdf = sg.to_gdf([MultiPoint([(0, 0), (1, 1)]), Point(0, 0)])
    >>> gdf
                                            geometry
    0  MULTIPOINT (0.00000 0.00000, 1.00000 1.00000)
    1                        POINT (0.00000 0.00000)

    >>> sg.bounds_to_polygon(gdf)
                                                geometry
    0  POLYGON ((1.00000 0.00000, 1.00000 1.00000, 0....
    1  POLYGON ((0.00000 0.00000, 0.00000 0.00000, 0....

    """
    if copy:
        gdf = gdf.copy()
    gdf["geometry"] = [box(*arr) for arr in gdf.bounds.values]
    return gdf


def bounds_to_points(gdf: GeoDataFrame, copy: bool = True) -> GeoDataFrame:
    """Creates a 4-noded multipoint around the geometry in each row of a GeoDataFrame.

    Args:
        gdf: The GeoDataFrame.
        copy: Defaults to True.

    Returns:
        GeoDataFrame of multipoints with same length and index as 'gdf'.

    Examples
    --------
    >>> gdf = sg.to_gdf([MultiPoint([(0, 0), (1, 1)]), Point(0, 0)])
    >>> gdf
                                            geometry
    0  MULTIPOINT (0.00000 0.00000, 1.00000 1.00000)
    1                        POINT (0.00000 0.00000)

    >>> sg.bounds_to_points(gdf)
                                                geometry
    0  MULTIPOINT (1.00000 0.00000, 1.00000 1.00000, ...
    1                       MULTIPOINT (0.00000 0.00000)
    """
    gdf = bounds_to_polygon(gdf, copy=copy)
    gdf["geometry"] = extract_unique_points(gdf)
    return gdf


def to_bbox(
    obj: GeoDataFrame | GeoSeries | Geometry | Iterable | dict,
) -> tuple[float, float, float, float]:
    """Returns 4-length tuple of bounds if possible, else raises ValueError.

    Args:
        obj: Object to be converted to bounding box. Can be geopandas or shapely
            objects, iterables of exactly four numbers or dictionary like/class
            with a the keys/attributes "minx", "miny", "maxx", "maxy" or
            "xmin", "ymin", "xmax", "ymax".
    """
    if isinstance(obj, (GeoDataFrame, GeoSeries)):
        return tuple(obj.total_bounds)
    if isinstance(obj, Geometry):
        return tuple(obj.bounds)
    if (
        hasattr(obj, "__iter__")
        and len(obj) == 4
        and all(isinstance(x, numbers.Number) for x in obj)
    ):
        return tuple(obj)

    if is_dict_like(obj) and all(x in obj for x in ["minx", "miny", "maxx", "maxy"]):
        try:
            minx = np.min(obj["minx"])
            miny = np.min(obj["miny"])
            maxx = np.max(obj["maxx"])
            maxy = np.max(obj["maxy"])
        except TypeError:
            minx = np.min(obj.minx)
            miny = np.min(obj.miny)
            maxx = np.max(obj.maxx)
            maxy = np.max(obj.maxy)
        return minx, miny, maxx, maxy
    if is_dict_like(obj) and all(x in obj for x in ["xmin", "ymin", "xmax", "ymax"]):
        try:
            xmin = np.min(obj["xmin"])
            ymin = np.min(obj["ymin"])
            xmax = np.max(obj["xmax"])
            ymax = np.max(obj["ymax"])
        except TypeError:
            xmin = np.min(obj.xmin)
            ymin = np.min(obj.ymin)
            xmax = np.max(obj.xmax)
            ymax = np.max(obj.ymax)
        return xmin, ymin, xmax, ymax
    if is_dict_like(obj) and hasattr(obj, "geometry"):
        try:
            return tuple(GeoSeries(obj["geometry"]).total_bounds)
        except Exception:
            return tuple(GeoSeries(obj.geometry).total_bounds)
    try:
        of_length = f" of length {len(obj)}"
    except TypeError:
        of_length = ""
    raise TypeError(f"Cannot convert type {obj.__class__.__name__}{of_length} to bbox")


def get_total_bounds(
    *geometries: GeoDataFrame | GeoSeries | Geometry,
) -> tuple[float, float, float, float]:
    """Get a combined total bounds of multiple geometry objects."""
    xs, ys = [], []
    for obj in geometries:
        minx, miny, maxx, maxy = to_bbox(obj)
        xs += [minx, maxx]
        ys += [miny, maxy]

    return min(xs), min(ys), max(xs), max(ys)


def points_in_bounds(gdf: GeoDataFrame | GeoSeries, n2: int):
    if not isinstance(gdf, (GeoDataFrame, GeoSeries)) and is_bbox_like(gdf):
        minx, miny, maxx, maxy = gdf
    else:
        minx, miny, maxx, maxy = gdf.total_bounds
    xs = np.linspace(minx, maxx, num=n2)
    ys = np.linspace(miny, maxy, num=n2)
    x_coords, y_coords = np.meshgrid(xs, ys, indexing="ij")
    coords = np.concatenate((x_coords.reshape(-1, 1), y_coords.reshape(-1, 1)), axis=1)
    return to_gdf(coords, crs=gdf.crs)

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from pyproj import CRS
from shapely import Geometry
from shapely import box
from shapely import extract_unique_points
from shapely.geometry import Polygon

from ..parallel.parallel import Parallel
from .conversion import to_bbox
from .conversion import to_gdf
from .general import clean_clip
from .general import get_common_crs


@dataclass
class Gridlooper:
    """Run functions in a loop cellwise based on a grid.

    Args:
        gridsize: Size of the grid cells in units of the crs (meters, degrees).
        mask: Geometry object to create a grid around.
        gridbuffer: Units to buffer each gridcell by. For edge cases.
            Defaults to 0.
        clip: If True (default) geometries are clipped by the grid cells.
            If False, all geometries that intersect will be selected in each iteration.
        verbose: Whether to print progress. Defaults to False.
        keep_geom_type: Whether to keep only the input geometry types after clipping.
            Defaults to True.

    Examples:
    ---------
    Get some points and some polygons.

    >>> import sgis as sg
    >>> points = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet")
    >>> points["idx"] = points.index
    >>> buffered = sg.buff(points, 100)
    >>> buffered
        idx                                           geometry
    0      0  POLYGON ((263222.700 6651184.900, 263222.651 6...
    1      1  POLYGON ((272556.100 6653369.500, 272556.051 6...
    2      2  POLYGON ((270182.300 6653032.700, 270182.251 6...
        3      3  POLYGON ((259904.800 6650339.700, 259904.751 6...
    4      4  POLYGON ((272976.200 6652889.100, 272976.151 6...
    ..   ...                                                ...
    995  995  POLYGON ((266901.700 6647844.500, 266901.651 6...
    996  996  POLYGON ((261374.000 6653593.400, 261373.951 6...
    997  997  POLYGON ((263642.900 6645427.000, 263642.851 6...
    998  998  POLYGON ((269326.700 6650628.000, 269326.651 6...
    999  999  POLYGON ((264670.300 6644239.500, 264670.251 6...

    [1000 rows x 2 columns]

    Instantiate a gridlooper.

    >>> looper = sg.Gridlooper(gridsize=200, mask=buffered, concat=True, parallelizer=sg.Parallel(1, backend="multiprocessing"))

    Run the function clean_overlay in a gridloop.

    >>> results = looper.run(
    ...     sg.clean_overlay,
    ...     points,
    ...     buffered,
    ... )
    >>> results
        idx_1 idx_2                        geometry
    0      220   220  POINT (254575.200 6661631.500)
    1      735   735  POINT (256337.400 6649931.700)
    2      575   575  POINT (256369.200 6650413.300)
    3       39    39  POINT (256142.300 6650526.300)
    4      235   235  POINT (256231.300 6650720.200)
    ...    ...   ...                             ...
    1481   711   795  POINT (272845.500 6655048.800)
    1482   711   711  POINT (272845.500 6655048.800)
    1483   757   757  POINT (273507.600 6652806.600)
    1484   457   457  POINT (273524.400 6652979.900)
    1485   284   284  POINT (273650.800 6653000.500)

    [1486 rows x 3 columns]

    """

    gridsize: int
    mask: GeoDataFrame | GeoSeries | Geometry
    gridbuffer: int = 0
    parallelizer: Parallel | None = None
    concat: bool = False
    clip: bool = True
    keep_geom_type: bool = True
    verbose: bool = False

    def __post_init__(self) -> None:
        """Fix types."""
        if not isinstance(self.mask, GeoDataFrame):
            self.mask = to_gdf(self.mask)

    def run(self, func: Callable, *args, **kwargs) -> Any | list[Any]:
        """Run a function for each grid cell in a loop.

        Returns a list of the return values from the function,
        or a (Geo)DataFrame if self.concat is True.

        """

        def _intersects_mask(df: GeoDataFrame) -> pd.Series:
            return df.index.isin(df.sjoin(self.mask).index)

        grid: GeoSeries = (
            make_grid(self.mask, gridsize=self.gridsize).loc[_intersects_mask].geometry
        )

        n = len(grid)

        buffered_grid = grid.buffer(self.gridbuffer, resolution=1, join_style=2)

        if self.parallelizer is not None:
            func_with_clip = functools.partial(
                _clip_and_run_func,
                func=func,
                args=args,
                kwargs=kwargs,
                keep_geom_type=self.keep_geom_type,
                clip=self.clip,
            )
            results = self.parallelizer.map(func_with_clip, buffered_grid)
            if not self.gridbuffer or not self.clip:
                return self._return(results, args, kwargs)
            out = []
            for cell_res, unbuffered in zip(results, grid, strict=True):
                out.append(
                    _clip_back_to_unbuffered_grid(
                        cell_res, unbuffered, self.keep_geom_type
                    )
                )
            return self._return(out, args, kwargs)

        results = []
        for i, (unbuffered, buffered) in enumerate(
            zip(grid, buffered_grid, strict=False)
        ):
            cell_kwargs = {
                key: _clip_if_isinstance(
                    value, buffered, self.keep_geom_type, self.clip
                )
                for key, value in kwargs.items()
            }
            cell_args = tuple(
                _clip_if_isinstance(value, buffered, self.keep_geom_type, self.clip)
                for value in args
            )

            cell_res = func(*cell_args, **cell_kwargs)

            # clip back to original
            if self.gridbuffer and self.clip:
                cell_res = _clip_back_to_unbuffered_grid(
                    cell_res, unbuffered, self.keep_geom_type
                )

            results.append(cell_res)

            if self.verbose:
                print(f"Done with {i+1} of {n} grid cells", end="\r")

        return self._return(results, args, kwargs)

    def _return(
        self, results: list[Any], args: tuple[Any], kwargs: dict[str, Any]
    ) -> list[Any] | GeoDataFrame:
        if self.concat and len(results):
            return pd.concat(results, ignore_index=True)
        elif self.concat:
            crs = get_common_crs(list(args) + list(kwargs.values()))
            return GeoDataFrame({"geometry": []}, crs=crs)
        else:
            return results


def gridloop(
    func: Callable,
    gridsize: int,
    mask: GeoDataFrame | GeoSeries | Geometry,
    gridbuffer: int = 0,
    clip: bool = True,
    keep_geom_type: bool = True,
    verbose: bool = False,
    args: tuple | None = None,
    kwargs: dict | None = None,
    parallelizer: Parallel | None = None,
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
        args: Positional arguments to pass to the function. Arguments
            of type GeoDataFrame or GeoSeries will be clipped by the grid cells in
            a loop.
        kwargs: Keyword arguments to pass to the function. Arguments
            of type GeoDataFrame or GeoSeries will be clipped by the grid cells in
            a loop.
        parallelizer: Optional instance of sgis.Parallel, to run the function in parallel.

    Returns:
        List of results with the same length as number of grid cells.

    Raises:
        TypeError: If args or kwargs has a wrong type

    Examples:
    ---------
    Get some points and some polygons.

    >>> import sgis as sg
    >>> points = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet")
    >>> points["idx"] = points.index
    >>> buffered = sg.buff(points, 100)
    >>> buffered
         idx                                           geometry
    0      0  POLYGON ((263222.700 6651184.900, 263222.651 6...
    1      1  POLYGON ((272556.100 6653369.500, 272556.051 6...
    2      2  POLYGON ((270182.300 6653032.700, 270182.251 6...
    3      3  POLYGON ((259904.800 6650339.700, 259904.751 6...
    4      4  POLYGON ((272976.200 6652889.100, 272976.151 6...
    ..   ...                                                ...
    995  995  POLYGON ((266901.700 6647844.500, 266901.651 6...
    996  996  POLYGON ((261374.000 6653593.400, 261373.951 6...
    997  997  POLYGON ((263642.900 6645427.000, 263642.851 6...
    998  998  POLYGON ((269326.700 6650628.000, 269326.651 6...
    999  999  POLYGON ((264670.300 6644239.500, 264670.251 6...

    [1000 rows x 2 columns]

    Run the function clean_overlay where the data is clipped to a grid
    of 1000x1000 meters. Args are the first two arguments of clean_overlay,
    kwargs are additional keyword arguments.

    >>> resultslist = sg.gridloop(
    ...     func=sg.clean_overlay,
    ...     mask=buffered,
    ...     gridsize=1000,
    ...     args=(points, buffered),
    ...     kwargs={"how": "intersection"}
    ... )
    >>> type(resultslist)
    list

    >>> results = pd.concat(resultslist, ignore_index=True)
    >>> results
         idx_1 idx_2                        geometry
    0      220   220  POINT (254575.200 6661631.500)
    1      735   735  POINT (256337.400 6649931.700)
    2      575   575  POINT (256369.200 6650413.300)
    3       39    39  POINT (256142.300 6650526.300)
    4      235   235  POINT (256231.300 6650720.200)
    ...    ...   ...                             ...
    1481   711   795  POINT (272845.500 6655048.800)
    1482   711   711  POINT (272845.500 6655048.800)
    1483   757   757  POINT (273507.600 6652806.600)
    1484   457   457  POINT (273524.400 6652979.900)
    1485   284   284  POINT (273650.800 6653000.500)

    [1486 rows x 3 columns]

    """
    if kwargs is None:
        kwargs = {}
    elif not isinstance(kwargs, dict):
        raise TypeError("kwargs should be a dict")

    if args is None:
        args = ()
    elif not isinstance(args, tuple):
        raise TypeError("args should be a tuple")

    if not isinstance(mask, GeoDataFrame):
        mask = to_gdf(mask)

    def _intersects_mask(df: GeoDataFrame) -> pd.Series:
        return df.index.isin(df.sjoin(mask).index)

    grid: GeoSeries = make_grid(mask, gridsize=gridsize).loc[_intersects_mask].geometry

    n = len(grid)

    buffered_grid = grid.buffer(gridbuffer, resolution=1, join_style=2)

    if parallelizer is not None:
        func_with_clip = functools.partial(
            _clip_and_run_func,
            func=func,
            args=args,
            kwargs=kwargs,
            keep_geom_type=keep_geom_type,
            clip=clip,
        )
        results = parallelizer.map(func_with_clip, buffered_grid)
        if not gridbuffer or not clip:
            return results
        out = []
        for cell_res, unbuffered in zip(results, grid, strict=True):
            out.append(
                _clip_back_to_unbuffered_grid(cell_res, unbuffered, keep_geom_type)
            )
        return out

    results = []
    for i, (unbuffered, buffered) in enumerate(zip(grid, buffered_grid, strict=False)):
        cell_kwargs = {
            key: _clip_if_isinstance(value, buffered, keep_geom_type, clip)
            for key, value in kwargs.items()
        }
        cell_args = tuple(
            _clip_if_isinstance(value, buffered, keep_geom_type, clip) for value in args
        )

        cell_res = func(*cell_args, **cell_kwargs)

        # clip back to original
        if gridbuffer and clip:
            cell_res = _clip_back_to_unbuffered_grid(
                cell_res, unbuffered, keep_geom_type
            )

        results.append(cell_res)

        if verbose:
            print(f"Done with {i+1} of {n} grid cells", end="\r")

    return results


def _clip_and_run_func(
    grid_cell: Polygon,
    func: Callable,
    args: tuple,
    kwargs: dict,
    keep_geom_type: bool,
    clip: bool,
) -> Any:
    cell_args = tuple(
        _clip_if_isinstance(value, grid_cell, keep_geom_type, clip) for value in args
    )
    cell_kwargs = {
        key: _clip_if_isinstance(value, grid_cell, keep_geom_type, clip)
        for key, value in kwargs.items()
    }

    return func(*cell_args, **cell_kwargs)


def _clip_if_isinstance(
    value: Any, cell: Geometry, keep_geom_type: bool, clip: bool
) -> Any:
    if not isinstance(value, (gpd.GeoDataFrame | gpd.GeoSeries | Geometry)):
        return value

    if isinstance(value, (gpd.GeoDataFrame | gpd.GeoSeries)):
        if clip:
            return clean_clip(value, cell, keep_geom_type=keep_geom_type)
        return value.loc[value.intersects(cell)]

    return value.intersection(cell).make_valid()


def _clip_back_to_unbuffered_grid(
    results: Any, mask: GeoDataFrame, keep_geom_type: bool
) -> Any:
    if isinstance(results, (gpd.GeoDataFrame | gpd.GeoSeries | Geometry)):
        return _clip_if_isinstance(results, mask, keep_geom_type, clip=True)
    elif isinstance(results, (pd.DataFrame | pd.Series | np.ndarray)):
        return results
    try:
        for key, value in results.items():
            results[key] = _clip_if_isinstance(value, mask, keep_geom_type, clip=True)
    except AttributeError:
        try:
            return [
                _clip_if_isinstance(res, mask, keep_geom_type, clip=True)
                for res in results
            ]
        except TypeError:
            pass
    return results


def make_grid_from_bbox(
    minx: int | float,
    miny: int | float,
    maxx: int | float,
    maxy: int | float,
    *_,
    gridsize: int | float,
    crs: CRS | int | str,
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
    crs: CRS | None = None,
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

    Raises:
        ValueError: crs can only be None if obj is GeoDataFrame/GeoSeries.

    """
    if isinstance(obj, (GeoDataFrame | GeoSeries)):
        crs = obj.crs or crs
    if hasattr(obj, "__len__") and not len(obj):
        return GeoDataFrame({"geometry": []}, crs=crs)

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
        add: Number of grid cells to add on each side,
            to make sure all data is covered by the grid.

    Returns:
        GeoDataFrame with grid geometries and a column 'SSBID'.

    Raises:
        ValueError: If the GeoDataFrame does not have 25833 as crs.
        TypeError: if gdf has wrong type.
    """
    if not isinstance(gdf, (GeoDataFrame | GeoSeries)):
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
        out_column: Name of column for the grid id.

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


def bounds_to_polygon(
    gdf: GeoDataFrame | GeoSeries, copy: bool = True
) -> GeoDataFrame | GeoSeries:
    """Creates a box around the geometry in each row of a GeoDataFrame.

    Args:
        gdf: The GeoDataFrame.
        copy: Defaults to True.

    Returns:
        GeoDataFrame of box polygons with length and index of 'gdf'.

    Examples:
    ---------
    >>> import sgis as sg
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
    if isinstance(gdf, GeoSeries):
        return GeoSeries([box(*arr) for arr in gdf.bounds.values], index=gdf.index)
    if copy:
        gdf = gdf.copy()
    gdf.geometry = [box(*arr) for arr in gdf.bounds.values]
    return gdf


def bounds_to_points(
    gdf: GeoDataFrame | GeoSeries, copy: bool = True
) -> GeoDataFrame | GeoSeries:
    """Creates a 4-noded multipoint around the geometry in each row of a GeoDataFrame.

    Args:
        gdf: The GeoDataFrame.
        copy: Defaults to True.

    Returns:
        GeoDataFrame of multipoints with same length and index as 'gdf'.

    Examples:
    ---------
    >>> import sgis as sg
    >>> from shapely.geometry import MultiPoint, Point
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
    as_bounds = bounds_to_polygon(gdf, copy=copy)
    if isinstance(gdf, GeoSeries):
        return GeoSeries(extract_unique_points(as_bounds), index=gdf.index)
    gdf.geometry = extract_unique_points(as_bounds.geometry)
    return gdf


def get_total_bounds(
    *geometries: GeoDataFrame | GeoSeries | Geometry, strict: bool = False
) -> tuple[float, float, float, float]:
    """Get a combined total bounds of multiple geometry objects."""
    xs, ys = [], []
    for obj in geometries:
        try:
            minx, miny, maxx, maxy = to_bbox(obj)
            xs += [minx, maxx]
            ys += [miny, maxy]
        except Exception as e:
            try:
                for x in obj:
                    minx, miny, maxx, maxy = to_bbox(x)
                    xs += [minx, maxx]
                    ys += [miny, maxy]
            except Exception as e2:
                if strict:
                    raise e2 from e
                else:
                    continue
    return min(xs), min(ys), max(xs), max(ys)

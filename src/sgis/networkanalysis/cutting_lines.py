"""Cutting and splitting line geometries."""
import warnings

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame, Series
from shapely import (
    buffer,
    extract_unique_points,
    force_2d,
    get_coordinates,
    get_parts,
    linestrings,
    touches,
    unary_union,
)
from shapely.geometry import LineString, Point

from ..geopandas_tools.buffer_dissolve_explode import buff
from ..geopandas_tools.conversion import to_gdf
from ..geopandas_tools.geometry_types import get_geom_type
from ..geopandas_tools.neighbors import get_k_nearest_neighbors
from ..geopandas_tools.point_operations import (
    _shapely_snap,
    snap_all,
    snap_within_distance,
)
from ..geopandas_tools.sfilter import sfilter_split
from .nodes import make_edge_coords_cols


def split_lines_by_nearest_point(
    gdf: GeoDataFrame,
    points: GeoDataFrame,
    max_distance: int | float | None = None,
    splitted_col: str | None = None,
) -> DataFrame:
    """Split lines that are closest to s point.

    Snaps points to nearest lines and splits the lines in two at the snap point.
    The splitting is done pointwise, meaning each point splits one line in two.
    The line will not be split if the point is closest to the endpoint of the line.

    This function is used in NetworkAnalysis if split_lines is set to True.

    Args:
        gdf: GeoDataFrame of lines that will be split.
        points: GeoDataFrame of points to split the lines with.
        max_distance: the maximum distance between the point and the line. Points
            further away than max_distance will not split any lines. Defaults to None.
        splitted_col: Optionally add a column

    Returns:
        A GeoDataFrame with the same columns as the input lines, but with the lines
        split at the closest point to the points. If no lines are within 'max_distance'
        from any points, 'gdf' is returned as it came.

    Raises:
        ValueError: If the crs of the input data differs.

    Examples
    --------
    >>> from sgis import read_parquet_url, split_lines_by_nearest_point
    >>> roads = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")
    >>> points = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet")
    >>> rows = len(roads)
    >>> rows
    93395

    Splitting lines for points closer than 10 meters from the lines.

    >>> roads = split_lines_by_nearest_point(roads, points, max_distance=10)
    >>> print("number of lines that were split:", len(roads) - rows)
    number of lines that were split: 380

    Splitting lines by all points.

    >>> roads = split_lines_by_nearest_point(roads, points)
    >>> print("number of lines that were split:", len(roads) - rows)
    number of lines that were split: 848

    Not all lines were split. That is because some points were closest to an endpoint
    of a line.
    """
    PRECISION = 1e-6

    if not len(gdf):
        return gdf

    if (points.crs is not None and gdf.crs is not None) and not points.crs.equals(
        gdf.crs
    ):
        raise ValueError("crs mismatch:", points.crs, "and", gdf.crs)

    if get_geom_type(gdf) != "line":
        raise ValueError("'gdf' should only have line geometries.", gdf.geom_type)

    if get_geom_type(points) != "point":
        raise ValueError("'points' should only have point geometries.")

    gdf = gdf.copy()

    # move the points to the nearest exact location on the line
    if max_distance:
        snapped = snap_within_distance(points, gdf, max_distance=max_distance)
    else:
        snapped = snap_all(points, gdf)

    # find the lines that were snapped to (or are very close because of float rounding)
    snapped_buff = buff(snapped, PRECISION, resolution=16)
    relevant_lines, the_other_lines = sfilter_split(gdf, snapped_buff)

    if max_distance and not len(relevant_lines):
        if splitted_col:
            return gdf.assign(**{splitted_col: 1})
        return gdf

    # need consistent coordinate dimensions later
    # (doing it down here to not overwrite the original data)
    relevant_lines.geometry = force_2d(relevant_lines.geometry)
    snapped.geometry = force_2d(snapped.geometry)

    # split the lines with buffer + difference, since shaply.split usually doesn't work
    # relevant_lines["_idx"] = range(len(relevant_lines))
    splitted = relevant_lines.overlay(snapped_buff, how="difference").explode(
        ignore_index=True
    )

    # linearrings (maybe coded as linestrings) that were not split,
    # do not have edges and must be added in the end
    boundaries = splitted.geometry.boundary
    circles = splitted[boundaries.is_empty]
    splitted = splitted[~boundaries.is_empty]

    if not len(splitted):
        return pd.concat([the_other_lines, circles], ignore_index=True)

    # the endpoints of the new lines are now sligtly off. Using get_k_nearest_neighbors
    # to get the exact snapped point coordinates, . This will map the sligtly
    # wrong line endpoints with the point the line was split by.

    snapped["point_coords"] = [(geom.x, geom.y) for geom in snapped.geometry]

    # get line endpoints as columns (source_coords and target_coords)
    splitted = make_edge_coords_cols(splitted)

    splitted_source = to_gdf(splitted["source_coords"], crs=gdf.crs)
    splitted_target = to_gdf(splitted["target_coords"], crs=gdf.crs)

    def get_nearest(splitted, snapped) -> pd.DataFrame:
        """find the nearest snapped point for each source and target of the lines"""
        return get_k_nearest_neighbors(splitted, snapped, k=1).loc[
            lambda x: x["distance"] <= PRECISION * 2
        ]

    # snapped = snapped.set_index("point_coords")
    snapped.index = snapped.geometry
    dists_source = get_nearest(splitted_source, snapped)
    dists_target = get_nearest(splitted_target, snapped)

    # neighbor_index: point coordinates as tuple
    pointmapper_source: pd.Series = dists_source["neighbor_index"]
    pointmapper_target: pd.Series = dists_target["neighbor_index"]

    # now, we can replace the source/target coordinate with the coordinates of
    # the snapped points.

    splitted = change_line_endpoint(
        splitted,
        indices=dists_source.index,
        pointmapper=pointmapper_source,
        change_what="first",
    )  # i=0)

    # same for the lines where the target was split, but change the last coordinate
    splitted = change_line_endpoint(
        splitted,
        indices=dists_target.index,
        pointmapper=pointmapper_target,
        change_what="last",
    )  # , i=-1)

    if splitted_col:
        splitted[splitted_col] = 1

    return pd.concat([the_other_lines, splitted, circles], ignore_index=True).drop(
        ["source_coords", "target_coords"], axis=1
    )


def change_line_endpoint(
    gdf: GeoDataFrame,
    indices: pd.Index,
    pointmapper: pd.Series,
    change_what: str | int,
) -> GeoDataFrame:
    """
    Loop for each line where the source is the endpoint that was split
    change the first point of the line to the point it was split by
    """
    assert gdf.index.is_unique

    if change_what == "first" or change_what == 0:
        to_be_changed = lambda x: ~x.index.duplicated(keep="first")
    elif change_what == "last" or change_what == -1:
        to_be_changed = lambda x: ~x.index.duplicated(keep="last")
    else:
        raise ValueError(
            f"change_what should be 'first' or 'last' or 0 or -1. Got {change_what}"
        )

    is_relevant = gdf.index.isin(indices)
    relevant_lines = gdf.loc[is_relevant]

    relevant_lines.geometry = extract_unique_points(relevant_lines.geometry)
    relevant_lines = relevant_lines.explode(index_parts=False)

    relevant_lines.loc[to_be_changed, "geometry"] = (
        relevant_lines.loc[to_be_changed].index.map(pointmapper).values
    )

    relevant_lines_mapped = relevant_lines.groupby(level=0)["geometry"].agg(LineString)

    gdf.loc[is_relevant, "geometry"] = relevant_lines_mapped

    return gdf


def cut_lines(gdf: GeoDataFrame, max_length: int, ignore_index=False) -> GeoDataFrame:
    """Cuts lines of a GeoDataFrame into pieces of a given length.

    Args:
        gdf: GeoDataFrame.
        max_length: The maximum length of the lines in the output GeoDataFrame.
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Defaults to False.

    Returns:
        A GeoDataFrame with lines cut to the maximum distance.

    Note:
        This method is time consuming for large networks and low 'max_length'.

    Examples
    --------
    >>> from sgis import read_parquet_url, cut_lines
    >>> roads = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")
    >>> roads.length.describe().round(1)
    count    93395.0
    mean        41.2
    std         78.5
    min          0.2
    25%         14.0
    50%         27.7
    75%         47.5
    max       5213.7
    dtype: float64
    >>> roads = cut_lines(roads, max_length=100)
    >>> roads.length.describe().round(1)
    count    126304.0
    mean         30.5
    std          30.1
    min           0.0
    25%           5.7
    50%          22.5
    75%          44.7
    max         100.0
    dtype: float64
    """
    if get_geom_type(gdf) != "line":
        raise ValueError("'gdf' should only have line geometries.")

    gdf["geometry"] = force_2d(gdf.geometry)

    gdf = gdf.explode(ignore_index=ignore_index, index_parts=False)

    long_lines = gdf.loc[gdf.length > max_length]

    if not len(long_lines):
        return gdf

    for x in [10, 5, 1]:
        max_ = max(long_lines.length)
        while max_ > max_length * x + 1:
            max_ = max(long_lines.length)

            long_lines = cut_lines_once(long_lines, max_length)

            if max_ == max(long_lines.length):
                break

    long_lines = long_lines.explode(ignore_index=ignore_index, index_parts=False)

    short_lines = gdf.loc[gdf.length <= max_length]

    return pd.concat([short_lines, long_lines], ignore_index=ignore_index)


def cut_lines_once(
    gdf: GeoDataFrame,
    distances: int | float | str | Series,
    ignore_index: bool = False,
) -> GeoDataFrame:
    """Cuts lines of a GeoDataFrame in two at the given distance or distances.

    Takes a GeoDataFrame of lines and cuts each line in two. If distances is a number,
    all lines will be cut at the same length.

    Args:
        gdf: GeoDataFrame.
        distances: The distance from the start of the lines to cut at. Either a number,
            the name of a column or array-like of same length as the line GeoDataFrame.
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Defaults to False.

    Examples
    --------
    >>> from sgis import cut_lines_once, to_gdf
    >>> import pandas as pd
    >>> from shapely.geometry import LineString
    >>> gdf = to_gdf(LineString([(0, 0), (1, 1), (2, 2)]))
    >>> gdf = pd.concat([gdf, gdf, gdf])
    >>> gdf
                                                geometry
    0  LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...
    0  LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...
    0  LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...

    Cut all lines at 1 unit from the start of the lines.

    >>> cut_lines_once(gdf, 1)
                                                geometry
    0      LINESTRING (0.00000 0.00000, 0.70711 0.70711)
    1  LINESTRING (0.70711 0.70711, 1.00000 1.00000, ...
    2      LINESTRING (0.00000 0.00000, 0.70711 0.70711)
    3  LINESTRING (0.70711 0.70711, 1.00000 1.00000, ...
    4      LINESTRING (0.00000 0.00000, 0.70711 0.70711)
    5  LINESTRING (0.70711 0.70711, 1.00000 1.00000, ...

    Cut distance as column.

    >>> gdf["dist"] = [1, 2, 3]
    >>> cut_lines_once(gdf, "dist")
                                                geometry  dist
    0      LINESTRING (0.00000 0.00000, 0.70711 0.70711)     1
    1  LINESTRING (0.70711 0.70711, 1.00000 1.00000, ...     1
    2  LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...     2
    3      LINESTRING (1.41421 1.41421, 2.00000 2.00000)     2
    4  LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...     3

    Cut distance as list (same result as above).

    >>> cut_lines_once(gdf, [1, 2, 3])
                                                geometry  dist
    0      LINESTRING (0.00000 0.00000, 0.70711 0.70711)     1
    1  LINESTRING (0.70711 0.70711, 1.00000 1.00000, ...     1
    2  LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...     2
    3      LINESTRING (1.41421 1.41421, 2.00000 2.00000)     2
    4  LINESTRING (0.00000 0.00000, 1.00000 1.00000, ...     3
    """

    def _cut(line: LineString, distance: int | float) -> list[LineString]:
        """From the shapely docs"""
        if distance <= 0.0 or distance >= line.length:
            return line
        coords = list(line.coords)
        for i, p in enumerate(coords):
            prd = line.project(Point(p))
            if prd == distance:
                return [LineString(coords[: i + 1]), LineString(coords[i:])]
            if prd > distance:
                cp = line.interpolate(distance)
                return [
                    LineString(coords[:i] + [(cp.x, cp.y)]),
                    LineString([(cp.x, cp.y)] + coords[i:]),
                ]

    crs = gdf.crs
    geom_col = gdf._geometry_column_name

    gdf = gdf.copy()

    # cutting lines will give lists of linestrings in the geometry column. Ignoring
    # the warning it triggers
    warnings.filterwarnings(
        "ignore", message="Geometry column does not contain geometry."
    )

    if isinstance(distances, str):
        gdf[geom_col] = np.vectorize(_cut)(gdf[geom_col], gdf[distances])
    else:
        gdf[geom_col] = np.vectorize(_cut)(gdf[geom_col], distances)

    # explode will give pandas df if not gdf is constructed
    gdf = GeoDataFrame(
        gdf.explode(ignore_index=ignore_index, index_parts=False),
        geometry=geom_col,
        crs=crs,
    )

    return gdf

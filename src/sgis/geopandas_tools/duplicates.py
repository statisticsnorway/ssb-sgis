from typing import Callable, Iterable

import networkx as nx
import pandas as pd
from geopandas import GeoDataFrame
from shapely import STRtree, difference, intersection, make_valid, unary_union, union
from shapely.errors import GEOSException
from shapely.geometry import Polygon

from .general import _push_geom_col, clean_geoms
from .geometry_types import get_geom_type, to_single_geom_type
from .overlay import clean_overlay


def update_geometries(
    gdf: GeoDataFrame,
    keep_geom_type: bool = True,
    grid_size: int | None = None,
    copy: bool = True,
) -> GeoDataFrame:
    """Puts geometries on top of each other rowwise.

    Since this operation is done rowwise, it's important to
    first sort the GeoDataFrame approriately. See example below.

    Args:
        gdf: The GeoDataFrame to be updated.
        keep_geom_type: If True, return only geometries of original type in case
            of intersection resulting in multiple geometry types or
            GeometryCollections. If False, return all resulting geometries
            (potentially mixed types).
        grid_size: Precision grid size to round the geometries. Will use the highest
            precision of the inputs by default.
        copy: Defaults to True.

    Example
    ------
    Create two circles and get the overlap.

    >>> import sgis as sg
    >>> circles = sg.to_gdf([(0, 0), (1, 1)]).pipe(sg.buff, 1)
    >>> duplicates = sg.get_intersections(circles)
    >>> duplicates
       idx                                           geometry
    0    1  POLYGON ((0.03141 0.99951, 0.06279 0.99803, 0....
    1    2  POLYGON ((1.00000 0.00000, 0.96859 0.00049, 0....

    The polygons are identical except for the order of the coordinates.

    >>> poly1, poly2 = duplicates.geometry
    >>> poly1.equals(poly2)
    True

    'update_geometries' gives different results based on the order
    of the GeoDataFrame.

    >>> sg.update_geometries(duplicates)
        idx                                           geometry
    0    1  POLYGON ((0.03141 0.99951, 0.06279 0.99803, 0....

    >>> dups_rev = duplicates.iloc[::-1]
    >>> sg.update_geometries(dups_rev)
        idx                                           geometry
    1    2  POLYGON ((1.00000 0.00000, 0.96859 0.00049, 0....

    It might be appropriate to put the largest polygons on top
    and sort all NaNs to the bottom.

    >>> updated = (
    ...     sg.sort_large_first(duplicates)
    ...     .pipe(sg.sort_nans_last)
    ...     .pipe(sg.update_geometries)
    >>> updated
        idx                                           geometry
    0    1  POLYGON ((0.03141 0.99951, 0.06279 0.99803, 0....

    """
    if len(gdf) <= 1:
        return gdf

    df = pd.DataFrame(gdf, copy=copy)

    unioned = Polygon()
    out_rows, indices, geometries = [], [], []

    if keep_geom_type:
        geom_type = get_geom_type(gdf)
        if geom_type == "mixed":
            raise ValueError("Cannot have mixed geometries when keep_geom_type is True")

    for i, row in df.iterrows():
        geom = row.pop("geometry")

        if any(geom.equals(geom2) for geom2 in geometries):
            continue

        try:
            new = difference(geom, unioned, grid_size=grid_size)
        except GEOSException:
            try:
                geom = make_valid(geom)
                new = difference(geom, unioned, grid_size=grid_size)
            except GEOSException:
                print("\n\nunioned")
                print(unioned)
                unioned = to_single_geom_type(unioned, geom_type=geom_type)
                print(unioned)
                new = difference(geom, unioned, grid_size=grid_size)

        if not new:
            continue

        try:
            unioned = unary_union([new, unioned], grid_size=grid_size)
        except GEOSException:
            new = make_valid(new)
            unioned = unary_union([new, unioned], grid_size=grid_size)

        unioned = make_valid(unioned)

        out_rows.append(row)
        geometries.append(new)
        indices.append(i)

    out = GeoDataFrame(out_rows, geometry=geometries, index=indices, crs=gdf.crs)

    if keep_geom_type:
        out = to_single_geom_type(out, geom_type)

    return out


def get_intersections(gdf: GeoDataFrame) -> GeoDataFrame:
    """Find geometries that intersect in a GeoDataFrame.

    Does an intersection with itself and keeps only the geometries that appear
    more than once.

    Note that the returned GeoDataFrame in most cases contain two rows per
    intersection pair. It might also contain more than two overlapping polygons
    if there were multiple overlapping. These can be removed with
    update_geometries. See example below.

    Args:
        gdf: GeoDataFrame of polygons.

    Returns:
        A GeoDataFrame of the overlapping polygons.

    Examples
    --------
    Create three partially overlapping polygons.

    >>> import sgis as sg
    >>> circles = sg.to_gdf([(0, 0), (1, 0), (2, 0)]).pipe(sg.buff, 1.2)
    >>> circles.area
    0    4.523149
    1    4.523149
    2    4.523149
    dtype: float64

    Get the duplicates.

    >>> duplicates = sg.get_intersections(circles)
    >>> duplicates["area"] = duplicates.area
    >>> duplicates
                                                geometry      area
    0  POLYGON ((1.19941 -0.03769, 1.19763 -0.07535, ...  2.194730
    0  POLYGON ((1.19941 -0.03769, 1.19763 -0.07535, ...  0.359846
    1  POLYGON ((0.48906 -1.08579, 0.45521 -1.06921, ...  2.194730
    1  POLYGON ((2.19941 -0.03769, 2.19763 -0.07535, ...  2.194730
    2  POLYGON ((0.98681 -0.64299, 0.96711 -0.61085, ...  0.359846
    2  POLYGON ((1.48906 -1.08579, 1.45521 -1.06921, ...  2.194730

    We get two rows for each intersection pair.

    To get no overlapping geometries without , we can put geometries
    on top of each other rowwise.

    >>> updated = sg.update_geometries(duplicates)
    >>> updated["area"] = updated.area
    >>> updated
           area                                           geometry
    0  2.194730  POLYGON ((1.19941 -0.03769, 1.19763 -0.07535, ...
    1  1.834884  POLYGON ((2.19763 -0.07535, 2.19467 -0.11293, ...

    It might be appropriate to sort the dataframe by columns.
    Or put large polygons first and NaN values last.

    >>> updated = (
    ...     sg.sort_large_first(duplicates)
    ...     .pipe(sg.sort_nans_last)
    ...     .pipe(sg.update_geometries)
    ... )
    >>> updated
          area                                           geometry
    0  2.19473  POLYGON ((1.19941 -0.03769, 1.19763 -0.07535, ...
    1  2.19473  POLYGON ((2.19763 -0.07535, 2.19467 -0.11293, ...

    """

    idx_name = gdf.index.name
    duplicated_geoms = _get_intersecting_geometries(gdf).pipe(clean_geoms)

    duplicated_geoms.index = duplicated_geoms["orig_idx"].values
    duplicated_geoms.index.name = idx_name
    return duplicated_geoms.drop(columns="orig_idx")


def _get_intersecting_geometries(gdf: GeoDataFrame) -> GeoDataFrame:
    gdf = gdf.assign(orig_idx=gdf.index).reset_index(drop=True)

    right = gdf[[gdf._geometry_column_name]]
    right["idx_right"] = right.index
    left = gdf
    left["idx_left"] = left.index

    intersected = clean_overlay(left, right, how="intersection")

    not_from_same_poly = intersected.loc[lambda x: x["idx_left"] != x["idx_right"]]

    return not_from_same_poly.drop(columns=["idx_left", "idx_right"])


def _drop_duplicate_geometries(gdf: GeoDataFrame, **kwargs) -> GeoDataFrame:
    """Drop geometries that are considered equal.

    Args:
        gdf: GeoDataFrame.
        **kwargs: Keyword arguments passed to pandas.DataFrame.drop_duplicates

    Returns:
        The GeoDataFrame with duplicate geometries dropped.
    """
    return (
        _get_duplicate_geometry_groups(gdf, group_col="dup_idx_")
        .drop_duplicates("dup_idx_", **kwargs)
        .drop(columns="dup_idx_")
    )


def _get_duplicate_geometry_groups(
    gdf: GeoDataFrame, group_col: str | Iterable[str] = "duplicate_index"
):
    idx_mapper = dict(enumerate(gdf.index))
    idx_name = gdf.index.name

    gdf = gdf.reset_index(drop=True)

    tree = STRtree(gdf.geometry.values)
    left, right = tree.query(gdf.geometry.values, predicate="within")

    edges = list(zip(left, right))

    graph = nx.Graph()
    graph.add_edges_from(edges)

    component_mapper = {
        j: i
        for i, component in enumerate(nx.connected_components(graph))
        for j in component
    }

    gdf[group_col] = component_mapper

    gdf = _push_geom_col(gdf)

    gdf.index = gdf.index.map(idx_mapper)
    gdf.index.name = idx_name

    return gdf

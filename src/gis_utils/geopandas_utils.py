import warnings
from random import random

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from shapely import (
    Geometry,
    area,
    get_exterior_ring,
    get_interior_ring,
    get_num_interior_rings,
    get_parts,
    polygons,
)
from shapely.ops import nearest_points, snap, unary_union
from shapely.wkt import loads

from .buffer_dissolve_explode import buff


def clean_geoms(
    gdf: GeoDataFrame | GeoSeries,
    single_geom_type: bool = True,
    ignore_index: bool = False,
) -> GeoDataFrame | GeoSeries:
    """
    Repairs geometries, removes geometries that are invalid, empty, NaN and None,
    keeps only the most common geometry type (multi- and singlepart).

    Args:
        gdf: GeoDataFrame or GeoSeries to be cleaned.
        single_geomtype: if only the most common geometry type should be kept.
            This will be either points, lines or polygons.
            Both multi- and singlepart geometries are included.
            GeometryCollections will be exploded first, so that no geometries are excluded.
            Defaults to True.
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Defaults to False

    Returns:
        GeoDataFrame or GeoSeries with fixed geometries and only the rows with valid,
        non-empty and not-NaN/-None geometries.

    """

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

    if single_geom_type:
        gdf = to_single_geom_type(gdf, ignore_index=ignore_index)

    return gdf


def to_single_geom_type(
    gdf: GeoDataFrame | GeoSeries, ignore_index: bool = False
) -> GeoDataFrame | GeoSeries:
    """
    It takes a GeoDataFrame or GeoSeries and returns a GeoDataFrame or GeoSeries
    with only one geometry type. This will either be points, lines or polygons.
    Both multipart and singlepart geometries are kept.
    LinearRings are considered as lines.

    GeometryCollections will be exploded to single-typed geometries, so that the
    correctly typed geometries in these collections are included in the output.

    Args:
      gdf (GeoDataFrame | GeoSeries): GeoDataFrame | GeoSeries
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
        Defaults to False

    Returns:
      A GeoDataFrame with a single geometry type.
    """

    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):
        raise TypeError(f"'gdf' should be GeoDataFrame or GeoSeries, got {type(gdf)}")

    # explode collections to single-typed geometries
    collections = gdf.loc[gdf.geom_type == "GeometryCollection"]
    if len(collections):
        collections = collections.explode(ignore_index=ignore_index)
        gdf = gdf_concat([gdf, collections])

    polys = ["Polygon", "MultiPolygon"]
    lines = ["LineString", "MultiLineString", "LinearRing"]
    points = ["Point", "MultiPoint"]

    poly_check = len(gdf.loc[gdf.geom_type.isin(polys)])
    lines_check = len(gdf.loc[gdf.geom_type.isin(lines)])
    points_check = len(gdf.loc[gdf.geom_type.isin(points)])

    _max = max([poly_check, lines_check, points_check])

    if _max == len(gdf):
        return gdf

    if poly_check == _max:
        gdf = gdf.loc[gdf.geom_type.isin(polys)]
    elif lines_check == _max:
        gdf = gdf.loc[gdf.geom_type.isin(lines)]
    elif points_check == _max:
        gdf = gdf.loc[gdf.geom_type.isin(points)]
    else:
        raise ValueError(
            "Mixed geometry types and equal amount of two or all the types."
        )

    if ignore_index:
        gdf = gdf.reset_index(drop=True)

    return gdf


def close_holes(
    gdf: GeoDataFrame | GeoSeries | Geometry,
    max_km2: int | None = None,
    copy: bool = True,
) -> GeoDataFrame | GeoSeries | Geometry:
    """
    It closes holes in polygons of a GeoDataFrame, GeoSeries or shapely Geometry.

    Args:
      gdf: GeoDataFrame, GeoSeries or shapely Geometry.
      max_km2 (int | None): if None (default), all holes are closed.
        Otherwise, closes holes with an area below the specified number in
        square kilometers if the crs unit is in meters.
      copy (bool): if True (default), the input GeoDataFrame or GeoSeries is copied.
        Defaults to True

    Returns:
      A GeoDataFrame, GeoSeries or shapely Geometry with closed holes in the geometry column
    """

    if copy:
        gdf = gdf.copy()

    if isinstance(gdf, GeoDataFrame):
        gdf["geometry"] = gdf.geometry.map(lambda x: _close_holes_geom(x, max_km2))

    elif isinstance(gdf, gpd.GeoSeries):
        gdf = gdf.map(lambda x: _close_holes_geom(x, max_km2))
        gdf = gpd.GeoSeries(gdf)

    else:
        gdf = _close_holes_geom(gdf, max_km2)

    return gdf


def _close_holes_geom(geom, max_km2=None):
    """closes holes within one shapely geometry."""

    # dissolve the exterior ring(s)
    if max_km2 is None:
        holes_closed = polygons(get_exterior_ring(get_parts(geom)))
        return unary_union(holes_closed)

    # start with a list containing the geometry,
    # then append all holes smaller than 'max_km2' to the list.
    holes_closed = [geom]
    singlepart = get_parts(geom)
    for part in singlepart:
        n_interior_rings = get_num_interior_rings(part)

        if not (n_interior_rings):
            continue

        for n in range(n_interior_rings):
            hole = polygons(get_interior_ring(part, n))

            if area(hole) / 1_000_000 < max_km2:
                holes_closed.append(hole)

    return unary_union(holes_closed)


def gdf_concat(
    gdfs: list[GeoDataFrame],
    crs: str | int | None = None,
    ignore_index: bool = True,
    geometry: str = "geometry",
    **kwargs,
) -> GeoDataFrame:
    """
    concatinates GeoDataFrames rowwise.
    Ignores index and changes to common crs.
    If no crs is given, chooses the first crs in the list of GeoDataFrames.

    Args:
        gdfs: list or tuple of GeoDataFrames to be concatinated.
        crs: common coordinate reference system each GeoDataFrames
            will be converted to before concatination. If None, it uses
            the crs of the first GeoDataFrame in the list or tuple.
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1. Defaults to True

    Returns:
        A GeoDataFrame.

    """

    gdfs = [gdf for gdf in gdfs if len(gdf)]

    if not len(gdfs):
        raise ValueError("All GeoDataFrames have 0 rows")

    if not crs:
        crs = gdfs[0].crs

    try:
        gdfs = [gdf.to_crs(crs) for gdf in gdfs]
    except ValueError:
        print(
            "Not all your GeoDataFrames have crs. If you are concatenating GeoDataFrames "
            "with different crs, the results will be wrong. First use set_crs to set the correct crs"
            "then the crs can be changed with to_crs()"
        )

    return GeoDataFrame(
        pd.concat(gdfs, ignore_index=ignore_index, **kwargs), geometry=geometry, crs=crs
    )


def to_gdf(
    geom: GeoSeries | Geometry | str | bytes, crs=None, **kwargs
) -> GeoDataFrame:
    """
    Converts a GeoSeries, shapely Geometry, wkt string or wkb bytes object to a
    GeoDataFrame.

    Args:
        geom: the object to be converted to a GeoDataFrame
        crs: if None (default), it uses the crs of the GeoSeries if GeoSeries
            is the input type. Otherwise, an exception is raised, saying that
            crs has to be specified.
    """

    if not crs:
        if not isinstance(geom, GeoSeries):
            raise ValueError("'crs' have to be specified when the input is a string.")
        crs = geom.crs

    if isinstance(geom, str):
        from shapely.wkt import loads

        geom = loads(geom)
        gdf = GeoDataFrame({"geometry": GeoSeries(geom)}, crs=crs, **kwargs)

    if isinstance(geom, bytes):
        from shapely.wkb import loads

        geom = loads(geom)
        gdf = GeoDataFrame({"geometry": GeoSeries(geom)}, crs=crs, **kwargs)

    else:
        gdf = GeoDataFrame({"geometry": GeoSeries(geom)}, crs=crs, **kwargs)

    return gdf


def push_geom_col(gdf: GeoDataFrame) -> GeoDataFrame:
    """Makes the geometry column the leftmost column in the GeoDataFrame."""
    geom_col = gdf._geometry_column_name
    return gdf.reindex(columns=[c for c in gdf.columns if c != geom_col] + [geom_col])


def clean_clip(
    gdf: GeoDataFrame | GeoSeries,
    mask: GeoDataFrame | GeoSeries | Geometry,
    keep_geom_type: bool = True,
    **kwargs,
) -> GeoDataFrame | GeoSeries:
    """
    Clips geometries to the mask extent, then cleans the geometries.
    geopandas.clip does a fast clipping, with no guarantee for valid outputs.
    Here, geometries are made valid, then invalid, empty, nan and None geometries are
    removed.

    """

    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):
        raise TypeError(f"'gdf' should be GeoDataFrame or GeoSeries, got {type(gdf)}")

    try:
        return gdf.clip(mask, keep_geom_type=keep_geom_type, **kwargs).pipe(clean_geoms)
    except Exception:
        gdf = clean_geoms(gdf)
        mask = clean_geoms(mask)
        return gdf.clip(mask, keep_geom_type=keep_geom_type, **kwargs).pipe(clean_geoms)


def sjoin(
    left_gdf: GeoDataFrame, right_gdf: GeoDataFrame, drop_dupcol: bool = True, **kwargs
) -> GeoDataFrame:
    """
    som gpd.sjoin bare at kolonner i right_gdf som også er i left_gdf fjernes
    (fordi det snart vil gi feilmelding i geopandas) og kolonner som har med index
     å gjøre fjernes, fordi sjoin returnerer index_right som kolonnenavn,
     som gir feilmelding ved neste join.
    """

    left_gdf = left_gdf.loc[:, ~left_gdf.columns.str.contains("index|level_")]
    right_gdf = right_gdf.loc[:, ~right_gdf.columns.str.contains("index|level_")]

    if drop_dupcol:
        right_gdf = right_gdf.loc[
            :,
            right_gdf.columns.difference(
                left_gdf.columns.difference([left_gdf._geometry_column_name])
            ),
        ]

    try:
        joined = left_gdf.sjoin(right_gdf, **kwargs)
    except Exception:
        right_gdf = right_gdf.to_crs(left_gdf.crs)
        left_gdf = clean_geoms(left_gdf, single_geom_type=True)
        right_gdf = clean_geoms(right_gdf, single_geom_type=True)
        joined = left_gdf.sjoin(right_gdf, **kwargs)

    return joined.loc[:, ~joined.columns.str.contains("index|level_")]


def snap_to(
    points: GeoDataFrame | GeoSeries,
    snap_to: GeoDataFrame | GeoSeries,
    max_dist: int | None = None,
    to_vertex: bool = False,
    copy: bool = False,
) -> GeoDataFrame | GeoSeries:
    """
    It takes a GeoDataFrame or GeoSeries of points and snaps them to the nearest point in a second
    GeoDataFrame or GeoSeries

    Args:
      points (GeoDataFrame | GeoSeries): The GeoDataFrame or GeoSeries of points to snap
      snap_to (GeoDataFrame | GeoSeries): The GeoDataFrame or GeoSeries to snap to
      max_dist (int): The maximum distance to snap to. Defaults to None.
      to_vertex (bool): If True, the points will snap to the nearest vertex of the snap_to geometry. If
        False, the points will snap to the nearest point on the snap_to geometry, which can be between two vertices
        if the snap_to geometry is line or polygon. Defaults to False
      copy (bool): If True, a copy of the GeoDataFrame is returned. Otherwise, the original
    GeoDataFrame. Defaults to False

    Returns:
      A GeoDataFrame or GeoSeries with the points snapped to the nearest point in the snap_to
    GeoDataFrame or GeoSeries.
    """

    unioned = snap_to.unary_union

    if copy:
        points = points.copy()

    def func(point, snap_to, to_vertex):
        if to_vertex:
            snap_to = to_multipoint(snap_to)

        if not max_dist:
            return nearest_points(point, snap_to)[1]

        nearest = nearest_points(point, snap_to)[1]
        return snap(point, nearest, tolerance=max_dist)

    if isinstance(points, GeoDataFrame):
        points[points._geometry_column_name] = points[
            points._geometry_column_name
        ].apply(lambda point: func(point, unioned, to_vertex))

    if isinstance(points, gpd.GeoSeries):
        points = points.apply(lambda point: func(point, unioned, to_vertex))

    return points


def to_multipoint(gdf, copy=False):
    """
    It takes a geometry and returns a multipoint geometry

    Args:
      gdf: The geometry to be converted. Can be a GeoDataFrame, GeoSeries or a shapely geometry.
      copy: If True, the geometry will be copied. Otherwise, it may be possible to modify the original
    geometry in-place, which can improve performance. Defaults to False

    Returns:
      A GeoDataFrame with the geometry column as a MultiPoint
    """
    from shapely import force_2d
    from shapely.wkt import loads

    if copy:
        gdf = gdf.copy()

    def _to_multipoint(gdf):
        koordinater = "".join(
            [x for x in gdf.wkt if x.isdigit() or x.isspace() or x == "." or x == ","]
        ).strip()

        alle_punkter = [
            loads(f"POINT ({punkt.strip()})") for punkt in koordinater.split(",")
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
        gdf = force_2d(gdf)
        gdf = _to_multipoint(unary_union(gdf))

    return gdf


def find_neighbours(
    gdf: GeoDataFrame | GeoSeries,
    possible_neighbours: GeoDataFrame | GeoSeries,
    id_col: str,
    within_distance: int = 0,
) -> list:
    """
    Finds all the geometries in another GeoDataFrame that intersects with the first geometry

    Args:
      gdf (GeoDataFrame | GeoSeries): the geometry
      possible_neighbours (GeoDataFrame | GeoSeries): the geometries that you want to find neighbours
    for
      id_col (str): The column in the GeoDataFrame that contains the unique identifier for each
    geometry.
      within_distance (int): The maximum distance between the two geometries. Defaults to 0

    Returns:
      A list of unique values from the id_col column in the joined dataframe.
    """

    if within_distance:
        if gdf.crs == 4326:
            warnings.warn(
                "'gdf' has latlon crs, meaning the 'within_distance' paramter "
                "will not be in meters, but degrees."
            )
        gdf = gdf.buffer(within_distance).to_frame()

    possible_neighbours = possible_neighbours.to_crs(gdf.crs)

    joined = gdf.sjoin(possible_neighbours, how="inner")

    return [x for x in joined[id_col].unique()]


def find_neighbors(
    gdf: GeoDataFrame | GeoSeries,
    possible_neighbors: GeoDataFrame | GeoSeries,
    id_col: str,
    within_distance: int = 0,
):
    """American alias for find_neighbours."""
    return find_neighbours(gdf, possible_neighbors, id_col, within_distance)


def gridish(
    gdf: GeoDataFrame, meters: int, x2: bool = False, minmax: bool = False
) -> GeoDataFrame:
    """
    Enkel rutedeling av dataene, for å kunne loope tunge greier for områder i valgfri
    størrelse. Gir dataene kolonne med avrundede minimum-xy-koordinater, altså det
    sørvestlige hjørnets koordinater avrundet.

    minmax=True gir kolonnen 'gridish_max', som er avrundede maksimum-koordinater,
    altså koordinatene i det nordøstlige hjørtet.
    Hvis man skal gjøre en overlay/sjoin og dataene er større i utstrekning enn
    områdene man vil loope for.

    x2=True gir kolonnen 'gridish2', med ruter 1/2 hakk nedover og bortover.
    Hvis grensetilfeller er viktig, kan man måtte loope for begge gridish-kolonnene.

    """

    # rund ned koordinatene og sett sammen til kolonne
    gdf["gridish"] = [
        f"{round(minx/meters)}_{round(miny/meters)}"
        for minx, miny in zip(gdf.geometry.bounds.minx, gdf.geometry.bounds.miny)
    ]

    if minmax:
        gdf["gridish_max"] = [
            f"{round(maxx/meters)}_{round(maxy/meters)}"
            for maxx, maxy in zip(gdf.geometry.bounds.maxx, gdf.geometry.bounds.maxy)
        ]

    if x2:
        gdf["gridish_x"] = gdf.geometry.bounds.minx / meters

        unike_x = gdf["gridish_x"].astype(int).unique()
        unike_x.sort()

        for x in unike_x:
            gdf.loc[
                (gdf["gridish_x"] >= x - 0.5) & (gdf["gridish_x"] < x + 0.5),
                "gridish_x2",
            ] = (
                x + 0.5
            )

        # samme for y
        gdf["gridish_y"] = gdf.geometry.bounds.miny / meters
        unike_y = gdf["gridish_y"].astype(int).unique()
        unike_y.sort()
        for y in unike_y:
            gdf.loc[
                (gdf["gridish_y"] >= y - 0.5) & (gdf["gridish_y"] < y + 0.5),
                "gridish_y2",
            ] = (
                y + 0.5
            )

        gdf["gridish2"] = (
            gdf["gridish_x2"].astype(str) + "_" + gdf["gridish_y2"].astype(str)
        )

        gdf = gdf.drop(["gridish_x", "gridish_y", "gridish_x2", "gridish_y2"], axis=1)

    return gdf


def random_points(n: int, mask=None) -> GeoDataFrame:
    """lager n tilfeldige punkter innenfor et gitt område (mask)."""

    if mask is None:
        x = np.array([random() * 10**7 for _ in range(n * 1000)])
        y = np.array([random() * 10**8 for _ in range(n * 1000)])
        punkter = to_gdf([loads(f"POINT ({x} {y})") for x, y in zip(x, y)], crs=25833)
        return punkter
    mask_kopi = mask.copy()
    mask_kopi = mask_kopi.to_crs(25833)
    out = GeoDataFrame({"geometry": []}, geometry="geometry", crs=25833)
    while len(out) < n:
        x = np.array([random() * 10**7 for _ in range(n * 1000)])
        x = x[(x > mask_kopi.bounds.minx.iloc[0]) & (x < mask_kopi.bounds.maxx.iloc[0])]

        y = np.array([random() * 10**8 for _ in range(n * 1000)])
        y = y[(y > mask_kopi.bounds.miny.iloc[0]) & (y < mask_kopi.bounds.maxy.iloc[0])]

        punkter = to_gdf([loads(f"POINT ({x} {y})") for x, y in zip(x, y)], crs=25833)
        overlapper = punkter.clip(mask_kopi)
        out = gdf_concat([out, overlapper])
    out = out.sample(n).reset_index(drop=True).to_crs(mask.crs)
    out["idx"] = out.index
    return out


def count_within_distance(
    gdf1: GeoDataFrame, gdf2: GeoDataFrame, distance=0, col_name="n"
) -> GeoDataFrame:
    """
    Teller opp antall nærliggende eller overlappende (hvis avstan=0) geometrier i
    to geodataframes. gdf1 returneres med en ny kolonne ('antall') som forteller hvor
    mange geometrier (rader) fra gdf2 som er innen spesifisert distance.
    """

    gdf1["temp_idx"] = range(len(gdf1))
    gdf2["temp_idx2"] = range(len(gdf2))

    if distance > 0:
        gdf2 = buff(gdf2[["geometry"]], distance)

    joined = (
        gdf1[["temp_idx", "geometry"]]
        .sjoin(gdf2[["geometry"]], how="inner")["temp_idx"]
        .value_counts()
    )

    gdf1[col_name] = gdf1["temp_idx"].map(joined).fillna(0)

    return gdf1.drop("temp_idx", axis=1)

import re
import warnings

import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryArray
from numpy.typing import NDArray
from shapely import (
    buffer,
    extract_unique_points,
    force_2d,
    get_coordinates,
    get_exterior_ring,
    get_parts,
    intersection,
    linearrings,
    linestrings,
    make_valid,
    polygons,
    voronoi_polygons,
)
from shapely.errors import GEOSException
from shapely.geometry import LinearRing, LineString, Point

from ..networkanalysis.closing_network_holes import get_angle
from .buffer_dissolve_explode import buff, dissexp, dissexp_by_cluster
from .conversion import coordinate_array, to_geoseries
from .duplicates import get_intersections, update_geometries
from .general import (
    clean_geoms,
    sort_large_first,
    sort_long_first,
    sort_small_first,
    to_lines,
)
from .geometry_types import get_geom_type, make_all_singlepart, to_single_geom_type
from .overlay import clean_overlay
from .polygon_operations import (
    close_all_holes,
    close_thin_holes,
    eliminate_by_longest,
    get_cluster_mapper,
    get_gaps,
    get_polygon_clusters,
)
from .polygons_as_rings import PolygonsAsRings
from .sfilter import sfilter, sfilter_inverse


warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


PRECISION = 1e-4
BUFFER_RES = 50


def coverage_clean(
    gdf: GeoDataFrame,
    tolerance: int | float,
    duplicate_action: str = "fix",
    spike_action: str = "ignore",
    grid_sizes: tuple[None | int] = (
        None,
        # 1e-6,
        # 1e-5,
        # 1e-4,
    ),
    logger=None,
) -> GeoDataFrame:
    """Fix thin gaps, holes, slivers and double surfaces.

    Rules:
    - Holes (interiors) thinner than the tolerance are closed.
    - Gaps between polygons are filled if thinner than the tolerance.
    - Sliver polygons thinner than the tolerance are eliminated
    into the neighbor polygon with the longest shared border.
    - Double surfaces thinner than the tolerance are eliminated.
    If duplicate_action is "fix", thicker double surfaces will
    be updated from top to bottom of the GeoDataFrame's rows.
    - Line and point geometries are removed.
    - MultiPolygons are exploded to Polygons.
    - Index is reset.

    Args:
        gdf: GeoDataFrame to be cleaned.
        tolerance: distance (usually meters) used as the minimum thickness
            for polygons to be eliminated. Any gap, hole, sliver or double
            surface that are empty after a negative buffer of tolerance / 2
            are eliminated into the neighbor with the longest shared border.
        duplicate action: Either "fix", "error" or "ignore".
            If "fix" (default), double surfaces thicker than the
            tolerance will be updated from top to bottom (function update_geometries)
            and then dissolved into the neighbor polygon with the longest shared border.
            If "error", an Exception is raised if there are any double surfaces thicker
            than the tolerance. If "ignore", double surfaces are kept as is.
        spike_action: Either "fix", "ignore" or "try".

    Returns:
        A GeoDataFrame with cleaned polygons.

    """

    _cleaning_checks(gdf, tolerance, duplicate_action, spike_action)

    if not gdf.index.is_unique:
        gdf = gdf.reset_index(drop=True)

    gdf = make_all_singlepart(close_thin_holes(gdf, tolerance)).loc[
        lambda x: x.geom_type.isin(["Polygon", "MultiPolygon"])
    ]
    gdf.geometry = shapely.simplify(gdf.geometry, PRECISION)

    gdf = clean_geoms(gdf)

    try:
        gaps = get_gaps(gdf, include_interiors=True)
    except GEOSException:
        for i, grid_size in enumerate(grid_sizes):
            try:
                gaps = get_gaps(gdf, include_interiors=True, grid_size=grid_size)
                break
            except GEOSException as e:
                if i == len(grid_sizes) - 1:
                    explore_geosexception(gdf, e)
                    raise e

    if duplicate_action == "ignore":
        double = GeoDataFrame({"geometry": []}, crs=gdf.crs)
        double["_double_idx"] = None
    else:
        double = get_intersections(gdf)
        double["_double_idx"] = range(len(double))

    if spike_action != "ignore":
        try:
            gdf = split_spiky_polygons(gdf, tolerance=tolerance, grid_sizes=grid_sizes)
        except GEOSException as e:
            if spike_action == "fix":
                raise e.__class__(
                    e, "Set spike_action='try' to ignore this error message."
                )

    gdf, slivers = split_out_slivers(gdf, tolerance)

    thin_gaps_and_double = pd.concat([gaps, double]).loc[
        lambda x: x.buffer(-tolerance / 2).is_empty
    ]

    all_are_thin = double["_double_idx"].isin(thin_gaps_and_double["_double_idx"]).all()

    if not all_are_thin and duplicate_action == "fix":
        gdf, thin_gaps_and_double, slivers = _properly_fix_duplicates(
            gdf, double, slivers, thin_gaps_and_double, tolerance
        )

        # gaps = pd.concat([gaps, more_gaps], ignore_index=True)
        # double = pd.concat([double, more_double], ignore_index=True)
    elif not all_are_thin and duplicate_action == "error":
        raise ValueError("Large double surfaces.")

    from ..maps.maps import explore
    from .conversion import to_gdf

    # x1 = sfilter(
    #     slivers, to_gdf([6.25326819, 62.60568757], 4326).to_crs(25833).buffer(5)
    # )
    # explore(x1, x2=buff(x1, -PRECISION / 2))
    # print([x.wkt for x in x1.geometry])
    # explore(
    #     gdf,
    #     slivers,
    #     thin_gaps_and_double,
    #     to_el=pd.concat([thin_gaps_and_double, slivers], ignore_index=True).loc[
    #         lambda x: ~x.buffer(-PRECISION / 10).is_empty
    #     ],
    #     to_el2=pd.concat([thin_gaps_and_double, slivers], ignore_index=True),
    #     mask=to_gdf([6.25326819, 62.60568757], 4326).to_crs(25833).buffer(100),
    # )

    to_eliminate = pd.concat([thin_gaps_and_double, slivers], ignore_index=True)
    to_eliminate.geometry = shapely.simplify(to_eliminate.geometry, PRECISION)

    # eliminate super-thin slivers causing weird geometries
    is_thin = to_eliminate.buffer(-PRECISION).is_empty
    thick, thin = to_eliminate[~is_thin], to_eliminate[is_thin]
    to_eliminate = eliminate_by_longest(
        thick, thin, remove_isolated=False, ignore_index=True
    )

    to_eliminate = to_eliminate.loc[lambda x: ~x.buffer(-PRECISION / 10).is_empty]

    to_eliminate["_eliminate_idx"] = range(len(to_eliminate))
    gdf["_poly_idx"] = range(len(gdf))

    to_eliminate["_cluster"] = get_cluster_mapper(to_eliminate.buffer(PRECISION))

    gdf_geoms_idx = gdf[["_poly_idx", "geometry"]]

    joined = to_eliminate.sjoin(gdf_geoms_idx, how="left")
    isolated = joined[lambda x: x["_poly_idx"].isna()]
    intersecting = joined[lambda x: x["_poly_idx"].notna()]

    # from ..maps.maps import explore
    # from .conversion import to_gdf

    # explore(
    #     joined,
    #     isolated,
    #     to_eliminate,
    #     gdf,
    #     mask=to_gdf([6.25326819, 62.60568757], 4326).to_crs(25833).buffer(100),
    # )

    poly_idx_mapper: pd.Series = (
        clean_overlay(
            intersecting[["_eliminate_idx", "geometry"]],
            buff(gdf_geoms_idx, tolerance, resolution=BUFFER_RES),
            geom_type="polygon",
        )
        .pipe(sort_large_first)
        .drop_duplicates("_eliminate_idx")
        .set_index("_eliminate_idx")["_poly_idx"]
    )
    intersecting["_poly_idx"] = intersecting["_eliminate_idx"].map(poly_idx_mapper)

    for i, grid_size in enumerate(grid_sizes):
        try:
            without_double = update_geometries(
                intersecting, geom_type="polygon", grid_size=grid_size
            ).drop(columns=["_eliminate_idx", "_double_idx", "index_right"])
            break
        except GEOSException as e:
            intersecting.geometry = shapely.simplify(
                intersecting.geometry, PRECISION * (10 * i + 1)
            )
            if i == len(grid_sizes) - 1:
                explore_geosexception(gdf, e)
                raise e

    not_really_isolated = isolated.drop(
        columns=[
            "_double_idx",
            "index_right",
        ]
    ).merge(without_double, on="_cluster", how="inner")

    really_isolated = isolated.loc[
        lambda x: ~x["_eliminate_idx"].isin(not_really_isolated["_eliminate_idx"])
    ]

    really_isolated["_poly_idx"] = (
        really_isolated["_cluster"] + gdf["_poly_idx"].max() + 1
    )

    # # dissolve overlapping isolated
    # _, _, isolated = get_polygon_clusters(
    #     gdf, without_double, isolated, cluster_col="_cluster"
    # )
    # # isolated["_cluster"] = get_cluster_mapper(isolated.buffer(PRECISION))

    # isolated["_poly_idx"] = (
    #     isolated["_cluster"]
    #     + max(gdf["_poly_idx"].max(), without_double["_poly_idx"].max())
    #     + 1
    # )
    # isolated = isolated.drop(
    #     columns=[
    #         "_double_idx",
    #         "_eliminate_idx",
    #         "_cluster",
    #         "index_right",
    #     ]
    # )

    # from ..maps.maps import explore
    # from .conversion import to_gdf

    # explore(
    #     without_double,
    #     not_really_isolated,
    #     really_isolated,
    #     isolated,
    #     intersecting,
    #     to_eliminate,
    #     gdf_cl=gdf,
    #     column="_poly_idx",
    #     mask=to_gdf([6.25326819, 62.60568757], 4326).to_crs(25833).buffer(100),
    # )

    cleaned = (
        dissexp(
            pd.concat([gdf, without_double, isolated, really_isolated]).drop(
                columns=["_cluster", "_eliminate_idx", "index_right", "_double_idx"],
                errors="ignore",
            ),
            by="_poly_idx",
            aggfunc="first",
            dropna=False,
        )
        .sort_index()
        .reset_index(drop=True)
        # .loc[lambda x: ~x.buffer(-PRECISION / 10).is_empty]
    )

    cleaned.geometry = shapely.make_valid(shapely.simplify(cleaned.geometry, PRECISION))

    for i, grid_size in enumerate(grid_sizes):
        try:
            cleaned = update_geometries(
                cleaned, geom_type="polygon", grid_size=grid_size
            )
            break
        except GEOSException as e:
            cleaned.geometry = shapely.simplify(
                cleaned.geometry, PRECISION * (10 * i + 1)
            )
            if i == len(grid_sizes) - 1:
                explore_geosexception(gdf, e)
                raise e

    #     from ..maps.maps import explore
    #     from .conversion import to_gdf
    #     explore(
    #         slivers,
    #         gdf,
    #         cleaned,
    #         joined,
    #         isolated,
    #         without_double,
    #         cleaned_d=get_intersections(cleaned),
    #         isolatedd=get_intersections(isolated),
    #         without_doubled=get_intersections(without_double),
    #         mask=to_gdf([4.70993435, 60.76502035
    # ], 4326)
    #                     .to_crs(25833)
    #                     .buffer(30),
    #                 )

    # isolated["_cluster"] = get_cluster_mapper(isolated)  # component_mapper
    # isolated = dissexp_by_cluster(sort_large_first(isolated), by=list(isolated.columns.difference(isolated._geometry_column_name)))

    # if logger:
    #     logger.debug("cleaned1", get_intersections(cleaned).area.sum())

    missing_indices: pd.Index = sfilter_inverse(
        gdf.representative_point(), cleaned
    ).index

    missing = clean_overlay(
        gdf.loc[missing_indices].drop(columns="_poly_idx"),
        cleaned,
        how="difference",
        geom_type="polygon",
    )

    # explore(
    #     without_double,
    #     not_really_isolated,
    #     really_isolated,
    #     isolated,
    #     intersecting,
    #     missing,
    #     gdf_cl=gdf,
    #     mask=to_gdf([6.25326819, 62.60568757], 4326).to_crs(25833).buffer(100),
    # )

    # if logger:
    #     logger.debug(
    #         "cleaned2",
    #         get_intersections(
    #             pd.concat([cleaned, missing], ignore_index=True)
    #         ).area.sum(),
    #     )

    cleaned = pd.concat([cleaned, missing], ignore_index=True)
    cleaned.geometry = shapely.make_valid(shapely.simplify(cleaned.geometry, PRECISION))

    return cleaned


def split_spiky_polygons(
    gdf: GeoDataFrame, tolerance: int | float, grid_sizes: tuple[None | int] = (None,)
) -> GeoDataFrame:
    if not len(gdf):
        return gdf

    if get_geom_type(gdf) != "polygon":
        raise ValueError("Geometries must be polygons.")

    gdf = make_all_singlepart(gdf)

    if not gdf.index.is_unique:
        gdf = gdf.reset_index(drop=True)

    polygons_without_spikes = gdf.buffer(tolerance / 2, join_style=2).buffer(
        -tolerance / 2, join_style=2
    )

    donuts_around_polygons = to_lines(
        polygons_without_spikes.to_frame("geometry")
    ).pipe(buff, 1e-3, copy=False)

    def remove_spikes(df):
        df = df.to_frame("geometry")
        df["_ring_idx"] = range(len(df))
        df = df.reset_index(drop=True)

        points = df.copy()
        points.geometry = extract_unique_points(points.geometry)
        points = points.explode(index_parts=False)

        # coords, indices = shapely.get_coordinates(df.geometry, return_index=True)
        # points = GeoDataFrame(
        #     {
        #         "geometry": GeoSeries(shapely.points(coords), index=indices),
        #     },
        #     crs=df.crs,
        # )
        points["_idx"] = range(len(points))
        # points["_ring_idx"] = indices

        not_spikes = points.sjoin(donuts_around_polygons).loc[
            lambda x: x["_ring_idx"] == x["index_right"]
        ]
        can_be_polygons = not_spikes.iloc[
            (not_spikes.groupby("_ring_idx").transform("size") >= 3).values
        ]

        without_spikes = (
            can_be_polygons.sort_values("_idx")
            .groupby("_ring_idx")["geometry"]
            .agg(LinearRing)
        )

        missing = df[~df["_ring_idx"].isin(without_spikes.index)].geometry

        # missing.loc[lambda x: x.buffer(-tolerance / 2).is_empty] = None

        # from ..maps.maps import explore
        # from .conversion import to_gdf

        # explore(
        #     without_spikes0hee=without_spikes,
        #     missing0=missing,
        #     can_be_polygons=can_be_polygons,
        #     mask=to_gdf([6.2532728, 62.605688], 4326).to_crs(25833).buffer(20),
        # )

        return pd.concat([without_spikes, missing]).sort_index()

    without_spikes = GeoDataFrame(
        {
            "geometry": PolygonsAsRings(gdf.geometry)
            .apply_geoseries_func(remove_spikes)
            .to_numpy()
        },
        crs=gdf.crs,
    ).pipe(to_single_geom_type, "polygon")

    is_thin = without_spikes.buffer(-tolerance / 2).is_empty
    without_spikes = pd.concat(
        [
            split_by_neighbors(
                without_spikes[is_thin], without_spikes, tolerance=tolerance
            ),
            without_spikes[~is_thin],
        ]
    )

    # from .conversion import to_gdf
    # from ..maps.maps import explore

    # explore(
    #     polygons_without_spikes,
    #     donuts_around_polygons,
    #     gdf1=(gdf),
    #     without_spikes1=without_spikes,
    # )

    # def try_for_grid_sizes(func):
    #     for i, grid_size in enumerate(grid_sizes):
    #         try:
    #             return func(grid_size=grid_size)
    #         except GEOSException as e:
    #             if i == len(grid_sizes) - 1:
    #                 raise e

    # without_spikes = try_for_grid_sizes(
    #     lambda grid_size: update_geometries(
    #         sort_small_first(without_spikes), grid_size=grid_size, geom_type="polygon"
    #     )
    # )
    # without_spikes = try_for_grid_sizes(
    #     lambda grid_size: update_geometries(
    #         sort_small_first(without_spikes), grid_size=grid_size, geom_type="polygon"
    #     )
    # )
    # return try_for_grid_sizes(
    #     lambda grid_size: clean_overlay(
    #         gdf, without_spikes, how="identity", grid_size=grid_size
    #     )
    # )

    # without_spikes = update_geometries(
    #     sort_small_first(without_spikes), geom_type="polygon"
    # )

    # put the spikes on top
    # without_spikes = coverage_clean(
    #     sort_small_first(without_spikes),
    #     tolerance=tolerance,
    #     spike_action="ignore",
    #     grid_sizes=grid_sizes,
    # )
    for _ in range(2):
        for i, grid_size in enumerate(grid_sizes):
            try:
                without_spikes = update_geometries(
                    sort_small_first(without_spikes), geom_type="polygon"
                )
                # without_spikes = coverage_clean(
                #     sort_small_first(without_spikes),
                #     tolerance=tolerance,
                #     spike_action="ignore",
                #     grid_sizes=grid_sizes,
                # )
                break
            except GEOSException as e:
                if i == len(grid_sizes) - 1:
                    raise e

    for i, grid_size in enumerate(grid_sizes):
        try:
            # explore(
            #     # polygons_without_spikes,
            #     # donuts_around_polygons,
            #     # gdf1=(gdf),
            #     # without_spikes1=without_spikes,
            #     **{"without_spikes_upd" + str(_): without_spikes},
            #     identityed=clean_overlay(
            #             gdf, without_spikes, how="identity", grid_size=grid_size
            #         ),
            #     mask=to_gdf([6.25327192, 62.60568806], 4326).to_crs(25833).buffer(50),
            # )
            return clean_overlay(
                gdf, without_spikes, how="identity", grid_size=grid_size
            )
        except GEOSException as e:
            if i == len(grid_sizes) - 1:
                raise e

    #     explore(
    #         **{f"without_spikes_grid{grid_size}": without_spikes},
    #         mask=to_gdf([4.70993435, 60.76502035], 4326).to_crs(25833).buffer(50),
    #     )
    # # if without_spikes.area.sum() <= gdf.area.sum() - tolerance:
    #     return clean_overlay(gdf, without_spikes, how="identity")

    # duplicates = get_intersections(without_spikes).loc[
    #     lambda x: ~x.buffer(-PRECISION * 10).is_empty
    # ]

    # explore(
    #     gdf2=(gdf),
    #     without_spikes2=without_spikes,
    #     duplicates2=duplicates,
    #     mask=to_gdf([8.00822745, 58.27210215], 4326).to_crs(25833).buffer(100),
    # )
    # from ..maps.maps import explore
    # from .conversion import to_gdf

    # explore(
    #     gdf=(gdf),
    #     without_spikes=(without_spikes),
    #     donuts_around_polygons=(donuts_around_polygons),
    #     polygons_without_spikes=(polygons_without_spikes),
    #     mask=to_gdf([4.72034128, 60.75264796], 4326).to_crs(25833).buffer(30),
    # )

    # if len(
    #     duplicates := get_intersections(without_spikes).loc[
    #         lambda x: ~x.buffer(-PRECISION * 10).is_empty
    #     ]
    # ):
    #     from ..maps.maps import explore

    #     new_duplicates = clean_overlay(
    #         without_spikes,
    #         duplicates,
    #         how="difference",
    #         geom_type="polygon",
    #     )
    #     print(new_duplicates)

    #     if len(new_duplicates.loc[lambda x: ~x.buffer(-PRECISION * 10).is_empty]):
    #         explore(
    #             gdf3=(gdf),
    #             without_spikes3=(without_spikes),
    #             new_duplicate3s=new_duplicates,
    #             duplicat3es=(duplicates),
    #         )
    #         raise ValueError(
    #             "duplicates", new_duplicates.area.sum(), new_duplicates.area.max()
    #         )

    # return clean_overlay(gdf, without_spikes, how="identity")


def remove_spikes(gdf: GeoDataFrame, tolerance: int | float) -> GeoDataFrame:
    """Remove thin spikes in polygons.

    Note that this function might be slow. Should only be used if nessecary.

    Args:
        gdf: GeoDataFrame of polygons
        tolerance: distance (usually meters) used as the minimum thickness
            for polygons to be eliminated. Any spike thinner than the tolerance
            will be removed.

    Returns:
        A GeoDataFrame of polygons without spikes thinner.
    """

    gdf.geometry = (
        PolygonsAsRings(gdf.geometry)
        .apply_numpy_func(_remove_spikes, args=(tolerance,))
        .to_numpy()
    )
    return gdf


def _remove_spikes(
    geoms: NDArray[LinearRing], tolerance: int | float
) -> NDArray[LinearRing]:
    if not len(geoms):
        return geoms
    geoms = to_geoseries(geoms).reset_index(drop=True)

    points = (
        extract_unique_points(geoms).explode(index_parts=False).to_frame("geometry")
    )

    points = get_angle_between_indexed_points(points)

    indices_with_spikes = points[
        lambda x: (x["angle_diff"] >= 180) & (x["angle_diff"] < 180.01)
    ].index.unique()

    rings_with_spikes = geoms  # [geoms.index.isin(indices_with_spikes)]
    # rings_without_spikes = geoms[~geoms.index.isin(indices_with_spikes)]

    def to_buffered_rings_without_spikes(x):
        polys = GeoSeries(make_valid(polygons(get_exterior_ring(x))))

        return (
            polys.buffer(-tolerance, resolution=BUFFER_RES)
            .explode(index_parts=False)
            .pipe(close_all_holes)
            .pipe(get_exterior_ring)
            .buffer(tolerance * 10)
        )

    buffered = to_buffered_rings_without_spikes(
        rings_with_spikes.buffer(tolerance / 2, resolution=BUFFER_RES)
    )

    points_without_spikes = (
        extract_unique_points(rings_with_spikes)
        .explode(index_parts=False)
        .loc[lambda x: x.index.isin(sfilter(x, buffered).index)]
    )

    # linearrings require at least 4 coordinate pairs, or three unique
    points_without_spikes = points_without_spikes.loc[
        lambda x: x.groupby(level=0).size() >= 3
    ]

    # need an index from 0 to n-1 in 'linearrings'
    to_int_index = {
        ring_idx: i
        for i, ring_idx in enumerate(sorted(set(points_without_spikes.index)))
    }
    int_indices = points_without_spikes.index.map(to_int_index)

    as_lines = pd.Series(
        linearrings(
            get_coordinates(points_without_spikes.geometry.values),
            indices=int_indices,
        ),
        index=points_without_spikes.index.unique(),
    )
    # as_lines = pd.concat([as_lines, rings_without_spikes])

    # the missing polygons are thin and/or spiky. Let's remove them
    missing = geoms.loc[~geoms.index.isin(as_lines.index)]

    missing = pd.Series(
        [None] * len(missing),
        index=missing.index.values,
    )

    return pd.concat([as_lines, missing]).sort_index()


def get_angle_between_indexed_points(point_df: GeoDataFrame):
    """ "Get angle difference between the two lines"""

    point_df["next"] = point_df.groupby(level=0)["geometry"].shift(-1)

    notna = point_df["next"].notna()

    this = coordinate_array(point_df.loc[notna, "geometry"].values)
    next_ = coordinate_array(point_df.loc[notna, "next"].values)

    point_df.loc[notna, "angle"] = get_angle(this, next_)
    point_df["prev_angle"] = point_df.groupby(level=0)["angle"].shift(1)

    point_df["angle_diff"] = point_df["angle"] - point_df["prev_angle"]

    return point_df


def _properly_fix_duplicates(gdf, double, slivers, thin_gaps_and_double, tolerance):
    # gdf = update_geometries(gdf)
    # gdf, more_slivers = split_out_slivers(gdf, tolerance)
    # slivers = pd.concat([slivers, more_slivers], ignore_index=True)
    # gaps = get_gaps(gdf, include_interiors=True)
    # double = get_intersections(gdf).pipe(update_geometries, geom_type="polygon")
    # double["_double_idx"] = range(len(double))
    # thin_gaps_and_double = pd.concat([gaps, double]).loc[
    #     lambda x: x.buffer(-tolerance / 2).is_empty
    # ]
    # return gdf, thin_gaps_and_double, slivers

    gdf = _dissolve_thick_double_and_update(gdf, double, thin_gaps_and_double)
    gdf, more_slivers = split_out_slivers(gdf, tolerance)
    slivers = pd.concat([slivers, more_slivers], ignore_index=True)
    gaps = get_gaps(gdf, include_interiors=True)
    assert "_double_idx" not in gaps
    double = get_intersections(gdf)
    double["_double_idx"] = range(len(double))
    thin_gaps_and_double = pd.concat([gaps, double], ignore_index=True).loc[
        lambda x: x.buffer(-tolerance / 2).is_empty
    ]

    return gdf, thin_gaps_and_double, slivers

    for _ in range(3):
        gdf = _dissolve_thick_double_and_update(gdf, double, thin_gaps_and_double)
        gdf, more_slivers = split_out_slivers(gdf, tolerance)
        slivers = pd.concat([slivers, more_slivers], ignore_index=True)
        gaps = get_gaps(gdf, include_interiors=True)
        assert "_double_idx" not in gaps
        double = get_intersections(gdf)
        double["_double_idx"] = range(len(double))
        thin_gaps_and_double = pd.concat([gaps, double], ignore_index=True).loc[
            lambda x: x.buffer(-tolerance / 2).is_empty
        ]
        all_are_thin = (
            double["_double_idx"].isin(thin_gaps_and_double["_double_idx"]).all()
        )
        if all_are_thin:
            return gdf, thin_gaps_and_double, slivers

    not_thin = double[
        lambda x: ~x["_double_idx"].isin(thin_gaps_and_double["_double_idx"])
    ]
    raise ValueError(
        "Failed to properly fix thick double surfaces",
        [x.wkt for x in not_thin.geometry],
    )


def _dissolve_thick_double_and_update(gdf, double, thin_double):
    large = (
        double.loc[~double["_double_idx"].isin(thin_double["_double_idx"])]
        .drop(columns="_double_idx")
        .pipe(sort_large_first)
        .pipe(update_geometries, geom_type="polygon")
    )
    return (
        clean_overlay(gdf, large, how="update")
        .pipe(sort_large_first)
        .pipe(update_geometries, geom_type="polygon")
    )


def _cleaning_checks(gdf, tolerance, duplicate_action, spike_action):
    if not len(gdf) or not tolerance:
        return gdf
    if get_geom_type(gdf) != "polygon":
        raise ValueError("Must be polygons.")
    if tolerance < PRECISION:
        raise ValueError(
            f"'tolerance' must be larger than {PRECISION} to avoid "
            "problems with floating point precision."
        )
    if duplicate_action not in ["fix", "error", "ignore"]:
        raise ValueError("duplicate_action must be 'fix', 'error' or 'ignore'")
    if spike_action not in ["fix", "try", "ignore"]:
        raise ValueError("duplicate_action must be 'fix', 'try' or 'ignore'")


def split_out_slivers(
    gdf: GeoDataFrame | GeoSeries, tolerance: float | int
) -> tuple[GeoDataFrame, GeoDataFrame] | tuple[GeoSeries, GeoSeries]:
    is_sliver = gdf.buffer(-tolerance / 2).is_empty
    slivers = gdf.loc[is_sliver]
    gdf = gdf.loc[~is_sliver]
    return gdf, slivers


def split_by_neighbors(df, split_by, tolerance):
    if not len(df):
        return df

    split_by = split_by.copy()
    split_by.geometry = shapely.simplify(split_by.geometry, tolerance)

    intersecting_lines = (
        clean_overlay(to_lines(split_by), buff(df, tolerance), how="identity")
        .pipe(get_line_segments)
        .reset_index(drop=True)
    )

    endpoints = intersecting_lines.boundary.explode(index_parts=False)

    extended_lines = GeoDataFrame(
        {
            "geometry": extend_lines(
                endpoints.loc[lambda x: ~x.index.duplicated(keep="first")].values,
                endpoints.loc[lambda x: ~x.index.duplicated(keep="last")].values,
                distance=tolerance * 3,
            )
        },
        crs=df.crs,
    )

    buffered = buff(extended_lines, tolerance, single_sided=True)

    from ..maps.maps import explore
    from .conversion import to_gdf

    explore(
        df,
        intersecting_lines,
        extended_lines,
        buffered,
        endpoints=endpoints.to_frame(),
        mask=to_gdf([4.69931532, 60.777209065], 4326).to_crs(25833).buffer(10),
    )
    return clean_overlay(df, buffered, how="identity")


def extend_lines(arr1, arr2, distance):
    if len(arr1) != len(arr2):
        raise ValueError
    if not len(arr1):
        return arr1

    arr1, arr2 = arr2, arr1  # TODO fix

    coords1 = coordinate_array(arr1)
    coords2 = coordinate_array(arr2)

    dx = coords2[:, 0] - coords1[:, 0]
    dy = coords2[:, 1] - coords1[:, 1]
    len_xy = np.sqrt((dx**2.0) + (dy**2.0))
    x = coords1[:, 0] + (coords1[:, 0] - coords2[:, 0]) / len_xy * distance
    y = coords1[:, 1] + (coords1[:, 1] - coords2[:, 1]) / len_xy * distance

    new_points = np.array([None for _ in range(len(arr1))])
    new_points[~np.isnan(x)] = shapely.points(x[~np.isnan(x)], y[~np.isnan(x)])

    new_points[~np.isnan(x)] = make_lines_between_points(
        arr2[~np.isnan(x)], new_points[~np.isnan(x)]
    )
    return new_points


def make_lines_between_points(
    arr1: NDArray[Point], arr2: NDArray[Point]
) -> NDArray[LineString]:
    if arr1.shape != arr2.shape:
        raise ValueError(
            f"Arrays must have equal shape. Got {arr1.shape} and {arr2.shape}"
        )
    coords: pd.DataFrame = pd.concat(
        [
            pd.DataFrame(get_coordinates(arr1), columns=["x", "y"]),
            pd.DataFrame(get_coordinates(arr2), columns=["x", "y"]),
        ]
    ).sort_index()

    return linestrings(coords.values, indices=coords.index)


def get_line_segments(lines) -> GeoDataFrame:
    assert lines.index.is_unique
    if isinstance(lines, GeoDataFrame):
        multipoints = lines.assign(
            **{
                lines._geometry_column_name: force_2d(
                    extract_unique_points(lines.geometry.values)
                )
            }
        )
        return multipoints_to_line_segments(multipoints.geometry)

    multipoints = GeoSeries(extract_unique_points(lines.values), index=lines.index)

    return multipoints_to_line_segments(multipoints)


def multipoints_to_line_segments(multipoints: GeoSeries) -> GeoDataFrame:
    if not len(multipoints):
        return GeoDataFrame({"geometry": multipoints}, index=multipoints.index)

    try:
        crs = multipoints.crs
    except AttributeError:
        crs = None

    try:
        point_df = multipoints.explode(index_parts=False)
        if isinstance(point_df, GeoSeries):
            point_df = point_df.to_frame("geometry")
    except AttributeError:
        points, indices = get_parts(multipoints, return_index=True)
        if isinstance(multipoints.index, pd.MultiIndex):
            indices = pd.MultiIndex.from_arrays(indices, names=multipoints.index.names)

        point_df = pd.DataFrame({"geometry": GeometryArray(points)}, index=indices)

    point_df["next"] = point_df.groupby(level=0)["geometry"].shift(-1)

    first_points = point_df.loc[lambda x: ~x.index.duplicated(), "geometry"]
    is_last_point = point_df["next"].isna()

    point_df.loc[is_last_point, "next"] = first_points
    assert point_df["next"].notna().all()

    point_df["geometry"] = [
        LineString([x1, x2]) for x1, x2 in zip(point_df["geometry"], point_df["next"])
    ]
    return GeoDataFrame(point_df.drop(columns=["next"]), geometry="geometry", crs=crs)


def explore_geosexception(gdf, e: GEOSException):
    from ..maps.maps import explore
    from .conversion import to_gdf

    pattern = r"(\d+\.\d+)\s+(\d+\.\d+)"
    matches = re.findall(pattern, str(e))
    coords_in_error_message = [(float(match[0]), float(match[1])) for match in matches]
    exception_point = to_gdf(coords_in_error_message, crs=gdf.crs)
    if len(exception_point):
        exception_point["wkt"] = exception_point.to_wkt()
        explore(gdf, exception_point, mask=exception_point.buffer(100))
    else:
        explore(gdf)

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors
from shapely import line_merge, union, force_2d, shortest_line
from geopandas import GeoDataFrame, GeoSeries

from geopandas_util import clean_geoms, gdf_concat


class ZeroRoadsError(Exception):
    "The roads have 0 rows."


def find_isolated_networks(roads: GeoDataFrame, max_length: int | None) -> GeoDataFrame:

    roads = roads.reset_index(drop=True)

    roads, nodes = make_node_ids(roads)

    deadends = roads.loc[
        (roads["n_source"] <= 1) |
        (roads["n_target"] <= 1)
    ]

    isolated = ()
    not_isolated = ()
    for i in deadends.index:

        if i in isolated or i in not_isolated:
            continue

        # find road(s) that intersects with current deadend
        intersects = roads.loc[
            roads.intersects(roads.loc[i, "geometry"])
        ]

        # then find the roads that intersects with 'intersects', working our way outwards in the network
        indices = (i,)
        while True:

            if len(
                i for i in indices
                or i in not_isolated
            ):
                not_isolated = not_isolated + tuple(
                    i for i in indices if i not in not_isolated
                )
                break

            new_indices = tuple(i for i in intersects.index
                             if i not in indices)

            if not len(new_indices):
                isolated = isolated + indices
                break

            indices = indices + new_indices

            length_now = sum(roads.loc[roads.index.isin(indices)].length)

            if length_now > max_length:
                not_isolated = not_isolated + tuple(
                    i for i in indices if i not in not_isolated
                )
                break
            
            intersects_new_only = roads.loc[intersects_new_only]

            intersects = roads.loc[
                roads.intersects(intersects_new_only.unary_union)
            ]

    roads["isolated"] = np.where(
        roads.index.isin(not_isolated),
        0,
        1
    )

    return roads


def make_edge_wkt_cols(roads):
    # hent ut linjenes endpoints.
    # men først: sirkler har ingen endpoints og heller ingen funksjon. Disse må fjernes
    roads["temp_idx"] = roads.index
    endpoints = roads.copy()
    endpoints["geometry"] = endpoints.geometry.boundary
    circles = endpoints.loc[endpoints.is_empty, "temp_idx"]  # sirkler har tom boundary
    roads = (roads
                .loc[~roads.temp_idx.isin(circles)]
                .drop("temp_idx", axis=1)
                .reset_index(drop=True)
    )

    endpoints = roads.geometry.boundary.explode(ignore_index=True)  

    assert (
        len(endpoints) / len(roads) == 2
    ), "The lines should have only two endpoints each. Try splitting multilinestrings with explode."

    wkt_geom = [f"POINT ({x} {y})" for x, y in zip(endpoints.x, endpoints.y)]
    roads["source_wkt"], roads["target_wkt"] = (
        wkt_geom[0::2],
        wkt_geom[1::2],
    )

    return roads


def make_node_ids(roads: GeoDataFrame) -> (GeoDataFrame, GeoDataFrame):
    """Nye node-id-er som følger index (fordi indexes med numpy arrays i avstand_til_nodes())"""

    roads = make_edge_wkt_cols(roads)

    sources = roads[["source_wkt"]].rename(columns={"source_wkt": "wkt"})
    targets = roads[["target_wkt"]].rename(columns={"target_wkt": "wkt"})

    nodes = pd.concat([sources, targets], axis=0, ignore_index=True)

    nodes["n"] = (nodes
                    .assign(n=1)
                    .groupby("wkt")
                    ["n"]
                    .transform("sum")
                    )
    
    nodes = nodes.drop_duplicates(subset=["wkt"]).reset_index(drop=True)
    
    nodes["node_id"] = nodes.index
    nodes["node_id"] = nodes["node_id"].astype(str)

    id_dict = {wkt: node_id for wkt, node_id in zip(nodes["wkt"], nodes["node_id"])}
    roads["source"] = roads["source_wkt"].map(id_dict)
    roads["target"] = roads["target_wkt"].map(id_dict)

    n_dict = {wkt: n for wkt, n in zip(nodes["wkt"], nodes["n"])}
    roads["n_source"] = roads["source_wkt"].map(n_dict)
    roads["n_target"] = roads["target_wkt"].map(n_dict)

    nodes["geometry"] = gpd.GeoSeries.from_wkt(nodes.wkt, crs=roads.crs)
    nodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs=roads.crs)
    nodes = nodes.reset_index(drop=True)

    return roads, nodes


def close_network_holes(roads, max_dist, min_dist=0, deadends_only=False, hole_col: str | None = "hole"):
    """
    Lager rette linjer der det er små hull i networket.
    Bruker NearestNeighbors fra scikit-learn, fordi det er utrolig raskt. Man trenger ikke loope for områder en gang.
    scikit-learn bruker numpy arrays, som må konverteres tilbake til geopandas via index-ene.
    """

    roads, nodes = make_node_ids(roads)

    if deadends_only:
        new_roads = find_holes_deadends(nodes, max_dist, min_dist)
    else:
        new_roads = find_holes_all_roads(roads, nodes, max_dist, min_dist)

    wkt_id_dict = {wkt: id for wkt, id in zip(nodes["wkt"], nodes["node_id"])}
    new_roads["source"] = new_roads["source_wkt"].map(wkt_id_dict)
    new_roads["target"] = new_roads["target_wkt"].map(wkt_id_dict)

    if any(new_roads.source.isna()) or any(new_roads.target.isna()):
        raise ValueError("Missing source/target ids.")

    if hole_col:
        new_roads[hole_col] = 1
    
    return gdf_concat([roads, new_roads])


def find_holes_all_roads(roads, nodes, max_dist, min_dist=0, n=10):

    crs = nodes.crs

    # velger ut nodene som kun finnes i én lenke. Altså blindveier i en networksanalyse.
    deadends_source = roads.loc[roads.n_source <= 1].rename(
        columns={"source_wkt": "wkt", "target_wkt": "wkt_andre_ende"}
    )
    deadends_source["geometry"] = gpd.GeoSeries.from_wkt(
        deadends_source["wkt"], crs=crs
    )
    deadends_target = roads.loc[roads.n_target <= 1].rename(
        columns={"source_wkt": "wkt_andre_ende", "target_wkt": "wkt"}
    )
    deadends_target["geometry"] = gpd.GeoSeries.from_wkt(
        deadends_target["wkt"], crs=crs
    )

    deadends = pd.concat([deadends_source, deadends_target], ignore_index=True)
    deadends_andre_ende = deadends.copy()
    deadends_andre_ende["geometry"] = gpd.GeoSeries.from_wkt(
        deadends_andre_ende["wkt_andre_ende"], crs=crs
    )

    deadends_lengder = deadends.length
    deadends_array = np.array(
        [(x, y) for x, y in zip(deadends.geometry.x, deadends.geometry.y)]
    )

    nodes_array = np.array([(x, y) for x, y in zip(nodes.geometry.x, nodes.geometry.y)])

    if len(deadends) <= 1:
        deadends["minutter"] = -1
        return deadends

    # finn nærmeste naboer
    n = n if len(deadends) >= n else len(deadends)
    nbr = NearestNeighbors(n_neighbors=n, algorithm="ball_tree").fit(nodes_array)
    avstander, idxs = nbr.kneighbors(deadends_array)

    fra = []
    til = []
    for i in np.arange(1, n):
        len_naa = len(fra)

        idxs1 = idxs[:, i]
        avstander1 = avstander[:, i]

        avstander_andre_ende1 = deadends_andre_ende.distance(
            nodes.loc[idxs1], align=False
        )

        # henter ut blindveg-wkt og node-wkt hvis avstanden er mellom max-min og i riktig retning (
        fratil = [
            (geom, nodes.loc[idx, "wkt"])
            for geom, idx, dist, dist_andre, lengde in zip(
                deadends["wkt"],
                idxs1,
                avstander1,
                avstander_andre_ende1,
                deadends_lengder,
            )
            if dist < max_dist and dist > min_dist and dist < dist_andre - lengde * 0.25
        ]

        til = til + [t for f, t in fratil if f not in fra]
        fra = fra + [f for f, t in fratil if f not in fra]

        if len_naa == len(fra):
            break

    # lag GeoDataFrame med rette linjer
    fra = gpd.GeoSeries.from_wkt(fra, crs=crs)
    til = gpd.GeoSeries.from_wkt(til, crs=crs)
    new_roads = shortest_line(fra, til)
    new_roads = gpd.GeoDataFrame(
        {"geometry": new_roads}, geometry="geometry", crs=crs
    )

    if not len(new_roads):
        return new_roads

    new_roads = make_edge_wkt_cols(new_roads)

    return new_roads


def find_holes_deadends(nodes, max_dist, min_dist=0):
    """
    Lager rette linjer der det er små hull i networket.
    Bruker NearestNeighbors fra scikit-learn, fordi det er utrolig raskt. Man trenger ikke loope for områder en gang.
    scikit-learn bruker numpy arrays, som må konverteres tilbake til geopandas via index-ene.
    """

    crs = nodes.crs

    # velger ut nodene som kun finnes i én lenke. Altså blindveier i en networksanalyse.
    deadends = nodes[nodes["n"] <= 1]

    # viktig å nullstille index siden sklearn kneighbors gir oss en numpy.array med indekser
    deadends = deadends.reset_index(drop=True)

    if len(deadends) <= 1:
        deadends["minutter"] = -1
        return deadends

    # koordinater i tuple som numpy array
    deadends_array = np.array(
        [(x, y) for x, y in zip(deadends.geometry.x, deadends.geometry.y)]
    )

    # finn nærmeste to naboer
    nbr = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(deadends_array)
    avstander, idxs = nbr.kneighbors(deadends_array)

    # velg ut nest nærmeste (nærmeste er fra og til samme punkt)
    avstander = avstander[:, 1]
    idxs = idxs[:, 1]

    """
    Nå har vi 1d-numpy arrays av lik lengde som blindvegene. 
    'idxs' inneholder numpy-indeksen for vegen som er nærmest, altså endepunktene for de nye lenkene.
    For å konvertere dette fra numpy til geopandas, trengs geometri og node-id. 
    """

    # fra-geometrien kan hentes direkte siden avstandene og blindvegene har samme rekkefølge
    fra = np.array(
        [
            geom.wkt
            for dist, geom in zip(avstander, deadends.geometry)
            if dist < max_dist and dist > min_dist
        ]
    )

    # til-geometrien må hentes via index-en
    til = np.array(
        [
            deadends.loc[idx, "wkt"]
            for dist, idx in zip(avstander, idxs)
            if dist < max_dist and dist > min_dist
        ]
    )

    # lag GeoDataFrame med rette linjer
    fra = gpd.GeoSeries.from_wkt(fra, crs=crs)
    til = gpd.GeoSeries.from_wkt(til, crs=crs)
    new_roads = shortest_line(fra, til)
    new_roads = gpd.GeoDataFrame(
        {"geometry": new_roads}, geometry="geometry", crs=crs
    )

    if not len(new_roads):
        return new_roads

    new_roads = make_edge_wkt_cols(new_roads)

    return new_roads


def cut_lines(gdf, avstand):
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union
    from shapely import force_2d

    def cut(line, distance):
        """fra shapely-dokumentasjonen"""
        if distance <= 0.0 or distance >= line.length:
            return line
        coords = list(line.coords)
        for i, p in enumerate(coords):
            pd = line.project(Point(p))
            if pd == distance:
                return unary_union(
                    [LineString(coords[: i + 1]), LineString(coords[i:])]
                )
            if pd > distance:
                cp = line.interpolate(distance)
                return unary_union(
                    [
                        LineString(coords[:i] + [(cp.x, cp.y)]),
                        LineString([(cp.x, cp.y)] + coords[i:]),
                    ]
                )

    cut_vektorisert = np.vectorize(cut)

    gdf["geometry"] = force_2d(gdf.geometry)

    gdf = gdf.explode(ignore_index=True)

    over_avstand = gdf.loc[gdf.length > avstand]
    under_avstand = gdf.loc[gdf.length <= avstand]

    for x in [10, 5, 1]:
        maks_lengde = max(over_avstand.length)

        while maks_lengde > avstand * x + 1:
            maks_lengde = over_avstand.length.max()

            over_avstand["geometry"] = cut_vektorisert(over_avstand.geometry, avstand)

            over_avstand = over_avstand.explode(ignore_index=True)

            if maks_lengde == max(over_avstand.length):
                break

    over_avstand = over_avstand.explode(ignore_index=True)

    return pd.concat([under_avstand, over_avstand], ignore_index=True)


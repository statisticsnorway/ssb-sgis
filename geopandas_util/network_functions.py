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































# funksjon som tilpasser vegnettet til classen Graf().
# mange parametre for at den skal kunne brukes for alle network, men burde nok vært splitta litt opp.
def make_network(
    roads: GeoDataFrame,
    directed: bool = True,
    source: str = "fromnode",
    target: str = "tonode",
    linkid: str = "linkid",
    minutter=("drivetime_fw", "drivetime_bw"),
    vegkategori: str = "category",
    oneway="oneway",
    find_isolated=False,
    extend_lines: int = None,  # meter
    turn_restrictions=None,
    stigningsprosent=False,
    keep_cols: None | list | tuple | str = None,
    copy=True,
) -> GeoDataFrame:

    """
    """

    if not isinstance(roads, GeoDataFrame):
        raise TypeError(f"'roads' should be GeoDataFrame, got {type(roads)}")

    if not len(roads):
        raise ZeroRoadsError

    if not roads._geometry_column_name == "geometry":
        roads = roads.rename_geometry('geometry')

    if copy:
        roads = roads.copy()

    if keep_cols:
        if isinstance(keep_cols, str):
            keep_cols = [keep_cols]
        assert isinstance(
            keep_cols, (list, tuple)
        ), "'keep_cols' må være string/liste/tuple med kolonnenavn."
        keep_cols = list(set(keep_cols))
    else:
        keep_cols = []

    cols_to_keep = ["geometry"]

    roads.columns = [col.lower() for col in roads.columns]

    for col in [source, target, linkid]:
        col = col.lower()
        if col in roads.columns:
            cols_to_keep = cols_to_keep + [col]

    # hvis ikke angitte kolonner finnes i vegdataene, sjekk om andre kolonner matcher.
    # lager ny kolonne hvis ingen matcher. Gir feilmelding hvis flere enn én matcher.
    roads["source"] = find_source(roads, source)
    roads["target"] = find_target(roads, target)
    roads["linkid"] = find_linkid(roads, linkid)
    roads["category"] = find_vegkategori(roads, vegkategori)
    roads["drivetime_fw"], roads["drivetime_bw"] = find_minutter(
        roads, minutter
    )

    if not "oneway" in roads.columns:
        roads["oneway"] = np.nan

    if not "sperring" in roads.columns:
        roads["sperring"] = -1

    # litt opprydning
    cols_to_keep = [
        "idx",
        "source",
        "target",
        "linkid",
        "drivetime_fw",
        "drivetime_bw",
        "oneway",
        "sperring",
        "category",
        "KOMMUNENR",
        "geometry",
    ] + keep_cols

    roads = (
        roads.to_crs(25833)
        [cols_to_keep]
        .pipe(clean_geoms)
        .assign(geometry=lambda x: line_merge(x.geometry))
        .reset_index(drop=True)
    )

    if not len(roads):
        raise ZeroRoadsError

    roads["idx"] = roads.index

    # hvis noen lenker fortsatt er multilinestrings, må de splittes for å ikke ha flere enn to endpoints
    n = len(roads)
    roads = roads.explode(ignore_index=True)
    if len(roads) < n:
        if n-len(roads)==1:
            print(f"Warning: 1 multi-geometry was split into single part geometries. Minute columns will be wrong for these rows.")
        else:
            print(
                f"Warning: {n-len(roads)} multi-geometries were split into single part geometries. Minute columns will be wrong for these rows."
            )

    if find_isolated:
        roads = find_isolated_networks(roads, lengde=10000, ruteloop_m=2250)
    else:
        roads["isolert"] = np.nan

    roads = make_edge_wkt_cols(roads)

    if not len(roads):
        raise ZeroRoadsError

    """
    del opp lange linjer?
    while np.max(veglenker.length) > 1001:
        veglenker = kutt_linjer(veglenker, 1000)
    while np.max(veglenker.length) > 301:
        veglenker = kutt_linjer(veglenker, 300)
    while np.max(veglenker.length) > 101:
        veglenker = kutt_linjer(veglenker, 100)
    while np.max(veglenker.length) > 51:
        veglenker = kutt_linjer(veglenker, 50)
    """

    # lag kolonner med geometritekst (wkt) for source og target

    if stigningsprosent:
        roads = beregn_stigningsprosent(roads, endpoints)
        keep_cols = keep_cols + ["stigningsprosent"]

    roads["geometry"] = force_2d(roads.geometry)

    if not directed or all(roads["oneway"].isna()):
        veglenker = roads
        veglenker["minutter"] = np.where(
            veglenker["drivetime_fw"].fillna(0) > 0,
            veglenker["drivetime_fw"],
            veglenker["drivetime_bw"],
        )
    else:
        veglenker = snu_og_samle_veglenker(roads, stigningsprosent)

    if not len(roads):
        raise ZeroRoadsError

    if turn_restrictions:
        veglenker = lag_turn_restrictions(veglenker, turn_restrictions)
    else:
        veglenker["turn_restriction"] = np.nan

    if utvid:
        if not isinstance(utvid, (float, int)):
            raise ValueError(
                "utvid må være et tall (antall meter man vil utvide linjer)"
            )
        veglenker = close_network_holes(veglenker, utvid)
    
    # nye node-id-er som følger index (fordi det indexes med numpy arrays i avstand_til_nodes())
    veglenker, nodes = make_node_ids(veglenker)

    veglenker["meter"] = veglenker.length

    veglenker = veglenker.loc[
        (veglenker.minutter > 0) | (veglenker.minutter.isna()),
        [
            "source",
            "target",
            "minutter",
            "meter",
            "turn_restriction",
            "oneway",
            "sperring",
            vegkategori,
            "isolert",
            "KOMMUNENR",
            "source_wkt",
            "target_wkt",
            "geometry",
        ]
        + keep_cols,
    ]

    # fjern kolonner som ikke ble brukt
    for col in [
        "minutter",
        "turn_restriction",
        "isolert",
        vegkategori,
        "sperring",
        "KOMMUNENR",
        "oneway",
    ]:
        if (
            len(
                veglenker[
                    ~(
                        (veglenker[col].isna())
                        | (veglenker[col] == 0.02)
                        | (veglenker[col] == -1)
                    )
                ]
            )
            == 0
        ):
            veglenker = veglenker.drop(col, axis=1)

    return veglenker





def snu_og_samle_veglenker(roads, stigningsprosent):
    # velg ut de enveiskjørte og snu source og target for lenkene som går "feil" vei
    ft = roads.loc[(roads.oneway == "FT") | (roads.oneway == "F")]
    tf = roads.loc[(roads.oneway == "TF") | (roads.oneway == "T")]
    tf = tf.rename(
        columns={
            "source": "target",
            "target": "source",
            "source_wkt": "target_wkt",
            "target_wkt": "source_wkt",
        }
    )

    # dupliser lenkene som går begge veier og snu source og target i den ene
    begge_retninger1 = roads[roads.oneway == "B"]
    begge_retninger2 = begge_retninger1.rename(
        columns={
            "source": "target",
            "target": "source",
            "source_wkt": "target_wkt",
            "target_wkt": "source_wkt",
        }
    )

    # lag minutt-kolonne
    begge_retninger1 = begge_retninger1.rename(columns={"drivetime_fw": "minutter"})
    begge_retninger2 = begge_retninger2.rename(columns={"drivetime_bw": "minutter"})
    ft = ft.rename(columns={"drivetime_fw": "minutter"})
    tf = tf.rename(columns={"drivetime_bw": "minutter"})

    if stigningsprosent:
        tf["stigningsprosent"] = tf["stigningsprosent"] * -1
        begge_retninger2["stigningsprosent"] = begge_retninger2["stigningsprosent"] * -1

    # oneway=="N" er sperringer fram til og med 2021
    n = roads.loc[roads.oneway == "N"]
    if len(n) > 0:
        n["minutter"] = np.where(
            (n["drivetime_fw"].isna())
            | (n["drivetime_fw"] == 0)
            | (n["drivetime_fw"] == ""),
            n["drivetime_bw"],
            n["drivetime_fw"],
        )
        n2 = n.rename(
            columns={
                "source": "target",
                "target": "source",
                "source_wkt": "target_wkt",
                "target_wkt": "source_wkt",
            }
        )
        veglenker = gdf_concat([begge_retninger1, begge_retninger2, ft, tf, n, n2])
    else:
        veglenker = gdf_concat([begge_retninger1, begge_retninger2, ft, tf])

    return veglenker





def find_isolated_networks(roads, lengde: int, ruteloop_m: int):
    """
    Gir vegdataene kolonnen 'isolert', som indikerer hvilke roads som er adskilt fra hovedvegnettet med bom/sperring e.l..
    Dette gjøres ved å samle roads som nesten overlapper (innen 0.001 meter), så velge ut ansamlingene som er under en viss størrelse og utstrekning.
    Dette er fryktelig tungt, så det gjøres i loop for et lite område av gangen.
    Så gjentas loopen for områder som er halvveis forskjøvet på grunn av grensetilfeller.
    Vegene som er med i begge loops, anses som isolerte.
    """

    if not "idx" in roads.columns:
        roads = roads.reset_index(drop=True)
        roads["idx"] = roads.index

    # gir vegdataene to kolonner med koordinatkategorier. Den andre kolonnen inneholder rutekategorier som er halvveis forskøvet
    roads = gridish(roads, meter=ruteloop_m, x2=True)

    # fjerner sperringer før beregningen
    if "sperring" in roads.columns:
        ikke_sperringer = roads.loc[roads.sperring.astype(int) != 1]
        sperringer = roads.loc[roads.sperring.astype(int) == 1]
    else:
        ikke_sperringer = roads
        sperringer = None

    # samle nesten overlappende med buffer, dissolve og explode. Loopes for hver rute.
    def buffdissexp_gridish_loop(roads, sperringer, lengde, kolonne):
        # lagrer veg-indeksene (idx) i tuple fordi det tar mindre plass enn lister
        kanskje_isolerte = ()
        for i in roads[kolonne].unique():
            vegene = roads.loc[roads[kolonne] == i, ["geometry"]]
            if sperringer is not None:
                sperringene = sperringer.loc[
                    sperringer[kolonne] == i, ["geometry", "idx"]
                ]

            vegene["geometry"] = vegene.buffer(
                0.001, resolution=1
            )  # lavest mulig oppløsning for å få det fort
            dissolvet = vegene.dissolve()
            singlepart = dissolvet.explode(ignore_index=True)

            # velger network under gitt lengde - hvis de er under halvparten av total lengde og mindre utstrekning enn total lengde
            sum_lengde = dissolvet.length.sum()
            singlepart[
                "utstrekning"
            ] = (
                singlepart.convex_hull.area
            )  # fordi lange, isolerte roads, gjerne skogsbilroads, kan være brukbare for turer innad i skogen.
            lite_network = singlepart[
                (singlepart.length < lengde * 2)
                & (singlepart.length < sum_lengde * 0.5)
                & (singlepart["utstrekning"] < sum_lengde)
            ]

            # legg til nye idx-er i tuple-en med kanskje isolerte network
            for geom in lite_network.geometry:
                nye_kanskje_isolerte = tuple(roads.loc[roads.within(geom), "idx"])

                if sperringer is not None:
                    nye_kanskje_isolerte = nye_kanskje_isolerte + tuple(
                        sperringene.loc[sperringene.intersects(geom), "idx"]
                    )

                nye_kanskje_isolerte = tuple(
                    x for x in nye_kanskje_isolerte if x not in kanskje_isolerte
                )

                kanskje_isolerte = kanskje_isolerte + nye_kanskje_isolerte

        return kanskje_isolerte

    kanskje_isolerte = buffdissexp_gridish_loop(
        ikke_sperringer, sperringer, lengde, "gridish"
    )
    kanskje_isolerte2 = buffdissexp_gridish_loop(
        ikke_sperringer, sperringer, lengde, "gridish2"
    )

    isolerte_network = [idx for idx in kanskje_isolerte if idx in kanskje_isolerte2]

    roads["isolert"] = np.where(roads.idx.isin(isolerte_network), 1, 0)

    return roads.loc[:, ~roads.columns.str.contains("gridish")]


def gridish(gdf, meter, x2=False):
    """
    Enkel rutedeling av dataene. For å fleksibelt kunne loope for små områder sånn at ting blir håndterbart.
    Gir dataene kolonne med avrundede xy-koordinater. Rundes av til valgfritt antall meter.
    x2=True gir en kolonne til med ruter 1/2 hakk nedover og bortover. Hvis grensetilfeller er viktig, kan man loope en gang per rutekategorikolonne.
    """

    # rund ned koordinatene og sett sammen til kolonne
    gdf["gridish"] = [
        f"{round(minx/meter)}_{round(miny/meter)}"
        for minx, miny in zip(gdf.geometry.bounds.minx, gdf.geometry.bounds.miny)
    ]

    if x2:
        gdf["gridish_x"] = gdf.geometry.bounds.minx / meter

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
        gdf["gridish_y"] = gdf.geometry.bounds.miny / meter
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


def tilpass_roads_sykkelfot(roads):
    roads = roads.loc[roads.sykkelforbud != "Ja"]
    roads = roads.rename(columns={"trafikkretning": "oneway"})
    roads["oneway"] = roads.oneway.map({"MED": "FT", "MOT": "TF", "BEGGE": "B"})
    return roads


def beregn_stigningsprosent(roads, endpoints):
    assert all(
        endpoints.has_z
    ), "Vegdataene må ha z-koordinater for å kunne beregne stigning."

    hoyde = [z for z in endpoints.geometry.z]
    roads["hoyde_source"], roads["hoyde_target"] = hoyde[0::2], hoyde[1::2]

    roads["stigningsprosent"] = (
        (roads.hoyde_target - roads.hoyde_source) / roads.length * 100
    )

    roads.loc[
        (roads.stigningsprosent > 100) | (roads.stigningsprosent < -100),
        "stigningsprosent",
    ] = 0

    return roads


def lag_turn_restrictions(roads, turn_restrictions):
    # FID starter på  1
    roads["fid"] = roads["idx"] + 1

    turn_restrictions.columns = [col.lower() for col in turn_restrictions.columns]

    for col in turn_restrictions.columns:
        turn_restrictions[col] = turn_restrictions[col].astype(str)

    # hvis 2021-data
    if "edge1fid" in turn_restrictions.columns:
        roads["fid"] = roads["fid"].astype(str)
        turn_restrictions1 = turn_restrictions.loc[
            turn_restrictions.edge1end == "Y", ["edge1fid", "edge2fid"]
        ].rename(columns={"edge1fid": "edge2fid", "edge2fid": "edge1fid"})
        turn_restrictions2 = turn_restrictions.loc[
            turn_restrictions.edge1end == "N", ["edge1fid", "edge2fid"]
        ]
        turn_restrictions = pd.concat(
            [turn_restrictions1, turn_restrictions2], axis=0, ignore_index=True
        )
        lenker_med_restriction = roads.merge(
            turn_restrictions, left_on="fid", right_on="edge1fid", how="inner"
        )

    # hvis 2022
    else:
        roads["linkid"] = roads["linkid"].astype(str)
        lenker_med_restriction = roads.merge(
            turn_restrictions, left_on="linkid", right_on="fromlinkid", how="inner"
        )
    #      lenker_med_restriction = roads.merge(turn_restrictions, left_on = ["source", "target"], right_on = ["fromfromnode", "fromtonode"], how = "inner")
    #    lenker_med_restriction2 = roads.merge(turn_restrictions, left_on = ["source", "target", "linkid"], right_on = ["fromfromnode", "fromtonode", "fromlinkid"], how = "inner")

    # gjør lenkene med restrictions til første del av nye dobbellenker som skal lages
    lenker_med_restriction = (
        lenker_med_restriction.drop("edge1fid", axis=1, errors="ignore")
        .rename(
            columns={
                "target": "middlenode",
                "minutter": "minutter1",
                "meter": "meter1",
                "geometry": "geom1",
                "fid": "edge1fid",
            }
        )
        .loc[
            :,
            [
                "source",
                "source_wkt",
                "middlenode",
                "minutter1",
                "meter1",
                "geom1",
                "edge1fid",
            ],
        ]
    )
    # klargjør tabell som skal bli andre del av dobbellenkene
    restrictions = roads.rename(
        columns={
            "source": "middlenode",
            "minutter": "minutter2",
            "meter": "meter2",
            "geometry": "geom2",
            "fid": "edge2fid",
        }
    ).loc[
        :,
        [
            "middlenode",
            "target",
            "target_wkt",
            "minutter2",
            "meter2",
            "geom2",
            "edge2fid",
        ],
    ]

    # koble basert på den nye kolonnen 'middlenode', som blir midterste node i dobbellenkene
    fra_nodes_med_restriction = lenker_med_restriction.merge(
        restrictions, on="middlenode", how="inner"
    )

    # vi har nå alle dobbellenker som starter der et svingforbud starter.
    # fjern nå dobbellenkene det faktisk er svingforbud
    if "edge1fid" in turn_restrictions.columns:
        dobbellenker = fra_nodes_med_restriction[
            ~(
                (
                    fra_nodes_med_restriction["edge1fid"].isin(
                        turn_restrictions["edge1fid"]
                    )
                )
                & (
                    fra_nodes_med_restriction["edge2fid"].isin(
                        turn_restrictions["edge2fid"]
                    )
                )
            )
        ]
    else:
        dobbellenker = fra_nodes_med_restriction[
            ~(
                (
                    fra_nodes_med_restriction["source"].isin(
                        turn_restrictions["fromfromnode"]
                    )
                )
                &
                #                                   (fra_nodes_med_restriction["middlenode"].isin(turn_restrictions["fromtonode"])) &
                (
                    fra_nodes_med_restriction["target"].isin(
                        turn_restrictions["totonode"]
                    )
                )
            )
        ]

    # smelt lenkeparene sammen
    dobbellenker["minutter"] = dobbellenker["minutter1"] + dobbellenker["minutter2"]
    dobbellenker["meter"] = dobbellenker["meter1"] + dobbellenker["meter2"]
    dobbellenker["geometry"] = line_merge(union(dobbellenker.geom1, dobbellenker.geom2))
    dobbellenker = gpd.GeoDataFrame(dobbellenker, geometry="geometry", crs=25833)
    dobbellenker["turn_restriction"] = True

    if "edge1fid" in turn_restrictions.columns:
        roads.loc[
            (roads["fid"].isin(turn_restrictions["edge1fid"])), "turn_restriction"
        ] = False
    else:
        roads.loc[
            (roads["linkid"].isin(turn_restrictions["fromlinkid"])), "turn_restriction"
        ] = False

    return gdf_concat([roads, dobbellenker])


def _find_source(roads, source):

    if source in roads.columns:
        return source
    
    possible_cols = []
    for col in roads.columns:
        if "from" in col and "node" in col or "source" in col:
            source = col
            possible_cols.append(col)
        elif "start" in col and "node" in col or "fra" in col and "node" in col:
            source = col
            possible_cols.append(col)
    if len(possible_cols) == 1:
        print(f"Bruker '{source}' som source-kolonne")
    elif len(possible_cols) == 0:
        roads[source] = np.nan
    elif len(possible_cols) > 1:
        raise ValueError(
            f"Flere kolonner kan inneholde source-id-er: {', '.join(possible_cols)}"
        )
    return roads[source]


def find_target(roads, target):
    if not target in roads.columns:
        mulige = []
        for col in roads.columns:
            if "to" in col and "node" in col or "target" in col:
                target = col
                mulige.append(col)
            elif "end" in col and "node" in col or "slutt" in col and "node" in col:
                target = col
                mulige.append(col)
        if len(mulige) == 1:
            print(f"Bruker '{target}' som target-kolonne")
        elif len(mulige) == 0:
            roads[target] = np.nan
        elif len(mulige) > 1:
            raise ValueError(
                f"Flere kolonner kan inneholde target-id-er: {', '.join(mulige)}"
            )
    return roads[target]


def find_linkid(roads, linkid):
    if not linkid in roads.columns:
        n = 0
        for col in roads.columns:
            if "link" in col and "id" in col:
                linkid = col
                n += 1
        if n == 1:
            print(f"Bruker '{linkid}' som linkid-kolonne")
        elif n == 0:
            roads[linkid] = np.nan
        elif n > 1:
            raise ValueError("Flere kolonner kan inneholde linkid-id-er")
    return roads[linkid]


def find_minutter(roads, minutter) -> tuple:
    if isinstance(minutter, str) and minutter in roads.columns:
        return roads[minutter], roads[minutter]
    elif minutter[0] in roads.columns and minutter[1] in roads.columns:
        return roads[minutter[0]], roads[minutter[1]]
    elif "drivetime_fw" in roads.columns and "drivetime_bw" in roads.columns:
        return roads["drivetime_fw"], roads["drivetime_bw"]
    elif "ft_minutes" in roads.columns and "tf_minutes" in roads.columns:
        return roads["ft_minutes"], roads["tf_minutes"]
    else:
        return np.nan, np.nan


def find_vegkategori(roads, vegkategori):
    if vegkategori in roads.columns:
        return roads[vegkategori]
    elif "category" in roads.columns:
        return roads["category"]
    elif "vegtype" in roads.columns:
        return roads["vegtype"]
    elif "roadid" in roads.columns:
        roads["category"] = roads["roadid"].map(
            lambda x: x.replace("{", "").replace("}", "")[0]
        )
        return roads["category"]
    else:
        return np.nan
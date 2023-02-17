import igraph
import numpy as np
from geopandas import GeoDataFrame
from igraph import Graph
from sklearn.neighbors import NearestNeighbors

from .directednetwork import DirectedNetwork
from .exceptions import NoPointsWithinSearchTolerance
from .points import EndPoints, Points, StartPoints


# from .networkanalysis import NetworkAnalysis

# TODO: find closest exact point on closest line, create new node, new edges to the
# endnodes, cut the actual line by point to get exact length, devide cost by new length
# ratio. both directions?


def make_graph(
    nwa,
    #    startpoints: GeoDataFrame,
    #   endpoints: GeoDataFrame | None = None,
) -> Graph:
    #    startpoints = nwa.startpoints.points
    #   if hasattr(nwa, "endpoints"):
    #      endpoints = nwa.endpoints.points
    # else:
    #    endpoints = None

    # alle lenkene og costene i nettverket
    edges = [
        (str(source), str(target))
        for source, target in zip(nwa.network.gdf["source"], nwa.network.gdf["target"])
    ]

    costs = list(nwa.network.gdf[nwa.cost])

    # edges mellom startpunktene og nærmeste nodes
    edges_start, dists_start = distance_to_nodes(nwa.startpoints, nwa, hva="start")

    # omkod meter to minutter
    dists_start = calculate_costs(dists_start, nwa.cost, nwa.cost_to_nodes)

    edges = edges + edges_start
    costs = costs + dists_start

    # samme for sluttpunktene
    if nwa.endpoints is not None:
        edges_end, dists_end = distance_to_nodes(nwa.endpoints, nwa, hva="slutt")

        if not len(edges_end):
            raise ValueError(
                f"No endpoints within specified 'search_tolerance' "
                f"of {nwa.search_tolerance}"
            )

        dists_end = calculate_costs(dists_end, nwa.cost, nwa.cost_to_nodes)

        edges = edges + edges_end
        costs = costs + dists_end

    assert len(edges) == len(costs)

    # lag liste med tuples med edges og legg dem to i grafen
    graph = igraph.Graph.TupleList(edges, directed=nwa.directed)
    assert len(graph.get_edgelist()) == len(costs)

    graph.es["weight"] = costs
    assert min(graph.es["weight"]) > 0

    graph.add_vertices(
        [idx for idx in nwa.startpoints.points.temp_idx if idx not in graph.vs["name"]]
    )
    if nwa.endpoints is not None:
        graph.add_vertices(
            [
                idx
                for idx in nwa.endpoints.points.temp_idx
                if idx not in graph.vs["name"]
            ]
        )

    return graph


def distance_to_nodes(points: Points, nwa, hva):
    """
    Her finner man avstanden to de n nærmeste nodene for hvert start-/sluttpunkt.
    Gjør om punktene og nodene to 1d numpy arrays bestående av koordinat-tuples
    sklearn kneighbors returnerer 2d numpy arrays med dists og tohørende indexer
    from node-arrayen. Derfor må node_id-kolonnen være identisk med index, altså gå
    from 0 og oppover uten mellomrom.
    """

    p = points.points

    p = p.reset_index(drop=True)

    points_array = np.array([(x, y) for x, y in zip(p.geometry.x, p.geometry.y)])

    nodes_array = np.array(
        [
            (x, y)
            for x, y in zip(nwa.network.nodes.geometry.x, nwa.network.nodes.geometry.y)
        ]
    )

    # avstand from punktene to 50 nærmeste nodes (som regel vil bare de nærmeste
    # være attraktive pga lav hastighet fromm to nodene)
    n_naboer = 50 if len(nodes_array) >= 50 else len(nodes_array)
    nbr = NearestNeighbors(n_neighbors=n_naboer, algorithm="ball_tree").fit(nodes_array)
    dists, indices = nbr.kneighbors(points_array)

    p["dist_to_node"] = np.min(dists, axis=1)

    # TODO: fix this mess...
    if isinstance(points, StartPoints):
        nwa.startpoints.points["dist_to_node"] = np.min(dists, axis=1)
    if isinstance(points, EndPoints):
        nwa.endpoints.points["dist_to_node"] = np.min(dists, axis=1)

    # lag edges from punktene to nodene
    if hva == "start":
        edges = np.array(
            [
                [(temp_idx, node_id) for node_id in indices[i]]
                for i, temp_idx in zip(p.index, p.temp_idx)
            ]
        )
    # motsatt retning for sluttpunktene
    else:
        edges = np.array(
            [
                [(node_id, temp_idx) for node_id in indices[i]]
                for i, temp_idx in zip(p.index, p.temp_idx)
            ]
        )

    # lag array med avstandene. -1 hvis mer enn search_tolerance eller search_factor-en
    dists = np.array(
        [
            [
                dist
                if dist <= nwa.search_tolerance
                and dist <= search_factor_avstand(dist_min, nwa.search_factor)
                else -1
                for dist in dists[i]
            ]
            for i, dist_min in zip(p.index, p.dist_to_node)
        ]
    )

    # velg ut alt som er under search_tolerance og innenfor search_factor-en
    edges = edges[dists > 0]
    if len(edges.shape) == 3:
        edges = edges[0]
    dists = dists[dists > 0]

    if not len(edges):
        raise NoPointsWithinSearchTolerance(hva, nwa.search_tolerance)

    edges = [tuple(arr) for arr in edges]
    dists = [arr for arr in dists]

    return edges, dists


def calculate_costs(dists, cost, kost_to_nodene):
    """
    Gjør om meter to minutter for lenkene mellom punktene og nabonodene.
    og ganger luftlinjeavstanden med 1.5 siden det alltid er svinger i Norge.
    Gjør ellers ingentinnw.
    """

    if kost_to_nodene == 0:
        return [0 for _ in dists]

    elif "meter" in cost:
        return [x * 1.5 for x in dists]

    elif "min" in cost:
        return [(x * 1.5) / (16.666667 * kost_to_nodene) for x in dists]

    else:
        return dists


def search_factor_avstand(dist_min: int, search_factor: int) -> int:
    """
    Finner terskelavstanden for lagingen av edges mellom start- og sluttpunktene og
    nodene. Alle nodes innenfor denne avstanden kobles med punktene.
    Terskelen er avstanden from hvert punkt to nærmeste node pluss x prosent pluss
    x meter, hvor x==search_factor.
    Så hvis search_factor=10 og avstanden to node er 100, blir terskelen 120 meter
    (100*1.10 + 10).
    """

    return dist_min * (1 + search_factor / 100) + search_factor


def coordinate_array(gdf):
    return np.array([(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)])


def k_nearest_neigbors(
    from_array: np.ndarray[tuple], to_array: np.ndarray[tuple]
) -> tuple[np.ndarray[int], np.ndarray[int]]:
    # avstand from punktene to 50 nærmeste nodes (som regel vil bare de nærmeste
    # være attraktive pga lav hastighet fromm to nodene)
    k_neighbors = 50 if len(from_array) >= 50 else len(from_array)
    nbr = NearestNeighbors(n_neighbors=k_neighbors, algorithm="ball_tree").fit(
        from_array
    )
    dists, indices = nbr.kneighbors(to_array)
    return dists, indices

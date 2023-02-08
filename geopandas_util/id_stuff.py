import numpy as np


def make_temp_ids(noder, startpoints, endpoints=None):
    """
    Lager id-kolonne som brukes som node-id-er i igraph.Graph().
    Fordi start- og sluttpunktene må ha node-id-er som ikke finnes i nettverket.
    """
    start = max(self.nodes.node_id.astype(int)) + 1
    stop = start + range(len(points))
    return [str(idx) for idx in np.arange(start, stop)]

    startpoints["temp_idx"] = range(len(startpoints))
    startpoints["temp_idx"] = (
        startpoints["temp_idx"] + np.max(noder.node_id.astype(int)) + 1
    )
    startpoints["temp_idx"] = startpoints["temp_idx"].astype(str)

    if endpoints is None:
        return startpoints["temp_idx"]

    endpoints["temp_idx"] = range(len(endpoints))
    endpoints["temp_idx"] = (
        endpoints["temp_idx"] + np.max(startpoints.temp_idx.astype(int)) + 1
    )
    endpoints["temp_idx"] = endpoints["temp_idx"].astype(str)

    return startpoints["temp_idx"], endpoints["temp_idx"]


def map_ids(self, col, points, id_col):
    """From temp to original ids."""

    id_dict = {
        temp_idx: idx
        for idx, temp_idx in zip(points[id_col], points["temp_idx"])
    }

    return col.map(id_dict_slutt)


def map_ids(df, id_col, startpoints, endpoints=None):
    """From temp to original ids."""

    if isinstance(id_col, str):
        id_col = (id_col, id_col)

    id_dict_start = {
        temp_idx: idx
        for idx, temp_idx in zip(startpoints[id_col[0]], startpoints["temp_idx"])
    }

    if endpoints is None:
        df[id_col[0]] = df[id_col[0]].map(id_dict_start)
        return df

    df["from"] = df["from"].map(id_dict_slutt)

    id_dict_slutt = {
        temp_idx: idx
        for idx, temp_idx in zip(endpoints[id_col[1]], endpoints["temp_idx"])
    }
    df["to"] = df["to"].map(id_dict_slutt)

    return df

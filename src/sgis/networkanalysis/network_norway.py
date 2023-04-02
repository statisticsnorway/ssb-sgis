from geopandas import GeoDataFrame
from pandas import DataFrame
from shapely import force_2d

from ..geopandas_tools.construction import gdf_concat
from .directednetwork import DirectedNetwork
from .network import Network


class NetworkDaplaCar(DirectedNetwork):
    def __init__(
        self,
        gdf: GeoDataFrame | str = "daplasti_nyeste",
        turn_restrictions: bool = True,
        fill_holes: bool = False,
        kommuner: str | tuple[str] | None = None,
        **kwargs,
    ):
        if isinstance(gdf, str):
            gdf = read_network_from_dapla(gdf, kommuner)

        if turn_restrictions:
            self.gdf = self.gdf.loc[gdf.turn_restriction == 0]

        if not fill_holes:
            self.gdf = self.gdf.loc[gdf.hole == 0]

        super().__init__(gdf, **kwargs)


class NetworkDaplaBike(DirectedNetwork):
    def __init__(
        self,
        gdf: GeoDataFrame | str = "daplasti_nyeste",
        speed: int | None = 20,
        fill_holes: bool = False,
        kommuner: str | list | tuple | None = None,
        **kwargs,
    ):
        self.speed = speed

        if isinstance(gdf, str):
            gdf = read_network_from_dapla(gdf, kommuner)

        if not fill_holes:
            gdf = gdf.loc[gdf.hole == 0]

        if self.weight == "minutes":
            self.network["minutes"] = self.network.length / speed * 16.6666667

        super().__init__(gdf, **kwargs)


class NetworkDaplaFoot(Network):
    def __init__(
        self,
        gdf: GeoDataFrame | str = "daplasti_nyeste",
        speed: int | None = 5,
        fill_holes: bool = False,
        kommuner: str | list | tuple | None = None,
        **kwargs,
    ):
        self.speed = speed

        if isinstance(gdf, str):
            gdf = read_network_from_dapla(gdf, kommuner)

        if not fill_holes:
            gdf = gdf.loc[gdf.hole == 0]

        if self.weight == "minutes":
            self.network["minutes"] = self.network.length / speed * 16.6666667

        super().__init__(gdf, **kwargs)


def prepare_network_norway(
    roads: GeoDataFrame,
    out_path: str,
    source_col="fromnode",
    target_col="tonode",
    direction_col="oneway",
    minute_cols=("drivetime_fw", "drivetime_bw"),
    muni_col="municipality",
    turn_restrictions: DataFrame | None = None,
):
    roads.columns = [col.lower() for col in roads.columns]

    roads = (
        roads.to_crs(25833)
        .loc[:, [source_col, target_col, direction_col, "geometry"] + list(minute_cols)]
        .rename(
            columns={
                muni_col: "KOMMUNENR",
            }
        )
    )

    roads["geometry"] = force_2d(roads.geometry)

    nw = DirectedNetwork(roads)

    nw = nw.make_directed_network_norway()

    if turn_restrictions:
        nw.network = find_turn_restrictions(nw.network, turn_restrictions)

    nw = nw.close_network_holes(max_distance=1.1, hole_col="hole")

    nw = nw.get_largest_component()

    write_geopandas(nw.gdf, out_path)

    return nw


def read_network_from_dapla(roads, kommuner):
    if kommuner:
        if isinstance(kommuner, str):
            kommuner = [kommuner]
        filters = [("KOMMUNENR", "in", kommuner)]
    else:
        filters = None

    return read_geopandas(roads, filters=filters)


def find_turn_restrictions(roads, turn_restrictions):
    """Not sure which version is correct because of errors in the source data."""

    # FID starter på  1
    roads["fid"] = roads["idx"] + 1

    turn_restrictions.columns = [col.lower() for col in turn_restrictions.columns]

    for col in turn_restrictions.columns:
        turn_restrictions[col] = turn_restrictions[col].astype(str)

    roads["linkid"] = roads["linkid"].astype(str)
    roads_with_restriction = roads.merge(
        turn_restrictions, left_on="linkid", right_on="fromlinkid", how="inner"
    )
    #    roads_with_restriction = roads.merge(turn_restrictions, left_on = ["source", "target"], right_on = ["fromfromnode", "fromtonode"], how = "inner")
    #    roads_with_restriction2 = roads.merge(turn_restrictions, left_on = "source", "target", "linkid"], right_on = ["fromfromnode", "fromtonode", "fromlinkid"], how = "inner")

    # gjør lenkene med restrictions til første del av nye double_roads som skal lages
    roads_with_restriction = roads_with_restriction.rename(
        columns={
            "target": "middlenode",
            "minutes": "minutes1",
            "meter": "meter1",
            "geometry": "geom1",
        }
    ).loc[:, ["source", "source_wkt", "middlenode", "minutes1", "meter1", "geom1"]]
    # klargjør tabell som skal bli andre del av dobbellenkene
    restrictions = roads.rename(
        columns={
            "source": "middlenode",
            "minutes": "minutes2",
            "meter": "meter2",
            "geometry": "geom2",
        }
    ).loc[:, ["middlenode", "target", "target_wkt", "minutes2", "meter2", "geom2"]]

    # koble basert på den nye kolonnen 'middlenode', som blir midterste node i
    # dobbellenkene
    fra_noder_med_restriction = roads_with_restriction.merge(
        restrictions, on="middlenode", how="inner"
    )

    double_roads = fra_noder_med_restriction[
        ~(
            (
                fra_noder_med_restriction["source"].isin(
                    turn_restrictions["fromfromnode"]
                )
            )
            &
            # (fra_noder_med_restriction["middlenode"].isin(turn_restrictions["fromtonode"])) &
            (fra_noder_med_restriction["target"].isin(turn_restrictions["totonode"]))
        )
    ]

    # smelt lenkeparene sammen
    double_roads["minutes"] = double_roads["minutes1"] + double_roads["minutes2"]
    double_roads["meter"] = double_roads["meter1"] + double_roads["meter2"]
    double_roads["geometry"] = line_merge(union(double_roads.geom1, double_roads.geom2))
    double_roads = gpd.GeoDataFrame(double_roads, geometry="geometry", crs=roads.crs)

    double_roads["turn_restriction"] = 1
    roads.loc[
        (roads["linkid"].isin(turn_restrictions["fromlinkid"])), "turn_restriction"
    ] = 0

    return gdf_concat([roads, double_roads])

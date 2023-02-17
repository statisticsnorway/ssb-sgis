# %%
import warnings

import geopandas as gpd
import pyogrio

import gis_utils as gs


"""
from tests.test_od_cost_matrix import test_od_cost_matrix
from tests.test_shortest_path import test_shortest_path
from tests.test_service_area import test_service_area
"""


def not_test_osm(nw):
    test_shortest_path(nw)
    test_service_area(nw)
    test_od_cost_matrix(nw)


def main():
    osm_path = r"C:\Users\ort\Downloads\norway-latest-free.shp\gis_osm_roads_free_1.shp"

    crs = pyogrio.read_info(osm_path)["crs"]

    oslo = (
        gpd.read_file(
            r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\Basisdata_0000_Norge_25833_Kommuner_FGDB.gdb",
            layer="kommune",
        )
        .query("kommunenummer == '0301'")
        .to_crs(crs)
    )

    r = gpd.read_file(osm_path, engine="pyogrio", bbox=oslo)
    r = r.to_crs(25833)

    nw = gs.DirectedNetwork(r).make_directed_network_osm().remove_isolated()
    nw = gs.Network(r).remove_isolated()

    nwa = gs.NetworkAnalysis(nw, cost="meters")

    not_test_osm(nwa)


if __name__ == "__main__":
    main()

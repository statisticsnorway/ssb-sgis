# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

import os


# %%
import geopandas as gpd


while "networkz" not in os.listdir():
    os.chdir("../")

import networkz as nz


nz.__version__

# %%
punkter = nz.les_geopandas(
    f"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/tilfeldige_adresser_1000.parquet"
)

G = nz.Graf()
G

# %%
display(punkter.head(3))
punkter = punkter.to_crs(25832)
od = G.od_cost_matrix(
    startpunkter=punkter.sample(5), sluttpunkter=punkter.sample(5), id_kolonne="idx"
)
assert punkter.crs == 25832
od = G.od_cost_matrix(
    startpunkter=punkter.sample(5), sluttpunkter=punkter.sample(5), id_kolonne=None
)
od = G.od_cost_matrix(
    startpunkter=punkter.sample(5), sluttpunkter=punkter.sample(5), id_kolonne=False
)
od = G.od_cost_matrix(
    startpunkter=punkter.sample(5), sluttpunkter=punkter.sample(5), id_kolonne=0
)
display(od.head(1))

assert "idx" in punkter.columns

od = G.od_cost_matrix(
    startpunkter=punkter.sample(5),
    sluttpunkter=punkter.sample(5),
    id_kolonne=("idx", "idx"),
)
od = G.od_cost_matrix(
    startpunkter=punkter.sample(5),
    sluttpunkter=punkter.sample(5),
    id_kolonne=("idx", "idx"),
)
od = G.od_cost_matrix(
    startpunkter=punkter.sample(5),
    sluttpunkter=punkter.sample(5),
    id_kolonne=("idx", "idx"),
)
display(od.head(1))

od = G.od_cost_matrix(
    startpunkter=punkter.sample(5),
    sluttpunkter=punkter.sample(5),
    id_kolonne=["idx", "idx"],
)
od = G.od_cost_matrix(
    startpunkter=punkter.sample(5),
    sluttpunkter=punkter.sample(5),
    id_kolonne=["idx", "idx"],
)
od = G.od_cost_matrix(
    startpunkter=punkter.sample(5),
    sluttpunkter=punkter.sample(5),
    id_kolonne=["idx", "idx"],
)
display(od.head(1))
display(punkter.head(3))

assert "idx" in punkter.columns
assert punkter.crs == 25832
punkter = punkter.to_crs(25833)

# %%
G = nz.Graf()
nett = G.nettverk
nett.loc[nett.isolert != 0, "isolert"] = 1
nett.sjoin(nz.til_gdf(punkter.buffer(1000).iloc[0], crs=25833)).plot(
    "isolert", cmap="bwr"
)

# %%
korteste_ruter = G.get_route(
    startpunkter=punkter.sample(10), sluttpunkter=punkter.sample(10), id_kolonne="idx"
)
korteste_ruter.plot()
korteste_ruter

# %%
relevante_veglenker = G.get_route(
    startpunkter=punkter.sample(10), sluttpunkter=punkter.sample(10), tell_opp=True
)

import matplotlib.pyplot as plt


fig, ax = plt.subplots(1, figsize=(15, 15))
ax.set_axis_off()
ax.set_title("Antall ganger hver veglenke ble brukt", fontsize=18)
relevante_veglenker["geometry"] = relevante_veglenker.buffer(20)
relevante_veglenker.plot(
    "antall", scheme="NaturalBreaks", cmap="RdPu", k=7, legend=True, alpha=0.8, ax=ax
)

# %%
korteste_ruter = G.get_route(
    startpunkter=punkter.sample(10),
    sluttpunkter=punkter.sample(10),
    id_kolonne="idx",
    radvis=True,
)
korteste_ruter

# %%
od = G.od_cost_matrix(
    startpunkter=punkter,
    sluttpunkter=punkter,
    id_kolonne="idx",
)
od

# %%
od = G.od_cost_matrix(
    startpunkter=punkter.sample(1), sluttpunkter=punkter, id_kolonne="idx", linjer=True
)

od.plot("minutter", scheme="Quantiles")

# %%
od = G.od_cost_matrix(
    startpunkter=punkter.sample(10),
    sluttpunkter=punkter.sample(10),
    id_kolonne="idx",
    radvis=True,
)
od

# %%
od = G.od_cost_matrix(
    startpunkter=punkter.sample(10),
    sluttpunkter=punkter.sample(10),
    id_kolonne="idx",
    destination_count=1,
)
od

# %%
od = G.od_cost_matrix(
    startpunkter=punkter, sluttpunkter=punkter, id_kolonne="idx", cutoff=5
)
od

# %%
G.kostnad = ["minutter", "meter"]

od = G.od_cost_matrix(startpunkter=punkter, sluttpunkter=punkter, id_kolonne="idx")
G.kostnad = "minutter"
od

# %%
service_areas = G.service_area(
    startpunkter=punkter.sample(5),
    breaks=5,  # antall minutter/meter
    id_kolonne="idx",
)
service_areas.plot()
service_areas

# %%
service_areas = G.service_area(
    startpunkter=punkter.sample(1),
    breaks=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],  # antall minutter/meter/annet
    id_kolonne="idx",
)
service_areas.plot(G.kostnad)

# %%
import pandas as pd


resultater = pd.DataFrame()
for kjoretoy in ["fot", "sykkel", "bil"]:
    G = nz.Graf(kjoretoy=kjoretoy, kostnad=["minutter", "meter"], kommuner="0301")

    od = G.od_cost_matrix(
        startpunkter=punkter.sample(10),
        sluttpunkter=punkter.sample(10),
        id_kolonne="idx",
    )

    od["kjoretoy"] = kjoretoy

    resultater = pd.concat([resultater, od], ignore_index=True)

resultater["km"] = resultater.meter / 1000
resultater["km_t"] = (resultater.meter / 1000) / (resultater.minutter / 60)

gruppert = resultater.groupby("kjoretoy").agg(
    minutter_mean=("minutter", "mean"),
    km_mean=("km", "mean"),
    km_t_mean=("km_t", "mean"),
)
gruppert

# %%
from shapely.wkt import loads


storo = gpd.GeoDataFrame(
    {"geometry": gpd.GeoSeries(loads("POINT (10.7777979 59.9451632)"))}, crs=4326
).to_crs(25833)
storo["idx"] = "storo"
grefsenkollen = gpd.GeoDataFrame(
    {"geometry": gpd.GeoSeries(loads("POINT (10.8038165 59.9590036)"))}, crs=4326
).to_crs(25833)
grefsenkollen["idx"] = "grefsenkollen"

G = nz.Graf(kjoretoy="sykkel")

oppover = G.get_route(startpunkter=storo, sluttpunkter=grefsenkollen, id_kolonne="idx")
nedover = G.get_route(startpunkter=grefsenkollen, sluttpunkter=storo, id_kolonne="idx")
nz.gdf_concat([oppover, nedover])

# %%
G = nz.Graf(kjoretoy="sykkel", kostnad="minutter")
med_sykkel = G.get_route(
    startpunkter=storo, sluttpunkter=grefsenkollen, id_kolonne="idx"
)
med_sykkel["hva"] = "sykkel"

G = nz.Graf(kjoretoy="fot", kostnad="minutter")
til_fots = G.get_route(startpunkter=storo, sluttpunkter=grefsenkollen, id_kolonne="idx")
til_fots["hva"] = "fot"

begge = nz.gdf_concat([med_sykkel, til_fots])
begge.plot("hva", cmap="bwr")
begge

# %%
from shapely.wkt import loads


akersveien = gpd.GeoDataFrame(
    {"geometry": gpd.GeoSeries(loads("POINT (10.7476913 59.9222196)"))}, crs=4326
).to_crs(25833)
akersveien["geometry"] = akersveien.buffer(500)
punkter_rundt_akersveien = punkter.sjoin(akersveien)

resultater = pd.DataFrame()
for kjoretoy in ["fot", "sykkel"]:
    G = nz.Graf(kjoretoy=kjoretoy, kostnad="minutter", kommuner="0301")

    korteste_ruter = G.get_route(
        startpunkter=punkter_rundt_akersveien,
        sluttpunkter=punkter_rundt_akersveien,
        id_kolonne="idx",
    )

    korteste_ruter["kjoretoy"] = kjoretoy

    resultater = gpd.GeoDataFrame(
        pd.concat([resultater, korteste_ruter], axis=0, ignore_index=True),
        geometry="geometry",
        crs=25833,
    )

resultater.plot("kjoretoy", cmap="bwr")

# %% [markdown]
# ### lag_nettverk

# %%
veger = gpd.read_parquet(
    r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\vegdata\veger_oslo_og_naboer_2022.parquet"
)

# %%
nettverk = nz.lag_nettverk(veger)

G = nz.Graf(nettverk=nettverk)
G.nettverk.head()


# %%
def test_nettverk(G, punkter, kostnad):
    nett = G.nettverk
    nett.loc[nett.isolert != 0, "isolert"] = 1
    nett.sjoin(nz.til_gdf(punkter.buffer(1000).iloc[0], crs=25833)).plot(
        "isolert", cmap="bwr"
    )

    korteste_ruter = G.get_route(
        startpunkter=punkter.sample(35),
        sluttpunkter=punkter.sample(35),
        id_kolonne="idx",
    )
    korteste_ruter.plot()

    od = G.od_cost_matrix(
        startpunkter=punkter.sample(1),
        sluttpunkter=punkter,
        id_kolonne="idx",
        linjer=True,
    )
    od.plot(G.kostnad, scheme="quantiles")

    if kostnad == "minutter":
        breaks = (7.5, 6, 4, 2.5, 1)
    else:
        breaks = (3000, 2000, 1000, 500, 200)

    service_area = G.service_area(
        startpunkter=punkter.sample(5), breaks=breaks, id_kolonne="idx"
    )
    service_area.plot(G.kostnad)

    display(od)


# %%
test_nettverk(G, punkter, "minutter")

# %%
nettverk = nz.lag_nettverk(veger[["geometry"]])
G = nz.Graf(nettverk=nettverk, kostnad="meter", directed=False)
G.nettverk.head()

# %%
test_nettverk(G, punkter, "meter")

# %%
import pandas as pd


utvalg = punkter.sample(150)
resultater = pd.DataFrame()
for kjoretoy in ["bil", "sykkel", "fot"]:
    G = nz.Graf(kjoretoy=kjoretoy, kommuner="0301")

    mest_brukte_gater = G.get_route(
        startpunkter=utvalg, sluttpunkter=utvalg, tell_opp=True
    )

    mest_brukte_gater["geometry"] = mest_brukte_gater.buffer(7)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.set_axis_off()
    ax.set_title(f"Antall ganger brukt. Kjøretøy: {kjoretoy}", fontsize=16)
    mest_brukte_gater.plot(
        "antall", scheme="NaturalBreaks", cmap="RdPu", k=7, legend=True, alpha=1, ax=ax
    )

# %%

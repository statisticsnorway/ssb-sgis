"""
Test om funksjonene gir forventede resultater. 
Bruker en fast gdf som aldri må endres. 
Funksjonen test_alt kjøres når man importerer geopandasgreier. Gir advarsel hvis en av testene feilet.
"""

import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.wkt import loads

import sys
while "geopandasgreier" not in os.listdir():
    os.chdir("../")
sys.path.append(os.getcwd())

from geopandasgreier.buffer_dissolve_explode import buff, diss, buffdiss, dissexp, buffdissexp
from geopandasgreier.generelt import gdf_concat, til_gdf, fiks_geometrier, min_sjoin, overlay_update
from geopandasgreier.spesifikt import finn_naboer, gridish, snap_til, antall_innen_avstand, til_multipunkt


def lag_gdf():
    """ 
    Lager en gdf med 9 rader, bestående av sju punkter, en linje og ett polygon. 
    tester samtidig funksjonen til_gdf
    OBS: denne må aldri endres.
    """
    
    xs = [10.7497196, 10.7484624, 10.7480624, 10.7384624, 10.7374624, 10.7324624, 10.7284624]
    ys = [59.9281407, 59.9275268, 59.9272268, 59.9175268, 59.9165268, 59.9365268, 59.9075268]
    punkter = [f'POINT ({x} {y})' for x, y in zip(xs, ys)]

    linje = ["LINESTRING (10.7284623 59.9075267, 10.7184623 59.9175267, 10.7114623 59.9135267, 10.7143623 59.8975267, 10.7384623 59.900000, 10.720000 59.9075200)"]

    polygon = ["POLYGON ((10.74 59.92, 10.735 59.915, 10.73 59.91, 10.725 59.905, 10.72 59.9, 10.72 59.91, 10.72 59.91, 10.74 59.92))"]

    geometrier = [loads(x) for x in punkter + linje + polygon]

    gdf = gpd.GeoDataFrame({'geometry': gpd.GeoSeries(geometrier)}, geometry="geometry", crs=4326).to_crs(25833)
    
    gdf2 = til_gdf(geometrier, crs=4326).to_crs(25833)
    
    assert gdf.equals(gdf2), "til_gdf gir ikke samme gdf som å lage gdf manuelt"
    
    gdf["numkol"] = [1,2,3,4,5,6,7,8,9]
    gdf["txtkol"] = [*'aaaabbbcc']

    assert len(gdf)==9, "feil lengde. Er testdataene endret?"
    
    x = round(gdf.dissolve().centroid.x.iloc[0], 5)
    y = round(gdf.dissolve().centroid.y.iloc[0], 5)
    assert  f'POINT ({x} {y})' == 'POINT (261106.48628 6649101.81219)', "feil midtpunkt. Er testdataene endret?"    

    return gdf


def test_buffdissexp(gdf):
    for avstand in [1, 10, 100, 1000, 10000]:
        kopi = gdf.copy()
        kopi = kopi[["geometry"]]
        kopi = buff(kopi, avstand)
        kopi = diss(kopi)
        kopi = kopi.explode(ignore_index=True)

        areal1 = kopi.area.sum()
        lengde1 = kopi.length.sum()

        kopi = buff(gdf, avstand, copy=True)
        kopi = diss(kopi[["geometry"]])
        kopi = kopi.explode(ignore_index=True)

        assert areal1==kopi.area.sum() and lengde1==kopi.length.sum(), "ulik lengde/areal"

        kopi = buffdissexp(gdf, avstand, copy=True)

        assert areal1==kopi.area.sum() and lengde1==kopi.length.sum(), "ulik lengde/areal"


def test_geos(gdf):
    kopi = buffdissexp(gdf, 25, copy=True)
    assert len(kopi)==4,  "feil antall rader. Noe galt/nytt med GEOS' GIS-algoritmer?"
    assert round(kopi.area.sum(), 5) == 1035381.10389, "feil areal. Noe galt/nytt med GEOS' GIS-algoritmer?"
    assert round(kopi.length.sum(), 5) == 16689.46148, "feil lengde. Noe galt/nytt med GEOS' GIS-algoritmer?"


def test_aggfuncs(gdf):
    kopi = dissexp(gdf, by="txtkol", aggfunc="sum")
    assert len(kopi)==11, "dissexp by txtkol skal gi 11 rader, tre stykk linestrings..."

    kopi = buffdiss(gdf, 100, by="txtkol", aggfunc="sum", copy=True)
    assert kopi.numkol.sum() == gdf.numkol.sum() == sum([1,2,3,4,5,6,7,8,9])

    kopi = buffdissexp(gdf, 100, by="txtkol", aggfunc=["sum", "mean"], copy=True)
    assert "numkol_sum" in kopi.columns and "numkol_sum" in kopi.columns, "kolonnene følger ikke mønstret 'kolonnenavn_aggfunc'"
    assert len(kopi)==6, "feil lengde"

    kopi = buffdissexp(gdf, 1000, by="txtkol", aggfunc=["sum", "mean"], copy=True)
    assert len(kopi)==4, "feil lengde"

    kopi = buffdissexp(gdf, 100, by="numkol", copy=True)
    assert len(kopi)==9, "feil lengde"

    kopi = buffdissexp(gdf, 100, by=["numkol", "txtkol"],copy=True)
    assert "numkol" in kopi.columns and "txtkol" in kopi.columns, "kolonnene mangler. Er de index?"
    assert len(kopi)==9, "feil lengde"
    

def test_fix(gdf):
    missing = gpd.GeoDataFrame({'geometry': [None, np.nan]}, geometry="geometry", crs=25833)
    empty = gpd.GeoDataFrame({'geometry':  gpd.GeoSeries(loads("POINT (0 0)")).buffer(0)}, geometry="geometry", crs=25833)
    gdf = gdf_concat([gdf, manglende, tomme])
    assert len(gdf)==12
    gdf2 = fix_geoms(gdf)
    ser = fix_geoms(gdf.geometry)
    assert len(gdf2)==9
    assert len(ser)==9

    
def sjoin_overlay(gdf):
    gdf1 = buff(gdf, 25, copy=True)
    gdf2 = buff(gdf, 100, copy=True)
    gdf2["nykoll"] = 1
    gdf = min_sjoin(gdf1, gdf2)
    assert list(gdf.columns)==['geometry', 'numkol', 'txtkol', 'nykoll']
    assert len(gdf)==25

    gdf = overlay_update(gdf2, gdf1)
    assert list(gdf.columns)==['geometry', 'numkol', 'txtkol', 'nykoll']
    assert len(gdf)==18
    
    
def test_copy(gdf):
    """
    Sjekk at copy-parametret i buff funker. Og sjekk pandas' copy-regler samtidig.
    """
    
    kopi = gdf[gdf.area==0]
    assert gdf.area.sum() != 0
    
    kopi = gdf.loc[gdf.area==0]    
    assert gdf.area.sum() != 0
    assert kopi.area.sum() == 0
    
    bufret = buff(kopi, 10, copy=True)
    assert kopi.area.sum() == 0

    bufret = buff(kopi, 10, copy=False)
    assert kopi.area.sum() != 0
    
    
def test_naboer(gdf):
    naboer = finn_naboer(gdf.iloc[[0]], mulige_naboer=gdf, id_kolonne="numkol", innen_meter = 100)
    naboer.sort()
    assert naboer==[1, 2], "feil i finn_naboer"
    naboer = finn_naboer(gdf.iloc[[8]], mulige_naboer=gdf, id_kolonne="numkol", innen_meter = 100)
    naboer.sort()
    assert naboer==[4, 5, 7, 8, 9], "feil i finn_naboer"
    

def test_gridish(gdf):
    kopi = gdf.copy()
    kopi = gridish(kopi, 2000)
    assert len(kopi.gridish.unique()) == 4
    
    kopi = gridish(kopi, 5000)
    assert len(kopi.gridish.unique()) == 2

    kopi = gridish(kopi, 1000, x2=True)
    assert len(kopi.gridish.unique()) == 7
    assert len(kopi.gridish2.unique()) == 7


def test_snap(gdf):
    punkter = gdf[gdf.length==0]
    annet = gdf[gdf.length!=0]
    snappet = snap_til(punkter, annet, maks_distanse=50000, copy=True)
    assert all(snappet.intersects(annet.buffer(1).unary_union))
    snappet = snap_til(punkter, annet, maks_distanse=50)
    assert sum(snappet.intersects(annet.buffer(1).unary_union)) == 3


def test_antall_innen_avstand(gdf):
    innen = antall_innen_avstand(gdf, gdf)
    assert innen.antall.sum() == 11
    innen = antall_innen_avstand(gdf, gdf, 50)
    assert innen.antall.sum() == 21
    innen = antall_innen_avstand(gdf, gdf, 10000)
    assert innen.antall.sum() == 81
    
    
def test_til_multipunkt(gdf):
    mp = til_multipunkt(gdf)
    assert mp.length.sum() == 0
    mp = til_multipunkt(gdf.geometry)
    assert mp.length.sum() == 0
    mp = til_multipunkt(gdf.unary_union)
    assert mp.length == 0
    

def test_alt():
    
    gdf = lag_gdf()
    
    test_fix(gdf)
    
    test_buffdissexp(gdf)

    test_geos(gdf)

    test_aggfuncs(gdf)
    
    sjoin_overlay(gdf)
    
    test_naboer(gdf)
    
    test_gridish(gdf)
    
    test_copy(gdf)
    
    test_snap(gdf)
    
    test_antall_innen_avstand(gdf)
    
    test_til_multipunkt(gdf)
    
    
def main():
    info = """
    Testen ble lagd 08.01.2023 med følgende versjoner.
    Fra C++: GEOS 3.11.1, PROJ 9.1.0, GDAL 3.6.1. 
    Fra Python: geopandas 0.12.2, shapely 2.0.0, pyproj 3.4.1, pandas 1.5.2 og numpy 1.24.
    """
    print(info)
    
    print("Versjoner nå:")
    from shapely.geos import geos_version
    geos_versjon = ".".join([str(x) for x in geos_version])
    print(f'{gpd.__version__ = }')
    print(f'{geos_versjon    = }')
    print(f'{pd.__version__  = }')
    print(f'{np.__version__  = }')

    print(test_fix())

    test_alt()

    print("vellykket")
    
    
if __name__=="__main__":
    main()

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
sys.path.append("C:/Users/ort/git/ssb-gis-utils")

import gis_utils as giu
import cProfile


def make_gdf():
    """ 
    Lager en gdf med 9 rader, bestående av sju punkter, en linje og ett polygon. 
    tester samtidig funksjonen giu.to_gdf
    OBS: denne må aldri endres.
    """
    
    xs = [10.7497196, 10.7484624, 10.7480624, 10.7384624, 10.7374624, 10.7324624, 10.7284624]
    ys = [59.9281407, 59.9275268, 59.9272268, 59.9175268, 59.9165268, 59.9365268, 59.9075268]
    punkter = [f'POINT ({x} {y})' for x, y in zip(xs, ys)]

    linje = ["LINESTRING (10.7284623 59.9075267, 10.7184623 59.9175267, 10.7114623 59.9135267, 10.7143623 59.8975267, 10.7384623 59.900000, 10.720000 59.9075200)"]

    polygon = ["POLYGON ((10.74 59.92, 10.735 59.915, 10.73 59.91, 10.725 59.905, 10.72 59.9, 10.72 59.91, 10.72 59.91, 10.74 59.92))"]

    geometrier = [loads(x) for x in punkter + linje + polygon]

    gdf = gpd.GeoDataFrame({'geometry': gpd.GeoSeries(geometrier)}, geometry="geometry", crs=4326).to_crs(25833)
    
    gdf2 = giu.to_gdf(geometrier, crs=4326).to_crs(25833)
    
    assert gdf.equals(gdf2), "giu.to_gdf gir ikke samme gdf som å lage gdf manuelt"
    
    gdf["numkol"] = [1,2,3,4,5,6,7,8,9]
    gdf["txtkol"] = [*'aaaabbbcc']

    assert len(gdf)==9, "feil lengde. Er testdataene endret?"
    
    x = round(gdf.dissolve().centroid.x.iloc[0], 5)
    y = round(gdf.dissolve().centroid.y.iloc[0], 5)
    assert  f'POINT ({x} {y})' == 'POINT (261106.48628 6649101.81219)', "feil midtpunkt. Er testdataene endret?"    

    return gdf


def test_buffdissexp(gdf):
    from shapely import area

    for distance in [1, 10, 100, 1000, 10000]:
        copy = gdf.copy()[["geometry"]]
        copy = giu.buff(copy, distance)
        copy = giu.diss(copy)
        copy = copy.explode(ignore_index=True)

        areal1 = copy.area.sum()
        lengde1 = copy.length.sum()

        copy = giu.buffdissexp(gdf, distance, copy=True)

        assert areal1==copy.area.sum() and lengde1==copy.length.sum(), "ulik lengde/areal"

    giu.buffdissexp(gdf, 100)
    giu.buffdissexp(gdf.geometry, 100)
    giu.buffdissexp(gdf.unary_union, 100)


def test_geos(gdf):
    copy = giu.buffdissexp(gdf, 25, copy=True)
    assert len(copy)==4,  "feil antall rader. Noe galt/nytt med GEOS' GIS-algoritmer?"
    assert round(copy.area.sum(), 5) == 1035381.10389, "feil areal. Noe galt/nytt med GEOS' GIS-algoritmer?"
    assert round(copy.length.sum(), 5) == 16689.46148, "feil lengde. Noe galt/nytt med GEOS' GIS-algoritmer?"


def test_aggfuncs(gdf):
    copy = giu.dissexp(gdf, by="txtkol", aggfunc="sum")
    assert len(copy)==11, "dissexp by txtkol skal gi 11 rader, tre stykk linestrings..."

    copy = giu.buffdiss(gdf, 100, by="txtkol", aggfunc="sum", copy=True)
    assert copy.numkol.sum() == gdf.numkol.sum() == sum([1,2,3,4,5,6,7,8,9])

    copy = giu.buffdissexp(gdf, 100, by="txtkol", aggfunc=["sum", "mean"], copy=True)
    assert "numkol_sum" in copy.columns and "numkol_sum" in copy.columns, "kolonnene følger ikke mønstret 'kolonnenavn_aggfunc'"
    assert len(copy)==6, "feil lengde"

    copy = giu.buffdissexp(gdf, 1000, by="txtkol", aggfunc=["sum", "mean"], copy=True)
    assert len(copy)==4, "feil lengde"

    copy = giu.buffdissexp(gdf, 100, by="numkol", copy=True)
    assert len(copy)==9, "feil lengde"

    copy = giu.buffdissexp(gdf, 100, by=["numkol", "txtkol"],copy=True)
    assert "numkol" in copy.columns and "txtkol" in copy.columns, "kolonnene mangler. Er de index?"
    assert len(copy)==9, "feil lengde"
    

def test_clean(gdf):
    missing = gpd.GeoDataFrame({'geometry': [None, np.nan]}, geometry="geometry", crs=25833)
    empty = gpd.GeoDataFrame({'geometry':  gpd.GeoSeries(loads("POINT (0 0)")).buffer(0)}, geometry="geometry", crs=25833)
    gdf = giu.gdf_concat([gdf, missing, empty])
    assert len(gdf)==12
    gdf2 = giu.clean_geoms(gdf, single_geom_type=False)
    ser = giu.clean_geoms(gdf.geometry, single_geom_type=False)
    assert len(gdf2)==9
    assert len(ser)==9

    
def sjoin_overlay(gdf):
    gdf1 = giu.buff(gdf, 25, copy=True)
    gdf2 = giu.buff(gdf, 100, copy=True)
    gdf2["nykoll"] = 1
    gdf = giu.sjoin(gdf1, gdf2)
    assert all(col in ['geometry', 'numkol', 'txtkol', 'nykoll'] for col in gdf.columns)
    assert not any(col not in list(gdf.columns) for col in ['geometry', 'numkol', 'txtkol', 'nykoll'])
    assert len(gdf)==25
    gdf = giu.overlay(gdf1, gdf2)
    assert all(col in ['geometry', 'numkol', 'txtkol', 'nykoll'] for col in gdf.columns)
    assert not any(col not in list(gdf.columns) for col in ['geometry', 'numkol', 'txtkol', 'nykoll'])
    assert len(gdf)==25

    gdf = giu.overlay_update(gdf2, gdf1)
    assert list(gdf.columns)==['geometry', 'numkol', 'txtkol', 'nykoll']
    assert len(gdf)==18
    

def test_copy(gdf):
    """
    Sjekk at copy-parametret i buff funker. Og sjekk pandas' copy-regler samtidig.
    """
    
    copy = gdf[gdf.area==0]
    assert gdf.area.sum() != 0
    
    copy = gdf.loc[gdf.area==0]    
    assert gdf.area.sum() != 0
    assert copy.area.sum() == 0
    
    bufret = giu.buff(copy, 10, copy=True)
    assert copy.area.sum() == 0

    bufret = giu.buff(copy, 10, copy=False)
    assert copy.area.sum() != 0
    
    
def test_neighbors(gdf):
    naboer = giu.find_neighbors(gdf.iloc[[0]], possible_neighbors=gdf, id_col="numkol", within_distance = 100)
    naboer.sort()
    assert naboer==[1, 2], "feil i find_neighbors"
    naboer = giu.find_neighbors(gdf.iloc[[8]], possible_neighbors=gdf, id_col="numkol", within_distance = 100)
    naboer.sort()
    assert naboer==[4, 5, 7, 8, 9], "feil i find_neighbors"
    

def test_gridish(gdf):
    copy = gdf.copy()
    copy = giu.gridish(copy, 2000)
    assert len(copy.gridish.unique()) == 4
    
    copy = giu.gridish(copy, 5000)
    assert len(copy.gridish.unique()) == 2

    copy = giu.gridish(copy, 1000, x2=True)
    assert len(copy.gridish.unique()) == 7
    assert len(copy.gridish2.unique()) == 7


def test_snap(gdf):
    punkter = gdf[gdf.length==0]
    annet = gdf[gdf.length!=0]
    snappet = giu.snap_to(punkter, annet, maks_distanse=50000, copy=True)
    assert all(snappet.intersects(annet.buffer(1).unary_union))
    snappet = giu.snap_to(punkter, annet, maks_distanse=50)
    assert sum(snappet.intersects(annet.buffer(1).unary_union)) == 3


def test_count_within_distance(gdf):
    innen = giu.count_within_distance(gdf, gdf)
    assert innen.n.sum() == 11
    innen = giu.count_within_distance(gdf, gdf, 50)
    assert innen.n.sum() == 21
    innen = giu.count_within_distance(gdf, gdf, 10000)
    assert innen.n.sum() == 81
    
    
def test_to_multipoint(gdf):
    mp = giu.to_multipoint(gdf)
    assert mp.length.sum() == 0
    mp = giu.to_multipoint(gdf.geometry)
    assert mp.length.sum() == 0
    mp = giu.to_multipoint(gdf.unary_union)
    assert mp.length == 0
    

def test_alt():
    
    gdf = make_gdf()
    
    test_clean(gdf)
    
    test_buffdissexp(gdf)

    test_geos(gdf)

    test_aggfuncs(gdf)
    
    sjoin_overlay(gdf)
    
    test_neighbors(gdf)
    
    test_gridish(gdf)
    
    test_copy(gdf)
    
    test_snap(gdf)
    
    test_count_within_distance(gdf)
    
    test_to_multipoint(gdf)
    
    
def main():
    info = """
    The test was created 08.01.2023 with the following package versions.
    From C++: GEOS 3.11.1, PROJ 9.1.0, GDAL 3.6.1. 
    From Python: geopandas 0.12.2, shapely 2.0.0, pyproj 3.4.1, pandas 1.5.2, numpy 1.24.
    """
    print(info)
    
    print("Versjoner nå:")
    from shapely.geos import geos_version
    geos_versjon = ".".join([str(x) for x in geos_version])
    print(f'{gpd.__version__ = }')
    print(f'{geos_versjon    = }')
    print(f'{pd.__version__  = }')
    print(f'{np.__version__  = }')

    test_alt()

    print("Success")
    
    
if __name__=="__main__":
    main()

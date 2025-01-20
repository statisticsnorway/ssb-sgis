import os
import random
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa

src = str(Path(__file__).parent).replace("tests", "") + "src"

import sys

sys.path.insert(0, src)
sys.path.insert(0, "c:/users/ort/git/daplapath")


from daplapath.path import LocalFileSystem

import sgis as sg

sg.config["file_system"] = LocalFileSystem

cols_should_be = ["komm_nr", "fylke_nr", "x", "geometry"]
n_should_be = [1, 10, 100, 1000, 10000, 2]
df = pd.concat(
    [
        sg.random_points(n_should_be[0]),
        sg.random_points(n_should_be[1], loc=0.5),
        sg.random_points(n_should_be[2], loc=5),
        sg.random_points(n_should_be[3], loc=50),
        sg.random_points(n_should_be[4], loc=50000),
    ],
)
df.index = pd.MultiIndex.from_arrays(
    [np.full(len(df), 0), [random.choice([*"abc"]) for _ in range(len(df))]]
)
df.crs = 25833
df["komm_nr"] = (
    ["0301"] * n_should_be[0]
    + ["4601"] * n_should_be[1]
    + ["4602"] * n_should_be[2]
    + ["3201"] * n_should_be[3]
    + ["5501"] * n_should_be[4]
)
df["fylke_nr"] = df["komm_nr"].str[:2]
df["x"] = range(len(df))
df["x"] = df["x"].astype("UInt32")
df["komm_nr"] = df["komm_nr"].astype(pd.StringDtype("pyarrow"))

df = df[cols_should_be]

# to be written separately
new_df = df.iloc[:2]
new_df["komm_nr"] = "5001"
new_df["fylke_nr"] = "50"

path = "c:/users/ort/downloads/test_partitioning.parquet"
path2 = "c:/users/ort/downloads/test_non_partitioning.parquet"

no_rows = df[~df.index.isin(df.index)]
assert not len(no_rows)

sg.write_geopandas(
    no_rows,
    path,
    partition_cols=["komm_nr", "fylke_nr"],
    pandas_fallback=True,
)

sg.write_geopandas(
    df,
    path,
    partition_cols=["komm_nr", "fylke_nr"],
)

sg.write_geopandas(
    new_df,
    path,
    partition_cols=["komm_nr", "fylke_nr"],
)
sg.write_geopandas(
    no_rows,
    path,
    partition_cols=["komm_nr", "fylke_nr"],
    file_system=LocalFileSystem(),
    pandas_fallback=True,
)

df2 = sg.read_geopandas(path)
assert len(df2) == sum(n_should_be), len(df2)
assert list(df2) == cols_should_be, list(df2)
assert isinstance(df2, gpd.GeoDataFrame)

df2 = sg.read_geopandas(
    path, filters=["komm_nr = 5001".split()], columns=["komm_nr", "geometry"]
)

assert len(df2) == 2, len(df2)
assert list(df2) == ["komm_nr", "geometry"], list(df2)
assert isinstance(df2, gpd.GeoDataFrame)

df2 = sg.read_geopandas(path, filters=["komm_nr = 5001".split()])
assert len(df2) == 2, len(df2)
assert list(df2) == cols_should_be, list(df2)
assert isinstance(df2, gpd.GeoDataFrame)

df2 = sg.read_geopandas(path, filters=["fylke_nr = 46".split()])
assert len(df2) == 110, len(df2)
assert list(df2) == cols_should_be, list(df2)
assert isinstance(df2, gpd.GeoDataFrame)
df2 = sg.read_geopandas(path, filters=["fylke_nr = 460".split()])
assert len(df2) == 0, len(df2)
assert isinstance(df2, gpd.GeoDataFrame)
assert list(sorted(df2)) == ["fylke_nr", "geometry", "komm_nr", "x"], list(df2)
df2 = sg.read_geopandas(
    path,
    filters=[("komm_nr", "=", "4601"), ("fylke_nr", "=", "46")],
    file_system=LocalFileSystem(),
)
assert len(df2) == 10, len(df2)
assert list(df2) == cols_should_be, list(df2)
assert isinstance(df2, gpd.GeoDataFrame)
df2 = sg.read_geopandas(
    path,
    filters=[("fylke_nr", "=", "46"), ("komm_nr", "=", "4601")],
    file_system=LocalFileSystem(),
)
assert len(df2) == 10, len(df2)
assert list(df2) == cols_should_be, list(df2)
assert isinstance(df2, gpd.GeoDataFrame)


df2 = sg.read_geopandas(
    path,
    filters=[("fylke_nr", "=", "03"), ("komm_nr", "=", "4601")],
    file_system=LocalFileSystem(),
)
assert not len(df2), len(df2)
assert len(df2.columns) == len(cols_should_be), list(df2)
assert isinstance(df2, gpd.GeoDataFrame)


df2 = sg.read_geopandas(
    path,
    filters=[("fylke_nr", "in", "(03, 46,1,  10x)")],
    file_system=LocalFileSystem(),
)
assert len(df2) == 111, len(df2)
assert len(df2.columns) == len(cols_should_be), list(df2)
assert isinstance(df2, gpd.GeoDataFrame)

df2 = sg.read_geopandas(
    Path(path) / "komm_nr=0301",
    file_system=LocalFileSystem(),
).convert_dtypes()
assert len(df2) == 1, len(df2)
assert len(df2.columns) == len(cols_should_be), list(df2)
assert isinstance(df2, gpd.GeoDataFrame)
assert [str(x) for x in df2.dtypes] == ["string", "string", "UInt32", "geometry"], [
    str(x) for x in df2.dtypes
]


# with vanilla pandas
df2 = pd.read_parquet(
    path,
    schema=pa.schema(
        [
            ("komm_nr", pa.string()),
            ("fylke_nr", pa.string()),
            ("x", pa.int16()),
            ("geometry", pa.binary()),
        ]
    ),
).convert_dtypes()
assert len(df2) == sum(n_should_be), (len(df2), sum(n_should_be))
assert list(df2) == cols_should_be, list(df2)
assert [str(x) for x in df2.dtypes] == ["string", "string", "Int16", "object"], [
    str(x) for x in df2.dtypes
]

for n, kommnr in zip(
    n_should_be, ["0301", "4601", "4602", "3201", "5501", "5001"], strict=True
):
    df2 = sg.read_geopandas(path, filters=[f"komm_nr = {kommnr}".split()])
    assert len(df2) == n
    assert list(df2) == cols_should_be, list(df2)
    assert isinstance(df2, gpd.GeoDataFrame)

    df2 = pd.read_parquet(
        path,
        schema=pa.schema(
            [
                ("komm_nr", pa.string()),
                ("fylke_nr", pa.string()),
                ("x", pa.int16()),
                ("geometry", pa.binary()),
            ]
        ),
        filters=[f"komm_nr = {kommnr}".split()],
    ).convert_dtypes()
    assert len(df2) == n
    assert list(df2) == cols_should_be, list(df2)


df2 = sg.read_geopandas(
    path,
    filters=[("fylke_nr", "=", "46"), ("komm_nr", "=", "4601")],
    schema=pa.schema(
        [
            ("komm_nr", pa.string()),
            ("x", pa.uint16()),
            ("geometry", pa.binary()),
        ]
    ),
    mask=df[df["komm_nr"] == "4601"].geometry.iloc[0].buffer(0.1),
    file_system=LocalFileSystem(),
).convert_dtypes()

assert len(df2), df2
assert [str(x) for x in df2.dtypes] == ["string", "UInt16", "geometry"], [
    str(x) for x in df2.dtypes
]
assert isinstance(df2, gpd.GeoDataFrame)


df2 = sg.read_geopandas(
    path,
    mask=df.geometry.iloc[0].buffer(0.1),
    file_system=LocalFileSystem(),
)
assert len(df2), df2
assert list(df2) == cols_should_be, list(df2)
assert isinstance(df2, gpd.GeoDataFrame)


df2 = sg.read_geopandas(
    path,
    mask=df.geometry.buffer(0),
    file_system=LocalFileSystem(),
)
assert not len(df2), df2
assert len(df2.columns) == len(cols_should_be), list(df2)
assert isinstance(df2, gpd.GeoDataFrame)


# without partitions
try:
    os.remove(path2)
except FileNotFoundError:
    pass
pd.concat([df, new_df]).to_parquet(path2)

df2 = sg.read_geopandas(
    path2,
    file_system=LocalFileSystem(),
)
assert len(df2) == sum(n_should_be), len(df2)
assert list(df2) == cols_should_be, list(df2)
assert isinstance(df2, gpd.GeoDataFrame)


# without partitions

df2 = sg.read_geopandas(
    path,
    mask=df.geometry.iloc[0].buffer(0.01),
    file_system=LocalFileSystem(),
)
assert len(df2), df2
assert list(df2) == cols_should_be, list(df2)
assert isinstance(df2, gpd.GeoDataFrame)

df2 = sg.read_geopandas(
    path2,
    filters=[("fylke_nr", "=", "46"), ("komm_nr", "=", "4601")],
    file_system=LocalFileSystem(),
)
assert len(df2) == 10, len(df2)
assert list(df2) == cols_should_be, list(df2)
assert isinstance(df2, gpd.GeoDataFrame)


raise ValueError("fjern sys.path.insert")

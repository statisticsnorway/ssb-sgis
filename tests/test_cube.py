# %%
import glob
import os
import sys
from pathlib import Path

import numpy as np
import shapely
import xarray as xr
from IPython.display import display

src = str(Path(__file__).parent.parent) + "/src"
testdata = str(Path(__file__).parent.parent) + "/tests/testdata/raster"

sys.path.insert(0, src)

import sgis as sg

path_singleband = testdata + "/dtm_10.tif"
path_two_bands = testdata + "/dtm_10_two_bands.tif"
path_sentinel = testdata + "/sentinel2"
# sg.DataCube._test = True


def x2(x):
    return x * 2


def not_test_xdataset():
    cube = sg.DataCube.from_root(
        testdata,
        filename_regex=sg.raster.SENTINEL2_FILENAME_REGEX,
        res=10,
        nodata=-32767,
    )[lambda x: x.date.notna()].load()

    print(cube)
    xdataset = cube.to_xarray()
    assert isinstance(xdataset, xr.Dataset)
    print(xdataset)
    print(xdataset.attrs)
    print(dir(xdataset))


def test_query():
    cube = sg.DataCube.from_root(testdata, endswith=".tif", nodata=0, res=10)

    for i in cube.date.unique():
        c1 = cube[cube.date == i]
        assert isinstance(c1, sg.DataCube)


def not_test_shape():
    cube = sg.DataCube.from_root(
        testdata,
        endswith=".tif",
        res=10,
        nodata=0,
    )[lambda x: [Path(r.path).parent != "sentinel2" for r in x]]
    print(list(cube))

    cube = cube.to_crs(25833).load()
    assert cube.res == 10, cube.res
    cube.res = 30
    cube = cube.to_crs(25833).load()
    assert cube.res == 30, cube.res

    cube.res = 10
    c = cube.centroid.iloc[1].buffer(100)
    cube = cube.clip(c)
    assert cube.res == 10, cube.res
    assert len(cube) == 4, len(cube)

    cube.res = 20
    cube = cube.clip(c).explode()
    assert cube.res == 20, cube.res
    assert (cube.shape == (20, 20)).all(), cube.shape


def test_copy():
    cube = sg.DataCube.from_root(
        testdata,
        endswith=".tif",
        res=10,
        crs=25833,
        filename_regex=sg.raster.SENTINEL2_FILENAME_REGEX,
    )

    assert [arr is None for arr in cube.arrays]
    cube2 = cube.load().load().load()
    assert [arr is None for arr in cube.arrays]
    assert [arr is not None for arr in cube2.arrays]

    print(cube2.max())
    cube2.parallelizer = sg.Parallel(2)
    cube3 = cube2.copy().map(x2)
    print(cube2.max())
    print(cube3.max())
    assert int(cube3.max().max()) == int(cube2.max().max() * 2), (
        int(cube3.max().max()),
        int(cube2.max().max() * 2),
    )


def not_test_gradient():
    cube = sg.DataCube.from_root(
        testdata,
        endswith=".tif",
        nodata=0,
        res=10,
    )[lambda x: x.path.str.lower().str.contains("dtm")].load()
    gradient = cube.copy().gradient()
    assert int(gradient.max().max()) == 17, gradient.max().max()

    degrees = cube.copy().gradient(degrees=True)
    assert int(degrees.max().max()) == 86, degrees.max().max()


def test_sentinel():
    cube = sg.DataCube.from_root(
        testdata,
        endswith=".tif",
        filename_regex=sg.raster.SENTINEL2_FILENAME_REGEX,
        res=10,
        nodata=-32767,
    )
    assert all(r.name is not None for r in cube), [r.name for r in cube]
    assert len(cube) == 10, len(cube)

    assert cube.band.str.startswith("B").all()
    assert ((x := cube.date.str[-1]).isin([*"0123456789"])).all(), x
    assert (cube.date.str.len() == 8).all()

    ndvi = cube.calculate_index(sg.indices.ndvi, band_name1="B4", band_name2="B8")
    assert len(ndvi) == 1, len(ndvi)

    cube.parallelizer = sg.Parallel(3)
    ndvi_parallelized = cube.calculate_index(
        sg.indices.ndvi, band_name1="B4", band_name2="B8"
    )

    assert ndvi.min().equals(ndvi_parallelized.min()), (
        ndvi.min(),
        ndvi_parallelized.min(),
    )


def not_test_df():
    df_path = testdata + "/cube_df.parquet"
    try:
        os.remove(df_path)
    except FileNotFoundError:
        pass
    try:
        cube = sg.DataCube.from_root(
            testdata, endswith=".tif", res=10, nodata=0
        ).explode()
        df = cube._prepare_df_for_parquet()
        df["my_idx"] = range(len(df))
        df.to_parquet(df_path)
        cube_from_cube_df = sg.DataCube.from_cube_df(df_path).explode()
        display(cube_from_cube_df.df)
        assert "my_idx" in cube_from_cube_df.df
        assert hasattr(cube_from_cube_df, "_from_cube_df")
        assert cube_from_cube_df.boxes.intersects(cube.unary_union).all()

        cube_from_cube_df = sg.DataCube.from_root(
            testdata, endswith=".tif", res=10, nodata=0
        ).explode()
        assert hasattr(cube_from_cube_df, "_from_cube_df")
        assert cube_from_cube_df.boxes.intersects(cube.unary_union).all()
        os.remove(df_path)
    except Exception as e:
        os.remove(df_path)
        raise e


def test_from_root():

    files = [file for file in glob.glob(str(Path(testdata)) + "/*") if ".tif" in file]
    cube = sg.DataCube.from_paths(files, res=10)

    cube = sg.DataCube.from_root(testdata, endswith=".tif", res=10, nodata=0).explode()
    assert len(cube) == 22, cube
    display(cube)

    cube = sg.DataCube.from_root(
        testdata, filename_regex=r"\.tif$", res=10, nodata=0
    ).explode()
    assert len(cube) == 22, cube
    display(cube)


def test_getitem():
    cube = sg.DataCube.from_root(
        testdata, endswith=".tif", crs=25833, res=10, nodata=0
    ).load()
    assert len(cube) == 20
    assert isinstance(cube[0], sg.Raster), type(cube[0])
    assert len(cube[1:3]) == 2
    assert len(cube[:3]) == 3
    assert (
        n := len(cube[lambda x: (x.length > 0) & (x.path.str.contains("dtm"))])
    ) == 2, n
    assert (
        n := len(cube[(cube.length > 0) & (cube.path.str.contains("FRC_B"))])
    ) == 10, n


def test_to_gdf():
    cube = (
        sg.DataCube.from_root(testdata, endswith=".tif", crs=25833, res=10, nodata=0)
        .load()[lambda x: x.path.str.contains("two_bands")]
        .explode()
    )
    assert len(cube) == 2, len(cube)


def not_test_to_crs():
    cube = sg.DataCube.from_root(testdata, endswith=".tif", nodata=0, res=10).explode()[
        lambda x: x.path.str.contains("two_bands")
    ]
    assert len(cube) == 2, len(cube)

    area_was = cube.area.sum()
    old = cube.copy()
    print(old.crs)
    cube = cube.to_crs(4326).to_crs(25833)  # .to_crs(old.crs)
    area_is = cube.area.sum()
    sg.explore(
        new_bounds=sg.to_gdf(old.bounds, 25833).to_crs(4326).to_crs(25833),
        cube_boxes=sg.to_gdf(cube.boxes, 25833),
        old_boxes=sg.to_gdf(old.boxes, 25833),
    )

    assert area_was == area_is, (area_was, area_is)

    merged_rio = cube.merge().to_gdf()

    merged_xarr = cube.load().merge().to_gdf()

    point = shapely.affinity.translate(cube.unary_union.centroid, xoff=0, yoff=300)
    buffered = sg.to_gdf(point, crs=25833).buffer(100)

    assert (
        merged_rio.clip(buffered)
        .geom_almost_equals(merged_xarr.clip(buffered), decimal=3)
        .all()
    )
    assert tuple(merged_rio.total_bounds) == tuple(merged_xarr.total_bounds)

    cube_25832 = sg.DataCube.from_root(
        testdata, endswith=".tif", crs=25832, nodata=0
    ).explode()[lambda x: x.name.str.contains("two_bands")]

    assert cube_25832.crs == 25832
    wpd_rio = cube_25832.load().to_crs(25833).to_gdf()

    assert tuple(merged_rio.total_bounds) == tuple(wpd_rio.total_bounds)
    equal_geoms = merged_rio.clip(buffered).geom_almost_equals(
        wpd_rio.clip(buffered), decimal=0
    )

    assert equal_geoms.all(), equal_geoms.value_counts()

    merged_xarr_32 = cube.load().to_crs(25832).merge().to_crs(25833).to_gdf()
    assert (
        merged_rio.clip(buffered)
        .geom_almost_equals(merged_xarr_32.clip(buffered), decimal=0)
        .all()
    )

    mergexarr32 = cube.to_crs(25832).merge().to_crs(25833).to_gdf()
    assert (
        merged_rio.clip(buffered)
        .geom_almost_equals(mergexarr32.clip(buffered), decimal=0)
        .all()
    )

    sg.explore(
        wpd_rio,
        merged_rio,
        # merged_xarr,
        mergexarr32,
        merged_xarr_32,
        "value",
    )
    assert [arr is None for arr in cube.arrays]


def not_test_merge():
    cube = sg.DataCube.from_root(
        testdata, crs=25833, endswith=".tif", res=10, nodata=0
    ).explode()
    assert len(cube) == 22, len(cube)

    all_merged = cube.copy().merge()
    assert len(all_merged) == 1, len(all_merged)

    only_multiband_merged = cube.copy().merge(by=["path"])
    assert len(only_multiband_merged) == 20, len(only_multiband_merged)

    # cube2 = cube.copy().load().merge_by_bounds()
    # assert len(cube2) == 3, len(cube2)

    cube.subfolder = cube.path.apply(lambda x: Path(x).parent)

    merge_by = cube.copy().load().merge(by="subfolder")
    assert len(merge_by) == 3, len(merge_by)
    assert merge_by.res == 10, merge_by.res

    # xarray_merge_by = cube.copy().load().merge(by="subfolder")
    # assert len(xarray_merge_by) == 3, len(xarray_merge_by)
    # assert xarray_merge_by.res == 10, xarray_merge_by.res

    # x_mean = int(np.mean([r.array.mean() for r in xarray_merge_by]))
    # r_mean = int(np.mean([r.array.mean() for r in merge_by]))
    # assert x_mean == r_mean, (x_mean, r_mean)


def not_test_dissolve():
    cube = sg.DataCube.from_root(testdata, endswith=".tif", res=10, nodata=0)
    cube = cube.merge_by_bounds()
    assert list(cube.shape) == [(1, 201, 201), (2, 201, 201)]
    print(cube)
    cube = cube.dissolve_bands("mean")
    assert list(cube.shape) == [(201, 201), (201, 201)]
    print(cube)


def test_explode():
    cube = sg.DataCube.from_root(testdata, endswith=".tif", res=10, nodata=0)[
        lambda x: ~x.path.str.lower().str.contains("sentinel")
    ]
    assert cube.shape.notna().all()
    assert cube.res == 10
    assert cube.boxes.notna().all().all()

    exploded = cube.copy().explode()
    assert len(cube) == 4
    assert len(exploded) == 6
    assert exploded.shape.notna().all()
    assert cube.res == 10
    assert exploded.boxes.notna().all().all()

    exploded2 = cube.copy().load().explode()
    assert len(cube) == 4
    assert len(exploded2) == 6
    assert exploded2.shape.notna().all()
    assert exploded2.boxes.notna().all().all()


def not_test_merge_from_array():
    should_give = np.array(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17]],
    )

    arr1 = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    arr2 = np.array([[9, 10, 11], [12, 13, 14], [15, 16, 17]])

    arr_3d = np.array(
        [
            [
                [10, 10, 10],
                [10, 10, 10],
                [10, 10, 10],
                [10, 10, 10],
                [10, 10, 10],
                [10, 10, 10],
            ]
        ]
    )

    assert arr1.shape == (3, 3)
    assert arr2.shape == (3, 3)
    assert arr_3d.shape == (1, 6, 3), arr_3d.shape

    r2 = sg.Raster.from_array(arr1, bounds=(0, 3, 3, 6), crs=25833)
    r1 = sg.Raster.from_array(arr2, bounds=(0, 0, 3, 3), crs=25833)
    r3 = sg.Raster.from_array(arr_3d, bounds=(0, 0, 3, 6), crs=25833)

    def should_be_same():
        cube = sg.DataCube([r1])
        merged = cube.merge()
        assert cube.total_bounds == merged.total_bounds
        assert cube.res.equals(merged.res)
        assert cube.area.sum() == merged.area.sum()
        assert cube.equals(merged), (cube.arrays.iloc[0], merged.arrays.iloc[0])

        cube = sg.DataCube([r1, r2])
        assert cube.area.sum() == 18, cube.area.sum()

        merged = cube.merge()
        assert np.array_equal(merged.arrays.iloc[0], should_give)
        assert len(merged) == 1
        assert merged.total_bounds == cube.total_bounds, (
            merged.total_bounds,
            cube.total_bounds,
        )
        assert merged.area.sum() == 18, merged.area.sum()

    should_be_same()

    def zeros_should_be_10(should_give):
        should_give = np.where(should_give == 0, 10, should_give)
        should_give = np.array([should_give])
        assert should_give.shape == (1, 6, 3)

        cube = sg.DataCube([r1, r2, r3])
        assert cube.area.sum() == 18 * 2, cube.area.sum()

        merged = cube.merge()
        print(merged.arrays.iloc[0])
        assert np.array_equal(merged.arrays.iloc[0], should_give), merged.arrays.iloc[0]
        sg.explore(merged.to_gdf(), "value")
        assert len(merged) == 1
        assert merged.area.sum() == 18, merged.area.sum()

    zeros_should_be_10(should_give)

    def len_should_be_2_and_3():
        cube = sg.DataCube([r1, r2, r3])
        cube.df["numcol"] = [0, 1, 2]
        cube.df["txtcol"] = [*"aab"]

        merged = cube.merge(by="txtcol")
        assert len(merged) == 2
        assert merged.area.sum() == 18 * 2, merged.area.sum()

        merged = cube.merge(by="numcol")
        assert len(merged) == 3
        assert merged.area.sum() == 18 * 2, merged.area.sum()

    def all_should_be_10():
        merged = sg.DataCube([r3, r2, r1]).merge()
        print(arr_3d.shape)
        print(merged.arrays.iloc[0].shape)
        assert np.array_equal(merged.arrays.iloc[0], arr_3d), merged.arrays.iloc[0]

    all_should_be_10()


def not_test_from_gdf():
    cube = sg.DataCube.from_root(
        testdata,
        endswith=".tif",
        res=10,
        filename_regex=sg.raster.SENTINEL2_FILENAME_REGEX,
    )
    gdf = cube[0].load().to_gdf("val")
    print(gdf)
    cube = sg.DataCube.from_gdf(gdf, tilesize=100, processes=1, columns=["val"], res=10)
    print(cube.df)

    cube = sg.DataCube.from_gdf(gdf, tilesize=100, processes=4, columns=["val"], res=10)
    print(cube.df)


def not_test_zonal():
    arr_1d = np.arange(0, 8)
    arr_2d = arr_1d + np.arange(8).reshape(-1, 1)
    print(arr_2d)
    assert arr_2d.shape == (8, 8)

    r = sg.Raster.from_array(arr_2d, crs=25833, bounds=(0, 0, 9, 9), nodata=0)
    r2 = sg.Raster.from_array(arr_2d, crs=25833, bounds=(0, 9, 9, 19), nodata=0)
    cube = sg.DataCube([r, r2])

    grid = sg.make_grid_from_bbox(*cube.total_bounds, gridsize=4, crs=25833)
    assert len(grid) == 24

    touch_or_overlap = grid.overlaps(cube.unary_union) | grid.within(cube.unary_union)
    assert sum(touch_or_overlap) == 15

    zonal = cube.zonal(grid, aggfunc="sum")
    print(zonal)
    assert len(zonal) == 15, len(zonal)

    print(zonal)
    assert list(sorted(zonal["sum"])) == [
        1,
        3,
        6,
        6,
        9,
        16,
        16,
        24,
        24,
        30,
        30,
        48,
        48,
        48,
        48,
    ], list(sorted(zonal["sum"]))


def not_test_parallel():
    cube = sg.DataCube.from_root(
        testdata,
        endswith=".tif",
        filename_regex=sg.raster.SENTINEL2_FILENAME_REGEX,
        res=10,
        crs=25833,
        nodata=0,
    )  # [lambda x: ~x.path.str.contains("entinel")]

    center = cube.unary_union.centroid.buffer(200)

    results = (cube.copy().clip(center)).map(x2).map(np.float32).explode()

    cube.parallelizer = sg.Parallel(3)
    results_parallelized = (cube.clip(center)).map(x2).map(np.float32).explode()

    assert len(results) == len(results_parallelized)
    for r1, r2 in zip(results, results_parallelized, strict=False):
        assert r1.equals(r2)
    # assert results.equals(results_parallelized)

    cube.parallelizer = None
    grid = sg.make_grid(center, gridsize=100, crs=cube.crs)
    zonal = cube.copy().zonal(grid, aggfunc=["sum", np.mean], by_date=False)
    print(zonal)
    # assert int(zonal["sum"].max()) == 10, int(zonal["sum"].max())
    # assert int(zonal["sum"].mean()) == 10, int(zonal["sum"].mean())

    cube.parallelizer = sg.Parallel(3)
    zonal_parallelized = cube.zonal(grid, aggfunc=["sum", np.mean], by_date=False)
    assert zonal.equals(zonal_parallelized)


def write_sentinel():
    src_path_sentinel = r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data"

    cube = sg.DataCube.from_root(
        src_path_sentinel,
        endswith=".tif",
        filename_regex=sg.raster.SENTINEL2_FILENAME_REGEX,
    )

    mask = sg.to_gdf(cube.unary_union.centroid.buffer(1000))

    def _save_raster(file, src_path_sentinel):
        if ".tif" not in file:
            return
        path = Path(src_path_sentinel) / file
        r = sg.Raster.from_path(path).clip(mask)
        out_path = str(Path(path_sentinel) / Path(file).stem) + "_clipped.tif"
        print(out_path)
        r.write(out_path)

    for file in os.listdir(src_path_sentinel):
        _save_raster(file, src_path_sentinel)

    for file in os.listdir(src_path_sentinel + "/MASKS"):
        _save_raster(file, src_path_sentinel + "/MASKS")


if __name__ == "__main__":
    # write_sentinel()

    def test_cube():
        test_sentinel()
        test_explode()
        test_getitem()
        test_to_gdf()
        test_query()
        test_copy()
        not_test_dissolve()
        test_from_root()
        not_test_parallel()
        not_test_from_gdf()
        not_test_shape()
        not_test_xdataset()
        not_test_merge()
        not_test_zonal()
        not_test_to_crs()
        not_test_merge_from_array()
        not_test_gradient()
        # not_test_df()

    # cProfile.run("test_merge_performance()", sort="cumtime")
    test_cube()


# %%

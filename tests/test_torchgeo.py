# %%
import multiprocessing
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import shapely
import xarray as xr
from IPython.display import display
from pyproj import CRS
from shapely import box

src = str(Path(__file__).parent.parent) + "/src"
testdata = str(Path(__file__).parent.parent) + "/tests/testdata/raster"

sys.path.insert(0, src)

import sgis as sg

path_sentinel = testdata + "/sentinel2"


def test_torch():

    from torch.utils.data import DataLoader
    from torchgeo.datasets import stack_samples
    from torchgeo.samplers import RandomGeoSampler

    torch_dataset = sg.torchgeo.Sentinel2(path_sentinel, res=10)
    data_loaded = torch_dataset[torch_dataset.bounds]
    torch_dataset.plot(data_loaded)

    sg.explore(torch_dataset, "value")
    gdf = sg.to_gdf(torch_dataset)
    assert len(gdf)

    assert len(torch_dataset) == 10, len(torch_dataset)

    sampler = RandomGeoSampler(torch_dataset, size=16, length=10)
    dataloader = DataLoader(
        torch_dataset, batch_size=2, sampler=sampler, collate_fn=stack_samples
    )

    for batch in dataloader:
        image = batch["image"]
        mask = batch["mask"]
        print(image)


if __name__ == "__main__":
    test_torch()


# %%

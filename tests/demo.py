from mdutils import Html
from mdutils.mdutils import MdUtils


mdFile = MdUtils(file_name="demo", title="Network analysis with gis_utils")

mdFile.new_paragraph(
    """
Network analysis with igraph, integrated with geopandas.

The package supports three types of network analysis:
- od_cost_matrix: fast many-to-many travel times/distances
- shortest_path: returns the geometry of the lowest-cost paths,
    or counts the number of times each road segment was used.
- service_area: returns the roads that can be reached within one or more breaks.
"""
)

mdFile.insert_code(
    """
import warnings
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gis_utils as gs

# ignoring some warnings:
pd.options.mode.chained_assignment = None  # ignore SettingWithCopyWarning for now
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
""",
    language="python",
)

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


src = str(Path(__file__).parent).strip("tests") + "src"
import sys


sys.path.append(src)
import gis_utils as gs


# ignoring some warnings:
pd.options.mode.chained_assignment = None  # ignore SettingWithCopyWarning for now
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)

mdFile.new_paragraph(
    """
Before network analysis can start, we need three things:
- a network
- some network analysis rules
- points from which the distance calculations will start and end

Let's start by loading the data.
"""
)

mdFile.insert_code(
    """
points = gpd.read_parquet("tests/testdata/random_points.parquet")
"""
)

points = gpd.read_parquet("tests/testdata/random_points.parquet")
table = []
for col in points.columns:
    table.append(col)
rows = 5
for i in points.index:
    if i > rows - 1:
        break
    for col in points.columns:
        table.append(points.loc[i, col])
mdFile.new_table(
    columns=len(points.columns), rows=rows + 1, text=table, text_align="right"
)

mdFile.insert_code(
    """
roads = gpd.read_parquet("tests/testdata/roads_oslo_2022.parquet")
roads = roads[["oneway", "drivetime_fw", "drivetime_bw", "geometry"]]
roads.head(3)
"""
)

mdFile.new_header(level=1, title="The Network")
mdFile.new_paragraph(
    """
The road data can be made into a network like this
"""
)


mdFile.insert_code(
    """
nw = gs.Network(roads)
nw
"""
)

path = "./tests/demo_files/demo_22_0.png"

mdFile.new_paragraph(Html.image(path=path, size="300"))


mdFile.insert_code(
    """

"""
)

mdFile.new_paragraph(
    """
"""
)

mdFile.new_paragraph(
    """
"""
)

mdFile.new_paragraph(
    """
"""
)


mdFile.create_md_file()

# sgis

```python

import os
import warnings

import numpy as np
import pandas as pd


os.chdir("../../src")
import sgis as sg


# ignore some warnings to make it cleaner
pd.options.mode.chained_assignment = None
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
import geopandas as gpd


points = sg.read_parquet_url(
    "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet"
)

### !!!
### COPY EVERYTHING BELOW INTO readme.md and change the png paths.
### !!!
```

sgis builds on the geopandas package and provides functions that make it easier to do advanced GIS in python.
Features include network analysis, functions for exploring multiple GeoDataFrames in a layered interactive map,
and vector operations like finding k-nearest neighbours, splitting lines by points, snapping and closing holes
in polygons by size.

## Network analysis examples

Preparing for network analysis:

```python
import sgis as sg


roads = sg.read_parquet_url(
    "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet"
)

nw = (
    sg.DirectedNetwork(roads)
    .remove_isolated()
    .make_directed_network(
        direction_col="oneway",
        direction_vals_bft=("B", "FT", "TF"),
        minute_cols=("drivetime_fw", "drivetime_bw"),
    )
)

rules = sg.NetworkAnalysisRules(weight="minutes")

nwa = sg.NetworkAnalysis(network=nw, rules=rules)

nwa
```

    NetworkAnalysis(
        network=DirectedNetwork(6364 km, percent_bidirectional=87),
        rules=NetworkAnalysisRules(weight=minutes, search_tolerance=250, search_factor=0, split_lines=False, ...),
        log=True, detailed_log=True,
    )

Get number of times each line segment was visited, with optional weighting.

```python
origins = points.iloc[:75]
destinations = points.iloc[75:150]

weights = pd.DataFrame(
    index=pd.MultiIndex.from_product([origins.index, destinations.index])
)
weights["weight"] = 10

frequencies = nwa.get_route_frequencies(origins, destinations, weight_df=weights)

# plot the results
m = sg.ThematicMap(sg.buff(frequencies, 15), column="frequency", black=True)
m.cmap = "plasma"
m.title = "Number of times each road was used,\nweighted * 10"
m.plot()
```

![png](network_analysis_examples_files/network_analysis_examples_5_0.png)

Fast many-to-many travel times/distances.

```python
od = nwa.od_cost_matrix(points, points)

print(od)
```

            origin  destination    minutes
    0            0            0   0.000000
    1            0            1  13.039830
    2            0            2  10.902453
    3            0            3   8.297021
    4            0            4  14.742294
    ...        ...          ...        ...
    999995     999          995  11.038673
    999996     999          996  17.820664
    999997     999          997  10.288465
    999998     999          998  14.798257
    999999     999          999   0.000000

    [1000000 rows x 3 columns]

Get the area that can be reached within one or more breaks.

```python
service_areas = nwa.service_area(
    points.iloc[[0]],
    breaks=np.arange(1, 11),
)

# plot the results
m = sg.ThematicMap(service_areas, column="minutes", black=True, size=10)
m.k = 10
m.title = "Roads that can be reached within 1 to 10 minutes"
m.plot()
```

![png](network_analysis_examples_files/network_analysis_examples_9_0.png)

Get one or more route per origin-destination pair.

```python
routes = nwa.get_k_routes(
    points.iloc[[0]], points.iloc[[1]], k=4, drop_middle_percent=50
)

m = sg.ThematicMap(sg.buff(routes, 15), column="k", black=True)
m.title = "Four fastest routes from A to B"
m.legend.title = "Rank"
m.plot()
```

![png](network_analysis_examples_files/network_analysis_examples_11_0.png)

More network analysis examples can be found here: https://github.com/statisticsnorway/ssb-sgis/blob/main/docs/network_analysis_demo_template.md

Road data for Norway can be downloaded here: https://kartkatalog.geonorge.no/metadata/nvdb-ruteplan-nettverksdatasett/8d0f9066-34f9-4423-be12-8e8523089313

```python
import os
```

```python
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd


os.chdir("../src")
import gis_utils as gs


os.chdir("..")

# ignore some warnings to make it cleaner
pd.options.mode.chained_assignment = None
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
plot_kwargs = {
    "facecolor": "#0f0f0f",
    "title_color": "#f7f7f7",
}
```

The package offers functions that simplify and ... geopandas code for long, repetitive code.

Also network analysis...

## Network analysis

The package supports three types of network analysis, and methods for customising and optimising your road data.

Analysis can start by initialising a NetworkAnalysis instance:

```python
from gis_utils import DirectedNetwork, NetworkAnalysis, NetworkAnalysisRules


roads = gpd.read_parquet("tests/testdata/roads_oslo_2022.parquet")

points = gpd.read_parquet("tests/testdata/random_points.parquet")
roads = gs.clean_clip(roads, points.geometry.iloc[0].buffer(1250))

nw = (
    DirectedNetwork(roads)
    .remove_isolated()
    .make_directed_network(
        direction_col="oneway",
        direction_vals_bft=("B", "FT", "TF"),
        minute_cols=("drivetime_fw", "drivetime_bw"),
    )
)

rules = NetworkAnalysisRules(weight="minutes")

nwa = NetworkAnalysis(network=nw, rules=rules)

nwa
```

    1 multi-geometry was split into single part geometries. Minute column(s) will be wrong for these rows.





    NetworkAnalysis(network=DirectedNetwork(219 km, directed=True), rules=NetworkAnalysisRules(weight='minutes', search_tolerance=250, search_factor=10, split_lines, ...))

```python
points = gpd.read_parquet("tests/testdata/random_points.parquet")
p1 = points.iloc[[0]]
```

### od_cost_matrix

Fast many-to-many travel times/distances

```python
od = nwa.od_cost_matrix(p1, points, lines=True)

print(od.head(3))

gs.qtm(
    od,
    "minutes",
    title="Travel time (minutes) from 1 to 1000 points.",
    **plot_kwargs,
)
```

      origin destination  minutes  \
    0   2662        2663      0.0
    1   2662        2664      NaN
    2   2662        2665      NaN

                                                geometry
    0  LINESTRING (263122.700 6651184.900, 263122.700...
    1  LINESTRING (263122.700 6651184.900, 272456.100...
    2  LINESTRING (263122.700 6651184.900, 270082.300...

![png](examples_files/examples_7_1.png)

### get_route

Get the actual paths:

```python
routes = nwa.get_route(points.iloc[[0]], points.sample(100), id_col="idx")

gs.qtm(
    gs.buff(routes, 12),
    "minutes",
    cmap="plasma",
    title="Travel times (minutes)",
    **plot_kwargs,
)

routes
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin</th>
      <th>destination</th>
      <th>minutes</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>311</td>
      <td>2.624607</td>
      <td>MULTILINESTRING Z ((263171.800 6651250.200 46....</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>935</td>
      <td>3.346286</td>
      <td>MULTILINESTRING Z ((263171.800 6651250.200 46....</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>228</td>
      <td>2.902736</td>
      <td>MULTILINESTRING Z ((262058.100 6651556.100 74....</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>316</td>
      <td>1.262305</td>
      <td>MULTILINESTRING Z ((263171.800 6651250.200 46....</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>484</td>
      <td>2.278675</td>
      <td>MULTILINESTRING Z ((263033.800 6650711.600 25....</td>
    </tr>
  </tbody>
</table>
</div>

![png](examples_files/examples_9_1.png)

### get_route_frequencies

Get the number of times each line segment was used:

```python
freq = nwa.get_route_frequencies(points.sample(100), points.sample(100))

gs.qtm(
    gs.buff(freq, 15),
    "n",
    scheme="naturalbreaks",
    cmap="plasma",
    title="Number of times each road was used.",
    **plot_kwargs,
)
```

![png](examples_files/examples_11_0.png)

### Service area

Get the area that can be reached within one or more breaks

```python
sa = nwa.service_area(p1, breaks=np.arange(1, 11), dissolve=False)

sa = sa.drop_duplicates(["source", "target"])

gs.qtm(
    sa,
    "minutes",
    k=10,
    title="Roads that can be reached within 1 to 10 minutes",
    legend=False,
    **plot_kwargs,
)
```

![png](examples_files/examples_13_0.png)

Check the log:

```python
nwa.log
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>endtime</th>
      <th>minutes_elapsed</th>
      <th>function</th>
      <th>origins_count</th>
      <th>destinations_count</th>
      <th>percent_missing</th>
      <th>cost_mean</th>
      <th>isolated_removed</th>
      <th>percent_directional</th>
      <th>weight</th>
      <th>...</th>
      <th>cost_p25</th>
      <th>cost_median</th>
      <th>cost_p75</th>
      <th>cost_std</th>
      <th>lines</th>
      <th>cutoff</th>
      <th>destination_count</th>
      <th>rowwise</th>
      <th>breaks</th>
      <th>dissolve</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-02-27 00:36:46</td>
      <td>0.0</td>
      <td>od_cost_matrix</td>
      <td>1</td>
      <td>1000.0</td>
      <td>93.7</td>
      <td>2.429468</td>
      <td>True</td>
      <td>44</td>
      <td>minutes</td>
      <td>...</td>
      <td>2.010145</td>
      <td>2.644186</td>
      <td>2.882840</td>
      <td>0.815339</td>
      <td>True</td>
      <td>None</td>
      <td>None</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-02-27 00:36:48</td>
      <td>0.0</td>
      <td>get_route</td>
      <td>1</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>2.482922</td>
      <td>True</td>
      <td>44</td>
      <td>minutes</td>
      <td>...</td>
      <td>2.278675</td>
      <td>2.624607</td>
      <td>2.902736</td>
      <td>0.786343</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-02-27 00:36:48</td>
      <td>0.0</td>
      <td>get_route_frequencies</td>
      <td>100</td>
      <td>100.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>44</td>
      <td>minutes</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-02-27 00:36:51</td>
      <td>0.0</td>
      <td>service_area</td>
      <td>1</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>6.371417</td>
      <td>True</td>
      <td>44</td>
      <td>minutes</td>
      <td>...</td>
      <td>4.000000</td>
      <td>6.000000</td>
      <td>8.000000</td>
      <td>2.420119</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1, 2, 3, 4, 5, 6, 7, 8, 9, 10</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 26 columns</p>
</div>

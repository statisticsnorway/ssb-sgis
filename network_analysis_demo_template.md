```python
# %%--
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
```

## Network analysis in ssb-gis-utils

##### igraph integrated with geopandas

The package supports three types of network analysis, and methods for customising and optimising your road data.

The package supports four types of network analysis:

- od_cost_matrix: fast many-to-many travel times/distances
- get_route: returns the geometry of the lowest-cost paths.
- get_route_frequencies: summarises how many times each road/line segment was used.
- service_area: returns the roads/lines that can be reached within one or more breaks.

```python
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd


os.chdir("../src")

import gis_utils as gs


os.chdir("..")
```

```python
# ignore some warnings to make it cleaner
pd.options.mode.chained_assignment = None
# warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
```

At the core of the network analysis, is the NetworkAnalysis class.

It takes a network and a set of rules for the analysis:

The rules can be instantiated like this:

```python

rules = gs.NetworkAnalysisRules(weight="minutes")
rules
```

    NetworkAnalysisRules(weight='minutes', search_tolerance=250, search_factor=10, split_lines=False, weight_to_nodes_dist=False, weight_to_nodes_kmh=None, weight_to_nodes_mph=None)

To create the network, we need some road data:

```python

roads = gpd.read_parquet("tests/testdata/roads_oslo_2022.parquet")
roads = roads[["oneway", "drivetime_fw", "drivetime_bw", "geometry"]]
roads.head(3)
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
      <th>oneway</th>
      <th>drivetime_fw</th>
      <th>drivetime_bw</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>119702</th>
      <td>B</td>
      <td>0.216611</td>
      <td>0.216611</td>
      <td>MULTILINESTRING Z ((258028.440 6674249.890 413...</td>
    </tr>
    <tr>
      <th>199710</th>
      <td>FT</td>
      <td>0.099323</td>
      <td>-1.000000</td>
      <td>MULTILINESTRING Z ((271778.700 6653238.900 138...</td>
    </tr>
    <tr>
      <th>199725</th>
      <td>FT</td>
      <td>0.173963</td>
      <td>-1.000000</td>
      <td>MULTILINESTRING Z ((271884.510 6653207.540 142...</td>
    </tr>
  </tbody>
</table>
</div>

The road data can be made into a Network instance like this:

```python
nw = gs.Network(roads).close_network_holes(1.5).remove_isolated().cut_lines(100)

nw = gs.Network(roads)
nw
```

    Network(3851 km, undirected)

The Network is now ready for undirected network analysis. The network can also be optimises with methods stored in the Network class. More about this further down in this notebook.

```python

nw = nw.close_network_holes(1.5).remove_isolated().cut_lines(250)
nw
```

    Network(3832 km, undirected)

For directed network analysis, the DirectedNetwork class can be used. This inherits all methods from the Network class, and also includes methods for making a directed network.

```python

nw = gs.DirectedNetwork(roads).remove_isolated()
nw
```

    DirectedNetwork(3407 km, percent_bidirectional=0)

We now have a DirectedNetwork instance. However, the network isn't actually directed yet, as indicated by the percent_bidirectional attribute above.

The roads going both ways have to be duplicated and the geometry of the new lines have to be flipped. The roads going backwards also have to be flipped.

This can be done with the make_directed_network method:

```python

nw = nw.make_directed_network(
    direction_col="oneway",
    direction_vals_bft=("B", "FT", "TF"),
    speed_col=None,
    minute_cols=("drivetime_fw", "drivetime_bw"),
    flat_speed=None,
)
nw
```

    DirectedNetwork(6364 km, percent_bidirectional=87)

The network has now almost doubled in length, since most roads are bidirectional in this network.

Norwegian road data can be made directional with a custom method:

```python

nw = gs.DirectedNetwork(roads).remove_isolated().make_directed_network_norway()
nw
```

    DirectedNetwork(6364 km, percent_bidirectional=87)

## NetworkAnalysis

To start the network analysis, we put our network and our rules into the NetworkAnalysis class:

```python

nwa = gs.NetworkAnalysis(network=nw, rules=rules)
nwa
```

    NetworkAnalysis(
        network=DirectedNetwork(6364 km, percent_bidirectional=87),
        rules=NetworkAnalysisRules(weight='minutes', search_tolerance=250, search_factor=10, split_lines=False, ...)
    )

We also need some points that will be our origins and destinations:

```python

points = gpd.read_parquet("tests/testdata/random_points.parquet")
points
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
      <th>idx</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>POINT (263122.700 6651184.900)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>POINT (272456.100 6653369.500)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>POINT (270082.300 6653032.700)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>POINT (259804.800 6650339.700)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>POINT (272876.200 6652889.100)</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>996</td>
      <td>POINT (266801.700 6647844.500)</td>
    </tr>
    <tr>
      <th>996</th>
      <td>997</td>
      <td>POINT (261274.000 6653593.400)</td>
    </tr>
    <tr>
      <th>997</th>
      <td>998</td>
      <td>POINT (263542.900 6645427.000)</td>
    </tr>
    <tr>
      <th>998</th>
      <td>999</td>
      <td>POINT (269226.700 6650628.000)</td>
    </tr>
    <tr>
      <th>999</th>
      <td>1000</td>
      <td>POINT (264570.300 6644239.500)</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 2 columns</p>
</div>

### OD cost matrix

od_cost_matrix calculates the traveltime from a set of origins to a set of destinations:

```python

od = nwa.od_cost_matrix(origins=points, destinations=points, id_col="idx")
od
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>12.930588</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>10.867076</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>8.075722</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>14.659333</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>999995</th>
      <td>1000</td>
      <td>996</td>
      <td>10.936357</td>
    </tr>
    <tr>
      <th>999996</th>
      <td>1000</td>
      <td>997</td>
      <td>17.476000</td>
    </tr>
    <tr>
      <th>999997</th>
      <td>1000</td>
      <td>998</td>
      <td>10.288465</td>
    </tr>
    <tr>
      <th>999998</th>
      <td>1000</td>
      <td>999</td>
      <td>14.798257</td>
    </tr>
    <tr>
      <th>999999</th>
      <td>1000</td>
      <td>1000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>1000000 rows × 3 columns</p>
</div>

Set 'lines' to True to get straight lines between origin and destination:

```python
od = nwa.od_cost_matrix(points.iloc[[0]], points, lines=True)

print(od.head(3))

gs.qtm(
    od,
    "minutes",
    title="Travel time (minutes) from 1 to 1000 addresses.",
)
```

      origin destination    minutes  \
    0  79166       79167   0.000000
    1  79166       79168  12.930588
    2  79166       79169  10.867076

                                                geometry
    0  LINESTRING (263122.700 6651184.900, 263122.700...
    1  LINESTRING (263122.700 6651184.900, 272456.100...
    2  LINESTRING (263122.700 6651184.900, 270082.300...

![png](network_analysis_demo_template_files/network_analysis_demo_template_25_1.png)

Information about the analyses are stored in a DataFrame in the 'log' attribute:

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
      <th>method</th>
      <th>origins_count</th>
      <th>destinations_count</th>
      <th>percent_missing</th>
      <th>cost_mean</th>
      <th>isolated_removed</th>
      <th>percent_bidirectional</th>
      <th>weight</th>
      <th>...</th>
      <th>weight_to_nodes_kmh</th>
      <th>weight_to_nodes_mph</th>
      <th>cost_p25</th>
      <th>cost_median</th>
      <th>cost_p75</th>
      <th>cost_std</th>
      <th>lines</th>
      <th>cutoff</th>
      <th>destination_count</th>
      <th>rowwise</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-03-01 20:18:58</td>
      <td>0.6</td>
      <td>od_cost_matrix</td>
      <td>1000</td>
      <td>1000</td>
      <td>0.5987</td>
      <td>15.063283</td>
      <td>True</td>
      <td>87</td>
      <td>minutes</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>10.362621</td>
      <td>14.572965</td>
      <td>19.124486</td>
      <td>6.648103</td>
      <td>False</td>
      <td>None</td>
      <td>None</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-03-01 20:19:09</td>
      <td>0.2</td>
      <td>od_cost_matrix</td>
      <td>1</td>
      <td>1000</td>
      <td>0.2000</td>
      <td>11.286299</td>
      <td>True</td>
      <td>87</td>
      <td>minutes</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>7.660459</td>
      <td>11.573666</td>
      <td>14.151198</td>
      <td>5.091459</td>
      <td>True</td>
      <td>None</td>
      <td>None</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 24 columns</p>
</div>

### Get route

The get_route method can be used to get the actual lowest cost path:

```python
routes = nwa.get_route(points.iloc[[0]], points.sample(100), id_col="idx")

gs.qtm(
    gs.buff(routes, 12),
    "minutes",
    cmap="plasma",
    title="Travel times (minutes)",
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
      <td>665</td>
      <td>5.741593</td>
      <td>MULTILINESTRING Z ((264822.621 6652289.866 125...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>285</td>
      <td>17.053776</td>
      <td>MULTILINESTRING Z ((273597.950 6652941.630 198...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>15</td>
      <td>14.734647</td>
      <td>MULTILINESTRING Z ((263171.800 6651250.200 46....</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>392</td>
      <td>14.155139</td>
      <td>MULTILINESTRING Z ((265675.300 6645750.199 144...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>281</td>
      <td>12.328688</td>
      <td>MULTILINESTRING Z ((263171.800 6651250.200 46....</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1</td>
      <td>476</td>
      <td>8.006030</td>
      <td>MULTILINESTRING Z ((268091.187 6652321.443 134...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>1</td>
      <td>352</td>
      <td>19.546321</td>
      <td>MULTILINESTRING Z ((266999.100 6640759.200 133...</td>
    </tr>
    <tr>
      <th>97</th>
      <td>1</td>
      <td>64</td>
      <td>7.665240</td>
      <td>MULTILINESTRING Z ((263171.800 6651250.200 46....</td>
    </tr>
    <tr>
      <th>98</th>
      <td>1</td>
      <td>460</td>
      <td>2.745346</td>
      <td>MULTILINESTRING Z ((262841.200 6651029.403 30....</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1</td>
      <td>104</td>
      <td>6.374744</td>
      <td>MULTILINESTRING Z ((265611.900 6651045.190 94....</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 4 columns</p>
</div>

![png](network_analysis_demo_template_files/network_analysis_demo_template_29_1.png)

## Get route frequencies

get_route_frequencies finds the number of times each road segment was used.

```python
pointsample = points.sample(100)
freq = nwa.get_route_frequencies(pointsample, pointsample)

gs.qtm(
    gs.buff(freq, 15),
    "n",
    scheme="naturalbreaks",
    cmap="plasma",
    title="Number of times each road was used (weight='minutes')",
)
```

![png](network_analysis_demo_template_files/network_analysis_demo_template_31_0.png)

The results will be quite different when it is the shortest, rather than the fastest, route that is used:

```python
nwa.rules.weight = "meters"

freq = nwa.get_route_frequencies(pointsample, pointsample)

gs.qtm(
    gs.buff(freq, 15),
    "n",
    scheme="naturalbreaks",
    cmap="plasma",
    title="Number of times each road was used (weight='meters')",
)
```

![png](network_analysis_demo_template_files/network_analysis_demo_template_33_0.png)

```python
nwa.rules.weight = "minutes"
```

### Service area

The service_area method finds the area that can be reached within one or more breaks.

Here, we find the areas that can be reached within 5, 10 and 15 minutes for five random points:

```python

sa = nwa.service_area(points.sample(5), breaks=(5, 10, 15), id_col="idx")
sa
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
      <th>minutes</th>
      <th>idx</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>42</td>
      <td>MULTILINESTRING Z ((261699.300 6650469.800 37....</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>42</td>
      <td>MULTILINESTRING Z ((264348.673 6648271.134 17....</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15</td>
      <td>42</td>
      <td>MULTILINESTRING Z ((266903.800 6651086.560 114...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>816</td>
      <td>MULTILINESTRING Z ((258314.423 6653716.416 128...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>816</td>
      <td>MULTILINESTRING Z ((257636.090 6655993.340 388...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15</td>
      <td>816</td>
      <td>MULTILINESTRING Z ((255703.040 6661235.800 176...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5</td>
      <td>592</td>
      <td>MULTILINESTRING Z ((259185.676 6652656.707 76....</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10</td>
      <td>592</td>
      <td>MULTILINESTRING Z ((256747.750 6655744.370 182...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>15</td>
      <td>592</td>
      <td>MULTILINESTRING Z ((264348.673 6648271.134 17....</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5</td>
      <td>208</td>
      <td>MULTILINESTRING Z ((267287.202 6646656.305 155...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>208</td>
      <td>MULTILINESTRING Z ((264348.673 6648271.134 17....</td>
    </tr>
    <tr>
      <th>11</th>
      <td>15</td>
      <td>208</td>
      <td>MULTILINESTRING Z ((266903.800 6651086.560 114...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5</td>
      <td>906</td>
      <td>MULTILINESTRING Z ((258314.423 6653716.416 128...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>10</td>
      <td>906</td>
      <td>MULTILINESTRING Z ((256747.750 6655744.370 182...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>906</td>
      <td>MULTILINESTRING Z ((266909.769 6651075.250 114...</td>
    </tr>
  </tbody>
</table>
</div>

```python
sa = nwa.service_area(points.iloc[[0]], breaks=np.arange(1, 11), id_col="idx")

gs.qtm(
    sa,
    "minutes",
    k=10,
    title="Roads that can be reached within 1 to 10 minutes",
)
sa
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
      <th>minutes</th>
      <th>idx</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>MULTILINESTRING Z ((263405.137 6651352.537 56....</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>MULTILINESTRING Z ((263273.605 6652079.666 67....</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>MULTILINESTRING Z ((262750.930 6651840.320 63....</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>MULTILINESTRING Z ((261870.921 6651358.348 77....</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>MULTILINESTRING Z ((265378.000 6650581.600 85....</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>1</td>
      <td>MULTILINESTRING Z ((265262.829 6650519.242 78....</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>1</td>
      <td>MULTILINESTRING Z ((264348.673 6648271.134 17....</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>1</td>
      <td>MULTILINESTRING Z ((266903.800 6651086.560 114...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1</td>
      <td>MULTILINESTRING Z ((263683.620 6648172.520 20....</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1</td>
      <td>MULTILINESTRING Z ((265657.500 6654844.860 296...</td>
    </tr>
  </tbody>
</table>
</div>

![png](network_analysis_demo_template_files/network_analysis_demo_template_37_1.png)

By default, only the lowest break is kept for overlapping areas from the same origin, meaning the area for minutes=10
covers not the entire area, only the outermost ring:

```python
gs.qtm(
    sa.query("minutes == 10"),
    color="yellow",
    title="Roads that can be reached within 10 minutes",
)
```

![png](network_analysis_demo_template_files/network_analysis_demo_template_39_0.png)

This behaviour can be changed by setting drop_duplicates to False.

Duplicate lines from different origins will not be removed. To drop all duplicates, if many
origins in close proximity, set 'dissolve' to False to get each individual road or line returned,
and then drop rows afterwards:

```python
sa = nwa.service_area(points.sample(100), breaks=5, dissolve=False)
print("rows before drop_duplicates:", len(sa))
sa = sa.drop_duplicates(["source", "target"])
print("rows after drop_duplicates:", len(sa))
```

    rows before drop_duplicates: 612026
    rows after drop_duplicates: 142986

Let's check the log:

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
      <th>method</th>
      <th>origins_count</th>
      <th>destinations_count</th>
      <th>percent_missing</th>
      <th>cost_mean</th>
      <th>isolated_removed</th>
      <th>percent_bidirectional</th>
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
      <td>2023-03-01 20:18:58</td>
      <td>0.6</td>
      <td>od_cost_matrix</td>
      <td>1000</td>
      <td>1000.0</td>
      <td>0.5987</td>
      <td>15.063283</td>
      <td>True</td>
      <td>87</td>
      <td>minutes</td>
      <td>...</td>
      <td>10.362621</td>
      <td>14.572965</td>
      <td>19.124486</td>
      <td>6.648103</td>
      <td>False</td>
      <td>None</td>
      <td>None</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-03-01 20:19:09</td>
      <td>0.2</td>
      <td>od_cost_matrix</td>
      <td>1</td>
      <td>1000.0</td>
      <td>0.2000</td>
      <td>11.286299</td>
      <td>True</td>
      <td>87</td>
      <td>minutes</td>
      <td>...</td>
      <td>7.660459</td>
      <td>11.573666</td>
      <td>14.151198</td>
      <td>5.091459</td>
      <td>True</td>
      <td>None</td>
      <td>None</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-03-01 20:19:50</td>
      <td>0.7</td>
      <td>get_route</td>
      <td>1</td>
      <td>100.0</td>
      <td>0.0000</td>
      <td>11.298471</td>
      <td>True</td>
      <td>87</td>
      <td>minutes</td>
      <td>...</td>
      <td>7.916974</td>
      <td>11.455953</td>
      <td>14.137341</td>
      <td>4.632490</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-03-01 20:24:36</td>
      <td>4.7</td>
      <td>get_route_frequencies</td>
      <td>100</td>
      <td>100.0</td>
      <td>0.0000</td>
      <td>0.074777</td>
      <td>True</td>
      <td>87</td>
      <td>minutes</td>
      <td>...</td>
      <td>0.016905</td>
      <td>0.047149</td>
      <td>0.097674</td>
      <td>0.091438</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-03-01 20:28:59</td>
      <td>4.3</td>
      <td>get_route_frequencies</td>
      <td>100</td>
      <td>100.0</td>
      <td>0.0000</td>
      <td>39.394569</td>
      <td>True</td>
      <td>87</td>
      <td>meters</td>
      <td>...</td>
      <td>9.645208</td>
      <td>24.764460</td>
      <td>47.777602</td>
      <td>54.211882</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-03-01 20:29:18</td>
      <td>0.2</td>
      <td>service_area</td>
      <td>5</td>
      <td>NaN</td>
      <td>0.0000</td>
      <td>10.000000</td>
      <td>True</td>
      <td>87</td>
      <td>minutes</td>
      <td>...</td>
      <td>5.000000</td>
      <td>10.000000</td>
      <td>15.000000</td>
      <td>4.225771</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>5, 10, 15</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-03-01 20:29:30</td>
      <td>0.1</td>
      <td>service_area</td>
      <td>1</td>
      <td>NaN</td>
      <td>0.0000</td>
      <td>5.500000</td>
      <td>True</td>
      <td>87</td>
      <td>minutes</td>
      <td>...</td>
      <td>3.250000</td>
      <td>5.500000</td>
      <td>7.750000</td>
      <td>3.027650</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>1, 2, 3, 4, 5, 6, 7, 8, 9, 10</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-03-01 20:29:53</td>
      <td>0.3</td>
      <td>service_area</td>
      <td>100</td>
      <td>NaN</td>
      <td>0.0000</td>
      <td>5.000000</td>
      <td>True</td>
      <td>87</td>
      <td>minutes</td>
      <td>...</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 26 columns</p>
</div>

### Customising the network

```python
nw = gs.Network(roads)
nw
```

    Network(3851 km, undirected)

If you want to manipulate the roads after instantiating the Network, you can access the GeoDataFrame in the 'gdf' attribute:

```python
nw.gdf.head(3)
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
      <th>oneway</th>
      <th>drivetime_fw</th>
      <th>drivetime_bw</th>
      <th>idx_orig</th>
      <th>meters</th>
      <th>source_wkt</th>
      <th>target_wkt</th>
      <th>source</th>
      <th>target</th>
      <th>n_source</th>
      <th>n_target</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B</td>
      <td>0.216611</td>
      <td>0.216611</td>
      <td>119702</td>
      <td>36.838554</td>
      <td>POINT (258028.4400000004 6674249.890000001)</td>
      <td>POINT (258023.5 6674213.59)</td>
      <td>0</td>
      <td>49540</td>
      <td>1</td>
      <td>2</td>
      <td>LINESTRING Z (258028.440 6674249.890 413.751, ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FT</td>
      <td>0.099323</td>
      <td>-1.000000</td>
      <td>199710</td>
      <td>110.359439</td>
      <td>POINT (271778.7000000002 6653238.9)</td>
      <td>POINT (271884.5099999998 6653207.539999999)</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>LINESTRING Z (271778.700 6653238.900 138.671, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FT</td>
      <td>0.173963</td>
      <td>-1.000000</td>
      <td>199725</td>
      <td>193.291970</td>
      <td>POINT (271884.5099999998 6653207.539999999)</td>
      <td>POINT (272069.7999999998 6653152.5)</td>
      <td>2</td>
      <td>60241</td>
      <td>4</td>
      <td>2</td>
      <td>LINESTRING Z (271884.510 6653207.540 142.211, ...</td>
    </tr>
  </tbody>
</table>
</div>

### remove_isolated

The above log file has a column called 'isolated_removed'. This is set to True because the method 'remove_isolated' was used before the analyses.

Networks often consist of one large, connected network and many small, isolated "network islands".

origins and destinations located inside these isolated networks, will have a hard time finding their way out.

The large, connected network component can be found (not removed) with the method get_largest_component:

```python
nw = nw.get_largest_component()

# the GeoDataFrame of the network is stored in the gdf attribute:
nw.gdf["connected_str"] = np.where(nw.gdf.connected == 1, "connected", "isolated")

gs.clipmap(
    nw.gdf,
    points.iloc[[0]].buffer(1000),
    column="connected_str",
    title="Connected and isolated networks",
    cmap="bwr",
    explore=False,
)
```

![png](network_analysis_demo_template_files/network_analysis_demo_template_49_0.png)

Use the remove_isolated method to remove the unconnected roads:

```python

nwa = gs.NetworkAnalysis(network=nw, rules=gs.NetworkAnalysisRules(weight="meters"))
od = nwa.od_cost_matrix(points, points)
percent_missing = od[nwa.rules.weight].isna().mean() * 100
print(f"Before removing isolated: {percent_missing=:.2f}")
```

    Before removing isolated: percent_missing=11.44

```python
nwa.network = nwa.network.remove_isolated()

od = nwa.od_cost_matrix(points, points)
percent_missing = od[nwa.rules.weight].isna().mean() * 100
print(f"After removing isolated: {percent_missing=:.2f}")
```

    After removing isolated: percent_missing=0.40

If the road data has some gaps between the segments, these can be filled with straight lines:

```python

nw = nw.close_network_holes(max_dist=1.5)  # meters
nw
```

    Network(3408 km, undirected)

The network analysis is done from node to node. In a service area analysis, the results will be inaccurate for long lines, since the destination will either be reached or not within the breaks. This can be fixed by cutting all lines to a maximum distance.

Note: cutting the lines can take a lot of time for large networks and low cut distances.

```python
nw = nw.cut_lines(100)  # meters
nw.gdf.length.max()
```

    100.00000000046512

## DirectedNetwork

Using the DirectedNetwork instead of the Network class, doesn't do anything to the network initially.

But if we use it directly in the NetworkAnalysis, we see that 0 percent of the lines go in both directions:

```python

nw = gs.DirectedNetwork(roads)
rules = gs.NetworkAnalysisRules(weight="metres")
nwa = gs.NetworkAnalysis(nw, rules=rules)
```

    c:\Users\ort\git\ssb-gis-utils\src\gis_utils\directednetwork.py:257: UserWarning: Your network is likely not directed. Only 0.0 percent of the lines go both ways. Try setting direction_col='oneway' in the 'make_directed_network' method
      warnings.warn(mess)

To make this network bidirectional, roads going both ways have to be duplicated and flipped. Roads going the opposite way also need to be flipped.

The key here is in the 'oneway' column:

```python
nw.gdf.oneway.value_counts()
```

    B     82230
    FT     7353
    TF     3812
    Name: oneway, dtype: int64

We use this to make the network bidirectional with the 'make_directed_network' method.

If we want a minute column, we also have to specify how to calculate this. Here, I use the two minute columns in the data:

```python
nw.gdf[["oneway", "drivetime_fw", "drivetime_bw"]].drop_duplicates(
    "oneway"
)  # dropping duplicates for illustration's sake
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
      <th>oneway</th>
      <th>drivetime_fw</th>
      <th>drivetime_bw</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B</td>
      <td>0.216611</td>
      <td>0.216611</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FT</td>
      <td>0.099323</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>TF</td>
      <td>-1.000000</td>
      <td>0.030141</td>
    </tr>
  </tbody>
</table>
</div>

Specify the values of the direction column in a tuple/list with the order "both ways", "from/forward", "to/backward".

```python
nw = nw.make_directed_network(
    direction_col="oneway",
    direction_vals_bft=("B", "FT", "TF"),
    minute_cols=("drivetime_fw", "drivetime_bw"),
)

nw.gdf["minutes"]
```

    0         0.216611
    1         0.028421
    2         0.047592
    3         0.026180
    4         0.023978
                ...
    175620    0.007564
    175621    0.020246
    175622    0.036810
    175623    0.003019
    175624    0.036975
    Name: minutes, Length: 175541, dtype: float64

You can also calculate minutes from a speed limit column. But you might want to do some manual adjusting, since keeping the speed limit at all times is unrealistic in most cases.

You can set a flat speed that will be used for the entire network. Decent if the travel mode is walking, bike, boat etc.

```python
bike_nw = nw.make_directed_network(
    direction_col="oneway",
    direction_vals_bft=("B", "FT", "TF"),
    flat_speed=20,
)

bike_nw.gdf["minutes"]
```

    0         30.698795
    1          8.055821
    2         13.489731
    3          7.420639
    4          6.796531
                ...
    339912     5.252557
    339913     5.778109
    339914    10.505145
    339915     0.861647
    339916    10.552264
    Name: minutes, Length: 339917, dtype: float64

## The NetworkAnalysisRules

#### weight

The weight parameter has to be specified. The weight can be the name of any numeric column in network.gdf.

Or, if the weight is 'meters' or 'metres', a meter column will be created. The coordinate reference system of the network has to be meters as well.

```python
rules = gs.NetworkAnalysisRules(weight="metres")
gs.NetworkAnalysis(nw, rules=rules).network.gdf["metres"]
```

    0         36.838554
    1          9.666985
    2         16.187677
    3          8.904767
    4          8.155837
                ...
    339912     6.303068
    339913     6.933731
    339914    12.606175
    339915     1.033976
    339916    12.662717
    Name: metres, Length: 339917, dtype: float64

If you want other distance units, create the column beforehand.

```python

nw.gdf = nw.gdf.to_crs(6576).assign(feet=lambda x: x.length)
rules = gs.NetworkAnalysisRules(weight="feet")
gs.NetworkAnalysis(nw, rules=rules).network.gdf.feet
```

    0         134.327479
    1          35.146303
    2          58.852473
    3          32.373751
    4          29.651735
                 ...
    339912     22.941131
    339913     25.236597
    339914     45.882497
    339915      3.763345
    339916     46.088281
    Name: feet, Length: 339917, dtype: float64

A minute column can be created through the 'make_directed_network' or 'make_directed_network_norway' methods.

```python
nw = (
    gs.DirectedNetwork(roads)
    .remove_isolated()
    .make_directed_network(
        direction_col="oneway",
        direction_vals_bft=("B", "FT", "TF"),
        minute_cols=("drivetime_fw", "drivetime_bw"),
    )
)

rules = gs.NetworkAnalysisRules(weight="minutes")

nwa = gs.NetworkAnalysis(network=nw, rules=rules)

nwa
```

    NetworkAnalysis(
        network=DirectedNetwork(6364 km, percent_bidirectional=87),
        rules=NetworkAnalysisRules(weight='minutes', search_tolerance=250, search_factor=10, split_lines=False, ...)
    )

### split_lines

By default, the origins and destinations are connected to the closest nodes of the network:

```python

nwa.rules.split_lines
```

    False

By setting 'split_lines' to True, the line closest to each point will be split in two where the point is closest to the line. The points can then start their travels in the middle of lines. This makes things more accurate, but it takes a little more time.

The split lines stays with the network until it is re-instantiated.

Splitting the lines will have a larger effect if the lines in the network are long, and/or if the distances to be calculated are short.

In this road network, most lines are short enough that splitting the lines usually doesn't do much. The longest lines are all in the forest.

```python
nwa.network.gdf.length.describe()
```

    count    160137.000000
    mean         39.741307
    std          71.096080
    min           0.213581
    25%          14.571123
    50%          28.016534
    75%          47.031438
    max        5213.749178
    dtype: float64

It has a minimal impact on the results. Here comes one example (get_route) and the average travel minutes (od_cost_matrix).

```python
nwa.rules.search_factor = 0

nwa.rules.split_lines = False

od = nwa.od_cost_matrix(points, points)
sp1 = nwa.get_route(points.iloc[[97]], points.iloc[[135]])
sp1["split_lines"] = "Not splitted"

nwa.rules.split_lines = True

od = nwa.od_cost_matrix(points, points)
sp2 = nwa.get_route(points.iloc[[97]], points.iloc[[135]])
sp2["split_lines"] = "Splitted"
```

In the get_route example, when the lines are split, the trip starts a bit further up in the bottom-right corner (when the search_factor is 0). The trip also ends in a roundtrip, since the line that is split is a oneway street. So you're allowed to go to the intersection where the blue line goes, but not to the point where the line is cut.

```python

gs.qtm(gs.gdf_concat([sp1, sp2]), column="split_lines", cmap="bwr")
```

![png](network_analysis_demo_template_files/network_analysis_demo_template_80_0.png)

But these kinds of deviations doesn't have much of an impact on the results in total here, where the mean is about 15 minutes. For shorter trips, the difference will be relatively larger, of course.

```python

nwa.log.loc[
    nwa.log.method == "od_cost_matrix",
    ["split_lines", "cost_mean", "cost_p25", "cost_median", "cost_p75", "cost_std"],
]

nwa.rules.split_lines = False
```

If the point is located in the middle of a very long line, it has to travel all the way to the end of the line and then, half the time, traverse the whole line.

#### search_factor

Since the closest node might be intraversable, the points are connected to all nodes within a given search_factor. The default is 10, which means that 10 meters and 10 percent is added to the closest distance to a node.

So if the closest node is 1 meter away, the point will be connected to all nodes within 11.1 meters.

If the closest node is 100 meters away, the point will be connected to all nodes within 120 meters.

And 0 means that only the closest node will be used.

Let's check how the search_factor influences the number of missing values:

```python
for search_factor in [0, 10, 50, 100]:
    nwa.rules.search_factor = search_factor
    od = nwa.od_cost_matrix(points, points)

nwa.rules.search_factor = 10  # back to default

nwa.log.iloc[-4:][["search_factor", "percent_missing"]]
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
      <th>search_factor</th>
      <th>percent_missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.8973</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10</td>
      <td>0.6983</td>
    </tr>
    <tr>
      <th>6</th>
      <td>50</td>
      <td>0.3994</td>
    </tr>
    <tr>
      <th>7</th>
      <td>100</td>
      <td>0.3994</td>
    </tr>
  </tbody>
</table>
</div>

The remaining missing points are far away from the network. It might not be desirable to get results for these points. But if it is, it can be done with the search_tolerance parameter.

#### search_tolerance

search_tolerance is the maximum distance a start- or destination can be from the network. If the closest node is above the search_tolerance, this point will not be eligable for the analysis.

The default is:

```python
rules.search_tolerance
```

    250

The search_tolerance unit is meters if the units of the crs is meters, which it is in this case:

```python
nw.gdf.crs.axis_info[0].unit_name
```

    'metre'

Let's check how the search_tolerance influences the number of missing values:

```python
for search_tolerance in [100, 250, 500, 5_000]:
    nwa.rules.search_tolerance = search_tolerance
    od = nwa.od_cost_matrix(points, points)

nwa.log.iloc[-4:][["search_tolerance", "percent_missing"]]
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
      <th>search_tolerance</th>
      <th>percent_missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>100</td>
      <td>2.0878</td>
    </tr>
    <tr>
      <th>9</th>
      <td>250</td>
      <td>0.6983</td>
    </tr>
    <tr>
      <th>10</th>
      <td>500</td>
      <td>0.4990</td>
    </tr>
    <tr>
      <th>11</th>
      <td>5000</td>
      <td>0.2995</td>
    </tr>
  </tbody>
</table>
</div>

The remaining 0.2 percent are two points trapped behind oneway streets going the wrong way. A high search_tolerance won't help here, since the points are only connected to the closest node and the nodes within the search_factor. So the fix here (if a fix is desirable), is a higher search_tolerance (see above), but this will give more inaccurate results for the rest of the points. So consider using strict rules at first, then loosen up for only the points that give you problems.

```python
nwa.rules.search_factor = 100

od = nwa.od_cost_matrix(points, points)

nwa.log.iloc[[-1]][["search_tolerance", "percent_missing"]]
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
      <th>search_tolerance</th>
      <th>percent_missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>5000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
# back to default:
nwa.rules.search_tolerance = 250
nwa.rules.search_factor = 10
```

Note: one of the points that had all missing values at a search*tolerance of 500, is on an island without a car ferry (but a regular ferry). With a search_tolerance of 5000, trips from this point will originate at the mainland with 0 weight penalty. If you want to include trips like this, it might be a good idea to give a weight for the trip to the mainland. this can be done with one of the 'weight_to_nodes*' parameters.

#### weight*to_nodes*

The class has three 'weight*to_nodes*' parameters. This is about the cost between the origins and destinations and the network nodes. All three paramters are set to False or None by default, meaning the cost will be 0.

This will produce inaccurate results for points that are far away from the network. Especially when the search_factor is high.

Therefore, you can set one of the 'weight*to_nodes*' parameters. If the weight is 'meters' (i.e. the length unit of the crs), setting 'weight_to_nodes_dist' to True will make the weight equivelant to the straight-line distance:

```python

gs.NetworkAnalysisRules(weight="meters", weight_to_nodes_dist=True)
```

    NetworkAnalysisRules(weight='meters', search_tolerance=250, search_factor=10, split_lines=False, weight_to_nodes_dist=True, weight_to_nodes_kmh=None, weight_to_nodes_mph=None)

If the weight is "minutes", you specify the speed in kilometers or miles per hour:

```python

gs.NetworkAnalysisRules(weight="minutes", weight_to_nodes_kmh=5)
```

    NetworkAnalysisRules(weight='minutes', search_tolerance=250, search_factor=10, split_lines=False, weight_to_nodes_dist=False, weight_to_nodes_kmh=5, weight_to_nodes_mph=None)

```python

gs.NetworkAnalysisRules(weight="minutes", weight_to_nodes_mph=3)
```

    NetworkAnalysisRules(weight='minutes', search_tolerance=250, search_factor=10, split_lines=False, weight_to_nodes_dist=False, weight_to_nodes_kmh=None, weight_to_nodes_mph=3)

Let's check how the speed to the nodes influences the average speed:

```python
for weight_to_nodes_kmh in [5, 20, 50, 0]:
    nwa.rules.weight_to_nodes_kmh = weight_to_nodes_kmh
    od = nwa.od_cost_matrix(points, points)

nwa.log.iloc[-4:][["weight_to_nodes_kmh", "cost_mean"]]
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
      <th>weight_to_nodes_kmh</th>
      <th>cost_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>5</td>
      <td>15.621510</td>
    </tr>
    <tr>
      <th>14</th>
      <td>20</td>
      <td>15.303529</td>
    </tr>
    <tr>
      <th>15</th>
      <td>50</td>
      <td>15.235100</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>15.189379</td>
    </tr>
  </tbody>
</table>
</div>

```python

```

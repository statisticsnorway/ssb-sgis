from io import BytesIO

import geopandas as gpd
import requests
from geopandas import GeoDataFrame


def read_parquet_url(url: str) -> GeoDataFrame:
    """Reads geoparquet from a URL.

    Args:
        url: URL containing a geoparquet file.

    Returns:
        A GeoDataFrame.

    Examples:
    ---------
    >>> from sgis import read_parquet_url
    >>> url = "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet"
    >>> points = read_parquet_url(url)
    >>> points
          idx                        geometry
    0       1  POINT (263122.700 6651184.900)
    1       2  POINT (272456.100 6653369.500)
    2       3  POINT (270082.300 6653032.700)
    3       4  POINT (259804.800 6650339.700)
    4       5  POINT (272876.200 6652889.100)
    ..    ...                             ...
    995   996  POINT (266801.700 6647844.500)
    996   997  POINT (261274.000 6653593.400)
    997   998  POINT (263542.900 6645427.000)
    998   999  POINT (269226.700 6650628.000)
    999  1000  POINT (264570.300 6644239.500)
    <BLANKLINE>
    [1000 rows x 2 columns]

    >>> url = "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet"
    >>> roads = read_parquet_url(url)
    >>> roads
              linkid fromnode   tonode         streetname fromangle  ... estimateddrivetimebus_fw estimateddrivetimebus_bw lengde SHAPE_Length                                           geometry
    119702    121690   135529   135537       skogsbilsveg       203  ...                 0.188357                 0.188357    NaN    36.838554  MULTILINESTRING Z ((258028.440 6674249.890 413...
    199710    203624   223290   223291              Rv163       107  ...                 0.086368                -1.000000    NaN   110.359439  MULTILINESTRING Z ((271778.700 6653238.900 138...
    199725    203640   223291   223306              Rv163       107  ...                 0.151272                -1.000000    NaN   193.291970  MULTILINESTRING Z ((271884.510 6653207.540 142...
    199726    203641   223291   223307              Rv163        85  ...                 0.010284                -1.000000    NaN    13.141298  MULTILINESTRING Z ((271884.510 6653207.540 142...
    199733    203648   223313   223291              Rv163       135  ...                 0.007910                -1.000000    NaN    10.107394  MULTILINESTRING Z ((271877.373 6653214.697 141...
    ...          ...      ...      ...                ...       ...  ...                      ...                      ...    ...          ...                                                ...
    1944129  2137454  1836098  1836031              Rv191        74  ...                 0.027344                -1.000000    NaN     8.735034  MULTILINESTRING Z ((268649.503 6651869.320 111...
    1944392  2137723   575129  1836232     Bygdøynesveien       333  ...                 0.117879                 0.117879    NaN    46.109289  MULTILINESTRING Z ((259501.570 6648459.580 3.2...
    1944398  2137729  1836234  1836235  Fredriksborgveien       183  ...                 0.059338                 0.059338    NaN    23.210556  MULTILINESTRING Z ((258292.600 6648313.440 18....
    1944409  2137741  1836240  1481147  Fredriksborgveien       178  ...                 0.020547                 0.020547    NaN     8.036981  MULTILINESTRING Z ((258291.452 6648289.258 19....
    1944415  2137747  1493921  1836244         privat veg       234  ...                 0.717860                 0.717860    NaN    59.821648  MULTILINESTRING Z ((260762.830 6650240.620 43....
    <BLANKLINE>
    [93395 rows x 46 columns]
    """
    r = requests.get(url, timeout=10)
    return gpd.read_parquet(BytesIO(r.content))

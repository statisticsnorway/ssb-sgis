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
    >>> print(points.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
          idx                    geometry
    0       1  POINT (263122.7 6651184.9)
    1       2  POINT (272456.1 6653369.5)
    2       3  POINT (270082.3 6653032.7)
    3       4  POINT (259804.8 6650339.7)
    4       5  POINT (272876.2 6652889.1)
    ..    ...                         ...
    995   996  POINT (266801.7 6647844.5)
    996   997    POINT (261274 6653593.4)
    997   998    POINT (263542.9 6645427)
    998   999    POINT (269226.7 6650628)
    999  1000  POINT (264570.3 6644239.5)
    <BLANKLINE>
    [1000 rows x 2 columns]

    >>> url = "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet"
    >>> roads = read_parquet_url(url)
    >>> print(roads.to_string(max_rows=10, max_cols=2, max_colwidth=50, show_dimensions=True))
              linkid  ...                                           geometry
    119702    121690  ...  MULTILINESTRING Z ((258028.44 6674249.89 413.7...
    199710    203624  ...  MULTILINESTRING Z ((271778.7 6653238.9 138.671...
    199725    203640  ...  MULTILINESTRING Z ((271884.51 6653207.54 142.2...
    199726    203641  ...  MULTILINESTRING Z ((271884.51 6653207.54 142.2...
    199733    203648  ...  MULTILINESTRING Z ((271877.373 6653214.697 141...
    ...          ...  ...                                                ...
    1944129  2137454  ...  MULTILINESTRING Z ((268649.503 6651869.32 111....
    1944392  2137723  ...  MULTILINESTRING Z ((259501.57 6648459.58 3.29,...
    1944398  2137729  ...  MULTILINESTRING Z ((258292.6 6648313.44 18.53,...
    1944409  2137741  ...  MULTILINESTRING Z ((258291.452 6648289.258 19....
    1944415  2137747  ...  MULTILINESTRING Z ((260762.83 6650240.62 43.72...
    <BLANKLINE>
    [93395 rows x 46 columns]
    """
    r = requests.get(url, timeout=10)
    return gpd.read_parquet(BytesIO(r.content))

"""Contains the NetworkAnalysis class.

The class has five methods: od_cost_matrix, get_route, get_k_routes,
get_route_frequencies and service_area.
"""


from datetime import datetime
from time import perf_counter

import igraph
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from igraph import Graph
from pandas import DataFrame

from ._get_route import _get_route
from ._od_cost_matrix import _od_cost_matrix
from ._points import Destinations, Origins
from ._service_area import _service_area
from .directednetwork import DirectedNetwork
from ..geopandas_tools.general import gdf_concat, _push_geom_col
from .network import Network, _edge_ids
from ..geopandas_tools.line_operations import split_lines_at_closest_point
from .networkanalysisrules import NetworkAnalysisRules


class NetworkAnalysis:
    """Class for doing network analysis.

    It takes a (Directed)Network and rules (NetworkAnalysisRules).

    Args:
        network: either the base Network class or a subclass, chiefly the
            DirectedNetwork class. The network should be customized beforehand, but
            can also be accessed through the 'network' attribute of this class.
        rules: NetworkAnalysisRules class instance.
        log: If True (the default), a DataFrame with information about each
            analysis run will be stored in the 'log' attribute.
        detailed_log: If True (the default), the log DataFrame will include columns for
            all arguments held by the NetworkAnalysisRules class and the analysis
            method used. Will also include standard deviation, 25th, 50th and 75th
            percentile of the weight column in the results.

    Attributes:
        network: the Network instance
        rules: the NetworkAnalysisRules instance
        log: A DataFrame with information about each analysis run
        origins: the origins used in the latest analysis run, in the form of an Origins
            class instance. The GeoDataFrame is stored in the 'gdf' attribute, with a
            column 'missing' that can be used for investigation/debugging. So, write
            e.g.: nw.origins.gdf.missing.value_counts()
        destinations: the destinations used in the latest analysis run, in the form of
            a Destinations class instance. The GeoDataFrame is stored in the 'gdf'
            attribute, with a column 'missing' that can be used for
            investigation/debugging. So, write e.g.:
            nw.destinations.gdf.missing.value_counts()

    Raises:
        TypeError: if 'rules' is not of type NetworkAnalysisRules
        TypeError: if 'network' is not of type Network (subclasses are)

    See also
    --------
    DirectedNetwork : for customising and optimising line data before directed network
        analysis
    Network : for customising and optimising line data before undirected network
        analysis

    Examples
    --------
    Read example data.

    >>> from sgis import read_parquet_url
    >>> roads = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_eidskog_2022.parquet")
    >>> points = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_eidskog.parquet")

    Creating a NetworkAnalysis class instance.

    >>> from sgis import DirectedNetwork, NetworkAnalysisRules, NetworkAnalysis
    >>> nw = (
    ...     DirectedNetwork(roads)
    ...     .remove_isolated()
    ...     .make_directed_network(
    ...         direction_col="oneway",
    ...         direction_vals_bft=("B", "FT", "TF"),
    ...         minute_cols=("drivetime_fw", "drivetime_bw"),
    ...     )
    ... )
    >>> rules = NetworkAnalysisRules(weight="minutes")
    >>> nwa = NetworkAnalysis(network=nw, rules=rules, detailed_log=False)
    >>> nwa
    NetworkAnalysis(
        network=DirectedNetwork(6364 km, percent_bidirectional=87),
        rules=NetworkAnalysisRules(weight='minutes', search_tolerance=250, search_factor=10, split_lines=False, ...)
    )

    od_cost_matrix: fast many-to-many travel time/distance calculation.

    >>> od = nwa.od_cost_matrix(points, points, id_col="idx")
    >>> od
            origin  destination    minutes
    0            1            1   0.000000
    1            1            2  11.983871
    2            1            3   9.822048
    3            1            4   7.838012
    4            1            5  13.708064
    ...        ...          ...        ...
    999995    1000          996  10.315319
    999996    1000          997  16.839220
    999997    1000          998   6.539792
    999998    1000          999  14.182613
    999999    1000         1000   0.000000

    [1000000 rows x 3 columns]

    get_route: get the geometry of the routes.

    >>> routes = nwa.get_route(points.sample(1), points, id_col="idx")
    >>> routes
        origin  destination    minutes                                           geometry
    0       432            1   9.550767  MULTILINESTRING Z ((268999.800 6653318.400 186...
    1       432            2   6.865927  MULTILINESTRING Z ((268999.800 6653318.400 186...
    2       432            3   4.250621  MULTILINESTRING Z ((270054.367 6653367.774 144...
    3       432            4  16.726048  MULTILINESTRING Z ((259735.774 6650362.886 24....
    4       432            5   8.590120  MULTILINESTRING Z ((272816.062 6652789.578 163...
    ..      ...          ...        ...                                                ...
    995     432          996  14.356265  MULTILINESTRING Z ((266881.100 6647824.860 132...
    996     432          997  11.840131  MULTILINESTRING Z ((268999.800 6653318.400 186...
    997     432          998  17.317844  MULTILINESTRING Z ((263489.330 6645655.330 11....
    998     432          999   8.577831  MULTILINESTRING Z ((269217.997 6650654.895 166...
    999     432         1000  18.385282  MULTILINESTRING Z ((268999.800 6653318.400 186...

    [1000 rows x 4 columns]

    get_route_frequencies: get the number of times each line segment was used.

    >>> freq = nwa.get_route_frequencies(points.sample(25), points.sample(25))
    >>> freq
        source target      n                                           geometry
    137866  19095  44962    1.0  LINESTRING Z (265476.114 6645475.318 160.724, ...
    138905  30597  16266    1.0  LINESTRING Z (272648.400 6652234.800 178.170, ...
    138903  16266  45388    1.0  LINESTRING Z (272642.602 6652236.229 178.687, ...
    138894  43025  30588    1.0  LINESTRING Z (272446.600 6652253.700 162.970, ...
    138892  30588  16021    1.0  LINESTRING Z (272414.400 6652263.100 161.170, ...
    ...       ...    ...    ...                                                ...
    158287  78157  78156  176.0  LINESTRING Z (263975.482 6653605.092 132.739, ...
    149697  72562  72563  180.0  LINESTRING Z (265179.202 6651549.723 81.532, 2...
    149698  72563  72564  180.0  LINESTRING Z (265178.761 6651549.956 81.561, 2...
    149695  72560  72561  180.0  LINESTRING Z (265457.755 6651249.238 76.502, 2...
    149696  72561  72562  180.0  LINESTRING Z (265180.086 6651549.259 81.473, 2...

    [12231 rows x 4 columns]

    service_area: get the area that can be reached within one or more breaks.

    >>> sa = nwa.service_area(
    ...         points.iloc[:3],
    ...         breaks=[5, 10, 15],
    ...         id_col="idx",
    ...     )
    >>> sa
    idx  minutes                                           geometry
    0    1        5  MULTILINESTRING Z ((265378.000 6650581.600 85....
    1    1       10  MULTILINESTRING Z ((264348.673 6648271.134 17....
    2    1       15  MULTILINESTRING Z ((263110.060 6658296.870 154...
    3    2        5  MULTILINESTRING Z ((273330.930 6653248.870 208...
    4    2       10  MULTILINESTRING Z ((266909.769 6651075.250 114...
    5    2       15  MULTILINESTRING Z ((264348.673 6648271.134 17....
    6    3        5  MULTILINESTRING Z ((266909.769 6651075.250 114...
    7    3       10  MULTILINESTRING Z ((264348.673 6648271.134 17....
    8    3       15  MULTILINESTRING Z ((273161.140 6654455.240 229...

    get_k_routes: get the geometry of the k low-cost routes for each od pair.

    >>> k_routes = nwa.get_k_routes(
    ...    points.iloc[[0]],
    ...    points.iloc[1:3],
    ...    k=3,
    ...    drop_middle_percent=50,
    ...    id_col="idx"
    ...    )
    >>> k_routes
    origin  destination    minutes  k                                           geometry
    0       1            2  12.930588  1  MULTILINESTRING Z ((272281.367 6653079.745 160...
    1       1            2  14.128866  2  MULTILINESTRING Z ((272281.367 6653079.745 160...
    2       1            2  20.030052  3  MULTILINESTRING Z ((272281.367 6653079.745 160...
    3       1            3  10.867076  1  MULTILINESTRING Z ((270054.367 6653367.774 144...
    4       1            3  11.535946  2  MULTILINESTRING Z ((270074.933 6653001.553 118...
    5       1            3  14.867076  3  MULTILINESTRING Z ((265313.000 6650960.400 97....

    Check the log.

    >>> nwa.log
                endtime  minutes_elapsed                 method  origins_count  destinations_count  percent_missing  ...  search_tolerance  search_factor  split_lines weight_to_nodes_dist  weight_to_nodes_kmh  weight_to_nodes_mph
    0 2023-03-01 16:58:37              0.4         od_cost_matrix           1000              1000.0           0.5987  ...               250             10        False                False                 None                 None
    1 2023-03-01 17:05:26              6.7              get_route              1              1000.0           0.0000  ...               250             10        False                False                 None                 None
    2 2023-03-01 17:06:21              0.3  get_route_frequencies             25                25.0           0.0000  ...               250             10        False                False                 None                 None
    3 2023-03-01 17:07:25              0.2           service_area              3                 NaN           0.0000  ...               250             10        False                False                 None                 None
    4 2023-03-01 17:07:46              0.1           get_k_routes              1                 2.0           0.0000  ...               250             10        False                False                 None                 None

    [5 rows x 16 columns]
    """

    def __init__(
        self,
        network: Network | DirectedNetwork,
        rules: NetworkAnalysisRules,
        log: bool = True,
        detailed_log: bool = True,
    ):
        """Checks types and does some validation."""
        self.network = network
        self.rules = rules
        self._log = log
        self.detailed_log = detailed_log

        if not isinstance(rules, NetworkAnalysisRules):
            raise TypeError(
                f"'rules' should be of type NetworkAnalysisRules. Got {type(rules)}"
            )

        if not isinstance(network, Network):
            raise TypeError(
                "'network' should of type DirectedNetwork or Network. "
                f"Got {type(network)}"
            )

        self.network.gdf = self.rules._validate_weight(
            self.network.gdf, raise_error=False
        )

        if isinstance(self.network, DirectedNetwork):
            self.network._warn_if_undirected()

        self._update_wkts()
        self.rules._update_rules()

        if log:
            self.log = DataFrame()

    def od_cost_matrix(
        self,
        origins: GeoDataFrame,
        destinations: GeoDataFrame,
        id_col: str | tuple[str, str] | None = None,
        *,
        lines: bool = False,
        rowwise: bool = False,
        cutoff: int | None = None,
        destination_count: int | None = None,
    ) -> DataFrame | GeoDataFrame:
        """Fast calculation of many-to-many travel costs.

        Finds the the lowest cost (minutes, meters, etc.) from a set of origins to a
        set of destinations. If the weight is meters, the shortest route will be
        found. If the weight is minutes, the fastest route will be found.

        Args:
            origins: GeoDataFrame of points from where the trips will originate
            destinations: GeoDataFrame of points from where the trips will terminate
            id_col: column(s) to be used as identifier for the origins and
                destinations. If two different columns, put it in a tuple as
                ("origin_col", "destination_col") If None, an arbitrary id will be
                returned.
            lines: if True, returns a geometry column with straight lines between
                origin and destination. Defaults to False.
            rowwise: if False (the default), it will calculate the cost from each
                origins to each destination. If true, it will calculate the cost from
                origin 1 to destination 1, origin 2 to destination 2 and so on.
            cutoff: the maximum cost (weight) for the trips. Defaults to None,
                meaning all rows will be included. NaNs will also be removed if cutoff
                is specified.
            destination_count: number of closest destinations to keep for each origin.
                If None (the default), all trips will be included. The number of
                destinations might be higher than the destination count if trips have
                equal cost.

        Returns:
            A DataFrame with the columns 'origin', 'destination' and the weight column.
            If lines is True, adds a geometry column with straight lines between origin
            and destination.

        Examples
        --------
        Travel time from 1000 to 1000 points. Rows where origin and destination is the
        the same has 0 in cost.

        >>> nwa = NetworkAnalysis(network=nw, rules=rules)
        >>> od = nwa.od_cost_matrix(points, points, id_col="idx")
        >>> od
                origin  destination    minutes
        0            1            1   0.000000
        1            1            2  11.983871
        2            1            3   9.822048
        3            1            4   7.838012
        4            1            5  13.708064
        ...        ...          ...        ...
        999995    1000          996  10.315319
        999996    1000          997  16.839220
        999997    1000          998   6.539792
        999998    1000          999  14.182613
        999999    1000         1000   0.000000

        [1000000 rows x 3 columns]

        Travel time from 1000 to 1000 points rowwise.

        >>> points_reversed = points.iloc[::-1]
        >>> od = nwa.od_cost_matrix(points, points_reversed, rowwise=True, id_col="idx")
        >>> od
            origin  destination    minutes
        0         1         1000  14.657289
        1         2          999   8.378826
        2         3          998  15.147861
        3         4          997   8.889927
        4         5          996  16.371447
        ..      ...          ...        ...
        995     996            5  16.644710
        996     997            4   9.015495
        997     998            3  18.342336
        998     999            2   9.410509
        999    1000            1  14.892648

        [1000 rows x 3 columns]

        Get only five lowest costs for each origin. The rows don't add up to 5000
        because some origins cannot find (m)any destinations with the default
        NetworkAnalysisRules.

        >>> od = nwa.od_cost_matrix(points, points, destination_count=5, id_col="idx")
        >>> od
            origin  destination   minutes
        0          1            1  0.000000
        1          1           98  0.810943
        2          1          136  0.966702
        3          1          318  1.075858
        4          1          675  1.176377
        ...      ...          ...       ...
        4962    1000           95  0.000000
        4963    1000          304  1.065610
        4964    1000          334  1.180584
        4965    1000          400  0.484851
        4966    1000         1000  0.000000

        [4967 rows x 3 columns]

        Get costs less than ten minutes.

        >>> od = nwa.od_cost_matrix(points, points, cutoff=10, id_col="idx")
        >>> od
                origin  destination   minutes
        0            1            1  0.000000
        1            1            4  8.075722
        2            1            8  4.037207
        3            1           10  8.243380
        4            1           11  5.486970
        ...        ...          ...       ...
        228574    1000          985  7.725878
        228575    1000          988  9.801021
        228576    1000          989  2.446085
        228577    1000          990  7.968874
        228578    1000         1000  0.000000

        [228579 rows x 3 columns]

        """
        if self._log:
            time_ = perf_counter()

        self._prepare_network_analysis(origins, destinations, id_col)

        results = _od_cost_matrix(
            graph=self.graph,
            origins=self.origins.gdf,
            destinations=self.destinations.gdf,
            weight=self.rules.weight,
            lines=lines,
            cutoff=cutoff,
            destination_count=destination_count,
            rowwise=rowwise,
        )

        self.origins._get_n_missing(results, "origin")
        self.destinations._get_n_missing(results, "destination")

        if id_col:
            results["origin"] = results["origin"].map(self.origins.id_dict)
            results["destination"] = results["destination"].map(
                self.destinations.id_dict
            )

        if lines:
            results = _push_geom_col(results)

        if self.rules.split_lines:
            self._unsplit_network()

        if self._log:
            minutes_elapsed = round((perf_counter() - time_) / 60, 1)
            self._runlog(
                "od_cost_matrix",
                results,
                minutes_elapsed,
                lines=lines,
                cutoff=cutoff,
                destination_count=destination_count,
                rowwise=rowwise,
            )

        return results

    def get_route(
        self,
        origins: GeoDataFrame,
        destinations: GeoDataFrame,
        id_col: str | tuple[str, str] | None = None,
        *,
        rowwise: bool = False,
        cutoff: int | None = None,
        destination_count: int | None = None,
    ) -> GeoDataFrame:
        """Returns the geometry of the low-cost route between origins and destinations.

        Finds the route with the lowest cost (minutes, meters, etc.) from a set of
        origins to a set of destinations. If the weight is meters, the shortest route
        will be found. If the weight is minutes, the fastest route will be found.

        Args:
            origins: GeoDataFrame of points from where the routes will originate
            destinations: GeoDataFrame of points from where the routes will terminate
            id_col: optional column to be used as identifier of the service areas. If
                None, an arbitrary id will be used.
            rowwise: if False (the default), it will calculate the cost from each
                origins to each destination. If true, it will calculate the cost from
                origin 1 to destination 1, origin 2 to destination 2 and so on.
            cutoff: the maximum cost (weight) for the trips. Defaults to None,
                meaning all rows will be included. NaNs will also be removed if cutoff
                is specified.
            destination_count: number of closest destinations to keep for each origin.
                If None (the default), all trips will be included. The number of
                destinations might be higher than the destination count if trips have
                equal cost.

        Returns:
            A GeoDataFrame with the columns 'origin', 'destination', the weight
            column and the geometry of the route between origin and destination.

        Raises:
            ValueError: if no paths were found.

        Examples
        --------
        Get routes from 1 to 1000 points.

        >>> nwa = NetworkAnalysis(network=nw, rules=rules)
        >>> routes = nwa.get_route(points.iloc[[0]], points, id_col="idx")
        >>> routes
            origin  destination    minutes                                           geometry
        0         1            2  12.930588  MULTILINESTRING Z ((272281.367 6653079.745 160...
        1         1            3  10.867076  MULTILINESTRING Z ((270054.367 6653367.774 144...
        2         1            4   8.075722  MULTILINESTRING Z ((259735.774 6650362.886 24....
        3         1            5  14.659333  MULTILINESTRING Z ((272281.367 6653079.745 160...
        4         1            6  14.406460  MULTILINESTRING Z ((257034.948 6652685.595 156...
        ..      ...          ...        ...                                                ...
        992       1          996  10.858519  MULTILINESTRING Z ((266881.100 6647824.860 132...
        993       1          997   7.461032  MULTILINESTRING Z ((262623.190 6652506.640 79....
        994       1          998  10.698588  MULTILINESTRING Z ((263489.330 6645655.330 11....
        995       1          999  10.109855  MULTILINESTRING Z ((269217.997 6650654.895 166...
        996       1         1000  14.657289  MULTILINESTRING Z ((264475.675 6644245.782 114...

        [997 rows x 4 columns]
        """
        if self._log:
            time_ = perf_counter()

        self._prepare_network_analysis(origins, destinations, id_col)

        results = _get_route(
            graph=self.graph,
            origins=self.origins.gdf,
            destinations=self.destinations.gdf,
            weight=self.rules.weight,
            roads=self.network.gdf,
            cutoff=cutoff,
            destination_count=destination_count,
            rowwise=rowwise,
        )

        self.origins._get_n_missing(results, "origin")
        self.destinations._get_n_missing(results, "destination")

        if id_col:
            results["origin"] = results["origin"].map(self.origins.id_dict)
            results["destination"] = results["destination"].map(
                self.destinations.id_dict
            )

        results = _push_geom_col(results)

        if self.rules.split_lines:
            self._unsplit_network()

        if self._log:
            minutes_elapsed = round((perf_counter() - time_) / 60, 1)
            self._runlog(
                "get_route",
                results,
                minutes_elapsed,
                cutoff=cutoff,
                destination_count=destination_count,
                rowwise=rowwise,
            )

        return results

    def get_k_routes(
        self,
        origins: GeoDataFrame,
        destinations: GeoDataFrame,
        *,
        k: int,
        drop_middle_percent: int,
        id_col: str | tuple[str, str] | None = None,
        rowwise=False,
        cutoff: int = None,
        destination_count: int = None,
    ) -> GeoDataFrame:
        """Returns the geometry of 1 or more routes between origins and destinations.

        Finds the route with the lowest cost (minutes, meters, etc.) from a set of
        origins to a set of destinations. Then the middle part of the route is removed
        from the graph the new low-cost path is found. Repeats k times. If k=1, it is
        identical to the get_route method.

        Note:
            How many percent of the route to drop from the graph, will determine how
            many k routes will be found. If 100 percent of the route is dropped, it is
            very hard to find more than one path for each OD pair. If
            'drop_middle_percent' is 1, the resulting routes might be very similar,
            depending on the layout of the network.

        Args:
            origins: GeoDataFrame of points from where the routes will originate
            destinations: GeoDataFrame of points from where the routes will terminate
            k: the number of low-cost routes to find
            drop_middle_percent: how many percent of the middle part of the routes
                that should be removed from the graph before the next k route is
                calculated. If set to 100, only the median edge will be removed.
                If set to 0, all but the first and last edge will be removed. The
                graph is copied for each od pair.
            id_col: optional column to be used as identifier of the service areas. If
                None, an arbitrary id will be used.
            rowwise: if False (the default), it will calculate the cost from each
                origins to each destination. If true, it will calculate the cost from
                origin 1 to destination 1, origin 2 to destination 2 and so on.
            cutoff: the maximum cost (weight) for the trips. Defaults to None,
                meaning all rows will be included. NaNs will also be removed if cutoff
                is specified.
            destination_count: number of closest destinations to keep for each origin.
                If None (the default), all trips will be included. The number of
                destinations might be higher than the destination count if trips have
                equal cost.

        Returns:
            A GeoDataFrame with the columns 'origin', 'destination', the weight
            column and the geometry of the route between origin and destination.

        Raises:
            ValueError: if no paths were found.
            ValueError: if drop_middle_percent is not between 0 and 100.

        Examples
        --------

        Getting 10 fastest routes from one point to another point.

        >>> k_routes = nwa.get_k_routes(
        ...             point1,
        ...             point2,
        ...             k=10,
        ...             drop_middle_percent=1
        ...         )
        >>> k_routes
        origin destination    minutes   k                                           geometry
        0  79166       79167  12.930588   1  MULTILINESTRING Z ((272281.367 6653079.745 160...
        1  79166       79167  13.975082   2  MULTILINESTRING Z ((272281.367 6653079.745 160...
        2  79166       79167  14.128866   3  MULTILINESTRING Z ((272281.367 6653079.745 160...
        3  79166       79167  14.788440   4  MULTILINESTRING Z ((263171.800 6651250.200 46....
        4  79166       79167  14.853351   5  MULTILINESTRING Z ((263171.800 6651250.200 46....
        5  79166       79167  15.314692   6  MULTILINESTRING Z ((272281.367 6653079.745 160...
        6  79166       79167  16.108029   7  MULTILINESTRING Z ((272281.367 6653079.745 160...
        7  79166       79167  16.374740   8  MULTILINESTRING Z ((272281.367 6653079.745 160...
        8  79166       79167  16.404011   9  MULTILINESTRING Z ((272281.367 6653079.745 160...
        9  79166       79167  17.677964  10  MULTILINESTRING Z ((272281.367 6653079.745 160...

        We got all 10 routes because only the middle 1 percent of the routes are removed in
        each iteration. Let's compare with dropping middle 50 and middle 100 percent.

        >>> point1 = points.iloc[[0]]
        >>> point2 = points.iloc[[1]]
        >>> nwa = NetworkAnalysis(network=nw, rules=rules)
        >>> k_routes = nwa.get_k_routes(
        ...             point1,
        ...             point2,
        ...             k=10,
        ...             drop_middle_percent=50
        ...         )
        >>> k_routes
        origin destination    minutes  k                                           geometry
        0  79166       79167  12.930588  1  MULTILINESTRING Z ((272281.367 6653079.745 160...
        1  79166       79167  14.128866  2  MULTILINESTRING Z ((272281.367 6653079.745 160...
        2  79166       79167  20.030052  3  MULTILINESTRING Z ((272281.367 6653079.745 160...
        3  79166       79167  23.397536  4  MULTILINESTRING Z ((265226.515 6650674.617 88....

        >>> k_routes = nwa.get_k_routes(
        ...             point1,
        ...             point2,
        ...             k=10,
        ...             drop_middle_percent=100
        ...         )
        >>> k_routes
        origin destination    minutes  k                                           geometry
        0  79166       79167  12.930588  1  MULTILINESTRING Z ((272281.367 6653079.745 160...

        """
        if not 0 <= drop_middle_percent <= 100:
            raise ValueError("'drop_middle_percent' should be between 0 and 100")

        if self._log:
            time_ = perf_counter()

        self._prepare_network_analysis(origins, destinations, id_col)

        results = _get_route(
            graph=self.graph,
            origins=self.origins.gdf,
            destinations=self.destinations.gdf,
            weight=self.rules.weight,
            roads=self.network.gdf,
            cutoff=cutoff,
            destination_count=destination_count,
            rowwise=rowwise,
            k=k,
            drop_middle_percent=drop_middle_percent,
        )

        self.origins._get_n_missing(results, "origin")
        self.destinations._get_n_missing(results, "destination")

        if id_col:
            results["origin"] = results["origin"].map(self.origins.id_dict)
            results["destination"] = results["destination"].map(
                self.destinations.id_dict
            )

        results = _push_geom_col(results)

        if self.rules.split_lines:
            self._unsplit_network()

        if self._log:
            minutes_elapsed = round((perf_counter() - time_) / 60, 1)
            self._runlog(
                "get_k_routes",
                results,
                minutes_elapsed,
                cutoff=cutoff,
                destination_count=destination_count,
                rowwise=rowwise,
            )

        return results

    def get_route_frequencies(
        self,
        origins: GeoDataFrame,
        destinations: GeoDataFrame,
    ) -> GeoDataFrame:
        """Finds the number of times each line segment was visited in all trips.

        Finds the route with the lowest cost (minutes, meters, etc.) from a set of
        origins to a set of destinations. If the weight is meters, the shortest route
        will be found. If the weight is minutes, the fastest route will be found.

        Args:
            origins: GeoDataFrame of points from where the routes will originate
            destinations: GeoDataFrame of points from where the routes will terminate

        Returns:
            A GeoDataFrame with all line segments that were visited at least once,
            with the column 'n', which is the number of times the segment was visited
            for all the trips.

        Note:
            The resulting lines will keep all columns of the 'gdf' of the Network.

        Raises:
            ValueError: if no paths were found.

        Examples
        --------
        Get number of times each road was visited for trips from 25 to 25 points.

        >>> nwa = NetworkAnalysis(network=nw, rules=rules)
        >>> freq = nwa.get_route_frequencies(points.sample(25), points.sample(25))
        >>> freq
            source target      n                                           geometry
        137866  19095  44962    1.0  LINESTRING Z (265476.114 6645475.318 160.724, ...
        138905  30597  16266    1.0  LINESTRING Z (272648.400 6652234.800 178.170, ...
        138903  16266  45388    1.0  LINESTRING Z (272642.602 6652236.229 178.687, ...
        138894  43025  30588    1.0  LINESTRING Z (272446.600 6652253.700 162.970, ...
        138892  30588  16021    1.0  LINESTRING Z (272414.400 6652263.100 161.170, ...
        ...       ...    ...    ...                                                ...
        158287  78157  78156  176.0  LINESTRING Z (263975.482 6653605.092 132.739, ...
        149697  72562  72563  180.0  LINESTRING Z (265179.202 6651549.723 81.532, 2...
        149698  72563  72564  180.0  LINESTRING Z (265178.761 6651549.956 81.561, 2...
        149695  72560  72561  180.0  LINESTRING Z (265457.755 6651249.238 76.502, 2...
        149696  72561  72562  180.0  LINESTRING Z (265180.086 6651549.259 81.473, 2...

        [12231 rows x 4 columns]
        """
        if self._log:
            time_ = perf_counter()

        self._prepare_network_analysis(origins, destinations, None)

        results = _get_route(
            graph=self.graph,
            origins=self.origins.gdf,
            destinations=self.destinations.gdf,
            weight=self.rules.weight,
            roads=self.network.gdf,
            summarise=True,
        )

        results = _push_geom_col(results)

        results = results.sort_values("n")

        if self.rules.split_lines:
            self._unsplit_network()

        if self._log:
            minutes_elapsed = round((perf_counter() - time_) / 60, 1)
            self._runlog(
                "get_route_frequencies",
                results,
                minutes_elapsed,
            )

        return results

    def service_area(
        self,
        origins: GeoDataFrame,
        breaks: int | float | tuple[int | float],
        *,
        id_col: str | None = None,
        drop_duplicates: bool = True,
        dissolve: bool = True,
    ) -> GeoDataFrame:
        """Returns the lines that can be reached within breaks (weight values).

        It finds all the network lines that can be reached within each weight
        impedance, given in the breaks argument as one or more integers/floats.
        The breaks are sorted in ascending order, and duplicate lines from

        Args:
            origins: GeoDataFrame of points from where the service areas will
                originate
            breaks: one or more integers or floats which will be the
                maximum weight for the service areas. Calculates multiple areas for
                each origins if multiple breaks.
            id_col: optional column to be used as identifier of the service areas.
                If None, an arbitrary id will be used.
            drop_duplicates: If True (the default), duplicate lines from the same
                origin will be removed. Priority is given to the lower break values,
                meaning the highest break will only cover the outermost ring of the
                total service area for the origin. If False, the higher breaks will
                also cover the inner rings of the origin's service area.
            dissolve: If True (the default), each service area will be dissolved into
                one long multilinestring. If False, the individual line segments will
                be returned. Duplicate lines can then be removed, or occurences
                counted.

        Returns:
            A GeoDataFrame with one row per origin and break, with a dissolved line
            geometry of the lines that can be reached within the break. Duplicate lines
            from an origin will be removed, with priority given to the lower break.
            the roads that can be reached within the break
            for each origin. If dissolve is False, the columns will be the weight
            column, which contains the relevant break, and the if_col if specified,
            or the column 'origin' if not. If dissolve is False, it will return all
            the columns of the network.gdf as well. The columns 'source' and 'target'
            can be used to remove duplicates, or count occurences.

        Examples
        --------
        10 minute service area for one origin point.

        >>> nwa = NetworkAnalysis(network=nw, rules=rules)
        >>> sa = nwa.service_area(
        ...         points.iloc[[0]],
        ...         breaks=10,
        ...         id_col="idx",
        ...     )
        >>> sa
            idx  minutes                                           geometry
        0    1       10  MULTILINESTRING Z ((264348.673 6648271.134 17....

        Service areas of 5, 10 and 15 minutes from three origin points.

        >>> nwa = NetworkAnalysis(network=nw, rules=rules)
        >>> sa = nwa.service_area(
        ...         points.iloc[:2],
        ...         breaks=[5, 10, 15],
        ...         id_col="idx",
        ...     )
        >>> sa
            idx  minutes                                           geometry
        0    1        5  MULTILINESTRING Z ((265378.000 6650581.600 85....
        1    1       10  MULTILINESTRING Z ((264348.673 6648271.134 17....
        2    1       15  MULTILINESTRING Z ((263110.060 6658296.870 154...
        3    2        5  MULTILINESTRING Z ((273330.930 6653248.870 208...
        4    2       10  MULTILINESTRING Z ((266909.769 6651075.250 114...
        5    2       15  MULTILINESTRING Z ((264348.673 6648271.134 17....
        """
        if self._log:
            time_ = perf_counter()

        self._prepare_network_analysis(origins, id_col=id_col)

        # sort the breaks as an np.ndarray
        breaks = self._sort_breaks(breaks)

        results = _service_area(
            graph=self.graph,
            origins=self.origins.gdf,
            weight=self.rules.weight,
            lines=self.network.gdf,
            breaks=breaks,
        )

        if drop_duplicates:
            results = results.drop_duplicates(["source", "target", "origin"])

        if dissolve:
            results = (
                results.dissolve(by=["origin", self.rules.weight])
                .reset_index()
                .loc[:, ["origin", self.rules.weight, "geometry"]]
            )

        # add missing rows as NaNs
        missing = self.origins.gdf.loc[
            ~self.origins.gdf["temp_idx"].isin(results["origin"])
        ].rename(columns={"temp_idx": "origin"})[["origin"]]

        if len(missing):
            missing["geometry"] = np.nan
            results = pd.concat([results, missing], ignore_index=True)

        if id_col:
            results[id_col] = results["origin"].map(self.origins.id_dict)
            results = results.drop("origin", axis=1)

        results = _push_geom_col(results)

        if self.rules.split_lines:
            self._unsplit_network()

        if self._log:
            minutes_elapsed = round((perf_counter() - time_) / 60, 1)
            self._runlog(
                "service_area",
                results,
                minutes_elapsed,
                breaks=breaks,
                dissolve=dissolve,
            )

        return results

    def _log_df_template(self, method: str, minutes_elapsed: float) -> DataFrame:
        """Creates a DataFrame with one row and the main columns.

        To be run after each network analysis.

        Args:
            method: the name of the network analysis method used
            minutes_elapsed: time use of the method

        Returns:
            A one-row DataFrame with log info columns

        Note:
            The 'isolated_removed' column does not account for
            preperation done before initialising the (Directed)Network class.
        """
        data = {
            "endtime": pd.to_datetime(datetime.now()).floor("S").to_pydatetime(),
            "minutes_elapsed": minutes_elapsed,
            "method": method,
            "origins_count": np.nan,
            "destinations_count": np.nan,
            "percent_missing": np.nan,
            "cost_mean": np.nan,
        }
        if self.detailed_log:
            data = data | {
                "isolated_removed": self.network._isolated_removed,
                "percent_bidirectional": self.network.percent_bidirectional,
            }

        df = DataFrame(data, index=[0])

        if not self.detailed_log:
            return df

        for key, value in self.rules.__dict__.items():
            if key.startswith("_") or key.endswith("_"):
                continue
            df = pd.concat([df, pd.DataFrame({key: [value]})], axis=1)

        return df

    def _runlog(
        self,
        fun: str,
        results: DataFrame | GeoDataFrame,
        minutes_elapsed: float,
        **kwargs,
    ) -> None:
        df = self._log_df_template(fun, minutes_elapsed)

        df["origins_count"] = len(self.origins.gdf)

        if self.rules.weight in results.columns:
            df["percent_missing"] = results[self.rules.weight].isna().mean() * 100
            df["cost_mean"] = results[self.rules.weight].mean()
            if self.detailed_log:
                df["cost_p25"] = results[self.rules.weight].quantile(0.25)
                df["cost_median"] = results[self.rules.weight].median()
                df["cost_p75"] = results[self.rules.weight].quantile(0.75)
                df["cost_std"] = results[self.rules.weight].std()

        if fun == "service_area":
            df["percent_missing"] = results["geometry"].isna().mean() * 100
        else:
            df["destinations_count"] = len(self.destinations.gdf)

        if self.detailed_log:
            for key, value in kwargs.items():
                if isinstance(value, np.ndarray):
                    value = list(value)
                if isinstance(value, (list, tuple)):
                    value = [str(x) for x in value]
                    value = ", ".join(value)
                df[key] = value

        self.log = pd.concat([self.log, df], ignore_index=True)

    def _prepare_network_analysis(
        self, origins, destinations=None, id_col: str | None = None
    ) -> None:
        """Prepares the weight column, node ids, origins, destinations and graph.

        Updates the graph only if it is not yet created and no parts of the analysis
        has changed. this method is run inside od_cost_matrix, get_route and
        service_area.
        """
        self.network.gdf = self.rules._validate_weight(
            self.network.gdf, raise_error=True
        )

        self.origins = Origins(
            origins,
            id_col=id_col,
            temp_idx_start=max(self.network.nodes.node_id.astype(int)) + 1,
        )

        if destinations is not None:
            self.destinations = Destinations(
                destinations,
                id_col=id_col,
                temp_idx_start=max(self.origins.gdf.temp_idx.astype(int)) + 1,
            )

        else:
            self.destinations = None

        if not self._graph_is_up_to_date() or not self.network._nodes_are_up_to_date():
            self.network._update_nodes_if()

            edges, weights = self._get_edges_and_weights()

            self.graph = self._make_graph(
                edges=edges, weights=weights, directed=self.network._as_directed
            )

            self._add_missing_vertices()

        self._update_wkts()
        self.rules._update_rules()

    def _get_edges_and_weights(self) -> tuple[list[tuple[str, str]], list[float]]:
        """Creates lists of edges and weights which will be used to make the graph.

        Edges and weights between origins and nodes and nodes and destinations are
        also added.
        """
        if self.rules.split_lines:
            self._split_lines()
            self.network._make_node_ids()

        edges = [
            (str(source), str(target))
            for source, target in zip(
                self.network.gdf["source"], self.network.gdf["target"], strict=True
            )
        ]

        weights = list(self.network.gdf[self.rules.weight])

        edges_start, weights_start = self.origins._get_edges_and_weights(
            nodes=self.network.nodes,
            rules=self.rules,
        )
        edges = edges + edges_start
        weights = weights + weights_start

        if self.destinations is None:
            return edges, weights

        edges_end, weights_end = self.destinations._get_edges_and_weights(
            nodes=self.network.nodes,
            rules=self.rules,
        )

        edges = edges + edges_end
        weights = weights + weights_end

        return edges, weights

    def _split_lines(self) -> None:
        if self.destinations is not None:
            points = gdf_concat([self.origins.gdf, self.destinations.gdf])
        else:
            points = self.origins.gdf

        points = points.drop_duplicates("geometry")

        self.network.gdf["meters"] = self.network.gdf.length

        # create an id from before the split to be able to revert the split later
        self.network.gdf["temp_idx__"] = range(len(self.network.gdf))

        lines = split_lines_at_closest_point(
            lines=self.network.gdf,
            points=points,
            max_dist=self.rules.search_tolerance,
        )

        # save the unsplit lines for later
        splitted = lines.loc[lines["splitted"] == 1, "temp_idx__"]
        self.network._not_splitted = self.network.gdf.loc[
            self.network.gdf["temp_idx__"].isin(splitted)
        ]

        self.network.gdf = lines

    def _unsplit_network(self):
        """Remove the splitted lines and add the unsplitted ones."""
        lines = self.network.gdf.loc[self.network.gdf.splitted != 1]
        self.network.gdf = gdf_concat([lines, self.network._not_splitted]).drop(
            "temp_idx__", axis=1
        )
        del self.network._not_splitted

    def _add_missing_vertices(self):
        """Adds the missing points.

        Nodes that had no nodes within the search_tolerance are added to the graph.
        To not get an error when running the distance calculation.
        """
        # TODO: either check if any() beforehand, or add fictional edges before
        # making the graph, to make things faster
        # (this method took 64.660 out of 500 seconds)
        self.graph.add_vertices(
            [
                idx
                for idx in self.origins.gdf["temp_idx"]
                if idx not in self.graph.vs["name"]
            ]
        )
        if self.destinations is not None:
            self.graph.add_vertices(
                [
                    idx
                    for idx in self.destinations.gdf["temp_idx"]
                    if idx not in self.graph.vs["name"]
                ]
            )

    @staticmethod
    def _make_graph(
        edges: list[tuple[str, ...]] | np.ndarray[tuple[str, ...]],
        weights: list[float] | np.ndarray[float],
        directed: bool,
    ) -> Graph:
        """Creates an igraph Graph from a list of edges and weights."""
        assert len(edges) == len(weights)

        graph = igraph.Graph.TupleList(edges, directed=directed)

        graph.es["weight"] = weights
        graph.es["source_target_weight"] = _edge_ids(edges, weights)
        graph.es["edge_tuples"] = edges
        graph.es["source"] = [edge[0] for edge in edges]
        graph.es["target"] = [edge[1] for edge in edges]

        assert min(graph.es["weight"]) >= 0

        return graph

    def _graph_is_up_to_date(self) -> bool:
        """Checks if the network or rules have changed.

        Returns False if the rules of the graphmaking has changed,
        or if the points have changed.
        """
        if not hasattr(self, "graph") or not hasattr(self, "wkts"):
            return False

        if self.rules._rules_have_changed():
            return False

        for points in ["origins", "destinations"]:
            if not hasattr(self.wkts, points):
                return False
            if self._points_have_changed(self[points].gdf, what=points):
                return False

        return True

    def _points_have_changed(self, points: GeoDataFrame, what: str) -> bool:
        """Check if the origins or destinations have changed.

        This method is best stored in the NetworkAnalysis class,
        since the point classes are instantiated each time an analysis is run.
        """
        if self.wkts[what] != [geom.wkt for geom in points.geometry]:
            return True

        if not all(x in self.graph.vs["name"] for x in list(points.temp_idx.values)):
            return True

        return False

    def _update_wkts(self) -> None:
        """Creates a dict of wkt lists.

        This method is run after the graph is created. If the wkts haven't updated
        since the last run, the graph doesn't have to be remade.
        """
        self.wkts = {}

        self.wkts["network"] = [geom.wkt for geom in self.network.gdf.geometry]

        if not hasattr(self, "origins"):
            return

        self.wkts["origins"] = [geom.wkt for geom in self.origins.gdf.geometry]

        if self.destinations is not None:
            self.wkts["destinations"] = [
                geom.wkt for geom in self.destinations.gdf.geometry
            ]

    @staticmethod
    def _sort_breaks(breaks):
        if isinstance(breaks, (list, tuple)):
            breaks = np.array(breaks)

        if isinstance(breaks, str):
            breaks = float(breaks)

        if not isinstance(breaks, (int, float)):
            breaks = np.sort(breaks)

        return breaks

    def __repr__(self) -> str:
        """The print representation."""
        # drop the 'weight_to_nodes_' parameters in the repr of rules to avoid clutter
        rules = (
            f"{self.rules.__class__.__name__}(weight={self.rules.weight}, "
            f"search_tolerance={self.rules.search_tolerance}, "
            f"search_factor={self.rules.search_factor}, "
            f"split_lines={self.rules.split_lines}, "
        )

        # add one 'weight_to_nodes_' parameter if used,
        # else inform that there are more parameters with '...'
        if self.rules.weight_to_nodes_dist:
            x = f"weight_to_nodes_dist={self.rules.weight_to_nodes_dist}"
        elif self.rules.weight_to_nodes_kmh:
            x = f"weight_to_nodes_kmh={self.rules.weight_to_nodes_kmh}"
        elif self.rules.weight_to_nodes_mph:
            x = f"weight_to_nodes_mph={self.rules.weight_to_nodes_mph}"
        else:
            x = "..."

        return (
            f"{self.__class__.__name__}(\n"
            f"    network={self.network.__repr__()},\n"
            f"    rules={rules}{x})\n)"
        )

    def __getitem__(self, item):
        """To be able to write self['origins'] as well as self.origins."""
        return getattr(self, item)

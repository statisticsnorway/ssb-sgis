"""Contains the NetworkAnalysis class.

The class has five analysis methods: od_cost_matrix, get_route, get_k_routes,
get_route_frequencies and service_area.
"""

from copy import copy
from copy import deepcopy
from datetime import datetime
from time import perf_counter
from typing import Any

import igraph
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from igraph import Graph
from pandas import DataFrame
from pandas import MultiIndex
from shapely import force_2d

from ..geopandas_tools.general import _push_geom_col
from ._get_route import _get_k_routes
from ._get_route import _get_route
from ._get_route import _get_route_frequencies
from ._od_cost_matrix import _od_cost_matrix
from ._points import Destinations
from ._points import Origins
from ._service_area import _service_area
from .cutting_lines import split_lines_by_nearest_point
from .network import Network
from .networkanalysisrules import NetworkAnalysisRules


class NetworkAnalysis:
    """Class for doing network analysis.

    Args:
        network: A GeoDataFrame of line geometries.
        rules: The rules for the analysis, either as an instance of
            NetworkAnalysisRules or a dictionary with the parameters
            as keys.
        log: If True (default), a DataFrame with information about each
            analysis run will be stored in the 'log' attribute.
        detailed_log: If True, the log DataFrame will include columns for
            all arguments passed to the analysis method, plus standard deviation and
            percentiles (25th, 50th, 75th) of the weight column in the results.
            Defaults to False.

    The class implements methods for doing network analysis based on
    GeoDataFrames of origin and destination points.

    The 'od_cost_matrix' method is the fastest, and returns a DataFrame with only
    indices and travel costs between each origin-destination pair.

    The 'get_route' method does the same, but also returns the line geometry of the
    routes. 'get_k_routes' can be used to find multiple routes between each OD pair.

    The service_area methods only take a set of origins, and return the lines that
    can be reached within one or more breaks.

    The 'get_route_frequencies' method is a bit different. It returns the individual
    line segments that were visited with an added column for how many times the
    segments were used.

    Attributes:
        network: A Network instance that holds the lines and nodes (points).
        rules: NetworkAnalysisRules instance.
        log: A DataFrame with information about each analysis run.

    Examples:
    ---------
    Read example data.

    >>> import sgis as sg
    >>> roads = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")

    Preparing the lines for directed network analysis.

    >>> connected_roads = sg.get_connected_components(roads).query("connected == 1")

    >>> directed_roads = sg.make_directed_network(
    ...     connected_roads,
    ...     direction_col="oneway",
    ...     direction_vals_bft=("B", "FT", "TF"),
    ...     minute_cols=("drivetime_fw", "drivetime_bw"),
    ...     dropnegative=True,
    ...     dropna=True,
    ... )

    >>> rules = sg.NetworkAnalysisRules(weight="minutes", directed=True)
    >>> nwa = sg.NetworkAnalysis(network=directed_roads, rules=rules, detailed_log=False)
    >>> nwa
    NetworkAnalysis(
        network=Network(6364 km, percent_bidirectional=87),
        rules=NetworkAnalysisRules(weight=minutes, directed=True, search_tolerance=250, search_factor=0, split_lines=True, ...),
        log=True, detailed_log=False,
    )

    Now we're ready for network analysis.

    """

    def __init__(
        self,
        network: GeoDataFrame,
        rules: NetworkAnalysisRules | dict,
        log: bool = True,
        detailed_log: bool = False,
    ) -> None:
        """Initialise NetworkAnalysis instance."""
        if not isinstance(rules, NetworkAnalysisRules):
            rules = NetworkAnalysisRules(**rules)

        if not isinstance(network, Network):
            network = Network(network)

        self.network = network
        self.rules = rules.copy()
        self._log = log
        self.detailed_log = detailed_log

        self._check_if_holes_are_nan()

        self.network.gdf = self.rules._validate_weight(self.network.gdf)

        self._update_wkts()
        self.rules._update_rules()

        if log:
            self.log = DataFrame()

        self._graph_updated_count = 0
        self._k_nearest_points = 50

    def _check_if_holes_are_nan(self) -> None:
        holes_are_nan: str = (
            "Network holes have been filled by straigt lines, but the rows have "
            f"NaN values in the {self.rules.weight!r} column. Either remove NaNs "
            "or fill these values with a numeric value (e.g. 0)."
        )
        if (
            hasattr(self.network.gdf, "hole")
            and len(self.network.gdf.loc[lambda x: x["hole"] == 1])
            and (
                self.network.gdf.loc[lambda x: x["hole"] == 1, self.rules.weight]
                .isna()
                .all()
            )
        ):
            raise ValueError(holes_are_nan)

    def od_cost_matrix(
        self,
        origins: GeoDataFrame,
        destinations: GeoDataFrame,
        *,
        rowwise: bool = False,
        destination_count: int | None = None,
        cutoff: int | float | None = None,
        lines: bool = False,
    ) -> DataFrame | GeoDataFrame:
        """Fast calculation of many-to-many travel costs.

        Finds the the lowest cost (minutes, meters, etc.) from a set of origins to a
        set of destinations. The index of the origins and destinations are used as
        values for the returned columns 'origins' and 'destinations'.

        Args:
            origins: GeoDataFrame of points from where the trips will originate
            destinations: GeoDataFrame of points from where the trips will terminate
            rowwise: if False (default), it will calculate the cost from each
                origins to each destination. If true, it will calculate the cost from
                origin 1 to destination 1, origin 2 to destination 2 and so on.
            destination_count: number of closest destinations to keep for each origin.
                If None (default), all trips will be included. The number of
                destinations might be higher than the destination_count if trips have
                equal cost.
            cutoff: the maximum cost (weight) for the trips. Defaults to None,
                meaning all rows will be included. NaNs will also be removed if cutoff
                is specified.
            lines: if True, returns a geometry column with straight lines between
                origin and destination. Defaults to False.

        Returns:
            A DataFrame with the weight column and the columns 'origin' and
            'destination', containing the indices of the origins and destinations
            GeoDataFrames. If lines is True, also returns a geometry column with
            straight lines between origin and destination.

        Examples:
        ---------
        Create the NetworkAnalysis instance.

        >>> import sgis as sg
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> roads = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")
        >>> directed_roads = sg.get_connected_components(roads).loc[lambda x: x["connected"] == 1].pipe(sg.make_directed_network_norway, dropnegative=True)
        >>> rules = sg.NetworkAnalysisRules(weight="minutes", directed=True)
        >>> nwa = sg.NetworkAnalysis(network=directed_roads, rules=rules, detailed_log=False)

        Create some origin and destination points.

        >>> points = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet")
        >>> origins = points.loc[:9, ["geometry"]]
        >>> print(origins.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
                             geometry
        0  POINT (263122.7 6651184.9)
        1  POINT (272456.1 6653369.5)
        2  POINT (270082.3 6653032.7)
        3  POINT (259804.8 6650339.7)
        4  POINT (272876.2 6652889.1)
        5  POINT (257076.8 6652598.4)
        6  POINT (268426.2 6649159.6)
        7    POINT (261664.6 6650990)
        8  POINT (258505.6 6654539.4)
        9      POINT (259736 6650048)
        <BLANKLINE>
        [10 rows x 1 columns]

        >>> destinations = points.loc[10:19, ["geometry"]]
        >>> print(destinations.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
                              geometry
        10  POINT (265479.7 6652146.1)
        11  POINT (267812.3 6646693.7)
        12    POINT (265624 6646283.7)
        13    POINT (258666.6 6648408)
        14  POINT (266136.8 6641975.4)
        15  POINT (261829.1 6653277.7)
        16  POINT (266549.9 6652070.9)
        17  POINT (257867.8 6651037.6)
        18  POINT (257347.7 6656069.5)
        19  POINT (267030.2 6642630.6)
        <BLANKLINE>
        [10 rows x 1 columns]

        Travel time from ten origins to ten destinations.

        >>> od = nwa.od_cost_matrix(origins, destinations)
        >>> print(od.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
            origin  destination    minutes
        0        0           10   5.614867
        1        1           10  12.098424
        2        2           10  10.776401
        3        3           10  12.359597
        4        4           10  13.736730
        ..     ...          ...        ...
        95       5           19  26.021529
        96       6           19  16.017797
        97       7           19  16.982346
        98       8           19  24.561074
        99       9           19  16.844023
        <BLANKLINE>
        [100 rows x 3 columns]

        Assign aggregated values onto the origins (or destinations).

        >>> origins["minutes_min"] = od.groupby("origin")["minutes"].min()
        >>> origins["minutes_mean"] = od.groupby("origin")["minutes"].mean()
        >>> origins["n_missing"] = len(destinations) - od.groupby("origin")["minutes"].count()
        >>> print(origins.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
                             geometry  minutes_min  minutes_mean  n_missing
        0  POINT (263122.7 6651184.9)     5.614867     11.711857          0
        1  POINT (272456.1 6653369.5)     8.338750     17.323004          0
        2  POINT (270082.3 6653032.7)     7.423471     16.285906          0
        3  POINT (259804.8 6650339.7)     4.983162     12.202607          0
        4  POINT (272876.2 6652889.1)     9.977056     18.961310          0
        5  POINT (257076.8 6652598.4)    10.316722     17.903136          0
        6  POINT (268426.2 6649159.6)     7.239519     15.471286          0
        7    POINT (261664.6 6650990)     5.256121     12.346193          0
        8  POINT (258505.6 6654539.4)     4.992840     15.739859          0
        9      POINT (259736 6650048)     4.808400     12.308181          0
        <BLANKLINE>
        [10 rows x 4 columns]

        Join the results onto the 'origins' via the index.

        >>> joined = origins.join(od.set_index("origin"))
        >>> print(joined.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
                              geometry  minutes_min  minutes_mean  n_missing  destination    minutes
        0   POINT (263122.7 6651184.9)     5.614867     11.711857          0           10   5.614867
        0   POINT (263122.7 6651184.9)     5.614867     11.711857          0           11  13.491910
        0   POINT (263122.7 6651184.9)     5.614867     11.711857          0           12  11.642500
        0   POINT (263122.7 6651184.9)     5.614867     11.711857          0           13  12.996674
        0   POINT (263122.7 6651184.9)     5.614867     11.711857          0           14  14.862544
        ..                         ...          ...           ...        ...          ...        ...
        9       POINT (259736 6650048)     4.808400     12.308181          0           15   7.935885
        9       POINT (259736 6650048)     4.808400     12.308181          0           16  13.470447
        9       POINT (259736 6650048)     4.808400     12.308181          0           17   4.808400
        9       POINT (259736 6650048)     4.808400     12.308181          0           18  17.019517
        9       POINT (259736 6650048)     4.808400     12.308181          0           19  16.844023
        <BLANKLINE>
        [100 rows x 6 columns]

        Keep only travel times of 10 minutes or less. This is the same as using the
        cutoff parameter.

        >>> ten_min_or_less = od.loc[od.minutes <= 10]
        >>> joined = origins.join(ten_min_or_less.set_index("origin"))
        >>> print(joined.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
                              geometry  minutes_min  minutes_mean  n_missing  destination   minutes
        0   POINT (263122.7 6651184.9)     5.614867     11.711857          0         10.0  5.614867
        0   POINT (263122.7 6651184.9)     5.614867     11.711857          0         15.0  5.913632
        0   POINT (263122.7 6651184.9)     5.614867     11.711857          0         16.0  6.199637
        1   POINT (272456.1 6653369.5)     8.338750     17.323004          0         16.0  8.338750
        2   POINT (270082.3 6653032.7)     7.423471     16.285906          0         16.0  7.423471
        ..                         ...          ...           ...        ...          ...       ...
        8   POINT (258505.6 6654539.4)     4.992840     15.739859          0         15.0  7.980121
        8   POINT (258505.6 6654539.4)     4.992840     15.739859          0         18.0  4.992840
        9       POINT (259736 6650048)     4.808400     12.308181          0         13.0  5.660898
        9       POINT (259736 6650048)     4.808400     12.308181          0         15.0  7.935885
        9       POINT (259736 6650048)     4.808400     12.308181          0         17.0  4.808400
        <BLANKLINE>
        [20 rows x 6 columns]

        Keep the three fastest times from each origin. This is the same as using the
        destination_count parameter.

        >>> three_fastest = od.loc[od.groupby("origin")["minutes"].rank() <= 3]
        >>> joined = origins.join(three_fastest.set_index("origin"))
        >>> print(joined.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
                              geometry  minutes_min  minutes_mean  n_missing  destination    minutes
        0   POINT (263122.7 6651184.9)     5.614867     11.711857          0           10   5.614867
        0   POINT (263122.7 6651184.9)     5.614867     11.711857          0           15   5.913632
        0   POINT (263122.7 6651184.9)     5.614867     11.711857          0           16   6.199637
        1   POINT (272456.1 6653369.5)     8.338750     17.323004          0           10  12.098424
        1   POINT (272456.1 6653369.5)     8.338750     17.323004          0           11  13.974151
        ..                         ...          ...           ...        ...          ...        ...
        8   POINT (258505.6 6654539.4)     4.992840     15.739859          0           17  10.804265
        8   POINT (258505.6 6654539.4)     4.992840     15.739859          0           18   4.992840
        9       POINT (259736 6650048)     4.808400     12.308181          0           13   5.660898
        9       POINT (259736 6650048)     4.808400     12.308181          0           15   7.935885
        9       POINT (259736 6650048)     4.808400     12.308181          0           17   4.808400
        <BLANKLINE>
        [30 rows x 6 columns]

        Use set_index to use column as identifier insted of the index.

        >>> origins["areacode"] = np.random.choice(["0301", "3401"], len(origins))
        >>> od = nwa.od_cost_matrix(
        ...    origins.set_index("areacode"),
        ...    destinations
        ... )
        >>> print(od.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
           origin  destination    minutes
        0    0301           10   5.614867
        1    3401           10  12.098424
        2    3401           10  10.776401
        3    0301           10  12.359597
        4    3401           10  13.736730
        ..    ...          ...        ...
        95   3401           19  26.021529
        96   3401           19  16.017797
        97   3401           19  16.982346
        98   3401           19  24.561074
        99   3401           19  16.844023
        <BLANKLINE>
        [100 rows x 3 columns]

        Pair ten origins with ten destinations rowwise.

        >>> rowwise_points = points.iloc[:10]
        >>> points_reversed = rowwise_points.iloc[::-1]
        >>> od = nwa.od_cost_matrix(rowwise_points, points_reversed, rowwise=True)
        >>> print(od.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
           origin  destination    minutes
        0       0            9   8.275748
        1       1            8  20.526351
        2       2            7  14.464923
        3       3            6  17.637484
        4       4            5  24.663967
        5       5            4  25.611493
        6       6            3  17.687899
        7       7            2  14.507884
        8       8            1  21.634552
        9       9            0   8.189336
        <BLANKLINE>
        [10 rows x 3 columns]

        """
        if self._log:
            time_ = perf_counter()

        self._prepare_network_analysis(origins, destinations, rowwise)

        ori = self.origins.gdf.set_index("temp_idx")
        des = self.destinations.gdf.set_index("temp_idx")
        results = _od_cost_matrix(
            graph=self.graph,
            origins=ori,
            destinations=des,
            weight=self.rules.weight,
            lines=lines,
            rowwise=rowwise,
        )

        if cutoff is not None:
            results = results.loc[results[self.rules.weight] <= cutoff]

        if destination_count:
            results = results.loc[
                results.groupby("origin")[self.rules.weight].rank() <= destination_count
            ]

        results["origin"] = results["origin"].map(self.origins.idx_dict)
        results["destination"] = results["destination"].map(self.destinations.idx_dict)

        if lines:
            results = _push_geom_col(results)

        # if self.rules.split_lines:
        #     self._unsplit_network()

        if self._log:
            minutes_elapsed = round((perf_counter() - time_) / 60, 1)
            self._runlog(
                "od_cost_matrix",
                results,
                minutes_elapsed,
                lines=lines,
                rowwise=rowwise,
            )

        return results

    def get_route_frequencies(
        self,
        origins: GeoDataFrame,
        destinations: GeoDataFrame,
        weight_df: DataFrame | None = None,
        default_weight: int | float | None = None,
        rowwise: bool = False,
        strict: bool = False,
        frequency_col: str = "frequency",
        n_jobs: int | None = None,
    ) -> GeoDataFrame:
        """Finds the number of times each line segment was visited in all trips.

        Finds the route with the lowest cost (minutes, meters, etc.) from a set of
        origins to a set of destinations and summarises the number of times each
        segment was used. The aggregation is done on the line indices, which is much
        faster than getting the geometries and then dissolving.

        The trip frequencies can be weighted (multiplied) based on 'weight_df'. See
        example below.

        Args:
            origins: GeoDataFrame of points from where the routes will originate.
            destinations: GeoDataFrame of points from where the routes will terminate.
            weight_df: A long formated DataFrame where each row contains the indices of
                an origin-destination pair and the number to multiply the frequency for
                this route by. The DataFrame can either contain three columns (origin
                index, destination index and weight. In that order) or only a weight
                column and a MultiIndex where level 0 is origin index and level 1 is
                destination index.
            default_weight: If set, OD pairs not represented in 'weight_df'
                will be given a default weight value.
            rowwise: if False (default), it will calculate the cost from each
                origins to each destination. If true, it will calculate the cost from
                origin 1 to destination 1, origin 2 to destination 2 and so on.
            strict: If True, all OD pairs must be in weigth_df if specified. Defaults
                to False.
            frequency_col: Name of column with the number of times each road was
                visited. Defaults to 'frequency'.
            n_jobs: Number of parallell jobs.

        Returns:
            A GeoDataFrame with all line segments that were visited at least once,
            with a column with the number of times the line segment was used in the
            individual routes.

        Note:
            The resulting lines will keep all columns of the 'gdf' of the Network.

        Raises:
            ValueError: If weight_df is not a DataFrame with one or three columns
                that contain weights and all indices of 'origins' and 'destinations'.

        Examples:
        ---------
        Create the NetworkAnalysis instance.

        >>> import sgis as sg
        >>> import pandas as pd
        >>> roads = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")
        >>> directed_roads = sg.get_connected_components(roads).loc[lambda x: x["connected"] == 1].pipe(sg.make_directed_network_norway, dropnegative=True)
        >>> rules = sg.NetworkAnalysisRules(weight="minutes", directed=True)
        >>> nwa = sg.NetworkAnalysis(network=directed_roads, rules=rules, detailed_log=False)

        Get some points.

        >>> points = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet")
        >>> origins = points.iloc[:3]
        >>> destinations = points.iloc[3:6]

        Get number of times each road was visited for trips from three to three points.

        >>> frequencies = nwa.get_route_frequencies(origins, destinations)
        >>> frequencies[["source", "target", "frequency", "geometry"]]
               source  ...                                           geometry
        152415  74650  ...  LINESTRING (270320.2 6653034.3, 270320.5 6653030)
        152102  74415  ...  LINESTRING (264579.835 6651954.573, 264580.014...
        152100  74414  ...  LINESTRING (264545.722 6651923.505, 264549.637...
        152093  74408  ...  LINESTRING (264571.192 6651978.362, 264563.3 6...
        152092  74407  ...  LINESTRING (264571.803 6651977.604, 264571.192...
        ...       ...  ...                                                ...
        155281  76617  ...  LINESTRING (260732.767 6651321.461, 260719.064...
        155282  76618  ...  LINESTRING (260719.064 6651311.891, 260700.041...
        155283  76619  ...  LINESTRING (260700.041 6651299.939, 260681.4 6...
        155284  43949  ...  LINESTRING (260681.4 6651291.3, 260664.9 66512...
        155285  76620  ...  LINESTRING (260657.9 6651280.6, 260629.8 66512...
        <BLANKLINE>
        [987 rows x 4 columns]

        The frequencies can be weighted for each origin-destination pair by specifying
        'weight_df'. This can be a DataFrame with three columns, where the first two
        contain the indices of the origin and destination (in that order), and the
        third the number to multiply the frequency by. 'weight_df' can also be a
        DataFrame with a 2-leveled MultiIndex, where level 0 is the origin index and
        level 1 is the destination.

        Constructing a DataFrame with all od-pair combinations and give all rows a
        weight of 10.

        >>> od_pairs = pd.MultiIndex.from_product(
        ...     [origins.index, destinations.index], names=["origin", "destination"]
        ... )
        >>> weight_df = pd.DataFrame(index=od_pairs).reset_index()
        >>> weight_df["weight"] = 10
        >>> print(weight_df.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
           origin  destination  weight
        0       0            3      10
        1       0            4      10
        2       0            5      10
        3       1            3      10
        4       1            4      10
        5       1            5      10
        6       2            3      10
        7       2            4      10
        8       2            5      10
        <BLANKLINE>
        [9 rows x 3 columns]

        All frequencies will now be multiplied by 10.

        >>> frequencies = nwa.get_route_frequencies(origins, destinations, weight_df=weight_df)
        >>> frequencies[["source", "target", "frequency", "geometry"]]
               source  ...                                           geometry
        152415  74650  ...  LINESTRING (270320.2 6653034.3, 270320.5 6653030)
        152102  74415  ...  LINESTRING (264579.835 6651954.573, 264580.014...
        152100  74414  ...  LINESTRING (264545.722 6651923.505, 264549.637...
        152093  74408  ...  LINESTRING (264571.192 6651978.362, 264563.3 6...
        152092  74407  ...  LINESTRING (264571.803 6651977.604, 264571.192...
        ...       ...  ...                                                ...
        155281  76617  ...  LINESTRING (260732.767 6651321.461, 260719.064...
        155282  76618  ...  LINESTRING (260719.064 6651311.891, 260700.041...
        155283  76619  ...  LINESTRING (260700.041 6651299.939, 260681.4 6...
        155284  43949  ...  LINESTRING (260681.4 6651291.3, 260664.9 66512...
        155285  76620  ...  LINESTRING (260657.9 6651280.6, 260629.8 66512...
        <BLANKLINE>
        [987 rows x 4 columns]

        'weight_df' can also be a DataFrame with one column (the weight) and a
        MultiIndex.

        >>> weight_df = pd.DataFrame(index=od_pairs)
        >>> weight_df["weight"] = 10
        >>> weight_df
                            weight
        origin destination
        0      3                10
               4                10
               5                10
        1      3                10
               4                10
               5                10
        2      3                10
               4                10
               5                10
        """
        if self._log:
            time_ = perf_counter()

        if weight_df is not None:
            weight_df: DataFrame = self._prepare_weight_df(weight_df)
            od_pairs: MultiIndex = self._create_od_pairs(
                origins, destinations, rowwise=rowwise
            )
            self._make_sure_unique(weight_df, od_pairs)

            weights_mapped = od_pairs.map(weight_df.iloc[:, 0])
            if default_weight:
                if not weight_df.index.isin(od_pairs).all():
                    raise ValueError(
                        "All origin-destination indices in 'weight_df' must "
                        "be in 'origins' and 'destinations'."
                    )
                weights_mapped = weights_mapped.fillna(default_weight)
            elif strict:
                self._make_sure_index_match(weight_df, od_pairs)
            weight_df = DataFrame(index=od_pairs)
            weight_df["weight"] = weights_mapped

        self._prepare_network_analysis(origins, destinations, rowwise)

        if weight_df is not None:
            # map to temporary ids
            ori_idx_mapper = {v: k for k, v in self.origins.idx_dict.items()}
            des_idx_mapper = {v: k for k, v in self.destinations.idx_dict.items()}

            def multiindex_mapper(x: tuple[int, int]) -> tuple[int, int]:
                return (
                    ori_idx_mapper.get(x[0]),
                    des_idx_mapper.get(x[1]),
                )

            weight_df.index = weight_df.index.map(multiindex_mapper)
        else:
            od_pairs = self._create_od_pairs(
                self.origins.gdf.set_index("temp_idx"),
                self.destinations.gdf.set_index("temp_idx"),
                rowwise=rowwise,
            )
            weight_df = DataFrame(index=od_pairs)
            weight_df["weight"] = 1

        results = _get_route_frequencies(
            graph=self.graph,
            roads=self.network.gdf,
            weight_df=weight_df,
            n_jobs=n_jobs,
        )

        if isinstance(results, GeoDataFrame):
            results = _push_geom_col(results)

        results = results.rename(columns={"frequency": frequency_col}).sort_values(
            frequency_col
        )

        results.geometry = force_2d(results.geometry)

        # if self.rules.split_lines:
        #     self._unsplit_network()

        if self._log:
            minutes_elapsed = round((perf_counter() - time_) / 60, 1)
            self._runlog(
                "get_route_frequencies",
                results,
                minutes_elapsed,
            )

        return results

    def get_route(
        self,
        origins: GeoDataFrame,
        destinations: GeoDataFrame,
        *,
        rowwise: bool = False,
        destination_count: int | None = None,
        cutoff: int | float | None = None,
        n_jobs: int | None = None,
    ) -> GeoDataFrame:
        """Returns the geometry of the low-cost route between origins and destinations.

        Finds the route with the lowest cost (minutes, meters, etc.) from a set of
        origins to a set of destinations. If the weight is meters, the shortest route
        will be found. If the weight is minutes, the fastest route will be found.

        Args:
            origins: GeoDataFrame of points from where the routes will originate
            destinations: GeoDataFrame of points from where the routes will terminate.
            rowwise: if False (default), it will calculate the cost from each
                origins to each destination. If true, it will calculate the cost from
                origin 1 to destination 1, origin 2 to destination 2 and so on.
            destination_count: number of closest destinations to keep for each origin.
                If None (default), all trips will be included. The number of
                destinations might be higher than the destination_count if trips have
                equal cost.
            cutoff: the maximum cost (weight) for the trips. Defaults to None,
                meaning all rows will be included. NaNs will also be removed if cutoff
                is specified.
            n_jobs: Number of parallell jobs.

        Returns:
            A DataFrame with the geometry of the routes between origin and destination.
            Also returns a weight column and the columns 'origin' and 'destination',
            containing the indices of the origins and destinations GeoDataFrames.

        Examples:
        ---------
        Create the NetworkAnalysis instance.

        >>> import sgis as sg
        >>> roads = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")
        >>> directed_roads = sg.get_connected_components(roads).loc[lambda x: x["connected"] == 1].pipe(sg.make_directed_network_norway, dropnegative=True)
        >>> rules = sg.NetworkAnalysisRules(weight="minutes", directed=True)
        >>> nwa = sg.NetworkAnalysis(network=directed_roads, rules=rules, detailed_log=False)

        Get routes from one origin to ten destination points.

        >>> points = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet")

        >>> routes = nwa.get_route(points.iloc[[0]], points.iloc[:10])
        >>> print(routes.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
           origin  destination    minutes                                           geometry
        0       0            1  13.209125  MULTILINESTRING ((263167.581 6651182.119, 2631...
        1       0            2  10.994973  MULTILINESTRING ((263167.581 6651182.119, 2631...
        2       0            3   7.749668  MULTILINESTRING ((263167.581 6651182.119, 2631...
        3       0            4  14.849464  MULTILINESTRING ((263167.581 6651182.119, 2631...
        4       0            5  14.391621  MULTILINESTRING ((263167.581 6651182.119, 2631...
        5       0            6  13.071382  MULTILINESTRING ((263167.581 6651182.119, 2631...
        6       0            7   4.439742  MULTILINESTRING ((263167.581 6651182.119, 2631...
        7       0            8  13.673451  MULTILINESTRING ((263167.581 6651182.119, 2631...
        8       0            9   8.275748  MULTILINESTRING ((263167.581 6651182.119, 2631...
        <BLANKLINE>
        [9 rows x 4 columns]
        """
        if self._log:
            time_ = perf_counter()

        self._prepare_network_analysis(origins, destinations, rowwise)

        od_pairs = self._create_od_pairs(
            self.origins.gdf.set_index("temp_idx"),
            self.destinations.gdf.set_index("temp_idx"),
            rowwise=rowwise,
        )

        results = _get_route(
            graph=self.graph,
            weight=self.rules.weight,
            roads=self.network.gdf,
            od_pairs=od_pairs,
            n_jobs=n_jobs,
        )

        if cutoff is not None:
            results = results.loc[results[self.rules.weight] <= cutoff]

        if destination_count:
            results = results.loc[
                results.groupby("origin")[self.rules.weight].rank() <= destination_count
            ]

        results["origin"] = results["origin"].map(self.origins.idx_dict)
        results["destination"] = results["destination"].map(self.destinations.idx_dict)

        # if self.rules.split_lines:
        #     self._unsplit_network()

        if self._log:
            minutes_elapsed = round((perf_counter() - time_) / 60, 1)
            self._runlog(
                "get_route",
                results,
                minutes_elapsed,
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
        rowwise: bool = False,
        destination_count: int | None = None,
        cutoff: int | float | None = None,
    ) -> GeoDataFrame:
        r"""Returns the geometry of 1 or more routes between origins and destinations.

        Finds the route with the lowest cost (minutes, meters, etc.) from a set of
        origins to a set of destinations. Then the middle part of the route is removed
        from the graph the new low-cost path is found. Repeats k times. If k=1, it is
        identical to the get_route method.

        Args:
            origins: GeoDataFrame of points from where the routes will originate.
            destinations: GeoDataFrame of points from where the routes will terminate.
            k: the number of low-cost routes to find.
            drop_middle_percent: how many percent of the middle part of the routes
                that should be removed from the graph before the next k route is
                calculated. If set to 100, only the median edge will be removed.
                If set to 0, all but the first and last edge will be removed. The
                graph is copied for each od pair.
            rowwise: if False (default), it will calculate the cost from each
                origins to each destination. If true, it will calculate the cost from
                origin 1 to destination 1, origin 2 to destination 2 and so on.
            destination_count: number of closest destinations to keep for each origin.
                If None (default), all trips will be included. The number of
                destinations might be higher than the destination_count if trips have
                equal cost.
            cutoff: the maximum cost (weight) for the trips. Defaults to None,
                meaning all rows will be included. NaNs will also be removed if cutoff
                is specified.

        Returns:
            A DataFrame with the geometry of the k routes between origin and
            destination. Also returns the column 'k', a weight column and the columns
            'origin' and 'destination', containing the indices of the origins and
            destinations GeoDataFrames.

        Note:
            How many percent of the route to drop from the graph, will determine how
            many k routes will be found. If 100 percent of the route is dropped, it is
            very hard to find more than one path for each OD pair.
            If 'drop_middle_percent' is 1, the resulting routes might be very similar,
            depending on the layout of the network.

        Raises:
            ValueError: if drop_middle_percent is not between 0 and 100.

        Examples:
        ---------
        Create the NetworkAnalysis instance.

        >>> import sgis as sg
        >>> roads = sg.read_parquet_url('https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet')
        >>> directed_roads = sg.get_connected_components(roads).loc[lambda x: x["connected"] == 1].pipe(sg.make_directed_network_norway, dropnegative=True)
        >>> rules = sg.NetworkAnalysisRules(weight="minutes", directed=True)
        >>> nwa = sg.NetworkAnalysis(network=directed_roads, rules=rules, detailed_log=False)

        Getting 10 fastest routes from one point to another point.

        >>> points = sg.read_parquet_url('https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet')
        >>> point1 = points.iloc[[0]]
        >>> point2 = points.iloc[[1]]

        >>> k_routes = nwa.get_k_routes(
        ...             point1,
        ...             point2,
        ...             k=10,
        ...             drop_middle_percent=1
        ... )
        >>> print(k_routes.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
           origin  destination    minutes   k                                           geometry
        0       0            1  13.209125   1  MULTILINESTRING Z ((263167.581 6651182.119 NaN...
        1       0            1  14.253618   2  MULTILINESTRING Z ((263167.581 6651182.119 NaN...
        2       0            1  14.407402   3  MULTILINESTRING Z ((263167.581 6651182.119 NaN...
        3       0            1  15.066977   4  MULTILINESTRING Z ((263167.581 6651182.119 NaN...
        4       0            1  15.131888   5  MULTILINESTRING Z ((263167.581 6651182.119 NaN...
        5       0            1  15.593228   6  MULTILINESTRING Z ((263167.581 6651182.119 NaN...
        6       0            1  16.386565   7  MULTILINESTRING Z ((263167.581 6651182.119 NaN...
        7       0            1  16.653277   8  MULTILINESTRING Z ((263167.581 6651182.119 NaN...
        8       0            1  16.682548   9  MULTILINESTRING Z ((263167.581 6651182.119 NaN...
        9       0            1  17.956501  10  MULTILINESTRING Z ((263167.581 6651182.119 NaN...
        <BLANKLINE>
        [10 rows x 5 columns]

        We got all 10 routes because only the middle 1 percent of the routes are
        removed in each iteration. Let's compare with dropping middle 50 and middle
        100 percent.

        >>> k_routes = nwa.get_k_routes(
        ...             point1,
        ...             point2,
        ...             k=10,
        ...             drop_middle_percent=50
        ...         )
        >>> print(k_routes.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
           origin  destination    minutes  k                                           geometry
        0       0            1  13.209125  1  MULTILINESTRING Z ((263167.581 6651182.119 NaN...
        1       0            1  14.445367  2  MULTILINESTRING Z ((263167.581 6651182.119 NaN...
        2       0            1  20.308589  3  MULTILINESTRING Z ((263167.581 6651182.119 NaN...
        3       0            1  23.676072  4  MULTILINESTRING Z ((263167.581 6651182.119 NaN...
        <BLANKLINE>
        [4 rows x 5 columns]

        >>> k_routes = nwa.get_k_routes(
        ...             point1,
        ...             point2,
        ...             k=10,
        ...             drop_middle_percent=100
        ...         )
        >>> print(k_routes.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
           origin  destination    minutes  k                                           geometry
        0       0            1  13.209125  1  MULTILINESTRING Z ((263167.581 6651182.119 NaN...
        <BLANKLINE>
        [1 rows x 5 columns]

        """
        if not 0 <= drop_middle_percent <= 100:
            raise ValueError("'drop_middle_percent' should be between 0 and 100")

        if self._log:
            time_ = perf_counter()

        self._prepare_network_analysis(origins, destinations, rowwise)

        od_pairs = self._create_od_pairs(
            self.origins.gdf.set_index("temp_idx"),
            self.destinations.gdf.set_index("temp_idx"),
            rowwise=rowwise,
        )

        results = _get_k_routes(
            graph=self.graph,
            weight=self.rules.weight,
            roads=self.network.gdf,
            od_pairs=od_pairs,
            k=k,
            drop_middle_percent=drop_middle_percent,
        )

        if cutoff is not None:
            results = results.loc[results[self.rules.weight] <= cutoff]

        if destination_count:
            results = results.loc[
                results.groupby("origin")[self.rules.weight].rank() <= destination_count
            ]

        results["origin"] = results["origin"].map(self.origins.idx_dict)
        results["destination"] = results["destination"].map(self.destinations.idx_dict)

        if isinstance(results, GeoDataFrame):
            results = _push_geom_col(results)

        # if self.rules.split_lines:
        #     self._unsplit_network()

        if self._log:
            minutes_elapsed = round((perf_counter() - time_) / 60, 1)
            self._runlog(
                "get_k_routes",
                results,
                minutes_elapsed,
                rowwise=rowwise,
            )

        return results

    def service_area(
        self,
        origins: GeoDataFrame,
        breaks: int | float | tuple[int | float],
        *,
        dissolve: bool = True,
    ) -> GeoDataFrame:
        """Returns the lines that can be reached within breaks (weight values).

        It finds all the network lines that can be reached within each break. Lines
        that are only partly within the break will not be included. The index of the
        origins is used as values in the 'origins' column.

        Args:
            origins: GeoDataFrame of points from where the service areas will
                originate
            breaks: one or more integers or floats which will be the
                maximum weight for the service areas. Calculates multiple areas for
                each origins if multiple breaks.
            dissolve: If True (default), each service area will be dissolved into
                one long multilinestring. If False, the individual line segments will
                be returned.

        Returns:
            A GeoDataFrame with one row per break per origin, with the origin index and
            a dissolved line geometry. If dissolve is False, it will return each line
            that is part of the service area.

        See Also:
            precice_service_area: Equivelent method where lines are also cut to get
            precice results.

        Examples:
        ---------
        Create the NetworkAnalysis instance.

        >>> import sgis as sg
        >>> from shapely.geometry import LineString
        >>> roads = sg.to_gdf([
        ...     LineString([(0, 0), (1000, 0)]),
        ...     LineString([(1000, 0), (2000, 0)]),
        ...     LineString([(2000, 0), (3000, 0)]),
        ... ], crs=25833)
        >>> roads["minutes"] = 5.0
        >>> rules = sg.NetworkAnalysisRules(weight="minutes", directed=False)
        >>> nwa = sg.NetworkAnalysis(network=roads, rules=rules)

        10 minute service area for three origin points.

        >>> points = sg.to_gdf([(0, 0), (500, 0), (1000, 0)], crs=25833)
        >>> service_areas = nwa.service_area(
        ...         points.loc[:2],
        ...         breaks=10,
        ... )
        >>> print(service_areas.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
           origin  minutes                                           geometry
        0       0       10  MULTILINESTRING ((0 0, 500 0), (500 0, 1000 0)...
        1       1       10  MULTILINESTRING ((0 0, 500 0), (500 0, 1000 0)...
        2       2       10  MULTILINESTRING ((2000 0, 3000 0), (0 0, 500 0...
        <BLANKLINE>
        [3 rows x 3 columns]

        Service areas of 5, 10 and 15 minutes from two origin points.

        >>> service_areas = nwa.service_area(
        ...         points.iloc[:2],
        ...         breaks=[5, 10, 15],
        ... )
        >>> print(service_areas.to_string(max_rows=10, max_colwidth=50, show_dimensions=True))
           origin  minutes                                         geometry
        0       0        5  MULTILINESTRING ((0 0, 500 0), (500 0, 1000 0))
        1       0       10                      LINESTRING (1000 0, 2000 0)
        2       0       15                      LINESTRING (2000 0, 3000 0)
        3       1        5  MULTILINESTRING ((0 0, 500 0), (500 0, 1000 0))
        4       1       10                      LINESTRING (1000 0, 2000 0)
        5       1       15                      LINESTRING (2000 0, 3000 0)
        <BLANKLINE>
        [6 rows x 3 columns]
        """
        if self._log:
            time_ = perf_counter()

        self._prepare_network_analysis(origins)

        # sort the breaks as an np.ndarray
        breaks = self._sort_breaks(breaks)

        results = _service_area(
            graph=self.graph,
            origins=self.origins.gdf,
            breaks=breaks,
            weight=self.rules.weight,
            lines=self.network.gdf,
            nodes=self.network.nodes,
            directed=self.rules.directed,
            precice=False,
        )

        if not all(results.geometry.isna()):
            results = results.drop_duplicates(["src_tgt_wt", "origin"])

            if dissolve:
                results = results.dissolve(by=["origin", self.rules.weight]).loc[
                    :, [results.geometry.name]
                ]

            results = results.reset_index()

            # add missing rows as NaNs
            missing = self.origins.gdf.loc[
                ~self.origins.gdf["temp_idx"].isin(results["origin"])
            ].rename(columns={"temp_idx": "origin"})[["origin"]]

            if len(missing):
                missing[results.geometry.name] = pd.NA
                results = pd.concat([results, missing], ignore_index=True)

            results["origin"] = results["origin"].map(self.origins.idx_dict)

            results = _push_geom_col(results)

        # if self.rules.split_lines:
        #     self._unsplit_network()

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

    def precice_service_area(
        self,
        origins: GeoDataFrame,
        breaks: int | float | tuple[int | float],
        *,
        dissolve: bool = True,
    ) -> GeoDataFrame:
        """Precice, but slow version of the service_area method.

        It finds all the network lines that can be reached within each break. Lines
        that are partly within the break will be split at the point where the weight
        value is exactly correct. Note that this takes more time than the regular
        'service_area' method.

        Args:
            origins: GeoDataFrame of points from where the service areas will
                originate
            breaks: one or more integers or floats which will be the
                maximum weight for the service areas. Calculates multiple areas for
                each origins if multiple breaks.
            dissolve: If True (default), each service area will be dissolved into
                one long multilinestring. If False, the individual line segments will
                be returned.

        Returns:
            A GeoDataFrame with one row per break per origin, with a dissolved line
            geometry. If dissolve is False, it will return all the columns of the
            network.gdf as well.

        See Also:
            service_area: Faster method where lines are not cut to get precice results.

        Examples:
        ---------
        Create the NetworkAnalysis instance.

        >>> import sgis as sg
        >>> from shapely.geometry import LineString
        >>> roads = sg.to_gdf([
        ...     LineString([(0, 0), (1000, 0)]),
        ...     LineString([(1000, 0), (2000, 0)]),
        ...     LineString([(2000, 0), (3000, 0)]),
        ... ], crs=25833)
        >>> roads["minutes"] = 5.0
        >>> rules = sg.NetworkAnalysisRules(weight="minutes", directed=False)
        >>> nwa = sg.NetworkAnalysis(network=roads, rules=rules)

        10 minute service area for one origin point.

        >>> points = sg.to_gdf([(0, 0), (500, 0), (1000, 0)], crs=25833)

        >>> sa = nwa.precice_service_area(
        ...         points.iloc[[0]],
        ...         breaks=10,
        ...     )
        >>> sa
           minutes                                           geometry
        0       10  MULTILINESTRING ((1000 0, 1005 0), (1005 0, 19...

        Service areas of 5, 10 and 15 minutes from two origin points.

        >>> sa = nwa.precice_service_area(
        ...         points.iloc[:2],
        ...         breaks=[5, 10, 15],
        ...     )
        >>> sa
           minutes                                           geometry
        0        5  MULTILINESTRING ((0 0, 2.5 0), (2.5 0, 5 0), (...
        1       10  MULTILINESTRING ((1000 0, 1005 0), (1005 0, 19...
        2       15  MULTILINESTRING ((1000 0, 1005 0), (1005 0, 10...
        3        5  MULTILINESTRING ((0 0, 2.5 0), (2.5 0, 5 0), (...
        4       10  MULTILINESTRING ((1000 0, 1002.5 0), (1002.5 0...
        5       15  MULTILINESTRING ((1000 0, 1007.5 0), (1007.5 0...
        """
        if self._log:
            time_ = perf_counter()

        self._prepare_network_analysis(origins)

        # sort the breaks as an np.ndarray
        breaks = self._sort_breaks(breaks)

        results = _service_area(
            graph=self.graph,
            origins=self.origins.gdf,
            breaks=breaks,
            weight=self.rules.weight,
            lines=self.network.gdf,
            nodes=self.network.nodes,
            directed=self.rules.directed,
            precice=True,
        )

        if not all(results.geometry.isna()):
            if dissolve:
                results = results.dissolve(by=["origin", self.rules.weight]).loc[
                    :, [results.geometry.name]
                ]
            else:
                results = results.dissolve(
                    by=["src_tgt_wt", "origin", self.rules.weight]
                )

            results = results.reset_index()

            # add missing rows as NaNs
            missing = self.origins.gdf.loc[
                ~self.origins.gdf["temp_idx"].isin(results["origin"])
            ].rename(columns={"temp_idx": "origin"})[["origin"]]

            if len(missing):
                missing[results.geometry.name] = pd.NA
                results = pd.concat([results, missing], ignore_index=True)

            results["origin"] = results["origin"].map(self.origins.idx_dict)
            results = results.drop("origin", axis=1)

            results = _push_geom_col(results)

        # if self.rules.split_lines:
        #     self._unsplit_network()

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

    @staticmethod
    def _prepare_weight_df(weight_df: DataFrame) -> DataFrame:
        """Copy weight_df, convert to MultiIndex (if needed), check if correct shape.

        The weight_df needs to have a very specific shape and index. If a 3-columned df
        is given, convert the first two to a MultiIndex.

        """
        error_message = (
            "'weight_df' should be a DataFrame with the columns "
            "'origin', 'destination' and 'weight', where the first "
            "two contain the indices of the origins and destinations "
            "and the weight column contains the number to multiply "
            "the trip frequency for this origin-destination pair."
        )

        if not isinstance(weight_df, (DataFrame | pd.Series)):
            raise ValueError(error_message)

        if isinstance(weight_df, pd.Series):
            weight_df = weight_df.to_frame()

        weight_df = weight_df.copy()

        if len(weight_df.columns) == 3:
            weight_df = weight_df.set_index(list(weight_df.columns[:2]))

        if len(weight_df.columns) != 1 and isinstance(weight_df.index, MultiIndex):
            raise ValueError(error_message)

        return weight_df

    @staticmethod
    def _make_sure_unique(weight_df: DataFrame, od_pairs: MultiIndex) -> None:
        """It's nesseccary with unique index when using weight_df."""
        if not weight_df.index.is_unique:
            raise ValueError("'weight_df' must contain only unique OD combinations.")

        if not od_pairs.is_unique:
            raise ValueError(
                "'origins' and 'destinations must contain only unique "
                "indices when weight_df is specified."
            )

    @staticmethod
    def _make_sure_index_match(
        weight_df: DataFrame,
        od_pairs: MultiIndex,
    ) -> None:
        """Make sure this index matches the index of origins and destinations."""
        if not od_pairs.isin(weight_df.index).all():
            if not od_pairs.isin(weight_df.index).any():
                raise ValueError(
                    "None of the origin-destination pair indices are in 'weight_df'."
                )
            raise ValueError(
                "Not all origin-destination pair indices are in 'weight_df'."
            )

    @staticmethod
    def _create_od_pairs(
        origins: GeoDataFrame, destinations: GeoDataFrame, rowwise: bool
    ) -> MultiIndex:
        """Get all OD combinaions without identical origin-destination geometry.

        Returns a MultiIndex to be iterated over in get_route, get_k_routes and
        get_route_frequencies. In get_route_frequencies, the MultiIndex is turned
        into a DataFrame with a weight column.
        """
        if rowwise:
            od_pairs = MultiIndex.from_arrays([origins.index, destinations.index])
        else:
            od_pairs = MultiIndex.from_product([origins.index, destinations.index])

        geoms_ori = od_pairs.get_level_values(0).map(origins.geometry)
        geoms_des = od_pairs.get_level_values(1).map(destinations.geometry)
        no_identical_geoms = od_pairs[geoms_ori != geoms_des]

        if not len(no_identical_geoms) and len(origins) and len(destinations):
            raise ValueError("All origin-destination pairs have identical geometries.")

        return no_identical_geoms

    def _log_df_template(self, method: str, minutes_elapsed: float) -> DataFrame:
        """Creates a DataFrame with one row and the main columns.

        To be run after each network analysis.

        Args:
            method: the name of the network analysis method used
            minutes_elapsed: time use of the method

        Returns:
            A one-row DataFrame with log info columns
        """
        data = {
            "endtime": pd.to_datetime(datetime.now()).floor("s").to_pydatetime(),
            "minutes_elapsed": minutes_elapsed,
            "method": method,
            "origins_count": pd.NA,
            "destinations_count": pd.NA,
            "percent_missing": pd.NA,
            "cost_mean": pd.NA,
        }
        if self.rules.directed:
            data["percent_bidirectional"] = self.network.percent_bidirectional

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
            df["percent_missing"] = results[results.geometry.name].isna().mean() * 100
        else:
            df["destinations_count"] = len(self.destinations.gdf)

        if self.detailed_log:
            for key, value in kwargs.items():
                if isinstance(value, np.ndarray):
                    value = list(value)
                if isinstance(value, (list | tuple)):
                    value = [str(x) for x in value]
                    value = ", ".join(value)
                df[key] = value

        self.log = pd.concat([self.log, df], ignore_index=True)

    def _prepare_network_analysis(
        self,
        origins: GeoDataFrame,
        destinations: GeoDataFrame | None = None,
        rowwise: bool = False,
    ) -> None:
        """Prepares the weight column, node ids, origins, destinations and graph.

        Updates the graph only if it is not yet created and no parts of the analysis
        has changed. this method is run inside od_cost_matrix, get_route and
        service_area.
        """
        if rowwise and len(origins) != len(destinations):
            raise ValueError(
                "'origins' and 'destinations' must have the same length when "
                "rowwise=True"
            )

        self._unsplit_network()

        self.network.gdf = self.rules._validate_weight(self.network.gdf)

        self.origins = Origins(origins)
        self.origins._make_temp_idx(
            start=max(self.network.nodes["node_id"].astype(int)) + 1
        )

        if destinations is not None:
            self.destinations = Destinations(destinations)
            self.destinations._make_temp_idx(
                start=max(self.origins.gdf["temp_idx"].astype(int)) + 1
            )

        else:
            self.destinations = None

        if not self._graph_is_up_to_date() or not self.network._nodes_are_up_to_date():
            self.network._update_nodes_if()

            edges, weights, ids = self._get_edges_and_weights()

            self.graph = self._make_graph(
                edges=edges,
                weights=weights,
                edge_ids=ids,
                directed=self.rules.directed,
            )

            self._graph_updated_count += 1

        self._update_wkts()
        self.rules._update_rules()

    def _get_edges_and_weights(
        self,
    ) -> tuple[list[tuple[str, str]], list[float], list[str]]:
        """Creates lists of edges and weights which will be used to make the graph.

        Edges and weights between origins and nodes and nodes and destinations are
        also added.
        """
        if self.rules.split_lines:
            self._split_lines()
            self.network._make_node_ids()
            self.origins._make_temp_idx(
                start=max(self.network.nodes["node_id"].astype(int)) + 1
            )
            if self.destinations is not None:
                self.destinations._make_temp_idx(
                    start=max(self.origins.gdf["temp_idx"].astype(int)) + 1
                )

        edges: list[tuple[str, str]] = self.network.get_edges()

        weights = list(self.network.gdf[self.rules.weight])

        self.network.gdf["src_tgt_wt"] = self.network._create_edge_ids(edges, weights)

        edges_start, weights_start = self.origins._get_edges_and_weights(
            nodes=self.network.nodes,
            rules=self.rules,
            k=self._k_nearest_points,
        )

        edges = edges + edges_start
        weights = weights + weights_start

        if self.destinations is None:
            edges, weights = self._add_fake_edges(edges, weights)
            edge_ids = self.network._create_edge_ids(edges, weights)
            return edges, weights, edge_ids

        edges_end, weights_end = self.destinations._get_edges_and_weights(
            nodes=self.network.nodes,
            rules=self.rules,
            k=self._k_nearest_points,
        )

        edges = edges + edges_end
        weights = weights + weights_end

        edges, weights = self._add_fake_edges(edges, weights)
        edge_ids = self.network._create_edge_ids(edges, weights)

        return edges, weights, edge_ids

    def _add_fake_edges(self, edges, weights):
        """Add edges for origins and destinations to not get error in igraph because of non-existing nodes."""
        nodes = {x[0] for x in edges} | {x[1] for x in edges}

        fake_edges = [
            (idx, idx)
            for idx in list(self.origins.gdf["temp_idx"])
            + list(
                self.destinations.gdf["temp_idx"]
                if self.destinations is not None
                else []
            )
            if idx not in nodes
        ]
        edges = edges + fake_edges
        weights = weights + [0 for _ in fake_edges]
        return edges, weights

    def _split_lines(self) -> None:
        if self.destinations is not None:
            points = pd.concat(
                [self.origins.gdf, self.destinations.gdf], ignore_index=True
            )
        else:
            points = self.origins.gdf

        points = points.drop_duplicates(points.geometry.name)

        self.network.gdf["meters_"] = self.network.gdf.length

        # create an id from before the split, used to revert the split later
        self.network.gdf["temp_idx__"] = range(len(self.network.gdf))

        lines = split_lines_by_nearest_point(
            gdf=self.network.gdf,
            points=points,
            max_distance=self.rules.search_tolerance,
            splitted_col="splitted",
        )

        # save the unsplit lines for later
        splitted = lines.loc[lines["splitted"] == 1, "temp_idx__"]
        self.network._not_splitted = self.network.gdf.loc[
            self.network.gdf["temp_idx__"].isin(splitted)
        ]

        # adjust weight to new length
        lines[self.rules.weight] = lines[self.rules.weight] * (
            lines.length / lines["meters_"]
        )

        self.network.gdf = lines

    def _unsplit_network(self):
        """Remove the splitted lines and add the unsplitted ones."""
        if not hasattr(self.network, "_not_splitted"):
            return
        lines = self.network.gdf.loc[self.network.gdf["splitted"] != 1]
        self.network.gdf = pd.concat(
            [lines, self.network._not_splitted], ignore_index=True
        ).drop("temp_idx__", axis=1)
        del self.network._not_splitted

    @staticmethod
    def _make_graph(
        edges: list[tuple[str, ...]] | np.ndarray[tuple[str, ...]],
        weights: list[float] | np.ndarray[float],
        edge_ids: np.ndarray,
        directed: bool,
    ) -> Graph:
        """Creates an igraph Graph from a list of edges and weights."""
        assert len(edges) == len(weights) == len(edge_ids)
        graph = igraph.Graph.TupleList(edges, directed=directed)
        graph.es["weight"] = weights
        graph.es["src_tgt_wt"] = edge_ids
        graph.es["edge_tuples"] = edges
        graph.es["source"] = [edge[0] for edge in edges]
        graph.es["target"] = [edge[1] for edge in edges]

        if min(graph.es["weight"]) < 0:
            n = sum([1 for w in graph.es["weight"] if w < 0])
            raise ValueError(
                f"The graph has been built with {n} negative weight values."
            )

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

        if self.network.gdf["src_tgt_wt"].isna().any():
            return False

        for points in ["origins", "destinations"]:
            if self[points] is None:
                continue
            if points not in self.wkts:
                return False
            if self._points_have_changed(self[points].gdf, what=points):
                return False

        return True

    def _points_have_changed(self, points: GeoDataFrame, what: str) -> bool:
        """Check if the origins or destinations have changed.

        This method is best stored in the NetworkAnalysis class,
        since the point classes are instantiated each time an analysis is run.
        """
        if not np.array_equal(self.wkts[what], points.geometry.to_wkt().values):
            return True

        if not all(x in self.graph.vs["name"] for x in list(points["temp_idx"].values)):
            return True

        return False

    def _update_wkts(self) -> None:
        """Creates a dict of wkt lists.

        This method is run after the graph is created. If the wkts haven't updated
        since the last run, the graph doesn't have to be remade.

        """
        self.wkts = {}

        self.wkts["network"] = self.network.gdf.geometry.to_wkt().values

        if not hasattr(self, "origins"):
            return

        self.wkts["origins"] = self.origins.gdf.geometry.to_wkt().values

        if self.destinations is not None:
            self.wkts["destinations"] = self.destinations.gdf.geometry.to_wkt().values

    @staticmethod
    def _sort_breaks(breaks: str | list | tuple | int | float) -> list[float | int]:
        if isinstance(breaks, str):
            breaks = float(breaks)

        if hasattr(breaks, "__iter__"):
            return list(sorted(list(breaks)))

        if isinstance(breaks, (int | float)):
            return [breaks]

        raise ValueError(
            "'breaks' should be integer, float, string or an iterable of "
            f" one of these. Got {type(breaks)!r}"
        )

    def __repr__(self) -> str:
        """The print representation."""
        # drop the 'weight_to_nodes_' parameters in the repr of rules to avoid clutter
        rules = (
            f"{self.rules.__class__.__name__}(weight={self.rules.weight}, "
            f"directed={self.rules.directed}, "
            f"search_tolerance={self.rules.search_tolerance}, "
            f"search_factor={self.rules.search_factor}, "
            f"split_lines={self.rules.split_lines}, "
        )

        # add one 'weight_to_nodes_' parameter if used,
        # else inform that there are more parameters with '...'
        if self.rules.nodedist_multiplier:
            x = f"nodedist_multiplier={self.rules.nodedist_multiplier}"
        elif self.rules.nodedist_kmh:
            x = f"nodedist_kmh={self.rules.nodedist_kmh}"
        else:
            x = "..."

        return (
            f"{self.__class__.__name__}(\n"
            f"    network={self.network.__repr__()},\n"
            f"    rules={rules}{x}),\n"
            f"    log={self._log}, detailed_log={self.detailed_log},"
            "\n)"
        )

    def __getitem__(self, item: str) -> Any:
        """To be able to write self['origins'] as well as self.origins."""
        return getattr(self, item)

    def copy(self, deep: bool = True) -> "NetworkAnalysis":
        """Returns a (deep) copy of the class instance.

        Args:
            deep: Whether to return a deep or shallow copy. Defaults to True.
        """
        if deep:
            return deepcopy(self)
        else:
            return copy(self)

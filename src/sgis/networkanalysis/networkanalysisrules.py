"""The NetworkAnalysisRules class sets the rules for the network analysis.

The class is to be used as the 'rules' parameter in the NetworkAnalysis
class.
"""
import warnings
from dataclasses import dataclass

from geopandas import GeoDataFrame

from ..helpers import unit_is_meters


@dataclass
class NetworkAnalysisRules:
    """Sets the rules for the network analysis.

    To be used as the 'rules' parameter in the NetworkAnalysis class. The 'weight'
    should be either 'meters'/'metres' or a column in the network, for instance
    'minutes'.

    By default, the edges between the origins/destinations and closeby network
    nodes get a weight of 0. If you want to make the travel between the points
    and the network nodes count more than 0, you can specify one of the
    ``weight_to_nodes_`` parameters. If the weight is 'meters', setting
    'weight_to_nodes_dist' to True will give these edges a weight equivelant to
    its length in a straight line. If the weight is 'minutes', you can set a speed
    for these edges in either kilometers or miles per hour. What speed to set depends
    on whether it is likely that these distances have to be covered on foot or faster
    transport modes, and how much shorter the straight line distance is compared to the
    real-life distance.

    Args:
        weight: either a column in the gdf of the network or 'meters'/'metres'. A
            minute column can be created with the make_directed_network method of the
            DirectedNetwork class.
        search_tolerance: distance to search for nodes in the network. Origins and
            destinations further away from the network than the search_tolerance will
            not find any paths.
        search_factor: number of meters and percent to add to the closest distance to a
            node when connecting origins and destinations to the network. Defaults to
            0, meaning only the closest node is used. If search_factor is 10 and the
            closest node is 1 meter away, paths will be created from the point and all
            nodes within 11.1 meters. If the closest node is 100 meters away, paths
            will be created with all nodes within 120 meters.

            It can be wise to set a higher search_factor only for the origins and
            destinations that are causing problems in a separate analysis run.
        split_lines: If False (the default), points will be connected to the nodes of
            the network, i.e. the destinations of the lines. If True, the closest line
            to each point will be split in two at the closest part of the line to the
            point. The weight of the split lines are then adjusted to fit the new
            length. Defaults to False because it's faster and doesn't make a huge
            difference in most cases. Note: the split lines stays with the network
            until it is re-instantiated.
        weight_to_nodes_dist: If the weight is 'meters', setting this to True will make
            the edge between origins/destinations and the network count equal to its
            straight line distance.
        weight_to_nodes_kmh: if the weight is 'minutes', this will give a weight for
            the edge between the origins/destinations and the network nodes,
            where the weight will be its straight-line distance converted to minutes
            in the speed specified.
        weight_to_nodes_mph: same as weight_to_nodes_kmh, only that you speficy the
            speed in miles per hour

    Note:
        Whether the network analysis will be directed or undirected is not stored here,
        although it can be considered a 'rule'. Whether to do directed network
        analysis, depends on the network. So if the graph is made directed or not, is
        decided by which network class you use, DirectedNetwork to make a directed
        graph and the base Network class to make an undirected graph.

    Examples
    --------
    Read testdata.

    >>> import sgis as sg
    >>> roads = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")
    >>> points = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet")

    Let's start by setting the default rules. 'weight' is the only parameter with no
    default.

    >>> nw = (sg.DirectedNetwork(roads)
    ...       .remove_isolated()
    ...       .make_directed_network_norway()
    ... )
    >>> rules = sg.NetworkAnalysisRules(weight="minutes")
    >>> nwa = sg.NetworkAnalysis(network=nw, rules=rules)
    >>>
    >>> nwa
    NetworkAnalysis(
        network=DirectedNetwork(6364 km, percent_bidirectional=87),
        rules=NetworkAnalysisRules(weight=minutes, search_tolerance=250, search_factor=0, split_lines=False, ...)
    )

    Setting 'split_lines' to True, means the points will be connected to the closest
    part of the closest network line. If False, the lines are connected to the closest
    endpoint of the lines. split_lines defaults to False, since splitting lines takes
    some time and doesn't make a huge difference in most cases.

    >>> od = nwa.od_cost_matrix(points, points)
    >>> nwa.rules.split_lines = True
    >>> od = nwa.od_cost_matrix(points, points)
    >>>
    >>> nwa.log[['split_lines', 'percent_missing', 'cost_mean']]
       split_lines  percent_missing  cost_mean
    0        False           0.9966  15.270462
    1         True           0.2995   3.306094

    Setting a high search_tolerance will make faraway points find their way to the
    network.

    >>> for i in [100, 250, 500, 1000]:
    ...     nwa.rules.search_tolerance = i
    ...     od = nwa.od_cost_matrix(points, points)
    >>>
    >>> nwa.log.iloc[-4:][['percent_missing', 'cost_mean', 'search_tolerance', 'search_factor']]
       percent_missing  cost_mean  search_tolerance  search_factor
    2           2.2854  15.538083               100              0
    3           0.7977  15.178683               250              0
    4           0.6982  15.570546               500              0
    5           0.4989  15.565538              1000              0

    High search_tolerance won't affect how the points close to the network are
    connected to network nodes. Points trapped behind deadend oneway streets, can find
    their way out with a higher search_factor.

    >>> nwa.rules.search_tolerance = 250
    >>> for i in [0, 10, 35, 100]:
    ...     nwa.rules.search_factor = i
    ...     od = nwa.od_cost_matrix(points, points)
    >>>
    >>> nwa.log.iloc[-4:][['percent_missing', 'cost_mean', 'search_tolerance', 'search_factor']]
       percent_missing  cost_mean  search_tolerance  search_factor
    6           0.8973  15.566827               250              0
    7           0.6983  15.375234               250             10
    8           0.4991  14.890791               250             35
    9           0.3994  13.910521               250            100

    The remaining 0.4 percent missing are from/to two points, one on an island with no
    brigde and one at the edge of the road network (would require a larger network).
    These two points only find themselves, and thus has 999 missing values.

    >>> nwa.origins.gdf.sort_values("missing").tail(3)
          idx                        geometry temp_idx  missing
    999  1000  POINT (264570.300 6644239.500)    80957        2
    510   511  POINT (261319.300 6647824.800)    80468      999
    59     60  POINT (271816.400 6650812.500)    80017      999

    By default, the distance from origin/destination to the network nodes is given a
    weight of 0. This means, if the search_tolerance is high, points far away from the
    network will get unrealisticly low travel times/distances. This can be changed with
    one of the ``weight_to_nodes_`` parameters.

    If the weight is 'minutes', you can set the speed for the travel between
    origin/destination and nodes, either in kilometers or miles per hour. What speed to
    set, will depend on how the distance is likely to be covered (foot, boat, car etc.).

    >>> nwa.rules.search_tolerance = 5000
    >>> for i in [3, 10, 50]:
    ...     nwa.rules.weight_to_nodes_kmh = i
    ...     od = nwa.od_cost_matrix(points, points)
    ...
    >>> nwa.log.iloc[-3:][['weight_to_nodes_kmh', 'cost_mean']]
       weight_to_nodes_kmh  cost_mean
    10                   3  15.898794
    11                  10  14.945977
    12                  50  14.164665

    If the weight is 'meters', setting weight_to_nodes_dist=True will make the distance
    to nodes count as its straight line distance.

    >>> rules = NetworkAnalysisRules(
    ...     weight="meters",
    ...     search_tolerance=5000,
    ... )
    >>> nwa = NetworkAnalysis(network=nw, rules=rules)
    >>> od = nwa.od_cost_matrix(points, points)
    >>> nwa.rules.weight_to_nodes_dist = True
    >>> od = nwa.od_cost_matrix(points, points)
    >>>
    >>> nwa.log[['weight_to_nodes_dist', 'cost_mean']]
       weight_to_nodes_dist     cost_mean
    0                 False  10228.400228
    1                  True  10277.926186
    """

    weight: str
    search_tolerance: int = 250
    search_factor: int = 0
    split_lines: bool = False
    weight_to_nodes_dist: bool = False
    weight_to_nodes_kmh: int | None = None
    weight_to_nodes_mph: int | None = None

    def _update_rules(self):
        """Stores the rules as separate attributes.

        Used for checking whether the rules have changed.
        """
        self._weight = self.weight
        self._search_tolerance = self.search_tolerance
        self._search_factor = self.search_factor
        self._split_lines = self.split_lines
        self._weight_to_nodes_dist = self.weight_to_nodes_dist
        self._weight_to_nodes_kmh = self.weight_to_nodes_kmh
        self._weight_to_nodes_mph = self.weight_to_nodes_mph

    def _rules_have_changed(self):
        """Checks if any of the rules have changed since the graph was last created.

        If no rules have changed, time can be saved by not remaking the graph
        (the network and the points have to be unchanged as well).
        """
        if self.weight != self._weight:
            return True
        if self.search_factor != self._search_factor:
            return True
        if self.search_tolerance != self._search_tolerance:
            return True
        if self.split_lines != self._split_lines:
            return True
        if self.weight_to_nodes_dist != self._weight_to_nodes_dist:
            return True
        if self.weight_to_nodes_kmh != self._weight_to_nodes_kmh:
            return True
        if self.weight_to_nodes_mph != self._weight_to_nodes_mph:
            return True

    def _validate_weight(
        self, gdf: GeoDataFrame, raise_error: bool = True
    ) -> GeoDataFrame:
        if "meter" in self.weight or "metre" in self.weight and unit_is_meters(gdf):
            gdf[self.weight] = gdf.length
            return gdf
        elif self.weight in gdf.columns:
            gdf[self.weight] = gdf[self.weight].astype(float)
            gdf = self._check_for_nans(gdf, self.weight)
            gdf = self._check_for_negative_values(gdf, self.weight)
            gdf = self._try_to_float(gdf, self.weight)
            return gdf
        elif (
            self.weight == "min" or "minut" in self.weight and "minutes" in gdf.columns
        ):
            self.weight = "minutes"
            gdf["minutes"] = gdf[self.weight]
            return gdf

        # at this point, the weight is wrong.
        if "meter" in self.weight or "metre" in self.weight:
            raise ValueError(
                "the crs of the roads have to have units in 'meters' when the "
                "weight is 'meters'."
            )

        if self.weight == "minutes":
            incorrect_weight_column = (
                "Cannot find 'weight' column for minutes. "
                "Try running one of the 'make_directed_network_' methods"
                ", or set 'weight' to 'meters'"
            )

        else:
            incorrect_weight_column = f"Cannot find 'weight' column {self.weight}"

        if raise_error:
            raise KeyError(incorrect_weight_column)
        else:
            warnings.warn(incorrect_weight_column, stacklevel=2)

        return gdf

    @staticmethod
    def _check_for_nans(df, col):
        """Remove NaNs and give warning if there are any."""
        if all(df[col].isna()):
            raise ValueError(f"All values in the {col!r} column are NaN.")

        nans = sum(df[col].isna())
        if nans:
            warnings.warn(
                f"Warning: {nans} rows have missing values in the {col!r} column. "
                "Removing these rows.",
                stacklevel=2,
            )
            df = df.loc[df[col].notna()]

        return df

    @staticmethod
    def _check_for_negative_values(df, col):
        """Remove negative values and give warning if there are any."""
        negative = sum(df[col] < 0)
        if negative:
            warnings.warn(
                f"Warning: {negative} rows have a 'col' less than 0. Removing these "
                "rows.",
                stacklevel=2,
            )
            df = df.loc[df[col] >= 0]

        return df

    @staticmethod
    def _try_to_float(df, col):
        """Try to convert weight column to float, raise ValueError if it fails."""
        try:
            df[col] = df[col].astype(float)
        except ValueError as e:
            raise ValueError(
                f"The {col!r} column must be numeric. Got characters that couldn't be "
                "interpreted as numbers."
            ) from e
        return df

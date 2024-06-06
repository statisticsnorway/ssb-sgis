"""The NetworkAnalysisRules class sets the rules for the network analysis.

The class is to be used as the 'rules' parameter in the NetworkAnalysis
class.
"""

from copy import copy
from copy import deepcopy
from dataclasses import dataclass

from geopandas import GeoDataFrame

from ..helpers import unit_is_meters


@dataclass
class NetworkAnalysisRules:
    """Sets the rules for the network analysis.

    To be used as the 'rules' parameter in the NetworkAnalysis class.

    Args:
        weight: Either a column in the GeoDataFrame of the Network or
            'meters'/'metres'. A 'minutes' column can be created with the
            'make_directed_network' or 'make_directed_network_norway' functions.
        directed: Whether the lines will be considered traversable in both directions.
        search_tolerance: distance to search for nodes in the network. Origins and
            destinations further away from the network than the search_tolerance will
            not find any paths. Defaults to 250.
        search_factor: number of meters and percent to add to the closest distance to a
            node when connecting origins and destinations to the network. Defaults to
            0, meaning only the closest node is used. If search_factor is 10 and the
            closest node is 1 meter away, paths will be created from the point and all
            nodes within 11.1 meters. If the closest node is 100 meters away, paths
            will be created with all nodes within 120 meters.

            It can be wise to set a higher search_factor only for the origins and
            destinations that are causing problems in a separate analysis run.
        split_lines: If True (default), the closest line  to each point will be
            split in two at the nearest excact point. The weight of the split lines
            are then adjusted to the new length. If False, the points will be
            connected to the endpoints of the network lines.
        nodedist_kmh: When using "minutes" as weight, this sets the speed in kilometers
            per hour for the edges between origins/destinations and the network nodes
            that connect them. Defaults to None, meaning 0 weight is added for the
            edges.
        nodedist_multiplier: When using "meters" as weight, this sets the weight for
            the edges between origins/destinations and the network nodes that connect
            them. Defaults to None, meaning 0 weight is added for these edges. If set
            to 1, the weight will be equal to the straigt line distance.

    Examples:
    ---------
    Read testdata.

    >>> import sgis as sg
    >>> roads = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")
    >>> points = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet")

    Let's start by setting the default rules. 'weight' and 'directed' have no default
    values.

    >>> rules = sg.NetworkAnalysisRules(weight="minutes", directed=True)
    >>> directed_roads = sg.get_connected_components(roads).loc[lambda x: x["connected"] == 1].pipe(sg.make_directed_network_norway, dropnegative=True)
    >>> nwa = sg.NetworkAnalysis(network=directed_roads, rules=rules, detailed_log=True)
    >>> nwa
    NetworkAnalysis(
        network=Network(6364 km, percent_bidirectional=87),
        rules=NetworkAnalysisRules(weight=minutes, directed=True, search_tolerance=250, search_factor=0, split_lines=True, ...),
        log=True, detailed_log=True,
    )

    Setting 'split_lines' to True, means the points will be connected to the closest
    part of the closest network line. If False, the lines are connected to the closest
    endpoint of the lines. split_lines defaults to False, since splitting lines takes
    some time and doesn't make a huge difference in most cases.

    >>> od = nwa.od_cost_matrix(points, points)
    >>> nwa.rules.split_lines = True
    >>> od = nwa.od_cost_matrix(points, points)

    >>> nwa.log[['split_lines', 'percent_missing', 'cost_mean']]
       split_lines  percent_missing  cost_mean
    0        False           0.9966  15.270462
    1         True           0.8973  15.249900

    >>> nwa.rules.split_lines = False

    Setting a high search_tolerance will make faraway points find their way to the
    network.

    >>> for i in [100, 250, 500, 1000]:
    ...     nwa.rules.search_tolerance = i
    ...     od = nwa.od_cost_matrix(points, points)

    >>> nwa.log.iloc[-4:][['percent_missing', 'cost_mean', 'search_tolerance', 'search_factor']]
       percent_missing  cost_mean  search_tolerance  search_factor
    2           2.3840  15.235559               100              0
    3           0.9966  15.270462               250              0
    4           0.7976  15.273579               500              0
    5           0.5984  15.268614              1000              0

    High search_tolerance won't affect how the points close to the network are
    connected to network nodes. Points trapped behind deadend oneway streets, can find
    their way out with a higher search_factor.

    >>> nwa.rules.search_tolerance = 250
    >>> for i in [0, 10, 35, 100]:
    ...     nwa.rules.search_factor = i
    ...     od = nwa.od_cost_matrix(points, points)

    >>> nwa.log.iloc[-4:][['percent_missing', 'cost_mean', 'search_tolerance', 'search_factor']]
       percent_missing  cost_mean  search_tolerance  search_factor
    6           0.9966  15.270462               250              0
    7           0.5987  15.063283               250             10
    8           0.4991  14.636172               250             35
    9           0.3994  13.680307               250            100

    The remaining 0.4 percent missing are from/to two points, one on an island with no
    brigde and one at the edge of the road network (would require a larger network).
    These two points only find themselves, and thus has 999 missing values.

    >>> n_missing = od.groupby("origin").minutes.agg(lambda x: x.isna().sum())
    >>> n_missing.n_largest(3)
    59     999
    510    999
    0        2
    Name: minutes, dtype: int64

    By default, the distance from origin/destination to the network nodes is given a
    weight of 0. This means, if the search_tolerance is high, points far away from the
    network will get unrealisticly low travel times/distances. The weight from origin/
    destination to the network nodes can be set with the 'nodedist_kmh' parameter if
    the weight is 'minutes', and the 'nodedist_multiplier' if the weight is 'meters'.

    If the weight is 'minutes', setting 'nodedist_kmh' to 5 means a distance of 1000
    meters will get a weight of 12 minutes.

    >>> nwa.rules.search_tolerance = 5000
    >>> for i in [3, 10, 50]:
    ...     nwa.rules.nodedist_kmh = i
    ...     od = nwa.od_cost_matrix(points, points)
    ...
    >>> nwa.log.iloc[-3:][['nodedist_kmh', 'cost_mean']]
       nodedist_kmh  cost_mean
    10            3  15.924197
    11           10  14.817717
    12           50  13.964457

    If the weight is 'meters', setting nodedist_multiplier=1 will make the distance
    to nodes count as its straight line distance.

    >>> rules = NetworkAnalysisRules(
    ...     weight="meters",
    ...     search_tolerance=5000,
    ... )
    >>> nwa = NetworkAnalysis(network=directed_roads, rules=rules)
    >>> od = nwa.od_cost_matrix(points, points)
    >>> nwa.rules.nodedist_multiplier = 1
    >>> od = nwa.od_cost_matrix(points, points)

    >>> nwa.log[['nodedist_multiplier', 'cost_mean']]
      nodedist_multiplier     cost_mean
    0                None  10228.400228
    1                   1  10277.926186
    """

    directed: bool
    weight: str
    search_tolerance: int = 250
    search_factor: int = 0
    split_lines: bool = True
    nodedist_multiplier: int | float | None = None
    nodedist_kmh: int | float | None = None

    def _update_rules(self) -> None:
        """Stores the rules as separate attributes.

        Used for checking whether the rules have changed and the graph have to be
        remade.
        """
        self._directed = self.directed
        self._weight = self.weight
        self._search_tolerance = self.search_tolerance
        self._search_factor = self.search_factor
        self._split_lines = self.split_lines
        self._nodedist_multiplier = self.nodedist_multiplier
        self._nodedist_kmh = self.nodedist_kmh

    def _rules_have_changed(self) -> bool:
        """Checks if any of the rules have changed since the graph was last created.

        If no rules have changed, time can be saved by not remaking the graph
        (the network and the points have to be unchanged as well).
        """
        if self.directed != self._directed:
            return True
        if self.weight != self._weight:
            return True
        if self.search_factor != self._search_factor:
            return True
        if self.search_tolerance != self._search_tolerance:
            return True
        if self.split_lines != self._split_lines:
            return True
        if self.nodedist_multiplier != self._nodedist_multiplier:
            return True
        if self.nodedist_kmh != self._nodedist_kmh:
            return True

    def _validate_weight(self, gdf: GeoDataFrame) -> GeoDataFrame:
        if "meter" in self.weight or "metre" in self.weight and unit_is_meters(gdf):
            if self.nodedist_kmh:
                raise ValueError("Cannot set 'nodedist_kmh' when 'weight' is meters.")
            gdf[self.weight] = gdf.length
            return gdf

        # allow abbreviation of 'minutes' to be nice
        elif (
            self.weight == "min" or "minut" in self.weight and "minutes" in gdf.columns
        ):
            if self.nodedist_multiplier:
                raise ValueError(
                    "Cannot set 'nodedist_multiplier' when 'weight' is minutes. "
                    "Set 'nodedist_kmh' instead."
                )
            self.weight = "minutes"
            gdf["minutes"] = gdf[self.weight]
            self._check_for_nans(gdf, self.weight)
            gdf = self._try_to_float(gdf, self.weight)
            self._check_for_negative_values(gdf, self.weight)
            return gdf

        elif self.weight in gdf.columns:
            self._check_for_nans(gdf, self.weight)
            gdf = self._try_to_float(gdf, self.weight)
            self._check_for_negative_values(gdf, self.weight)
            return gdf

        # at this point, the weight is wrong.
        # Now to determine the error message

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

        raise KeyError(incorrect_weight_column)

    @staticmethod
    def _check_for_nans(df: GeoDataFrame, col: str) -> None:
        """Remove NaNs and give warning if there are any."""
        if all(df[col].isna()):
            raise ValueError(f"All values in the {col!r} column are NaN.")

        nans = sum(df[col].isna())
        if nans:
            raise ValueError(
                f"{nans} rows have missing values in the {col!r} column. "
                "Fill these rows with 0 or another number.",
            )

    @staticmethod
    def _check_for_negative_values(df: GeoDataFrame, col: str) -> None:
        """Remove negative values and give warning if there are any."""
        negative = sum(df[col] < 0)
        if negative:
            raise ValueError(
                f"{negative} negative values found in the {col!r} column. Fill these "
                "with a number greater than or equal to zero.",
            )

    @staticmethod
    def _try_to_float(df: GeoDataFrame, col: str) -> GeoDataFrame:
        """Try to convert weight column to float, raise ValueError if it fails."""
        try:
            df[col] = df[col].astype(float)
        except ValueError as e:
            raise ValueError(
                f"The {col!r} column must be numeric. Got characters that couldn't be "
                "interpreted as numbers."
            ) from e
        return df

    def copy(self) -> "NetworkAnalysisRules":
        """Return a shallow copy the instance."""
        return copy(self)

    def deepcopy(self) -> "NetworkAnalysisRules":
        """Return a deep copy the instance."""
        return deepcopy(self)

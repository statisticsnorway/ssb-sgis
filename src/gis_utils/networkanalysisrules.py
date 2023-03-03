import warnings
from dataclasses import dataclass

from geopandas import GeoDataFrame


@dataclass
class NetworkAnalysisRules:
    weight: str
    search_tolerance: int = 250
    search_factor: int = 10
    split_lines: bool = False
    weight_to_nodes_dist: bool = False
    weight_to_nodes_kmh: int | None = None
    weight_to_nodes_mph: int | None = None

    """Rules for how the network analysis should be executed

    To be used as the 'rules' parameter in the NetworkAnalysis class. The 'weight'
    should be either 'meters'/'metres' or a column in the network, for instance
    'minutes'.

    By default, the edges between the origins/destinations and closeby network
    nodes get a weight of 0. If you want to make the travel between the points
    and the network nodes count more than 0, you can specify one of the
    'weight_to_nodes_' parameters. If the weight is 'meters', setting
    'weight_to_nodes_dist' to True will give these edges a weight equivelant to
    its length in a straight line. If the weight is 'minutes', you can set a speed
    for these edges in either kilometers or miles per hour. What speed to set depends
    on whether it is likely that these distances have to be covered on foot or faster
    transport modes, and how much shorter the straight line distance is compared to the
    real-life distance.

    Note:
        Whether the network analysis will be directed or undirected is not stored in
        this class, although it can be considered a 'rule'. Whether to do directed
        network analysis, depends on the network. So if the graph is made directed or
        not, is decided by which network class you use, DirectedNetwork to make a
        directed graph and the base Network class to make an undirected graph.

    Args:
        weight: either a column in the gdf of the network or 'meters'/'metres'. A
            minute column can be created with the make_directed_network method of the
            DirectedNetwork class.
        search_tolerance: distance to search for nodes in the network.
            Points further away than the search_tolerance will not find any paths
        search_factor: number of meters and percent to add to the closest distance to a
            node. So if the closest node is 1 meter away, paths will be created from
            the point and all nodes within 11.1 meters. If the closest node is 100
            meters away, paths will be created with all nodes within 120 meters.
        split_lines: If False (the default), points will be connected to the nodes of
            the network, i.e. the destinations of the lines. If True, the closest line
            to each point will be split in two at the closest part of the line to the
            point. The weight of the split lines are then adjusted to the ratio to the
            original length. Defaults to False because it's faster and doesn't make a
            huge difference in most cases. Note: the split lines stays with the network
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
    """

    def _update_rules(self):
        """Stores the rules as separate attributes
        used for checking whether the rules have changed.
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
        if self.weight in gdf.columns:
            gdf = self._check_for_nans(gdf, self.weight)
            gdf = self._check_for_negative_values(gdf, self.weight)
            gdf = self._try_to_float(gdf, self.weight)

        elif "meter" in self.weight or "metre" in self.weight:
            if gdf.crs.axis_info[0].unit_name != "metre":
                raise ValueError(
                    "the crs of the roads have to have units in 'meters' when the "
                    "weight is 'meters'."
                )

            gdf[self.weight] = gdf.length

        elif (
            self.weight == "min" or "minut" in self.weight and "minutes" in gdf.columns
        ):
            self.weight = "minutes"

        if self.weight not in gdf.columns:
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
                warnings.warn(incorrect_weight_column)

        return gdf

    @staticmethod
    def _check_for_nans(df, col):
        """Remove NaNs and give warning if there are any"""
        if all(df[col].isna()):
            raise ValueError(f"All values in the '{col}' column are NaN.")

        nans = sum(df[col].isna())
        if nans:
            warnings.warn(
                f"Warning: {nans} rows have missing values in the '{col}' column. "
                "Removing these rows."
            )
            df = df.loc[df[col].notna()]

        return df

    @staticmethod
    def _check_for_negative_values(df, col):
        negative = sum(df[col] < 0)
        if negative:
            warnings.warn(
                f"Warning: {negative} rows have a 'col' less than 0. Removing these "
                "rows."
            )
            df = df.loc[df[col] >= 0]

        return df

    @staticmethod
    def _try_to_float(df, col):
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            raise ValueError(
                f"The '{col}' column must be numeric. Got characters that couldn't be "
                "interpreted as numbers."
            )
        return df

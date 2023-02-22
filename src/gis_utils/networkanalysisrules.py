import warnings
from dataclasses import dataclass

from geopandas import GeoDataFrame


@dataclass
class NetworkAnalysisRules:
    weight: str
    search_tolerance: int = 250
    search_factor: int = 10
    weight_to_nodes_dist: bool = False
    weight_to_nodes_kmh: int | None = None
    weight_to_nodes_mph: int | None = None

    """
    Args:
        weight: either 'meters'/'metres' or a column in the network.gdf
        search_tolerance: distance to search for
    """

    def update_rules(self):
        """
        Stores the rules as separate attributes to
        be able to check whether the rules have changed.
        """

        self._weight = self.weight
        self._search_tolerance = self.search_tolerance
        self._search_factor = self.search_factor
        self._weight_to_nodes_dist = self.weight_to_nodes_dist
        self._weight_to_nodes_kmh = self.weight_to_nodes_kmh
        self._weight_to_nodes_mph = self.weight_to_nodes_mph

    def rules_have_changed(self):
        """Checks if any of the rules have changed since the graph was made last.
        If no rules have changed, the graph doesn't have to be remade (the network and
        points have to be the same as well).
        """
        if self.weight != self._weight:
            return True
        if self.search_factor != self._search_factor:
            return True
        if self.search_tolerance != self._search_tolerance:
            return True
        if self.weight_to_nodes_dist != self._weight_to_nodes_dist:
            return True
        if self.weight_to_nodes_kmh != self._weight_to_nodes_kmh:
            return True
        if self.weight_to_nodes_mph != self._weight_to_nodes_mph:
            return True

    def validate_weight(
        self, gdf: GeoDataFrame, raise_error: bool = True
    ) -> GeoDataFrame:
        if self.weight in gdf.columns:
            gdf = self.check_for_nans(gdf, self.weight)
            gdf = self.check_for_negative_values(gdf, self.weight)
            gdf = self.try_to_float(gdf, self.weight)

        elif "meter" in self.weight or "metre" in self.weight:
            if not gdf.crs.axis_info[0].unit_name == "metre":
                raise ValueError(
                    "the crs of the roads have to have units in 'meters' when the weight is 'meters'."
                )

            gdf[self.weight] = gdf.length

        elif (
            self.weight == "min" or "minut" in self.weight and "minutes" in gdf.columns
        ):
            self.weight = "minutes"

        if raise_error and self.weight not in gdf.columns:
            if self.weight == "minutes":
                raise KeyError(
                    f"Cannot find 'cost' column for minutes. Try running one of the 'make_directed_network_' methods, or set 'cost' to 'meters'."
                )
            else:
                raise KeyError(f"Cannot find 'cost' column {self.weight}")

        return gdf

    @staticmethod
    def check_for_nans(gdf, cost):
        if all(gdf[cost].isna()):
            raise ValueError("All values in the 'cost' column are NaN.")

        nans = sum(gdf[cost].isna())
        if nans:
            if nans > len(gdf) * 0.05:
                warnings.warn(
                    f"Warning: {nans} rows have missing values in the 'cost' column. Removing these rows."
                )
            gdf = gdf.loc[gdf[cost].notna()]

        return gdf

    @staticmethod
    def check_for_negative_values(gdf, cost):
        negative = sum(gdf[cost] < 0)
        if negative:
            if negative > len(gdf) * 0.05:
                warnings.warn(
                    f"Warning: {negative} rows have a 'cost' less than 0. Removing these rows."
                )
            gdf = gdf.loc[gdf[cost] >= 0]

        return gdf

    @staticmethod
    def try_to_float(gdf, cost):
        try:
            gdf[cost] = gdf[cost].astype(float)
        except ValueError:
            raise ValueError(
                f"The 'cost' column must be numeric. Got characters that couldn't be interpreted as numbers."
            )
        return gdf

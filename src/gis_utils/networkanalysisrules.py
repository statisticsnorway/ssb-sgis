import warnings
from dataclasses import dataclass

from geopandas import GeoDataFrame


@dataclass
class NetworkAnalysisRules:
    cost: str
    search_tolerance: int = 250
    search_factor: int = 10
    cost_to_nodes: int = 5

    def rules_have_changed(self):
        if self.search_factor != self._search_factor:
            return True
        if self.search_tolerance != self._search_tolerance:
            return True
        if self.cost_to_nodes != self._cost_to_nodes:
            return True

    def update_rules(self):
        self._search_tolerance = self.search_tolerance
        self._search_factor = self.search_factor
        self._cost_to_nodes = self.cost_to_nodes

    def validate_cost(
        self, gdf: GeoDataFrame, raise_error: bool = True
    ) -> GeoDataFrame:
        if "meter" in self.cost or "metre" in self.cost:
            if gdf.crs == 4326:
                raise ValueError(
                    "'roads' cannot have crs 4326 (latlon) when cost is 'meters'."
                )

            gdf[self.cost] = gdf.length

        elif self.cost in gdf.columns:
            gdf = self.check_for_nans(gdf, self.cost)
            gdf = self.check_for_negative_values(gdf, self.cost)
            gdf = self.try_to_float(gdf, self.cost)

        elif "min" in self.cost and "minutes" in gdf.columns:
            self.cost = "minutes"

        elif raise_error:
            if "min" in self.cost:
                raise KeyError(
                    f"Cannot find 'cost' column for minutes. Try running one of the 'make_directed_network_' methods, or set 'cost' to 'meters'."
                )
            else:
                raise KeyError(f"Cannot find 'cost' column {self.cost}")

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cost={self.cost}, search_tolerance={self.search_tolerance}, search_factor={self.search_factor}, cost_to_nodes={self.cost_to_nodes})"

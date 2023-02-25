from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from pandas import DataFrame


@dataclass
class Log:
    def _make_log(self) -> DataFrame:
        if hasattr(self.network, "_isolated_removed"):
            self.network._isolated_removed = "yes"
        elif "connected" in self.network.gdf:
            self.network._isolated_removed = (
                "yes" if all(self.network.gdf["connected"] == 1) else "no"
            )
        else:
            self.network._isolated_removed = np.nan

        if hasattr(self.network, "_is_directed"):
            self.network._is_directed = self.network._is_directed
        else:
            self.network._is_directed = "no"

        return self._log_df_template("__init__")

    def _log_df_template(self, fun: str) -> DataFrame:
        if not hasattr(self, "log"):
            nth_run = 0
        else:
            nth_run = max(self.log["nth_run"]) + 1

        df = DataFrame(
            {
                "endtime": pd.to_datetime(datetime.now()).floor("S").to_pydatetime(),
                "nth_run": nth_run,
                "function": fun,
                "percent_missing": np.nan,
                "mean_weight": np.nan,
                "n_origins": np.nan,
                "n_destinations": np.nan,
                "is_directed": self.network._is_directed,
                "isolated_removed": self.network._isolated_removed,
            },
            index=[0],
        )

        for key, value in self.rules.__dict__.items():
            if key.startswith("_") or key.endswith("_"):
                continue
            df = pd.concat([df, pd.DataFrame({key: [value]})], axis=1)

        return df

    def _runlog(self, fun: str, results: DataFrame | GeoDataFrame, **kwargs) -> None:
        df = self._log_df_template(fun)

        df["n_origins"] = len(self.origins.gdf)

        if fun != "service_area":
            df["n_destinations"] = len(self.destinations.gdf)

        if self.rules.weight in results.columns:
            df["percent_missing"] = results[self.rules.weight].isna().mean() * 100
            df["mean_weight"] = results[self.rules.weight].mean()

        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                value = list(value)
            if isinstance(value, (list, tuple)):
                value = [str(x) for x in value]
                value = ", ".join(value)
            df[key] = value

        self.log = pd.concat([self.log, df])

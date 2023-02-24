import warnings
from typing import Tuple

import numpy as np
from geopandas import GeoDataFrame
from pandas import DataFrame
from shapely.constructive import reverse

from .geopandas_utils import gdf_concat
from .network import Network


class DirectedNetwork(Network):
    def __init__(
        self,
        gdf: GeoDataFrame,
        merge_lines: bool = True,
        **kwargs,
    ):
        super().__init__(gdf, merge_lines, **kwargs)

        self.directed = True

    def _warn_if_not_directed(self):
        """Road data often have to be duplicated and flipped to make it directed."""

        if self._percent_directional > 5:
            return

        mess = (
            "Your network is likely not directed. "
            f"Only {self._percent_directional:.1f} percent of the lines go both ways."
        )
        if "oneway" in [col.lower() for col in self.gdf.columns]:
            mess = mess + (
                " Try setting direction_col='oneway' in the 'make_directed_network' method"
            )
        else:
            mess = mess + "Try running 'make_directed_network'"

        warnings.warn(mess)

    def make_directed_network_osm(
        self,
        direction_col: str = "oneway",
        direction_vals_bft: list | tuple = ("B", "F", "T"),
        speed_col: str = "maxspeed",
    ):
        return self.make_directed_network(
            direction_col=direction_col,
            direction_vals_bft=direction_vals_bft,
            speed_col=speed_col,
        )

    def make_directed_network_norway(
        self,
        direction_col: str = "oneway",
        direction_vals_bft: Tuple[str, str, str] = ("B", "FT", "TF"),
        minute_cols: Tuple[str, str] = ("drivetime_fw", "drivetime_bw"),
    ):
        """https://kartkatalog.geonorge.no/metadata/nvdb-ruteplan-nettverksdatasett/8d0f9066-34f9-4423-be12-8e8523089313"""

        return self.make_directed_network(
            direction_col=direction_col,
            direction_vals_bft=direction_vals_bft,
            minute_cols=minute_cols,
        )

    def make_directed_network(
        self,
        direction_col: str,
        direction_vals_bft: Tuple[str, str, str],
        speed_col: str | None = None,
        minute_cols: Tuple[str, str] | str | None = None,
        flat_speed: int | None = None,
    ):
        """Flips the line geometries of roads going backwards and in both directions.

        Args:
            direction_col: name of column specifying the direction of the line geometry.
            direction_vals_bft: tuple or list with the values of the direction column.
                Must be in the order 'both directions', 'from', 'to'. E.g. ('B', 'F', 'T').
            speed_col (optional): name of column with the road speed limit.
            minute_cols (optional): column or columns containing the number of minutes it takes to traverse the line.
                If one column name is given, this will be used for both directions. If tuple/list with two column names,
                the first column will be used as the minute column for the forward direction, and the second column for roads going backwards.
            flat_speed (optional): Speed in kilometers per hour to use as the speed for all roads.

        Returns:
            The Network class, with the network attribute updated with flipped geometries for lines going backwards and both directions.
            Adds the column 'minutes' if either 'speed_col', 'minute_col' or 'flat_speed' is specified.

        """

        if len(direction_vals_bft) != 3:
            raise ValueError(
                "'direction_vals_bft' should be tuple/list with values of directions "
                "both, from and to. E.g. ('B', 'F', 'T')"
            )

        if not minute_cols and not speed_col and not flat_speed:
            warnings.warn(
                "Minute column will not be calculated when both 'minute_cols', "
                "'speed_col' and 'flat_speed' is None"
            )

        if sum([bool(minute_cols), bool(speed_col), bool(flat_speed)]) > 1:
            raise ValueError(
                "Can only calculate minutes from either 'speed_col', "
                "'minute_cols' or 'flat_speed'."
            )

        nw = self.gdf
        b, f, t = direction_vals_bft

        if "b" in t.lower() and "t" in b.lower() and "f" in f.lower():
            warnings.warn(
                f"The 'direction_vals_bft' should be in the order 'both ways', "
                f"'from', 'to'. Got {b, f, t}. Is this correct?"
            )

        if minute_cols:
            nw = nw.drop("minutes", axis=1, errors="ignore")

        ft = nw.loc[nw[direction_col] == f]
        tf = nw.loc[nw[direction_col] == t]
        both_ways = nw.loc[nw[direction_col] == b]
        both_ways2 = both_ways.copy()

        if minute_cols:
            if isinstance(minute_cols, str):
                min_f, min_t = minute_cols, minute_cols
            if len(minute_cols) > 2:
                raise ValueError(
                    "'minute_cols' should be column name (string) or tuple/list with "
                    "values of directions forwards and backwards, in that order."
                )

            if len(minute_cols) == 2:
                min_f, min_t = minute_cols

            both_ways = both_ways.rename(columns={min_f: "minutes"})
            both_ways2 = both_ways2.rename(columns={min_t: "minutes"})

            ft = ft.rename(columns={min_f: "minutes"})
            tf = tf.rename(columns={min_t: "minutes"})

        tf.geometry = reverse(tf.geometry)
        both_ways2.geometry = reverse(both_ways2.geometry)

        self.gdf = gdf_concat([both_ways, both_ways2, ft, tf])

        unit = self.gdf.crs.axis_info[0].unit_name
        if (flat_speed or speed_col) and unit != "metre":
            raise ValueError(
                f"The crs must have 'metre' as units when calculating minutes. Got '{unit}'. "
                "Change crs or calculate minutes manually."
            )

        if speed_col:
            self.gdf["minutes"] = (
                self.gdf.length / self.gdf[speed_col].astype(float) * 16.6666666667
            )

        if flat_speed:
            self.gdf["minutes"] = self.gdf.length / flat_speed * 16.6666666667

        if "minutes" in self.gdf.columns:
            self.gdf = self.gdf.loc[self.gdf["minutes"] >= 0]

        self.make_node_ids()

        self._is_directed = True
        self._percent_directional = self._check_percent_directional()

        return self

"""DirectedNetwork class for directed network analysis."""

import warnings

import pandas as pd
from geopandas import GeoDataFrame
from shapely.constructive import reverse

from ..helpers import return_two_vals, unit_is_meters
from .network import Network


class DirectedNetwork(Network):
    """Subclass of Network with methods for making the network directed.

    Can be used as the 'network' parameter in the NetworkAnalysis class for directed
    network analysis.

    The DirectedNetwork class differs from the Network base class in two ways:
    1) using a DirectedNetwork in the NetworkAnalysis class means the network graph
    will be directed, meaning you can only travel in one direction on each line.
    2) the class offers methods for making the network directed, mainly the
    'make_directed_network' method, which reverses lines going the wrong direction
    and duplicates and flips lines going both directions. It also creates a 'minute'
    column.

    Args:
        gdf: a GeoDataFrame of line geometries.
        **kwargs: keyword arguments taken by the base class Network.

    Attributes:
        gdf: the GeoDataFrame of lines with source and target ids.
        nodes: GeoDataFrame of points with unique ids. The ids are dynamic, and changes
            every time the lines change. The nodes cannot be changed directly.

    Examples
    --------
    Read example data.

    >>> from sgis import read_parquet_url, DirectedNetwork
    >>> roads = read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_eidskog_2022.parquet")

    Create a DirectedNetwork instance. Notice that percent_bidirectional is 0.

    >>> nw = DirectedNetwork(roads).remove_isolated()
    >>> nw
    DirectedNetwork(3407 km, percent_bidirectional=0)

    Make the network bidirectional by specifying the direction column and its values.

    >>> nw = nw.make_directed_network(
    ...     direction_col="oneway",
    ...     direction_vals_bft=("B", "FT", "TF"),
    ...     minute_cols=("drivetime_fw", "drivetime_bw"),
    ... )
    >>> nw
    DirectedNetwork(6364 km, percent_bidirectional=87)

    Custom method for Norwegian road data (https://kartkatalog.geonorge.no/metadata/nvdb-ruteplan-nettverksdatasett/8d0f9066-34f9-4423-be12-8e8523089313),
    which does the same as the example above.

    >>> nw = (
    ...     DirectedNetwork(roads)
    ...     .remove_isolated()
    ...     .make_directed_network_norway()
    ...     )
    >>> nw
    DirectedNetwork(6364 km, percent_bidirectional=87)
    """

    def __init__(
        self,
        gdf: GeoDataFrame,
        **kwargs,
    ):
        """Initialiser passed to the Network base class."""
        super().__init__(gdf, **kwargs)

        # for the DirectedNetwork class, the graph will be directed in network analysis
        self._as_directed = True

    def make_directed_network(
        self,
        direction_col: str,
        direction_vals_bft: tuple[str, str, str],
        minute_cols: tuple[str, str] | str | None = None,
        speed_col: str | None = None,
        flat_speed: int | None = None,
        reverse_tofrom: bool = True,
    ):
        """Flips the line geometries of roads going backwards and in both directions.

        Args:
            direction_col: name of column specifying the direction of the line
                geometry.
            direction_vals_bft: tuple or list with the values of the direction
                column. Must be in the order 'both directions', 'from', 'to'.
                E.g. ('B', 'F', 'T').
            minute_cols (optional): column or columns containing the number of minutes
                it takes to traverse the line. If one column name is given, this will
                be used for both directions. If tuple/list with two column names,
                the first column will be used as the minute column for the forward
                direction, and the second column for roads going backwards.
            speed_col (optional): name of column with the road speed limit.
            flat_speed (optional): Speed in kilometers per hour to use as the speed for
                all roads.
            reverse_tofrom: If the geometries of the lines going backwards
                (i.e. has the last value in 'direction_vals_bft'). Defaults to True.

        Returns:
            The Network class, with the network attribute updated with flipped
                geometries for lines going backwards and both directions.
            Adds the column 'minutes' if either 'speed_col', 'minute_col' or
                'flat_speed' is specified.

        Raises:
            ValueError: If 'flat_speed' or 'speed_col' is specified and the unit of the
                coordinate reference system is not 'metre'
        """
        self._validate_minute_args(minute_cols, speed_col, flat_speed)

        self._validate_direction_args(direction_col, direction_vals_bft)

        if (flat_speed or speed_col) and not unit_is_meters(self.gdf):
            raise ValueError(
                "The crs must have 'metre' as units when calculating minutes."
                "Change crs or calculate minutes manually."
            )

        gdf = self.gdf
        b, f, t = direction_vals_bft

        if minute_cols:
            gdf = gdf.drop("minutes", axis=1, errors="ignore")

        # select the directional and bidirectional rows.
        ft = gdf.loc[gdf[direction_col] == f]
        tf = gdf.loc[gdf[direction_col] == t]
        both_ways = gdf.loc[gdf[direction_col] == b]
        both_ways2 = both_ways.copy()

        if minute_cols:
            try:
                min_f, min_t = return_two_vals(minute_cols)
            except ValueError as e:
                raise ValueError(
                    "'minute_cols' should be column name (string) or tuple/list with "
                    "values of directions forwards and backwards, in that order."
                ) from e

            # rename the two minute cols
            both_ways = both_ways.rename(columns={min_f: "minutes"}, errors="raise")
            both_ways2 = both_ways2.rename(columns={min_t: "minutes"}, errors="raise")

            ft = ft.rename(columns={min_f: "minutes"}, errors="raise")
            tf = tf.rename(columns={min_t: "minutes"}, errors="raise")

            for gdf in [ft, tf, both_ways, both_ways2]:
                if all(gdf["minutes"].fillna(0) <= 0):
                    raise ValueError("All values in minute col is NaN or less than 0.")

        both_ways2.geometry = reverse(both_ways2.geometry)

        if reverse_tofrom:
            tf.geometry = reverse(tf.geometry)

        self.gdf = pd.concat([both_ways, both_ways2, ft, tf], ignore_index=True)

        if speed_col:
            self._get_speed_from_col(speed_col)

        if flat_speed:
            self.gdf["minutes"] = self.gdf.length / flat_speed * 16.6666666667

        if "minutes" in self.gdf.columns:
            self.gdf = self.gdf.loc[self.gdf["minutes"] >= 0]

        self._make_node_ids()

        self._percent_bidirectional = self._check_percent_bidirectional()

        return self

    def make_directed_network_norway(
        self,
        direction_col: str = "oneway",
        direction_vals_bft: tuple[str, str, str] = ("B", "FT", "TF"),
        minute_cols: tuple[str, str] = ("drivetime_fw", "drivetime_bw"),
    ):
        """Runs method make_directed_network for Norwegian road data.

        Runs the method make_directed_network with default arguments to fit
        Norwegian road data:
        https://kartkatalog.geonorge.no/metadata/nvdb-ruteplan-nettverksdatasett/8d0f9066-34f9-4423-be12-8e8523089313
        """
        return self.make_directed_network(
            direction_col=direction_col,
            direction_vals_bft=direction_vals_bft,
            minute_cols=minute_cols,
        )

    def _warn_if_undirected(self):
        """Road data often have to be duplicated and flipped to make it directed."""
        if self.percent_bidirectional > 5:
            return

        mess = (
            "Your network is likely not directed. "
            f"Only {self.percent_bidirectional:.1f} percent of the lines go both ways."
        )
        if "oneway" in [col.lower() for col in self.gdf.columns]:
            mess = mess + (
                " Try setting direction_col='oneway' in the 'make_directed_network' "
                "method"
            )
        else:
            mess = mess + "Try running 'make_directed_network'"

        warnings.warn(mess, stacklevel=2)

    @staticmethod
    def _validate_minute_args(minute_cols, speed_col, flat_speed):
        if not minute_cols and not speed_col and not flat_speed:
            warnings.warn(
                "Minute column will not be calculated when both 'minute_cols', "
                "'speed_col' and 'flat_speed' is None",
                stacklevel=2,
            )

        if sum([bool(minute_cols), bool(speed_col), bool(flat_speed)]) > 1:
            raise ValueError(
                "Can only calculate minutes from either 'speed_col', "
                "'minute_cols' or 'flat_speed'."
            )

    def _validate_direction_args(self, direction_col, direction_vals_bft):
        if len(direction_vals_bft) != 3:
            raise ValueError(
                "'direction_vals_bft' should be tuple/list with values of directions "
                "both, from and to. E.g. ('B', 'F', 'T')"
            )

        gdf = self.gdf
        b, f, t = direction_vals_bft

        if not all(gdf.loc[gdf[direction_col].isin(direction_vals_bft)]):
            raise ValueError(
                f"direction_col '{direction_col}' should have only three unique",
                f"values. unique values in {direction_col}:",
                ", ".join(gdf[direction_col].fillna("nan").unique()),
                f"Got {direction_vals_bft}.",
            )

        if "b" in t.lower() and "t" in b.lower() and "f" in f.lower():
            warnings.warn(
                f"The 'direction_vals_bft' should be in the order 'both ways', "
                f"'from', 'to'. Got {b, f, t}. Is this correct?",
                stacklevel=2,
            )

    def _get_speed_from_col(self, speed_col):
        if (
            len(self.gdf.loc[(self.gdf[speed_col].isna()) | (self.gdf[speed_col] == 0)])
            > len(self.gdf) * 0.05
        ):
            raise ValueError(
                f"speed_col {speed_col!r} has a lot of missing or 0 values. Fill these "
                "with appropriate values"
            )
        self.gdf["minutes"] = (
            self.gdf.length / self.gdf[speed_col].astype(float) * 16.6666666667
        )

    def _make_directed_network_osm(
        self,
        direction_col: str = "oneway",
        direction_vals_bft: tuple[str, str, str] = ("B", "F", "T"),
        speed_col: str = "maxspeed",
    ):
        """Currently not working."""
        return self.make_directed_network(
            direction_col=direction_col,
            direction_vals_bft=direction_vals_bft,
            speed_col=speed_col,
        )

    def __repr__(self) -> str:
        """The print representation."""
        cl = self.__class__.__name__
        km = int(sum(self.gdf.length) / 1000)
        return f"{cl}({km} km, percent_bidirectional={self.percent_bidirectional})"

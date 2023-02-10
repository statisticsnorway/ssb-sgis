
import warnings
from shapely.constructive import reverse
from geopandas import GeoDataFrame
import numpy as np
from pandas import DataFrame

from .network import Network

from .gis import gdf_concat


class DirectedNetwork(Network):
    def __init__(
        self,
        roads: GeoDataFrame,
        cost: str = "minutes",
        **kwargs,
        ):
        
        super().__init__(roads, cost=cost, **kwargs)

        self.prepare_network()

        self.check_if_directed()

        self.directed = True

    def check_if_directed(self):

        no_dups = DataFrame(np.sort(self.network[["source", "target"]].values, axis=1), columns=[["source", "target"]]).drop_duplicates()
        if len(self.network) * 0.9 < len(no_dups):
            warnings.warn("""
Your network does not seem to be directed. 
Try running 'make_directed_network' or 'make_directed_network_osm'.
With 'make_directed_network', specify the direction column (e.g. 'oneway'), and the values of directions 'both', 'from', 'to' in a tuple (e.g. ("B", "F", "T")).
            """)
        else:
            self.validate_cost(raise_error=False)

    def make_directed_network_osm(
        self,
        direction_col: str = "oneway",
        direction_vals: list | tuple = ("B", "F", "T"),
        speed_col: str = "maxspeed",
        ):

        return self.make_directed_network(
            direction_col=direction_col,
            direction_vals=direction_vals,
            speed_col=speed_col,
        )

    def make_directed_network_norway(
        self,
        direction_col: str = "oneway",
        direction_vals: list | tuple = ("B", "FT", "TF"),
        minute_cols: list | tuple = ("drivetime_fw", "drivetime_bw"),
        ):

        return self.make_directed_network(
            direction_col=direction_col,
            direction_vals=direction_vals,
            minute_cols=minute_cols,
        )

    def make_directed_network(
        self,
        direction_col: str,
        direction_vals: list | tuple,
        speed_col: str | None = None,
        minute_cols: list | tuple | str | None = None,
        ):
        
        if len(direction_vals) != 3:
            raise ValueError("'direction_vals' should be tuple/list with values of directions both, forwards and backwards. E.g. ('B', 'F', 'T')")
        
        if self.cost == "minutes" and not minute_cols and not speed_col:
            raise ValueError("Must specify either 'speed_col' or 'minute_cols' when 'cost' is minutes.")

        if minute_cols and speed_col:
            raise ValueError("Can only calculate minutes from 'speed_col' or 'minute_cols', not both.")

        nw = self.network
        b, f, t = direction_vals
        
        ft = nw.loc[nw[direction_col] == f]
        tf = nw.loc[nw[direction_col] == t]
        both_ways = nw[nw[direction_col] == b]
        both_ways2 = both_ways.copy()

        tf.geometry = reverse(tf.geometry)
        both_ways2.geometry = reverse(both_ways2.geometry)
        
        if minute_cols:
            if isinstance(minute_cols, str):
                min_f, min_t = minute_cols, minute_cols
            if len(minute_cols) > 2:
                raise ValueError("'minute_cols' should be column name (string) or tuple/list with values of directions both, forwards and backwards. E.g. ('B', 'F', 'T')")
            if len(minute_cols) == 2:
                min_f, min_t = minute_cols

            both_ways = both_ways.rename(columns={min_f: "minutes"})
            both_ways2 = both_ways2.rename(columns={min_t: "minutes"})
            ft = ft.rename(columns={min_f: "minutes"})
            tf = tf.rename(columns={min_t: "minutes"})

        self.network = gdf_concat([both_ways, both_ways2, ft, tf])

        if speed_col:
            self.network["minutes"] = self.network.length / self.network[speed_col] * 16.6666666667
        
        return self

    def __repr__(self) -> str:
        return f"""
DirectedNetwork class instance with {len(self.network)} rows.
- cost: {self.cost}
- search_tolerance: {self.search_tolerance} meters
- search_factor: {self.search_factor} % + m
- cost_to_nodes: {self.cost_to_nodes} km/h
"""
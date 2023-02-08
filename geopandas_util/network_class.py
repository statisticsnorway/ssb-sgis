
import warnings
from shapely import line_merge
from shapely.constructive import reverse
from igraph import Graph
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame

"""
from .core import (
    clean_geoms,
)
from .network_functions import (
    make_node_ids,
    close_network_holes,
    find_isolated_networks,
    ZeroRoadsError,
)
from .od_cost_matrix import od_cost_matrix
"""


class NetworkAnalysis:
    def __init__(
        self,
        search_tolerance: int = 1000,
        search_factor: int = 10,
        cost_to_nodes: int = 5,
    ):
        self.search_tolerance = search_tolerance
        self.search_factor = search_factor
        self.cost_to_nodes = cost_to_nodes

    def prepare_network_analysis(self, startpoints, endpoints):
        pass
    
    def mer():
        pass


class Network(NetworkAnalysis):
    """Parent network class containing methods common for directed and undirected networks. """

    def __init__(
        self,
        roads: GeoDataFrame,
        cost: str,
        **kwargs,
        ):

        if not isinstance(roads, GeoDataFrame):
            raise TypeError(f"'roads' should be GeoDataFrame, got {type(roads)}")
        
        if not len(roads):
            raise ZeroRoadsError

        super().__init__(**kwargs)

        self.network = roads
        self.network["idx_orig"] = roads.index

        self.cost = cost

        self.make_node_ids()

    def prepare_network(self) -> None:
        """Make the necessary adjustments to the road network before network analysis can start. """
        
        if not self.network._geometry_column_name == "geometry":
            self.network = self.network.rename_geometry('geometry')
        
        self.network = (
            self.network
            .pipe(clean_geoms)
            .assign(geometry=lambda x: line_merge(x.geometry))
        )

        if not len(self.network):
            raise ZeroRoadsError

        n = len(self.network)
        self.network = self.network.explode(ignore_index=True)
        if len(self.network) < n:
            if n-len(self.network)==1:
                print(
                    f"1 multi-geometry was split into single part geometries. Minute column(s) will be wrong for these rows."
                    )
            else:
                print(
                    f"{n-len(self.network)} multi-geometries were split into single part geometries. Minute column(s) will be wrong for these rows."
                )
            
        self.network["idx"] = self.network.index

    def validate_cost(self, raise_error: bool = True) -> None:

        if self.cost in self.network.columns:

            if all(self.network[self.cost].isna()):
                raise ValueError("All values in the 'cost' column are NaN.")

            if (n := sum(self.network[self.cost].isna())):
                warnings.warn(f"Warning: {n} rows have missing values in the 'cost' column. Removing NaNs.")
                self.network = self.network.loc[self.network[self.cost].notna()]
            
            if (n := sum(self.network[self.cost] < 0)):
                warnings.warn(f"Warning: {n} rows have a 'cost' less than 0. Removing these rows.")
                self.network = self.network.loc[self.network[self.cost] > 0]

            try:
                self.network[self.cost] = self.network[self.cost].astype(float)    
            except ValueError as e:
                raise ValueError(f"There are alphabetical characters in the 'cost' column: {str(e)}")

            if "min" in self.cost:
                self.cost = "minutes"
                    
        if "meter" in self.cost or "metre" in self.cost:

            if self.network.crs == 4326:
                raise ValueError("'roads' cannot have crs 4326 (latlon) when cost is 'meters'.")

            self.cost = "meters"
            
            return

        if self.cost == "minutes" and "minutes" not in self.network.columns:
            if raise_error:
                raise KeyError(f"Cannot find 'cost' column for minutes.")
            else:
                warnings.warn("Warning: Cannot find 'cost' column for minutes. Try running one of the 'make_directed_network_' methods, or set 'cost' to 'meters'.")

    def make_node_ids(self) -> None:
        self.network, self._nodes = make_node_ids(self.network)

    def close_network_holes(self, max_dist, only_deadends=False, min_dist=0, hole_col = "hole") -> None:
        close_network_holes(self.network, max_dist, only_deadends, min_dist=0, hole_col = "hole")
        return self
        roads, max_dist, min_dist=0, deadends_only=False, hole_col: str | None = "hole"

    def find_isolated(self, max_length: int, remove=False) -> None:
        if "isolated" in self.network.columns:
            warning.warn("Warning: there is already a column named 'isolated' in your network. Try running .remove_isolated(max_length) to remove the isolated networks.")
        self.network = find_isolated_networks(self.network, max_length)
        if remove:
            self.network = self.network.loc[self.network.isolated == 0]

        return self

    def remove_isolated(self, max_length: int = None) -> None:
        if not "isolated" in self.network.columns:
            if not max_length:
                raise ValueError("'max_length' has to be specified when there is no column 'isolated' in the network.")
            self.network = find_isolated_networks(self.network, max_length)
        self.network = self.network.loc[self.network.isolated == 0]

        return self

    def cut_lines(self) -> None:
        pass
    
    def prepare_network_analysis(self, startpoints, endpoints=None, id_col: str | None = None) -> None:

        self.validate_cost(raise_error=True)

        self.validate_points(startpoints, endpoints, id_col)

        startpoints, endpoints = self.prepare_points(startpoints, endpoints)
        
        if isinstance(id_col, str):
            id_col = id_col, id_col

        if self.graph_is_up_to_date(startpoints, endpoints):
            if endpoints is None:
                return startpoints, id_col
            return startpoints, endpoints, id_col
                    
        self.make_node_ids()

        startpoints, endpoints = self.prepare_points(startpoints, endpoints)

        self.graph = self.make_graph(startpoints, endpoints)
        
        self.update_graph_info(startpoints, endpoints)
        
        if endpoints is None:
            return startpoints, id_col
        
        return startpoints, endpoints, id_col

    def graph_is_up_to_date(self, startpoints, endpoints):

        if not hasattr(self, "graph"):
            return False
    
        if any(
            True if x not in self.graph.vs["name"] else False for x in startpoints.temp_idx
            ):
            return False

        if endpoints is not None:
            if any(
                True if x not in self.graph.vs["name"] else False for x in endpoints.temp_idx
                ):
                return False
            if self._temp_idx_end_max == max(endpoints.temp_idx):
                return True
                        
        if self._network_len == len(self.network):
            return True
        
        if self._temp_idx_start_max == max(startpoints.temp_idx):
            return True
            
        idx = self.network.index
        if not isinstance(idx, RangeIndex):
            return False
        if idx.start != 0:
            return False
        if idx.stop != len(self.network):
            return False
        
        return True

    def update_graph_info(self, startpoints, endpoints):

        self._network_len = len(self.network)

        self._temp_idx_start_max = max(startpoints.temp_idx)

        if endpoints is not None:
            self._temp_idx_end_max = max(endpoints.temp_idx)

    def make_graph(
        self,
        startpoints: GeoDataFrame,
        endpoints: GeoDataFrame | None = None,
        ) -> Graph:

        return make_graph(self, startpoints, endpoints)

    def od_cost_matrix(
        self, 
        startpoints: GeoDataFrame, 
        endpoints: GeoDataFrame,
        id_col: str | list | tuple = None, 
        **kwargs
        ) -> DataFrame | GeoDataFrame:

        startpoints, endpoints, id_cols = self.prepare_network_analysis(
            startpoints, endpoints, id_col
        )

        res = od_cost_matrix(self, startpoints, endpoints, **kwargs)

        if id_col:
            res["origin"] = self.map_ids(
                res["origin"], startpoints, id_cols[0],
            )
            res["destination"] = self.map_ids(
                res["destination"], endpoints, id_cols[1],
            )

        return res

    def shortest_path(
        self, 
        startpoints: GeoDataFrame, 
        endpoints: GeoDataFrame,
        id_col: str | list | tuple = None, 
        **kwargs
        ) -> GeoDataFrame:

        startpoints, endpoints, id_cols = self.prepare_network_analysis(
            startpoints, endpoints, id_col
        )

        res = shortest_path(self, startpoints, endpoints, **kwargs)

        res["from"] = self.map_ids(
            res["from"], startpoints[id_cols[0]]
        ) 
        res["to"] = self.map_ids(
            res["to"], endpoints[id_cols[1]]
        )

        return res

    def service_area(
        self, 
        startpoints: GeoDataFrame, 
        id_col: str | list | tuple = None, 
        **kwargs
        ) -> GeoDataFrame:

        startpoints, id_cols = self.prepare_network_analysis(
            startpoints, id_col
        )

        res = service_area(self, startpoints, **kwargs)

        res["from"] = self.map_ids(
            res["from"], startpoints[id_cols[0]]
        )

        return res

    def make_temp_ids(self, points, plus=0):
        """
        Lager id-kolonne som brukes som node-id-er i igraph.Graph().
        Fordi start- og sluttpunktene må ha node-id-er som ikke finnes i networket.
        """
        start = max(self.nodes.node_id.astype(int)) + 1 + plus
        stop = start + len(points)
        return [str(idx) for idx in np.arange(start, stop)]

    def map_ids(self, col, points, id_col):
        """From temp to original ids."""

        id_dict = {
            temp_idx: idx
            for temp_idx, idx in zip(points["temp_idx"], points[id_col])
        }

        return col.map(id_dict)
        
    @staticmethod
    def validate_points(
        startpoints: GeoDataFrame,
        endpoints: GeoDataFrame | None = None,
        id_col: str | list | tuple = None,
        ) -> None: 

        if isinstance(id_col, str):
            if not id_col in startpoints.columns:
                raise KeyError(f"'startpoints' has no attribute '{id_col}'")
            if endpoints is not None:
                if not id_col in endpoints.columns:
                    raise KeyError(f"'endpoints' has no attribute '{id_col}'")
        elif isinstance(id_col, (list, tuple)):
            if not id_col[0] in startpoints.columns:
                raise KeyError(f"'startpoints' has no attribute '{id_col[0]}'")
            if endpoints is None:
                warnings.warn(f"'id_col' is of type {type(id_col)} even though there are no endpoints")
            else:
                if not id_col[1] in endpoints.columns:
                    raise KeyError(f"'endpoints' has no attribute '{id_col[1]}'")
        
    def prepare_points(self, startpoints, endpoints):
        startpoints = startpoints.to_crs(self.network.crs)
        startpoints["temp_idx"] = self.make_temp_ids(startpoints)

        if endpoints is not None:
            endpoints = endpoints.to_crs(self.network.crs)
            endpoints["temp_idx"] = self.make_temp_ids(endpoints, plus=len(startpoints))
        
        return startpoints, endpoints

    @property
    def nodes(self):
        return self._nodes

    def copy(self):
        return copy(self)

    def deepcopy(self):
        return deepcopy(self)


class UndirectedNetwork(Network):
    def __init__(
        self,
        roads: GeoDataFrame,
        cost: str = "meters",
        **network_analysis_rules,
        ):

        super().__init__(roads, cost)

        self.prepare_network()

        self.validate_cost(raise_error=True)


class DirectedNetwork(Network):
    def __init__(
        self,
        roads: GeoDataFrame,
        cost: str = "minutes",
        **network_analysis_rules,
        ):
        
        super().__init__(roads, cost)

        self.prepare_network()

        self.check_if_directed()

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
        ) -> DirectedNetwork:

        return self.make_directed_network(
            direction_col=direction_col,
            direction_vals=direction_vals,
            speed_col=speed_col,
        )

    def make_directed_network_norway(
        self,
        source_col: str = "fromnode",
        target_col: str = "tonode",
        direction_col: str = "oneway",
        direction_vals: list | tuple = ("B", "FT", "TF"),
        minute_cols: list | tuple = ("drivetime_fw", "drivetime_bw"),
        ) -> DirectedNetwork:

        return self.make_directed_network(
            source_col=source_col,
            target_col=target_col,
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
        source_col: str | None = None,
        target_col: str | None = None,
        ) -> DirectedNetwork:
        
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
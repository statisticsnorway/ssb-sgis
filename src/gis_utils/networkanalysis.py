from datetime import datetime
from time import perf_counter

import igraph
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from igraph import Graph
from pandas import DataFrame

from .directednetwork import DirectedNetwork
from .geopandas_utils import gdf_concat, push_geom_col
from .get_route import _get_route
from .network import Network
from .network_functions import make_node_ids, split_lines_at_closest_point
from .networkanalysisrules import NetworkAnalysisRules
from .od_cost_matrix import _od_cost_matrix
from .points import Destinations, Origins
from .service_area import _service_area


class NetworkAnalysis:
    """Class that holds the actual network analysis methods.

    It takes a (Directed)Network and rules (NetworkAnalysisRules).

    Args:
        network: either the base Network class or a subclass, chiefly the DirectedNetwork class.
            The network should be customized beforehand, but can also be accessed through
            the 'network' attribute of this class.
        rules: NetworkAnalysisRules class instance.
        log: If True (the default), a DataFrame with information about each analysis run
            will be stored in the 'log' attribute.
        detailed_log: If True (the default), will include all arguments passed to the
            analysis methods and the standard deviation, 25th, 50th and 75th percentile
            of the weight column in the results.

    Attributes:
        network: the Network instance
        rules: the NetworkAnalysisRules instance
        log: A DataFrame with information about each analysis run
        origins: the origins used in the latest analysis run,
            contained in a origins class instance, with the GeoDataFrame stored
            in the 'gdf' attribute.

    Examples:

    roads = gpd.read_parquet(filepath_roads)
    points = gpd.read_parquet(filepath_points)

    nw = (
        DirectedNetwork(roads)
        .make_directed_network_osm()
        .remove_isolated()
        )

    nwa = NetworkAnalysis(nw, weight="minutes")

    od = nwa.od_cost_matrix(points, points)

    """

    def __init__(
        self,
        network: Network | DirectedNetwork,
        rules: NetworkAnalysisRules,
        log: bool = True,
        detailed_log: bool = True,
    ):
        self.network = network
        self.rules = rules
        self._log = log
        self.detailed_log = detailed_log

        if not isinstance(rules, NetworkAnalysisRules):
            raise ValueError(
                f"'rules' should be of type NetworkAnalysisRules. Got {type(rules)}"
            )

        if not isinstance(network, Network):
            raise ValueError(
                f"'network' should of type DirectedNetwork or Network. Got {type(network)}"
            )

        self.network.gdf = self.rules._validate_weight(
            self.network.gdf, raise_error=False
        )

        if isinstance(self.network, DirectedNetwork):
            self.network._warn_if_undirected()

        self._update_wkts()
        self.rules._update_rules()

        if log:
            self.log = DataFrame()

    def od_cost_matrix(
        self,
        origins: GeoDataFrame,
        destinations: GeoDataFrame,
        id_col: str | tuple[str, str] | None = None,
        *,
        lines=False,
        rowwise=False,
        cutoff: int = None,
        destination_count: int = None,
    ) -> DataFrame | GeoDataFrame:
        """Fast calculation of many-to-many travel costs

        Args:
            origins: GeoDataFrame of points from where the trips will originate
            destinations: GeoDataFrame of points from where the trips will terminate
            id_col: column(s) to be used as identifier for the origins and destinations.
                If two different columns, put it in a tuple as ("origin_col", "destination_col")
                If None, an arbitrary id will be returned.
            lines: if True, returns a geometry column with straight lines between
                origin and destination. Defaults to False.
            rowwise: if False (the default), it will calculate the cost from each origins
                to each destination. If true, it will calculate the cost from origin 1 to destination 1,
                origin 2 to destination 2 and so on.
            cutoff: the maximum cost (weight) for the trips. Defaults to None,
                meaning all rows will be included. NaNs will also be removed if cutoff
                is specified.
            destination_count: number of closest destinations to keep for each origin.
                If None (the default), all trips will be included. The number of destinations
                might be higher than the destination count if trips have equal cost.

        Returns:
            A DataFrame with the columns 'origin', 'destination' and the weight column.
            If lines is True, adds a geometry column with straight lines between origin
            and destination.

        """

        if self._log:
            time_ = perf_counter()

        self._prepare_network_analysis(origins, destinations, id_col)

        results = _od_cost_matrix(
            graph=self.graph,
            origins=self.origins.gdf,
            destinations=self.destinations.gdf,
            weight=self.rules.weight,
            lines=lines,
            cutoff=cutoff,
            destination_count=destination_count,
            rowwise=rowwise,
        )

        self.origins._get_n_missing(results, "origin")
        self.destinations._get_n_missing(results, "destination")

        if id_col:
            results["origin"] = results["origin"].map(self.origins.id_dict)
            results["destination"] = results["destination"].map(
                self.destinations.id_dict
            )

        if lines:
            results = push_geom_col(results)

        if self._log:
            minutes_elapsed_ = round((perf_counter() - time_) / 60, 1)
            self._runlog(
                "od_cost_matrix",
                results,
                minutes_elapsed_,
                lines=lines,
                cutoff=cutoff,
                destination_count=destination_count,
                rowwise=rowwise,
            )

        return results

    def get_route(
        self,
        origins: GeoDataFrame,
        destinations: GeoDataFrame,
        id_col: str | tuple[str, str] | None = None,
        *,
        rowwise=False,
        cutoff: int = None,
        destination_count: int = None,
    ) -> GeoDataFrame:
        """Returns the geometry of the low-cost route between origins and destinations

        Finds the route with the lowest cost (minutes, meters, etc.) from a set of origins
        to a set of destinations. If the weight is meters, the shortest route will be found.
        If the weight is minutes, the fastest route will be found.

        Args:
            origins: GeoDataFrame of points from where the routes will originate
            destinations: GeoDataFrame of points from where the routes will terminate
            id_col: optional column to be used as identifier of the service areas. If None,
                an arbitrary id will be used.
            rowwise: if False (the default), it will calculate the cost from each origins
                to each destination. If true, it will calculate the cost from origin 1 to destination 1,
                origin 2 to destination 2 and so on.
            cutoff: the maximum cost (weight) for the trips. Defaults to None,
                meaning all rows will be included. NaNs will also be removed if cutoff
                is specified.
            destination_count: number of closest destinations to keep for each origin.
                If None (the default), all trips will be included. The number of destinations
                might be higher than the destination count if trips have equal cost.

        Returns:
            A GeoDataFrame with the columns 'origin', 'destination', the weight
            column and the geometry of the route between origin and destination.

        Raises:
            ValueError if no paths were found.
        """
        if self._log:
            time_ = perf_counter()

        self._prepare_network_analysis(origins, destinations, id_col)

        results = _get_route(
            graph=self.graph,
            origins=self.origins.gdf,
            destinations=self.destinations.gdf,
            weight=self.rules.weight,
            roads=self.network.gdf,
            cutoff=cutoff,
            destination_count=destination_count,
            rowwise=rowwise,
        )

        self.origins._get_n_missing(results, "origin")
        self.destinations._get_n_missing(results, "destination")

        if id_col:
            results["origin"] = results["origin"].map(self.origins.id_dict)
            results["destination"] = results["destination"].map(
                self.destinations.id_dict
            )

        results = push_geom_col(results)

        if self._log:
            minutes_elapsed_ = round((perf_counter() - time_) / 60, 1)
            self._runlog(
                "get_route",
                results,
                minutes_elapsed_,
                cutoff=cutoff,
                destination_count=destination_count,
                rowwise=rowwise,
            )

        return results

    def get_route_frequencies(
        self,
        origins: GeoDataFrame,
        destinations: GeoDataFrame,
    ) -> GeoDataFrame:
        """Finds the number of times each line segment was visited in all trips

        Finds the route with the lowest cost (minutes, meters, etc.) from a set of origins
        to a set of destinations. If the weight is meters, the shortest route will be found.
        If the weight is minutes, the fastest route will be found.

        Args:
            origins: GeoDataFrame of points from where the routes will originate
            destinations: GeoDataFrame of points from where the routes will terminate

        Returns:
            A GeoDataFrame with all line segments that were visited at least once,
            with the column 'n', which is the number of times the segment was visited
            for all the trips.

        Raises:
            ValueError if no paths were found.
        """
        if self._log:
            time_ = perf_counter()

        self._prepare_network_analysis(origins, destinations, None)

        results = _get_route(
            graph=self.graph,
            origins=self.origins.gdf,
            destinations=self.destinations.gdf,
            weight=self.rules.weight,
            roads=self.network.gdf,
            summarise=True,
        )

        results = push_geom_col(results)

        if self._log:
            minutes_elapsed_ = round((perf_counter() - time_) / 60, 1)
            self._runlog(
                "get_route_frequencies",
                results,
                minutes_elapsed_,
            )

        return results

    def service_area(
        self,
        origins: GeoDataFrame,
        breaks: int | float | list[int | float] | tuple[int | float],
        *,
        id_col: str | None = None,
        dissolve: bool = True,
    ) -> GeoDataFrame:
        """Returns the lines that can be reached within breaks (weight values)

        It finds all the network lines that can be reached within each weight impedance,
        given in the breaks argument as one or more integers/floats.

        Args:
            origins: GeoDataFrame of points from where the service areas will originate
            breaks: one or more integers or floats which will be the maximum weight for
                the service areas. Calculates multiple areas for each origins if
                multiple breaks.
            id_col: optional column to be used as identifier of the service areas. If None,
                an arbitrary id will be used.
            dissolve: If True (the default), each service area will be dissolved into one long
                multilinestring. If False, the individual line segments will be returned. Duplicate
                lines can then be removed, or occurences counted.

        Returns:
            A GeoDataFrame with the roads that can be reached within the break
            for each origin. If dissolve is False, the columns will be the weight
            column, which contains the relevant break, and the if_col if specified,
            or the column 'origin' if not. If dissolve is False, it will return all
            the columns of the network.gdf as well. The columns 'source' and 'target'
            can be used to remove duplicates, or count occurences.

        """

        if self._log:
            time_ = perf_counter()

        self._prepare_network_analysis(origins, id_col=id_col)

        results = _service_area(
            graph=self.graph,
            origins=self.origins.gdf,
            weight=self.rules.weight,
            lines=self.network.gdf,
            breaks=breaks,
            dissolve=dissolve,
        )

        # add missing rows as NaNs
        missing = self.origins.gdf.loc[
            ~self.origins.gdf["temp_idx"].isin(results["origin"])
        ].rename(columns={"temp_idx": "origin"})[["origin"]]

        if len(missing):
            missing["geometry"] = np.nan
            results = pd.concat([results, missing], ignore_index=True)

        if id_col:
            results[id_col] = results["origin"].map(self.origins.id_dict)
            results = results.drop("origin", axis=1)

        results = push_geom_col(results)

        if self._log:
            minutes_elapsed_ = round((perf_counter() - time_) / 60, 1)
            self._runlog(
                "service_area",
                results,
                minutes_elapsed_,
                breaks=breaks,
                dissolve=dissolve,
            )

        return results

    def _log_df_template(self, fun: str, minutes_elapsed: int) -> DataFrame:
        """Creates a df with one row
        The 'isolated_removed' column does not account for
        preperation done before initialising the (Directed)Network class.
        """

        df = DataFrame(
            {
                "endtime": pd.to_datetime(datetime.now()).floor("S").to_pydatetime(),
                "minutes_elapsed": minutes_elapsed,
                "function": fun,
                "origins_count": np.nan,
                "destinations_count": np.nan,
                "percent_missing": np.nan,
                "cost_mean": np.nan,
                "isolated_removed": self.network._isolated_removed,
                "percent_directional": self.network._percent_directional,
            },
            index=[0],
        )

        for key, value in self.rules.__dict__.items():
            if key.startswith("_") or key.endswith("_"):
                continue
            df = pd.concat([df, pd.DataFrame({key: [value]})], axis=1)

        return df

    def _runlog(
        self,
        fun: str,
        results: DataFrame | GeoDataFrame,
        minutes_elapsed: int,
        **kwargs,
    ) -> None:
        df = self._log_df_template(fun, minutes_elapsed)

        df["origins_count"] = len(self.origins.gdf)

        if self.rules.weight in results.columns:
            df["percent_missing"] = results[self.rules.weight].isna().mean() * 100
            df["cost_mean"] = results[self.rules.weight].mean()
            if self.detailed_log:
                df["cost_p25"] = results[self.rules.weight].quantile(0.25)
                df["cost_median"] = results[self.rules.weight].median()
                df["cost_p75"] = results[self.rules.weight].quantile(0.75)
                df["cost_std"] = results[self.rules.weight].std()

        if fun == "service_area":
            df["percent_missing"] = results["geometry"].isna().mean() * 100
        else:
            df["destinations_count"] = len(self.destinations.gdf)

        if self.detailed_log:
            for key, value in kwargs.items():
                if isinstance(value, np.ndarray):
                    value = list(value)
                if isinstance(value, (list, tuple)):
                    value = [str(x) for x in value]
                    value = ", ".join(value)
                df[key] = value

        self.log = pd.concat([self.log, df], ignore_index=True)

    def _prepare_network_analysis(
        self, origins, destinations=None, id_col: str | None = None
    ) -> None:
        """Prepares the weight column, node ids and origins and destinations.
        Also updates the graph if it is not yet created and no parts of the analysis has changed.
        this method is run inside od_cost_matrix, get_route and service_area.
        """

        self.network.gdf = self.rules._validate_weight(
            self.network.gdf, raise_error=True
        )

        self.origins = Origins(
            origins,
            id_col=id_col,
            temp_idx_start=max(self.network.nodes.node_id.astype(int)) + 1,
        )

        if destinations is not None:
            self.destinations = Destinations(
                destinations,
                id_col=id_col,
                temp_idx_start=max(self.origins.gdf.temp_idx.astype(int)) + 1,
            )

        else:
            self.destinations = None

        if not self._graph_is_up_to_date() or not self.network._nodes_are_up_to_date():
            self.network._update_nodes_if()

            edges, weights = self._get_edges_and_weights()

            self.graph = self._make_graph(
                edges=edges, weights=weights, directed=self.network.directed
            )

            self._add_missing_vertices()

        self._update_wkts()
        self.rules._update_rules()

    def _get_edges_and_weights(self) -> tuple[list[tuple[str, ...]], list[float]]:
        """Creates lists of edges and weights which will be used to make the graph.
        Edges and weights between origins and nodes and nodes and destinations are also added.
        """

        if self.rules.split_lines:
            if self.destinations is not None:
                points = gdf_concat([self.origins.gdf, self.destinations.gdf])
            else:
                points = self.origins.gdf

            points = points.drop_duplicates("geometry")

            self.network.gdf["meters"] = self.network.gdf.length

            lines = split_lines_at_closest_point(
                lines=self.network.gdf,
                points=points,
                max_dist=self.rules.search_tolerance,
            )

            # adjust the weight to new splitted length
            lines.loc[lines["splitted"] == 1, self.rules.weight] = lines[
                self.rules.weight
            ] * (lines.length / lines["meters"])

            self.network.gdf = lines
            self.network._make_node_ids()

            # remake the temp_idx
            # TODO: consider changing how this thing works
            self.origins.temp_idx_start = (
                max(self.network.nodes.node_id.astype(int)) + 1
            )
            self.origins._make_temp_idx()
            if self.destinations is not None:
                self.destinations.temp_idx_start = (
                    max(self.origins.gdf.temp_idx.astype(int)) + 1
                )
                self.destinations._make_temp_idx()

        edges = [
            (str(source), str(target))
            for source, target in zip(
                self.network.gdf["source"], self.network.gdf["target"]
            )
        ]

        weights = list(self.network.gdf[self.rules.weight])

        edges_start, weights_start = self.origins._get_edges_and_weights(
            nodes=self.network.nodes, rules=self.rules
        )
        edges = edges + edges_start
        weights = weights + weights_start

        if self.destinations is None:
            return edges, weights

        edges_end, weights_end = self.destinations._get_edges_and_weights(
            nodes=self.network.nodes, rules=self.rules
        )

        edges = edges + edges_end
        weights = weights + weights_end

        return edges, weights

    def _add_missing_vertices(self):
        """Adds the points that had no nodes within the search_tolerance
        to the graph. To prevent error when running the distance calculation.
        """
        # TODO: either check if any() beforehand, or add fictional edges before making the graph,
        # to make things faster (this method took 64.660 out of 500 seconds)
        self.graph.add_vertices(
            [
                idx
                for idx in self.origins.gdf["temp_idx"]
                if idx not in self.graph.vs["name"]
            ]
        )
        if self.destinations is not None:
            self.graph.add_vertices(
                [
                    idx
                    for idx in self.destinations.gdf["temp_idx"]
                    if idx not in self.graph.vs["name"]
                ]
            )

    @staticmethod
    def _make_graph(
        edges: list[tuple[str, ...]] | np.ndarray[tuple[str, ...]],
        weights: list[float] | np.ndarray[float],
        directed: bool,
    ) -> Graph:
        """Creates an igraph Graph from a list of edges and weights."""

        assert len(edges) == len(weights)

        graph = igraph.Graph.TupleList(edges, directed=directed)

        graph.es["weight"] = weights

        assert min(graph.es["weight"]) >= 0

        return graph

    def _graph_is_up_to_date(self) -> bool:
        """Returns False if the rules of the graphmaking has changed,
        or if the points have changed"""

        if not hasattr(self, "graph") or not hasattr(self, "wkts"):
            return False

        if self.rules._rules_have_changed():
            return False

        for points in ["origins", "destinations"]:
            if not hasattr(self.wkts, points):
                return False
            if self._points_have_changed(self[points].gdf, what=points):
                return False

        return True

    def _points_have_changed(self, points: GeoDataFrame, what: str) -> bool:
        """This method is best stored in the NetworkAnalysis class,
        since the point classes are instantiated each time an analysis is run."""
        if self.wkts[what] != [geom.wkt for geom in points.geometry]:
            return True

        if not all(x in self.graph.vs["name"] for x in list(points.temp_idx.values)):
            return True

        return False

    def _update_wkts(self) -> None:
        """Creates a dict of wkt lists. This method is run after the graph is created.
        If the wkts haven't updated since the last run, the graph doesn't have to be remade.
        """
        self.wkts = {}

        self.wkts["network"] = [geom.wkt for geom in self.network.gdf.geometry]

        if not hasattr(self, "origins"):
            return

        self.wkts["origins"] = [geom.wkt for geom in self.origins.gdf.geometry]

        if self.destinations is not None:
            self.wkts["destinations"] = [
                geom.wkt for geom in self.destinations.gdf.geometry
            ]

    def __repr__(self) -> str:
        """The print representation"""
        # remove 'weight_to_nodes_' arguments in the repr of the NetworkAnalysisRules instance
        rules = self.rules.__repr__()
        for txt in ["weight_to_nodes_", "dist", "kmh", "mph", "=None", "=False"]:
            rules = rules.replace(txt, "")
        for txt in [", )"] * 5:
            rules = rules.replace(txt, ")")
        rules = rules.strip(")")

        # add a 'weight_to_nodes_' argument if used,
        # else inform that there are more with some '...'
        if self.rules.weight_to_nodes_dist:
            x = f", weight_to_nodes_dist={self.rules.weight_to_nodes_dist}"
        elif self.rules.weight_to_nodes_kmh:
            x = f", weight_to_nodes_dist={self.rules.weight_to_nodes_kmh}"
        elif self.rules.weight_to_nodes_mph:
            x = f", weight_to_nodes_dist={self.rules.weight_to_nodes_mph}"
        else:
            x = ", ..."

        return (
            f"{self.__class__.__name__}("
            f"network={self.network.__repr__()}, "
            f"rules={rules}{x}))"
        )

    def __getitem__(self, item):
        return getattr(self, item)

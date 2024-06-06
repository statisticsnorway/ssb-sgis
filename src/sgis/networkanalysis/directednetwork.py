"""Prepare a GeoDataFrame of line geometries for directed network analysis."""

import warnings
from collections.abc import Sequence

import pandas as pd
from geopandas import GeoDataFrame
from shapely.constructive import reverse

from ..helpers import return_two_vals
from ..helpers import unit_is_meters


def make_directed_network_norway(gdf: GeoDataFrame, dropnegative: bool) -> GeoDataFrame:
    """Runs the method make_directed_network for Norwegian road data.

    The parameters in make_directed_network are set to the correct values for
    Norwegian road data as of 2023.

    The data can be downloaded here:
    https://kartkatalog.geonorge.no/metadata/nvdb-ruteplan-nettverksdatasett/8d0f9066-34f9-4423-be12-8e8523089313

    Args:
        gdf: Road GeoDataFrame.
        dropnegative: Whether to keep rows with negative minute values in both directions.
            These are mostly circles with no impact on the analyses, but there might be
            exceptions. Keeping negative values will result in an error when building the
            network graph. Recode these rows to a non-negative values if you want
            to keep them.

    Examples:
    ---------
    2022 data for the municipalities of Oslo and Eidskog can be read directly like this:

    >>> import sgis as sg
    >>> roads = sg.read_parquet_url(
    ...     "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet"
    ... )
    >>> roads[["oneway", "drivetime_fw", "drivetime_bw", "geometry"]]
            oneway  drivetime_fw  drivetime_bw                                           geometry
    119702       B      0.216611      0.216611  MULTILINESTRING Z ((258028.440 6674249.890 413...
    199710      FT      0.099323     -1.000000  MULTILINESTRING Z ((271778.700 6653238.900 138...
    199725      FT      0.173963     -1.000000  MULTILINESTRING Z ((271884.510 6653207.540 142...
    199726      FT      0.011827     -1.000000  MULTILINESTRING Z ((271884.510 6653207.540 142...
    199733      FT      0.009097     -1.000000  MULTILINESTRING Z ((271877.373 6653214.697 141...
    ...        ...           ...           ...                                                ...
    1944129     FT      0.010482     -1.000000  MULTILINESTRING Z ((268649.503 6651869.320 111...
    1944392      B      0.135561      0.135561  MULTILINESTRING Z ((259501.570 6648459.580 3.2...
    1944398      B      0.068239      0.068239  MULTILINESTRING Z ((258292.600 6648313.440 18....
    1944409      B      0.023629      0.023629  MULTILINESTRING Z ((258291.452 6648289.258 19....
    1944415      B      0.175876      0.175876  MULTILINESTRING Z ((260762.830 6650240.620 43....
    <BLANKLINE>
    [93395 rows x 4 columns]

    And converted to a directed network like this:

    >>> roads_directed = sg.make_directed_network_norway(roads, dropnegative=True)
    >>> roads_directed[["minutes", "geometry"]]
             minutes                                           geometry
    0       0.216611  MULTILINESTRING Z ((258028.440 6674249.890 413...
    1       0.028421  MULTILINESTRING Z ((266382.600 6639604.600 -99...
    2       0.047592  MULTILINESTRING Z ((266349.200 6639372.499 171...
    3       0.026180  MULTILINESTRING Z ((266360.515 6639120.493 172...
    4       0.023978  MULTILINESTRING Z ((266351.263 6639416.154 169...
    ...          ...                                                ...
    175620  0.007564  MULTILINESTRING Z ((268641.225 6651871.624 111...
    175621  0.020246  MULTILINESTRING Z ((268669.748 6651890.291 110...
    175622  0.036810  MULTILINESTRING Z ((268681.757 6651886.457 110...
    175623  0.003019  MULTILINESTRING Z ((268682.748 6651886.162 110...
    175624  0.036975  MULTILINESTRING Z ((268694.594 6651881.688 111...
    <BLANKLINE>
    [175541 rows x 2 columns]
    """
    if gdf["drivetime_fw"].isna().any():
        raise ValueError("Missing values in the columns 'drivetime_fw'")
    if gdf["drivetime_bw"].isna().any():
        raise ValueError("Missing values in the columns 'drivetime_bw'")

    return make_directed_network(
        gdf,
        direction_col="oneway",
        direction_vals_bft=("B", "FT", "TF"),
        minute_cols=("drivetime_fw", "drivetime_bw"),
        dropnegative=dropnegative,
        dropna=False,
    )


def make_directed_network(
    gdf: GeoDataFrame,
    direction_col: str,
    direction_vals_bft: tuple[str, str, str],
    dropna: bool | None = None,
    dropnegative: bool | None = None,
    minute_cols: tuple[str, str] | str | None = None,
    speed_col_kmh: str | None = None,
    flat_speed_kmh: int | None = None,
    reverse_tofrom: bool = True,
) -> GeoDataFrame:
    """Flips the line geometries of roads going backwards and in both directions.

    Args:
        gdf: GeoDataFrame of line geometries.
        direction_col: name of column specifying the direction of the line
            geometry.
        direction_vals_bft: tuple or list with the values of the direction
            column. Must be in the order 'both directions', 'from', 'to'.
            E.g. ('B', 'F', 'T').
        dropna: When minutes_cols is set, whether to drop rows with missing values
            in both columns. Alternatively, set these rows to a positive value before
            making the directed network.
        dropnegative: When minutes_cols is set, whether to drop rows with negative values
            in both columns. Not dropping them will cause an error when building a
            network graph. Alternatively, set these rows to a positive value before
            making the directed network.
        minute_cols (optional): column or columns containing the number of minutes
            it takes to traverse the line. If one column name is given, this will
            be used for both directions. If tuple/list with two column names,
            the first column will be used as the minute column for the forward
            direction, and the second column for roads going backwards.
        speed_col_kmh (optional): name of column with the road speed limit.
        flat_speed_kmh (optional): Speed in kilometers per hour to use as the speed for
            all roads.
        reverse_tofrom: If the geometries of the lines going backwards
            (i.e. has the last value in 'direction_vals_bft'). Defaults to True.

    Returns:
        The Network class, with the network attribute updated with flipped
            geometries for lines going backwards and both directions.
        Adds the column 'minutes' if either 'speed_col_kmh', 'minute_col' or
            'flat_speed_kmh' is specified.

    Raises:
        ValueError: If 'flat_speed_kmh' or 'speed_col_kmh' is specified and the unit of
            the coordinate reference system is not 'metre'.
    """
    _validate_minute_args(minute_cols, speed_col_kmh, flat_speed_kmh)
    _validate_direction_args(gdf, direction_col, direction_vals_bft)

    if minute_cols is not None and any(x is None for x in [dropnegative, dropna]):
        raise ValueError(
            "Must specify 'dropna' and 'dropnegative' when 'minute_cols' is given."
        )

    if speed_col_kmh is not None and any(x is None for x in [dropnegative, dropna]):
        raise ValueError(
            "Must specify 'dropna' and 'dropnegative' when 'speed_col_kmh' is given."
        )

    if (flat_speed_kmh or speed_col_kmh) and not unit_is_meters(gdf):
        raise ValueError(
            "The crs must have 'metre' as units when calculating minutes."
            "Change crs or calculate minutes manually."
        )

    b, f, t = direction_vals_bft

    if minute_cols and minute_cols != "minutes" and minute_cols[0] != "minutes":
        gdf = gdf.drop("minutes", axis=1, errors="ignore")

    if minute_cols:
        try:
            min_f, min_t = return_two_vals(minute_cols)
        except ValueError as e:
            raise ValueError(
                "'minute_cols' should be column name (string) or tuple/list with "
                "values of directions forwards and backwards, in that order."
            ) from e

        if dropna:
            gdf = gdf.loc[~((gdf[min_f].isna()) & (gdf[min_t].isna()))]
        if dropnegative:
            gdf = gdf.loc[~((gdf[min_f] < 0) & (gdf[min_t] < 0))]

    # select the directional and bidirectional rows.
    ft = gdf.loc[gdf[direction_col] == f]
    tf = gdf.loc[gdf[direction_col] == t]
    both_ways = gdf.loc[gdf[direction_col] == b]
    both_ways2 = both_ways.copy()

    if minute_cols:
        # to single minute column
        both_ways = both_ways.rename(columns={min_f: "minutes"}, errors="raise")
        both_ways2 = both_ways2.rename(columns={min_t: "minutes"}, errors="raise")

        ft = ft.rename(columns={min_f: "minutes"}, errors="raise")
        tf = tf.rename(columns={min_t: "minutes"}, errors="raise")

    both_ways2.geometry = reverse(both_ways2.geometry)

    if reverse_tofrom:
        tf.geometry = reverse(tf.geometry)

    gdf = pd.concat([both_ways, both_ways2, ft, tf], ignore_index=True)

    if minute_cols and minute_cols != "minutes" and minute_cols[0] != "minutes":
        gdf = gdf.drop([min_f, min_t], axis=1, errors="ignore")

    if speed_col_kmh:
        gdf = _get_speed_from_col(gdf, speed_col_kmh)

    if flat_speed_kmh:
        meters_per_min = (flat_speed_kmh / 60) * 1000
        gdf["minutes"] = gdf.length / meters_per_min

    return gdf


def _validate_minute_args(
    minute_cols: Sequence[str], speed_col_kmh: str, flat_speed_kmh: str
) -> None:
    if not minute_cols and not speed_col_kmh and not flat_speed_kmh:
        warnings.warn(
            "Minute column will not be calculated when both 'minute_cols', "
            "'speed_col_kmh' and 'flat_speed_kmh' is None",
            stacklevel=2,
        )

    if sum([bool(minute_cols), bool(speed_col_kmh), bool(flat_speed_kmh)]) > 1:
        raise ValueError(
            "Can only calculate minutes from either 'speed_col_kmh', "
            "'minute_cols' or 'flat_speed_kmh'."
        )


def _validate_direction_args(
    gdf: GeoDataFrame, direction_col: str, direction_vals_bft: Sequence[str]
) -> None:
    if len(direction_vals_bft) != 3:
        raise ValueError(
            "'direction_vals_bft' should be tuple/list with values of directions "
            "both, from and to. E.g. ('B', 'F', 'T')"
        )

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


def _get_speed_from_col(gdf: GeoDataFrame, speed_col_kmh: str) -> GeoDataFrame:
    if len(gdf.loc[(gdf[speed_col_kmh].isna()) | (gdf[speed_col_kmh] == 0)]):
        raise ValueError(
            f"speed_col_kmh {speed_col_kmh!r} cannot have missing values or zeros"
        )

    gdf["minutes"] = gdf.length / (gdf[speed_col_kmh].astype(float) * 1000 / 60)

    return gdf

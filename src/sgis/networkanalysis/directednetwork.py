"""Prepare a GeoDataFrame of line geometries for directed network analysis."""

import warnings

import pandas as pd
from geopandas import GeoDataFrame
from shapely.constructive import reverse

from ..helpers import return_two_vals, unit_is_meters


def make_directed_network_norway(gdf: GeoDataFrame) -> GeoDataFrame:
    """Runs the method make_directed_network for Norwegian road data.

    The parameters in make_directed_network are set to the correct values for
    Norwegian road data as of 2023.

    The data can be downloaded here:
    https://kartkatalog.geonorge.no/metadata/nvdb-ruteplan-nettverksdatasett/8d0f9066-34f9-4423-be12-8e8523089313

    Examples
    --------
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
    [93395 rows x 46 columns]

    And converted to a directed network like this:

    >>> roads_directed = sg.make_directed_network_norway(roads)
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
    [175541 rows x 45 columns]

    """
    return make_directed_network(
        gdf,
        direction_col="oneway",
        direction_vals_bft=("B", "FT", "TF"),
        minute_cols=("drivetime_fw", "drivetime_bw"),
    )


def make_directed_network(
    gdf: GeoDataFrame,
    direction_col: str,
    direction_vals_bft: tuple[str, str, str],
    minute_cols: tuple[str, str] | str | None = None,
    speed_col: str | None = None,
    flat_speed: int | None = None,
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
    _validate_minute_args(minute_cols, speed_col, flat_speed)
    _validate_direction_args(gdf, direction_col, direction_vals_bft)

    if (flat_speed or speed_col) and not unit_is_meters(gdf):
        raise ValueError(
            "The crs must have 'metre' as units when calculating minutes."
            "Change crs or calculate minutes manually."
        )

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

    gdf = pd.concat([both_ways, both_ways2, ft, tf], ignore_index=True)

    gdf = gdf.drop([min_f, min_t], axis=1)

    if speed_col:
        _get_speed_from_col(gdf, speed_col)

    if flat_speed:
        gdf["minutes"] = gdf.length / flat_speed * 16.6666666667

    if "minutes" in gdf.columns:
        gdf = gdf.loc[gdf["minutes"] >= 0]

    return gdf


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


def _validate_direction_args(gdf, direction_col, direction_vals_bft):
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


def _get_speed_from_col(gdf, speed_col):
    if len(gdf.loc[(gdf[speed_col].isna()) | (gdf[speed_col] == 0)]) > len(gdf) * 0.05:
        raise ValueError(
            f"speed_col {speed_col!r} has a lot of missing or 0 values. Fill these "
            "with appropriate values"
        )
    gdf["minutes"] = gdf.length / gdf[speed_col].astype(float) * 16.6666666667

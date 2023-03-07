from geopandas import GeoDataFrame


def return_two_vals(
    vals: tuple[str, str] | list[str] | str | int | float
) -> tuple[str | int | tuple, str | int | tuple]:
    """Return a tuple of two values from a str/int/float or list/tuple of length 1 or 2

    Returns 'vals' as a 2-length tuple. If the input is a string, return
    a tuple of two strings. If the input is a list or tuple of two
    strings, return the list or tuple. Otherwise, raise a ValueError

    Args:
        vals: the item to be converted to a tuple of two values.

    Returns:
        A tuple of two strings, integers or floats.

    Raises:
        ValueError: If input type is not str/int/float or tuple/list with length of 1
            or 2.
    """
    if isinstance(vals, (tuple, list)):
        if len(vals) == 2:
            return vals[0], vals[1]
        if len(vals) == 1:
            return vals[0], vals[0]
        raise ValueError("list/tuple should be of length 1 or 2.")

    if isinstance(vals, (str, int, float)):
        return vals, vals

    raise ValueError(
        "Input type should be str/int/float or a tuple/list with a length of 1 or 2"
    )


def unit_is_meters(gdf: GeoDataFrame) -> bool:
    unit = gdf.crs.axis_info[0].unit_name
    if unit != "metre":
        return False
    return True


def unit_is_metres(gdf: GeoDataFrame) -> bool:
    return unit_is_meters(gdf)

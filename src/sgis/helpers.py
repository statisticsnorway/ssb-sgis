"""Small helper functions."""
import inspect
import warnings

from geopandas import GeoDataFrame


def return_two_vals(
    vals: tuple[str | None, str | None] | list[str] | str | int | float
) -> tuple[str | int | float, str | int | float | None]:
    """Return a two-length tuple from a str/int/float or list/tuple of length 1 or 2.

    Returns 'vals' as a 2-length tuple. If the input is a string, return
    a tuple of two strings. If the input is a list or tuple of two
    strings, return the list or tuple. Otherwise, raise a ValueError

    Args:
        vals: the item to be converted to a tuple of two values.

    Returns:
        A tuple of two strings, integers or floats.
    """
    if isinstance(vals, str):
        return vals, vals
    if hasattr(vals, "__iter__"):
        if len(vals) == 2:
            return vals[0], vals[1]
        if len(vals) == 1:
            return vals[0], vals[0]
        raise ValueError("list/tuple should be of length 1 or 2.")

    return vals, vals


def unit_is_meters(gdf: GeoDataFrame) -> bool:
    """Returns True if the crs unit is 'metre''."""
    unit = gdf.crs.axis_info[0].unit_name
    if unit != "metre":
        return False
    return True


def unit_is_metres(gdf: GeoDataFrame) -> bool:
    """Returns True if the crs unit is 'metre''."""
    return unit_is_meters(gdf)


def unit_is_degrees(gdf: GeoDataFrame) -> bool:
    """Returns True if the crs unit is 'metre''."""
    unit = gdf.crs.axis_info[0].unit_name
    if "degree" in unit:
        return True
    return False


def get_name(var: object, n: int = 5) -> str | None:
    """Searches through the local variables down one level at a time."""
    frame = inspect.currentframe().f_back.f_back

    for _ in range(n):
        locals_ = frame.f_locals

        names = [var_name for var_name, var_val in locals_.items() if var_val is var]
        if names and len(names) == 1:
            return names[0]

        names = [name for name in names if not name.startswith("_")]

        if names and len(names) == 1:
            return names[0]

        if names and len(names) > 1:
            warnings.warn(
                "More than one local variable matches the object. Name might be wrong."
            )
            return names[0]

        frame = frame.f_back

        if not frame:
            return


def make_namedict(gdfs: tuple[GeoDataFrame]) -> dict[int, str]:
    namedict = {}
    for i, gdf in enumerate(gdfs):
        if hasattr(gdf, "name"):
            namedict[i] = gdf.name
        else:
            name = get_name(gdf)
            if not name:
                name = str(i)
            namedict[i] = name
    return namedict

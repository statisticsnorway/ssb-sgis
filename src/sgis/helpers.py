"""Small helper functions."""
import glob
import inspect
import os
import warnings
from collections.abc import Callable

import numpy as np
from geopandas import GeoDataFrame


def get_numpy_func(text, error_message: str | None = None) -> Callable:
    f = getattr(np, text, None)
    if f is not None:
        return f
    f = getattr(np.ndarray, text, None)
    if f is not None:
        return f
    raise ValueError(error_message)


def get_func_name(func):
    try:
        return func.__name__
    except AttributeError:
        return str(func)


def get_non_numpy_func_name(f):
    if callable(f):
        return f.__name__
    return str(f).replace("np.", "").replace("numpy.", "")


def to_numpy_func(text):
    f = getattr(np, text, None)
    if f is not None:
        return f
    f = getattr(np.ndarray, text, None)
    if f is not None:
        return f
    raise ValueError


def is_property(obj, attribute) -> bool:
    return hasattr(obj.__class__, attribute) and isinstance(
        getattr(obj.__class__, attribute), property
    )


def dict_zip_intersection(*dicts):
    """From mCoding (YouTube)."""
    if not dicts:
        return

    keys = set(dicts[0]).intersection(*dicts[1:])
    for key in keys:
        yield key, *(d[key] for d in dicts)


def dict_zip_union(*dicts, fillvalue=None):
    """From mCoding (YouTube)."""
    if not dicts:
        return

    keys = set(dicts[0]).union(*dicts[1:])
    for key in keys:
        yield key, *(d.get(key, fillvalue) for d in dicts)


def dict_zip(*dicts):
    """From mCoding (YouTube)."""
    if not dicts:
        return

    n = len(dicts[0])
    if any(len(d) != n for d in dicts):
        raise ValueError("arguments must have the same length")

    for key, first_val in dicts[0].items():
        yield key, first_val, *(other[key] for other in dicts[1:])


def in_jupyter():
    try:
        get_ipython
        return True
    except NameError:
        return False


def get_all_files(root, recursive=True):
    if not recursive:
        return [path for path in glob.glob(str(Path(root)) + "/*")]
    paths = []
    for root_dir, _, files in os.walk(root):
        for file in files:
            path = os.path.join(root_dir, file)
            paths.append(path)
    return paths


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
    if not gdf.crs:
        return False
    unit = gdf.crs.axis_info[0].unit_name
    if unit != "metre":
        return False
    return True


def unit_is_metres(gdf: GeoDataFrame) -> bool:
    """Returns True if the crs unit is 'metre''."""
    return unit_is_meters(gdf)


def unit_is_degrees(gdf: GeoDataFrame) -> bool:
    """Returns True if the crs unit is 'degree''."""
    if not gdf.crs:
        return False

    unit = gdf.crs.axis_info[0].unit_name
    if "degree" in unit:
        return True
    return False


def get_object_name(
    var: object, start: int = 2, stop: int = 7, ignore_self: bool = True
) -> str | None:
    """Searches through the local variables down one level at a time."""
    frame = inspect.currentframe()

    for _ in range(start):
        frame = frame.f_back

    for _ in np.arange(start, stop):
        names = [
            var_name for var_name, var_val in frame.f_locals.items() if var_val is var
        ]
        if names and len(names) == 1:
            if ignore_self and names[0] == "self":
                frame = frame.f_back
                continue
            return names[0]

        names = [name for name in names if not name.startswith("_")]

        if names and len(names) == 1:
            if ignore_self and names[0] == "self":
                frame = frame.f_back
                continue

            return names[0]

        if names and len(names) > 1:
            if ignore_self and names[0] == "self":
                frame = frame.f_back
                continue
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
            name = get_object_name(gdf) or str(i)
            namedict[i] = name
    return namedict


def sort_nans_last(df, ignore_index: bool = False):
    if not len(df):
        return df
    df["n_nan"] = df.isna().sum(axis=1).values

    df["_idx"] = range(len(df))

    df = df.sort_values(["n_nan", "_idx"]).drop(columns=["n_nan", "_idx"])

    return df.reset_index(drop=True) if ignore_index else df


def is_number(text) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


class LocalFunctionError(ValueError):
    def __init__(self, func: str):
        self.func = func.__name__

    def __str__(self):
        return (
            f"{self.func}. "
            "In Jupyter, functions to be parallelized must \n"
            "be defined in and imported from another file when context='spawn'. \n"
            "Note that setting context='fork' might cause freezing processes.\n"
        )

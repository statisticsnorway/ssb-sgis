"""Small helper functions."""

import glob
import inspect
import os
import warnings
from collections.abc import Callable
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame


def get_numpy_func(text: str, error_message: str | None = None) -> Callable:
    """Fetch a numpy function based on its name.

    Args:
        text: The name of the numpy function to retrieve.
        error_message: Custom error message if the function is not found.

    Returns:
        The numpy function corresponding to the provided text.
    """
    f = getattr(np, text, None)
    if f is not None:
        return f
    f = getattr(np.ndarray, text, None)
    if f is not None:
        return f
    raise ValueError(error_message)


def get_func_name(func: Callable) -> str:
    """Return the name of a function.

    Args:
        func: The function object whose name is to be retrieved.

    Returns:
        The name of the function.
    """
    try:
        return func.__name__
    except AttributeError:
        return str(func)


def get_non_numpy_func_name(f: Callable | str) -> str:
    if callable(f):
        return f.__name__
    return str(f).replace("np.", "").replace("numpy.", "")


def to_numpy_func(text: str) -> Callable:
    """Convert a text identifier into a numpy function.

    Args:
        text: Name of the numpy function.

    Returns:
        The numpy function.
    """
    f = getattr(np, text, None)
    if f is not None:
        return f
    f = getattr(np.ndarray, text, None)
    if f is not None:
        return f
    raise ValueError


def is_property(obj: object, attr: str) -> bool:
    """Determine if a class attribute is a property.

    Args:
        obj: The object to check.
        attr: The attribute name to check on the object.

    Returns:
        True if the attribute is a property, False otherwise.
    """
    if not hasattr(obj.__class__, attr):
        return False
    if isinstance(obj, type):
        return isinstance(getattr(obj, attr), property)
    else:
        return isinstance(getattr(obj.__class__, attr), property)


def is_method(obj: Any, attr: str) -> bool:
    if isinstance(obj, type):
        return inspect.ismethod(getattr(obj, attr, None))
    else:
        return inspect.ismethod(getattr(obj.__class__, attr, None))


def dict_zip_intersection(*dicts: dict) -> Generator[tuple[Any, ...], None, None]:
    """From mCoding (YouTube)."""
    if not dicts:
        return

    keys = set(dicts[0]).intersection(*dicts[1:])
    for key in keys:
        yield key, *(d[key] for d in dicts)


def dict_zip_union(
    *dicts: dict, fillvalue: Any | None = None
) -> Generator[tuple[Any, ...], None, None]:
    """From mCoding (YouTube)."""
    if not dicts:
        return

    keys = set(dicts[0]).union(*dicts[1:])
    for key in keys:
        yield key, *(d.get(key, fillvalue) for d in dicts)


def dict_zip(*dicts: dict) -> Generator[tuple[Any, ...], None, None]:
    """From mCoding (YouTube)."""
    if not dicts:
        return

    n = len(dicts[0])
    if any(len(d) != n for d in dicts):
        raise ValueError("arguments must have the same length")

    for key, first_val in dicts[0].items():
        yield key, first_val, *(other[key] for other in dicts[1:])


def in_jupyter() -> bool:
    try:
        get_ipython  # type: ignore[name-defined]
        return True
    except NameError:
        return False


def _fix_path(path: str) -> str:
    return (
        str(path).replace("\\", "/").replace(r"\"", "/").replace("//", "/").rstrip("/")
    )


def get_all_files(root: str, recursive: bool = True) -> list[str]:
    """Fetch all files in a directory.

    Args:
        root: The root directory path.
        recursive: Whether to include subdirectories.

    Returns:
        A list of file paths.
    """
    if not recursive:
        return [_fix_path(path) for path in glob.glob(str(Path(root)) + "/**")]
    paths = []
    for root_dir, _, files in os.walk(root):
        for file in files:
            path = _fix_path(os.path.join(root_dir, file))
            paths.append(path)
    return paths


def return_two_vals(
    vals: tuple[str, str] | list[str] | str | int | float
) -> tuple[str | int | float, str | int | float]:
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
    if isinstance(vals, (tuple, list)):
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
) -> str:
    frame = inspect.currentframe()  # frame can be FrameType or None
    if frame:
        try:
            for _ in range(start):
                frame = frame.f_back if frame else None
            for _ in range(start, stop):
                if frame:
                    names = [
                        var_name
                        for var_name, var_val in frame.f_locals.items()
                        if var_val is var and not (ignore_self and var_name == "self")
                    ]
                    names = [name for name in names if not name.startswith("_")]
                    if names:
                        if len(names) != 1:
                            warnings.warn(
                                "More than one local variable matches the object. Name might be wrong.",
                                stacklevel=2,
                            )
                        return names[0]
                frame = frame.f_back if frame else None
        finally:
            if frame:
                del frame  # Explicitly delete frame reference to assist with garbage collection
    raise ValueError(f"Couldn't find name for {var}")


def make_namedict(gdfs: tuple[GeoDataFrame]) -> dict[int, str]:
    namedict = {}
    for i, gdf in enumerate(gdfs):
        if hasattr(gdf, "name"):
            namedict[i] = gdf.name
        else:
            name = get_object_name(gdf) or str(i)
            namedict[i] = name
    return namedict


def sort_nans_last(df: pd.DataFrame, ignore_index: bool = False) -> pd.DataFrame:
    """Sort a DataFrame placing rows with the most NaNs last.

    Args:
        df: DataFrame to sort.
        ignore_index: If True, the index will be reset.

    Returns:
        Sorted DataFrame with NaNs last.
    """
    if not len(df):
        return df
    df["n_nan"] = df.isna().sum(axis=1).values

    df["_idx"] = range(len(df))

    df = df.sort_values(["n_nan", "_idx"]).drop(columns=["n_nan", "_idx"])

    return df.reset_index(drop=True) if ignore_index else df


def is_number(text: str) -> bool:
    """Check if a string can be converted to a number.

    Args:
        text: The string to check.

    Returns:
        True if the string can be converted to a number, False otherwise.
    """
    try:
        float(text)
        return True
    except ValueError:
        return False


class LocalFunctionError(ValueError):
    """Exception for when a locally defined function is used in Jupyter, which is incompatible with multiprocessing."""

    def __init__(self, func: Callable) -> None:
        """Initialiser."""
        self.func = func.__name__

    def __str__(self) -> str:
        """Error message representation."""
        return (
            f"{self.func}. "
            "In Jupyter, functions to be parallelized must \n"
            "be defined in and imported from another file when context='spawn'. \n"
            "Note that setting context='fork' might cause freezing processes.\n"
        )

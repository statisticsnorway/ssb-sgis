"""Interactive or static map of one or more GeoDataFrames.

This module holds the Map class, which is the basis for the Explore class.
"""

import warnings
from collections.abc import Sequence
from statistics import mean
from typing import Any

import matplotlib
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import GeoSeries

try:
    from jenkspy import jenks_breaks
except ImportError:
    pass
from mapclassify import classify
from pandas.errors import PerformanceWarning
from shapely import Geometry

from ..geopandas_tools.conversion import to_gdf
from ..geopandas_tools.general import _rename_geometry_if
from ..geopandas_tools.general import clean_geoms
from ..geopandas_tools.general import drop_inactive_geometry_columns
from ..geopandas_tools.general import get_common_crs
from ..helpers import get_object_name
from ..helpers import unit_is_meters
from ..raster.image_collection import Band
from ..raster.image_collection import Image
from ..raster.image_collection import ImageCollection

try:
    from torchgeo.datasets.geo import RasterDataset
except ImportError:

    class RasterDataset:
        """Placeholder."""


# the geopandas._explore raises a deprication warning. Ignoring for now.
warnings.filterwarnings(
    action="ignore", category=matplotlib.MatplotlibDeprecationWarning
)
warnings.filterwarnings(action="ignore", category=PerformanceWarning)

pd.options.mode.chained_assignment = None


# custom default colors for non-numeric data, because the geopandas default has very
# similar colors. The palette is like the "Set2" cmap from matplotlib, but with more
# colors. If more than 14 categories, the geopandas default cmap is used.
_CATEGORICAL_CMAP = {
    0: "#3b93ff",
    1: "#ff3370",
    2: "#f7cf19",
    3: "#60e825",
    4: "#ff8cc9",
    5: "#804e00",
    6: "#e3dc00",
    7: "#00ffee",
    9: "#870062",
    10: "#751500",
    11: "#1c6b00",
    8: "#7cebb9",
}

DEFAULT_SCHEME = "quantiles"


def proper_fillna(val: Any, fill_val: Any) -> Any:
    """Manually handle missing values when fillna doesn't work as expected.

    Args:
        val: The value to check and fill.
        fill_val: The value to fill in.

    Returns:
        The original value or the filled value if conditions are met.
    """
    try:
        if "NAType" in val.__class__.__name__:
            return fill_val
    except Exception:
        if fill_val is None:
            return fill_val
        if fill_val != fill_val:
            return fill_val

    return val


class Map:
    """Base class that prepares one or more GeoDataFrames for mapping.

    The class has no public methods.
    """

    def __init__(
        self,
        *gdfs: GeoDataFrame,
        column: str | None = None,
        k: int = 5,
        bins: tuple[float] | None = None,
        nan_label: str = "Missing",
        nan_color="#c2c2c2",
        scheme: str = DEFAULT_SCHEME,
        cmap: str | None = None,
        **kwargs,
    ) -> None:
        """Initialiser.

        Args:
            *gdfs: Variable length GeoDataFrame list.
            column: The column name to work with.
            k: Number of bins or classes for classification (default: 5).
            bins: Predefined bins for data classification.
            nan_label: Label for missing data.
            nan_color: Color for missing data.
            scheme: Classification scheme to be used.
            cmap (str): Colormap of the plot. See:
                https://matplotlib.org/stable/tutorials/colors/colormaps.html
            **kwargs: Arbitrary keyword arguments.
        """
        gdfs, column, kwargs = self._separate_args(gdfs, column, kwargs)

        self._column = column
        self.bins = bins
        self._k = k
        self.nan_label = nan_label
        self.nan_color = nan_color
        self._cmap = cmap
        self.scheme = scheme

        # need to get the object names of the gdfs before copying. Only getting,
        # not setting, labels. So the original gdfs don't get the label column.
        self.labels: list[str] = [
            _determine_best_name(gdf, column, i) for i, gdf in enumerate(gdfs)
        ]

        show = kwargs.pop("show", True)
        if isinstance(show, (int, bool)):
            show_temp = [bool(show) for _ in range(len(gdfs))]
        elif not hasattr(show, "__iter__"):
            raise ValueError(
                "'show' must be boolean or an iterable of boleans same "
                f"length as gdfs ({len(gdfs)}). Got len {len(show)}"
            )
        else:
            show_temp = show

        show_args = show_temp[: len(gdfs)]
        # gdfs that are in kwargs
        show_kwargs = show_temp[len(gdfs) :]
        self._gdfs = []
        new_labels = []
        self.show = []
        for label, gdf, show in zip(self.labels, gdfs, show_args, strict=False):
            if not len(gdf):
                continue

            gdf = clean_geoms(gdf).reset_index(drop=True)
            if not len(gdf):
                continue

            self._gdfs.append(to_gdf(gdf))
            new_labels.append(label)
            self.show.append(show)
        self.labels = new_labels

        # pop all geometry-like items from kwargs into self._gdfs
        self.kwargs = {}
        i = 0
        for key, value in kwargs.items():
            try:
                if isinstance(value, Geometry):
                    value = to_gdf(value)
                if not len(value):
                    continue
                self._gdfs.append(to_gdf(value))
                self.labels.append(key)
                try:
                    show = show_kwargs[i]
                    i += 1
                except IndexError:
                    pass
                self.show.append(show)
            except Exception:
                self.kwargs[key] = value

        if hasattr(self.show, "__iter__") and len(self.show) != len(self._gdfs):
            raise ValueError(
                "'show' must be boolean or an iterable of boleans same "
                f"length as gdfs ({len(gdfs)}). Got len {len(show)}"
            )

        if not self._gdfs or not any(len(gdf) for gdf in self._gdfs):
            self._gdfs = []
            self._is_categorical = True
            self._unique_values = []
            self._nan_idx = []
            return

        if not self.labels:
            self._set_labels()

        self._gdfs = self._to_common_crs_and_one_geom_col(self._gdfs)
        self._is_categorical = self._check_if_categorical()

        if self._column:
            self._fillna_if_col_is_missing()
        else:
            gdfs = []
            for gdf, label in zip(self._gdfs, self.labels, strict=True):
                gdf["label"] = label
                gdfs.append(gdf)
            self._column = "label"
            self._gdfs = gdfs

        try:
            self._gdf = pd.concat(self._gdfs, ignore_index=True)
        except ValueError:
            crs = get_common_crs(self._gdfs)
            for gdf in self._gdfs:
                gdf.crs = crs
            self._gdf = pd.concat(self._gdfs, ignore_index=True)

        self._nan_idx = self._gdf[self._column].isna()
        self._get_unique_values()

    def __getattr__(self, attr: str) -> Any:
        """Search for attribute in kwargs."""
        return self.kwargs.get(attr, super().__getattribute__(attr))

    def __bool__(self) -> bool:
        """True of any gdfs with more than 0 rows."""
        return bool(len(self._gdfs) + len(self._gdf))

    def _get_unique_values(self) -> None:
        if not self._is_categorical:
            self._unique_values = self._get_unique_floats()
        else:
            unique = list(self._gdf[self._column].unique())
            try:
                self._unique_values = sorted(unique)
            except TypeError:
                self._unique_values = [
                    x for x in unique if proper_fillna(x, None) is not None
                ]

        self._k = min(self._k, len(self._unique_values))

    def _get_unique_floats(self) -> np.array:
        """Get unique floats by multiplying, then converting to integer.

        Find a multiplier that makes the max value greater than +- 1_000_000.
        Because floats don't always equal each other. This will make very
        similar values count as the same value in the color classification.
        """
        array = self._gdf.loc[list(~self._nan_idx), self._column]
        self._min = np.min(array)
        self._max = np.max(array)
        self._get_multiplier(array)

        unique = array.reset_index(drop=True).drop_duplicates()
        as_int = self._array_to_large_int(unique)
        no_duplicates = as_int.drop_duplicates()

        return np.sort(np.array(unique.loc[no_duplicates.index]))

    def _array_to_large_int(self, array: np.ndarray | pd.Series) -> pd.Series:
        """Multiply values in float array, then convert to integer."""
        if not isinstance(array, pd.Series):
            array = pd.Series(array)

        notna = array[array.notna()]
        isna = array[array.isna()]

        unique_multiplied = (notna * self._multiplier).astype(np.int64)

        return pd.concat([unique_multiplied, isna]).sort_index()

    def _get_multiplier(self, array: np.ndarray) -> None:
        """Find the number of zeros needed to push the max value of the array above +-1_000_000.

        Adding this as an attribute to use later in _classify_from_bins.
        """
        if np.max(array) == 0:
            self._multiplier: int = 1
            return

        multiplier = 10
        max_ = np.max(array * multiplier)

        if self._max > 0:
            while max_ < 1_000_000:
                multiplier *= 10
                max_ = np.max(array * multiplier)
        else:
            while max_ > -1_000_000:
                multiplier *= 10
                max_ = np.max(array * multiplier)

        self._multiplier: int = multiplier

    def _add_minmax_to_bins(self, bins: list[float | int]) -> list[float | int]:
        """If values are outside the bin range, add max and/or min values of array."""
        # make sure they are lists
        bins = [bin_ for bin_ in bins]

        if min(bins) > 0 and min(
            self._gdf.loc[list(~self._nan_idx), self._column]
        ) < min(bins):
            num = min(self._gdf.loc[list(~self._nan_idx), self._column])
            # if isinstance(num, float):
            #     num -= (
            #         float(f"1e-{abs(self.legend.rounding)}")
            #         if self.legend and self.legend.rounding
            #         else 0
            #     )
            bins = [num] + bins

        if min(bins) < 0 and min(
            self._gdf.loc[list(~self._nan_idx), self._column]
        ) < min(bins):
            num = min(self._gdf.loc[list(~self._nan_idx), self._column])
            # if isinstance(num, float):
            #     num -= (
            #         float(f"1e-{abs(self.legend.rounding)}")
            #         if self.legend and self.legend.rounding
            #         else 0
            #     )
            bins = [num] + bins

        if max(bins) > 0 and max(
            self._gdf.loc[self._gdf[self._column].notna(), self._column]
        ) > max(bins):
            num = max(self._gdf.loc[self._gdf[self._column].notna(), self._column])
            # if isinstance(num, float):
            #     num += (
            #         float(f"1e-{abs(self.legend.rounding)}")
            #         if self.legend and self.legend.rounding
            #         else 0
            #     )
            bins = bins + [num]

        if max(bins) < 0 and max(
            self._gdf.loc[self._gdf[self._column].notna(), self._column]
        ) < max(bins):
            num = max(self._gdf.loc[self._gdf[self._column].notna(), self._column])
            # if isinstance(num, float):
            #     num += (
            #         float(f"1e-{abs(self.legend.rounding)}")
            #         if self.legend and self.legend.rounding
            #         else 0
            #     )

            bins = bins + [num]

        def adjust_bin(num: int | float, i: int) -> int | float:
            if isinstance(num, int):
                return num
            adjuster = (
                float(f"1e-{abs(self.legend.rounding)}")
                if self.legend and self.legend.rounding
                else 0
            )
            if i == 0:
                return num - adjuster
            elif i == len(bins) - 1:
                return num + adjuster
            return num

        bins = [adjust_bin(x, i) for i, x in enumerate(bins)]

        return bins

    @staticmethod
    def _separate_args(
        args: tuple,
        column: str | None,
        kwargs: dict,
    ) -> tuple[tuple[GeoDataFrame], str]:
        """Separate GeoDataFrames from string (column argument)."""

        def as_dict(obj) -> dict:
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            elif isinstance(obj, dict):
                return obj
            raise TypeError(type(obj))

        allowed_types = (
            GeoDataFrame,
            GeoSeries,
            Geometry,
            RasterDataset,
            ImageCollection,
            Image,
            Band,
        )

        gdfs = ()
        more_gdfs = {}
        i = 0
        for arg in args:
            if isinstance(arg, str):
                if column is None:
                    column = arg
                else:
                    raise ValueError(
                        "Can specify at most one string as a positional argument."
                    )
            elif isinstance(arg, allowed_types):
                gdfs = gdfs + (arg,)
            # elif isinstance(arg, Sequence) and not isinstance(arg, str):
            elif isinstance(arg, dict) or hasattr(arg, "__dict__"):
                # add dicts or classes with GeoDataFrames to kwargs
                for key, value in as_dict(arg).items():
                    if isinstance(value, allowed_types):
                        more_gdfs[key] = value
                    elif isinstance(value, dict) or hasattr(value, "__dict__"):
                        # elif isinstance(value, Sequence) and not isinstance(value, str):
                        try:
                            # same as above, one level down
                            more_gdfs |= {
                                k: v
                                for k, v in as_dict(value).items()
                                if isinstance(v, allowed_types)
                            }
                        except Exception:
                            # ignore all exceptions
                            pass

            elif isinstance(arg, Sequence) and not isinstance(arg, str):
                # add dicts or classes with GeoDataFrames to kwargs
                for value in arg:
                    if isinstance(value, allowed_types):
                        name = _determine_best_name(value, column, i)
                        more_gdfs[name] = value
                    elif isinstance(value, dict) or hasattr(value, "__dict__"):
                        try:
                            # same as above, one level down
                            more_gdfs |= {
                                k: v
                                for k, v in value.items()
                                if isinstance(v, allowed_types)
                            }
                        except Exception:
                            # no need to raise here
                            pass
                    elif isinstance(value, Sequence) and not isinstance(value, str):
                        for x in value:
                            if not isinstance(x, allowed_types):
                                continue
                            name = _determine_best_name(value, column, i)
                            more_gdfs[name] = x
                    i += 1

        kwargs |= more_gdfs

        return gdfs, column, kwargs

    def _prepare_continous_map(self) -> None:
        """Create bins if not already done and adjust k if needed."""
        if self.scheme is None:
            return

        if self.bins is None:
            self.bins = self._create_bins(self._gdf, self._column)
            if len(self.bins) <= self._k and len(self.bins) != len(self._unique_values):
                self._k = len(self.bins)
        elif not all(self._gdf[self._column].isna()):
            self.bins = self._add_minmax_to_bins(self.bins)
            if len(self._unique_values) <= len(self.bins):
                self._k = len(self.bins)  # - 1
        else:
            self._unique_values = self.nan_label
            self._k = 1

    def _set_labels(self) -> None:
        """Setting the labels after copying the gdfs."""
        gdfs = []
        for i, gdf in enumerate(self._gdfs):
            gdf["label"] = self.labels[i]
            gdfs.append(gdf)
        self._gdfs = gdfs

    def _to_common_crs_and_one_geom_col(
        self, gdfs: list[GeoDataFrame]
    ) -> list[GeoDataFrame]:
        """Need common crs and max one geometry column."""
        crs_list = list({gdf.crs for gdf in gdfs if gdf.crs is not None})
        if crs_list:
            self.crs = crs_list[0]
        new_gdfs = []
        for gdf in gdfs:
            gdf = gdf.reset_index(drop=True)
            gdf = drop_inactive_geometry_columns(gdf).pipe(_rename_geometry_if)
            if crs_list:
                try:
                    gdf = gdf.to_crs(self.crs)
                except ValueError:
                    gdf = gdf.set_crs(self.crs)
            new_gdfs.append(gdf)
        return new_gdfs

    def _fillna_if_col_is_missing(self) -> None:
        n = 0
        for gdf in self._gdfs:
            if self._column in gdf.columns:
                gdf[self._column] = gdf[self._column].fillna(pd.NA)
                n += 1
            else:
                gdf[self._column] = pd.NA

        maybe_area = 1 if "area" in self._column else 0
        maybe_length = (
            1 if any(x in self._column for x in ["meter", "metre", "leng"]) else 0
        )
        n = n + maybe_area + maybe_length

        if n == 0:
            raise ValueError(
                f"The column {self._column!r} is not present in any "
                "of the passed GeoDataFrames."
            )

    def _check_if_categorical(self) -> bool:
        """Quite messy this..."""
        if not self._column or not self._gdfs:
            return True

        def is_maybe_km2():
            if "area" in self._column and (
                "km2" in self._column
                or "kilomet" in self._column
                and ("sq" in self._column or "2" in self._column)
            ):
                return True
            else:
                return False

        maybe_area = 1 if "area" in self._column else 0
        maybe_area_km2 = 1 if is_maybe_km2() else 0
        maybe_length = (
            1 if any(x in self._column for x in ["meter", "metre", "leng"]) else 0
        )

        all_nan = 0
        col_not_present = 0
        for gdf in self._gdfs:
            if self._column not in gdf:
                if maybe_area_km2 and unit_is_meters(gdf):
                    gdf["area_km2"] = gdf.area / 1_000_000
                    maybe_area_km2 += 1
                elif maybe_area:
                    gdf["area"] = gdf.area
                    maybe_area += 1
                elif maybe_length:
                    gdf["length"] = gdf.length
                    maybe_length += 1
                else:
                    col_not_present += 1
            elif not pd.api.types.is_numeric_dtype(gdf[self._column]):
                if all(gdf[self._column].isna()):
                    all_nan += 1
                return True

        if maybe_area_km2 > 1:
            self._column = "area_km2"
            return False
        if maybe_area > 1:
            self._column = "area"
            return False
        if maybe_length > 1:
            self._column = "length"
            return False

        if all_nan == len(self._gdfs):
            raise ValueError(
                f"All values are NaN in column {self.column!r}. {self._gdfs}"
            )

        if col_not_present == len(self._gdfs):
            raise ValueError(f"{self.column} not found.")

        return False

    def _make_categories_colors_dict(self) -> None:
        # custom categorical cmap
        if not self._cmap and len(self._unique_values) <= len(_CATEGORICAL_CMAP):
            self._categories_colors_dict = {
                category: _CATEGORICAL_CMAP[i]
                for i, category in enumerate(self._unique_values)
            }
        elif self._cmap:
            cmap = matplotlib.colormaps.get_cmap(self._cmap)

            self._categories_colors_dict = {
                category: colors.to_hex(cmap(int(i)))
                for i, category in enumerate(self._unique_values)
            }
        else:
            cmap = matplotlib.colormaps.get_cmap("tab20")

            self._categories_colors_dict = {
                category: colors.to_hex(cmap(int(i)))
                for i, category in enumerate(self._unique_values)
            }

    def _fix_nans(self) -> None:
        if any(self._nan_idx):
            self._gdf[self._column] = self._gdf[self._column].fillna(self.nan_label)
            self._categories_colors_dict[self.nan_label] = self.nan_color

        new_gdfs = []
        for gdf in self._gdfs:
            gdf["color"] = gdf[self._column].map(self._categories_colors_dict)
            new_gdfs.append(gdf)
        self._gdfs = new_gdfs

        self._gdf["color"] = self._gdf[self._column].map(self._categories_colors_dict)

    def _create_bins(self, gdf: GeoDataFrame, column: str) -> np.ndarray:
        """Make bin list of length k + 1, or length of unique values.

        The returned bins sometimes have two almost identical

        If 'scheme' is not specified, the jenks_breaks function is used, which is
        much faster than the one from Mapclassifier.
        """
        if not len(gdf.loc[list(~self._nan_idx), column]):
            return np.array([0])

        n_classes = (
            self._k if len(self._unique_values) > self._k else len(self._unique_values)
        )

        if self._k == len(self._unique_values) - 1:
            n_classes = self._k - 1
            self._k = self._k - 1

        if self._k > len(self._unique_values):
            self._k = len(self._unique_values)
            n_classes = len(self._unique_values)

        if self.scheme == "jenks":
            bins = jenks_breaks(
                gdf.loc[list(~self._nan_idx), column], n_classes=n_classes
            )
        else:
            binning = classify(
                np.asarray(gdf.loc[list(~self._nan_idx), column]),
                scheme=self.scheme,
                # k=self._k,
                k=n_classes,
            )
            bins = binning.bins

        bins = self._add_minmax_to_bins(bins)

        unique_bins = list({round(bin_, 5) for bin_ in bins})
        unique_bins.sort()

        if self._k == len(self._unique_values) - 1 or len(unique_bins) == len(
            self._unique_values
        ):
            return np.array(unique_bins)

        if len(unique_bins) == len(bins) - 1:
            self._k -= 1

        return np.array(bins)

    def change_cmap(self, cmap: str, start: int = 0, stop: int = 256) -> "Map":
        """Change the color palette of the plot.

        Args:
            cmap: The colormap.
                https://matplotlib.org/stable/tutorials/colors/colormaps.html
            start: Start position for the color palette. Defaults to 0.
            stop: End position for the color palette. Defaults to 256, which
                is the end of the color range.
        """
        self.cmap_start = start
        self.cmap_stop = stop
        self._cmap = cmap
        self._cmap_has_been_set = True
        return self

    def _get_continous_colors(self, n: int) -> np.ndarray:
        cmap = matplotlib.colormaps.get_cmap(self._cmap)
        colors_ = [
            colors.to_hex(cmap(int(i)))
            #            for i in np.linspace(self.cmap_start, self.cmap_stop, num=self._k)
            for i in np.linspace(self.cmap_start, self.cmap_stop, num=n)
        ]
        if any(self._nan_idx):
            colors_ = colors_ + [self.nan_color]
        return np.array(colors_)

    def _classify_from_bins(self, gdf: GeoDataFrame, bins: np.ndarray) -> np.ndarray:
        """Place the column values into groups."""
        bins = bins.copy()

        # if equal lenght, convert to integer and check for equality
        if len(bins) == len(self._unique_values):
            if gdf[self._column].isna().all():
                return np.repeat(len(bins), len(gdf))

            gdf["col_as_int"] = self._array_to_large_int(gdf[self._column])
            bins = self._array_to_large_int(self._unique_values)
            gdf["col_as_int"] = gdf["col_as_int"].fillna(np.nan)
            classified = np.searchsorted(bins, gdf["col_as_int"])

        else:
            if len(bins) == self._k + 1:
                bins = bins[1:]

            if (
                self.legend
                and self.legend.rounding
                and (self.legend.rounding or 1) <= 0
            ):
                bins[0] = bins[0] - 1
                bins[-1] = bins[-1] + 1

            if gdf[self._column].isna().all():
                return np.repeat(len(bins), len(gdf))

            # need numpy.nan instead of pd.NA
            gdf[self._column] = gdf[self._column].fillna(np.nan)

            gdf[self._column] = gdf[self._column].apply(
                lambda x: proper_fillna(x, np.nan)
            )

            classified = np.searchsorted(bins, gdf[self._column])

        return classified

    def _push_classification(self, classified: np.ndarray) -> np.ndarray:
        """Push classes downwards if gaps in classification sequence.

        So from e.g. [0,2,4] to [0,1,2].

        Otherwise, will get index error when classifying colors.
        """
        rank_dict = {val: rank for rank, val in enumerate(np.unique(classified))}

        return np.array([rank_dict[val] for val in classified])

    @property
    def k(self) -> int:
        """Number of bins."""
        return self._k

    @k.setter
    def k(self, new_value: int) -> None:
        if not self._is_categorical and new_value > len(self._unique_values):
            raise ValueError(
                "'k' cannot be greater than the number of unique values in the column.'"
                f"Got new k={new_value} and previous k={len(self._unique_values)}.'"
                #  f"{''.join(self._unique_values)}"
            )
        self._k = int(new_value)

    @property
    def cmap(self) -> str:
        """Colormap."""
        return self._cmap

    @cmap.setter
    def cmap(self, new_value: str) -> None:
        self._cmap = new_value
        if not self._is_categorical:
            self.change_cmap(cmap=new_value, start=self.cmap_start, stop=self.cmap_stop)

    @property
    def gdf(self) -> GeoDataFrame:
        """All GeoDataFrames concated."""
        return self._gdf

    @gdf.setter
    def gdf(self, _) -> None:
        raise ValueError(
            "Cannot change 'gdf' after init. Put the GeoDataFrames into "
            "the class initialiser."
        )

    @property
    def gdfs(self) -> list[GeoDataFrame]:
        """All GeoDataFrames as a list."""
        return self._gdfs

    @gdfs.setter
    def gdfs(self, _) -> None:
        raise ValueError(
            "Cannot change 'gdfs' after init. Put the GeoDataFrames into "
            "the class initialiser."
        )

    @property
    def column(self) -> str | None:
        """Column to use as colormap."""
        return self._column

    @column.setter
    def column(self, _) -> None:
        raise ValueError(
            "Cannot change 'column' after init. Specify 'column' in the "
            "class initialiser."
        )

    def __setitem__(self, item: Any, new_item: Any) -> None:
        """Set an attribute with square brackets."""
        return setattr(self, item, new_item)

    def __getitem__(self, item: Any) -> Any:
        """Get an attribute with square brackets."""
        return getattr(self, item)

    def get(self, key: Any, default: Any | None = None) -> Any:
        """Get an attribute with default value if not present."""
        try:
            return self[key]
        except (KeyError, ValueError, IndexError, AttributeError):
            return default


def _determine_best_name(obj: Any, column: str | None, i: int) -> str:
    try:
        # Frame 3: actual object name Frame 2: maps.py:explore(). Frame 1: __init__. Frame 0: this function.
        return str(get_object_name(obj, start=3))
    except ValueError:
        if isinstance(obj, GeoSeries) and obj.name:
            return str(obj.name)
        elif isinstance(obj, GeoDataFrame) and len(obj.columns) == 2 and not column:
            series = obj.drop(columns=obj._geometry_column_name).iloc[:, 0]
            if (
                len(series.unique()) == 1
                and mean(isinstance(x, str) for x in series) > 0.5
            ):
                return str(next(iter(series)))
            elif series.name:
                return str(series.name)
        else:
            # generic label e.g. Image(1)
            return f"{obj.__class__.__name__}({i})"

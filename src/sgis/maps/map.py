"""Interactive map of one or more GeoDataFrames with layers that can be toggles on/off.

This module holds the Explore class, which is the basis for the explore, samplemap and
clipmap functions from the 'maps' module.
"""
import warnings

import matplotlib
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from jenkspy import jenks_breaks
from mapclassify import classify
from shapely import Geometry

from ..geopandas_tools.general import drop_inactive_geometry_columns, rename_geometry_if
from ..helpers import get_name


# the geopandas._explore raises a deprication warning. Ignoring for now.
warnings.filterwarnings(
    action="ignore", category=matplotlib.MatplotlibDeprecationWarning
)
pd.options.mode.chained_assignment = None


# custom default colors for non-numeric data, because the geopandas default has very
# similar colors. The palette is like the "Set2" cmap from matplotlib, but with more
# colors. If more than 14 categories, the geopandas default cmap is used.
_CATEGORICAL_CMAP = {
    0: "#4576ff",
    1: "#ff455e",
    2: "#59d45f",
    3: "#b51d8b",
    4: "#ffa514",
    5: "#f2dc4e",
    6: "#ff8cc9",
    7: "#6bf2eb",
    8: "#916209",
    9: "#008d94",
    10: "#8a030a",
    11: "#9c65db",
    12: "#228000",
    13: "#80ff00",
}


class Map:
    """Base class that prepares one or more GeoDataFrames for mapping.

    The class has no public methods.
    """

    def __init__(
        self,
        *gdfs: GeoDataFrame,
        column: str | None = None,
        labels: tuple[str] | None = None,
        k: int = 5,
        bins: tuple[float] | None = None,
        nan_label: str = "Missing",
        nan_color="#c2c2c2",
        **kwargs,
    ):
        if not all(isinstance(gdf, GeoDataFrame) for gdf in gdfs):
            gdfs, column = self._separate_args(gdfs, column)

        self._column = column
        self.bins = bins
        self._k = k
        self.nan_label = nan_label
        self.nan_color = nan_color
        self._cmap = kwargs.pop("cmap", None)

        if not all(isinstance(gdf, GeoDataFrame) for gdf in gdfs):
            raise ValueError("gdfs must be GeoDataFrames.")

        if not any(len(gdf) for gdf in gdfs):
            raise ValueError("None of the GeoDataFrames have rows.")

        if "namedict" in kwargs:
            for i, gdf in enumerate(gdfs):
                gdf.name = kwargs["namedict"][i]
            kwargs.pop("namedict")

        # need to get the object names of the gdfs before copying. Only getting,
        # not setting, labels. So the original gdfs don't get the label column.
        self.labels = labels
        if not self.labels:
            self._get_labels(gdfs)

        self._gdfs: list[GeoDataFrame] = [gdf.copy() for gdf in gdfs]
        self.kwargs = kwargs

        if not self.labels:
            self._set_labels()

        if not self._column:
            for gdf, label in zip(self._gdfs, self.labels, strict=True):
                gdf["label"] = label
            self._column = "label"

        self._gdfs = self._to_common_crs_and_one_geom_col(self._gdfs)
        self._is_categorical = self._check_if_categorical()

        self._fillna_if_col_is_missing()

        self._gdf = pd.concat(self._gdfs, ignore_index=True)

        self._nan_idx = self._gdf[self._column].isna()

        if not self._is_categorical:
            self._unique_values = self._get_unique_floats()
            if self._k > len(self._unique_values):
                self._k = len(self._unique_values)
        else:
            self._unique_values = sorted(
                list(self._gdf.loc[~self._nan_idx, self._column].unique())
            )

    def _get_unique_floats(self) -> list[int | float]:
        """Converting floats to large integers, then getting unique values.

        Also making a column of the large integers to use in the bin classifying later.
        """
        array = self._gdf.loc[~self._nan_idx, self._column]

        self._gdf["col_as_int"] = self._array_to_large_int(array)

        unique = array.reset_index(drop=True).drop_duplicates()

        as_int = self._array_to_large_int(unique)
        no_duplicates = as_int.drop_duplicates()
        return list(sorted(unique.loc[no_duplicates.index]))

    @staticmethod
    def _array_to_large_int(array: np.ndarray):
        """Multiply values in float array, then convert to integer."""
        max_ = np.max(array)
        min_ = np.min(array)

        if max_ > 1 or min_ < -1:
            unique_multiplied = array * np.emath.logn(1.25, np.abs(np.mean(array)) + 1)
        else:
            unique_multiplied = array
            while max_ < 1_000_000:
                unique_multiplied = unique_multiplied * 10
                max_ = np.max(unique_multiplied)

        return unique_multiplied.astype(np.int64)

    def _add_minmax_to_bins(self, bins: list[float | int]) -> list[float | int]:
        """If values are outside the bin range, add max and/or min values of array."""
        # make sure they are lists
        bins = [bin for bin in bins]

        if min(bins) > 0 and min(self._gdf[self._column]) < min(bins) * 0.999:
            bins = [min(self._gdf[self._column]) * 0.9999] + bins

        if min(bins) < 0 and min(self._gdf[self._column]) < min(bins) * 1.0001:
            bins = [min(self._gdf[self._column]) * 1.0001] + bins

        if max(bins) > 0 and max(self._gdf[self._column]) > max(bins) * 1.0001:
            bins = bins + [max(self._gdf[self._column]) * 1.0001]

        if max(bins) < 0 and max(self._gdf[self._column]) < max(bins) * 1.0001:
            bins = bins + [max(self._gdf[self._column]) * 1.0001]

        return bins

    @staticmethod
    def _separate_args(
        args: tuple,
        column: str | None,
    ) -> tuple[tuple[GeoDataFrame], str]:
        """Separate GeoDataFrames from string (column argument)."""

        gdfs: tuple[GeoDataFrame] = ()
        for arg in args:
            if isinstance(arg, str):
                if column is None:
                    column = arg
                else:
                    raise ValueError(
                        "Can specify at most one string as a positional argument."
                    )
            elif isinstance(arg, (GeoDataFrame, GeoSeries, Geometry)):
                gdfs = gdfs + (arg,)

        return gdfs, column

    def _prepare_continous_map(self):
        """Create bins if not already done and adjust k if needed."""

        if not hasattr(self, "scheme"):
            self.scheme = self.kwargs.get("scheme", "fisherjenks")

        if self.scheme is None:
            return

        if not self.bins:
            self.bins = self._create_bins(self._gdf, self._column)
            if len(self.bins) <= self._k and len(self.bins) != len(self._unique_values):
                warnings.warn(f"Could not create {self._k} classes.")
                self._k = len(self.bins)
            self.bins = self._add_minmax_to_bins(self.bins)
        else:
            self.bins = self._add_minmax_to_bins(self.bins)
            if len(self._unique_values) > len(self.bins):
                self._k = len(self.bins) - 1

    def _get_labels(self, gdfs: tuple[GeoDataFrame]) -> None:
        """Putting the labels/names in a list before copying the gdfs."""
        self.labels: list[str] = []
        for i, gdf in enumerate(gdfs):
            if hasattr(gdf, "name"):
                name = gdf.name
            else:
                name = get_name(gdf)
                if not name:
                    name = str(i)
            self.labels.append(name)

    def _set_labels(self) -> None:
        """Setting the labels after copying the gdfs."""
        for i, gdf in enumerate(self._gdfs):
            gdf["label"] = self.labels[i]

    def _to_common_crs_and_one_geom_col(self, gdfs: list[GeoDataFrame]):
        """Need common crs and max one geometry column."""
        crs_list = list({gdf.crs for gdf in gdfs if gdf.crs is not None})
        if not crs_list:
            return gdfs
        self.crs = crs_list[0]
        new_gdfs = []
        for gdf in gdfs:
            gdf = drop_inactive_geometry_columns(gdf).pipe(rename_geometry_if)
            if crs_list:
                try:
                    gdf = gdf.to_crs(self.crs)
                except ValueError:
                    gdf = gdf.set_crs(self.crs)
            new_gdfs.append(gdf)
        return new_gdfs

    def _fillna_if_col_is_missing(self) -> None:
        for gdf in self._gdfs:
            if self._column in gdf.columns:
                continue
            gdf[self._column] = pd.NA

    def _check_if_categorical(self) -> bool:
        """Quite messy this..."""
        if not self._column:
            return True

        maybe_area = 1 if "area" in self._column else 0
        maybe_length = (
            1 if any(x in self._column for x in ["meter", "metre", "leng"]) else 0
        )

        all_nan = 0
        col_not_present = 0
        for gdf in self._gdfs:
            if self._column not in gdf:
                if maybe_area:
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

        if maybe_area > 1:
            self._column = "area"
            return False
        if maybe_length > 1:
            self._column = "length"
            return False

        if all_nan == len(self._gdfs):
            raise ValueError(f"All values are NaN in column {self.kwargs['column']!r}.")

        if col_not_present == len(self._gdfs):
            raise ValueError(f"{self.kwargs['column']} not found.")

        return False

    def _get_categorical_colors(self) -> None:
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

        if any(self._nan_idx):
            self._gdf[self._column] = self._gdf[self._column].fillna(self.nan_label)
            self._categories_colors_dict[self.nan_label] = self.nan_color

        for gdf in self._gdfs:
            gdf["color"] = gdf[self._column].map(self._categories_colors_dict)

        self._gdf["color"] = self._gdf[self._column].map(self._categories_colors_dict)

    def _create_bins(self, gdf, column) -> np.ndarray:
        """Make bin list of length k + 1, or length of unique values.

        The returned bins sometimes have two almost identical

        If 'scheme' is not specified, the jenks_breaks function is used, which is
        much faster than the one from Mapclassifier.
        """

        if hasattr(self, "scheme"):
            scheme = self.scheme
        else:
            scheme = self.kwargs.get("scheme", "fisherjenks")

        if scheme is None:
            return

        n_classes = (
            self._k if len(self._unique_values) > self._k else len(self._unique_values)
        )

        if self._k == len(self._unique_values) - 1:
            n_classes = self._k - 1

        if scheme == "fisherjenks":
            bins = jenks_breaks(gdf.loc[~self._nan_idx, column], n_classes=n_classes)
        else:
            binning = classify(
                np.asarray(gdf.loc[~self._nan_idx, column]),
                scheme=scheme,
                k=self._k,
            )
            bins = binning.bins
            bins = self._add_minmax_to_bins(bins)

        unique_bins = list({round(bin, 5) for bin in bins})
        unique_bins.sort()

        if self._k == len(self._unique_values) - 1:
            return np.array(unique_bins)

        if len(unique_bins) == len(self._unique_values):
            return np.array(unique_bins)

        binarray = np.array(bins)
        binarray = np.where(
            binarray > 0,
            binarray + binarray / 100_000,
            binarray - binarray / 100_000,
        )
        return binarray

    def change_cmap(self, cmap: str, start: int = 0, stop: int = 256):
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

    def _get_continous_colors(self) -> list[str]:
        cmap = matplotlib.colormaps.get_cmap(self._cmap)
        colors_ = [
            colors.to_hex(cmap(int(i)))
            for i in np.linspace(self.cmap_start, self.cmap_stop, num=self._k)
        ]
        if any(self._nan_idx):
            colors_ = colors_ + [self.nan_color]
        return colors_

    def _classify_from_bins(self, gdf: GeoDataFrame) -> np.ndarray:
        """Place the values of the column into groups."""
        # if equal lenght, use integer column to check for equality
        # since long floats are unpredictable
        if len(self.bins) == len(self._unique_values):
            if "col_as_int" not in gdf.columns:
                gdf["col_as_int"] = self._array_to_large_int(gdf[self._column])
            bins = np.array(sorted(gdf["col_as_int"].unique()))
            classified = np.searchsorted(bins, gdf["col_as_int"])
        else:
            if any(self._nan_idx) and len(self.bins) == len(self.colorlist):
                bins = self.bins[1:]
            elif not any(self._nan_idx) and len(self.bins) == len(self.colorlist) + 1:
                bins = self.bins[1:]
            else:
                bins = self.bins

            classified = np.searchsorted(bins, gdf[self._column])

        # storing unique values to use in legend labels
        self._bins_unique_values = {
            i: list(set(gdf.loc[classified == i, self._column]))
            for i, _ in enumerate(bins)
        }

        colors_ = np.array(self.colorlist)

        # nans are sorted to the end, so nans will get NAN_COLOR
        colors_classified = colors_[classified]

        return colors_classified

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, new_value: bool):
        if not self._is_categorical and new_value > len(self._unique_values):
            raise ValueError(
                "'k' cannot be greater than the number of unique values in the column.'"
                f"Got new k={new_value} and previous k={len(self._unique_values)}.'"
                #  f"{''.join(self._unique_values)}"
            )
        self._k = int(new_value)

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, new_value: bool):
        self._cmap = new_value
        self.change_cmap(cmap=new_value, start=self.cmap_start, stop=self.cmap_stop)

    @property
    def gdf(self):
        return self._gdf

    @gdf.setter
    def gdf(self, _):
        raise ValueError(
            "Cannot change 'gdf' after init. Put the GeoDataFrames into "
            "the class initialiser."
        )

    @property
    def gdfs(self):
        return self._gdfs

    @gdfs.setter
    def gdfs(self, _):
        raise ValueError(
            "Cannot change 'gdfs' after init. Put the GeoDataFrames into "
            "the class initialiser."
        )

    @property
    def column(self):
        return self._column

    @column.setter
    def column(self, _):
        raise ValueError(
            "Cannot change 'column' after init. Specify 'column' in the "
            "class initialiser."
        )

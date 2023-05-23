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
    2: "#ffa617",
    3: "#ff8cc9",
    4: "#804e00",
    5: "#99ff00",
    6: "#fff700",
    7: "#00ffee",
    8: "#36d19b",
    9: "#94006b",
    10: "#750000",
    11: "#1c6b00",
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

        self._gdfs: list[GeoDataFrame] = [gdf.reset_index(drop=True) for gdf in gdfs]
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

        self._get_unique_values()

    def _get_unique_values(self):
        if not self._is_categorical:
            self._unique_values = self._get_unique_floats()
            if self._k > len(self._unique_values):
                self._k = len(self._unique_values)
        else:
            self._unique_values = sorted(
                list(self._gdf.loc[~self._nan_idx, self._column].unique())
            )

    def _get_unique_floats(self) -> np.array:
        """Get unique floats by multiplying, then converting to integer.

        Find a multiplier that makes the max value greater than +- 1_000_000.
        Because floats don't always equal each other. This will make very
        similar values count as the same value in the color classification.
        """
        array = self._gdf.loc[~self._nan_idx, self._column]

        self._min = np.min(array)
        self._max = np.max(array)
        self._get_multiplier(array)

        unique = array.reset_index(drop=True).drop_duplicates()
        as_int = self._array_to_large_int(unique)
        no_duplicates = as_int.drop_duplicates()

        return np.sort(np.array(unique.loc[no_duplicates.index]))

    def _array_to_large_int(self, array):
        """Multiply values in float array, then convert to integer."""
        if not isinstance(array, pd.Series):
            array = pd.Series(array)

        notna = array[array.notna()]
        isna = array[array.isna()]

        unique_multiplied = (notna * self._multiplier).astype(np.int64)

        return pd.concat([unique_multiplied, isna]).sort_index()

    def _get_multiplier(self, array: np.ndarray):
        """Find the number of zeros needed to push the max value of the array above
        +-1_000_000.

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
        bins = [bin for bin in bins]

        if min(bins) > 0 and min(self._gdf.loc[~self._nan_idx, self._column]) < min(
            bins
        ):
            bins = [min(self._gdf.loc[~self._nan_idx, self._column])] + bins

        if min(bins) < 0 and min(self._gdf.loc[~self._nan_idx, self._column]) < min(
            bins
        ):
            bins = [min(self._gdf.loc[~self._nan_idx, self._column])] + bins

        if max(bins) > 0 and max(self._gdf[self._column]) > max(bins):
            bins = bins + [max(self._gdf[self._column])]

        if max(bins) < 0 and max(self._gdf[self._column]) < max(bins):
            bins = bins + [max(self._gdf[self._column])]

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

        default_scheme = "fisherjenks"

        if not hasattr(self, "scheme"):
            self.scheme = self.kwargs.pop("scheme", default_scheme)

        if self.scheme is None:
            return

        if not self.bins:
            self.bins = self._create_bins(self._gdf, self._column)
            if len(self.bins) <= self._k and len(self.bins) != len(self._unique_values):
                self._k = len(self.bins)
        else:
            self.bins = self._add_minmax_to_bins(self.bins)
            if len(self._unique_values) <= len(self.bins):
                self._k = len(self.bins)  # - 1

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
        if crs_list:
            self.crs = crs_list[0]
        new_gdfs = []
        for gdf in gdfs:
            gdf = gdf.reset_index(drop=True)
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
            raise ValueError(f"All values are NaN in column {self.column!r}.")

        if col_not_present == len(self._gdfs):
            raise ValueError(f"{self.column} not found.")

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

        n_classes = (
            self._k if len(self._unique_values) > self._k else len(self._unique_values)
        )

        if self._k == len(self._unique_values) - 1:
            n_classes = self._k - 1

        if self.scheme == "jenks":
            bins = jenks_breaks(gdf.loc[~self._nan_idx, column], n_classes=n_classes)
        else:
            binning = classify(
                np.asarray(gdf.loc[~self._nan_idx, column]),
                scheme=self.scheme,
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

        return np.array(bins)

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

        # if equal lenght, convert to integer and check for equality
        if len(bins) == len(self._unique_values):
            if gdf[self._column].isna().all():
                return np.repeat(len(bins), len(gdf))

            gdf["col_as_int"] = self._array_to_large_int(gdf[self._column])
            bins = self._array_to_large_int(self._unique_values)
            classified = np.searchsorted(bins, gdf["col_as_int"])
        else:
            if len(bins) == self._k + 1:
                bins = bins[1:]

            if gdf[self._column].isna().all():
                return np.repeat(len(bins), len(gdf))

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

    def __setitem__(self, item, new_item):
        return setattr(self, item, new_item)

    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, ValueError, IndexError, AttributeError):
            return default

"""Attributes of the legend of the ThematicMap class.

The Legend class is best accessed through the 'legend' attribute of the ThematicMap
class.

"""

import itertools
import warnings
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from matplotlib.lines import Line2D
from pandas import Series

from ..geopandas_tools.bounds import bounds_to_points
from ..geopandas_tools.general import points_in_bounds

# the geopandas._explore raises a deprication warning. Ignoring for now.
warnings.filterwarnings(
    action="ignore", category=matplotlib.MatplotlibDeprecationWarning
)
pd.options.mode.chained_assignment = None


LEGEND_KWARGS = {
    "title",
    "size",
    "position",
    "fontsize",
    "title_fontsize",
    "markersize",
    "framealpha",
    "edgecolor",
    "kwargs",
    "labelspacing",
    "title_color",
    "width",
    "height",
    "labels",
    "pretty_labels",
    "thousand_sep",
    "decimal_mark",
    "label_sep",
    "label_suffix",
    "rounding",
    "facecolor",
    "labelcolor",
}

LOWERCASE_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "but",
    "by",
    "for",
    "in",
    "nor",
    "of",
    "on",
    "or",
    "the",
    "up",
}


def prettify_label(label: str) -> str:
    """Replace underscores with spaces and capitalize words that are all lowecase."""
    if len(label) == 1:
        return label
    return " ".join(
        (word.title() if word.islower() and word not in LOWERCASE_WORDS else word)
        for word in label.replace("_", " ").split()
    )


def prettify_number(x: int | float, rounding: int) -> int:
    rounding = int(float(f"1e+{abs(rounding)}"))
    rounded_down = int(x // rounding * rounding)
    rounded_up = rounded_down + rounding
    diff_up = abs(x - rounded_up)
    diff_down = abs(x - rounded_down)
    if diff_up < diff_down:
        return rounded_up
    else:
        return rounded_down


def prettify_bins(bins: list[int | float], rounding: int) -> list[int]:
    return [
        (
            prettify_number(x, rounding)
            if i != len(bins) - 1
            else int(x)
            # else prettify_number(x, rounding) + abs(rounding)
        )
        for i, x in enumerate(bins)
    ]


class Legend:
    """Holds the general attributes of the legend in the ThematicMap class.

    This class holds attributes of the 'legend' attribute of the ThematicMap class.
    The fontsize, title_fontsize and markersize attributes are adjusted
    according to the size attribute of the ThematicMap.

    If a numeric column is used, additional attributes can be found in the
    ContinousLegend class.

    Examples:
    ---------
    Create ten points with a numeric column from 0 to 9.

    >>> import sgis as sg
    >>> points = sg.to_gdf(
    ...     [
    ...         (0, 1),
    ...         (1, 0),
    ...         (1, 1),
    ...         (0, 0),
    ...         (0.5, 0.5),
    ...         (0.5, 0.25),
    ...         (0.25, 0.25),
    ...         (0.75, 0.75),
    ...         (0.25, 0.75),
    ...         (0.75, 0.25),
    ...     ]
    ... )
    >>> points["number"] = range(10)
    >>> points
                      geometry  number
    0  POINT (0.00000 1.00000)       0
    1  POINT (1.00000 0.00000)       1
    2  POINT (1.00000 1.00000)       2
    3  POINT (0.00000 0.00000)       3
    4  POINT (0.50000 0.50000)       4
    5  POINT (0.50000 0.25000)       5
    6  POINT (0.25000 0.25000)       6
    7  POINT (0.75000 0.75000)       7
    8  POINT (0.25000 0.75000)       8
    9  POINT (0.75000 0.25000)       9

    Creating the ThematicMap instance will also create the legend. Since we
    pass a numeric column, a ContinousLegend is created.

    >>> m = sg.ThematicMap(
    ...     points,
    ...     column="number"
    ...     legend_kwargs=dict(
    ...         title="Meters",
    ...         label_sep="to",
    ...         label_suffix="num",
    ...         rounding=2,
    ...         position = (0.35, 0.28),
    ...         title_fontsize=11,
    ...         fontsize=9,
    ...         markersize=7.5,
    ...     ),
    ... )
    >>> m.plot()
    >>> m.legend
    <sgis.maps.legend.ContinousLegend object at 0x00000222206738D0>
    """

    def __init__(
        self,
        title: str | None = None,
        pretty_labels: bool = True,
        labels: list[str] | None = None,
        position: tuple[float] | None = None,
        markersize: int | None = None,
        fontsize: int | None = None,
        title_fontsize: int | None = None,
        framealpha: float = 1.0,
        edgecolor: str = "#0f0f0f",
        **kwargs,
    ) -> None:
        """Initialiser.

        Args:
            title: Legend title. Defaults to the column name if used in the
                ThematicMap class.
            pretty_labels: If True, words will be capitalized and underscores turned to spaces.
                If continous values, numbers will be rounded.
            labels: Labels of the categories.
            position: The legend's x and y position in the plot, specified as a tuple of
                x and y position between 0 and 1. E.g. position=(0.8, 0.2) for a position
                in the bottom right corner, (0.2, 0.8) for the upper left corner.
            fontsize: Text size of the legend labels. Defaults to the size of
                the ThematicMap class.
            title_fontsize: Text size of the legend title. Defaults to the
                size * 1.2 of the ThematicMap class.
            markersize: Size of the color circles in the legend. Defaults to the size of
                the ThematicMap class.
            framealpha: Transparency of the legend background.
            edgecolor: Color of the legend border. Defaults to #0f0f0f (almost black).
            kwargs: Stores additional keyword arguments taken by the matplotlib legend
                method. Specify this as e.g. m.legend.kwargs["labelcolor"] = "red", where
                'm' is the name of the ThematicMap instance. See here:
                https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html

        """
        self.title = title

        if "size" in kwargs:
            size = kwargs.pop("size")
            self._get_legend_sizes(size, kwargs)
        else:
            self._title_fontsize = title_fontsize
            self._fontsize = fontsize
            self._markersize = markersize

        self.pretty_labels = pretty_labels
        self.framealpha = framealpha
        self.edgecolor = edgecolor
        self.width = kwargs.pop("width", 0.1)
        self.height = kwargs.pop("height", 0.1)
        self.title_color = kwargs.pop("title_color", None)
        self.labelspacing = kwargs.pop("labelspacing", 0.8)

        self.labels = labels
        self._position = position
        self._position_has_been_set = True if position else False

        self.kwargs = {}
        for key, value in kwargs.items():
            if key not in LEGEND_KWARGS:
                self.kwargs[key] = value
            else:
                try:
                    setattr(self, key, value)
                except Exception:
                    setattr(self, f"_{key}", value)

    @property
    def valid_keywords(self) -> set[str]:
        """List all valid keywords for the class initialiser."""
        return LEGEND_KWARGS

    def _get_legend_sizes(self, size: int | float, kwargs: dict) -> None:
        """Adjust fontsize and markersize to size kwarg."""
        if "title_fontsize" in kwargs:
            self._title_fontsize = kwargs["title_fontsize"]
            self._title_fontsize_has_been_set = True
        else:
            self._title_fontsize = size * 1.2

        if "fontsize" in kwargs:
            self._fontsize = kwargs["fontsize"]
            self._fontsize_has_been_set = True
        else:
            self._fontsize = size

        if "markersize" in kwargs:
            self._markersize = kwargs["markersize"]
            self._markersize_has_been_set = True
        else:
            self._markersize = size

    def _prepare_categorical_legend(
        self, categories_colors: dict, nan_label: str
    ) -> None:
        for attr in self.__dict__.keys():
            if attr in self.kwargs:
                self[attr] = self.kwargs.pop(attr)

        # swap column values with label values if labels is dict
        if self.labels and isinstance(self.labels, dict):
            categories_colors = {
                self.labels[cat]: color for cat, color in categories_colors.items()
            }
        # swap column values with label list and hope it's in the correct order
        elif self.labels:
            categories_colors = {
                label: color
                for label, color in zip(
                    self.labels, categories_colors.values(), strict=True
                )
            }

        self._patches, self._categories = [], []
        for category, color in categories_colors.items():
            if self.pretty_labels:
                category = prettify_label(category)
            if category == nan_label:
                self._categories.append(nan_label)
            else:
                self._categories.append(category)
            self._patches.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker="o",
                    alpha=self.kwargs.get("alpha", 1),
                    markersize=self._markersize,
                    markerfacecolor=color,
                    markeredgewidth=0,
                )
            )

    def _actually_add_legend(self, ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
        if self.pretty_labels:
            self.title = prettify_label(self.title)
        legend = ax.legend(
            self._patches,
            self._categories,
            fontsize=self._fontsize,
            title=self.title,
            title_fontsize=self._title_fontsize,
            bbox_to_anchor=self._position + (self.width, self.height),
            fancybox=False,
            framealpha=self.framealpha,
            edgecolor=self.edgecolor,
            labelspacing=self.labelspacing,
            **self.kwargs,
        )

        if self.title_color:
            plt.setp(legend.get_title(), color=self.title_color)

        return ax

    def _get_best_legend_position(
        self, gdf: GeoDataFrame, k: int
    ) -> tuple[float, float]:
        minx, miny, maxx, maxy = gdf.total_bounds
        diffx = maxx - minx
        diffy = maxy - miny

        points = pd.concat(
            [
                points_in_bounds(gdf, 30),
                bounds_to_points(gdf)
                .geometry.explode(ignore_index=True)
                .to_frame("geometry"),
            ]
        )

        gdf = gdf.loc[:, ~gdf.columns.str.contains("index|level_")]
        joined = points.sjoin_nearest(gdf, distance_col="nearest")

        max_distance = max(joined["nearest"])

        best_position = joined.loc[joined["nearest"] == max_distance].drop_duplicates(
            "geometry"
        )

        bestx, besty = (
            best_position.geometry.x.iloc[0],
            best_position.geometry.y.iloc[0],
        )

        bestx_01 = (bestx - minx) / (diffx)
        besty_01 = (besty - miny) / (diffy)

        bestx_01 = 0.1 if bestx_01 < 0.5 else 0.90
        besty_01 = 0.0375 * k if besty_01 < 0.5 else 1

        return bestx_01, besty_01

    def __getitem__(self, item: str) -> Any:
        """Get attribute with square brackets."""
        return getattr(self, item)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set attribute with square brackets."""
        setattr(self, key, value)

    def get(self, key: Any, default: Any = None) -> Any:
        """Get value of an attribute of the Legend."""
        try:
            return self[key]
        except (KeyError, ValueError, IndexError, AttributeError):
            return default

    @property
    def position(self) -> tuple[float, float]:
        """Legend position in x, y."""
        return self._position

    @position.setter
    def position(self, new_value: tuple[float, float]) -> None:
        self._position = new_value
        self._position_has_been_set = True

    @property
    def title_fontsize(self) -> int:
        """Legend title fontsize."""
        return self._title_fontsize

    @title_fontsize.setter
    def title_fontsize(self, new_value: int) -> None:
        self._title_fontsize = new_value
        self._title_fontsize_has_been_set = True

    @property
    def fontsize(self) -> int:
        """Legend fontsize."""
        return self._fontsize

    @fontsize.setter
    def fontsize(self, new_value: int) -> None:
        self._fontsize = new_value
        self._fontsize_has_been_set = True

    @property
    def markersize(self) -> int:
        """Legend markersize."""
        return self._markersize

    @markersize.setter
    def markersize(self, new_value: int) -> None:
        self._markersize = new_value
        self._markersize_has_been_set = True


class ContinousLegend(Legend):
    """Holds the legend attributes specific to numeric columns.

    The attributes consern the labeling of the groups in the legend.
    Labels can be set manually with the 'labels' attribute, or the format
    of the labels can be changed with the remaining attributes.

    Attributes:
        labels: To manually set labels. If set, all other labeling attributes are
            ignored. Should be given as a list of strings with the same length as
            the number of color groups.
        pretty_labels: If False (default), the minimum and maximum values of each
            color group will be used as legend labels. If True, the labels will end
            with the maximum value, but start at 1 + the maximum value of the previous
            group. The labels will be correct but inaccurate.
        label_suffix: The text to put after each number in the legend labels.
            Defaults to None.
        label_sep: Text to put in between the two numbers in each color group in
            the legend. Defaults to '-'.
        thousand_sep: Separator between each thousand for large numbers. Defaults to
            None, meaning no separator.
        decimal_mark: Text to use as decimal point. Defaults to None, meaning '.' (dot)
            unless 'thousand_sep' is '.'. In this case, ',' (comma) will be used as
            decimal mark.

    Examples:
    ---------
    Create ten random points with a numeric column from 0 to 9.

    >>> import sgis as sg
    >>> points = sg.random_points(10)
    >>> points["number"] = range(10)
    >>> points
                    geometry  number
    0  POINT (0.59780 0.50425)       0
    1  POINT (0.07019 0.26167)       1
    2  POINT (0.56475 0.15422)       2
    3  POINT (0.87293 0.60316)       3
    4  POINT (0.47373 0.20040)       4
    5  POINT (0.98661 0.15614)       5
    6  POINT (0.30951 0.77057)       6
    7  POINT (0.47802 0.52824)       7
    8  POINT (0.12215 0.96588)       8
    9  POINT (0.02938 0.93467)       9

    Creating the ThematicMap instance with a numeric column.

    >>> m = sg.ThematicMap(points, column="number")

    Changing the attributes that apply to both numeric and categorical columns.

    >>> m.legend.title = "Meters"
    >>> m.legend.position = (0.35, 0.28)

    Change the attributes that only apply to numeric columns.

    >>> m.label_sep = "to"
    >>> m.label_suffix = "num"
    >>> m.rounding = 2
    >>> m.plot()

    Setting labels manually. For better control, it might be wise to also set the bins
    manually. The following bins will create a plot with the color groups
    0-2, 3-5, 6-7 and 8-9. The legend labels can then be set accordingly.

    >>> m = sg.ThematicMap(points, column="number")
    >>> m.bins = [0, 2, 5, 7, 9]
    >>> m.legend.labels = ["0 to 2 num", "3 to 5 num", "6 to 7 num", "8 to 9 num"]
    >>> m.plot()

    We will get the same groups if we exclude the first and last bin values. The
    minimum and maximum values will be filled anyway.

    >>> m = sg.ThematicMap(points, column="number")
    >>> m.bins = [2, 5, 7]
    >>> m.legend.labels = ["0 to 2 num", "3 to 5 num", "6 to 7 num", "8 to 9 num"]
    >>> m.plot()

    """

    def __init__(
        self,
        labels: list[str] | None = None,
        pretty_labels: bool = True,
        label_suffix: str | None = None,
        label_sep: str = "-",
        rounding: int | None = None,
        thousand_sep: str | None = None,
        decimal_mark: str | None = None,
        **kwargs,
    ) -> None:
        """Initialiser.

        Args:
            labels: To manually set labels. If set, all other labeling attributes are
                ignored. Should be given as a list of strings with the same length as
                the number of color groups.
            pretty_labels: If False (default), the minimum and maximum values of each
                color group will be used as legend labels. If True, the labels will end
                with the maximum value, but start at 1 + the maximum value of the previous
                group. The labels will be correct but inaccurate.
            label_suffix: The text to put after each number in the legend labels.
                Defaults to None.
            label_sep: Text to put in between the two numbers in each color group in
                the legend. Defaults to '-'.
            rounding: Number of decimals in the labels. By default, the rounding
                depends on the column's maximum value and standard deviation.
                OBS: The bins will not be rounded, meaning the labels might be wrong
                if not bins are set manually.
            thousand_sep: Separator between each thousand for large numbers. Defaults to
                None, meaning no separator.
            decimal_mark: Text to use as decimal point. Defaults to None, meaning '.' (dot)
                unless 'thousand_sep' is '.'. In this case, ',' (comma) will be used as
                decimal mark.
            kwargs: Stores additional keyword arguments taken by the matplotlib legend
                method. Specify this as e.g. m.legend.kwargs["labelcolor"] = "red", where
                'm' is the name of the ThematicMap instance. See here:
                https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html

        """
        super().__init__(**kwargs)

        self.pretty_labels = pretty_labels
        self.thousand_sep = thousand_sep
        self.decimal_mark = decimal_mark

        self.label_sep = label_sep
        self.label_suffix = "" if not label_suffix else label_suffix
        self._rounding = rounding
        # self._rounding_has_been_set = True if rounding else False

    def _get_rounding(self, array: Series | np.ndarray) -> int:
        def isinteger(x):
            return np.equal(np.mod(x, 1), 0)

        if np.all(isinteger(array)):
            return 0

        closest_to_zero_idx = np.argmin(np.abs(array))
        closest_to_zero = np.abs(array[closest_to_zero_idx])

        between_1_and_0 = 1 > closest_to_zero > 0
        if between_1_and_0:
            return int(abs(np.log10(abs(closest_to_zero)))) + 1

        std_ = np.std(array)
        max_ = np.max(array)
        if max_ > 30 and std_ > 5:
            return 0
        if max_ > 5 and std_ > 1:
            return 1
        if max_ > 1 and std_ > 0.1:
            return 2
        return int(abs(np.log10(std_))) + 1

    @staticmethod
    def _set_rounding(bins, rounding: int | float) -> list[int | float]:
        if not rounding:
            return [int(round(bin_, 0)) for bin_ in bins]
        elif rounding <= 0:
            return [int(round(bin_, rounding)) for bin_ in bins]
        else:
            return [round(bin_, rounding) for bin_ in bins]

    def _remove_max_legend_value(self) -> None:
        if not self._legend:
            raise ValueError("Cannot modify legend before it is created.")

    def _prepare_continous_legend(
        self,
        bins: list[float],
        colors: list[str],
        nan_label: str,
        bin_values: dict,
    ) -> None:
        # TODO: clean up this messy method

        for attr in self.kwargs:
            if attr in self.__dict__:
                self[attr] = self.kwargs.pop(attr)

        self._patches, self._categories = [], []

        for color in colors:
            self._patches.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker="o",
                    alpha=self.kwargs.get("alpha", 1),
                    markersize=self._markersize,
                    markerfacecolor=color,
                    markeredgewidth=0,
                )
            )

        if self.labels:
            if len(self.labels) != len(colors):
                raise ValueError(
                    "Label list must be same length as the number of groups. "
                    f"Got k={len(colors)} and labels={len(self.labels)}."
                    f"labels: {', '.join(self.labels)}"
                    f"colors: {', '.join(colors)}"
                    f"bins: {bins}"
                )
            self._categories = self.labels

        elif len(bins) == len(colors):
            for i, _ in enumerate(bins):
                min_ = np.min(bin_values[i])
                max_ = np.max(bin_values[i])
                try:
                    min_rounded = self._set_rounding([min_], self._rounding)[0]
                    max_rounded = self._set_rounding([max_], self._rounding)[0]
                except ValueError as e:
                    if "nan" in str(e).lower():
                        min_rounded, max_rounded == "NaN", "NaN"
                if min_ == max_:
                    self._categories.append(f"{min_rounded} {self.label_suffix}")
                else:
                    self._categories.append(
                        f"{min_rounded} {self.label_suffix} {self.label_sep} "
                        f"{max_rounded} {self.label_suffix}"
                    )

        else:
            for i, (cat1, cat2) in enumerate(itertools.pairwise(bins)):
                if nan_label in str(cat1) or nan_label in str(cat2):
                    self._categories.append(nan_label)
                    continue

                min_ = np.min(bin_values[i])
                max_ = np.max(bin_values[i])

                if self.pretty_labels:
                    if i == 0:
                        cat1 = int(min_) if (self.rounding or 0) <= 0 else min_

                    is_last = i == len(bins) - 2
                    if is_last:
                        cat2 = int(max_) if (self.rounding or 0) <= 0 else max_

                    if (self.rounding or 0) <= 0:
                        cat1 = int(cat1)
                        cat2 = int(cat2 - 1) if not is_last else int(cat2)
                    elif (self.rounding or 0) > 0:
                        cat1 = round(cat1, self._rounding)
                        cat2 = round(
                            cat2 - float(f"1e-{self._rounding}"), self._rounding
                        )
                    else:
                        cat1 = round(cat1, self._rounding)
                        cat2 = round(cat2, self._rounding)

                    cat1 = self._format_number(cat1)
                    cat2 = self._format_number(cat2)

                    if min_ == max_:
                        label = self._get_two_value_label(cat1, cat2)
                        self._categories.append(label)
                        continue

                    label = self._get_two_value_label(cat1, cat2)
                    self._categories.append(label)

                    continue

                min_rounded = self._set_rounding([min_], self._rounding)[0]
                max_rounded = self._set_rounding([max_], self._rounding)[0]
                if min_ == max_:
                    min_rounded = self._format_number(min_rounded)
                    label = self._get_one_value_label(min_rounded)
                    self._categories.append(label)
                else:
                    min_rounded = self._format_number(min_rounded)
                    max_rounded = self._format_number(max_rounded)
                    label = self._get_two_value_label(min_rounded, max_rounded)
                    self._categories.append(label)

    def _get_two_value_label(self, value1: int | float, value2: int | float) -> str:
        return (
            f"{value1} {self.label_suffix} {self.label_sep} "
            f"{value2} {self.label_suffix}"
        )

    def _get_one_value_label(self, value1: int | float) -> str:
        return f"{value1} {self.label_suffix}"

    def _format_number(self, number: int | float) -> int | float:
        if not self.thousand_sep and not self.decimal_mark:
            return number

        if self.thousand_sep:
            number = f"{number:,}".replace(",", "*temp_thousand*")

        if self.decimal_mark:
            number = str(number).replace(".", "*temp_decimal*")

        if self.thousand_sep == "." and not self.decimal_mark:
            number = number.replace(".", ",").replace(
                "*temp_thousand*", self.thousand_sep
            )
        elif not self.thousand_sep:
            number = number.replace("*temp_decimal*", self.decimal_mark)
        elif not self.decimal_mark:
            number = number.replace("*temp_thousand*", self.thousand_sep)
        else:
            number = number.replace("*temp_thousand*", self.thousand_sep).replace(
                "*temp_decimal*", self.decimal_mark
            )
        return number

    @property
    def rounding(self) -> int:
        """Number of decimals in the labels.

        By default, the rounding
        depends on the column's maximum value and standard deviation.
        OBS: The bins will not be rounded, meaning the labels might be wrong
        if not bins are set manually.
        """
        return self._rounding

    @rounding.setter
    def rounding(self, new_value: int) -> None:
        self._rounding = new_value

"""Attributes of the legend of the ThematicMap class.

The Legend class is best accessed through the 'legend' attribute of the ThematicMap
class.

"""
import warnings
from statistics import mean

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from matplotlib.lines import Line2D
from pandas import Series

from ..geopandas_tools.general import points_in_bounds, to_gdf
from ..geopandas_tools.point_operations import snap_all


# the geopandas._explore raises a deprication warning. Ignoring for now.
warnings.filterwarnings(
    action="ignore", category=matplotlib.MatplotlibDeprecationWarning
)
pd.options.mode.chained_assignment = None


class Legend:
    """Holds the attributes of the legend in the ThematicMap class.

    This class is stored in the 'legend' attribute of the ThematicMap class.
    The fontsize, title_fontsize and markersize attributes are adjusted
    according to the size attribute of the ThematicMap.

    The attributes 'label_suffix', 'label_sep' and 'rounding' only apply to plots
    of numeric columns.

    The 'labels' attribute can be used to set labels manually. By default, the
    maximum and minimum values of each color group is used as label for numeric
    columns. For categorical columns, the column values are used.

    Attributes:
        title: Legend title. Defaults to the column name if used in the
            ThematicMap class.
        labels: To manually set labels for the color groups. Must be a list/tuple of
            same length as the number of color groups (k).
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
        label_suffix: For numeric columns. The text to put after each number in the
            legend labels.
        label_sep: For numeric columns. Text to put in between the two numbers in each
            color group in the legend.
        rounding: For numeric columns. Number of decimals in the legend labels. By
            default the rounding depends on the column's maximum value and standard
            deviation.
        kwargs: Stores additional keyword arguments taken by the matplotlib legend
            method. Specify this as e.g. m.legend.kwargs["labelcolor"] = "red", where
            'm' is the name of the ThematicMap instance. See here:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html

    Examples
    --------
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

    Creating the ThematicMap instance.

    >>> m = sg.ThematicMap(points, column="number")

    Changing the attributes that apply to both numeric and categorical columns.

    >>> m.legend.title = "Meters"
    >>> m.legend.title_fontsize = 11
    >>> m.legend.fontsize = 9
    >>> m.legend.markersize = 7.5
    >>> m.legend.position = (0.35, 0.28)
    >>> m.legend.kwargs["labelcolor"] = "red"
    >>> m.plot()

    Changing the additional attributes that only apply only to numeric columns.

    >>> m = sg.ThematicMap(points, column="number")
    >>> m.label_sep = "to"
    >>> m.label_suffix = "num"
    >>> m.rounding = 2
    >>> m.plot()

    The final attribute, labels, should be changed along with the bins attribute
    of the ThematicMap class. The following bins will create a plot with the color
    groups 0-2, 3-5, 6-7 and 8-9. The legend labels can then be set accordingly.

    >>> m.bins = [2, 5, 7]
    >>> m.legend.labels = ["0 to 2 num", "3 to 5 num", "6 to 7 num", "8 to 9 num"]
    >>> m.plot()

    """

    def __init__(
        self,
        title: str | None = None,
        labels: list[str] | None = None,
        label_suffix: str = "",
        label_sep: str = "-",
        rounding: int | None = None,
        position: tuple[float] | None = None,
        markersize: int | None = None,
        fontsize: int | None = None,
        title_fontsize: int | None = None,
        framealpha: float = 1.0,
        edgecolor: str = "#0f0f0f",
        **kwargs,
    ):
        self.title = title

        if "size" in kwargs:
            size = kwargs.pop("size")
            self._get_legend_sizes(size, kwargs)
        else:
            self._title_fontsize = title_fontsize
            self._fontsize = fontsize
            self._markersize = markersize

        self.framealpha = framealpha
        self.edgecolor = edgecolor
        self.width = kwargs.pop("width", 0.1)
        self.height = kwargs.pop("height", 0.1)
        self.title_color = kwargs.pop("title_color", None)
        self.labelspacing = kwargs.pop("labelspacing", 0.8)

        self.label_suffix = label_suffix
        self.label_sep = label_sep
        self.labels = labels
        self._rounding = rounding
        self._position = position
        self.kwargs = kwargs
        self._position_has_been_set = True if position else False
        self._rounding_has_been_set = True if rounding else False

    def _get_legend_sizes(self, size, kwargs):
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

    def _get_rounding(self, array: Series | np.ndarray) -> int:
        def isinteger(x):
            return np.equal(np.mod(x, 1), 0)

        if np.all(isinteger(array)):
            return 0
        if np.max(array) > 30 and np.std(array) > 5:
            return 0
        if np.max(array) > 5 and np.std(array) > 1:
            return 1
        if np.max(array) > 1 and np.std(array) > 0.1:
            return 2
        return int(abs(np.log10(np.std(array)))) + 1

    @staticmethod
    def _set_rounding(bins, rounding: int | float):
        if rounding == 0:
            return [int(round(bin, 0)) for bin in bins]
        else:
            return [round(bin, rounding) for bin in bins]

    def _remove_max_legend_value(self):
        if not self._legend:
            raise ValueError("Cannot modify legend before it is created.")

    def _actually_add_continous_legend(
        self,
        ax,
        bins: list[float],
        colors: list[str],
        nan_label: str,
        bin_values: dict,
    ):
        for attr in self.__dict__.keys():
            if attr in self.kwargs:
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
                    f"Label list must be same length as 'k'. Got k={len(colors)} and "
                    f"labels={len(self.labels)}"
                )
            self._categories = self.labels

        elif len(bins) == len(colors):
            for i, _ in enumerate(bins):
                min_ = np.min(bin_values[i])
                max_ = np.max(bin_values[i])
                min_rounded = self._set_rounding([min_], self._rounding)[0]
                max_rounded = self._set_rounding([max_], self._rounding)[0]
                if min_ == max_:
                    self._categories.append(f"{min_rounded} {self.label_suffix}")
                else:
                    self._categories.append(
                        f"{min_rounded} {self.label_suffix} {self.label_sep} "
                        f"{max_rounded} {self.label_suffix}"
                    )

        else:
            for i, (cat1, cat2) in enumerate(zip(bins[:-1], bins[1:], strict=True)):
                if nan_label in str(cat1) or nan_label in str(cat2):
                    self._categories.append(nan_label)
                else:
                    min_ = np.min(bin_values[i])
                    max_ = np.max(bin_values[i])
                    min_rounded = self._set_rounding([min_], self._rounding)[0]
                    max_rounded = self._set_rounding([max_], self._rounding)[0]
                    if min_ == max_:
                        self._categories.append(f"{min_rounded} {self.label_suffix}")
                    else:
                        self._categories.append(
                            f"{min_rounded} {self.label_suffix} {self.label_sep} "
                            f"{max_rounded} {self.label_suffix}"
                        )

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

    def _actually_add_categorical_legend(
        self, ax, categories_colors: dict, nan_label: str
    ):
        for attr in self.__dict__.keys():
            if attr in self.kwargs:
                self[attr] = self.kwargs.pop(attr)

        if self.labels and isinstance(self.labels, dict):
            categories_colors = {
                self.labels[cat]: color for cat, color in categories_colors.items()
            }
        elif self.labels:
            categories_colors = {
                label: color
                for label, color in zip(
                    self.labels, categories_colors.values(), strict=True
                )
            }

        self._patches, self._categories = [], []
        for category, color in categories_colors.items():
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

    def _get_best_legend_position(self, gdf, k: int):
        minx, miny, maxx, maxy = gdf.total_bounds
        diffx = maxx - minx
        diffy = maxy - miny

        points = points_in_bounds(gdf, 30)
        gdf = gdf.loc[:, ~gdf.columns.str.contains("index|level_")]
        joined = points.sjoin_nearest(gdf, distance_col="nearest")

        max_distance = max(joined.nearest)

        best_position = joined.loc[joined.nearest == max_distance].drop_duplicates(
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

    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, ValueError, IndexError, AttributeError):
            return default

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, new_value: bool):
        self._position = new_value
        self._position_has_been_set = True

    @property
    def rounding(self):
        return self._rounding

    @rounding.setter
    def rounding(self, new_value: bool):
        self._rounding = new_value
        self._rounding_has_been_set = True

    @property
    def title_fontsize(self):
        return self._title_fontsize

    @title_fontsize.setter
    def title_fontsize(self, new_value: bool):
        self._title_fontsize = new_value
        self._title_fontsize_has_been_set = True

    @property
    def fontsize(self):
        return self._fontsize

    @fontsize.setter
    def fontsize(self, new_value: bool):
        self._fontsize = new_value
        self._fontsize_has_been_set = True

    @property
    def markersize(self):
        return self._markersize

    @markersize.setter
    def markersize(self, new_value: bool):
        self._markersize = new_value
        self._markersize_has_been_set = True


class _ContinousLegend(Legend):
    """Holds the legend attributes specific to numeric columns.

    This class is stored in the 'legend' attribute of the ThematicMap class.
    The fontsize, title_fontsize and markersize attributes are adjusted
    according to the size attribute of the ThematicMap.

    The 'labels' attribute can be used to set labels manually. By default, the
    maximum and minimum values of each color group is used as label.

    Attributes:
        labels: To manually set labels for the color groups. Must be a list/tuple of
            same length as the number of color groups (k).
        label_suffix: For numeric columns. The text to put after each number in the
            legend labels.
        label_sep: For numeric columns. Text to put in between the two numbers in each
            color group in the legend.
        rounding: Number of decimals for numeric columns. By default the rounding
            depends on the column's

    Examples
    --------
    """

    pass


class _CategoricalLegend(Legend):
    """Holds the attributes of the legend in the ThematicMap class.

    This class is stored in the 'legend' attribute of the ThematicMap class.
    The fontsize, title_fontsize and markersize attributes are adjusted
    according to the size attribute of the ThematicMap.

    Attributes:
        title: Legend title. Defaults to the column name if used in the
            ThematicMap class.
        position: The legend's x and y position in the plot, specified as a tuple of
            x and y position between 0 and 1. E.g. position=(0.8, 0.2) for a position
            in the bottom right corner, (0.2, 0.8) for the upper left corner.
        fontsize: Text size of the legend labels. Defaults to the size of
            the ThematicMap class.
        title_fontsize: Text size of the legend title. Defaults to the
            size * 1.2 of the ThematicMap class.
        markersize: Size of the color circles in the legend. Defaults to the size of
            the ThematicMap class.
        kwargs: Stores additional keyword arguments taken by the matplotlib legend
            method. Specify this as e.g. m.legend.kwargs["labelcolor"] = "red", where
            'm' is the name of the ThematicMap instance.

    Examples
    --------
    """

    pass

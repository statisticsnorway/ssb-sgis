"""Interactive map of one or more GeoDataFrames with layers that can be toggles on/off.

This module holds the Explore class, which is the basis for the explore, samplemap and
clipmap functions from the 'maps' module.
"""
import warnings

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from pandas import Series

from ..geopandas_tools.general import points_in_bounds


# the geopandas._explore raises a deprication warning. Ignoring for now.
warnings.filterwarnings(
    action="ignore", category=matplotlib.MatplotlibDeprecationWarning
)
pd.options.mode.chained_assignment = None


class Legend:
    def __init__(
        self,
        title: str | None = None,
        markersize: int | None = None,
        fontsize: int | None = None,
        title_fontsize: int | None = None,
        labels: list[str] | None = None,
        label_suffix: str = "",
        label_sep: str = "-",
        rounding: int | None = None,
        position: tuple[float] | None = None,
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
        self, ax, bins: list[float], colors: list[str], nan_label: str, bin_values: dict
    ):
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
                    f"Label list must be same length as 'k'. Got k={len(colors)} and labels={self.labels}"
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
                        f"{min_rounded} {self.label_suffix} {self.label_sep} {max_rounded} {self.label_suffix}"
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
                            f"{min_rounded} {self.label_suffix} {self.label_sep} {max_rounded} {self.label_suffix}"
                        )

        ax.legend(
            self._patches,
            self._categories,
            fontsize=self._fontsize,
            title=self.title,
            title_fontsize=self._title_fontsize,
            bbox_to_anchor=self._position,
            fancybox=False,
            **self.kwargs,
        )
        return ax

    def _actually_add_categorical_legend(
        self, ax, categories_colors: dict, nan_label: str
    ):
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
        ax.legend(
            self._patches,
            self._categories,
            fontsize=self._fontsize,
            title=self.title,
            title_fontsize=self._title_fontsize,
            bbox_to_anchor=self._position,
            fancybox=False,
            **self.kwargs,
        )
        return ax

    def _get_best_legend_position(self, gdf):
        points = points_in_bounds(gdf, 30)
        gdf = gdf.loc[:, ~gdf.columns.str.contains("index|level_")]
        joined = points.sjoin_nearest(gdf, distance_col="nearest")
        best_position = joined.loc[
            joined.nearest == max(joined.nearest)
        ].drop_duplicates("geometry")

        x, y = best_position.geometry.x.iloc[0], best_position.geometry.y.iloc[0]

        minx, miny, maxx, maxy = gdf.total_bounds

        bestx = (x - minx) / (maxx - minx)
        besty = (y - miny) / (maxy - miny)

        if bestx < 0.2:
            bestx = bestx + 0.2 - bestx
        if besty < 0.2:
            besty = besty + 0.2 - besty

        self._position = bestx, besty

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

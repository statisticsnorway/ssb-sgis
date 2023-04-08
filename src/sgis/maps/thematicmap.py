"""Interactive map of one or more GeoDataFrames with layers that can be toggles on/off.

This module holds the Explore class, which is the basis for the explore, samplemap and
clipmap functions from the 'maps' module.
"""
import warnings

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame

from .legend import Legend
from .map import Map


# the geopandas._explore raises a deprication warning. Ignoring for now.
warnings.filterwarnings(
    action="ignore", category=matplotlib.MatplotlibDeprecationWarning
)
pd.options.mode.chained_assignment = None


class ThematicMap(Map):
    """Class for customising static maps with geopandas.

    The class takes one or more GeoDataFrames, a column name
    """

    def __init__(
        self,
        *gdfs: GeoDataFrame,
        column: str | None = None,
        title: str | None = None,
        size: int = 10,
        black: bool = False,
        bins: tuple[float] | None = None,
        **kwargs,
    ):
        super().__init__(*gdfs, column=column, bins=bins, **kwargs)

        self._size = size
        self._black = black
        self.background_gdfs = []

        self.title = title

        if "title_fontsize" in kwargs:
            self.title_fontsize = kwargs["title_fontsize"]
            self._title_fontsize_has_been_set = True
        else:
            self._title_fontsize = self._size * 2

        self._black_or_white()

        if not self._is_categorical:
            cmap_kwargs = {}
            if "cmap_start" in kwargs:
                cmap_kwargs["cmap_start"] = kwargs.pop("cmap_start")
            if "cmap_stop" in kwargs:
                cmap_kwargs["cmap_stop"] = kwargs.pop("cmap_stop")
            self._choose_cmap(
                cmap=self._cmap,
                **cmap_kwargs,
            )

    def _choose_cmap(self, cmap: str | None, **kwargs):
        """kwargs is to catch start and stop points for the cmap in __init__."""
        if cmap:
            self._cmap = cmap
            self.cmap_start = kwargs.get("cmap_start", 0)
            self.cmap_stop = kwargs.get("cmap_stop", 256)
        elif self._black:
            self._cmap = "viridis"
            self.cmap_start = kwargs.get("cmap_start", 0)
            self.cmap_stop = kwargs.get("cmap_stop", 256)
        else:
            self._cmap = "RdPu"
            self.cmap_start = kwargs.get("cmap_start", 25)
            self.cmap_stop = kwargs.get("cmap_stop", 256)

    def change_cmap(self, cmap: str, start: int = 0, stop: int = 256):
        self.cmap_start = start
        self.cmap_stop = stop
        self._cmap = cmap
        return self

    def add_background(self, gdf, color: str | None = None):
        if color:
            self.bg_gdf_color = color
        if not hasattr(self, "_background_gdfs"):
            self._background_gdfs = gdf
        else:
            self._background_gdfs = pd.concat(
                [self._background_gdfs, gdf], ignore_index=True
            )
        self.minx, self.miny, self.maxx, self.maxy = self._gdf.total_bounds
        self.diffx = self.maxx - self.minx
        self.diffy = self.maxy - self.miny

    def plot(self):
        """Creates the final plot. This method should be run last."""

        self.fig, self.ax = self._get_matplotlib_figure_and_axix(
            figsize=(self._size, self._size)
        )
        self.fig.patch.set_facecolor(self.facecolor)
        self.ax.set_axis_off()

        if hasattr(self, "_background_gdfs"):
            self._actually_add_background()

        if hasattr(self, "title") and self.title:
            self.ax.set_title(
                self.title, fontsize=self.title_fontsize, color=self.title_color
            )

        if not self._is_categorical:
            self._prepare_continous_map()
            self.colorlist = self._get_continous_colors()
            self.colors = self._classify_from_bins(self._gdf)
        else:
            self._get_categorical_colors()
            self.colors = self._gdf["color"]

        if hasattr(self, "legend") and self._is_categorical:
            self.add_categorical_legend()
            self.ax = self.legend._actually_add_categorical_legend(
                ax=self.ax,
                categories_colors=self._categories_colors_dict,
                nan_label=self.nan_label,
            )
        elif hasattr(self, "legend") and not self._is_categorical:
            self.add_continous_legend()

            if self.legend._rounding is not None:
                self.bins = self.legend._set_rounding(
                    bins=self.bins, rounding=self.legend._rounding
                )

            if any(self._nan_idx):
                self.bins = self.bins + [self.nan_label]

            self.ax = self.legend._actually_add_continous_legend(
                ax=self.ax,
                bins=self.bins,
                colors=self.colorlist,
                nan_label=self.nan_label,
                bin_values=self._bins_unique_values,
            )

        self._gdf.plot(color=self.colors, legend=hasattr(self, "legend"), ax=self.ax)

    def _remove_max_legend_value(self):
        if not self.legend:
            raise ValueError("Cannot modify legend before it is created.")

    def add_continous_legend(
        self,
        **kwargs,
    ):
        if hasattr(self, "legend"):
            for key, value in kwargs.items():
                self.legend[key] = value
        else:
            title = kwargs.pop("title", self._column)

            self.legend = Legend(
                title=title,
                size=self._size,
                **kwargs,
            )

        if "rounding" in kwargs:
            self.legend._rounding = kwargs["rounding"]
        elif not self.legend._rounding_has_been_set:
            self.legend._rounding = self.legend._get_rounding(
                array=self._gdf.loc[~self._nan_idx, self._column]
            )

        if "position" in kwargs:
            self.legend._position = kwargs["position"]
        elif not self.legend._position_has_been_set:
            self.legend._get_best_legend_position(self._gdf)

    def add_categorical_legend(
        self,
        **kwargs,
    ):
        if hasattr(self, "legend"):
            for key, value in kwargs.items():
                self.legend[key] = value
        else:
            title = kwargs.pop("title", self._column)

            self.legend = Legend(
                title=title,
                size=self._size,
                **kwargs,
            )

        if "position" in kwargs:
            self.legend._position = kwargs["position"]
        elif not self.legend._position_has_been_set:
            self.legend._get_best_legend_position(self._gdf)

    def save(self, path):
        try:
            plt.savefig(path)
        except Exception:
            from dapla import FileClient

            fs = FileClient.get_gcs_file_system()
            with fs.open(path, "wb") as file:
                plt.savefig(file)

    def _actually_add_background(self):
        self.ax.set_xlim([self.minx - self.diffx * 0.03, self.maxx + self.diffx * 0.03])
        self.ax.set_ylim([self.miny - self.diffy * 0.03, self.maxy + self.diffy * 0.03])
        self._background_gdfs.plot(ax=self.ax, color=self.bg_gdf_color)

    @staticmethod
    def _get_matplotlib_figure_and_axix(figsize: tuple[int, int]):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        return fig, ax

    def _black_or_white(self):
        if self._black:
            self.facecolor, self.title_color, self.bg_gdf_color = (
                "#0f0f0f",
                "#fefefe",
                "#383834",
            )
        else:
            self.facecolor, self.title_color, self.bg_gdf_color = (
                "#fefefe",
                "#0f0f0f",
                "#d1d1cd",
            )

    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, ValueError, IndexError, AttributeError):
            return default

    @property
    def black(self):
        return self._black

    @black.setter
    def black(self, new_value: bool):
        self._black = new_value
        self._black_or_white()

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, new_value: bool):
        self._cmap = new_value
        self.change_cmap(cmap=new_value)

    @property
    def title_fontsize(self):
        return self._title_fontsize

    @title_fontsize.setter
    def title_fontsize(self, new_value: bool):
        self._title_fontsize = new_value
        self._title_fontsize_has_been_set = True

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, new_value: bool):
        """Adjust font and marker size if not actively set."""
        self._size = new_value
        if not hasattr(self, "_title_fontsize_has_been_set"):
            self._title_fontsize = self._size * 2
        if not hasattr(self, "legend"):
            return
        if not hasattr(self.legend, "_title_fontsize_has_been_set"):
            self.legend._title_fontsize = self._size * 1.2
        if not hasattr(self.legend, "_fontsize_has_been_set"):
            self.legend._fontsize = self._size
        if not hasattr(self.legend, "_markersize_has_been_set"):
            self.legend._markersize = self._size

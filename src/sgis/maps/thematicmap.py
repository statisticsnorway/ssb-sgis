"""Make static maps with geopandas and matplotlib."""
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from .legend import ContinousLegend, Legend
from .map import Map


# the geopandas._explore raises a deprication warning. Ignoring for now.
warnings.filterwarnings(
    action="ignore", category=matplotlib.MatplotlibDeprecationWarning
)
pd.options.mode.chained_assignment = None


class ThematicMap(Map):
    """Class for creating static maps with geopandas and matplotlib.

    The class takes one or more GeoDataFrames and a column name. The class attributes
    can then be set to customise the map before plotting.

    Args:
        *gdfs: One or more GeoDataFrames.
        column: The name of the column to plot.
        size: Width and height of the plot in inches. Fontsize of title and legend is
            adjusted accordingly. Defaults to 25.
        black: If False (default), the background will be white and the text black. If
            True, the background will be black and the text white. When True, the
            default cmap is "viridis", and when False, the default is red to purple
            (RdPu).

    Attributes:
        size (int): Width and height of the plot in inches.
        k (int): Number of color groups.
        legend (Legend): The legend object of the map. The legend holds its own set of
            attributes. See the Legend class for details.
        title (str): Title of the plot.
        title_color (str): Color of the title font.
        title_fontsize (int): Color of the title font.
        bins (list[int | float]): For numeric columns. List of numbers that define the
            maximum value for the color groups.
        cmap (str): Colormap of the plot. See:
            https://matplotlib.org/stable/tutorials/colors/colormaps.html
        cmap_start (int): Start position for the color palette.
        cmap_stop (int): End position for the color palette.
        facecolor (str): Background color.

    Examples
    --------
    >>> import sgis as sg
    >>> points = sg.random_points(100).pipe(sg.buff, np.random.rand(100))
    >>> points2 = sg.random_points(100).pipe(sg.buff, np.random.rand(100))

    Simple plot with legend and title.

    >>> m = sg.ThematicMap(points, points2, "area")
    >>> m.title = "Area of random circles"
    >>> m.plot()

    Plot with custom legend units (label_suffix) and separator (label_sep).

    >>> m = sg.ThematicMap(points, points2, "area")
    >>> m.title = "Area of random circles"
    >>> m.legend.label_suffix = "m2"
    >>> m.legend.label_sep = "to"
    >>> m.plot()

    With custom bins and legend labels.

    >>> m = sg.ThematicMap(points, points2, "area")
    >>> m.title = "Area of random circles"
    >>> m.bins = [1, 2, 3]
    >>> m.legend.labels = [
    ...     f"{int(round(min(points.length),0))} to 1",
    ...     "1 to 2",
    ...     "2 to 3",
    ...     f"3 to {int(round(max(points.length),0))}",
    ... ]
    >>> m.plot()
    """

    def __init__(
        self,
        *gdfs: GeoDataFrame,
        column: str | None = None,
        size: int = 25,
        black: bool = False,
    ):
        super().__init__(*gdfs, column=column)

        self._size = size
        self._black = black
        self.background_gdfs = []

        self._title_fontsize = self._size * 2

        self.black = black

        if not self._is_categorical:
            self._choose_cmap()

        self._create_legend()

    def change_cmap(self, cmap: str, start: int = 0, stop: int = 256):
        """Change the color palette of the plot.

        Args:
            cmap: The colormap.
                https://matplotlib.org/stable/tutorials/colors/colormaps.html
            start: Start position for the color palette. Defaults to 0.
            stop: End position for the color palette. Defaults to 256, which
                is the end of the color range.
        """
        super().change_cmap(cmap, start, stop)
        return self

    def add_background(self, gdf, color: str | None = None):
        """Add a GeoDataFrame as a background layer.

        Args:
            gdf: a GeoDataFrame.
            color: Single color. Defaults to gray (shade depends on whether the map
                facecolor is black or white).
        """
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
        return self

    def plot(self, **kwargs) -> None:
        """Creates the final plot.

        This method should be run after customising the map, but before saving.
        """

        __test = kwargs.pop("__test", False)
        include_legend = bool(kwargs.pop("legend", self.legend))

        if "color" in kwargs:
            kwargs.pop("column", None)
            self.legend = None
            include_legend = False
        elif hasattr(self, "color"):
            kwargs.pop("column", None)
            kwargs["color"] = self.color
            self.legend = None
            include_legend = False

        elif self._is_categorical:
            kwargs = self._prepare_categorical_plot(kwargs)
            if self.legend:
                self.legend._prepare_categorical_legend(
                    categories_colors=self._categories_colors_dict,
                    nan_label=self.nan_label,
                )

        else:
            kwargs = self._prepare_continous_plot(kwargs)
            if self.legend:
                if not self.legend._rounding_has_been_set:
                    self.legend._rounding = self.legend._get_rounding(
                        array=self._gdf.loc[~self._nan_idx, self._column]
                    )

                self.legend._prepare_continous_legend(
                    bins=self.bins,
                    colors=self._unique_colors,
                    nan_label=self.nan_label,
                    bin_values=self._bins_unique_values,
                )

        if self.legend and not self.legend._position_has_been_set:
            self.legend._position = self.legend._get_best_legend_position(
                self._gdf, k=self._k + bool(len(self._nan_idx))
            )

        if __test:
            return

        self._prepare_plot(**kwargs)

        if self.legend:
            self.ax = self.legend._actually_add_legend(ax=self.ax)

        self._gdf.plot(legend=include_legend, ax=self.ax, **kwargs)

    def save(self, path: str) -> None:
        """Save figure as image file.

        To be run after the plot method.

        Args:
            path: File path.
        """
        try:
            plt.savefig(path)
        except FileNotFoundError:
            from dapla import FileClient

            fs = FileClient.get_gcs_file_system()
            with fs.open(path, "wb") as file:
                plt.savefig(file)

    def _prepare_plot(self, **kwargs):
        """Add figure and axis, title and background gdf."""
        for attr in self.__dict__.keys():
            if attr in self.kwargs:
                self[attr] = self.kwargs.pop(attr)
            if attr in kwargs:
                self[attr] = kwargs.pop(attr)

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

    def _prepare_continous_plot(self, kwargs) -> dict:
        """Create bins and colors."""
        self._prepare_continous_map()

        if self.scheme is None:
            self.legend = None
            kwargs["column"] = self.column
            return kwargs

        elif self.bins is None:
            kwargs["column"] = self.column
            return kwargs

        else:
            classified = self._classify_from_bins(self._gdf, bins=self.bins)
            classified_sequential = self._push_classification(classified)
            n_colors = len(np.unique(classified_sequential)) - any(self._nan_idx)
            self._unique_colors = self._get_continous_colors(n=n_colors)
            self._bins_unique_values = self._make_bin_value_dict(
                self._gdf, classified_sequential
            )
            colorarray = self._unique_colors[classified_sequential]
            kwargs["color"] = colorarray

        if self.legend and not self.legend._rounding_has_been_set:
            self.bins = self.legend._set_rounding(
                bins=self.bins, rounding=self.legend._rounding
            )

            if any(self._nan_idx):
                self.bins = self.bins + [self.nan_label]

        return kwargs

    def _prepare_categorical_plot(self, kwargs) -> dict:
        """Map values to colors."""
        self._get_categorical_colors()
        colorarray = self._gdf["color"]

        kwargs["color"] = colorarray
        return kwargs

    def _actually_add_legend(self) -> None:
        """Add legend to the axis and fill it with colors and labels."""
        if not self.legend._position_has_been_set:
            self.legend._position = self.legend._get_best_legend_position(
                self._gdf, k=self._k + bool(len(self._nan_idx))
            )

        if self._is_categorical:
            self.ax = self.legend._actually_add_categorical_legend(
                ax=self.ax,
                categories_colors=self._categories_colors_dict,
                nan_label=self.nan_label,
            )
        else:
            self.ax = self.legend._actually_add_continous_legend(
                ax=self.ax,
                bins=self.bins,
                colors=self._unique_colors,
                nan_label=self.nan_label,
                bin_values=self._bins_unique_values,
            )

    def _create_legend(self):
        """Instantiate the Legend class."""
        kwargs = {}
        if self._black:
            kwargs["facecolor"] = "#0f0f0f"
            kwargs["labelcolor"] = "#fefefe"
            kwargs["title_color"] = "#fefefe"

        if self._is_categorical:
            self.legend = Legend(title=self._column, size=self._size, **kwargs)
        else:
            self.legend = ContinousLegend(title=self._column, size=self._size, **kwargs)

    def _choose_cmap(self):
        """kwargs is to catch start and stop points for the cmap in __init__."""
        if self._black:
            self._cmap = "viridis"
            self.cmap_start = 0
            self.cmap_stop = 256
        else:
            self._cmap = "RdPu"
            self.cmap_start = 23
            self.cmap_stop = 256

    def _make_bin_value_dict(self, gdf, classified) -> dict:
        """Dict with unique values of all bins. Used in labels in ContinousLegend."""
        bins_unique_values = {
            i: list(set(gdf.loc[classified == i, self._column]))
            for i, _ in enumerate(np.unique(classified))
        }
        return bins_unique_values

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
                "#383836",
            )
            self.nan_color = "#666666"
            if not self._is_categorical:
                self.change_cmap("viridis")

        else:
            self.facecolor, self.title_color, self.bg_gdf_color = (
                "#fefefe",
                "#0f0f0f",
                "#ebebeb",
            )
            self.nan_color = "#c2c2c2"
            if not self._is_categorical:
                self.change_cmap("RdPu", start=23)

        self._create_legend()

    @property
    def black(self):
        return self._black

    @black.setter
    def black(self, new_value: bool):
        self._black = new_value
        self._black_or_white()

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

    def __setattr__(self, __name: str, __value) -> None:
        if "legend_" in __name:
            last_part = __name.split("legend_")[-1]
            raise AttributeError(
                f"Invalid attribute {__name!r}. Did you mean 'legend.{last_part}'?"
            )
        return super().__setattr__(__name, __value)

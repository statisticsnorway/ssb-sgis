"""Creating static maps with geopandas and matplotlib."""
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
    """Class for creating static maps with geopandas and matplotlib.

    The class takes one or more GeoDataFrames and a column name. The class attributes
    can then be set to customise the map before plotting.

    Args:
        *gdfs: One or more GeoDataFrames.
        column: The name of the column to plot.
        size: Width and height of the plot in inches. Fontsize of title and legend is
            adjusted accordingly.
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
        title_color (str): Color of the title.

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
        size: int = 10,
        black: bool = False,
    ):
        super().__init__(*gdfs, column=column)

        self._size = size
        self._black = black
        self.background_gdfs = []

        self._title_fontsize = self._size * 2

        self._black_or_white()

        if not self._is_categorical:
            self._choose_cmap(cmap=self._cmap)

        self._add_legend()

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

    def plot(self) -> None:
        """Creates the final plot.

        This method should be run after customising the map, but before saving.
        """

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

        if self.legend and self._is_categorical:
            self.ax = self.legend._actually_add_categorical_legend(
                ax=self.ax,
                categories_colors=self._categories_colors_dict,
                nan_label=self.nan_label,
            )
        elif self.legend and not self._is_categorical:
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

        self._gdf.plot(color=self.colors, legend=bool(self.legend), ax=self.ax)

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

    def _add_legend(self):
        self.legend = Legend(title=self._column, size=self._size)

        self.legend._get_best_legend_position(self._gdf)

        if not self._is_categorical:
            self.legend._rounding = self.legend._get_rounding(
                array=self._gdf.loc[~self._nan_idx, self._column]
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
            self.cmap_start = kwargs.get("cmap_start", 33)
            self.cmap_stop = kwargs.get("cmap_stop", 256)

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
        else:
            self.facecolor, self.title_color, self.bg_gdf_color = (
                "#fefefe",
                "#0f0f0f",
                "#ebebeb",
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

"""Make static maps with geopandas and matplotlib."""

import warnings
from typing import Any

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from ..geopandas_tools.conversion import to_bbox
from ..geopandas_tools.conversion import to_gdf
from ..helpers import is_property
from .legend import LEGEND_KWARGS
from .legend import ContinousLegend
from .legend import Legend
from .legend import prettify_bins
from .map import Map
from .map import _determine_best_name

# the geopandas._explore raises a deprication warning. Ignoring for now.
warnings.filterwarnings(
    action="ignore", category=matplotlib.MatplotlibDeprecationWarning
)
pd.options.mode.chained_assignment = None

MAP_KWARGS = {
    "bins",
    "title",
    "title_fontsize",
    "size",
    "cmap",
    "cmap_start",
    "cmap_stop",
    "scheme",
    "k",
    "column",
    "title_color",
    "facecolor",
    "labelcolor",
    "nan_color",
    # "alpha",
    "title_kwargs",
    "bg_gdf_color",
    "title_position",
    # "linewidth",
}


class ThematicMap(Map):
    """Class for making static maps.

    Args:
        *gdfs: One or more GeoDataFrames.
        column: The name of the column to plot.
        bounds: Optional bounding box for the map.
        title: Title of the plot.
        title_position: Title position. Either "center" (default), "left" or "right".
        size: Width and height of the plot in inches. Fontsize of title and legend is
            adjusted accordingly. Defaults to 25.
        dark: If False (default), the background will be white and the text black. If
            True, the background will be black and the text white. When True, the
            default cmap is "viridis", and when False, the default is red to purple
            (RdPu).
        cmap: Colormap of the plot. See:
            https://matplotlib.org/stable/tutorials/colors/colormaps.html
        scheme: How to devide numeric values into categories. Defaults to
            "naturalbreaks".
        k: Number of color groups.
        bins: For numeric columns. List of numbers that define the
            maximum value for the color groups.
        nan_label: Label for missing data.
        nan_hatch: Hatch for missing data. See https://matplotlib.org/stable/gallery/shapes_and_collections/hatch_style_reference.html.
        legend_kwargs: dictionary with attributes for the legend. E.g.:
            title: Legend title. Defaults to the column name.
            rounding: If positive number, it will round floats to n decimals.
            If negative, eg. -2, the number 3429 is rounded to 3400.
            By default, the rounding depends on the column's maximum value
            and standard deviation.
            position: The legend's x and y position in the plot. By default, it's
            decided dynamically by finding the space with most distance to
            the geometries. To be specified as a tuple of
            x and y position between 0 and 1. E.g. position=(0.8, 0.2) for a position
            in the bottom right corner, (0.2, 0.8) for the upper left corner.
            pretty_labels: Whether to capitalize words in text categories.
            label_suffix: For numeric columns. The text to put after each number
            in the legend labels. Defaults to None.
            label_sep: For numeric columns. Text to put in between the two numbers
            in each color group in the legend. Defaults to '-'.
            thousand_sep: For numeric columns. Separator between each thousand for
            large numbers. Defaults to None, meaning no separator.
            decimal_mark: For numeric columns. Text to use as decimal point.
            Defaults to None, meaning '.' (dot) unless 'thousand_sep' is
            '.'. In this case, ',' (comma) will be used as decimal mark.
        **kwargs: Additional attributes for the map. E.g.:
            title_color (str): Color of the title font.
            title_fontsize (int): Color of the title font.
            cmap_start (int): Start position for the color palette.
            cmap_stop (int): End position for the color palette.
            facecolor (str): Background color.
            labelcolor (str): Color for the labels.
            nan_color: Color for missing data.

    Examples:
    ---------
    >>> import sgis as sg
    >>> points = sg.random_points(100, loc=1000).pipe(sg.buff, np.random.rand(100) * 100)
    >>> points2 = sg.random_points(100, loc=1000).pipe(sg.buff, np.random.rand(100) * 100)


    Simple plot with legend and title.

    >>> m = sg.ThematicMap(points, points2, column="area", title="Area of random circles")
    >>> m.plot()

    Plot with custom legend units (label_suffix) and thousand separator.
    And with rounding set to -2, meaning e.g. 3429 is rounded to 3400.
    If rounding was set to positive 2, 3429 would be rounded to 3429.00.

    >>> m = sg.ThematicMap(
    ...     points,
    ...     points2,
    ...     column="area",
    ...     title = "Area of random circles",
    ...     legend_kwargs=dict(
    ...         rounding=-2,
    ...         thousand_sep=" ",
    ...         label_sep="to",
    ...     ),
    ... )
    >>> m.plot()

    With custom bins for the categories, and other customizations.

    >>> m = sg.ThematicMap(
    ...     points,
    ...     points2,
    ...     column="area",
    ...     cmap="Greens",
    ...     cmap_start=50,
    ...     cmap_stop=255,
    ...     nan_label="Missing",
    ...     title = "Area of random circles",
    ...     bins = [5000, 10000, 15000, 20000],
    ...     title_kwargs=dict(
    ...         loc="left",
    ...         y=0.93,
    ...         x=0.025,
    ...     ),
    ...     legend_kwargs=dict(
    ...         thousand_sep=" ",
    ...         label_sep="to",
    ...         decimal_mark=".",
    ...         label_suffix="m2",
    ...     ),
    ... )
    >>> m.plot()
    """

    def __init__(
        self,
        *gdfs: GeoDataFrame,
        column: str | None = None,
        bounds: tuple | None = None,
        title: str | None = None,
        title_position: tuple[float, float] | None = None,
        size: int = 25,
        dark: bool = False,
        cmap: str | None = None,
        scheme: str = "naturalbreaks",
        k: int = 5,
        bins: tuple[float] | None = None,
        nan_label: str = "Missing",
        nan_color: str | None = None,
        nan_hatch: str | None = None,
        hatch: str | None = None,
        legend_kwargs: dict | None = None,
        title_kwargs: dict | None = None,
        legend: bool = True,
        **kwargs,
    ) -> None:
        """Initializer."""
        new_gdfs = {}
        for i, gdf in enumerate(gdfs):
            if isinstance(gdf, str):
                raise ValueError("gdfs cannot be a string in ThematicMap.")
            name = _determine_best_name(gdf, column, i)
            if name in new_gdfs:
                name += str(i)
            try:
                new_gdfs[name] = to_gdf(gdf)
            except Exception:
                continue

        new_kwargs = {}
        self.kwargs = {}
        for key, value in kwargs.items():
            try:
                new_gdfs[key] = to_gdf(value)
            except Exception:
                new_kwargs[key] = value

        super().__init__(
            column=column,
            scheme=scheme,
            k=k,
            bins=bins,
            nan_label=nan_label,
            nan_color=nan_color,
            **new_gdfs,
        )

        self.kwargs = self.kwargs | new_kwargs

        self.title = title
        self._size = size
        self.hatch = hatch
        self._dark = dark
        self.title_kwargs = title_kwargs or {}
        if title_position and "position" in self.title_kwargs:
            raise TypeError(
                "Specify either 'title_position' or title_kwargs position, not both."
            )
        if title_position or "position" in self.title_kwargs:
            position = self.title_kwargs.pop("position", title_position)
            error_mess = (
                "legend_kwargs position should be a two length tuple/list with two numbers between "
                "0 and 1 (x, y position)"
            )
            if not hasattr(position, "__len__"):
                raise TypeError(error_mess)
            if len(position) != 2:
                raise ValueError(error_mess)
            x, y = position
            if "loc" not in self.title_kwargs:
                if x < 0.4:
                    self.title_kwargs["loc"] = "left"
                elif x > 0.6:
                    self.title_kwargs["loc"] = "right"
                else:
                    self.title_kwargs["loc"] = "center"

            self.title_kwargs["x"], self.title_kwargs["y"] = x, y
        self.background_gdfs = []

        legend_kwargs = legend_kwargs or {}

        self._title_fontsize = self._size * 1.9

        black = self.kwargs.pop("black", None)
        self._dark = self._dark or black

        if not self.cmap and not self._is_categorical:
            self._choose_cmap(**kwargs)

        if not legend:
            self.legend = None
        else:
            self._create_legend()

        new_kwargs = {}
        for key, value in self.kwargs.items():
            if key not in MAP_KWARGS:
                new_kwargs[key] = value
            elif is_property(self, key):
                setattr(self, f"_{key}", value)
            else:
                setattr(self, key, value)
        self.kwargs = new_kwargs

        for key, value in legend_kwargs.items():
            if key not in LEGEND_KWARGS:
                raise TypeError(
                    f"{self.__class__.__name__} legend_kwargs got an unexpected key {key}"
                )
            if self.legend is not None:
                try:
                    setattr(self.legend, key, value)
                except Exception:
                    setattr(self.legend, f"_{key}", value)

        self._dark_or_light()

        if cmap:
            self._cmap = cmap

        self.bounds = (
            to_bbox(bounds) if bounds is not None else to_bbox(self._gdf.total_bounds)
        )
        self.minx, self.miny, self.maxx, self.maxy = self.bounds
        self.diffx = self.maxx - self.minx
        self.diffy = self.maxy - self.miny

        if self._gdf[self._column].isna().any():
            isnas = []
            for label, gdf in self._gdfs.items():
                isnas.append(gdf[gdf[self._column].isna()])
                self._gdfs[label] = gdf[gdf[self._column].notna()]
            self._more_data[nan_label] = {
                "gdf": pd.concat(isnas, ignore_index=True),
                "color": self.nan_color,
                "hatch": nan_hatch,
            } | new_kwargs
            self._gdf = pd.concat(self.gdfs.values(), ignore_index=True)

    @property
    def valid_keywords(self) -> set[str]:
        """List all valid keywords for the class initialiser."""
        return MAP_KWARGS

    def change_cmap(self, cmap: str, start: int = 0, stop: int = 256) -> "ThematicMap":
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

    def add_data(
        self,
        *,
        color: str | None = None,
        hatch: str | None = None,
        **kwargs,
    ) -> "ThematicMap":
        """Add Geometric Data of a given color or hatch.

        The geodata must be passed as keyword argument.
        The keyword will be used as label in the legend.

        Args:
            color: Color of the data.
            hatch: Hatch of the data. See https://matplotlib.org/stable/gallery/shapes_and_collections/hatch_style_reference.html.
            **kwargs: Must include exactly one GeoDataFrame or object that can be converted to GeoDataFrame.
                Additional kwargs are passed to geopandas.GeoDataFrame.plot and matplotlib.patches.Patch.
        """
        new_kwargs = {}
        n = 0
        for key, value in kwargs.items():
            try:
                gdf = to_gdf(value)
                label = key
                n += 1
            except Exception:
                new_kwargs[key] = value

        if n != 1:
            raise ValueError(
                "Must specify a single geometry object as keyword argument."
            )
        if not color and not hatch:
            raise TypeError("Must pass either 'color' or 'hatch'.")
        if hatch is not None:
            color = self.facecolor
        gdf["label"] = label
        self._more_data[label] = {
            "gdf": gdf,
            "color": color,
            "hatch": hatch,
        } | new_kwargs
        if self.bounds is None:
            self.bounds = to_bbox(self._gdf.total_bounds)
        return self

    def add_background(
        self,
        gdf: GeoDataFrame,
        color: str | None = None,
        **kwargs,
    ) -> "ThematicMap":
        """Add a GeoDataFrame as a background layer.

        Args:
            gdf: a GeoDataFrame.
            color: Single color. Defaults to gray (shade depends on whether the map
                facecolor is black or white).
            **kwargs: Keyword arguments sent to GeoDataFrame.plot.
        """
        if color:
            self.bg_gdf_color = color
        if not hasattr(self, "_background_gdfs"):
            self._background_gdfs = gdf
        else:
            self._background_gdfs = pd.concat(
                [self._background_gdfs, gdf], ignore_index=True
            )
        if self.bounds is None:
            self.bounds = to_bbox(self._gdf.total_bounds)
        self.bg_gdf_kwargs = kwargs
        return self

    def plot(self, **kwargs) -> None:
        """Creates the final plot.

        This method should be run after customising the map, but before saving.

        """
        kwargs = kwargs | self.kwargs
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
            self._prepare_categorical_plot()
            if self._gdf is not None:
                kwargs["color"] = self._gdf["color"]
            if self.legend:
                self.legend._prepare_categorical_legend(
                    categories_colors=self._categories_colors_dict,
                    hatch=self.hatch,
                )

        else:
            kwargs = self._prepare_continous_plot(kwargs)
            if self.legend:
                if not self.legend.rounding:
                    self.legend._rounding = self.legend._get_rounding(
                        array=self._gdf[self._column].dropna()
                    )

                self.legend._prepare_continous_legend(
                    bins=self.bins,
                    colors=self._unique_colors,
                    bin_values=self._bins_unique_values,
                    nan_label=self.nan_label,
                    hatch=self.hatch,
                )

        if self.legend and self._more_data:
            self.legend._add_more_data_to_legend(self._more_data)

        if self.legend and not self.legend._position_has_been_set:
            self.legend._position = self.legend._get_best_legend_position(
                self._gdf, k=self._k + self._gdf[self._column].isna().any()
            )

        self._prepare_plot(**kwargs)

        if self.legend:
            self.ax = self.legend._actually_add_legend(ax=self.ax)

        self.ax = self._gdf.plot(
            legend=include_legend, ax=self.ax, hatch=self.hatch, **kwargs
        )

        if __test:
            return self

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

    def _prepare_plot(self, **kwargs) -> None:
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
        self.ax.set_xlim([self.minx - self.diffx * 0.03, self.maxx + self.diffx * 0.03])
        self.ax.set_ylim([self.miny - self.diffy * 0.03, self.maxy + self.diffy * 0.03])

        if hasattr(self, "_background_gdfs"):
            self._actually_add_background()
        # if self.bounds is not None:

        for datadict in self._more_data.values():
            gdf = datadict["gdf"]
            datadict = {key: value for key, value in datadict.items() if key != "gdf"}
            gdf.plot(ax=self.ax, **datadict)

        if self.title:
            self.ax.set_title(
                self.title,
                **(
                    dict(fontsize=self.title_fontsize, color=self.title_color)
                    | self.title_kwargs
                ),
            )

    def _prepare_continous_plot(self, kwargs: dict) -> dict:
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
            if self.legend and self.legend.rounding and self.legend.rounding < 0:
                self.bins = prettify_bins(self.bins, self.legend.rounding)
                self.bins = list({round(bin_, 5) for bin_ in self.bins})
                self.bins.sort()
                # self.legend._rounding_was = self.legend.rounding
                # self.legend.rounding = None

            classified = self._classify_from_bins(self._gdf, bins=self.bins)
            classified_sequential = self._push_classification(classified)
            n_colors = (
                len(np.unique(classified_sequential))
                - self._gdf[self._column].isna().any()
            )
            self._unique_colors = self._get_continous_colors(n=n_colors)
            self._bins_unique_values = self._make_bin_value_dict(
                self._gdf, classified_sequential
            )

            colorarray = self._unique_colors[classified_sequential]
            kwargs["color"] = colorarray

        if (
            self.legend and self.legend.rounding
        ):  # not self.legend._rounding_has_been_set:
            self.bins = self.legend._set_rounding(
                bins=self.bins, rounding=self.legend._rounding
            )

            if self._gdf[self._column].isna().any():
                self.bins = self.bins + [self.nan_label]

        return kwargs

    def _actually_add_legend(self) -> None:
        """Add legend to the axis and fill it with colors and labels."""
        if not self.legend._position_has_been_set:
            self.legend._position = self.legend._get_best_legend_position(
                self._gdf, k=self._k + self._gdf[self._column].isna().any()
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

    def _create_legend(self) -> None:
        """Instantiate the Legend class."""
        if self._is_categorical:
            self.legend = Legend(title=self._column, size=self._size)
        else:
            self.legend = ContinousLegend(title=self._column, size=self._size)

    def _choose_cmap(self, **kwargs) -> None:
        """Kwargs is to catch start and stop points for the cmap in __init__."""
        if self._dark:
            self._cmap = "viridis"
            self.cmap_start = self.kwargs.pop("cmap_start", 0)
            self.cmap_stop = self.kwargs.pop("cmap_stop", 256)
        else:
            self._cmap = "RdPu"
            self.cmap_start = self.kwargs.pop("cmap_start", 23)
            self.cmap_stop = self.kwargs.pop("cmap_stop", 256)

    def _make_bin_value_dict(self, gdf: GeoDataFrame, classified: np.ndarray) -> dict:
        """Dict with unique values of all bins. Used in labels in ContinousLegend."""
        bins_unique_values = {
            i: list(set(gdf.loc[classified == i, self._column]))
            for i, _ in enumerate(np.unique(classified))
        }
        return bins_unique_values

    def _actually_add_background(self) -> None:
        self._background_gdfs.plot(
            ax=self.ax, color=self.bg_gdf_color, **self.bg_gdf_kwargs
        )

    @staticmethod
    def _get_matplotlib_figure_and_axix(
        figsize: tuple[int, int],
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        return fig, ax

    def _dark_or_light(self) -> None:
        if self._dark:
            self.facecolor, self.title_color, self.bg_gdf_color = (
                "#0f0f0f",
                "#fefefe",
                "#383836",
            )
            self.nan_color = "#666666" if self._nan_color_was_none else self.nan_color
            if not self._is_categorical:
                self._cmap = self.kwargs.get("cmap", "viridis")

            if self.legend is not None:
                for key, color in {
                    "facecolor": "#0f0f0f",
                    "labelcolor": "#fefefe",
                    "title_color": "#fefefe",
                    "edgecolor": "#0f0f0f",
                }.items():
                    setattr(self.legend, key, color)

        else:
            self.facecolor, self.title_color, self.bg_gdf_color = (
                "#fefefe",
                "#0f0f0f",
                "#e8e6e6",
            )
            self.nan_color = "#c2c2c2" if self._nan_color_was_none else self.nan_color
            if not self._is_categorical:
                self._cmap = self.kwargs.get("cmap", "RdPu")

            if self.legend is not None:
                for key, color in {
                    "facecolor": "#fefefe",
                    "labelcolor": "#0f0f0f",
                    "title_color": "#0f0f0f",
                    "edgecolor": "#0f0f0f",
                }.items():
                    setattr(self.legend, key, color)

    @property
    def title_fontsize(self) -> int:
        """Title fontsize, not to be confused with legend.title_fontsize."""
        return self._title_fontsize

    @title_fontsize.setter
    def title_fontsize(self, new_value: int) -> None:
        self._title_fontsize = new_value
        self._title_fontsize_has_been_set = True

    @property
    def size(self) -> int:
        """Size of the image."""
        return self._size

    @size.setter
    def size(self, new_value: bool) -> None:
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

    def __setattr__(self, __name: str, __value: Any) -> None:
        """Set an attribute with square brackets."""
        if "legend_" in __name:
            last_part = __name.split("legend_")[-1]
            raise AttributeError(
                f"Invalid attribute {__name!r}. Did you mean 'legend.{last_part}'?"
            )
        return super().__setattr__(__name, __value)

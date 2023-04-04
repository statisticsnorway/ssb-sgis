"""Interactive map of one or more GeoDataFrames with layers that can be toggles on/off.

This module holds the Explore class, which is the basis for the explore, samplemap and
clipmap functions from the 'maps' module.
"""
import warnings
from statistics import mean

import branca as bc
import folium
import geopandas
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from jenkspy import jenks_breaks
from mapclassify import classify
from matplotlib.lines import Line2D
from shapely import Geometry
from shapely.geometry import LineString, Polygon

from ..geopandas_tools.general import (
    clean_geoms,
    drop_inactive_geometry_columns,
    random_points_in_polygons,
    rename_geometry_if,
    to_gdf,
)
from ..geopandas_tools.geometry_types import get_geom_type
from ..helpers import get_name
from .map import Map


# the geopandas._explore raises a deprication warning. Ignoring for now.
warnings.filterwarnings(
    action="ignore", category=matplotlib.MatplotlibDeprecationWarning
)
pd.options.mode.chained_assignment = None


class ThematicMap(Map):
    def __init__(
        self,
        *gdfs: GeoDataFrame,
        column: str | None = None,
        size: int = 10,
        black: bool = False,
        bins: tuple[float] | None = None,
        **kwargs,
    ):
        super().__init__(*gdfs, column=column, bins=bins, **kwargs)

        self.size = size
        self.black = black
        self._legend = False
        self._has_been_plotted = False
        self._legend_fontsize = size
        self.background_gdfs = []

        if self.black:
            self.facecolor, self.title_color, self.bg_gdf_color = (
                "#0f0f0f",
                "#fefefe",
                "#383834",
            )
            self.cmap = kwargs.get("cmap", "viridis")
            self._cmap_start = 0
        else:
            self.facecolor, self.title_color, self.bg_gdf_color = (
                "#fefefe",
                "#0f0f0f",
                "#d1d1cd",
            )
            self.cmap = kwargs.get("cmap", "RdPu")
            self._cmap_start = 25

        if not self._is_categorical:
            if not self.bins:
                self.bins = self._create_bins(self.gdf, self.column)

                if len(self.bins) < self.k:
                    self.k = len(self.bins)

            self.colorlist = self._get_continous_colors(
                self.cmap, k=self.k, start=self._cmap_start
            )
            self.colors = self._classify_from_bins(
                self.bins, self.colorlist, self.gdf[self.column]
            )
        else:
            self._get_categorical_colors()
            self.colors = self.gdf["color"]

        self.fig, self.ax = self._get_matplotlib_figure_and_axix(
            figsize=(self.size, self.size)
        )

        self.fig.patch.set_facecolor(self.facecolor)
        self.ax.set_axis_off()

    def add_background(self, gdf):
        self.background_gdfs.append(gdf)
        minx, miny, maxx, maxy = self.gdf.total_bounds
        diffx = maxx - minx
        diffy = maxy - miny
        self.ax.set_xlim([minx - diffx * 0.03, maxx + diffx * 0.03])
        self.ax.set_ylim([miny - diffy * 0.03, maxy + diffy * 0.03])
        gdf.plot(ax=self.ax, color=self.bg_gdf_color)

    def plot(self):
        if not self._has_been_plotted:
            self._has_been_plotted = True
            self.gdf.plot(color=self.colors, legend=self._legend, ax=self.ax)
            self.legend = plt.gca().get_legend()
        #  else:
        # plt.show()

    def add_title(self, title, size: int | None = None):
        self.title = title
        if not size:
            size = self.size * 2
        self.ax.set_title(self.title, fontsize=size, color=self.title_color)

    def change_title_size(self, size=int):
        self.ax.set_title(self.title, fontsize=size, color=self.title_color)

    def change_legend_title(self, title: str):
        self.legend_title = title
        self.ax.legend(
            self.patches,
            self.categories,
            fontsize=self.fontsize,
            title=title,
            title_fontsize=self.title_fontsize,
            bbox_to_anchor=self.bbox_to_anchor,
            fancybox=False,
        )

    def change_legend_size(self, fontsize: int, title_fontsize: int, markersize: int):
        self.new_patches = []
        for patch in self.patches:
            patch.set_markersize(markersize)
            self.new_patches.append(patch)
        self.patches = self.new_patches
        self.ax.legend(
            self.patches,
            self.categories,
            fontsize=fontsize,
            title=self.legend_title,
            title_fontsize=title_fontsize,
            bbox_to_anchor=self.bbox_to_anchor,
            fancybox=False,
        )

    def _remove_max_legend_value(self):
        pass

    def move_legend(self, x, y):
        self.legend = self.ax.get_legend()
        self.legend.set_bbox_to_anchor((x, y))
        self.ax.plot()

    def create_bins(self, bins):
        self.bins = bins
        if len(bins) < self.k:
            self.k = len(bins)

        colors_ = self._get_continous_colors(cmap=self.cmap, k=self.k)
        self.colors = self._classify_from_bins(bins, colors_, self.gdf[self.column])

    def add_continous_legend(
        self,
        title: str | None = None,
        markersize: int | None = None,
        fontsize: int | None = None,
        title_fontsize: int | None = None,
        label_suffix: str = "",
        label_sep: str = "-",
        bin_precicion: int | float | None = None,
        **kwargs,
    ):
        self._legend = True
        self.fontsize = self.size if not fontsize else fontsize
        self.markersize = self.size if not markersize else markersize
        self.title_fontsize = self.size if not title_fontsize else title_fontsize
        self.legend_title = self.column if not title else title

        if bin_precicion is None:
            bin_precicion = self._get_bin_precicion()

        self._set_bin_precicion(bin_precicion)
        self._get_best_legend_position()

        if len(self.bins) == len(self.colorlist):
            self.patches = [
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker="o",
                    alpha=self.kwargs.get("alpha", 1),
                    markersize=self.markersize,
                    markerfacecolor=color,
                    markeredgewidth=0,
                )
                for color in self.colorlist
            ]
            self.categories = self.bins

        else:
            self.patches, self.categories = [], []
            for cat1, cat2, color in zip(
                self.bins[:-1], self.bins[1:], self.colorlist, strict=True
            ):
                self.categories.append(
                    f"{cat1} {label_suffix} {label_sep} {cat2} {label_suffix}"
                )
                self.patches.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle="none",
                        marker="o",
                        alpha=self.kwargs.get("alpha", 1),
                        markersize=self.markersize,
                        markerfacecolor=color,
                        markeredgewidth=0,
                    )
                )

        self.ax.legend(
            self.patches,
            self.categories,
            fontsize=self.fontsize,
            title=self.legend_title,
            title_fontsize=self.title_fontsize,
            loc="best",
            bbox_to_anchor=self.bbox_to_anchor,
            fancybox=False,
            **kwargs,
        )

    def add_categorical_legend(
        self,
        title: str | None = None,
        markersize: int | None = None,
        fontsize: int | None = None,
        title_fontsize: int | None = None,
        **kwargs,
    ):
        self._legend = True
        fontsize = self.size if not fontsize else fontsize
        markersize = self.size if not markersize else markersize
        title_fontsize = self.size if not title_fontsize else title_fontsize
        self.legend_title = self.column if not title else title

        self._get_best_legend_position()

        self.patches, self.categories = [], []
        for category, color in self._categories_colors_dict.items():
            self.categories.append(category)
            self.patches.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker="o",
                    alpha=self.kwargs.get("alpha", 1),
                    markersize=markersize,
                    markerfacecolor=color,
                    markeredgewidth=0,
                )
            )
        self.ax.legend(
            self.patches,
            self.categories,
            fontsize=fontsize,
            title=self.legend_title,
            title_fontsize=title_fontsize,
            bbox_to_anchor=self.bbox_to_anchor,
            **kwargs,
            fancybox=False,
        )

    @staticmethod
    def _get_matplotlib_figure_and_axix(figsize: tuple[int, int]):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        return fig, ax

    def _get_best_legend_position(self):
        minx, miny, maxx, maxy = self.gdf.total_bounds
        bbox = to_gdf(
            Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]),
            crs=self.crs,
        )
        xs = np.linspace(minx, maxx, num=20)
        ys = np.linspace(miny, maxy, num=20)
        x_coords, y_coords = np.meshgrid(xs, ys, indexing="ij")
        coords = np.concatenate(
            (x_coords.reshape(-1, 1), y_coords.reshape(-1, 1)), axis=1
        )
        points = to_gdf(coords, crs=self.crs)
        self.gdf = self.gdf.loc[:, ~self.gdf.columns.str.contains("index|level_")]
        joined = points.sjoin_nearest(self.gdf, distance_col="nearest")
        best_position = joined.loc[
            joined.nearest == max(joined.nearest)
        ].drop_duplicates("geometry")

        x, y = best_position.geometry.x.iloc[0], best_position.geometry.y.iloc[0]

        bestx = (x - minx) / (maxx - minx)
        besty = (y - miny) / (maxy - miny)
        if bestx < 0.2:
            bestx = bestx + 0.2 - bestx
        if besty < 0.2:
            besty = besty + 0.2 - besty

        self.bbox_to_anchor = bestx, besty

    def _set_bin_precicion(self, precicion: int | float):
        if precicion == 0:
            self.bins = [int(bin) for bin in self.bins]
        elif precicion > 1:
            self.bins = [int(int(bin / precicion) * precicion) for bin in self.bins]
        else:
            precicion = int(abs(np.log10(precicion)))
            self.bins = [round(bin, precicion) for bin in self.bins]

    def _get_bin_precicion(self):
        max_ = max(self.bins)
        if max_ > 30:
            return 0
        if max_ > 5:
            return 1
        if max_ > 1:
            return 2
        if max_ > 0.1:
            return 3
        if max_ > 0.01:
            return 4
        return int(abs(np.log10(max_)))

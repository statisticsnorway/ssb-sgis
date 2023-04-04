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
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from jenkspy import jenks_breaks
from mapclassify import classify
from shapely import Geometry
from shapely.geometry import LineString

from ..geopandas_tools.general import (
    clean_geoms,
    drop_inactive_geometry_columns,
    random_points_in_polygons,
    rename_geometry_if,
)
from ..geopandas_tools.geometry_types import get_geom_type
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

# gray for NaNs
NAN_COLOR = "#969696"


class Map:
    def __init__(
        self,
        *gdfs: GeoDataFrame,
        column: str | None = None,
        labels: tuple[str] | None = None,
        k: int = 5,
        nan_label: str = "Missing",
        bins: tuple[float] | None = None,
        **kwargs,
    ) -> None:
        if not all(isinstance(gdf, GeoDataFrame) for gdf in gdfs):
            gdfs, column = self._separate_args(gdfs, column)

        self.column = column
        self.k = k
        self.bins = bins
        self.nan_label = nan_label

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

        self.gdfs: list[GeoDataFrame] = [gdf.copy() for gdf in gdfs]
        self.kwargs = kwargs

        if not self.labels:
            self._set_labels()

        if not self.column:
            for gdf, label in zip(self.gdfs, self.labels, strict=True):
                gdf["label"] = label
            self.column = "label"

        self.gdfs = self._to_common_crs_and_one_geom_col(self.gdfs)
        self._is_categorical = self._check_if_categorical()

        self._fill_missings()

        self.gdf = pd.concat(self.gdfs, ignore_index=True)

        if self.bins:
            self.bins = [bin for bin in self.bins]
            print(self.bins)
            if min(self.gdf[self.column]) < self.bins[0]:
                self.bins = [min(self.gdf[self.column])] + self.bins
            if max(self.gdf[self.column]) > self.bins[-1]:
                self.bins = self.bins + [max(self.gdf[self.column])]
            print(self.bins)

            self.k = len(self.bins) - 1

    def _prepare_bins_and_colors(self, cmap: str):
        if not self.bins:
            self.bins = self._create_bins(self.gdf, self.column)

            if len(self.bins) < self.k:
                self.k = len(self.bins)

        self.colorlist = self._get_continous_colors(cmap, self.k)

    @staticmethod
    def _separate_args(
        args: tuple,
        column: str | None,
    ) -> tuple[tuple[GeoDataFrame], str]:
        """Separate GeoDataFrames from string (column)."""

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

    def _get_labels(self, gdfs: tuple[GeoDataFrame]) -> None:
        """Putting the labels/names in a list before copying the gdfs"""
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
        for i, gdf in enumerate(self.gdfs):
            gdf["label"] = self.labels[i]

    def _to_common_crs_and_one_geom_col(self, gdfs: list[GeoDataFrame]):
        """Need common crs and max one geometry column."""
        crss = list({gdf.crs for gdf in gdfs if gdf.crs is not None})
        self.crs = crss[0]
        new_gdfs = []
        for gdf in gdfs:
            gdf = drop_inactive_geometry_columns(gdf).pipe(rename_geometry_if)
            if crss:
                try:
                    gdf = gdf.to_crs(self.crs)
                except ValueError:
                    gdf = gdf.set_crs(self.crs)
            new_gdfs.append(gdf)
        return new_gdfs

    def _fill_missings(self) -> None:
        for gdf in self.gdfs:
            if self.column in gdf.columns:
                continue
            if self._is_categorical:
                gdf[self.column] = self.nan_label
            else:
                gdf[self.column] = pd.NA

    def _check_if_categorical(self) -> bool:
        """Quite messy this..."""
        if not self.column:
            return True

        maybe_area = 1 if "area" in self.column else 0
        maybe_length = (
            1 if any(x in self.column for x in ["meter", "metre", "leng"]) else 0
        )

        all_nan = 0
        col_not_present = 0
        for gdf in self.gdfs:
            if self.column not in gdf:
                if maybe_area:
                    gdf["area"] = gdf.area
                    maybe_area += 1
                elif maybe_length:
                    gdf["length"] = gdf.length
                    maybe_length += 1
                else:
                    col_not_present += 1
            elif not pd.api.types.is_numeric_dtype(gdf[self.column]):
                if all(gdf[self.column].isna()):
                    all_nan += 1
                return True

        if maybe_area > 1:
            self.column = "area"
            return False
        if maybe_length > 1:
            self.column = "length"
            return False

        if all_nan == len(self.gdfs):
            raise ValueError(f"All values are NaN in column {self.kwargs['column']!r}.")

        if col_not_present == len(self.gdfs):
            raise ValueError(f"{self.kwargs['column']} not found.")

        return False

    def _get_categorical_colors(self) -> None:
        cat_col = self.column
        self._unique_categories = sorted(
            list(self.gdf.loc[self.gdf[cat_col] != self.nan_label, cat_col].unique())
        )
        # custom categorical cmap
        if len(self._unique_categories) <= len(_CATEGORICAL_CMAP):
            self._categories_colors_dict = {
                category: _CATEGORICAL_CMAP[i]
                for i, category in enumerate(self._unique_categories)
            }
        else:
            cmap = matplotlib.colormaps.get_cmap("tab20")

            self._categories_colors_dict = {
                category: colors.to_hex(cmap(int(i)))
                for i, category in enumerate(self._unique_categories)
            }

        if any(self.gdf[self.column].isna()) or any(
            self.gdf[self.column] == self.nan_label
        ):
            self._categories_colors_dict[self.nan_label] = NAN_COLOR

        for gdf in self.gdfs:
            gdf["color"] = gdf[self.column].map(self._categories_colors_dict)

        self.gdf["color"] = self.gdf[self.column].map(self._categories_colors_dict)

    def _create_bins(self, gdf, column):
        n_unique = len(gdf[column].unique())

        if n_unique <= self.k:
            self.k = n_unique - 1  #!!!

        if self.kwargs.get("scheme", "fisherjenks") == "fisherjenks":
            bins = jenks_breaks(gdf.loc[gdf[column].notna(), column], n_classes=self.k)
        else:
            binning = classify(
                np.asarray(gdf.loc[gdf[column].notna(), column]),
                scheme=self.scheme,
                k=self.k,
            )
            bins = binning.bins

        unique_bins = list({round(bin, 5) for bin in bins})
        unique_bins.sort()

        # adding a small amount to get the colors correct. Weird that this is
        # nessecary...
        return [bin + bin / 10_000 for bin in unique_bins]

    @staticmethod
    def _get_continous_colors(
        cmap: str, k: int, start: int = 0, stop: int = 256
    ) -> list[str]:
        cmap = matplotlib.colormaps.get_cmap(cmap)
        return [colors.to_hex(cmap(int(i))) for i in np.linspace(start, stop, num=k)]

    @staticmethod
    def _classify_from_bins(
        bins: list[float], colors_: list[str], array: np.ndarray
    ) -> np.ndarray:
        print(bins)
        if len(bins) == len(colors_) + 1:
            bins = bins[1:]
        print(bins)

        bins = np.array(bins)
        colors_ = np.array(colors_)
        classified = np.searchsorted(bins, array)
        print(colors_)
        print(classified)
        print(array)
        colors_ = colors_[classified]
        return colors_

        #        drop = []
        #       if min(array) < bins[0]:
        #          bins = [min(array)] + bins
        #         drop = drop + [0]
        #    if max(array) > bins[-1]:
        #       bins = bins + [max(array)]
        #      drop = drop + [len(bins)]
        # print(drop)
        bins = np.array(bins)
        colors_ = np.array(colors_)
        print(bins)
        classified = np.searchsorted(bins, array)
        print(classified)

        print(array < bins[0])
        print(array[array < bins[0]])
        print(array[array > bins[-1]])
        classified[array < bins[0]] = -1
        classified[array > bins[-1]] = -1
        print(classified)

        #        classified = [x for x in classified if x not in drop]
        print(max(classified))
        print(len(colors_))

        colors_ = colors_[classified]

        print(colors_)
        return colors_

    @staticmethod
    def _get_continous_color_idx(gdf, column, bins):
        gdf.loc[gdf[column] < bins[0], "color_idx"] = 0
        for i, (prev_bin, this_bin) in enumerate(zip(bins[:-1], bins[1:], strict=True)):
            gdf.loc[
                (gdf[column] >= prev_bin) & (gdf[column] < this_bin), "color_idx"
            ] = i

        return gdf

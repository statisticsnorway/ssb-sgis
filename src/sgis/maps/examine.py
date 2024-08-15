import geopandas as gpd
import numpy as np

from ..geopandas_tools.bounds import get_total_bounds
from ..helpers import unit_is_degrees
from ..raster.image_collection import Band
from ..raster.image_collection import Image
from ..raster.image_collection import ImageCollection
from .explore import Explore
from .map import Map
from .maps import clipmap
from .maps import explore
from .maps import samplemap


class Examine:
    """Explore geometries one row at a time.

    It takes one or more GeoDataFrames and shows an interactive map
    of one area at the time with the 'next' method or random areas with
    the 'sample' method.

    After creating the examiner object, the 'next' method will create a map
    showing all geometries within a given radius (the size parameter) of the
    first geometry in 'mask_gdf' (or the first speficied gdf). The 'next' method
    can then be repeated.

    Examples:
    ---------
    Create the examiner.

    >>> import sgis as sg
    >>> roads = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")
    >>> points = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet")
    >>> e = sg.Examine(points, roads)
    >>> e
    Examine(indices=1000, current=0, n_gdfs=2)

    Then the line below can be repeated for all rows if 'points'. This has to be
    in a separate notebook cell to the previous.

    >>> e.next()
    i == 1 (of 1000)
    <folium.folium.Map object at 0x000002AC73ACC090>

    Previous geometry:

    >>> e.next(-1)
    i == 0 (of 1000)
    <folium.folium.Map object at 0x0000020F3D68BE50>

    Repeating -1 will display the last row of 'points'.

    >>> e.next(-1)
    i == -1 (of 1000)
    <folium.folium.Map object at 0x0000020F3E46FB50>

    Show index 100 and color the map by 'idx':

    >>> e.next(100, column="idx")
    i == 100 (of 1000)
    <folium.folium.Map object at 0x0000020F3DD73F50>

    """

    def __init__(
        self,
        *gdfs: gpd.GeoDataFrame,
        column: str | None = None,
        mask_gdf: gpd.GeoDataFrame | None = None,
        sort_values: str | None = None,
        size: int | float = 1000,
        only_show_mask: bool = True,
        **kwargs,
    ) -> None:
        """Initialiser.

        Args:
            *gdfs: One or more GeoDataFrames. The rows of the first GeoDataFrame
                will be used as masks, unless 'mask_gdf' is specified.
            column: Column to use as colors.
            mask_gdf: Optional GeoDataFrame to use as mask iterator. The geometries
                of mask_gdf will not be shown.
            size: Number of meters (or other crs unit) to buffer the mask geometry
                before clipping.
            sort_values: Optional sorting column(s) of the mask GeoDataFrame. Rows
                will be iterated through from the top.
            only_show_mask: If True (default), show only the mask GeoDataFrame by default.
                The other layers can be toggled on.
            **kwargs: Additional keyword arguments passed to sgis.clipmap.

        """
        if mask_gdf is None:
            self.mask_gdf = gdfs[0]
        else:
            self.mask_gdf = mask_gdf

        gdfs, column, kwargs = Map._separate_args(gdfs, column, kwargs)

        m = Explore(*gdfs, column=column, **kwargs)

        # m = Map(*gdfs, column=column, **kwargs)
        self._gdfs: dict[str, gpd.GeoDataFrame] = dict(
            zip(m.labels, m.gdfs, strict=False)
        )

        self.rasters: dict[str, ImageCollection | Image | Band] = m.rasters

        self.indices = list(range(len(self.mask_gdf)))
        self.i = 0
        self.column = column
        self.size = size
        self.kwargs = {
            key: value for key, value in kwargs.items() if key not in self._gdfs
        }

        if not len(self.mask_gdf):
            return

        if unit_is_degrees(self.mask_gdf) and size > 360:
            raise ValueError(
                "CRS unit is degrees. Use geopandas' "
                "to_crs method to change crs to e.g. UTM. "
                "Or set 'size' to a smaller number."
            )

        if sort_values is not None:
            if (
                sort_values == "area"
                or not isinstance(sort_values, str)
                and "area" in sort_values
            ):
                self.mask_gdf["area"] = self.mask_gdf.area
            self.mask_gdf = self.mask_gdf.sort_values(sort_values)

        if only_show_mask:
            self.kwargs["show"] = [True] + [False] * (len(self._gdfs) - 1)
        elif not kwargs.get("show", True):
            self.kwargs["show"] = [False] * len(self._gdfs)

    def next(self, i: int | None = None, **kwargs) -> None:
        """Displays a map of geometries within the next row of the mask gdf.

        Args:
            i: Index to display.
            **kwargs: Additional keyword arguments passed to sgis.clipmap.

        """
        if kwargs:
            kwargs = self._fix_kwargs(kwargs)
            self.kwargs = self.kwargs | kwargs

        if i and i < 0:
            self.i += i - 1
        elif i:
            self.i = i

        if self.i >= len(self.mask_gdf):
            print("All rows are shown.")
            return

        print(f"i == {self.i} (max. {len(self.mask_gdf)- 1})")
        self.explorer = clipmap(
            self.column,
            *list(self.rasters.values()),
            **self._gdfs,
            mask=self.mask_gdf.iloc[[self.i]].buffer(self.size),
            **self.kwargs,
        )

        self.i += 1

    def sample(self, **kwargs) -> None:
        """Takes a sample index of the mask and displays a map of this area.

        Args:
            **kwargs: Additional keyword arguments passed to sgis.clipmap.
        """
        if kwargs:
            kwargs = self._fix_kwargs(kwargs)
            self.kwargs = self.kwargs | kwargs

        i = np.random.randint(0, len(self.mask_gdf))

        print(f"Showing index {i}")
        self.explorer = clipmap(
            self.column,
            *list(self.rasters.values()),
            **self._gdfs,
            mask=self.mask_gdf.iloc[[i]].buffer(self.size),
            **self.kwargs,
        )

    def current(self, i: int | None = None, **kwargs) -> None:
        """Repeat the last shown map."""
        if kwargs:
            kwargs = self._fix_kwargs(kwargs)
            self.kwargs = self.kwargs | kwargs

        if i and i < 0:
            self.i -= i
        elif i:
            self.i = i

        print(f"{self.i + 1} of {len(self.mask_gdf)}")
        self.explorer = clipmap(
            self.column,
            *list(self.rasters.values()),
            **self._gdfs,
            mask=self.mask_gdf.iloc[[self.i]].buffer(self.size),
            **self.kwargs,
        )

    def explore(self, **kwargs) -> None:
        """Show all rows like the function explore."""
        if kwargs:
            kwargs = self._fix_kwargs(kwargs)
            self.kwargs = self.kwargs | kwargs

        self.explorer = explore(
            *list(self.rasters.values()),
            **self._gdfs,
            column=self.column,
            **self.kwargs,
        )

    def clipmap(self, **kwargs) -> None:
        """Show all rows like the function clipmap."""
        if kwargs:
            kwargs = self._fix_kwargs(kwargs)
            self.kwargs = self.kwargs | kwargs

        self.explorer = clipmap(
            *list(self.rasters.values()),
            **self._gdfs,
            column=self.column,
            **self.kwargs,
        )

    def samplemap(self, **kwargs) -> None:
        """Show all rows like the function samplemap."""
        if kwargs:
            kwargs = self._fix_kwargs(kwargs)
            self.kwargs = self.kwargs | kwargs

        self.explorer = samplemap(
            *list(self.rasters.values()),
            **self._gdfs,
            column=self.column,
            **self.kwargs,
        )

    @property
    def mask(self) -> gpd.GeoDataFrame:
        """Returns a GeoDataFrame of the last shown mask geometry."""
        return self.mask_gdf.iloc[[self.i - 1]]

    @property
    def gdfs(self) -> dict[str, gpd.GeoDataFrame]:
        """Returns all GeoDataFrames in the area of the last shown mask geometry."""
        mask = self.mask_gdf.iloc[[self.i]]
        gdfs = {}
        for label, gdf in self._gdfs.items():
            gdfs[label] = gdf.clip(mask.buffer(self.size))
        return gdfs

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Total bounds of all GeoDataFrames."""
        return get_total_bounds(*list(self.gdfs.values()))

    def _fix_kwargs(self, kwargs: dict) -> dict:
        self.size = kwargs.pop("size", self.size)
        self.column = kwargs.pop("column", self.column)
        return kwargs

    def __repr__(self) -> str:
        """Representation."""
        return f"{self.__class__.__name__}(indices={len(self.indices)}, current={self.i}, n_gdfs={len(self._gdfs)})"

    def __add__(self, scalar: int) -> "Examine":
        """Add a number to the index."""
        self.i += scalar
        return self

    def __sub__(self, scalar: int) -> "Examine":
        """Subtract a number from the index."""
        self.i -= scalar
        return self

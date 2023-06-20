import geopandas as gpd

from ..helpers import unit_is_degrees
from .maps import clipmap


class Examine:
    """Explore geometries one row at a time.

    It takes one or more GeoDataFrames and shows an interactive map
    of one area at the time with the 'next', 'prev' and 'current' methods.

    After creating the examiner object, the 'next' method will create a map
    showing all geometries within a given radius (the size parameter) of the
    first geometry in 'mask_gdf' (or the first speficied gdf). The 'next' method
    can then be repeated.

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
        **kwargs: Additional keyword arguments passed to sgis.clipmap.

    Examples
    --------
    Create the examiner.

    >>> import sgis as sg
    >>> roads = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet")
    >>> points = sg.read_parquet_url("https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet")
    >>> e = sg.Examine(points, roads)
    >>> e

    Then the line below can be repeated for all rows if 'points'. This has to be
    in a separate notebook cell to the previous.

    >>> e.next()

    Previous geometry:

    >>> e.prev()

    Repeating the current area with another layer and new column:

    >>> some_points = points.sample(100)
    >>> e.current(some_points, column="idx")

    The row number can also be specified manually.
    Can be done in 'next', 'prev' and 'current'.

    >>> e.next(i=101)

    This will create an examiner where 'points' is not shown, only used as mask.

    >>>  e = sg.Examine(roads, mask_gdf=points, column="oneway")
    """

    def __init__(
        self,
        *gdfs: gpd.GeoDataFrame,
        column: str | None = None,
        mask_gdf: gpd.GeoDataFrame | None = None,
        sort_values: str | None = None,
        size: int | float = 1000,
        **kwargs,
    ):
        if not all(isinstance(gdf, gpd.GeoDataFrame) for gdf in gdfs):
            raise ValueError("gdfs must be of type GeoDataFrame.")

        self.gdfs = gdfs
        if mask_gdf is None:
            self.mask_gdf = gdfs[0]
        else:
            self.mask_gdf = mask_gdf

        if unit_is_degrees(self.mask_gdf) and size > 360:
            raise ValueError(
                "CRS unit is degrees. Use geopandas' "
                "to_crs method to change crs to e.g. UTM. "
                "Or set 'size' to a smaller number."
            )

        if sort_values is not None:
            self.mask_gdf = self.mask_gdf.sort_values(sort_values)

        self.indices = list(range(len(gdfs[0])))
        self.i = 0
        self.column = column
        self.size = size
        self.kwargs = kwargs

    def next(self, *gdfs, i: int | None = None, **kwargs):
        """Displays a map of geometries within the next row of the mask gdf.

        Args:
            *gdfs: Optional GeoDataFrames to be added on top of the current.
            i: Optionally set the integer index of which row to use as mask.
            **kwargs: Additional keyword arguments passed to sgis.clipmap.
        """
        gdfs = () if not gdfs else gdfs
        self.gdfs = self.gdfs + gdfs
        if kwargs:
            kwargs = self._fix_kwargs(kwargs)
            self.kwargs = self.kwargs | kwargs

        if i:
            self.i = i

        if self.i >= len(self.mask_gdf):
            print("All rows are shown.")
            return

        print(f"{self.i + 1} of {len(self.mask_gdf)}")
        clipmap(
            *self.gdfs,
            self.column,
            mask=self.mask_gdf.iloc[[self.i]].buffer(self.size),
            **self.kwargs,
        )
        self.i += 1

    def prev(self, *gdfs, i: int | None = None, **kwargs):
        """Displays a map of geometries within the previus row of the mask gdf.

        Args:
            *gdfs: Optional GeoDataFrames to be added on top of the current.
            i: Optionally set the integer index of which row to use as mask.
            **kwargs: Additional keyword arguments passed to sgis.clipmap.
        """
        gdfs = () if not gdfs else gdfs
        self.gdfs = self.gdfs + gdfs
        if kwargs:
            kwargs = self._fix_kwargs(kwargs)
            self.kwargs = self.kwargs | kwargs

        self.i -= 2

        if i:
            self.i = i

        print(f"{self.i + 1} of {len(self.mask_gdf)}")
        clipmap(
            *self.gdfs,
            self.column,
            mask=self.mask_gdf.iloc[[self.i]].buffer(self.size),
            **self.kwargs,
        )

    def current(self, *gdfs, i: int | None = None, **kwargs):
        """Repeat the last shown map."""
        gdfs = () if not gdfs else gdfs
        self.gdfs = self.gdfs + gdfs
        if kwargs:
            kwargs = self._fix_kwargs(kwargs)
            self.kwargs = self.kwargs | kwargs

        if i:
            self.i = i

        print(f"{self.i + 1} of {len(self.mask_gdf)}")
        clipmap(
            *self.gdfs,
            self.column,
            mask=self.mask_gdf.iloc[[self.i]].buffer(self.size),
            **self.kwargs,
        )

    def get_current_mask(self) -> gpd.GeoDataFrame:
        """Returns a GeoDataFrame of the last shown mask geometry."""
        return self.mask_gdf.iloc[[self.i]]

    def get_current_geoms(self) -> tuple[gpd.GeoDataFrame]:
        """Returns all GeoDataFrames in the area of the last shown mask geometry."""
        mask = self.mask_gdf.iloc[[self.i]]
        gdfs = ()
        for gdf in self.gdfs:
            gdfs = gdfs + (gdf.clip(mask.buffer(self.size)),)
        return gdfs

    def _fix_kwargs(self, kwargs) -> dict:
        self.size = kwargs.pop("size", self.size)
        self.column = kwargs.pop("column", self.column)
        return kwargs

    def __repr__(self) -> str:
        return f"{self.__class__}(indices={len(self.indices)}, current={self.i}, n_gdfs={len(self.gdfs)})"
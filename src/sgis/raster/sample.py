import pandas as pd
import shapely
from geopandas import GeoDataFrame, GeoSeries
from shapely import Geometry


class RandomCubeSample:
    def __init__(self, cube, n=1, buffer=1000, mask=None, **kwargs):
        self.cubes = []

        if mask is not None:
            points = (
                GeoSeries(cube.unary_union, crs=cube.crs)
                .clip(mask, keep_geom_type=False)
                .sample_points(n)
                .explode(ignore_index=True)
            )
        else:
            points = (
                GeoSeries(cube.unary_union, crs=cube.crs)
                .sample_points(n)
                .explode(ignore_index=True)
            )
        buffered = points.buffer(buffer)

        boxes = [shapely.box(*arr) for arr in buffered.bounds.values]

        for box in boxes:
            clipped = cube.clip(box, **kwargs)
            self.cubes.append(clipped)

    def to_cube(self):
        cube = self.cubes.__class__()
        cube._crs = self._crs
        cube.df = pd.concat([cube.df for cube in self.cubes])
        return cube

    @property
    def index(self):
        pass

    @property
    def arrays(self):
        pass

    def __iter__(self):
        return iter(self.cubes)

    def __len__(self):
        return len(self.cubes)

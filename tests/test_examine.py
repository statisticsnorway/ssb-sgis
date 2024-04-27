# %%
from pathlib import Path

src = str(Path(__file__).parent).strip("tests") + "src"

import sys

sys.path.insert(0, src)

import sgis as sg

# %%
if __name__ == "__main__":
    from oslo import points_oslo
    from oslo import roads_oslo

    points = points_oslo()
    roads = roads_oslo()
# %%
if __name__ == "__main__":
    e = sg.Examine(points, roads)
    print(e)

# %%
if __name__ == "__main__":
    e.next()
# %%
if __name__ == "__main__":
    e.prev()

# %%
if __name__ == "__main__":
    e.current()

# %%
if __name__ == "__main__":
    some_points = points.sample(100)
    e.next(some_points, column="idx")

# %%
if __name__ == "__main__":
    e.next(cmap="plasma")

# %%
if __name__ == "__main__":
    e = sg.Examine(roads, mask_gdf=points, column="oneway")

# %%
if __name__ == "__main__":
    e.next()

# %%
if __name__ == "__main__":
    print(e.get_current_mask())

# %%
if __name__ == "__main__":
    print(e.get_current_geoms())

# %%

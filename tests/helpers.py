import cProfile
import io
import sys
from pathlib import Path

import pandas as pd

src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


def cprofile_df(call: str) -> pd.DataFrame:
    old_stdout = sys.stdout  # Memorize the default stdout stream
    sys.stdout = buffer = io.StringIO()

    cProfile.run(call, sort="cumtime")

    sys.stdout = old_stdout  # Put the old stream back in place

    what_was_printed = (
        buffer.getvalue()
    )  # Return a str containing the entire contents of the buffer.
    print(what_was_printed)

    lines = what_was_printed.split("Ordered by: cumulative time")[-1].split("\n")
    lines = [line.strip() for line in lines if line]
    lines = [line.split(" ") for line in lines]

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass

    cols = ["ncalls", "tottime", "percall", "cumtime", "percall_cumtime", "name"]

    out = []
    for line in lines[1:]:
        line = [x.split("/")[-1] for x in line if x.split("/")[-1]]
        numbers = [float(x) for x in line if is_number(x)]
        not_number = "".join([x for x in line if not is_number(x)])
        if len(numbers) + 1 == len(cols):
            out_line = numbers + [not_number]
            out.append(out_line)
            continue
        raise ValueError(line)

    df = pd.DataFrame(out, columns=cols)
    df = df[["ncalls", "tottime", "cumtime", "percall", "name"]]
    df["percall"] = df["cumtime"] / df["ncalls"]
    return df


def create_all_geometry_types():
    from shapely.geometry import LinearRing
    from shapely.geometry import LineString

    point = sg.to_gdf([(0, 0)])
    multipoint = sg.to_gdf([(10, 10), (11, 11)]).dissolve()
    line = sg.to_gdf(LineString([(20, 20), (21, 21)]))
    multiline = sg.to_gdf(
        [LineString([(30, 30), (31, 31)]), LineString([(32, 32), (33, 33)])]
    ).dissolve()
    polygon = sg.buff(sg.to_gdf([(40, 40)]), 0.25)
    multipolygon = sg.to_gdf([(50, 50), (51, 51)]).dissolve().pipe(sg.buff, 0.25)
    ring = sg.to_gdf(LinearRing([(60, 60), (60, 61), (61, 61), (61, 60), (60, 60)]))
    gdf = pd.concat([point, multipoint, ring, line, multiline, polygon, multipolygon])
    collection = gdf.dissolve()
    return pd.concat([gdf, collection], ignore_index=True)

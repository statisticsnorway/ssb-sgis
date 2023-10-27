from pathlib import Path

import dapla as dp
import numpy as np
import pandas as pd

import sgis as sg


class DaplaPath(str):
    """Path that works like a string, with methods like exists and ls for Dapla."""

    def __init__(self, path: str | Path):
        try:
            self = str(path)
        except Exception as e:
            raise TypeError from e

    def exists(self) -> bool:
        return dp.FileClient.get_gcs_file_system().exists(self)

    def get_highest_versions(self, include_versionless: bool = True) -> pd.Series:
        """Strips all version numbers off the file paths in the folder and keeps only the highest.

        Does a regex search for the pattern '_v' plus any integer.

        """
        ser = self.ls(time_as_index=True).sort_values()
        return self._drop_version_number_and_keep_last(ser, include_versionless)

    def get_newest_versions(self, include_versionless: bool = True) -> pd.Series:
        """Strips all version numbers off the file paths in the folder and keeps only the newest.

        Does a regex search for the pattern '_v' plus any integer.

        """
        ser = self.ls(time_as_index=True)
        return self._drop_version_number_and_keep_last(ser, include_versionless)

    def ls_contains(self, contains: str, time_as_index: bool = True) -> pd.Series:
        """Returns a list of files containing the given string."""
        ser = self.ls(time_as_index=time_as_index)
        return ser.loc[lambda x: x.str.contains(contains)]

    def ls_within(self, within: int) -> pd.Series:
        """Returns a list of files with a timestamp within the given amount of minutes."""
        ser = self.ls(time_as_index=True)

        time_now = pd.Timestamp.now() - pd.Timedelta(minutes=within)

        return ser.loc[lambda x: x.index > time_now]

    def ls(self, time_as_index: bool = True) -> pd.Series:
        """Returns a list of files in the directory."""
        info: list[dict] = dp.FileClient.get_gcs_file_system().ls(self, detail=True)

        index_col = "updated" if time_as_index else "size"

        fileinfo = np.array(
            [
                (x[index_col], x["name"])
                for x in info
                if x["storageClass"] != "DIRECTORY"
            ]
        )

        ser = pd.Series(fileinfo[:, 1], index=fileinfo[:, 0], name="path").loc[
            lambda x: ~x.str.endswith("/")
        ]

        if time_as_index:
            ser.index = (
                pd.to_datetime(ser.index)
                .round("s")
                .tz_convert("Europe/Oslo")
                .tz_localize(None)
                .round("s")
            )
        else:
            ser.index = ser.index.astype(int)

        return ser.sort_index()

    def check_files(
        self, contains: str | None = None, within_minutes: int | None = None
    ) -> pd.DataFrame:
        return sg.check_files(self, contains, within_minutes)

    @staticmethod
    def _drop_version_number_and_keep_last(ser, include_versionless: bool) -> pd.Series:
        stems = ser.reset_index(drop=True).apply(lambda x: Path(x).stem)

        version_pattern = r"_v\d+"

        if not include_versionless:
            stems = stems.loc[stems.str.contains(version_pattern)]

        without_version_number = stems.str.replace(version_pattern, "", regex=True)

        only_newest = without_version_number.loc[lambda x: ~x.duplicated(keep="last")]

        return ser.iloc[only_newest.index]


class Kartbucket:
    def __init__(self, year: int):
        self.year = year

    @property
    def delt(self) -> DaplaPath:
        return DaplaPath("ssb-prod-kart-data-delt")

    @property
    def analyse(self) -> DaplaPath:
        return DaplaPath(f"{self.delt}/kartdata_analyse/klargjorte-data/{self.year}")

    @property
    def visualisering(self) -> DaplaPath:
        return DaplaPath(
            f"{self.delt}/kartdata_visualisering/klargjorte-data/{self.year}"
        )

    @property
    def rutenett(self) -> DaplaPath:
        return DaplaPath(f"{self.delt}/kartdata_rutenett/klargjorte-data/{self.year}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.delt})"


class PathSeries:
    def __init__(self, data: list[str], index=None):
        try:
            self.paths = pd.Series(data, index=index)
        except Exception as e:
            raise TypeError from e

        folders = self.paths.iloc[0].split("/")
        base = []
        for folder in folders:
            if not self.paths.str.contains(folder).all():
                continue
            base.append(folder)

        self.base = "/".join(base).strip("/")

        self.paths.name = self.base

    @property
    def loc(self):
        return self.paths.loc

    def query(self, *args, **kwargs):
        return self.paths.query(*args, **kwargs)

    def contains(self, text: str):
        return self.paths.loc[lambda x: x.str.contains(text)]

    def sort(self, text: str):
        return self.paths.loc[lambda x: x.str.contains(text)]

    def __iter__(self):
        return iter(self.paths)

    def __repr__(self):
        return self.paths.str.replace(self.base, "{self.base}").__repr__()

    def __str__(self):
        return self.paths.str.replace(self.base, "{self.base}").__str__()

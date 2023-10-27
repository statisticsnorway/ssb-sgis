from pathlib import Path

import dapla as dp
import numpy as np
import pandas as pd

import sgis as sg


class DaplaPath:
    def __init__(self, path: str | Path):
        try:
            self.path = Path(path)
        except Exception as e:
            raise TypeError from e

    def exists(self) -> bool:
        return dp.FileClient.get_gcs_file_system().exists(self.path)

    def get_highest_versions(self, include_versionless: bool = True) -> pd.Series:
        ser = self.ls(time_as_index=True).sort_values()
        return self._drop_version_number_and_keep_last(ser, include_versionless)

    def get_newest_versions(self, include_versionless: bool = True) -> pd.Series:
        ser = self.ls(time_as_index=True)
        return self._drop_version_number_and_keep_last(ser, include_versionless)

    @staticmethod
    def _drop_version_number_and_keep_last(ser, include_versionless) -> pd.Series:
        stems = ser.reset_index(drop=True).apply(lambda x: Path(x).stem)

        version_pattern = r"_v\d+"

        if not include_versionless:
            stems = stems.loc[stems.str.contains(version_pattern)]

        without_version_number = stems.str.replace(version_pattern, "", regex=True)

        only_newest = without_version_number.loc[lambda x: ~x.duplicated(keep="last")]

        return ser.iloc[only_newest.index]

    def ls_contains(self, contains: str, time_as_index: bool = True) -> pd.Series:
        ser = self.ls(time_as_index=time_as_index)
        return ser.loc[lambda x: x.str.contains(contains)]

    def ls_within(self, within: int) -> pd.Series:
        ser = self.ls(time_as_index=True)

        time_now = pd.Timestamp.now() - pd.Timedelta(minutes=within)

        return ser.loc[lambda x: x.index > time_now]

    def ls(self, time_as_index: bool = True) -> pd.Series:
        info: list[dict] = dp.FileClient.get_gcs_file_system().ls(
            self.path, detail=True
        )

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

        return ser.sort_index()

    def check_files(
        self, contains: str | None = None, within_minutes: int | None = None
    ) -> pd.DataFrame:
        return sg.check_files(self.path, contains, within_minutes)

    def __repr__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)

    def __iter__(self):
        return iter(str(self.path))


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

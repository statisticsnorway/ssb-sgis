try:
    from gcsfs import GCSFileSystem

    file_system = GCSFileSystem
except ImportError:
    import datetime
    import glob
    import io
    import os
    import pathlib
    from concurrent.futures import ThreadPoolExecutor
    from typing import Any

    class LocalFileSystem:
        """Mimicks GCS's FileSystem but using standard library (os, glob)."""

        @staticmethod
        def glob(
            path: str,
            detail: bool = False,
            recursive: bool = True,
            include_hidden: bool = True,
            **kwargs,
        ) -> list[dict] | list[str]:
            """Like GCSFileSystem.glob."""
            relevant_paths = glob.iglob(
                path, recursive=recursive, include_hidden=include_hidden, **kwargs
            )

            if not detail:
                return list(relevant_paths)
            with ThreadPoolExecutor() as executor:
                return list(executor.map(get_file_info, relevant_paths))

        @classmethod
        def ls(
            cls, path: str, detail: bool = False, **kwargs
        ) -> list[dict] | list[str]:
            """Like GCSFileSystem.ls."""
            return cls().glob(
                str(pathlib.Path(path) / "**"), detail=detail, recursive=False, **kwargs
            )

        @staticmethod
        def info(path) -> dict[str, Any]:
            """Like GCSFileSystem.info."""
            return get_file_info(path)

        @staticmethod
        def open(path: str, *args, **kwargs) -> io.TextIOWrapper:
            """Built in open."""
            return open(path, *args, **kwargs)

        @staticmethod
        def exists(path: str) -> bool:
            """os.path.exists."""
            return os.path.exists(path)

    def get_file_info(path) -> dict[str, str | float]:
        return {
            "updated": datetime.datetime.fromtimestamp(os.path.getmtime(path)),
            "size": os.path.getsize(path),
            "name": path,
            "type": "directory" if os.path.isdir(path) else "file",
        }

    file_system = LocalFileSystem

from .geopandas_tools.runners import RTreeQueryRunner
from .geopandas_tools.runners import OverlayRunner
from .geopandas_tools.runners import UnionRunner


class Config:
    def __init__(self, data: dict):
        self.data = data

    def get_instance(self, key: str, *args, **kwargs):
        x = self.data[key]
        if callable(x):
            return x(*args, **kwargs)
        return x

    def __getattr__(self, attr: str):
        return getattr(self.data, attr)

    def __setitem__(self, key: str, value):
        self.data[key] = value

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __str__(self) -> str:
        return str(self.data)


config = Config(
    {
        "n_jobs": 1,
        "file_system": file_system,
        "rtree_runner": RTreeQueryRunner(1),
        "overlay_runner": OverlayRunner,
        "union_runner": UnionRunner,
    }
)

from typing import Any

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

from .geopandas_tools.runners import OverlayRunner
from .geopandas_tools.runners import RTreeQueryRunner
from .geopandas_tools.runners import UnionRunner


def _get_instance(data: dict, key: str, **kwargs) -> Any:
    """Get the dict value and call it if callable."""
    x = data[key]
    if callable(x):
        return x(**kwargs)
    return x


config = {
    "n_jobs": 1,
    "file_system": file_system,
    "rtree_runner": RTreeQueryRunner,
    "overlay_runner": OverlayRunner,
    "union_runner": UnionRunner,
}

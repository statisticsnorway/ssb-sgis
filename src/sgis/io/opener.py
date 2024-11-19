from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

try:
    from dapla import FileClient
    from dapla.gcs import GCSFileSystem
except ImportError:

    class GCSFileSystem:  # type: ignore[no-redef]
        """Placeholder."""


from ._is_dapla import is_dapla


@contextmanager
def opener(
    path, mode: str = "rb", file_system: GCSFileSystem | None = None
) -> Generator[str | Any, None, None]:
    """Yields a gcs buffer if in Dapla, otherwise yields the path.

    Example:
    -------
    >>> with opener(path) as file:
    >>>     with rasterio.open(file) as src:
    >>>         array = src.read()
    """
    if is_dapla():
        if file_system is None:
            file_system = FileClient.get_gcs_file_system()
        yield file_system.open(str(path), mode=mode)
    else:
        yield str(path)

from contextlib import contextmanager


try:
    import dapla as dp
except ImportError:
    pass

from ._is_dapla import is_dapla


@contextmanager
def opener(path, mode="rb"):
    """Yields a gcs buffer if in Dapla, otherwise yields the path.

    Example
    -------
    >>> with opener(path) as file:
    >>>     with rasterio.open(file) as src:
    >>>         array = src.read()
    """
    if is_dapla():
        fs = dp.FileClient.get_gcs_file_system()
        yield fs.open(str(path), mode=mode)
    else:
        yield str(path)

try:
    from gcsfs import GCSFileSystem
except ImportError:

    class GCSFileSystem:
        """Placeholder."""

        raise ImportError


config = {
    "n_jobs": 1,
    "file_system": GCSFileSystem,
}

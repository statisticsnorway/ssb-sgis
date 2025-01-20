try:
    from gcsfs import GCSFileSystem
except ImportError:

    class GCSFileSystem:
        """Placeholder."""

        def __init__(self, *args, **kwargs) -> None:
            """Placeholder."""
            raise ImportError("gcsfs")


config = {
    "n_jobs": 1,
    "file_system": GCSFileSystem,
}

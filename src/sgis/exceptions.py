"""Some small exception classes."""


class NoPointsWithinSearchToleranceError(ValueError):
    """Exception for when the points are too far away from the network."""

    def __init__(
        self, what: str | None = None, search_tolerance: str | None = None
    ) -> None:
        """Initialise error class."""
        self.what = what
        self.search_tolerance = search_tolerance

    def __str__(self) -> str:
        """String representation."""
        return (
            f"No {self.what} within specified 'search_tolerance' "
            f"of {self.search_tolerance}"
        )


class ZeroLinesError(ValueError):
    """DataFrame has 0 rows."""

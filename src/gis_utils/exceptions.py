

class NoPointsWithinSearchTolerance(Exception):
    def __init__(self, what: str | None = None, search_tolerance: str | None = None) -> None:
        self.what = what
        self.search_tolerance = search_tolerance
    def __str__(self):
        return f"No {self.what}points within specified 'search_tolerance' of {self.search_tolerance}"


class ZeroRowsError(Exception):
    "DataFrame has 0 rows."


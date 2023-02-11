

class NoPointsWithinSearchTolerance(Exception):
    def __init__(self, what: str | None = None, search_tolerance: str | None = None) -> None:
        f"No {what}startpoints within specified 'search_tolerance' of {search_tolerance}"


class ZeroRowsError(Exception):
    "The roads have 0 rows."

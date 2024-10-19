from typing import Any


class _NoExplore:
    """Simply so signal that explore functions should be immediately exited."""


_DEBUG_CONFIG = {
    # "center": (5.3719398, 59.00999914, 0.01),
    # "center": (5.27306727, 59.44232754, 200),
    # "center": (5.85575588, 62.33991158, 200),
    # "center": (12.11270809, 66.55499008, 10),
    # "center": (26.02870514, 70.68108478, 200),
    "center": _NoExplore(),
    "print": False,
}


def _try_debug_print(*args: Any) -> None:
    if not _DEBUG_CONFIG["print"]:
        return
    try:
        print(*args)
    except Exception:
        pass

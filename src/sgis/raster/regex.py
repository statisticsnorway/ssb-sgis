import itertools
import re
from collections.abc import Sequence

import pandas as pd

from ..io._is_dapla import is_dapla

try:
    import dapla as dp
except ImportError:
    pass

try:
    from gcsfs.core import GCSFile
except ImportError:

    class GCSFile:
        """Placeholder."""


if is_dapla():

    def _open_func(*args, **kwargs) -> GCSFile:
        return dp.FileClient.get_gcs_file_system().open(*args, **kwargs)

else:
    _open_func = open


class _RegexError(ValueError):
    pass


def _any_regex_matches(xml_file: str, regexes: tuple[str]) -> bool | None:
    n_matches = 0
    for regex in regexes:
        try:
            if bool(re.search(regex, xml_file)):
                return True
            n_matches += 1
        except (TypeError, AttributeError):
            continue

    if not n_matches:
        return None
    else:
        return False


def _get_regex_match_from_xml_in_local_dir(
    paths: list[str], regexes: str | tuple[str]
) -> str | dict[str, str]:
    for i, path in enumerate(paths):
        if ".xml" not in path:
            continue
        with _open_func(path, "rb") as file:
            filebytes: bytes = file.read()
            try:
                return _extract_regex_match_from_string(
                    filebytes.decode("utf-8"), regexes
                )
            except _RegexError as e:
                if i == len(paths) - 1:
                    raise e


def _extract_regex_match_from_string(
    xml_file: str, regexes: tuple[str | re.Pattern]
) -> str | dict[str, str]:
    if all(isinstance(x, str) for x in regexes):
        for regex in regexes:
            try:
                return re.search(regex, xml_file).group(1)
            except (TypeError, AttributeError, IndexError):
                continue
        raise _RegexError(regexes)

    out = {}
    for regex in regexes:
        try:
            matches = re.search(regex, xml_file)
            out |= matches.groupdict()
        except (TypeError, AttributeError):
            continue
    if not out:
        raise _RegexError(regexes)
    return out


def _get_regexes_matches_for_df(
    df, match_col: str, patterns: Sequence[re.Pattern]
) -> pd.DataFrame:
    if not len(df):
        return df

    non_optional_groups = list(
        set(
            itertools.chain.from_iterable(
                [_get_non_optional_groups(pat) for pat in patterns]
            )
        )
    )

    if not non_optional_groups:
        return df

    assert df.index.is_unique
    keep = []
    for pat in patterns:
        for i, row in df[match_col].items():
            matches = _get_first_group_match(pat, row)
            if all(group in matches for group in non_optional_groups):
                keep.append(i)

    return df.loc[keep]


def _get_non_optional_groups(pat: re.Pattern | str) -> list[str]:
    return [
        x
        for x in [
            _extract_group_name(group)
            for group in pat.pattern.split("\n")
            if group
            and not group.replace(" ", "").startswith("#")
            and not group.replace(" ", "").split("#")[0].endswith("?")
        ]
        if x is not None
    ]


def _extract_group_name(txt: str) -> str | None:
    try:
        return re.search(r"\(\?P<(\w+)>", txt)[1]
    except TypeError:
        return None


def _get_first_group_match(pat: re.Pattern, text: str) -> dict[str, str]:
    groups = pat.groupindex.keys()
    all_matches: dict[str, str] = {}
    for x in pat.findall(text):
        if not x:
            continue
        if isinstance(x, str):
            x = [x]
        for group, value in zip(groups, x, strict=True):
            if value and group not in all_matches:
                all_matches[group] = value
    return all_matches

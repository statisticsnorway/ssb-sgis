SENTINEL2_FILENAME_REGEX = r"""
    ^(?P<tile>T\d{2}[A-Z]{3})
    _(?P<date>\d{8})T\d{6}
    _(?P<band>B[018][\dA]|SCL)
    (?:_(?P<resolution>\d+)m)?
    .*
    \..*$
"""


SENTINEL2_IMAGE_REGEX = r"""
    ^(?P<mission_id>S2[AB])
    _MSI(?P<level>[A-Z]\d{1}[A-Z])
    _(?P<date>\d{8})T\d{6}
    _(?P<baseline>N\d{4})
    _(?P<orbit>R\d{3})
    _(?P<tile>T\d{2}[A-Z]{3})
    _\d{8}T\d{6}
    .*
    .*$
"""

SENTINEL2_MOSAIC_FILENAME_REGEX = r"""
    ^SENTINEL2X_
    (?P<date>\d{8})
    .*?
    (?P<tile>T\d{2}[A-Z]{3})
    .*?
    _(?P<band>B\d{1,2}A?)_
    .*?
    (?:_(?P<resolution>\d+m))?
    .*?\.tif$
"""

SENTINEL2_MOSAIC_IMAGE_REGEX = r"""
    ^SENTINEL2X_
    (?P<date>\d{8})
    -\d{6}
    -\d{3}
    _(?P<level>[A-Z]\d{1}[A-Z])
    .*(?P<tile>T\d{2}[A-Z]{3})
    .*.*$
"""


SENTINEL2_SCL_CLASSES = {
    0: "No Data (Missing data)",  # 000000
    1: "Saturated or defective pixel",  # ff0000
    2: "Topographic casted shadows",  # 2f2f2f
    3: "Cloud shadows",  # 643200
    4: "Vegetation",  # 00a000
    5: "Not-vegetated",  # ffe65a
    6: "Water",  # 0000ff
    7: "Unclassified",  # 808080
    8: "Cloud medium probability",  # c0c0c0
    9: "Cloud high probability",  # ffffff
    10: "Thin cirrus",  # 64c8ff
    11: "Snow or ice",  # ff96ff
}

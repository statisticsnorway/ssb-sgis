import re

SENTINEL2_FILENAME_REGEX = r"""
    ^(?P<tile>T\d{2}[A-Z]{3})
    _(?P<date>\d{8})T\d{6}
    _(?P<band>B[018][\dA])
    (?:_(?P<resolution>\d+)m)?
    .*
    \..*$
"""


SENTINEL2_CLOUD_FILENAME_REGEX = r"""
    ^(?P<tile>T\d{2}[A-Z]{3})
    _(?P<date>\d{8})T\d{6}
    _(?P<band>SCL)
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


# multiple regex searches because there are different xml files with same info, but different naming
CLOUD_COVERAGE_REGEXES: tuple[str] = (
    r"<Cloud_Coverage_Assessment>([\d.]+)</Cloud_Coverage_Assessment>",
    r"<CLOUDY_PIXEL_OVER_LAND_PERCENTAGE>([\d.]+)</CLOUDY_PIXEL_OVER_LAND_PERCENTAGE>",
)

BOA_QUANTIFICATION_VALUE_REGEXES: tuple[str] = (
    r'<BOA_QUANTIFICATION_VALUE unit="none">(\d+)</BOA_QUANTIFICATION_VALUE>',
)

PROCESSING_BASELINE_REGEXES: tuple[str] = (
    r"<PROCESSING_BASELINE>(.*?)</PROCESSING_BASELINE>",
)


CRS_REGEXES: tuple[str] = (r"<HORIZONTAL_CS_CODE>EPSG:(\d+)</HORIZONTAL_CS_CODE>",)

BOUNDS_REGEXES: tuple[dict[str, str]] = (
    {"minx": r"<ULX>(\d+)</ULX>", "maxy": r"<ULY>(\d+)</ULY>"},
)
BOUNDS_REGEXES: tuple[re.Pattern] = (
    re.compile(r"<ULX>(?P<minx>\d+)</ULX>"),
    re.compile(r"<ULY>(?P<maxy>\d+)</ULY>"),
    # )
    # SHAPE_PATTERNS: tuple[re.Pattern] = (
    re.compile(
        r'<Size resolution="(?P<resolution>\d+)">\s*<NROWS>(?P<nrows>\d+)</NROWS>\s*<NCOLS>(?P<ncols>\d+)</NCOLS>\s*</Size>'
    ),
    re.compile(
        r"<Cloud_Coverage_Assessment>(?P<cloud_coverage_percentage>[\d.]+)</Cloud_Coverage_Assessment>"
    ),
    re.compile(
        r"<CLOUDY_PIXEL_OVER_LAND_PERCENTAGE>(?P<cloud_coverage_percentage>[\d.]+)</CLOUDY_PIXEL_OVER_LAND_PERCENTAGE>"
    ),
)

SENTINEL2_L2A_BANDS = {
    "B01": 60,
    "B02": 10,
    "B03": 10,
    "B04": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B08": 10,
    "B8A": 20,
    "B09": 60,
    "B11": 20,
    "B12": 20,
}
SENTINEL2_L1C_BANDS = SENTINEL2_L2A_BANDS | {"B10": 60}
SENTINEL2_CLOUD_BANDS = {
    "SCL": 20,  # SCL: scene classification
}

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

SENTINEL2_BANDS = SENTINEL2_L1C_BANDS | SENTINEL2_CLOUD_BANDS
SENTINEL2_RBG_BANDS = ["B02", "B03", "B04"]
SENTINEL2_NDVI_BANDS = ["B04", "B08"]

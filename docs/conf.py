"""Sphinx configuration."""
project = "ssb-gis-utils"
author = "Statistics Norway"
copyright = "2023, Statistics Norway"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"

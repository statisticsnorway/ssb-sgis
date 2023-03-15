# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import date


sys.path.insert(0, os.path.abspath("../src"))


# -- Project information -----------------------------------------------------

project = "ssb-sgis"
copyright = f"{date.today().year}, Statistics Norway"
author = "Statistics Norway"

# The full version, including alpha/beta/rc tags
# release = "0.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "monokai"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
# html_theme = "sphinx_book_theme"
# html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = []


# -- Other configuration ---------------------------------------------------

autodoc_typehints = "description"
autodoc_mock_imports = ["dapla"]

# html_logo = "_static/python-logo-generic.svg"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
    "css/pandas.css",
]
"""
html_theme_options = {
    "rightsidebar": True,
    "header_inverse": True,
    "relbar_inverse": True,
    "noresponsive": False,
}

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "red",
        "color-brand-content": "#CC3333",
        "color-admonition-background": "orange",
    },
}
"""

# Show typehints as content of the function or method
autodoc_typehints = "description"

numpydoc_show_class_members = False

source_suffix = [".rst", ".md"]
master_doc = "reference/index"

# List of source files, relative to this directory.
source_files = [
    "index.md",
    "reference/index.rst",  # make sure this matches the name of the referenced document
]

add_module_names = False

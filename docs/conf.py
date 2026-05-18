import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))

project = "AstraFlow"
author = "AstraFlow Team"
copyright = f"2025-{datetime.now().year}, {author}"
version = "0.1.0"
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx_tabs.tabs",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

language = os.environ.get("PROJECT_DOC_LANG", "en")
exclude_patterns = ["_build", "build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.jpg"
html_theme_options = {
    "logo": {
        "image_light": "_static/logo.jpg",
        "image_dark": "_static/logo.jpg",
    },
    "max_navbar_depth": 1,
}
html_css_files = ["css/custom.css"]

# Notebook policy: render notebooks but do not execute during docs build.
nbsphinx_allow_errors = True
nbsphinx_execute = "never"

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "html_image",
    "substitution",
]
myst_heading_anchors = 3

# Prefix auto-generated section labels with document path to avoid duplicates.
autosectionlabel_prefix_document = True

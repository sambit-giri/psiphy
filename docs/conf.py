import sys
import os

sys.path.insert(0, os.path.abspath("../src"))

project = "psiphy"
author = "Sambit Giri"
copyright = "2024, Sambit Giri"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "numpydoc",
    "nbsphinx",
]

html_theme = "sphinx_rtd_theme"
autodoc_member_order = "bysource"
numpydoc_show_class_members = False
nbsphinx_execute = "never"   # notebooks are pre-rendered; don't re-execute during build

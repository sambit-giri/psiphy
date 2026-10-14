import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../src"))

# The tutorials live in ../examples. Copy them next to this file at build time:
# nbsphinx resolves image paths relative to the notebook's real location, so a
# symlink (or a build directory inside docs/) leaves every figure pointing to a
# file that is never published.
_here = Path(__file__).resolve().parent
_nb_src, _nb_dst = _here.parent / "examples", _here / "examples"
if _nb_dst.is_symlink():
    _nb_dst.unlink()
_nb_dst.mkdir(exist_ok=True)
for _nb in _nb_src.glob("*.ipynb"):
    shutil.copy2(_nb, _nb_dst / _nb.name)

project = "psiphy"
author = "Sambit Giri"
copyright = "2024-2026, Sambit Giri"

try:
    from importlib.metadata import version as _version
    release = _version("psiphy")
except Exception:
    release = ""
version = ".".join(release.split(".")[:2])

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
nbsphinx_allow_errors = True  # don't fail on error cells in pre-executed notebooks

exclude_patterns = ["_build", "examples/line_fitting.ipynb"]

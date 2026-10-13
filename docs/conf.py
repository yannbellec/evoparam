# SPDX-License-Identifier: Apache-2.0
"""Sphinx configuration of the evoparam documentation (spec §9.7; completed at step S20).

``sphinx-build -b html docs docs/_build/html`` builds the site and runs the tutorials of
``docs/tutorials`` (sphinx-gallery); ``-D sphinx_gallery_conf.plot_gallery=0`` skips running
them.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import evoparam  # noqa: E402  (after the sys.path insert above)

project = "evoparam"
author = "Yann V. Bellec"
copyright = "Yann V. Bellec"  # noqa: A001  (the Sphinx configuration name)
version = release = evoparam.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
]
templates_path = []
exclude_patterns = ["_build", "spec", "tutorials/README.txt", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
html_theme = "pydata_sphinx_theme"
# version switcher (REQ-302): ``latest`` is main, ``stable`` the last tag. The JSON file is
# served with the published site, which waits for the owner's HIIE decision (REQ-398).
html_theme_options = {
    "switcher": {
        "json_url": "_static/switcher.json",
        "version_match": "latest" if "dev" in release else "stable",
    },
    "navbar_end": ["version-switcher", "navbar-icon-links"],
    "check_switcher": False,
}
html_static_path = ["_static"]
html_title = f"evoparam {release}"
numpydoc_show_class_members = False
autosummary_generate = True
# every literature key [LIT-<axis>-<n>] cited in the docs (REQ-302), generated from
# docs/spec/LITERATURE.md by tools/gen_references_bib.py
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"
# linkcheck (REQ-302): 0 broken links; DOIs behind paywalls answer 403 to robots
linkcheck_ignore = [
    r"https://doi\.org/10\.1016/.*",
    r"https://doi\.org/10\.1002/.*",
    r"https://doi\.org/10\.1093/.*",
    r"https://doi\.org/10\.1152/.*",
    r"https://doi\.org/10\.1109/.*",
    r"https://www\.sciencedirect\.com/.*",
]
linkcheck_timeout = 30
linkcheck_retries = 2
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "mne": ("https://mne.tools/stable", None),
    "pynwb": ("https://pynwb.readthedocs.io/en/stable", None),
}
sphinx_gallery_conf = {
    "examples_dirs": ["tutorials", "../examples"],
    "gallery_dirs": ["auto_tutorials", "auto_examples"],
    "filename_pattern": r"/(preset|integration|reporting|plot|ccep)_",
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": False,
    "run_stale_examples": False,
    "remove_config_comments": True,
}

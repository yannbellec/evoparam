"""Sphinx configuration of the evoparam documentation (spec §9.7; completed at step S20).

``sphinx-build -b html docs docs/_build/html`` builds the site and runs the tutorials of
``docs/tutorials`` (sphinx-gallery); ``-D sphinx_gallery_conf.plot_gallery=0`` skips running
them.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import evoparam  # noqa: E402

project = "evoparam"
author = "Yann V. Bellec"
copyright = "Yann V. Bellec"
version = release = evoparam.__version__

extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.intersphinx",
              "numpydoc", "sphinx_gallery.gen_gallery", "myst_parser"]
templates_path = []
exclude_patterns = ["_build", "spec", "tutorials/README.txt", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
html_theme = "pydata_sphinx_theme"
html_title = f"evoparam {release}"
numpydoc_show_class_members = False
autosummary_generate = True
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "mne": ("https://mne.tools/stable", None),
    "pynwb": ("https://pynwb.readthedocs.io/en/stable", None),
}
sphinx_gallery_conf = {
    "examples_dirs": ["tutorials"],
    "gallery_dirs": ["auto_tutorials"],
    "filename_pattern": r"/(preset|integration|reporting)_",
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": False,
    "run_stale_examples": False,
    "remove_config_comments": True,
}

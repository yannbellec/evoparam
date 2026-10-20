# SPDX-License-Identifier: Apache-2.0
"""Sphinx configuration of the evoparam website (home page and tutorials).

Build: ``python -m sphinx -b html site site/_build`` from the repository root; the
pages are published from the ``gh-pages`` branch.
"""

import sphinx_bootstrap_theme

project = "evoparam"
copyright = "the evoparam developers"  # noqa: A001  (Sphinx setting)
extensions = ["sphinx_gallery.gen_gallery", "sphinx.ext.mathjax"]
exclude_patterns = ["_build"]
templates_path = ["_templates"]

html_theme = "bootstrap"
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
html_theme_options = {
    "navbar_title": "evoparam",
    "navbar_sidebarrel": False,
    "navbar_pagenav": False,
    "bootswatch_theme": "flatly",
    "navbar_links": [
        ("Tutorials", "auto_tutorials/index"),
        ("GitHub", "https://github.com/yannbellec/evoparam", True),
    ],
}
html_sidebars = {"**": []}
html_show_sourcelink = False
html_title = "evoparam"

# The tutorials written for newcomers (plot_*); the preset and integration walkthroughs
# stay in the full documentation (docs/).
sphinx_gallery_conf = {
    "examples_dirs": ["../docs/tutorials"],
    "gallery_dirs": ["auto_tutorials"],
    "filename_pattern": r"/plot_",
    "ignore_pattern": r"(preset|integration|reporting)_",
    "capture_repr": ("_repr_html_", "__repr__"),
    "ignore_repr_types": r"matplotlib\.(text|axes|figure)",
    "download_all_examples": False,
}

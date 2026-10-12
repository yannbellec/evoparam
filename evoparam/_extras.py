# SPDX-License-Identifier: Apache-2.0
"""The extras table of spec §9.2 (REQ-292), as data.

``EXTRAS`` is the table of the specification: extra name -> {distribution: floor or None}.
``ALTERNATIVES`` are the packages the table names as an alternative ("or"), which the
pyproject may use instead. ``ADDITIONS`` are the packages ``pyproject.toml`` declares beyond
the table, each with the reason. ``UNIONS`` are the extras that collect others (``all``,
``cli``) or tool chains (``docs``, ``tests``), which the "exactly one extra" rule does not
count. ``evoparam/tests/test_dependencies.py`` checks ``pyproject.toml`` against this file.
"""

EXTRAS = {
    "mne": {"mne": "1.6"},
    "bids": {"mne-bids": "0.15", "pybids": "0.16"},
    "nwb": {"pynwb": "2.6", "hdmf": "3.12", "ndx-events": "0.2"},
    "neo": {"neo": "0.13"},
    "edf": {"edfio": "0.4"},
    "bayes": {"numpyro": "0.14", "jax": "0.4.30", "arviz": "0.18"},
    "gpu": {"jax": "0.4.30"},
    "fast": {"numba": "0.59"},
    "plot": {"matplotlib": "3.7"},
    "data": {"pandas": "2.0", "pyarrow": "14"},
    "xarray": {"xarray": "2023.1"},
    "stats": {"statsmodels": "0.14", "formulaic": "1.0"},
    "validate": {"bidsschematools": "0.7", "nwbinspector": "0.5", "jsonschema": "4.18"},
    "cli": {"pandas": "2.0", "pyarrow": "14", "tqdm": "4.66"},
    "progress": {"tqdm": "4.66"},
    "parallel": {"joblib": "1.3"},
    "stream": {"pylsl": "1.16"},
    "docs": {
        "sphinx": "7",
        "pydata-sphinx-theme": "0.15",
        "sphinx-gallery": "0.15",
        "numpydoc": "1.6",
        "sphinx-copybutton": None,
        "sphinxcontrib-bibtex": None,
    },
    "tests": {
        "pytest": "8",
        "pytest-cov": "5",
        "pytest-timeout": "2.3",
        "pytest-xdist": "3.5",
        "hypothesis": "6.100",
        "scikit-learn": "1.4",
        "mypy": "1.10",
        "ruff": "0.5",
        "interrogate": "1.7",
    },
}

# "edfio>=0.4 (preferred) or pyedflib>=0.1.36"; "(alternative backend pymc>=5.10)";
# "jax[cuda12]>=0.4.30 or cupy-cuda12x>=13"
ALTERNATIVES = {
    "edf": {"pyedflib": "0.1.36"},
    "bayes": {"pymc": "5.10"},
    "gpu": {"cupy-cuda12x": "13"},
}

ADDITIONS = {
    "nwb": {
        "h5py": "the NWB reader opens HDF5 files directly (remote reads, §5.1)",
        "remfile": "remote NWB reads by byte ranges (§5.1, DANDI)",
    },
    "neo": {"quantities": "Neo returns quantities arrays; units are converted on read"},
    "brain": {
        "nilearn": "glass-brain backend of spec §5.9 method 3 ('the nilearn plotting "
        "backend is used when installed')"
    },
    "sklearn": {
        "scikit-learn": "scikit-learn >= 1.6 for the estimator tags API of §10.4 "
        "(the tests extra keeps it for check_estimator)"
    },
    "cli": {"tomli": "TOML configuration files of the command line on Python 3.10"},
    "docs": {
        "myst-parser": "Markdown pages of the docs (presets.md)",
        "matplotlib": "gallery figures",
        "pandas": "gallery tables",
        "mne": "MNE integration tutorial",
        "pynwb": "NWB integration tutorial",
        "neo": "Neo integration tutorial",
        "scikit-learn": "scikit-learn tutorial",
    },
    "tests": {
        "bidsschematools": "BIDS schema checks of written derivatives (REQ-315)",
        "mpmath": "high-precision reference values of the shape tests",
        "pyyaml": "static checks of the CI workflows and CITATION.cff (REQ-297, REQ-308, REQ-398)",
        "tomli": "pyproject parsing in the tests on Python 3.10",
    },
}

UNIONS = {"all", "cli", "docs", "tests"}

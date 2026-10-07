"""Registry of public datasets of evoked responses.

The registry is a versioned JSON file shipped with the package
(``evoparam/io/registry.json``), read with the standard library. Each entry gives
the identifier, title, repository, DOI, modality, species, sampling frequency,
number of subjects, size, licence, events status, loader entry point and known
quirks. A value that the sources do not state is ``null`` (None).
"""

import copy
import functools
import json
from importlib import resources

from evoparam.errors import ReaderError
from evoparam.utils._optional import require

__all__ = ["list_datasets", "dataset_info", "EVENTS_STATUS"]

EVENTS_STATUS = ("present", "absent", "unknown")

# Columns of list_datasets, in order.
_COLUMNS = ("id", "title", "repository", "modality", "recording", "species", "sfreq",
            "n_subjects", "size_gb", "license", "events_status", "format", "loader", "doi")


@functools.lru_cache(maxsize=1)
def _load_registry():
    text = resources.files("evoparam.io").joinpath("registry.json").read_text(encoding="utf-8")
    return json.loads(text)


def _find(dataset_id):
    key = str(dataset_id).strip().lower()
    for entry in _load_registry()["datasets"]:
        if key == entry["id"].lower() or key in (alias.lower() for alias in entry["aliases"]):
            return entry
    return None


def dataset_info(dataset_id):
    """Return the registry entry of one dataset.

    Parameters
    ----------
    dataset_id : str
        Identifier or alias, case-insensitive (for example ``"ds004080"`` or ``"erp-core"``).

    Returns
    -------
    info : dict
        A copy of the entry. Keys: ``id``, ``aliases``, ``title``, ``repository``, ``url``,
        ``doi``, ``version``, ``publications`` (DOIs), ``modality``, ``recording``,
        ``species``, ``sfreq`` (Hz), ``n_subjects``, ``size_gb``, ``license``,
        ``events_status`` (``"present"``, ``"absent"`` or ``"unknown"``), ``format``,
        ``loader`` (dotted name of the reading function, or None), ``quirks``,
        ``references`` (citation keys of the specification) and ``files``
        (path → SHA-256). Values the sources do not state are None.

    Raises
    ------
    ReaderError
        If the dataset is not in the registry.
    """

    entry = _find(dataset_id)
    if entry is None:
        known = ", ".join(item["id"] for item in _load_registry()["datasets"])
        raise ReaderError(f"Dataset {dataset_id!r} is not in the registry. Known: {known}.")
    return copy.deepcopy(entry)


def list_datasets(*, modality=None, species=None, max_size_gb=None):
    """List the datasets of the registry.

    Parameters
    ----------
    modality : str or list of str, optional
        Keep these modalities (for example ``"spes"`` or ``["erp", "tep"]``).
    species : str or list of str, optional
        Keep these species (for example ``"human"``).
    max_size_gb : float, optional
        Keep datasets whose stated size is at most this many GB; datasets of
        unknown size are left out.

    Returns
    -------
    table : pandas.DataFrame
        One row per dataset with the columns ``id``, ``title``, ``repository``,
        ``modality``, ``recording``, ``species``, ``sfreq``, ``n_subjects``, ``size_gb``,
        ``license``, ``events_status``, ``format``, ``loader`` and ``doi``.
        ``table.attrs["registry_version"]`` holds the registry version.
    """

    pd = require("pandas", extra="data", caller="evoparam.io.list_datasets")

    def as_set(value):
        if value is None:
            return None
        return {value} if isinstance(value, str) else set(value)

    modalities, species_set = as_set(modality), as_set(species)
    rows = []
    for entry in _load_registry()["datasets"]:
        if modalities is not None and entry["modality"] not in modalities:
            continue
        if species_set is not None and entry["species"] not in species_set:
            continue
        if max_size_gb is not None and (entry["size_gb"] is None
                                        or entry["size_gb"] > max_size_gb):
            continue
        rows.append({column: entry[column] for column in _COLUMNS})
    table = pd.DataFrame(rows, columns=list(_COLUMNS))
    for column in ("sfreq", "size_gb"):
        table[column] = table[column].astype(float)
    table.attrs["registry_version"] = _load_registry()["version"]
    return table


def match_dataset(path):
    """Registry entry whose identifier appears in a path or URL, or None."""

    text = str(path).lower().replace("\\", "/")
    for entry in _load_registry()["datasets"]:
        for name in [entry["id"]] + list(entry["aliases"]):
            name = name.lower()
            if text == name or f"/{name}/" in f"/{text}/" or text.endswith("/" + name):
                return copy.deepcopy(entry)
    return None

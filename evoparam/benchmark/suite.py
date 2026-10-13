# SPDX-License-Identifier: Apache-2.0
"""Frozen simulated suites: cells, seeds, the hidden split, task inputs and truth (spec §8.2).

A suite file (``suites/core.json``, ``suites/full.json``; schema
``evoparam-benchmark-suite/1``) lists groups of cells. A group names a generator of
:mod:`evoparam.sim` (or ``"mixed"``, each cell then naming its own), the settings shared by
its cells and the tasks its cells feed; each cell has an id, a split (``public`` or
``hidden``), its own settings and, for the public split, its seed. The settings key ``_``
holds what the benchmark needs besides the generator's arguments (hints given to methods,
the T5 target, the T6 component). The data are never stored: they are generated on the
fly from the seed.

The hidden split (20 % of the cells, spec §8.2) has no seed in the suite file. Its seeds
are kept in a separate document (schema ``evoparam-benchmark-hidden-seeds/1``: a random
256-bit ``nonce`` and the seeds of the hidden cells of every suite) that is not in the
repository; the suite file holds the SHA-256 of that document's canonical JSON
(``hidden_split.commitment_sha256``), so that the hidden cells cannot be changed after
submissions are seen (REQ-356). The nonce makes the hash impossible to invert by trying
every seed.
"""

import copy
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources

import numpy as np

from evoparam._typing import Any
from evoparam.errors import DataError, SettingsError

from ._canon import canonical_bytes, sha256_bytes
from .tasks import TASKS, TaskInput, make_descriptor

__all__ = [
    "Suite",
    "Cell",
    "load_suite",
    "list_suites",
    "simulate_cell",
    "make_inputs",
    "hidden_commitment",
    "load_hidden_seeds",
    "GENERATOR_NAMES",
    "SUITE_SCHEMA",
    "HIDDEN_SCHEMA",
    "ERPCORE",
]

SUITE_SCHEMA = "evoparam-benchmark-suite/1"
HIDDEN_SCHEMA = "evoparam-benchmark-hidden-seeds/1"
GENERATOR_NAMES = ("simulate_ccep", "simulate_erp", "simulate_bursts")


def _erpcore() -> Any:
    text = resources.files("evoparam.benchmark").joinpath("erpcore.json").read_text("utf-8")
    return json.loads(text)


ERPCORE = _erpcore()


@dataclass(frozen=True)
class Cell:
    """One simulated cell of a suite."""

    suite: str
    version: str
    group: str
    id: str
    split: str
    seed: int | None
    generator: str
    settings: dict[str, Any] = field(hash=False, compare=False)
    tasks: tuple[Any, ...] = ()
    kind: str = ""

    def generator_settings(self) -> dict[Any, Any]:
        """Keyword arguments of the generator (shared settings, then the cell's), with the seed."""
        if self.seed is None:
            raise DataError(
                f"cell {self.id}: hidden cell without its seed (give the hidden seeds document)."
            )
        out = copy.deepcopy(self.settings)
        out.pop("_", None)
        out["seed"] = int(self.seed)
        return out


@dataclass
class Suite:
    """A loaded suite file."""

    name: str
    version: str
    description: str
    groups: dict[str, Any]
    tasks: dict[str, Any]
    hidden_split: dict[str, Any]
    sha256: str
    path: str = ""
    hidden_seeds: dict[str, Any] | None = None

    def cells(self, task: Any = None, split: str = "public") -> list[Any]:
        """Return the cells feeding ``task`` (all tasks when None) of a split.

        The split is ``"public"``, ``"hidden"`` or ``"all"``. Hidden cells carry their seed only
        when the hidden seeds document was given (:meth:`with_hidden_seeds`).
        """
        out = []
        for gname, group in self.groups.items():
            if task is not None and task not in group["tasks"]:
                continue
            for raw in group["cells"]:
                if split != "all" and raw["split"] != split:
                    continue
                seed = raw.get("seed")
                if raw["split"] == "hidden":
                    seed = (
                        None
                        if self.hidden_seeds is None
                        else self.hidden_seeds[self.name][raw["id"]]
                    )
                settings = copy.deepcopy(group.get("settings", {}))
                settings.update(copy.deepcopy(raw.get("settings", {})))
                out.append(
                    Cell(
                        suite=self.name,
                        version=self.version,
                        group=gname,
                        id=raw["id"],
                        split=raw["split"],
                        seed=seed,
                        generator=raw.get("generator", group["generator"]),
                        settings=settings,
                        tasks=tuple(group["tasks"]),
                        kind=raw.get("kind", ""),
                    )
                )
        return out

    def task_ids(self) -> list[str]:
        """Return the tasks of the suite, in order."""
        return [t for t in TASKS if t in self.tasks]

    def with_hidden_seeds(self, document: Any) -> "Suite":
        """Return a copy whose hidden cells carry their seeds.

        ``document`` (a dict or a path) is first checked against the commitment of the suite file.

        Raises
        ------
        DataError
            The document's SHA-256 is not the commitment, or a hidden cell has no seed.
        """
        document = load_hidden_seeds(document)
        digest = hidden_commitment(document)
        if digest != self.hidden_split["commitment_sha256"]:
            raise DataError(
                f"hidden seeds: SHA-256 {digest} is not the commitment of suite "
                f"{self.name!r} ({self.hidden_split['commitment_sha256']})."
            )
        seeds = document["seeds"].get(self.name, {})
        missing = [c.id for c in self.cells(split="hidden") if c.id not in seeds]
        if missing:
            raise DataError(f"hidden seeds: no seed for {missing}.")
        out = copy.copy(self)
        out.hidden_seeds = document["seeds"]
        return out


def list_suites() -> list[str]:
    """Names of the suites shipped with the package."""
    folder = resources.files("evoparam.benchmark").joinpath("suites")
    return sorted(p.name[:-5] for p in folder.iterdir() if p.name.endswith(".json"))


def _check_suite(doc: Any, where: Any) -> None:
    if not isinstance(doc, dict) or doc.get("schema") != SUITE_SCHEMA:
        raise DataError(f"{where}: not a suite file (schema {SUITE_SCHEMA}).")
    ids = set()
    for gname, group in doc["groups"].items():
        if group.get("generator") not in GENERATOR_NAMES + ("mixed",):
            raise DataError(
                f"{where}: group {gname}: generator must be one of {GENERATOR_NAMES} or 'mixed'."
            )
        for task in group["tasks"]:
            if task not in TASKS:
                raise DataError(f"{where}: group {gname}: unknown task {task}.")
        for cell in group["cells"]:
            if cell["id"] in ids:
                raise DataError(f"{where}: duplicate cell id {cell['id']}.")
            ids.add(cell["id"])
            if cell.get("generator", group["generator"]) not in GENERATOR_NAMES:
                raise DataError(
                    f"{where}: cell {cell['id']}: generator must be one of {GENERATOR_NAMES}."
                )
            if cell["split"] == "public" and not isinstance(cell.get("seed"), int):
                raise DataError(f"{where}: public cell {cell['id']} without an integer seed.")
            if cell["split"] == "hidden" and "seed" in cell:
                raise DataError(f"{where}: hidden cell {cell['id']} must not hold its seed.")


def load_suite(name_or_path: str, hidden_seeds: Any = None) -> "Suite":
    """Load a suite: a shipped name (``"core"``, ``"full"``) or the path of a suite file.

    Parameters
    ----------
    name_or_path : str
    hidden_seeds : dict or str, optional
        The hidden seeds document (or its path), checked against the commitment.

    Returns
    -------
    Suite
    """
    if os.path.isfile(str(name_or_path)):
        path = str(name_or_path)
        with open(path, "rb") as fobj:
            raw = fobj.read()
    else:
        if name_or_path not in list_suites():
            raise SettingsError(
                f"Unknown suite {name_or_path!r}; shipped suites: "
                f"{list_suites()}, or the path of a suite file."
            )
        res = (
            resources.files("evoparam.benchmark")
            .joinpath("suites")
            .joinpath(f"{name_or_path}.json")
        )
        raw = res.read_bytes()
        path = str(res)
    doc = json.loads(raw.decode("utf-8"))
    _check_suite(doc, path)
    suite = Suite(
        name=doc["name"],
        version=doc["version"],
        description=doc["description"],
        groups=doc["groups"],
        tasks=doc["tasks"],
        hidden_split=doc["hidden_split"],
        sha256=sha256_bytes(raw),
        path=path,
    )
    if hidden_seeds is not None:
        suite = suite.with_hidden_seeds(hidden_seeds)
    return suite


def load_hidden_seeds(document: Any) -> Any:
    """Return the hidden seeds document from a dict or a path (checked for its schema)."""
    if not isinstance(document, dict):
        with open(document, encoding="utf-8") as fobj:
            document = json.load(fobj)
    if (
        document.get("schema") != HIDDEN_SCHEMA
        or "nonce" not in document
        or "seeds" not in document
    ):
        raise DataError(f"not a hidden seeds document (schema {HIDDEN_SCHEMA}).")
    return document


def hidden_commitment(document: Any) -> str:
    """SHA-256 of the canonical JSON of the hidden seeds document (the commitment)."""
    return sha256_bytes(canonical_bytes(load_hidden_seeds(document)))


###################################################################################################
## Simulation


def simulate_cell(cell: Any) -> Any:
    """(TrialSet, GroundTruth) of a cell, from its generator, settings and seed."""
    from evoparam import sim

    generator = getattr(sim, cell.generator)
    return generator(**cell.generator_settings())


def _hints(cell: Any, **extra: Any) -> dict[Any, Any]:
    hints = dict(cell.settings.get("_", {}).get("hints", {}))
    hints.update(extra)
    return hints


def _base_fields(cell: Any, ts: Any, _task: Any) -> dict[str, Any]:
    return {
        "suite": {"name": cell.suite, "version": cell.version},
        "cell": cell.id,
        "split": cell.split,
        "sfreq": float(ts.sfreq),
        "tmin": float(ts.times[0]),
        "units": ts.units,
        "ch_names": list(ts.ch_names),
        "n_trials": int(ts.n_trials),
        "modality": ts.modality or "",
    }


def _truth_components(truth: Any) -> Any:
    table = truth.to_dataframe("component")
    return table[table["role"] == "target"] if "role" in table else table


def make_inputs(
    cell: "Cell", ts: Any = None, truth: Any = None, tasks: Sequence[str] | None = None
) -> dict[Any, Any]:
    """Return the task inputs of a cell, with their truth attached (``TaskInput.truth``).

    Parameters
    ----------
    cell : Cell
    ts, truth : optional
        The simulation (generated when not given).
    tasks : sequence of str, optional
        Default: every task of the cell's group.

    Returns
    -------
    dict
        Task id -> :class:`~evoparam.benchmark.tasks.TaskInput`.
    """
    if ts is None:
        ts, truth = simulate_cell(cell)
    meta = cell.settings.get("_", {})
    out = {}
    for task in tasks or cell.tasks:
        if task not in cell.tasks:
            raise DataError(f"cell {cell.id} does not feed task {task}.")
        fields = _base_fields(cell, ts, task)
        if task in ("T1", "T2"):
            arrays = {"data": ts.data}
            site = truth.settings["stim_site"]
            fields.update(site=site, stim_channels=site.split("-"), hints=_hints(cell))
            comps = _truth_components(truth)
            n1 = comps[comps["label"] == "N1"]
            truth_t = {
                "stim_channels": site.split("-"),
                "responders": sorted(set(truth.settings["responders"])),
                "n1": {
                    r.channel: {"latency": float(r.latency), "amplitude": float(r.amplitude)}
                    for r in n1.itertuples()
                },
            }
        elif task == "T3":
            arrays = {"evoked": ts.data.mean(axis=0)}
            fields.update(hints=_hints(cell))
            comps = _truth_components(truth)
            truth_t = {
                "components": comps[
                    ["channel", "label", "latency", "amplitude", "fwhm"]
                ].reset_index(drop=True)
            }
        elif task == "T4":
            arrays = {"data": ts.data}
            site = truth.settings["stim_site"]
            fields.update(
                site=site,
                stim_channels=site.split("-"),
                artifact_window=list(meta.get("artifact_window", [-0.002, 0.008])),
                hints=_hints(cell),
            )
            comps = _truth_components(truth)
            truth_t = {
                "clean": ts.data - truth.artifact_data,
                "artifact": truth.artifact_data,
                "stim_channels": site.split("-"),
                "responders": sorted(set(comps[comps["label"] == "N1"]["channel"])),
            }
        elif task == "T5":
            arrays = {"data": ts.data}
            label, window = meta["target_label"], meta["target_window"]
            trial_table = truth.to_dataframe("trial")
            trial_table = trial_table[trial_table["label"] == label]
            channels = [c for c in ts.ch_names if c in set(trial_table["channel"])]
            fields.update(
                targets=[{"channel": c, "label": label, "window": list(window)} for c in channels],
                hints=_hints(cell),
            )
            truth_t = {
                "trials": trial_table[["channel", "trial", "latency", "amplitude"]].reset_index(
                    drop=True
                )
            }
        elif task == "T6":
            average = ts.data.mean(axis=0)
            arrays = {"data": ts.data, "average": average}
            label = meta["component"]
            definition = ERPCORE["components"][label]
            fields.update(
                definitions=[
                    {
                        "condition": label,
                        "channel": definition["channel"],
                        "window": list(definition["window"]),
                        "polarity": definition["polarity"],
                        "measures": list(ERPCORE["measures"]),
                        "source": ERPCORE["source"],
                    }
                ],
                hints=_hints(cell),
            )
            truth_t = {"signal_average": truth.signal.mean(axis=0)}
        elif task == "T7":
            arrays = {"data": ts.data}
            fields.update(hints=_hints(cell))
            truth_t = {"bursts": copy.deepcopy(truth.settings.get("bursts", []))}
        else:  # pragma: no cover
            raise DataError(task)
        descriptor = make_descriptor(task, arrays, **fields)
        out[task] = TaskInput(
            descriptor=descriptor,
            arrays={k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()},
            truth=truth_t,
        )
    return out

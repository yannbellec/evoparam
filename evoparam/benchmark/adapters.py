# SPDX-License-Identifier: Apache-2.0
"""External methods run as separate processes on task files (spec §8.3 item 6, REQ-354).

An adapter wraps an external command-line tool: ``Adapter.run(task_input_path) ->
task_output_path`` runs the tool in a subprocess on the task's input file and returns the
path of the output it wrote, after validating that output against the task's schema.
Nothing of the tool is imported into this process, so a tool under any licence (GPL tools
included) can be compared without being linked with evoparam; the adapter scripts of such
tools live outside the package tree, with their own licence notice (``adapters/`` at the
root of the repository).

The command receives the input and output paths: either as the placeholders ``{input}``
and ``{output}`` in its arguments, or appended as the last two arguments. ``{python}``
stands for the running interpreter and ``{here}`` for the folder of an adapter
specification file. A specification file is JSON::

    {"name": "er-detect", "version": "2.0", "citation": "...",
     "tasks": ["T1", "T2"], "command": ["{python}", "{here}/erdetect_adapter.py"],
     "timeout": 3600}
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

from evoparam.errors import DataError, EvoparamError

from .tasks import TASKS, output_format, read_input, read_output

__all__ = ["Adapter", "AdapterError", "load_adapter"]


class AdapterError(EvoparamError):
    """The external command failed (non-zero exit, time-out, no output)."""


@dataclass
class Adapter:
    """An external method run in a subprocess on task files.

    Parameters
    ----------
    name : str
        Method name in the results.
    command : list of str
        Program and arguments; ``{input}`` and ``{output}`` are replaced by the paths (else
        both are appended), ``{python}`` by the running interpreter, ``{here}`` by
        ``here``.
    tasks : sequence of str
        Tasks the tool answers.
    version, citation : str
        Recorded in the results.
    timeout : float
        Seconds before the command is stopped.
    env : dict, optional
        Variables added to the environment of the command.
    here : str
        Folder that ``{here}`` stands for (the folder of the specification file).
    """

    name: str
    command: list
    tasks: tuple = ()
    version: str = ""
    citation: str = ""
    timeout: float = 3600.0
    env: dict = field(default_factory=dict)
    here: str = ""

    def __post_init__(self):
        if not self.command or not all(isinstance(c, str) for c in self.command):
            raise DataError(f"adapter {self.name!r}: command must be a list of strings.")
        self.tasks = tuple(self.tasks)
        unknown = [t for t in self.tasks if t not in TASKS]
        if unknown:
            raise DataError(f"adapter {self.name!r}: unknown tasks {unknown}.")

    def _argv(self, input_path, output_path):
        values = {
            "python": sys.executable,
            "here": self.here,
            "input": input_path,
            "output": output_path,
        }
        argv = [part.format(**values) for part in self.command]
        if not any("{input}" in part for part in self.command):
            argv += [input_path, output_path]
        return argv

    def check(self):
        """Raise :class:`AdapterError` when the command's program cannot be found."""

        import shutil

        program = self._argv("", "")[0]
        if (
            not (os.path.isfile(program) and os.access(program, os.X_OK))
            and shutil.which(program) is None
        ):
            raise AdapterError(f"adapter {self.name!r}: program {program!r} not found.")

    def run(self, task_input_path, output_dir=None):
        """Run the tool on one task input and return the path of its validated output.

        Parameters
        ----------
        task_input_path : str
            The ``<cell>.json`` descriptor of a task input.
        output_dir : str, optional
            Folder of the output (default: ``<input folder>/adapter-<name>``); the output
            is ``<cell>.tsv`` (``<cell>.npz`` for T4).

        Returns
        -------
        str
            Path of the output file.

        Raises
        ------
        AdapterError
            The command failed, timed out or wrote no output.
        BenchmarkValidationError
            The output does not follow the task's schema.
        """

        task_input = read_input(task_input_path)
        task = task_input.task
        if self.tasks and task not in self.tasks:
            raise DataError(f"adapter {self.name!r} does not answer task {task}.")
        output_dir = output_dir or os.path.join(
            os.path.dirname(task_input_path), f"adapter-{self.name}"
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{task_input.cell}.{output_format(task)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        argv = self._argv(os.path.abspath(task_input_path), os.path.abspath(output_path))
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, **{k: str(v) for k, v in self.env.items()}},
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise AdapterError(
                f"adapter {self.name!r}: {argv[0]} did not finish within "
                f"{self.timeout} s on {task_input.cell}."
            ) from None
        except OSError as err:
            raise AdapterError(f"adapter {self.name!r}: cannot run {argv[0]} ({err}).") from None
        self.last_duration = time.perf_counter() - start
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
            raise AdapterError(
                f"adapter {self.name!r}: exit code {proc.returncode} on {task_input.cell}: {tail}"
            )
        if not os.path.isfile(output_path):
            raise AdapterError(f"adapter {self.name!r}: no output written at {output_path}.")
        read_output(task, output_path)
        return output_path

    def info(self):
        """The ``method`` record of a result document."""

        return {
            "name": self.name,
            "version": self.version,
            "kind": "adapter",
            "citation": self.citation,
            "description": "external tool run in a separate process",
            "settings": {"command": list(self.command)},
        }


def load_adapter(path):
    """An :class:`Adapter` from a JSON specification file."""

    with open(path, encoding="utf-8") as fobj:
        spec = json.load(fobj)
    missing = [k for k in ("name", "command") if k not in spec]
    if missing:
        raise DataError(f"{path}: adapter specification without {missing}.")
    return Adapter(
        name=spec["name"],
        command=list(spec["command"]),
        tasks=tuple(spec.get("tasks", ())),
        version=str(spec.get("version", "")),
        citation=spec.get("citation", ""),
        timeout=float(spec.get("timeout", 3600.0)),
        env=spec.get("env", {}),
        here=os.path.dirname(os.path.abspath(path)),
    )

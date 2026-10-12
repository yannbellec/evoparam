# SPDX-License-Identifier: Apache-2.0
"""REQ-351: every task's input and output schemas are versioned (``benchmark-task/<id>/1``)
and validated on read; scoring rejects outputs that do not validate, naming the first
violation."""

import json
import shutil

import numpy as np
import pytest

from evoparam.benchmark import (
    TASKS,
    BenchmarkValidationError,
    get_baseline,
    load_schema,
    read_input,
    read_output,
    schema_id,
    score,
    validate,
    write_input,
    write_output,
)
from evoparam.benchmark.tasks import output_columns, output_format

from .helpers import inputs, perfect_t1, suite

jsonschema = pytest.importorskip("jsonschema")


@pytest.mark.parametrize("task", list(TASKS))
def test_req351_schemas_are_versioned_and_valid(task):
    schema = load_schema(task)
    assert schema_id(task) == f"benchmark-task/{task}/1"
    assert schema["$id"] == schema_id(task)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    jsonschema.Draft202012Validator.check_schema(schema)
    assert {"input", "output"} | ({"output_row"} if task != "T4" else set()) <= set(
        schema["$defs"]
    )
    assert schema["$defs"]["input"]["properties"]["schema"]["const"] == schema_id(task)
    assert output_format(task) == ("npz" if task == "T4" else "tsv")
    if task != "T4":
        required, optional = output_columns(task)
        assert required
        assert set(required) <= set(schema["$defs"]["output_row"]["required"])


@pytest.mark.parametrize("task", list(TASKS))
def test_req351_inputs_validate_and_round_trip(task, tmp_path):
    ti = inputs(task, 1)[0]
    path = write_input(ti, tmp_path)
    back = read_input(path)
    assert back.descriptor == ti.descriptor
    for name, array in ti.arrays.items():
        np.testing.assert_allclose(back.arrays[name], array, rtol=1e-9, atol=0)
    assert back.sha256() == ti.sha256()


def _broken_input(tmp_path, change):
    ti = inputs("T1", 1)[0]
    path = write_input(ti, tmp_path)
    with open(path, encoding="utf-8") as fobj:
        descriptor = json.load(fobj)
    change(descriptor)
    with open(path, "w", encoding="utf-8") as fobj:
        json.dump(descriptor, fobj)
    return path


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.pop("sfreq"), "(document): 'sfreq' is a required property"),
        (lambda d: d.update(sfreq=-1), "sfreq: -1 is less than or equal to the minimum of 0"),
        (
            lambda d: d.update(schema="benchmark-task/T1/2"),
            "schema: 'benchmark-task/T1/1' was expected",
        ),
        (lambda d: d["ch_names"].append(3), "ch_names/24: 3 is not of type 'string'"),
        (lambda d: d.update(split="secret"), "split: 'secret' is not one of"),
    ],
)
def test_req351_invalid_input_names_the_first_violation(tmp_path, change, message):
    path = _broken_input(tmp_path, change)
    with pytest.raises(BenchmarkValidationError) as info:
        read_input(path)
    text = str(info.value)
    assert text.startswith(str(path) + ": ")
    assert message in text
    assert "benchmark-task/T1/1" in text


def test_req351_input_arrays_are_checked_against_their_hash(tmp_path):
    ti = inputs("T1", 1)[0]
    path = write_input(ti, tmp_path)
    npz = tmp_path / ti.descriptor["data_file"]
    data = ti.arrays["data"].copy()
    data[0, 0, 0] += 1.0
    np.savez(npz, data=data)
    with pytest.raises(BenchmarkValidationError, match="differs from the shape or SHA-256"):
        read_input(path)


def test_req351_first_violation_is_stable():
    """Several violations: the first in document order is reported, every time."""

    ti = inputs("T1", 1)[0]
    descriptor = dict(ti.descriptor, sfreq="fast", tmin="early", units=5)
    messages = set()
    for _ in range(3):
        with pytest.raises(BenchmarkValidationError) as info:
            validate(descriptor, "T1", "input", where="x")
        messages.add(str(info.value))
    assert len(messages) == 1
    assert messages.pop().startswith("x: sfreq: 'fast' is not of type 'number'")


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("site\tchannel\tresponsive\n", "header: missing required column 'p_value'"),
        ("site\tchannel\tresponsive\tp_value\tcolour\n", "header: unknown column 'colour'"),
        ("site\tchannel\tp_value\tresponsive\n", "header: columns must be in the order"),
        ("site\tchannel\tresponsive\tp_value\nA1-A2\tA3\t1\n", "line 2: 3 fields for 4 columns"),
        (
            "site\tchannel\tresponsive\tp_value\nA1-A2\tA3\t2\t0.01\n",
            "line 2, column 'responsive': 2 is not one of [0, 1]",
        ),
        (
            "site\tchannel\tresponsive\tp_value\nA1-A2\tA3\t1\t0.01\nA1-A2\tA4\t0\t1.5\n",
            "line 3, column 'p_value': 1.5 is greater than the maximum of 1",
        ),
        (
            "site\tchannel\tresponsive\tp_value\nA1-A2\tA3\t1\tnan\n",
            "line 2, column 'p_value': 'nan' is not a finite number",
        ),
        (
            "site\tchannel\tresponsive\tp_value\nA1-A2\tA3\tyes\t0.01\n",
            "line 2, column 'responsive': 'yes' is not one of [0, 1]",
        ),
        ("", "empty file"),
    ],
)
def test_req351_invalid_tsv_names_the_first_violation(tmp_path, text, message):
    path = _write(tmp_path / "cell.tsv", text)
    with pytest.raises(BenchmarkValidationError) as info:
        read_output("T1", str(path))
    assert str(info.value).startswith(str(path))
    assert message in str(info.value)


def test_req351_valid_tsv_is_typed():
    rows = [
        {"site": "A1-A2", "channel": "A3", "responsive": 1, "p_value": 0.01, "score": 2.5},
        {"site": "A1-A2", "channel": "A4", "responsive": 0, "p_value": None},
    ]
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = write_output("T1", rows, f"{tmp}/c.tsv")
        back = read_output("T1", path)
    assert back[0] == rows[0]
    assert back[1] == {**rows[1], "score": None}


@pytest.mark.parametrize(
    ("task", "header", "row", "message"),
    [
        (
            "T2",
            "site\tchannel\tlatency\tamplitude",
            "S\tA1\tsoon\t-50",
            "column 'latency': 'soon' is not of type 'number'",
        ),
        (
            "T3",
            "channel\tcomponent\tlatency\tamplitude\tfwhm",
            "A1\t-1\t0.02\t-50\t0.01",
            "column 'component': -1 is less than the minimum of 0",
        ),
        (
            "T3",
            "channel\tcomponent\tlatency\tamplitude\tfwhm",
            "A1\t0\t0.02\t-50\t0",
            "column 'fwhm': 0 is less than or equal to the minimum of 0",
        ),
        (
            "T5",
            "channel\ttrial\tlatency\tamplitude",
            "A1\t0.5\t0.02\t-50",
            "column 'trial': 0.5 is not of type 'integer'",
        ),
        (
            "T6",
            "channel\tcondition\tmeasure\tvalue\tsme",
            "Pz\tP3\tpeak\t5\t1",
            "column 'measure': 'peak' is not one of",
        ),
        (
            "T7",
            "channel\tzone\tsign\tfreq_centroid\ttime_centroid",
            "A1\t0\t0\t80\t0.2",
            "column 'sign': 0 is not one of [-1, 1]",
        ),
    ],
)
def test_req351_each_task_checks_its_rows(tmp_path, task, header, row, message):
    path = _write(tmp_path / "cell.tsv", f"{header}\n{row}\n")
    with pytest.raises(BenchmarkValidationError) as info:
        read_output(task, str(path))
    assert f"{path}: line 2, " in str(info.value)
    assert message in str(info.value)
    assert schema_id(task) in str(info.value)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (np.zeros((2, 3, 4, 5)), "arrays/data/shape: [2, 3, 4, 5] is too long"),
        (np.array([[[np.nan]]]), "arrays/data/finite: True was expected"),
        (None, "arrays: Additional properties are not allowed ('cleaned' was unexpected)"),
        ("empty", "arrays: 'data' is a required property"),
    ],
)
def test_req351_t4_output_is_validated(tmp_path, data, message):
    path = tmp_path / "cell.npz"
    if data is None:
        np.savez(path, cleaned=np.zeros((1, 1, 1)))
    elif isinstance(data, str):
        np.savez(path)
    else:
        np.savez(path, data=data)
    with pytest.raises(BenchmarkValidationError) as info:
        read_output("T4", str(path))
    assert message in str(info.value)


def test_req351_scoring_rejects_an_invalid_output(tmp_path):
    """One invalid file among valid ones: scoring stops and names that file and its first
    violation."""

    cells = inputs("T1")
    folder = tmp_path / "pred"
    for ti in cells:
        write_output("T1", perfect_t1(ti), str(folder / f"{ti.cell}.tsv"))
    bad = folder / f"{cells[3].cell}.tsv"
    lines = bad.read_text(encoding="utf-8").split("\n")
    fields = lines[5].split("\t")
    fields[3] = "-0.2"
    lines[5] = "\t".join(fields)
    bad.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(BenchmarkValidationError) as info:
        score("T1", str(folder), suite("core"), inputs=list(cells))
    assert str(info.value).startswith(
        f"{bad}: line 6, column 'p_value': -0.2 is less than the minimum of 0"
    )
    shutil.rmtree(folder)


def test_req351_scoring_rejects_rows_for_unknown_channels(tmp_path):
    ti = inputs("T1", 1)[0]
    rows = perfect_t1(ti) + [
        {"site": ti.descriptor["site"], "channel": "Z99", "responsive": 1, "p_value": 0.01}
    ]
    write_output("T1", rows, str(tmp_path / f"{ti.cell}.tsv"))
    with pytest.raises(BenchmarkValidationError, match="Z99"):
        score("T1", str(tmp_path), suite("core"), inputs=[ti])


def test_req351_baseline_outputs_validate(tmp_path):
    """The outputs written by a baseline follow the schema (read back without error)."""

    ti = inputs("T1", 1)[0]
    out = get_baseline("vanblooijs2023").run({"T1": ti, "T2": inputs("T2", 1)[0]})
    for task, rows in out.items():
        path = write_output(task, rows, str(tmp_path / f"{task}.tsv"))
        assert len(read_output(task, path)) == len(rows)

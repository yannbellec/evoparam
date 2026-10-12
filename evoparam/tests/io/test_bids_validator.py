# SPDX-License-Identifier: Apache-2.0
"""REQ-314: the stimulation-metadata validator over a BIDS tree and ``evoparam validate-bids``."""

import json
import os

import pytest

pd = pytest.importorskip("pandas")

from evoparam.cli import main
from evoparam.errors import ReaderError
from evoparam.io import format_stim_report, validate_stim_metadata
from evoparam.io.bids import validate_stim_metadata as from_bids

from .helpers import make_bids


def _tsv(path, header, rows):
    with open(path, "w", encoding="utf-8") as fobj:
        fobj.write("\t".join(header) + "\n")
        for row in rows:
            fobj.write("\t".join(str(v) for v in row) + "\n")


def _json(path, content):
    with open(path, "w", encoding="utf-8") as fobj:
        json.dump(content, fobj)


def _fixtures(root):
    """Three subjects shaped like the deposits of REQ-313.

    sub-utrecht (ds004080): currents in A and pulse widths in s without Units, polarity
    'monophasic' without phase order, no channel reference, no electrodes.tsv.
    sub-mayo (ds003708/ds004977): site, current (Units mA) and pulse width (Units us) in the
    sidecar, ElectricalStimulation false (inconsistent), references and coordinates in m.
    sub-minimal: only trial_type; no sidecar, no channels.tsv.
    """

    os.makedirs(root)
    _json(os.path.join(root, "dataset_description.json"), {"Name": "f", "BIDSVersion": "1.9.0"})
    folder = os.path.join(root, "sub-utrecht", "ses-1", "ieeg")
    os.makedirs(folder)
    stem = "sub-utrecht_ses-1_task-SPESclin_run-021"
    _tsv(
        os.path.join(folder, stem + "_events.tsv"),
        [
            "onset",
            "duration",
            "trial_type",
            "electrical_stimulation_type",
            "electrical_stimulation_site",
            "electrical_stimulation_current",
            "electrical_stimulation_frequency",
            "electrical_stimulation_pulsewidth",
        ],
        [[507.0, 0.23, "artefact", "n/a", "n/a", "n/a", "n/a", "n/a"]]
        + [
            [
                2881.78 + 5 * k,
                0.001,
                "electrical_stimulation",
                "monophasic",
                "PT01-PT02",
                0.008,
                0.2,
                0.001,
            ]
            for k in range(4)
        ],
    )
    _json(os.path.join(folder, stem + "_ieeg.json"), {"ElectricalStimulation": True})
    _tsv(
        os.path.join(folder, stem + "_channels.tsv"),
        ["name", "type", "units"],
        [["PT01", "ECOG", "uV"], ["PT02", "ECOG", "uV"], ["EKG", "n/a", "uV"]],
    )

    folder = os.path.join(root, "sub-mayo", "ses-ieeg01", "ieeg")
    os.makedirs(folder)
    stem = "sub-mayo_ses-ieeg01_task-ccep_run-01"
    _tsv(
        os.path.join(folder, stem + "_events.tsv"),
        [
            "onset",
            "duration",
            "trial_type",
            "electrical_stimulation_site",
            "electrical_stimulation_current",
            "electrical_stimulation_pulsewidth",
        ],
        [[1.0 + k, 0.0002, "electrical_stimulation", "LA1-LA2", 6.0, 200] for k in range(3)],
    )
    _json(
        os.path.join(folder, stem + "_events.json"),
        {
            "electrical_stimulation_current": {"Units": "mA"},
            "electrical_stimulation_pulsewidth": {"Units": "us"},
        },
    )
    _json(os.path.join(folder, stem + "_ieeg.json"), {"ElectricalStimulation": False})
    _tsv(
        os.path.join(folder, stem + "_channels.tsv"),
        ["name", "type", "reference"],
        [["LA1", "SEEG", "LB1"], ["LA2", "SEEG", "LB1"]],
    )
    _tsv(
        os.path.join(folder, "sub-mayo_ses-ieeg01_electrodes.tsv"),
        ["name", "x", "y", "z"],
        [["LA1", 0.01, 0.02, 0.03], ["LA2", 0.011, 0.02, 0.03]],
    )
    _json(
        os.path.join(folder, "sub-mayo_ses-ieeg01_coordsystem.json"),
        {"iEEGCoordinateSystem": "Other", "iEEGCoordinateUnits": "m"},
    )

    folder = os.path.join(root, "sub-minimal", "ieeg")
    os.makedirs(folder)
    _tsv(
        os.path.join(folder, "sub-minimal_task-spes_events.tsv"),
        ["onset", "duration", "trial_type"],
        [[1.0, 0.001, "electrical_stimulation"], [2.0, 0.001, "electrical_stimulation"]],
    )
    return root


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    return _fixtures(str(tmp_path_factory.mktemp("validator") / "bids"))


def _status(table, subject):
    run = table[table["subject"] == subject]
    return dict(zip(run["field"], run["status"], strict=False))


def test_statuses_on_the_three_fixtures(fixtures):

    table = validate_stim_metadata(fixtures)
    assert from_bids is validate_stim_metadata
    assert list(table.columns) == [
        "subject",
        "session",
        "task",
        "acquisition",
        "run",
        "datatype",
        "events_file",
        "field",
        "status",
        "n_events",
        "message",
    ]
    assert set(table["status"]) <= {"present", "inferred", "missing", "inconsistent"}
    assert len(table) == 3 * 10

    utrecht = _status(table, "utrecht")
    assert utrecht["stim_site"] == "present"
    assert utrecht["stim_current_mA"] == "inferred"  # 0.008 read as A (8 mA)
    assert utrecht["stim_pulse_width_us"] == "inferred"  # 0.001 read as s (1000 us)
    assert utrecht["stim_frequency_hz"] == "inferred"  # 0.2 without unit, read as Hz
    assert utrecht["stim_polarity"] == "inconsistent"  # monophasic, phase order unknown
    assert utrecht["stim_n_pulses"] == "missing"
    assert utrecht["ElectricalStimulation"] == "present"
    assert utrecht["reference"] == "missing"
    assert utrecht["type"] == "inconsistent"  # EKG typed n/a
    assert utrecht["coordinates"] == "missing"
    assert set(table.loc[table["subject"] == "utrecht", "n_events"]) == {4}

    mayo = _status(table, "mayo")
    assert mayo["stim_current_mA"] == "present"
    assert mayo["stim_pulse_width_us"] == "present"
    assert mayo["stim_site"] == "present"
    assert mayo["stim_polarity"] == "missing"
    assert mayo["ElectricalStimulation"] == "inconsistent"
    assert mayo["reference"] == "present"
    assert mayo["type"] == "present"
    assert mayo["coordinates"] == "present"

    minimal = _status(table, "minimal")
    assert {
        minimal[f]
        for f in (
            "stim_site",
            "stim_current_mA",
            "stim_pulse_width_us",
            "stim_polarity",
            "stim_frequency_hz",
            "stim_n_pulses",
            "ElectricalStimulation",
            "reference",
            "type",
            "coordinates",
        )
    } == {"missing"}


def test_text_report_lists_the_missing_pulse_width(fixtures):

    text = format_stim_report(validate_stim_metadata(fixtures, subject="minimal"))
    lines = text.splitlines()
    assert lines[0].startswith("sub-minimal/ieeg/sub-minimal_task-spes_events.tsv")
    assert any(line.split()[:2] == ["missing", "stim_pulse_width_us"] for line in lines)
    assert lines[-1] == "summary: 0 present, 0 inferred, 10 missing, 0 inconsistent"


def test_partial_and_unreadable_values_are_inconsistent(tmp_path):

    root = _fixtures(str(tmp_path / "bids"))
    path = os.path.join(
        root, "sub-mayo", "ses-ieeg01", "ieeg", "sub-mayo_ses-ieeg01_task-ccep_run-01_events.tsv"
    )
    _tsv(
        path,
        [
            "onset",
            "duration",
            "trial_type",
            "electrical_stimulation_site",
            "electrical_stimulation_current",
            "electrical_stimulation_pulsewidth",
        ],
        [
            [1.0, 0.0002, "electrical_stimulation", "LA1-LA2", 6.0, 200],
            [2.0, 0.0002, "electrical_stimulation", "n/a", "high", 200],
        ],
    )
    mayo = validate_stim_metadata(root, subject="sub-mayo")
    status = dict(zip(mayo["field"], mayo["status"], strict=False))
    assert status["stim_site"] == "inconsistent"
    assert status["stim_current_mA"] == "inconsistent"
    message = mayo.set_index("field").loc["stim_current_mA", "message"]
    assert "1 of 2" in message
    assert "unparsable" in message


def test_strict_raises_on_missing_fields(fixtures):

    with pytest.raises(ReaderError, match="stim_pulse_width_us"):
        validate_stim_metadata(fixtures, subject="minimal", strict=True)
    with pytest.raises(ReaderError, match="No \\*_events.tsv"):
        validate_stim_metadata(fixtures, subject="nobody")
    with pytest.raises(ReaderError, match="not a local BIDS folder"):
        validate_stim_metadata(os.path.join(fixtures, "absent"))


def test_mini_dataset_of_the_reader_tests(tmp_path):

    root = make_bids(str(tmp_path / "mini"))
    table = validate_stim_metadata(root)
    sub01 = _status(table, "01")
    assert sub01["stim_current_mA"] == "present"  # Units A in the root events.json
    assert sub01["stim_pulse_width_us"] == "inferred"  # 0.0002 without units
    assert sub01["coordinates"] == "present"
    assert sub01["reference"] == "present"
    assert _status(table, "02")["coordinates"] == "missing"


def test_cli_prints_the_report_and_exits_5_with_strict(fixtures, capsys):

    assert main(["validate-bids", fixtures]) == 0
    out = capsys.readouterr().out
    assert "missing      stim_pulse_width_us" in out
    assert out.strip().splitlines()[-1].startswith("summary:")
    assert main(["validate-bids", fixtures, "--sub", "mayo", "--strict"]) == 5
    capsys.readouterr()
    assert main(["validate-bids", fixtures, "--json"]) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["status"] == "ok"
    assert len(summary["result"]["fields"]) == 30
    assert summary["result"]["counts"]["missing"] >= 10

# SPDX-License-Identifier: Apache-2.0
"""REQ-313: stimulation-event harmonisation on the three deposit shapes, and read_bids.

Fixtures copy the column layout of the deposits: ds004080 (Utrecht: currents in A and pulse
widths in s, without Units), ds003708 / ds004977 (Mayo Clinic: current written "6.0 mA",
``biphasic`` without phase order, no pulse-width column; headers as published on
OpenNeuro) and a minimal file with only ``trial_type``.
"""

import io
import os
import shutil

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from evoparam.io import harmonize_stim_events, read_bids

from .helpers import make_bids


def _table(text):
    return pd.read_csv(io.StringIO(text), sep="\t", na_values=["n/a"], keep_default_na=False)


UTRECHT = _table(
    "onset\tduration\ttrial_type\tsub_type\telectrodes_involved_onset\t"
    "electrical_stimulation_type\telectrical_stimulation_site\t"
    "electrical_stimulation_current\telectrical_stimulation_frequency\t"
    "electrical_stimulation_pulsewidth\tnotes\n"
    "507.0\t0.23\tartefact\tn/a\tall\tn/a\tn/a\tn/a\tn/a\tn/a\tn/a\n"
    "2881.78\t0.001\telectrical_stimulation\tSPESclin\tn/a\tmonophasic\tPT01-PT02\t0.008\t0.2"
    "\t0.001\tn/a\n"
    "2950.10\t0.001\telectrical_stimulation\tSPESclin\tn/a\tmonophasic\tPT03_PT02\t0.004\t0.2"
    "\t0.001\tn/a\n")

# ds003708 sub-01 run-01 (first rows, published header)
MAYO_DS003708 = _table(
    "onset\tduration\ttrial_type\telectrodes_involved_onset\telectrodes_involved_offset\t"
    "sample_start\tsample_end\telectrical_stimulation_type\telectrical_stimulation_site\t"
    "electrical_stimulation_site_num\telectrical_stimulation_current\tnotes\tstatus\t"
    "status_description\n"
    "510.07373046875\t0.0002\telectrical_stimulation\tn/a\tn/a\t1044631\tn/a\tbiphasic\t"
    "LTG1-LTG2\t13-14\t6.0 mA\tn/a\tgood\tn/a\n"
    "594.72705078125\t0.0002\telectrical_stimulation\tn/a\tn/a\t1218001\tn/a\tbiphasic\t"
    "LTG1-LTG2\t13-14\t6.0 mA\tn/a\tgood\tn/a\n")

# ds004977 sub-1 run-01 (first rows, published header)
MAYO_DS004977 = _table(
    "onset\tduration\tsample_start\ttrial_type\telectrical_stimulation_type\t"
    "electrical_stimulation_site\telectrical_stimulation_site_num\t"
    "electrical_stimulation_current\tstatus\tstatus_description\n"
    "12.12125\t0.0002\t58182\telectrical_stimulation\tbiphasic\tROC1-ROC2\t175-176\t6.0 mA\t"
    "good\tn/a\n"
    "16.26625\t0.0002\t78078\telectrical_stimulation\tbiphasic\tROC1-ROC2\t175-176\t6.0 mA\t"
    "good\tn/a\n")

MINIMAL = _table("onset\tduration\ttrial_type\n1.0\t0.001\telectrical_stimulation\n"
                 "2.0\t0.001\telectrical_stimulation\n")


def test_utrecht_shape():

    out, report = harmonize_stim_events(UTRECHT)
    stim = out[out["trial_type"] == "electrical_stimulation"]
    assert list(stim["stim_site"]) == ["PT01-PT02", "PT03-PT02"]
    assert list(stim["stim_current_mA"]) == pytest.approx([8.0, 4.0])
    assert any(w.startswith("unit_inferred: electrical_stimulation_current")
               for w in report.warnings)
    assert list(stim["stim_pulse_width_us"]) == pytest.approx([1000.0, 1000.0])
    assert list(stim["stim_polarity"]) == ["unknown"] * 2
    assert list(stim["evoparam_stim_type_raw"]) == ["monophasic"] * 2
    # originals kept under their own names, trial_type unchanged
    assert {"electrical_stimulation_current", "sub_type", "notes"} <= set(out.columns)
    assert list(out["trial_type"]) == list(UTRECHT["trial_type"])


@pytest.mark.parametrize("table", [MAYO_DS003708, MAYO_DS004977], ids=["ds003708", "ds004977"])
def test_mayo_shape(table):

    out, report = harmonize_stim_events(table)
    assert set(out["stim_site"]) == {table["electrical_stimulation_site"][0]}
    assert list(out["stim_current_mA"]) == pytest.approx([6.0, 6.0])     # "6.0 mA" inline
    assert report.units["stim_current_mA"] == "inline"
    assert not any(w.startswith("unit_inferred") for w in report.warnings)
    assert out["stim_pulse_width_us"].isna().all()                       # no column
    assert "stim_pulse_width_us" in report.missing
    assert set(out["stim_polarity"]) == {"unknown"}                      # order not stated
    assert "electrical_stimulation_site_num" in out.columns


def test_mayo_phase_order_from_the_sidecar():

    sidecar = {"ElectricalStimulationParameters": "biphasic pulses, cathodic phase first"}
    out, _ = harmonize_stim_events(MAYO_DS004977, sidecar)
    assert set(out["stim_polarity"]) == {"biphasic_cathodic_first"}


def test_minimal_shape():

    out, report = harmonize_stim_events(MINIMAL)
    assert list(out["stim_site"]) == ["n/a", "n/a"]
    assert out[["stim_current_mA", "stim_pulse_width_us"]].isna().all().all()
    assert set(report.missing) >= {"stim_site", "stim_current_mA", "stim_pulse_width_us"}
    assert list(out["condition"]) == ["electrical_stimulation"] * 2


def test_user_mapping_overrides_the_default():

    table = UTRECHT.assign(my_ma=[np.nan, 2.5, 3.5])
    out, report = harmonize_stim_events(table, {"my_ma": {"Units": "mA"}},
                                        mapping={"my_ma": "stim_current_mA"})
    stim = out[out["trial_type"] == "electrical_stimulation"]
    assert list(stim["stim_current_mA"]) == pytest.approx([2.5, 3.5])
    assert report.mapping["stim_current_mA"] == "my_ma"
    assert report.user_mapping == {"my_ma": "stim_current_mA"}


def test_read_bids_harmonises_every_events_file(tmp_path):

    root = make_bids(str(tmp_path / "bids"))
    ts = read_bids(root, "01", task="spes")
    meta = ts.metadata
    assert {"stim_site", "stim_current_mA", "stim_pulse_width_us", "stim_polarity",
            "electrical_stimulation_current"} <= set(meta.columns)
    assert meta["stim_current_mA"].tolist() == pytest.approx([4.0] * len(meta))
    harmonization = ts.provenance.steps[-1]["runs"][0]["harmonization"]
    assert harmonization["mapping"]["stim_current_mA"] == "electrical_stimulation_current"

    # a user mapping through read_bids: the run-level column "dose" becomes the current
    folder = os.path.join(root, "sub-01", "ieeg")
    path = os.path.join(folder, "sub-01_task-spes_run-01_events.tsv")
    events = pd.read_csv(path, sep="\t", keep_default_na=False)
    events["dose"] = ["2.5 mA" if t == "electrical_stimulation" else "n/a"
                      for t in events["trial_type"]]
    events.to_csv(path, sep="\t", index=False)
    mapped = read_bids(root, "01", task="spes", stim_mapping={"dose": "stim_current_mA"})
    assert mapped.metadata["stim_current_mA"].tolist() == pytest.approx([2.5] * len(meta))
    run = mapped.provenance.steps[-1]["runs"][0]["harmonization"]
    assert run["user_mapping"] == {"dose": "stim_current_mA"}
    shutil.rmtree(root)

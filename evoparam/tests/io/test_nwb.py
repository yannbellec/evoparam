# SPDX-License-Identifier: Apache-2.0
"""NWB reader and writer (REQ-317, REQ-318)."""

import warnings
from typing import Any

import numpy as np
import pytest

pynwb = pytest.importorskip("pynwb")

from evoparam.io import read_nwb, read_nwb_results, write_nwb  # noqa: E402

SFREQ = 1000.0
N_CH = 4
DURATION = 30.0
ONSETS = np.arange(2.0, 28.0, 2.0)


def _signal() -> Any:
    rng = np.random.default_rng(0)
    n = int(DURATION * SFREQ)
    data = rng.normal(0, 1e-6, (n, N_CH))
    for t in ONSETS:
        i = int(round(t * SFREQ))
        data[i + 20 : i + 40, 0] += -50e-6
    return data


def _nwbfile(source: Any) -> Any:
    from datetime import datetime, timezone

    from pynwb import NWBFile
    from pynwb.ecephys import ElectricalSeries

    nwb = NWBFile(
        session_description="test",
        identifier=f"evoparam-{source}",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    device = nwb.create_device(name="probe")
    group = nwb.create_electrode_group(
        name="shank0", description="shank", location="VISp", device=device
    )
    for i in range(N_CH):
        nwb.add_electrode(
            group=group, location="VISp" if i < 2 else "CA1", x=float(i), y=0.0, z=0.0
        )
    region = nwb.create_electrode_table_region(list(range(N_CH)), "all")
    data = _signal()
    if source == "icephys":
        nwb.add_acquisition(
            ElectricalSeries(
                name="lfp", data=data, electrodes=region, rate=SFREQ, starting_time=0.0
            )
        )
    else:
        nwb.add_acquisition(
            ElectricalSeries(
                name="lfp", data=data, electrodes=region, rate=SFREQ, starting_time=0.0
            )
        )
    if source == "events_table":
        from pynwb.event import EventsTable

        table = EventsTable(name="stim_events", description="pulses")
        table.add_column("amplitude", "current (A)")
        table.add_column("pulse_width", "pulse width (s)")
        for t in ONSETS:
            table.add_row(timestamp=float(t), duration=0.001, amplitude=0.006, pulse_width=0.0002)
        nwb.add_acquisition(table)
    elif source == "trials":
        nwb.add_trial_column("stim_onset", "stimulus onset")
        for t in ONSETS:
            nwb.add_trial(start_time=float(t) - 0.5, stop_time=float(t) + 1.0, stim_onset=float(t))
    elif source == "intervals":
        from pynwb.epoch import TimeIntervals

        table = TimeIntervals(name="flashes", description="flashes")
        table.add_column("color", "flash colour")
        for t in ONSETS:
            table.add_row(start_time=float(t), stop_time=float(t) + 0.25, color=1.0)
        nwb.add_time_intervals(table)
    elif source == "stim_table":
        stim_table_cls = _stim_table_class()
        table = stim_table_cls(name="stimulation", description="biphasic pulses")
        for t in ONSETS:
            table.add_row(
                start_time=float(t),
                stop_time=float(t) + 0.0004,
                bipolar_pair="e0-e1",
                frequency=0.5,
                amplitude=0.004,
                pulse_width=0.0002,
            )
        nwb.add_acquisition(table)
    elif source == "icephys":
        from pynwb.icephys import CurrentClampStimulusSeries

        electrode = nwb.create_icephys_electrode(name="cell", description="patch", device=device)
        for k, t in enumerate(ONSETS[:4]):
            wave = np.zeros(int(SFREQ))
            wave[100:300] = 1e-10
            nwb.add_stimulus(
                CurrentClampStimulusSeries(
                    name=f"sweep{k}",
                    data=wave,
                    rate=SFREQ,
                    starting_time=float(t) - 0.1,
                    electrode=electrode,
                    gain=1.0,
                    sweep_number=k,
                )
            )
    return nwb


_STIM_TABLE = {}


def _stim_table_class() -> Any:
    """``StimTable`` of ndx-electrical-stim (dormant, not on PyPI): the same type, declared
    with the NWB extension builder in a folder of the test session and loaded into pynwb."""

    if _STIM_TABLE:
        return _STIM_TABLE["cls"]
    import tempfile

    from pynwb import get_class, load_namespaces
    from pynwb.spec import NWBDatasetSpec, NWBGroupSpec, NWBNamespaceBuilder

    folder = tempfile.mkdtemp(prefix="ndx-electrical-stim-")
    builder = NWBNamespaceBuilder(
        doc="electrical stimulation (test copy)",
        name="ndx-electrical-stim",
        version="0.1.0",
        author=["test"],
        contact=["test"],
    )
    builder.include_namespace("core")
    columns = [
        NWBDatasetSpec(name=n, neurodata_type_inc="VectorData", dtype=d, doc=n)
        for n, d in (
            ("bipolar_pair", "text"),
            ("frequency", "float64"),
            ("amplitude", "float64"),
            ("pulse_width", "float64"),
        )
    ]
    builder.add_spec(
        "ndx-electrical-stim.extensions.yaml",
        NWBGroupSpec(
            neurodata_type_def="StimTable",
            neurodata_type_inc="TimeIntervals",
            doc="stimulation table",
            datasets=columns,
        ),
    )
    builder.export("ndx-electrical-stim.namespace.yaml", outdir=folder)
    load_namespaces(f"{folder}/ndx-electrical-stim.namespace.yaml")
    _STIM_TABLE["cls"] = get_class("StimTable", "ndx-electrical-stim")
    return _STIM_TABLE["cls"]


def _write(tmp_path: Any, source: Any) -> Any:
    path = tmp_path / f"{source}.nwb"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nwbfile = _nwbfile(source)  # first: a source may load an extension namespace
        with pynwb.NWBHDF5IO(str(path), "w") as io:
            io.write(nwbfile)
    return path


@pytest.mark.parametrize(
    "source", ["events_table", "trials", "intervals", "stim_table", "icephys"]
)
def test_req317_event_sources(tmp_path: Any, source: Any) -> None:

    path = _write(tmp_path, source)
    ts = read_nwb(str(path), tmin=-0.1, tmax=0.2)
    assert ts.provenance.source["events"] == source
    expected = ONSETS[:4] if source == "icephys" else ONSETS
    assert np.allclose(ts.event_times, expected, atol=1 / SFREQ)
    assert ts.data.shape == (len(expected), N_CH, int(0.3 * SFREQ))
    # the response of channel 0 (-50 µV, 20 to 40 ms after each onset) in µV
    window = (ts.times >= 0.022) & (ts.times <= 0.038)
    assert np.allclose(ts.data[:, 0, window].mean(), -50, atol=2)
    assert list(ts.ch_info["location"][:2]) == ["VISp", "VISp"]


def test_req317_stimulation_units(tmp_path: Any) -> None:

    ts = read_nwb(str(_write(tmp_path, "events_table")), tmin=-0.1, tmax=0.2)
    assert np.allclose(ts.metadata["stim_current_mA"], 6.0)
    assert np.allclose(ts.metadata["stim_pulse_width_us"], 200.0)
    stim = read_nwb(str(_write(tmp_path, "stim_table")), tmin=-0.1, tmax=0.2)
    assert stim.provenance.source["events"] == "stim_table"
    assert np.allclose(stim.metadata["stim_current_mA"], 4.0)
    assert set(stim.metadata["stim_site"]) == {"e0-e1"}


def test_req317_equal_to_the_trialset_built_in_memory(tmp_path: Any) -> None:

    from evoparam.io import epoch

    ts = read_nwb(str(_write(tmp_path, "trials")), tmin=-0.1, tmax=0.2)
    memory = epoch(
        _signal().T * 1e6, SFREQ, ONSETS, -0.1, 0.2, ch_names=[f"e{i}" for i in range(N_CH)]
    )
    assert ts.ch_names == memory.ch_names
    assert ts.units == memory.units == "uV"
    np.testing.assert_allclose(ts.times, memory.times, atol=1e-12)
    np.testing.assert_allclose(ts.event_times, memory.event_times, atol=1e-12)
    np.testing.assert_allclose(ts.data, memory.data, rtol=1e-12, atol=1e-9)


def test_events_given_as_a_table_and_select(tmp_path: Any) -> None:

    import pandas as pd

    path = _write(tmp_path, "trials")
    kinds = ["a" if k % 2 == 0 else "b" for k in range(len(ONSETS))]
    table = pd.DataFrame({"start_time": ONSETS, "kind": kinds})
    ts = read_nwb(
        str(path), events=table, select="kind == 'a'", tmin=-0.05, tmax=0.1, picks=["e0", "e1"]
    )
    assert ts.n_trials == kinds.count("a")
    assert ts.ch_names == ["e0", "e1"]
    assert set(ts.metadata["kind"]) == {"a"}


def _results(path: Any) -> Any:
    from evoparam.existence import test_response
    from evoparam.models import fit
    from evoparam.trials import fit_trials

    ts = read_nwb(str(path), tmin=-0.5, tmax=0.6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = fit(ts, "rodent_ep")
        test = test_response(ts, n_permutations=50, random_state=0, by=())
        trial_fits = fit_trials(ts, group, by=(), random_state=0)
    return ts, group, test, trial_fits


def _inspect(path: Any) -> Any:
    inspector = pytest.importorskip("nwbinspector")
    messages = list(inspector.inspect_nwbfile(nwbfile_path=str(path)))
    # messages about what evoparam wrote (the test file itself has no subject)
    return [
        m.message
        for m in messages
        if m.importance.name in ("CRITICAL", "BEST_PRACTICE_VIOLATION")
        and "evoparam" in str(m.location)
    ]


@pytest.mark.parametrize("extension", [False, True])
def test_req318_write_and_inspect(tmp_path: Any, extension: Any) -> None:

    path = _write(tmp_path, "trials")
    ts, group, test, trial_fits = _results(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_nwb([group, test, trial_fits, ts], str(path), extension=extension)
    with pynwb.NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        module = nwb.processing["evoparam"]
        assert set(module.data_interfaces) == {
            "evoparam_components",
            "evoparam_existence",
            "evoparam_trials",
            "evoparam_model",
            "evoparam_results",
        }
        components = module["evoparam_components"]
        frame = components.to_dataframe()
        for column in (
            "electrode",
            "stim_site",
            "condition",
            "component",
            "shape",
            "status",
            "present",
            "latency",
            "latency_se",
            "amplitude",
            "amplitude_se",
            "polarity",
            "fwhm",
            "onset",
            "offset",
            "area",
            "r2",
            "n_trials",
            "ci_level",
        ):
            assert column in components.colnames, column
        assert set(frame["channel"]) == set(ts.ch_names)
        # electrode: a region into the electrodes table, rows matching the channel names
        assert components["electrode"].table is nwb.electrodes
        rows = [int(r) for r in components["electrode"].data[:]]
        assert [f"e{r}" for r in rows] == list(frame["channel"])
        first = group.fits[0].components[0]
        assert np.isclose(frame["amplitude"].iloc[0], first.amplitude * 1e-6)
        assert set(frame["present"]) <= {"true", "false"}
        existence = module["evoparam_existence"]
        assert {"electrode", "stim_site", "condition"} <= set(existence.colnames)
        trials = module["evoparam_trials"]
        assert trials["trial"].table is nwb.trials
        assert "electrode" in trials.colnames
        model = module["evoparam_model"]
        assert model.data.shape[1] == len(group.fits)
        peak = np.max(np.abs(model.data[:]))
        assert 1e-6 < peak < 1e-3  # volts
        assert '"settings"' in module.description
        if extension:
            assert type(components).__name__ == "EvokedResponseFitTable"
            assert type(existence).__name__ == "ResponseTestTable"
            assert type(trials).__name__ == "TrialFitTable"
            assert {"params", "param_names"} <= set(components.colnames)
            assert list(components["param_names"][0]) == list(first.param_names)
            assert np.allclose(components["params"][0], first.params)
            record = nwb.lab_meta_data["evoparam_settings"]
            assert record.preset_name == "rodent_ep"
            assert record.package_version
    assert _inspect(path) == []


@pytest.mark.parametrize("extension", [False, True])
def test_req024_write_nwb_rereads_and_passes_pynwb_validate(tmp_path: Any, extension: Any) -> None:
    """REQ-024: the output of ``write_nwb`` re-reads with pynwb and passes
    ``pynwb.validate`` (against the namespaces cached in the file, with the evoparam
    extension or without it)."""

    path = _write(tmp_path, "trials")
    ts, group, test, trial_fits = _results(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_nwb([group, test, trial_fits, ts], str(path), extension=extension)
    with pynwb.NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        assert "evoparam" in nwb.processing
        assert len(nwb.processing["evoparam"]["evoparam_components"].to_dataframe()) > 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        errors = pynwb.validate(path=str(path))
    errors = errors[0] if isinstance(errors, tuple) else errors  # pynwb < 3: (errors, status)
    assert list(errors) == []


def test_req318_read_back_equal_results(tmp_path: Any) -> None:

    path = _write(tmp_path, "trials")
    ts, group, test, trial_fits = _results(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_nwb([group, test, trial_fits], str(path))
    assert read_nwb_results(str(path)) == group
    everything = read_nwb_results(str(path), kind="all")
    assert everything["existence"] == test
    assert everything["trials"] == trial_fits.fits
    assert everything["components"].provenance == group.provenance


def test_req318_edited_table_is_detected(tmp_path: Any) -> None:

    from evoparam.errors import DataError

    path = _write(tmp_path, "trials")
    _, group, _, _ = _results(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_nwb(group, str(path))
    import h5py

    with h5py.File(str(path), "a") as handle:
        latency = handle["processing/evoparam/evoparam_components/latency"]
        values = latency[:]
        values[0] += 0.001
        latency[...] = values
    with pytest.raises(DataError, match="edited"):
        read_nwb_results(str(path))
    assert read_nwb_results(str(path), verify=False) == group


def test_ndx_spec_files_are_current(tmp_path: Any) -> None:

    from evoparam.io import ndx_evoparam

    built = ndx_evoparam.build_spec(str(tmp_path))
    for name in ("ndx-evoparam.namespace.yaml", "ndx-evoparam.extensions.yaml"):
        with (
            open(tmp_path / name, encoding="utf-8") as a,
            open(f"{ndx_evoparam.SPEC_DIR}/{name}", encoding="utf-8") as b,
        ):
            assert a.read() == b.read(), name
    assert built.endswith("ndx-evoparam.namespace.yaml")


def test_events_in_a_gap_of_the_timestamps_are_dropped(tmp_path: Any) -> None:

    from datetime import datetime, timezone

    from pynwb import NWBFile
    from pynwb.ecephys import ElectricalSeries

    nwb = NWBFile(
        session_description="gap",
        identifier="gap",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    device = nwb.create_device(name="probe")
    group = nwb.create_electrode_group(name="g", description="g", location="x", device=device)
    nwb.add_electrode(group=group, location="x")
    region = nwb.create_electrode_table_region([0], "e")
    stamps = np.r_[np.arange(0, 10, 0.001), np.arange(20, 30, 0.001)]
    data = np.sin(stamps)[:, None] * 1e-6
    nwb.add_acquisition(
        ElectricalSeries(name="lfp", data=data, electrodes=region, timestamps=stamps)
    )
    path = tmp_path / "gap.nwb"
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    with pytest.warns(UserWarning, match="gaps"):
        ts = read_nwb(str(path), events=np.array([5.0, 15.0, 25.0]), tmin=-0.1, tmax=0.1)
    assert np.allclose(ts.event_times, [5.0, 25.0])
    assert np.allclose(ts.data[1, 0, 100], np.sin(25.0) * 1e0, atol=1e-6)

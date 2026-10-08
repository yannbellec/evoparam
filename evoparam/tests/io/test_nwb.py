"""NWB reader and writer (REQ-317, REQ-318)."""

import warnings

import numpy as np
import pytest

pynwb = pytest.importorskip("pynwb")

from evoparam.io import read_nwb, write_nwb  # noqa: E402

SFREQ = 1000.0
N_CH = 4
DURATION = 30.0
ONSETS = np.arange(2.0, 28.0, 2.0)


def _signal():
    rng = np.random.default_rng(0)
    n = int(DURATION * SFREQ)
    data = rng.normal(0, 1e-6, (n, N_CH))
    for t in ONSETS:
        i = int(round(t * SFREQ))
        data[i + 20:i + 40, 0] += -50e-6
    return data


def _nwbfile(source):
    from datetime import datetime, timezone

    from pynwb import NWBFile
    from pynwb.ecephys import ElectricalSeries

    nwb = NWBFile(session_description="test", identifier=f"evoparam-{source}",
                  session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc))
    device = nwb.create_device(name="probe")
    group = nwb.create_electrode_group(name="shank0", description="shank", location="VISp",
                                       device=device)
    for i in range(N_CH):
        nwb.add_electrode(group=group, location="VISp" if i < 2 else "CA1", x=float(i),
                          y=0.0, z=0.0)
    region = nwb.create_electrode_table_region(list(range(N_CH)), "all")
    data = _signal()
    if source == "icephys":
        nwb.add_acquisition(ElectricalSeries(name="lfp", data=data, electrodes=region,
                                             rate=SFREQ, starting_time=0.0))
    else:
        nwb.add_acquisition(ElectricalSeries(name="lfp", data=data, electrodes=region,
                                             rate=SFREQ, starting_time=0.0))
    if source == "events_table":
        from pynwb.event import EventsTable

        table = EventsTable(name="stim_events", description="pulses")
        table.add_column("amplitude", "current (A)")
        table.add_column("pulse_width", "pulse width (s)")
        for t in ONSETS:
            table.add_row(timestamp=float(t), duration=0.001, amplitude=0.006,
                          pulse_width=0.0002)
        nwb.add_acquisition(table)
    elif source == "trials":
        nwb.add_trial_column("stim_onset", "stimulus onset")
        for t in ONSETS:
            nwb.add_trial(start_time=float(t) - 0.5, stop_time=float(t) + 1.0,
                          stim_onset=float(t))
    elif source == "intervals":
        from pynwb.epoch import TimeIntervals

        table = TimeIntervals(name="flashes", description="flashes")
        table.add_column("color", "flash colour")
        for t in ONSETS:
            table.add_row(start_time=float(t), stop_time=float(t) + 0.25, color=1.0)
        nwb.add_time_intervals(table)
    elif source == "icephys":
        from pynwb.icephys import CurrentClampStimulusSeries

        electrode = nwb.create_icephys_electrode(name="cell", description="patch",
                                                 device=device)
        for k, t in enumerate(ONSETS[:4]):
            wave = np.zeros(int(SFREQ))
            wave[100:300] = 1e-10
            nwb.add_stimulus(CurrentClampStimulusSeries(
                name=f"sweep{k}", data=wave, rate=SFREQ, starting_time=float(t) - 0.1,
                electrode=electrode, gain=1.0, sweep_number=k))
    return nwb


def _write(tmp_path, source):
    path = tmp_path / f"{source}.nwb"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pynwb.NWBHDF5IO(str(path), "w") as io:
            io.write(_nwbfile(source))
    return path


@pytest.mark.parametrize("source", ["events_table", "trials", "intervals", "icephys"])
def test_req317_event_sources(tmp_path, source):

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


def test_req317_stimulation_units(tmp_path):

    ts = read_nwb(str(_write(tmp_path, "events_table")), tmin=-0.1, tmax=0.2)
    assert np.allclose(ts.metadata["stim_current_mA"], 6.0)
    assert np.allclose(ts.metadata["stim_pulse_width_us"], 200.0)


def test_events_given_as_a_table_and_select(tmp_path):

    import pandas as pd

    path = _write(tmp_path, "trials")
    kinds = ["a" if k % 2 == 0 else "b" for k in range(len(ONSETS))]
    table = pd.DataFrame({"start_time": ONSETS, "kind": kinds})
    ts = read_nwb(str(path), events=table, select="kind == 'a'", tmin=-0.05, tmax=0.1,
                  picks=["e0", "e1"])
    assert ts.n_trials == kinds.count("a") and ts.ch_names == ["e0", "e1"]
    assert set(ts.metadata["kind"]) == {"a"}


def test_req318_write_and_inspect(tmp_path):

    from evoparam.existence import test_response
    from evoparam.models import fit

    path = _write(tmp_path, "trials")
    ts = read_nwb(str(path), tmin=-0.5, tmax=0.6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = fit(ts, "rodent_ep")
        test = test_response(ts, n_permutations=50, random_state=0, by=())
        write_nwb([group, test], str(path))
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        module = nwb.processing["evoparam"]
        components = module["evoparam_components"].to_dataframe()
        assert set(components["channel"]) == set(ts.ch_names)
        assert "evoparam_existence" in module.data_interfaces
        assert '"settings"' in module.description
    inspector = pytest.importorskip("nwbinspector")
    messages = list(inspector.inspect_nwbfile(nwbfile_path=str(path)))
    # messages about the tables evoparam wrote (the test file itself has no subject)
    bad = [m for m in messages if m.importance.name in ("CRITICAL", "BEST_PRACTICE_VIOLATION")
           and "evoparam" in str(m.location)]
    assert not bad, [m.message for m in bad]


def test_events_in_a_gap_of_the_timestamps_are_dropped(tmp_path):

    from datetime import datetime, timezone

    from pynwb import NWBFile
    from pynwb.ecephys import ElectricalSeries

    nwb = NWBFile(session_description="gap", identifier="gap",
                  session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc))
    device = nwb.create_device(name="probe")
    group = nwb.create_electrode_group(name="g", description="g", location="x", device=device)
    nwb.add_electrode(group=group, location="x")
    region = nwb.create_electrode_table_region([0], "e")
    stamps = np.r_[np.arange(0, 10, 0.001), np.arange(20, 30, 0.001)]
    data = np.sin(stamps)[:, None] * 1e-6
    nwb.add_acquisition(ElectricalSeries(name="lfp", data=data, electrodes=region,
                                         timestamps=stamps))
    path = tmp_path / "gap.nwb"
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    with pytest.warns(UserWarning, match="gaps"):
        ts = read_nwb(str(path), events=np.array([5.0, 15.0, 25.0]), tmin=-0.1, tmax=0.1)
    assert np.allclose(ts.event_times, [5.0, 25.0])
    assert np.allclose(ts.data[1, 0, 100], np.sin(25.0) * 1e0, atol=1e-6)

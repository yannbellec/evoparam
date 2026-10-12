# SPDX-License-Identifier: Apache-2.0
"""REQ-253: simulated stimulation artifacts and their removal (link of §5.3 and §5.12)."""

import numpy as np
import pytest

from evoparam import sim
from evoparam.artifact import paradigm_subtract, remove_artifact
from evoparam.errors import SettingsError
from evoparam.sim.artifacts import (
    dipole_gain,
    grid_layout,
    line_layout,
    pulse_shape,
    scalp_layout,
    seeg_layout,
    tail_shape,
)

pytest.importorskip("pandas")

WINDOW = (-0.002, 0.008)


def _ratio(ts, cleaned, valid=None):
    """Residual RMS in the artifact window relative to the RMS before removal."""

    win = (ts.times >= WINDOW[0]) & (ts.times <= WINDOW[1])
    raw, out = ts.data[..., win], cleaned.data[..., win]
    if valid is not None:
        raw, out = raw[valid[..., win]], out[valid[..., win]]
    return float(np.sqrt(np.mean(out**2) / np.mean(raw**2)))


###################################################################################################
## REQ-253


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_template_removal_leaves_at_most_ten_percent(seed):
    """Default simulate_ccep: residual RMS in the window <= 10 % of the RMS before removal,
    pooled over trials and channels (and also on the non-stimulated contacts alone)."""

    ts, truth = sim.simulate_ccep(seed=seed)
    res = remove_artifact(ts, method="template", window=WINDOW)
    assert _ratio(ts, res.cleaned) <= 0.10
    away = ~ts.ch_info["stim_contact"].to_numpy()
    win = (ts.times >= WINDOW[0]) & (ts.times <= WINDOW[1])
    raw = ts.data[:, away][..., win]
    out = res.cleaned.data[:, away][..., win]
    assert np.sqrt(np.mean(out**2) / np.mean(raw**2)) <= 0.10
    # What remains is mostly the response and the noise, not the artifact.
    clean_truth = (ts.data - truth.artifact_data)[..., win]
    left = res.cleaned.data[..., win] - clean_truth
    assert np.sqrt(np.mean(left**2) / np.mean(ts.data[..., win] ** 2)) <= 0.10


def test_template_removal_with_saturation():
    """Clipped samples are masked by the simulator and left out of the ratio."""

    ts, truth = sim.simulate_ccep(seed=0, saturation=True)
    assert ts.masks is not None and ts.masks.any()
    assert truth.artifact["n_clipped_samples"] == int(ts.masks.sum())
    assert np.max(np.abs(ts.data)) <= truth.artifact["rail"] + 1e-9
    res = remove_artifact(ts, method="template", window=WINDOW)
    assert _ratio(ts, res.cleaned, valid=~ts.masks) <= 0.10


def test_saturation_value_in_microvolts():
    ts, truth = sim.simulate_ccep(seed=3, saturation=500.0, n_trials=4, n_channels=10, tmax=0.2)
    assert truth.artifact["rail"] == 500.0
    assert np.max(np.abs(ts.data)) <= 500.0
    assert np.array_equal(ts.masks, np.abs(ts.data) >= 500.0)


###################################################################################################
## Artifact model


def test_ccep_artifact_geometry():
    """The artifact is largest at the stimulating contacts, has opposite signs there and
    decays with the distance to the site."""

    ts, truth = sim.simulate_ccep(seed=5, noise="none", n_trials=3, tmax=0.2)
    info = truth.artifact
    stim = [ts.ch_names.index(name) for name in info["stim_contacts"]]
    peak = np.asarray(info["peak_uV"])
    assert set(np.argsort(peak)[-2:]) == set(stim)
    first = truth.artifact_data[0]
    t_peak = np.argmax(np.abs(first[stim[0]]))
    assert np.sign(first[stim[0], t_peak]) == -np.sign(first[stim[1], t_peak])
    coords = np.column_stack([ts.ch_info[c].to_numpy() for c in ("x", "y", "z")])
    centre = coords[stim].mean(axis=0)
    distance = np.linalg.norm(coords - centre, axis=1)
    others = np.setdiff1d(np.arange(ts.n_channels), stim)
    near, far = others[np.argsort(distance[others])[:3]], others[np.argsort(distance[others])[-3:]]
    assert peak[near].mean() > 10 * peak[far].mean()
    assert info["filter"]["cutoff_hz"] == 0.25 * ts.sfreq
    np.testing.assert_allclose(
        info["trial_scale"],
        np.max(np.abs(truth.artifact_data), axis=(1, 2))
        / np.max(np.abs(truth.artifact_data[0]))
        * info["trial_scale"][0],
    )
    assert not np.any(truth.artifact_data[..., ts.times < -0.001])


def test_ccep_artifact_off_and_scaling():
    ts, truth = sim.simulate_ccep(
        seed=6, artifact=False, noise="none", n_trials=2, n_channels=8, tmax=0.2
    )
    assert truth.artifact is None and truth.artifact_data is None
    np.testing.assert_allclose(ts.data, truth.signal)
    _, one = sim.simulate_ccep(seed=6, stim_current_mA=1.0, n_trials=2, n_channels=8, tmax=0.2)
    _, four = sim.simulate_ccep(seed=6, stim_current_mA=4.0, n_trials=2, n_channels=8, tmax=0.2)
    np.testing.assert_allclose(four.artifact_data, 4.0 * one.artifact_data, rtol=1e-12)


def test_layouts_and_waveforms():
    names, coords, groups = seeg_layout(30)
    assert names[0] == "A01" and len(set(groups)) == 3 and coords.shape == (30, 3)
    assert np.isclose(np.linalg.norm(coords[1] - coords[0]), 3.5)
    names, coords, _ = grid_layout(16)
    assert names[0] == "G01" and np.isclose(np.linalg.norm(coords[1] - coords[0]), 10.0)
    names, coords, _ = line_layout(4, 0.04, "L")
    assert names == ["L00", "L01", "L02", "L03"]
    names, coords, _ = scalp_layout(40)
    assert names[:3] == ["Cz", "Pz", "Fz"] and len(set(names)) == 40
    assert np.all(np.linalg.norm(coords[:, :2], axis=1) <= 1.0 + 1e-9)

    contacts = np.array([[0, 0, 0], [3.5, 0, 0], [50, 0, 0]])
    gain = dipole_gain(contacts, contacts[0], contacts[1])
    assert np.isclose(gain[0], 1 - 1 / 3.5) and np.isclose(gain[1], -gain[0])
    assert abs(gain[2]) < 0.01 * gain[0]

    sfreq = 2048.0
    times = np.arange(-100, 400) / sfreq
    mono = pulse_shape(times, sfreq, pulse_width_us=200.0, kind="monophasic", polarity=-1)
    bi = pulse_shape(times, sfreq, pulse_width_us=200.0, kind="biphasic", polarity=-1)
    assert np.all(mono[times < -0.003] == 0) and np.all(bi[times < -0.003] == 0)
    # The anti-aliasing filter passes DC: a monophasic pulse integrates to its charge.
    assert np.isclose(mono.sum() / sfreq, -200e-6, rtol=0.02)
    assert abs(bi.sum()) / sfreq < 0.02 * 200e-6  # charge-balanced
    tail = tail_shape(times, sfreq, start=0.0, tau1=0.002, tau2=0.02)
    assert tail[0] == 0 and np.all(tail >= 0) and tail.max() <= 1.0
    with pytest.raises(SettingsError):
        pulse_shape(times, sfreq, pulse_width_us=200.0, kind="square")
    with pytest.raises(SettingsError):
        tail_shape(times, sfreq, start=0.0, tau1=-1.0, tau2=0.01)


###################################################################################################
## Paradigm-based cancellation (ECAP)


def test_ecap_alternating_cancels_the_artifact():
    ts, truth = sim.simulate_ecap(8, paradigm="alternating", noise="none", seed=0)
    assert set(ts.metadata["stim_polarity"]) == {
        "biphasic_cathodic_first",
        "biphasic_anodic_first",
    }
    out = paradigm_subtract(ts, scheme="alternating")
    assert out.n_trials == 4
    np.testing.assert_allclose(out.data, truth.signal[::2], atol=1e-9 * np.abs(ts.data).max())
    res = remove_artifact(ts, method="paradigm", scheme="alternating", quality=False)
    np.testing.assert_allclose(res.cleaned.data, out.data)


def test_ecap_forward_masking_recovers_the_probe():
    """A − B + (C − D) cancels the probe and masker artifacts, the switching artifact and
    the masker response, leaving the probe response."""

    ts, truth = sim.simulate_ecap(8, paradigm="forward_masking", noise="none", seed=0)
    assert list(ts.metadata["paradigm_condition"][:4]) == ["A", "B", "C", "D"]
    assert truth.conditions[:4] == ["A", "B", "C", "D"]
    assert {c["role"] for c in truth.components} == {"target", "masker"}
    out = paradigm_subtract(ts, scheme="forward_masking")
    assert out.n_trials == 2
    probe = truth.signal[0]
    np.testing.assert_allclose(out.data[0], probe, atol=1e-9 * np.abs(ts.data).max())
    assert np.abs(probe).max() > 0
    assert truth.artifact["switch_artifact_uV"] == 20.0


def test_ecap_kinds_and_errors():
    ts, truth = sim.simulate_ecap(4, kind="cochlear", seed=0, noise="none", artifact=False)
    assert ts.modality == "ecap"
    latencies = [c["features"]["latency"] for c in truth.components if c["label"] == "N1"]
    assert all(0 < latency < 0.0015 for latency in latencies)
    with pytest.raises(SettingsError):
        sim.simulate_ecap(4, kind="retinal")
    with pytest.raises(SettingsError):
        sim.simulate_ecap(4, paradigm="sandwich")


###################################################################################################
## TEP and ERNA artifacts


def test_tep_artifact_parts():
    ts, truth = sim.simulate_tep(seed=0, n_trials=4, n_channels=8, sfreq=2000.0, noise="none")
    info = truth.artifact
    assert info["model"] == "tms" and info["stim_site"] == "C3"
    for part in ("pulse", "decay", "muscle"):
        assert part in info and "trial_cv" in info[part]
    c3 = ts.ch_names.index("C3")
    peak = np.max(np.abs(truth.artifact_data[0]), axis=-1)
    assert np.argmax(peak) == c3
    _, none = sim.simulate_tep(
        seed=0,
        n_trials=4,
        n_channels=8,
        sfreq=2000.0,
        pulse_artifact=False,
        decay_artifact=False,
        muscle_artifact=False,
    )
    assert none.artifact is None and none.artifact_data is None
    roles = {c["label"]: c["role"] for c in truth.components}
    assert roles["AEP_N100"] == "confound" and roles["N100"] == "target"


def test_erna_artifact_is_a_pulse_train():
    ts, truth = sim.simulate_erna(n_pulses=4, n_trials=3, noise="none", seed=0)
    assert truth.artifact["n_pulses"] == 4
    stim = ts.ch_names.index(truth.artifact["stim_contact"])
    trace = np.abs(truth.artifact_data[0, stim])
    period = 1.0 / truth.artifact["pulse_rate_hz"]
    for pulse in range(4):
        near = np.abs(ts.times - pulse * period) < 0.0005
        assert trace[near].max() > 0.5 * trace.max()
    with pytest.raises(SettingsError):
        sim.simulate_erna(n_pulses=2, sfreq=2048.0)

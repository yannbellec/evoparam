# SPDX-License-Identifier: Apache-2.0
"""Specified exports and listings: TrialSet.to_xarray (spec §4.3.1), TFFit.to_mne (§5.10),
HierarchicalFit.to_arviz (§5.8), list_shapes and list_backgrounds (§5.5.8)."""

from typing import Any

import numpy as np
import pytest

from evoparam.data import TrialSet
from evoparam.errors import DataError

pd = pytest.importorskip("pandas")

###################################################################################################
###################################################################################################


def _trialset() -> Any:
    rng = np.random.default_rng(0)
    times = np.arange(-50, 150) / 1000.0
    data = rng.normal(0, 1, (4, 3, len(times)))
    masks = np.zeros(data.shape, dtype=bool)
    masks[:, :, 50:55] = True
    meta = pd.DataFrame(
        {"stim_site": ["A1-A2"] * 2 + ["B1-B2"] * 2, "stim_current_mA": [4.0, 4.0, 6.0, 6.0]}
    )
    info = pd.DataFrame({"name": ["a", "b", "c"], "bad": [False, True, False]})
    return TrialSet(
        data=data,
        times=times,
        sfreq=1000.0,
        ch_names=["a", "b", "c"],
        metadata=meta,
        ch_info=info,
        masks=masks,
    )


def test_list_shapes_and_backgrounds() -> None:

    from evoparam.models import list_backgrounds, list_shapes
    from evoparam.shapes import BACKGROUNDS, SHAPES

    shapes, backgrounds = list_shapes(), list_backgrounds()
    assert shapes == sorted(SHAPES)
    assert backgrounds == sorted(BACKGROUNDS)
    assert {"gaussian", "skewed_gaussian", "exgaussian", "gamma", "template"} <= set(shapes)
    assert {"none", "constant", "linear", "exponential", "pspline"} <= set(backgrounds)
    assert all(isinstance(name, str) for name in shapes + backgrounds)


def test_trialset_to_xarray() -> None:

    xr = pytest.importorskip("xarray")
    ts = _trialset()
    arr = ts.to_xarray()
    assert isinstance(arr, xr.DataArray)
    assert arr.dims == ("trial", "channel", "time")
    assert np.array_equal(arr.values, ts.data)
    assert arr["channel"].values.tolist() == ts.ch_names
    assert np.array_equal(arr["time"].values, ts.times)
    assert arr["stim_site"].dims == ("trial",)
    assert arr["stim_current_mA"].values.tolist() == [4.0, 4.0, 6.0, 6.0]
    assert arr["bad"].dims == ("channel",)
    assert np.array_equal(arr["mask"].values, ts.masks)
    assert arr.attrs["sfreq"] == 1000.0
    assert arr.attrs["units"] == ts.units
    # Selection by label works as in xarray
    assert np.array_equal(arr.sel(channel="b").values, ts.data[:, 1])


def test_trialset_to_xarray_without_xarray(monkeypatch: Any) -> None:

    import builtins

    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == "xarray":
            raise ImportError("no xarray")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match="evoparam\\[xarray\\]"):
        _trialset().to_xarray()


def test_tffit_to_mne() -> None:

    mne = pytest.importorskip("mne")
    from evoparam.tests.tf.helpers import TIMES, ar1, burst, trialset
    from evoparam.tf import tf_fit

    rng = np.random.default_rng(0)
    data = ar1(rng, (20, 2, len(TIMES)))
    data[:, 0] += burst(rng, 20, freq=80.0, amp=10.0)
    tf = tf_fit(trialset(data), freqs=(4, 250, 12), zones="none", random_state=0)
    out = tf.to_mne()
    assert isinstance(out, mne.time_frequency.AverageTFR)
    assert np.allclose(out.data, tf.maps["total"], equal_nan=True)
    assert np.allclose(out.freqs, tf.freqs)
    assert np.allclose(out.times, tf.times)
    assert out.ch_names == [row[0] for row in tf.settings["rows"]]
    assert out.nave == min(tf.settings["n_trials"])
    norm = tf.to_mne("total_norm")
    assert np.allclose(norm.data, tf.maps["total_norm"], equal_nan=True)
    with pytest.raises(DataError, match="unknown map"):
        tf.to_mne("nothing")


def test_hierarchical_to_arviz_eb() -> None:

    az = pytest.importorskip("arviz")
    from evoparam.hierarchical import fit_hierarchical
    from evoparam.tests.hierarchical.helpers import simulate_group

    sets, _ = simulate_group(
        n_subjects=3, n_channels=8, n_trials=15, p_respond=1.0, snr=3.0, seed=4
    )
    fit = fit_hierarchical(sets, components=["N1"])
    data = fit.to_arviz(n_draws=4000, random_state=0)
    posterior = data["posterior"] if not hasattr(data, "posterior") else data.posterior
    channel = fit.level_tables["channel"]
    channel = channel[channel["component"] == "N1"]
    draws = np.asarray(posterior["N1_latency_channel"])
    assert draws.shape == (1, 4000, len(channel))
    # The draws follow the normal approximation of the table: mean and SD
    finite = np.isfinite(channel["latency"].to_numpy(dtype=float))
    assert np.allclose(
        draws.mean(axis=(0, 1))[finite],
        channel["latency"].to_numpy(dtype=float)[finite],
        atol=4 * channel["latency_se"].to_numpy(dtype=float)[finite].max() / np.sqrt(4000) + 1e-12,
    )
    sites = np.asarray(posterior["site_N1"]).tolist()
    assert sites == [
        f"{a}/{b}" for a, b in zip(channel["subject"], channel["channel"], strict=False)
    ]
    assert "N1_amplitude_subject" in posterior
    assert "normal" in data.attrs["approximation"]
    # ArviZ summaries work on it
    assert az.summary(data) is not None


def test_hierarchical_to_arviz_uses_stored_draws() -> None:

    pytest.importorskip("arviz")
    from evoparam.data import HierarchicalFit

    rng = np.random.default_rng(1)
    draws = {
        "N1": {
            "sites": ["s1/a", "s1/b"],
            "subjects": ["s1"],
            "latency_channel": rng.normal(0.02, 1e-3, (2, 50, 2)),
            "amplitude_channel": rng.normal(-50, 5, (2, 50, 2)),
            "latency_subject": rng.normal(0.02, 1e-3, (2, 50, 1)),
            "amplitude_subject": rng.normal(-50, 5, (2, 50, 1)),
        }
    }
    fit = HierarchicalFit(
        levels=("channel", "subject"),
        summary=None,
        level_tables={},
        p_response=None,
        variance_components=None,
        effects=None,
        engine="numpyro",
        diagnostics={"draws": draws},
    )
    data = fit.to_arviz()
    posterior = data["posterior"] if not hasattr(data, "posterior") else data.posterior
    assert np.array_equal(
        np.asarray(posterior["N1_latency_channel"]), draws["N1"]["latency_channel"]
    )
    assert np.asarray(posterior["site_N1"]).tolist() == ["s1/a", "s1/b"]
    assert "draws" in data.attrs["approximation"]

# SPDX-License-Identifier: Apache-2.0
"""Shared shapes, shared-parameter fits and the volume-conduction screen on simulations
(REQ-190 to REQ-193)."""

import copy
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from evoparam.data import ShapeBasis, TrialSet
from evoparam.errors import DataError, SettingsError
from evoparam.models import ResponseModel
from evoparam.spatial import (
    adjusted_rand_index,
    as_template,
    convergent,
    divergent,
    fit_shared,
    neighbours,
    screen_volume_conduction,
    shared_shapes,
)

from .helpers import (
    BASELINE,
    SFREQ,
    SHAPES,
    TIMES,
    independent_fits,
    multi_site,
    shared_channels,
    volume_scene,
)

###################################################################################################
###################################################################################################


def _matched_correlations(basis: Any, shapes: Any = SHAPES) -> Any:
    """Best correlation of each true shape with a recovered curve (one curve per shape)."""

    from itertools import permutations

    win = (basis.times[0] - 1e-9 <= TIMES) & (basis.times[-1] + 1e-9 >= TIMES)
    corr = np.array(
        [[np.corrcoef(curve, shape[win])[0, 1] for shape in shapes] for curve in basis.curves]
    )
    best = max(
        permutations(range(len(basis.curves)), len(shapes)),
        key=lambda p: sum(corr[p[j], j] for j in range(len(shapes))),
    )
    return np.array([corr[best[j], j] for j in range(len(shapes))])


@pytest.mark.parametrize("method", ["nmf", "kpca"])
def test_req190_recovers_three_shapes(method: Any) -> None:

    found, correlations, agreement = [], [], []
    for seed in range(10):
        ts, which = multi_site(seed)
        basis = shared_shapes(ts, method=method, random_state=seed)
        found.append(basis.n_shapes == 3)
        if basis.n_shapes == 3:
            correlations += list(_matched_correlations(basis))
            agreement.append(adjusted_rand_index(basis.assignment["shape_id"], which))
    # n_shapes = 3 in >= 90 % of runs; recovered curves correlate >= 0.95 with the truth
    assert np.mean(found) >= 0.9
    assert np.min(correlations) >= 0.95
    assert np.median(agreement) == 1.0


@pytest.mark.parametrize("method", ["nmf", "kpca"])
def test_weights_are_the_clustering_loadings(method: Any) -> None:
    """Spec §4.3.4 and §5.9 (audit F56): ``weights`` (group x shape) are the non-negative
    loadings of the clustering, which assign each group to its shape; the group-level
    weight alpha_m of method 1 (mean projection on the shape) is ``assignment["weight"]``."""

    ts, which = multi_site(3)
    # site S19 keeps 2 trials: it is projected, not used to define the bases
    meta = ts.metadata
    keep = [
        i
        for i, site in enumerate(meta["stim_site"])
        if site != "S19-T19" or sum(meta["stim_site"][:i] == site) < 2
    ]
    ts = ts.select(trials=keep)
    basis = convergent(ts, "R01", method=method, random_state=0)
    table = basis.assignment
    w = basis.weights
    assert w.shape == (20, 3)
    assert np.isfinite(w).all()
    assert (w >= 0).all()
    used = table["used"].to_numpy(bool)
    assert not used[-1]
    assert used[:-1].all()
    ids = table["shape_id"].to_numpy()
    assert (np.argmax(w[used], axis=1) == ids[used]).all()
    assert np.allclose(table["nmf_loading"], w[np.arange(20), ids])
    if method == "nmf":
        # columns of H: unit-norm rows over the defining groups
        assert np.allclose(np.linalg.norm(w[used], axis=0), 1.0, atol=1e-6)
        # the projected group's loadings put it with the sites of its true shape
        assert np.argmax(w[-1]) == ids[-1]
    else:
        assert set(np.unique(w)) <= {0.0, 1.0}
        assert (w.sum(axis=1) == 1).all()
    # alpha_m: the mean projection of the group's trials on its shape, data units
    v = ts.data[np.asarray(ts.metadata["stim_site"] == "S00-T00"), 0]
    win = (ts.times >= basis.times[0] - 1e-9) & (ts.times <= basis.times[-1] + 1e-9)
    alpha = (v[:, win] @ basis.curves[ids[0]]).mean()
    assert np.isclose(table["weight"].iloc[0], alpha)


def test_shape_basis_contents_and_round_trip(tmp_path: Any) -> None:

    ts, which = multi_site(3)
    basis = convergent(ts, "R01", random_state=0)
    assert isinstance(basis, ShapeBasis)
    assert basis.curves.shape == (3, basis.times.size)
    assert np.allclose(np.linalg.norm(basis.curves, axis=1), 1.0)
    assert basis.weights.shape == (20, 3)
    assert basis.r2.shape == (20,)
    assert basis.stability.shape == (3,)
    assert (basis.stability >= 0.8).all()
    table = basis.assignment
    assert list(table["group"]) == list(dict.fromkeys(ts.metadata["stim_site"]))
    assert (table["shape_id"] >= 0).all()
    assert (table["r2"] > 0).all()
    assert table["r2"].median() > 0.3
    assert (table["weight"] > 0).all()
    assert table["used"].all()
    assert basis.settings["by"] == "stim_site"
    assert basis.settings["channel"] == "R01"
    assert basis.provenance.steps[-1]["function"] == "evoparam.spatial.shared_shapes"
    path = basis.save(tmp_path / "basis.evoparam")
    loaded = ShapeBasis.load(path)
    assert np.allclose(loaded.curves, basis.curves)
    assert list(loaded.assignment["shape_id"]) == list(table["shape_id"])
    assert copy.deepcopy(basis) == basis


def test_same_result_with_the_same_seed() -> None:

    ts, _ = multi_site(4)
    a = shared_shapes(ts, random_state=11)
    b = shared_shapes(ts, random_state=11)
    assert a.n_shapes == b.n_shapes
    assert np.allclose(a.curves, b.curves, atol=1e-10)


def test_divergent_groups_channels() -> None:

    # One site into 12 channels: two shapes, six channels each, plus the stimulated contacts
    rng = np.random.default_rng(0)
    names = [f"C{i:02d}" for i in range(14)]
    data = rng.normal(0, 20, (10, 14, len(TIMES)))
    for c in range(12):
        data[:, c] += rng.uniform(50, 100) * SHAPES[c % 2]
    info = pd.DataFrame({"name": names, "bad": [False] * 13 + [True]})
    ts = TrialSet(
        data=data,
        times=TIMES,
        sfreq=SFREQ,
        ch_names=names,
        ch_info=info,
        metadata=pd.DataFrame({"stim_site": ["C12-C99"] * 10}),
    )
    basis = divergent(ts, "C12-C99", random_state=0)
    assert basis.n_shapes == 2
    groups = list(basis.assignment["group"])
    assert "C12" not in groups
    assert "C13" not in groups
    ids = basis.assignment["shape_id"].to_numpy()
    assert adjusted_rand_index(ids, np.arange(12) % 2) == 1.0
    assert basis.settings["stim_site"] == "C12-C99"


def test_small_groups_are_projected_not_used() -> None:

    ts, which = multi_site(5)
    keep = np.ones(ts.n_trials, bool)
    keep[np.flatnonzero(ts.metadata["stim_site"] == "S00-T00")[2:]] = False  # 2 trials
    small = ts.select(trials=np.flatnonzero(keep))
    basis = shared_shapes(small, random_state=0)
    row = basis.assignment.set_index("group").loc["S00-T00"]
    assert not row["used"]
    assert row["n_trials"] == 2
    assert "S00-T00" not in basis.settings["defining_groups"]
    # Projected on the shape that explains it best: the one of its own true shape
    others = basis.assignment[basis.assignment["used"]]
    same = others[which[1:] == which[0]]["shape_id"]
    assert row["shape_id"] == same.mode()[0]


def test_recording_contact_sites_are_left_out() -> None:

    ts, _ = multi_site(6)
    sites = ts.metadata["stim_site"].replace({"S01-T01": "R01-T01"})
    ts = TrialSet(
        data=ts.data,
        times=ts.times,
        sfreq=ts.sfreq,
        ch_names=ts.ch_names,
        metadata=pd.DataFrame({"stim_site": sites}),
    )
    basis = shared_shapes(ts, random_state=0)
    assert "R01-T01" not in list(basis.assignment["group"])


def test_edge_cases() -> None:

    ts, _ = multi_site(0, n_sites=2)
    with pytest.raises(DataError):
        shared_shapes(ts)  # fewer than 3 groups
    ts, _ = multi_site(0)
    with pytest.raises(SettingsError):
        shared_shapes(ts, method="pca")
    with pytest.raises(SettingsError):
        shared_shapes(ts, by="channel")  # many sites: stim_site needed
    two = TrialSet(
        data=np.repeat(ts.data, 2, axis=1),
        times=ts.times,
        sfreq=ts.sfreq,
        ch_names=["a", "b"],
        metadata=ts.metadata,
    )
    with pytest.raises(SettingsError):
        shared_shapes(two)  # channel needed


def test_unstable_basis_returns_one_shape_with_a_warning() -> None:

    # Every site has its own random waveform: no shared structure beyond chance
    rng = np.random.default_rng(1)
    data, sites = [], []
    for s in range(12):
        wave = np.convolve(rng.normal(0, 1, len(TIMES)), np.hanning(60), "same") * 20
        for _ in range(8):
            data.append(wave + rng.normal(0, 30, len(TIMES)))
            sites.append(f"S{s}-T{s}")
    ts = TrialSet(
        data=np.array(data)[:, None],
        times=TIMES,
        sfreq=SFREQ,
        ch_names=["R1"],
        metadata=pd.DataFrame({"stim_site": sites}),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        basis = shared_shapes(ts, random_state=0, stability_min=0.99)
    if basis.n_shapes == 1:
        assert "unstable_basis" in basis.settings["warnings"]
        assert any("unstable_basis" in str(w.message) for w in caught)
    else:  # a stable decomposition was found instead
        assert all(
            row["stability"] >= 0.99 for row in basis.settings["selection"] if row["accepted"]
        )


def test_as_template_returns_a_template_shape() -> None:

    ts, _ = multi_site(2)
    basis = shared_shapes(ts, random_state=0)
    shape = as_template(basis, 0, max_shift=0.01)
    assert shape.name == "template"
    assert shape.param_names == ("shift",)
    values = shape.evaluate(basis.times, np.array([0.0]))
    assert np.max(np.abs(values)) == pytest.approx(1.0, abs=0.02)
    assert abs(values[0]) < 1e-12
    assert abs(values[-1]) < 1e-12
    with pytest.raises(SettingsError):
        as_template(basis, 5)


###################################################################################################
## Shared-parameter fits


def _closest_latency(fit: Any, latency: Any) -> Any:
    if fit.status != "ok" or not fit.components:
        return np.nan
    return min(fit.components, key=lambda c: abs(c.latency - latency)).latency


def test_req191_shared_fit_reduces_latency_rmse() -> None:

    ratios = []
    for seed in range(3):
        draw, _, _ = shared_channels(seed, snr=2.0)
        evoked = draw()
        independent = independent_fits(evoked)
        shared = fit_shared(evoked, preset="ccep", baseline=BASELINE, group_fit=independent)
        ind = np.array([_closest_latency(f, 0.03) - 0.03 for f in independent.fits])
        sha = np.array([_closest_latency(f, 0.03) - 0.03 for f in shared.fits])
        ok = np.isfinite(ind)  # channels with an independent estimate
        assert ok.sum() >= 5
        assert np.isfinite(sha).all()
        ratios.append(np.sqrt(np.mean(sha[ok] ** 2)) / np.sqrt(np.mean(ind[ok] ** 2)))
    # RMSE reduced by >= 30 % in every run (0.01 to 0.37 at the time of writing)
    assert max(ratios) <= 0.7


def test_fit_shared_result_and_flags() -> None:

    draw, clean, _ = shared_channels(0, n=10, snr=6.0)
    evoked = draw()
    group = fit_shared(evoked, preset="ccep", baseline=BASELINE)
    info = group.settings["fit_shared"]
    assert info["modal_k"] == 1
    assert info["shapes"] in (["gaussian"], ["skewed_gaussian"])
    assert len(group.fits) == 10
    assert all(f.status == "ok" for f in group.fits)
    latencies = [f.components[0].latency for f in group.fits]
    assert np.ptp(latencies) < 1e-6  # one shared latency
    assert abs(latencies[0] - 0.03) < 0.002
    for f, truth in zip(group.fits, clean, strict=False):
        amp = f.components[0].amplitude
        assert abs(amp - truth[np.argmax(np.abs(truth))]) < 0.2 * abs(amp) + 5
        assert f.gof.r2 > 0.3
        assert f.model.shape == f.residual.shape
    table = group.to_dataframe()
    assert len(table) == 10
    assert table["amplitude"].notna().all()
    assert group.provenance.steps[-1]["function"] == "evoparam.spatial.fit_shared"
    # A channel without response is fitted with the modal K and flagged
    evoked.data[3] = np.random.default_rng(1).normal(0, 5, evoked.data.shape[1])
    group = fit_shared(evoked, preset="ccep", baseline=BASELINE)
    flagged = group.settings["fit_shared"]["flagged"]
    assert evoked.ch_names[3] in flagged
    assert any(w.startswith("k_mismatch") for w in group.get(evoked.ch_names[3]).warnings)


def test_free_latency_recovers_offsets() -> None:

    offsets = np.tile(np.linspace(-0.003, 0.003, 10), 1)
    draw, _, _ = shared_channels(2, n=10, snr=8.0, offsets=offsets)
    group = fit_shared(
        draw(), preset="ccep", baseline=BASELINE, free_latency=True, max_shift=0.005
    )
    latencies = np.array([f.components[0].latency for f in group.fits])
    assert np.max(np.abs(latencies - 0.03 - offsets)) < 0.0015
    fwhm = [f.components[0].fwhm for f in group.fits]
    assert np.ptp(fwhm) < 1e-6  # width still shared


def test_req192_penalty_never_hurts_held_out_prediction() -> None:

    win = (TIMES >= 0.009) & (TIMES <= 0.5)
    rng = np.random.default_rng(0)
    for seed in range(2):
        smooth_field = seed % 2 == 0
        if smooth_field:  # latency drifting along each shaft
            offsets = np.concatenate([np.linspace(-0.004, 0.004, 10) * s for s in (1, -1, 1)])
        else:  # no spatial structure
            offsets = rng.uniform(-0.004, 0.004, 30)
        draw, _, _ = shared_channels(seed, snr=3.0, offsets=offsets)
        train, test = draw(), draw()
        independent = independent_fits(train)
        kw = {
            "preset": "ccep",
            "baseline": BASELINE,
            "group_fit": independent,
            "free_latency": True,
            "smooth": True,
        }
        chosen = fit_shared(train, **kw)
        off = fit_shared(train, smooth_lambda=0.0, **kw)

        def err(g):
            return sum(np.sum((test.data[i, win] - f.model) ** 2) for i, f in enumerate(g.fits))  # noqa: B023  (called within this iteration only)

        smooth = chosen.settings["fit_shared"]["smooth"]
        assert smooth["grid"][0] == 0.0
        assert len(smooth["loco_error"]) == len(smooth["grid"])
        assert min(smooth["loco_error"]) <= smooth["loco_error"][0]
        # Held-out error (fresh noise) with the chosen lambda <= 1.01 x the one of lambda = 0
        assert err(chosen) <= 1.01 * err(off)


def test_smooth_needs_free_latency_and_neighbours() -> None:

    draw, _, info = shared_channels(0, n=10, snr=6.0)
    evoked = draw()
    with pytest.raises(SettingsError):
        fit_shared(evoked, preset="ccep", baseline=BASELINE, smooth=True)
    bare = type(evoked)(
        data=evoked.data, times=evoked.times, sfreq=evoked.sfreq, ch_names=evoked.ch_names
    )
    with pytest.warns(UserWarning, match="no neighbouring"):
        group = fit_shared(bare, preset="ccep", baseline=BASELINE, free_latency=True, smooth=True)
    assert group.settings["fit_shared"]["smooth"]["lambda"] == 0.0
    weights = neighbours(info, list(info["name"]), radius_mm=10.0)
    # Contacts 3.5 mm apart on a shaft: two neighbours on each side, none across shafts
    assert weights.sum(axis=1).max() == 4
    assert weights[0, 1] == 1
    assert weights[0, 3] == 0


###################################################################################################
## Volume conduction


def test_req193_volume_conduction_screen() -> None:

    high, low = [], []
    for seed in range(5):
        ts, volume, local = volume_scene(seed)
        table = screen_volume_conduction(ts, window=(0.009, 0.5), baseline=BASELINE)
        table = table.set_index("channel")
        high += list(table.loc[volume, "volume_conduction_risk"] == "high")
        low += list(table.loc[local, "volume_conduction_risk"] == "low")
    assert len(high) >= 50
    assert len(low) == 20
    assert np.mean(high) >= 0.9
    assert np.mean(low) >= 0.9


def test_volume_screen_columns_and_markers() -> None:

    ts, volume, local = volume_scene(0)
    group = ResponseModel("ccep", baseline=BASELINE).fit_group(
        ts.select(picks=volume[:3] + local[:1])
    )
    table = screen_volume_conduction(ts, group, decay_model="none")
    assert set(table["volume_conduction_risk"]) <= {"", "low", "medium", "high"}
    rows = table.set_index("channel")
    assert not rows.loc["A04", "evaluated"]
    assert rows.loc["A04", "volume_conduction_risk"] == ""
    assert rows.loc[volume[0], "response"]
    assert not rows.loc["F10", "response"]
    assert rows.loc[local[0], "reversal"]
    assert rows.loc[volume[0], "screens"].count(",") + 1 == rows.loc[volume[0], "n_screens"]
    with pytest.raises(SettingsError):
        screen_volume_conduction(ts, decay_model="exponential")
    with pytest.raises(SettingsError):
        screen_volume_conduction(ts, reference_schemes=("unknown",))

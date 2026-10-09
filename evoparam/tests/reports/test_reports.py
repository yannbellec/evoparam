"""Reports: methods text (REQ-338), JSON with its schema (REQ-341), quality summary, HTML,
Markdown, PDF (REQ-343), citation (REQ-346). The frozen methods texts of the reference
pipelines (REQ-340) are in test_reports_text.py."""

import json
import os
import re

import numpy as np
import pandas as pd
import pytest

from evoparam import reports


def test_methods_text_follows_the_processing_steps(results):

    from evoparam.artifact import remove_artifact
    from evoparam.models import fit
    from evoparam.prep import baseline_correct

    cleaned = remove_artifact(results["ts"], method="interpolate", window=(-0.002, 0.008)).cleaned
    cleaned = baseline_correct(cleaned, (-0.5, -0.05), check=False)
    group = fit(cleaned, "ccep", min_fwhm=0.004)
    text = reports.methods_text(group)
    assert text.index("stimulation artifact was removed") < text.index("baseline-corrected") \
        < text.index("each waveform was modelled")
    assert "method 'interpolate'" in text and "-2 to 8 ms" in text
    assert "preset 'ccep' v1.0" in text and "4 to 250 ms" in text
    assert "Settings changed from the preset (times in s): min_fwhm = 0.004" in text
    assert reports.citation() in text
    # deterministic, and the same from the saved result
    assert reports.methods_text(group) == text


def test_methods_text_of_other_results(results):

    assert "presence of a response" in reports.methods_text(results["test"])
    assert "Time-frequency power" in reports.methods_text(results["tf"])


def test_req341_json_round_trip_and_schema(results, tmp_path):

    group = results["group"]
    path = reports.to_json(group, tmp_path / "group.json", validate=True)
    text = open(path, encoding="utf-8").read()
    json.loads(text, parse_constant=lambda name: (_ for _ in ()).throw(ValueError(name)))
    assert reports.validate_json(path)
    back = reports.from_json(path)
    pd.testing.assert_frame_equal(back.to_dataframe(), group.to_dataframe())
    for name in ("tf", "test"):
        obj = results[name]
        again = reports.from_json(reports.to_json(obj))
        assert type(again) is type(obj)
    tf_back = reports.from_json(reports.to_json(results["tf"]))
    np.testing.assert_array_equal(tf_back.significant, results["tf"].significant)


def test_invalid_json_is_named():

    from evoparam.errors import DataError

    with pytest.raises(DataError, match="at object"):
        reports.validate_json({"schema": "evoparam-result/1", "object": "Nope",
                               "package_version": "x", "structure": {}, "arrays": {},
                               "tables": {}, "provenance": None, "settings": {}})


def test_quality_summary(results):

    table = reports.qc(results["group"], trials=results["artifact"].cleaned,
                       artifact=results["artifact"], trial_fits=results["trials"])
    quantities = set(table["quantity"])
    assert {"ill-conditioned fits", "masked fraction", "artifact quality, median",
            "trials not trusted"} <= quantities
    assert any(q.startswith("fits: ") for q in quantities)


def test_req343_html_is_self_contained(results, tmp_path):

    path = reports.html(results["group"], tmp_path / "report.html", data=results["ts"])
    text = open(path, encoding="utf-8").read()
    assert "data:image/png;base64," in text
    assert not re.search(r"(src|href)=['\"]https?://", text)
    assert "For research use only" in text and "Methods" in text


def test_req343_html_size_for_100_channels(tmp_path):

    import warnings

    from evoparam.models import fit
    from evoparam.tests.trials.helpers import simulate

    ts, _, _ = simulate(n=10, channels=100, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = fit(ts, "ccep", n_jobs=2)
    path = reports.html(group, tmp_path / "big.html", data=ts, max_channels=100)
    assert (tmp_path / "big.html").stat().st_size <= 20e6


def test_markdown_and_pdf(results, tmp_path):

    text = reports.markdown(results["group"])
    assert text.startswith("# evoparam report") and "## Methods" in text
    path = reports.pdf(results["group"], tmp_path / "report.pdf", data=results["ts"])
    assert open(path, "rb").read(4) == b"%PDF"


def _pages(path):
    import re

    return len(re.findall(rb"/Type\s*/Page\b", open(path, "rb").read()))


def test_pdf_has_one_page_per_stimulation_site(tmp_path):
    """Spec §5.13: report.pdf has one page per stimulation site (after the text page)."""

    import dataclasses
    import warnings

    from evoparam import sim
    from evoparam.existence import test_response
    from evoparam.models import fit

    ts, _ = sim.simulate_ccep(n_trials=18, n_channels=4, sfreq=1024.0, seed=2)
    sites = ["A01-A02", "B01-B02", "C01-C02"]
    meta = ts.metadata.copy()
    meta["stim_site"] = [sites[i % 3] for i in range(ts.n_trials)]
    ts = dataclasses.replace(ts, metadata=meta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = fit(ts, "ccep", by="stim_site")
        pages = reports._documents.site_pages(group, ts)
        assert [site for site, _ in pages] == sites
        assert all(len(members) == 4 for _, members in pages)
        path = reports.pdf(group, tmp_path / "sites.pdf", data=ts, max_channels=0)
        assert _pages(path) == 1 + len(sites)
        # a test split by site (the default of test_response): one page per site too
        test = test_response(ts, n_permutations=50, random_state=0)
        assert test.settings["groups"] == sites
        path = reports.pdf(test, tmp_path / "test.pdf")
        assert _pages(path) == 1 + len(sites)
        # the fits of one site, fitted on all its trials: one page named after the site
        one = ts.select(trials=[i for i in range(ts.n_trials) if i % 3 == 0])
        group = fit(one, "ccep")
        assert [site for site, _ in reports._documents.site_pages(group, one)] == ["A01-A02"]


def test_req346_citation_from_citation_cff():

    assert reports.citation().startswith("Bellec Y.")
    bib = reports.citation(bibtex=True)
    assert bib.startswith("@software{bellec_evoparam,") and "version = {" in bib


def test_methods_text_roundtrip(ccep_run, tmp_path):
    """REQ-338: the paragraph comes from provenance and settings only, so the saved result
    gives the same one, with the version, preset, fit window, artifact method, existence
    method and the reproducibility sentence."""

    from evoparam import __version__, load_object

    group = ccep_run.group
    assert load_object(group.save(str(tmp_path / "group"))).methods_text() == \
        group.methods_text()
    fit = group.fits[0]
    assert load_object(fit.save(str(tmp_path / "fit"))).methods_text() == fit.methods_text()
    text = group.methods_text()
    assert text == reports.methods_text(group)
    assert f"evoparam (version {__version__})" in text
    assert "preset 'ccep' v1.0" in text
    assert "In the window 9 to 500 ms" in text                       # fit window
    assert "method 'template'" in text                               # artifact method
    assert "sign-flip permutation test" in text                      # existence method
    assert "Random draws used the seed 0;" in text                   # reproducibility
    assert "evoparam.reproduce" in text


def test_methods_text_reads_only_provenance_and_settings(ccep_run):

    group = ccep_run.group

    class Record:
        provenance = group.provenance
        settings = group.settings

    assert reports.methods_text(Record()) == group.methods_text()


def test_pipeline_methods_text_merges_every_result(ccep_run):

    text = reports.methods_text(ccep_run)
    assert "Single-trial parameters were estimated" in text           # trial fits
    assert text.count("The stimulation artifact was removed") == 1
    assert reports.methods_text(ccep_run.results) == text
    assert "Software:" not in reports.methods_text(ccep_run, cite=False)


def test_methods_text_without_seed_says_so(ccep_run):

    from evoparam.data import Provenance

    group = ccep_run.group
    steps = [dict(s, settings={k: v for k, v in (s.get("settings") or {}).items()
                               if k != "random_state"}) for s in group.provenance.steps]

    class Record:
        provenance = Provenance(steps=steps, random_state=None)
        settings = {k: v for k, v in group.settings.items() if k != "random_state"}

    text = reports.methods_text(Record())
    assert "No random seed was set" in text and "Random draws used the seed" not in text


def test_req346_citation_single_source():
    """evoparam/CITATION.cff (the packaged copy) is the root CITATION.cff, byte for byte."""

    from evoparam.data._citation import parse_cff, read_citation

    package = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(package, "CITATION.cff"), "rb") as a, \
            open(os.path.join(os.path.dirname(package), "CITATION.cff"), "rb") as b:
        assert a.read() == b.read(), "evoparam/CITATION.cff must be a copy of CITATION.cff"
    cff = read_citation()
    with open(os.path.join(os.path.dirname(package), "CITATION.cff"), encoding="utf-8") as f:
        assert parse_cff(f.read()) == cff
    text = reports.citation()
    assert cff["title"] in text and f"version {cff['version']}" in text
    assert all(author["family-names"] in text for author in cff["authors"])


def _recorded_steps():
    """Every function name the package records as a provenance step."""

    import pathlib

    import evoparam

    root = pathlib.Path(evoparam.__file__).parent
    pattern = re.compile(r"(?:with_step|new|finish|_new)\(\s*(?:[\w.\[\]]+,\s*){0,3}"
                         r"[\"'](evoparam\.[\w.]+)[\"']")
    names = set()
    for path in root.rglob("*.py"):
        if "tests" not in path.relative_to(root).parts:
            names |= set(pattern.findall(path.read_text("utf-8")))
    # the filters record f"evoparam.prep.{name}"
    return names | {"evoparam.prep.highpass", "evoparam.prep.lowpass", "evoparam.prep.notch"}


def test_methods_text_has_a_template_for_every_recorded_step():
    """Spec §5.13: one paragraph per processing step (events, artifact, preparation, fit,
    existence, trials, hierarchical, tf, stats); a step without a template would only be
    named, so every recorded step has one, or brings data in without changing a number."""

    from evoparam.data._methods import _TEMPLATES, SILENT_PREFIXES

    names = _recorded_steps()
    assert {"evoparam.events.detect_pulses", "evoparam.stats.reliability",
            "evoparam.prep.reference_sensitivity", "evoparam.prep.train_baseline"} <= names
    missing = sorted(n for n in names if n not in _TEMPLATES
                     and not n.startswith(SILENT_PREFIXES))
    assert not missing, missing


def test_methods_text_events_and_stats_paragraphs(results):

    from evoparam import stats
    from evoparam.data._methods import step_sentence
    from evoparam.events import detect_pulses
    from evoparam.io import epoch

    # events: the detection step travels from the event table into the trials
    rng = np.random.default_rng(0)
    sfreq = 2048.0
    raw = rng.normal(0, 1, (4, int(sfreq * 20)))
    for onset in np.arange(2.0, 18.0, 2.0):
        i = int(onset * sfreq)
        raw[0, i:i + 4] += 800
        raw[1, i:i + 4] -= 800
    events = detect_pulses(raw, sfreq, ch_names=["A1", "A2", "A3", "A4"], expected_rate=0.5)
    trials = epoch(raw, sfreq, events, -0.5, 1.0, ch_names=["A1", "A2", "A3", "A4"])
    assert [s["function"] for s in trials.provenance.steps] == [
        "evoparam.events.detect_pulses", "evoparam.io.epoch"]
    text = reports.methods_text(trials)
    assert "Stimulation pulses were detected from their artifact" in text
    assert "exceeded 20" in text and "expected rate of 0.5 Hz" in text
    assert "grouped into trains" in text
    # stats: a table of evoparam.stats carries its step
    table = stats.bootstrap(results["ts"], "peak_amplitude", n_boot=50, by=(),
                            window=(0.01, 0.1), random_state=0)
    text = reports.methods_text(table)
    assert "were obtained by bootstrap over trials (50 resamples" in text
    # and the other statistics have their own sentence (no generic "processed with")
    steps = {
        "evoparam.stats.compare_conditions": {"parameter": "latency", "condition": "drug",
                                              "unit": "subject", "paired": True,
                                              "n_permutations": 5000, "correction": "fdr_bh",
                                              "conditions": ["off", "on"], "ci_level": 0.95,
                                              "n_boot": 2000},
        "evoparam.stats.reliability": {"scorer": "mean_amplitude", "facets": ["trial",
                                                                             "subject"],
                                       "method": "both", "n_splits": 1000,
                                       "target_phi": 0.7, "window": [0.11, 0.15]},
        "evoparam.stats.trials_needed": {"scorer": "existence", "n_resamples": 200},
        "evoparam.stats.jackknife_latency": {"scorer": "fractional_area_latency",
                                             "retrieve": True, "n_subjects": 12},
        "evoparam.stats.group_summary": {"by": ["region"], "n_boot": 1000, "ci_level": 0.95},
        "evoparam.prep.notch": {"freq": 50.0, "q": 30.0, "harmonics": 3,
                                "frequencies": [50.0, 100.0, 150.0]},
        "evoparam.prep.highpass": {"cutoff": 0.5, "order": 4},
        "evoparam.prep.train_baseline": {"length": 0.05, "mode": "mean", "source": "epochs"},
        "evoparam.prep.reference_sensitivity": {"schemes": ["bipolar", "car",
                                                            "car_adaptive"]},
    }
    for function, settings in steps.items():
        sentence = step_sentence({"function": function, "settings": settings})
        assert "processed with" not in sentence and sentence.endswith("."), function
    assert "Benjamini-Hochberg" in step_sentence(
        {"function": "evoparam.stats.compare_conditions",
         "settings": steps["evoparam.stats.compare_conditions"]})
    assert "50, 100 and 150 Hz" in step_sentence(
        {"function": "evoparam.prep.notch", "settings": steps["evoparam.prep.notch"]})
    assert "last 50 ms before the train" in step_sentence(
        {"function": "evoparam.prep.train_baseline",
         "settings": steps["evoparam.prep.train_baseline"]})

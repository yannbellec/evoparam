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

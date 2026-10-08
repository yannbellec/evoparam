"""Reports: methods text, JSON with its schema, quality summary, HTML, Markdown, PDF
(REQ-340, REQ-341, REQ-343, REQ-346)."""

import json
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
        < text.index("parameterized")
    assert "'interpolate'" in text and "-2 to 8 ms" in text
    assert "preset 'ccep' v1.0" in text and "4 to 250 ms" in text
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
        group = fit(ts, "ccep", n_jobs=-1)
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

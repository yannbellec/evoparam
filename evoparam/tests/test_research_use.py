"""REQ-015: the package documentation, the CLI ``--version`` output, the HTML and PDF reports
and the stream outputs carry the statement "research use only; not a medical device"
(§13.3)."""

import contextlib
import importlib.util
import io
import pathlib
import re

import pytest

import evoparam

ROOT = pathlib.Path(evoparam.__file__).resolve().parent.parent


def carries_statement(text):
    text = re.sub(r"\s+", " ", text).lower()
    return "research use only" in text and "not a medical device" in text


def test_req015_documentation():
    assert carries_statement((ROOT / "README.rst").read_text(encoding="utf-8"))
    assert carries_statement(evoparam.__doc__)


def test_req015_cli_version():
    from evoparam.cli import main

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(["--version"])
    assert code == 0
    assert out.getvalue().startswith(f"evoparam {evoparam.__version__}")
    assert carries_statement(out.getvalue())


@pytest.fixture(scope="module")
def group():
    from evoparam import sim

    trials, _ = sim.simulate_ccep(n_trials=8, n_channels=3, seed=2)
    return evoparam.fit(trials, preset="ccep", random_state=0)


def test_req015_html_and_markdown_reports(group, tmp_path):
    pytest.importorskip("pandas")
    from evoparam import reports

    path = reports.html(group, tmp_path / "report.html")
    assert carries_statement(pathlib.Path(path).read_text(encoding="utf-8"))
    assert carries_statement(reports.markdown(group))


def test_req015_pdf_report(group, tmp_path):
    pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    from evoparam import reports

    # uncompressed content and TrueType text, so that the words can be read from the file
    with matplotlib.rc_context({"pdf.compression": 0, "pdf.fonttype": 42}):
        reports.pdf(group, tmp_path / "report.pdf")
    raw = (tmp_path / "report.pdf").read_bytes().decode("latin-1")
    shown = " ".join(re.findall(r"\(((?:[^()\\]|\\.)*)\)\s*Tj", raw)) + " ".join(
        "".join(re.findall(r"\(((?:[^()\\]|\\.)*)\)", block))
        for block in re.findall(r"\[(.*?)\]\s*TJ", raw, re.DOTALL))
    shown = shown.replace("\x00", "")              # TrueType text is UTF-16 in the file
    assert carries_statement(shown) or carries_statement(raw), \
        "the statement is not in the PDF text"


def test_req015_stream_outputs():
    if importlib.util.find_spec("evoparam.stream") is None:
        pytest.xfail("evoparam.stream (§10.6) is built on branch s18-en-ligne; its outputs "
                     "must carry the statement once it is merged.")
    import evoparam.stream as stream

    text = " ".join(str(getattr(stream, name, "")) for name in ("RESEARCH_USE", "__doc__"))
    assert carries_statement(text)

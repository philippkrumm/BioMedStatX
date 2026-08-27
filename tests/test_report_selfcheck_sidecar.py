"""The export-path self-check: silent when the report is sound, loud when it is not.

Two properties matter more than the checks themselves, because both are ways a
diagnostic quietly becomes worthless:

* a sidecar beside every export is noise, so a clean report must leave none;
* a diagnostic that can break the thing it diagnoses is worse than none, so a
  check that raises must cost the export nothing.

The third property is what the user reads: the file carries flags and counts,
and no values from the data. It describes properties of the report; it is not a
second export of what the report contains.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

from export.report_selfcheck import (CHECK_SUBJECTS, REPORT_CHECKS,  # noqa: E402
                                     REQUIRED_SECTIONS, SELFCHECK_ENV_VAR,
                                     SIDECAR_SUFFIX, run_report_checks,
                                     write_sidecar)


@pytest.fixture(autouse=True)
def _selfcheck_on(monkeypatch):
    """The self-check is off by default; these tests are about what it does when on.

    Made explicit rather than left to the ambient environment, because a suite
    that only passes on a developer's machine with the variable already exported
    is a suite that proves nothing about anyone else's.
    """
    monkeypatch.setenv(SELFCHECK_ENV_VAR, "1")

# A sound report has every section. Leaving them out would make the "clean"
# fixture fail sections_present, and the test would then be asserting that a
# broken report writes no sidecar -- the opposite of the point.
_SECTIONS = "".join(f'<section id="{name}"></section>' for name in REQUIRED_SECTIONS)


def _report(tmp_path, body: str = "", name: str = "report.html"):
    path = tmp_path / name
    path.write_text("<html><body>" + _SECTIONS + body + "</body></html>",
                    encoding="utf-8")
    return str(path)


def test_every_check_has_a_data_free_subject_line():
    """The sidecar prints these; a missing one would print an empty column."""
    for name, _ in REPORT_CHECKS:
        assert CHECK_SUBJECTS.get(name), f"{name} has no subject line"


def test_a_check_that_cannot_apply_is_reported_as_such(tmp_path):
    """"n-a" and "pass" must stay distinguishable.

    Collapsing them is how a run reports coverage it does not have: a report
    with no figure builder in it would otherwise come out as a row of passes on
    checks that never looked at anything.
    """
    outcome = dict((name, verdict) for name, verdict, _ in
                   run_report_checks(_report(tmp_path), {}))

    assert outcome["payloads_parse"] == "n-a", outcome
    assert outcome["sections_present"] == "pass", outcome


def test_a_report_with_nothing_to_check_writes_no_sidecar(tmp_path):
    """Silence is the normal case, and it has to stay the normal case."""
    path = _report(tmp_path)

    assert write_sidecar(path, {}) is None
    assert os.listdir(tmp_path) == ["report.html"]


def test_an_unparseable_payload_produces_a_sidecar(tmp_path):
    """A payload that is not JSON is the cheapest real failure to construct."""
    path = _report(tmp_path,
                   '<script id="pd-data-plot">{not json}</script>'
                   '<script id="pd-data-order">["A","B"]</script>')

    sidecar = write_sidecar(path, {})

    assert sidecar and sidecar.endswith(SIDECAR_SUFFIX)
    assert os.path.exists(sidecar)


def test_the_sidecar_carries_flags_and_counts_but_no_data(tmp_path):
    path = _report(tmp_path,
                   '<script id="pd-data-plot">{not json}</script>'
                   '<script id="pd-data-order">["Vehicle","Dose10"]</script>')

    sidecar = write_sidecar(path, {"p_value": 0.0031, "test": "Welch ANOVA",
                                   "raw_data": {"Vehicle": [1.0], "Dose10": [2.0]}})
    text = open(sidecar, encoding="utf-8").read()

    # Every check named, with a verdict.
    for name, _ in REPORT_CHECKS:
        assert name in text
    assert "fail" in text

    # And nothing from the data: not a group label, not a p-value, not a test
    # name. The report is next to it for that.
    for leaked in ("Vehicle", "Dose10", "0.0031", "Welch"):
        assert leaked not in text, f"{leaked!r} leaked into the sidecar"


def test_the_sidecar_says_the_report_is_unaffected(tmp_path):
    """The file is read by someone who just found it; it has to say what it is."""
    path = _report(tmp_path, '<script id="pd-data-plot">{not json}</script>')

    text = open(write_sidecar(path, {}), encoding="utf-8").read()

    assert "informational" in text.lower()
    assert "report.html" in text


def test_a_raising_check_costs_the_export_nothing(tmp_path, monkeypatch):
    """A check that blows up must be recorded, not propagated."""
    import export.report_selfcheck as selfcheck

    def _explode(report, result, violations):
        raise RuntimeError("check is broken")

    monkeypatch.setattr(selfcheck, "REPORT_CHECKS", (("exploding_check", _explode),))
    monkeypatch.setattr(selfcheck, "CHECK_SUBJECTS", {"exploding_check": "a broken check"})

    path = _report(tmp_path)
    sidecar = write_sidecar(path, {})

    assert sidecar is not None
    assert "error" in open(sidecar, encoding="utf-8").read()


def test_an_unreadable_report_is_not_an_export_failure(tmp_path):
    """The sidecar runs after a successful write; it may never undo that."""
    assert write_sidecar(str(tmp_path / "does_not_exist.html"), {}) is None


@pytest.fixture
def _real_exporter(monkeypatch):
    from export.html_exporter import HTMLExporter

    monkeypatch.setattr(HTMLExporter, "_render_template",
                        staticmethod(lambda context, mode: "<html></html>"))
    monkeypatch.setattr(HTMLExporter, "_prepare_single_report_context",
                        staticmethod(lambda results, analysis_log=None: {}))
    return HTMLExporter


def test_the_export_path_runs_the_self_check_when_it_is_on(tmp_path, _real_exporter):
    """The hook is in the export path, not only in the checker.

    Verified through ``export_results_to_html`` with the real ``write_sidecar``
    rather than a stand-in, because the thing worth pinning is that an export
    end to end produces the file -- and that it still returns the report path,
    so nothing upstream changes behaviour.
    """
    out = str(tmp_path / "on.html")
    returned = _real_exporter.export_results_to_html({"test": "t"}, out)

    assert returned == os.path.abspath(out)
    assert sorted(os.listdir(tmp_path)) == ["on.html", "on_selfcheck.txt"]


def test_the_export_path_leaves_no_trace_when_the_self_check_is_off(
        tmp_path, monkeypatch, _real_exporter):
    """The default, and the whole point of the gate.

    Same report as the test above -- one that does fail a check -- so the only
    difference between the two is the variable. An installed copy gets this
    case: no sidecar, and the export otherwise identical.
    """
    monkeypatch.delenv(SELFCHECK_ENV_VAR, raising=False)

    out = str(tmp_path / "off.html")
    returned = _real_exporter.export_results_to_html({"test": "t"}, out)

    assert returned == os.path.abspath(out)
    assert os.listdir(tmp_path) == ["off.html"]


def test_an_unset_variable_is_off_and_an_empty_one_too(tmp_path, monkeypatch):
    """"Set but empty" is how a shell profile half-configures something."""
    path = _report(tmp_path, '<script id="pd-data-plot">{not json}</script>')

    for value in ("", "0", "no", "  "):
        monkeypatch.setenv(SELFCHECK_ENV_VAR, value)
        assert write_sidecar(path, {}) is None, f"{value!r} enabled the self-check"

    monkeypatch.setenv(SELFCHECK_ENV_VAR, "1")
    assert write_sidecar(path, {}) is not None

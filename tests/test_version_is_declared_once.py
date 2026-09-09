"""The version is written in three places and they have to agree.

`updater.py` compares CURRENT_VERSION against the tag of the latest GitHub
release. If a release is tagged v2.1 while the shipped constant still reads 2.0,
every installed copy of that very build is told an update is available and
offered itself -- forever. The Windows resource block is the third copy, and it
is the one nobody looks at.

Three declarations of one fact, enforced nowhere, is the same shape as the
figure builder's min/max: the bound was shown to the user and ignored by the
code. This is the check that would have caught it at release time rather than
in a support mail.
"""
import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _updater_version():
    match = re.search(r'CURRENT_VERSION\s*=\s*"([^"]+)"', _read("src", "core", "updater.py"))
    assert match, "updater.py no longer declares CURRENT_VERSION"
    return match.group(1)


def _windows_versions():
    text = _read("tools", "win_version_info.txt")
    tuples = re.findall(r"(?:file|prod)vers=\((\d+), (\d+), (\d+), (\d+)\)", text)
    strings = re.findall(r"u'(\d+\.\d+\.\d+\.\d+)'", text)
    assert tuples and strings, "the Windows resource block changed shape"
    return {".".join(t) for t in tuples} | set(strings)


def test_the_windows_resource_agrees_with_the_app():
    app = _updater_version()
    parts = app.split(".")
    while len(parts) < 4:
        parts.append("0")
    expected = ".".join(parts)
    mismatched = {v for v in _windows_versions() if v != expected}
    assert not mismatched, (
        f"updater.py says {app}, the Windows resource says {sorted(mismatched)}")


def test_the_changelog_has_a_section_for_the_shipped_version():
    """A release with no notes is a release nobody can read."""
    app = _updater_version()
    changelog = _read("CHANGELOG.md")
    assert re.search(r"^## \[%s\]" % re.escape(app), changelog, re.M), (
        f"CHANGELOG.md has no '## [{app}]' section")


def test_unreleased_stays_at_the_top_for_the_next_change():
    changelog = _read("CHANGELOG.md")
    assert "## [Unreleased]" in changelog
    assert changelog.index("## [Unreleased]") < changelog.index("## [%s]" % _updater_version())

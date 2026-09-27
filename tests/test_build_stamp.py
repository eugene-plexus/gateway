"""The package knows the commit it was built from.

Stamped by `git archive` through `export-subst` (see `_build.py`). Two
things can silently stop the stamping: the file moving without its
attribute, or the attribute naming a file that no longer exists. Either
leaves every install reporting "development build", and nothing would
fail. This fails.
"""

from __future__ import annotations

from pathlib import Path

import eugene_plexus_gateway._build as build

REPO = Path(__file__).resolve().parents[1]


def test_the_stamped_file_is_marked_for_git_to_stamp() -> None:
    marked = [
        line.split()[0]
        for line in (REPO / ".gitattributes").read_text(encoding="utf-8").splitlines()
        if "export-subst" in line.split()[1:]
    ]
    source = Path(build.__file__).resolve().relative_to(REPO).as_posix()
    assert source in marked


def test_a_checkout_is_a_development_build_and_an_archive_is_its_commit(monkeypatch) -> None:
    # The placeholder, unsubstituted: a git checkout, not an archive.
    assert build.COMMIT.startswith("$Format")
    assert build.commit() is None
    stamped = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setattr(build, "COMMIT", stamped)
    assert build.commit() == stamped
    monkeypatch.setattr(build, "COMMIT", "0123456")
    assert build.commit() is None

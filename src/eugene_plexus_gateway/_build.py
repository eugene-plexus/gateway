"""The commit this package was built from, stamped by git.

Every Eugene Plexus install is built from GitHub source archives at pinned
commits, and GitHub makes those archives with `git archive`, which
replaces the placeholder below with the commit's full id because
`.gitattributes` marks this file `export-subst`. So the version is in the
code, put there by git when the archive was made: nothing of ours writes
it, and it says what was actually installed rather than what an installer
meant to install. A package that half-failed to upgrade reports its old
commit, which is how a mixed install becomes visible.

A development checkout is not an archive and keeps the placeholder;
`commit()` answers None for it.
"""

from __future__ import annotations

import re

COMMIT = "$Format:%H$"


def commit() -> str | None:
    """The full commit id, or None for a development checkout."""
    return COMMIT if re.fullmatch(r"[0-9a-f]{40}", COMMIT) else None

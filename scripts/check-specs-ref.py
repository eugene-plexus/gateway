"""Offline half of CI's "Codegen freshness" job.

The CI job downloads the specs commit named in SPECS_REF, regenerates the
models and fails on any diff. That needs the network and minutes. This checks
the commonest cause of that failure in milliseconds: SPECS_REF was bumped
without re-running codegen, or a `+local` dev stamp was left in the generated
code. It does not detect hand edits to the generated files.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAMP = re.compile(r'SPECS_REF\s*=\s*"([^"]*)"')


def stamp_files(root: Path) -> list[Path]:
    """The generated file that records the ref: Python packages or the UI's src/generated."""
    return sorted(root.glob("src/*/_generated/__init__.py")) + sorted(
        root.glob("src/generated/index.ts")
    )


def check(root: Path) -> list[str]:
    pinned = (root / "SPECS_REF").read_text(encoding="utf-8").strip()
    files = stamp_files(root)
    if not files:
        return ["no generated code records a specs ref; expected src/*/_generated/__init__.py"]
    problems = []
    for path in files:
        match = STAMP.search(path.read_text(encoding="utf-8"))
        recorded = match.group(1) if match else "(none)"
        if recorded != pinned:
            problems.append(
                f"{path.relative_to(root).as_posix()} records {recorded}, SPECS_REF is {pinned}"
            )
    return problems


if __name__ == "__main__":
    found = check(ROOT)
    if found:
        print("Generated code is out of sync with SPECS_REF (run the codegen script):")
        for line in found:
            print(f"  {line}")
        sys.exit(1)
    print("Generated code records the pinned specs ref")

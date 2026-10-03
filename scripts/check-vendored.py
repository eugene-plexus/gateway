"""Offline integrity check for generated platform modules (copied to consumers)."""

import hashlib
import json
from pathlib import Path


def check(root: Path) -> None:
    manifest = json.loads((root / "VENDORED.json").read_text(encoding="utf-8"))
    if manifest["format"] != 1:
        raise ValueError("unsupported vendoring manifest")
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"path outside repository: {name}")
        actual = hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
        if actual != expected["sha256"]:
            raise ValueError(f"{name}: edited vendored code; update specs/platform and regenerate")


if __name__ == "__main__":
    check(Path(__file__).resolve().parent.parent)
    print("Vendored platform integrity verified")

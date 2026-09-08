"""Regenerate Pydantic models from the pinned specs commit.

Reads `SPECS_REF` (a single line: the git SHA of `eugene-plexus/specs`),
downloads the OpenAPI tree at that SHA, and runs `datamodel-code-generator`
to produce Pydantic v2 models under `src/eugene_plexus_gateway/_generated/`.

The generated files are committed to the repo so builds are reproducible
without network access. CI re-runs this script and fails the build if the
working tree differs.

Usage:
    python scripts/codegen.py

To bump to a newer specs commit, overwrite `SPECS_REF` and re-run.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPECS_REF_FILE = REPO_ROOT / "SPECS_REF"
GENERATED_DIR = REPO_ROOT / "src" / "eugene_plexus_gateway" / "_generated"
WORKING_DIR = REPO_ROOT / ".codegen-cache"

SPECS_TARBALL_URL_TEMPLATE = "https://github.com/eugene-plexus/specs/archive/{ref}.tar.gz"

# Dev-only escape hatch: when this env var points at a local `specs`
# checkout, codegen reads that working tree instead of fetching the
# pinned commit from GitHub. Used to iterate on an unpublished spec
# (e.g. validating a provisional schema before it's committed/pushed).
# The generated code is marked `<ref>+local` and WILL differ from the
# pinned commit — CI regenerates from the pinned ref and will fail, so
# never merge local-generated output without first publishing the spec
# and bumping SPECS_REF.
SPECS_LOCAL_ENV = "EUGENE_PLEXUS_SPECS_LOCAL_PATH"

# The gateway's own wire surface is gateway.yaml (the OpenAI-compatible
# front door, config, admin). It also *consumes* inference-driver.yaml as
# an HTTP client, so we generate a second module of those types. Each
# input transitively pulls components/common.yaml types into its own
# module — we live with duplicating the common types between the two
# modules rather than introducing shared-code coupling.
#
SPECS_TO_GENERATE: list[tuple[str, str, list[str] | None]] = [
    ("openapi/gateway.yaml", "models.py", None),
    ("openapi/inference-driver.yaml", "driver_models.py", None),
]


def read_specs_ref() -> str:
    ref = SPECS_REF_FILE.read_text().strip()
    if not ref:
        sys.exit(f"error: {SPECS_REF_FILE} is empty")
    return ref


def download_specs(ref: str) -> Path:
    """Download specs tarball and extract to WORKING_DIR/specs."""
    url = SPECS_TARBALL_URL_TEMPLATE.format(ref=ref)
    print(f"fetching {url}")

    if WORKING_DIR.exists():
        shutil.rmtree(WORKING_DIR)
    WORKING_DIR.mkdir(parents=True)

    with urllib.request.urlopen(url) as response:
        data = response.read()

    extracted_root = None
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(WORKING_DIR)
        for member in tar.getnames():
            top_level = member.split("/", 1)[0]
            if top_level.startswith("specs-"):
                extracted_root = WORKING_DIR / top_level
                break

    if extracted_root is None or not extracted_root.is_dir():
        sys.exit(f"error: could not locate specs-* directory in tarball from {url}")

    return extracted_root


def write_init(generated_dir: Path, ref: str) -> None:
    """Write the package __init__.py with the pinned ref recorded."""
    content = f'''"""Generated Pydantic v2 models — DO NOT EDIT BY HAND.

Regenerate with:

    python scripts/codegen.py

Source: https://github.com/eugene-plexus/specs at commit {ref}
"""

SPECS_REF = "{ref}"
'''
    (generated_dir / "__init__.py").write_text(content, encoding="utf-8")


def run_codegen(specs_root: Path, ref: str) -> None:
    """Run datamodel-code-generator for each configured spec."""
    if GENERATED_DIR.exists():
        shutil.rmtree(GENERATED_DIR)
    GENERATED_DIR.mkdir(parents=True)

    write_init(GENERATED_DIR, ref)

    for input_rel, output_name, scopes in SPECS_TO_GENERATE:
        input_path = specs_root / input_rel
        if not input_path.is_file():
            sys.exit(f"error: expected {input_path} in extracted specs tarball")

        output_path = GENERATED_DIR / output_name
        print(f"generating {output_path.relative_to(REPO_ROOT)} from {input_rel}")

        cmd = [
            sys.executable,
            "-m",
            "datamodel_code_generator",
            "--input",
            str(input_path),
            "--input-file-type",
            "openapi",
            "--output",
            str(output_path),
            "--output-model-type",
            "pydantic_v2.BaseModel",
            "--target-python-version",
            "3.12",
            "--use-standard-collections",
            "--use-union-operator",
            "--use-schema-description",
            "--field-constraints",
            "--collapse-root-models",
            "--disable-timestamp",
        ]
        if scopes:
            cmd.append("--openapi-scopes")
            cmd.extend(scopes)
        subprocess.run(cmd, check=True)


def main() -> None:
    ref = read_specs_ref()

    local = os.environ.get(SPECS_LOCAL_ENV)
    if local:
        specs_root = Path(local).resolve()
        if not (specs_root / "openapi").is_dir():
            sys.exit(f"error: {SPECS_LOCAL_ENV}={local} has no openapi/ dir")
        print(f"using LOCAL specs at {specs_root} (pin {ref}; output is DEV-ONLY)")
        run_codegen(specs_root, f"{ref}+local")
        print(
            f"wrote {GENERATED_DIR.relative_to(REPO_ROOT)} — LOCAL/dev output; "
            "do not merge without publishing the spec and bumping SPECS_REF"
        )
        return

    print(f"specs ref: {ref}")
    specs_root = download_specs(ref)

    try:
        run_codegen(specs_root, ref)
    finally:
        if WORKING_DIR.exists():
            shutil.rmtree(WORKING_DIR)

    print(f"wrote {GENERATED_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

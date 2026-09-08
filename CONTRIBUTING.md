# Contributing to Eugene Plexus `gateway`

Thanks for your interest. This service implements the `gateway` OpenAPI contract from [`eugene-plexus/specs`](https://github.com/eugene-plexus/specs) — please read this before opening a PR.

## Developer Certificate of Origin (DCO)

We use the [Developer Certificate of Origin](https://developercertificate.org/) instead of a CLA. **Every commit must be signed off** with `git commit -s`:

```
Signed-off-by: Your Name <your.email@example.com>
```

The name and email must match your `git config user.name` and `git config user.email`. CI blocks PRs whose commits are missing matching sign-offs.

If you forgot to sign off:

```bash
git commit --amend -s --no-edit       # most recent commit
git rebase --signoff main             # whole branch
```

The full DCO text is in [the specs CONTRIBUTING.md](https://github.com/eugene-plexus/specs/blob/main/CONTRIBUTING.md).

## Wire contract changes go in `specs`, not here

If your change touches the HTTP API — endpoints, request/response shapes, schemas — it belongs in [`eugene-plexus/specs`](https://github.com/eugene-plexus/specs), not here. Land that PR first; bump `SPECS_REF` and re-run codegen here in a follow-up.

PRs to this repo should generally cover one or more of:

- **Implementation** of an existing spec endpoint
- **Bicameral loop work** — corpus-callosum blends, NT modulation, multi-pass policy
- **Memory work** — once the in-process stub is replaced by a real backend
- **Adapter / hemisphere-client work** — e.g. streaming, retries, circuit breakers
- **Tooling** — CI, type-checking, lint config, codegen script

## Local setup

```bash
git clone https://github.com/eugene-plexus/gateway
cd gateway
python -m venv .venv
. .venv/bin/activate           # or: .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

To exercise the full stack you also need two `inference-driver` instances reachable. See its [README](https://github.com/eugene-plexus/inference-driver) for setup; configure gateway's `leftDriverUrl` and `rightDriverUrl` accordingly.

## Git hooks

We use [pre-commit](https://pre-commit.com/) to auto-format staged files with Ruff before they reach CI. Enable it once per clone:

```bash
pip install pre-commit
pre-commit install
```

After that, `git commit` runs `ruff check --fix` and `ruff format` on staged Python files; if a hook reformats anything, re-stage and commit again.

## Style

- **Python 3.12+**. Modern syntax; no `from __future__ import annotations` unless materially helpful.
- **Ruff** for lint and format.
- **Mypy** with the project's relaxed settings (see `pyproject.toml`).
- **Async-first.** Every route handler and external call is `async`.
- **No comments explaining what code does.** Reserve comments for *why* a non-obvious choice was made.

## Running checks

```bash
ruff check .
ruff format --check .
mypy src/
pytest
python scripts/codegen.py
git diff --exit-code src/eugene_plexus_gateway/_generated/
```

## Reporting issues

File issues at <https://github.com/eugene-plexus/gateway/issues>. This repo is the right place for broader architectural questions about Eugene Plexus too.

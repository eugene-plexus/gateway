# Eugene Plexus — `gateway`

[![CI](https://github.com/eugene-plexus/gateway/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/eugene-plexus/gateway/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org)

Front door of [Eugene Plexus](https://github.com/eugene-plexus). Owns the bicameral loop: dispatches each turn to **two** [`inference-driver`](https://github.com/eugene-plexus/inference-driver) instances in parallel (typically configured with different model families), runs the corpus-callosum blend on their outputs, decides terminate or another pass, and returns the final response.

## Status

**v0.1, working bicameral loop.** Calls two configured hemisphere drivers, runs a trivial agreement-based corpus callosum, writes conversation history to in-process memory, and returns blended responses. Streaming (`/v1/chat/stream`) is stubbed pending the UI consumer.

## Wire contract

This service implements the [`gateway.yaml`](https://github.com/eugene-plexus/specs/blob/main/openapi/gateway.yaml) OpenAPI 3.1 spec from [`eugene-plexus/specs`](https://github.com/eugene-plexus/specs). Pydantic models in `src/eugene_plexus_gateway/_generated/` are produced via codegen — see [Codegen](#codegen).

Endpoints:

| Method | Path                            | Status      |
|--------|---------------------------------|-------------|
| GET    | `/healthz`                      | ✅           |
| POST   | `/v1/chat`                      | ✅           |
| POST   | `/v1/chat/stream`               | stub (501)  |
| GET    | `/v1/conversations/{id}`        | ✅           |
| GET    | `/v1/admin/drivers`             | ✅           |
| POST   | `/v1/admin/drivers/probe`       | ✅           |
| GET    | `/v1/admin/nt-state`            | ✅           |
| POST   | `/v1/admin/restart`             | ✅           |
| GET    | `/v1/config`                    | ✅           |
| GET    | `/v1/config/schema`             | ✅           |
| PATCH  | `/v1/config`                    | ✅           |
| POST   | `/v1/config/test`               | ✅           |

## What v0.1 does

- **A `drivers` list** in config (operator-supplied name + URL per entry). The gateway calls each in parallel for every turn. v0.1 requires exactly two; the agreement and blend functions are pairwise.
- **Trivial corpus callosum:** agreement = Jaccard similarity of word-sets between the two outputs. Above the configured `agreementThreshold` → terminate; below → another pass; cap at `defaultMaxPasses`.
- **Trivial blend:** when terminating, picks the longer of the two hemispheres' responses as the final assistant message. (Honest about the lack of a smart blend; deferred until we have data on what real disagreements look like.)
- **Static neutral NT state.** The endpoint and schema exist so consumers can begin reading NT now; modulation lands in v0.2.
- **Memory client.** Calls a separate `memory` component over HTTP (configurable `memoryUrl`); fails gracefully into degraded mode when memory is unreachable so chat doesn't fully die when storage hiccups.
- **Safe-mode boot.** `EUGENE_PLEXUS_GATEWAY_SAFE_MODE=1` (set by the watchdog when a previous boot failed) skips loading the persisted config and keeps the gateway reachable for config edits while `/v1/chat` returns 503. Recovery path when bad config breaks startup.
- **Restart endpoint.** `POST /v1/admin/restart` schedules a graceful process exit; an external supervisor (the watchdog, or systemd/docker for non-personal-use installs) respawns. Used by the UI's Restart Now flow after config changes that require it.
- **DEBUG-level content trace.** Set `logLevel: DEBUG` in config and the bicameral loop logs every outgoing message (role + driverName + content), each hemisphere's response (finish reason + latency + content), the callosum agreement, and the final blended text. Useful when cross-vendor disagreements need debugging.

## What v0.1 does NOT do

- Streaming (the gateway's `/v1/chat/stream` endpoint and the underlying inference-driver streaming both wait on the UI consumer).
- Drives, sleep consolidation, plasticity, EWC, autonomous thinking, real vector memory / RAG, tool/MCP integration, auth (assumes Tailscale tailnet).

## Running

```bash
pip install -e ".[dev]"
python -m eugene_plexus_gateway
```

Default port: `8080`, overridable via `EUGENE_PLEXUS_GATEWAY_BIND_PORT` (the watchdog uses this when supervising). Other startup behavior is configured via env vars (12-factor) or by editing `config.yaml` (auto-created in the working directory on first run).

You'll need two `inference-driver` instances reachable. Typical local dev:

```bash
# in one shell
EUGENE_PLEXUS_HD_CONFIG_FILE=./left.yaml EUGENE_PLEXUS_HD_BIND_PORT=8081 \
  python -m eugene_plexus_inference_driver
# in another
EUGENE_PLEXUS_HD_CONFIG_FILE=./right.yaml EUGENE_PLEXUS_HD_BIND_PORT=8082 \
  python -m eugene_plexus_inference_driver
```

…then add both to the gateway's `drivers` config (operator-chosen names; URLs match the driver bind ports). For personal-use installs the watchdog handles all of this automatically — this is the manual path for tinkering.

> **v0.1 has no auth.** Deployment assumption: behind a [Tailscale](https://tailscale.com/) tailnet or equivalent network boundary. Anyone reachable on the network can hit any endpoint. Auth lands in v0.2.

## Codegen

Pydantic models for the wire contract are generated from the pinned commit of `eugene-plexus/specs` recorded in [`SPECS_REF`](SPECS_REF):

```bash
python scripts/codegen.py
```

The script downloads the specs at the pinned SHA and runs `datamodel-code-generator`. Generated files are committed for reproducibility; CI re-runs codegen and fails on diff.

## Development

```bash
pip install -e ".[dev]"

ruff check .
ruff format --check .
mypy src/
pytest

python scripts/codegen.py
git diff --exit-code src/eugene_plexus_gateway/_generated/
```

## License

Apache 2.0 — see [`LICENSE`](LICENSE).

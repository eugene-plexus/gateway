# Eugene Plexus — `gateway`

[![CI](https://github.com/eugene-plexus/gateway/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/eugene-plexus/gateway/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org)

Front door of [Eugene Plexus](https://github.com/eugene-plexus/specs#readme): **one OpenAI-compatible endpoint over configured local and cloud backends.** It resolves requested models, balances replicas, cascades across priority tiers, and decides idle unload and wake on demand. The owning node's agent executes lifecycle actions.

**It holds no engine-specific knowledge.** One [`inference-driver`](https://github.com/eugene-plexus/inference-driver) runs per backend and owns provider choice, model id, secrets and protocol adaptation. The gateway owns output-affecting request parameters. The two layers are never collapsed; see the [design doc](https://github.com/eugene-plexus/specs/blob/main/docs/design/local-inference-control-plane.md) for why.

## The routing table is derived, not configured

The gateway reads configured agents for `inference-driver` entries, asks each
driver's `/v1/info` what it serves, and groups replicas by model id. It joins
supervised drivers to runtimes by name and routes only to `ready` runtimes.
Peer connections prefer `advertiseUrl` over the agent-local bind/probe URL.

- Backend addresses come from agent topology, not a second driver URL list.
- The agent declares a companion driver for each runtime. A ready launch becomes
  routable when discovered; an empty eligible set triggers an on-demand refresh.
- Within a model's replica set, balancing uses outstanding requests weighted by
  `parallelSlots`. Replicas are not distinct fallback tiers.
- `modelSlots` configures ordered tiers of **model ids**. Each id resolves to its
  replica set; a cloud-backed model can be a fallback tier like a local one.

Idle unload and start on demand do not depend on the control root being available.
See the [M6 design](https://github.com/eugene-plexus/specs/blob/main/docs/design/m6-lifecycle-policy.md).

## Verification Status

M6 passed a live six-process run with two llama.cpp replicas on one GPU. M7 passed
with two agents on one host. A short post-unload routing window remains open;
real two-machine and two-GPU runs remain unverified. See the
[current project status](https://github.com/eugene-plexus/specs#current-status)
and [M7 acceptance record](https://github.com/eugene-plexus/specs/blob/main/docs/acceptance/m7-two-agent-run.md).

## Wire contract

This service implements the [`gateway.yaml`](https://github.com/eugene-plexus/specs/blob/main/openapi/gateway.yaml) OpenAPI 3.1 spec from [`eugene-plexus/specs`](https://github.com/eugene-plexus/specs). Pydantic models in `src/eugene_plexus_gateway/_generated/` are produced via codegen — see [Codegen](#codegen).

| Method | Path                      | Notes                                          |
| ------ | ------------------------- | ---------------------------------------------- |
| GET    | `/healthz`                | unauthenticated                                |
| POST   | `/v1/chat/completions`    | OpenAI-compatible; `stream: true` for SSE      |
| GET    | `/v1/models`              | OpenAI-compatible; a view of the routing table |
| GET    | `/v1/admin/drivers`       | operator-only                                  |
| POST   | `/v1/admin/drivers/probe` | operator-only                                  |
| POST   | `/v1/admin/restart`       | operator-only                                  |
| GET    | `/v1/config`              | operator-only                                  |
| GET    | `/v1/config/schema`       | operator-only                                  |
| PATCH  | `/v1/config`              | operator-only                                  |
| POST   | `/v1/config/test`         | operator-only                                  |

`/v1/chat/completions` and `/v1/models` are the **only** operations in Eugene Plexus that use snake_case field names and OpenAI's error envelope. That is deliberate and not negotiable: "OpenAI-compatible" is worth nothing unless an unmodified OpenAI SDK can point its `base_url` here and work, and those SDKs parse the error shape to build their exceptions. Renaming `max_tokens` to `maxTokens` for house consistency would break the entire audience.

Both carry a namespaced `x_eugene_plexus` extension reporting which driver and backend served a request, its latency, and how many backends were tried. Clients ignore unknown fields, so it costs nothing — and the failure mode of a routing layer is opacity, so `attempts > 1` being visible is the point.

## The gateway owns every output-affecting parameter

Temperature, max tokens and stop sequences ride on every request the gateway makes downstream. A driver never substitutes a local default: if a value reaches a backend, the gateway put it there. Omitted values use gateway config defaults. Library profiles configure engine launches; they are not automatic per-model sampling defaults.

Correspondingly nothing here refuses a model. A backend that rejects `temperature`, as some reasoning models do, has the parameter dropped with a warning by the driver. You own the model; routing to it is the job.

## Failure taxonomy

The status codes distinguish the cases that differ operationally, because "it didn't work" is not an actionable answer:

| Code | Meaning                                                                                                                                                   | Retry?                |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| 404  | Nothing serves the requested model at all                                                                                                                 | No — fix the topology |
| 400  | A backend rejected the request (4xx). Does **not** cascade: the next backend would reject it identically, and cascading past it would bury the real error | No — fix the request  |
| 503  | A driver serves the model but isn't ready — usually an engine still loading its weights                                                                   | Yes                   |
| 502  | The cascade ran and every eligible backend failed                                                                                                         | Maybe                 |

Transport errors, timeouts and 5xx cascade to the next backend; 4xx fails hard.

## Safe mode

`EUGENE_PLEXUS_GATEWAY_SAFE_MODE=1` (set by the agent after a failed boot) skips the persisted config and builds no routing table. `PATCH /v1/config` still writes to disk so the repair survives the next normal boot; `/v1/chat/completions` returns 503 until then, and `/v1/models` returns an empty list — "nothing available" being a valid answer to "what have you got".

## Running

```bash
pip install -e ".[dev]"
python -m eugene_plexus_gateway
```

Default port: `8080`, overridable via `EUGENE_PLEXUS_GATEWAY_BIND_PORT` (the agent uses this when supervising). Other startup behavior is configured via env vars (12-factor) or by editing `config.yaml` (auto-created in the working directory on first run).

The gateway needs reachable **agents**, where it reads topology, and a driver
serving the requested model. Supervised engine runtimes receive companion drivers
automatically; declare standalone cloud/CLI drivers on the agent separately.
Configure additional agent URLs and `modelSlots` through the gateway's config
trio. `EUGENE_PLEXUS_GATEWAY_AGENT_URL` supplies the startup agent address when
it is not on loopback.

> **Auth is on by default when supervised.** The agent threads a signing key and a service token in at spawn; every endpoint except `/healthz` then requires a bearer token. Run standalone without those env vars and the gateway serves unauthenticated — dev only. A mesh VPN (Tailscale/WireGuard) is still the network boundary between hosts.

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

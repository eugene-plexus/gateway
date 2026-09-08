"""The continuous-loop runtime — Eugene's seat of consciousness.

One Eugene = one consciousness = one workspace = one attention. A single
long-lived asyncio task (the `ConsciousnessLoop`) is the only thing that
thinks; the HTTP endpoints are thin doors that inject afferent events
onto its queue and subscribe to its observability stream.

This package supersedes the v0.2 request-response `chat` path. See the
design doc `docs/design/m1-continuous-runtime.md` in the `specs` repo.
"""

"""The videos doors (P5, 2026-09-28): OpenAI's requests read, routed, answered.

Measured first (`provider-accounts-measurement.md` section 10) and decided by
Troy (design section 11):

* **OpenAI shut its video API down on 2026-09-24**; OpenRouter is the only
  live backend. OpenAI's shape is still what the SDK's `client.videos`
  speaks, so this door is OpenAI's and the driver translates.
* **The handle is the job** (call #5): `video_` and a signed payload naming
  the driver, its node, the model, the backend's job, the owner, and the
  `seconds` and `size` asked for (OpenRouter's poll carries neither, and
  OpenAI's `VideoResource` requires both). It is signed with the gateway's
  own secret file (P5-4), so a restart loses nothing and another install's,
  or a forged, handle reads as not found.
* **The owner is the key that made the job** (P5-3): a client key's job is
  readable only with that key, an operator session's by any operator
  session of the install (sessions rotate at every sign-in).
* **`seconds` is any whole number a model lists** (P5-1), `size` one it
  lists, and `input_reference` only to a model that takes a first frame.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ._generated.driver_models import VideoCapabilities, VideoJob, VideoRequest
from ._generated.models import VideoCreateRequest
from ._private_files import write_private_text
from .chat_contract import Refusal, TooLarge, _field_name, _most_specific
from .image_doors import MAX_UPLOAD_BYTES, Upload, _data_url, _upload

#: The handle's prefix, as OpenAI's ids carry theirs (`video_...`).
PREFIX = "video_"


@dataclass(frozen=True)
class VideoAsk:
    """A video request, read from whichever form it came in."""

    model: str
    prompt: str
    seconds: int | None = None
    size: str | None = None
    reference: Upload | None = None


def _seconds(raw: str | None) -> int | None:
    if raw is None:
        return None
    if not raw.isdigit() or not 1 <= int(raw) <= 120:
        raise Refusal("seconds", "must be a whole number of seconds, as a string, from 1 to 120")
    return int(raw)


def parse_create(raw: Any) -> VideoAsk:
    """`POST /v1/videos`' JSON body."""
    if not isinstance(raw, dict):
        raise Refusal("body", "must be a JSON object")
    try:
        parsed = VideoCreateRequest.model_validate(raw)
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_context=False)
        for error in errors:
            if error["type"] == "extra_forbidden" and error["loc"]:
                raise Refusal(
                    _field_name(str(error["loc"][-1])), "is not a field of this request"
                ) from None
        parts = _most_specific(errors)
        where = ".".join(_field_name(p) if isinstance(p, str) else str(p) for p in parts)
        raise Refusal(where or "body", "has an invalid or missing value") from None
    reference = (
        _data_url(parsed.input_reference, "input_reference")
        if parsed.input_reference is not None
        else None
    )
    if reference is not None and len(reference.raw) > MAX_UPLOAD_BYTES:
        raise TooLarge("input_reference", "is over 25 MiB")
    return VideoAsk(
        model=parsed.model,
        prompt=parsed.prompt,
        seconds=_seconds(parsed.seconds),
        size=parsed.size,
        reference=reference,
    )


_FORM_FIELDS = {"model", "prompt", "seconds", "size", "input_reference"}


async def read_create_form(form: Any) -> VideoAsk:
    """`POST /v1/videos`' multipart form, `input_reference` a file."""
    for key in form:
        if key not in _FORM_FIELDS:
            raise Refusal(_field_name(key), "is not a field of this form")

    def text(key: str, *, limit: int) -> str | None:
        value = form.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise Refusal(key, "must be a text field")
        if len(value) > limit:
            raise Refusal(key, f"is longer than {limit} characters")
        return value

    model = text("model", limit=512)
    if not model:
        raise Refusal("model", "is required")
    prompt = text("prompt", limit=32000)
    if not prompt:
        raise Refusal("prompt", "is required")
    reference: Upload | None = None
    upload = form.get("input_reference")
    if upload is not None:
        if isinstance(upload, str):
            raise Refusal(
                "input_reference",
                "must be an image file; a URL is not fetched and a file_id "
                "names a store this install does not have",
            )
        raw = await upload.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise TooLarge("input_reference", "is over 25 MiB")
        reference = _upload(raw, "input_reference")
    return VideoAsk(
        model=model,
        prompt=prompt,
        seconds=_seconds(text("seconds", limit=8)),
        size=text("size", limit=32),
        reference=reference,
    )


def rules_out(caps: VideoCapabilities | None, ask: VideoAsk) -> tuple[str, str] | None:
    """`(field, why)` when this model's listing rules the request out; None
    when it may take it. Unknown lists are the backend's to check; a first
    frame must be confirmed."""
    c = caps or VideoCapabilities()
    if ask.reference is not None and not c.firstFrame:
        return "input_reference", "takes no first frame"
    if ask.seconds is not None and c.durations is not None and ask.seconds not in c.durations:
        listed = ", ".join(str(d) for d in c.durations)
        return "seconds", f"does not make {ask.seconds} s; it makes {listed}"
    if ask.size is not None and c.sizes is not None and ask.size not in c.sizes:
        return "size", f"does not make {ask.size}; it makes {', '.join(c.sizes)}"
    return None


def to_driver(ask: VideoAsk, *, local_only: bool, request_id: Any) -> VideoRequest:
    return VideoRequest(
        prompt=ask.prompt,
        seconds=ask.seconds,
        size=ask.size,
        firstFrame=ask.reference.as_data() if ask.reference is not None else None,
        localOnly=local_only,
        requestId=request_id,
    )


# ---------------------------------------------------------------------------
# The handle
# ---------------------------------------------------------------------------


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class Handles:
    """Signs and reads job handles with the gateway's own secret (P5-4).

    The secret is made on first use and kept owner-only beside the gateway's
    config, so a restart keeps it and every handle issued before the restart
    still reads. A reinstall makes a new one, and older handles then read as
    not found -- the honest answer, since nothing can say otherwise.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._key: bytes | None = None

    def _secret(self) -> bytes:
        if self._key is None:
            if self._path.exists():
                self._key = bytes.fromhex(self._path.read_text(encoding="ascii").strip())
            else:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                write_private_text(self._path, secrets.token_hex(32) + "\n", encoding="ascii")
                self._key = bytes.fromhex(self._path.read_text(encoding="ascii").strip())
        return self._key

    def issue(self, payload: dict[str, Any]) -> str:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        mac = hmac.new(self._secret(), body, hashlib.sha256).digest()
        return f"{PREFIX}{_b64(body)}.{_b64(mac)}"

    def read(self, handle: str) -> dict[str, Any] | None:
        """The payload, or None for anything this install did not sign."""
        if not handle.startswith(PREFIX) or "." not in handle:
            return None
        encoded, _, mac = handle[len(PREFIX) :].partition(".")
        try:
            body, given = _unb64(encoded), _unb64(mac)
        except (binascii.Error, ValueError):
            return None
        expected = hmac.new(self._secret(), body, hashlib.sha256).digest()
        if not hmac.compare_digest(given, expected):
            return None
        try:
            payload = json.loads(body)
        except ValueError:
            return None
        return payload if isinstance(payload, dict) else None


def owner_of(key_id: str | None) -> str:
    """P5-3: a client key owns its jobs; operator sessions share the install's."""
    return f"key:{key_id}" if key_id else "install"


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


def resource(handle: str, payload: dict[str, Any], job: VideoJob | None) -> dict[str, Any]:
    """OpenAI's `VideoResource`, every field its schema requires. Before the
    first poll there is only the submit's answer; `prompt` is not kept."""
    status = job.status.value if job is not None else "queued"
    progress = 100 if status == "completed" else (job.progress if job and job.progress else 0)
    error = None
    if status == "failed":
        error = {
            "code": "video_generation_failed",
            "message": (
                job.error if job and job.error else "The provider reported the job failed."
            ),
        }
    body: dict[str, Any] = {
        "id": handle,
        "object": "video",
        "model": payload.get("m"),
        "status": status,
        "progress": progress,
        "created_at": payload.get("c"),
        "completed_at": int(time.time()) if status == "completed" else None,
        "expires_at": None,
        "prompt": None,
        "size": payload.get("z") or "auto",
        "seconds": str(payload["s"]) if payload.get("s") is not None else "auto",
        "remixed_from_video_id": None,
        "error": error,
    }
    # The gateway's first money (2026-10-08): what the provider says it
    # billed, once the job has ended and it says. Absent is unknown.
    if job is not None and job.cost is not None and status in ("completed", "failed"):
        body["x_eugene_plexus"] = {"cost_usd": job.cost}
    return body

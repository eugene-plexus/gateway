"""The images doors (P4, 2026-09-28): OpenAI's requests read, routed, answered.

Measured first (`provider-accounts-measurement.md` section 9) and decided by
Troy (design section 10):

* **Two edit forms.** The OpenAI SDK sends an edit as multipart only --
  `image` for one image, `image[]` for several, `mask` -- and labels a
  `BytesIO` `application/octet-stream`, so an image's type is read from its
  bytes. OpenAI's spec also takes JSON with `images[].image_url`; there only
  a `data:` URL is taken (A4: no URL is fetched; a `file_id` names a store
  Eugene does not have).
* **A setting routes by the model's own listing** (`ImageCapabilities`): a
  listed value is enforced, `[]` is not taken, null is the backend's to
  check. `auto` asks for nothing and is never a reason to route away.
* **`stream: true` routes only to a model that streams** (P4-3).
* **`response_format: url` is accepted and ignored** (P4-1): every answer is
  `b64_json`.
* **The answer is OpenAI's shape** whatever the backend: `created` from this
  clock where the backend gave none, `output_format` from the bytes, and
  every stream event carrying the fields OpenAI's schema requires, which
  OpenRouter's omit.

Duplicated rather than shared with the driver where the two need the same
fact (the byte sniffer): components share schemas, not code.
"""

from __future__ import annotations

import base64
import binascii
import json
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ._generated.driver_models import (
    GeneratedImage,
    ImageCapabilities,
    ImageData,
    ImageRequest,
    ImageResponse,
    ImageUsage,
)
from ._generated.models import (
    ImageBackground,
    ImageEditJsonRequest,
    ImageGenerationRequest,
    ImageInputFidelity,
    ImageModeration,
    ImageOutputFormat,
    ImageQuality,
    ImageResponseFormat,
)
from .chat_contract import Refusal, TooLarge, _field_name, _most_specific

#: OpenAI's limits: 25 MiB decoded across every image and the mask, 16 images.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_INPUT_IMAGES = 16

#: What a backend here reads as an input image.
_READS = ("image/png", "image/jpeg", "image/webp", "image/gif")

#: OpenRouter's own fields, refused by name (P4-4) with the OpenAI way round.
_NOT_OPENAI = {
    "aspect_ratio": "is OpenRouter's own field, not OpenAI's; send size (WIDTHxHEIGHT), "
    "which reaches every backend",
    "resolution": "is OpenRouter's own field, not OpenAI's; send size (WIDTHxHEIGHT), "
    "which reaches every backend",
    "seed": "is not carried: it is OpenRouter's own field, not OpenAI's",
    "input_references": "is OpenRouter's own field; send the images to /v1/images/edits",
}


def sniff(raw: bytes) -> str | None:
    """The media type an image's bytes carry."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    head = raw[:512].lstrip()
    if head.startswith(b"<svg") or (head.startswith(b"<?xml") and b"<svg" in raw[:4096]):
        return "image/svg+xml"
    return None


def format_name(media_type: str) -> str:
    """`image/jpeg` -> `jpeg`, `image/svg+xml` -> `svg`: OpenAI's spelling."""
    return media_type.split("/", 1)[-1].split("+", 1)[0]


@dataclass(frozen=True)
class Upload:
    """One input image, its type read from its bytes."""

    raw: bytes
    media_type: str

    def as_data(self) -> ImageData:
        return ImageData(data=base64.b64encode(self.raw).decode("ascii"), mediaType=self.media_type)


@dataclass(frozen=True)
class ImageAsk:
    """An images request, read from whichever form it came in."""

    door: str  # "generations" or "edits"
    model: str
    prompt: str
    n: int | None = None
    size: str | None = None
    quality: str | None = None
    background: str | None = None
    output_format: str | None = None
    output_compression: int | None = None
    moderation: str | None = None
    style: str | None = None
    user: str | None = None
    input_fidelity: str | None = None
    stream: bool = False
    partial_images: int | None = None
    references: tuple[Upload, ...] = field(default_factory=tuple)
    mask: Upload | None = None

    @property
    def event_prefix(self) -> str:
        return "image_edit" if self.door == "edits" else "image_generation"


def _value(member: Any) -> Any:
    return getattr(member, "value", member)


def _named(exc: ValidationError) -> Refusal:
    """The field the caller meant, in their words; an unknown one said so."""
    errors = exc.errors(include_input=False, include_context=False)
    for error in errors:
        if error["type"] == "extra_forbidden" and error["loc"]:
            name = str(error["loc"][-1])
            return Refusal(
                _field_name(name), _NOT_OPENAI.get(name, "is not a field of this request")
            )
    parts = _most_specific(errors)
    where = ".".join(_field_name(p) if isinstance(p, str) else str(p) for p in parts)
    return Refusal(where or "body", "has an invalid or missing value")


def _upload(raw: bytes, what: str) -> Upload:
    if not raw:
        raise Refusal(what, "is empty")
    media = sniff(raw)
    if media not in _READS:
        raise Refusal(what, "is not a PNG, JPEG, WebP or GIF image")
    return Upload(raw=raw, media_type=media)


def _within_limits(references: list[Upload], mask: Upload | None) -> None:
    if len(references) > MAX_INPUT_IMAGES:
        raise Refusal("image", f"takes at most {MAX_INPUT_IMAGES} images, as OpenAI's does")
    total = sum(len(u.raw) for u in references) + (len(mask.raw) if mask else 0)
    if total > MAX_UPLOAD_BYTES:
        raise TooLarge("image", f"is {total} bytes in all; the limit is 25 MiB, as OpenAI's is")


def parse_generation(raw: Any) -> ImageAsk:
    """`/v1/images/generations`' JSON body."""
    if not isinstance(raw, dict):
        raise Refusal("body", "must be a JSON object")
    try:
        parsed = ImageGenerationRequest.model_validate(raw)
    except ValidationError as exc:
        raise _named(exc) from None
    return ImageAsk(
        door="generations",
        model=parsed.model,
        prompt=parsed.prompt,
        n=parsed.n,
        size=parsed.size,
        quality=_value(parsed.quality),
        background=_value(parsed.background),
        output_format=_value(parsed.output_format),
        output_compression=parsed.output_compression,
        moderation=_value(parsed.moderation),
        style=_value(parsed.style),
        user=parsed.user,
        stream=parsed.stream is True,
        partial_images=parsed.partial_images,
    )


def _data_url(ref: Any, where: str) -> Upload:
    """One `ImageRef`: a `data:` URL only."""
    if ref.file_id is not None:
        raise Refusal(
            f"{where}.file_id",
            "names a stored file, and this install keeps none; send the image inline as a "
            "data: URL",
        )
    url = ref.image_url
    if not url:
        raise Refusal(where, "needs image_url, a data: URL")
    if not url.startswith("data:"):
        raise Refusal(f"{where}.image_url", "is not fetched: send the image inline as a data: URL")
    head, sep, payload = url.partition(",")
    if not sep or not head.endswith(";base64"):
        raise Refusal(f"{where}.image_url", "must be a base64 data: URL")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise Refusal(f"{where}.image_url", "is not valid base64") from None
    return _upload(raw, f"{where}.image_url")


def parse_edit_json(raw: Any) -> ImageAsk:
    """`/v1/images/edits`' JSON form."""
    if not isinstance(raw, dict):
        raise Refusal("body", "must be a JSON object")
    try:
        parsed = ImageEditJsonRequest.model_validate(raw)
    except ValidationError as exc:
        raise _named(exc) from None
    references = [_data_url(ref, f"images[{i}]") for i, ref in enumerate(parsed.images)]
    mask = _data_url(parsed.mask, "mask") if parsed.mask is not None else None
    _within_limits(references, mask)
    return ImageAsk(
        door="edits",
        model=parsed.model,
        prompt=parsed.prompt,
        n=parsed.n,
        size=parsed.size,
        quality=_value(parsed.quality),
        background=_value(parsed.background),
        output_format=_value(parsed.output_format),
        output_compression=parsed.output_compression,
        moderation=_value(parsed.moderation),
        user=parsed.user,
        input_fidelity=_value(parsed.input_fidelity),
        stream=parsed.stream is True,
        partial_images=parsed.partial_images,
        references=tuple(references),
        mask=mask,
    )


#: The multipart edit form's text fields, and each one's values where named.
_FORM_ENUMS: dict[str, tuple[str, ...]] = {
    "quality": tuple(m.value for m in ImageQuality),
    "background": tuple(m.value for m in ImageBackground),
    "output_format": tuple(m.value for m in ImageOutputFormat),
    "moderation": tuple(m.value for m in ImageModeration),
    "input_fidelity": tuple(m.value for m in ImageInputFidelity),
    "response_format": tuple(m.value for m in ImageResponseFormat),
}
_FORM_FIELDS = {
    "image",
    "image[]",
    "mask",
    "model",
    "prompt",
    "n",
    "size",
    "output_compression",
    "stream",
    "partial_images",
    "user",
    *_FORM_ENUMS,
}


async def read_edit_form(form: Any) -> ImageAsk:
    """`/v1/images/edits`' multipart form, as the OpenAI SDK sends it."""
    for key in form:
        if key not in _FORM_FIELDS:
            raise Refusal(_field_name(key), _NOT_OPENAI.get(key, "is not a field of this form"))

    def text(key: str, *, limit: int) -> str | None:
        value = form.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise Refusal(key, "must be a text field")
        if len(value) > limit:
            raise Refusal(key, f"is longer than {limit} characters")
        return value

    def whole(key: str, low: int, high: int) -> int | None:
        raw = text(key, limit=8)
        if raw is None:
            return None
        try:
            value = int(raw)
        except ValueError:
            raise Refusal(key, "must be a whole number") from None
        if not low <= value <= high:
            raise Refusal(key, f"must be between {low} and {high}")
        return value

    def named(key: str) -> str | None:
        value = text(key, limit=32)
        if value is not None and value not in _FORM_ENUMS[key]:
            raise Refusal(key, f"must be one of {', '.join(_FORM_ENUMS[key])}")
        return value

    model = text("model", limit=512)
    if not model:
        raise Refusal("model", "is required")
    prompt = text("prompt", limit=32000)
    if not prompt:
        raise Refusal("prompt", "is required")
    files = [*form.getlist("image"), *form.getlist("image[]")]
    if not files:
        raise Refusal("image", "is required: the image to edit, as a multipart file")
    references: list[Upload] = []
    for index, item in enumerate(files):
        if isinstance(item, str):
            raise Refusal("image", "must be a file, not a text field")
        references.append(_upload(await item.read(), f"image[{index}]"))
    mask: Upload | None = None
    raw_mask = form.get("mask")
    if raw_mask is not None:
        if isinstance(raw_mask, str):
            raise Refusal("mask", "must be a file, not a text field")
        mask = _upload(await raw_mask.read(), "mask")
    _within_limits(references, mask)
    stream = text("stream", limit=8)
    if stream is not None and stream.lower() not in ("true", "false", "1", "0"):
        raise Refusal("stream", "must be true or false")
    named("response_format")  # checked, then ignored (P4-1)
    return ImageAsk(
        door="edits",
        model=model,
        prompt=prompt,
        n=whole("n", 1, 10),
        size=text("size", limit=32),
        quality=named("quality"),
        background=named("background"),
        output_format=named("output_format"),
        output_compression=whole("output_compression", 0, 100),
        moderation=named("moderation"),
        user=text("user", limit=256),
        input_fidelity=named("input_fidelity"),
        stream=stream is not None and stream.lower() in ("true", "1"),
        partial_images=whole("partial_images", 0, 3),
        references=tuple(references),
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def rules_out(caps: ImageCapabilities | None, ask: ImageAsk) -> tuple[str, str] | None:
    """`(field, why)` when this model's listing rules the request out; None
    when it may take it. Unknown capabilities are the backend's to check,
    except the three that must be confirmed: `stream`, `mask`, and editing."""
    c = caps or ImageCapabilities()
    refs = len(ask.references)
    if ask.stream and not c.streaming:
        return "stream", "does not stream partial images"
    if ask.mask is not None and not c.mask:
        return "mask", "does not honour a mask (only OpenAI's own API does)"
    if refs < (c.minReferences or 0):
        return "image", "only edits an image, and makes nothing without one"
    if c.maxReferences is not None and refs > c.maxReferences:
        if c.maxReferences == 0:
            return "image", "does not edit images"
        return "image", f"takes at most {c.maxReferences} images"
    if ask.n is not None and c.maxImages is not None and ask.n > c.maxImages:
        return "n", f"makes at most {c.maxImages} per request"
    for name, value, allowed in (
        ("quality", ask.quality, c.qualities),
        ("background", ask.background, c.backgrounds),
        ("output_format", ask.output_format, c.outputFormats),
    ):
        if value is None or value == "auto" or allowed is None or value in allowed:
            continue
        if not allowed:
            return name, "does not take it"
        return name, f"does not take {value}; it takes {', '.join(allowed)}"
    return None


def to_driver(ask: ImageAsk, *, local_only: bool, request_id: Any) -> ImageRequest:
    return ImageRequest(
        prompt=ask.prompt,
        n=ask.n,
        size=ask.size,
        quality=ask.quality,
        background=ask.background,
        outputFormat=ask.output_format,
        outputCompression=ask.output_compression,
        moderation=ask.moderation,
        style=ask.style,
        user=ask.user,
        inputFidelity=ask.input_fidelity,
        references=[u.as_data() for u in ask.references] or None,
        mask=ask.mask.as_data() if ask.mask is not None else None,
        partialImages=ask.partial_images if ask.stream else None,
        localOnly=local_only,
        requestId=request_id,
    )


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


def _usage(usage: ImageUsage | None) -> dict[str, Any] | None:
    if usage is None or (usage.inputTokens is None and usage.outputTokens is None):
        return None
    total = usage.totalTokens
    if total is None and usage.inputTokens is not None and usage.outputTokens is not None:
        total = usage.inputTokens + usage.outputTokens
    return {
        "input_tokens": usage.inputTokens,
        "output_tokens": usage.outputTokens,
        "total_tokens": total,
    }


def images_body(result: ImageResponse, *, routing: dict[str, Any]) -> dict[str, Any]:
    """OpenAI's `ImagesResponse`, always `b64_json` (P4-1)."""
    body: dict[str, Any] = {
        "created": result.created or int(time.time()),
        "data": [
            {
                "b64_json": image.data,
                **({"revised_prompt": image.revisedPrompt} if image.revisedPrompt else {}),
            }
            for image in result.images
        ],
        "output_format": format_name(result.images[0].mediaType),
    }
    for key in ("size", "quality", "background"):
        value = getattr(result, key)
        if value:
            body[key] = value
    usage = _usage(result.usage)
    if usage is not None:
        body["usage"] = usage
    body["x_eugene_plexus"] = routing
    return body


def stream_event(
    ask: ImageAsk,
    kind: str,
    image: GeneratedImage,
    *,
    index: int | None = None,
    usage: ImageUsage | None = None,
    routing: dict[str, Any] | None = None,
) -> str:
    """One SSE event in OpenAI's shape, every field its schema requires
    filled: the settings as asked (`auto` when not), the format from the
    bytes."""
    payload: dict[str, Any] = {
        "type": f"{ask.event_prefix}.{kind}",
        "b64_json": image.data,
        "created_at": int(time.time()),
        "size": ask.size or "auto",
        "quality": ask.quality or "auto",
        "background": ask.background or "auto",
        "output_format": format_name(image.mediaType),
    }
    if kind == "partial_image":
        payload["partial_image_index"] = index or 0
    else:
        payload["usage"] = _usage(usage) or {}
        if routing is not None:
            payload["x_eugene_plexus"] = routing
    return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n"

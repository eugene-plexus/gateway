"""Bounded inline image validation. Never resolves a URL or reflects its value."""

from __future__ import annotations

import base64
import binascii
import io
import warnings
from typing import Any

from PIL import Image

#: The default for the gateway's `maxImagesPerRequest`. It was a fixed four
#: until 2026-09-23, when it was measured to refuse a Claude Code session on
#: every turn after its fifth screenshot: a client resends its history, so the
#: count is the whole conversation's.
DEFAULT_MAX_IMAGES = 12
#: The most the setting accepts, and the inference-driver's own ceiling.
MAX_IMAGES_CEILING = 64
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_DIMENSION = 8192


class ImageRefusal(ValueError):
    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"{field}: {reason}.")


def content_wire(content: Any) -> Any:
    return (
        content.model_dump(mode="json", exclude_none=True)
        if hasattr(content, "model_dump")
        else content
    )


def has_images(messages: Any) -> bool:
    return any(
        isinstance(content := content_wire(message.content), list)
        and any(part.get("type") == "image_url" for part in content)
        for message in messages or []
    )


class ImageBudget:
    """One request's image count and decoded total, whichever door it came in by.

    The Anthropic door names a picture in its caller's own coordinates and
    the OpenAI door in its, so the limits live here and the field name is
    the caller's to supply.
    """

    def __init__(self, max_images: int = DEFAULT_MAX_IMAGES) -> None:
        self.count = 0
        self.total = 0
        self.max_images = max_images

    def admit(self, url: str, field: str) -> None:
        self.count += 1
        if self.count > self.max_images:
            raise ImageRefusal(
                field,
                f"at most {self.max_images} images are allowed per request, counted across "
                "the whole conversation (the gateway's maxImagesPerRequest)",
            )
        self.total += _validate_image(url, field)
        if self.total > MAX_TOTAL_BYTES:
            raise ImageRefusal(field, "images exceed the 10 MiB decoded request limit")


def max_images(store: Any) -> int:
    """The operator's limit, or the default when the store says nothing usable."""
    try:
        value = int(store.get("maxImagesPerRequest")) if store is not None else DEFAULT_MAX_IMAGES
    except (TypeError, ValueError):
        return DEFAULT_MAX_IMAGES
    return min(max(value, 1), MAX_IMAGES_CEILING)


def validate_messages(messages: Any, max_images: int = DEFAULT_MAX_IMAGES) -> None:
    """Validate typed messages, then flatten only arrays consisting entirely of text."""
    budget = ImageBudget(max_images)
    for index, message in enumerate(messages or []):
        content = content_wire(message.content)
        if not isinstance(content, list):
            continue
        contains_image = False
        for part_index, part in enumerate(content):
            if part["type"] == "text":
                continue
            field = f"messages[{index}].content[{part_index}].image_url"
            if getattr(message.role, "value", message.role) != "user":
                raise ImageRefusal(field, "images are supported only on user messages")
            contains_image = True
            budget.admit(part["image_url"]["url"], field)
        if not contains_image:
            message.content = "".join(part["text"] for part in content)


# Anthropic accepts four image types. Local engines, and this gateway's own
# contract, take two; the other two are re-encoded rather than refused,
# because Claude Code sends a small `.webp` or `.gif` exactly as it found it
# (captured 2026-09-23) and a refusal would fail its `Read` of either.
# PNG is lossless, so the model sees the same pixels the caller sent.
_REENCODED = {"image/gif": "GIF", "image/webp": "WEBP"}


def data_url_from_base64(media_type: Any, data: Any, field: str) -> str:
    """An Anthropic base64 image source as the data URL the shared path carries."""
    if not isinstance(media_type, str) or not isinstance(data, str) or not data:
        raise ImageRefusal(field, "an image source needs base64 `data` and a `media_type`")
    if media_type in ("image/png", "image/jpeg"):
        return f"data:{media_type};base64,{data}"
    fmt = _REENCODED.get(media_type)
    if fmt is None:
        raise ImageRefusal(field, "use a PNG, JPEG, GIF or WebP image")
    if len(data) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ImageRefusal(field, "image exceeds the 5 MiB decoded limit")
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error):
        raise ImageRefusal(field, "image has invalid base64") from None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw), formats=[fmt]) as picture:
                width, height = picture.size
                if max(width, height) > MAX_DIMENSION or width * height > MAX_PIXELS:
                    raise ImageRefusal(
                        field, "image exceeds 8192 pixels per side or 16 million pixels"
                    )
                if getattr(picture, "is_animated", False):
                    raise ImageRefusal(field, "animated images are not supported")
                out = io.BytesIO()
                picture.save(out, format="PNG")
    except ImageRefusal:
        raise
    except (
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        raise ImageRefusal(field, "image is invalid or does not match its GIF/WebP type") from None
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def _validate_image(url: str, field: str) -> int:
    header, separator, encoded = url.partition(",")
    formats = {"data:image/png;base64": "PNG", "data:image/jpeg;base64": "JPEG"}
    if not separator or header not in formats:
        raise ImageRefusal(field, "use an inline base64 PNG or JPEG; URLs are not fetched")
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ImageRefusal(field, "image exceeds the 5 MiB decoded limit")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ImageRefusal(field, "image has invalid base64") from None
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ImageRefusal(field, "image is empty or exceeds the 5 MiB decoded limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=[formats[header]]) as picture:
                width, height = picture.size
                if (
                    max(width, height) > MAX_DIMENSION
                    or width * height > MAX_PIXELS
                    or min(width, height) < 1
                ):
                    raise ImageRefusal(
                        field, "image exceeds 8192 pixels per side or 16 million pixels"
                    )
                if getattr(picture, "is_animated", False):
                    raise ImageRefusal(field, "animated images are not supported")
                picture.verify()
            # JPEG verify alone does not decode truncated pixel data.
            with Image.open(io.BytesIO(data), formats=[formats[header]]) as picture:
                picture.load()
    except ImageRefusal:
        raise
    except (
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        raise ImageRefusal(field, "image is invalid or does not match its PNG/JPEG type") from None
    return len(data)

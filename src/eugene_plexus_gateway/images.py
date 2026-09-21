"""Bounded inline image validation. Never resolves a URL or reflects its value."""

from __future__ import annotations

import base64
import binascii
import io
import warnings
from typing import Any

from PIL import Image

MAX_IMAGES = 4
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


def validate_messages(messages: Any) -> None:
    """Validate typed messages, then flatten only arrays consisting entirely of text."""
    count = total = 0
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
            count += 1
            if count > MAX_IMAGES:
                raise ImageRefusal(field, "at most four images are allowed per request")
            total += _validate_image(part["image_url"]["url"], field)
            if total > MAX_TOTAL_BYTES:
                raise ImageRefusal(field, "images exceed the 10 MiB decoded request limit")
        if not contains_image:
            message.content = "".join(part["text"] for part in content)


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

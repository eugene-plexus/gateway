"""Bounded inline attachment validation: images, audio and PDFs.

Never resolves a URL or reflects a value it was sent. The name is the
module's history: images came first (2026-09-11), audio and files in P2.
"""

from __future__ import annotations

import base64
import binascii
import io
import warnings
from typing import Any

from PIL import Image
from pydantic import RootModel

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
#: One audio clip or file, decoded (P2, 2026-09-28).
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
#: Every attachment in a request together, images included: what fits in the
#: 16 MiB JSON body once base64 has grown it by a third.
MAX_ATTACHMENTS_TOTAL = 11 * 1024 * 1024
#: A content part's `type` and the input kind it asks the model to take.
PART_KINDS = {"image_url": "image", "input_audio": "audio", "file": "file"}
_PDF_DATA_URL = "data:application/pdf;base64,"


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


def attachment_kinds(messages: Any) -> frozenset[str]:
    """Which kinds of input a request asks its model to take: `image`,
    `audio`, `file`. Empty for text alone. Routing reads this: a request
    goes only to a backend that confirms every kind in it."""
    return frozenset(
        PART_KINDS[part.get("type")]
        for message in messages or []
        if isinstance(content := content_wire(message.content), list)
        for part in content
        if part.get("type") in PART_KINDS
    )


def moved_note(kinds: list[str]) -> tuple[str, str]:
    """What a tool message says about attachments its tool returned, and the
    label they carry on the next user message.

    A `tool` message carries text only and attachments ride on user
    messages, so they move, and both ends say so: a model reading an empty
    result and then an unexplained picture cannot tell which call made it.
    """
    nouns = {"image": "image", "file": "document", "audio": "recording"}
    word = nouns[kinds[0]] if len(set(kinds)) == 1 else "attachment"
    if len(kinds) == 1:
        article = "an" if word[0] in "aeiou" else "a"
        return (
            f"[The tool returned {article} {word}; it is attached to the next user message.]",
            f"The {word}",
        )
    return (
        f"[The tool returned {len(kinds)} {word}s; they are attached to the next user message.]",
        f"The {word}s",
    )


def has_images(messages: Any) -> bool:
    return "image" in attachment_kinds(messages)


class ImageBudget:
    """One request's attachments -- image count, image total and the total
    of every kind -- whichever door they came in by.

    The Anthropic door names a picture in its caller's own coordinates and
    the OpenAI door in its, so the limits live here and the field name is
    the caller's to supply.
    """

    def __init__(self, max_images: int = DEFAULT_MAX_IMAGES) -> None:
        self.count = 0
        self.total = 0
        self.attached = 0
        self.max_images = max_images

    def admit(self, url: str, field: str) -> None:
        self.count += 1
        if self.count > self.max_images:
            raise ImageRefusal(
                field,
                f"at most {self.max_images} images are allowed per request, counted across "
                "the whole conversation (the gateway's maxImagesPerRequest)",
            )
        size = _validate_image(url, field)
        self.total += size
        if self.total > MAX_TOTAL_BYTES:
            raise ImageRefusal(field, "images exceed the 10 MiB decoded request limit")
        self._attach(size, field)

    def admit_audio(self, data: str, fmt: str, field: str) -> None:
        self._attach(validate_audio(data, fmt, field), field)

    def admit_file(self, file: dict[str, Any], field: str) -> str:
        """The file's PDF as the data URL every backend accepts."""
        url, size = validate_file(file, field)
        self._attach(size, field)
        return url

    def _attach(self, size: int, field: str) -> None:
        self.attached += size
        if self.attached > MAX_ATTACHMENTS_TOTAL:
            raise ImageRefusal(field, "attachments exceed the 11 MiB decoded request limit")


def max_images(store: Any) -> int:
    """The operator's limit, or the default when the store says nothing usable."""
    try:
        value = int(store.get("maxImagesPerRequest")) if store is not None else DEFAULT_MAX_IMAGES
    except (TypeError, ValueError):
        return DEFAULT_MAX_IMAGES
    return min(max(value, 1), MAX_IMAGES_CEILING)


def _typed_parts(content: Any) -> list[Any] | None:
    """The content's own part objects, so a normalised value can be written back."""
    while isinstance(content, RootModel):
        content = content.root
    return content if isinstance(content, list) else None


def validate_messages(messages: Any, max_images: int = DEFAULT_MAX_IMAGES) -> None:
    """Validate typed messages, then flatten only arrays consisting entirely of text.

    A PDF sent as bare base64 is rewritten in place as the data URL every
    backend accepts; nothing else is changed.
    """
    budget = ImageBudget(max_images)
    for index, message in enumerate(messages or []):
        content = content_wire(message.content)
        if not isinstance(content, list):
            continue
        typed = _typed_parts(message.content)
        contains_attachment = False
        for part_index, part in enumerate(content):
            kind = PART_KINDS.get(part["type"])
            if kind is None:
                continue
            field = f"messages[{index}].content[{part_index}].{part['type']}"
            if getattr(message.role, "value", message.role) != "user":
                raise ImageRefusal(
                    field,
                    "images are supported only on user messages"
                    if kind == "image"
                    else "attachments are supported only on user messages",
                )
            contains_attachment = True
            if kind == "image":
                budget.admit(part["image_url"]["url"], field)
            elif kind == "audio":
                budget.admit_audio(
                    part["input_audio"]["data"], part["input_audio"]["format"], field
                )
            else:
                url = budget.admit_file(part["file"], field)
                if url != part["file"].get("file_data") and typed is not None:
                    typed[part_index].file.file_data = url
        if not contains_attachment:
            message.content = "".join(part["text"] for part in content)


def validate_audio(data: str, fmt: str, field: str) -> int:
    """An `input_audio` clip's decoded size, once its bytes match its format."""
    if len(data) > 4 * ((MAX_ATTACHMENT_BYTES + 2) // 3):
        raise ImageRefusal(field, "audio exceeds the 10 MiB decoded limit")
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error):
        raise ImageRefusal(
            field, "audio has invalid base64; send it without a data: prefix"
        ) from None
    if not raw or len(raw) > MAX_ATTACHMENT_BYTES:
        raise ImageRefusal(field, "audio is empty or exceeds the 10 MiB decoded limit")
    if fmt == "wav":
        matches = raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    else:
        # An ID3 tag, or straight into an MPEG frame: eleven set sync bits.
        matches = raw[:3] == b"ID3" or (len(raw) > 1 and raw[0] == 0xFF and raw[1] & 0xE0 == 0xE0)
    if not matches:
        raise ImageRefusal(field, f"audio does not match its declared {fmt} format")
    return len(raw)


def validate_file(file: Any, field: str) -> tuple[str, int]:
    """A `file` part's PDF as a data URL, and its decoded size.

    Bare base64 is accepted: OpenAI's schema calls the field base64, and
    OpenRouter refuses anything but the data URL (measured 2026-09-28).
    """
    if not isinstance(file, dict):
        raise ImageRefusal(field, "must be an object")
    if file.get("file_id"):
        raise ImageRefusal(
            f"{field}.file_id",
            "names an uploaded file, and this install has no file store; send file_data",
        )
    data = file.get("file_data")
    if not isinstance(data, str) or not data:
        raise ImageRefusal(f"{field}.file_data", "is required: send the PDF inline")
    if data.startswith("data:"):
        header, separator, encoded = data.partition(",")
        if not separator or f"{header}," != _PDF_DATA_URL:
            raise ImageRefusal(
                f"{field}.file_data",
                "use a base64 PDF data URL (data:application/pdf;base64,...); "
                "other file types are not supported",
            )
    else:
        encoded = data
    if len(encoded) > 4 * ((MAX_ATTACHMENT_BYTES + 2) // 3):
        raise ImageRefusal(f"{field}.file_data", "file exceeds the 10 MiB decoded limit")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ImageRefusal(
            f"{field}.file_data", "is neither a PDF data URL nor base64; URLs are not fetched"
        ) from None
    if not raw.startswith(b"%PDF-"):
        raise ImageRefusal(f"{field}.file_data", "is not a PDF")
    return _PDF_DATA_URL + encoded, len(raw)


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

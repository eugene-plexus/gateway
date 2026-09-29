"""/v1/moderations (P6, 2026-09-28): OpenAI's request, read and checked.

OpenAI's `input` is a string, an array of strings (one result each), or an
array of parts (one result for the whole): `text`, and `image_url` whose
`url` must be a `data:` URL here (A4: the gateway never forwards a remote
one, though OpenAI would fetch it). Everything else is refused naming the
field, as the other doors refuse an unknown one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._generated.driver_models import ModerationPart, ModerationPartType
from .images import ImageBudget, ImageRefusal

_FIELDS = {"model", "input"}
_PART_FIELDS = {"type", "text", "image_url"}


class Refusal(ValueError):
    """A request this door refuses, naming the field."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = f"{field}: {message}"


@dataclass(frozen=True)
class ModerationAsk:
    """The request, read: `model` left out is None (P6-1)."""

    model: str | None
    texts: list[str] | None
    parts: list[ModerationPart] | None


def parse(body: Any, *, max_images: int) -> ModerationAsk:
    if not isinstance(body, dict):
        raise Refusal("body", "must be a JSON object")
    for key in body:
        if key not in _FIELDS:
            raise Refusal(str(key), "is not a field of this request")
    model = body.get("model")
    if model is not None and (not isinstance(model, str) or not model):
        raise Refusal("model", "must be a model id")
    raw = body.get("input")
    if isinstance(raw, str):
        return ModerationAsk(model, [raw], None)
    shape = "must be a string, an array of strings, or an array of parts"
    if not isinstance(raw, list) or not raw:
        raise Refusal("input", shape)
    if all(isinstance(item, str) for item in raw):
        return ModerationAsk(model, list(raw), None)
    if not all(isinstance(item, dict) for item in raw):
        raise Refusal("input", "mixes strings and parts; send one or the other")
    budget = ImageBudget(max_images)
    parts: list[ModerationPart] = []
    for index, part in enumerate(raw):
        field = f"input[{index}]"
        for key in part:
            if key not in _PART_FIELDS:
                raise Refusal(f"{field}.{key}", "is not a field of a part")
        kind = part.get("type")
        if kind == "text":
            if not isinstance(part.get("text"), str):
                raise Refusal(f"{field}.text", "a text part carries its text")
            parts.append(ModerationPart(type=ModerationPartType.text, text=part["text"]))
        elif kind == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else None
            if not isinstance(url, str):
                raise Refusal(f"{field}.image_url.url", "an image part carries its URL")
            try:
                budget.admit(url, f"{field}.image_url.url")
            except ImageRefusal as e:
                raise Refusal(e.field, e.reason) from None
            parts.append(ModerationPart(type=ModerationPartType.image, image=url))
        else:
            raise Refusal(f"{field}.type", "must be text or image_url")
    return ModerationAsk(model, None, parts)

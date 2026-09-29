"""/v1/completions (P6, 2026-09-28): OpenAI's legacy request, read and checked.

The door is a translation at the edge, as `/v1/messages` and `/v1/responses`
are: the sampling fields become a chat-shaped request so the shared path
(`_prepare`, the tiers, the commit point, the recording) serves it
unchanged, and the prompt itself travels to the driver as
`GenerateRequest.completion`, continued as written. The chat request's one
message is a stand-in the driver never sees.

Refused naming the field, in P6: `n` or `best_of` above 1 (one answer per
request), `echo`, `logprobs`, a `prompt` of more than one string or of token
ids, and anything OpenAI's form does not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from . import chat_contract
from ._generated.driver_models import CompletionPrompt
from ._generated.models import ChatCompletionRequest

_FIELDS = {
    "model",
    "prompt",
    "suffix",
    "max_tokens",
    "temperature",
    "top_p",
    "n",
    "stream",
    "stream_options",
    "logprobs",
    "echo",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "best_of",
    "logit_bias",
    "seed",
    "user",
}
#: Carried to the chat-shaped request under the same names.
_SAMPLING = (
    "max_tokens",
    "temperature",
    "top_p",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "seed",
    "user",
)


class Refusal(ValueError):
    """A request this door refuses, naming the field."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = f"{field}: {message}"


@dataclass(frozen=True)
class CompletionAsk:
    body: ChatCompletionRequest
    prompt: CompletionPrompt
    stream: bool
    include_usage: bool


def parse(raw: Any, *, max_images: int) -> CompletionAsk:
    if not isinstance(raw, dict):
        raise Refusal("body", "must be a JSON object")
    for key in raw:
        if key not in _FIELDS:
            raise Refusal(str(key), "is not a field of this request")
    prompt = raw.get("prompt")
    if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str):
        prompt = prompt[0]
    if isinstance(prompt, list):
        raise Refusal(
            "prompt",
            "one string only here: send each prompt as its own request (token ids are not taken)",
        )
    if not isinstance(prompt, str):
        raise Refusal("prompt", "is required: the text to continue")
    suffix = raw.get("suffix")
    if suffix is not None and not isinstance(suffix, str):
        raise Refusal("suffix", "must be a string")
    for field in ("n", "best_of"):
        if raw.get(field) not in (None, 1):
            raise Refusal(field, "only 1 here: one answer per request")
    if raw.get("echo") not in (None, False):
        raise Refusal(
            "echo", "is not carried: llama-server ignores it (measured), so it is refused"
        )
    if raw.get("logprobs") is not None:
        raise Refusal("logprobs", "is not carried on this door yet")
    stream = raw.get("stream", False)
    if not isinstance(stream, bool):
        raise Refusal("stream", "must be true or false")
    options = raw.get("stream_options")
    include_usage = False
    if options is not None:
        if not isinstance(options, dict) or set(options) - {"include_usage"}:
            raise Refusal("stream_options", "takes include_usage only")
        include_usage = options.get("include_usage") is True
    chat: dict[str, Any] = {
        "model": raw.get("model"),
        # A stand-in: the driver is sent `completion`, never this message.
        "messages": [{"role": "user", "content": prompt or " "}],
        "stream": stream,
    }
    if stream:
        chat["stream_options"] = {"include_usage": True}
    chat.update({k: raw[k] for k in _SAMPLING if raw.get(k) is not None})
    try:
        body = chat_contract.parse_request(chat, max_images=max_images)
    except chat_contract.Refusal as exc:
        raise Refusal(exc.field, exc.message.split(": ", 1)[-1]) from None
    except ValidationError as exc:
        where = ".".join(str(p) for p in exc.errors()[0].get("loc", ())) or "body"
        raise Refusal(where, "has an invalid value") from None
    return CompletionAsk(
        body=body,
        prompt=CompletionPrompt(prompt=prompt, suffix=suffix),
        stream=stream,
        include_usage=include_usage,
    )

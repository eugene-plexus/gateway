"""Where a system message may stand in the prompt a backend is sent (PC2).

**Claude Code sends `system` messages inside the conversation**, after user
turns (`# Environment ...`, `<total_tokens>...`), and the Responses door
turns a `developer` message into `system` in place. The Qwen 3.5-and-later
chat templates refuse a system message anywhere but first ("System message
must be at the beginning"), so llama-server answered 500 and Claude Code
could not use three of the five starter classes (measured 2026-10-02 on
Qwen3.5-4B, Qwen3.6-35B-A3B and Qwen3.8-27B; gateway#6). Eight template
families accept the shape below, the same three included.

**`user_turn`, the default:** the leading run of system messages is one
system message, and every later one becomes a user turn **in its own
place**, its text wrapped as `<system-reminder>`, the form Claude Code uses
for reminders it puts in user turns. Not merged into the first system
message: that would change the start of the prompt on every turn, and an
engine's prompt cache can reuse a prompt only up to its first difference.
**`system`:** every message as the client sent it, for an operator whose
backends all accept that and who wants the role kept. Troy's call
(2026-10-02): the operator chooses.
"""

from __future__ import annotations

from typing import Any

from ._generated.driver_models import Message, Role

USER_TURN = "user_turn"
SYSTEM = "system"
PLACEMENTS = [USER_TURN, SYSTEM]


def _text(content: Any) -> str | None:
    """A system message's text, or None when it is not text alone."""
    # The generated model wraps a part list (and may wrap a string) in a
    # RootModel.
    content = getattr(content, "root", content)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            # The generated `Message` parses parts into models; a caller
            # building one by hand may pass plain dicts.
            if hasattr(part, "model_dump"):
                part = part.model_dump(mode="json", exclude_none=True)
            if not isinstance(part, dict) or part.get("type") != "text":
                return None
            parts.append(str(part.get("text") or ""))
        return "\n".join(parts)
    return None


def place(messages: list[Message], placement: str | None) -> list[Message]:
    """`messages` with each system message where `placement` puts it."""
    if placement == SYSTEM:
        return messages
    out: list[Message] = []
    leading = True
    for message in messages:
        if message.role != Role.system:
            leading = False
            out.append(message)
            continue
        text = _text(message.content)
        if text is None:
            # Something other than text in a system message: leave it be
            # rather than drop what we cannot read.
            out.append(message)
            continue
        if leading:
            if out:
                first = _text(out[0].content) or ""
                out[0] = out[0].model_copy(
                    update={"content": f"{first}\n\n{text}" if first else text}
                )
            else:
                out.append(message.model_copy(update={"content": text}))
            continue
        out.append(
            Message(role=Role.user, content=f"<system-reminder>\n{text}\n</system-reminder>")
        )
    return out

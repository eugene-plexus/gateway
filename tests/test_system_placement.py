"""Where a system message stands in the prompt a backend gets (PC2, gateway#6)."""

from __future__ import annotations

from eugene_plexus_gateway import system_placement
from eugene_plexus_gateway._generated.driver_models import Message, Role


def m(role: str, content: object) -> Message:
    return Message(role=Role(role), content=content)


def roles(messages: list[Message]) -> list[str]:
    return [x.role.value for x in messages]


def test_a_later_system_message_becomes_a_user_turn_where_it_was() -> None:
    out = system_placement.place(
        [m("system", "rules"), m("user", "hi"), m("system", "env"), m("user", "go")], None
    )
    assert roles(out) == ["system", "user", "user", "user"]
    assert out[2].content == "<system-reminder>\nenv\n</system-reminder>"
    assert out[0].content == "rules"


def test_a_leading_run_of_system_messages_is_one() -> None:
    """Two at the start is the second at index 1, which Qwen 3.5+ also refuses."""
    out = system_placement.place([m("system", "a"), m("system", "b"), m("user", "hi")], "user_turn")
    assert roles(out) == ["system", "user"]
    assert out[0].content == "a\n\nb"


def test_text_parts_are_read_as_text() -> None:
    out = system_placement.place(
        [
            m("user", "hi"),
            m("system", [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]),
        ],
        None,
    )
    assert out[1].role.value == "user"
    assert out[1].content == "<system-reminder>\nx\ny\n</system-reminder>"


def test_a_system_message_that_is_not_text_is_left_alone() -> None:
    odd = m("system", [{"type": "image_url", "image_url": {"url": "data:,"}}])
    out = system_placement.place([m("user", "hi"), odd], None)
    assert out[1] is odd


def test_system_placement_sends_everything_as_the_client_did() -> None:
    sent = [m("system", "a"), m("system", "b"), m("user", "hi"), m("system", "env")]
    assert system_placement.place(sent, "system") == sent


def test_each_turn_is_a_prefix_of_the_next() -> None:
    """**The property the engine's cache needs.** A conversation that grows
    by appending reaches the backend as a prompt that grows by appending:
    nothing earlier changes, so every earlier turn can be reused. Merging a
    later system message into the first one would break exactly this."""
    turn1 = [m("system", "rules"), m("user", "hi"), m("system", "env 1")]
    turn2 = [*turn1, m("assistant", "hello"), m("user", "next"), m("system", "env 2")]
    a = system_placement.place(turn1, None)
    b = system_placement.place(turn2, None)
    assert b[: len(a)] == a

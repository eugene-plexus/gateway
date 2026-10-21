"""The closed role catalogue and advisory readings of driver capabilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ._generated.models import (
    ModelRole,
    ModelRoleCapability,
    ModelRoleCapabilityStatus,
    ModelRoleInfo,
)

if TYPE_CHECKING:
    from .routing import _Backend


ROLE_DETAILS: dict[ModelRole, tuple[str, ModelRoleCapability]] = {
    ModelRole.chat: ("Chat", ModelRoleCapability.chat),
    ModelRole.agentic: ("Agentic (tools and code)", ModelRoleCapability.toolCalling),
    ModelRole.vision: ("Sees images", ModelRoleCapability.imageInput),
    ModelRole.reads_pages: ("Reads pages (OCR)", ModelRoleCapability.imageInput),
    ModelRole.image: ("Makes images", ModelRoleCapability.image),
    ModelRole.speech_to_text: ("Hears speech", ModelRoleCapability.transcription),
    ModelRole.text_to_speech: ("Speaks", ModelRoleCapability.speech),
    ModelRole.music: ("Makes music", ModelRoleCapability.audioOutput),
    ModelRole.embeddings: ("Embeddings", ModelRoleCapability.embeddings),
    ModelRole.rerank: ("Reranks", ModelRoleCapability.rerank),
    ModelRole.decision: ("Decisions (yes/no)", ModelRoleCapability.decision),
}


def role_info(model: str) -> ModelRoleInfo | None:
    if not model.startswith("role:"):
        return None
    try:
        role = ModelRole(model.removeprefix("role:"))
    except ValueError:
        return None
    return ModelRoleInfo(role=role, label=ROLE_DETAILS[role][0])


def capability_finding(
    backends: Sequence[_Backend], capability: ModelRoleCapability
) -> tuple[ModelRoleCapabilityStatus, str | None]:
    """Report evidence, without using the routing policy's unknown-as-yes defaults.

    One supporting replica establishes support. An unreported replica keeps an
    otherwise negative finding unknown. No assignment or inference is gated here.
    """
    findings: list[bool | None] = []
    for backend in backends:
        model = backend.model
        surfaces = model.surfaces if model is not None else []
        if capability in {
            ModelRoleCapability.toolCalling,
            ModelRoleCapability.imageInput,
            ModelRoleCapability.audioOutput,
        }:
            needs_chat = capability != ModelRoleCapability.audioOutput
            if needs_chat and not surfaces:
                findings.append(None)
            elif needs_chat and "chat" not in surfaces:
                findings.append(False)
            else:
                findings.append(getattr(backend.caps, capability.value, None))
        else:
            surface = (
                "decisions" if capability == ModelRoleCapability.decision else capability.value
            )
            findings.append(surface in surfaces if surfaces else None)
    if any(f is True for f in findings):
        return ModelRoleCapabilityStatus.yes, None
    if not findings:
        return ModelRoleCapabilityStatus.unknown, "No reachable driver lists this model."
    if any(f is None for f in findings):
        return (
            ModelRoleCapabilityStatus.unknown,
            f"A driver listing does not report {capability.value} for this model.",
        )
    return (
        ModelRoleCapabilityStatus.no,
        f"The driver listings explicitly lack the required {capability.value} "
        "capability or surface.",
    )

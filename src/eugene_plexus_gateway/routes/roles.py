"""The install's model roles, read from the existing slot resolution."""

from fastapi import APIRouter, Request

from .._generated.models import ModelRoleDefinition, ModelRoleList, ModelRoleModel
from ..model_roles import ROLE_DETAILS, capability_finding
from ..routing import RoutingTable

router = APIRouter(tags=["roles"])


@router.get("/v1/roles", response_model=ModelRoleList)
async def list_roles(request: Request) -> ModelRoleList:
    table: RoutingTable | None = getattr(request.app.state, "routing", None)
    roles = []
    for role, (label, capability) in ROLE_DETAILS.items():
        models = []
        if table is not None:
            resolution = table.resolve(f"role:{role.value}")
            if resolution.configured:
                for tier in resolution.tiers:
                    if tier.target == resolution.model:
                        continue
                    status, reason = capability_finding(tier.backends, capability)
                    models.append(
                        ModelRoleModel(model=tier.target, capability=status, reason=reason)
                    )
        roles.append(
            ModelRoleDefinition(
                role=role, label=label, requiredCapability=capability, models=models
            )
        )
    return ModelRoleList(roles=roles)

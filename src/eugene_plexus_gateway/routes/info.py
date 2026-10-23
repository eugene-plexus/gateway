"""The last observed Library state; this read starts no network work."""

from fastapi import APIRouter, Request

from .._generated.models import GatewayInfo, GatewayLibraryStatus
from ..library_cache import LibraryCache
from ..profiles import ProfileDefaults

router = APIRouter(tags=["meta"])


@router.get("/v1/info", response_model=GatewayInfo)
async def info(request: Request) -> GatewayInfo:
    profiles: ProfileDefaults | None = getattr(request.app.state, "profile_defaults", None)
    saved: LibraryCache = profiles.library if profiles else request.app.state.library_cache
    return GatewayInfo(
        library=GatewayLibraryStatus(
            reachable=saved.reachable,
            since=saved.since,
            cause=saved.cause,
            modelsOnSavedSettings=profiles.models_on_saved_settings() if profiles else 0,
        )
    )

"""OpenAI's Files and Uploads APIs, and Anthropic's Files API, over the install's store (row 53).

**Whose file it is.** Every file belongs to one owner, decided here from
what this gateway verified and never from the request's words:

* `person:<id>` when the root verified a person for an app's key
  (`X-Eugene-Person-Token`, checked with the admission authority);
* `key:<lineage>` for a client key with no person -- the key's lineage
  when the authority reports one (row 24), so a rotated key keeps its
  files, else the key's id;
* `install` for the owner's own console session;
* `component:<sub>` for a service token of one of this machine's components.

The store believes that name only beside this gateway's service token,
and answers another owner's id exactly as a missing one.

**Client admission.** These doors are a check against the admission
authority, never a reservation: a file request runs no model, so it
holds no lease and counts against no rate. The check is made through the
request's own `ClientRequest` (`check`), the same call the inference
doors make first, so a revoked key, a removed person or a person token
without its app key is refused here as there. They are not in
`CLIENT_ADMISSION_PATHS` (admission.py belongs to another row now); the
paths are bounded by `FilesBodyLimit` below instead of the shared one.

**Which shape.** A request with `anthropic-version` is Anthropic's Files
API (every Anthropic SDK sends that header); anything else is OpenAI's.
Both read and write the same files.

**`file_id` in a request** (`resolve_file_ids`): on the chat, Responses
and Messages doors an id is read from the caller's own files and carried
inline -- a PDF as file data, an image as an image, text as text -- within
the attachment limits those doors already have. Another owner's id is
refused as one that does not exist.
"""

from __future__ import annotations

import base64
import contextlib
import json
import mimetypes
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials
from starlette.datastructures import UploadFile
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import admission, door_paths, images, tokens
from .store_client import StoreClient, StoreRefusal, StoreUnavailable

router = APIRouter(tags=["files"])

#: A file sent whole, and one part of an upload: the gateway's body limit.
MAX_FILE_BYTES = 16 * 1024 * 1024
#: The multipart wrapping around that file: boundaries, headers, form fields.
_FORM_ROOM = 256 * 1024
_JSON_BYTES = 1024 * 1024
PURPOSES = ("assistants", "batch", "fine-tune", "vision", "user_data", "evals")
#: OpenAI's bounds on `expires_after.seconds`. A file with none is kept until
#: deleted, as OpenAI keeps it.
EXPIRES_MIN, EXPIRES_MAX = 3600, 30 * 24 * 3600

#: Every path these doors answer, as `door_paths` templates, with the most
#: a body may be.
BODY_LIMITS: dict[str, int] = {
    "/v1/files": MAX_FILE_BYTES + _FORM_ROOM,
    "/v1/uploads": _JSON_BYTES,
    "/v1/uploads/{upload_id}/parts": MAX_FILE_BYTES + _FORM_ROOM,
    "/v1/uploads/{upload_id}/complete": _JSON_BYTES,
    "/v1/uploads/{upload_id}/cancel": _JSON_BYTES,
}


class DoorRefusal(Exception):
    """A refusal said in the asking door's own envelope."""

    def __init__(
        self, status: int, message: str, *, param: str | None = None, code: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param
        self.code = code


def is_anthropic(request: Request | Scope) -> bool:
    headers = request.headers if isinstance(request, Request) else _scope_headers(request)
    return bool(headers.get("anthropic-version"))


def _scope_headers(scope: Scope) -> dict[str, str]:
    return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}


_OPENAI_TYPE = {401: "authentication_error", 403: "permission_error", 404: "invalid_request_error"}
_ANTHROPIC_TYPE = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    503: "overloaded_error",
}


def error_response(anthropic: bool, refusal: DoorRefusal) -> JSONResponse:
    status = refusal.status
    if anthropic:
        body: dict[str, Any] = {
            "type": "error",
            "error": {"type": _ANTHROPIC_TYPE.get(status, "api_error"), "message": refusal.message},
        }
    else:
        kind = (
            "service_unavailable"
            if status >= 500
            else _OPENAI_TYPE.get(status, "invalid_request_error")
        )
        body = {
            "error": {
                "message": refusal.message,
                "type": kind,
                "param": refusal.param,
                "code": refusal.code,
            }
        }
    return JSONResponse(body, status_code=status)


def _unavailable(exc: StoreUnavailable) -> DoorRefusal:
    return DoorRefusal(503, exc.message(), code="store_unavailable")


def _refused(exc: StoreRefusal, *, kind: str, ident: str | None = None) -> DoorRefusal:
    if exc.status == 404:
        name = {"file": "File", "upload": "Upload", "part": "Part"}.get(kind, kind)
        return DoorRefusal(404, f"No such {name} object: {ident}", param="id")
    if exc.status in (401,) or exc.title == "Only the gateway":
        # The store refused this gateway's own credential: the install's
        # fault, not the caller's.
        return DoorRefusal(
            503,
            StoreUnavailable(f"it refused the gateway's credential ({exc.detail})").message(),
            code="store_unavailable",
        )
    return DoorRefusal(exc.status, exc.detail)


# --------------------------------------------------------------------------- #
# Who owns what this request keeps or reads
# --------------------------------------------------------------------------- #


def _credential(request: Request, anthropic: bool) -> str | None:
    if anthropic:
        api_key = request.headers.get("x-api-key")
        if api_key and api_key.strip():
            return api_key.strip()
    header = request.headers.get("authorization") or ""
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()
    return None


async def owner_for(request: Request, *, anthropic: bool) -> str:
    """The owner this request acts for, verified; `DoorRefusal` when it cannot be."""
    from .dependencies import front_door_claims, require_authorized

    auth = request.app.state.auth_state
    if auth.auth_disabled:
        return "install"
    token = _credential(request, anthropic)
    if token is None:
        raise DoorRefusal(
            401,
            "No credential. Send an Eugene Plexus client key as `Authorization: Bearer`"
            + (" or `x-api-key`." if anthropic else "."),
            code="invalid_api_key",
        )
    try:
        claims = front_door_claims(auth, token)
    except tokens.TokenError as exc:
        raise DoorRefusal(401, f"That key was rejected: {exc}", code="invalid_api_key") from exc
    if claims.is_session:
        if request.headers.get("x-eugene-person-token"):
            raise DoorRefusal(401, "A person token must accompany exactly one app key.")
        return "install"
    if claims.is_service:
        return f"component:{claims.sub}"
    context = admission.current.get()
    own = None
    if context is None:
        context = admission.ClientRequest(request.scope)
        own = admission.current.set(context)
    try:
        try:
            await require_authorized(
                request, HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            )
        except HTTPException as exc:
            detail: dict[str, Any] = exc.detail if isinstance(exc.detail, dict) else {}
            raise DoorRefusal(
                exc.status_code, str(detail.get("detail") or detail.get("title") or exc.detail)
            ) from exc
        try:
            await context.authorize(request, None)
        except admission.AdmissionFailure as exc:
            raise DoorRefusal(exc.status, exc.message) from exc
    finally:
        if own is not None:
            admission.current.reset(own)
    if context.person_id:
        return f"person:{context.person_id}"
    lineage = (context.access or {}).get("lineage") or context.key_id
    return f"key:{lineage}"


def _store(request: Request) -> StoreClient:
    store: StoreClient | None = getattr(request.app.state, "store_client", None)
    if store is None:
        settings = request.app.state.settings
        store = StoreClient(
            settings.agent_url,
            request.app.state.auth_state.service_token,
            transport=getattr(request.app.state, "store_transport", None),
        )
        request.app.state.store_client = store
    return store


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #


def _iso(seconds: int) -> str:
    return datetime.fromtimestamp(seconds, UTC).isoformat().replace("+00:00", "Z")


def openai_file(kept: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": kept["id"],
        "object": "file",
        "bytes": kept["bytes"],
        "created_at": kept["createdAt"],
        "expires_at": kept.get("expiresAt"),
        "filename": kept["filename"],
        "purpose": kept["purpose"],
        "status": "processed",
        "status_details": None,
    }


def anthropic_file(kept: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": kept["id"],
        "type": "file",
        "filename": kept["filename"],
        "mime_type": kept["mimeType"],
        "size_bytes": kept["bytes"],
        "created_at": _iso(kept["createdAt"]),
        "downloadable": True,
    }


def openai_upload(upload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": upload["id"],
        "object": "upload",
        "bytes": upload["bytes"],
        "created_at": upload["createdAt"],
        "filename": upload["filename"],
        "purpose": upload["purpose"],
        "status": upload["status"],
        "expires_at": upload["expiresAt"],
        "file": openai_file(upload["file"]) if upload.get("file") else None,
    }


# --------------------------------------------------------------------------- #
# The doors
# --------------------------------------------------------------------------- #


Handler = Callable[[Request, bool, str], Any]


async def _serve(request: Request, handler: Handler) -> Response:
    anthropic = is_anthropic(request)
    try:
        owner = await owner_for(request, anthropic=anthropic)
        answer = await handler(request, anthropic, owner)
    except DoorRefusal as refusal:
        return error_response(anthropic, refusal)
    except StoreUnavailable as exc:
        return error_response(anthropic, _unavailable(exc))
    if isinstance(answer, Response):
        return answer
    return JSONResponse(answer)


def _expires_after(form: Any) -> int | None:
    anchor = form.get("expires_after[anchor]")
    seconds = form.get("expires_after[seconds]")
    if anchor is None and seconds is None:
        return None
    if anchor not in (None, "created_at"):
        raise DoorRefusal(400, "expires_after.anchor: must be created_at.", param="expires_after")
    try:
        value = int(str(seconds))
    except ValueError:
        raise DoorRefusal(
            400, "expires_after.seconds: a whole number of seconds.", param="expires_after"
        ) from None
    if not EXPIRES_MIN <= value <= EXPIRES_MAX:
        raise DoorRefusal(
            400,
            f"expires_after.seconds: from {EXPIRES_MIN} (an hour) to {EXPIRES_MAX} (30 days), "
            "as OpenAI allows. Without it a file is kept until it is deleted.",
            param="expires_after",
        )
    return value


def _purpose(value: Any, *, anthropic: bool) -> str:
    if value is None and anthropic:
        return "user_data"
    if not isinstance(value, str) or value not in PURPOSES:
        raise DoorRefusal(400, f"purpose: must be one of {', '.join(PURPOSES)}.", param="purpose")
    return value


async def _form_file(request: Request, field: str) -> tuple[bytes, str, str | None, Any]:
    try:
        form = await request.form(max_part_size=MAX_FILE_BYTES + 1)
    except Exception as exc:
        raise DoorRefusal(400, f"body: must be multipart/form-data with a `{field}` file.") from exc
    upload = form.get(field)
    if not isinstance(upload, UploadFile):
        raise DoorRefusal(400, f"{field}: required, a file.", param=field)
    data = await upload.read()
    if len(data) > MAX_FILE_BYTES:
        raise DoorRefusal(
            413,
            f"{field}: at most 16 MiB in one request; send a larger file through /v1/uploads "
            "in parts.",
            param=field,
        )
    name = (upload.filename or "file").replace("\\", "/").rsplit("/", 1)[-1] or "file"
    media = upload.content_type
    if not media or media == "application/octet-stream":
        media = mimetypes.guess_type(name)[0] or media
    return data, name[:255], media, form


@router.post("/v1/files")
async def create_file(request: Request) -> Response:
    async def handler(request: Request, anthropic: bool, owner: str) -> Any:
        data, name, media, form = await _form_file(request, "file")
        purpose = _purpose(form.get("purpose"), anthropic=anthropic)
        try:
            kept = await _store(request).call(
                "POST",
                "/v1/files",
                owner,
                params={
                    "filename": name,
                    "purpose": purpose,
                    "mimeType": media,
                    "expiresAfterSeconds": _expires_after(form),
                },
                content=data,
            )
        except StoreRefusal as exc:
            raise _refused(exc, kind="file") from exc
        return anthropic_file(kept) if anthropic else openai_file(kept)

    return await _serve(request, handler)


def _int_param(request: Request, name: str, low: int, high: int) -> int | None:
    value = request.query_params.get(name)
    if value is None:
        return None
    try:
        number = int(value)
    except ValueError:
        raise DoorRefusal(400, f"{name}: a whole number.", param=name) from None
    if not low <= number <= high:
        raise DoorRefusal(400, f"{name}: from {low} to {high}.", param=name)
    return number


@router.get("/v1/files")
async def list_files(request: Request) -> Response:
    async def handler(request: Request, anthropic: bool, owner: str) -> Any:
        query = request.query_params
        if anthropic:
            params = {
                "limit": _int_param(request, "limit", 1, 1000) or 20,
                "after": query.get("after_id"),
                "before": query.get("before_id"),
            }
        else:
            order = query.get("order") or "desc"
            if order not in ("asc", "desc"):
                raise DoorRefusal(400, "order: asc or desc.", param="order")
            purpose = query.get("purpose")
            if purpose is not None:
                _purpose(purpose, anthropic=False)
            params = {
                "purpose": purpose,
                "limit": _int_param(request, "limit", 1, 10000) or 10000,
                "order": order,
                "after": query.get("after"),
            }
        try:
            listing = await _store(request).call("GET", "/v1/files", owner, params=params)
        except StoreRefusal as exc:
            cursor = params.get("after") or params.get("before")
            raise _refused(exc, kind="file", ident=str(cursor)) from exc
        render = anthropic_file if anthropic else openai_file
        data = [render(kept) for kept in listing["data"]]
        page: dict[str, Any] = {
            "data": data,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
            "has_more": bool(listing["hasMore"]),
        }
        return page if anthropic else {"object": "list", **page}

    return await _serve(request, handler)


@router.get("/v1/files/{file_id}")
async def retrieve_file(request: Request, file_id: str) -> Response:
    async def handler(request: Request, anthropic: bool, owner: str) -> Any:
        try:
            kept = await _store(request).call("GET", f"/v1/files/{_safe(file_id)}", owner)
        except StoreRefusal as exc:
            raise _refused(exc, kind="file", ident=file_id) from exc
        return anthropic_file(kept) if anthropic else openai_file(kept)

    return await _serve(request, handler)


@router.delete("/v1/files/{file_id}")
async def delete_file(request: Request, file_id: str) -> Response:
    async def handler(request: Request, anthropic: bool, owner: str) -> Any:
        try:
            gone = await _store(request).call("DELETE", f"/v1/files/{_safe(file_id)}", owner)
        except StoreRefusal as exc:
            raise _refused(exc, kind="file", ident=file_id) from exc
        if anthropic:
            return {"id": gone["id"], "type": "file_deleted"}
        return {"id": gone["id"], "object": "file", "deleted": True}

    return await _serve(request, handler)


@router.get("/v1/files/{file_id}/content")
async def file_content(request: Request, file_id: str) -> Response:
    async def handler(request: Request, anthropic: bool, owner: str) -> Any:
        try:
            upstream, chunks = await _store(request).stream(
                f"/v1/files/{_safe(file_id)}/content", owner
            )
        except StoreRefusal as exc:
            raise _refused(exc, kind="file", ident=file_id) from exc

        async def body() -> Any:
            try:
                async for chunk in chunks:
                    yield chunk
            finally:
                await upstream.aclose()

        headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower() in ("content-length", "x-eugene-plexus-sha256")
        }
        return StreamingResponse(
            body(),
            media_type=upstream.headers.get("content-type", "application/octet-stream"),
            headers=headers,
        )

    return await _serve(request, handler)


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = json.loads(await request.body() or b"null")
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise DoorRefusal(400, "body: must be a JSON object.", param="body")
    return body


@router.post("/v1/uploads")
async def create_upload(request: Request) -> Response:
    async def handler(request: Request, anthropic: bool, owner: str) -> Any:
        body = await _json_body(request)
        size = body.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise DoorRefusal(400, "bytes: the file's size, a whole number above 0.", param="bytes")
        name = body.get("filename")
        if not isinstance(name, str) or not name.strip():
            raise DoorRefusal(400, "filename: required.", param="filename")
        media = body.get("mime_type")
        if not isinstance(media, str) or not media:
            raise DoorRefusal(400, "mime_type: required.", param="mime_type")
        expires = body.get("expires_after")
        seconds = None
        if expires is not None:
            if not isinstance(expires, dict):
                raise DoorRefusal(400, "expires_after: an object.", param="expires_after")
            seconds = _expires_after(
                {
                    "expires_after[anchor]": expires.get("anchor"),
                    "expires_after[seconds]": expires.get("seconds"),
                }
            )
        try:
            upload = await _store(request).call(
                "POST",
                "/v1/uploads",
                owner,
                json={
                    "filename": name.strip()[:255],
                    "purpose": _purpose(body.get("purpose"), anthropic=False),
                    "bytes": size,
                    "mimeType": media[:255],
                    "expiresAfterSeconds": seconds,
                },
            )
        except StoreRefusal as exc:
            raise _refused(exc, kind="upload") from exc
        return openai_upload(upload)

    return await _serve(request, handler)


@router.post("/v1/uploads/{upload_id}/parts")
async def add_upload_part(request: Request, upload_id: str) -> Response:
    async def handler(request: Request, anthropic: bool, owner: str) -> Any:
        data, _, _, _ = await _form_file(request, "data")
        try:
            part = await _store(request).call(
                "POST", f"/v1/uploads/{_safe(upload_id)}/parts", owner, content=data
            )
        except StoreRefusal as exc:
            raise _refused(exc, kind="upload", ident=upload_id) from exc
        return {
            "id": part["id"],
            "object": "upload.part",
            "created_at": part["createdAt"],
            "upload_id": part["uploadId"],
        }

    return await _serve(request, handler)


@router.post("/v1/uploads/{upload_id}/complete")
async def complete_upload(request: Request, upload_id: str) -> Response:
    async def handler(request: Request, anthropic: bool, owner: str) -> Any:
        body = await _json_body(request)
        parts = body.get("part_ids")
        if not isinstance(parts, list) or not parts or not all(isinstance(p, str) for p in parts):
            raise DoorRefusal(400, "part_ids: the parts, in order.", param="part_ids")
        sent: dict[str, Any] = {"partIds": parts}
        for key in ("md5", "sha256"):
            if body.get(key) is not None:
                if not isinstance(body[key], str):
                    raise DoorRefusal(400, f"{key}: a hex digest.", param=key)
                sent[key] = body[key]
        try:
            upload = await _store(request).call(
                "POST", f"/v1/uploads/{_safe(upload_id)}/complete", owner, json=sent
            )
        except StoreRefusal as exc:
            raise _refused(exc, kind="upload", ident=upload_id) from exc
        return openai_upload(upload)

    return await _serve(request, handler)


@router.post("/v1/uploads/{upload_id}/cancel")
async def cancel_upload(request: Request, upload_id: str) -> Response:
    async def handler(request: Request, anthropic: bool, owner: str) -> Any:
        try:
            upload = await _store(request).call(
                "POST", f"/v1/uploads/{_safe(upload_id)}/cancel", owner
            )
        except StoreRefusal as exc:
            raise _refused(exc, kind="upload", ident=upload_id) from exc
        return openai_upload(upload)

    return await _serve(request, handler)


def _safe(ident: str) -> str:
    """An id as one path segment. Anything that is not one of the store's
    id shapes cannot name a file, so it is sent as a name that matches
    none, and the store answers it as a missing id."""
    if ident and all(c.isalnum() or c in "-_" for c in ident) and len(ident) <= 64:
        return ident
    return "not-an-id"


# --------------------------------------------------------------------------- #
# The body limit for these doors
# --------------------------------------------------------------------------- #


class FilesBodyLimit:
    """`BODY_LIMITS`, for paths that carry an id (the shared limit matches
    exact paths only). Refused in the asking door's envelope."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        limit = None
        if scope["type"] == "http" and scope.get("method") == "POST":
            for template, most in BODY_LIMITS.items():
                if door_paths.matches(path, [template]):
                    limit = most
                    break
        if limit is None:
            await self.app(scope, receive, send)
            return
        for key, value in scope.get("headers", []):
            if key == b"content-length" and value.isdigit() and int(value) > limit:
                await self._refuse(scope, receive, send, limit)
                return
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            body.extend(event.get("body", b""))
            if len(body) > limit:
                await self._refuse(scope, receive, send, limit)
                return
            if not event.get("more_body", False):
                break
        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)

    async def _refuse(self, scope: Scope, receive: Receive, send: Send, limit: int) -> None:
        refusal = DoorRefusal(
            413,
            "body: exceeds the 16 MiB request limit; send a larger file through /v1/uploads "
            "in parts.",
            param="body",
        )
        await error_response(is_anthropic(scope), refusal)(scope, receive, send)


# --------------------------------------------------------------------------- #
# file_id on the chat, Responses and Messages doors
# --------------------------------------------------------------------------- #


def _sniff(data: bytes, said: str) -> str:
    """What the bytes are, from their first bytes. Text only when they are
    UTF-8 indeed: a caller's `text/plain` on bytes that are not is not text."""
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    with contextlib.suppress(UnicodeDecodeError):
        if b"\x00" not in data:
            data.decode("utf-8")
            return "text/plain"
    recorded = said.split(";", 1)[0].strip().lower()
    if not recorded or recorded.startswith("text/"):
        return "application/octet-stream"
    return recorded


class _Found:
    def __init__(self, ident: str, filename: str, media: str, data: bytes) -> None:
        self.ident, self.filename, self.media, self.data = ident, filename, media, data

    @property
    def b64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    def text(self) -> str:
        return self.data.decode("utf-8")


class _Resolving:
    """One request's file ids: whose they are, and the one budget they share.

    **Every file a request names counts against one total** (review of row
    53): a file named twice expands twice, and text becomes text, which the
    attachment checks after this one never see. So the total is checked
    from each file's recorded size before its bytes are read, and the
    download is cut at what is left.
    """

    def __init__(self, request: Request, owner: str) -> None:
        self.request = request
        self.owner = owner
        self.used = 0
        #: Bytes of stored text read in place of a file id: the inline checks never
        #: see them, so they are handed on to the request's one budget.
        self.text = 0

    def take(self, size: int, ident: str, where: str) -> int:
        """Room for `size` more bytes, or the refusal; returns what is left after."""
        if size > images.MAX_ATTACHMENT_BYTES:
            raise DoorRefusal(
                400,
                f"{where}: {ident} is {size} bytes; a request carries at most 10 MiB "
                "from one file.",
                param=where,
            )
        if self.used + size > images.MAX_ATTACHMENTS_TOTAL:
            raise DoorRefusal(
                400,
                f"{where}: the files this request names come to more than "
                f"{images.MAX_ATTACHMENTS_TOTAL // (1024 * 1024)} MiB together "
                f"({self.used + size} bytes, each named file counted each time).",
                param=where,
            )
        self.used += size
        return size


async def _fetch(ctx: _Resolving, ident: str, where: str) -> _Found:
    store = _store(ctx.request)
    try:
        kept = await store.call("GET", f"/v1/files/{_safe(ident)}", ctx.owner)
    except StoreRefusal as exc:
        if exc.status == 404:
            raise DoorRefusal(400, f"{where}: no such file: {ident}.", param=where) from exc
        raise _refused(exc, kind="file", ident=ident) from exc
    allowed = ctx.take(int(kept["bytes"]), ident, where)
    try:
        data, media = await store.read(
            f"/v1/files/{_safe(ident)}/content", ctx.owner, limit=allowed
        )
    except StoreRefusal as exc:
        if exc.status == 404:
            raise DoorRefusal(400, f"{where}: no such file: {ident}.", param=where) from exc
        if exc.status == 413:
            raise DoorRefusal(
                400, f"{where}: {ident} is larger than its record says.", param=where
            ) from exc
        raise _refused(exc, kind="file", ident=ident) from exc
    found = _Found(ident, str(kept["filename"]), _sniff(data, media), data)
    if found.media == "text/plain":
        ctx.text += len(data)
    return found


def _unreadable(found: _Found, where: str, door: str) -> DoorRefusal:
    return DoorRefusal(
        400,
        f"{where}: {found.ident} ({found.filename}) is {found.media}; {door} carries a PDF, an "
        "image (PNG, JPEG, GIF or WebP) or UTF-8 text from the store.",
        param=where,
    )


def _image_url(found: _Found, where: str) -> str:
    try:
        return images.data_url_from_base64(found.media, found.b64, where)
    except images.ImageRefusal as exc:
        raise DoorRefusal(400, str(exc), param=where) from None


async def _chat_part(ctx: _Resolving, part: dict, where: str) -> dict:
    file = part.get("file")
    if part.get("type") != "file" or not isinstance(file, dict):
        return part
    ident = file.get("file_id")
    if not isinstance(ident, str):
        return part
    if file.get("file_data"):
        raise DoorRefusal(400, f"{where}.file: send file_id or file_data, not both.", param=where)
    found = await _fetch(ctx, ident, f"{where}.file.file_id")
    name = file.get("filename") or found.filename
    if found.media == "application/pdf":
        return {
            "type": "file",
            "file": {"filename": name, "file_data": "data:application/pdf;base64," + found.b64},
        }
    if found.media.startswith("image/"):
        return {"type": "image_url", "image_url": {"url": _image_url(found, where)}}
    if found.media == "text/plain":
        return {"type": "text", "text": f"{name}\n\n{found.text()}"}
    raise _unreadable(found, where, "a chat request")


async def _responses_part(ctx: _Resolving, part: dict, where: str) -> dict:
    kind = part.get("type")
    ident = part.get("file_id")
    if kind not in ("input_file", "input_image") or not isinstance(ident, str):
        return part
    found = await _fetch(ctx, ident, f"{where}.file_id")
    name = part.get("filename") or found.filename
    if found.media == "application/pdf" and kind == "input_file":
        out: dict[str, Any] = {
            "type": "input_file",
            "filename": name,
            "file_data": "data:application/pdf;base64," + found.b64,
        }
    elif found.media.startswith("image/"):
        out = {"type": "input_image", "image_url": _image_url(found, where)}
    elif found.media == "text/plain" and kind == "input_file":
        out = {"type": "input_text", "text": f"{name}\n\n{found.text()}"}
    else:
        raise _unreadable(found, where, "a Responses request")
    if part.get("detail") is not None:
        out["detail"] = part["detail"]
    return out


async def _anthropic_block(ctx: _Resolving, block: dict, where: str) -> dict:
    kind = block.get("type")
    source = block.get("source")
    if (
        kind not in ("image", "document")
        or not isinstance(source, dict)
        or source.get("type") != "file"
    ):
        if kind == "tool_result" and isinstance(block.get("content"), list):
            inner = [
                await _anthropic_block(ctx, b, f"{where}.content.{i}") if isinstance(b, dict) else b
                for i, b in enumerate(block["content"])
            ]
            return {**block, "content": inner}
        return block
    ident = source.get("file_id")
    if not isinstance(ident, str):
        raise DoorRefusal(400, f"{where}.source.file_id: required.")
    found = await _fetch(ctx, ident, f"{where}.source.file_id")
    out = dict(block)
    if kind == "image" and found.media.startswith("image/"):
        out["source"] = {"type": "base64", "media_type": found.media, "data": found.b64}
    elif kind == "document" and found.media == "application/pdf":
        out["source"] = {"type": "base64", "media_type": "application/pdf", "data": found.b64}
    elif kind == "document" and found.media == "text/plain":
        out["source"] = {"type": "text", "media_type": "text/plain", "data": found.text()}
        out.setdefault("title", found.filename)
    else:
        raise _unreadable(found, where, "a Messages request")
    return out


def _has_file_id(value: Any) -> bool:
    if isinstance(value, dict):
        if isinstance(value.get("file_id"), str):
            return True
        return any(_has_file_id(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_file_id(v) for v in value)
    return False


async def resolve_file_ids(request: Request, raw: Any, *, door: str) -> Any | Response:
    """`raw` with every file_id the door reads replaced by the file it names,
    or the refusal as the door's own response. Untouched, and the store
    never asked, when the request names no file."""
    if not isinstance(raw, dict) or not _has_file_id(raw):
        return raw
    anthropic = door == "messages"
    try:
        owner = await owner_for(request, anthropic=anthropic)
        ctx = _Resolving(request, owner)
        resolved = await _resolve(ctx, raw, door)
        # One budget per request: the text just read counts with the inline attachments
        # the validators count next.
        images.carried_attachment_bytes.set(ctx.text)
        return resolved
    except DoorRefusal as refusal:
        return error_response(anthropic, refusal)
    except StoreUnavailable as exc:
        return error_response(anthropic, _unavailable(exc))


async def _resolve(ctx: _Resolving, raw: dict, door: str) -> dict:
    out = dict(raw)
    if door == "chat":
        messages = raw.get("messages")
        if isinstance(messages, list):
            out["messages"] = [
                await _message(ctx, m, f"messages.{i}", _chat_part) for i, m in enumerate(messages)
            ]
    elif door == "responses":
        items = raw.get("input")
        if isinstance(items, list):
            out["input"] = [
                await _message(ctx, m, f"input[{i}]", _responses_part, brackets=True)
                for i, m in enumerate(items)
            ]
    elif door == "messages":
        messages = raw.get("messages")
        if isinstance(messages, list):
            out["messages"] = [
                await _message(ctx, m, f"messages.{i}", _anthropic_block)
                for i, m in enumerate(messages)
            ]
    return out


async def _message(
    ctx: _Resolving, message: Any, where: str, part: Any, *, brackets: bool = False
) -> Any:
    """Each content part of one message; `where` names it as the door names fields."""
    if not isinstance(message, dict) or not isinstance(message.get("content"), list):
        return message
    content = [
        await part(ctx, p, f"{where}.content[{j}]" if brackets else f"{where}.content.{j}")
        if isinstance(p, dict)
        else p
        for j, p in enumerate(message["content"])
    ]
    return {**message, "content": content}

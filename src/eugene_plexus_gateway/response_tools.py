"""Client-run tools adapted to chat functions; no command or patch executes here."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError
from pydantic_core import from_json

from ._generated.models import ResponsesPatchOperation, ResponsesToolAction
from .chat_contract import _field_name

KINDS = frozenset({"custom", "local_shell", "shell", "apply_patch"})
REFUSALS = {
    "computer_use_preview": "no local model is trained for it; use a function tool instead",
    "computer": "no local model is trained for it; use a function tool instead",
    "code_interpreter": "Eugene has no sandbox; use a function tool and run it yourself",
    "mcp": "remote MCP execution needs row 59; use a client-run function tool instead",
    "file_search": "file search needs row 56; use a client-run function tool instead",
}


class InvalidTool(ValueError):
    """The caller's definition or the backend's adapter arguments are invalid."""


def definition(raw: Mapping[str, Any], where: str, ignored: list[str]) -> dict[str, Any]:
    kind = raw["type"]
    allowed = (
        {"type", "name", "description", "format"}
        if kind == "custom"
        else {"type", "environment"}
        if kind == "shell"
        else {"type"}
    )
    for key in raw:
        if key not in allowed:
            raise InvalidTool(f"{where}.{_field_name(key)}: unsupported setting.")
    name = raw.get("name") if kind == "custom" else kind
    if not isinstance(name, str) or not name:
        raise InvalidTool(f"{where}.name: required for a custom tool.")
    description = raw.get("description") or f"Ask the client to run {kind}."
    if not isinstance(description, str):
        raise InvalidTool(f"{where}.description: must be a string.")
    if kind == "shell" and raw.get("environment") is not None:
        environment = raw["environment"]
        if not isinstance(environment, Mapping) or environment.get("type") != "local":
            raise InvalidTool(
                f"{where}.environment: this gateway has no shell execution service; "
                "use the client's local environment."
            )
        if set(environment) - {"type"}:
            unsupported = next(key for key in environment if key != "type")
            raise InvalidTool(
                f"{where}.environment.{_field_name(unsupported)}: unsupported setting."
            )
    if kind == "custom" and raw.get("format") is not None:
        fmt = raw["format"]
        if not isinstance(fmt, Mapping) or fmt.get("type") not in ("text", "grammar"):
            raise InvalidTool(f"{where}.format: must be text or grammar.")
        if fmt["type"] == "grammar":
            if fmt.get("syntax") not in ("lark", "regex") or not isinstance(
                fmt.get("definition"), str
            ):
                raise InvalidTool(f"{where}.format: a grammar needs syntax and definition.")
            description += (
                f"\nRequested {fmt['syntax']} grammar (not enforced):\n{fmt['definition']}"
            )
            ignored.append(f"{where}.format")
    string = {"type": "string"}
    array = {"type": "array", "items": string, "minItems": 1}
    positive = {"type": "integer", "minimum": 1}
    properties: dict[str, Any]
    if kind in ("custom", "apply_patch"):
        properties, required = {"input": string}, ["input"]
        if kind == "apply_patch":
            description += (
                "\ninput is one file's patch: *** Begin Patch, *** Add File: path, "
                "*** Update File: path, or *** Delete File: path, then *** End Patch. "
                "Alternatively input may be a JSON operation with type (create_file, "
                "update_file, delete_file), path, and diff (V4A for create or update). "
                "Use separate calls for separate files."
            )
    elif kind == "local_shell":
        properties = {
            "command": array,
            "workdir": string,
            "timeout_ms": positive,
            "env": {"type": "object", "additionalProperties": string},
        }
        required = ["command"]
    else:
        properties = {"commands": array, "timeout_ms": positive, "max_output_length": positive}
        required = ["commands"]
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def patch_operation(text: str) -> dict[str, Any]:
    """One client operation from a patch; preserve its diff, never apply it."""
    try:
        operation = json.loads(text)
    except ValueError:
        lines = text.splitlines()
        if len(lines) < 3 or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
            raise InvalidTool(
                "apply_patch.input: expected one file patch or a JSON operation."
            ) from None
        markers = {
            "*** Add File: ": "create_file",
            "*** Update File: ": "update_file",
            "*** Delete File: ": "delete_file",
        }
        marker = next((prefix for prefix in markers if lines[1].startswith(prefix)), None)
        if marker is None or any(line.startswith("*** ") for line in lines[2:-1]):
            raise InvalidTool(
                "apply_patch.input: expected one file operation; "
                "use separate calls for separate files."
            ) from None
        operation = {"type": markers[marker], "path": lines[1][len(marker) :]}
        if operation["type"] != "delete_file":
            operation["diff"] = "\n".join(lines[2:-1])
    try:
        parsed = ResponsesPatchOperation.model_validate(operation)
    except ValidationError as exc:
        raise InvalidTool("apply_patch.input: invalid operation type, path or diff.") from exc
    if not parsed.path or (parsed.type != "delete_file" and parsed.diff is None):
        raise InvalidTool("apply_patch.input: path and a create/update diff are required.")
    return parsed.model_dump(mode="json", exclude_none=True)


def item(
    kind: str, call_id: str, name: str, arguments: str, *, item_id: str, status: str
) -> dict[str, Any]:
    try:
        args = json.loads(arguments)
    except ValueError as exc:
        raise InvalidTool(f"{name}: backend returned invalid JSON arguments.") from exc
    if not isinstance(args, dict):
        raise InvalidTool(f"{name}: backend arguments must be an object.")
    out = {"id": item_id, "call_id": call_id, "status": status}
    if kind in ("custom", "apply_patch"):
        text = args.get("input")
        if not isinstance(text, str):
            raise InvalidTool(f"{name}.input: backend must return a string.")
        if kind == "custom":
            return {**out, "type": "custom_tool_call", "name": name, "input": text}
        return {**out, "type": "apply_patch_call", "operation": patch_operation(text)}
    key = "command" if kind == "local_shell" else "commands"
    commands = args.get(key)
    if (
        not isinstance(commands, list)
        or not commands
        or any(not isinstance(c, str) for c in commands)
    ):
        raise InvalidTool(f"{name}.{key}: backend must return a nonempty array of strings.")
    allowed = (
        {key, "timeout_ms", "workdir", "env"}
        if kind == "local_shell"
        else {key, "timeout_ms", "max_output_length"}
    )
    if set(args) - allowed:
        raise InvalidTool(f"{name}: backend returned unsupported action fields.")
    for field in ("timeout_ms", "max_output_length"):
        if (
            field in args
            and args[field] is not None
            and (type(args[field]) is not int or args[field] < 1)
        ):
            raise InvalidTool(f"{name}.{field}: backend must return a positive integer.")
    action = dict(args)
    if kind == "local_shell":
        action["type"] = "exec"
        action["env"] = action.get("env") or {}
        if "workdir" in action:
            action["working_directory"] = action.pop("workdir")
    try:
        ResponsesToolAction.model_validate(action)
    except ValidationError as exc:
        raise InvalidTool(f"{name}: backend returned an invalid shell action.") from exc
    return {**out, "type": kind + "_call", "action": action}


def input_item(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a chat function call/result from a Responses history item."""
    kind = raw["type"]
    if kind.endswith("_output"):
        output = raw.get("output")
        if kind == "shell_call_output":
            output = json.dumps(
                {
                    "output": output,
                    **(
                        {"max_output_length": raw["max_output_length"]}
                        if "max_output_length" in raw
                        else {}
                    ),
                },
                ensure_ascii=False,
            )
        elif kind == "apply_patch_call_output":
            output = json.dumps({"status": raw.get("status"), "output": output}, ensure_ascii=False)
        return {
            "type": "function_call_output",
            "call_id": raw.get("call_id")
            or (raw.get("id") if kind == "local_shell_call_output" else None),
            "output": output,
        }
    if kind == "custom_tool_call":
        name, args = raw.get("name"), {"input": raw.get("input")}
    elif kind == "apply_patch_call":
        try:
            operation = ResponsesPatchOperation.model_validate(raw.get("operation"))
        except ValidationError as exc:
            raise InvalidTool("apply_patch_call.operation: invalid type, path or diff.") from exc
        op = operation.model_dump(mode="json", exclude_none=True)
        prefix = {"create_file": "Add", "update_file": "Update", "delete_file": "Delete"}[
            op["type"]
        ]
        patch = f"*** Begin Patch\n*** {prefix} File: {op['path']}\n"
        if "diff" in op:
            patch += op["diff"] + "\n"
        name, args = "apply_patch", {"input": patch + "*** End Patch"}
    else:
        action = raw.get("action")
        if not isinstance(action, Mapping):
            raise InvalidTool(f"{kind}.action: must be an object.")
        args = dict(action)
        name = kind.removesuffix("_call")
        if kind == "local_shell_call":
            args.pop("type", None)
            if "working_directory" in args:
                args["workdir"] = args.pop("working_directory")
    return {
        "type": "function_call",
        "call_id": raw.get("call_id"),
        "name": name,
        "arguments": json.dumps(args, ensure_ascii=False),
    }


def partial(arguments: str) -> dict[str, Any]:
    """Decode complete strings and the current string prefix across JSON escapes."""
    try:
        value = from_json(arguments, allow_partial="trailing-strings")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}

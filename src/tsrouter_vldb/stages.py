from __future__ import annotations

from typing import Any

from .artifacts import check_artifacts
from .commands import COMMAND_ARTIFACT_GROUPS
from .execution import (
    artifact_bundles_for_reuse,
    build_release_command_plan,
    command_reuses_outputs,
    execute_release_command_plan,
    normalize_reuse_mode,
)


class StageCommandError(RuntimeError):
    pass


def _artifact_bundles(command: str) -> tuple[str, ...]:
    try:
        return tuple(COMMAND_ARTIFACT_GROUPS[command])
    except KeyError as exc:
        raise StageCommandError(f"unsupported public command: {command}") from exc


def build_stage_plan(command: str, args: Any) -> dict[str, Any]:
    return build_release_command_plan(command, args)


def check_stage(command: str, args: Any) -> dict[str, Any]:
    bundles = artifact_bundles_for_reuse(getattr(args, "reuse", "results"))
    payload = build_stage_plan(command, args)
    payload["artifact_check"] = check_artifacts(
        group=f"{command}_required",
        bundle_names=bundles,
        check_archives=False,
        check_contents=True,
    )
    payload["ok"] = bool(payload["artifact_check"]["ok"])
    return payload


def run_stage(command: str, args: Any) -> dict[str, Any]:
    payload = build_stage_plan(command, args)
    if bool(getattr(args, "execute", False)):
        bundles = artifact_bundles_for_reuse(getattr(args, "reuse", "results"))
        artifact_check = check_artifacts(
            group=f"{command}_required",
            bundle_names=bundles,
            check_archives=False,
            check_contents=True,
        )
        payload["artifact_check"] = artifact_check
        reuse = normalize_reuse_mode(getattr(args, "reuse", "results"))
        if command_reuses_outputs(reuse, command) and artifact_check["ok"]:
            payload["execution_results"] = [
                {
                    "operation": item.get("operation", ""),
                    "returncode": 0,
                    "skipped": True,
                    "reason": f"reuse level: {reuse}",
                }
                for item in payload.get("_execution_commands", [])
                if isinstance(item, dict)
            ]
            payload["ok"] = True
            return payload
        payload["execution_results"] = execute_release_command_plan(payload)
    return payload

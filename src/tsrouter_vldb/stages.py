from __future__ import annotations

from typing import Any

from .artifacts import check_artifacts
from .commands import COMMAND_ARTIFACT_GROUPS
from .legacy import build_release_command_plan, execute_release_command_plan


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
    bundles = _artifact_bundles(command)
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
        payload["execution_results"] = execute_release_command_plan(payload)
    return payload

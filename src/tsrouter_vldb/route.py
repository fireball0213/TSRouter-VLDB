from __future__ import annotations

from typing import Any

from .stages import build_stage_plan, check_stage, run_stage


COMMAND = "route"


def build_plan(args: Any) -> dict[str, Any]:
    return build_stage_plan(COMMAND, args)


def check(args: Any) -> dict[str, Any]:
    return check_stage(COMMAND, args)


def run(args: Any) -> dict[str, Any]:
    return run_stage(COMMAND, args)

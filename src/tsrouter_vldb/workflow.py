from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .artifacts import check_artifacts
from .checks import check_layout
from .commands import COMMAND_ARTIFACT_GROUPS
from .execution import command_reuses_outputs, execute_release_command_plan, normalize_reuse_mode
from .stages import build_stage_plan


@dataclass(frozen=True)
class WorkflowStep:
    command: str
    action: str
    stage: int = 20
    variant: str = ""
    methods: str = ""
    start_stage: int | None = None
    end_stage: int | None = None
    write: bool = False


FAST_STEPS = (
    WorkflowStep("tsfm", "run"),
    WorkflowStep("profile", "run", variant="main,fast"),
    WorkflowStep("route", "run", variant="main,fast"),
    WorkflowStep("summary", "tables", write=True),
)

BASELINE_STEPS = (
    WorkflowStep("baselines", "run", methods="all"),
    WorkflowStep("summary", "tables", write=True),
)


WORKFLOWS = {
    "fast": FAST_STEPS,
    "baselines": BASELINE_STEPS,
}


class WorkflowError(RuntimeError):
    pass


def _reuse_mode(args: Any) -> str:
    return normalize_reuse_mode(getattr(args, "reuse", "results"))


def _workflow_artifact_group(mode: str, reuse: str) -> str:
    if reuse in {"results", "route", "core"}:
        return reuse
    return "results"


def _namespace_for_step(step: WorkflowStep, args: Any) -> Any:
    class Namespace:
        pass

    ns = Namespace()
    ns.action = step.action
    ns.stage = int(getattr(args, "stage", step.stage) or step.stage)
    ns.reuse = _reuse_mode(args)
    ns.execute = bool(getattr(args, "execute", False))
    ns.python_bin = str(getattr(args, "python_bin", "") or "")
    ns.workspace_root = getattr(args, "workspace_root", None) or getattr(args, "root", None)
    ns.devices = str(getattr(args, "devices", "") or "")
    ns.quick_test = bool(getattr(args, "quick_test", False))
    ns.variant = step.variant
    ns.methods = step.methods
    ns.table = ""
    ns.start_stage = step.start_stage
    ns.end_stage = step.end_stage
    ns.write = step.write
    return ns


def build_workflow_plan(args: Any) -> dict[str, Any]:
    mode = str(getattr(args, "mode", "fast") or "fast")
    if mode not in WORKFLOWS:
        raise WorkflowError(f"unknown workflow mode: {mode}")
    reuse = _reuse_mode(args)

    checks = []
    if bool(getattr(args, "check_layout", True)):
        layout_results = check_layout()
        checks.append(
            {
                "name": "layout",
                "ok": all(item.ok for item in layout_results),
                "results": [item.__dict__ for item in layout_results],
            }
        )
    if bool(getattr(args, "check_artifacts", True)):
        artifact_group = _workflow_artifact_group(mode, reuse)
        checks.append(
            {
                "name": "artifacts",
                **check_artifacts(group=artifact_group, check_archives=False, check_contents=True),
            }
        )

    step_plans = []
    for step in WORKFLOWS[mode]:
        step_args = _namespace_for_step(step, args)
        step_plans.append(build_stage_plan(step.command, step_args))

    return {
        "workflow": mode,
        "execution_mode": "execute" if bool(getattr(args, "execute", False)) else "plan",
        "reuse": reuse,
        "quick_test": bool(getattr(args, "quick_test", False)),
        "devices": str(getattr(args, "devices", "") or ""),
        "checks": checks,
        "steps": step_plans,
    }


def _artifact_backed_reuse_results(step: dict[str, Any]) -> list[dict[str, Any]] | None:
    reuse = normalize_reuse_mode(str(step.get("reuse", "results") or "results"))

    command = str(step.get("command", "") or "")
    if not command_reuses_outputs(reuse, command):
        return None

    return [
        {
            "operation": item.get("operation", ""),
            "returncode": 0,
            "skipped": True,
            "reason": f"reuse level: {reuse}",
        }
        for item in step.get("_execution_commands", [])
        if isinstance(item, dict)
    ]


def execute_workflow_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for index, step in enumerate(plan.get("steps", []), start=1):
        started = time.time()
        execution_results = _artifact_backed_reuse_results(step)
        artifact_backed_reuse = execution_results is not None
        if execution_results is None:
            execution_results = execute_release_command_plan(step)
        results.append(
            {
                "index": index,
                "command": step.get("command"),
                "action": step.get("action"),
                "returncode": 0,
                "workflow_wall_seconds": round(time.time() - started, 3),
                "artifact_backed_reuse": artifact_backed_reuse,
                "execution_results": execution_results,
            }
        )
    return results

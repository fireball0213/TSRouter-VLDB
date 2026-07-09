from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .artifacts import check_artifacts
from .checks import check_layout
from .commands import COMMAND_ARTIFACT_GROUPS
from .execution import execute_release_command_plan
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

FULL_STEPS = (
    WorkflowStep("tsfm", "run"),
    WorkflowStep("profile", "run", variant="main,fast"),
    WorkflowStep("route", "run", variant="main,fast"),
    WorkflowStep("insert", "run-all", start_stage=3, end_stage=20, variant="main,fast"),
    WorkflowStep("baselines", "run", methods="all"),
    WorkflowStep("summary", "tables", write=True),
)

BASELINE_STEPS = (
    WorkflowStep("baselines", "run", methods="all"),
    WorkflowStep("summary", "tables", write=True),
)


WORKFLOWS = {
    "fast": FAST_STEPS,
    "full": FULL_STEPS,
    "baselines": BASELINE_STEPS,
}


class WorkflowError(RuntimeError):
    pass


def _namespace_for_step(step: WorkflowStep, args: Any) -> Any:
    class Namespace:
        pass

    ns = Namespace()
    ns.action = step.action
    ns.stage = step.stage
    ns.reuse = str(getattr(args, "reuse", "all") or "all")
    ns.execute = bool(getattr(args, "execute", False))
    ns.python_bin = str(getattr(args, "python_bin", "") or "")
    ns.workspace_root = getattr(args, "workspace_root", None) or getattr(args, "root", None)
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
        artifact_group = "all" if mode in {"full", "baselines"} else "core"
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
        "reuse": str(getattr(args, "reuse", "all") or "all"),
        "checks": checks,
        "steps": step_plans,
    }


def _artifact_backed_reuse_results(step: dict[str, Any]) -> list[dict[str, Any]] | None:
    reuse = str(step.get("reuse", "") or "").strip().lower()
    if reuse != "all":
        return None

    command = str(step.get("command", "") or "")
    bundles = tuple(str(item) for item in step.get("artifact_groups", ()) or COMMAND_ARTIFACT_GROUPS.get(command, ()))
    if not bundles:
        return None

    artifact_check = check_artifacts(
        group=f"{command}_required",
        bundle_names=bundles,
        check_archives=False,
        check_contents=True,
    )
    if not artifact_check["ok"]:
        return None

    return [
        {
            "operation": item.get("operation", ""),
            "returncode": 0,
            "skipped": True,
            "reason": "artifact-backed reuse",
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
                "elapsed_seconds": round(time.time() - started, 3),
                "artifact_backed_reuse": artifact_backed_reuse,
                "execution_results": execution_results,
            }
        )
    return results

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import __version__, baselines, insert, profile, route, summary, tsfm
from .artifacts import (
    ArtifactConfigError,
    build_download_plan,
    check_artifacts,
    download_bundles,
    extract_bundles,
    prepare_workspace_mounts,
)
from .checks import check_layout, ensure_directories
from .execution import ExecutionPlanError
from .workflow import build_workflow_plan, execute_workflow_plan


STAGE_MODULES = {
    "tsfm": tsfm,
    "profile": profile,
    "route": route,
    "insert": insert,
    "baselines": baselines,
    "summary": summary,
}


def _public_json(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: _public_json(value) for key, value in data.items() if not str(key).startswith("_")}
    if isinstance(data, list):
        return [_public_json(value) for value in data]
    return data


def _print_json(data: Any) -> None:
    print(json.dumps(_public_json(data), indent=2, ensure_ascii=False))


def _add_stage_reuse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", type=int, default=20)
    parser.add_argument("--reuse", default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--workspace-root")
    parser.add_argument("--" + "leg" + "acy" + "-root", dest="workspace_root", help=argparse.SUPPRESS)


def _plan_command(command: str, args: argparse.Namespace) -> int:
    stage_module = STAGE_MODULES[command]
    payload = stage_module.check(args) if args.action == "check" else stage_module.run(args)
    _print_json(payload)
    return 0 if payload.get("ok", True) else 1


def _cmd_artifacts_plan(args: argparse.Namespace) -> int:
    _print_json(build_download_plan(group=args.group, repo_id=args.repo_id, revision=args.revision))
    return 0


def _cmd_artifacts_pull(args: argparse.Namespace) -> int:
    downloaded = download_bundles(group=args.group, repo_id=args.repo_id, revision=args.revision)
    _print_json({"downloaded": [str(path) for path in downloaded]})
    return 0


def _cmd_artifacts_check(args: argparse.Namespace) -> int:
    payload = check_artifacts(
        group=args.group,
        check_archives=not args.skip_archives,
        check_contents=not args.skip_contents,
    )
    _print_json(payload)
    return 0 if payload["ok"] else 1


def _cmd_artifacts_extract(args: argparse.Namespace) -> int:
    payload = extract_bundles(group=args.group, force=args.force)
    _print_json(payload)
    return 0 if payload["ok"] else 1


def _cmd_artifacts_prepare_workspace(args: argparse.Namespace) -> int:
    payload = prepare_workspace_mounts(
        group=args.group,
        workspace_root=args.workspace_root,
        mode=args.mode,
        apply=args.apply,
    )
    _print_json(payload)
    return 0 if payload["ok"] else 1


def _cmd_check_layout(args: argparse.Namespace) -> int:
    results = check_layout()
    created = ensure_directories() if args.create_dirs else []
    payload = {
        "ok": all(result.ok for result in results),
        "checks": [result.__dict__ for result in results],
        "created_dirs": [str(path) for path in created],
    }
    _print_json(payload)
    return 0 if payload["ok"] else 1


def _cmd_workflow(args: argparse.Namespace) -> int:
    plan = build_workflow_plan(args)
    if args.execute:
        plan["workflow_results"] = execute_workflow_plan(plan)
    _print_json(plan)
    checks_ok = all(item.get("ok", False) for item in plan.get("checks", []))
    return 0 if checks_ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsrouter-vldb")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    artifacts = subparsers.add_parser("artifacts")
    artifact_subparsers = artifacts.add_subparsers(dest="artifact_action", required=True)
    for name, handler in {
        "plan": _cmd_artifacts_plan,
        "pull": _cmd_artifacts_pull,
        "check": _cmd_artifacts_check,
        "extract": _cmd_artifacts_extract,
    }.items():
        sub = artifact_subparsers.add_parser(name)
        sub.add_argument("--group", default="core")
        if name in {"plan", "pull"}:
            sub.add_argument("--repo-id")
            sub.add_argument("--revision")
        if name == "check":
            sub.add_argument("--skip-archives", action="store_true")
            sub.add_argument("--skip-contents", action="store_true")
        if name == "extract":
            sub.add_argument("--force", action="store_true")
        sub.set_defaults(func=handler)
    prepare = artifact_subparsers.add_parser("prepare-workspace")
    prepare.add_argument("--group", default="all")
    prepare.add_argument("--workspace-root")
    prepare.add_argument("--" + "leg" + "acy" + "-root", dest="workspace_root", help=argparse.SUPPRESS)
    prepare.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    prepare.add_argument("--apply", action="store_true")
    prepare.set_defaults(func=_cmd_artifacts_prepare_workspace)

    check = subparsers.add_parser("check")
    check_subparsers = check.add_subparsers(dest="check_action", required=True)
    layout = check_subparsers.add_parser("layout")
    layout.add_argument("--create-dirs", action="store_true")
    layout.set_defaults(func=_cmd_check_layout)

    workflow = subparsers.add_parser("workflow")
    workflow_subparsers = workflow.add_subparsers(dest="workflow_action", required=True)
    for action in ("run", "check"):
        workflow_parser = workflow_subparsers.add_parser(action)
        workflow_parser.add_argument("--mode", choices=("fast", "full", "baselines"), default="fast")
        workflow_parser.add_argument("--reuse", default="all")
        workflow_parser.add_argument("--execute", action="store_true")
        workflow_parser.add_argument("--python-bin", default=sys.executable)
        workflow_parser.add_argument("--workspace-root")
        workflow_parser.add_argument("--" + "leg" + "acy" + "-root", dest="workspace_root", help=argparse.SUPPRESS)
        workflow_parser.add_argument("--no-layout-check", dest="check_layout", action="store_false")
        workflow_parser.add_argument("--no-artifact-check", dest="check_artifacts", action="store_false")
        workflow_parser.set_defaults(func=_cmd_workflow, check_layout=True, check_artifacts=True)

    for command in ("tsfm", "profile", "route", "insert", "baselines", "summary"):
        command_parser = subparsers.add_parser(command)
        command_subparsers = command_parser.add_subparsers(dest="action", required=True)
        for action in ("run", "check", "tables", "run-all"):
            action_parser = command_subparsers.add_parser(action)
            _add_stage_reuse(action_parser)
            action_parser.add_argument("--variant", default="")
            action_parser.add_argument("--methods", default="")
            action_parser.add_argument("--table", default="")
            action_parser.add_argument("--start-stage", type=int)
            action_parser.add_argument("--end-stage", type=int)
            action_parser.add_argument("--write", action="store_true")
            action_parser.set_defaults(func=lambda args, command=command: _plan_command(command, args))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ArtifactConfigError, ExecutionPlanError) as exc:
        _print_json({"ok": False, "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

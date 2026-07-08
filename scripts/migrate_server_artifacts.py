#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = RELEASE_ROOT.parent


def run_command(argv: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=str(cwd),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check and stage TSRouter-VLDB release artifacts on the server.")
    parser.add_argument("--legacy-root", default=".", help="Original TSRouter-v0 workspace root on the server.")
    parser.add_argument("--group", default="all", help="Artifact group to stage.")
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink", help="Use symlinks or real copies.")
    parser.add_argument("--clean", action="store_true", help="Clean staged bundle roots before staging.")
    parser.add_argument("--strict-zoo-span", default="20-20")
    parser.add_argument("--compatible-zoo-spans", default="20-21,20-23")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--snapshot-json", default="TSRouter-VLDB/docs/server_snapshot_check.json")
    parser.add_argument("--snapshot-md", default="TSRouter-VLDB/docs/server_snapshot_check.md")
    parser.add_argument("--snapshot-stdout", default="TSRouter-VLDB/docs/server_snapshot_check.stdout.json")
    parser.add_argument("--stage-json", default="TSRouter-VLDB/docs/server_stage_artifacts.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    legacy_root = Path(args.legacy_root).resolve()
    snapshot_json = Path(args.snapshot_json)
    snapshot_md = Path(args.snapshot_md)
    snapshot_stdout = Path(args.snapshot_stdout)
    stage_json = Path(args.stage_json)
    if not snapshot_json.is_absolute():
        snapshot_json = legacy_root / snapshot_json
    if not snapshot_md.is_absolute():
        snapshot_md = legacy_root / snapshot_md
    if not snapshot_stdout.is_absolute():
        snapshot_stdout = legacy_root / snapshot_stdout
    if not stage_json.is_absolute():
        stage_json = legacy_root / stage_json

    check_cmd = [
        sys.executable,
        str(RELEASE_ROOT / "scripts" / "check_server_migration_snapshot.py"),
        "--legacy-root",
        str(legacy_root),
        "--strict-zoo-span",
        str(args.strict_zoo_span),
        "--compatible-zoo-spans",
        str(args.compatible_zoo_spans),
        "--max-samples",
        str(args.max_samples),
        "--json-out",
        str(snapshot_json),
        "--md-out",
        str(snapshot_md),
    ]
    check_result = run_command(check_cmd, cwd=legacy_root)
    snapshot_stdout.parent.mkdir(parents=True, exist_ok=True)
    snapshot_stdout.write_text(check_result.stdout or "", encoding="utf-8")
    if check_result.returncode != 0:
        print(f"[migration] snapshot check failed; see {snapshot_json}", file=sys.stderr)
        return check_result.returncode

    snapshot = json.loads(snapshot_json.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "snapshot_ok": snapshot.get("ok"),
                "migration_safe": snapshot.get("migration_safe"),
                "summary": snapshot.get("summary", {}),
                "missing_required": snapshot.get("missing_required", []),
                "compatible_only": snapshot.get("compatible_only", []),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not snapshot.get("migration_safe", False):
        print(f"[migration] snapshot is not migration-safe; see {snapshot_json}", file=sys.stderr)
        return 2

    stage_cmd = [
        sys.executable,
        str(RELEASE_ROOT / "scripts" / "stage_local_artifacts.py"),
        "--group",
        str(args.group),
        "--mode",
        str(args.mode),
    ]
    if args.clean:
        stage_cmd.append("--clean")
    stage_result = run_command(stage_cmd, cwd=legacy_root)
    if stage_result.stdout:
        print(stage_result.stdout, end="")
    if stage_result.returncode != 0:
        print("[migration] artifact staging failed", file=sys.stderr)
        return stage_result.returncode

    try:
        stage_payload = json.loads(stage_result.stdout)
    except json.JSONDecodeError:
        stage_payload = {"ok": False, "raw_output": stage_result.stdout}
    stage_json.parent.mkdir(parents=True, exist_ok=True)
    stage_json.write_text(json.dumps(stage_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if not stage_payload.get("ok", False):
        print(f"[migration] staging reported missing required sources; see {stage_json}", file=sys.stderr)
        return 3

    print(f"[migration] staged artifacts written with mode={args.mode}; report={stage_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

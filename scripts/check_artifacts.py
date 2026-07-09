#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RELEASE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tsrouter_vldb.artifacts import check_artifacts, prepare_workspace_mounts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check TSRouter-VLDB artifact bundles and workspace mounts.")
    parser.add_argument("--group", default="core")
    parser.add_argument("--skip-archives", action="store_true")
    parser.add_argument("--skip-contents", action="store_true")
    parser.add_argument("--prepare-workspace", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--workspace-root")
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = {
        "artifact_check": check_artifacts(
            group=args.group,
            check_archives=not args.skip_archives,
            check_contents=not args.skip_contents,
        )
    }
    if args.prepare_workspace:
        payload["workspace_prepare"] = prepare_workspace_mounts(
            group=args.group,
            workspace_root=args.workspace_root,
            mode=args.mode,
            apply=args.apply,
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    ok = payload["artifact_check"]["ok"]
    if "workspace_prepare" in payload:
        ok = ok and payload["workspace_prepare"]["ok"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

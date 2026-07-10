from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "src"))

from config.model_zoo_config import Model_zoo_details
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local TSFM checkpoint availability.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--checkpoint-root", default="")
    parser.add_argument("--model", action="append", default=[])
    return parser.parse_args()


def records(selected: list[str]) -> list[dict[str, str]]:
    available: dict[str, dict[str, str]] = {}
    for family, variants in Model_zoo_details.items():
        for variant, config in variants.items():
            model_key = f"{family}_{variant}"
            available[model_key] = {
                "model": model_key,
                "source": str(config["module_name"]),
                "configured_path": str(config["model_local_path"]),
            }
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"unknown model identifiers: {', '.join(unknown)}")
    keys = selected or sorted(available)
    return [available[key] for key in keys]


def resolve_path(configured_path: str, checkpoint_root: Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == "checkpoints":
        return checkpoint_root.joinpath(*path.parts[1:]).resolve()
    return path.resolve()


def main() -> int:
    args = parse_args()
    workspace_root = Path(args.workspace_root or Path.cwd()).resolve()
    checkpoint_root = Path(args.checkpoint_root or workspace_root / "checkpoints").resolve()

    rows = []
    for item in records(args.model):
        path = resolve_path(item["configured_path"], checkpoint_root)
        rows.append({**item, "path": str(path), "available": path.is_dir() and any(path.iterdir())})

    missing = [row["model"] for row in rows if not row["available"]]
    print(json.dumps({"ok": not missing, "models": rows, "missing": missing}, indent=2))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())

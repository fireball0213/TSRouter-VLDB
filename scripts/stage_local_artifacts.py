#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from pack_hf_artifacts import (
    RELEASE_ROOT,
    WORKSPACE_ROOT,
    format_bytes,
    load_yaml,
    pattern_files,
    source_specs,
)


def place_file(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "symlink":
        try:
            os.symlink(source.resolve(), target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def target_for_source(bundle_id: str, spec: dict[str, Any], source: Path) -> Path:
    release_path = Path(str(spec["release_path"]))
    if bundle_id == "profile_sources":
        return RELEASE_ROOT / release_path

    pattern = str(spec["pattern"])
    source_root = WORKSPACE_ROOT / pattern.rstrip("/")
    base_path = str(spec.get("base_path", "") or "").strip("/")
    if base_path:
        return RELEASE_ROOT / release_path / source.relative_to(WORKSPACE_ROOT / base_path)
    if source_root.is_dir():
        return RELEASE_ROOT / release_path / source.relative_to(source_root)
    if source_root.is_file():
        return RELEASE_ROOT / release_path
    return RELEASE_ROOT / release_path.parent / source.name


def stage_bundle(bundle_id: str, bundle_data: dict[str, Any], profile_config: dict[str, Any], mode: str) -> dict[str, Any]:
    file_count = 0
    total_bytes = 0
    missing_required: list[str] = []
    missing_optional: list[str] = []
    staged_samples: list[str] = []

    for spec in source_specs(bundle_id, bundle_data, profile_config):
        files = pattern_files(spec["pattern"], spec["exclude_patterns"], spec["include_patterns"])
        if not files:
            if bool(spec.get("required", True)):
                missing_required.append(spec["pattern"])
            else:
                missing_optional.append(spec["pattern"])
            continue
        for source in files:
            target = target_for_source(bundle_id, spec, source)
            place_file(source, target, mode)
            file_count += 1
            total_bytes += source.stat().st_size
            if len(staged_samples) < 20:
                staged_samples.append(str(target.relative_to(RELEASE_ROOT)))

    return {
        "bundle_id": bundle_id,
        "ok": not missing_required,
        "mode": mode,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "total_human": format_bytes(total_bytes),
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "staged_samples": staged_samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage local TSRouter-VLDB artifacts from legacy workspace paths.")
    parser.add_argument("--group", default="core")
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    layout = load_yaml(RELEASE_ROOT / "configs" / "artifact_layout.yaml")
    profile_config = load_yaml(RELEASE_ROOT / "configs" / "profile_sources.yaml")
    group_ids = layout["groups"][args.group]["bundles"] if args.group in layout.get("groups", {}) else [args.group]

    if args.clean:
        for bundle_id in group_ids:
            staging_root = RELEASE_ROOT / str(layout["bundles"][bundle_id].get("staging_root", "."))
            if staging_root.exists() and staging_root != RELEASE_ROOT:
                shutil.rmtree(staging_root)

    results = [
        stage_bundle(bundle_id, layout["bundles"][bundle_id], profile_config, args.mode)
        for bundle_id in group_ids
    ]
    payload = {"ok": all(item["ok"] for item in results), "group": args.group, "results": results}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

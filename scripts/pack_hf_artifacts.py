#!/usr/bin/env python
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


RELEASE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = RELEASE_ROOT.parent


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required to pack artifacts.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration file must contain a mapping: {path}")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def as_rel(path: Path, root: Path = WORKSPACE_ROOT) -> str:
    path_abs = Path(os.path.abspath(path))
    root_abs = Path(os.path.abspath(root))
    try:
        return path_abs.relative_to(root_abs).as_posix()
    except ValueError:
        return path_abs.as_posix()


def excluded(path: Path, patterns: Iterable[str]) -> bool:
    rel = as_rel(path)
    return any(fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern) for pattern in patterns)


def included(path: Path, patterns: Iterable[str]) -> bool:
    values = list(patterns)
    if not values:
        return True
    rel = as_rel(path)
    return any(fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern) for pattern in values)


def pattern_files(
    pattern: str,
    exclude_patterns: Iterable[str] = (),
    include_patterns: Iterable[str] = (),
) -> list[Path]:
    normalized = pattern.replace("\\", "/").rstrip("/")
    if any(char in normalized for char in "*?[]"):
        matches = list(WORKSPACE_ROOT.glob(normalized))
    else:
        candidate = WORKSPACE_ROOT / normalized
        matches = [candidate] if candidate.exists() else []
    files: list[Path] = []
    for match in matches:
        if match.is_file():
            files.append(match)
        elif match.is_dir():
            files.extend(path for path in match.rglob("*") if path.is_file())
    return sorted(
        {
            path
            for path in files
            if included(path, include_patterns) and not excluded(path, exclude_patterns)
        },
        key=lambda item: item.as_posix(),
    )


def source_specs(bundle_id: str, bundle_data: dict[str, Any], profile_config: dict[str, Any]) -> list[dict[str, Any]]:
    if bundle_id == "profile_sources":
        specs = []
        for item in profile_config.get("sources", []):
            pattern = item.get("backend_path_hint")
            if not pattern:
                raise KeyError(f"profile source {item.get('release_id', '<unknown>')} must define backend_path_hint")
            specs.append(
                {
                    "pattern": str(pattern),
                    "release_path": str(item["release_path"]),
                    "exclude_patterns": [],
                    "include_patterns": [],
                    "base_path": "",
                    "required": bool(item.get("required_for_main_results", True)),
                }
            )
        return specs
    specs = []
    staging_root = str(bundle_data.get("staging_root", ".")).strip("/")
    for item in bundle_data.get("legacy_sources", []):
        if isinstance(item, dict):
            pattern = str(item["pattern"])
            exclude_patterns = [str(value) for value in item.get("exclude_patterns", [])]
            include_patterns = [str(value) for value in item.get("include_patterns", [])]
            base_path = str(item.get("base_path", "") or "")
            release_path = str(item.get("release_path", "") or "")
            required = bool(item.get("required", True))
        else:
            pattern = str(item)
            exclude_patterns = []
            include_patterns = []
            base_path = ""
            release_path = ""
            required = True
        specs.append(
            {
                "pattern": pattern,
                "release_path": release_path
                or str(Path(staging_root) / pattern.rstrip("/")).replace("\\", "/"),
                "exclude_patterns": exclude_patterns,
                "include_patterns": include_patterns,
                "base_path": base_path,
                "required": required,
            }
        )
    return specs


def link_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.symlink(source.resolve(), target)
    except OSError:
        shutil.copy2(source, target)


def _has_glob(pattern: str) -> bool:
    return any(char in pattern for char in "*?[]")


def _release_matches(pattern: str) -> list[Path]:
    normalized = pattern.replace("\\", "/").rstrip("/")
    if _has_glob(normalized):
        return [path for path in RELEASE_ROOT.glob(normalized) if path.exists()]
    candidate = RELEASE_ROOT / normalized
    return [candidate] if candidate.exists() else []


def staged_contents_ready(bundle_data: dict[str, Any]) -> bool:
    required_paths = [str(value) for value in bundle_data.get("required_paths", [])]
    if required_paths:
        return all(_release_matches(pattern) for pattern in required_paths)
    contents = [str(value).replace("\\", "/").rstrip("/") for value in bundle_data.get("contents", [])]
    return bool(contents) and all((RELEASE_ROOT / content).exists() for content in contents)


def stage_existing_contents(bundle_id: str, bundle_data: dict[str, Any], staging_dir: Path) -> dict[str, Any]:
    file_count = 0
    total_bytes = 0
    missing_sources = []
    optional_missing_sources = []
    for content in [str(value).replace("\\", "/").rstrip("/") for value in bundle_data.get("contents", [])]:
        source_root = RELEASE_ROOT / content
        if not source_root.exists():
            missing_sources.append(content)
            continue
        if source_root.is_file():
            files = [source_root]
        else:
            files = [path for path in source_root.rglob("*") if path.is_file()]
        for source in files:
            target = staging_dir / source.relative_to(RELEASE_ROOT)
            link_file(source, target)
            file_count += 1
            total_bytes += source.stat().st_size
    return {
        "bundle_id": bundle_id,
        "source_mode": "staged_contents",
        "file_count": file_count,
        "total_bytes": total_bytes,
        "total_human": format_bytes(total_bytes),
        "missing_sources": missing_sources,
        "optional_missing_sources": optional_missing_sources,
    }


def stage_bundle(bundle_id: str, bundle_data: dict[str, Any], profile_config: dict[str, Any], staging_dir: Path) -> dict[str, Any]:
    if staged_contents_ready(bundle_data):
        return stage_existing_contents(bundle_id, bundle_data, staging_dir)

    file_count = 0
    total_bytes = 0
    missing_sources = []
    optional_missing_sources = []
    for spec in source_specs(bundle_id, bundle_data, profile_config):
        files = pattern_files(spec["pattern"], spec["exclude_patterns"], spec["include_patterns"])
        if not files:
            if bool(spec.get("required", True)):
                missing_sources.append(spec["pattern"])
            else:
                optional_missing_sources.append(spec["pattern"])
            continue
        source_root = WORKSPACE_ROOT / spec["pattern"].rstrip("/")
        base_path = str(spec.get("base_path", "") or "").strip("/")
        base_root = WORKSPACE_ROOT / base_path if base_path else None
        for source in files:
            if any(char in spec["pattern"] for char in "*?[]") or source.is_file():
                if bundle_id == "profile_sources":
                    rel_target = Path(spec["release_path"])
                elif base_root is not None:
                    rel_target = Path(spec["release_path"]) / source.relative_to(base_root)
                elif source_root.is_dir():
                    rel_target = Path(spec["release_path"]) / source.relative_to(source_root)
                elif source_root.is_file():
                    rel_target = Path(spec["release_path"])
                else:
                    rel_target = Path(spec["release_path"]).parent / source.name
                target = staging_dir / rel_target
                link_file(source, target)
                file_count += 1
                total_bytes += source.stat().st_size
    return {
        "bundle_id": bundle_id,
        "source_mode": "legacy_sources",
        "file_count": file_count,
        "total_bytes": total_bytes,
        "total_human": format_bytes(total_bytes),
        "missing_sources": missing_sources,
        "optional_missing_sources": optional_missing_sources,
    }


def make_archive(staging_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["tar", "--zstd", "-chf", str(archive_path), "-C", str(staging_dir), "."],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"tar failed for {archive_path}")


def write_dataset_readme(path: Path, repo_id: str, manifest: dict[str, Any]) -> None:
    lines = [
        "---",
        "license: other",
        "pretty_name: TSRouter-VLDB Artifacts",
        "tags:",
        "- time-series",
        "- benchmark",
        "- reproducibility",
        "- paper-artifact",
        "- vldb",
        "---",
        "",
        "# TSRouter-VLDB Artifacts",
        "",
        f"This Hugging Face Dataset repository stores reproduction artifacts for `{repo_id}`.",
        "",
        "Use the public GitHub repository code to download, extract, verify, and reproduce the released TSRouter-VLDB main paper tables.",
        "",
        "GitHub code: https://github.com/fireball0213/TSRouter-VLDB",
        "",
        "## Scope",
        "",
        "- Main experiment: stage 20.",
        "- Paper-facing TSRouter configuration: `repr_size=3000`, `repr_v=4`, `zoo_repr_set=c-e-n-h-w-s`.",
        "- Methods covered: TSFM reuse results, TSRouter-main, TSRouter-fast, and paper baseline artifacts.",
        "- Supported workflow: artifact-backed Step1-Step4 reproduction, result checking, and released table preview.",
        "- Excluded from this artifact release: Step0 source sampling, auto-cl, v5/search runs, ablation caches, and full raw benchmark/model archives.",
        "",
        "## Bundles",
        "",
    ]
    for bundle in manifest["bundles"]:
        lines.append(f"- `{bundle['filename']}`: {bundle['size_human']}, sha256 `{bundle['sha256']}`")
    lines.extend(
        [
            "",
            "## Integrity Files",
            "",
            "- `manifest.json`: bundle metadata, source mode, sizes, checksums, and expected contents.",
            "- `checksums.sha256`: SHA-256 checksums for each bundle and the manifest.",
            "",
            "## Local Use",
            "",
            "```bash",
            "export TSROUTER_VLDB_HF_REPO=\"" + repo_id + "\"",
            "python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts pull --repo-id \"$TSROUTER_VLDB_HF_REPO\" --group all",
            "python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts extract --group all",
            "python TSRouter-VLDB/src/cli/tsrouter_vldb.py artifacts check --group all",
            "bash TSRouter-VLDB/scripts/run_public_reproduction.sh --root \"$PWD\" --python-bin \"$(which python)\" --mode full",
            "```",
            "",
            "## Notes",
            "",
            "This repository is an artifact bundle rather than a standalone raw dataset. The public workflow consumes the archives through the TSRouter-VLDB GitHub code and validates the released main-results contract before running the artifact-backed workflow.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack TSRouter-VLDB Hugging Face Dataset artifacts.")
    parser.add_argument("--out", default="TSRouter-VLDB_hf_upload")
    parser.add_argument("--group", default="all")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    layout = load_yaml(RELEASE_ROOT / "configs" / "artifact_layout.yaml")
    profile_config = load_yaml(RELEASE_ROOT / "configs" / "profile_sources.yaml")
    group_ids = layout["groups"][args.group]["bundles"] if args.group in layout.get("groups", {}) else [args.group]
    out_dir = Path(args.out).resolve()
    staging_root = out_dir / ".staging"
    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "repo_id": args.repo_id,
        "bundles": [],
    }
    checksum_lines = []

    for bundle_id in group_ids:
        bundle_data = layout["bundles"][bundle_id]
        staging_dir = staging_root / bundle_id
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)
        stage_info = stage_bundle(bundle_id, bundle_data, profile_config, staging_dir)
        if stage_info["missing_sources"]:
            raise SystemExit(f"missing sources for {bundle_id}: {stage_info['missing_sources']}")
        archive_path = out_dir / bundle_data["filename"]
        make_archive(staging_dir, archive_path)
        digest = sha256_file(archive_path)
        size = archive_path.stat().st_size
        manifest["bundles"].append(
            {
                **stage_info,
                "filename": bundle_data["filename"],
                "sha256": digest,
                "size_bytes": size,
                "size_human": format_bytes(size),
                "extract_to": str(bundle_data.get("extract_to", ".")),
                "contents": list(bundle_data.get("contents", [])),
            }
        )
        checksum_lines.append(f"{digest}  {bundle_data['filename']}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_digest = sha256_file(manifest_path)
    checksum_lines.append(f"{manifest_digest}  manifest.json")
    (out_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    write_dataset_readme(out_dir / "README.md", args.repo_id, manifest)
    shutil.rmtree(staging_root)
    print(json.dumps({"ok": True, "out": str(out_dir), "manifest": str(manifest_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


RELEASE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = RELEASE_ROOT.parent


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required to build the artifact manifest.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration file must contain a mapping: {path}")
    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def as_posix_relative(path: Path, root: Path = WORKSPACE_ROOT) -> str:
    path_abs = Path(os.path.abspath(path))
    root_abs = Path(os.path.abspath(root))
    try:
        return path_abs.relative_to(root_abs).as_posix()
    except ValueError:
        return path_abs.as_posix()


def excluded_by_patterns(path: Path, exclude_patterns: Iterable[str]) -> bool:
    relative = as_posix_relative(path)
    name = path.name
    return any(fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(name, pattern) for pattern in exclude_patterns)


def included_by_patterns(path: Path, include_patterns: Iterable[str]) -> bool:
    values = list(include_patterns)
    if not values:
        return True
    relative = as_posix_relative(path)
    name = path.name
    return any(fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(name, pattern) for pattern in values)


def iter_pattern_matches(
    pattern: str,
    exclude_patterns: Iterable[str] = (),
    include_patterns: Iterable[str] = (),
) -> list[Path]:
    normalized = pattern.replace("\\", "/").rstrip("/")
    if any(char in normalized for char in "*?[]"):
        matches = list(WORKSPACE_ROOT.glob(normalized))
    else:
        path = WORKSPACE_ROOT / normalized
        matches = [path] if path.exists() else []

    files: list[Path] = []
    for match in matches:
        if match.is_file():
            files.append(match)
        elif match.is_dir():
            files.extend(child for child in match.rglob("*") if child.is_file())
    filtered = [
        path
        for path in files
        if included_by_patterns(path, include_patterns) and not excluded_by_patterns(path, exclude_patterns)
    ]
    return sorted(set(filtered), key=lambda item: item.as_posix())


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    source_pattern: str
    release_path: str | None = None
    base_path: str | None = None
    platform: str = "huggingface"
    required: bool = True
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()


def profile_source_specs(profile_config: dict[str, Any]) -> list[SourceSpec]:
    specs: list[SourceSpec] = []
    for item in profile_config.get("sources", []):
        source_pattern = item.get("backend_path_hint")
        if not source_pattern:
            raise KeyError(f"profile source {item.get('release_id', '<unknown>')} must define backend_path_hint")
        specs.append(
            SourceSpec(
                source_id=str(item["release_id"]),
                source_pattern=str(source_pattern),
                release_path=str(item["release_path"]),
                base_path=None,
                required=bool(item.get("required_for_main_results", True)),
            )
        )
    return specs


def bundle_source_specs(bundle_id: str, bundle_data: dict[str, Any], profile_config: dict[str, Any]) -> list[SourceSpec]:
    if bundle_id == "profile_sources":
        return profile_source_specs(profile_config)
    specs: list[SourceSpec] = []
    for idx, item in enumerate(bundle_data.get("legacy_sources", [])):
        if isinstance(item, dict):
            specs.append(
                SourceSpec(
                    source_id=str(item.get("source_id", f"{bundle_id}:{idx + 1}")),
                    source_pattern=str(item["pattern"]),
                    release_path=str(item.get("release_path", "")) or None,
                    base_path=str(item.get("base_path", "")) or None,
                    required=bool(item.get("required", True)),
                    include_patterns=tuple(str(value) for value in item.get("include_patterns", [])),
                    exclude_patterns=tuple(str(value) for value in item.get("exclude_patterns", [])),
                )
            )
        else:
            specs.append(SourceSpec(source_id=f"{bundle_id}:{idx + 1}", source_pattern=str(item)))
    return specs


def should_hash(path: Path, mode: str, threshold_bytes: int) -> bool:
    if mode == "none":
        return False
    if mode == "all":
        return True
    return path.stat().st_size <= threshold_bytes


def scan_source(spec: SourceSpec, *, hash_mode: str, hash_threshold_bytes: int) -> dict[str, Any]:
    files = iter_pattern_matches(spec.source_pattern, spec.exclude_patterns, spec.include_patterns)
    total_bytes = sum(path.stat().st_size for path in files)
    hashed_files = 0
    sample_entries = []
    largest_entries = []

    for path in files[:20]:
        size = path.stat().st_size
        sample_entries.append(
            {
                "path": as_posix_relative(path),
                "size_bytes": size,
                "sha256": sha256_file(path) if should_hash(path, hash_mode, hash_threshold_bytes) else None,
            }
        )
        if sample_entries[-1]["sha256"]:
            hashed_files += 1

    for path in sorted(files, key=lambda item: item.stat().st_size, reverse=True)[:10]:
        largest_entries.append(
            {
                "path": as_posix_relative(path),
                "size_bytes": path.stat().st_size,
                "size_human": format_bytes(path.stat().st_size),
            }
        )

    if len(files) > len(sample_entries) and hash_mode != "none":
        for path in files[len(sample_entries):]:
            if should_hash(path, hash_mode, hash_threshold_bytes):
                hashed_files += 1

    return {
        "source_id": spec.source_id,
        "source_pattern": spec.source_pattern,
        "release_path": spec.release_path,
        "base_path": spec.base_path,
        "platform": spec.platform,
        "required": spec.required,
        "include_patterns": list(spec.include_patterns),
        "exclude_patterns": list(spec.exclude_patterns),
        "status": "present" if files else "missing",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "total_human": format_bytes(total_bytes),
        "hashed_file_count": hashed_files,
        "hash_mode": hash_mode,
        "sample_files": sample_entries,
        "largest_files": largest_entries,
    }


def build_manifest(hash_mode: str, hash_threshold_mb: int) -> dict[str, Any]:
    artifact_layout = load_yaml(RELEASE_ROOT / "configs" / "artifact_layout.yaml")
    profile_config = load_yaml(RELEASE_ROOT / "configs" / "profile_sources.yaml")
    threshold_bytes = int(hash_threshold_mb) * 1024 * 1024
    bundles = []

    for bundle_id, bundle_data in artifact_layout.get("bundles", {}).items():
        sources = [
            scan_source(spec, hash_mode=hash_mode, hash_threshold_bytes=threshold_bytes)
            for spec in bundle_source_specs(bundle_id, bundle_data, profile_config)
        ]
        total_bytes = sum(item["total_bytes"] for item in sources)
        file_count = sum(item["file_count"] for item in sources)
        missing_required = [
            item["source_id"]
            for item in sources
            if item["required"] and item["status"] != "present"
        ]
        bundles.append(
            {
                "bundle_id": bundle_id,
                "hf_filename": bundle_data["filename"],
                "required": bool(bundle_data.get("required", True)),
                "extract_to": str(bundle_data.get("extract_to", ".")),
                "contents": list(bundle_data.get("contents", [])),
                "status": "ready" if not missing_required else "missing_required_sources",
                "file_count": file_count,
                "total_bytes": total_bytes,
                "total_human": format_bytes(total_bytes),
                "missing_required_sources": missing_required,
                "sources": sources,
            }
        )

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "workspace_root": ".",
        "hash_mode": hash_mode,
        "hash_threshold_mb": hash_threshold_mb,
        "source_configs": [
            "configs/artifact_layout.yaml",
            "configs/profile_sources.yaml",
        ],
        "bundles": bundles,
    }


def md_table_row(values: Iterable[Any]) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Artifact 迁移清单",
        "",
        "本文档由 `configs/artifact_layout.yaml` 和 `configs/profile_sources.yaml` 生成，用于核对本地源文件如何进入 Hugging Face artifact bundle。",
        "",
        "机器可读版本是 `configs/artifact_manifest.yaml`。公开仓库不包含 `docs/` 下的维护文档。",
        "",
        "## Bundle 汇总",
        "",
        md_table_row(["Bundle", "Hugging Face 文件", "状态", "文件数", "大小", "解压后的目标内容"]),
        md_table_row(["---", "---", "---", "---", "---", "---"]),
    ]

    for bundle in manifest["bundles"]:
        lines.append(
            md_table_row(
                [
                    bundle["bundle_id"],
                    bundle["hf_filename"],
                    bundle["status"],
                    bundle["file_count"],
                    bundle["total_human"],
                    "<br>".join(bundle["contents"]),
                ]
            )
        )

    lines.extend(["", "## 源文件汇总", ""])
    for bundle in manifest["bundles"]:
        lines.extend([f"### {bundle['bundle_id']}", ""])
        lines.append(md_table_row(["Source", "源模式", "必须", "状态", "文件数", "大小", "发布路径"]))
        lines.append(md_table_row(["---", "---", "---", "---", "---", "---", "---"]))
        for source in bundle["sources"]:
            lines.append(
                md_table_row(
                    [
                        source["source_id"],
                        source["source_pattern"],
                        str(source["required"]).lower(),
                        source["status"],
                        source["file_count"],
                        source["total_human"],
                        source["release_path"] or "",
                    ]
                )
            )
        lines.append("")

    lines.extend(
        [
            "## 迁移流程",
            "",
            "1. 在服务器上核对每个 bundle 的源模式只命中论文采用参数。",
            "2. 使用 `scripts/stage_local_artifacts.py` 本地暂存，或按 `configs/artifact_layout.yaml` 的 `release_path` 手工迁移到同样结构。",
            "3. 使用 `scripts/pack_hf_artifacts.py` 生成压缩包，并上传 `bundles/*.tar.zst`、`manifest.json`、`checksums.sha256` 与 Hugging Face Dataset Card。",
            "4. 下载或解压后运行 artifact check、backend prepare 和 workflow 测试。",
            "",
            "## 校验策略",
            "",
            "本地迁移清单可以只对小文件计算 checksum。公开发布时必须为每个压缩包和 `manifest.json` 提供 checksum。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bundle_id",
                "source_id",
                "source_pattern",
                "release_path",
                "status",
                "file_count",
                "total_bytes",
            ],
        )
        writer.writeheader()
        for bundle in manifest["bundles"]:
            for source in bundle["sources"]:
                writer.writerow(
                    {
                        "bundle_id": bundle["bundle_id"],
                        "source_id": source["source_id"],
                        "source_pattern": source["source_pattern"],
                        "release_path": source["release_path"] or "",
                        "status": source["status"],
                        "file_count": source["file_count"],
                        "total_bytes": source["total_bytes"],
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the TSRouter-VLDB local artifact migration manifest.")
    parser.add_argument("--hash-mode", choices=("none", "small", "all"), default="small")
    parser.add_argument("--hash-threshold-mb", type=int, default=64)
    parser.add_argument("--yaml-out", default="configs/artifact_manifest.yaml")
    parser.add_argument("--md-out", default="docs/artifact_manifest.md")
    parser.add_argument("--csv-out", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args.hash_mode, args.hash_threshold_mb)
    dump_yaml(RELEASE_ROOT / args.yaml_out, manifest)
    write_markdown(RELEASE_ROOT / args.md_out, manifest)
    if args.csv_out:
        write_csv(RELEASE_ROOT / args.csv_out, manifest)
    if args.json:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

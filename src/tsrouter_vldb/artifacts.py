from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .paths import ReleasePaths, default_hf_repo


class ArtifactConfigError(RuntimeError):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ArtifactConfigError("PyYAML is required to read release configuration files.") from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ArtifactConfigError(f"configuration file must contain a mapping: {path}")
    return data


@dataclass(frozen=True)
class ArtifactBundle:
    bundle_id: str
    filename: str
    required: bool
    extract_to: str
    contents: tuple[str, ...]
    required_paths: tuple[str, ...]
    staging_root: str
    backend_mounts: tuple[tuple[str, str, bool], ...]

    @classmethod
    def from_mapping(cls, bundle_id: str, data: dict[str, Any]) -> "ArtifactBundle":
        mounts = tuple(
            (str(item["source"]), str(item["target"]), bool(item.get("required", True)))
            for item in data.get("backend_mounts", [])
            if isinstance(item, dict)
        )
        return cls(
            bundle_id=bundle_id,
            filename=str(data["filename"]),
            required=bool(data.get("required", True)),
            extract_to=str(data.get("extract_to", ".")),
            contents=tuple(str(value) for value in data.get("contents", [])),
            required_paths=tuple(str(value) for value in data.get("required_paths", [])),
            staging_root=str(data.get("staging_root", ".")),
            backend_mounts=mounts,
        )


@dataclass(frozen=True)
class ArtifactLayout:
    default_hf_repo: str
    default_revision: str
    groups: dict[str, tuple[str, ...]]
    bundles: dict[str, ArtifactBundle]

    @classmethod
    def load(cls, paths: ReleasePaths | None = None) -> "ArtifactLayout":
        release_paths = paths or ReleasePaths.from_env()
        raw = load_yaml(release_paths.config_path("artifact_layout.yaml"))
        groups = {
            str(group_id): tuple(str(bundle_id) for bundle_id in group_data.get("bundles", []))
            for group_id, group_data in raw.get("groups", {}).items()
        }
        bundles = {
            str(bundle_id): ArtifactBundle.from_mapping(str(bundle_id), bundle_data)
            for bundle_id, bundle_data in raw.get("bundles", {}).items()
        }
        return cls(
            default_hf_repo=str(raw.get("default_hf_repo", "")),
            default_revision=str(raw.get("default_revision", "main")),
            groups=groups,
            bundles=bundles,
        )

    def select(self, group: str = "all", names: Iterable[str] | None = None) -> list[ArtifactBundle]:
        selected_ids = list(names) if names is not None else list(self.groups.get(group, ()))
        if not selected_ids:
            raise ArtifactConfigError(f"unknown or empty artifact group: {group}")
        missing = [bundle_id for bundle_id in selected_ids if bundle_id not in self.bundles]
        if missing:
            raise ArtifactConfigError(f"artifact layout references unknown bundles: {missing}")
        return [self.bundles[bundle_id] for bundle_id in selected_ids]


def build_download_plan(
    *,
    group: str = "all",
    repo_id: str | None = None,
    revision: str | None = None,
    paths: ReleasePaths | None = None,
) -> dict[str, Any]:
    release_paths = paths or ReleasePaths.from_env()
    layout = ArtifactLayout.load(release_paths)
    resolved_repo = repo_id or default_hf_repo() or layout.default_hf_repo
    resolved_revision = revision or layout.default_revision
    bundles = layout.select(group=group)
    return {
        "repo_id": resolved_repo,
        "revision": resolved_revision,
        "artifact_root": str(release_paths.artifact_root),
        "group": group,
        "bundles": [
            {
                "bundle_id": bundle.bundle_id,
                "filename": bundle.filename,
                "required": bundle.required,
                "extract_to": bundle.extract_to,
                "staging_root": bundle.staging_root,
                "contents": list(bundle.contents),
                "required_paths": list(bundle.required_paths),
                "backend_mounts": [
                    {"source": source, "target": target, "required": required}
                    for source, target, required in bundle.backend_mounts
                ],
            }
            for bundle in bundles
        ],
    }


def download_bundles(
    *,
    group: str = "all",
    repo_id: str | None = None,
    revision: str | None = None,
    paths: ReleasePaths | None = None,
) -> list[Path]:
    release_paths = paths or ReleasePaths.from_env()
    plan = build_download_plan(group=group, repo_id=repo_id, revision=revision, paths=release_paths)
    if not plan["repo_id"]:
        raise ArtifactConfigError("missing Hugging Face Dataset repo: pass --repo-id or set TSROUTER_VLDB_HF_REPO.")
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ArtifactConfigError("huggingface-hub is required to download release artifacts.") from exc

    downloaded: list[Path] = []
    for bundle in plan["bundles"]:
        local_path = hf_hub_download(
            repo_id=plan["repo_id"],
            filename=bundle["filename"],
            repo_type="dataset",
            revision=plan["revision"],
            cache_dir=str(release_paths.hf_cache_dir),
            local_dir=str(release_paths.artifact_root),
        )
        downloaded.append(Path(local_path))
    return downloaded


def _bundle_archive_path(paths: ReleasePaths, bundle: ArtifactBundle) -> Path:
    return paths.artifact_root / bundle.filename


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size if path.is_file() else None
    except OSError:
        return None


def _path_status(path: Path) -> dict[str, Any]:
    exists = _safe_exists(path)
    size = _safe_file_size(path) if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "is_file": _safe_is_file(path) if exists else False,
        "is_dir": _safe_is_dir(path) if exists else False,
        "size_bytes": size,
        "size_human": _format_bytes(size) if size is not None else "",
    }


def _pattern_status(root: Path, pattern: str) -> dict[str, Any]:
    normalized = pattern.replace("\\", "/").rstrip("/")
    has_glob = any(char in normalized for char in "*?[]")
    matches = sorted(root.glob(normalized)) if has_glob else [root / normalized]
    existing = [path for path in matches if _safe_exists(path)]
    total_bytes = sum(size for path in existing if (size := _safe_file_size(path)) is not None)
    return {
        "pattern": pattern,
        "exists": bool(existing),
        "match_count": len(existing),
        "size_bytes": total_bytes,
        "size_human": _format_bytes(total_bytes) if total_bytes else "0 B",
        "sample_matches": [str(path) for path in existing[:20]],
    }


def check_artifacts(
    *,
    group: str = "core",
    bundle_names: Iterable[str] | None = None,
    paths: ReleasePaths | None = None,
    check_archives: bool = True,
    check_contents: bool = True,
) -> dict[str, Any]:
    release_paths = paths or ReleasePaths.from_env()
    layout = ArtifactLayout.load(release_paths)
    bundles = layout.select(group=group, names=bundle_names)
    bundle_results = []

    for bundle in bundles:
        archive_status = _path_status(_bundle_archive_path(release_paths, bundle)) if check_archives else None
        content_statuses = [
            _path_status(release_paths.artifact_root / content)
            for content in bundle.contents
        ] if check_contents else []
        required_statuses = [
            _pattern_status(release_paths.artifact_root, pattern)
            for pattern in bundle.required_paths
        ] if check_contents else []
        mount_sources = [
            {
                **_path_status(release_paths.artifact_root / source),
                "target": target,
                "required": required,
            }
            for source, target, required in bundle.backend_mounts
        ]
        archive_ok = True if archive_status is None else bool(archive_status["exists"])
        contents_ok = all(item["exists"] for item in content_statuses) if content_statuses else True
        required_ok = all(item["exists"] for item in required_statuses) if required_statuses else True
        mounts_ok = all(item["exists"] or not item["required"] for item in mount_sources) if mount_sources else True
        bundle_results.append(
            {
                "bundle_id": bundle.bundle_id,
                "required": bundle.required,
                "archive": archive_status,
                "contents": content_statuses,
                "required_paths": required_statuses,
                "backend_mount_sources": mount_sources,
                "ok": archive_ok and contents_ok and required_ok and mounts_ok,
            }
        )

    return {
        "ok": all(item["ok"] or not item["required"] for item in bundle_results),
        "artifact_root": str(release_paths.artifact_root),
        "group": group,
        "bundle_names": list(bundle_names or []),
        "bundles": bundle_results,
    }


def extract_bundles(
    *,
    group: str = "core",
    paths: ReleasePaths | None = None,
    force: bool = False,
) -> dict[str, Any]:
    release_paths = paths or ReleasePaths.from_env()
    layout = ArtifactLayout.load(release_paths)
    results = []

    for bundle in layout.select(group=group):
        archive = _bundle_archive_path(release_paths, bundle)
        target_dir = (release_paths.artifact_root / bundle.extract_to).resolve()
        if not archive.exists():
            results.append({"bundle_id": bundle.bundle_id, "ok": False, "detail": f"missing archive: {archive}"})
            continue
        if not force and all((release_paths.artifact_root / content).exists() for content in bundle.contents):
            results.append({"bundle_id": bundle.bundle_id, "ok": True, "detail": "contents already present"})
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["tar", "--zstd", "-xf", str(archive), "-C", str(target_dir)],
            check=False,
            capture_output=True,
            text=True,
        )
        results.append(
            {
                "bundle_id": bundle.bundle_id,
                "ok": proc.returncode == 0,
                "archive": str(archive),
                "extract_to": str(target_dir),
                "stderr": proc.stderr.strip(),
            }
        )

    return {
        "ok": all(item["ok"] for item in results),
        "artifact_root": str(release_paths.artifact_root),
        "group": group,
        "results": results,
    }


def _same_link(target: Path, source: Path) -> bool:
    if not target.is_symlink():
        return False
    try:
        return target.resolve() == source.resolve()
    except OSError:
        return False


def _copy_path(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def prepare_backend_mounts(
    *,
    group: str = "all",
    paths: ReleasePaths | None = None,
    legacy_root: str | os.PathLike[str] | None = None,
    mode: str = "symlink",
    apply: bool = False,
) -> dict[str, Any]:
    if mode not in {"symlink", "copy"}:
        raise ArtifactConfigError("backend prepare mode must be 'symlink' or 'copy'")

    release_paths = paths or ReleasePaths.from_env()
    backend_root = Path(legacy_root).resolve() if legacy_root else release_paths.root.parent.resolve()
    layout = ArtifactLayout.load(release_paths)
    actions = []

    for bundle in layout.select(group=group):
        for source_rel, target_rel, required in bundle.backend_mounts:
            source = (release_paths.artifact_root / source_rel).resolve()
            target = (backend_root / target_rel).resolve()
            action = {
                "bundle_id": bundle.bundle_id,
                "source": str(source),
                "target": str(target),
                "required": required,
                "mode": mode,
                "source_exists": source.exists(),
                "target_exists": target.exists() or target.is_symlink(),
                "applied": False,
                "ok": False,
                "detail": "",
            }
            if _same_link(target, source):
                action["ok"] = True
                action["detail"] = "already linked"
            elif target.exists() or target.is_symlink():
                action["ok"] = True
                action["detail"] = "target already exists"
            elif not source.exists() and not required:
                action["ok"] = True
                action["detail"] = "optional source missing"
            elif not source.exists():
                action["detail"] = "missing source"
            elif not apply:
                action["ok"] = True
                action["detail"] = "planned"
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                if mode == "symlink":
                    os.symlink(source, target, target_is_directory=source.is_dir())
                else:
                    _copy_path(source, target)
                action["applied"] = True
                action["ok"] = True
                action["detail"] = "created"
            actions.append(action)

    return {
        "ok": all(item["ok"] for item in actions),
        "artifact_root": str(release_paths.artifact_root),
        "legacy_root": str(backend_root),
        "group": group,
        "apply": apply,
        "actions": actions,
    }

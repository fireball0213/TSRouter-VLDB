from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .artifacts import ArtifactLayout, load_yaml
from .paths import ARTIFACT_DIRS, ReleasePaths
from .registry import load_model_registry


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


REQUIRED_CONFIGS = (
    "artifact_layout.yaml",
    "artifact_manifest.yaml",
    "legacy_run_contract.yaml",
    "model_registry.yaml",
    "paper_run_profiles.yaml",
    "paper_tables.yaml",
    "profile_sources.yaml",
    "release_visibility.yaml",
)


def check_layout(paths: ReleasePaths | None = None) -> list[CheckResult]:
    release_paths = paths or ReleasePaths.from_env()
    results: list[CheckResult] = []

    for filename in REQUIRED_CONFIGS:
        path = release_paths.config_path(filename)
        results.append(CheckResult(f"config:{filename}", path.exists(), str(path)))

    try:
        layout = ArtifactLayout.load(release_paths)
        results.append(CheckResult("artifact_layout", True, f"{len(layout.bundles)} bundles, {len(layout.groups)} groups"))
    except Exception as exc:
        results.append(CheckResult("artifact_layout", False, str(exc)))
        layout = None

    if layout is not None:
        try:
            manifest = load_yaml(release_paths.config_path("artifact_manifest.yaml"))
            layout_bundle_ids = set(layout.bundles)
            manifest_bundle_ids = {str(item.get("bundle_id", "")) for item in manifest.get("bundles", [])}
            bundle_match = layout_bundle_ids == manifest_bundle_ids
            not_ready = [
                str(item.get("bundle_id", ""))
                for item in manifest.get("bundles", [])
                if str(item.get("status", "")) != "ready"
            ]
            detail = f"{len(manifest_bundle_ids)} bundles"
            if not bundle_match:
                detail += f", layout_only={sorted(layout_bundle_ids - manifest_bundle_ids)}, manifest_only={sorted(manifest_bundle_ids - layout_bundle_ids)}"
            if not_ready:
                detail += f", not_ready={not_ready}"
            results.append(CheckResult("artifact_manifest", bundle_match, detail))
        except Exception as exc:
            results.append(CheckResult("artifact_manifest", False, str(exc)))

    try:
        models = load_model_registry(release_paths)
        results.append(CheckResult("model_registry", len(models) == 20, f"{len(models)} models"))
    except Exception as exc:
        results.append(CheckResult("model_registry", False, str(exc)))

    return results


def ensure_directories(paths: ReleasePaths | None = None) -> list[Path]:
    release_paths = paths or ReleasePaths.from_env()
    created: list[Path] = []
    for relative_path in ARTIFACT_DIRS.values():
        path = release_paths.artifact_root / relative_path
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created

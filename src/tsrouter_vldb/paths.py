from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ARTIFACT_ROOT_ENV = "TSROUTER_VLDB_ARTIFACT_ROOT"
HF_REPO_ENV = "TSROUTER_VLDB_HF_REPO"


ARTIFACT_DIRS = {
    "tsfm_results": Path("artifacts/tsfm_results"),
    "profile_anchors": Path("artifacts/profile_anchors"),
    "profile_forwards": Path("artifacts/profile_forwards"),
    "capability_indexes": Path("artifacts/capability_indexes"),
    "route_results": Path("artifacts/route_results"),
    "insert_logs": Path("artifacts/insert_logs"),
    "baseline_results": Path("artifacts/baseline_results"),
    "task_sample_cache": Path("artifacts/task_sample_cache"),
    "summary_tables": Path("artifacts/summary_tables"),
    "figure_sources": Path("artifacts/figure_sources"),
    "profile_sources": Path("data/profile_sources"),
}


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ReleasePaths:
    root: Path
    config_dir: Path
    artifact_root: Path
    hf_cache_dir: Path

    @classmethod
    def from_env(cls, root: str | os.PathLike[str] | None = None) -> "ReleasePaths":
        release_root = Path(root).resolve() if root else package_root()
        artifact_root = Path(os.environ.get(ARTIFACT_ROOT_ENV, release_root)).resolve()
        return cls(
            root=release_root,
            config_dir=release_root / "configs",
            artifact_root=artifact_root,
            hf_cache_dir=artifact_root / ".cache" / "huggingface",
        )

    def artifact_path(self, key: str) -> Path:
        try:
            relative_path = ARTIFACT_DIRS[key]
        except KeyError as exc:
            valid = ", ".join(sorted(ARTIFACT_DIRS))
            raise KeyError(f"unknown artifact key {key!r}; valid keys: {valid}") from exc
        return self.artifact_root / relative_path

    def config_path(self, filename: str) -> Path:
        return self.config_dir / filename


def default_hf_repo() -> str | None:
    value = os.environ.get(HF_REPO_ENV)
    return value.strip() if value else None

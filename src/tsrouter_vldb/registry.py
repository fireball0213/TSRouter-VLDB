from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .artifacts import ArtifactConfigError, load_yaml
from .paths import ReleasePaths


@dataclass(frozen=True)
class ModelEntry:
    registry_key: str
    stage_id: int
    family: str
    variant: str
    name: str
    abbreviation: str
    source_repo: str
    release_date: str

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ModelEntry":
        return cls(
            registry_key=str(data["registry_key"]),
            stage_id=int(data["stage_id"]),
            family=str(data["family"]),
            variant=str(data["variant"]),
            name=str(data["name"]),
            abbreviation=str(data["abbreviation"]),
            source_repo=str(data["source_repo"]),
            release_date=str(data["release_date"]),
        )


def load_model_registry(paths: ReleasePaths | None = None) -> list[ModelEntry]:
    release_paths = paths or ReleasePaths.from_env()
    raw = load_yaml(release_paths.config_path("model_registry.yaml"))
    models = [ModelEntry.from_mapping(item) for item in raw.get("models", [])]
    expected_size = int(raw.get("zoo_size", len(models)))
    if len(models) != expected_size:
        raise ArtifactConfigError(f"model registry size mismatch: expected {expected_size}, found {len(models)}")
    return sorted(models, key=lambda item: item.stage_id)


def model_order(stage: int | None = None, paths: ReleasePaths | None = None) -> list[str]:
    models = load_model_registry(paths)
    if stage is not None:
        models = [model for model in models if model.stage_id <= int(stage)]
    return [model.abbreviation for model in models]

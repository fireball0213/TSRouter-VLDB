from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("TSROUTER_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])).resolve()
SRC_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_METADATA_ROOT = RELEASE_ROOT / "data" / "benchmark_metadata"

CHECKPOINT_ROOT = Path(os.environ.get("TSROUTER_CHECKPOINT_ROOT", PROJECT_ROOT / "checkpoints")).resolve()
ENCODER_CHECKPOINT_ROOT = CHECKPOINT_ROOT / "encoders"

DATASET_ROOT = PROJECT_ROOT / "Dataset"
DATASET_PROPERTIES_PATH = Path(
    os.environ.get("TSROUTER_DATASET_PROPERTIES_PATH", BENCHMARK_METADATA_ROOT / "dataset_properties.json")
).resolve()
CHANNEL_META_PATH = Path(
    os.environ.get("TSROUTER_CHANNEL_META_PATH", BENCHMARK_METADATA_ROOT / "channel_meta.csv")
).resolve()
REPR_DATA_SOURCE_ROOT = Path(
    os.environ.get("TSROUTER_PROFILE_SOURCE_ROOT", DATASET_ROOT / "Repr_data_sourse")
).resolve()

RESULTS_CSV_ROOT = PROJECT_ROOT / "results_csv"
RESULTS_ARTIFACTS_ROOT = PROJECT_ROOT / "results_artifacts"

TSFM_CSV_ROOT = RESULTS_CSV_ROOT / "TSFM"
TSFM_ARTIFACT_ROOT = RESULTS_ARTIFACTS_ROOT / "TSFM"

TSROUTER_CSV_ROOT = RESULTS_CSV_ROOT / "TSRouter"
TSROUTER_ARTIFACT_ROOT = RESULTS_ARTIFACTS_ROOT / "TSRouter"
TSROUTER_SELECTOR_CSV_ROOT = TSROUTER_CSV_ROOT / "Selector_results"
TSROUTER_SELECTOR_ARTIFACT_ROOT = TSROUTER_ARTIFACT_ROOT / "Selector_results"
TSROUTER_REPR_FORWARD_CSV_ROOT = TSROUTER_CSV_ROOT / "Repr_forward"
TSROUTER_REPR_FORWARD_ARTIFACT_ROOT = TSROUTER_ARTIFACT_ROOT / "Repr_forward"
TSROUTER_ANCHOR_ROOT = TSROUTER_ARTIFACT_ROOT / "Sampled_repr_anchor"
TSROUTER_MODEL_REPR_ROOT = TSROUTER_ARTIFACT_ROOT / "Model_zoo_repr"
TSROUTER_TRAINED_ENCODER_ROOT = TSROUTER_ARTIFACT_ROOT / "trained_encoders"
TSROUTER_ROUTE_ARTIFACT_ROOT = TSROUTER_ARTIFACT_ROOT / "route"
TSROUTER_INSERT_ARTIFACT_ROOT = TSROUTER_ARTIFACT_ROOT / "insert"
TSROUTER_PROFILE_ARTIFACT_ROOT = TSROUTER_ARTIFACT_ROOT / "profile"

BASELINE_CSV_ROOT = RESULTS_CSV_ROOT / "baselines"
BASELINE_ARTIFACT_ROOT = RESULTS_ARTIFACTS_ROOT / "baselines"

CACHE_ROOT = RESULTS_ARTIFACTS_ROOT / "caches"
GE_TEST_SAMPLE_CACHE_ROOT = CACHE_ROOT / "GE_test_sample"
SAMPLED_REPR_POOL_CACHE_ROOT = CACHE_ROOT / "Sampled_repr_pool"

ANALYSIS_CSV_ROOT = RESULTS_CSV_ROOT / "analysis"
ANALYSIS_ARTIFACT_ROOT = RESULTS_ARTIFACTS_ROOT / "analysis"

TSROUTER_VLDB_CSV_ROOT = TSROUTER_CSV_ROOT / "vldb"
TSROUTER_VLDB_LOG_ROOT = TSROUTER_VLDB_CSV_ROOT / "logs"
TSROUTER_VLDB_TABLE_ROOT = TSROUTER_VLDB_CSV_ROOT / "tables"
TSROUTER_VLDB_MANIFEST_ROOT = TSROUTER_VLDB_CSV_ROOT / "manifests"


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def resolve_checkpoint_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    parts = path.parts
    if parts and parts[0] == "checkpoints":
        return CHECKPOINT_ROOT.joinpath(*parts[1:]).resolve()
    return (PROJECT_ROOT / path).resolve()


def ensure_project_dirs() -> None:
    for path in [
        CHECKPOINT_ROOT,
        ENCODER_CHECKPOINT_ROOT,
        DATASET_ROOT,
        REPR_DATA_SOURCE_ROOT,
        TSFM_CSV_ROOT,
        TSFM_ARTIFACT_ROOT,
        TSROUTER_SELECTOR_CSV_ROOT,
        TSROUTER_SELECTOR_ARTIFACT_ROOT,
        TSROUTER_REPR_FORWARD_CSV_ROOT,
        TSROUTER_REPR_FORWARD_ARTIFACT_ROOT,
        TSROUTER_ANCHOR_ROOT,
        TSROUTER_MODEL_REPR_ROOT,
        TSROUTER_TRAINED_ENCODER_ROOT,
        TSROUTER_ROUTE_ARTIFACT_ROOT,
        TSROUTER_INSERT_ARTIFACT_ROOT,
        TSROUTER_PROFILE_ARTIFACT_ROOT,
        BASELINE_CSV_ROOT,
        BASELINE_ARTIFACT_ROOT,
        GE_TEST_SAMPLE_CACHE_ROOT,
        SAMPLED_REPR_POOL_CACHE_ROOT,
        ANALYSIS_CSV_ROOT,
        ANALYSIS_ARTIFACT_ROOT,
        TSROUTER_VLDB_LOG_ROOT,
        TSROUTER_VLDB_TABLE_ROOT,
        TSROUTER_VLDB_MANIFEST_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)

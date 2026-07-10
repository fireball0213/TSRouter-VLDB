import os
import sys
import pickle
import json
import hashlib
import importlib
import shutil
import time
import warnings
import random
import argparse
import csv
import glob
from pathlib import Path
from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist, pdist, squareform

from statsmodels.tsa.seasonal import STL
from gluonts.dataset.common import ListDataset

# =========================
               
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
parent_parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
sys.path.append(parent_dir)
sys.path.append(parent_parent_dir)
sys.path.append("..")                      

# =========================
         
# =========================
from config.dataset_config import ALL_DATASETS, Short_Fast_datasets
from config.model_zoo_config import (
    All_sorted_model_names,
    Model_abbrev_map,
    Model_zoo_details,
    build_model_family_metadata,
    validate_model_family_metadata,
)
from utils.decomposition import decomposition_method
from utils.path_utils import (
    TSROUTER_SAMPLED_REPR_POOL_DIR,
    build_repr_eval_pool_name,
    build_repr_eval_pool_forward_stem,
    build_repr_forward_all_results_stem,
    build_repr_forward_stem,
    build_repr_set_name,
    get_auto_cl_mode,
    get_auto_cl_profile_name,
    get_advanced_baseline_train_scope,
    get_gift_eval_task_repr_cache_path,
    get_tsrouter_repr_forward_dir,
    get_repr_save_path,
)
from utils.io_lock import atomic_pickle_dump
from utils.tsrouter_metrics import (
    COMPETENCE_REGION_SCHEMA_VERSION,
    encoder_enrichment_paths,
    encoder_enrichment_suffix,
    parse_order_string,
    rank_decay_coef,
    rank_position_scores,
    real_channel_rank_cache_path,
    resolve_process_metrics_region_rule,
    save_competence_region_report,
    save_encoder_enrichment_report,
)
from utils.project_paths import BASELINE_CSV_ROOT, TSROUTER_CSV_ROOT

warnings.filterwarnings("ignore")          


STEP2_RUNTIME_WEIGHT_COLUMNS = [
    "insert_runtime_seconds",
    "forward_runtime_seconds",
    "non_eval_runtime_seconds",
    "runtime_seconds",
]

STRICT_RANK_WEIGHT_ALGORITHM = "strict_rank_lower_better_v1"
STRICT_RANK_WEIGHT_TIE_BREAKER = "model_order"
POOL_EMBEDDING_CACHE_SCHEMA_VERSION = 2


def _route_efficiency_mode_enabled(args) -> bool:
    raw = getattr(args, "route_efficiency_mode", False)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"true", "1", "yes", "y", "t"}


def _auto_cl_sidecar_fields(args) -> dict:
    source_len = getattr(args, "repr_source_exact_length", None)
    return {
        "auto_cl_mode": get_auto_cl_mode(args),
        "adaptive_profile": get_auto_cl_profile_name(args),
        "repr_source_exact_length": int(source_len) if source_len is not None else None,
        "repr_input_dim": int(getattr(args, "repr_input_dim", 0)),
        "repr_output_dim": int(getattr(args, "repr_output_dim", 0)),
        "repr_sub_pred_len": int(getattr(args, "repr_sub_pred_len", 0)),
    }


def _effective_repr_weight_ratio(args) -> float:
    return float(getattr(args, "repr_weight_ratio", 0.0))


def _base_metric_display_name(args) -> str:
    code = str(getattr(args, "base_metrics", "M")).strip().upper()
    return {"C": "CRPS", "M": "MASE", "S": "sMAPE"}.get(code, code or "metric")


def _strict_lower_better_rank_weights(
    values,
    model_names: list[str],
    use_perf_scale: bool,
    perf_scale: float,
) -> np.ndarray:
    """Map lower-is-better scores to unique rank-spaced weights."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape[0] != len(model_names):
        raise ValueError(
            f"weight value/model count mismatch: values={values.shape[0]}, models={len(model_names)}"
        )
    if values.size == 0:
        return values
    if not np.isfinite(values).all():
        bad = {
            str(name): float(value)
            for name, value in zip(model_names, values)
            if not np.isfinite(value)
        }
        raise ValueError(f"non-finite values for weight ranking: {bad}")

    order = np.lexsort((np.arange(values.size, dtype=np.int64), values))
    rank_scores = np.empty(values.size, dtype=np.float64)
    if values.size == 1:
        rank_scores[order] = 1.0
    else:
        rank_scores[order] = np.linspace(1.0, 0.0, values.size, dtype=np.float64)
    if use_perf_scale:
        rank_scores = rank_scores + float(perf_scale)
    return rank_scores


def _print_step3_weight_prior(
    *,
    source: str,
    metric: str,
    model_names: list[str],
    score_values,
    weights,
) -> None:
    score_values = np.asarray(score_values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if score_values.shape[0] != len(model_names) or weights.shape[0] != len(model_names):
        print(
            f"[Step3][weight-prior] unable to print weights: "
            f"models={len(model_names)}, scores={score_values.shape[0]}, weights={weights.shape[0]}",
            flush=True,
        )
        return

    ranked = [
        (idx, name, float(score), float(weight))
        for idx, (name, score, weight) in enumerate(zip(model_names, score_values, weights))
    ]
    ranked.sort(key=lambda item: (-item[3], item[0]))
    by_rank = ", ".join(
        f"{rank}:{name}(score={score:.6g},weight={weight:.6g})"
        for rank, (_, name, score, weight) in enumerate(ranked, start=1)
    )
    by_model = ", ".join(
        f"{name}={float(weight):.6g}"
        for name, weight in zip(model_names, weights)
    )
    print(
        f"[Step3][weight-prior] source={source}, metric={metric}, lower_better=true, "
        f"algorithm={STRICT_RANK_WEIGHT_ALGORITHM}, tie_breaker={STRICT_RANK_WEIGHT_TIE_BREAKER}",
        flush=True,
    )
    print(f"[Step3][weight-prior] weights_by_rank: {by_rank}", flush=True)
    print(f"[Step3][weight-prior] weights_by_model_order: {by_model}", flush=True)


def _repr_v5_enabled(args) -> bool:
    return str(getattr(args, "repr_v", ""))[:1] == "5"


def _step3_subset_assign_path(args, model_repr_path: str) -> str:
    suffix = encoder_enrichment_suffix(args) if _repr_v5_enabled(args) else ""
    return str(model_repr_path).replace(".pkl", f"{suffix}_subset_assign.pkl")


def _step3_manifest_path(model_repr_path: str) -> str:
    return str(model_repr_path).replace(".pkl", "_model_manifest.json")


def _step3_artifact_paths(args, weight_path: str, model_repr_path: str) -> dict[str, str]:
    summary_path, by_model_path = encoder_enrichment_paths(model_repr_path, args=args)
    return {
        "model_repr": model_repr_path,
        "weight": weight_path,
        "subset": _step3_subset_assign_path(args, model_repr_path),
        "manifest": _step3_manifest_path(model_repr_path),
        "enrichment_summary": summary_path,
        "enrichment_by_model": by_model_path,
    }


def _step3_timing_csv_path() -> str:
    return str(TSROUTER_CSV_ROOT / "Model_zoo_repr" / "step3_insert_timing.csv")


def _step3_timing_row_key(model_repr_path: str) -> str:
    return Path(str(model_repr_path)).stem


def _ordered_missing(expected: list[str], available) -> list[str]:
    available_set = {str(x) for x in available}
    return [str(x) for x in expected if str(x) and str(x) not in available_set]


def _format_missing_by_metric(missing_by_metric: dict[str, list[str]]) -> str:
    parts = []
    for metric_name, missing in missing_by_metric.items():
        if missing:
            parts.append(f"{metric_name}:{' '.join(str(x) for x in missing)}")
    return ";".join(parts)


def _step3_coverage_csv_fields(coverage: dict | None) -> dict:
    if not coverage:
        return {
            "step2_coverage_status": "",
            "step2_metric_complete": "",
            "step2_runtime_complete": "",
            "step2_expected_model_count": "",
            "step2_metric_covered_model_count": "",
            "step2_runtime_covered_model_count": "",
            "step2_missing_metric_models": "",
            "step2_missing_runtime_models": "",
            "step2_missing_by_metric": "",
            "step2_profile_coverage_note": "",
        }
    return {
        "step2_coverage_status": str(coverage.get("status", "")),
        "step2_metric_complete": bool(coverage.get("metric_complete", False)),
        "step2_runtime_complete": bool(coverage.get("runtime_complete", False)),
        "step2_expected_model_count": int(coverage.get("expected_model_count", 0) or 0),
        "step2_metric_covered_model_count": int(coverage.get("metric_covered_model_count", 0) or 0),
        "step2_runtime_covered_model_count": int(coverage.get("runtime_covered_model_count", 0) or 0),
        "step2_missing_metric_models": " ".join(str(x) for x in coverage.get("missing_metric_models", []) or []),
        "step2_missing_runtime_models": " ".join(str(x) for x in coverage.get("missing_runtime_models", []) or []),
        "step2_missing_by_metric": _format_missing_by_metric(coverage.get("missing_by_metric", {}) or {}),
        "step2_profile_coverage_note": str(coverage.get("note", "")),
    }


def _read_step3_timing_row(model_repr_path: str) -> pd.Series | None:
    path = _step3_timing_csv_path()
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "row_key" not in df.columns:
        return None
    row_key = _step3_timing_row_key(model_repr_path)
    matched = df[df["row_key"].astype(str).eq(row_key)].copy()
    if matched.empty:
        return None
    return matched.iloc[-1]


def _existing_step3_timing_status(model_repr_path: str) -> dict:
    path = _step3_timing_csv_path()
    row = _read_step3_timing_row(model_repr_path)
    if row is None:
        return {"csv": os.path.exists(path), "row": False, "index_refresh_nan": None, "status": ""}
    value = pd.to_numeric(pd.Series([row.get("index_refresh_seconds", np.nan)]), errors="coerce").iloc[0]
    return {
        "csv": True,
        "row": True,
        "index_refresh_nan": bool(pd.isna(value) or not np.isfinite(float(value))),
        "status": str(row.get("status", "")),
    }


def _upsert_step3_timing_row(row: dict) -> None:
    path = _step3_timing_csv_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_df = pd.DataFrame([row])
    if os.path.exists(path):
        try:
            old_df = pd.read_csv(path)
        except Exception:
            old_df = pd.DataFrame()
    else:
        old_df = pd.DataFrame()
    if not old_df.empty and "row_key" in old_df.columns:
        old_df = old_df[~old_df["row_key"].astype(str).eq(str(row["row_key"]))].copy()
    all_cols = list(dict.fromkeys(list(old_df.columns) + list(new_df.columns)))
    old_df = old_df.reindex(columns=all_cols)
    new_df = new_df.reindex(columns=all_cols)
    pd.concat([old_df, new_df], ignore_index=True).to_csv(path, index=False)


def _paths_for_model_repr_path(args, model_repr_path: str) -> dict[str, str]:
    model_repr_path = str(model_repr_path)
    base_dir = os.path.dirname(model_repr_path)
    stem = Path(model_repr_path).stem
    weight_path = os.path.join(base_dir, f"weight_{stem}.pkl")
    return _step3_artifact_paths(args, weight_path, model_repr_path)


def _file_ready(path: str) -> bool:
    return bool(path) and os.path.exists(path) and os.path.getsize(path) > 0


def _read_json_file(path: str) -> dict:
    if not _file_ready(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _read_pickle_file(path: str):
    if not _file_ready(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _model_order_from_artifacts(paths: dict[str, str]) -> tuple[list[str], str]:
    manifest = _read_json_file(paths.get("manifest", ""))
    for key in ("model_abbr_order", "model_names"):
        value = manifest.get(key)
        if isinstance(value, list) and value:
            return [str(x) for x in value], "manifest"

    subset_payload = _read_pickle_file(paths.get("subset", ""))
    if isinstance(subset_payload, dict):
        for key in ("model_abbr_order", "model_names"):
            value = subset_payload.get(key)
            if isinstance(value, list) and value:
                return [str(x) for x in value], "subset"

    repr_payload = _read_pickle_file(paths.get("model_repr", ""))
    if isinstance(repr_payload, dict):
        if repr_payload.get("__repr_format__") == "v5_rank_centers":
            value = repr_payload.get("model_abbr_order")
            if isinstance(value, list) and value:
                return [str(x) for x in value], "model_repr"
        keys = [str(k) for k in repr_payload.keys() if not str(k).startswith("__")]
        if keys:
            return keys, "model_repr"
    return [], ""


def _pool_phase_done_from_artifacts(
    paths: dict[str, str],
    enable_process_metrics: bool,
    require_competence_metrics: bool = False,
) -> bool:
    if not enable_process_metrics:
        return True
    manifest = _read_json_file(paths.get("manifest", ""))
    if "pool_phase_done" in manifest:
        done = bool(manifest.get("pool_phase_done"))
        if require_competence_metrics:
            done = done and int(manifest.get("process_metrics_schema_version", 0) or 0) >= COMPETENCE_REGION_SCHEMA_VERSION
        return done
    subset_payload = _read_pickle_file(paths.get("subset", ""))
    if isinstance(subset_payload, dict):
        if "pool_phase_done" in subset_payload:
            done = bool(subset_payload.get("pool_phase_done"))
            if require_competence_metrics:
                done = done and int(subset_payload.get("process_metrics_schema_version", 0) or 0) >= COMPETENCE_REGION_SCHEMA_VERSION
            return done
        return "encoder_quality" in subset_payload or "encoder_quality_indices_dict" in subset_payload
    return False


def _cluster_phase_done_from_artifacts(paths: dict[str, str]) -> bool:
    manifest = _read_json_file(paths.get("manifest", ""))
    if "cluster_phase_done" in manifest:
        return bool(manifest.get("cluster_phase_done"))
    subset_payload = _read_pickle_file(paths.get("subset", ""))
    if isinstance(subset_payload, dict) and "cluster_phase_done" in subset_payload:
        return bool(subset_payload.get("cluster_phase_done"))
    return True


def _merge_existing_pool_payload(payload: dict, existing_payload: dict | None) -> dict:
    if not isinstance(existing_payload, dict) or not bool(existing_payload.get("pool_phase_done", False)):
        return payload
    for key in [
        "selected_indices_dict",
        "encoder_quality_indices_dict",
        "encoder_quality",
        "total_repr_pool",
        "mode",
        "competence_region",
        "process_metrics_schema_version",
        "anchor_pool_embedding_alignment",
        "assignment_rule_diagnostics",
    ]:
        if key in existing_payload:
            payload[key] = existing_payload[key]
    payload["pool_phase_done"] = True
    return payload


def _step3_core_status(
    args,
    paths: dict[str, str],
    expected_model_order: list[str],
    require_manifest: bool,
) -> dict:
    core_keys = ["model_repr", "weight", "subset"]
    if require_manifest:
        core_keys.append("manifest")
    file_status = {key: _file_ready(paths.get(key, "")) for key in core_keys}
    model_order, model_order_source = _model_order_from_artifacts(paths)
    models_match = bool(model_order) and list(model_order) == list(expected_model_order)
    cluster_done = _cluster_phase_done_from_artifacts(paths)
    cluster_complete = all(file_status.values()) and models_match and cluster_done
    pool_done = _pool_phase_done_from_artifacts(
        paths,
        enable_process_metrics=bool(getattr(args, "enable_process_metrics", True)),
        require_competence_metrics=str(getattr(args, "repr_v", ""))[:1] in {"0", "1", "2", "3", "4"},
    )
    timing_status = _existing_step3_timing_status(paths.get("model_repr", ""))
    timing_is_skip_or_reuse = str(timing_status.get("status", "")).lower() == "skip_or_reuse"
    complete = cluster_complete and pool_done and not timing_is_skip_or_reuse
    return {
        "complete": bool(complete),
        "file_status": file_status,
        "model_order": model_order,
        "model_order_source": model_order_source,
        "models_match": bool(models_match),
        "cluster_phase_done": bool(cluster_done),
        "cluster_complete": bool(cluster_complete),
        "pool_phase_done": bool(pool_done),
        "timing_status": timing_status,
        "timing_is_skip_or_reuse": timing_is_skip_or_reuse,
    }


def _format_bool_map(values: dict[str, bool]) -> str:
    return ", ".join(f"{key}={'Y' if val else 'N'}" for key, val in values.items())


def _print_step3_status(label: str, status: dict, paths: dict[str, str]) -> None:
    report_status = {
        "enrichment_summary": _file_ready(paths.get("enrichment_summary", "")),
        "enrichment_by_model": _file_ready(paths.get("enrichment_by_model", "")),
    }
    timing = status.get("timing_status", {})
    timing_nan = timing.get("index_refresh_nan")
    timing_nan_text = "NA" if timing_nan is None else ("Y" if timing_nan else "N")
    print(
        f"[Step3][skip-save] {label}: core=({_format_bool_map(status['file_status'])}), "
        f"models_match={status['models_match']}({status['model_order_source'] or 'none'}), "
        f"cluster_complete={status.get('cluster_complete')}, "
        f"pool_phase_done={status['pool_phase_done']}, complete={status['complete']}, "
        f"reports=({_format_bool_map(report_status)}), "
        f"timing=(csv={'Y' if timing.get('csv') else 'N'}, row={'Y' if timing.get('row') else 'N'}, "
        f"index_refresh_nan={timing_nan_text}, status={timing.get('status') or 'NA'})"
    )


def _warn_step3_timing_if_nan(status: dict) -> None:
    timing = status.get("timing_status", {})
    if timing.get("index_refresh_nan") is True:
        print(
            "⚠️ [Step3][skip-save] existing timing has index_refresh_seconds=NaN; "
            "skip build as requested, but Step2 insert_runtime_seconds should be backfilled."
        )
    if status.get("timing_is_skip_or_reuse"):
        print(
            "⚠️ [Step3][skip-save] existing timing status=skip_or_reuse is not accepted; "
            "Step3 will rebuild timing instead of skipping."
        )


def _write_step3_manifest(
    args,
    repr_set_name: str,
    weight_path: str,
    model_repr_path: str,
    subset_assign_path: str,
    model_names: list[str],
    pool_phase_done: bool,
    reused_from: dict | None = None,
) -> None:
    stage = int(getattr(args, "current_zoo_num", len(model_names)) or len(model_names))
    total = int(getattr(args, "zoo_total_num", stage) or stage)
    subset_payload = _read_pickle_file(subset_assign_path)
    process_metrics_schema_version = (
        int(subset_payload.get("process_metrics_schema_version", 0) or 0)
        if isinstance(subset_payload, dict)
        else 0
    )
    manifest = {
        "schema_version": 2,
        "stage": stage,
        "zoo_total_num": total,
        "zoo_tag": f"zoo{stage}-{total}",
        "model_abbr_order": list(model_names),
        "model_count": int(len(model_names)),
        "repr_set_name": repr_set_name,
        "repr_encoder": str(getattr(args, "repr_encoder", "")),
        "simplets_ts2vec_checkpoint": str(
            getattr(args, "simplets_ts2vec_checkpoint", "") or ""
        ),
        "simplets_ts2vec_checkpoint_fingerprint": str(
            getattr(args, "simplets_ts2vec_checkpoint_fingerprint", "") or ""
        ),
        "simplets_ts2vec_source_repr_set_name": str(
            getattr(args, "simplets_ts2vec_source_repr_set_name", "") or ""
        ),
        "model_repr_name": Path(model_repr_path).stem,
        "weight_file": os.path.basename(weight_path),
        "subset_assign_file": os.path.basename(subset_assign_path),
        "cluster_phase_done": True,
        "pool_phase_done": bool(pool_phase_done),
        "enable_process_metrics": bool(getattr(args, "enable_process_metrics", True)),
        "process_metrics_schema_version": process_metrics_schema_version,
        **build_model_family_metadata(model_names),
        **_auto_cl_sidecar_fields(args),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if reused_from:
        manifest["reused_from"] = dict(reused_from)
    path = _step3_manifest_path(model_repr_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _candidate_same_stage_model_repr_paths(args, model_repr_path: str) -> list[Path]:
    target = Path(model_repr_path)
    target_stem = target.stem
    if "_" not in target_stem:
        return []
    stage = int(getattr(args, "current_zoo_num", 0) or 0)
    tail = target_stem.split("_", 1)[1]
    roots = [target.parent, Path(str(getattr(args, "save_model_repr_path", "")))]
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root or not root.exists():
            continue
        for path in root.glob(f"zoo{stage}-*_{tail}.pkl"):
            try:
                resolved = str(path.resolve())
                target_resolved = str(target.resolve())
            except Exception:
                resolved = str(path)
                target_resolved = str(target)
            if resolved == target_resolved or resolved in seen:
                continue
            seen.add(resolved)
            out.append(path)
    return sorted(out, key=lambda p: p.name)


def _copy_pickle_with_metadata(src: str, dst: str, updates: dict) -> None:
    payload = _read_pickle_file(src)
    if isinstance(payload, dict):
        payload = dict(payload)
        payload.update(updates)
        atomic_pickle_dump(payload, dst)
    else:
        shutil.copy2(src, dst)


def _try_reuse_same_stage_artifacts(
    args,
    repr_set_name: str,
    weight_path: str,
    model_repr_path: str,
    expected_model_order: list[str],
) -> bool:
    target_paths = _step3_artifact_paths(args, weight_path, model_repr_path)
    for candidate in _candidate_same_stage_model_repr_paths(args, model_repr_path):
        source_paths = _paths_for_model_repr_path(args, str(candidate))
        status = _step3_core_status(args, source_paths, expected_model_order, require_manifest=False)
        if not status["complete"]:
            _print_step3_status(f"reuse-candidate {candidate.name}", status, source_paths)
            continue

        os.makedirs(os.path.dirname(model_repr_path), exist_ok=True)
        shutil.copy2(source_paths["model_repr"], target_paths["model_repr"])
        if _file_ready(source_paths["weight"]):
            _copy_pickle_with_metadata(
                source_paths["weight"],
                target_paths["weight"],
                {"reused_from_weight_file": os.path.basename(source_paths["weight"])},
            )
        if _file_ready(source_paths["subset"]):
            _copy_pickle_with_metadata(
                source_paths["subset"],
                target_paths["subset"],
                {
                    "model_abbr_order": list(expected_model_order),
                    "model_names": list(expected_model_order),
                    "stage": int(getattr(args, "current_zoo_num", len(expected_model_order))),
                    "zoo_total_num": int(getattr(args, "zoo_total_num", len(expected_model_order))),
                    "reused_from_subset_file": os.path.basename(source_paths["subset"]),
                },
            )
        reused_from = {
            "model_repr_path": str(candidate),
            "model_repr_file": candidate.name,
            "model_order_source": status.get("model_order_source", ""),
        }
        _write_step3_manifest(
            args=args,
            repr_set_name=repr_set_name,
            weight_path=weight_path,
            model_repr_path=model_repr_path,
            subset_assign_path=target_paths["subset"],
            model_names=list(expected_model_order),
            pool_phase_done=True,
            reused_from=reused_from,
        )
        print(
            f"[Step3][reuse] same-stage artifacts reused: {candidate.name} -> {Path(model_repr_path).name}"
        )
        return True
    return False


def _skip_or_reuse_step3_if_ready(
    args,
    repr_set_name: str,
    weight_path: str,
    model_repr_path: str,
    expected_model_order: list[str],
) -> bool:
    paths = _step3_artifact_paths(args, weight_path, model_repr_path)
    status = _step3_core_status(args, paths, expected_model_order, require_manifest=True)
    _print_step3_status("target", status, paths)
    _warn_step3_timing_if_nan(status)
    if status["complete"]:
        manifest = _read_json_file(paths.get("manifest", ""))
        family_problems = validate_model_family_metadata(
            manifest,
            expected_model_order,
        )
        if family_problems:
            _write_step3_manifest(
                args=args,
                repr_set_name=repr_set_name,
                weight_path=weight_path,
                model_repr_path=model_repr_path,
                subset_assign_path=paths["subset"],
                model_names=list(expected_model_order),
                pool_phase_done=bool(status.get("pool_phase_done", False)),
                reused_from=manifest.get("reused_from") if isinstance(manifest, dict) else None,
            )
            print(
                "[Step3][skip-save] repaired model-family metadata in existing manifest; "
                f"model_repr={model_repr_path}"
            )
        print(f"⏭️ [Step3][skip-save] all required core artifacts exist; skip build: {model_repr_path}")
        return True

    legacy_status = _step3_core_status(args, paths, expected_model_order, require_manifest=False)
    if legacy_status["complete"]:
        _write_step3_manifest(
            args=args,
            repr_set_name=repr_set_name,
            weight_path=weight_path,
            model_repr_path=model_repr_path,
            subset_assign_path=paths["subset"],
            model_names=list(expected_model_order),
            pool_phase_done=True,
        )
        print(f"[Step3][skip-save] legacy core artifacts matched; manifest repaired and build skipped.")
        return True

    if _try_reuse_same_stage_artifacts(
        args=args,
        repr_set_name=repr_set_name,
        weight_path=weight_path,
        model_repr_path=model_repr_path,
        expected_model_order=list(expected_model_order),
    ):
        return True

    print("[Step3][skip-save] build required; no complete target or reusable same-stage artifacts found.")
    return False

'TSRouter runtime message.'

def _dedupe_per_sample_metric_file(file_path: str) -> int:
    """Keep the latest row for each (model, metric) pair in repr-forward per-sample CSV."""
    if not os.path.exists(file_path):
        return 0
    with open(file_path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return 0

    has_header = len(rows[0]) >= 2 and rows[0][0] == "model" and rows[0][1] == "metric"
    header = rows[0] if has_header else None
    data_rows = rows[1:] if has_header else rows
    latest = OrderedDict()
    malformed = []
    duplicate_count = 0
    for row in data_rows:
        if len(row) < 2:
            malformed.append(row)
            continue
        key = (row[0].strip(), row[1].strip())
        if key in latest:
            duplicate_count += 1
        latest[key] = row

    if duplicate_count <= 0:
        return 0

    out_rows = []
    if header is not None:
        out_rows.append(header)
    out_rows.extend(latest.values())
    out_rows.extend(malformed)
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)
    print(
        f"[PER-SAMPLE CLEAN] removed {duplicate_count} duplicate rows from {file_path}; "
        "keep latest row by (model, metric)"
    )
    return duplicate_count


def _distance_metric_name(args) -> str:
    abbr_map = {
        "euc": "euclidean",
        "cos": "cosine",
        "cor": "correlation",
        "cit": "cityblock",
    }
    return abbr_map.get(str(getattr(args, "repr_distance_metric", "cos")), "cosine")


def _encoder_sim_param_text(args) -> str:
    return (
        f"repr_weight_ratio={float(getattr(args, 'repr_weight_ratio', 0.0))}, "
        f"repr_v5_nearest_k={int(getattr(args, 'repr_v5_nearest_k', 10))}, "
        f"rank_decay_coef={rank_decay_coef(args):g}, "
        f"repr_v5_distance_power={float(getattr(args, 'repr_v5_distance_power', 1.0)):g}, "
        f"repr_distance_metric={getattr(args, 'repr_distance_metric', 'cos')}, "
        f"model_repr_agg={getattr(args, 'model_repr_agg', 'min')}"
    )


def _encoder_sim_param_extra(args) -> dict:
    return {
        "repr_weight_ratio": float(getattr(args, "repr_weight_ratio", 0.0)),
        "repr_v5_nearest_k": int(getattr(args, "repr_v5_nearest_k", 10)),
        "rank_decay_coef": rank_decay_coef(args),
        "repr_v5_distance_power": float(getattr(args, "repr_v5_distance_power", 1.0)),
        "repr_distance_metric": str(getattr(args, "repr_distance_metric", "cos")),
        "model_repr_agg": str(getattr(args, "model_repr_agg", "min")),
    }


def _indices_from_assignment(assignments: np.ndarray, model_names: list[str]) -> OrderedDict:
    assignments = np.asarray(assignments, dtype=np.int64).reshape(-1)
    return OrderedDict(
        (name, np.where(assignments == i)[0].astype(np.int64))
        for i, name in enumerate(model_names)
    )


def _assign_repr_points_by_model_repr(
    center_repr: np.ndarray,
    final_repr_dict: OrderedDict,
    model_names: list[str],
    model_weights: dict,
    args,
    *,
    weight_ratio_override: float | None = None,
    agg_override: str | None = None,
    log_label: str = "v1-v4",
) -> OrderedDict:
    """Assign every repr-set point to a model using the same nearest-repr rule as v1-v4 search."""
    metric = _distance_metric_name(args)
    agg = str(agg_override or getattr(args, "model_repr_agg", "min"))
    weight_ratio = (
        _effective_repr_weight_ratio(args)
        if weight_ratio_override is None
        else float(weight_ratio_override)
    )
    eps = 1e-8
    query_batch_size = max(
        1,
        int(getattr(args, "process_metrics_distance_batch_size", 2048) or 2048),
    )
    dist_cols = []
    for name in model_names:
        model_repr = np.asarray(final_repr_dict[name], dtype=np.float32)
        if model_repr.ndim == 1:
            model_repr = model_repr.reshape(1, -1)
        distance_parts = []
        for st in range(0, int(center_repr.shape[0]), query_batch_size):
            ed = min(st + query_batch_size, int(center_repr.shape[0]))
            dists = cdist(center_repr[st:ed], model_repr, metric=metric)
            if agg == "min":
                d_part = dists.min(axis=1)
            elif agg == "min3":
                if dists.shape[1] >= 3:
                    d_part = np.partition(dists, 2, axis=1)[:, :3].mean(axis=1)
                else:
                    d_part = dists.min(axis=1)
            elif agg == "mean":
                d_part = dists.mean(axis=1)
            elif agg == "median":
                d_part = np.median(dists, axis=1)
            else:
                raise ValueError(f"Unknown model_repr_agg: {agg}")
            distance_parts.append(np.asarray(d_part, dtype=np.float64))
        d = np.concatenate(distance_parts, axis=0)
        if weight_ratio != 0:
            weight = float(model_weights.get(name, 1.0))
            d = d / (weight ** weight_ratio + eps)
        dist_cols.append(d)
    assignments = np.argmin(np.stack(dist_cols, axis=1), axis=1)
    print(
        f"[ENC_METRIC] {log_label} pool assignment: query={center_repr.shape[0]}, "
        f"metric={metric}, agg={agg}, repr_weight_ratio={weight_ratio}, "
        f"query_batch_size={query_batch_size}"
    )
    return _indices_from_assignment(assignments, model_names)


def _assignment_array_from_indices(
    selected_indices_dict: OrderedDict,
    model_names: list[str],
    n_points: int,
) -> np.ndarray:
    assignments = np.full(int(n_points), -1, dtype=np.int64)
    for owner_id, name in enumerate(model_names):
        idx = np.asarray(selected_indices_dict.get(name, []), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < n_points)]
        if idx.size:
            if np.any(assignments[idx] >= 0):
                raise ValueError(f"duplicate competence-region assignments for model={name}")
            assignments[idx] = int(owner_id)
    if np.any(assignments < 0):
        missing = np.where(assignments < 0)[0]
        raise ValueError(
            f"competence-region assignments incomplete: missing={missing.size}/{n_points}, "
            f"first={missing[:8].tolist()}"
        )
    return assignments


def _assign_repr_points_by_v5_neighbors(
    center_repr: np.ndarray,
    center_rank: np.ndarray,
    model_names: list[str],
    args,
) -> OrderedDict:
    """Assign repr-set points with the same rank-fusion idea as v5, using leave-one-out diagnostics."""
    metric = _distance_metric_name(args)
    n_points = int(center_repr.shape[0])
    n_models = len(model_names)
    topk_raw = int(getattr(args, "repr_v5_nearest_k", 10))
    leave_self = bool(getattr(args, "repr_eval_leave_self", True))
    topk = max(1, min(topk_raw, n_points - 1 if leave_self and n_points > 1 else n_points))
    power = float(getattr(args, "repr_v5_distance_power", 1.0))
    dmat = cdist(center_repr, center_repr, metric=metric)
    if leave_self and n_points > 1:
        np.fill_diagonal(dmat, np.inf)
    score_template = rank_position_scores(n_models, decay_coef=rank_decay_coef(args))
    assignments = np.zeros(n_points, dtype=np.int64)
    eps = 1e-8
    for i in range(n_points):
        idx = np.argpartition(dmat[i], kth=topk - 1)[:topk]
        d = dmat[i, idx]
        w = 1.0 / np.power(d + eps, power)
        scores = np.zeros(n_models, dtype=np.float32)
        for j, center_id in enumerate(idx):
            order = center_rank[int(center_id)]
            valid = (order >= 0) & (order < n_models)
            if np.any(valid):
                scores[order[valid]] += w[j] * score_template[:np.sum(valid)]
        assignments[i] = int(np.argmax(scores))
    print(
        f"[ENC_METRIC] v5 diagnostic assignment: topk={topk}, "
        f"leave_self={leave_self}, metric={metric}, power={power}"
    )
    return _indices_from_assignment(assignments, model_names)


def _assign_query_points_by_v5_neighbors(
    query_repr: np.ndarray,
    center_repr: np.ndarray,
    center_rank: np.ndarray,
    center_distance_weights: np.ndarray | None,
    model_names: list[str],
    args,
) -> OrderedDict:
    """Assign query points by the same nearest-center rank fusion used by Step4 v5."""
    metric = _distance_metric_name(args)
    n_query = int(query_repr.shape[0])
    n_center = int(center_repr.shape[0])
    n_models = len(model_names)
    topk = int(getattr(args, "repr_v5_nearest_k", 10))
    topk = max(1, min(topk, n_center))
    power = float(getattr(args, "repr_v5_distance_power", 1.0))
    weight_ratio = _effective_repr_weight_ratio(args)
    if center_distance_weights is None:
        center_distance_weights = np.ones(n_center, dtype=np.float32)
    center_distance_weights = np.asarray(center_distance_weights, dtype=np.float32).reshape(-1)
    if center_distance_weights.shape[0] != n_center:
        center_distance_weights = np.ones(n_center, dtype=np.float32)

    score_template = rank_position_scores(n_models, decay_coef=rank_decay_coef(args))
    assignments = np.zeros(n_query, dtype=np.int64)
    eps = 1e-8

    def _fuse_from_neighbors(idx: np.ndarray, dist: np.ndarray) -> np.ndarray:
        weights = 1.0 / np.power(dist.astype(np.float32) + eps, power)
        scores = np.zeros((idx.shape[0], n_models), dtype=np.float32)
        rows = np.arange(idx.shape[0])[:, None]
        for j in range(idx.shape[1]):
            order = center_rank[idx[:, j]]  # (batch, n_models)
            contrib = weights[:, j:j + 1] * score_template[None, :]
            np.add.at(scores, (rows, order), contrib)
        return np.argmax(scores, axis=1).astype(np.int64)

    if weight_ratio == 0:
        nn_metric = metric
        print(
            f"[ENC_METRIC] v5 nearest-neighbor search: exact sklearn topk, "
            f"query={n_query}, centers={n_center}, topk={topk}"
        )
        nn = NearestNeighbors(n_neighbors=topk, metric=nn_metric, algorithm="brute")
        nn.fit(center_repr)
        dist, idx = nn.kneighbors(query_repr, return_distance=True)
        assignments = _fuse_from_neighbors(idx.astype(np.int64), dist)
    else:
        batch = int(getattr(args, "repr_v5_pool_batch_size", 4096))
        batch = max(1, batch)
        print(
            f"[ENC_METRIC] v5 weighted distance search: exact chunked cdist, "
            f"query={n_query}, centers={n_center}, topk={topk}, batch={batch}"
        )
        print(
            "[ENC_METRIC] v5 weighted distance search work: computing batched "
            "query-to-center distances, topk neighbors, and rank-fusion assignments."
        )
        denom = np.power(center_distance_weights[None, :], weight_ratio) + eps
        for st in range(0, n_query, batch):
            ed = min(st + batch, n_query)
            dmat = cdist(query_repr[st:ed], center_repr, metric=metric)
            dmat = dmat / denom
            idx = np.argpartition(dmat, kth=topk - 1, axis=1)[:, :topk].astype(np.int64)
            dist = np.take_along_axis(dmat, idx, axis=1)
            assignments[st:ed] = _fuse_from_neighbors(idx, dist)
    print(
        f"[ENC_METRIC] v5 pool assignment: query={n_query}, centers={n_center}, "
        f"topk={topk}, metric={metric}, power={power}, repr_weight_ratio={weight_ratio}"
    )
    return _indices_from_assignment(assignments, model_names)


def _assign_pool_by_center_cluster(
    cluster_labels: np.ndarray,
    center_selected_indices_dict: OrderedDict,
    model_names: list[str],
) -> OrderedDict:
    """Propagate each center's assigned model to all candidate-pool members in its Step1 cluster."""
    labels = np.asarray(cluster_labels, dtype=np.int64).reshape(-1)
    n_centers = int(labels.max()) + 1 if labels.size else 0
    center_to_model = np.full(n_centers, -1, dtype=np.int64)
    for mi, name in enumerate(model_names):
        idx = np.asarray(center_selected_indices_dict.get(name, []), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < n_centers)]
        center_to_model[idx] = mi
    assignments = np.full(labels.shape, -1, dtype=np.int64)
    valid = (labels >= 0) & (labels < n_centers)
    assignments[valid] = center_to_model[labels[valid]]
    return _indices_from_assignment(assignments, model_names)


def _load_repr_pool_meta(args, repr_set_name: str) -> dict | None:
    pool_name = build_repr_eval_pool_name(args)
    candidates = [
        os.path.join(TSROUTER_SAMPLED_REPR_POOL_DIR, f"{pool_name}_meta.pkl"),
        os.path.join(args.save_repr_data_path, f"{pool_name}_meta.pkl"),
        os.path.join(TSROUTER_SAMPLED_REPR_POOL_DIR, f"{repr_set_name}_pool_meta.pkl"),
        os.path.join(args.save_repr_data_path, f"{repr_set_name}_pool_meta.pkl"),
    ]
    meta_path = next((p for p in candidates if os.path.exists(p)), None)
    if meta_path is None:
        print(f"⚠️ [ENC_METRIC] pool meta missing; checked: {candidates[:2]}")
        return None
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    print(f"[ENC_METRIC] loaded pool meta -> {meta_path}, seq_shape={meta.get('seq_shape')}")
    return meta


def _load_pool_metrics_matrix(args, current_zoo_abbr_order_list, model_names: list[str]) -> np.ndarray | None:
    pool_stem = build_repr_eval_pool_forward_stem(args)
    pool_csv = os.path.join(get_tsrouter_repr_forward_dir(args), pool_stem + "_per_sample_results.csv")
    if not os.path.exists(pool_csv):
        print(f"⚠️ [ENC_METRIC] pool per-sample metrics missing: {pool_csv}")
        return None
    try:
        metric_dicts = {
            "C": load_metrics_matrix(pool_csv, current_zoo_abbr_order_list, metric="CRPS", log_summary=False),
            "M": load_metrics_matrix(pool_csv, current_zoo_abbr_order_list, metric="MASE", log_summary=False),
            "S": load_metrics_matrix(pool_csv, current_zoo_abbr_order_list, metric="sMAPE", log_summary=False),
        }
    except (FileNotFoundError, ValueError) as exc:
        print(f"⚠️ [ENC_METRIC] pool per-sample metrics unavailable: {pool_csv}: {exc}")
        return None
    _print_loaded_metric_summary(
        "pool per-sample metrics",
        {"CRPS": metric_dicts["C"], "MASE": metric_dicts["M"], "sMAPE": metric_dicts["S"]},
    )
    missing_by_metric = {
        metric_name: [name for name in model_names if name not in metric_dict]
        for metric_name, metric_dict in metric_dicts.items()
    }
    missing_by_metric = {
        metric_name: missing
        for metric_name, missing in missing_by_metric.items()
        if missing
    }
    if missing_by_metric:
        print(
            f"⚠️ [ENC_METRIC] pool per-sample metrics incomplete for current Step3 models: "
            f"{missing_by_metric}; defer pool enrichment."
        )
        return None
    chosen = metric_dicts.get(str(args.base_metrics), metric_dicts["M"])
    matrix = np.stack([chosen[name] for name in model_names])
    print(f"[ENC_METRIC] loaded pool metrics -> {pool_csv}, matrix_shape={matrix.shape}")
    return matrix


def _save_pool_encoder_enrichment_if_ready(
    args,
    current_zoo_abbr_order_list,
    model_repr_path: str,
    model_names: list[str],
    selected_indices_dict: OrderedDict,
    extra: dict,
    pool_matrix: np.ndarray | None = None,
) -> dict | None:
    if pool_matrix is None:
        pool_matrix = _load_pool_metrics_matrix(args, current_zoo_abbr_order_list, model_names)
    if pool_matrix is None:
        return None
    return save_encoder_enrichment_report(
        model_repr_path=model_repr_path,
        metrics_matrix=pool_matrix,
        model_names=model_names,
        selected_indices_dict=selected_indices_dict,
        extra=extra,
        args=args,
    )


def _load_repr_data_and_encoder_for_step3(args, repr_set_name: str):
    repr_data_path = args.save_repr_data_path + "/" + repr_set_name + ".pkl"
    with open(repr_data_path, "rb") as f:
        repr_data = pickle.load(f)
    if isinstance(repr_data, list):
        repr_data = np.array(repr_data, dtype=np.float32)

    from encoder.base_encoder import EncoderFactory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repr_model, scaler, configs = EncoderFactory.build_encoder(args, device=device)
    repr_model = repr_model.to(device)
    repr_model.eval()
    return repr_data, repr_model, scaler, configs.input_dim


def _pool_embedding_cache_path(args) -> str:
    cache_dir = os.path.join(TSROUTER_SAMPLED_REPR_POOL_DIR, "embeddings")
    return os.path.join(cache_dir, f"{build_repr_set_name(args)}__pool_embeddings.npy")


def _pool_embedding_cache_meta_path(cache_path: str) -> str:
    return f"{cache_path}.meta.json"


def _small_file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pool_source_signature(pool_path: str) -> dict:
    stat = os.stat(pool_path)
    return {
        "path": os.path.abspath(pool_path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _encoder_implementation_signature(args) -> dict:
    """Fingerprint the executable encoder contract, including Stats* helpers."""
    from encoder.encoder_config import ENCODER_CONFIG

    encoder_name = str(getattr(args, "repr_encoder", ""))
    cfg = dict(ENCODER_CONFIG.get(encoder_name, {}))
    source_paths: list[str] = []
    module_path = str(cfg.get("module_path", "") or "")
    if module_path:
        module = importlib.import_module(module_path)
        module_file = getattr(module, "__file__", None)
        if module_file:
            source_paths.append(os.path.abspath(module_file))

    configured_fusion = getattr(args, "random_stats_fusion", None)
    if configured_fusion is None:
        configured_fusion = cfg.get("random_stats_fusion", "none")
    if str(configured_fusion or "none").lower() != "none":
        stats_module = importlib.import_module("encoder.baseline.random_stats_features")
        stats_file = getattr(stats_module, "__file__", None)
        if stats_file:
            source_paths.append(os.path.abspath(stats_file))

    source_hashes = {
        os.path.basename(path): _small_file_sha256(path)
        for path in sorted(set(source_paths))
        if os.path.isfile(path)
    }
    contract = {
        "encoder": encoder_name,
        "class_name": str(cfg.get("class_name", "")),
        "module_path": module_path,
        "source_hashes": source_hashes,
        "repr_input_dim": int(getattr(args, "repr_input_dim", cfg.get("default_input_dim", 0)) or 0),
        "repr_output_dim": int(getattr(args, "repr_output_dim", cfg.get("default_embedding_dim", 0)) or 0),
        "repr_encoder_seed": int(getattr(args, "repr_encoder_seed", 0) or 0),
        "encoder_type": str(getattr(args, "encoder_type", cfg.get("encoder_type", "")) or ""),
        "encoder_structure": str(getattr(args, "encoder_structure", "") or ""),
        "random_stats_fusion": str(configured_fusion or "none").lower(),
        "random_stats_normalize": bool(
            getattr(args, "random_stats_normalize", cfg.get("random_stats_normalize", True))
        ),
        "simplets_checkpoint_fingerprint": str(
            getattr(args, "simplets_ts2vec_checkpoint_fingerprint", "") or ""
        ),
    }
    encoded = json.dumps(contract, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "contract": contract,
    }


def _expected_pool_embedding_cache_meta(args, pool_path: str, expected_rows: int | None) -> dict:
    return {
        "schema_version": POOL_EMBEDDING_CACHE_SCHEMA_VERSION,
        "repr_set_name": build_repr_set_name(args),
        "expected_rows": expected_rows,
        "pool_source": _pool_source_signature(pool_path),
        "encoder_implementation": _encoder_implementation_signature(args),
    }


def _pool_embedding_cache_meta_matches(actual: dict, expected: dict) -> tuple[bool, str]:
    if not isinstance(actual, dict):
        return False, "metadata_missing_or_invalid"
    for key in ("schema_version", "repr_set_name", "expected_rows", "pool_source", "encoder_implementation"):
        if actual.get(key) != expected.get(key):
            return False, f"metadata_changed:{key}"
    return True, "matched"


def _atomic_json_dump(payload: dict, target_path: str) -> None:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    tmp_path = f"{target_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, sort_keys=True, indent=2)
    os.replace(tmp_path, target_path)


def _load_or_encode_repr_pool_embeddings(args, pool_meta: dict) -> np.ndarray | None:
    expected_rows = None
    seq_shape = pool_meta.get("seq_shape") if isinstance(pool_meta, dict) else None
    if isinstance(seq_shape, (tuple, list)) and seq_shape:
        expected_rows = int(seq_shape[0])

    pool_name = str(pool_meta.get("pool_name") or build_repr_eval_pool_name(args))
    pool_candidates = [
        os.path.join(TSROUTER_SAMPLED_REPR_POOL_DIR, f"{pool_name}.pkl"),
        os.path.join(args.save_repr_data_path, f"{pool_name}.pkl"),
    ]
    pool_path = next((path for path in pool_candidates if os.path.exists(path)), None)
    if pool_path is None:
        print(f"⚠️ [ENC_METRIC] candidate pool data missing; checked: {pool_candidates}")
        return None

    cache_path = _pool_embedding_cache_path(args)
    cache_meta_path = _pool_embedding_cache_meta_path(cache_path)
    expected_meta = _expected_pool_embedding_cache_meta(args, pool_path, expected_rows)
    if os.path.exists(cache_path):
        try:
            actual_meta = {}
            if os.path.exists(cache_meta_path):
                with open(cache_meta_path, "r", encoding="utf-8") as f:
                    actual_meta = json.load(f)
            meta_matches, mismatch_reason = _pool_embedding_cache_meta_matches(actual_meta, expected_meta)
            cached = np.load(cache_path, mmap_mode="r")
            shape_matches = cached.ndim == 2 and (expected_rows is None or cached.shape[0] == expected_rows)
            finite = bool(np.isfinite(cached).all()) if shape_matches else False
            if meta_matches and shape_matches and finite:
                print(f"[ENC_METRIC] loaded pool embeddings cache -> {cache_path}, shape={cached.shape}")
                return np.array(cached, dtype=np.float32, copy=True)
            print(
                f"⚠️ [ENC_METRIC] ignore stale pool embeddings cache: {cache_path}, "
                f"reason={mismatch_reason}, shape={getattr(cached, 'shape', None)}, "
                f"expected_rows={expected_rows}, finite={finite}"
            )
        except Exception as exc:
            print(f"⚠️ [ENC_METRIC] failed to read pool embeddings cache {cache_path}: {exc}")

    with open(pool_path, "rb") as f:
        pool_data = pickle.load(f)
    pool_data = np.asarray(pool_data, dtype=np.float32)
    if pool_data.ndim != 2 or pool_data.shape[0] == 0:
        print(f"⚠️ [ENC_METRIC] invalid candidate pool data: {pool_path}, shape={pool_data.shape}")
        return None

    from encoder.base_encoder import EncoderFactory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repr_model, scaler, configs = EncoderFactory.build_encoder(args, device=device)
    repr_model = repr_model.to(device)
    repr_model.eval()
    pool_input = pool_data[:, :int(configs.input_dim)]
    if scaler:
        pool_input = scaler.transform(pool_input)

    dataset = TensorDataset(torch.from_numpy(np.asarray(pool_input, dtype=np.float32)))
    loader = DataLoader(
        dataset,
        batch_size=min(int(args.batch_size), len(dataset)),
        shuffle=False,
        drop_last=False,
    )
    feats = []
    print(f"[ENC_METRIC] encoding candidate pool -> {pool_path}, input_shape={pool_input.shape}")
    for batch_x in loader:
        batch_data = batch_x[0].unsqueeze(-1).to(device)
        with torch.no_grad():
            feats.append(repr_model.encode(batch_data).detach().cpu().numpy())
    embeddings = np.concatenate(feats, axis=0).astype(np.float32)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.tmp.{os.getpid()}.npy"
    np.save(tmp_path, embeddings)
    os.replace(tmp_path, cache_path)
    _atomic_json_dump(expected_meta, cache_meta_path)
    print(f"[ENC_METRIC] saved pool embeddings cache -> {cache_path}, shape={embeddings.shape}")
    return embeddings


def _load_anchor_pool_member_indices(args, repr_set_name: str) -> tuple[np.ndarray | None, str]:
    candidates = [
        os.path.join(args.save_repr_data_path, f"{repr_set_name}_meta.pkl"),
        os.path.join(TSROUTER_SAMPLED_REPR_POOL_DIR, f"{repr_set_name}_meta.pkl"),
    ]
    meta_path = next((path for path in candidates if os.path.exists(path)), None)
    if meta_path is None:
        return None, ""
    try:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
    except Exception as exc:
        print(f"⚠️ [ENC_METRIC][anchor-pool-check] cannot read anchor meta {meta_path}: {exc}")
        return None, meta_path
    raw = meta.get("center_member_idx_in_pool") if isinstance(meta, dict) else None
    if raw is None:
        return None, meta_path
    return np.asarray(raw, dtype=np.int64).reshape(-1), meta_path


def _validate_anchor_pool_embedding_alignment(
    args,
    repr_set_name: str,
    pool_embeddings: np.ndarray,
    final_repr_dict: OrderedDict,
    selected_indices_dict: OrderedDict,
    model_names: list[str],
) -> dict:
    """Fail fast when Step3 anchors and the cached/full candidate pool do not share one embedding space."""
    if str(getattr(args, "model_repr_mode", "all")).lower() != "all":
        return {"status": "skipped", "reason": "model_repr_mode_not_all"}

    member_indices, meta_path = _load_anchor_pool_member_indices(args, repr_set_name)
    if member_indices is None:
        print(
            "⚠️ [ENC_METRIC][anchor-pool-check] center_member_idx_in_pool unavailable; "
            f"cannot verify anchor/pool lineage (meta={meta_path or 'missing'})."
        )
        return {"status": "unavailable", "reason": "center_member_idx_missing", "meta_path": meta_path}

    n_anchors = int(member_indices.shape[0])
    embedding_dim = int(pool_embeddings.shape[1])
    reconstructed = np.full((n_anchors, embedding_dim), np.nan, dtype=np.float32)
    filled = np.zeros(n_anchors, dtype=bool)
    for name in model_names:
        anchor_indices = np.asarray(selected_indices_dict.get(name, []), dtype=np.int64).reshape(-1)
        model_repr = np.asarray(final_repr_dict.get(name, []), dtype=np.float32)
        if model_repr.ndim == 1 and model_repr.size:
            model_repr = model_repr.reshape(1, -1)
        if model_repr.shape != (anchor_indices.size, embedding_dim):
            print(
                "⚠️ [ENC_METRIC][anchor-pool-check] cannot reconstruct anchor embeddings: "
                f"model={name}, indices={anchor_indices.size}, repr_shape={model_repr.shape}, "
                f"expected_dim={embedding_dim}"
            )
            return {"status": "unavailable", "reason": "model_repr_shape_not_anchor_aligned"}
        if anchor_indices.size == 0:
            continue
        if np.any((anchor_indices < 0) | (anchor_indices >= n_anchors)):
            raise ValueError(f"anchor indices out of range for model={name}: n_anchors={n_anchors}")
        if np.any(filled[anchor_indices]):
            raise ValueError(f"duplicate anchor ownership while validating pool alignment: model={name}")
        reconstructed[anchor_indices] = model_repr
        filled[anchor_indices] = True

    valid = filled & (member_indices >= 0) & (member_indices < int(pool_embeddings.shape[0]))
    if not np.any(valid):
        print("⚠️ [ENC_METRIC][anchor-pool-check] no valid mapped anchors; lineage check unavailable")
        return {"status": "unavailable", "reason": "no_valid_mapped_anchors"}

    anchor_repr = reconstructed[valid].astype(np.float64)
    mapped_pool_repr = np.asarray(pool_embeddings[member_indices[valid]], dtype=np.float64)
    delta_norm = np.linalg.norm(anchor_repr - mapped_pool_repr, axis=1)
    base_norm = np.maximum(np.linalg.norm(anchor_repr, axis=1), 1.0)
    relative_l2 = delta_norm / base_norm
    mismatch_threshold = 5e-4
    mismatch_count = int(np.sum(~np.isfinite(relative_l2) | (relative_l2 > mismatch_threshold)))
    result = {
        "status": "matched" if mismatch_count == 0 else "mismatch",
        "checked": int(relative_l2.size),
        "mismatch_count": mismatch_count,
        "max_relative_l2": float(np.nanmax(relative_l2)),
        "p99_relative_l2": float(np.nanquantile(relative_l2, 0.99)),
        "meta_path": meta_path,
    }
    print(
        "[ENC_METRIC][anchor-pool-check] "
        f"status={result['status']}, checked={result['checked']}, mismatches={mismatch_count}, "
        f"p99_rel_l2={result['p99_relative_l2']:.3g}, max_rel_l2={result['max_relative_l2']:.3g}"
    )
    if mismatch_count:
        raise RuntimeError(
            "[ENC_METRIC] anchor/pool embedding lineage mismatch. Step3 model advantage anchors and "
            "candidate-pool embeddings are not in the same representation space; reporting competence "
            "regions would be invalid. Rebuild Step1 anchor+pool artifacts together with skip_saved=false, "
            "then rerun Step2 and Step3."
        )
    return result


def _assignment_distribution(assignments: np.ndarray, model_names: list[str]) -> dict:
    counts = np.bincount(np.asarray(assignments, dtype=np.int64), minlength=len(model_names))
    total = int(counts.sum())
    dominant_idx = int(np.argmax(counts)) if counts.size else -1
    return {
        "counts": {name: int(counts[i]) for i, name in enumerate(model_names)},
        "active_models": int(np.sum(counts > 0)),
        "dominant_model": model_names[dominant_idx] if dominant_idx >= 0 else "",
        "dominant_share": float(counts[dominant_idx] / total) if total and dominant_idx >= 0 else 0.0,
    }


def _log_assignment_rule_comparison(
    strict_assignments: np.ndarray,
    effective_assignments: np.ndarray,
    model_names: list[str],
) -> dict:
    strict = _assignment_distribution(strict_assignments, model_names)
    effective = _assignment_distribution(effective_assignments, model_names)
    changed_rate = float(np.mean(strict_assignments != effective_assignments)) if strict_assignments.size else 0.0
    print(f"[ENC_METRIC] strict-1NN subset sizes: {strict['counts']}")
    print(
        "[ENC_METRIC][assignment-shift] "
        f"strict_active={strict['active_models']}/{len(model_names)}, "
        f"strict_top={strict['dominant_model']}:{strict['dominant_share']:.2%}, "
        f"effective_active={effective['active_models']}/{len(model_names)}, "
        f"effective_top={effective['dominant_model']}:{effective['dominant_share']:.2%}, "
        f"changed={changed_rate:.2%}"
    )
    if effective["dominant_share"] >= 0.95 and effective["active_models"] <= 2:
        cause = (
            "weight prior / effective distance"
            if strict["dominant_share"] < 0.95 or strict["active_models"] > 2
            else "embedding geometry"
        )
        print(
            "⚠️ [ENC_METRIC][assignment-collapse] effective competence regions collapsed: "
            f"likely_cause={cause}; inspect strict/effective reports separately."
        )
    return {"strict": strict, "effective": effective, "changed_rate": changed_rate}


def _run_non_v5_pool_phase(
    args,
    repr_set_name: str,
    weight_path: str,
    model_repr_path: str,
    subset_assign_path: str,
    current_zoo_abbr_order_list,
    model_names: list[str],
    final_repr_dict: OrderedDict,
    model_weights: dict,
    selected_indices_dict: OrderedDict,
    subset_assign_payload: dict,
    metrics_matrix: np.ndarray,
    pool_phase_skip: bool,
    repr_context: tuple | None = None,
) -> tuple[bool, float]:
    step3_pool_start = time.perf_counter()
    enable_process_metrics = bool(getattr(args, "enable_process_metrics", True))
    pool_phase_done = not enable_process_metrics

    if pool_phase_skip:
        print(f"[Step3][pool] skipped: existing pool artifacts are complete -> {subset_assign_path}")
        return True, 0.0

    if enable_process_metrics:
        print("[Step3][pool] begin: assign candidate pool and compute Table-5 process metrics")
        pool_meta = _load_repr_pool_meta(args, repr_set_name)
        encoder_quality_indices_dict = None
        pool_embeddings = None
        if pool_meta is not None and "embeddings" in pool_meta:
            pool_embeddings = np.asarray(pool_meta["embeddings"], dtype=np.float32)
        elif pool_meta is not None:
            pool_embeddings = _load_or_encode_repr_pool_embeddings(args, pool_meta)
        if pool_embeddings is not None:
            subset_assign_payload["anchor_pool_embedding_alignment"] = _validate_anchor_pool_embedding_alignment(
                args=args,
                repr_set_name=repr_set_name,
                pool_embeddings=pool_embeddings,
                final_repr_dict=final_repr_dict,
                selected_indices_dict=selected_indices_dict,
                model_names=list(model_names),
            )
            encoder_quality_indices_dict = _assign_repr_points_by_model_repr(
                center_repr=pool_embeddings,
                final_repr_dict=final_repr_dict,
                model_names=list(model_names),
                model_weights=model_weights,
                args=args,
            )
        elif pool_meta is not None and "cluster_labels" in pool_meta:
            print(
                "[ENC_METRIC] pool embeddings missing; fallback to cluster-label propagation "
                "(repr_weight_ratio has no effect in this fallback)"
            )
            encoder_quality_indices_dict = _assign_pool_by_center_cluster(
                cluster_labels=np.asarray(pool_meta["cluster_labels"], dtype=np.int64),
                center_selected_indices_dict=selected_indices_dict,
                model_names=list(model_names),
            )
        if encoder_quality_indices_dict is None:
            subset_assign_payload["encoder_quality"] = {
                "status": "unavailable",
                "reason": "pool_assignment_unavailable",
            }
            print(
                "⚠️ [ENC_METRIC] pool enrichment unavailable: pool_assignment_unavailable; "
                "do not report center construction-derived Sub/PWW metrics."
            )
        else:
            print("[ENC_METRIC] construction subset sizes:", {k: int(len(v)) for k, v in selected_indices_dict.items()})
            print(
                f"[ENC_METRIC] pool-assigned subset sizes ({_encoder_sim_param_text(args)}):",
                {k: int(len(v)) for k, v in encoder_quality_indices_dict.items()},
            )
            subset_assign_payload["encoder_quality_indices_dict"] = {
                k: np.asarray(v, dtype=np.int64) for k, v in encoder_quality_indices_dict.items()
            }
            pool_matrix = _load_pool_metrics_matrix(
                args,
                current_zoo_abbr_order_list,
                list(model_names),
            )
            enrichment_summary = _save_pool_encoder_enrichment_if_ready(
                args=args,
                current_zoo_abbr_order_list=current_zoo_abbr_order_list,
                model_repr_path=model_repr_path,
                model_names=list(model_names),
                selected_indices_dict=encoder_quality_indices_dict,
                extra={
                    "metric": str(args.base_metrics),
                    "repr_v": int(args.repr_v),
                    "mode": "pool_cluster_assignment",
                    **_encoder_sim_param_extra(args),
                },
                pool_matrix=pool_matrix,
            )
            competence_reports = None
            if pool_embeddings is not None and pool_matrix is not None:
                strict_indices_dict = _assign_repr_points_by_model_repr(
                    center_repr=pool_embeddings,
                    final_repr_dict=final_repr_dict,
                    model_names=list(model_names),
                    model_weights={name: 1.0 for name in model_names},
                    args=args,
                    weight_ratio_override=0.0,
                    agg_override="min",
                    log_label="strict-1NN",
                )
                assignments_by_rule = {
                    "strict": _assignment_array_from_indices(
                        strict_indices_dict,
                        list(model_names),
                        int(pool_embeddings.shape[0]),
                    ),
                    "effective": _assignment_array_from_indices(
                        encoder_quality_indices_dict,
                        list(model_names),
                        int(pool_embeddings.shape[0]),
                    ),
                }
                subset_assign_payload["assignment_rule_diagnostics"] = _log_assignment_rule_comparison(
                    assignments_by_rule["strict"],
                    assignments_by_rule["effective"],
                    list(model_names),
                )
                competence_reports = {}
                for region_rule, assignments in assignments_by_rule.items():
                    competence_reports[region_rule] = save_competence_region_report(
                        model_repr_path=model_repr_path,
                        metrics_matrix=pool_matrix,
                        model_names=list(model_names),
                        assignments=assignments,
                        region_rule=region_rule,
                        args=args,
                    )
                primary_rule = resolve_process_metrics_region_rule(args)
                subset_assign_payload["competence_region"] = {
                    "schema_version": COMPETENCE_REGION_SCHEMA_VERSION,
                    "primary_rule": primary_rule,
                    "reports": competence_reports,
                }
                subset_assign_payload["process_metrics_schema_version"] = COMPETENCE_REGION_SCHEMA_VERSION
                print(
                    f"[REGION_METRIC] primary rule for selector summary: {primary_rule}; "
                    "both strict/effective reports were saved"
                )
            if enrichment_summary is None or competence_reports is None:
                subset_assign_payload["encoder_quality"] = {
                    "status": "unavailable",
                    "reason": (
                        "pool_per_sample_metrics_incomplete"
                        if pool_matrix is None
                        else "competence_region_assignment_unavailable"
                    ),
                }
                print(
                    "⚠️ [ENC_METRIC] Table-5 process metrics unavailable; "
                    "do not report CA/WP/DiagRank/DeltaRank."
                )
            else:
                subset_assign_payload["encoder_quality"] = enrichment_summary
                subset_assign_payload["pool_phase_done"] = True
                pool_phase_done = True
    else:
        print("[Step3][pool] skipped: enable_process_metrics=false")

    atomic_pickle_dump(subset_assign_payload, subset_assign_path)
    _write_step3_manifest(
        args=args,
        repr_set_name=repr_set_name,
        weight_path=weight_path,
        model_repr_path=model_repr_path,
        subset_assign_path=subset_assign_path,
        model_names=list(model_names),
        pool_phase_done=pool_phase_done,
    )
    print(f"[Step3][pool] done: pool_phase_done={pool_phase_done}, subset_assign={subset_assign_path}")
    return pool_phase_done, time.perf_counter() - step3_pool_start


def _check_metric_sanity(
    metric_name: str,
    model_perf_dict: OrderedDict,
    file_path: str,
    *,
    cleaning_scope: str = "global",
):  # [MOD]
    'TSRouter runtime message.'
    if not model_perf_dict:
        return

    model_names = list(model_perf_dict.keys())
    values = np.stack(list(model_perf_dict.values()))  # (n_models, n_data)
    ABS_MAX = 1e3
    RATIO_MAX = 1e5

    if cleaning_scope == "per_model":
        values_clean = values.copy()
        replacement_details: list[str] = []
        total_outliers = 0
        for model_idx, model_name in enumerate(model_names):
            model_values = values[model_idx]
            finite_mask = np.isfinite(model_values)
            finite_values = model_values[finite_mask]
            if finite_values.size == 0:
                print(
                    f"[METRIC SANITY][per_model] {metric_name} model={model_name} "
                    "has no finite values; leave unchanged"
                )
                continue
            median = float(np.median(finite_values))
            outlier_mask = ~finite_mask
            outlier_mask |= model_values > ABS_MAX
            if median > 0:
                outlier_mask |= model_values > median * RATIO_MAX
            if not outlier_mask.any():
                continue
            inlier_values = model_values[finite_mask & ~outlier_mask]
            replacement = (
                float(np.median(inlier_values))
                if inlier_values.size
                else float(min(median, ABS_MAX))
            )
            count = int(outlier_mask.sum())
            values_clean[model_idx, outlier_mask] = replacement
            total_outliers += count
            replacement_details.append(f"{model_name}:{count}@{replacement:.4g}")
        if total_outliers:
            print(
                f"  [METRIC CLEAN][per_model] Repr_forward file {metric_name}: "
                f"replaced {total_outliers} outliers with model-local medians "
                f"({', '.join(replacement_details)})"
            )
            for model_idx, model_name in enumerate(model_names):
                model_perf_dict[model_name] = values_clean[model_idx].astype(np.float32)
        return
    if cleaning_scope != "global":
        raise ValueError(f"unsupported metric cleaning_scope={cleaning_scope!r}")

                                           
    finite_mask = np.isfinite(values)
    if not finite_mask.all():
        print(f"TSRouter runtime message: {file_path}TSRouter runtime message: {metric_name}TSRouter runtime message: ")
        bad_idx = np.argwhere(~finite_mask)
        for (mi, sj) in bad_idx[:10]:
            print(f"TSRouter runtime message: {model_names[mi]}TSRouter runtime message: {sj}TSRouter runtime message: {values[mi, sj]}")
        print('TSRouter runtime message.')

    finite_values = values[finite_mask]
    if finite_values.size == 0:
        print(f"TSRouter runtime message: {file_path}TSRouter runtime message: {metric_name}TSRouter runtime message: ")
        return

    v_min = float(np.min(finite_values))
    v_max = float(np.max(finite_values))
    v_mean = float(np.mean(finite_values))
    v_median = float(np.median(finite_values))

    # print(
                                                 
    #     f"min={v_min:.6g}, max={v_max:.6g}, mean={v_mean:.6g}, median={v_median:.6g}"
    # )

                                    
    ABS_MAX = 1e3                                            
    RATIO_MAX = 1e5                                       

    suspicious = False
    reason = None
    if v_max > ABS_MAX:
        suspicious = True
        reason = f"max={v_max:.6g} > ABS_MAX={ABS_MAX}"
    elif v_median > 0 and v_max > v_median * RATIO_MAX:
        suspicious = True
        reason = f"max={v_max:.6g}TSRouter runtime message: {v_median:.6g} (ratio>{RATIO_MAX})"

                                        
                              
    outlier_mask = ~finite_mask.copy()

    if suspicious:
                             
        big_abs = (values > ABS_MAX)
        big_ratio = np.zeros_like(values, dtype=bool)
        if v_median > 0:
            big_ratio = (values > v_median * RATIO_MAX)

        outlier_mask |= big_abs | big_ratio

                  
    if not outlier_mask.any():
        return

                                          
    global_median = v_median
    num_outliers = int(outlier_mask.sum())
    print(
        f"TSRouter runtime message: {metric_name}TSRouter runtime message: {num_outliers}TSRouter runtime message: "
        f"TSRouter runtime message: {global_median:.4g}TSRouter runtime message: "
    )

        
    values_clean = values.copy()
    values_clean[outlier_mask] = global_median

                           
    for i, name in enumerate(model_names):
        model_perf_dict[name] = values_clean[i].astype(np.float32)

def load_metrics_matrix(
    file_path,
    current_zoo_abbr_order_list=None,
    metric="MASE",
    log_summary: bool = True,
    *,
    cleaning_scope: str = "global",
):
    'TSRouter runtime message.'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TSRouter runtime message: {file_path}")

    _dedupe_per_sample_metric_file(file_path)

    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    metric_lines = [
        line for line in lines
        if len(line.split(",")) >= 2 and line.split(",")[1].strip() == metric
    ]

    if not metric_lines:
        print('TSRouter runtime message.', lines[:10])
        raise ValueError(f"TSRouter runtime message: {file_path}TSRouter runtime message: {metric}TSRouter runtime message: ")

    model_perf_dict = OrderedDict()
    for line in metric_lines:
        parts = [p.strip() for p in line.split(",")]
        full_model_name = parts[0]
        model_name = Model_abbrev_map.get(full_model_name, full_model_name)

        if current_zoo_abbr_order_list is not None and model_name not in current_zoo_abbr_order_list:
            continue

                                      
        metric_values_str = parts[2:]
        if len(metric_values_str) == 0:
            print(f"TSRouter runtime message: {len(parts)}）：{line}")
            continue

        try:
            metric_values = [float(x) for x in metric_values_str]
        except ValueError as e:
            print(f"TSRouter runtime message: {line}")
            raise

        metric_arr = np.array(metric_values, dtype=np.float32)

        if model_name in model_perf_dict and len(model_perf_dict[model_name]) == len(metric_arr):
            print(f"[WARN] duplicate per-sample row for model={model_name}, metric={metric}; keep latest row")
            model_perf_dict[model_name] = metric_arr
            continue

        if model_name in model_perf_dict:
                                           
            old_len = len(model_perf_dict[model_name])
            if old_len != len(metric_arr):
                print(f"TSRouter runtime message: {model_name}TSRouter runtime message: {old_len}TSRouter runtime message: {len(metric_arr)}")
                print('TSRouter runtime message.', line)
                raise ValueError('TSRouter runtime message.')
                       
        else:
            model_perf_dict[model_name] = metric_arr

    if not model_perf_dict:
        raise ValueError(f"TSRouter runtime message: {file_path}TSRouter runtime message: ")

                             
    lengths = {name: len(v) for name, v in model_perf_dict.items()}
                                                     

                                              
    if len(set(lengths.values())) != 1:
        raise ValueError(f"TSRouter runtime message: {lengths}")

    if log_summary:
        print(f"TSRouter runtime message: {len(model_perf_dict)}TSRouter runtime message: {metric}TSRouter runtime message: {next(iter(lengths.values()))}")

    _check_metric_sanity(
        metric,
        model_perf_dict,
        file_path,
        cleaning_scope=cleaning_scope,
    )  # [NEW]
    return model_perf_dict


def _print_loaded_metric_summary(label: str, metric_dicts: dict[str, OrderedDict]) -> None:
    first = next(iter(metric_dicts.values()), OrderedDict())
    print(f"[Step3][metric-check] {label}: models={list(first.keys())}")


def _format_rank_list(names: list[str], values: np.ndarray) -> str:
    order = np.argsort(values)
    return "[" + ", ".join(repr(str(names[i])) for i in order) + "]"


def _step2_all_results_path(args) -> str:
    stem = build_repr_forward_all_results_stem(args)
    return os.path.join(get_tsrouter_repr_forward_dir(args), stem + "_all_results.csv")


def _filter_step2_rows_to_current_profile(args, df: pd.DataFrame) -> pd.DataFrame:
    if get_auto_cl_mode(args) == "v0" or "dataset" not in df.columns:
        return df.copy()
    dataset_key = f"{build_repr_set_name(args)}_freqH"
    return df[df["dataset"].astype(str).eq(dataset_key)].copy()


def _runtime_by_model_for_summary(
    args,
    current_zoo_abbr_order_list,
    model_names: list[str],
) -> tuple[dict[str, float], list[str], str]:
    runtime_path = _step2_all_results_path(args)
    if not os.path.exists(runtime_path):
        return {}, list(model_names), runtime_path
    try:
        df = pd.read_csv(runtime_path)
    except Exception:
        return {}, list(model_names), runtime_path
    if df.empty or "model" not in df.columns:
        return {}, list(model_names), runtime_path

    raw = _filter_step2_rows_to_current_profile(args, df)
    raw["_file_order"] = np.arange(len(raw), dtype=np.int64)
    raw["_model_abbr"] = raw["model"].astype(str).map(lambda name: Model_abbrev_map.get(name, name))
    if current_zoo_abbr_order_list is not None:
        raw = raw[raw["_model_abbr"].isin(set(current_zoo_abbr_order_list))].copy()
    raw = raw[raw["_model_abbr"].isin(set(model_names))].copy()
    if raw.empty:
        return {}, list(model_names), runtime_path
    raw["_dataset_key"] = raw["dataset"].astype(str) if "dataset" in raw.columns else "__step2_profile__"
    raw["_summary_runtime_seconds"] = np.nan
    for col in STEP2_RUNTIME_WEIGHT_COLUMNS:
        if col not in raw.columns:
            continue
        vals = pd.to_numeric(raw[col], errors="coerce")
        fill = raw["_summary_runtime_seconds"].isna() & vals.notna() & np.isfinite(vals) & vals.ge(0)
        raw.loc[fill, "_summary_runtime_seconds"] = vals[fill].astype(float)
    usable = raw.dropna(subset=["_summary_runtime_seconds"]).copy()
    usable = usable.sort_values("_file_order").drop_duplicates(["_model_abbr", "_dataset_key"], keep="last")
    runtime_by_model = {
        str(name): float(pd.to_numeric(group["_summary_runtime_seconds"], errors="coerce").sum())
        for name, group in usable.groupby("_model_abbr", sort=False)
    }
    missing = [name for name in model_names if name not in runtime_by_model]
    return runtime_by_model, missing, runtime_path


def _strict_forward_runtime_by_model_from_step2(
    *,
    args,
    model_names: list[str],
) -> dict:
    """Read AutoXPCR resource costs from the exact Step2 forward-runtime field."""
    runtime_path = _step2_all_results_path(args)
    if not os.path.exists(runtime_path):
        raise FileNotFoundError(
            f"[AutoXPCR] missing Step2 runtime CSV: {runtime_path}"
        )
    try:
        df = pd.read_csv(runtime_path)
    except Exception as exc:
        raise RuntimeError(
            f"[AutoXPCR] cannot read Step2 runtime CSV {runtime_path}: {exc}"
        ) from exc
    if df.empty:
        raise ValueError(f"[AutoXPCR] Step2 runtime CSV is empty: {runtime_path}")
    required = {"model", "forward_runtime_seconds"}
    missing_columns = sorted(required.difference(df.columns))
    if missing_columns:
        raise ValueError(
            f"[AutoXPCR] Step2 runtime CSV missing required columns={missing_columns}: "
            f"{runtime_path}; no fallback runtime fields are allowed"
        )

    raw = _filter_step2_rows_to_current_profile(args, df)
    raw = raw.copy()
    raw["_file_order"] = np.arange(len(raw), dtype=np.int64)
    raw["_model_abbr"] = raw["model"].astype(str).map(
        lambda name: Model_abbrev_map.get(name, name)
    )
    expected = [str(name) for name in model_names]
    raw = raw[raw["_model_abbr"].isin(set(expected))].copy()
    raw["_dataset_key"] = (
        raw["dataset"].astype(str)
        if "dataset" in raw.columns
        else "__step2_profile__"
    )
    raw["_forward_runtime_seconds"] = pd.to_numeric(
        raw["forward_runtime_seconds"], errors="coerce"
    )
    raw = raw.sort_values("_file_order").drop_duplicates(
        ["_model_abbr", "_dataset_key"], keep="last"
    )
    usable = raw[
        raw["_forward_runtime_seconds"].notna()
        & np.isfinite(raw["_forward_runtime_seconds"])
        & raw["_forward_runtime_seconds"].ge(0)
    ].copy()
    runtime_by_model = {
        str(name): float(group["_forward_runtime_seconds"].sum())
        for name, group in usable.groupby("_model_abbr", sort=False)
    }
    missing_models = [name for name in expected if name not in runtime_by_model]
    if missing_models:
        raise ValueError(
            f"[AutoXPCR] missing finite forward_runtime_seconds for models={missing_models} "
            f"in {runtime_path}; no insert/runtime/non-eval fallback is allowed"
        )
    ordered_runtime = {name: runtime_by_model[name] for name in expected}
    return {
        "runtime_path": runtime_path,
        "runtime_column": "forward_runtime_seconds",
        "runtime_by_model_seconds": ordered_runtime,
        "rows_used": int(len(usable)),
        "datasets_by_model": {
            str(name): int(group["_dataset_key"].nunique())
            for name, group in usable.groupby("_model_abbr", sort=False)
        },
    }


def _build_step3_step2_coverage(
    *,
    expected_model_order: list[str],
    metric_dicts: dict[str, OrderedDict],
    repr_forward_path: str,
    runtime_by_model: dict[str, float],
    runtime_missing_models: list[str],
    runtime_path: str,
    load_errors: dict[str, str] | None = None,
) -> dict:
    expected = [str(x) for x in expected_model_order if str(x)]
    load_errors = load_errors or {}
    missing_by_metric: dict[str, list[str]] = {}
    metric_available_sets: list[set[str]] = []
    for metric_name, metric_dict in metric_dicts.items():
        available = {str(x) for x in metric_dict.keys()}
        metric_available_sets.append(available)
        missing = _ordered_missing(expected, available)
        if missing:
            missing_by_metric[str(metric_name)] = missing

    if metric_available_sets:
        metric_covered_models = [name for name in expected if all(name in available for available in metric_available_sets)]
    else:
        metric_covered_models = []
    missing_metric_models = _ordered_missing(expected, metric_covered_models)
    missing_runtime_models = _ordered_missing(expected, runtime_by_model.keys())
    if runtime_missing_models:
        missing_runtime_models = list(dict.fromkeys(missing_runtime_models + [str(x) for x in runtime_missing_models]))

    metric_complete = bool(expected) and not missing_metric_models and not load_errors
    runtime_complete = bool(expected) and not missing_runtime_models
    status = "ok" if metric_complete and runtime_complete else "incomplete"
    if load_errors:
        status = "load_error"

    note_parts = []
    if load_errors:
        note_parts.append(
            "load_errors=" + ";".join(f"{name}:{msg}" for name, msg in sorted(load_errors.items()))
        )
    if missing_metric_models:
        note_parts.append("missing_metric_models=" + " ".join(missing_metric_models))
    if missing_runtime_models:
        note_parts.append("missing_runtime_models=" + " ".join(missing_runtime_models))
    if not note_parts:
        note_parts.append("complete")

    return {
        "status": status,
        "metric_complete": metric_complete,
        "runtime_complete": runtime_complete,
        "expected_model_order": expected,
        "expected_model_count": int(len(expected)),
        "metric_covered_models": metric_covered_models,
        "metric_covered_model_count": int(len(metric_covered_models)),
        "runtime_covered_models": [name for name in expected if name in runtime_by_model],
        "runtime_covered_model_count": int(len([name for name in expected if name in runtime_by_model])),
        "missing_metric_models": missing_metric_models,
        "missing_runtime_models": missing_runtime_models,
        "missing_by_metric": missing_by_metric,
        "repr_forward_path": repr_forward_path,
        "runtime_path": runtime_path,
        "load_errors": dict(load_errors),
        "note": " | ".join(note_parts),
    }


def _load_step3_cluster_metric_dicts(
    args,
    current_zoo_abbr_order_list,
    repr_forward_csv_path: str,
) -> tuple[dict[str, OrderedDict], dict[str, str], dict]:
    metric_dicts: dict[str, OrderedDict] = {}
    load_errors: dict[str, str] = {}
    cleaning_scope = "per_model" if str(getattr(args, "repr_v", ""))[:1] == "7" else "global"
    for metric_name in ("MASE", "sMAPE", "CRPS"):
        try:
            print(f"[Step3][labels] loading metric={metric_name}", flush=True)
            metric_dicts[metric_name] = load_metrics_matrix(
                repr_forward_csv_path,
                current_zoo_abbr_order_list,
                metric=metric_name,
                log_summary=False,
                cleaning_scope=cleaning_scope,
            )
            print(
                f"[Step3][labels] loaded metric={metric_name}, "
                f"models={len(metric_dicts[metric_name])}",
                flush=True,
            )
        except Exception as exc:
            metric_dicts[metric_name] = OrderedDict()
            load_errors[metric_name] = f"{type(exc).__name__}:{exc}"

    expected_model_order = [str(x) for x in current_zoo_abbr_order_list if str(x)]
    runtime_by_model, runtime_missing_models, runtime_path = _runtime_by_model_for_summary(
        args=args,
        current_zoo_abbr_order_list=current_zoo_abbr_order_list,
        model_names=expected_model_order,
    )
    coverage = _build_step3_step2_coverage(
        expected_model_order=expected_model_order,
        metric_dicts=metric_dicts,
        repr_forward_path=repr_forward_csv_path,
        runtime_by_model=runtime_by_model,
        runtime_missing_models=runtime_missing_models,
        runtime_path=runtime_path,
        load_errors=load_errors,
    )
    return metric_dicts, load_errors, coverage


def _print_step3_step2_coverage(coverage: dict, *, will_skip: bool) -> None:
    if coverage.get("status") == "ok":
        print(
            f"[Step3][coverage] Step2 profile complete: "
            f"models={coverage.get('expected_model_count')} "
            f"per_sample={coverage.get('repr_forward_path')} "
            f"runtime={coverage.get('runtime_path')}"
        )
        return

    action = "skip current Step3 profile" if will_skip else "continue with incomplete runtime evidence"
    print(
        f"⚠️ [Step3][coverage] Step2 profile incomplete; {action}. "
        f"expected={coverage.get('expected_model_count')}, "
        f"metric_covered={coverage.get('metric_covered_model_count')}, "
        f"runtime_covered={coverage.get('runtime_covered_model_count')}, "
        f"missing_metric={coverage.get('missing_metric_models')}, "
        f"missing_runtime={coverage.get('missing_runtime_models')}, "
        f"per_sample={coverage.get('repr_forward_path')}, "
        f"runtime={coverage.get('runtime_path')}, "
        f"note={coverage.get('note')}"
    )


def _latest_model_full_name(model_abbr: str) -> str:
    abbr_to_full = {abbr: full_name for full_name, abbr in Model_abbrev_map.items()}
    return str(abbr_to_full.get(model_abbr, model_abbr))


def _read_latest_model_step2_insert_runtime(args, latest_model_abbr: str) -> tuple[float, str, str]:
    runtime_path = _step2_all_results_path(args)
    if not os.path.exists(runtime_path):
        return float("nan"), runtime_path, "missing_runtime_csv"
    try:
        df = pd.read_csv(runtime_path)
    except Exception as exc:
        return float("nan"), runtime_path, f"read_error:{exc}"
    if df.empty:
        return float("nan"), runtime_path, "empty_runtime_csv"
    if "model" not in df.columns:
        return float("nan"), runtime_path, "missing_model_column"
    if "insert_runtime_seconds" not in df.columns:
        return float("nan"), runtime_path, "missing_insert_runtime_seconds_column"

    raw = _filter_step2_rows_to_current_profile(args, df)
    raw["_file_order"] = np.arange(len(raw), dtype=np.int64)
    raw["_model_abbr"] = raw["model"].astype(str).map(lambda name: Model_abbrev_map.get(name, name))
    raw = raw[raw["_model_abbr"].astype(str).eq(str(latest_model_abbr))].copy()
    if raw.empty:
        return float("nan"), runtime_path, f"missing_latest_model:{latest_model_abbr}"
    raw["_dataset_key"] = raw["dataset"].astype(str) if "dataset" in raw.columns else "__step2_profile__"
    raw = raw.sort_values("_file_order").drop_duplicates(["_model_abbr", "_dataset_key"], keep="last")
    vals = pd.to_numeric(raw["insert_runtime_seconds"], errors="coerce")
    if vals.empty or vals.isna().any() or not np.isfinite(vals.to_numpy(dtype=float)).all():
        return float("nan"), runtime_path, f"nan_insert_runtime_seconds:{latest_model_abbr}"
    return float(vals.sum()), runtime_path, "ok"


def _read_latest_model_advanced_baseline_profile_runtime(
    args,
    latest_model_abbr: str,
) -> tuple[float, str, str]:
    """Read the profile-forward cost required by the advanced baseline scope."""
    if get_advanced_baseline_train_scope(args) != "full_pool":
        return _read_latest_model_step2_insert_runtime(args, latest_model_abbr)

    runtime_path = os.path.join(
        get_tsrouter_repr_forward_dir(args),
        build_repr_eval_pool_forward_stem(args) + "_all_results.csv",
    )
    if not os.path.exists(runtime_path):
        return float("nan"), runtime_path, "missing_full_pool_runtime_csv"
    try:
        df = pd.read_csv(runtime_path)
    except Exception as exc:
        return float("nan"), runtime_path, f"full_pool_read_error:{exc}"
    if df.empty:
        return float("nan"), runtime_path, "empty_full_pool_runtime_csv"
    if "model" not in df.columns:
        return float("nan"), runtime_path, "missing_full_pool_model_column"
    if "forward_runtime_seconds" not in df.columns:
        return float("nan"), runtime_path, "missing_full_pool_forward_runtime_seconds_column"

    raw = df.copy()
    raw["_file_order"] = np.arange(len(raw), dtype=np.int64)
    raw["_model_abbr"] = raw["model"].astype(str).map(
        lambda name: Model_abbrev_map.get(name, name)
    )
    raw = raw[raw["_model_abbr"].astype(str).eq(str(latest_model_abbr))].copy()
    if raw.empty:
        return float("nan"), runtime_path, f"missing_full_pool_latest_model:{latest_model_abbr}"
    raw["_dataset_key"] = (
        raw["dataset"].astype(str)
        if "dataset" in raw.columns
        else "__full_pool_profile__"
    )
    raw = raw.sort_values("_file_order").drop_duplicates(
        ["_model_abbr", "_dataset_key"], keep="last"
    )
    vals = pd.to_numeric(raw["forward_runtime_seconds"], errors="coerce")
    if (
        vals.empty
        or vals.isna().any()
        or not np.isfinite(vals.to_numpy(dtype=float)).all()
        or vals.lt(0).any()
    ):
        return (
            float("nan"),
            runtime_path,
            f"invalid_full_pool_forward_runtime_seconds:{latest_model_abbr}",
        )
    return float(vals.sum()), runtime_path, "ok_full_pool_forward"


def _write_step3_insert_timing(
    args,
    repr_set_name: str,
    model_repr_path: str,
    model_names: list[str],
    step3_skip_reuse_seconds: float,
    step3_cluster_seconds: float,
    step3_pool_seconds: float,
    status: str,
    step2_coverage: dict | None = None,
) -> None:
    latest_model_abbr = str(model_names[-1]) if model_names else ""
    latest_model_full = _latest_model_full_name(latest_model_abbr) if latest_model_abbr else ""
    insert_runtime_seconds, runtime_path, runtime_status = _read_latest_model_step2_insert_runtime(
        args, latest_model_abbr
    )
    if status == "built" and step2_coverage and not bool(step2_coverage.get("runtime_complete", True)):
        status = "built_step2_runtime_incomplete"
    measured_index_refresh_seconds = float(step3_skip_reuse_seconds) + float(step3_cluster_seconds)
    if str(status).startswith("skipped"):
        index_refresh_seconds = float("nan")
        insert_total_seconds = float("nan")
    elif np.isfinite(insert_runtime_seconds):
        index_refresh_seconds = measured_index_refresh_seconds
        insert_total_seconds = index_refresh_seconds + float(insert_runtime_seconds)
    else:
        index_refresh_seconds = float("nan")
        insert_total_seconds = float("nan")

    row = {
        "row_key": _step3_timing_row_key(model_repr_path),
        "status": status,
        "stage": int(getattr(args, "current_zoo_num", len(model_names)) or len(model_names)),
        "zoo_total_num": int(getattr(args, "zoo_total_num", len(model_names)) or len(model_names)),
        "latest_model_abbr": latest_model_abbr,
        "latest_model_full_name": latest_model_full,
        "model_abbr_order": " ".join(str(x) for x in model_names),
        "model_count": int(len(model_names)),
        "repr_set_name": repr_set_name,
        "repr_forward_stem": build_repr_forward_stem(args),
        "model_repr_name": Path(model_repr_path).stem,
        **_auto_cl_sidecar_fields(args),
        **_step3_coverage_csv_fields(step2_coverage),
        "repr_v": getattr(args, "repr_v", ""),
        "base_metrics": getattr(args, "base_metrics", ""),
        "model_repr_mode": getattr(args, "model_repr_mode", ""),
        "subset_top_k": getattr(args, "subset_top_k", ""),
        "subset_perf_scale": getattr(args, "subset_perf_scale", ""),
        "route_efficiency_mode": getattr(args, "route_efficiency_mode", ""),
        "repr_weight_ratio": getattr(args, "repr_weight_ratio", ""),
        "repr_distance_metric": getattr(args, "repr_distance_metric", ""),
        "repr_v5_nearest_k": getattr(args, "repr_v5_nearest_k", ""),
        "rank_decay_coef": rank_decay_coef(args),
        "repr_v5_distance_power": getattr(args, "repr_v5_distance_power", ""),
        "model_repr_agg": getattr(args, "model_repr_agg", ""),
        "step3_skip_reuse_seconds": float(step3_skip_reuse_seconds),
        "step3_cluster_seconds": float(step3_cluster_seconds),
        "step3_pool_seconds": float(step3_pool_seconds),
        "measured_index_refresh_seconds": measured_index_refresh_seconds,
        "insert_runtime_seconds": insert_runtime_seconds,
        "index_refresh_seconds": index_refresh_seconds,
        "insert_total_seconds": insert_total_seconds,
        "step2_runtime_status": runtime_status,
        "step2_runtime_path": runtime_path,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _upsert_step3_timing_row(row)
    print(
        f"[Step3][timing] saved -> {_step3_timing_csv_path()} "
        f"(index_refresh_seconds={index_refresh_seconds}, insert_total_seconds={insert_total_seconds})"
    )
    if not np.isfinite(insert_runtime_seconds):
        print(
            "⚠️ [Step3][timing] latest model insert_runtime_seconds is NaN/missing; "
            "index_refresh_seconds and insert_total_seconds are recorded as NaN. "
            "Please backfill Step2 insert_runtime_seconds."
        )


def _print_cluster_forward_summary(
    args,
    current_zoo_abbr_order_list,
    model_names: list[str],
    metric_matrix_by_name: dict[str, np.ndarray],
) -> None:
    for metric_name, matrix in metric_matrix_by_name.items():
        values = np.nanmean(np.asarray(matrix, dtype=np.float64), axis=1)
        print(
            f"[Step3][cluster_forward] metric_rank {metric_name} "
            f"(per-sample mean, lower better): {_format_rank_list(model_names, values)}"
        )
    runtime_by_model, missing, runtime_path = _runtime_by_model_for_summary(
        args=args,
        current_zoo_abbr_order_list=current_zoo_abbr_order_list,
        model_names=model_names,
    )
    if runtime_by_model:
        names = [name for name in model_names if name in runtime_by_model]
        values = np.asarray([runtime_by_model[name] for name in names], dtype=np.float64)
        missing_text = f" ⚠️ missing_models={missing}" if missing else ""
        print(
            f"[Step3][cluster_forward] time_efficiency "
            f"(Step2 insert runtime, lower faster): {_format_rank_list(names, values)}{missing_text}"
        )
    else:
        print(f"[Step3][cluster_forward] time_efficiency unavailable: {runtime_path} ⚠️ missing_models={missing}")


def _runtime_weights_from_step2(
    args,
    current_zoo_abbr_order_list,
    model_names: list[str],
    use_perf_scale: bool,
    perf_scale: float,
) -> tuple[dict[str, float], dict]:
    """Build route-fast weights from saved Step2 profile-forward runtime."""
    runtime_path = _step2_all_results_path(args)
    print(f"[route-fast] Step2 runtime source path: {runtime_path}")
    if not os.path.exists(runtime_path):
        raise FileNotFoundError(f"[route-fast] missing Step2 runtime CSV: {runtime_path}")

    try:
        df = pd.read_csv(runtime_path)
    except Exception as exc:
        raise RuntimeError(f"[route-fast] cannot read Step2 runtime CSV: {runtime_path}: {exc}") from exc
    if df.empty:
        raise ValueError(f"[route-fast] Step2 runtime CSV is empty: {runtime_path}")
    if "model" not in df.columns:
        raise ValueError(f"[route-fast] Step2 runtime CSV has no model column: {runtime_path}")
    available_runtime_cols = [col for col in STEP2_RUNTIME_WEIGHT_COLUMNS if col in df.columns]
    if not available_runtime_cols:
        raise ValueError(
            f"[route-fast] Step2 runtime CSV has no runtime columns {STEP2_RUNTIME_WEIGHT_COLUMNS}: "
            f"{runtime_path}; columns={list(df.columns)}"
        )

    raw = _filter_step2_rows_to_current_profile(args, df)
    raw["_file_order"] = np.arange(len(raw), dtype=np.int64)
    raw["_model_abbr"] = raw["model"].astype(str).map(lambda name: Model_abbrev_map.get(name, name))
    if current_zoo_abbr_order_list is not None:
        raw = raw[raw["_model_abbr"].isin(set(current_zoo_abbr_order_list))].copy()
    raw = raw[raw["_model_abbr"].isin(set(model_names))].copy()
    if "dataset" in raw.columns:
        raw["_dataset_key"] = raw["dataset"].astype(str)
    else:
        raw["_dataset_key"] = "__step2_profile__"

    raw["_step2_runtime_seconds"] = np.nan
    raw["_step2_runtime_column"] = ""
    for col in STEP2_RUNTIME_WEIGHT_COLUMNS:
        if col not in raw.columns:
            continue
        vals = pd.to_numeric(raw[col], errors="coerce")
        fill = raw["_step2_runtime_seconds"].isna() & vals.notna() & np.isfinite(vals) & vals.ge(0)
        raw.loc[fill, "_step2_runtime_seconds"] = vals[fill].astype(float)
        raw.loc[fill, "_step2_runtime_column"] = col

    usable = raw.dropna(subset=["_step2_runtime_seconds"]).copy()
    usable = usable[np.isfinite(pd.to_numeric(usable["_step2_runtime_seconds"], errors="coerce"))].copy()
    if usable.empty:
        raise ValueError(
            f"[route-fast] no finite Step2 runtime values in {runtime_path}; "
            f"runtime_columns={available_runtime_cols}"
        )
    usable = usable.sort_values("_file_order").drop_duplicates(["_model_abbr", "_dataset_key"], keep="last")

    runtime_by_model: dict[str, float] = {}
    detail_by_model: dict[str, dict] = {}
    for name, group in usable.groupby("_model_abbr", sort=False):
        runtime_seconds = pd.to_numeric(group["_step2_runtime_seconds"], errors="coerce")
        runtime_seconds = runtime_seconds.replace([np.inf, -np.inf], np.nan).dropna()
        if runtime_seconds.empty:
            continue
        runtime_by_model[str(name)] = float(runtime_seconds.sum())
        detail_by_model[str(name)] = {
            "runtime_seconds": float(runtime_seconds.sum()),
            "datasets": int(group["_dataset_key"].nunique()),
            "rows": int(len(group)),
            "columns": ",".join(sorted(set(group["_step2_runtime_column"].dropna().astype(str)))),
        }

    missing = [name for name in model_names if name not in runtime_by_model]
    if missing:
        preview = usable[["_model_abbr", "_dataset_key", "_step2_runtime_seconds", "_step2_runtime_column"]].head(20)
        fallback_runtime = float(max(runtime_by_model.values())) if runtime_by_model else 0.0
        print(
            f"⚠️ [route-fast] Step2 runtime missing models={missing} in {runtime_path}; "
            f"use slowest observed runtime fallback={fallback_runtime:.6f}s. "
            f"available_models={sorted(runtime_by_model)}\n{preview.to_string(index=False)}"
        )
        for name in missing:
            runtime_by_model[name] = fallback_runtime
            detail_by_model[name] = {
                "runtime_seconds": fallback_runtime,
                "datasets": 0,
                "rows": 0,
                "columns": "missing_fallback_slowest_observed",
                "missing_runtime": True,
            }

    runtime_arr = np.asarray([runtime_by_model[name] for name in model_names], dtype=np.float64)
    if not np.isfinite(runtime_arr).all():
        raise ValueError(f"[route-fast] non-finite Step2 runtime by model: {runtime_by_model}")
    r_min = float(runtime_arr.min())
    r_max = float(runtime_arr.max())
    base_weights = _strict_lower_better_rank_weights(
        runtime_arr,
        model_names=list(model_names),
        use_perf_scale=use_perf_scale,
        perf_scale=perf_scale,
    )
    weights = {name: float(weight) for name, weight in zip(model_names, base_weights)}

    print(
        f"[route-fast] Step2 runtime rows used={len(usable)}, models={len(model_names)}, "
        f"scale={'strict_rank_plus_perf_scale' if use_perf_scale else 'strict_rank_0_1'}, "
        f"runtime_min={r_min:.6f}s, runtime_max={r_max:.6f}s"
    )
    _print_step3_weight_prior(
        source="efficiency",
        metric="Step2_runtime_seconds",
        model_names=list(model_names),
        score_values=runtime_arr,
        weights=base_weights,
    )
    # for name in model_names:
    #     detail = detail_by_model[name]
    #     print(
    #         f"[route-fast] model={name} runtime_s={detail['runtime_seconds']:.6f} "
    #         f"weight={weights[name]:.6f} columns={detail['columns']} "
    #         f"datasets={detail['datasets']} rows={detail['rows']}"
    #     )

    return weights, {
        "source": "step2_runtime",
        "runtime_path": runtime_path,
        "runtime_columns_priority": list(STEP2_RUNTIME_WEIGHT_COLUMNS),
        "runtime_by_model_seconds": dict(runtime_by_model),
        "runtime_min_seconds": r_min,
        "runtime_max_seconds": r_max,
        "detail_by_model": detail_by_model,
        "weights": dict(weights),
        "weight_algorithm": STRICT_RANK_WEIGHT_ALGORITHM,
        "weight_tie_breaker": STRICT_RANK_WEIGHT_TIE_BREAKER,
        "use_perf_scale": bool(use_perf_scale),
        "perf_scale": float(perf_scale),
        "missing_models": list(missing),
        "missing_runtime_fallback_seconds": float(max(runtime_by_model.values())) if missing else None,
    }


def _source_tokens(args) -> set[str]:
    raw = getattr(args, "zoo_repr_set", [])
    raw_values = [raw] if isinstance(raw, str) else list(raw)
    tokens: set[str] = set()
    for value in raw_values:
        value = str(value).lower()
        tokens.add(value)
        for part in value.replace("_", "-").split("-"):
            part = part.strip()
            if part:
                tokens.add(part)
    return tokens


def _is_oracle_sample_source(args) -> bool:
    return bool(_source_tokens(args).intersection({"os", "oracle_sample", "sample_oracle", "oraclesample", "sampleoracle"}))


def _normalize_task_cache_key(dataset_key: str) -> str:
    s = str(dataset_key)
    if "/" in s:
        return s
    parts = s.rsplit("_", 2)
    if len(parts) == 3:
        return f"{parts[0]}/{parts[1]}/{parts[2]}"
    return s

def calculate_combined_scores(model_perf_dict):
    'TSRouter runtime message.'

    'TSRouter runtime message.'
    model_names = list(model_perf_dict.keys())
    mse_matrix = np.stack(list(model_perf_dict.values()))  # shape: (n_models, n_data)

    perf_advantage = np.zeros_like(mse_matrix)

    for model_idx, name in enumerate(model_names):
                      
        other_avg = np.mean(np.delete(mse_matrix, model_idx, axis=0), axis=0)
                                    
        perf_advantage[model_idx] = other_avg - mse_matrix[model_idx]

    specificity = np.std(mse_matrix, axis=0)                   

                    
    # standard_specificity = (specificity - specificity.mean()) / (specificity.std() + 1e-8)
    norm_specificity = (specificity - specificity.min()) / (specificity.max() - specificity.min() + 1e-8)

              
    # decile_results = analyze_performance_by_specificity(mse_matrix, specificity, model_names, n_deciles=10, plot_path=fig_save_path,title=plot_Model_repr_name)

                                                                                      
    selection_scores_matrix = perf_advantage * norm_specificity *100        
                                                                         

                                
    score_dict = OrderedDict()
    perf_adv_dict = OrderedDict()                                
    metrics_dict = OrderedDict()                          

    for model_idx, name in enumerate(model_names):
            score_dict[name] = selection_scores_matrix[model_idx]
            perf_adv_dict[name] = float(np.mean(perf_advantage[model_idx], axis=0))                    
            metrics_dict[name] = float(np.mean(mse_matrix[model_idx], axis=0))                                              
    print(f"[repr_score] computed selection scores for {len(model_names)} models, samples={mse_matrix.shape[1]}")

    return score_dict, perf_adv_dict, metrics_dict



def cluster_analysis(repr_data, zoo_dataset, max_clusters, fig_save_path='results_new/figs/', silhouette_threshold=0.1,
                     Repr_v=5):
    'TSRouter runtime message.'
           
    silhouette_scores = []
    possible_ks = range(2, max_clusters + 1)
    kmeans_models = {}

              
    for k in possible_ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(repr_data)
        score = silhouette_score(repr_data, labels)
        silhouette_scores.append(score)
        kmeans_models[k] = (kmeans, labels)           

            
                        
    print(f"{zoo_dataset}TSRouter runtime message: ", end=' ')
    for k, score in zip(possible_ks, silhouette_scores):
        print(f" ({k},{score:.3f})", end=' ')

           
    best_k = possible_ks[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)

              
    if best_score < silhouette_threshold and Repr_v != 3:  #
        print(f"⚠️ {zoo_dataset}TSRouter runtime message: {best_score:.3f}TSRouter runtime message: {silhouette_threshold}TSRouter runtime message: ")
        return 1, np.zeros(len(repr_data), dtype=int)
    else:
        print(f"{zoo_dataset}TSRouter runtime message: {best_k}TSRouter runtime message: {best_score:.3f})")
        return best_k, kmeans_models[best_k][1]


def get_model_zoo_repr(args, current_zoo_abbr_order_list):
    'TSRouter runtime message.'

                                        
    if _is_oracle_sample_source(args):
        raise NotImplementedError(
            "os/oracle_sample needs task-sample future metadata to recompute per-window forward ranks. "
            "The existing Step4 task cache stores contexts only, so os cannot be built losslessly from current artifacts."
        )
    final_repr_dict = OrderedDict()

                                    
    Dec_mode=False

    def decode_repr_version(args):
        'TSRouter runtime message.'
        use_wta = False
        use_repr_order = False
        use_perf_scale = False
        Use_rand_subset= False
        repr_v_head = str(args.repr_v)[0]
        if repr_v_head == '2':
            use_wta = True
            use_perf_scale = True
        if repr_v_head == '3':
            use_wta = True
            use_repr_order = True
        if repr_v_head == '4':
            use_wta = True
            use_repr_order = True
            use_perf_scale = True
        if repr_v_head == '0':
            Use_rand_subset = True
            print('TSRouter runtime message.')
        use_continuous_rank = False
        if repr_v_head == '5':
                                               
                                         
            use_continuous_rank = True
            use_wta = False
            use_repr_order = True
            use_perf_scale = True
            # print("🆕 [repr_v5] continuous rank mode enabled + repr_order/perf_scale enabled.")
        return use_wta, use_repr_order, use_perf_scale, Use_rand_subset, use_continuous_rank

    Use_wta, Use_repr_order, Use_perf_scale, Use_rand_subset, Use_continuous_rank = decode_repr_version(args)


                                                         
    repr_set_name, weight_path, model_repr_path,_ = get_repr_save_path(args)
    if str(getattr(args, "repr_v", "") or "")[:1] == "6":
        from selector.TSRouter_Select.simplets_select import build_simplets_step3

        build_simplets_step3(
            args,
            list(current_zoo_abbr_order_list),
            repr_set_name=repr_set_name,
            weight_path=weight_path,
            model_repr_path=model_repr_path,
            load_metric_dicts=_load_step3_cluster_metric_dicts,
            read_latest_model_runtime=_read_latest_model_advanced_baseline_profile_runtime,
        )
        return
    if str(getattr(args, "repr_v", "") or "")[:1] == "7":
        from selector.TSRouter_Select.autoforecast_select import build_autoforecast_step3

        build_autoforecast_step3(
            args,
            list(current_zoo_abbr_order_list),
            repr_set_name=repr_set_name,
            weight_path=weight_path,
            model_repr_path=model_repr_path,
            load_metric_dicts=_load_step3_cluster_metric_dicts,
            read_latest_model_runtime=_read_latest_model_advanced_baseline_profile_runtime,
            load_model_forward_runtime=_strict_forward_runtime_by_model_from_step2,
        )
        return
    skip_saved = bool(getattr(args, "skip_saved", False))
    step3_skip_reuse_start = time.perf_counter()
    step3_skip_or_reuse_hit = False
    if skip_saved:
        step3_skip_or_reuse_hit = _skip_or_reuse_step3_if_ready(
            args=args,
            repr_set_name=repr_set_name,
            weight_path=weight_path,
            model_repr_path=model_repr_path,
            expected_model_order=list(current_zoo_abbr_order_list),
        )
    else:
        print("[Step3][skip-save] disabled; rebuild cluster and pool artifacts.")
    step3_skip_reuse_seconds = time.perf_counter() - step3_skip_reuse_start
    if step3_skip_or_reuse_hit:
        print("[Step3][timing] skip-save hit, timing.csv row unchanged.")
        return
    phase_paths = _step3_artifact_paths(args, weight_path, model_repr_path)
    if skip_saved:
        phase_status = _step3_core_status(
            args=args,
            paths=phase_paths,
            expected_model_order=list(current_zoo_abbr_order_list),
            require_manifest=True,
        )
        cluster_phase_skip = bool(phase_status.get("cluster_complete")) and not bool(phase_status.get("timing_is_skip_or_reuse"))
        pool_phase_skip = bool(phase_status.get("pool_phase_done")) and bool(getattr(args, "enable_process_metrics", True))
        existing_subset_payload = _read_pickle_file(phase_paths["subset"]) if pool_phase_skip else None
        print(
            f"[Step3][cluster][skip-save] cluster_complete={phase_status.get('cluster_complete')}, "
            f"timing_status={phase_status.get('timing_status', {}).get('status') or 'NA'}, "
            f"skip={cluster_phase_skip}"
        )
        print(
            f"[Step3][pool][skip-save] pool_phase_done={phase_status.get('pool_phase_done')}, "
            f"skip={pool_phase_skip}"
        )
    else:
        cluster_phase_skip = False
        pool_phase_skip = False
        existing_subset_payload = None
    step3_cluster_start = time.perf_counter()

    # ======================
                           
    # ======================
    repr_forward_stem = build_repr_forward_stem(args)
    repr_forward_csv_path = os.path.join(get_tsrouter_repr_forward_dir(args), repr_forward_stem + "_per_sample_results.csv")
    metric_perf_dict_by_name, _metric_load_errors, step2_coverage = _load_step3_cluster_metric_dicts(
        args=args,
        current_zoo_abbr_order_list=current_zoo_abbr_order_list,
        repr_forward_csv_path=repr_forward_csv_path,
    )
    metric_complete = bool(step2_coverage.get("metric_complete", False))
    _print_step3_step2_coverage(step2_coverage, will_skip=not metric_complete)
    if not metric_complete:
        step3_cluster_seconds = time.perf_counter() - step3_cluster_start
        _write_step3_insert_timing(
            args=args,
            repr_set_name=repr_set_name,
            model_repr_path=model_repr_path,
            model_names=list(step2_coverage.get("expected_model_order", []) or current_zoo_abbr_order_list),
            step3_skip_reuse_seconds=step3_skip_reuse_seconds,
            step3_cluster_seconds=step3_cluster_seconds,
            step3_pool_seconds=0.0,
            status="skipped_step2_incomplete",
            step2_coverage=step2_coverage,
        )
        print(
            f"⚠️ [Step3][skip] current profile Step2 per-sample coverage is incomplete; "
            f"skip building Step3 artifacts for {get_auto_cl_profile_name(args)}."
        )
        print('-' * 100)
        return

    model_MASE_perf_dict = metric_perf_dict_by_name["MASE"]
    model_sMAPE_perf_dict = metric_perf_dict_by_name["sMAPE"]
    model_CRPS_perf_dict = metric_perf_dict_by_name["CRPS"]
    _print_loaded_metric_summary(
        "cluster_forward per-sample metrics",
        metric_perf_dict_by_name,
    )

                                 
    model_perf_dict = {'C': model_CRPS_perf_dict, 'M': model_MASE_perf_dict, 'S': model_sMAPE_perf_dict}.get(args.base_metrics, model_MASE_perf_dict)

                                    
    abbr_to_full = {abbr: full_name for full_name, abbr in Model_abbrev_map.items()}

                                                       
    full_order_idx = {full_name: idx for idx, full_name in enumerate(All_sorted_model_names)}

    raw_model_names = list(model_perf_dict.keys())                                       

    def _sort_key(abbr: str) -> int:
        'TSRouter runtime message.'
        full_name = abbr_to_full.get(abbr)
        if full_name is None:
            return 10 ** 9
        return full_order_idx.get(full_name, 10 ** 8)

                   
    model_names = sorted(raw_model_names, key=_sort_key)
    ordered_model_perf_dict = OrderedDict((name, model_perf_dict[name]) for name in model_names)
    metric_perf_dict_by_name = {
        "MASE": model_MASE_perf_dict,
        "sMAPE": model_sMAPE_perf_dict,
        "CRPS": model_CRPS_perf_dict,
    }
    metric_matrix_by_name = {
        metric_name: np.stack([metric_dict[name] for name in model_names])
        for metric_name, metric_dict in metric_perf_dict_by_name.items()
    }
    _print_cluster_forward_summary(
        args=args,
        current_zoo_abbr_order_list=current_zoo_abbr_order_list,
        model_names=list(model_names),
        metric_matrix_by_name=metric_matrix_by_name,
    )

                                                
    metrics_matrix = np.stack(list(ordered_model_perf_dict.values()))
    print('TSRouter runtime message.', metrics_matrix.shape)  # (n_model,num_sample)
                                            

    route_fast_weight_dict = None
    route_fast_weight_info = None
    perf_scale = float(getattr(args, "subset_perf_scale", 1.0))
    if _route_efficiency_mode_enabled(args):
        route_fast_weight_dict, route_fast_weight_info = _runtime_weights_from_step2(
            args=args,
            current_zoo_abbr_order_list=current_zoo_abbr_order_list,
            model_names=list(model_names),
            use_perf_scale=bool(Use_perf_scale),
            perf_scale=perf_scale,
        )

    # ======================
                                    
    # ======================
    if Use_continuous_rank:
        subset_assign_path = model_repr_path.replace('.pkl', f'{encoder_enrichment_suffix(args)}_subset_assign.pkl')
        if cluster_phase_skip:
            v5_payload = _read_pickle_file(model_repr_path)
            if not isinstance(v5_payload, dict) or v5_payload.get("__repr_format__") != "v5_rank_centers":
                raise ValueError(f"[Step3][cluster] cannot skip cluster; invalid v5 payload: {model_repr_path}")
            center_repr = np.asarray(v5_payload["center_repr"], dtype=np.float32)
            center_rank = np.asarray(v5_payload["center_rank"], dtype=np.int64)
            center_distance_weights = np.asarray(
                v5_payload.get("center_distance_weights", np.ones(center_repr.shape[0])),
                dtype=np.float32,
            ).reshape(-1)
            subset_assign_payload = _read_pickle_file(subset_assign_path)
            if not isinstance(subset_assign_payload, dict):
                subset_assign_payload = {}
            raw_top1 = subset_assign_payload.get("raw_center_top1_indices_dict")
            if isinstance(raw_top1, dict):
                v5_raw_top1_indices_dict = OrderedDict(
                    (name, np.asarray(raw_top1.get(name, []), dtype=np.int64)) for name in model_names
                )
            else:
                v5_raw_top1_indices_dict = OrderedDict(
                    (name, np.where(center_rank[:, 0] == i)[0].astype(np.int64))
                    for i, name in enumerate(model_names)
                )
            step3_cluster_seconds = 0.0
            print(f"[Step3][cluster] skipped: existing v5 rank-center artifacts are complete -> {model_repr_path}")
        else:
            print("[Step3][cluster] begin: build v5 rank-center model representation")
                                              
                                                     
            if route_fast_weight_dict is not None:
                metric_weights = np.asarray([route_fast_weight_dict[name] for name in model_names], dtype=np.float32)
                print("[repr_v5] route-fast runtime prior weights:", {k: round(float(v), 4) for k, v in zip(model_names, metric_weights)})
            else:
                metric_array = np.array([ordered_model_perf_dict[name].mean() for name in model_names], dtype=float)
                metric_weights = _strict_lower_better_rank_weights(
                    metric_array,
                    model_names=list(model_names),
                    use_perf_scale=Use_perf_scale,
                    perf_scale=float(getattr(args, "subset_perf_scale", 1.0)),
                )
                print("[repr_v5] strict-rank metric prior weights:", {k: round(float(v), 4) for k, v in zip(model_names, metric_weights)})
                _print_step3_weight_prior(
                    source="performance",
                    metric=f"Step2_forward_{_base_metric_display_name(args)}",
                    model_names=list(model_names),
                    score_values=metric_array,
                    weights=metric_weights,
                )
            metric_weights = np.asarray(metric_weights, dtype=np.float32)

                             
            repr_data_path = args.save_repr_data_path + "/" + repr_set_name + ".pkl"
            with open(repr_data_path, "rb") as f:
                repr_data = pickle.load(f)
            if isinstance(repr_data, list):
                repr_data = np.array(repr_data, dtype=np.float32)
            print('TSRouter runtime message.', repr_data.shape)

                                   
            from encoder.base_encoder import EncoderFactory
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            repr_model, scaler, configs = EncoderFactory.build_encoder(args, device=device)
            repr_model = repr_model.to(device)
            repr_model.eval()
            INPUT_DIM = configs.input_dim

                           
            full_repr = repr_data[:, :INPUT_DIM]
            if scaler:
                full_repr_scaled = scaler.transform(full_repr)
            else:
                full_repr_scaled = full_repr

            full_dataset = TensorDataset(torch.from_numpy(full_repr_scaled).float())
            full_loader = DataLoader(
                full_dataset,
                batch_size=min(args.batch_size, len(full_dataset)),
                shuffle=False,
                drop_last=False
            )
            center_feats = []
            for batch_x in full_loader:
                batch_data = batch_x[0].unsqueeze(-1)  # (B,T,1)
                with torch.no_grad():
                    feats = repr_model.encode(batch_data.cuda()).cpu().numpy()
                center_feats.append(feats)
            center_repr = np.concatenate(center_feats, axis=0).astype(np.float32)

                                                         
            center_rank_by_metric = {
                metric_name: np.argsort(matrix, axis=0).T.astype(np.int64)
                for metric_name, matrix in metric_matrix_by_name.items()
            }
            center_rank = np.argsort(metrics_matrix, axis=0).T.astype(np.int64)  # (N_center, n_models)
            center_distance_weights = metric_weights[center_rank[:, 0]].astype(np.float32)
            v5_payload = {
                "__repr_format__": "v5_rank_centers",
                "center_repr": center_repr,
                "center_rank": center_rank,
                "center_rank_by_metric": center_rank_by_metric,
                "center_distance_weights": center_distance_weights,
                "model_metric_weights": {k: float(v) for k, v in zip(model_names, metric_weights)},
                "model_weight_algorithm": STRICT_RANK_WEIGHT_ALGORITHM,
                "model_weight_tie_breaker": STRICT_RANK_WEIGHT_TIE_BREAKER,
                "model_abbr_order": list(model_names),
                "metric": str(args.base_metrics),
                "repr_v": int(args.repr_v),
                **build_model_family_metadata(model_names),
                **_auto_cl_sidecar_fields(args),
            }
            if route_fast_weight_info is not None:
                v5_payload["route_efficiency_mode"] = True
                v5_payload["route_efficiency_weight_info"] = route_fast_weight_info

            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            os.makedirs(os.path.dirname(model_repr_path), exist_ok=True)
            weight_info = {
                "total_models": len(model_names),
                "model_weights": {k: float(v) for k, v in zip(model_names, metric_weights)},
                "weight_source": "step2_runtime" if route_fast_weight_dict is not None else "metric_prior",
                "model_weight_algorithm": STRICT_RANK_WEIGHT_ALGORITHM,
                "model_weight_tie_breaker": STRICT_RANK_WEIGHT_TIE_BREAKER,
                **build_model_family_metadata(model_names),
                **_auto_cl_sidecar_fields(args),
            }
            if route_fast_weight_info is not None:
                weight_info["route_efficiency_mode"] = True
                weight_info["route_efficiency_weight_info"] = route_fast_weight_info
            atomic_pickle_dump(weight_info, weight_path)
            atomic_pickle_dump(v5_payload, model_repr_path)

            v5_raw_top1_indices_dict = OrderedDict()
            for i, name in enumerate(model_names):
                v5_raw_top1_indices_dict[name] = np.where(center_rank[:, 0] == i)[0].astype(np.int64)
            subset_assign_payload = {
                "selected_indices_dict": {
                    k: np.asarray(v, dtype=np.int64) for k, v in v5_raw_top1_indices_dict.items()
                },
                "raw_center_top1_indices_dict": {
                    k: np.asarray(v, dtype=np.int64) for k, v in v5_raw_top1_indices_dict.items()
                },
                "model_names": list(model_names),
                "model_abbr_order": list(model_names),
                "stage": int(getattr(args, "current_zoo_num", len(model_names))),
                "zoo_total_num": int(getattr(args, "zoo_total_num", len(model_names))),
                "total_repr_centers": int(metrics_matrix.shape[1]),
                "total_repr_pool": None,
                "center_rank": center_rank,
                "mode": "v5_center_raw_rank_only",
                "cluster_phase_done": True,
                "pool_phase_done": False,
                **build_model_family_metadata(model_names),
                **_auto_cl_sidecar_fields(args),
            }
            if pool_phase_skip:
                subset_assign_payload = _merge_existing_pool_payload(subset_assign_payload, existing_subset_payload)
            atomic_pickle_dump(subset_assign_payload, subset_assign_path)
            _write_step3_manifest(
                args=args,
                repr_set_name=repr_set_name,
                weight_path=weight_path,
                model_repr_path=model_repr_path,
                subset_assign_path=subset_assign_path,
                model_names=list(model_names),
                pool_phase_done=bool(subset_assign_payload.get("pool_phase_done", False)),
            )
            step3_cluster_seconds = time.perf_counter() - step3_cluster_start
            print(f"[Step3][cluster] saved core artifacts: repr={model_repr_path}, weight={weight_path}")
            print(f"TSRouter runtime message: {model_repr_path}")

        step3_pool_start = time.perf_counter()
        if pool_phase_skip:
            v5_pool_phase_done = True
            step3_pool_seconds = 0.0
            print(f"[Step3][pool] skipped: existing pool artifacts are complete -> {subset_assign_path}")
        else:
            pool_meta = None
            pool_embeddings = None
            pool_assigned_indices_dict = None
            if bool(getattr(args, "enable_process_metrics", True)):
                print("[Step3][pool] begin: assign candidate pool and compute encoder enrichment")
                pool_meta = _load_repr_pool_meta(args, repr_set_name)
                if pool_meta is not None and "embeddings" in pool_meta:
                    pool_embeddings = np.asarray(pool_meta["embeddings"], dtype=np.float32)
                elif pool_meta is not None:
                    pool_embeddings = _load_or_encode_repr_pool_embeddings(args, pool_meta)
                if pool_embeddings is not None:
                    pool_assigned_indices_dict = _assign_query_points_by_v5_neighbors(
                        query_repr=pool_embeddings,
                        center_repr=center_repr,
                        center_rank=center_rank,
                        center_distance_weights=center_distance_weights,
                        model_names=list(model_names),
                        args=args,
                    )
            else:
                print("[Step3][pool] skipped: enable_process_metrics=false")
            print("[ENC_METRIC] v5 raw center-rank top1 sizes:", {k: int(len(v)) for k, v in v5_raw_top1_indices_dict.items()})
            if pool_assigned_indices_dict is not None:
                print(
                    f"[ENC_METRIC] v5 pool-search-assigned subset sizes ({_encoder_sim_param_text(args)}):",
                    {k: int(len(v)) for k, v in pool_assigned_indices_dict.items()},
                )

            subset_assign_payload = {
                "selected_indices_dict": {
                    k: np.asarray(v, dtype=np.int64) for k, v in (pool_assigned_indices_dict or v5_raw_top1_indices_dict).items()
                },
                "raw_center_top1_indices_dict": {
                    k: np.asarray(v, dtype=np.int64) for k, v in v5_raw_top1_indices_dict.items()
                },
                "model_names": list(model_names),
                "model_abbr_order": list(model_names),
                "stage": int(getattr(args, "current_zoo_num", len(model_names))),
                "zoo_total_num": int(getattr(args, "zoo_total_num", len(model_names))),
                "total_repr_centers": int(metrics_matrix.shape[1]),
                "total_repr_pool": int(pool_embeddings.shape[0]) if pool_assigned_indices_dict is not None else None,
                "center_rank": center_rank,
                "mode": "v5_pool_rank_fusion" if pool_assigned_indices_dict is not None else "v5_center_raw_rank_only",
                "cluster_phase_done": True,
                "pool_phase_done": False,
                **build_model_family_metadata(model_names),
                **_auto_cl_sidecar_fields(args),
            }
            enrichment_summary = None
            if bool(getattr(args, "enable_process_metrics", True)):
                if pool_assigned_indices_dict is not None:
                    enrichment_summary = _save_pool_encoder_enrichment_if_ready(
                        args=args,
                        current_zoo_abbr_order_list=current_zoo_abbr_order_list,
                        model_repr_path=model_repr_path,
                        model_names=list(model_names),
                        selected_indices_dict=pool_assigned_indices_dict,
                        extra={
                            "metric": str(args.base_metrics),
                            "repr_v": int(args.repr_v),
                            "mode": "v5_pool_rank_fusion",
                            **_encoder_sim_param_extra(args),
                        },
                    )
                if enrichment_summary is None:
                    reason = (
                        "pool_assignment_unavailable"
                        if pool_assigned_indices_dict is None
                        else "pool_per_sample_metrics_incomplete"
                    )
                    enrichment_summary = {"status": "unavailable", "reason": reason}
                    print(
                        f"⚠️ [ENC_METRIC] pool enrichment unavailable: {reason}; "
                        "do not report center Top1-derived Sub/PWW metrics."
                    )
                subset_assign_payload["encoder_quality"] = enrichment_summary
            v5_pool_phase_done = (
                not bool(getattr(args, "enable_process_metrics", True))
                or (
                    pool_assigned_indices_dict is not None
                    and isinstance(enrichment_summary, dict)
                    and enrichment_summary.get("status") != "unavailable"
                )
            )
            subset_assign_payload["pool_phase_done"] = v5_pool_phase_done
            atomic_pickle_dump(subset_assign_payload, subset_assign_path)
            _write_step3_manifest(
                args=args,
                repr_set_name=repr_set_name,
                weight_path=weight_path,
                model_repr_path=model_repr_path,
                subset_assign_path=subset_assign_path,
                model_names=list(model_names),
                pool_phase_done=v5_pool_phase_done,
            )
            print(
                f"[Step3][pool] done: pool_phase_done={v5_pool_phase_done}, "
                f"subset_assign={subset_assign_path}"
            )
            step3_pool_seconds = time.perf_counter() - step3_pool_start
        _write_step3_insert_timing(
            args=args,
            repr_set_name=repr_set_name,
            model_repr_path=model_repr_path,
            model_names=list(model_names),
            step3_skip_reuse_seconds=step3_skip_reuse_seconds,
            step3_cluster_seconds=step3_cluster_seconds,
            step3_pool_seconds=step3_pool_seconds,
            status="built",
            step2_coverage=step2_coverage,
        )
        print(f"TSRouter runtime message: {subset_assign_path}")
        print('-' * 100)
        return

    # ======================
                           
    # ======================
    subset_assign_path = model_repr_path.replace('.pkl', '_subset_assign.pkl')
    if cluster_phase_skip:
        final_repr_payload = _read_pickle_file(model_repr_path)
        if not isinstance(final_repr_payload, dict):
            raise ValueError(f"[Step3][cluster] cannot skip cluster; invalid model repr: {model_repr_path}")
        final_repr_dict = OrderedDict()
        for name in model_names:
            if name not in final_repr_payload:
                raise ValueError(f"[Step3][cluster] cannot skip cluster; model repr missing {name}: {model_repr_path}")
            final_repr_dict[name] = np.asarray(final_repr_payload[name], dtype=np.float32)

        weight_info = _read_pickle_file(weight_path)
        raw_weights = weight_info.get("model_weights", {}) if isinstance(weight_info, dict) else {}
        model_weights = {name: float(raw_weights.get(name, 1.0)) for name in model_names}

        subset_assign_payload = _read_pickle_file(subset_assign_path)
        if not isinstance(subset_assign_payload, dict):
            raise ValueError(f"[Step3][cluster] cannot skip cluster; invalid subset sidecar: {subset_assign_path}")
        raw_selected = subset_assign_payload.get("selected_indices_dict", {})
        if not isinstance(raw_selected, dict):
            raw_selected = {}
        selected_indices_dict = OrderedDict(
            (name, np.asarray(raw_selected.get(name, []), dtype=np.int64))
            for name in model_names
        )
        step3_cluster_seconds = 0.0
        print(f"[Step3][cluster] skipped: existing model repr/weight/subset artifacts are complete -> {model_repr_path}")
        _, step3_pool_seconds = _run_non_v5_pool_phase(
            args=args,
            repr_set_name=repr_set_name,
            weight_path=weight_path,
            model_repr_path=model_repr_path,
            subset_assign_path=subset_assign_path,
            current_zoo_abbr_order_list=current_zoo_abbr_order_list,
            model_names=list(model_names),
            final_repr_dict=final_repr_dict,
            model_weights=model_weights,
            selected_indices_dict=selected_indices_dict,
            subset_assign_payload=subset_assign_payload,
            metrics_matrix=metrics_matrix,
            pool_phase_skip=pool_phase_skip,
            repr_context=None,
        )
        _write_step3_insert_timing(
            args=args,
            repr_set_name=repr_set_name,
            model_repr_path=model_repr_path,
            model_names=list(model_names),
            step3_skip_reuse_seconds=step3_skip_reuse_seconds,
            step3_cluster_seconds=step3_cluster_seconds,
            step3_pool_seconds=step3_pool_seconds,
            status="built",
            step2_coverage=step2_coverage,
        )
        print(f"TSRouter runtime message: {model_repr_path}")
        print(f"TSRouter runtime message: {subset_assign_path}")
        print('-' * 100)
        return

    print("[Step3][cluster] begin: build model representation from advantage subsets")
    selection_score_dict, perf_adv_dict, metrics_dict = calculate_combined_scores(ordered_model_perf_dict)

    if Use_wta:
                                                   
                              
        winner_per_sample = np.argmin(metrics_matrix, axis=0)  # (n_data,)
    else:
        winner_per_sample = None

    # ======================
                          
    # ======================
    selected_indices_dict = OrderedDict()
    model_weights = {}             
    Top_k = int(args.subset_top_k)

    perf_scale = float(getattr(args, "subset_perf_scale", 1.0))

                                                     
    metric_weight_dict = None
    if route_fast_weight_dict is not None:
        metric_weight_dict = dict(route_fast_weight_dict)
        print("[route-fast] using Step2 efficiency prior weights:", metric_weight_dict)
    elif Use_repr_order:
        metric_array = np.array([metrics_dict[name] for name in model_names], dtype=float)
        metric_weights = _strict_lower_better_rank_weights(
            metric_array,
            model_names=list(model_names),
            use_perf_scale=Use_perf_scale,
            perf_scale=perf_scale,
        )
        metric_weight_dict = {
            name: float(w) for name, w in zip(model_names, metric_weights)
        }
        print("[Use_repr_order] strict-rank metric prior weights:", metric_weight_dict)
        _print_step3_weight_prior(
            source="performance",
            metric=f"Step2_forward_{_base_metric_display_name(args)}",
            model_names=list(model_names),
            score_values=metric_array,
            weights=metric_weights,
        )

    total_n = metrics_matrix.shape[1]         

    # =========
                    
    # =========
    if Use_rand_subset:
        rng = np.random.RandomState(int(getattr(args, "repr_encoder_seed", 2025)))
        for i, name in enumerate(model_names):
                                          
            k_rand = rng.randint(1, total_n + 1)
                                   
            indices = rng.choice(total_n, size=k_rand, replace=False)
            selected_indices_dict[name] = indices

                      
            if route_fast_weight_dict is not None:
                model_weights[name] = route_fast_weight_dict[name]
            elif Use_repr_order:
                                                                      
                model_weights[name] = metric_weight_dict[name]
            else:
                                              
                model_weights[name] = len(indices)

            print(f"TSRouter runtime message: {name}TSRouter runtime message: {len(indices)}")

    # =========
                       
    # =========
    else:
        for i, name in enumerate(model_names):
            scores = selection_score_dict[name]
            perf_scores = perf_adv_dict[name]
                                                                            

                                            
            if Use_repr_order:
                                                      
                indices = np.arange(len(scores))
            else:
                top_k = int(Top_k)
                if Use_perf_scale:                                                        
                    top_k = int(Top_k) * (1 - perf_scores * perf_scale)
                indices = np.where(scores > top_k)[0]

                                                        
            if Use_wta and winner_per_sample is not None:
                if len(indices) > 0:
                    mask = (winner_per_sample[indices] == i)
                    indices = indices[mask]

                                 
            if len(indices) == 0:
                if Use_wta and winner_per_sample is not None:
                                          
                    cand = np.where(winner_per_sample == i)[0]
                    if cand.size > 0:
                        best_j = int(cand[np.argmax(scores[cand])])
                        indices = [best_j]
                    else:
                                                   
                        best_j = int(np.argmax(scores))
                        indices = [best_j]
                else:
                                                                 
                    best_j = int(np.argmax(scores))
                    indices = [best_j]
                    if not Use_repr_order:
                        print(f"⚠️ Warning: No scores above threshold {top_k} for model {name}, selecting top 1 instead.")

                                 
            if Use_repr_order:
                print(f'TSRouter runtime message: {name}TSRouter runtime message: ', len(indices))
            else:
                print(f'TopK={top_k}TSRouter runtime message: {name}TSRouter runtime message: ', len(indices))

            selected_indices_dict[name] = indices

                                                           
            if route_fast_weight_dict is not None:
                model_weights[name] = route_fast_weight_dict[name]
            elif Use_repr_order:
                                                        
                model_weights[name] = metric_weight_dict[name]
            else:
                                    
                model_weights[name] = len(indices)

    print(model_weights)

                             
    abbr_to_id = {abbr: _sort_key(abbr)  for abbr in model_names}

    sorted_by_samples = sorted(
        model_weights.items(),
        key=lambda kv: (-kv[1], abbr_to_id[kv[0]]),
    )
                                
    model_sample_order = [abbr_to_id[name] for name, _ in sorted_by_samples]
    print('TSRouter runtime message.', model_sample_order)

    # ======================
                     
    # ======================
    repr_data_path = args.save_repr_data_path + "/" + repr_set_name + ".pkl"
    with open(repr_data_path, "rb") as f:
        repr_data = pickle.load(f)

    if isinstance(repr_data, list):
        repr_data = np.array(repr_data, dtype=np.float32)

    print('TSRouter runtime message.', repr_data.shape)

    # ======================
                           
    # ======================
    from encoder.base_encoder import EncoderFactory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repr_model, scaler, configs = EncoderFactory.build_encoder(args, device=device)
    repr_model = repr_model.to(device)                 
    repr_model.eval()              
    INPUT_DIM = configs.input_dim

    # ======================
                         
    # ======================
    cluster_results = {}          
    for zoo_dataset in model_names:
        if zoo_dataset in selected_indices_dict:
                        
            indices = selected_indices_dict[zoo_dataset]
            selected_repr = repr_data[indices, :INPUT_DIM]
            if scaler:
                selected_repr_scaled = scaler.transform(selected_repr)
            else:
                selected_repr_scaled = selected_repr

            dataset = TensorDataset(torch.from_numpy(selected_repr_scaled).float())
            loader = DataLoader(
                dataset,
                batch_size=min(args.batch_size, len(dataset)),
                shuffle=False,
                drop_last=False
            )

                  
            all_reprs = []
            for batch_x in loader:
                batch_data = batch_x[0].unsqueeze(-1)   # (B, T, 1)
                with torch.no_grad():
                    feats = repr_model.encode(batch_data.cuda()).cpu().numpy()
                all_reprs.append(feats)

            repr_array = np.concatenate(all_reprs, axis=0)

            # model_repr_mode ∈ {"mean", "all", "cluster"}：
                                                        
                                                        
                                                              
            model_repr_mode = getattr(args, "model_repr_mode", None)
            Max_cluster= getattr(args, "Max_cluster", 1)                   
            if model_repr_mode=="cluster":
                      
                best_k, cluster_labels = cluster_analysis(repr_array, zoo_dataset, max_clusters=Max_cluster,
                                                          silhouette_threshold=0.1, Repr_v=args.repr_v)

                if best_k == 1:          
                    # current_file_repr[zoo_dataset] = np.mean(repr, axis=0).reshape(1, -1)  # (1, DIM)
                    final_repr_dict[zoo_dataset] = repr_array.mean(axis=0).reshape(1, -1)
                else:           
                    cluster_centers = np.array(
                        [np.mean(repr_array[cluster_labels == k], axis=0) for k in range(best_k)])
                    # current_file_repr[zoo_dataset] = cluster_centers  # (n_clusters, DIM)
                    final_repr_dict[zoo_dataset] = cluster_centers  # shape: (k, dim)

                cluster_results[zoo_dataset] = {
                    'n_clusters': best_k,
                    'cluster_sizes': np.array([len(repr_array)]) if best_k == 1 else np.bincount(cluster_labels),
                    'silhouette_score': 0 if best_k == 1 else silhouette_score(repr_array, cluster_labels)
                }

                                                
                                             
                                                                                     
                                                                                          
                                                                        

                                   
            elif model_repr_mode == "mean":# shape: (1, dim)
                final_repr_dict[zoo_dataset] = repr_array.mean(axis=0).reshape(1, -1)
            elif model_repr_mode == "all":                       
                final_repr_dict[zoo_dataset] = repr_array
                print(f'{zoo_dataset} final_repr_array.shape:', repr_array.shape)
            else:
                raise ValueError(f"Unsupported model_repr_mode: {model_repr_mode}")



    # ======================
                
    # ======================
    weight_info = {
        "total_models": len(model_names),
        "model_weights": model_weights,
        "model_weight_algorithm": STRICT_RANK_WEIGHT_ALGORITHM
        if route_fast_weight_info is not None or Use_repr_order
        else "subset_size",
        "model_weight_tie_breaker": STRICT_RANK_WEIGHT_TIE_BREAKER
        if route_fast_weight_info is not None or Use_repr_order
        else None,
        **build_model_family_metadata(model_names),
        **_auto_cl_sidecar_fields(args),
    }
    if route_fast_weight_info is not None:
        weight_info["route_efficiency_mode"] = True
        weight_info["weight_source"] = "step2_runtime"
        weight_info["route_efficiency_weight_info"] = route_fast_weight_info
           
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_repr_path), exist_ok=True)
    atomic_pickle_dump(weight_info, weight_path)
    atomic_pickle_dump(final_repr_dict, model_repr_path)

                                      
    subset_assign_payload = {
        "selected_indices_dict": {
            k: np.asarray(v, dtype=np.int64) for k, v in selected_indices_dict.items()
        },
        "model_names": list(model_names),
        "model_abbr_order": list(model_names),
        "stage": int(getattr(args, "current_zoo_num", len(model_names))),
        "zoo_total_num": int(getattr(args, "zoo_total_num", len(model_names))),
        "total_repr_centers": int(total_n),
        "cluster_phase_done": True,
        "pool_phase_done": False,
        **build_model_family_metadata(model_names),
        **_auto_cl_sidecar_fields(args),
    }
    if pool_phase_skip:
        subset_assign_payload = _merge_existing_pool_payload(subset_assign_payload, existing_subset_payload)
    atomic_pickle_dump(subset_assign_payload, subset_assign_path)
    _write_step3_manifest(
        args=args,
        repr_set_name=repr_set_name,
        weight_path=weight_path,
        model_repr_path=model_repr_path,
        subset_assign_path=subset_assign_path,
        model_names=list(model_names),
        pool_phase_done=bool(subset_assign_payload.get("pool_phase_done", False)),
    )
    step3_cluster_seconds = time.perf_counter() - step3_cluster_start
    print(f"[Step3][cluster] saved core artifacts: repr={model_repr_path}, weight={weight_path}")
    _, step3_pool_seconds = _run_non_v5_pool_phase(
        args=args,
        repr_set_name=repr_set_name,
        weight_path=weight_path,
        model_repr_path=model_repr_path,
        subset_assign_path=subset_assign_path,
        current_zoo_abbr_order_list=current_zoo_abbr_order_list,
        model_names=list(model_names),
        final_repr_dict=final_repr_dict,
        model_weights=model_weights,
        selected_indices_dict=selected_indices_dict,
        subset_assign_payload=subset_assign_payload,
        metrics_matrix=metrics_matrix,
        pool_phase_skip=pool_phase_skip,
        repr_context=(repr_data, repr_model, scaler, INPUT_DIM),
    )
    _write_step3_insert_timing(
        args=args,
        repr_set_name=repr_set_name,
        model_repr_path=model_repr_path,
        model_names=list(model_names),
        step3_skip_reuse_seconds=step3_skip_reuse_seconds,
        step3_cluster_seconds=step3_cluster_seconds,
        step3_pool_seconds=step3_pool_seconds,
        status="built",
        step2_coverage=step2_coverage,
    )
                           
    # for model, arr in final_repr_dict.items():
                                                               
    print(f"TSRouter runtime message: {model_repr_path}")
    print(f"TSRouter runtime message: {subset_assign_path}")
    print('-' * 100)

    # ======================
                                      
    # ======================
    with open(model_repr_path, 'rb') as f:
        sampled_repr_dict = pickle.load(f)

    model_names = []
    vector_list = []

    for model_name, array in sampled_repr_dict.items():
        array = np.array(array)
        if array.ndim == 1:  # (128,)
            model_names.append(model_name)
            vector_list.append(array)
        elif array.ndim == 2:  # (N, 128)
            for i in range(array.shape[0]):
                model_names.append(f"{model_name}_v{i + 1}")
                vector_list.append(array[i])
        else:
            raise ValueError(f"Unsupported shape {array.shape} for model {model_name}")

    # vectors = np.stack(vector_list)  # shape: [num_vectors, dims]
    # visualize_similarity_heatmap(vectors, model_names, fig_save_path, title=plot_Model_repr_name)
    # visualize_radar_chart(vectors, model_names, fig_save_path, top_n=6, title=plot_Model_repr_name)
    # visualize_with_pca(vectors, model_names, fig_save_path, title=plot_Model_repr_name)
    # visualize_with_umap(vectors, model_names)
    # print('Successfully visualized!')

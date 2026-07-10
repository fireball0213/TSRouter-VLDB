import os
import pickle
import copy
import math
import csv
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import cuda
from scipy.spatial.distance import cdist

from config.model_zoo_config import build_route_family_target_by_model_id
from utils.data import numpy_to_gluonts, numpy_samples_to_gluonts
from selector.baselines.baseline_select import Baseline_Select_Model, Baseline_Select_Predictor
from encoder.encoder_config import ENCODER_CONFIG
from utils.io_lock import file_lock
from utils.path_utils import (
    auto_cl_enabled,
    get_auto_cl_mode,
    get_auto_cl_profile_by_name,
    get_repr_scale_protocol,
    normalize_route_family_mode,
    resolve_auto_cl_profile,
    route_efficiency_mode_enabled,
)
from utils.tsrouter_metrics import rank_decay_coef, rank_position_scores

VLDB_ROUTE_LATENCY_FIELDS = [
    "timestamp_utc",
    "route_id",
    "stage",
    "method",
    "profile_id",
    "route_family_mode",
    "dataset",
    "status",
    "zoo_size",
    "stable_model_ids",
    "selected_model_order",
    "step4_skip_saved",
    "cache_mode",
    "cache_hit",
    "timing_level",
    "timing_valid",
    "route_command_s",
    "cache_lookup_ms",
    "index_load_ms",
    "sample_ms",
    "sample_to_route_ms",
    "route_final_ms",
    "task_sampling_ms",
    "task_embedding_ms",
    "index_lookup_ms",
    "rank_ms",
    "route_overhead_ms",
    "selected_forecast_ms",
    "evaluate_ms",
    "metric_read_ms",
    "end_to_end_ms",
    "fast_eval_enabled",
    "evaluation_mode",
    "vldb_fast_sample",
    "vldb_fast_forward",
    "vldb_fast_eval",
    "task_sampling_timing_valid",
    "selected_forecast_timing_valid",
    "evaluate_timing_valid",
    "forward_mode",
    "plan_csv",
    "manifest_json",
    "timing_note",
]

TASK_SAMPLE_TIMING_FIELDS = [
    "timestamp_utc",
    "dataset",
    "sample_seconds",
    "sample_ms",
    "sample_repr_num",
    "task_sample_strategy",
    "task_window_sample_strategy",
    "sample_repr_ratio",
    "search_seed",
    "repr_scale_protocol",
    "cache_path",
    "cache_shape",
    "timing_valid",
]


def remap_route_family_rank_ids(rank, target_by_model_id) -> np.ndarray:
    """Replace every model id in a rank tensor with its family representative id."""
    arr = np.asarray(rank, dtype=np.int64)
    targets = np.asarray(target_by_model_id, dtype=np.int64).reshape(-1)
    if targets.size == 0:
        raise ValueError("route family target map is empty")
    invalid = arr[(arr >= targets.size)]
    if invalid.size:
        raise ValueError(
            f"route rank contains model ids outside family map: {np.unique(invalid).tolist()}"
        )
    out = arr.copy()
    valid = out >= 0
    out[valid] = targets[out[valid]]
    return out


def merge_route_family_task_rank_ids(
    rank,
    target_by_model_id,
    candidate_model_ids: list[int] | None = None,
) -> np.ndarray:
    """Merge a final task-level rank into one representative per model family.

    The raw rank is already fused at task level before this function runs.  Family
    modes therefore use the best final task position reached by any member, then
    replace that member with the requested representative size.  This avoids
    giving multi-size families extra votes during window/sample-level fusion.
    """
    arr = np.asarray(rank, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"route family task rank must be 2D, got shape={arr.shape}")
    targets = np.asarray(target_by_model_id, dtype=np.int64).reshape(-1)
    if targets.size == 0:
        raise ValueError("route family target map is empty")
    invalid = arr[(arr >= targets.size)]
    if invalid.size:
        raise ValueError(
            f"route rank contains model ids outside family map: {np.unique(invalid).tolist()}"
        )

    if candidate_model_ids is None:
        candidates = list(dict.fromkeys(int(value) for value in targets.tolist()))
    else:
        candidates = list(dict.fromkeys(int(value) for value in candidate_model_ids))
        invalid_candidates = [value for value in candidates if value < 0 or value >= targets.size]
        if invalid_candidates:
            raise ValueError(f"candidate_model_ids outside family map: {invalid_candidates}")
    if not candidates:
        raise ValueError("candidate_model_ids is empty")
    candidate_set = set(candidates)

    out = np.empty((len(candidates), arr.shape[1]), dtype=np.int64)
    for c in range(arr.shape[1]):
        seen: set[int] = set()
        merged: list[int] = []
        for raw_model_id in arr[:, c]:
            raw_model_id = int(raw_model_id)
            if raw_model_id < 0:
                continue
            target_id = int(targets[raw_model_id])
            if target_id not in candidate_set or target_id in seen:
                continue
            seen.add(target_id)
            merged.append(target_id)
        for target_id in candidates:
            if target_id not in seen:
                merged.append(target_id)
        out[:, c] = np.asarray(merged[: len(candidates)], dtype=np.int64)
    return out


def _ms(seconds: float) -> str:
    return f"{max(0.0, float(seconds)) * 1000.0:.3f}"


def _bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def _task_sample_timing_csv_path(cache_path: str | os.PathLike | None) -> str:
    if not cache_path:
        return ""
    base, _ = os.path.splitext(str(cache_path))
    return base + ".csv"


def _legacy_task_sample_timing_csv_path(cache_path: str | os.PathLike | None) -> str:
    path = _task_sample_timing_csv_path(cache_path)
    if not path:
        return ""
    base, ext = os.path.splitext(path)
    legacy_base = re.sub(r"_ws[^_]+(?:_sr[^_]+)?(?=_ss\d+$)", "", base)
    return legacy_base + ext if legacy_base != base else ""


def _read_task_sample_seconds_from_path(path: str, dataset_key: str) -> float | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    for row in reversed(rows):
        if str(row.get("dataset", "")) != str(dataset_key):
            continue
        if str(row.get("timing_valid", "true")).strip().lower() not in {"true", "1", "yes", "y", "t"}:
            continue
        for col in ["sample_seconds", "task_sampling_seconds"]:
            if col not in row:
                continue
            try:
                val = float(row.get(col, ""))
            except Exception:
                continue
            if np.isfinite(val) and val >= 0:
                return val
        try:
            val_ms = float(row.get("sample_ms", ""))
        except Exception:
            continue
        if np.isfinite(val_ms) and val_ms >= 0:
            return val_ms / 1000.0
    return None


def _read_task_sample_seconds(cache_path: str | os.PathLike | None, dataset_key: str) -> tuple[float | None, str]:
    exact_path = _task_sample_timing_csv_path(cache_path)
    exact_value = _read_task_sample_seconds_from_path(exact_path, dataset_key)
    if exact_value is not None:
        return exact_value, exact_path

    legacy_path = _legacy_task_sample_timing_csv_path(cache_path)
    legacy_value = _read_task_sample_seconds_from_path(legacy_path, dataset_key)
    if legacy_value is not None:
        return legacy_value, legacy_path
    return None, exact_path


def _upsert_task_sample_timing_row(
    cache_path: str | os.PathLike | None,
    dataset_key: str,
    args,
    sample_ms: float,
    cache_shape=None,
) -> None:
    path = _task_sample_timing_csv_path(cache_path)
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        sample_ms = float(sample_ms)
    except Exception:
        sample_ms = float("nan")
    sample_seconds = sample_ms / 1000.0 if np.isfinite(sample_ms) else float("nan")
    row = {
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset": str(dataset_key),
        "sample_seconds": f"{sample_seconds:.9f}" if np.isfinite(sample_seconds) else "",
        "sample_ms": f"{sample_ms:.3f}" if np.isfinite(sample_ms) else "",
        "sample_repr_num": str(int(getattr(args, "sample_repr_num", 0) or 0)),
        "task_sample_strategy": str(getattr(args, "task_sample_strategy", "latest_random")),
        "task_window_sample_strategy": str(getattr(args, "task_window_sample_strategy", "legacy")),
        "sample_repr_ratio": str(float(getattr(args, "sample_repr_ratio", 0.0) or 0.0)),
        "search_seed": str(int(getattr(args, "search_seed", 0) or 0)),
        "repr_scale_protocol": get_repr_scale_protocol(args),
        "cache_path": str(cache_path),
        "cache_shape": "" if cache_shape is None else "x".join(str(int(x)) for x in tuple(cache_shape)),
        "timing_valid": _bool_text(np.isfinite(sample_ms) and sample_ms >= 0),
    }
    with file_lock(path + ".lock"):
        rows = []
        if os.path.exists(path):
            try:
                with open(path, "r", newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
            except Exception:
                rows = []
        rows = [old for old in rows if str(old.get("dataset", "")) != str(dataset_key)]
        rows.append(row)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TASK_SAMPLE_TIMING_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for old in rows:
                writer.writerow({field: old.get(field, "") for field in TASK_SAMPLE_TIMING_FIELDS})


def _append_vldb_route_latency_row(args, row: dict) -> None:
    path = str(getattr(args, "vldb_route_latency_log", "") or "")
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {field: row.get(field, "") for field in VLDB_ROUTE_LATENCY_FIELDS}
    with file_lock(path + ".lock"):
        existing_rows = []
        if os.path.exists(path):
            try:
                with open(path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames == VLDB_ROUTE_LATENCY_FIELDS:
                        existing_rows = None
                    else:
                        existing_rows = list(reader)
            except Exception:
                existing_rows = []
        if existing_rows is not None:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=VLDB_ROUTE_LATENCY_FIELDS, extrasaction="ignore")
                writer.writeheader()
                for old_row in existing_rows:
                    writer.writerow({field: old_row.get(field, "") for field in VLDB_ROUTE_LATENCY_FIELDS})
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=VLDB_ROUTE_LATENCY_FIELDS, extrasaction="ignore")
            if os.path.getsize(path) == 0:
                writer.writeheader()
            writer.writerow(payload)
# =========================================================
                              
# =========================================================
class TSRouterModelSearcher:
    'TSRouter runtime message.'

    def __init__(self, args, zoo_model_size: dict, device: torch.device):
        self.args = args
        self.device = device
        self.ensemble_size = getattr(args, "ensemble_size", 1)

                       
        self.err_rate = getattr(args, "err_rate", 0)
        self.route_efficiency_mode = route_efficiency_mode_enabled(args)
        self.effective_repr_weight_ratio = float(getattr(args, "repr_weight_ratio", 0.0))
        if (
            self.route_efficiency_mode
            and self.effective_repr_weight_ratio == 0.0
            and str(getattr(args, "repr_v", "") or "")[:1] not in {"6", "7"}
        ):
            print("[route-fast] route_efficiency_mode=true, but repr_weight_ratio=0; fast prior is loaded but not applied to distances")

                                      
                           
        abbr_map = {
            "euc": "euclidean",
            "cos": "cosine",
            "cor": "correlation",
            "cit": "cityblock",
        }
        self.distance_metric = abbr_map[args.repr_distance_metric]
                           
        self.model_repr_agg = getattr(args, "model_repr_agg", "min")

                                       
        from utils.path_utils import get_repr_save_path
        _, weight_path, model_repr_path,_ = get_repr_save_path(self.args)
        model_repr_path = str(getattr(self.args, "model_repr_path_override", None) or model_repr_path)

        with open(model_repr_path, "rb") as f:
            print(f"[TSRouterSearcher] Load model repr from: {model_repr_path}")
            self.model_repr: Dict[str, np.ndarray] = pickle.load(f)

                                                                  
        self.sorted_models = self._flatten_and_sort_models(zoo_model_size)
        expected_abbr_order = [str(m["abbreviation"]) for m in self.sorted_models]
        self.route_family_mode = normalize_route_family_mode(
            getattr(args, "route_family_mode", "default")
        )
        self.route_family_target_by_model_id = self._load_route_family_target_map(
            model_repr_path=model_repr_path,
            expected_abbr_order=expected_abbr_order,
        )
        self.route_raw_candidate_model_ids = list(range(len(self.sorted_models)))
        self.route_candidate_model_ids = list(
            dict.fromkeys(self.route_family_target_by_model_id.tolist())
        )
        if self.route_family_mode != "default":
            print(
                "[TSRouterSearcher] route family merge: "
                f"mode={self.route_family_mode}, "
                f"target_by_model_id={self.route_family_target_by_model_id.tolist()}, "
                f"candidate_model_ids={self.route_candidate_model_ids}"
            )

                                                                         
        self.is_autoforecast_mode = (
            isinstance(self.model_repr, dict)
            and self.model_repr.get("__repr_format__") == "autoforecast_v1"
        )
        self.is_autoxpcr_mode = bool(
            self.is_autoforecast_mode
            and str(self.model_repr.get("selector_mode", "autoforecast")).strip().lower()
            == "autoxpcr"
        )
        self.is_simplets_mode = (
            isinstance(self.model_repr, dict)
            and self.model_repr.get("__repr_format__") == "simplets_v1"
        )
        self.is_v5_rank_mode = (
            isinstance(self.model_repr, dict)
            and self.model_repr.get("__repr_format__") == "v5_rank_centers"
        )
        if self.is_autoforecast_mode:
            local_abbr_order = list(
                self.model_repr.get("model_abbr_order", self.model_repr.get("model_names", []))
            )
            self._validate_model_repr_order(
                expected_abbr_order=expected_abbr_order,
                actual_abbr_order=[str(x) for x in local_abbr_order],
                model_repr_path=model_repr_path,
            )
            payload_model_weights = self.model_repr.get("model_metric_weights", {})
            self.model_weights = {
                model["abbreviation"]: float(payload_model_weights.get(model["abbreviation"], 1.0))
                for model in self.sorted_models
            }
            self.model_reprs_list = []
            input_dim = int(getattr(args, "repr_input_dim", getattr(args, "context_len", 512)) or 512)
            self.encoder_configs = SimpleNamespace(input_dim=input_dim)
            self.encoder_input_dim = input_dim
            self.repr_model = None
            self.scalers = None
            selector_label = "AutoXPCR" if self.is_autoxpcr_mode else "AutoForecast"
            print(
                f"[TSRouterSearcher] {selector_label} v7 loaded: "
                f"learner={self.model_repr.get('learner', 'unknown')}, "
                f"metric={self.model_repr.get('target_metric', self.model_repr.get('metric', 'unknown'))}, "
                f"train_samples={self.model_repr.get('train_samples', 'NA')}, "
                f"feature_dim={self.model_repr.get('feature_dim', 'NA')}"
            )
        elif self.is_simplets_mode:
            local_abbr_order = list(
                self.model_repr.get("model_abbr_order", self.model_repr.get("model_names", []))
            )
            self._validate_model_repr_order(
                expected_abbr_order=expected_abbr_order,
                actual_abbr_order=[str(x) for x in local_abbr_order],
                model_repr_path=model_repr_path,
            )
            payload_model_weights = self.model_repr.get("model_metric_weights", {})
            self.model_weights = {
                model["abbreviation"]: float(payload_model_weights.get(model["abbreviation"], 1.0))
                for model in self.sorted_models
            }
            self.model_reprs_list = []
            ts2vec_config = dict(self.model_repr.get("ts2vec_config", {}))
            input_dim = int(
                ts2vec_config.get(
                    "input_dim",
                    getattr(args, "repr_input_dim", getattr(args, "context_len", 512)),
                )
                or 512
            )
            self.encoder_configs = SimpleNamespace(input_dim=input_dim)
            self.encoder_input_dim = input_dim
            self.repr_model = None
            self.scalers = None
            from selector.TSRouter_Select.simplets_select import load_simplets_encoder_cached

            self.simplets_encoder, simplets_encoder_cache_hit = load_simplets_encoder_cached(
                self.model_repr,
                device=self.device,
            )
            print(
                "[TSRouterSearcher] SimpleTS v1 loaded: "
                f"metric={self.model_repr.get('target_metric', 'unknown')}, "
                f"clusters={self.model_repr.get('n_clusters', 'NA')}, "
                f"train_samples={self.model_repr.get('train_samples', 'NA')}, "
                f"embedding_dim={self.model_repr.get('embedding_dim', 'NA')}, "
                f"input_dim={input_dim}, encoder_cache_hit={str(simplets_encoder_cache_hit).lower()}, "
                f"source_repr={self.model_repr.get('source_repr_set_name', self.model_repr.get('repr_set_name', 'NA'))}, "
                f"selector_repr={self.model_repr.get('selector_repr_set_name', 'TS2Vec')}"
            )
        elif self.is_v5_rank_mode:
            self.center_repr = np.asarray(self.model_repr["center_repr"], dtype=np.float32)
            center_rank = self._select_v5_center_rank(self.model_repr)
            self.center_distance_weights = np.asarray(
                self.model_repr.get("center_distance_weights", np.ones(self.center_repr.shape[0])),
                dtype=np.float32,
            ).reshape(-1)
            if self.center_distance_weights.shape[0] != self.center_repr.shape[0]:
                self.center_distance_weights = np.ones(self.center_repr.shape[0], dtype=np.float32)
            local_abbr_order = list(self.model_repr["model_abbr_order"])
            self._validate_model_repr_order(
                expected_abbr_order=expected_abbr_order,
                actual_abbr_order=[str(x) for x in local_abbr_order],
                model_repr_path=model_repr_path,
            )
            global_idx_map = {m["abbreviation"]: i for i, m in enumerate(self.sorted_models)}
            local_pos_to_global = np.asarray([global_idx_map.get(abbr, -1) for abbr in local_abbr_order], dtype=np.int64)
            self.center_rank_global = local_pos_to_global[center_rank]
            weight_fallback_keys = {m["abbreviation"]: 1.0 for m in self.sorted_models}
            self.model_weights = self._load_model_weights(weight_path, weight_fallback_keys)
            payload_model_weights = self.model_repr.get("model_metric_weights", {})
            if isinstance(payload_model_weights, dict):
                raw_model_weights = {}
                for model in self.sorted_models:
                    abbr = model["abbreviation"]
                    if abbr not in payload_model_weights:
                        raw_model_weights = {}
                        break
                    raw_model_weights[abbr] = float(payload_model_weights[abbr])
                if raw_model_weights:
                    if raw_model_weights != self.model_weights:
                        print("[TSRouterSearcher] use raw v5 model weights from model repr payload")
                    self.model_weights = raw_model_weights
            print(f"[TSRouterSearcher] repr_v5 loaded: centers={self.center_repr.shape[0]}")
        else:
            self._validate_model_repr_order(
                expected_abbr_order=expected_abbr_order,
                actual_abbr_order=[str(x) for x in self.model_repr.keys()],
                model_repr_path=model_repr_path,
            )
            self.model_weights = self._load_model_weights(weight_path, self.model_repr)
            self.model_reprs_list = [self.model_repr[m["abbreviation"]] for m in self.sorted_models]

                                            
        if not (self.is_autoforecast_mode or self.is_simplets_mode):
            self.repr_model, self.scalers, self.encoder_configs = self._load_repr_encoder()
            self.encoder_input_dim = int(self.encoder_configs.input_dim)

    @staticmethod
    def _flatten_and_sort_models(zoo_model_size: dict) -> List[dict]:
        'TSRouter runtime message.'
        all_models = [
            details
            for family in zoo_model_size.values()
            for details in family.values()
        ]
        return sorted(all_models, key=lambda x: x["id"])

    def _load_route_family_target_map(
        self,
        *,
        model_repr_path: str,
        expected_abbr_order: List[str],
    ) -> np.ndarray:
        if self.route_family_mode == "default":
            return np.arange(len(expected_abbr_order), dtype=np.int64)

        manifest_path = str(model_repr_path).replace(".pkl", "_model_manifest.json")
        metadata_source = manifest_path
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except Exception as exc:
                raise ValueError(
                    f"cannot read Step3 model-family metadata: path={manifest_path}, "
                    f"error={type(exc).__name__}: {exc}"
                ) from exc
        elif isinstance(self.model_repr, dict) and "model_family_members" in self.model_repr:
            metadata = self.model_repr
            metadata_source = model_repr_path
        else:
            raise FileNotFoundError(
                "route_family_mode requires Step3 model-family metadata; "
                f"missing manifest={manifest_path} and metadata is absent from repr={model_repr_path}. "
                "Re-run Step3 once to backfill it."
            )
        manifest_order = [str(value) for value in metadata.get("model_abbr_order", [])]
        if manifest_order != list(expected_abbr_order):
            raise ValueError(
                "Step3 model-family metadata order mismatch; "
                f"expected={expected_abbr_order}, actual={manifest_order}, path={metadata_source}"
            )
        try:
            target_by_id = build_route_family_target_by_model_id(
                expected_abbr_order,
                metadata,
                self.route_family_mode,
            )
        except ValueError as exc:
            raise ValueError(
                f"invalid Step3 model-family metadata: path={metadata_source}; {exc}"
            ) from exc
        return np.asarray(target_by_id, dtype=np.int64)

    def apply_route_family_mode(self, rank) -> np.ndarray:
        return remap_route_family_rank_ids(
            rank,
            self.route_family_target_by_model_id,
        )

    def merge_route_family_task_rank(self, rank) -> np.ndarray:
        return merge_route_family_task_rank_ids(
            rank,
            self.route_family_target_by_model_id,
            candidate_model_ids=self.route_candidate_model_ids,
        )

    @staticmethod
    def _validate_model_repr_order(
        *,
        expected_abbr_order: List[str],
        actual_abbr_order: List[str],
        model_repr_path: str,
    ) -> None:
        if list(actual_abbr_order) == list(expected_abbr_order):
            return
        actual_set = set(actual_abbr_order)
        expected_set = set(expected_abbr_order)
        missing = [abbr for abbr in expected_abbr_order if abbr not in actual_set]
        extra = [abbr for abbr in actual_abbr_order if abbr not in expected_set]
        raise ValueError(
            "[TSRouterSearcher] model repr order mismatch; "
            f"missing={missing}, extra={extra}, "
            f"expected_count={len(expected_abbr_order)}, actual_count={len(actual_abbr_order)}, "
            f"model_repr_path={model_repr_path}"
        )

    def _target_rank_metric(self) -> str:
        raw = (
            getattr(self.args, "sgl_rank_metric", None)
            or getattr(self.args, "rank_metric", None)
            or getattr(self.args, "real_order_metric", None)
            or None
        )
        if raw is None:
            code = str(getattr(self.args, "base_metrics", "M")).strip().upper()
            return {"M": "MASE", "S": "sMAPE", "C": "CRPS"}.get(code, "MASE")
        raw = str(raw).strip()
        return {"M": "MASE", "S": "sMAPE", "C": "CRPS", "SMAPE": "sMAPE"}.get(raw.upper(), raw)

    def _select_v5_center_rank(self, model_repr: dict) -> np.ndarray:
        rank_by_metric = model_repr.get("center_rank_by_metric", {})
        metric = self._target_rank_metric()
        if isinstance(rank_by_metric, dict) and metric in rank_by_metric:
            print(f"[TSRouterSearcher] repr_v5 rank metric: {metric}")
            return np.asarray(rank_by_metric[metric], dtype=np.int64)
        if isinstance(rank_by_metric, dict):
            lower_map = {str(k).lower(): k for k in rank_by_metric.keys()}
            hit = lower_map.get(metric.lower())
            if hit is not None:
                print(f"[TSRouterSearcher] repr_v5 rank metric: {hit}")
                return np.asarray(rank_by_metric[hit], dtype=np.int64)
        saved_metric = model_repr.get("metric", "unknown")
        if saved_metric not in {None, metric}:
            print(f"[TSRouterSearcher] repr_v5 rank metric fallback: saved={saved_metric}, requested={metric}")
        return np.asarray(model_repr["center_rank"], dtype=np.int64)

    def _load_model_weights(self, weight_path: str, model_repr: Dict[str, np.ndarray]) -> Dict[str, float]:
        try:
            with open(weight_path, "rb") as f:
                weight_info = pickle.load(f)
                print(f"TSRouter runtime message: {weight_info['total_models']}")
                model_names = list(weight_info["model_weights"].keys())
                weights = [round(v, 2) for v in weight_info["model_weights"].values()]
                print('TSRouter runtime message.', model_names)
                print('TSRouter runtime message.', weights)
                return weight_info["model_weights"]
        except FileNotFoundError:
            print(f"TSRouter runtime message: {weight_path}TSRouter runtime message: ")
            return {name: 1.0 for name in model_repr.keys()}

    def _load_repr_encoder(self):
        from encoder.base_encoder import EncoderFactory
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        repr_model, scaler, configs= EncoderFactory.build_encoder(self.args, device=device)
        repr_model = repr_model.to(device)                 
        repr_model.eval()              

        return repr_model,scaler, configs

    def _compute_rank_from_v5_centers(self, samples_repr: np.ndarray) -> np.ndarray:
        'TSRouter runtime message.'
        C = samples_repr.shape[0]
        K = len(self.sorted_models)
        n_center = self.center_repr.shape[0]
        topk = int(getattr(self.args, "repr_v5_nearest_k", 10))
        topk = max(1, min(topk, n_center))
        power = float(getattr(self.args, "repr_v5_distance_power", 1.0))
        weight_ratio = float(getattr(self, "effective_repr_weight_ratio", getattr(self.args, "repr_weight_ratio", 0.0)))

        dmat = cdist(samples_repr, self.center_repr, metric=self.distance_metric)  # (C, N_center)
        if weight_ratio != 0:
            dmat = dmat / (np.power(self.center_distance_weights[None, :], weight_ratio) + 1e-8)
        out = np.zeros((C, K), dtype=np.int64)
        score_template = rank_position_scores(K, decay_coef=rank_decay_coef(self.args))
        eps = 1e-8

        for c in range(C):
            row = dmat[c]
            idx = np.argpartition(row, kth=topk - 1)[:topk]
            d = row[idx]
            w = 1.0 / np.power(d + eps, power)
            scores = np.zeros(K, dtype=np.float32)
            for j, center_id in enumerate(idx):
                order = self.center_rank_global[center_id]  # (K,)
                valid = order >= 0
                if np.any(valid):
                    scores[order[valid]] += w[j] * score_template[:np.sum(valid)]
            out[c] = np.argsort(-scores)
        return out


    # ----------------------------
                               
    # ----------------------------
    def _compute_distance_matrix(self, samples_repr: np.ndarray) -> np.ndarray:
        'TSRouter runtime message.'
        C = samples_repr.shape[0]
        K = len(self.model_reprs_list)                                          
        distances = np.zeros((C, K), dtype=np.float32)

                
        weight_ratio = float(getattr(self, "effective_repr_weight_ratio", getattr(self.args, "repr_weight_ratio", 0.0)))
        eps = 1e-8

        for model_idx in range(K):
            model_name = self.sorted_models[model_idx]["abbreviation"]
            weight = float(self.model_weights.get(model_name, 1.0))

            model_repr = self.model_reprs_list[model_idx]
            if model_repr.ndim == 1:
                model_repr = model_repr.reshape(1, -1)
            elif model_repr.ndim != 2:
                raise ValueError(f"TSRouter runtime message: {model_repr.ndim}")

                                        
            dists = cdist(samples_repr, model_repr, metric=self.distance_metric)

                               
            if self.model_repr_agg == "min":
                d = dists.min(axis=1)
            elif self.model_repr_agg == "min3":
                if dists.shape[1] >= 3:
                    nearest3 = np.partition(dists, 2, axis=1)[:, :3]
                    d = nearest3.mean(axis=1)
                else:                         
                    d = dists.min(axis=1)
            elif self.model_repr_agg == "mean":
                d = dists.mean(axis=1)
            elif self.model_repr_agg == "median":
                d = np.median(dists, axis=1)
            else:
                raise ValueError(f"Unknown model_repr_agg: {self.model_repr_agg}")

                                                          
            if weight_ratio != 0:
                d = d / (weight ** weight_ratio + eps)
            distances[:, model_idx] = d
            # print(model_name,weight,d.mean(),distances[:, model_idx].mean())
        return distances

    # ----------------------------
                                                
    # ----------------------------
    def fit(self, samples: torch.Tensor, use_norm: bool = False) -> np.ndarray:
        'TSRouter runtime message.'
        if isinstance(samples, list):
            samples = torch.cat(samples, dim=0)

        if not isinstance(samples, torch.Tensor):
            raise TypeError(f"samples should be torch.Tensor or List[Tensor], got {type(samples)}")
        if samples.ndim != 3:
            raise ValueError(f"samples ndim should be 3, got {samples.ndim} with shape={tuple(samples.shape)}")

        N, T, C = samples.shape
        if getattr(self, "is_simplets_mode", False):
            from selector.TSRouter_Select.simplets_select import predict_simplets_rank_tensor

            top_k_indices, timing = predict_simplets_rank_tensor(
                self.model_repr,
                samples.detach().cpu().numpy(),
                encoder=self.simplets_encoder,
            )
            self.last_task_embedding_ms = float(timing.get("feature_ms", 0.0))
            self.last_index_lookup_ms = float(timing.get("predict_ms", 0.0))
            self.last_rank_ms = float(timing.get("rank_ms", 0.0))
            self.last_simplets_timing = timing
            print(
                "[SimpleTS Route] "
                f"mode=aggregate samples={N} channels={C} context={T} "
                f"classifier={timing.get('classifier_backend', 'unknown')}/"
                f"{timing.get('classifier_name', 'unknown')} "
                f"feature={self.last_task_embedding_ms / 1000.0:.3f}s "
                f"predict={self.last_index_lookup_ms / 1000.0:.3f}s "
                f"(raw={float(timing.get('predict_raw_ms', 0.0)) / 1000.0:.3f}s, "
                f"post={float(timing.get('predict_postprocess_ms', 0.0)) / 1000.0:.3f}s) "
                f"rank={self.last_rank_ms / 1000.0:.3f}s",
                flush=True,
            )
            return top_k_indices

        if getattr(self, "is_autoforecast_mode", False):
            from selector.TSRouter_Select.autoforecast_select import predict_autoforecast_rank_tensor

            top_k_indices, timing = predict_autoforecast_rank_tensor(
                self.model_repr,
                samples.detach().cpu().numpy(),
            )
            self.last_task_embedding_ms = float(timing.get("feature_ms", 0.0))
            self.last_index_lookup_ms = float(timing.get("predict_ms", 0.0))
            self.last_rank_ms = float(timing.get("rank_ms", 0.0))
            self.last_autoforecast_timing = timing
            selector_label = "AutoXPCR" if self.is_autoxpcr_mode else "AutoForecast"
            print(
                f"[{selector_label} Route] "
                f"mode=aggregate samples={N} channels={C} context={T} "
                f"feature={self.last_task_embedding_ms / 1000.0:.3f}s "
                f"predict={self.last_index_lookup_ms / 1000.0:.3f}s "
                f"rank={self.last_rank_ms / 1000.0:.3f}s",
                flush=True,
            )
            return top_k_indices

                                                 
        samples_reshaped = samples.permute(0, 2, 1).reshape(N * C, T, 1)

                        
        t_encode0 = time.perf_counter()
        if use_norm:
            print("[TSRouterSearcher] Using scaler normalization for representation extraction")
                                                         
            samples_np = samples_reshaped.squeeze(-1).detach().cpu().numpy()   # (N*C, T)
            samples_norm_np = self.scalers.transform(samples_np)
            samples_norm = torch.from_numpy(samples_norm_np).to(samples.device).unsqueeze(-1)
            feat = self.repr_model.encode(samples_norm.float())
        else:
            feat = self.repr_model.encode(samples_reshaped.float())

        # feat: (N*C, D) -> (N, C, D) -> mean over N -> (C, D)
        feat_np = feat.detach().cpu().numpy()
        samples_repr = feat_np.reshape(N, C, -1).mean(axis=0)
        t_encode1 = time.perf_counter()

        t_lookup0 = time.perf_counter()
        if self.is_v5_rank_mode:
            top_k_indices = self._compute_rank_from_v5_centers(samples_repr)  # (C,K)
            distance_ref = np.min(top_k_indices.astype(np.float32), axis=1)
        else:
            distances = self._compute_distance_matrix(samples_repr)  # (C,K)
            top_k_indices = np.argsort(distances, axis=1)
            distance_ref = np.min(distances, axis=1)
        t_lookup1 = time.perf_counter()
        self.last_task_embedding_ms = (t_encode1 - t_encode0) * 1000.0
        self.last_index_lookup_ms = (t_lookup1 - t_lookup0) * 1000.0
        self.last_rank_ms = 0.0

                                              
        if self.err_rate != 0:
                                           
            min_distances = distance_ref
            if self.err_rate < 0:
                                                
                l_interval = np.mean(min_distances) - 0.5 * np.std(min_distances)
                r_interval = np.mean(min_distances) + 0.5 * np.std(min_distances)
                top_k_indices[min_distances < l_interval, :] = -1
            else:
                                          
                top_k_indices[min_distances < self.err_rate, :] = -1

        # (C, K) -> (K, 1, C)，priority=K
        top_k_indices = top_k_indices.T.reshape(-1, 1, C)
        return top_k_indices

    def fit_autoforecast_samples(self, samples) -> np.ndarray:
        if not getattr(self, "is_autoforecast_mode", False):
            raise RuntimeError("fit_autoforecast_samples requires an AutoForecast model artifact")
        if isinstance(samples, torch.Tensor):
            samples_np = samples.detach().cpu().numpy()
        else:
            samples_np = np.asarray(samples, dtype=np.float32)
        if samples_np.ndim != 3:
            raise ValueError(
                f"AutoForecast batched samples must have shape (N,T,C), got {samples_np.shape}"
            )

        from selector.TSRouter_Select.autoforecast_select import (
            predict_autoforecast_sample_rank_tensors,
        )

        rank_tensors, timing = predict_autoforecast_sample_rank_tensors(
            self.model_repr,
            samples_np,
        )
        self.last_task_embedding_ms = float(timing.get("feature_ms", 0.0))
        self.last_index_lookup_ms = float(timing.get("predict_ms", 0.0))
        self.last_rank_ms = float(timing.get("rank_ms", 0.0))
        self.last_autoforecast_timing = timing
        n, _k, c = rank_tensors.shape
        selector_label = "AutoXPCR" if self.is_autoxpcr_mode else "AutoForecast"
        print(
            f"[{selector_label} Route] "
            f"mode=per-sample-batch samples={n} channels={c} context={samples_np.shape[1]} "
            f"feature={self.last_task_embedding_ms / 1000.0:.3f}s "
            f"predict={self.last_index_lookup_ms / 1000.0:.3f}s "
            f"rank={self.last_rank_ms / 1000.0:.3f}s",
            flush=True,
        )
        return rank_tensors

    def fit_simplets_samples(self, samples) -> np.ndarray:
        if not getattr(self, "is_simplets_mode", False):
            raise RuntimeError("fit_simplets_samples requires a SimpleTS model artifact")
        if isinstance(samples, torch.Tensor):
            samples_np = samples.detach().cpu().numpy()
        else:
            samples_np = np.asarray(samples, dtype=np.float32)
        if samples_np.ndim != 3:
            raise ValueError(
                f"SimpleTS batched samples must have shape (N,T,C), got {samples_np.shape}"
            )

        from selector.TSRouter_Select.simplets_select import predict_simplets_sample_rank_tensors

        rank_tensors, timing = predict_simplets_sample_rank_tensors(
            self.model_repr,
            samples_np,
            encoder=self.simplets_encoder,
        )
        self.last_task_embedding_ms = float(timing.get("feature_ms", 0.0))
        self.last_index_lookup_ms = float(timing.get("predict_ms", 0.0))
        self.last_rank_ms = float(timing.get("rank_ms", 0.0))
        self.last_simplets_timing = timing
        n, _k, c = rank_tensors.shape
        print(
            "[SimpleTS Route] "
            f"mode=per-sample-batch samples={n} channels={c} context={samples_np.shape[1]} "
            f"classifier={timing.get('classifier_backend', 'unknown')}/"
            f"{timing.get('classifier_name', 'unknown')} "
            f"feature={self.last_task_embedding_ms / 1000.0:.3f}s "
            f"predict={self.last_index_lookup_ms / 1000.0:.3f}s "
            f"(raw={float(timing.get('predict_raw_ms', 0.0)) / 1000.0:.3f}s, "
            f"post={float(timing.get('predict_postprocess_ms', 0.0)) / 1000.0:.3f}s) "
            f"rank={self.last_rank_ms / 1000.0:.3f}s",
            flush=True,
        )
        return rank_tensors


# class TSRouterModelSearcher:
#     """
                   
                             
                           
                                     
#     """
#
#     def __init__(self, args, zoo_model_size: dict, device: torch.device):
#         self.args = args
#         self.device = device
#         self.ensemble_size = getattr(args, "ensemble_size", 1)
#         self.use_search_ensemble = bool(getattr(args, "enable_search_ensemble", False))
#
                         
#         self.err_rate = getattr(args, "err_rate", 0)
#
                                        
                             
#         abbr_map = {
#             "euc": "euclidean",
#             "cos": "cosine",
#             "cor": "correlation",
#             "cit": "cityblock",
#         }
#         self.distance_metric = abbr_map[args.repr_distance_metric]
                             
#         self.model_repr_agg = getattr(args, "model_repr_agg", "min")
#
                                                                 
#         all_models = [
#             details
#             for family in zoo_model_size.values()
#             for details in family.values()
#         ]
#         self.sorted_models = sorted(all_models, key=lambda x: x["id"])
                                 
#         self.search_units = self._build_search_units()
#         if len(self.search_units) == 0:
                                                                           
#
                                                        
#         self.encoder_input_dim = max(int(u["encoder_configs"].input_dim) for u in self.search_units)
#         active_names = [u["encoder_name"] for u in self.search_units]
#         print(f"[TSRouterSearcher] active encoders for search: {active_names}, max_input_dim={self.encoder_input_dim}")
#
#     def _load_model_weights(self, weight_path: str, model_repr: Dict[str, np.ndarray]) -> Dict[str, float]:
#         try:
#             with open(weight_path, "rb") as f:
#                 weight_info = pickle.load(f)
                                                                           
#                 model_names = list(weight_info["model_weights"].keys())
#                 weights = [round(v, 2) for v in weight_info["model_weights"].values()]
                                             
                                         
#                 return weight_info["model_weights"]
#         except FileNotFoundError:
                                                              
#             return {name: 1.0 for name in model_repr.keys()}
#
#     def _load_repr_encoder(self, args_for_encoder):
#         from encoder.base_encoder import EncoderFactory
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         repr_model, scaler, configs = EncoderFactory.build_encoder(args_for_encoder, device=device)
                                                             
                                         
#
#         return repr_model, scaler, configs
#
#     def _build_args_for_encoder(self, encoder_name: str):
#         """
                                    
                                      
                                                              
#         """
#         cfg = ENCODER_CONFIG[encoder_name]
#         a = copy.copy(self.args)
#         a.repr_encoder = encoder_name
#         if "fixed_input_dim" in cfg:
#             a.repr_input_dim = int(cfg["fixed_input_dim"])
#         else:
#             a.repr_input_dim = int(cfg.get("default_input_dim", getattr(self.args, "repr_input_dim", 96)))
#         a.repr_output_dim = int(cfg.get("default_embedding_dim", getattr(self.args, "repr_output_dim", 128)))
#         a.repr_sub_pred_len = int(cfg.get("default_sub_pred_len", getattr(self.args, "repr_sub_pred_len", 192)))
#         return a
#
#     def _iter_ensemble_encoder_names(self):
#         if not self.use_search_ensemble:
#             return [self.args.repr_encoder]
#         names = []
#         for name, cfg in ENCODER_CONFIG.items():
#             if bool(cfg.get("search_ensemble_enable", False)):
#                 names.append(name)
#         return names
#
#     def _build_search_units(self):
#         """
                 
                                                   
                             
#         """
#         from utils.path_utils import get_repr_save_path
#
#         units = []
#         for enc_name in self._iter_ensemble_encoder_names():
#             enc_args = self._build_args_for_encoder(enc_name)
#             _, weight_path, model_repr_path, _ = get_repr_save_path(enc_args)
#
#             if not os.path.exists(model_repr_path):
#                 print(f"⚠️ [SearchEnsemble] skip encoder={enc_name}, repr not found: {model_repr_path}")
#                 continue
#
#             with open(model_repr_path, "rb") as f:
#                 model_repr = pickle.load(f)
#             model_weights = self._load_model_weights(weight_path, model_repr)
#             model_reprs_list = [model_repr[m["abbreviation"]] for m in self.sorted_models]
#             repr_model, scaler, configs = self._load_repr_encoder(enc_args)
#
#             units.append({
#                 "encoder_name": enc_name,
#                 "args": enc_args,
#                 "repr_model": repr_model,
#                 "scaler": scaler,
#                 "encoder_configs": configs,
#                 "model_reprs_list": model_reprs_list,
#                 "model_weights": model_weights,
#             })
#             print(f"✅ [SearchEnsemble] add encoder={enc_name}, repr={model_repr_path}")
#         return units
#
#     def _adjust_samples_len(self, samples: torch.Tensor, target_len: int) -> torch.Tensor:
#         """
                                       
                          
                             
#         """
#         n, t, c = samples.shape
#         if t == target_len:
#             return samples
#         if t > target_len:
#             return samples[:, -target_len:, :]
#         pad = torch.zeros((n, target_len - t, c), dtype=samples.dtype, device=samples.device)
#         return torch.cat([pad, samples], dim=1)
#
#     # ----------------------------
                                 
#     # ----------------------------
#     def _compute_distance_matrix(self, samples_repr: np.ndarray, unit: dict) -> np.ndarray:
#         """
             
                                              
             
                                                    
#         """
#         C = samples_repr.shape[0]
#         model_reprs_list = unit["model_reprs_list"]
#         model_weights = unit["model_weights"]
                                                                             
#         distances = np.zeros((C, K), dtype=np.float32)
#
                  
#         weight_ratio = getattr(self.args, "repr_weight_ratio", 0.0)
#         eps = 1e-8
#
#         for model_idx in range(K):
#             model_name = self.sorted_models[model_idx]["abbreviation"]
#             weight = float(model_weights.get(model_name, 1.0))
#             model_repr = model_reprs_list[model_idx]
#             if model_repr.ndim == 1:
#                 model_repr = model_repr.reshape(1, -1)
#             elif model_repr.ndim != 2:
                                                                           
#
                                          
#             dists = cdist(samples_repr, model_repr, metric=self.distance_metric)
#
                                 
#             if self.model_repr_agg == "min":
#                 d = dists.min(axis=1)
#             elif self.model_repr_agg == "min3":
#                 if dists.shape[1] >= 3:
#                     nearest3 = np.partition(dists, 2, axis=1)[:, :3]
#                     d = nearest3.mean(axis=1)
                                                
#                     d = dists.min(axis=1)
#             elif self.model_repr_agg == "mean":
#                 d = dists.mean(axis=1)
#             elif self.model_repr_agg == "median":
#                 d = np.median(dists, axis=1)
#             else:
#                 raise ValueError(f"Unknown model_repr_agg: {self.model_repr_agg}")
#
                                                      
#             distances[:, model_idx] = d / (weight ** weight_ratio + eps)
#             # print(model_name,weight,d.mean(),distances[:, model_idx].mean())
#         return distances
#
#     # ----------------------------
                                                  
#     # ----------------------------
#     def fit(self, samples: torch.Tensor, use_norm: bool = False) -> np.ndarray:
#         """
             
                                                                    
             
#             top_k_indices: shape = (priority, 1, C)
                                                     
#         """
#         if isinstance(samples, list):
#             samples = torch.cat(samples, dim=0)
#
#         if not isinstance(samples, torch.Tensor):
#             raise TypeError(f"samples should be torch.Tensor or List[Tensor], got {type(samples)}")
#
#         N, _, C = samples.shape
#         unit_orders = []
#         for unit in self.search_units:
                                          
#             target_len = int(unit["encoder_configs"].input_dim)
#             s_unit = self._adjust_samples_len(samples, target_len)
#             # 2) (N,T,C)->(N*C,T,1)
#             samples_reshaped = s_unit.permute(0, 2, 1).reshape(N * C, target_len, 1)
                     
#             repr_model = unit["repr_model"]
#             scaler = unit["scaler"]
#             if use_norm and scaler is not None:
#                 samples_np = samples_reshaped.squeeze(-1).detach().cpu().numpy()
#                 samples_norm_np = scaler.transform(samples_np)
#                 samples_norm = torch.from_numpy(samples_norm_np).to(samples.device).unsqueeze(-1)
#                 feat = repr_model.encode(samples_norm.float())
#             else:
#                 feat = repr_model.encode(samples_reshaped.float())
#             feat_np = feat.detach().cpu().numpy()
#             samples_repr = feat_np.reshape(N, C, -1).mean(axis=0)  # (C,D)
#             distances = self._compute_distance_matrix(samples_repr, unit)  # (C,K)
#             unit_orders.append(np.argsort(distances, axis=1))  # (C,K)
#
#         if len(unit_orders) == 1:
#             top_k_indices = unit_orders[0]
#         else:
                                                             
#             K = unit_orders[0].shape[1]
#             scores = np.zeros((C, K), dtype=np.float32)
#             for order in unit_orders:
#                 print(f"[SearchEnsemble] unit order sample: {order}")  # debug
#                 inv_rank = np.empty_like(order)
#                 inv_rank[np.arange(C)[:, None], order] = np.arange(K)[None, :]
#                 scores += inv_rank
#             top_k_indices = np.argsort(scores, axis=1)
#             print(f"[SearchEnsemble] fused order sample: {top_k_indices}")  # debug
#
                                                
#         if self.err_rate != 0:
                                                 
#             min_distances = np.min(top_k_indices.astype(np.float32), axis=1)
#             if self.err_rate < 0:
                                                  
#                 l_interval = np.mean(min_distances) - 0.5 * np.std(min_distances)
#                 r_interval = np.mean(min_distances) + 0.5 * np.std(min_distances)
#                 top_k_indices[min_distances < l_interval, :] = -1
#             else:
                                            
#                 top_k_indices[min_distances < self.err_rate, :] = -1
#
#         # (C, K) -> (K, 1, C)，priority=K
#         top_k_indices = top_k_indices.T.reshape(-1, 1, C)
#         return top_k_indices

class TSRouter_Select_Model(Baseline_Select_Model):
    'TSRouter runtime message.'

    def __init__(self, args, model_name, Model_zoo_current):
        super().__init__(args, model_name, Model_zoo_current)
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")

                                                     
        self._searcher: Optional[TSRouterModelSearcher] = None
        self._searchers: dict[tuple, TSRouterModelSearcher] = {}
        self._task_repr_cache = None
        self._task_repr_cache_path = None
        self._task_repr_caches: dict[str, dict] = {}

    @staticmethod
    def _infer_term_from_key(dataset_key: str | None) -> str | None:
        if dataset_key is None:
            return None
        s = str(dataset_key)
        for term in ("short", "medium", "long"):
            if s.endswith(f"/{term}") or s.endswith(f"_{term}") or f"/{term}/" in s or f"_{term}_" in s:
                return term
        return None

    def _infer_context_len_avg(self, test_data_input=None) -> float | None:
        if not test_data_input:
            return None
        lengths = []
        for entry in test_data_input:
            if not isinstance(entry, dict):
                continue
            target = entry.get("target")
            if target is None:
                continue
            arr = np.asarray(target)
            if arr.ndim == 1:
                lengths.append(int(arr.shape[0]))
            elif arr.ndim >= 2:
                lengths.append(int(arr.shape[-1]))
        if not lengths:
            return None
        return float(np.mean(lengths))

    def _adaptive_search_args(self, dataset=None, dataset_name: str | None = None, prediction_length: int | None = None, test_data_input=None):
        cl_auto = auto_cl_enabled(self.args)
        pl_auto = bool(getattr(self.args, "enable_pred_len_adaptive_repr", False))
        if not (cl_auto or pl_auto):
            return self.args, "default", prediction_length, None

        pred_len = prediction_length
        if pred_len is None and dataset is not None and hasattr(dataset, "prediction_length"):
            pred_len = int(dataset.prediction_length)

        term = self._infer_term_from_key(dataset_name)
        context_len_avg = self._infer_context_len_avg(test_data_input)
        if auto_cl_enabled(self.args):
            profile_cfg, term_fallback_used, effective_context_len = resolve_auto_cl_profile(
                context_len_avg,
                term,
                self.args,
            )
            args_v = copy.deepcopy(self.args)
            args_v.repr_input_dim = int(profile_cfg["repr_input_dim"])
            args_v.repr_output_dim = int(profile_cfg["repr_output_dim"])
            args_v.repr_sub_pred_len = int(profile_cfg["repr_sub_pred_len"])
            args_v.repr_source_exact_length = int(profile_cfg["repr_source_exact_length"])
            args_v.search_context_len = int(args_v.repr_input_dim)
            args_v.adaptive_profile = str(profile_cfg["adaptive_profile"])
            args_v.adaptive_context_len_avg = context_len_avg
            args_v.adaptive_context_len_effective = float(effective_context_len)
            args_v.adaptive_pred_len = pred_len
            args_v.adaptive_task_term_fallback_used = bool(term_fallback_used)
            args_v.resolved_eval_cl = str(profile_cfg["tsfm_results_dir"])
            args_v.rank_truth_cl = str(profile_cfg["tsfm_results_dir"])
            args_v.TSFM_results_dir = str(profile_cfg["tsfm_results_dir"])
            if getattr(args_v, "fix_context_len", False):
                args_v.context_len = int(args_v.repr_input_dim)
            profile = str(profile_cfg["adaptive_profile"])
            return args_v, profile, pred_len, context_len_avg

        cl_threshold = float(getattr(self.args, "context_len_adaptive_threshold", 256.0))
        pl_threshold = int(getattr(self.args, "pred_len_adaptive_threshold", 96))

        if context_len_avg is not None:
            cl_is_short = context_len_avg < cl_threshold
        elif term in {"short", "medium", "long"}:
            cl_is_short = term == "short"
        else:
            cl_is_short = True

        if pred_len is not None:
            pl_is_short = int(pred_len) <= pl_threshold
        elif term in {"short", "medium", "long"}:
            pl_is_short = term == "short"
        else:
            pl_is_short = True

        args_v = copy.deepcopy(self.args)
        cl_profile = "short" if cl_is_short else "long"
        pl_profile = "short" if pl_is_short else "long"
        if cl_auto:
            args_v.repr_input_dim = int(
                getattr(
                    self.args,
                    "short_repr_input_dim" if cl_is_short else "long_repr_input_dim",
                    96 if cl_is_short else 512,
                )
            )
        if pl_auto:
            args_v.repr_sub_pred_len = int(
                getattr(
                    self.args,
                    "short_repr_sub_pred_len" if pl_is_short else "long_repr_sub_pred_len",
                    48 if pl_is_short else 480,
                )
            )

        args_v.search_context_len = int(args_v.repr_input_dim)
        if getattr(args_v, "fix_context_len", False):
            args_v.context_len = int(args_v.repr_input_dim)
        profile = f"cl_{cl_profile}_pl_{pl_profile}"
        return args_v, profile, pred_len, context_len_avg

    def _get_searcher(self, search_args=None) -> TSRouterModelSearcher:
        if search_args is None:
            search_args = self.args
        from utils.path_utils import get_repr_save_path
        _, _, model_repr_path, _ = get_repr_save_path(search_args)
        if not os.path.exists(model_repr_path):
            raise FileNotFoundError(
                f"[AdaptiveRepr] missing repr file for search: {model_repr_path} "
                f"(repr_input_dim={getattr(search_args, 'repr_input_dim', None)}, "
                f"repr_sub_pred_len={getattr(search_args, 'repr_sub_pred_len', None)})"
            )
        key = (
            str(model_repr_path),
            int(getattr(search_args, "repr_input_dim", 0)),
            int(getattr(search_args, "repr_sub_pred_len", 0)),
            int(getattr(search_args, "repr_output_dim", 0)),
        )
        if key not in self._searchers:
            self._searchers[key] = TSRouterModelSearcher(
                args=search_args,
                zoo_model_size=self.Model_sizes,
                device=self.device,
            )
            if self._searcher is None and search_args is self.args:
                self._searcher = self._searchers[key]
        return self._searchers[key]

    def _ensure_task_repr_cache_ready(self, searcher: TSRouterModelSearcher | None = None, search_args=None):
        if searcher is None:
            searcher = self._get_searcher(search_args)
        if search_args is None:
            search_args = self.args
        from utils.path_utils import get_gift_eval_task_repr_cache_path
        cache_path = get_gift_eval_task_repr_cache_path(
            search_args, search_context_len=int(searcher.encoder_configs.input_dim)
        )
        if cache_path in self._task_repr_caches:
            self._task_repr_cache_path = cache_path
            self._task_repr_cache = self._task_repr_caches[cache_path]
            return
        self._task_repr_cache_path = cache_path
        candidates = [p for p in [cache_path] if isinstance(p, (str, os.PathLike))]
        load_path = None
        for p in candidates:
            if os.path.exists(p):
                load_path = p
                break
        if load_path is not None and os.path.exists(load_path):
            try:
                with open(load_path, "rb") as f:
                    self._task_repr_cache = pickle.load(f)
                if not isinstance(self._task_repr_cache, dict):
                    self._task_repr_cache = {}
            except Exception:
                self._task_repr_cache = {}
        else:
            self._task_repr_cache = {}
        self._task_repr_caches[cache_path] = self._task_repr_cache

    @staticmethod
    def _normalize_cache_key(dataset_key: str) -> str:
        'TSRouter runtime message.'
        s = str(dataset_key)
        if "/" in s:
            return s
        parts = s.rsplit("_", 2)
        if len(parts) == 3:
            return f"{parts[0]}/{parts[1]}/{parts[2]}"
        return s

    def get_cached_task_repr(self, ds_config: str, searcher: TSRouterModelSearcher | None = None, search_args=None):
        if search_args is None:
            search_args, _, _, _ = self._adaptive_search_args(dataset_name=ds_config)
        self._ensure_task_repr_cache_ready(searcher=searcher, search_args=search_args)
        return self._task_repr_cache.get(self._normalize_cache_key(ds_config))

    def get_cached_task_sample_seconds(
        self,
        ds_config: str,
        searcher: TSRouterModelSearcher | None = None,
        search_args=None,
    ) -> tuple[float | None, str]:
        if search_args is None:
            search_args, _, _, _ = self._adaptive_search_args(dataset_name=ds_config)
        self._ensure_task_repr_cache_ready(searcher=searcher, search_args=search_args)
        key = self._normalize_cache_key(ds_config)
        return _read_task_sample_seconds(self._task_repr_cache_path, key)

    def _task_repr_cache_meta_path(self) -> str:
        if not self._task_repr_cache_path:
            return ""
        return f"{self._task_repr_cache_path}.meta.json"

    def _write_task_repr_cache_metadata(self, key: str, metadata: dict) -> None:
        meta_path = self._task_repr_cache_meta_path()
        if not meta_path or not metadata:
            return
        payload = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    payload = loaded
            except Exception:
                payload = {}
        payload[key] = metadata
        tmp_path = f"{meta_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, meta_path)

    def _upsert_task_repr_cache(
        self,
        ds_config: str,
        search_input: np.ndarray,
        searcher: TSRouterModelSearcher | None = None,
        search_args=None,
        task_sampling_ms: float | None = None,
    ):
        if search_args is None:
            search_args = self.args
        self._ensure_task_repr_cache_ready(searcher=searcher, search_args=search_args)
        key = self._normalize_cache_key(ds_config)
        arr = np.asarray(search_input, dtype=np.float32)
        self._validate_search_input_shape(arr, source=f"save:{key}", searcher=searcher)
        print(f"TSRouter runtime message: {key}, shape={arr.shape}")
        self._task_repr_cache[key] = arr
        with file_lock(self._task_repr_cache_path + ".lock"):
            current = {}
            if os.path.exists(self._task_repr_cache_path):
                try:
                    with open(self._task_repr_cache_path, "rb") as f:
                        loaded = pickle.load(f)
                    if isinstance(loaded, dict):
                        current = loaded
                except Exception:
                    current = {}
            current[key] = self._task_repr_cache[key]
            with open(self._task_repr_cache_path, "wb") as f:
                pickle.dump(current, f)
            self._task_repr_cache = current
            self._task_repr_caches[self._task_repr_cache_path] = current
            meta = dict(getattr(self, "_last_task_repr_sample_meta", {}) or {})
            if meta:
                meta.update(
                    {
                        "dataset": key,
                        "cache_path": self._task_repr_cache_path,
                        "cache_shape": list(arr.shape),
                        "repr_input_dim": int(arr.shape[1]),
                        "sample_repr_num": int(getattr(search_args, "sample_repr_num", 0)),
                        "task_sample_strategy": str(getattr(search_args, "task_sample_strategy", "latest_random")),
                        "task_window_sample_strategy": str(getattr(search_args, "task_window_sample_strategy", "legacy")),
                        "sample_repr_ratio": float(getattr(search_args, "sample_repr_ratio", 0.0) or 0.0),
                        "search_seed": int(getattr(search_args, "search_seed", 0) or 0),
                        "updated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    }
                )
                if task_sampling_ms is not None:
                    meta["task_sampling_ms"] = float(task_sampling_ms)
                    meta["task_sampling_timing_valid"] = True
                self._write_task_repr_cache_metadata(key, meta)
            if task_sampling_ms is not None:
                _upsert_task_sample_timing_row(
                    self._task_repr_cache_path,
                    key,
                    search_args,
                    float(task_sampling_ms),
                    cache_shape=arr.shape,
                )

    def _stable_model_ids(self) -> list[str]:
        return [
            str(details.get("abbreviation", ""))
            for family in self.Model_sizes.values()
            for details in family.values()
            if details.get("abbreviation", "")
        ]

    @staticmethod
    def _model_order_to_abbrs(model_order, searcher: TSRouterModelSearcher | None = None) -> list[str]:
        out: list[str] = []
        if model_order is None:
            return out
        for item in list(model_order):
            try:
                idx = int(item)
                if searcher is not None and 0 <= idx < len(searcher.sorted_models):
                    out.append(str(searcher.sorted_models[idx].get("abbreviation", idx)))
                else:
                    out.append(str(idx))
            except Exception:
                out.append(str(item))
        return out

    def _build_vldb_route_latency_row(
        self,
        dataset_name: str | None,
        searcher: TSRouterModelSearcher,
        model_order,
        cache_hit: bool,
        cache_lookup_ms: float,
        index_load_ms: float,
        task_sampling_ms: float,
        task_embedding_ms: float,
        index_lookup_ms: float,
        rank_ms: float,
        sample_to_route_ms: float | None = None,
        selected_forecast_ms: float | None = None,
    ) -> dict:
        selected_forecast_ms_value = "" if selected_forecast_ms is None else f"{float(selected_forecast_ms):.3f}"
        route_overhead_ms = cache_lookup_ms + index_load_ms + task_sampling_ms + task_embedding_ms + index_lookup_ms + rank_ms
        if sample_to_route_ms is None:
            sample_to_route_ms = task_embedding_ms + index_lookup_ms + rank_ms
        route_final_ms = float(task_sampling_ms) + float(sample_to_route_ms)
        end_to_end_ms = route_overhead_ms + (0.0 if selected_forecast_ms is None else float(selected_forecast_ms))
        skip_saved = bool(getattr(self.args, "skip_saved", False))
        timing_valid = (not skip_saved) and (not cache_hit)
        notes = []
        if skip_saved:
            notes.append("Step4 skip_saved is enabled.")
        if bool(getattr(self.args, "vldb_fast_sample", False)):
            notes.append("VLDB_FAST_SMAPLE is enabled; task sampling is reused from cache.")
        if bool(getattr(self.args, "vldb_fast_forward", True)):
            notes.append("VLDB_FAST_FORWARD is enabled; selected_forecast_ms is cached prediction load/compose time.")
        if bool(getattr(self.args, "vldb_skip_evaluate", False)):
            notes.append("VLDB_FAST_EVAL is enabled; evaluate_forecasts will be replaced by saved metrics.")
        if cache_hit:
            notes.append("task repr cache was reused; sample_ms is restored from the task-sample timing CSV.")
        if not notes:
            notes.append("fresh task sampling was used for this dataset-level route timing.")
        task_sampling_valid = np.isfinite(float(task_sampling_ms)) and float(task_sampling_ms) >= 0
        selected_forecast_valid = not bool(getattr(self.args, "vldb_fast_forward", True))
        evaluate_valid = not bool(getattr(self.args, "vldb_skip_evaluate", False))
        return {
            "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "route_id": str(getattr(self.args, "vldb_route_id", "") or ""),
            "stage": str(getattr(self.args, "vldb_route_stage", "") or ""),
            "method": str(getattr(self.args, "models", self.model_name)),
            "profile_id": str(getattr(self.args, "vldb_route_profile_id", "") or ""),
            "route_family_mode": searcher.route_family_mode,
            "dataset": str(dataset_name or ""),
            "status": "dataset_executed",
            "zoo_size": str(len(searcher.sorted_models)),
            "stable_model_ids": " ".join(self._stable_model_ids()),
            "selected_model_order": " ".join(self._model_order_to_abbrs(model_order, searcher=searcher)),
            "step4_skip_saved": _bool_text(skip_saved),
            "cache_mode": "task_repr_cache" if cache_hit else "fresh_task_repr",
            "cache_hit": _bool_text(cache_hit),
            "timing_level": "selector_dataset_internal",
            "timing_valid": _bool_text(timing_valid),
            "route_command_s": "",
            "cache_lookup_ms": f"{float(cache_lookup_ms):.3f}",
            "index_load_ms": f"{float(index_load_ms):.3f}",
            "sample_ms": f"{float(task_sampling_ms):.3f}",
            "sample_to_route_ms": f"{float(sample_to_route_ms):.3f}",
            "route_final_ms": f"{float(route_final_ms):.3f}",
            "task_sampling_ms": f"{float(task_sampling_ms):.3f}",
            "task_embedding_ms": f"{float(task_embedding_ms):.3f}",
            "index_lookup_ms": f"{float(index_lookup_ms):.3f}",
            "rank_ms": f"{float(rank_ms):.3f}",
            "route_overhead_ms": f"{float(route_overhead_ms):.3f}",
            "selected_forecast_ms": selected_forecast_ms_value,
            "evaluate_ms": "",
            "metric_read_ms": "",
            "end_to_end_ms": f"{float(end_to_end_ms):.3f}",
            "fast_eval_enabled": _bool_text(bool(getattr(self.args, "GE_fast_eval", False)) or bool(getattr(self.args, "vldb_skip_evaluate", False))),
            "evaluation_mode": "saved_metric_fast_eval" if (bool(getattr(self.args, "GE_fast_eval", False)) or bool(getattr(self.args, "vldb_skip_evaluate", False))) else "gluonts_evaluate",
            "vldb_fast_sample": _bool_text(bool(getattr(self.args, "vldb_fast_sample", False))),
            "vldb_fast_forward": _bool_text(bool(getattr(self.args, "vldb_fast_forward", True))),
            "vldb_fast_eval": _bool_text(bool(getattr(self.args, "vldb_skip_evaluate", False))),
            "task_sampling_timing_valid": _bool_text(task_sampling_valid),
            "selected_forecast_timing_valid": _bool_text(selected_forecast_valid),
            "evaluate_timing_valid": _bool_text(evaluate_valid),
            "forward_mode": "cached_prediction" if bool(getattr(self.args, "vldb_fast_forward", True)) else "saved_runtime_proxy",
            "timing_note": " ".join(notes),
        }



                                                   
    def get_predictor(self, dataset, batch_size):
        self.args.test_pred_len = dataset.prediction_length
        return Zoo_Select_Predictor(
            args=self.args,
            Model_sizes=self.Model_sizes,
            prediction_length=dataset.prediction_length,
            channels=dataset.target_dim,
            windows=dataset.windows,
            model_cl_name=self.model_cl_name,
            select_strategy=self._get_select_strategy(dataset),
        )

    # ----------------------------
                                       
    # ----------------------------

    def _fuse_ranks_across_task_samples(
        self,
        rank_list: list[np.ndarray],
        num_models: int,
        candidate_model_ids: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fuse per-sample rank orders and return both rank order and Borda scores."""
        if len(rank_list) == 0:
            raise ValueError("rank_list is empty")
        if isinstance(num_models, dict):
            num_models = sum(len(v) for v in num_models.values())
        num_models = int(num_models)
        K, C = rank_list[0].shape
        if candidate_model_ids is None:
            candidates = list(range(num_models))
        else:
            candidates = list(dict.fromkeys(int(value) for value in candidate_model_ids))
            invalid = [value for value in candidates if value < 0 or value >= num_models]
            if invalid:
                raise ValueError(f"candidate_model_ids outside model range: {invalid}")
        if not candidates:
            raise ValueError("candidate_model_ids is empty")
        out = np.zeros((len(candidates), C), dtype=np.int64)
        score_matrix = np.zeros((num_models, C), dtype=np.float32)
        score_template = rank_position_scores(K, decay_coef=rank_decay_coef(self.args))

        for c in range(C):
            scores = score_matrix[:, c]
            for rk in rank_list:
                order = rk[:, c]
                for p, m in enumerate(order):
                    m = int(m)
                    if 0 <= m < num_models:
                        scores[m] += float(score_template[p])
            out[:, c] = np.asarray(
                sorted(candidates, key=lambda m: (-scores[m], m)),
                dtype=np.int64,
            )
        return out, score_matrix

    def _vote_ranks_across_task_samples(self, rank_list: list[np.ndarray], num_models: int) -> np.ndarray:
        """Fuse per-sample rank orders with a shared positional decay template."""
        return self._fuse_ranks_across_task_samples(rank_list, num_models)[0]

    @staticmethod
    def _top3_instability_from_rank_list(rank_list: list[np.ndarray]) -> np.ndarray:
        """Mean pairwise Jaccard distance of Top3 sets for each channel."""
        if len(rank_list) < 2:
            if not rank_list:
                return np.zeros((0,), dtype=np.float32)
            return np.zeros((rank_list[0].shape[1],), dtype=np.float32)
        c_num = int(rank_list[0].shape[1])
        out = np.zeros((c_num,), dtype=np.float32)
        pair_count = 0
        for i in range(len(rank_list)):
            for j in range(i + 1, len(rank_list)):
                pair_count += 1
                a = rank_list[i]
                b = rank_list[j]
                for c in range(c_num):
                    set_a = set(int(x) for x in a[: min(3, a.shape[0]), c] if int(x) >= 0)
                    set_b = set(int(x) for x in b[: min(3, b.shape[0]), c] if int(x) >= 0)
                    union = set_a | set_b
                    dist = 0.0 if not union else 1.0 - (len(set_a & set_b) / len(union))
                    out[c] += float(dist)
        if pair_count > 0:
            out /= float(pair_count)
        return out

    @staticmethod
    def _rank1_consensus_from_rank_list(rank_list: list[np.ndarray]) -> dict[str, np.ndarray]:
        """
        Measure whether sampled windows agree on the Rank1 model.

        The instability score is:
        second-highest Rank1 vote count * total valid vote count / highest Rank1 vote count ** 2.
        """
        if not rank_list:
            empty_f = np.zeros((0,), dtype=np.float32)
            empty_i = np.zeros((0,), dtype=np.int64)
            return {
                "rank1_instability": empty_f,
                "rank1_count_ratio": empty_f,
                "rank1_count_score": empty_f,
                "rank1_top_count": empty_i,
                "rank1_second_count": empty_i,
                "rank1_top_model": empty_i,
                "rank1_sample_count": empty_i,
            }

        c_num = int(rank_list[0].shape[1])
        count_score = np.zeros((c_num,), dtype=np.float32)
        top_count = np.zeros((c_num,), dtype=np.int64)
        second_count = np.zeros((c_num,), dtype=np.int64)
        top_model = np.full((c_num,), -1, dtype=np.int64)
        sample_count = np.zeros((c_num,), dtype=np.int64)

        for c in range(c_num):
            winners: list[int] = []
            for rk in rank_list:
                arr = np.asarray(rk)
                if arr.ndim != 2 or arr.shape[0] == 0 or c >= arr.shape[1]:
                    continue
                mid = int(arr[0, c])
                if mid >= 0:
                    winners.append(mid)

            n = len(winners)
            sample_count[c] = n
            if n == 0:
                count_score[c] = 1.0
                continue

            models, counts = np.unique(np.asarray(winners, dtype=np.int64), return_counts=True)
            count_order = np.argsort(-counts, kind="stable")
            best_pos = int(count_order[0])
            best_count = int(counts[best_pos])
            runner_up_count = int(counts[int(count_order[1])]) if count_order.size > 1 else 0
            top_count[c] = best_count
            second_count[c] = runner_up_count
            top_model[c] = int(models[best_pos])
            count_score[c] = (
                float(runner_up_count * n) / float(best_count * best_count)
                if best_count > 0
                else 1.0
            )

        return {
            "rank1_instability": count_score,
            "rank1_count_ratio": count_score,
            "rank1_count_score": count_score,
            "rank1_top_count": top_count,
            "rank1_second_count": second_count,
            "rank1_top_model": top_model,
            "rank1_sample_count": sample_count,
        }

    def _rank_consistency_instability_from_rank_list(
        self,
        rank_list: list[np.ndarray],
    ) -> dict[str, np.ndarray]:
        rank1 = self._rank1_consensus_from_rank_list(rank_list)
        top3 = self._top3_instability_from_rank_list(rank_list)
        rank1_count_score = np.asarray(rank1["rank1_count_score"], dtype=np.float32)
        if top3.size < rank1_count_score.size:
            top3 = np.pad(top3, (0, rank1_count_score.size - top3.size), constant_values=0.0)
        top3 = np.asarray(top3[: rank1_count_score.size], dtype=np.float32)
        return {
            "rank1_instability": rank1_count_score,
            "rank1_count_ratio": rank1_count_score,
            "rank1_count_score": rank1_count_score,
            "rank1_top_count": rank1["rank1_top_count"],
            "rank1_second_count": rank1["rank1_second_count"],
            "rank1_top_model": rank1["rank1_top_model"],
            "rank1_sample_count": rank1["rank1_sample_count"],
            "rank_top3_instability": top3.astype(np.float32),
            "rank_consistency_instability": rank1_count_score,
        }

    @staticmethod
    def _repeat_to_len(indices: list[int], target_len: int) -> list[int]:
        if target_len <= 0:
            return []
        if not indices:
            return []
        out: list[int] = []
        while len(out) < target_len:
            out.extend(indices)
        return out[:target_len]

    def _plan_task_sample_indices(self, test_dataset, sample_num: int, rng: np.random.RandomState) -> list[int] | None:
        strategy = str(getattr(self.args, "task_window_sample_strategy", "legacy")).lower()
        ratio = float(getattr(self.args, "sample_repr_ratio", 0.0) or 0.0)
        if strategy == "legacy" and ratio <= 0:
            return None

        entries = list(test_dataset)
        n_entries = len(entries)
        if n_entries <= 0:
            raise ValueError("test_dataset is empty")
        target_n = max(1, int(sample_num or 1))
        if ratio > 0:
            target_n = max(target_n, int(math.ceil(n_entries * ratio)))

        if strategy == "legacy":
            strategy = "even"
        if strategy == "first":
            base = list(range(min(n_entries, target_n)))
            indices = self._repeat_to_len(base, target_n) if len(base) < target_n else base
        elif strategy == "last":
            start = max(0, n_entries - min(n_entries, target_n))
            base = list(range(start, n_entries))
            indices = self._repeat_to_len(base, target_n) if len(base) < target_n else base[-target_n:]
        elif strategy == "random":
            replace = target_n > n_entries
            indices = rng.choice(n_entries, size=target_n, replace=replace).astype(int).tolist()
        elif strategy == "even":
            if target_n == 1:
                indices = [0]
            elif target_n <= n_entries:
                indices = np.linspace(0, n_entries - 1, target_n).round().astype(int).tolist()
            else:
                base = list(range(n_entries))
                indices = self._repeat_to_len(base, target_n)
        else:
            raise ValueError(f"Unsupported task_window_sample_strategy={strategy}")

        unique_n = len(set(indices))
        dup_n = len(indices) - unique_n
        # print(
        #     f"[TS_Router] task_window_sample_strategy={strategy}, sample_repr_ratio={ratio:g}, "
        #     f"entries={n_entries}, effective_samples={len(indices)}, unique={unique_n}, duplicates={dup_n}",
        #     flush=True,
        # )
        return [int(x) for x in indices]

    def _build_search_input_array(self, test_dataset, searcher: TSRouterModelSearcher | None = None):

        sample_num = self.args.sample_repr_num               
        print(
            "[TS_Router] sample_num:",
            sample_num,
            "sample_repr_ratio:",
            getattr(self.args, "sample_repr_ratio", 0.0),
            "task_window_sample_strategy:",
            getattr(self.args, "task_window_sample_strategy", "legacy"),
            "search_seed:",
            self.args.search_seed,
        )

        if searcher is None:
            searcher = self._get_searcher()

                                                          
        search_context_len = searcher.encoder_configs.input_dim
                                                                                                                  
        # search_context_len = searcher.encoder_input_dim

        base_seed = int(self.args.search_seed if self.args.search_seed is not None else 0)
        plan_rng = np.random.RandomState(base_seed)
        planned_indices = self._plan_task_sample_indices(test_dataset, sample_num, plan_rng)
        effective_sample_num = len(planned_indices) if planned_indices is not None else int(sample_num)

        search_input_list = []
        sample_meta = []
        for k in range(effective_sample_num):
            rng = np.random.RandomState(base_seed + k)
            preferred_idx = None if planned_indices is None else planned_indices[k]
            x = self._get_search_input(
                test_dataset,
                search_context_len,
                rng=rng,
                sample_index=k,
                sample_num=effective_sample_num,
                preferred_entry_idx=preferred_idx,
            )
            x = self._apply_search_input_scale_protocol(x)
            search_input_list.append(x)
            sample_meta.append(dict(getattr(self, "_last_search_input_source", {}) or {}))
        out = np.concatenate(search_input_list, axis=0)  # (K,T,C)
        self._validate_search_input_shape(out, source="online_sample", searcher=searcher)
        self._last_task_repr_sample_meta = {
            "entry_indices": [
                int(m["entry_idx"]) for m in sample_meta
                if m.get("entry_idx") is not None
            ],
            "segment_start_indices": [
                None if m.get("segment_start_idx") is None else int(m.get("segment_start_idx"))
                for m in sample_meta
            ],
            "sample_sources": sample_meta,
        }
        return out

    def _validate_search_input_shape(self, raw: np.ndarray, source: str = "unknown", searcher: TSRouterModelSearcher | None = None) -> None:
        'TSRouter runtime message.'
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"[{source}] search_input ndim must be 3, got {arr.ndim}, shape={arr.shape}")
        sample_num = int(getattr(self.args, "sample_repr_num", 1))
        if searcher is None:
            searcher = self._get_searcher()
        target_t = int(searcher.encoder_configs.input_dim)
        k, t, c = arr.shape
        ratio = float(getattr(self.args, "sample_repr_ratio", 0.0) or 0.0)
        strategy = str(getattr(self.args, "task_window_sample_strategy", "legacy")).lower()
        if ratio <= 0 and strategy == "legacy" and k != sample_num:
            raise ValueError(f"[{source}] invalid K={k}, expected sample_repr_num={sample_num}, shape={arr.shape}")
        if (ratio > 0 or strategy != "legacy") and k < sample_num:
            raise ValueError(f"[{source}] invalid K={k}, expected at least sample_repr_num={sample_num}, shape={arr.shape}")
        if t != target_t:
            raise ValueError(f"[{source}] invalid T={t}, expected encoder_input_dim={target_t}, shape={arr.shape}")
        if c <= 0:
            raise ValueError(f"[{source}] invalid C={c}, shape={arr.shape}")

    def _apply_search_input_scale_protocol(self, search_input: np.ndarray) -> np.ndarray:
        protocol = get_repr_scale_protocol(self.args)
        arr = np.asarray(search_input, dtype=np.float32)
        if protocol == "raw":
            return arr
        out = arr.copy()
        mean = np.mean(out, axis=1, keepdims=True)
        std = np.std(out, axis=1, keepdims=True)
        out = (out - mean) / np.where(std < 1e-6, 1.0, std)
        out = np.where(std < 1e-6, 0.0, out)
        return out.astype(np.float32)

    def _get_model_order_list(self, test_dataset_or_search_input, searcher: TSRouterModelSearcher | None = None):
        if searcher is None:
            searcher = self._get_searcher()
        task_embedding_ms = 0.0
        index_lookup_ms = 0.0
        rank_ms = 0.0
        if isinstance(test_dataset_or_search_input, np.ndarray):
            search_input = test_dataset_or_search_input.astype(np.float32, copy=False)
            self._validate_search_input_shape(search_input, source="cached", searcher=searcher)
        else:
            search_input = self._normalize_search_input_array(test_dataset_or_search_input)

                           
        task_sample_version = int(getattr(self.args, "task_sample_version", 1))
        self._last_task_rank_instability = None
        self._last_task_top3_instability = None
        self._last_task_rank1_instability = None
        self._last_task_rank1_count_ratio = None
        self._last_task_rank1_count_score = None
        self._last_task_rank1_top_count = None
        self._last_task_rank1_second_count = None
        self._last_task_rank1_top_model = None
        self._last_task_rank1_sample_count = None
        self._last_task_sample_rankings = None
        self._last_route_candidate_model_ids = list(searcher.route_raw_candidate_model_ids)
        if task_sample_version == 1:
            print('[search-v1]: search_input shape:', search_input.shape)
            search_input = torch.tensor(search_input, dtype=torch.float32).to(self.device)
            selected_model_list = searcher.fit(search_input, use_norm=getattr(self.args, "use_norm", False))
            self._last_model_search_timing = {
                "task_embedding_ms": float(getattr(searcher, "last_task_embedding_ms", 0.0)),
                "index_lookup_ms": float(getattr(searcher, "last_index_lookup_ms", 0.0)),
                "rank_ms": float(getattr(searcher, "last_rank_ms", 0.0)),
            }
            return selected_model_list

                                        
        per_sample_rankings = []
        if getattr(searcher, "is_simplets_mode", False):
            batch_rankings = searcher.fit_simplets_samples(search_input)  # (N,K,C)
            task_embedding_ms = float(getattr(searcher, "last_task_embedding_ms", 0.0))
            index_lookup_ms = float(getattr(searcher, "last_index_lookup_ms", 0.0))
            rank_ms = float(getattr(searcher, "last_rank_ms", 0.0))
            print("[search-v2]:per_sample_rankings：", end=" ")
            for rk in batch_rankings:
                per_sample_rankings.append(rk)
                print(rk[0, :], end=" ")
        elif getattr(searcher, "is_autoforecast_mode", False):
            batch_rankings = searcher.fit_autoforecast_samples(search_input)  # (N,K,C)
            task_embedding_ms = float(getattr(searcher, "last_task_embedding_ms", 0.0))
            index_lookup_ms = float(getattr(searcher, "last_index_lookup_ms", 0.0))
            rank_ms = float(getattr(searcher, "last_rank_ms", 0.0))
            print("[search-v2]:per_sample_rankings：", end=" ")
            for rk in batch_rankings:
                per_sample_rankings.append(rk)
                print(rk[0, :], end=" ")
        else:
            print("[search-v2]:per_sample_rankings：", end=" ")
            for x in search_input:
                s = torch.tensor(x[None, ...], dtype=torch.float32).to(self.device)  # (1,T,C)
                rk = searcher.fit(s, use_norm=getattr(self.args, "use_norm", False))  # (K,1,C)
                task_embedding_ms += float(getattr(searcher, "last_task_embedding_ms", 0.0))
                index_lookup_ms += float(getattr(searcher, "last_index_lookup_ms", 0.0))
                rank_ms += float(getattr(searcher, "last_rank_ms", 0.0))
                per_sample_rankings.append(rk[:, 0, :])  # (K,C)
                print(rk[0, 0, :],end=" ")

        t_rank0 = time.perf_counter()
        fused_rank, _ = self._fuse_ranks_across_task_samples(
            per_sample_rankings,
            num_models=self.Model_sizes,
            candidate_model_ids=searcher.route_raw_candidate_model_ids,
        )
        rank_consistency = self._rank_consistency_instability_from_rank_list(per_sample_rankings)
        self._last_task_rank1_instability = rank_consistency["rank1_instability"]
        self._last_task_rank1_count_ratio = rank_consistency["rank1_count_ratio"]
        self._last_task_rank1_count_score = rank_consistency["rank1_count_score"]
        self._last_task_rank1_top_count = rank_consistency["rank1_top_count"]
        self._last_task_rank1_second_count = rank_consistency["rank1_second_count"]
        self._last_task_rank1_top_model = rank_consistency["rank1_top_model"]
        self._last_task_rank1_sample_count = rank_consistency["rank1_sample_count"]
        self._last_task_top3_instability = rank_consistency["rank_top3_instability"]
        self._last_task_rank_instability = rank_consistency["rank_consistency_instability"]
        rank_ms += (time.perf_counter() - t_rank0) * 1000.0
        self._last_task_sample_rankings = np.stack(per_sample_rankings, axis=0).astype(np.int64, copy=False)
        if self._last_task_rank_instability.size > 0:
            rank1_counts = np.asarray(self._last_task_rank1_top_count, dtype=np.int64)
            rank1_second_counts = np.asarray(self._last_task_rank1_second_count, dtype=np.int64)
            rank1_sample_counts = np.asarray(self._last_task_rank1_sample_count, dtype=np.int64)
            score_text = "[" + ", ".join(f"{float(value):.2f}" for value in self._last_task_rank_instability) + "]"
            top3_text = "[" + ", ".join(f"{float(value):.2f}" for value in self._last_task_top3_instability) + "]"
            count_by_channel = "[" + ", ".join(
                f"{int(top)}/{int(second)}/{int(total)}"
                for top, second, total in zip(
                    rank1_counts.tolist(),
                    rank1_second_counts.tolist(),
                    rank1_sample_counts.tolist(),
                )
            ) + "]"
            print(
                "[TS_Router] rank_consistency_by_channel:",
                f"scores={score_text}",
                f"| rank1_top_second_total_count={count_by_channel}",
                f"| top3_jaccard_diag={top3_text}",
                f"| mean={float(np.mean(self._last_task_rank_instability)):.2f}",
                f"max={float(np.max(self._last_task_rank_instability)):.2f}",
            )
        print("[TS_Router-fallback before] window_fused_rank_Top1_before_fallback:", fused_rank[0])
        self._last_model_search_timing = {
            "task_embedding_ms": task_embedding_ms,
            "index_lookup_ms": index_lookup_ms,
            "rank_ms": rank_ms,
        }
        return fused_rank[:, None, :]  # (K,1,C)

    def _channel_fuse_positions(self, selected_model_list_2d: np.ndarray) -> np.ndarray:
        c_num = int(selected_model_list_2d.shape[1])
        raw = str(getattr(self.args, "task_channel_fuse_limit", "all") or "all").strip().lower()
        if raw in {"", "all", "none"}:
            return np.arange(c_num, dtype=np.int64)
        try:
            limit = int(raw)
        except ValueError as exc:
            raise ValueError(f"task_channel_fuse_limit must be all or a positive integer, got {raw!r}") from exc
        if limit <= 0:
            raise ValueError(f"task_channel_fuse_limit must be positive when not all, got {raw!r}")
        limit = min(limit, c_num)
        positions = np.arange(limit, dtype=np.int64)
        print(f"[TS_Router] channel fusion cap: use {limit}/{c_num} channels for task-level rank fusion", flush=True)
        return positions

    def _restrict_model_order_with_channel_limit(self, selected_model_list_2d: np.ndarray) -> tuple[np.ndarray, list[int], np.ndarray]:
        selected_model_list_2d = np.asarray(selected_model_list_2d)
        fuse_positions = self._channel_fuse_positions(selected_model_list_2d)
        fuse_rank = selected_model_list_2d[:, fuse_positions]
        selected_limited, model_order = restrict_top_k_models(
            fuse_rank,
            k=self.args.restrict_top_model_num,
            rank_decay_coef=float(getattr(self.args, "rank_decay_coef", 1.0)),
            allowed_model_ids=getattr(self, "_last_route_candidate_model_ids", None),
        )
        selected_full = np.full(selected_model_list_2d.shape[1], int(model_order[0]) if model_order else -1, dtype=int)
        selected_full[fuse_positions] = selected_limited
        return selected_full, model_order, fuse_positions

    def _apply_route_family_mode_to_task_outputs(
        self,
        selected_model_list_2d: np.ndarray,
        selected_models_per_channel: np.ndarray,
        model_order: list[int],
        searcher: TSRouterModelSearcher,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        route_family_mode = normalize_route_family_mode(
            getattr(searcher, "route_family_mode", "default")
        )
        if route_family_mode == "default":
            return selected_model_list_2d, selected_models_per_channel, model_order

        family_rank = searcher.merge_route_family_task_rank(selected_model_list_2d)
        selected_family = remap_route_family_rank_ids(
            np.asarray(selected_models_per_channel, dtype=np.int64),
            searcher.route_family_target_by_model_id,
        )
        if model_order:
            family_order = searcher.merge_route_family_task_rank(
                np.asarray(model_order, dtype=np.int64)[:, None]
            )[:, 0].astype(int).tolist()
        else:
            family_order = []
        return family_rank, selected_family.astype(int, copy=False), family_order

    @staticmethod
    def _fallback_profile_label(args) -> str:
        if args is None or not auto_cl_enabled(args):
            return ""
        profile = str(getattr(args, "adaptive_profile", "") or "")
        resolved_cl = str(
            getattr(args, "resolved_eval_cl", "")
            or getattr(args, "rank_truth_cl", "")
            or getattr(args, "TSFM_results_dir", "")
            or getattr(args, "model_cl_name", "")
            or ""
        )
        parts = []
        if profile:
            parts.append(f"profile={profile}")
        if resolved_cl:
            parts.append(f"cl={resolved_cl}")
        return "[" + ",".join(parts) + "]" if parts else ""

    @staticmethod
    def _ensure_fallback_searcher_matches_task(
        searcher: TSRouterModelSearcher,
        search_args=None,
    ) -> None:
        if search_args is None or not auto_cl_enabled(search_args):
            return
        searcher_args = getattr(searcher, "args", None)
        expected_profile = str(getattr(search_args, "adaptive_profile", "") or "")
        actual_profile = str(getattr(searcher_args, "adaptive_profile", "") or "")
        expected_cl = str(
            getattr(search_args, "resolved_eval_cl", "")
            or getattr(search_args, "rank_truth_cl", "")
            or ""
        )
        actual_cl = str(
            getattr(searcher_args, "resolved_eval_cl", "")
            or getattr(searcher_args, "rank_truth_cl", "")
            or ""
        )
        mismatches = []
        if expected_profile and actual_profile and expected_profile != actual_profile:
            mismatches.append(f"profile expected={expected_profile!r} actual={actual_profile!r}")
        if expected_cl and actual_cl and expected_cl != actual_cl:
            mismatches.append(f"cl expected={expected_cl!r} actual={actual_cl!r}")
        if mismatches:
            raise ValueError(
                "[TS_Router-fallback] searcher/profile mismatch; "
                + "; ".join(mismatches)
            )

    @staticmethod
    def _weight_prior_order_from_searcher(
        searcher: TSRouterModelSearcher,
    ) -> tuple[list[int], str, list[tuple[str, float]]]:
        weighted_models: list[tuple[int, str, float]] = []
        for idx, model in enumerate(searcher.sorted_models):
            abbr = str(model.get("abbreviation", idx))
            weight = float(searcher.model_weights.get(abbr, 1.0))
            if not np.isfinite(weight):
                weight = 0.0
            weighted_models.append((idx, abbr, weight))
        if not weighted_models:
            return [], "model_weight_prior_empty", []

        route_family_mode = normalize_route_family_mode(
            getattr(searcher, "route_family_mode", "default")
        )
        if route_family_mode != "default":
            raw_targets = getattr(searcher, "route_family_target_by_model_id", None)
            targets = np.asarray(raw_targets, dtype=np.int64).reshape(-1)
            if targets.size != len(weighted_models):
                raise ValueError(
                    "route family fallback map size mismatch: "
                    f"targets={targets.size}, models={len(weighted_models)}"
                )
            weight_by_target: dict[int, float] = defaultdict(float)
            for idx, _abbr, weight in weighted_models:
                weight_by_target[int(targets[idx])] += float(weight)
            weighted_models = [
                (
                    target_id,
                    str(searcher.sorted_models[target_id].get("abbreviation", target_id)),
                    float(weight_by_target[target_id]),
                )
                for target_id in dict.fromkeys(targets.tolist())
            ]
        rounded = {round(weight, 12) for _, _, weight in weighted_models}
        source = "model_weight_prior_all_equal" if len(rounded) <= 1 else "model_weight_prior"
        if route_family_mode != "default":
            source += f"_family_{route_family_mode}"
        source += TSRouter_Select_Model._fallback_profile_label(getattr(searcher, "args", None))
        ordered = sorted(weighted_models, key=lambda item: (-item[2], item[0]))
        order = [idx for idx, _, _ in ordered]
        named_weights = [(abbr, weight) for _, abbr, weight in ordered]
        return order, source, named_weights

    def _apply_rank_consistency_fallback(
        self,
        selected_model_list_2d: np.ndarray,
        searcher: TSRouterModelSearcher,
        search_args=None,
    ) -> tuple[np.ndarray, np.ndarray | None, list[int] | None, str]:
        selected_model_list_2d = np.asarray(selected_model_list_2d, dtype=np.int64)
        rank_consistency_instability = getattr(self, "_last_task_rank_instability", None)
        raw_threshold = getattr(self.args, "task_rank_top3_instability_threshold", -1.0)
        threshold = float(raw_threshold) if raw_threshold is not None else -1.0
        if abs(threshold) < 1e-12:
            threshold = 0.0
        force_all_fallback = threshold == 0.0
        fallback_mask = None
        fallback_order = None
        fallback_source = ""

        if threshold < 0 or selected_model_list_2d.size == 0:
            return selected_model_list_2d, fallback_mask, fallback_order, fallback_source

        channel_count = int(selected_model_list_2d.shape[1])
        rank_consistency_values = None
        if rank_consistency_instability is not None:
            rank_consistency_values = np.asarray(rank_consistency_instability, dtype=np.float32)

        if force_all_fallback:
            fallback_mask = np.ones(channel_count, dtype=bool)
        elif rank_consistency_values is None:
            return selected_model_list_2d, fallback_mask, fallback_order, fallback_source
        elif rank_consistency_values.size != channel_count:
            print(
                f"⚠️ rank-consistency fallback skipped: instability channels={rank_consistency_values.size}, "
                f"rank channels={channel_count}",
                flush=True,
            )
            return selected_model_list_2d, fallback_mask, fallback_order, fallback_source
        else:
            fallback_mask = rank_consistency_values > threshold
        if not fallback_mask.any():
            return selected_model_list_2d, fallback_mask, fallback_order, fallback_source

        self._ensure_fallback_searcher_matches_task(searcher, search_args)
        fallback_order, fallback_source, named_weights = self._weight_prior_order_from_searcher(searcher)
        fallback_fill_order = list(fallback_order)
        if len(fallback_fill_order) < selected_model_list_2d.shape[0]:
            route_family_mode = normalize_route_family_mode(
                getattr(searcher, "route_family_mode", "default")
            )
            if route_family_mode != "default":
                for model_id in range(len(getattr(searcher, "sorted_models", []))):
                    model_id = int(model_id)
                    if model_id not in fallback_fill_order:
                        fallback_fill_order.append(model_id)
        if len(fallback_fill_order) < selected_model_list_2d.shape[0]:
            print(
                f"⚠️ rank-consistency fallback skipped: fallback rank length={len(fallback_fill_order)}, "
                f"required={selected_model_list_2d.shape[0]}",
                flush=True,
            )
            return selected_model_list_2d, None, None, ""

        fallback_channels = np.flatnonzero(fallback_mask)
        selected_model_list_2d = selected_model_list_2d.copy()
        selected_model_list_2d[:, fallback_channels] = np.asarray(
            fallback_fill_order[: selected_model_list_2d.shape[0]],
            dtype=np.int64,
        )[:, None]

        fallback_model = int(fallback_fill_order[0])
        fallback_abbr, fallback_weight = named_weights[0]
        if rank_consistency_values is not None and rank_consistency_values.size == channel_count:
            active_scores = "{" + ", ".join(
                f"{int(channel)}:{float(rank_consistency_values[channel]):.2f}"
                for channel in fallback_channels
            ) + "}"
        else:
            active_scores = "{all:forced}"
        top5_text = ", ".join(f"{abbr}={weight:.4g}" for abbr, weight in named_weights[:5])

        print(
            f"[TS_Router-fallback] unstable rank-consistency fallback: threshold={threshold:g}, "
            f"mode={'all' if force_all_fallback else 'threshold'}, "
            f"channels={int(fallback_mask.sum())}/{channel_count}, "
            f"fallback_order={fallback_source}, channel_scores={active_scores}",
            flush=True,
        )
        print(f"[TS_Router-weight list] model_weight_prior_top5: {top5_text}", flush=True)
        # print(
        #     f"[TS_Router-fallback] window-level rank consistency is poor on channels={fallback_channels.tolist()}; "
        #     f"fallback to highest-weight model {fallback_abbr}(id={fallback_model}, weight={fallback_weight:.4g})",
        #     flush=True,
        # )
        print(
            f"[TS_Router-fallback after] channel_final_fused_rank_Top1={selected_model_list_2d[0].tolist()}",
            flush=True,
        )
        return selected_model_list_2d, fallback_mask, fallback_order, fallback_source

    def _get_select_strategy(self, dataset):
        'TSRouter runtime message.'

        def select_strategy(dataset_name=None, test_data_input=None):
            route_t0 = time.perf_counter()
            search_args, profile, pred_len, context_len_avg = self._adaptive_search_args(dataset=dataset, dataset_name=dataset_name, test_data_input=test_data_input)
            t_index_load0 = time.perf_counter()
            searcher = self._get_searcher(search_args)
            index_load_ms = (time.perf_counter() - t_index_load0) * 1000.0
            if auto_cl_enabled(self.args) or bool(getattr(self.args, "enable_pred_len_adaptive_repr", False)):
                term_fallback_used = bool(
                    getattr(search_args, "adaptive_task_term_fallback_used", False)
                )
                print(
                    f"[AdaptiveRepr] pred_len={pred_len}, context_len_avg={context_len_avg if context_len_avg is not None else 'NA'}, "
                    f"task_term_fallback_used={str(term_fallback_used).lower()}, "
                    f"profile={profile}, repr_input_dim={int(getattr(search_args, 'repr_input_dim', 0))}, "
                    f"repr_sub_pred_len={int(getattr(search_args, 'repr_sub_pred_len', 0))}, "
                    f"resolved_eval_cl={getattr(search_args, 'resolved_eval_cl', self.model_cl_name)}"
                )
            force_fresh_task_repr = bool(getattr(self.args, "vldb_force_fresh_task_repr", False))
            cache_lookup_ms = 0.0
            cached_search_input = None
            if dataset_name is not None and not force_fresh_task_repr:
                t_cache0 = time.perf_counter()
                cached_search_input = self.get_cached_task_repr(dataset_name, searcher=searcher, search_args=search_args)
                cache_lookup_ms = (time.perf_counter() - t_cache0) * 1000.0
            elif dataset_name is not None and force_fresh_task_repr:
                print(f"[VLDB latency] force fresh task repr, ignore cache: {dataset_name}")
            cache_hit = cached_search_input is not None
            task_sampling_ms = 0.0
            sample_timing_source = ""
            if cached_search_input is not None:
                cached_sample_seconds, sample_timing_source = _read_task_sample_seconds(
                    self._task_repr_cache_path,
                    self._normalize_cache_key(dataset_name),
                )
                if cached_sample_seconds is None:
                    print(
                        f"TSRouter runtime message: "
                        f"{dataset_name}, cache_pkl={self._task_repr_cache_path}, timing_csv={sample_timing_source}"
                    )
                    cached_search_input = None
                    cache_hit = False
                else:
                    task_sampling_ms = float(cached_sample_seconds) * 1000.0

            route_search_input = None
            route_input_from_cache = False
            if cached_search_input is not None:
                route_search_input = cached_search_input
                route_input_from_cache = True
            else:
                if test_data_input is None:
                    raise ValueError("[TSRouter] need test_data_input to build search_input")
                t_sample0 = time.perf_counter()
                fresh_search_input = self._build_search_input_array(test_data_input, searcher=searcher)
                task_sampling_ms = (time.perf_counter() - t_sample0) * 1000.0
                if dataset_name is not None:
                    self._upsert_task_repr_cache(
                        dataset_name,
                        fresh_search_input,
                        searcher=searcher,
                        search_args=search_args,
                        task_sampling_ms=task_sampling_ms,
                    )
                    sample_timing_source = _task_sample_timing_csv_path(self._task_repr_cache_path)
                route_search_input = fresh_search_input

            t_sample_to_route0 = time.perf_counter()
            try:
                selected_model_list = self._get_model_order_list(route_search_input, searcher=searcher)
            except Exception as e:
                if not route_input_from_cache:
                    raise
                print(f"TSRouter runtime message: {dataset_name}, err={e}")
                if test_data_input is None:
                    raise
                cache_hit = False
                t_sample0 = time.perf_counter()
                fresh_search_input = self._build_search_input_array(test_data_input, searcher=searcher)
                task_sampling_ms = (time.perf_counter() - t_sample0) * 1000.0
                if dataset_name is not None:
                    self._upsert_task_repr_cache(
                        dataset_name,
                        fresh_search_input,
                        searcher=searcher,
                        search_args=search_args,
                        task_sampling_ms=task_sampling_ms,
                    )
                    sample_timing_source = _task_sample_timing_csv_path(self._task_repr_cache_path)
                t_sample_to_route0 = time.perf_counter()
                selected_model_list = self._get_model_order_list(fresh_search_input, searcher=searcher)
            # selected_model_list: priority x 1 x C  -> priority x C
            if selected_model_list.ndim == 3:
                selected_model_list_2d = selected_model_list[:, 0, :]
            else:
                selected_model_list_2d = selected_model_list

                                                                  
            t_rank0 = time.perf_counter()
            rank_consistency_instability = getattr(self, "_last_task_rank_instability", None)
            rank_top3_instability = getattr(self, "_last_task_top3_instability", None)
            rank1_instability = getattr(self, "_last_task_rank1_instability", None)
            rank1_count_ratio = getattr(self, "_last_task_rank1_count_ratio", None)
            rank1_count_score = getattr(self, "_last_task_rank1_count_score", None)
            rank1_top_count = getattr(self, "_last_task_rank1_top_count", None)
            rank1_second_count = getattr(self, "_last_task_rank1_second_count", None)
            rank1_top_model = getattr(self, "_last_task_rank1_top_model", None)
            rank1_sample_count = getattr(self, "_last_task_rank1_sample_count", None)
            (
                selected_model_list_2d,
                fallback_mask,
                fallback_order,
                fallback_source,
            ) = self._apply_rank_consistency_fallback(
                selected_model_list_2d,
                searcher,
                search_args=search_args,
            )
            selected_models_per_channel, model_order, fuse_positions = self._restrict_model_order_with_channel_limit(selected_model_list_2d)
            (
                selected_model_list_2d,
                selected_models_per_channel,
                model_order,
            ) = self._apply_route_family_mode_to_task_outputs(
                selected_model_list_2d,
                selected_models_per_channel,
                model_order,
                searcher,
            )
            if fallback_mask is not None and fallback_mask.any():
                print(
                    f"[TS_Router-task fusion] task rank after normal channel fusion: model_order_top5={model_order[:5]}",
                    flush=True,
                )

            ensemble_size = self.args.ensemble_size
            restrict_rank_ms = (time.perf_counter() - t_rank0) * 1000.0
            search_timing = getattr(self, "_last_model_search_timing", {}) or {}
            task_embedding_ms = float(search_timing.get("task_embedding_ms", 0.0))
            index_lookup_ms = float(search_timing.get("index_lookup_ms", 0.0))
            rank_ms = float(search_timing.get("rank_ms", 0.0)) + restrict_rank_ms
            sample_to_route_ms = (time.perf_counter() - t_sample_to_route0) * 1000.0
            route_final_ms = float(task_sampling_ms) + float(sample_to_route_ms)
            task_cache_path = str(getattr(self, "_task_repr_cache_path", "") or "")

                                                                 
            extra = {
                "selected_model_list_2d": selected_model_list_2d,
                "selected_models_per_channel": selected_models_per_channel,
                "model_order": model_order,
                "task_sample_rankings": getattr(self, "_last_task_sample_rankings", None),
                "search_args": search_args,
                "adaptive_profile": profile,
                "route_family_mode": searcher.route_family_mode,
                "route_family_target_by_model_id": searcher.route_family_target_by_model_id.copy(),
                "adaptive_pred_len": pred_len,
                "adaptive_context_len_avg": context_len_avg,
                "adaptive_task_term_fallback_used": bool(
                    getattr(search_args, "adaptive_task_term_fallback_used", False)
                ),
                "resolved_eval_cl": str(
                    getattr(search_args, "resolved_eval_cl", self.model_cl_name)
                ),
                "rank_truth_cl": str(
                    getattr(search_args, "rank_truth_cl", self.model_cl_name)
                ),
                "task_sample_cache_path": task_cache_path,
                "task_sample_cache_hit": bool(cache_hit),
                "sample_timing_source": str(sample_timing_source or ""),
                "eval_cl_fallback_used": False,
                "rank1_instability": rank1_instability,
                "rank1_count_ratio": rank1_count_ratio,
                "rank1_count_score": rank1_count_score,
                "rank1_top_count": rank1_top_count,
                "rank1_second_count": rank1_second_count,
                "rank1_top_model": rank1_top_model,
                "rank1_sample_count": rank1_sample_count,
                "rank_top3_instability": rank_top3_instability,
                "rank_consistency_instability": rank_consistency_instability,
                "rank_consistency_fallback_mask": fallback_mask,
                "rank_top3_fallback_mask": fallback_mask,
                "rank_fallback_order": fallback_order,
                "rank_fallback_source": fallback_source,
                "task_channel_fuse_limit": str(getattr(self.args, "task_channel_fuse_limit", "all") or "all"),
                "task_channel_fuse_positions": fuse_positions,
                "step4_route_timing": {
                    "sample_seconds": float(task_sampling_ms) / 1000.0,
                    "sample_to_route_seconds": float(sample_to_route_ms) / 1000.0,
                    "route_final_seconds": float(route_final_ms) / 1000.0,
                    "task_feature_seconds": float(task_embedding_ms) / 1000.0,
                    "selector_predict_seconds": float(index_lookup_ms) / 1000.0,
                    "score_rank_seconds": float(rank_ms) / 1000.0,
                    "sample_timing_source": str(sample_timing_source or ""),
                    "cache_hit": bool(cache_hit),
                    "task_sample_cache_path": task_cache_path,
                },
            }
            route_latency_row = self._build_vldb_route_latency_row(
                dataset_name=dataset_name,
                searcher=searcher,
                model_order=model_order,
                cache_hit=cache_hit,
                cache_lookup_ms=cache_lookup_ms,
                index_load_ms=index_load_ms,
                task_sampling_ms=task_sampling_ms,
                task_embedding_ms=task_embedding_ms,
                index_lookup_ms=index_lookup_ms,
                rank_ms=rank_ms,
                sample_to_route_ms=sample_to_route_ms,
                selected_forecast_ms=None,
            )
            route_latency_row["route_overhead_ms"] = _ms(time.perf_counter() - route_t0)
            route_latency_row["end_to_end_ms"] = route_latency_row["route_overhead_ms"]
            extra["vldb_route_latency_row"] = route_latency_row
            self.last_selector_extra = extra
            return model_order, ensemble_size, extra

        return select_strategy

    def get_model_order_from_search_input(
        self,
        dataset_name: str,
        search_input: np.ndarray,
        dataset=None,
        test_data_input=None,
    ) -> List[int]:
        search_args, profile, pred_len, context_len_avg = self._adaptive_search_args(
            dataset=dataset,
            dataset_name=dataset_name,
            test_data_input=test_data_input,
        )
        searcher = self._get_searcher(search_args)
        if auto_cl_enabled(self.args) or bool(getattr(self.args, "enable_pred_len_adaptive_repr", False)):
            term_fallback_used = bool(
                getattr(search_args, "adaptive_task_term_fallback_used", False)
            )
            print(
                f"[AdaptiveRepr] pred_len={pred_len}, context_len_avg={context_len_avg if context_len_avg is not None else 'NA'}, "
                f"task_term_fallback_used={str(term_fallback_used).lower()}, "
                f"profile={profile}, repr_input_dim={int(getattr(search_args, 'repr_input_dim', 0))}, "
                f"repr_sub_pred_len={int(getattr(search_args, 'repr_sub_pred_len', 0))}, "
                f"resolved_eval_cl={getattr(search_args, 'resolved_eval_cl', self.model_cl_name)}"
            )
        sample_seconds, sample_timing_source = self.get_cached_task_sample_seconds(
            dataset_name,
            searcher=searcher,
            search_args=search_args,
        )
        task_cache_path = str(getattr(self, "_task_repr_cache_path", "") or "")
        route_search_input = search_input
        route_cache_hit = True
        if sample_seconds is None and test_data_input is not None:
            t_sample0 = time.perf_counter()
            route_search_input = self._build_search_input_array(test_data_input, searcher=searcher)
            sample_ms = (time.perf_counter() - t_sample0) * 1000.0
            sample_seconds = sample_ms / 1000.0
            self._upsert_task_repr_cache(
                dataset_name,
                route_search_input,
                searcher=searcher,
                search_args=search_args,
                task_sampling_ms=sample_ms,
            )
            sample_timing_source = _task_sample_timing_csv_path(self._task_repr_cache_path)
            route_cache_hit = False
        t_sample_to_route0 = time.perf_counter()
        selected_model_list = self._get_model_order_list(route_search_input, searcher=searcher)
        if selected_model_list.ndim == 3:
            selected_model_list_2d = selected_model_list[:, 0, :]
        else:
            selected_model_list_2d = selected_model_list
        rank_consistency_instability = getattr(self, "_last_task_rank_instability", None)
        rank_top3_instability = getattr(self, "_last_task_top3_instability", None)
        rank1_instability = getattr(self, "_last_task_rank1_instability", None)
        rank1_count_ratio = getattr(self, "_last_task_rank1_count_ratio", None)
        rank1_count_score = getattr(self, "_last_task_rank1_count_score", None)
        rank1_top_count = getattr(self, "_last_task_rank1_top_count", None)
        rank1_second_count = getattr(self, "_last_task_rank1_second_count", None)
        rank1_top_model = getattr(self, "_last_task_rank1_top_model", None)
        rank1_sample_count = getattr(self, "_last_task_rank1_sample_count", None)
        (
            selected_model_list_2d,
            fallback_mask,
            fallback_order,
            fallback_source,
        ) = self._apply_rank_consistency_fallback(
            selected_model_list_2d,
            searcher,
            search_args=search_args,
        )
        selected_models_per_channel, model_order, fuse_positions = self._restrict_model_order_with_channel_limit(selected_model_list_2d)
        (
            selected_model_list_2d,
            selected_models_per_channel,
            model_order,
        ) = self._apply_route_family_mode_to_task_outputs(
            selected_model_list_2d,
            selected_models_per_channel,
            model_order,
            searcher,
        )
        if fallback_mask is not None and fallback_mask.any():
            print(
                f"[TS_Router-task fusion] task rank after normal channel fusion: model_order_top5={model_order[:5]}",
                flush=True,
            )
        search_timing = getattr(self, "_last_model_search_timing", {}) or {}
        task_embedding_ms = float(search_timing.get("task_embedding_ms", 0.0))
        index_lookup_ms = float(search_timing.get("index_lookup_ms", 0.0))
        rank_ms = float(search_timing.get("rank_ms", 0.0))
        sample_to_route_seconds = (time.perf_counter() - t_sample_to_route0)
        sample_seconds_value = float(sample_seconds) if sample_seconds is not None else float("nan")
        route_final_seconds = (
            sample_seconds_value + sample_to_route_seconds
            if np.isfinite(sample_seconds_value)
            else float("nan")
        )
        self.last_selector_extra = {
            "selected_model_list_2d": selected_model_list_2d,
            "selected_models_per_channel": selected_models_per_channel,
            "model_order": model_order,
            "task_sample_rankings": getattr(self, "_last_task_sample_rankings", None),
            "search_args": search_args,
            "adaptive_profile": profile,
            "route_family_mode": searcher.route_family_mode,
            "route_family_target_by_model_id": searcher.route_family_target_by_model_id.copy(),
            "adaptive_pred_len": pred_len,
            "adaptive_context_len_avg": context_len_avg,
            "adaptive_task_term_fallback_used": bool(
                getattr(search_args, "adaptive_task_term_fallback_used", False)
            ),
            "resolved_eval_cl": str(
                getattr(search_args, "resolved_eval_cl", self.model_cl_name)
            ),
            "rank_truth_cl": str(
                getattr(search_args, "rank_truth_cl", self.model_cl_name)
            ),
            "task_sample_cache_path": task_cache_path,
            "task_sample_cache_hit": bool(route_cache_hit),
            "sample_timing_source": str(sample_timing_source or ""),
            "eval_cl_fallback_used": False,
            "rank1_instability": rank1_instability,
            "rank1_count_ratio": rank1_count_ratio,
            "rank1_count_score": rank1_count_score,
            "rank1_top_count": rank1_top_count,
            "rank1_second_count": rank1_second_count,
            "rank1_top_model": rank1_top_model,
            "rank1_sample_count": rank1_sample_count,
            "rank_top3_instability": rank_top3_instability,
            "rank_consistency_instability": rank_consistency_instability,
            "rank_consistency_fallback_mask": fallback_mask,
            "rank_top3_fallback_mask": fallback_mask,
            "rank_fallback_order": fallback_order,
            "rank_fallback_source": fallback_source,
            "task_channel_fuse_limit": str(getattr(self.args, "task_channel_fuse_limit", "all") or "all"),
            "task_channel_fuse_positions": fuse_positions,
            "step4_route_timing": {
                "sample_seconds": sample_seconds_value,
                "sample_to_route_seconds": sample_to_route_seconds,
                "route_final_seconds": route_final_seconds,
                "task_feature_seconds": float(task_embedding_ms) / 1000.0,
                "selector_predict_seconds": float(index_lookup_ms) / 1000.0,
                "score_rank_seconds": float(rank_ms) / 1000.0,
                "sample_timing_source": str(sample_timing_source or ""),
                "cache_hit": route_cache_hit,
                "task_sample_cache_path": task_cache_path,
            },
        }
        return model_order

                                           

    def _get_search_input(
        self,
        test_dataset,
        target_len,
        rng=None,
        sample_index: int = 0,
        sample_num: int = 1,
        preferred_entry_idx: int | None = None,
    ):
        if rng is None:
            rng = np.random

        entries = list(test_dataset)
        if len(entries) == 0:
            raise ValueError("test_dataset is empty")

        strategy = str(getattr(self.args, "task_sample_strategy", "latest_random")).lower()
        sample_num = max(1, int(sample_num or 1))
        sample_index = int(sample_index) % sample_num

        def _coverage_index(n: int) -> int:
            if n <= 1:
                return 0
            base = round((n - 1) * sample_index / max(sample_num - 1, 1))
            stride = max(1, (n - 1) // max(sample_num - 1, 1))
            jitter = int(rng.randint(-stride // 4, stride // 4 + 1)) if stride > 1 else 0
            return int(np.clip(base + jitter, 0, n - 1))

        if preferred_entry_idx is not None:
            first = int(preferred_entry_idx) % len(entries)
            indices = np.concatenate([np.arange(first, len(entries)), np.arange(0, first)])
        elif strategy == "time_coverage":
            first = _coverage_index(len(entries))
            indices = np.concatenate([np.arange(first, len(entries)), np.arange(0, first)])
        else:
            indices = rng.permutation(len(entries))
        best_target = None
        best_len = -1
        best_idx = 0

        def _clean_target(v: np.ndarray) -> np.ndarray | None:
            if v.ndim == 1:
                v = v[None, :]
            elif v.ndim != 2:
                raise ValueError(f"Unsupported target shape: {v.shape}")
            if np.isnan(v).all():
                return None
            mask = ~np.isnan(v).any(axis=0)
            if mask.any():
                return v[:, mask]
            return None

        def _pad_to_len(v: np.ndarray, length: int) -> np.ndarray:
            if v.shape[1] >= length:
                return v[:, -length:]
            if v.shape[1] == 0:
                raise ValueError("empty target after cleaning")
            pad_len = length - v.shape[1]
            base = v[:, ::-1]
            if base.shape[1] == 1:
                pad = np.repeat(base, pad_len, axis=1)
            else:
                rep = (pad_len + base.shape[1] - 1) // base.shape[1]
                pad = np.tile(base, rep)[:, :pad_len]
            return np.concatenate([pad, v], axis=1)[:, -length:]

        for idx in indices:
            entry = entries[idx]
            candidate = _clean_target(entry["target"])
            if candidate is None:
                continue
            cand_len = int(candidate.shape[1])
            if cand_len >= target_len:
                if strategy == "time_coverage":
                    max_start = cand_len - target_len
                    start_idx = _coverage_index(max_start + 1)
                    segment = candidate[:, start_idx:start_idx + target_len]
                else:
                    start_idx = cand_len - target_len
                    segment = candidate[:, -target_len:]
                if not np.isnan(segment).any():
                    self._last_search_input_source = {
                        "entry_idx": int(idx),
                        "segment_start_idx": int(start_idx),
                        "target_len_before_pad": int(cand_len),
                        "source": "observed_context",
                    }
                    return segment.T[None, :, :]
            if cand_len > best_len:
                best_target = candidate
                best_len = cand_len
                best_idx = int(idx)

        if best_target is None:
            raise ValueError("all targets are NaN")

        padded = _pad_to_len(best_target, target_len)
        if np.isnan(padded).any():
            print(" [TS_Router] NaN fill by channel mean")
            channel_means = np.nanmean(padded, axis=1, keepdims=True)
            channel_means = np.where(np.isnan(channel_means), 0.0, channel_means)
            padded = np.where(np.isnan(padded), channel_means, padded)

        self._last_search_input_source = {
            "entry_idx": int(best_idx),
            "segment_start_idx": 0,
            "target_len_before_pad": int(best_len),
            "source": "padded_observed_context",
        }
        return padded.T[None, :, :]



# =========================================================
                                                     
# =========================================================
class Zoo_Select_Predictor(Baseline_Select_Predictor):
    'TSRouter runtime message.'

    def _selected_forward_runtime_ms(self, selected_model_ids: list[int]) -> tuple[float | None, str]:
        runtimes = (getattr(self, "last_prediction_cache", {}) or {}).get("runtime_seconds", {}) or {}
        unique_ids = sorted({int(mid) for mid in selected_model_ids})
        vals = []
        missing = []
        for mid in unique_ids:
            val = runtimes.get(mid)
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                missing.append(mid)
            else:
                vals.append(float(val))
        if missing or not vals:
            return None, f"missing_runtime_for={missing}"
        return float(sum(vals) * 1000.0), "saved_prediction_meta_runtime"

    def _update_vldb_forecast_timing(self, extra: dict, cached_elapsed_ms: float) -> None:
        route_row = dict(extra.get("vldb_route_latency_row", {}) or {})
        if not route_row:
            return
        route_overhead = float(route_row.get("route_overhead_ms", 0.0) or 0.0)
        if bool(getattr(self.args, "vldb_fast_forward", True)):
            runtime_ms, runtime_source = self._selected_forward_runtime_ms(extra.get("selected_model_ids_for_forward", []))
            if runtime_ms is None:
                selected_forecast_ms = float(cached_elapsed_ms)
                route_row["forward_mode"] = "cached_prediction_missing_runtime"
                route_row["selected_forecast_timing_valid"] = "false"
                note = f"selected forward runtime unavailable ({runtime_source}); cached elapsed is recorded."
            else:
                selected_forecast_ms = runtime_ms
                route_row["forward_mode"] = runtime_source
                route_row["selected_forecast_timing_valid"] = "true"
                note = "VLDB_FAST_FORWARD reused saved predictions; selected_forecast_ms is charged from saved TSFM runtime metadata."
        else:
            runtime_ms, runtime_source = self._selected_forward_runtime_ms(extra.get("selected_model_ids_for_forward", []))
            if runtime_ms is None:
                selected_forecast_ms = float(cached_elapsed_ms)
                route_row["forward_mode"] = "cached_prediction_missing_runtime"
                route_row["selected_forecast_timing_valid"] = "false"
                note = f"selected forward runtime unavailable ({runtime_source}); cached elapsed is recorded."
            else:
                selected_forecast_ms = runtime_ms
                route_row["forward_mode"] = runtime_source
                route_row["selected_forecast_timing_valid"] = "true"
                note = "selected_forecast_ms uses saved TSFM runtime metadata as selected-model forward proxy."
        route_row["selected_forecast_ms"] = f"{selected_forecast_ms:.3f}"
        route_row["end_to_end_ms"] = f"{route_overhead + selected_forecast_ms:.3f}"
        route_row["vldb_fast_forward"] = _bool_text(bool(getattr(self.args, "vldb_fast_forward", True)))
        route_row["timing_note"] = (str(route_row.get("timing_note", "") or "") + " " + note).strip()
        extra["vldb_route_latency_row"] = route_row
        self.last_selector_extra = extra

    def predict(self, test_data_input: List[dict], dataset_name, model_order: Optional[List[int]] = None):
                                         
        if model_order is None:
            try:
                model_order, ensemble_size, extra = self.select_strategy(dataset_name, test_data_input)
            except TypeError:
                model_order, ensemble_size = self.select_strategy(dataset_name)
                extra = {}
        else:
            try:
                _, ensemble_size, extra = self.select_strategy(dataset_name, test_data_input)
            except TypeError:
                _, ensemble_size = self.select_strategy(dataset_name)
                extra = {}

        search_args = extra.get("search_args") if isinstance(extra, dict) else None
        if auto_cl_enabled(self.args):
            profile = str(extra.get("adaptive_profile", "") or "")
            profile_cfg = get_auto_cl_profile_by_name(profile, self.args)
            if profile_cfg is None:
                raise ValueError(
                    f"[AutoCL Step4] invalid adaptive profile for dataset={dataset_name}: {profile!r}"
                )
            resolved_eval_cl = str(
                getattr(search_args, "resolved_eval_cl", "")
                or profile_cfg["tsfm_results_dir"]
            )
            expected_eval_cl = str(profile_cfg["tsfm_results_dir"])
            if resolved_eval_cl != expected_eval_cl:
                raise ValueError(
                    f"[AutoCL Step4] profile/CL mismatch: dataset={dataset_name}, "
                    f"profile={profile}, resolved_eval_cl={resolved_eval_cl}, expected={expected_eval_cl}"
                )
            self.model_cl_name = resolved_eval_cl
            extra["resolved_eval_cl"] = resolved_eval_cl
            extra["rank_truth_cl"] = str(
                getattr(search_args, "rank_truth_cl", resolved_eval_cl)
            )
            extra["eval_cl_fallback_used"] = False

        try:
            loaded = self._load_predictions(dataset_name)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"[AutoCL Step4] missing TSFM prediction artifact: dataset={dataset_name}, "
                f"resolved_eval_cl={self.model_cl_name}, expected={exc}"
            ) from exc
        if self.use_sample_distribution:
            samples_dict, dataset_pred_per_model_gluonts = loaded
        else:
            preds_dict, dataset_pred_per_model_gluonts = loaded

        self.last_selector_extra = extra
        t_forecast0 = time.perf_counter()
        if self.use_sample_distribution:
            mix_samples = self._create_zoo_ensemble_distribution(
                samples_dict, model_order, ensemble_size, extra
            )  # (N, S, T, C)
            forecasts = numpy_samples_to_gluonts(mix_samples, dataset_pred_per_model_gluonts)
            self._update_vldb_forecast_timing(extra, (time.perf_counter() - t_forecast0) * 1000.0)
            return forecasts, model_order
        t_forecast0 = time.perf_counter()
        preds = self._create_zoo_ensemble(preds_dict, model_order, ensemble_size, extra)
        forecasts = numpy_to_gluonts(preds, dataset_pred_per_model_gluonts)
        self._update_vldb_forecast_timing(extra, (time.perf_counter() - t_forecast0) * 1000.0)
        return forecasts, model_order

    def _create_zoo_ensemble(self, preds_dict, model_order, ensemble_size, extra):
        C = self.channels
        selected_models_per_channel = extra.get("selected_models_per_channel", None)

                                                   
        if ensemble_size == 1:
            if selected_models_per_channel is None:
                                                
                chosen = [model_order[0]] * C
            else:
                chosen = list(selected_models_per_channel)

            print("[TS_Router] Only selected per-channel:", [self.id_to_abbr.get(i, i) for i in chosen])
            extra["selected_model_ids_for_forward"] = sorted({int(x) for x in chosen})

            ensemble_preds = []
            for c in range(C):
                mid = int(chosen[c])
                cur_pred_full = preds_dict[mid]            # (N, pred_len, C)
                ensemble_preds.append(cur_pred_full[:, :, c])
            preds = np.stack(ensemble_preds, axis=-1)      # (N, pred_len, C)
            return preds

                                                                   
        print(f"[TS_Router] Selected Top-{ensemble_size} Models: [", end=" ")
        topk = model_order[:ensemble_size]
        extra["selected_model_ids_for_forward"] = sorted({int(x) for x in topk})
        for mid in topk:
            print(self.id_to_abbr.get(int(mid), mid), end=" ")
        print("]")

        ensemble_preds = []
        for mid in topk:
            mid = int(mid)
            cur_pred_full = preds_dict[mid]   # (N, pred_len, C)
            ensemble_preds.append(cur_pred_full)

        ensemble_preds = np.stack(ensemble_preds, axis=-1)  # (N, pred_len, C, K)
        preds = self._aggregate_preds(ensemble_preds, ensemble_agg=self.args.ensemble_agg)            # (N, pred_len, C)
        return preds


    def _create_zoo_ensemble_distribution(self, samples_dict, model_order, ensemble_size, extra):
        'TSRouter runtime message.'
        C = self.channels
        rng = np.random.RandomState(getattr(self.args, "search_seed", 2025))
        selected_models_per_channel = extra.get("selected_models_per_channel", None)

                                  
        if ensemble_size == 1:
            if selected_models_per_channel is None:
                chosen = [model_order[0]] * C
            else:
                chosen = list(selected_models_per_channel)

            print("[TS_Router][dist] Only selected per-channel:",
                  [self.id_to_abbr.get(int(i), i) for i in chosen])
            extra["selected_model_ids_for_forward"] = sorted({int(x) for x in chosen})

                                          
            s_target = self.samples_per_model
            channel_samples = []
            channel_legacy_point = []
            for c in range(C):
                mid = int(chosen[c])
                arr = samples_dict[mid]  # (N,S,T,C)
                arr = self._resample_samples(arr, s_target, rng)
                                  
                channel_samples.append(arr[:, :, :, c])
                channel_legacy_point.append(np.median(samples_dict[mid][:, :, :, c], axis=1))  # (N,T)

                             
            mix = np.stack(channel_samples, axis=-1)
            legacy_point = np.stack(channel_legacy_point, axis=-1)  # (N,T,C)
            mix_median = np.median(mix, axis=1)
            mix = mix + (legacy_point - mix_median)[:, None, :, :]
            return mix

                                    
        print(f"[TS_Router][dist] Selected Top-{ensemble_size} Models: [", end=" ")
        topk = [int(i) for i in model_order[:ensemble_size]]
        extra["selected_model_ids_for_forward"] = sorted(set(topk))
        for mid in topk:
            print(self.id_to_abbr.get(mid, mid), end=" ")
        print("]")

        picked = []
        model_point_list = []
        for mid in topk:
            arr = samples_dict[mid]  # (N,S,T,C)
            model_point_list.append(np.median(arr, axis=1))  # (N,T,C)
            arr = self._resample_samples(arr, self.samples_per_model, rng)
            picked.append(arr)

                          
        mix = np.concatenate(picked, axis=1)  # (N, K*S_model, T, C)
        model_points = np.stack(model_point_list, axis=-1)  # (N,T,C,K)
        if self.args.ensemble_agg == "mean":
            legacy_point = np.mean(model_points, axis=-1)
        elif self.args.ensemble_agg == "median":
            legacy_point = np.median(model_points, axis=-1)
        else:
            raise ValueError(f"Unsupported ensemble_agg for TSRouter distribution mode: {self.args.ensemble_agg}")

        mix_median = np.median(mix, axis=1)
        mix = mix + (legacy_point - mix_median)[:, None, :, :]
        return mix



# =========================================================
                                            
# =========================================================
def restrict_top_k_models(
    selected_model_list_2d: np.ndarray,
    k: int,
    rank_decay_coef: float = 1.0,
    allowed_model_ids: list[int] | None = None,
):
    'TSRouter runtime message.'
    n_priority, C = selected_model_list_2d.shape
    valid_models = [int(x) for x in np.unique(selected_model_list_2d) if int(x) >= 0]
    n_models = (max(valid_models) + 1) if valid_models else 0
    if n_models <= 0:
        raise ValueError("selected_model_list_2d does not contain valid model ids")

    score_template = rank_position_scores(n_priority, decay_coef=rank_decay_coef)
    model_scores = defaultdict(float)
    for priority in range(n_priority):
        row = selected_model_list_2d[priority]
        unique, counts = np.unique(row[row >= 0], return_counts=True)
        for model, count in zip(unique, counts):
            model_scores[int(model)] += float(count) * float(score_template[priority])

    if allowed_model_ids is None:
        candidates = list(range(n_models))
    else:
        candidates = list(dict.fromkeys(int(value) for value in allowed_model_ids))
        if not candidates:
            raise ValueError("allowed_model_ids is empty")
        unexpected = sorted(set(model_scores) - set(candidates))
        if unexpected:
            raise ValueError(f"rank contains models outside allowed_model_ids: {unexpected}")
    for mid in candidates:
        model_scores.setdefault(mid, 0.0)
    if allowed_model_ids is not None:
        model_scores = defaultdict(
            float,
            {mid: float(model_scores[mid]) for mid in candidates},
        )
    model_order = [m for m, _ in sorted(model_scores.items(), key=lambda x: (-x[1], x[0]))]
    topk_models = model_order[: max(1, int(k))]

    filled = selected_model_list_2d.copy().astype(float)
    for p in range(n_priority):
        for c in range(C):
            if int(filled[p, c]) not in topk_models:
                filled[p, c] = np.nan

    final_rec = np.full(C, np.nan)
    for c in range(C):
        for p in range(n_priority):
            if not np.isnan(filled[p, c]):
                final_rec[c] = filled[p, c]
                break

    assert not np.any(np.isnan(final_rec)), 'TSRouter runtime message.'
    return final_rec.astype(int), model_order

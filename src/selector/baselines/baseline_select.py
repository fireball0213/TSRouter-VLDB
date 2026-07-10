import os
import csv
import hashlib
import json
import pickle
import subprocess
import sys
import time
import warnings
import torch
from torch import cuda
from typing import List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

from model_zoo.base_model import BaseModel
from utils.data import (
    gluonts_to_numpy,
    numpy_to_gluonts,
    load_gluonts_pred,
    load_gluonts_pred_distribution,
    numpy_samples_to_gluonts,
)
from utils.tsrouter_metrics import (
    channel_orders_from_error_matrix,
    load_per_channel_error_matrix,
    real_channel_rank_cache_path,
    save_channel_rank_orders,
)
from utils.path_utils import (
    build_repr_eval_pool_name,
    get_gift_eval_task_repr_cache_path,
    get_tsrouter_selector_result_dir,
    resolve_tsfm_artifact_path,
    resolve_tsfm_csv_path,
)
from utils.project_paths import (
    BASELINE_CSV_ROOT,
    SAMPLED_REPR_POOL_CACHE_ROOT,
    SRC_ROOT,
    TSFM_ARTIFACT_ROOT,
    TSROUTER_REPR_FORWARD_CSV_ROOT,
    TSROUTER_VLDB_LOG_ROOT,
)
from encoder.baseline.random_stats_features import compute_stats_features


class Baseline_Select_Model(BaseModel):
    def __init__(self, args, model_name, Model_zoo_current):
        self.args = args
        self.model_name = model_name
        self.Model_sizes = Model_zoo_current
        if self.model_name == "TSRouter":
            self.output_dir = get_tsrouter_selector_result_dir(args)
        else:
            self.output_dir = str(BASELINE_CSV_ROOT / "selectors" / self.model_name)
        super().__init__(self.model_name, args, self.output_dir)
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")

                        
        self.abbr_to_id = {
            model_info["abbreviation"]: model_info["id"]
            for family in self.Model_sizes.values()
            for model_info in family.values()
        }

        if self.args.fix_context_len:
            self.model_cl_name = f"{self.args.TSFM_results_dir}"
        else:
            self.model_cl_name = "cl_original"

    def get_predictor(self, dataset, batch_size):
        'TSRouter runtime message.'
        self.args.test_pred_len = dataset.prediction_length
        return Baseline_Select_Predictor(
            args=self.args,
            Model_sizes=self.Model_sizes,
            prediction_length=dataset.prediction_length,
            channels=dataset.target_dim,
            windows=dataset.windows,
            model_cl_name=self.model_cl_name,
            select_strategy=self._get_select_strategy(dataset)            
        )

    def _get_select_strategy(self, dataset):
        'TSRouter runtime message.'
        raise NotImplementedError


class Baseline_Select_Predictor:
    'TSRouter runtime message.'

    def __init__(
            self, args, Model_sizes,
            prediction_length: int,
            channels, windows, model_cl_name,
            select_strategy: callable          
    ):
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")
        self.prediction_length = prediction_length
        self.args = args
        self.Model_sizes = Model_sizes
        self.channels = channels
        self.windows = windows
        self.model_cl_name = model_cl_name
        self.select_strategy = select_strategy
                       
                                     
                                    
        self.use_sample_distribution = bool(getattr(args, "selector_use_sample_dist", True))
                                       
        self.samples_per_model = int(getattr(args, "selector_samples_per_model", 100))
                                     
        self.point_from_distribution = str(getattr(args, "selector_point_from", "median")).lower()

        self.id_to_abbr = {
            model_info["id"]: model_info["abbreviation"]
            for family in self.Model_sizes.values()
            for model_info in family.values()
        }

    def predict(self, test_data_input: List[dict], dataset_name, model_order: Optional[List[int]] = None) -> List:
        'TSRouter runtime message.'
        loaded = self._load_predictions(dataset_name)
        if self.use_sample_distribution:
            samples_dict, dataset_pred_per_model_gluonts = loaded
        else:
            preds_dict, dataset_pred_per_model_gluonts = loaded

        if model_order is None:
                                         
            sel_out = self.select_strategy(dataset_name, test_data_input)
            model_order, ensemble_size, selector_extra = self._parse_select_output(sel_out)
        else:
            sel_out = self.select_strategy(dataset_name, test_data_input)
            _, ensemble_size, selector_extra = self._parse_select_output(sel_out)
        self.last_selector_extra = {"model_order": model_order, **(selector_extra or {})}
        self._complete_vldb_route_row(dataset_name, model_order)

              
        if self.use_sample_distribution:
            forecasts = self._create_ensemble_distribution(
                samples_dict=samples_dict,
                model_order=model_order,
                ensemble_size=ensemble_size,
                template_dataset=dataset_pred_per_model_gluonts,
            )
            return forecasts, model_order

        preds = self._create_ensemble(preds_dict, model_order, ensemble_size)

        if self.args.debug_mode:
            print(f"Predictions shape: {np.shape(preds)}")

                          
        forecasts = numpy_to_gluonts(preds, dataset_pred_per_model_gluonts)

        return forecasts, model_order

    @staticmethod
    def _parse_select_output(sel_out):
        if isinstance(sel_out, tuple) and len(sel_out) >= 2:
            model_order = sel_out[0]
            ensemble_size = sel_out[1]
            selector_extra = sel_out[2] if len(sel_out) >= 3 and isinstance(sel_out[2], dict) else {}
            return model_order, ensemble_size, selector_extra
        raise ValueError('TSRouter runtime message.')

    def _runtime_ms_for_model_id(self, model_id: int) -> tuple[float, str]:
        runtimes = (getattr(self, "last_prediction_cache", {}) or {}).get("runtime_seconds", {}) or {}
        val = _numeric_or_nan(runtimes.get(int(model_id)))
        if np.isfinite(val):
            return float(val) * 1000.0, "prediction_meta_runtime_seconds"
        return float("nan"), "missing_prediction_runtime"

    def _complete_vldb_route_row(self, dataset_name: str, model_order) -> None:
        extra = getattr(self, "last_selector_extra", {}) or {}
        route_row = extra.get("vldb_route_latency_row")
        if not isinstance(route_row, dict) or not route_row:
            return
        selected_ms = float("nan")
        runtime_source = "missing_prediction_runtime"
        if model_order is not None and len(model_order) > 0:
            selected_ms, runtime_source = self._runtime_ms_for_model_id(int(model_order[0]))
        route_overhead = _numeric_or_nan(route_row.get("route_overhead_ms"))
        if not np.isfinite(route_overhead):
            route_overhead = 0.0
        if np.isfinite(selected_ms):
            route_row["selected_forecast_ms"] = f"{selected_ms:.3f}"
            route_row["selected_forecast_timing_valid"] = "true"
            route_row["end_to_end_ms"] = f"{route_overhead + selected_ms:.3f}"
        else:
            route_row["selected_forecast_ms"] = ""
            route_row["selected_forecast_timing_valid"] = "false"
            route_row["end_to_end_ms"] = f"{route_overhead:.3f}"
        note = str(route_row.get("timing_note", "") or "")
        route_row["selected_model_order"] = " ".join(map(str, model_order or []))
        route_row["timing_note"] = (
            f"{note} selected_forecast_source={runtime_source}; final evaluate_forecasts is not part of serving latency."
        ).strip()
        extra["vldb_route_latency_row"] = route_row
        self.last_selector_extra = extra


    def _load_predictions(self, dataset_name):
        'TSRouter runtime message.'
        if self.use_sample_distribution:
            return self._load_predictions_distribution(dataset_name)
        return self._load_predictions_point(dataset_name)

    def _load_predictions_point(self, dataset_name):
        'TSRouter runtime message.'
        preds_dict = {}
        runtime_dict = {}

        for family, size_dict in self.Model_sizes.items():
            for size_name in size_dict.keys():
                model_name = f"{family}_{size_name}"
                model_idx = size_dict[size_name]['id']

                try:
                    dataset_pred_per_model_gluonts, pred_npy, runtime_seconds = load_gluonts_pred(
                        str(TSFM_ARTIFACT_ROOT), model_name, self.model_cl_name, dataset_name, self.prediction_length,
                        self.channels, self.windows
                    )
                except FileNotFoundError as exc:
                    samples_path = resolve_tsfm_artifact_path(
                        model_name,
                        self.model_cl_name,
                        "npy",
                        f"{dataset_name}_samples.npy",
                    )
                    meta_path = resolve_tsfm_artifact_path(
                        model_name,
                        self.model_cl_name,
                        "meta",
                        f"{dataset_name}_meta.json",
                    )
                    raise FileNotFoundError(
                        f"model={model_name}, dataset={dataset_name}, CL={self.model_cl_name}, "
                        f"expected_samples={samples_path}, expected_meta={meta_path}"
                    ) from exc
                dataset_pred_per_model = gluonts_to_numpy(dataset_pred_per_model_gluonts)
                runtime_dict[model_idx] = runtime_seconds

                if dataset_pred_per_model.shape[-1] == 1 and self.channels > 1:
                    dataset_pred_per_model = dataset_pred_per_model.reshape(
                        -1, self.channels, dataset_pred_per_model.shape[1]
                    ).transpose(0, 2, 1)

                preds_dict[model_idx] = dataset_pred_per_model

        self.last_prediction_cache = {
            "kind": "point",
            "predictions": preds_dict,
            "runtime_seconds": runtime_dict,
        }
        return preds_dict, dataset_pred_per_model_gluonts

    def _load_predictions_distribution(self, dataset_name):
        'TSRouter runtime message.'
        samples_dict = {}
        runtime_dict = {}

        for family, size_dict in self.Model_sizes.items():
            for size_name in size_dict.keys():
                model_name = f"{family}_{size_name}"
                model_idx = size_dict[size_name]["id"]

                try:
                    dataset_pred_per_model_gluonts, pred_samples_4d, runtime_seconds = load_gluonts_pred_distribution(
                        str(TSFM_ARTIFACT_ROOT), model_name, self.model_cl_name, dataset_name, self.prediction_length,
                        self.channels, self.windows
                    )
                except FileNotFoundError as exc:
                    samples_path = resolve_tsfm_artifact_path(
                        model_name,
                        self.model_cl_name,
                        "npy",
                        f"{dataset_name}_samples.npy",
                    )
                    meta_path = resolve_tsfm_artifact_path(
                        model_name,
                        self.model_cl_name,
                        "meta",
                        f"{dataset_name}_meta.json",
                    )
                    raise FileNotFoundError(
                        f"model={model_name}, dataset={dataset_name}, CL={self.model_cl_name}, "
                        f"expected_samples={samples_path}, expected_meta={meta_path}"
                    ) from exc
                samples_dict[model_idx] = pred_samples_4d.astype(np.float32)
                runtime_dict[model_idx] = runtime_seconds

        self.last_prediction_cache = {
            "kind": "distribution",
            "predictions": samples_dict,
            "runtime_seconds": runtime_dict,
        }
        return samples_dict, dataset_pred_per_model_gluonts

    def _resample_samples(self, arr: np.ndarray, target_s: int, rng: np.random.RandomState) -> np.ndarray:
        'TSRouter runtime message.'
        if arr.ndim != 4:
            raise ValueError(f"TSRouter runtime message: {arr.shape}")
        s = arr.shape[1]
        if s == target_s:
            return arr
        if s > target_s:
            idx = rng.choice(s, size=target_s, replace=False)
        else:
            idx = rng.choice(s, size=target_s, replace=True)
        return arr[:, idx, :, :]

    def _aggregate_preds(self, ensemble_preds: np.ndarray, ensemble_agg) -> np.ndarray:
        'TSRouter runtime message.'

        if ensemble_agg == "mean":
            return np.mean(ensemble_preds, axis=-1)
        elif ensemble_agg == "median":
            return np.median(ensemble_preds, axis=-1)
        else:
            raise ValueError(
                f"[Predictor] Unsupported ensemble_agg='{ensemble_agg}', "
                f"please use 'mean' or 'median'"
            )

    def _create_ensemble(self, preds_dict, model_order, ensemble_size):
        'TSRouter runtime message.'
        ensemble_preds = []
        print(f"Selected Top-{ensemble_size} Models : [", end=' ')

        for model_idx in range(ensemble_size):
            channel_select_model_id = model_order[model_idx]
            select_model_name = self.id_to_abbr.get(channel_select_model_id)
            print(f"{select_model_name}", end=" ")

            cur_pred_full = preds_dict[channel_select_model_id]
            ensemble_size_pred = []

            for cur_channel in range(self.channels):
                cur_pred = cur_pred_full[:, :, cur_channel]  # (N, pred_len)
                ensemble_size_pred.append(cur_pred)

            ensemble_size_preds = np.stack(ensemble_size_pred, axis=-1)  # (N, pred_len, C)
            ensemble_preds.append(ensemble_size_preds)

        print(']')
        ensemble_preds_lst = np.stack(ensemble_preds, axis=-1)
        preds = self._aggregate_preds(ensemble_preds_lst,ensemble_agg=self.args.ensemble_agg)  # (N, pred_len, C)

        return preds

    def _create_ensemble_distribution(self, samples_dict, model_order, ensemble_size, template_dataset):
        'TSRouter runtime message.'
        rng = np.random.RandomState(getattr(self.args, "search_seed", 2025))
        picked = []
        model_point_list = []
        print(f"Selected Top-{ensemble_size} Models (distribution mode): [", end=" ")

        for model_idx in range(ensemble_size):
            channel_select_model_id = model_order[model_idx]
            select_model_name = self.id_to_abbr.get(channel_select_model_id)
            print(f"{select_model_name}", end=" ")
            cur = samples_dict[channel_select_model_id]  # (N,S,T,C)
                                 
            model_point_list.append(np.median(cur, axis=1))  # (N,T,C)
            cur = self._resample_samples(cur, self.samples_per_model, rng)
            picked.append(cur)
        print("]")

                                         
        mix = np.concatenate(picked, axis=1)

                                     
                                          
        model_points = np.stack(model_point_list, axis=-1)  # (N,T,C,K)
        if self.args.ensemble_agg == "mean":
            legacy_point = np.mean(model_points, axis=-1)
        elif self.args.ensemble_agg == "median":
            legacy_point = np.median(model_points, axis=-1)
        else:
            raise ValueError(f"Unsupported ensemble_agg for distribution mode: {self.args.ensemble_agg}")

        mix_median = np.median(mix, axis=1)  # (N,T,C)
        delta = legacy_point - mix_median
        mix = mix + delta[:, None, :, :]

        forecasts = numpy_samples_to_gluonts(mix, template_dataset)
        return forecasts


class All_Select_Model(Baseline_Select_Model):
    'TSRouter runtime message.'

    def _get_select_strategy(self, dataset):
        def select_strategy(dataset_name=None, test_data_input=None):                      
            model_order = list(range(self.args.current_zoo_num - 1, -1, -1))             
            ensemble_size = self.args.current_zoo_num             
            return model_order, ensemble_size

        return select_strategy


class Random_Select_Model(Baseline_Select_Model):
    'TSRouter runtime message.'

    def _get_select_strategy(self, dataset):
        def select_strategy(dataset_name=None, test_data_input=None):
            ensemble_size = self.args.ensemble_size
            return None, ensemble_size, {
                "vldb_route_latency_row": _baseline_route_row(
                    self.args,
                    "Random",
                    dataset_name or "",
                    [],
                    zoo_size=int(getattr(self.args, "current_zoo_num", 0)),
                    route_ms=0.0,
                    note="Random static selector has no route computation; E2E charges selected TSFM forward runtime only.",
                    timing_valid=True,
                    selected_forecast_timing_valid=True,
                    forward_mode="static_selected_runtime_metadata",
                    cache_mode="static_selector",
                    cache_hit=False,
                )
            }

        return select_strategy


class Recent_Select_Model(Baseline_Select_Model):
    'TSRouter runtime message.'

    def _get_select_strategy(self, dataset):
        def select_strategy(dataset_name=None, test_data_input=None):
            model_order = list(range(self.args.current_zoo_num - 1, -1, -1))                   
            ensemble_size = self.args.ensemble_size
            return model_order, ensemble_size, {
                "vldb_route_latency_row": _baseline_route_row(
                    self.args,
                    "Recent",
                    dataset_name or "",
                    model_order,
                    zoo_size=int(getattr(self.args, "current_zoo_num", len(model_order))),
                    route_ms=0.0,
                    note="Recent static selector has no route computation; E2E charges selected TSFM forward runtime only.",
                    timing_valid=True,
                    selected_forecast_timing_valid=True,
                    forward_mode="static_selected_runtime_metadata",
                    cache_mode="static_selector",
                    cache_hit=False,
                )
            }

        return select_strategy


def _dataset_name_to_config(dataset_name: str) -> str:
    try:
        ds_key, ds_freq, term = str(dataset_name).rsplit("_", 2)
        return f"{ds_key}/{ds_freq}/{term}"
    except ValueError:
        return str(dataset_name)


def _stable_hash01(text: str) -> float:
    h = hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8]
    return int(h, 16) / float(16**8 - 1)


def _simple_meta_features(ds_config: str) -> np.ndarray:
    parts = str(ds_config).split("/")
    ds_key = parts[0] if len(parts) > 0 else ""
    freq = parts[1] if len(parts) > 1 else ""
    term = parts[2] if len(parts) > 2 else ""
    freq_seconds = {
        "10S": 10, "5T": 300, "10T": 600, "15T": 900,
        "H": 3600, "D": 86400, "W": 604800, "M": 2629800,
        "Q": 7889400, "A": 31557600,
    }.get(freq, 0)
    horizon = {"short": 1.0, "medium": 2.0, "long": 3.0}.get(term, 1.0)
    return np.asarray([
        float(freq_seconds),
        float(horizon),
        _stable_hash01(ds_key),
        _stable_hash01(freq),
        _stable_hash01(term),
        _stable_hash01(f"{ds_key}:{freq}:{term}"),
    ], dtype=float)


    """Return the 16 StatsRandomFourier statistical features averaged over K x C windows."""
def _load_step4_task_cache(args, ds_config: str) -> tuple[np.ndarray | None, dict, str, str]:
    cache_path = get_gift_eval_task_repr_cache_path(
        args,
        search_context_len=int(getattr(args, "repr_input_dim", getattr(args, "context_len", 512))),
    )
    key = str(ds_config)
    if not Path(cache_path).exists():
        return None, {}, cache_path, f"missing_cache:{cache_path}"
    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    except Exception as e:
        return None, {}, cache_path, f"cannot_read_cache:{type(e).__name__}"
    if not isinstance(cache, dict):
        return None, {}, cache_path, "cache_not_dict"
    norm_key = key if "/" in key else _dataset_name_to_config(key)
    arr = cache.get(norm_key)
    if arr is None:
        arr = cache.get(key)
    meta_path = f"{cache_path}.meta.json"
    meta = {}
    if Path(meta_path).exists():
        try:
            loaded = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                meta = loaded.get(norm_key) or loaded.get(key) or {}
        except Exception:
            meta = {}
    return (None if arr is None else np.asarray(arr, dtype=np.float32)), meta, cache_path, ""


def _step4_window_ids_from_metadata(meta: dict, sample_n: int) -> list[int]:
    ids = meta.get("entry_indices", [])
    if not isinstance(ids, list):
        return []
    out = []
    for item in ids[:sample_n]:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def _task_sample_build_ms_from_route_log(args, ds_config: str) -> tuple[float, str]:
    log_path = _selector_log_dir(args) / "route_latency_log.csv"
    if not log_path.exists():
        return float("nan"), "missing_route_latency_log"
    try:
        df = pd.read_csv(
            log_path,
            usecols=lambda c: c in {
                "timestamp_utc",
                "stage",
                "method",
                "dataset",
                "timing_level",
                "task_sampling_ms",
                "task_sampling_timing_valid",
                "vldb_fast_sample",
                "route_id",
            },
        )
    except Exception as e:
        return float("nan"), f"route_latency_log_read_failed:{type(e).__name__}"
    if df.empty or "task_sampling_ms" not in df.columns:
        return float("nan"), "route_latency_log_missing_task_sampling_ms"
    ds_name = _dataset_config_to_name(ds_config)
    sub = df[df.get("dataset", pd.Series(index=df.index, dtype=str)).astype(str).isin({str(ds_config), ds_name})].copy()
    if "method" in sub.columns:
        sub = sub[sub["method"].astype(str).isin({"TSRouter", "TSRouter"})]
    if "timing_level" in sub.columns:
        sub = sub[sub["timing_level"].astype(str).eq("selector_dataset_internal")]
    if "stage" in sub.columns:
        stage = int(_effective_vldb_stage(args))
        if stage > 0:
            staged = sub[pd.to_numeric(sub["stage"], errors="coerce").eq(float(stage))]
            if not staged.empty:
                sub = staged
    if "task_sampling_timing_valid" in sub.columns:
        valid = sub["task_sampling_timing_valid"].astype(str).str.lower().eq("true")
        if valid.any():
            sub = sub[valid]
    if "vldb_fast_sample" in sub.columns:
        fresh = sub["vldb_fast_sample"].astype(str).str.lower().ne("true")
        if fresh.any():
            sub = sub[fresh]
    vals = pd.to_numeric(sub.get("task_sampling_ms", pd.Series(dtype=float)), errors="coerce")
    sub = sub[vals.notna()].copy()
    if sub.empty:
        return float("nan"), "route_latency_log_no_valid_task_sampling_ms"
    if "timestamp_utc" in sub.columns:
        sub = sub.sort_values("timestamp_utc")
    row = sub.iloc[-1]
    val = _numeric_or_nan(row.get("task_sampling_ms"))
    if not np.isfinite(val):
        return float("nan"), "route_latency_log_no_valid_task_sampling_ms"
    rid = str(row.get("route_id", "") or "")
    return float(val), f"route_latency_log:{rid or 'stage'}"


def _task_sample_build_ms_from_metadata(args, ds_config: str, meta: dict) -> tuple[float, str]:
    for key in ["task_sampling_ms", "sample_build_ms", "task_repr_build_ms"]:
        if key in meta:
            val = _numeric_or_nan(meta.get(key))
            if np.isfinite(val):
                return float(val), key
    return _task_sample_build_ms_from_route_log(args, ds_config)


def _bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def _dataset_config_to_name(ds_config: str) -> str:
    parts = str(ds_config).split("/")
    if len(parts) >= 3:
        return "_".join([parts[0], parts[1], parts[2]])
    return str(ds_config).replace("/", "_")


def _selector_log_dir(args) -> Path:
    route_log = str(getattr(args, "vldb_route_latency_log", "") or "")
    if route_log:
        return Path(route_log).parent
    return TSROUTER_VLDB_LOG_ROOT


def _fmt_int_list(values, limit: int = 20) -> str:
    raw = list(values or [])
    out = []
    for item in raw[: int(limit)]:
        try:
            out.append(str(int(item)))
        except Exception:
            out.append(str(item))
    suffix = "" if len(raw) <= int(limit) else " ..."
    return "[" + " ".join(out) + suffix + "]"


def _effective_vldb_stage(args) -> int:
    try:
        stage = int(getattr(args, "vldb_route_stage", -1))
    except Exception:
        stage = -1
    if stage > 0:
        return stage
    try:
        return int(getattr(args, "current_zoo_num", 0))
    except Exception:
        return 0


def _baseline_method_slug(method: str, args) -> str:
    text = str(method).strip().lower().replace("-", "_")
    text = "_".join(text.split())
    if text == "task_probe_forward":
        return f"task_probe_forward_task{int(getattr(args, 'sample_repr_num', 20))}"
    return text


def _effective_vldb_route_id(args, method: str) -> str:
    explicit = str(getattr(args, "vldb_route_id", "") or "").strip()
    try:
        explicit_stage = int(getattr(args, "vldb_route_stage", -1))
    except Exception:
        explicit_stage = -1
    if explicit and explicit_stage > 0:
        return explicit
    return f"stage{_effective_vldb_stage(args)}_{_baseline_method_slug(method, args)}_route"


def _append_csv_rows(path: Path, rows: list[dict], columns: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    exists = path.exists()
    schema_mismatch = False
    if exists:
        try:
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and list(reader.fieldnames) != list(columns):
                    schema_mismatch = True
                    merged = list(reader.fieldnames)
                    for col in columns:
                        if col not in merged:
                            merged.append(col)
                    columns = merged
                    existing_rows = list(reader)
        except Exception:
            existing_rows = []
    if schema_mismatch:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in existing_rows:
                writer.writerow({c: row.get(c, "") for c in columns})
            for row in rows:
                writer.writerow({c: row.get(c, "") for c in columns})
        return
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _numeric_or_nan(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _read_dataset_runtime_seconds(model_folder: str, model_cl_name: str, ds_config: str) -> tuple[float, str]:
    result_path = resolve_tsfm_csv_path(model_folder, model_cl_name, "all_results.csv")
    if result_path.exists():
        try:
            df = pd.read_csv(
                result_path,
                usecols=lambda c: c in {
                    "dataset",
                    "forward_runtime_seconds",
                },
            )
            if "dataset" in df.columns:
                sub = df[df["dataset"].astype(str).eq(str(ds_config))]
                if not sub.empty:
                    row = sub.iloc[-1]
                    val = _numeric_or_nan(row.get("forward_runtime_seconds"))
                    if np.isfinite(val):
                        return float(val), "all_results:forward_runtime_seconds"
        except Exception:
            pass
    meta_path = resolve_tsfm_artifact_path(
        model_folder,
        model_cl_name,
        "meta",
        f"{_dataset_config_to_name(ds_config)}_meta.json",
    )
    if meta_path.exists():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            perf = payload.get("performance", {}) or {}
            val = _numeric_or_nan(perf.get("forward_runtime_seconds"))
            if np.isfinite(val):
                return float(val), "meta:forward_runtime_seconds"
        except Exception:
            pass
    return float("nan"), "missing_runtime"


def _select_calibration_keys(sample_keys: list, args) -> list:
    keys = list(sample_keys)
    if not keys:
        return []
    sample_n = int(getattr(args, "sample_repr_num", 20))
    ratio = float(getattr(args, "sample_repr_ratio", 0.0) or 0.0)
    if ratio > 0:
        sample_n = max(1, int(np.ceil(len(keys) * ratio)))
    sample_n = min(sample_n, len(keys))
    strategy = str(getattr(args, "task_window_sample_strategy", "legacy") or "legacy").lower()
    if strategy in {"legacy", "last"}:
        return keys[-sample_n:]
    if strategy == "first":
        return keys[:sample_n]
    if strategy == "even":
        if sample_n == 1:
            return [keys[len(keys) // 2]]
        idx = np.linspace(0, len(keys) - 1, num=sample_n)
        return [keys[int(round(i))] for i in idx]
    if strategy == "random":
        rng = np.random.RandomState(int(getattr(args, "search_seed", 2025)))
        idx = sorted(rng.choice(len(keys), size=sample_n, replace=False).tolist())
        return [keys[i] for i in idx]
    return keys[-sample_n:]


def _baseline_route_row(args, method: str, dataset_name: str, model_order: list[int], zoo_size: int, route_ms: float, note: str, **parts) -> dict:
    selected_forecast_ms = parts.get("selected_forecast_ms", "")
    try:
        e2e_ms = float(route_ms) + float(selected_forecast_ms or 0.0)
    except Exception:
        e2e_ms = float(route_ms)
    return {
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "route_id": _effective_vldb_route_id(args, method),
        "stage": str(_effective_vldb_stage(args)),
        "method": method,
        "profile_id": str(getattr(args, "vldb_route_profile_id", "") or "selector_baseline"),
        "dataset": str(_dataset_name_to_config(dataset_name)),
        "status": "dataset_executed",
        "zoo_size": str(zoo_size),
        "stable_model_ids": "",
        "selected_model_order": " ".join(map(str, model_order or [])),
        "step4_skip_saved": _bool_text(bool(getattr(args, "skip_saved", False))),
        "cache_mode": str(parts.get("cache_mode", "selector_cache")),
        "cache_hit": _bool_text(bool(parts.get("cache_hit", False))),
        "timing_level": "selector_dataset_internal",
        "timing_valid": _bool_text(bool(parts.get("timing_valid", True))),
        "route_command_s": "",
        "cache_lookup_ms": f"{float(parts.get('cache_lookup_ms', 0.0) or 0.0):.3f}",
        "index_load_ms": f"{float(parts.get('index_load_ms', 0.0) or 0.0):.3f}",
        "task_sampling_ms": f"{float(parts.get('task_sampling_ms', 0.0) or 0.0):.3f}",
        "task_embedding_ms": f"{float(parts.get('task_embedding_ms', 0.0) or 0.0):.3f}",
        "index_lookup_ms": f"{float(parts.get('index_lookup_ms', 0.0) or 0.0):.3f}",
        "rank_ms": f"{float(parts.get('rank_ms', 0.0) or 0.0):.3f}",
        "route_overhead_ms": f"{float(route_ms):.3f}",
        "selected_forecast_ms": "" if selected_forecast_ms == "" else f"{float(selected_forecast_ms):.3f}",
        "evaluate_ms": "",
        "metric_read_ms": "",
        "end_to_end_ms": f"{e2e_ms:.3f}",
        "fast_eval_enabled": _bool_text(bool(getattr(args, "vldb_skip_evaluate", False))),
        "evaluation_mode": "saved_metric_fast_eval" if bool(getattr(args, "vldb_skip_evaluate", False)) else "gluonts_evaluate",
        "vldb_fast_sample": _bool_text(bool(getattr(args, "vldb_fast_sample", False))),
        "vldb_fast_forward": _bool_text(bool(getattr(args, "vldb_fast_forward", False))),
        "vldb_fast_eval": _bool_text(bool(getattr(args, "vldb_skip_evaluate", False))),
        "task_sampling_timing_valid": _bool_text(bool(parts.get("task_sampling_timing_valid", True))),
        "selected_forecast_timing_valid": _bool_text(bool(parts.get("selected_forecast_timing_valid", True))),
        "evaluate_timing_valid": _bool_text(not bool(getattr(args, "vldb_skip_evaluate", False))),
        "forward_mode": str(parts.get("forward_mode", "runtime_metadata")),
        "timing_note": note,
    }


class Task_Probe_Forward_Select_Model(Baseline_Select_Model):
    """Forward-based calibration selector driven by cached per-window probe rows."""

    def __init__(self, args, model_name, Model_zoo_current):
        super().__init__(args, model_name, Model_zoo_current)
        self._dataset_orders = None
        self._dataset_probe_extra = {}

    def _load_per_window_df(self, model_key: str) -> tuple[pd.DataFrame | None, str]:
        csv_path = resolve_tsfm_csv_path(model_key, self.model_cl_name, "per_window_results.csv")
        if not csv_path.exists():
            return None, f"missing file {csv_path}"
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in {"dataset", "entry", "window_id", "MASE"})
        except Exception as e:
            return None, f"read failed {csv_path}: {type(e).__name__}: {e}"
        if "dataset" not in df.columns or "MASE" not in df.columns:
            return None, f"missing required columns in {csv_path}"
        return df, str(csv_path)

    def _planned_dataset_configs(self) -> list[str]:
        out = []
        med_long = set(str(getattr(self.args, "med_long_datasets", "") or "").split())
        for ds_name in list(getattr(self.args, "all_datasets", []) or []):
            for term in ["short", "medium", "long"]:
                if term in {"medium", "long"} and str(ds_name) not in med_long:
                    continue
                try:
                    _, _, ds_config, _ = self._build_ds_meta(str(ds_name), term)
                    out.append(str(ds_config))
                except Exception:
                    continue
        return sorted(set(out))

    def _invoke_tsfm_per_window_forward(self, model_meta: dict[int, dict], missing_reasons: dict[int, str], dataset: str = "*") -> list[dict]:
        rows = []
        for mid, meta in sorted(model_meta.items()):
            if mid not in missing_reasons:
                continue
            family = str(meta.get("family", ""))
            size_name = str(meta.get("size_name", ""))
            model_key = str(meta.get("model_key", ""))
            start = time.perf_counter()
            cmd = [
                sys.executable,
                "-m",
                "cli.run_model_zoo",
                "--run_mode",
                "zoo",
                "--models",
                family,
                "--size_mode",
                size_name,
                "--fix_context_len",
                "--context_len",
                str(getattr(self.args, "context_len", 512)),
                "--TSFM_results_dir",
                self.model_cl_name,
                "--batch_size",
                str(getattr(self.args, "batch_size", 128)),
                "--forward_seed",
                str(getattr(self.args, "forward_seed", 2025)),
                "--enable_process_metrics",
                "true",
                "--enable_per_window_metrics",
                "true",
                "--vldb_skip_evaluate",
                "true",
                "--skip_saved",
                "--strict_phase_seed",
            ]
            if dataset and str(dataset) != "*":
                cmd.extend(["--only_dataset_config", str(dataset)])
            if bool(getattr(self.args, "quick_test", False)):
                cmd.append("--quick_test")
            print(
                f"[Task_Probe_Forward][probe_forward:fresh] dataset={dataset}, "
                f"model_id={mid}, model={meta.get('abbr', model_key)}({model_key}), "
                f"reason={missing_reasons.get(mid, '')}, cmd={' '.join(cmd)}"
            )
            status = "forward_invoked"
            error = ""
            try:
                env = os.environ.copy()
                src = str(SRC_ROOT)
                current = env.get("PYTHONPATH", "")
                parts = [p for p in current.split(os.pathsep) if p]
                if src not in parts:
                    env["PYTHONPATH"] = os.pathsep.join([src, *parts])
                proc = subprocess.run(cmd, cwd=SRC_ROOT.parent, text=True, capture_output=True, env=env)
                if proc.returncode == 0:
                    status = "forward_success"
                else:
                    status = "forward_failed"
                    err_text = (proc.stderr or proc.stdout or "").strip().replace("\n", " ")
                    error = err_text[:1000]
            except Exception as e:
                status = "forward_failed"
                error = f"{type(e).__name__}: {e}"
            wall_ms = (time.perf_counter() - start) * 1000.0
            print(
                f"[Task_Probe_Forward][probe_forward:fresh] dataset={dataset}, "
                f"model_id={mid}, model={meta.get('abbr', model_key)}({model_key}), "
                f"status={status}, wall_ms={wall_ms:.3f}"
                + (f", error={error[:240]}" if error else "")
            )
            rows.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    "method": "Task-Probe Forward",
                    "route_id": _effective_vldb_route_id(self.args, "Task-Probe Forward"),
                    "stage": str(_effective_vldb_stage(self.args)),
                    "dataset": str(dataset),
                    "sample_repr_num": int(getattr(self.args, "sample_repr_num", 20)),
                    "task_window_sample_strategy": str(getattr(self.args, "task_window_sample_strategy", "legacy")),
                    "step4_sample_reused": "",
                    "step4_task_cache_path": "",
                    "step4_task_cache_shape": "",
                    "model_id": mid,
                    "model_key": model_key,
                    "model_abbr": meta.get("abbr", model_key),
                    "selected_windows": "",
                    "total_windows": "",
                    "selected_window_ids": "",
                    "score_MASE": "",
                    "candidate_forward_ms": "",
                    "candidate_forward_runtime_source": "",
                    "internal_eval_ms": "",
                    "sample_select_ms": "",
                    "task_sample_build_ms": "",
                    "selected_model": "",
                    "probe_cache_status": status,
                    "probe_cache_error": error or missing_reasons.get(mid, ""),
                    "probe_forward_command": " ".join(cmd),
                    "probe_forward_wall_ms": f"{wall_ms:.3f}",
                }
            )
        return rows

    def _build_dataset_orders(self):
        if self._dataset_orders is not None:
            return self._dataset_orders
        per_model = {}
        per_model_source = {}
        model_meta = {}
        missing_reasons = {}
        forward_attempted_mids = set()
        for family, sizes_dict in self.Model_sizes.items():
            for size_name, model_info in sizes_dict.items():
                model_folder = f"{family}_{size_name}"
                model_id = int(model_info["id"])
                model_meta[model_id] = {
                    "model_key": model_folder,
                    "abbr": str(model_info.get("abbreviation", model_folder)),
                    "family": family,
                    "size_name": size_name,
                }
                df, reason = self._load_per_window_df(model_folder)
                if df is None:
                    missing_reasons[model_id] = reason
                    continue
                per_model[model_id] = df
                per_model_source[model_id] = reason
        detail_rows = []
        if missing_reasons:
            print(f"[Task_Probe_Forward] per-window cache missing/incomplete for {len(missing_reasons)} model(s); invoking TSFM forward fallback")
            detail_rows.extend(self._invoke_tsfm_per_window_forward(model_meta, missing_reasons))
            forward_attempted_mids.update(missing_reasons.keys())
            for mid in list(missing_reasons):
                df, reason = self._load_per_window_df(model_meta[mid]["model_key"])
                if df is not None:
                    per_model[mid] = df
                    per_model_source[mid] = reason
                    missing_reasons.pop(mid, None)
                else:
                    missing_reasons[mid] = reason
        if not per_model:
            if detail_rows:
                self._append_task_probe_detail_rows(detail_rows, {})
            raise RuntimeError("[Task_Probe_Forward] missing per_window_results.csv for all current-stage models after forward fallback")

        sample_n = int(getattr(self.args, "sample_repr_num", 20))
        orders = {}
        route_extra = {}
        cached_datasets = set().union(*[set(df["dataset"].dropna().astype(str)) for df in per_model.values()])
        planned_datasets = set(self._planned_dataset_configs())
        datasets = sorted(planned_datasets | cached_datasets)
        for ds_config in datasets:
            dataset_t0 = time.perf_counter()
            cached_task, task_meta, task_cache_path, task_cache_error = _load_step4_task_cache(self.args, ds_config)
            window_ids = _step4_window_ids_from_metadata(task_meta, sample_n)
            if cached_task is None or not window_ids:
                warnings.warn(
                    f"[Task_Probe_Forward] skip dataset={ds_config}: Step4 sample cache metadata is required "
                    f"for exact window alignment; cache={task_cache_path}; error={task_cache_error or 'missing_entry_indices'}"
                )
                continue
            cache_shape = tuple(np.asarray(cached_task).shape)
            task_sample_build_ms, task_sample_build_source = _task_sample_build_ms_from_metadata(self.args, ds_config, task_meta)
            sample_build_valid = np.isfinite(task_sample_build_ms)
            scores = {}
            forward_ms_total = 0.0
            internal_eval_ms_total = 0.0
            sample_select_ms_total = 0.0
            finite_forward = True
            dataset_missing = {}
            for mid, df in list(per_model.items()):
                sub = df[df["dataset"].astype(str).eq(ds_config)]
                if sub.empty:
                    dataset_missing[mid] = f"per_window_results has no dataset={ds_config}"
            retry_missing = {mid: reason for mid, reason in dataset_missing.items() if mid not in forward_attempted_mids}
            if retry_missing:
                print(
                    f"[Task_Probe_Forward] dataset={ds_config} missing per-window rows for "
                    f"{len(retry_missing)} model(s); invoking TSFM forward fallback"
                )
                detail_rows.extend(self._invoke_tsfm_per_window_forward(model_meta, retry_missing, dataset=ds_config))
                forward_attempted_mids.update(retry_missing.keys())
                for mid in list(retry_missing):
                    df, reason = self._load_per_window_df(model_meta[mid]["model_key"])
                    if df is not None:
                        per_model[mid] = df
                        per_model_source[mid] = reason
                    else:
                        dataset_missing[mid] = reason
            sample_build_text = f"{float(task_sample_build_ms):.3f}" if sample_build_valid else "nan"
            print(
                f"[Task_Probe_Forward][sample] dataset={ds_config}, sample_skipped=true, "
                f"cache_path={task_cache_path}, cache_shape={'x'.join(map(str, cache_shape))}, "
                f"sample_repr_num={sample_n}, windows={_fmt_int_list(window_ids, sample_n)}, "
                f"task_sample_build_ms={sample_build_text}, "
                f"task_sample_build_ms_source={task_sample_build_source}"
            )
            candidate_summaries = []
            for mid, df in per_model.items():
                score_t0 = time.perf_counter()
                sub = df[df["dataset"].astype(str).eq(ds_config)].copy()
                if sub.empty:
                    # print(
                    #     f"[Task_Probe_Forward][probe_forward:cache] dataset={ds_config}, "
                    #     f"model_id={mid}, model={model_meta[mid]['abbr']}({model_meta[mid]['model_key']}), "
                    #     f"source={per_model_source.get(mid, 'unknown')}, status=missing_dataset_rows"
                    # )
                    detail_rows.append(
                        {
                            "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                            "method": "Task-Probe Forward",
                            "route_id": _effective_vldb_route_id(self.args, "Task-Probe Forward"),
                            "stage": str(_effective_vldb_stage(self.args)),
                            "dataset": ds_config,
                            "sample_repr_num": sample_n,
                            "task_window_sample_strategy": str(getattr(self.args, "task_window_sample_strategy", "legacy")),
                            "step4_sample_reused": "true",
                            "step4_task_cache_path": task_cache_path,
                            "step4_task_cache_shape": "x".join(map(str, cache_shape)),
                            "model_id": mid,
                            "model_key": model_meta[mid]["model_key"],
                            "model_abbr": model_meta[mid]["abbr"],
                            "selected_windows": "",
                            "total_windows": "",
                            "selected_window_ids": "",
                            "score_MASE": "",
                            "candidate_forward_ms": "",
                            "candidate_forward_runtime_source": "",
                            "internal_eval_ms": "",
                            "sample_select_ms": "",
                            "task_sample_build_ms": "" if not sample_build_valid else f"{float(task_sample_build_ms):.3f}",
                            "probe_cache_status": "missing_dataset_rows",
                            "probe_cache_error": f"per_window_results has no dataset={ds_config}",
                        }
                    )
                    continue
                key_col = "entry" if "entry" in sub.columns else ("window_id" if "window_id" in sub.columns else "")
                if key_col:
                    key_values = pd.to_numeric(sub[key_col], errors="coerce")
                    all_keys = sorted(key_values.dropna().astype(int).unique().tolist())
                    sample_t0 = time.perf_counter()
                    wanted = {int(k) for k in window_ids}
                    keys = [int(k) for k in all_keys if int(k) in wanted]
                    sample_select_ms = (time.perf_counter() - sample_t0) * 1000.0
                    sub = sub[key_values.isin(keys)]
                else:
                    all_keys = list(range(len(sub)))
                    sample_t0 = time.perf_counter()
                    keys = [int(i) for i in window_ids if 0 <= int(i) < len(sub)]
                    sample_select_ms = (time.perf_counter() - sample_t0) * 1000.0
                    sub = sub.iloc[[int(i) for i in keys]]
                if len(keys) == 0 or sub.empty:
                    # print(
                    #     f"[Task_Probe_Forward][probe_forward:cache] dataset={ds_config}, "
                    #     f"model_id={mid}, model={model_meta[mid]['abbr']}({model_meta[mid]['model_key']}), "
                    #     f"source={per_model_source.get(mid, 'unknown')}, key_col={key_col or 'row_index'}, "
                    #     f"status=no_overlap, requested_windows={_fmt_int_list(window_ids, sample_n)}"
                    # )
                    continue
                val = pd.to_numeric(sub["MASE"], errors="coerce").mean()
                score_ms = (time.perf_counter() - score_t0) * 1000.0
                runtime_s, runtime_source = _read_dataset_runtime_seconds(
                    model_meta[mid]["model_key"], self.model_cl_name, ds_config
                )
                total_windows = max(len(all_keys), 1)
                selected_windows = len(keys)
                forward_ms = (
                    runtime_s * 1000.0 * float(selected_windows) / float(total_windows)
                    if np.isfinite(runtime_s)
                    else float("nan")
                )
                if np.isfinite(forward_ms):
                    forward_ms_total += float(forward_ms)
                else:
                    finite_forward = False
                internal_eval_ms_total += float(score_ms)
                sample_select_ms_total += float(sample_select_ms)
                if pd.notna(val):
                    scores[int(mid)] = float(val)
                    forward_text = f"{float(forward_ms):.3f}" if np.isfinite(forward_ms) else "nan"
                    candidate_summaries.append(
                        f"{model_meta[mid]['abbr']}:{float(val):.6g},fw_ms={forward_text}"
                    )
                score_text = f"{float(val):.8f}" if pd.notna(val) else "nan"
                forward_text = f"{float(forward_ms):.3f}" if np.isfinite(forward_ms) else "nan"
                # print(
                #     f"[Task_Probe_Forward][probe_forward:cache] dataset={ds_config}, "
                #     f"model_id={mid}, model={model_meta[mid]['abbr']}({model_meta[mid]['model_key']}), "
                #     f"source={per_model_source.get(mid, 'unknown')}, key_col={key_col or 'row_index'}, "
                #     f"selected_windows={selected_windows}/{total_windows}, "
                #     f"window_ids={_fmt_int_list(keys, sample_n)}, "
                #     f"score_MASE={score_text}, "
                #     f"candidate_forward_ms={forward_text}, "
                #     f"runtime_source={runtime_source}, sample_select_ms={sample_select_ms:.3f}, "
                #     f"internal_eval_ms={score_ms:.3f}"
                # )
                detail_rows.append(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                        "method": "Task-Probe Forward",
                        "route_id": _effective_vldb_route_id(self.args, "Task-Probe Forward"),
                        "stage": str(_effective_vldb_stage(self.args)),
                        "dataset": ds_config,
                        "sample_repr_num": sample_n,
                        "task_window_sample_strategy": str(getattr(self.args, "task_window_sample_strategy", "legacy")),
                        "step4_sample_reused": "true",
                        "step4_task_cache_path": task_cache_path,
                        "step4_task_cache_shape": "x".join(map(str, cache_shape)),
                        "model_id": mid,
                        "model_key": model_meta[mid]["model_key"],
                        "model_abbr": model_meta[mid]["abbr"],
                        "selected_windows": selected_windows,
                        "total_windows": total_windows,
                        "selected_window_ids": " ".join(map(str, keys)),
                        "score_MASE": "" if pd.isna(val) else f"{float(val):.8f}",
                        "candidate_forward_ms": "" if not np.isfinite(forward_ms) else f"{float(forward_ms):.3f}",
                        "candidate_forward_runtime_source": runtime_source,
                        "internal_eval_ms": f"{float(score_ms):.3f}",
                        "sample_select_ms": f"{float(sample_select_ms):.3f}",
                        "task_sample_build_ms": "" if not sample_build_valid else f"{float(task_sample_build_ms):.3f}",
                        "probe_cache_status": "cache_hit",
                        "probe_cache_error": "",
                    }
                )
            if scores:
                ordered = [mid for mid, _ in sorted(scores.items(), key=lambda kv: kv[1])]
                orders[ds_config] = ordered
                selected_mid = int(ordered[0])
                route_ms = (
                    (float(task_sample_build_ms) if sample_build_valid else 0.0)
                    + sample_select_ms_total
                    + forward_ms_total
                    + internal_eval_ms_total
                )
                route_extra[ds_config] = {
                    "row": _baseline_route_row(
                        self.args,
                        "Task-Probe Forward",
                        _dataset_config_to_name(ds_config),
                        ordered,
                        zoo_size=len(model_meta),
                        route_ms=route_ms,
                        note=(
                            "Task-Probe Forward uses cached per-window TSFM metrics for calibration; "
                            "window ids come from TSRouter Step4 task-repr sample cache metadata. "
                            f"candidate_forward=sum(per-model dataset runtime x selected_windows/total_windows), "
                            f"sample_repr_num={sample_n}, task_cache={task_cache_path}, "
                            f"task_sample_build_ms_source={task_sample_build_source}."
                        ),
                        task_sampling_ms=(float(task_sample_build_ms) if sample_build_valid else 0.0) + sample_select_ms_total,
                        index_lookup_ms=forward_ms_total,
                        rank_ms=internal_eval_ms_total,
                        timing_valid=finite_forward and sample_build_valid,
                        task_sampling_timing_valid=sample_build_valid,
                        selected_forecast_timing_valid=True,
                        forward_mode="candidate_calibration_forward_runtime_metadata",
                        cache_mode="per_window_metric_cache",
                        cache_hit=True,
                    ),
                    "forward_ms_total": forward_ms_total,
                    "internal_eval_ms_total": internal_eval_ms_total,
                    "sample_select_ms_total": sample_select_ms_total,
                    "task_sample_build_ms": task_sample_build_ms if sample_build_valid else np.nan,
                    "selected_mid": selected_mid,
                    "build_ms": (time.perf_counter() - dataset_t0) * 1000.0,
                }
                print(
                    f"[Task_Probe_Forward][probe_results] dataset={ds_config}, "
                    f"candidates={len(scores)}/{len(model_meta)}, "
                    f"results={'; '.join(candidate_summaries)}, "
                    f"order={[model_meta[mid]['abbr'] for mid in ordered]}, "
                    f"selected={model_meta[selected_mid]['abbr']}, "
                    f"candidate_forward={forward_ms_total / 1000.0:.3f}s, "
                    f"internal_eval={internal_eval_ms_total / 1000.0:.3f}s, "
                    f"sample_select={sample_select_ms_total:.3f}ms"
                )
            else:
                fallback_order = sorted(model_meta)
                orders[ds_config] = fallback_order
                route_extra[ds_config] = {
                    "row": _baseline_route_row(
                        self.args,
                        "Task-Probe Forward",
                        _dataset_config_to_name(ds_config),
                        fallback_order,
                        zoo_size=len(model_meta),
                        route_ms=(
                            (float(task_sample_build_ms) if sample_build_valid else 0.0)
                            + sample_select_ms_total
                            + forward_ms_total
                            + internal_eval_ms_total
                        ),
                        note=(
                            "Task-Probe Forward has no valid calibration score for this dataset after cache lookup "
                            "and forward fallback; using deterministic model-id fallback order so the selector run "
                            "can finish. This row is not paper-final for Task-Probe quality/efficiency."
                        ),
                        task_sampling_ms=(float(task_sample_build_ms) if sample_build_valid else 0.0) + sample_select_ms_total,
                        index_lookup_ms=forward_ms_total,
                        rank_ms=internal_eval_ms_total,
                        timing_valid=False,
                        task_sampling_timing_valid=sample_build_valid,
                        selected_forecast_timing_valid=False,
                        forward_mode="candidate_calibration_missing",
                        cache_mode="per_window_metric_cache_missing",
                        cache_hit=False,
                    ),
                    "forward_ms_total": forward_ms_total,
                    "internal_eval_ms_total": internal_eval_ms_total,
                    "sample_select_ms_total": sample_select_ms_total,
                    "task_sample_build_ms": task_sample_build_ms if sample_build_valid else np.nan,
                    "selected_mid": fallback_order[0] if fallback_order else None,
                    "build_ms": (time.perf_counter() - dataset_t0) * 1000.0,
                }
                detail_rows.append(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                        "method": "Task-Probe Forward",
                        "route_id": _effective_vldb_route_id(self.args, "Task-Probe Forward"),
                        "stage": str(_effective_vldb_stage(self.args)),
                        "dataset": ds_config,
                        "sample_repr_num": sample_n,
                        "task_window_sample_strategy": str(getattr(self.args, "task_window_sample_strategy", "legacy")),
                        "step4_sample_reused": "true",
                        "step4_task_cache_path": task_cache_path,
                        "step4_task_cache_shape": "x".join(map(str, cache_shape)),
                        "model_id": "",
                        "model_key": "",
                        "model_abbr": "",
                        "selected_windows": "",
                        "total_windows": "",
                        "selected_window_ids": " ".join(map(str, window_ids)),
                        "score_MASE": "",
                        "candidate_forward_ms": "",
                        "candidate_forward_runtime_source": "",
                        "internal_eval_ms": "",
                        "sample_select_ms": "",
                        "task_sample_build_ms": "" if not sample_build_valid else f"{float(task_sample_build_ms):.3f}",
                        "selected_model": "",
                        "probe_cache_status": "no_calibration_scores",
                        "probe_cache_error": "no current-stage model produced valid per-window scores for this dataset after fallback",
                    }
                )
                print(
                    f"[Task_Probe_Forward] dataset={ds_config}, no valid calibration scores after fallback; "
                    f"using fallback_order={fallback_order}"
                )
        self._append_task_probe_detail_rows(detail_rows, route_extra)
        self._dataset_probe_extra = route_extra
        self._dataset_orders = orders
        return orders

    def _append_task_probe_detail_rows(self, detail_rows: list[dict], route_extra: dict) -> None:
        if not detail_rows:
            return
        detail_path = _selector_log_dir(self.args) / "baseline_forward_detail_log.csv"
        for row in detail_rows:
            ds = row.get("dataset", "")
            selected_mid = route_extra.get(ds, {}).get("selected_mid")
            if "selected_model" not in row:
                row["selected_model"] = "true" if selected_mid is not None and int(row["model_id"]) == int(selected_mid) else "false"
        _append_csv_rows(
            detail_path,
            detail_rows,
            [
                "timestamp_utc",
                "method",
                "route_id",
                "stage",
                "dataset",
                "sample_repr_num",
                "task_window_sample_strategy",
                "step4_sample_reused",
                "step4_task_cache_path",
                "step4_task_cache_shape",
                "model_id",
                "model_key",
                "model_abbr",
                "selected_windows",
                "total_windows",
                "selected_window_ids",
                "score_MASE",
                "candidate_forward_ms",
                "candidate_forward_runtime_source",
                "internal_eval_ms",
                "sample_select_ms",
                "task_sample_build_ms",
                "selected_model",
                "probe_cache_status",
                "probe_cache_error",
                "probe_forward_command",
                "probe_forward_wall_ms",
            ],
        )
        print(f"[Task_Probe_Forward] detail saved: {detail_path}")

    def _get_select_strategy(self, dataset):
        orders = self._build_dataset_orders()

        def select_strategy(dataset_name=None, test_data_input=None):
            ds_config = _dataset_name_to_config(dataset_name)
            if ds_config not in orders:
                raise ValueError(f"[Task_Probe_Forward] no calibration rows for dataset={ds_config}")
            model_order = list(orders[ds_config])
            for mid in range(self.args.current_zoo_num):
                if mid not in model_order:
                    model_order.append(mid)
            extra = self._dataset_probe_extra.get(ds_config, {})
            return model_order, self.args.ensemble_size, {
                "vldb_route_latency_row": dict(extra.get("row", {}) or {}),
                "task_probe_forward_ms_total": extra.get("forward_ms_total", np.nan),
                "task_probe_internal_eval_ms_total": extra.get("internal_eval_ms_total", np.nan),
            }

        return select_strategy


class Real_Select_Model(Baseline_Select_Model):
    'TSRouter runtime message.'

    def __init__(self, args, model_name, Model_zoo_current):
        super().__init__(args, model_name, Model_zoo_current)
        self._dataset_fixed_orders = None
        self.real_order_metric = getattr(args, "real_order_metric", "sMAPE")

                                                                 
    def _build_dataset_fixed_orders(self):
        'TSRouter runtime message.'
        if self._dataset_fixed_orders is not None:
            return self._dataset_fixed_orders

        all_frames = []

                                           
        for family, sizes_dict in self.Model_sizes.items():
            for size_name, model_info in sizes_dict.items():
                model_folder = f"{family}_{size_name}"
                model_cl_name = self.model_cl_name
                csv_path = resolve_tsfm_csv_path(model_folder, model_cl_name, "all_results.csv")
                if not csv_path.exists():
                    warnings.warn(f"TSRouter runtime message: {csv_path}")
                    continue

                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    warnings.warn(f"TSRouter runtime message: {csv_path}: {e}")
                    continue

                if self.real_order_metric not in df.columns:
                    warnings.warn(f"[Real_Select] {csv_path}TSRouter runtime message: {self.real_order_metric}TSRouter runtime message: ")
                    continue

                model_abbr = model_info["abbreviation"]

                tmp = df[["dataset", self.real_order_metric]].copy()
                tmp[self.real_order_metric] = pd.to_numeric(tmp[self.real_order_metric], errors="coerce")
                tmp = tmp.dropna(subset=["dataset", self.real_order_metric])
                tmp = tmp.drop_duplicates(subset=["dataset"], keep="last")
                tmp["model_abbr"] = model_abbr
                if not tmp.empty:
                    all_frames.append(tmp)

        if not all_frames:
            raise RuntimeError('TSRouter runtime message.')

        baseline_df = pd.concat(all_frames, ignore_index=True)

        dataset_fixed_orders = {}

        for ds_config, ds_group in baseline_df.groupby("dataset"):
                                              
            metric_by_model = (
                ds_group.groupby("model_abbr")[self.real_order_metric]
                .mean()
                .dropna()
            )
            if metric_by_model.empty:
                continue

                                
            ordered_abbrs = list(metric_by_model.sort_values(ascending=True).index)

                                           
            model_ids = []
            for abbr in ordered_abbrs:
                if abbr not in self.abbr_to_id:
                    warnings.warn(f"TSRouter runtime message: {abbr}TSRouter runtime message: ")
                    continue
                model_ids.append(int(self.abbr_to_id[abbr]))

            if model_ids:
                dataset_fixed_orders[ds_config] = model_ids

        if not dataset_fixed_orders:
            raise RuntimeError('TSRouter runtime message.')

        self._dataset_fixed_orders = dataset_fixed_orders
        return dataset_fixed_orders

                                              
    def _get_select_strategy(self, dataset):
        'TSRouter runtime message.'
        dataset_fixed_orders = self._build_dataset_fixed_orders()

        def select_strategy(dataset_name=None, test_data_input=None):
            if dataset_name is None:
                raise ValueError('TSRouter runtime message.')

            try:
                ds_key, ds_freq, term = dataset_name.rsplit("_", 2)
            except ValueError:
                raise ValueError(f"TSRouter runtime message: {dataset_name}")

            ds_config = f"{ds_key}/{ds_freq}/{term}"

            if ds_config not in dataset_fixed_orders:
                raise ValueError(
                    f"[Real_Select] ds_config={ds_config}TSRouter runtime message: "
                    f"TSRouter runtime message: {ds_config}TSRouter runtime message: "
                )

            fixed_model_order = dataset_fixed_orders[ds_config]

            print(
                f"[Real_Select] sort_metric={self.args.real_order_metric}) for dataset_name={dataset_name}: "
                f"{fixed_model_order}"
            )

            ensemble_size = self.args.ensemble_size
            return fixed_model_order, ensemble_size

        return select_strategy


class Real_Channel_Select_Model(Baseline_Select_Model):
    """Per-channel oracle selector built from cached per-channel metrics."""

    def _get_select_strategy(self, dataset):
        def select_strategy(dataset_name=None, test_data_input=None):
            if dataset_name is None:
                raise ValueError("[Real_Channel_Select] need dataset_name")
            try:
                ds_key, ds_freq, term = dataset_name.rsplit("_", 2)
            except ValueError:
                raise ValueError(f"[Real_Channel_Select] unexpected dataset_name: {dataset_name}")
            ds_config = f"{ds_key}/{ds_freq}/{term}"
            metric = str(getattr(self.args, "real_order_metric", "MASE"))
            cache_path = real_channel_rank_cache_path(
                current_zoo_num=int(getattr(self.args, "current_zoo_num", 0)),
                zoo_total_num=int(getattr(self.args, "zoo_total_num", 0)),
                rank_metric=metric,
                model_cl_name=self.model_cl_name,
            )
            matrix, model_ids, valid_channels = load_per_channel_error_matrix(
                self.Model_sizes,
                ds_config,
                model_cl_name=self.model_cl_name,
                rank_metric=metric,
                require_complete=True,
            )
            orders = channel_orders_from_error_matrix(matrix, model_ids, valid_channels)
            save_channel_rank_orders(cache_path, ds_config, metric, orders)
            if not orders:
                raise FileNotFoundError(
                    f"[Real_Channel_Select] missing per-channel oracle cache for {ds_config}; "
                    "run zoo baselines with --enable_process_metrics first."
                )
            expected_model_count = len(model_ids)
            bad_orders = {
                ch: order
                for ch, order in orders.items()
                if len(order) < expected_model_count
            }
            if bad_orders:
                raise FileNotFoundError(
                    f"[Real_Channel_Select] incomplete rank orders for {ds_config}, "
                    f"metric={metric}, bad_channels={bad_orders}"
                )
            channels = sorted(orders.keys())
            max_rank = max(len(orders[ch]) for ch in channels)
            selected_model_list_2d = np.full((max_rank, len(channels)), -1, dtype=np.int64)
            for cpos, ch in enumerate(channels):
                order = orders[ch]
                selected_model_list_2d[: len(order), cpos] = order

            selected_models_per_channel = selected_model_list_2d[0].astype(int)
            model_order = []
            for mid in selected_models_per_channel.tolist():
                mid = int(mid)
                if mid >= 0 and mid not in model_order:
                    model_order.append(mid)
            for mid in model_ids:
                mid = int(mid)
                if mid not in model_order:
                    model_order.append(mid)
            selected_unique = sorted(set(int(x) for x in selected_models_per_channel.tolist()))
            print(
                f"[Real_Channel_Select] dataset={ds_config}, metric={metric}, channels={len(channels)}, "
                f"channel_top1_head={selected_models_per_channel[:10].tolist()}, "
                f"channel_top1_unique={selected_unique}, compatibility_order={model_order[:10]}"
            )
            extra = {
                "selected_model_list_2d": selected_model_list_2d,
                "selected_models_per_channel": selected_models_per_channel,
                "model_order": model_order,
            }
            return model_order, self.args.ensemble_size, extra

        return select_strategy

    def get_predictor(self, dataset, batch_size):
        self.args.test_pred_len = dataset.prediction_length
        return Real_Channel_Select_Predictor(
            args=self.args,
            Model_sizes=self.Model_sizes,
            prediction_length=dataset.prediction_length,
            channels=dataset.target_dim,
            windows=dataset.windows,
            model_cl_name=self.model_cl_name,
            select_strategy=self._get_select_strategy(dataset),
        )


class Real_Channel_Select_Predictor(Baseline_Select_Predictor):
    def predict(self, test_data_input: List[dict], dataset_name, model_order: Optional[List[int]] = None) -> List:
        loaded = self._load_predictions(dataset_name)
        if self.use_sample_distribution:
            samples_dict, template = loaded
        else:
            preds_dict, template = loaded

        if model_order is None:
            model_order, ensemble_size, extra = self.select_strategy(dataset_name, test_data_input)
        else:
            _, ensemble_size, extra = self.select_strategy(dataset_name, test_data_input)
        self.last_selector_extra = extra

        if self.use_sample_distribution:
            samples = self._create_channel_oracle_distribution(samples_dict, model_order, ensemble_size, extra)
            return numpy_samples_to_gluonts(samples, template), model_order
        preds = self._create_channel_oracle_point(preds_dict, model_order, ensemble_size, extra)
        return numpy_to_gluonts(preds, template), model_order

    def _create_channel_oracle_point(self, preds_dict, model_order, ensemble_size, extra):
        if ensemble_size != 1:
            return self._create_ensemble(preds_dict, model_order, ensemble_size)
        chosen = np.asarray(extra["selected_models_per_channel"], dtype=np.int64)
        out = []
        for c in range(self.channels):
            out.append(preds_dict[int(chosen[c])][:, :, c])
        return np.stack(out, axis=-1)

    def _create_channel_oracle_distribution(self, samples_dict, model_order, ensemble_size, extra):
        rng = np.random.RandomState(getattr(self.args, "search_seed", 2025))
        if ensemble_size != 1:
            topk = [int(i) for i in model_order[:ensemble_size]]
            picked = []
            points = []
            for mid in topk:
                arr = samples_dict[mid]
                points.append(np.median(arr, axis=1))
                picked.append(self._resample_samples(arr, self.samples_per_model, rng))
            mix = np.concatenate(picked, axis=1)
            model_points = np.stack(points, axis=-1)
            if self.args.ensemble_agg == "mean":
                legacy = np.mean(model_points, axis=-1)
            else:
                legacy = np.median(model_points, axis=-1)
            return mix + (legacy - np.median(mix, axis=1))[:, None, :, :]
        chosen = np.asarray(extra["selected_models_per_channel"], dtype=np.int64)
        channel_samples = []
        legacy_points = []
        for c in range(self.channels):
            arr = samples_dict[int(chosen[c])]
            arr_rs = self._resample_samples(arr, self.samples_per_model, rng)
            channel_samples.append(arr_rs[:, :, :, c])
            legacy_points.append(np.median(arr[:, :, :, c], axis=1))
        mix = np.stack(channel_samples, axis=-1)
        legacy = np.stack(legacy_points, axis=-1)
        mix = mix + (legacy - np.median(mix, axis=1))[:, None, :, :]
        return mix


# ========================
                         
# ========================
TRUE_ORDER_ABBR = {
    "sMAPE_Rank": ["Chr.2", "TiRex", "Kai.10", "Flo.r1", "TFM.25", "Moi2.S", "Chr.bT", "Toto", "Moi.S"],
    "sMAPE": ["Kai.10", "Chr.2", "Moi2.S", "Flo.r1", "TiRex", "TFM.25", "Chr.bT", "Toto", "Moi.S"],
    "MASE":  ["Moi2.S", "Chr.2", "Flo.r1", "TiRex", "Toto", "TFM.25", "Kai.10", "Chr.bT", "Moi.S"],
    "CRPS":  ["TiRex", "Moi2.S", "TFM.25", "Flo.r1", "Toto", "Chr.bT", "Chr.2", "Kai.10", "Moi.S"],
}


class CurrentBest_FixedOrder_Select_Model(Baseline_Select_Model):
    'TSRouter runtime message.'

    def __init__(self, args, model_name, Model_zoo_current, metric_name: str):
        super().__init__(args, model_name, Model_zoo_current)
        self.metric_name = metric_name

        if metric_name not in TRUE_ORDER_ABBR:
            raise ValueError(f"TSRouter runtime message: {metric_name}")

                                 
        self.fixed_model_order = []
        for abbr in TRUE_ORDER_ABBR[metric_name]:
            if abbr not in self.abbr_to_id:
                warnings.warn(
                    f"[CurrentBest-{metric_name}TSRouter runtime message: {abbr}TSRouter runtime message: "
                )
                continue
            self.fixed_model_order.append(int(self.abbr_to_id[abbr]))

        if not self.fixed_model_order:
            raise RuntimeError(
                f"[CurrentBest-{metric_name}TSRouter runtime message: "
            )

    def _get_select_strategy(self, dataset):
        def select_strategy(dataset_name=None, test_data_input=None):
            ensemble_size = self.args.ensemble_size

            print(
                f"[CurrentBest-{self.metric_name}] fixed order = {self.fixed_model_order}"
            )

            return self.fixed_model_order, ensemble_size

        return select_strategy


class Current_best_sMAPE_Rank_Select_Model(CurrentBest_FixedOrder_Select_Model):
    def __init__(self, args, model_name, Model_zoo_current):
        super().__init__(args, model_name, Model_zoo_current, metric_name="sMAPE_Rank")

class Current_best_sMAPE_Select_Model(CurrentBest_FixedOrder_Select_Model):
    def __init__(self, args, model_name, Model_zoo_current):
        super().__init__(args, model_name, Model_zoo_current, metric_name="sMAPE")


class Current_best_MASE_Select_Model(CurrentBest_FixedOrder_Select_Model):
    def __init__(self, args, model_name, Model_zoo_current):
        super().__init__(args, model_name, Model_zoo_current, metric_name="MASE")


class Current_best_CRPS_Select_Model(CurrentBest_FixedOrder_Select_Model):
    def __init__(self, args, model_name, Model_zoo_current):
        super().__init__(args, model_name, Model_zoo_current, metric_name="CRPS")


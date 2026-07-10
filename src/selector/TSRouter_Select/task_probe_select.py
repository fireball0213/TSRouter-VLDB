from __future__ import annotations

import ast
import copy
import csv
import importlib
import json
import os
import pickle
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pandas as pd

from config.model_zoo_config import All_sorted_model_names, Model_abbrev_map, MULTIVAR_TSFM_PREFIXES
from utils.data import Dataset
from utils.io_lock import file_lock
from utils.path_utils import (
    get_auto_cl_mode,
    get_auto_cl_profile_by_name,
    get_gift_eval_task_repr_cache_path,
    normalize_auto_cl_args,
    normalize_route_family_mode,
)
from utils.project_paths import BASELINE_CSV_ROOT, resolve_checkpoint_path
from utils.tsrouter_metrics import compute_per_window_metric_rows


TASK_PROBE_SELECT_DIRNAME = "Task_probe_Select"
TASK_PROBE_SELECT_FORWARD_SUMMARY = "forward_summary.csv"
TASK_PROBE_SELECT_RANK_SUMMARY = "rank_summary.csv"
TASK_PROBE_EVAL_PROTOCOL = "within_window_half_v1"

TASK_PROBE_PROFILE_COLUMNS = [
    "auto_cl_mode",
    "adaptive_profile",
    "adaptive_context_len_avg",
    "adaptive_pred_len",
    "resolved_eval_cl",
    "rank_truth_cl",
    "repr_input_dim",
    "repr_output_dim",
    "repr_sub_pred_len",
    "repr_source_exact_length",
    "task_sample_cache_path",
    "eval_cl_fallback_used",
]

TASK_PROBE_SELECT_WINDOW_COLUMNS = [
    "timestamp_utc",
    "cache_stem",
    "cache_path",
    "stage",
    "dataset",
    "sample_repr_num",
    "task_window_sample_strategy",
    "sample_repr_ratio",
    "task_sample_version",
    "route_family_mode",
    "search_seed",
    "repr_scale_protocol",
    "task_probe_eval_protocol",
    *TASK_PROBE_PROFILE_COLUMNS,
    "task_probe_cache_window_len",
    "task_probe_context_len",
    "task_probe_prediction_len",
    "model_id",
    "model_key",
    "model_abbr",
    "local_entry",
    "step4_sample_idx",
    "step4_entry_idx",
    "segment_start_idx",
    "series_id",
    "item_id",
    "forecast_start",
    "input_start",
    "channel",
    "window_id",
    "pred_len",
    "mase_lag",
    "METRIC_IMPL",
    "MASE",
    "sMAPE",
    "CRPS",
    "MASE_NUM",
    "MASE_DEN",
    "SMAPE_NUM",
    "SMAPE_DEN",
    "CRPS_NUM",
    "CRPS_DEN",
]

TASK_PROBE_SELECT_FORWARD_COLUMNS = [
    "timestamp_utc",
    "cache_stem",
    "cache_path",
    "stage",
    "dataset",
    "sample_repr_num",
    "task_window_sample_strategy",
    "sample_repr_ratio",
    "task_sample_version",
    "route_family_mode",
    "search_seed",
    "repr_scale_protocol",
    "task_probe_eval_protocol",
    *TASK_PROBE_PROFILE_COLUMNS,
    "task_probe_cache_window_len",
    "task_probe_context_len",
    "task_probe_prediction_len",
    "model_id",
    "model_key",
    "model_abbr",
    "selected_windows",
    "metric_rows",
    "model_load_ms",
    "forward_ms",
    "sample_forward_ms",
    "evaluate_ms",
    "glouts_MASE",
    "glouts_sMAPE",
    "glouts_CRPS",
    "status",
    "error",
    "per_model_csv",
]

TASK_PROBE_SELECT_RANK_COLUMNS = [
    "timestamp_utc",
    "cache_stem",
    "cache_path",
    "stage",
    "dataset",
    "sample_repr_num",
    "task_window_sample_strategy",
    "sample_repr_ratio",
    "task_sample_version",
    "route_family_mode",
    "search_seed",
    "repr_scale_protocol",
    "task_probe_eval_protocol",
    *TASK_PROBE_PROFILE_COLUMNS,
    "probe_model_num",
    "candidate_model_ids",
    "candidate_model_keys",
    "tsrouter_model_order",
    "model_rank_mase",
    "model_rank_crps",
    "model_rank_mase_keys",
    "model_rank_crps_keys",
    "selected_model_id_mase",
    "selected_model_key_mase",
    "selected_model_id_crps",
    "selected_model_key_crps",
    "sample_seconds",
    "sample_timing_source",
    "forward_seconds",
    "rank_seconds",
    "route_final_seconds",
    "status",
    "missing_model_ids",
    "error",
]


class _TaskProbeTestData(SimpleNamespace):
    def __len__(self) -> int:
        return len(self.label)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t", "on"}


def task_probe_select_enabled(args: Any) -> bool:
    return _truthy(getattr(args, "mix_route", False))


def task_probe_eval_protocol(args: Any) -> str:
    return TASK_PROBE_EVAL_PROTOCOL


def task_probe_select_root(args: Any) -> Path:
    raw = str(getattr(args, "task_probe_select_output_dir", "") or "").strip()
    if raw:
        return Path(raw)
    return BASELINE_CSV_ROOT / "selectors" / TASK_PROBE_SELECT_DIRNAME


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _csv_safe(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and not np.isfinite(value):
            return ""
    except Exception:
        pass
    return str(value)


def _format_float(value: Any, digits: int = 9) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if not np.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def _sample_config_row(args: Any) -> dict[str, Any]:
    return {
        "sample_repr_num": int(getattr(args, "sample_repr_num", 0) or 0),
        "task_window_sample_strategy": str(getattr(args, "task_window_sample_strategy", "legacy") or "legacy"),
        "sample_repr_ratio": str(getattr(args, "sample_repr_ratio", 0) or 0),
        "task_sample_version": str(getattr(args, "task_sample_version", "")),
        "route_family_mode": normalize_route_family_mode(
            getattr(args, "route_family_mode", "default")
        ),
        "search_seed": str(getattr(args, "search_seed", "")),
        "repr_scale_protocol": str(getattr(args, "repr_scale_protocol", "")),
        "task_probe_eval_protocol": task_probe_eval_protocol(args),
        "auto_cl_mode": get_auto_cl_mode(args),
        "adaptive_profile": str(getattr(args, "adaptive_profile", "") or ""),
        "adaptive_context_len_avg": _format_float(
            getattr(args, "adaptive_context_len_avg", np.nan), 9
        ),
        "adaptive_pred_len": _format_float(getattr(args, "adaptive_pred_len", np.nan), 9),
        "resolved_eval_cl": str(getattr(args, "resolved_eval_cl", "") or ""),
        "rank_truth_cl": str(getattr(args, "rank_truth_cl", "") or ""),
        "repr_input_dim": int(getattr(args, "repr_input_dim", 0) or 0),
        "repr_output_dim": int(getattr(args, "repr_output_dim", 0) or 0),
        "repr_sub_pred_len": int(getattr(args, "repr_sub_pred_len", 0) or 0),
        "repr_source_exact_length": int(
            getattr(args, "repr_source_exact_length", 0) or 0
        ),
        "task_sample_cache_path": str(
            getattr(args, "task_sample_cache_path", "") or ""
        ),
        "eval_cl_fallback_used": (
            "true" if _truthy(getattr(args, "eval_cl_fallback_used", False)) else "false"
        ),
    }


def task_probe_select_cache_path(args: Any) -> Path:
    search_context_len = int(getattr(args, "repr_input_dim", getattr(args, "context_len", 512)) or 512)
    return Path(get_gift_eval_task_repr_cache_path(args, search_context_len=search_context_len))


def task_probe_select_cache_stem(args: Any) -> str:
    return task_probe_select_cache_path(args).stem


def _read_task_cache(args: Any) -> tuple[dict, dict, Path]:
    cache_path = task_probe_select_cache_path(args)
    with cache_path.open("rb") as f:
        cache = pickle.load(f)
    if not isinstance(cache, dict):
        raise ValueError(f"Task-probe Select cache must be a dict: {cache_path}")
    meta_path = Path(f"{cache_path}.meta.json")
    meta = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            meta = loaded
    return cache, meta, cache_path


def _model_records(model_sizes: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, sizes in model_sizes.items():
        for size, info in sizes.items():
            rec = dict(info)
            rec["family"] = str(family)
            rec["size"] = str(size)
            rec["model_key"] = f"{family}_{size}"
            rec["model_id"] = int(info.get("id", len(rows)))
            rec["model_abbr"] = str(info.get("abbreviation", Model_abbrev_map.get(rec["model_key"], rec["model_key"])))
            rows.append(rec)
    return sorted(rows, key=lambda item: int(item["model_id"]))


def _complete_candidate_ids(model_order: Iterable[Any] | None, current_model_num: int, limit: int) -> list[int]:
    ids: list[int] = []
    for raw in _as_list(model_order):
        try:
            mid = int(raw)
        except Exception:
            continue
        if 0 <= mid < current_model_num and mid not in ids:
            ids.append(mid)
    for mid in range(int(current_model_num)):
        if mid not in ids:
            ids.append(mid)
    if limit > 0:
        ids = ids[: int(limit)]
    return ids


def _as_list(values: Iterable[Any] | None) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        return values.reshape(-1).tolist()
    if isinstance(values, (list, tuple)):
        return list(values)
    try:
        return list(values)
    except Exception:
        return [values]


def _shift_start(start: Any, offset: int | None) -> Any:
    try:
        if offset is None:
            return start
        return start + int(offset)
    except Exception:
        return start


def _dict_with(entry: Any, **updates: Any) -> dict[str, Any]:
    out = dict(entry) if isinstance(entry, dict) else {}
    out.update(updates)
    return out


def _build_sample_dataset(
    *,
    ds_name: str,
    term: str,
    ds_config: str,
    ds_freq: str,
    model_key: str,
    cache: dict,
    meta_all: dict,
    args: Any,
) -> Any:
    arr = cache.get(ds_config)
    if arr is None:
        arr = cache.get(str(ds_config).replace("_", "/", 2))
    if arr is None:
        raise FileNotFoundError(f"missing Step4 task sample cache for dataset={ds_config}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Step4 task sample cache expects (K,T,C), got {arr.shape} for {ds_config}")

    meta = meta_all.get(ds_config) or meta_all.get(str(ds_config).replace("_", "/", 2)) or {}
    entry_indices = meta.get("entry_indices", []) if isinstance(meta, dict) else []
    segment_starts = meta.get("segment_start_indices", []) if isinstance(meta, dict) else []
    if not isinstance(entry_indices, list):
        entry_indices = []
    if not isinstance(segment_starts, list):
        segment_starts = []

    raw_dataset = Dataset(name=ds_name, term=term, to_univariate=False)
    raw_inputs = list(raw_dataset.test_data.input)
    raw_labels = list(raw_dataset.test_data.label)
    if not raw_inputs:
        raise ValueError(f"empty raw test data for dataset={ds_config}")

    k, _t, channels = arr.shape
    cache_window_len = int(arr.shape[1])
    if cache_window_len < 2:
        raise ValueError(f"Task-probe Select requires cache window length >= 2, got {cache_window_len}")
    eval_protocol = task_probe_eval_protocol(args)
    probe_context_len = cache_window_len // 2
    probe_pred_len = cache_window_len - probe_context_len
    prefix = str(model_key).split("_", 1)[0].lower()
    use_multivar = prefix in MULTIVAR_TSFM_PREFIXES
    inputs: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    step4_entry_indices: list[int] = []
    step4_sample_indices: list[int] = []
    step4_channel_indices: list[Any] = []
    step4_segment_starts: list[Any] = []

    for sample_idx in range(k):
        original_idx = int(entry_indices[sample_idx]) if sample_idx < len(entry_indices) else int(sample_idx)
        original_idx = original_idx % len(raw_inputs)
        segment_start = segment_starts[sample_idx] if sample_idx < len(segment_starts) else None
        if segment_start is not None:
            try:
                segment_start = int(segment_start)
            except Exception:
                segment_start = None
        raw_input = raw_inputs[original_idx]
        raw_label = raw_labels[original_idx] if original_idx < len(raw_labels) else raw_input
        cache_t_c = np.asarray(arr[sample_idx], dtype=np.float32)
        input_start = _shift_start(raw_input.get("start") if isinstance(raw_input, dict) else None, segment_start)
        label_start = _shift_start(input_start, probe_context_len)
        context_t_c = cache_t_c[:probe_context_len, :]
        label_c_t = cache_t_c[probe_context_len:, :].T

        if use_multivar or channels == 1:
            target = context_t_c.T if channels > 1 else context_t_c[:, 0]
            label_target = label_c_t if channels > 1 else label_c_t[0]
            item_id = f"{ds_config}#{sample_idx}"
            inputs.append(_dict_with(raw_input, target=target, start=input_start, item_id=item_id))
            labels.append(_dict_with(raw_label, target=label_target, start=label_start, item_id=item_id))
            step4_entry_indices.append(int(original_idx))
            step4_sample_indices.append(int(sample_idx))
            step4_channel_indices.append(0 if channels == 1 else "")
            step4_segment_starts.append("" if segment_start is None else int(segment_start))
            continue

        for ch in range(channels):
            item_id = f"{ds_config}#{sample_idx}_dim{ch}"
            inputs.append(_dict_with(raw_input, target=context_t_c[:, ch], start=input_start, item_id=item_id))
            labels.append(
                _dict_with(
                    raw_label,
                    target=label_c_t[min(ch, label_c_t.shape[0] - 1)],
                    start=label_start,
                    item_id=item_id,
                )
            )
            step4_entry_indices.append(int(original_idx))
            step4_sample_indices.append(int(sample_idx))
            step4_channel_indices.append(int(ch))
            step4_segment_starts.append("" if segment_start is None else int(segment_start))

    return SimpleNamespace(
        name=ds_config,
        freq=str(ds_freq),
        prediction_length=int(probe_pred_len),
        target_dim=int(channels if use_multivar else 1),
        past_feat_dynamic_real_dim=0,
        windows=len(inputs),
        test_data=_TaskProbeTestData(input=inputs, label=labels),
        step4_entry_indices=step4_entry_indices,
        step4_sample_indices=step4_sample_indices,
        step4_channel_indices=step4_channel_indices,
        step4_segment_start_indices=step4_segment_starts,
        step4_cache_shape=list(arr.shape),
        task_probe_eval_protocol=eval_protocol,
        task_probe_cache_window_len=int(cache_window_len),
        task_probe_context_len=int(probe_context_len),
        task_probe_prediction_len=int(probe_pred_len),
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()


def _auto_cl_mode_mask(df: pd.DataFrame, auto_cl_mode: str) -> pd.Series:
    if "auto_cl_mode" not in df.columns:
        return pd.Series(str(auto_cl_mode) == "v0", index=df.index)
    modes = (
        df["auto_cl_mode"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"": "v0", "none": "v0"})
    )
    return modes.eq(str(auto_cl_mode).strip().lower())


def _route_family_mode_mask(df: pd.DataFrame, route_family_mode: str) -> pd.Series:
    expected = normalize_route_family_mode(route_family_mode)
    if "route_family_mode" not in df.columns:
        return pd.Series(expected == "default", index=df.index)
    modes = (
        df["route_family_mode"]
        .fillna("default")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("-", "_", regex=False)
        .replace({"": "default", "none": "default"})
    )
    return modes.eq(expected)


def _write_upsert(path: Path, rows: list[dict[str, Any]], columns: list[str], key_cols: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    for col in columns:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df[columns + [c for c in new_df.columns if c not in columns]]
    with file_lock(str(path) + ".lock"):
        old_df = _read_csv(path)
        out = pd.concat([old_df, new_df], ignore_index=True)
        ordered = columns + [c for c in out.columns if c not in columns]
        out = out.reindex(columns=ordered)
        if key_cols and all(col in out.columns for col in key_cols):
            def key_token(value: Any) -> str:
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    return ""
                text = str(value).strip()
                return re.sub(r"^(-?\d+)\.0$", r"\1", text)

            status_score = {"success": 3, "partial": 2, "failed": 1}
            key_df = out[key_cols].apply(lambda col: col.map(key_token))
            out["_dedup_key"] = key_df.astype(str).agg("\x1f".join, axis=1)
            out["_dedup_order"] = np.arange(len(out), dtype=np.int64)
            if "status" in out.columns:
                out["_dedup_status"] = out["status"].astype(str).str.lower().map(status_score).fillna(0).astype(int)
            else:
                out["_dedup_status"] = 0
            if "task_probe_eval_protocol" in out.columns:
                out["_dedup_protocol"] = (
                    out["task_probe_eval_protocol"].astype(str).str.strip().ne("").astype(int)
                )
            else:
                out["_dedup_protocol"] = 0
            out = (
                out.sort_values(["_dedup_key", "_dedup_protocol", "_dedup_status", "_dedup_order"], kind="mergesort")
                .drop_duplicates("_dedup_key", keep="last")
                .sort_values("_dedup_order", kind="mergesort")
                .drop(columns=["_dedup_key", "_dedup_order", "_dedup_status", "_dedup_protocol"])
            )
        out.to_csv(path, index=False)


@contextmanager
def _task_probe_relaxed_determinism(model_abbr: str, model_id: int):
    try:
        import torch
    except Exception:
        yield
        return
    enabled = bool(torch.are_deterministic_algorithms_enabled())
    warn_only = False
    if enabled:
        getter = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)
        if getter is not None:
            try:
                warn_only = bool(getter())
            except Exception:
                warn_only = False
        print(
            f"[TaskProbeSelect][determinism] model={model_abbr}({model_id}) "
            "temporarily disables torch deterministic algorithms for candidate forward"
        )
        torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        if enabled:
            try:
                torch.use_deterministic_algorithms(True, warn_only=warn_only)
            except TypeError:
                torch.use_deterministic_algorithms(True)


def _forward_summary_path(root: Path) -> Path:
    return root / TASK_PROBE_SELECT_FORWARD_SUMMARY


def _rank_summary_path(root: Path) -> Path:
    return root / TASK_PROBE_SELECT_RANK_SUMMARY


def _per_model_csv(root: Path, model_key: str, cache_stem: str) -> Path:
    return root / str(model_key) / f"{cache_stem}.csv"


def task_probe_sample_timing_csv_candidates(cache_path: Path | str) -> list[Path]:
    primary = Path(cache_path).with_suffix(".csv")
    candidates = [primary]
    fallback_stem = re.sub(r"_ws[^_]+(?:_sr[^_]+)?(?=_ss\d+$)", "", primary.stem)
    if fallback_stem != primary.stem:
        candidates.append(primary.with_name(f"{fallback_stem}.csv"))
    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        token = path.as_posix()
        if token not in seen:
            out.append(path)
            seen.add(token)
    return out


def task_probe_sample_seconds_for_dataset(cache_path: Path | str, ds_config: str) -> tuple[float | None, str]:
    checked: list[str] = []
    for path in task_probe_sample_timing_csv_candidates(cache_path):
        checked.append(path.as_posix())
        if not path.exists():
            continue
        try:
            with path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        for row in reversed(rows):
            if str(row.get("dataset", "")) != str(ds_config):
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
                    return float(val), f"{path.as_posix()}:{col}"
            try:
                val_ms = float(row.get("sample_ms", ""))
            except Exception:
                continue
            if np.isfinite(val_ms) and val_ms >= 0:
                return float(val_ms) / 1000.0, f"{path.as_posix()}:sample_ms"
    return None, ",".join(checked)


def _metric_values_from_res(res: dict) -> dict[str, float]:
    def first(*keys: str) -> float:
        for key in keys:
            if key not in res:
                continue
            try:
                arr = np.asarray(res[key], dtype=float)
                if arr.size:
                    val = float(arr.reshape(-1)[0])
                    if np.isfinite(val):
                        return val
            except Exception:
                continue
        return float("nan")

    return {
        "glouts_MASE": first("MASE[0.5]", "eval_metrics/MASE[0.5]"),
        "glouts_sMAPE": first("sMAPE[0.5]", "eval_metrics/sMAPE[0.5]"),
        "glouts_CRPS": first("mean_weighted_sum_quantile_loss", "eval_metrics/mean_weighted_sum_quantile_loss"),
    }


def _model_forward_complete(
    root: Path,
    cache_stem: str,
    ds_config: str,
    model_id: int,
    model_key: str,
    eval_protocol: str,
    auto_cl_mode: str,
) -> bool:
    per_csv = _per_model_csv(root, model_key, cache_stem)
    if not per_csv.exists():
        return False
    df = _read_csv(per_csv)
    required = {
        "cache_stem",
        "dataset",
        "model_id",
        "task_probe_eval_protocol",
        "MASE",
        "sMAPE",
        "CRPS",
        "MASE_DEN",
        "CRPS_DEN",
    }
    if df.empty or not required.issubset(set(df.columns)):
        return False
    sub = df[
        df["cache_stem"].astype(str).eq(str(cache_stem))
        & df["dataset"].astype(str).eq(str(ds_config))
        & pd.to_numeric(df["model_id"], errors="coerce").eq(float(model_id))
        & df["task_probe_eval_protocol"].astype(str).eq(str(eval_protocol))
        & _auto_cl_mode_mask(df, auto_cl_mode)
    ].copy()
    if sub.empty:
        return False
    vals = sub[["MASE", "sMAPE", "CRPS", "MASE_DEN", "CRPS_DEN"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if vals.size == 0 or not np.isfinite(vals).all():
        return False
    if (vals[:, 3:] <= 0).any():
        return False

    summary = _read_csv(_forward_summary_path(root))
    required_summary = {
        "cache_stem",
        "dataset",
        "model_id",
        "task_probe_eval_protocol",
        "sample_forward_ms",
        "glouts_MASE",
        "glouts_sMAPE",
        "glouts_CRPS",
        "status",
    }
    if summary.empty or not required_summary.issubset(set(summary.columns)):
        return False
    s = summary[
        summary["cache_stem"].astype(str).eq(str(cache_stem))
        & summary["dataset"].astype(str).eq(str(ds_config))
        & pd.to_numeric(summary["model_id"], errors="coerce").eq(float(model_id))
        & summary["task_probe_eval_protocol"].astype(str).eq(str(eval_protocol))
        & _auto_cl_mode_mask(summary, auto_cl_mode)
        & summary["status"].astype(str).str.lower().eq("success")
    ].copy()
    if s.empty:
        return False
    latest = s.iloc[-1]
    metric_vals = pd.to_numeric(latest[["sample_forward_ms", "glouts_MASE", "glouts_sMAPE", "glouts_CRPS"]], errors="coerce").to_numpy(dtype=float)
    if not bool(metric_vals.size and np.isfinite(metric_vals).all() and metric_vals[0] >= 0):
        return False
    expected_windows = pd.to_numeric(pd.Series([latest.get("selected_windows", np.nan)]), errors="coerce").iloc[0]
    if np.isfinite(float(expected_windows)) and float(expected_windows) > 0:
        window_key_cols = ["step4_sample_idx", "channel"]
        if not set(window_key_cols).issubset(set(sub.columns)):
            return False
        window_keys = sub[window_key_cols].apply(pd.to_numeric, errors="coerce").dropna().drop_duplicates()
        if int(len(window_keys)) < int(float(expected_windows)):
            return False
    return True


def _instantiate_probe_model(
    args: Any,
    rec: dict[str, Any],
    root: Path,
    per_csv: Path,
    context_len: int | None = None,
):
    model_args = copy.deepcopy(args)
    if context_len is not None and int(context_len) > 0:
        model_args.context_len = int(context_len)
    model_args.run_mode = "zoo_task_probe_select"
    model_args.models = str(rec["model_key"])
    model_args.output_dir = str(root)
    model_args.save_pred = False
    model_args.skip_saved = False
    model_args.clean_saved = False
    model_args.vldb_skip_evaluate = False
    model_args.GE_fast_eval = False
    model_args.enable_process_metrics = False
    model_args.enable_per_window_metrics = False
    model_args.task_probe_select_result_csv = str(per_csv)
    module = importlib.import_module(str(rec["model_module"]))
    model_class = getattr(module, str(rec["model_class"]))
    return model_class(
        model_args,
        module_name=rec["module_name"],
        model_name=rec["model_key"],
        model_local_path=str(resolve_checkpoint_path(rec["model_local_path"])),
    )


def _augment_window_rows(
    rows: list[dict[str, Any]],
    *,
    sample_dataset: Any,
    args: Any,
    cache_stem: str,
    cache_path: Path,
    stage: int,
    ds_config: str,
    model_id: int,
    model_key: str,
    model_abbr: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    step_entries = list(getattr(sample_dataset, "step4_entry_indices", []))
    step_samples = list(getattr(sample_dataset, "step4_sample_indices", []))
    step_channels = list(getattr(sample_dataset, "step4_channel_indices", []))
    step_segments = list(getattr(sample_dataset, "step4_segment_start_indices", []))
    base = {
        "timestamp_utc": _utc_now(),
        "cache_stem": cache_stem,
        "cache_path": str(cache_path),
        "stage": int(stage),
        "dataset": ds_config,
        "model_id": int(model_id),
        "model_key": model_key,
        "model_abbr": model_abbr,
        "task_probe_eval_protocol": task_probe_eval_protocol(args),
        "task_probe_cache_window_len": int(getattr(sample_dataset, "task_probe_cache_window_len", 0) or 0),
        "task_probe_context_len": int(getattr(sample_dataset, "task_probe_context_len", 0) or 0),
        "task_probe_prediction_len": int(getattr(sample_dataset, "task_probe_prediction_len", 0) or 0),
        **_sample_config_row(args),
    }
    for local_entry, row in enumerate(rows):
        entry_idx = int(row.get("entry", local_entry))
        item = dict(base)
        item.update(row)
        item["local_entry"] = int(entry_idx)
        item["step4_entry_idx"] = step_entries[entry_idx] if entry_idx < len(step_entries) else ""
        item["step4_sample_idx"] = step_samples[entry_idx] if entry_idx < len(step_samples) else ""
        if entry_idx < len(step_channels) and str(step_channels[entry_idx]).strip() != "":
            item["channel"] = int(step_channels[entry_idx])
        item["segment_start_idx"] = step_segments[entry_idx] if entry_idx < len(step_segments) else ""
        out.append(item)
    return out


def _forward_summary_row(
    *,
    args: Any,
    cache_stem: str,
    cache_path: Path,
    stage: int,
    ds_config: str,
    model_id: int,
    model_key: str,
    model_abbr: str,
    sample_dataset: Any | None,
    metric_rows: int,
    model_obj: Any | None,
    metrics: dict[str, float] | None,
    per_csv: Path,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    model_load_ms = float(getattr(model_obj, "_last_model_load_ms", np.nan)) if model_obj is not None else np.nan
    predict_ms = float(getattr(model_obj, "_last_predict_ms", np.nan)) if model_obj is not None else np.nan
    sample_forward_ms = float(getattr(model_obj, "_last_forward_ms", np.nan)) if model_obj is not None else np.nan
    evaluate_ms = float(getattr(model_obj, "_last_evaluate_ms", np.nan)) if model_obj is not None else np.nan
    metrics = metrics or {}
    return {
        "timestamp_utc": _utc_now(),
        "cache_stem": cache_stem,
        "cache_path": str(cache_path),
        "stage": int(stage),
        "dataset": ds_config,
        **_sample_config_row(args),
        "task_probe_cache_window_len": "" if sample_dataset is None else int(getattr(sample_dataset, "task_probe_cache_window_len", 0) or 0),
        "task_probe_context_len": "" if sample_dataset is None else int(getattr(sample_dataset, "task_probe_context_len", 0) or 0),
        "task_probe_prediction_len": "" if sample_dataset is None else int(getattr(sample_dataset, "task_probe_prediction_len", 0) or 0),
        "model_id": int(model_id),
        "model_key": model_key,
        "model_abbr": model_abbr,
        "selected_windows": "" if sample_dataset is None else len(sample_dataset.test_data.input),
        "metric_rows": int(metric_rows),
        "model_load_ms": _format_float(model_load_ms, 3),
        "forward_ms": _format_float(predict_ms, 3),
        "sample_forward_ms": _format_float(sample_forward_ms, 3),
        "evaluate_ms": _format_float(evaluate_ms, 3),
        "glouts_MASE": _format_float(metrics.get("glouts_MASE", np.nan), 12),
        "glouts_sMAPE": _format_float(metrics.get("glouts_sMAPE", np.nan), 12),
        "glouts_CRPS": _format_float(metrics.get("glouts_CRPS", np.nan), 12),
        "status": status,
        "error": str(error)[:1000],
        "per_model_csv": str(per_csv),
    }


def _forward_metrics_for_rank(
    root: Path,
    cache_stem: str,
    ds_config: str,
    candidate_ids: Iterable[int],
    eval_protocol: str,
    auto_cl_mode: str,
    summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if summary is None:
        summary = _read_csv(_forward_summary_path(root))
    if summary.empty:
        return pd.DataFrame()
    required = {
        "cache_stem",
        "dataset",
        "model_id",
        "task_probe_eval_protocol",
        "glouts_MASE",
        "glouts_CRPS",
        "sample_forward_ms",
        "status",
    }
    if not required.issubset(set(summary.columns)):
        return pd.DataFrame()
    ids = {int(x) for x in candidate_ids}
    sub = summary[
        summary["cache_stem"].astype(str).eq(str(cache_stem))
        & summary["dataset"].astype(str).eq(str(ds_config))
        & summary["task_probe_eval_protocol"].astype(str).eq(str(eval_protocol))
        & _auto_cl_mode_mask(summary, auto_cl_mode)
        & summary["status"].astype(str).str.lower().eq("success")
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["model_id"] = pd.to_numeric(sub["model_id"], errors="coerce")
    sub = sub[sub["model_id"].isin(ids)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["_file_order"] = np.arange(len(sub), dtype=np.int64)
    sub = sub.sort_values("_file_order").drop_duplicates("model_id", keep="last").drop(columns=["_file_order"])
    for col in ["glouts_MASE", "glouts_sMAPE", "glouts_CRPS", "sample_forward_ms"]:
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return sub


def _latest_success_forward_rows(
    root: Path,
    cache_stem: str,
    ds_config: str,
    candidate_ids: Iterable[int],
    eval_protocol: str,
    auto_cl_mode: str,
) -> pd.DataFrame:
    summary = _read_csv(_forward_summary_path(root))
    if summary.empty:
        return pd.DataFrame()
    required = {"cache_stem", "dataset", "model_id", "task_probe_eval_protocol", "status", "per_model_csv"}
    if not required.issubset(set(summary.columns)):
        return pd.DataFrame()
    ids = {int(x) for x in candidate_ids}
    sub = summary[
        summary["cache_stem"].astype(str).eq(str(cache_stem))
        & summary["dataset"].astype(str).eq(str(ds_config))
        & summary["task_probe_eval_protocol"].astype(str).eq(str(eval_protocol))
        & _auto_cl_mode_mask(summary, auto_cl_mode)
        & summary["status"].astype(str).str.lower().eq("success")
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["model_id"] = pd.to_numeric(sub["model_id"], errors="coerce")
    sub = sub[sub["model_id"].isin(ids)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["_file_order"] = np.arange(len(sub), dtype=np.int64)
    return sub.sort_values("_file_order").drop_duplicates("model_id", keep="last").drop(columns=["_file_order"])


def _candidate_ids_from_latest_rank(
    root: Path,
    cache_stem: str,
    ds_config: str,
    stage: int,
    eval_protocol: str,
    auto_cl_mode: str,
) -> list[int]:
    rank_df = _read_csv(_rank_summary_path(root))
    required = {"cache_stem", "dataset", "stage", "task_probe_eval_protocol", "candidate_model_ids", "missing_model_ids", "status"}
    if rank_df.empty or not required.issubset(set(rank_df.columns)):
        return []
    sub = rank_df[
        rank_df["cache_stem"].astype(str).eq(str(cache_stem))
        & rank_df["dataset"].astype(str).eq(str(ds_config))
        & pd.to_numeric(rank_df["stage"], errors="coerce").eq(float(stage))
        & rank_df["task_probe_eval_protocol"].astype(str).eq(str(eval_protocol))
        & _auto_cl_mode_mask(rank_df, auto_cl_mode)
        & rank_df["status"].astype(str).str.lower().isin({"success", "partial"})
    ].copy()
    if sub.empty:
        return []
    latest = sub.iloc[-1]
    missing = _parse_int_list(latest.get("missing_model_ids"))
    if missing:
        raise FileNotFoundError(
            f"Task-probe Select rank is partial for dataset={ds_config}, missing_model_ids={missing}"
        )
    return _parse_int_list(latest.get("candidate_model_ids"))


def _mase_rank_from_latest_summary(
    root: Path,
    cache_stem: str,
    ds_config: str,
    stage: int,
    eval_protocol: str,
    auto_cl_mode: str,
) -> list[int]:
    rank_df = _read_csv(_rank_summary_path(root))
    required = {"cache_stem", "dataset", "stage", "task_probe_eval_protocol", "model_rank_mase", "missing_model_ids", "status"}
    if rank_df.empty or not required.issubset(set(rank_df.columns)):
        return []
    sub = rank_df[
        rank_df["cache_stem"].astype(str).eq(str(cache_stem))
        & rank_df["dataset"].astype(str).eq(str(ds_config))
        & pd.to_numeric(rank_df["stage"], errors="coerce").eq(float(stage))
        & rank_df["task_probe_eval_protocol"].astype(str).eq(str(eval_protocol))
        & _auto_cl_mode_mask(rank_df, auto_cl_mode)
        & rank_df["status"].astype(str).str.lower().isin({"success", "partial"})
    ].copy()
    if sub.empty:
        return []
    latest = sub.iloc[-1]
    missing = _parse_int_list(latest.get("missing_model_ids"))
    if missing:
        return []
    return _parse_int_list(latest.get("model_rank_mase"))


def _rank_ids(metric_df: pd.DataFrame, metric_col: str, candidate_ids: list[int]) -> list[int]:
    if metric_df.empty or metric_col not in metric_df.columns:
        return []
    order_pos = {int(mid): pos for pos, mid in enumerate(candidate_ids)}
    sub = metric_df.dropna(subset=["model_id", metric_col]).copy()
    if sub.empty:
        return []
    sub["model_id"] = sub["model_id"].astype(int)
    sub = sub[sub["model_id"].isin(set(candidate_ids))].copy()
    if sub.empty:
        return []
    return [
        int(rec["model_id"])
        for _, rec in sorted(
            sub.iterrows(),
            key=lambda item: (float(item[1][metric_col]), order_pos.get(int(item[1]["model_id"]), 10**9)),
        )
    ]


def _model_keys_for_ids(ids: Iterable[int], id_to_rec: dict[int, dict[str, Any]]) -> list[str]:
    return [str(id_to_rec[int(mid)]["model_key"]) for mid in ids if int(mid) in id_to_rec]


def _unique_order_for_candidates(raw_order: Iterable[Any], candidate_ids: list[int]) -> list[int]:
    candidate_set = {int(x) for x in candidate_ids}
    out: list[int] = []
    for raw in _as_list(raw_order):
        try:
            mid = int(raw)
        except Exception:
            continue
        if mid in candidate_set and mid not in out:
            out.append(mid)
    for mid in candidate_ids:
        if int(mid) not in out:
            out.append(int(mid))
    return out


def _task_sample_top1_text(selector_extra: dict[str, Any] | None) -> str:
    sample_rankings = selector_extra.get("task_sample_rankings") if isinstance(selector_extra, dict) else None
    if sample_rankings is None:
        return ""
    arr = np.asarray(sample_rankings)
    if arr.ndim != 3 or arr.shape[0] <= 0 or arr.shape[1] <= 0 or arr.shape[2] <= 0:
        return ""
    tokens: list[str] = []
    for sample_idx in range(arr.shape[0]):
        vals = [int(x) for x in arr[sample_idx, 0, :].reshape(-1).tolist()]
        tokens.append("[" + " ".join(str(x) for x in vals) + "]")
    return " ".join(tokens)


def _predicted_order_for_task_probe_window(
    *,
    selector_extra: dict[str, Any],
    step4_sample_idx: int,
    channel: int,
    candidate_ids: list[int],
) -> list[int]:
    sample_rankings = selector_extra.get("task_sample_rankings") if isinstance(selector_extra, dict) else None
    if sample_rankings is None:
        raise FileNotFoundError("missing TSRouter task_sample_rankings; cannot evaluate window-level rank hits")
    arr = np.asarray(sample_rankings)
    if arr.ndim != 3:
        raise FileNotFoundError(f"invalid TSRouter task_sample_rankings shape={arr.shape}; expected (sample, rank, channel)")
    if not (0 <= int(step4_sample_idx) < arr.shape[0]) or not (0 <= int(channel) < arr.shape[2]):
        raise FileNotFoundError(
            f"Task-probe window key outside TSRouter task_sample_rankings: "
            f"sample={step4_sample_idx}, channel={channel}, shape={arr.shape}"
        )
    return _unique_order_for_candidates(arr[int(step4_sample_idx), :, int(channel)].tolist(), candidate_ids)


def compute_task_probe_window_hit_metrics(
    *,
    args: Any,
    ds_config: str,
    model_sizes: dict,
    selector_extra: dict[str, Any] | None,
    model_order: Iterable[Any] | None = None,
) -> dict[str, float]:
    selector_extra = selector_extra or {}
    search_args = selector_extra.get("search_args") if isinstance(selector_extra, dict) else None
    if search_args is None:
        search_args = args
    root = task_probe_select_root(args)
    cache_stem = task_probe_select_cache_stem(search_args)
    eval_protocol = task_probe_eval_protocol(search_args)
    auto_cl_mode = get_auto_cl_mode(search_args)
    stage = int(getattr(args, "current_zoo_num", 0) or 0)
    records = _model_records(model_sizes or {})
    id_to_rec = {int(rec["model_id"]): rec for rec in records}
    candidate_ids = [
        mid
        for mid in _candidate_ids_from_latest_rank(
            root, cache_stem, ds_config, stage, eval_protocol, auto_cl_mode
        )
        if mid in id_to_rec
    ]
    if not candidate_ids:
        summary = _read_csv(_forward_summary_path(root))
        if not summary.empty and {"cache_stem", "dataset", "model_id", "status"}.issubset(set(summary.columns)):
            sub = summary[
                summary["cache_stem"].astype(str).eq(str(cache_stem))
                & summary["dataset"].astype(str).eq(str(ds_config))
                & summary.get(
                    "task_probe_eval_protocol",
                    pd.Series("", index=summary.index),
                )
                .astype(str)
                .eq(str(eval_protocol))
                & _auto_cl_mode_mask(summary, auto_cl_mode)
                & summary["status"].astype(str).str.lower().eq("success")
            ].copy()
            candidate_ids = [
                int(x)
                for x in pd.to_numeric(sub.get("model_id"), errors="coerce").dropna().astype(int).tolist()
                if int(x) in id_to_rec
            ]
            candidate_ids = sorted(dict.fromkeys(candidate_ids))
    if len(candidate_ids) < 2:
        raise FileNotFoundError(
            f"Task-probe Select per-window truth needs at least 2 candidate models: "
            f"root={root}, cache={cache_stem}, dataset={ds_config}, "
            f"observed_candidate_ids={candidate_ids}"
        )

    forward_rows = _latest_success_forward_rows(
        root,
        cache_stem,
        ds_config,
        candidate_ids,
        eval_protocol,
        auto_cl_mode,
    )
    present_ids = {
        int(x)
        for x in pd.to_numeric(forward_rows.get("model_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist()
    }
    missing_ids = [int(mid) for mid in candidate_ids if int(mid) not in present_ids]
    if missing_ids:
        raise FileNotFoundError(
            f"Task-probe Select forward rows incomplete for dataset={ds_config}, missing_model_ids={missing_ids}"
        )
    expected_values = pd.to_numeric(
        forward_rows.get("selected_windows", pd.Series(dtype=float)),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).dropna()
    expected_windows = int(expected_values.max()) if not expected_values.empty and float(expected_values.max()) > 0 else 0

    metric_cols: list[str] = []
    window_df: pd.DataFrame | None = None
    key_cols = ["step4_sample_idx", "channel"]
    meta_cols = ["step4_sample_idx", "channel", "series_id", "window_id", "forecast_start"]
    for model_id in candidate_ids:
        rec = id_to_rec[int(model_id)]
        fwd = forward_rows[pd.to_numeric(forward_rows["model_id"], errors="coerce").eq(float(model_id))]
        if fwd.empty:
            raise FileNotFoundError(f"missing forward summary for model_id={model_id}")
        per_csv_raw = str(fwd.iloc[-1].get("per_model_csv", "") or "")
        per_csv = Path(per_csv_raw) if per_csv_raw else _per_model_csv(root, str(rec["model_key"]), cache_stem)
        if not per_csv.exists():
            raise FileNotFoundError(f"missing Task-probe Select per-window csv: {per_csv}")
        df = _read_csv(per_csv)
        required = {
            "cache_stem",
            "dataset",
            "model_id",
            "task_probe_eval_protocol",
            "step4_sample_idx",
            "channel",
            "window_id",
            "MASE",
        }
        if df.empty or not required.issubset(set(df.columns)):
            raise FileNotFoundError(f"invalid Task-probe Select per-window csv: {per_csv}")
        sub = df[
            df["cache_stem"].astype(str).eq(str(cache_stem))
            & df["dataset"].astype(str).eq(str(ds_config))
            & pd.to_numeric(df["model_id"], errors="coerce").eq(float(model_id))
            & df["task_probe_eval_protocol"].astype(str).eq(str(eval_protocol))
            & _auto_cl_mode_mask(df, auto_cl_mode)
        ].copy()
        if sub.empty:
            raise FileNotFoundError(f"missing dataset={ds_config}, model_id={model_id} rows in {per_csv}")
        sub["step4_sample_idx"] = pd.to_numeric(sub["step4_sample_idx"], errors="coerce")
        sub["channel"] = pd.to_numeric(sub["channel"], errors="coerce")
        sub["window_id"] = pd.to_numeric(sub["window_id"], errors="coerce")
        sub["MASE"] = pd.to_numeric(sub["MASE"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        sub = sub.dropna(subset=["step4_sample_idx", "channel", "window_id", "MASE"]).copy()
        if sub.empty:
            raise FileNotFoundError(f"no finite MASE rows in {per_csv}")
        sub["step4_sample_idx"] = sub["step4_sample_idx"].astype(int)
        sub["channel"] = sub["channel"].astype(int)
        sub["window_id"] = sub["window_id"].astype(int)
        sub = sub.sort_values(key_cols, kind="mergesort").drop_duplicates(key_cols, keep="last")
        metric_col = f"MASE_model_{int(model_id)}"
        keep_meta = list(dict.fromkeys([c for c in meta_cols if c in sub.columns] + key_cols))
        model_df = sub[keep_meta + ["MASE"]].rename(columns={"MASE": metric_col}).reset_index(drop=True)
        if window_df is None:
            window_df = model_df
        else:
            before = int(window_df[key_cols].drop_duplicates().shape[0])
            window_df = window_df.merge(
                model_df[key_cols + [metric_col]],
                on=key_cols,
                how="inner",
                validate="one_to_one",
            )
            after = int(window_df[key_cols].drop_duplicates().shape[0])
            if after < before:
                current = int(model_df[key_cols].drop_duplicates().shape[0])
                raise FileNotFoundError(
                    f"Task-probe Select sample-channel key mismatch for dataset={ds_config}, model_id={model_id}: "
                    f"matched={after}, previous_keys={before}, current_keys={current}, "
                    f"raw_rows={len(sub)}, align_keys={key_cols}, csv={per_csv}"
                )
        vals = pd.to_numeric(model_df[metric_col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            raise FileNotFoundError(f"non-finite Task-probe MASE rows in {per_csv}")
        metric_cols.append(metric_col)

    if window_df is None or not metric_cols:
        raise FileNotFoundError(f"empty Task-probe Select per-window truth for dataset={ds_config}")
    observed_windows = int(window_df[key_cols].drop_duplicates().shape[0])
    if expected_windows > 0 and observed_windows < expected_windows:
        raise FileNotFoundError(
            f"Task-probe Select per-window truth incomplete for dataset={ds_config}: "
            f"observed_windows={observed_windows}, expected_at_least={expected_windows}"
        )
    matrix = window_df[metric_cols].to_numpy(dtype=float).T
    if matrix.size == 0 or not np.isfinite(matrix).all():
        raise FileNotFoundError(f"invalid Task-probe Select MASE matrix for dataset={ds_config}")
    true_rank_mase = _mase_rank_from_latest_summary(
        root, cache_stem, ds_config, stage, eval_protocol, auto_cl_mode
    )
    if not true_rank_mase:
        metric_df = _forward_metrics_for_rank(
            root,
            cache_stem,
            ds_config,
            candidate_ids,
            eval_protocol,
            auto_cl_mode,
        )
        true_rank_mase = _rank_ids(metric_df, "glouts_MASE", candidate_ids)
    if not true_rank_mase:
        raise FileNotFoundError(f"missing Task-probe Select aggregate MASE rank for dataset={ds_config}")
    true_mid = int(true_rank_mase[0])
    true_top3 = {int(x) for x in true_rank_mase[:3]}
    top1_hits: list[bool] = []
    top3_hits: list[bool] = []
    repr_window_top1: list[list[int]] = []
    for col_idx in range(len(window_df)):
        meta = window_df.iloc[col_idx]
        pred_order = _predicted_order_for_task_probe_window(
            selector_extra=selector_extra,
            step4_sample_idx=int(meta.get("step4_sample_idx", 0)),
            channel=int(meta.get("channel", 0)),
            candidate_ids=candidate_ids,
        )
        if not pred_order:
            continue
        pred_top1 = int(pred_order[0])
        top1_hits.append(pred_top1 == int(true_mid))
        repr_window_top1.append([pred_top1])
        top3_hits.append(pred_top1 in true_top3)
    if not top1_hits:
        raise FileNotFoundError(f"no comparable Task-probe Select windows for dataset={ds_config}")
    return {
        "TEST_WINDOW_TOP1_ACC": float(np.mean(top1_hits)),
        "TEST_WINDOW_TOP3_HIT": float(np.mean(top3_hits)),
        "TEST_WINDOW_EVAL_N": float(len(top1_hits)),
        "_PROCESS_FORWARD_WINDOW_TOP3": [int(x) for x in true_rank_mase[:3]],
        "_PROCESS_REPR_WINDOW_TOP1": repr_window_top1,
    }


def _write_rank_summary(
    *,
    root: Path,
    args: Any,
    cache_stem: str,
    cache_path: Path,
    stage: int,
    ds_config: str,
    candidate_ids: list[int],
    tsrouter_model_order: Iterable[Any] | None,
    id_to_rec: dict[int, dict[str, Any]],
    status: str,
    missing_model_ids: list[int],
    error: str = "",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    eval_protocol = task_probe_eval_protocol(args)
    auto_cl_mode = get_auto_cl_mode(args)
    metric_df = _forward_metrics_for_rank(
        root,
        cache_stem,
        ds_config,
        candidate_ids,
        eval_protocol,
        auto_cl_mode,
    )
    rank_mase = _rank_ids(metric_df, "glouts_MASE", candidate_ids)
    rank_crps = _rank_ids(metric_df, "glouts_CRPS", candidate_ids)
    rank_seconds = time.perf_counter() - t0
    sample_seconds_value, sample_timing_source = task_probe_sample_seconds_for_dataset(cache_path, ds_config)
    sample_seconds = float(sample_seconds_value) if sample_seconds_value is not None else float("nan")
    forward_ms = pd.to_numeric(metric_df.get("sample_forward_ms", pd.Series(dtype=float)), errors="coerce")
    forward_seconds = float(forward_ms.sum() / 1000.0) if len(forward_ms.dropna()) == len(candidate_ids) else float("nan")
    route_final = (
        float(sample_seconds) + float(forward_seconds) + float(rank_seconds)
        if np.isfinite(sample_seconds) and np.isfinite(forward_seconds) and np.isfinite(rank_seconds)
        else float("nan")
    )
    if not rank_mase or not rank_crps:
        status = "failed"
        error = (error + " " if error else "") + "missing rank metrics"
    if not np.isfinite(sample_seconds):
        error = (error + " " if error else "") + "missing sample_seconds"
    row = {
        "timestamp_utc": _utc_now(),
        "cache_stem": cache_stem,
        "cache_path": str(cache_path),
        "stage": int(stage),
        "dataset": ds_config,
        **_sample_config_row(args),
        "probe_model_num": len(candidate_ids),
        "candidate_model_ids": " ".join(map(str, candidate_ids)),
        "candidate_model_keys": " ".join(_model_keys_for_ids(candidate_ids, id_to_rec)),
        "tsrouter_model_order": " ".join(
            str(int(x)) for x in _as_list(tsrouter_model_order) if str(x).strip() != ""
        ),
        "model_rank_mase": " ".join(map(str, rank_mase)),
        "model_rank_crps": " ".join(map(str, rank_crps)),
        "model_rank_mase_keys": " ".join(_model_keys_for_ids(rank_mase, id_to_rec)),
        "model_rank_crps_keys": " ".join(_model_keys_for_ids(rank_crps, id_to_rec)),
        "selected_model_id_mase": rank_mase[0] if rank_mase else "",
        "selected_model_key_mase": id_to_rec[rank_mase[0]]["model_key"] if rank_mase and rank_mase[0] in id_to_rec else "",
        "selected_model_id_crps": rank_crps[0] if rank_crps else "",
        "selected_model_key_crps": id_to_rec[rank_crps[0]]["model_key"] if rank_crps and rank_crps[0] in id_to_rec else "",
        "sample_seconds": _format_float(sample_seconds, 9),
        "sample_timing_source": sample_timing_source,
        "forward_seconds": _format_float(forward_seconds, 9),
        "rank_seconds": _format_float(rank_seconds, 9),
        "route_final_seconds": _format_float(route_final, 9),
        "status": status,
        "missing_model_ids": " ".join(map(str, missing_model_ids)),
        "error": str(error)[:1000],
    }
    rank_scope_key = [
        "cache_stem",
        "dataset",
        "stage",
        "probe_model_num",
        "candidate_model_ids",
        "task_probe_eval_protocol",
        "auto_cl_mode",
        "route_family_mode",
    ]
    _write_upsert(_rank_summary_path(root), [row], TASK_PROBE_SELECT_RANK_COLUMNS, rank_scope_key)
    return row


def run_task_probe_select_for_dataset(
    *,
    router_model: Any,
    dataset: Any,
    ds_key: str,
    ds_freq: str,
    ds_config: str,
    dataset_name: str,
    ds_name: str,
    term: str,
    model_order: Iterable[Any] | None,
) -> dict[str, Any] | None:
    args = router_model.args
    if not task_probe_select_enabled(args):
        return None

    extra = getattr(router_model, "_last_selector_extra", {}) or {}
    search_args = extra.get("search_args") if isinstance(extra, dict) else None
    if search_args is None:
        search_args = args

    root = task_probe_select_root(args)
    root.mkdir(parents=True, exist_ok=True)
    cache, meta_all, cache_path = _read_task_cache(search_args)
    search_args.task_sample_cache_path = str(cache_path)
    cache_stem = cache_path.stem
    eval_protocol = task_probe_eval_protocol(search_args)
    stage = int(getattr(args, "current_zoo_num", 0) or 0)
    records = _model_records(getattr(router_model, "Model_sizes", {}) or {})
    id_to_rec = {int(rec["model_id"]): rec for rec in records}
    current_model_num = len(records)
    requested = int(getattr(args, "mix_route_model_num", 0) or 0)
    if requested <= 0:
        requested = current_model_num
    candidate_ids = _complete_candidate_ids(model_order, current_model_num, requested)

    print(
        f"\n[TaskProbeSelect] dataset={ds_config}, cache={cache_stem}, "
        f"stage={stage}, probe_models={len(candidate_ids)}/{current_model_num}, "
        f"top_order={candidate_ids[:5]}"
    )

    missing: list[int] = []
    for pos, model_id in enumerate(candidate_ids, start=1):
        rec = id_to_rec.get(int(model_id))
        if rec is None:
            missing.append(int(model_id))
            continue
        model_key = str(rec["model_key"])
        model_abbr = str(rec["model_abbr"])
        per_csv = _per_model_csv(root, model_key, cache_stem)
        if bool(getattr(args, "skip_saved", False)) and _model_forward_complete(
            root,
            cache_stem,
            ds_config,
            int(model_id),
            model_key,
            eval_protocol,
            get_auto_cl_mode(search_args),
        ):
            print(
                f"[{pos:02d}/{len(candidate_ids):02d}]",end=""
            )
            continue

        model_obj = None
        try:
            sample_dataset = _build_sample_dataset(
                ds_name=ds_name,
                term=term,
                ds_config=ds_config,
                ds_freq=ds_freq,
                model_key=model_key,
                cache=cache,
                meta_all=meta_all,
                args=search_args,
            )
            print(
                f"[TaskProbeSelect][{pos:02d}/{len(candidate_ids):02d}] forward model {model_abbr}({model_id}) "
                # f" input_n={len(sample_dataset.test_data.input)}, "
                # f"cache_shape={sample_dataset.step4_cache_shape}, csv={per_csv.as_posix()}"
            )
            model_obj = _instantiate_probe_model(
                args,
                rec,
                root,
                per_csv,
                context_len=int(getattr(sample_dataset, "task_probe_context_len", 0) or 0),
            )
            with _task_probe_relaxed_determinism(model_abbr, int(model_id)):
                res, forecasts, _ = model_obj._make_forecasts(
                    dataset=sample_dataset,
                    dataset_name=dataset_name,
                    ds_config=ds_config,
                    fixed_model_order=None,
                    debug_mode=bool(getattr(args, "debug_mode", False)),
                )
            metrics = _metric_values_from_res(res)
            if (
                not np.isfinite(float(metrics.get("glouts_MASE", np.nan)))
                or not np.isfinite(float(metrics.get("glouts_CRPS", np.nan)))
            ):
                raise ValueError("missing finite GluonTS MASE/CRPS metrics")
            window_rows = compute_per_window_metric_rows(
                forecasts=forecasts,
                dataset=sample_dataset,
                dataset_name=ds_config,
                model_name=model_key,
            )
            if not window_rows:
                raise ValueError("no per-window metric rows computed")
            window_rows = _augment_window_rows(
                window_rows,
                sample_dataset=sample_dataset,
                args=search_args,
                cache_stem=cache_stem,
                cache_path=cache_path,
                stage=stage,
                ds_config=ds_config,
                model_id=int(model_id),
                model_key=model_key,
                model_abbr=model_abbr,
            )
            _write_upsert(
                per_csv,
                window_rows,
                TASK_PROBE_SELECT_WINDOW_COLUMNS,
                [
                    "cache_stem",
                    "dataset",
                    "model_id",
                    "task_probe_eval_protocol",
                    "auto_cl_mode",
                    "step4_sample_idx",
                    "channel",
                    "window_id",
                ],
            )
            summary_row = _forward_summary_row(
                args=search_args,
                cache_stem=cache_stem,
                cache_path=cache_path,
                stage=stage,
                ds_config=ds_config,
                model_id=int(model_id),
                model_key=model_key,
                model_abbr=model_abbr,
                sample_dataset=sample_dataset,
                metric_rows=len(window_rows),
                model_obj=model_obj,
                metrics=metrics,
                per_csv=per_csv,
                status="success",
            )
            _write_upsert(
                _forward_summary_path(root),
                [summary_row],
                TASK_PROBE_SELECT_FORWARD_COLUMNS,
                [
                    "cache_stem",
                    "dataset",
                    "model_id",
                    "task_probe_eval_protocol",
                    "auto_cl_mode",
                ],
            )
            print(
                f"👉 saved "
                f"model={model_abbr}({model_id}), rows={len(window_rows)}, "
                f"MASE={metrics.get('glouts_MASE', np.nan):.6g}, "
                f"CRPS={metrics.get('glouts_CRPS', np.nan):.6g}, csv={per_csv.as_posix()}"
            )
        except Exception as exc:
            missing.append(int(model_id))
            error = f"{type(exc).__name__}: {exc}"
            print(
                f"[TaskProbeSelect][{pos:02d}/{len(candidate_ids):02d}][failed] "
                f"dataset={ds_config}, model={model_abbr}({model_id}): {error}"
            )
            summary_row = _forward_summary_row(
                args=search_args,
                cache_stem=cache_stem,
                cache_path=cache_path,
                stage=stage,
                ds_config=ds_config,
                model_id=int(model_id),
                model_key=model_key,
                model_abbr=model_abbr,
                sample_dataset=None,
                metric_rows=0,
                model_obj=model_obj,
                metrics=None,
                per_csv=per_csv,
                status="failed",
                error=error,
            )
            _write_upsert(
                _forward_summary_path(root),
                [summary_row],
                TASK_PROBE_SELECT_FORWARD_COLUMNS,
                [
                    "cache_stem",
                    "dataset",
                    "model_id",
                    "task_probe_eval_protocol",
                    "auto_cl_mode",
                ],
            )

    top1_text = _task_sample_top1_text(extra)
    if top1_text:
        print(f"\n[TaskProbeWindow][router_top1_before_probe_rank] {top1_text}")

    rank_row = _write_rank_summary(
        root=root,
        args=search_args,
        cache_stem=cache_stem,
        cache_path=cache_path,
        stage=stage,
        ds_config=ds_config,
        candidate_ids=candidate_ids,
        tsrouter_model_order=model_order,
        id_to_rec=id_to_rec,
        status="success" if not missing else "partial",
        missing_model_ids=missing,
    )
    def seconds_text(value: Any) -> str:
        text = str(value or "").strip()
        return text if text else "NA"

    print(
        f"🏆 [TaskProbeSelect][rank] dataset={ds_config}, "
        f"M={rank_row.get('model_rank_mase', '')}, C={rank_row.get('model_rank_crps', '')}, "
        f"sample={seconds_text(rank_row.get('sample_seconds', ''))}s, "
        f"forward={seconds_text(rank_row.get('forward_seconds', ''))}s, "
        f"rank={seconds_text(rank_row.get('rank_seconds', ''))}s, "
        f"route={seconds_text(rank_row.get('route_final_seconds', ''))}s"
    )
    return rank_row


def _parse_int_list(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        raw = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            raw = list(parsed) if isinstance(parsed, (list, tuple, np.ndarray)) else []
        except Exception:
            raw = re.findall(r"-?\d+(?:\.\d+)?", text)
    out = []
    for item in raw:
        try:
            val = int(float(item))
        except Exception:
            continue
        out.append(val)
    return out


def _task_probe_concrete_args_from_selector_row(
    args: Any,
    latest: pd.Series,
) -> Any | None:
    mode = str(latest.get("auto_cl_mode", "") or get_auto_cl_mode(args)).strip().lower()
    profile = get_auto_cl_profile_by_name(
        str(latest.get("adaptive_profile", "") or ""),
        mode,
    )
    if profile is None:
        return None
    resolved_eval_cl = str(latest.get("resolved_eval_cl", "") or "")
    rank_truth_cl = str(latest.get("rank_truth_cl", "") or "")
    if (
        resolved_eval_cl != str(profile["tsfm_results_dir"])
        or rank_truth_cl != resolved_eval_cl
    ):
        return None
    out = copy.deepcopy(args)
    out.auto_cl_mode = mode
    out.enable_context_len_adaptive_repr = True
    for key in [
        "adaptive_profile",
        "repr_input_dim",
        "repr_output_dim",
        "repr_sub_pred_len",
        "repr_source_exact_length",
    ]:
        setattr(out, key, profile[key])
    out.resolved_eval_cl = resolved_eval_cl
    out.rank_truth_cl = rank_truth_cl
    out.adaptive_context_len_avg = latest.get("adaptive_context_len_avg", np.nan)
    out.adaptive_pred_len = latest.get("adaptive_pred_len", np.nan)
    out.eval_cl_fallback_used = _truthy(
        latest.get("eval_cl_fallback_used", False)
    )
    normalize_auto_cl_args(out)
    return out


def task_probe_concrete_args_for_saved_dataset(
    args: Any,
    ds_config: str,
    selector_result_path: str | Path | None,
) -> Any | None:
    if get_auto_cl_mode(args) == "v0":
        return args
    if selector_result_path is None:
        return None
    selector_df = _read_csv(Path(selector_result_path))
    if selector_df.empty or "dataset" not in selector_df.columns:
        return None
    sub = selector_df[selector_df["dataset"].astype(str).eq(str(ds_config))].copy()
    if sub.empty:
        return None
    return _task_probe_concrete_args_from_selector_row(args, sub.iloc[-1])


def _task_probe_auto_cl_row_complete(
    latest: pd.Series,
    concrete_args: Any,
    cache_stem: str,
) -> bool:
    if get_auto_cl_mode(concrete_args) == "v0":
        return True
    profile = get_auto_cl_profile_by_name(
        str(latest.get("adaptive_profile", "") or ""),
        concrete_args,
    )
    if profile is None:
        return False
    expected = {
        "resolved_eval_cl": str(profile["tsfm_results_dir"]),
        "rank_truth_cl": str(profile["tsfm_results_dir"]),
        "repr_input_dim": int(profile["repr_input_dim"]),
        "repr_output_dim": int(profile["repr_output_dim"]),
        "repr_sub_pred_len": int(profile["repr_sub_pred_len"]),
        "repr_source_exact_length": int(profile["repr_source_exact_length"]),
    }
    for key, value in expected.items():
        actual = latest.get(key, "")
        if isinstance(value, int):
            numeric = pd.to_numeric(pd.Series([actual]), errors="coerce").iloc[0]
            if pd.isna(numeric) or int(numeric) != value:
                return False
        elif str(actual or "") != value:
            return False
    if _truthy(latest.get("eval_cl_fallback_used", False)):
        return False
    cache_path = str(latest.get("task_sample_cache_path", "") or "")
    cl_token = f"cl{int(profile['repr_input_dim'])}"
    if cl_token not in cache_stem.lower() or cl_token not in Path(cache_path).stem.lower():
        return False
    return True


def task_probe_select_dataset_rank_complete(
    args: Any,
    ds_config: str,
    selector_result_path: str | Path | None = None,
) -> bool:
    if not task_probe_select_enabled(args):
        return True
    concrete_args = task_probe_concrete_args_for_saved_dataset(
        args, ds_config, selector_result_path
    )
    if concrete_args is None:
        return False
    root = task_probe_select_root(args)
    rank_path = _rank_summary_path(root)
    df = _read_csv(rank_path)
    required = {
        "cache_stem",
        "dataset",
        "stage",
        "probe_model_num",
        "task_probe_eval_protocol",
        "model_rank_mase",
        "model_rank_crps",
        "sample_seconds",
        "route_final_seconds",
        "status",
    }
    if df.empty or not required.issubset(set(df.columns)):
        return False
    cache_stem = task_probe_select_cache_stem(concrete_args)
    eval_protocol = task_probe_eval_protocol(concrete_args)
    stage = int(getattr(args, "current_zoo_num", 0) or 0)
    requested = int(getattr(args, "mix_route_model_num", 0) or 0)
    if requested <= 0:
        requested = stage
    sub = df[
        df["cache_stem"].astype(str).eq(str(cache_stem))
        & df["dataset"].astype(str).eq(str(ds_config))
        & pd.to_numeric(df["stage"], errors="coerce").eq(float(stage))
        & pd.to_numeric(df["probe_model_num"], errors="coerce").eq(float(requested))
        & df["task_probe_eval_protocol"].astype(str).eq(str(eval_protocol))
        & _auto_cl_mode_mask(df, get_auto_cl_mode(concrete_args))
        & _route_family_mode_mask(
            df,
            getattr(concrete_args, "route_family_mode", "default"),
        )
        & df["status"].astype(str).str.lower().isin({"success", "partial"})
    ].copy()
    if sub.empty:
        return False
    latest = sub.iloc[-1]
    if not _task_probe_auto_cl_row_complete(latest, concrete_args, cache_stem):
        return False
    if not _parse_int_list(latest.get("model_rank_mase")) or not _parse_int_list(latest.get("model_rank_crps")):
        return False
    sample = pd.to_numeric(pd.Series([latest.get("sample_seconds")]), errors="coerce").iloc[0]
    if pd.isna(sample) or not np.isfinite(float(sample)) or float(sample) < 0:
        return False
    forward = pd.to_numeric(pd.Series([latest.get("forward_seconds")]), errors="coerce").iloc[0]
    rank = pd.to_numeric(pd.Series([latest.get("rank_seconds")]), errors="coerce").iloc[0]
    route = pd.to_numeric(pd.Series([latest.get("route_final_seconds")]), errors="coerce").iloc[0]
    if pd.isna(route) or not np.isfinite(float(route)) or float(route) < 0:
        return False
    if (
        pd.isna(forward)
        or pd.isna(rank)
        or not np.isfinite(float(forward))
        or not np.isfinite(float(rank))
        or float(forward) < 0
        or float(rank) < 0
    ):
        return False
    expected_route = float(sample) + float(forward) + float(rank)
    return bool(abs(float(route) - expected_route) <= 1e-6)


def _latest_rank_row_for_dataset(
    df: pd.DataFrame,
    cache_stem: str,
    ds_config: str,
    stage: int,
    model_ids: set[int],
    metric_col: str,
    eval_protocol: str,
    auto_cl_mode: str,
) -> tuple[pd.Series | None, bool]:
    if df.empty:
        return None, False
    sub = df[
        df.get("cache_stem", pd.Series(dtype=str)).astype(str).eq(str(cache_stem))
        & df.get("dataset", pd.Series(dtype=str)).astype(str).eq(str(ds_config))
        & df.get("status", pd.Series(dtype=str)).astype(str).str.lower().isin({"success", "partial"})
    ].copy()
    if "task_probe_eval_protocol" in sub.columns:
        sub = sub[
            sub["task_probe_eval_protocol"].astype(str).eq(str(eval_protocol))
        ].copy()
    sub = sub[_auto_cl_mode_mask(sub, auto_cl_mode)].copy()
    if sub.empty or metric_col not in sub.columns:
        return None, False
    sub["_file_order"] = np.arange(len(sub), dtype=np.int64)
    exact = sub[pd.to_numeric(sub.get("stage"), errors="coerce").eq(float(stage))].copy()
    for candidate, is_exact in [(exact, True), (sub, False)]:
        if candidate.empty:
            continue
        candidate = candidate.sort_values(["_file_order"], ascending=True)
        for _, rec in candidate.iloc[::-1].iterrows():
            rank_ids = [mid for mid in _parse_int_list(rec.get(metric_col)) if mid in model_ids]
            if rank_ids:
                return rec, is_exact
    return None, False


def _rank_row_groups_for_selection(
    df: pd.DataFrame,
    cache_stem: str,
    metric_col: str,
    eval_protocol: str,
    auto_cl_mode: str,
) -> dict[str, pd.DataFrame]:
    if df.empty or metric_col not in df.columns:
        return {}
    required = {"cache_stem", "dataset", "status"}
    if not required.issubset(set(df.columns)):
        return {}
    mask = (
        df["cache_stem"].astype(str).eq(str(cache_stem))
        & df["status"].astype(str).str.lower().isin({"success", "partial"})
        & _auto_cl_mode_mask(df, auto_cl_mode)
    )
    if "task_probe_eval_protocol" in df.columns:
        mask &= df["task_probe_eval_protocol"].astype(str).eq(str(eval_protocol))
    sub = df.loc[mask].copy()
    if sub.empty:
        return {}
    sub["_file_order"] = np.arange(len(sub), dtype=np.int64)
    if "stage" in sub.columns:
        sub["_stage_num"] = pd.to_numeric(sub["stage"], errors="coerce")
    else:
        sub["_stage_num"] = np.nan
    groups: dict[str, pd.DataFrame] = {}
    for dataset, group in sub.groupby(sub["dataset"].astype(str), sort=False):
        groups[str(dataset)] = group.sort_values("_file_order", kind="mergesort")
    return groups


def _latest_rank_row_from_group(
    group: pd.DataFrame | None,
    stage: int,
    model_ids: set[int],
    metric_col: str,
) -> tuple[pd.Series | None, bool]:
    if group is None or group.empty or metric_col not in group.columns:
        return None, False
    stage_values = (
        group["_stage_num"]
        if "_stage_num" in group.columns
        else pd.Series(np.nan, index=group.index)
    )
    exact = group[pd.to_numeric(stage_values, errors="coerce").eq(float(stage))].copy()
    for candidate, is_exact in [(exact, True), (group, False)]:
        if candidate.empty:
            continue
        for _, rec in candidate.iloc[::-1].iterrows():
            rank_ids = [mid for mid in _parse_int_list(rec.get(metric_col)) if mid in model_ids]
            if rank_ids:
                return rec, is_exact
    return None, False


def _forward_groups_for_selection(
    summary: pd.DataFrame,
    cache_stem: str,
    candidate_ids: Iterable[int],
    eval_protocol: str,
    auto_cl_mode: str,
) -> dict[str, pd.DataFrame]:
    if summary.empty:
        return {}
    required = {
        "cache_stem",
        "dataset",
        "model_id",
        "task_probe_eval_protocol",
        "glouts_MASE",
        "glouts_CRPS",
        "sample_forward_ms",
        "status",
    }
    if not required.issubset(set(summary.columns)):
        return {}
    ids = {int(x) for x in candidate_ids}
    sub = summary[
        summary["cache_stem"].astype(str).eq(str(cache_stem))
        & summary["task_probe_eval_protocol"].astype(str).eq(str(eval_protocol))
        & _auto_cl_mode_mask(summary, auto_cl_mode)
        & summary["status"].astype(str).str.lower().eq("success")
    ].copy()
    if sub.empty:
        return {}
    sub["model_id"] = pd.to_numeric(sub["model_id"], errors="coerce")
    sub = sub[sub["model_id"].isin(ids)].copy()
    if sub.empty:
        return {}
    sub["_file_order"] = np.arange(len(sub), dtype=np.int64)
    sub = (
        sub.sort_values("_file_order", kind="mergesort")
        .drop_duplicates(["dataset", "model_id"], keep="last")
        .drop(columns=["_file_order"])
    )
    for col in ["glouts_MASE", "glouts_sMAPE", "glouts_CRPS", "sample_forward_ms"]:
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return {
        str(dataset): group.copy()
        for dataset, group in sub.groupby(sub["dataset"].astype(str), sort=False)
    }


def _forward_metrics_from_group(
    groups: dict[str, pd.DataFrame],
    ds_config: str,
    candidate_ids: Iterable[int],
) -> pd.DataFrame:
    group = groups.get(str(ds_config))
    if group is None or group.empty:
        return pd.DataFrame()
    ids = {int(x) for x in candidate_ids}
    sub = group[pd.to_numeric(group["model_id"], errors="coerce").isin(ids)].copy()
    return sub if not sub.empty else pd.DataFrame()


def _forward_seconds_from_metric_df(
    metric_df: pd.DataFrame,
    model_ids: Iterable[int],
) -> float:
    if metric_df.empty:
        return float("nan")
    wanted = {int(x) for x in model_ids}
    present = {int(x) for x in pd.to_numeric(metric_df.get("model_id"), errors="coerce").dropna().astype(int).tolist()}
    if not wanted.issubset(present):
        return float("nan")
    vals = pd.to_numeric(metric_df.get("sample_forward_ms"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    if vals.dropna().shape[0] != len(wanted):
        return float("nan")
    return float(vals.sum() / 1000.0)


def _forward_seconds_from_summary(
    root: Path,
    cache_stem: str,
    ds_config: str,
    model_ids: Iterable[int],
    eval_protocol: str,
    auto_cl_mode: str,
    summary: pd.DataFrame | None = None,
) -> float:
    metric_df = _forward_metrics_for_rank(
        root,
        cache_stem,
        ds_config,
        list(model_ids),
        eval_protocol,
        auto_cl_mode,
        summary,
    )
    if metric_df.empty:
        return float("nan")
    wanted = {int(x) for x in model_ids}
    present = {int(x) for x in pd.to_numeric(metric_df.get("model_id"), errors="coerce").dropna().astype(int).tolist()}
    if not wanted.issubset(present):
        return float("nan")
    vals = pd.to_numeric(metric_df.get("sample_forward_ms"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    if vals.dropna().shape[0] != len(wanted):
        return float("nan")
    return float(vals.sum() / 1000.0)


def _sample_seconds_from_rank_or_timing(rank_row: pd.Series | None, cache_path: Path, ds_config: str) -> tuple[float, str]:
    if rank_row is not None and "sample_seconds" in rank_row.index:
        val = pd.to_numeric(pd.Series([rank_row.get("sample_seconds")]), errors="coerce").iloc[0]
        if pd.notna(val) and np.isfinite(float(val)) and float(val) >= 0:
            source = str(rank_row.get("sample_timing_source", "") or "")
            return float(val), source or "rank_summary.csv:sample_seconds"
    val, source = task_probe_sample_seconds_for_dataset(cache_path, ds_config)
    if val is None:
        return float("nan"), source
    return float(val), source


def _rank_from_forward_summary(
    root: Path,
    cache_stem: str,
    ds_config: str,
    model_ids: list[int],
    metric: str,
    eval_protocol: str,
    auto_cl_mode: str,
    summary: pd.DataFrame | None = None,
) -> list[int]:
    metric_df = _forward_metrics_for_rank(
        root,
        cache_stem,
        ds_config,
        model_ids,
        eval_protocol,
        auto_cl_mode,
        summary,
    )
    metric_col = "glouts_MASE" if metric == "MASE" else "glouts_CRPS"
    rank_ids = _rank_ids(metric_df, metric_col, model_ids)
    return [int(mid) for mid in rank_ids if int(mid) in set(model_ids)]


def _rank_from_forward_summary_timed(
    root: Path,
    cache_stem: str,
    ds_config: str,
    model_ids: list[int],
    metric: str,
    eval_protocol: str,
    auto_cl_mode: str,
    summary: pd.DataFrame | None = None,
) -> tuple[list[int], float]:
    t0 = time.perf_counter()
    rank_ids = _rank_from_forward_summary(
        root,
        cache_stem,
        ds_config,
        model_ids,
        metric,
        eval_protocol,
        auto_cl_mode,
        summary,
    )
    return rank_ids, time.perf_counter() - t0


def task_probe_select_selection_for_stage(
    *,
    args: Any,
    stage: int,
    metric: str,
    baseline_df: pd.DataFrame,
    ordered_model_names: list[str],
    expected: set[str],
    selector_df: pd.DataFrame | None = None,
    baseline_row_resolver: Any | None = None,
) -> tuple[pd.DataFrame, str]:
    root = task_probe_select_root(args)
    rank_df = _read_csv(_rank_summary_path(root))
    forward_summary_df: pd.DataFrame | None = None
    rank_group_cache: dict[tuple[str, str, str, str], dict[str, pd.DataFrame]] = {}
    forward_group_cache: dict[tuple[str, str, str, tuple[int, ...]], dict[str, pd.DataFrame]] = {}

    def get_forward_summary() -> pd.DataFrame:
        nonlocal forward_summary_df
        if forward_summary_df is None:
            forward_summary_df = _read_csv(_forward_summary_path(root))
        return forward_summary_df

    def get_rank_groups(cache_stem: str, eval_protocol: str, auto_cl_mode: str) -> dict[str, pd.DataFrame]:
        key = (str(cache_stem), str(eval_protocol), str(auto_cl_mode), str(rank_col))
        if key not in rank_group_cache:
            rank_group_cache[key] = _rank_row_groups_for_selection(
                rank_df,
                cache_stem,
                rank_col,
                eval_protocol,
                auto_cl_mode,
            )
        return rank_group_cache[key]

    def get_forward_groups(cache_stem: str, eval_protocol: str, auto_cl_mode: str) -> dict[str, pd.DataFrame]:
        key = (str(cache_stem), str(eval_protocol), str(auto_cl_mode), tuple(model_ids))
        if key not in forward_group_cache:
            forward_group_cache[key] = _forward_groups_for_selection(
                get_forward_summary(),
                cache_stem,
                model_ids,
                eval_protocol,
                auto_cl_mode,
            )
        return forward_group_cache[key]

    model_ids = list(range(int(stage)))
    model_id_set = set(model_ids)
    id_to_abbr = {idx: str(name) for idx, name in enumerate(ordered_model_names)}
    baseline_lookup = {
        (str(rec["dataset"]), str(rec["model"])): rec
        for _, rec in baseline_df.drop_duplicates(["dataset", "model"], keep="last").iterrows()
        if pd.notna(rec.get("dataset")) and pd.notna(rec.get("model"))
    }
    rows = []
    sources = []
    rank_col = "model_rank_mase" if metric == "MASE" else "model_rank_crps"
    selected_col = "selected_model_id_mase" if metric == "MASE" else "selected_model_id_crps"
    for ds_config in sorted(expected):
        concrete_args = args
        selector_row = None
        if get_auto_cl_mode(args) != "v0":
            if (
                selector_df is None
                or selector_df.empty
                or "dataset" not in selector_df.columns
            ):
                continue
            selector_sub = selector_df[
                selector_df["dataset"].astype(str).eq(str(ds_config))
            ].copy()
            if selector_sub.empty:
                continue
            selector_row = selector_sub.iloc[-1]
            concrete_args = _task_probe_concrete_args_from_selector_row(
                args, selector_row
            )
            if concrete_args is None:
                continue
        cache_path = task_probe_select_cache_path(concrete_args)
        cache_stem = cache_path.stem
        eval_protocol = task_probe_eval_protocol(concrete_args)
        auto_cl_mode = get_auto_cl_mode(concrete_args)
        rank_row, exact = _latest_rank_row_from_group(
            get_rank_groups(cache_stem, eval_protocol, auto_cl_mode).get(str(ds_config)),
            int(stage),
            model_id_set,
            rank_col,
        )
        metric_df = _forward_metrics_from_group(
            get_forward_groups(cache_stem, eval_protocol, auto_cl_mode),
            ds_config,
            model_ids,
        )
        if rank_row is not None and exact:
            rank_ids = [mid for mid in _parse_int_list(rank_row.get(rank_col)) if mid in model_id_set]
            rank_seconds = pd.to_numeric(pd.Series([rank_row.get("rank_seconds")]), errors="coerce").iloc[0]
            if not set(model_ids).issubset(set(rank_ids)):
                t_rank = time.perf_counter()
                computed = _rank_ids(
                    metric_df,
                    "glouts_MASE" if metric == "MASE" else "glouts_CRPS",
                    model_ids,
                )
                computed_rank_seconds = time.perf_counter() - t_rank
                if computed:
                    rank_ids = computed
                    rank_seconds = computed_rank_seconds
            sources.append(f"{_rank_summary_path(root).as_posix()}:{'exact' if exact else 'reused'}")
        else:
            t_rank = time.perf_counter()
            rank_ids = _rank_ids(
                metric_df,
                "glouts_MASE" if metric == "MASE" else "glouts_CRPS",
                model_ids,
            )
            rank_seconds = time.perf_counter() - t_rank
            if rank_ids:
                sources.append(f"{_forward_summary_path(root).as_posix()}:computed_rank")
            elif rank_row is not None:
                rank_ids = [mid for mid in _parse_int_list(rank_row.get(rank_col)) if mid in model_id_set]
                rank_seconds = np.nan
                sources.append(f"{_rank_summary_path(root).as_posix()}:reused")
        rank_ids = [int(mid) for mid in rank_ids if int(mid) in model_id_set]
        if not rank_ids:
            continue
        selected_id = int(rank_ids[0])
        selected_model = id_to_abbr.get(selected_id)
        if selected_model is None:
            continue
        if baseline_row_resolver is not None:
            rec = baseline_row_resolver(
                str(ds_config), str(selected_model), concrete_args
            )
        else:
            rec = baseline_lookup.get((str(ds_config), str(selected_model)))
        if rec is None:
            continue
        row = rec.copy()
        sample_seconds, sample_source = _sample_seconds_from_rank_or_timing(
            rank_row, cache_path, ds_config
        )
        forward_seconds = _forward_seconds_from_metric_df(metric_df, model_ids)
        route_final = (
            float(sample_seconds) + float(forward_seconds) + float(rank_seconds)
            if np.isfinite(float(sample_seconds)) and np.isfinite(float(forward_seconds)) and np.isfinite(float(rank_seconds))
            else np.nan
        )
        if sample_source:
            sources.append(f"{sample_source}:sample")
        row["_selected_tsfm_model"] = selected_model
        row["selected_model_id"] = selected_id
        row["model_order"] = rank_ids
        row["sample_seconds"] = sample_seconds
        row["task_probe_forward_seconds"] = forward_seconds
        row["task_probe_rank_seconds"] = rank_seconds
        row["route_final_seconds"] = route_final
        row["model"] = "Task-probe-M" if metric == "MASE" else "Task-probe-C"
        row.update(_sample_config_row(concrete_args))
        row["task_sample_cache_path"] = str(cache_path)
        row["sample_timing_source"] = sample_source
        row["_tsfm_forward_cl_ok"] = bool(
            np.isfinite(float(forward_seconds)) and float(forward_seconds) >= 0
        )
        if selector_row is not None:
            for col in TASK_PROBE_PROFILE_COLUMNS:
                if col in selector_row.index and str(selector_row.get(col, "")).strip():
                    row[col] = selector_row.get(col)
        rows.append(row)
    cache_text = "per-dataset-auto-cl" if get_auto_cl_mode(args) != "v0" else task_probe_select_cache_stem(args)
    source = (
        f"{root.as_posix()} cache={cache_text}; "
        f"rank_summary={_rank_summary_path(root).as_posix()}; "
        f"forward_summary={_forward_summary_path(root).as_posix()}"
    )
    if sources:
        source += f"; sources={','.join(sorted(set(sources))[:3])}"
    return pd.DataFrame(rows), source

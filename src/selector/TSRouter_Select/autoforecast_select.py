from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import time
from collections import OrderedDict
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from config.model_zoo_config import Model_abbrev_map, build_model_family_metadata
from encoder.baseline.random_stats_features import compute_stats_features
from utils.io_lock import atomic_pickle_dump, file_lock
from utils.path_utils import (
    TSROUTER_SAMPLED_REPR_POOL_DIR,
    build_repr_eval_pool_forward_stem,
    build_repr_eval_pool_name,
    build_repr_forward_stem,
    get_advanced_baseline_train_scope,
    get_tsrouter_repr_forward_dir,
    route_efficiency_mode_enabled,
)
from utils.project_paths import BASELINE_ARTIFACT_ROOT, BASELINE_CSV_ROOT


AUTOFORECAST_REPR_FORMAT = "autoforecast_v1"
AUTOFORECAST_METHOD_VERSION = "autoforecast_lite_v2"
AUTOFORECAST_ARTIFACT_SCHEMA_VERSION = 3
AUTOXPCR_METHOD_VERSION = "autoxpcr_lite_v1"
AUTOXPCR_ARTIFACT_SCHEMA_VERSION = 1
AUTOFORECAST_FEATURE_CACHE_FORMAT = "autoforecast_feature_cache_v1"
AUTOFORECAST_SELECTOR_NAME = "AutoForecast_Select"
AUTOXPCR_SELECTOR_NAME = "AutoXPCR_Select"
AUTOFORECAST_DEFAULT_LEARNER = "LSTM"
AUTOFORECAST_LEARNERS = {"LSTM", "GBDT", "MLP"}
AUTOFORECAST_SELECTOR_MODE = "autoforecast"
AUTOXPCR_SELECTOR_MODE = "autoxpcr"
AUTOXPCR_P_WEIGHT = 0.8
AUTOXPCR_R_WEIGHT = 0.2
AUTOXPCR_C_WEIGHT = 0.0

STATS_FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "q25",
    "q50",
    "q75",
    "iqr",
    "first_value",
    "last_value",
    "slope",
    "mean_abs_diff",
    "diff_std",
    "max_abs_diff",
    "sign_change_rate",
    "mean_square",
]

SHAPE_FEATURE_NAMES = [
    "missing_ratio",
    "zero_ratio",
    "negative_ratio",
    "coefficient_of_variation",
    "relative_range",
    "last_minus_first",
    "absolute_last_minus_first",
    "mean_abs_level",
    "median_abs_level",
]

FREQUENCY_FEATURE_NAMES = [
    "acf_lag1",
    "acf_lag2",
    "acf_lag_seasonal_proxy",
    "fft_top1_power_ratio",
    "fft_top3_power_ratio",
    "spectral_entropy",
]

AGGREGATIONS = ["mean", "std", "min", "max"]


class ConstantRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, x):
        return np.full((len(x),), self.value, dtype=np.float64)


class _TorchMetaRegressorNet(nn.Module):
    def __init__(self, feature_dim: int, output_dim: int, hidden_dim: int, architecture: str):
        super().__init__()
        self.architecture = normalize_autoforecast_learner(architecture)
        if self.architecture == "LSTM":
            self.temporal = nn.LSTM(
                input_size=int(feature_dim),
                hidden_size=int(hidden_dim),
                num_layers=1,
                batch_first=True,
            )
            self.head = nn.Linear(int(hidden_dim), int(output_dim))
        elif self.architecture == "MLP":
            self.net = nn.Sequential(
                nn.Linear(int(feature_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(output_dim)),
            )
        else:
            raise ValueError(f"unsupported torch AutoForecast learner: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.architecture == "LSTM":
            seq = x.unsqueeze(1)
            out, _ = self.temporal(seq)
            return self.head(out[:, -1, :])
        return self.net(x)


class TorchMultiOutputRegressor:
    def __init__(
        self,
        *,
        learner: str,
        feature_dim: int,
        output_dim: int,
        hidden_dim: int,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        state_dict: dict,
    ):
        self.learner = normalize_autoforecast_learner(learner)
        self.feature_dim = int(feature_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.x_mean = np.asarray(x_mean, dtype=np.float32)
        self.x_std = np.asarray(x_std, dtype=np.float32)
        self.y_mean = np.asarray(y_mean, dtype=np.float32)
        self.y_std = np.asarray(y_std, dtype=np.float32)
        self.state_dict = {key: value.detach().cpu() for key, value in state_dict.items()}

    def _build_model(self) -> _TorchMetaRegressorNet:
        model = _TorchMetaRegressorNet(
            feature_dim=self.feature_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            architecture=self.learner,
        )
        model.load_state_dict(self.state_dict)
        model.eval()
        return model

    def predict(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.feature_dim:
            raise ValueError(
                f"AutoForecast {self.learner} expected feature matrix (*,{self.feature_dim}), got {arr.shape}"
            )
        x_norm = (arr - self.x_mean) / np.maximum(self.x_std, 1e-6)
        model = self._build_model()
        with torch.no_grad():
            pred_norm = model(torch.as_tensor(x_norm, dtype=torch.float32)).cpu().numpy()
        return pred_norm * np.maximum(self.y_std, 1e-6) + self.y_mean


def is_autoforecast_v7(args) -> bool:
    raw = str(getattr(args, "repr_v", "") or "")
    return raw[:1] == "7"


def is_autoxpcr_v7(args) -> bool:
    return is_autoforecast_v7(args) and route_efficiency_mode_enabled(args)


def autoforecast_selector_mode(args=None) -> str:
    return AUTOXPCR_SELECTOR_MODE if args is not None and is_autoxpcr_v7(args) else AUTOFORECAST_SELECTOR_MODE


def _normalize_selector_mode(value) -> str:
    mode = str(value or AUTOFORECAST_SELECTOR_MODE).strip().lower()
    if mode not in {AUTOFORECAST_SELECTOR_MODE, AUTOXPCR_SELECTOR_MODE}:
        raise ValueError(f"unsupported v7 selector_mode={value!r}")
    return mode


def _method_version_for_mode(selector_mode: str) -> str:
    return (
        AUTOXPCR_METHOD_VERSION
        if _normalize_selector_mode(selector_mode) == AUTOXPCR_SELECTOR_MODE
        else AUTOFORECAST_METHOD_VERSION
    )


def _artifact_schema_for_mode(selector_mode: str) -> int:
    return (
        AUTOXPCR_ARTIFACT_SCHEMA_VERSION
        if _normalize_selector_mode(selector_mode) == AUTOXPCR_SELECTOR_MODE
        else AUTOFORECAST_ARTIFACT_SCHEMA_VERSION
    )


def _selector_name_for_mode(selector_mode: str) -> str:
    return (
        AUTOXPCR_SELECTOR_NAME
        if _normalize_selector_mode(selector_mode) == AUTOXPCR_SELECTOR_MODE
        else AUTOFORECAST_SELECTOR_NAME
    )


def normalize_autoforecast_learner(value) -> str:
    raw = str(value or AUTOFORECAST_DEFAULT_LEARNER).strip().upper()
    if raw == "GDBT":
        raw = "GBDT"
    if raw in {"HGBDT", "HISTGBDT", "HISTGRADIENTBOOSTING"}:
        raw = "GBDT"
    if raw not in AUTOFORECAST_LEARNERS:
        raise ValueError(
            f"AutoForecast learner must be one of LSTM/GBDT/MLP, got {value!r}"
        )
    return raw


def autoforecast_learner_tag(args) -> str:
    learner = normalize_autoforecast_learner(getattr(args, "autoforecast_learner", AUTOFORECAST_DEFAULT_LEARNER))
    return f"af{learner.lower()}"


def normalize_autoforecast_metric(value) -> str:
    raw = str(value or "MASE").strip()
    upper = raw.upper()
    if upper in {"M", "MASE"}:
        return "MASE"
    if upper in {"C", "CRPS"}:
        return "CRPS"
    raise ValueError(
        f"AutoForecast v7 supports base_metrics=MASE/M or CRPS/C, got {raw!r}"
    )


def autoforecast_method_name(metric: str, selector_mode: str = AUTOFORECAST_SELECTOR_MODE) -> str:
    metric = normalize_autoforecast_metric(metric)
    if _normalize_selector_mode(selector_mode) == AUTOXPCR_SELECTOR_MODE:
        return "AutoXPCR-M" if metric == "MASE" else "AutoXPCR-C"
    return "AutoForecast-M" if metric == "MASE" else "AutoForecast-C"


def autoforecast_csv_root(selector_mode: str = AUTOFORECAST_SELECTOR_MODE) -> Path:
    return BASELINE_CSV_ROOT / "selectors" / _selector_name_for_mode(selector_mode)


def autoforecast_artifact_root(selector_mode: str = AUTOFORECAST_SELECTOR_MODE) -> Path:
    return BASELINE_ARTIFACT_ROOT / "selectors" / _selector_name_for_mode(selector_mode)


def _log_step3(message: str, selector_mode: str = AUTOFORECAST_SELECTOR_MODE) -> None:
    label = "AutoXPCR" if _normalize_selector_mode(selector_mode) == AUTOXPCR_SELECTOR_MODE else "AutoForecast"
    print(f"[{label} Step3] {message}", flush=True)


def _sklearn_threadpool_limit():
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return nullcontext()
    return threadpool_limits(limits=1)


def autoforecast_feature_names() -> list[str]:
    base_names = (
        [f"stats_{name}" for name in STATS_FEATURE_NAMES]
        + [f"shape_{name}" for name in SHAPE_FEATURE_NAMES]
        + [f"freq_{name}" for name in FREQUENCY_FEATURE_NAMES]
    )
    return [f"{name}_{agg}" for name in base_names for agg in AGGREGATIONS]


def _coerce_windows_array(windows) -> np.ndarray:
    x = np.asarray(windows, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :, None]
    elif x.ndim == 2:
        x = x[:, :, None]
    elif x.ndim == 3:
        pass
    else:
        raise ValueError(f"AutoForecast features expect 1D/2D/3D windows, got shape={x.shape}")
    if x.shape[1] <= 0 or x.shape[2] <= 0:
        raise ValueError(f"AutoForecast windows must have positive T and C, got shape={x.shape}")
    return x


def _fill_channel_nans(raw: np.ndarray) -> np.ndarray:
    clean = np.asarray(raw, dtype=np.float64).copy()
    if clean.ndim != 2:
        raise ValueError(f"channel matrix must be 2D, got shape={clean.shape}")
    for row_idx in range(clean.shape[0]):
        row = clean[row_idx]
        finite = np.isfinite(row)
        fill = float(np.nanmean(row[finite])) if finite.any() else 0.0
        row[~finite] = fill
        clean[row_idx] = row
    return clean


def _shape_features(raw: np.ndarray) -> np.ndarray:
    clean = _fill_channel_nans(raw)
    finite = np.isfinite(raw)
    missing_ratio = 1.0 - finite.mean(axis=1)
    abs_mean = np.mean(np.abs(clean), axis=1)
    std = np.std(clean, axis=1)
    min_v = np.min(clean, axis=1)
    max_v = np.max(clean, axis=1)
    return np.stack(
        [
            missing_ratio,
            np.mean(np.isclose(clean, 0.0, atol=1e-8), axis=1),
            np.mean(clean < 0.0, axis=1),
            std / np.maximum(abs_mean, 1e-8),
            (max_v - min_v) / np.maximum(abs_mean, 1e-8),
            clean[:, -1] - clean[:, 0],
            np.abs(clean[:, -1] - clean[:, 0]),
            abs_mean,
            np.median(np.abs(clean), axis=1),
        ],
        axis=1,
    )


def _acf_1d(values: np.ndarray, lag: int) -> float:
    lag = int(lag)
    if lag <= 0 or values.size <= lag:
        return 0.0
    centered = values - float(np.mean(values))
    denom = float(np.sum(centered * centered))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(centered[:-lag] * centered[lag:]) / denom)


def _frequency_features(raw: np.ndarray) -> np.ndarray:
    clean = _fill_channel_nans(raw)
    rows: list[list[float]] = []
    for row in clean:
        length = int(row.size)
        seasonal_lag = 24 if length > 24 else (7 if length > 7 else 1)
        centered = row - float(np.mean(row))
        power = np.abs(np.fft.rfft(centered)) ** 2
        if power.size > 1:
            power = power[1:]
        total = float(np.sum(power))
        if total <= 1e-12:
            top1 = 0.0
            top3 = 0.0
            entropy = 0.0
        else:
            sorted_power = np.sort(power)[::-1]
            top1 = float(sorted_power[0] / total)
            top3 = float(sorted_power[: min(3, sorted_power.size)].sum() / total)
            prob = power / total
            entropy_raw = -float(np.sum(prob * np.log(prob + 1e-12)))
            entropy = entropy_raw / float(np.log(max(2, prob.size)))
        rows.append(
            [
                _acf_1d(row, 1),
                _acf_1d(row, 2),
                _acf_1d(row, seasonal_lag),
                top1,
                top3,
                entropy,
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def _aggregate_channel_features(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate(
        [
            np.mean(arr, axis=0),
            np.std(arr, axis=0),
            np.min(arr, axis=0),
            np.max(arr, axis=0),
        ],
        axis=0,
    )


def feature_matrix_from_windows(
    windows,
    *,
    progress_label: str | None = None,
    log_every: int = 500,
) -> np.ndarray:
    x = _coerce_windows_array(windows)
    rows = []
    total = int(x.shape[0])
    progress_t0 = time.perf_counter()
    if progress_label:
        print(
            f"{progress_label} begin: windows={total}, shape={tuple(x.shape)}, log_every={log_every}",
            flush=True,
        )
    for sample_idx, sample in enumerate(x, start=1):
        raw_channels = np.transpose(sample, (1, 0)).astype(np.float32, copy=False)
        clean_channels = _fill_channel_nans(raw_channels)
        with torch.no_grad():
            stats = compute_stats_features(
                torch.as_tensor(clean_channels, dtype=torch.float32),
                normalize=True,
            )
        stats_np = stats.detach().cpu().numpy().astype(np.float64)
        all_channel_features = np.concatenate(
            [
                stats_np,
                _shape_features(raw_channels),
                _frequency_features(raw_channels),
            ],
            axis=1,
        )
        rows.append(_aggregate_channel_features(all_channel_features))
        if progress_label and (
            sample_idx == total
            or sample_idx == 1
            or (log_every > 0 and sample_idx % int(log_every) == 0)
        ):
            elapsed = time.perf_counter() - progress_t0
            rate = sample_idx / max(elapsed, 1e-9)
            print(
                f"{progress_label} progress: {sample_idx}/{total} windows, "
                f"elapsed={elapsed:.1f}s, rate={rate:.1f}/s",
                flush=True,
            )
    return np.asarray(rows, dtype=np.float32)


def feature_tensor_from_samples(samples) -> np.ndarray:
    x = _coerce_windows_array(samples)
    n, t, c = x.shape
    flat_windows = np.transpose(x, (0, 2, 1)).reshape(n * c, t, 1)
    features = feature_matrix_from_windows(flat_windows)
    return features.reshape(n, c, -1)


def _repr_forward_stem_for_train_scope(args, train_scope: str) -> str:
    if str(train_scope) == "full_pool":
        return build_repr_eval_pool_forward_stem(args)
    return build_repr_forward_stem(args)


def _repr_forward_csv_path_for_train_scope(args, train_scope: str) -> str:
    return os.path.join(
        get_tsrouter_repr_forward_dir(args),
        _repr_forward_stem_for_train_scope(args, train_scope) + "_per_sample_results.csv",
    )


def _training_feature_source_path(args, repr_set_name: str, train_scope: str) -> Path:
    if str(train_scope) == "full_pool":
        return Path(TSROUTER_SAMPLED_REPR_POOL_DIR) / f"{build_repr_eval_pool_name(args)}.pkl"
    return Path(str(getattr(args, "save_repr_data_path"))) / f"{repr_set_name}.pkl"


def _load_repr_windows(args, repr_set_name: str, train_scope: str = "center") -> tuple[np.ndarray, str]:
    path = _training_feature_source_path(args, repr_set_name, train_scope)
    if not path.exists():
        raise FileNotFoundError(
            f"AutoForecast v7 missing Step1 training windows for scope={train_scope}: {path}"
        )
    with path.open("rb") as f:
        payload = pickle.load(f)
    arr = _coerce_windows_array(payload)
    return arr, str(path)


def _file_signature(path_value: str) -> dict:
    if not str(path_value or ""):
        return {"path": "", "source_size": -1, "source_mtime_ns": -1}
    path = Path(path_value)
    signature = {"path": str(path)}
    try:
        stat = path.stat()
    except OSError:
        signature.update({"source_size": -1, "source_mtime_ns": -1})
    else:
        signature.update(
            {
                "source_size": int(stat.st_size),
                "source_mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return signature


def _feature_source_signature(feature_source: str, windows: np.ndarray) -> dict:
    signature = {
        **_file_signature(feature_source),
        "window_shape": [int(x) for x in np.asarray(windows).shape],
        "window_dtype": str(np.asarray(windows).dtype),
    }
    return signature


def _feature_cache_path(cache_root: Path, signature: dict) -> Path:
    cache_contract = {
        "format": AUTOFORECAST_FEATURE_CACHE_FORMAT,
        "feature_names": autoforecast_feature_names(),
        "source": signature,
    }
    digest = hashlib.sha256(
        json.dumps(cache_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    return cache_root / f"step3_features_{digest}.pkl"


def _load_or_build_feature_matrix(
    windows: np.ndarray,
    feature_source: str,
    *,
    cache_root: Path,
    log_every: int,
    selector_mode: str = AUTOFORECAST_SELECTOR_MODE,
) -> tuple[np.ndarray, str, str]:
    signature = _feature_source_signature(feature_source, windows)
    cache_path = _feature_cache_path(cache_root, signature)
    expected_feature_dim = len(autoforecast_feature_names())
    if cache_path.is_file():
        try:
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            matrix = np.asarray(cached.get("feature_matrix"), dtype=np.float32)
            valid = (
                cached.get("format") == AUTOFORECAST_FEATURE_CACHE_FORMAT
                and cached.get("source_signature") == signature
                and list(cached.get("feature_names", [])) == autoforecast_feature_names()
                and matrix.shape == (int(np.asarray(windows).shape[0]), expected_feature_dim)
            )
            if valid:
                _log_step3(
                    f"feature cache hit: shape={tuple(matrix.shape)}, path={cache_path}",
                    selector_mode,
                )
                return matrix, "hit", str(cache_path)
        except Exception as exc:
            _log_step3(
                f"feature cache rejected: reason={type(exc).__name__}, path={cache_path}",
                selector_mode,
            )

    _log_step3(f"feature cache miss: build features -> {cache_path}", selector_mode)
    selector_label = (
        "AutoXPCR"
        if _normalize_selector_mode(selector_mode) == AUTOXPCR_SELECTOR_MODE
        else "AutoForecast"
    )
    matrix = feature_matrix_from_windows(
        windows,
        progress_label=f"[{selector_label} Step3][features]",
        log_every=log_every,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_pickle_dump(
        {
            "format": AUTOFORECAST_FEATURE_CACHE_FORMAT,
            "source_signature": signature,
            "feature_names": autoforecast_feature_names(),
            "feature_matrix": matrix,
        },
        str(cache_path),
    )
    _log_step3(
        f"feature cache saved: shape={tuple(matrix.shape)}, path={cache_path}",
        selector_mode,
    )
    return matrix, "miss_built", str(cache_path)


def _ordered_metric_matrix(
    metric_dict: OrderedDict,
    model_order: list[str],
) -> np.ndarray:
    missing = [name for name in model_order if name not in metric_dict]
    if missing:
        raise ValueError(f"AutoForecast label matrix missing models: {missing}")
    matrix = np.stack([np.asarray(metric_dict[name], dtype=np.float64) for name in model_order], axis=1)
    return matrix


def _minmax_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"row MinMax expects a 2D matrix, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("row MinMax requires finite values")
    row_min = np.min(matrix, axis=1, keepdims=True)
    row_span = np.max(matrix, axis=1, keepdims=True) - row_min
    return np.divide(
        matrix - row_min,
        row_span,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_span > 1e-12,
    )


def _minmax_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(vector).all():
        raise ValueError("vector MinMax requires finite values")
    span = float(np.max(vector) - np.min(vector)) if vector.size else 0.0
    if span <= 1e-12:
        return np.zeros_like(vector, dtype=np.float64)
    return (vector - float(np.min(vector))) / span


def _build_autoxpcr_target_matrix(
    quality_matrix: np.ndarray,
    *,
    runtime_by_model_seconds: dict[str, float],
    model_order: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quality = np.asarray(quality_matrix, dtype=np.float64)
    if quality.ndim != 2 or quality.shape[1] != len(model_order):
        raise ValueError(
            f"AutoXPCR quality/model shape mismatch: quality={quality.shape}, models={len(model_order)}"
        )
    missing = [name for name in model_order if name not in runtime_by_model_seconds]
    if missing:
        raise ValueError(f"AutoXPCR forward_runtime_seconds missing models: {missing}")
    runtime = np.asarray(
        [runtime_by_model_seconds[name] for name in model_order],
        dtype=np.float64,
    )
    if not np.isfinite(runtime).all() or np.any(runtime < 0):
        raise ValueError(
            f"AutoXPCR forward_runtime_seconds must be finite and non-negative: "
            f"{dict(zip(model_order, runtime.tolist()))}"
        )
    p_score = _minmax_rows(quality)
    r_score = _minmax_vector(runtime)
    target = AUTOXPCR_P_WEIGHT * p_score + AUTOXPCR_R_WEIGHT * r_score.reshape(1, -1)
    return target, p_score, r_score


def _resource_score_fingerprint(r_score: np.ndarray, model_order: list[str]) -> str:
    vector = np.asarray(r_score, dtype="<f8").reshape(-1)
    if vector.size != len(model_order):
        raise ValueError(
            f"AutoXPCR resource fingerprint size mismatch: scores={vector.size}, models={len(model_order)}"
        )
    digest = hashlib.sha256()
    digest.update(b"autoxpcr-resource-score-v1\0")
    digest.update(json.dumps(list(model_order), separators=(",", ":")).encode("utf-8"))
    digest.update(vector.tobytes())
    return digest.hexdigest()


def _classify_incremental_measurement(
    *,
    selector_mode: str,
    incremental_source_stage: int,
    timing_valid: bool,
    feature_cache_status: str,
    reused_regressor_count: int,
    trained_regressor_count: int,
    model_count: int,
    label_changed_models: list[str],
    xpcr_target_changed_models: list[str],
) -> str:
    selector_mode = _normalize_selector_mode(selector_mode)
    if int(incremental_source_stage) <= 0:
        return "initial_build"
    if label_changed_models:
        return "corrected_old_labels"
    if selector_mode == AUTOXPCR_SELECTOR_MODE and xpcr_target_changed_models:
        return "stage_rescaled_retrain"
    if str(feature_cache_status).strip().lower() != "hit":
        return "cold_feature_rebuild"
    if (
        bool(timing_valid)
        and int(reused_regressor_count) == int(model_count) - 1
        and int(trained_regressor_count) == 1
    ):
        return "ordinary_incremental"
    return "nonstandard_rebuild"


def _target_fingerprints(
    metric_matrix: np.ndarray,
    finite_rows: np.ndarray,
    model_order: list[str],
) -> dict[str, str]:
    matrix = np.asarray(metric_matrix)
    row_mask = np.asarray(finite_rows, dtype=bool).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != row_mask.size:
        raise ValueError(
            f"AutoForecast target fingerprint shape mismatch: matrix={matrix.shape}, mask={row_mask.shape}"
        )
    if matrix.shape[1] != len(model_order):
        raise ValueError(
            f"AutoForecast target fingerprint model mismatch: columns={matrix.shape[1]}, models={len(model_order)}"
        )
    mask_bytes = np.ascontiguousarray(row_mask.astype(np.uint8)).tobytes()
    out: dict[str, str] = {}
    for model_idx, model_name in enumerate(model_order):
        target = np.ascontiguousarray(matrix[row_mask, model_idx], dtype="<f4")
        digest = hashlib.sha256()
        digest.update(b"autoforecast-target-v1\0")
        digest.update(mask_bytes)
        digest.update(str(target.shape).encode("ascii"))
        digest.update(target.tobytes())
        out[str(model_name)] = digest.hexdigest()
    return out


def _validate_legacy_target_reconstruction(
    payload: dict,
    metric_matrix: np.ndarray,
    finite_rows: np.ndarray,
) -> None:
    matrix = np.asarray(metric_matrix)
    row_mask = np.asarray(finite_rows, dtype=bool).reshape(-1)
    saved_means = np.asarray(payload.get("global_label_mean", []), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != row_mask.size:
        raise ValueError("legacy target reconstruction shape mismatch")
    reconstructed_means = np.nanmean(matrix[row_mask], axis=0).astype(np.float64)
    if saved_means.shape != reconstructed_means.shape:
        raise ValueError("legacy artifact target mean shape mismatch")
    if not np.allclose(saved_means, reconstructed_means, rtol=1e-6, atol=1e-7):
        raise ValueError("legacy artifact target means changed")


def _previous_stage_artifact_path(
    artifact_path: Path,
    *,
    current_stage: int,
    previous_stage: int,
) -> Path:
    current_token = f"zoo{int(current_stage)}-"
    if current_token not in artifact_path.name:
        raise ValueError(
            f"AutoForecast artifact name lacks current stage token {current_token!r}: {artifact_path}"
        )
    previous_name = artifact_path.name.replace(
        current_token,
        f"zoo{int(previous_stage)}-",
        1,
    )
    return artifact_path.parent.parent / f"stage{int(previous_stage)}" / previous_name


def _find_previous_gbdt_artifact(
    artifact_path: Path,
    *,
    current_stage: int,
    expected_order: list[str],
    target_metric: str,
    repr_set_name: str,
    train_scope: str = "center",
    selector_mode: str = AUTOFORECAST_SELECTOR_MODE,
) -> tuple[dict | None, Path | None, str]:
    selector_mode = _normalize_selector_mode(selector_mode)
    expected_method_version = _method_version_for_mode(selector_mode)
    expected_schema = _artifact_schema_for_mode(selector_mode)
    if current_stage <= 1:
        return None, None, "no_previous_stage"
    rejection_reasons: list[str] = []
    for previous_stage in range(current_stage - 1, 0, -1):
        try:
            candidate = _previous_stage_artifact_path(
                artifact_path,
                current_stage=current_stage,
                previous_stage=previous_stage,
            )
        except ValueError as exc:
            return None, None, str(exc)
        if not candidate.exists():
            continue
        try:
            with candidate.open("rb") as f:
                payload = pickle.load(f)
        except Exception as exc:
            rejection_reasons.append(f"stage{previous_stage}:load_error={type(exc).__name__}")
            continue
        old_order = [str(x) for x in payload.get("model_abbr_order", payload.get("model_names", []))]
        try:
            payload_learner = normalize_autoforecast_learner(payload.get("learner", ""))
            payload_metric = normalize_autoforecast_metric(payload.get("target_metric", ""))
            payload_mode = _normalize_selector_mode(
                payload.get("selector_mode", AUTOFORECAST_SELECTOR_MODE)
            )
            payload_schema = int(payload.get("artifact_schema_version", 0) or 0)
        except (TypeError, ValueError) as exc:
            rejection_reasons.append(
                f"stage{previous_stage}:invalid_metadata={type(exc).__name__}"
            )
            continue
        checks = {
            "repr_format": payload.get("__repr_format__") == AUTOFORECAST_REPR_FORMAT,
            "method_version": payload.get("method_version") == expected_method_version,
            "artifact_schema_version": payload_schema == expected_schema,
            "selector_mode": payload_mode == selector_mode,
            "learner": payload_learner == "GBDT",
            "target_metric": payload_metric == target_metric,
            "repr_set_name": str(payload.get("repr_set_name", "")) == str(repr_set_name),
            "advanced_baseline_train_scope": str(
                payload.get("advanced_baseline_train_scope", "center")
            ) == str(train_scope),
            "strict_prefix": bool(old_order) and old_order == expected_order[: len(old_order)] and len(old_order) < len(expected_order),
            "regressor_count": len(payload.get("regressors", [])) == len(old_order),
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            rejection_reasons.append(f"stage{previous_stage}:failed={','.join(failed)}")
            continue
        return payload, candidate, "compatible_prefix"
    reason = ";".join(rejection_reasons) if rejection_reasons else "no_previous_artifact"
    return None, None, reason


def _make_regressor(seed: int):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        HistGradientBoostingRegressor(max_iter=64, learning_rate=0.08, random_state=seed),
    )


def _fit_per_model_regressors(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    *,
    progress_label: str | None = None,
    reusable_regressors: dict[int, tuple[object, str]] | None = None,
) -> tuple[list[object], list[str]]:
    regressors: list[object] = []
    statuses: list[str] = []
    reusable_regressors = dict(reusable_regressors or {})
    t0 = time.perf_counter()
    if progress_label:
        print(
            f"{progress_label} begin: learner=GBDT, train_samples={x.shape[0]}, "
            f"feature_dim={x.shape[1]}, models={y.shape[1]}, "
            f"reuse={len(reusable_regressors)}, train={y.shape[1] - len(reusable_regressors)}",
            flush=True,
        )
    for model_idx in range(y.shape[1]):
        if model_idx in reusable_regressors:
            reg, old_status = reusable_regressors[model_idx]
            regressors.append(reg)
            statuses.append(str(old_status or type(reg).__name__))
            if progress_label:
                print(
                    f"{progress_label} progress: model={model_idx + 1}/{y.shape[1]}, "
                    f"action=reuse, status={statuses[-1]}, elapsed={time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
            continue
        target = np.asarray(y[:, model_idx], dtype=np.float64)
        finite = np.isfinite(target)
        if finite.sum() == 0:
            regressors.append(ConstantRegressor(float("nan")))
            statuses.append("constant_nan_no_finite_labels")
            if progress_label:
                print(
                    f"{progress_label} progress: model={model_idx + 1}/{y.shape[1]}, "
                    f"action=train, status={statuses[-1]}, elapsed={time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
            continue
        clean_target = target[finite]
        clean_x = x[finite]
        if clean_target.size < 2 or np.nanmax(clean_target) - np.nanmin(clean_target) <= 1e-12:
            regressors.append(ConstantRegressor(float(np.nanmean(clean_target))))
            statuses.append("constant")
            if progress_label:
                print(
                    f"{progress_label} progress: model={model_idx + 1}/{y.shape[1]}, "
                    f"action=train, status={statuses[-1]}, elapsed={time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
            continue
        reg = _make_regressor(seed + model_idx)
        try:
            with _sklearn_threadpool_limit():
                reg.fit(clean_x, clean_target)
            regressors.append(reg)
            statuses.append(type(reg[-1]).__name__ if hasattr(reg, "__getitem__") else type(reg).__name__)
        except Exception:
            regressors.append(ConstantRegressor(float(np.nanmean(clean_target))))
            statuses.append("constant_after_fit_error")
        if progress_label:
            print(
                f"{progress_label} progress: model={model_idx + 1}/{y.shape[1]}, "
                f"action=train, status={statuses[-1]}, elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )
    if progress_label:
        print(f"{progress_label} done: elapsed={time.perf_counter() - t0:.1f}s", flush=True)
    return regressors, statuses


def _fit_torch_multioutput_regressor(
    x: np.ndarray,
    y: np.ndarray,
    *,
    learner: str,
    seed: int,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    progress_label: str | None = None,
) -> TorchMultiOutputRegressor:
    learner = normalize_autoforecast_learner(learner)
    if learner not in {"LSTM", "MLP"}:
        raise ValueError(f"torch multi-output trainer only supports LSTM/MLP, got {learner}")
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    if x_arr.ndim != 2 or y_arr.ndim != 2:
        raise ValueError(f"AutoForecast torch trainer expects 2D x/y, got x={x_arr.shape}, y={y_arr.shape}")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"AutoForecast torch trainer row mismatch: x={x_arr.shape}, y={y_arr.shape}")

    x_mean = np.nanmean(x_arr, axis=0).astype(np.float32)
    x_std = np.nanstd(x_arr, axis=0).astype(np.float32)
    y_mean = np.nanmean(y_arr, axis=0).astype(np.float32)
    y_std = np.nanstd(y_arr, axis=0).astype(np.float32)
    x_norm = (np.nan_to_num(x_arr, nan=0.0) - x_mean) / np.maximum(x_std, 1e-6)
    y_norm = (np.nan_to_num(y_arr, nan=0.0) - y_mean) / np.maximum(y_std, 1e-6)

    torch.manual_seed(int(seed))
    model = _TorchMetaRegressorNet(
        feature_dim=x_arr.shape[1],
        output_dim=y_arr.shape[1],
        hidden_dim=max(1, int(hidden_dim)),
        architecture=learner,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loss_fn = nn.MSELoss()
    x_tensor = torch.as_tensor(x_norm, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_norm, dtype=torch.float32)
    n = int(x_tensor.shape[0])
    batch = max(1, int(batch_size))
    epochs = max(1, int(epochs))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    t0 = time.perf_counter()
    if progress_label:
        print(
            f"{progress_label} begin: learner={learner}, train_samples={n}, "
            f"feature_dim={x_arr.shape[1]}, models={y_arr.shape[1]}, "
            f"hidden_dim={max(1, int(hidden_dim))}, epochs={epochs}, batch_size={batch}, "
            f"lr={float(learning_rate):g}",
            flush=True,
        )
    log_interval = max(1, epochs // 5)
    model.train()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, generator=generator)
        loss_sum = 0.0
        loss_count = 0
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_tensor[idx]), y_tensor[idx])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * int(idx.numel())
            loss_count += int(idx.numel())
        if progress_label and (epoch == 1 or epoch == epochs or epoch % log_interval == 0):
            avg_loss = loss_sum / max(loss_count, 1)
            print(
                f"{progress_label} epoch {epoch}/{epochs}: loss={avg_loss:.6f}, "
                f"elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )
    model.eval()
    if progress_label:
        print(f"{progress_label} done: elapsed={time.perf_counter() - t0:.1f}s", flush=True)
    return TorchMultiOutputRegressor(
        learner=learner,
        feature_dim=x_arr.shape[1],
        output_dim=y_arr.shape[1],
        hidden_dim=max(1, int(hidden_dim)),
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        state_dict=model.state_dict(),
    )


def _predict_score_matrix(payload: dict, x: np.ndarray) -> np.ndarray:
    multi_output = payload.get("multi_output_regressor")
    if multi_output is not None:
        scores = np.asarray(multi_output.predict(x), dtype=np.float64)
        fallback = np.asarray(payload.get("global_label_mean", []), dtype=np.float64).reshape(-1)
        if fallback.size == scores.shape[1]:
            bad = ~np.isfinite(scores)
            if bad.any():
                scores = scores.copy()
                scores[bad] = np.take(fallback, np.where(bad)[1])
        return scores
    regressors = payload.get("regressors", [])
    if not regressors:
        raise ValueError("AutoForecast artifact has no regressors")
    cols = []
    with _sklearn_threadpool_limit():
        for reg in regressors:
            pred = np.asarray(reg.predict(x), dtype=np.float64).reshape(-1)
            cols.append(pred)
    scores = np.stack(cols, axis=1)
    fallback = np.asarray(payload.get("global_label_mean", []), dtype=np.float64).reshape(-1)
    if fallback.size == scores.shape[1]:
        bad = ~np.isfinite(scores)
        if bad.any():
            scores = scores.copy()
            scores[bad] = np.take(fallback, np.where(bad)[1])
    return scores


def _predict_autoforecast_score_tensor(payload: dict, samples) -> tuple[np.ndarray, dict]:
    t_feature0 = time.perf_counter()
    features = feature_tensor_from_samples(samples)
    n, c, f = features.shape
    flat_x = features.reshape(n * c, f)
    feature_ms = (time.perf_counter() - t_feature0) * 1000.0

    t_predict0 = time.perf_counter()
    score_flat = _predict_score_matrix(payload, flat_x)
    predict_ms = (time.perf_counter() - t_predict0) * 1000.0

    scores = score_flat.reshape(n, c, -1)
    return scores, {
        "feature_ms": feature_ms,
        "predict_ms": predict_ms,
        "score_shape": list(scores.shape),
        "feature_dim": int(f),
    }


def predict_autoforecast_sample_rank_tensors(payload: dict, samples) -> tuple[np.ndarray, dict]:
    scores, timing = _predict_autoforecast_score_tensor(payload, samples)
    t_rank0 = time.perf_counter()
    orders = np.argsort(scores, axis=2).transpose(0, 2, 1).astype(np.int64, copy=False)
    rank_ms = (time.perf_counter() - t_rank0) * 1000.0
    return orders, {
        **timing,
        "rank_ms": rank_ms,
    }


def predict_autoforecast_rank_tensor(payload: dict, samples) -> tuple[np.ndarray, dict]:
    scores, timing = _predict_autoforecast_score_tensor(payload, samples)
    _n, c, _model_count = scores.shape
    t_rank0 = time.perf_counter()
    mean_scores = np.nanmean(scores, axis=0)
    order = np.argsort(mean_scores, axis=1).astype(np.int64)
    rank_ms = (time.perf_counter() - t_rank0) * 1000.0

    return order.T.reshape(-1, 1, c), {
        **timing,
        "rank_ms": rank_ms,
    }


def _upsert_csv(path: Path, row: dict, fieldnames: list[str], key_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(str(path) + ".lock"):
        rows = []
        if path.exists():
            try:
                with path.open("r", newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
            except Exception:
                rows = []
        key = tuple(str(row.get(field, "")) for field in key_fields)
        rows = [
            old for old in rows
            if tuple(str(old.get(field, "")) for field in key_fields) != key
        ]
        rows.append(row)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for old in rows:
                writer.writerow({field: old.get(field, "") for field in fieldnames})


AUTOFORECAST_INSERT_FIELDS = [
    "row_key",
    "status",
    "method",
    "method_version",
    "artifact_schema_version",
    "selector_mode",
    "stage",
    "zoo_total_num",
    "latest_model_abbr",
    "latest_model_full_name",
    "model_abbr_order",
    "model_count",
    "repr_set_name",
    "repr_forward_stem",
    "advanced_baseline_train_scope",
    "training_repr_forward_stem",
    "model_repr_name",
    "repr_v",
    "base_metrics",
    "target_metric",
    "quality_target_metric",
    "label_cleaning_scope",
    "learner",
    "train_samples",
    "feature_dim",
    "label_refresh_seconds",
    "feature_refresh_seconds",
    "resource_refresh_seconds",
    "feature_cache_status",
    "feature_cache_path",
    "insert_measurement_kind",
    "incremental_measurement_status",
    "structure_refresh_seconds",
    "selector_retrain_seconds",
    "incremental_status",
    "incremental_source_stage",
    "incremental_source_artifact",
    "reused_regressor_count",
    "trained_regressor_count",
    "label_changed_regressor_count",
    "xpcr_target_changed_models",
    "incoming_profile_seconds",
    "insert_total_seconds",
    "old_model_forwards",
    "timing_valid",
    "feature_source",
    "label_source",
    "feature_source_signature",
    "label_source_signature",
    "resource_source_signature",
    "resource_score_fingerprint",
    "resource_source",
    "quality_normalization",
    "resource_normalization",
    "xpcr_p_weight",
    "xpcr_r_weight",
    "xpcr_c_weight",
    "model_artifact_path",
    "weight_artifact_path",
    "step2_coverage_status",
    "step2_metric_complete",
    "step2_runtime_complete",
    "step2_missing_by_metric",
    "step2_runtime_status",
    "step2_runtime_path",
    "created_at_utc",
]


def _latest_model_full_name(latest_model_abbr: str) -> str:
    abbr_to_full = {abbr: full for full, abbr in Model_abbrev_map.items()}
    return abbr_to_full.get(str(latest_model_abbr), "")


def _format_float(value: float) -> str:
    try:
        value = float(value)
    except Exception:
        return ""
    return f"{value:.9f}" if np.isfinite(value) else ""


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _autoforecast_skip_saved_ready(
    artifact_path: Path,
    manifest_path: Path,
    *,
    expected_order: list[str],
    target_metric: str,
    learner: str,
    expected_feature_source_signature: dict | None = None,
    expected_label_source_signature: dict | None = None,
    expected_resource_source_signature: dict | None = None,
    expected_train_scope: str = "center",
    selector_mode: str = AUTOFORECAST_SELECTOR_MODE,
) -> tuple[bool, str]:
    selector_mode = _normalize_selector_mode(selector_mode)
    expected_schema_version = _artifact_schema_for_mode(selector_mode)
    expected_method_version = _method_version_for_mode(selector_mode)
    if not artifact_path.is_file() or not manifest_path.is_file():
        return False, "missing_artifact_or_manifest"
    try:
        with artifact_path.open("rb") as f:
            payload = pickle.load(f)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"artifact_metadata_read_error:{type(exc).__name__}"

    def metadata_reason(data: dict, label: str, *, require_fingerprints: bool = True) -> str:
        if not isinstance(data, dict):
            return f"{label}_metadata_not_mapping"
        try:
            schema_version = int(data.get("artifact_schema_version", data.get("schema_version", 0)) or 0)
        except (TypeError, ValueError):
            return f"{label}_schema_invalid"
        if schema_version != expected_schema_version:
            return f"{label}_schema_outdated"
        if str(data.get("method_version", "") or "") != expected_method_version:
            return f"{label}_method_version_changed"
        try:
            data_mode = _normalize_selector_mode(
                data.get("selector_mode", AUTOFORECAST_SELECTOR_MODE)
            )
        except ValueError:
            return f"{label}_selector_mode_invalid"
        if data_mode != selector_mode:
            return f"{label}_selector_mode_changed"
        raw_order = data.get("model_abbr_order", data.get("model_names", []))
        if isinstance(raw_order, str):
            order = raw_order.split()
        elif isinstance(raw_order, (list, tuple)):
            order = [str(x) for x in raw_order]
        else:
            return f"{label}_model_order_invalid"
        if order != expected_order:
            return f"{label}_model_order_changed"
        try:
            data_metric = normalize_autoforecast_metric(data.get("target_metric", ""))
            data_learner = normalize_autoforecast_learner(data.get("learner", ""))
        except (TypeError, ValueError):
            return f"{label}_invalid_method_metadata"
        if data_metric != target_metric or data_learner != learner:
            return f"{label}_method_changed"
        if str(data.get("advanced_baseline_train_scope", "center") or "center") != str(expected_train_scope):
            return f"{label}_train_scope_changed"
        if str(data.get("label_cleaning_scope", "") or "") != "per_model":
            return f"{label}_label_cleaning_contract_outdated"
        if require_fingerprints:
            try:
                fingerprints = dict(data.get("target_fingerprints", {}))
            except (TypeError, ValueError):
                return f"{label}_invalid_target_fingerprints"
            if any(len(str(fingerprints.get(name, ""))) != 64 for name in expected_order):
                return f"{label}_missing_target_fingerprints"
        try:
            reused = int(data.get("reused_regressor_count", -1))
            trained = int(data.get("trained_regressor_count", -1))
        except (TypeError, ValueError):
            return f"{label}_invalid_incremental_counts"
        if reused < 0 or trained < 0 or reused + trained != len(expected_order):
            return f"{label}_invalid_incremental_counts"
        if not str(data.get("incremental_status", "") or "").strip():
            return f"{label}_missing_incremental_status"
        if str(data.get("insert_measurement_kind", "") or "") not in {
            "incremental_cache_hit",
            "cold_feature_build",
        }:
            return f"{label}_measurement_kind_invalid"
        if not str(data.get("incremental_measurement_status", "") or "").strip():
            return f"{label}_incremental_measurement_status_missing"
        return ""

    for data, label in ((payload, "artifact"), (manifest, "manifest")):
        reason = metadata_reason(data, label)
        if reason:
            return False, reason

    def source_signature_reason(data: dict, label: str) -> str:
        for field, expected in (
            ("feature_source_signature", expected_feature_source_signature),
            ("label_source_signature", expected_label_source_signature),
            ("resource_source_signature", expected_resource_source_signature),
        ):
            if expected is None:
                continue
            raw = data.get(field, {})
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except (TypeError, ValueError):
                    return f"{label}_{field}_invalid"
            if raw != expected:
                return f"{label}_{field}_changed"
        return ""

    for data, label in ((payload, "artifact"), (manifest, "manifest")):
        reason = source_signature_reason(data, label)
        if reason:
            return False, reason

    timing_path = autoforecast_csv_root(selector_mode) / "step3_insert_timing.csv"
    try:
        with timing_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        return False, "missing_timing_csv"
    except Exception as exc:
        return False, f"timing_read_error:{type(exc).__name__}"
    row_key = artifact_path.stem
    timing_row = next((row for row in reversed(rows) if str(row.get("row_key", "")) == row_key), None)
    if timing_row is None:
        return False, "missing_timing_row"
    reason = metadata_reason(timing_row, "timing", require_fingerprints=False)
    if reason:
        return False, reason
    reason = source_signature_reason(timing_row, "timing")
    if reason:
        return False, reason
    if str(timing_row.get("status", "") or "").strip().lower() != "built":
        return False, "timing_status_not_built"
    if str(timing_row.get("timing_valid", "") or "").strip().lower() != "true":
        return False, "timing_not_valid"
    try:
        insert_total_seconds = float(timing_row.get("insert_total_seconds", "nan"))
    except (TypeError, ValueError):
        insert_total_seconds = float("nan")
    if not np.isfinite(insert_total_seconds) or insert_total_seconds < 0:
        return False, "timing_total_invalid"
    return True, "current_schema_artifact_and_timing"


def build_autoforecast_step3(
    args,
    current_zoo_abbr_order_list: list[str],
    *,
    repr_set_name: str,
    weight_path: str,
    model_repr_path: str,
    load_metric_dicts: Callable,
    read_latest_model_runtime: Callable,
    load_model_forward_runtime: Callable | None = None,
) -> None:
    selector_mode = autoforecast_selector_mode(args)
    method_version = _method_version_for_mode(selector_mode)
    artifact_schema_version = _artifact_schema_for_mode(selector_mode)
    target_metric = normalize_autoforecast_metric(getattr(args, "base_metrics", "MASE"))
    method = autoforecast_method_name(target_metric, selector_mode)
    learner = normalize_autoforecast_learner(
        getattr(args, "autoforecast_learner", AUTOFORECAST_DEFAULT_LEARNER)
    )
    skip_saved = bool(getattr(args, "skip_saved", False))
    artifact_path = Path(model_repr_path)
    weight_artifact_path = Path(weight_path)
    manifest_path = artifact_path.with_name(f"{artifact_path.stem}_model_manifest.json")
    expected_order = [str(x) for x in current_zoo_abbr_order_list]
    train_scope = get_advanced_baseline_train_scope(args)
    training_repr_forward_stem = _repr_forward_stem_for_train_scope(args, train_scope)
    repr_forward_csv_path = _repr_forward_csv_path_for_train_scope(args, train_scope)
    expected_feature_source = str(
        _training_feature_source_path(args, repr_set_name, train_scope)
    )
    resource_info: dict = {}
    resource_source_signature: dict = {}
    resource_refresh_seconds = 0.0
    if selector_mode == AUTOXPCR_SELECTOR_MODE:
        if load_model_forward_runtime is None:
            raise ValueError("AutoXPCR Step3 requires a strict Step2 forward-runtime loader")
        resource_t0 = time.perf_counter()
        resource_info = dict(
            load_model_forward_runtime(
                args=args,
                model_names=expected_order,
            )
        )
        resource_refresh_seconds = time.perf_counter() - resource_t0
        resource_path = str(resource_info.get("runtime_path", "") or "")
        if not resource_path:
            raise ValueError("AutoXPCR Step3 runtime loader returned no runtime_path")
        resource_source_signature = _file_signature(resource_path)
    begin_message = (
        f"begin: method={method}, learner={learner}, stage={getattr(args, 'current_zoo_num', '')}, "
        f"models={len(expected_order)}, repr_set={repr_set_name}, "
        f"train_scope={train_scope}, artifact={artifact_path}"
    )
    if selector_mode == AUTOXPCR_SELECTOR_MODE:
        begin_message += f", resource_refresh_seconds={resource_refresh_seconds:.3f}"
    _log_step3(begin_message, selector_mode)
    if skip_saved:
        skip_ready, skip_reason = _autoforecast_skip_saved_ready(
            artifact_path,
            manifest_path,
            expected_order=expected_order,
            target_metric=target_metric,
            learner=learner,
            expected_feature_source_signature=(
                _file_signature(expected_feature_source)
                if expected_feature_source
                else None
            ),
            expected_label_source_signature=_file_signature(repr_forward_csv_path),
            expected_resource_source_signature=(
                resource_source_signature
                if selector_mode == AUTOXPCR_SELECTOR_MODE
                else None
            ),
            expected_train_scope=train_scope,
            selector_mode=selector_mode,
        )
        if skip_ready:
            _log_step3(f"skip-save ready: {artifact_path}", selector_mode)
            return
        if artifact_path.exists() or manifest_path.exists():
            _log_step3(
                f"skip-save rejected: reason={skip_reason}; rebuild={artifact_path}",
                selector_mode,
            )

    label_t0 = time.perf_counter()
    _log_step3(f"load Step2 per-sample labels: {repr_forward_csv_path}", selector_mode)
    metric_perf_dict_by_name, _metric_load_errors, step2_coverage = load_metric_dicts(
        args=args,
        current_zoo_abbr_order_list=current_zoo_abbr_order_list,
        repr_forward_csv_path=repr_forward_csv_path,
    )
    label_source_signature = _file_signature(repr_forward_csv_path)
    _log_step3(
        "Step2 labels loaded: "
        f"metric_complete={bool(step2_coverage.get('metric_complete', False))}, "
        f"runtime_complete={bool(step2_coverage.get('runtime_complete', False))}, "
        f"coverage_status={step2_coverage.get('status', '')}, "
        f"elapsed={time.perf_counter() - label_t0:.1f}s",
        selector_mode,
    )
    if not bool(step2_coverage.get("metric_complete", False)):
        label_refresh_seconds = time.perf_counter() - label_t0
        _write_autoforecast_insert_row(
            args=args,
            repr_set_name=repr_set_name,
            model_repr_path=str(artifact_path),
            weight_path=str(weight_artifact_path),
            model_names=expected_order,
            target_metric=target_metric,
            learner=learner,
            label_refresh_seconds=label_refresh_seconds,
            feature_refresh_seconds=0.0,
            selector_retrain_seconds=0.0,
            incoming_profile_seconds=float("nan"),
            step2_coverage=step2_coverage,
            runtime_status="skipped_step2_incomplete",
            runtime_path="",
            status="skipped_step2_incomplete",
            train_samples=0,
            feature_dim=0,
            feature_source="",
            label_source=repr_forward_csv_path,
            feature_cache_status="not_started",
            feature_cache_path="",
            resource_refresh_seconds=resource_refresh_seconds,
            selector_mode=selector_mode,
            resource_source_signature=resource_source_signature,
            advanced_baseline_train_scope=train_scope,
            training_repr_forward_stem=training_repr_forward_stem,
        )
        _log_step3("skip: Step2 metric coverage is incomplete.", selector_mode)
        return

    quality_matrix = _ordered_metric_matrix(metric_perf_dict_by_name[target_metric], expected_order)
    finite_rows = np.isfinite(quality_matrix).all(axis=1)
    _log_step3(
        f"target label matrix ready: metric={target_metric}, shape={tuple(quality_matrix.shape)}, "
        f"fully_finite_rows={int(finite_rows.sum())}/{int(finite_rows.size)}",
        selector_mode,
    )
    if not finite_rows.any():
        raise ValueError(f"{method} {target_metric} label matrix has no fully finite rows")
    quality_target_fingerprints = _target_fingerprints(
        quality_matrix,
        finite_rows,
        expected_order,
    )
    p_score_matrix = np.full_like(quality_matrix, np.nan, dtype=np.float64)
    r_score = np.zeros((len(expected_order),), dtype=np.float64)
    resource_score_fingerprint = ""
    target_matrix = quality_matrix
    if selector_mode == AUTOXPCR_SELECTOR_MODE:
        runtime_by_model_seconds = {
            str(name): float(value)
            for name, value in dict(
                resource_info.get("runtime_by_model_seconds", {})
            ).items()
        }
        xpcr_finite, p_finite, r_score = _build_autoxpcr_target_matrix(
            quality_matrix[finite_rows],
            runtime_by_model_seconds=runtime_by_model_seconds,
            model_order=expected_order,
        )
        target_matrix = np.full_like(quality_matrix, np.nan, dtype=np.float64)
        target_matrix[finite_rows] = xpcr_finite
        p_score_matrix[finite_rows] = p_finite
        resource_score_fingerprint = _resource_score_fingerprint(r_score, expected_order)
    target_fingerprints = _target_fingerprints(
        target_matrix,
        finite_rows,
        expected_order,
    )
    xpcr_target_fingerprints = (
        dict(target_fingerprints)
        if selector_mode == AUTOXPCR_SELECTOR_MODE
        else {}
    )
    label_refresh_seconds = time.perf_counter() - label_t0

    feature_t0 = time.perf_counter()
    _log_step3(f"load repr windows: repr_set={repr_set_name}", selector_mode)
    windows, feature_source = _load_repr_windows(args, repr_set_name, train_scope)
    feature_source_signature = _file_signature(feature_source)
    _log_step3(
        f"repr windows loaded: shape={tuple(windows.shape)}, source={feature_source}",
        selector_mode,
    )
    if windows.shape[0] != quality_matrix.shape[0]:
        raise ValueError(
            "AutoForecast feature/label row mismatch: "
            f"windows={windows.shape[0]}, labels={quality_matrix.shape[0]}, "
            f"feature_source={feature_source}, label_source={repr_forward_csv_path}"
        )
    x, feature_cache_status, feature_cache_path = _load_or_build_feature_matrix(
        windows,
        feature_source,
        cache_root=artifact_path.parent.parent / "caches" / "features",
        log_every=int(getattr(args, "autoforecast_feature_log_every", 500) or 500),
        selector_mode=selector_mode,
    )
    feature_refresh_seconds = time.perf_counter() - feature_t0
    x_train = x[finite_rows]
    y_train = target_matrix[finite_rows]
    _log_step3(
        f"training rows ready: x_train={tuple(x_train.shape)}, y_train={tuple(y_train.shape)}, "
        f"label_refresh_seconds={label_refresh_seconds:.3f}, "
        f"feature_refresh_seconds={feature_refresh_seconds:.3f}, "
        f"feature_cache_status={feature_cache_status}, "
        f"resource_refresh_seconds={resource_refresh_seconds:.3f}",
        selector_mode,
    )

    seed = int(getattr(args, "search_seed", 2025) or 2025)
    multi_output_regressor = None
    reusable_regressors: dict[int, tuple[object, str]] = {}
    incremental_status = f"not_supported_for_{learner.lower()}"
    incremental_source_artifact = ""
    incremental_source_stage = 0
    label_changed_models: list[str] = []
    xpcr_target_changed_models: list[str] = []
    if learner == "GBDT":
        # This is intentionally independent of skip_saved.  A non-skip-save
        # Step3 insert is a fresh timing measurement, but the fair incremental
        # operation is still "reuse all unchanged old per-model regressors and
        # train only the newly inserted model".
        incremental_plan_t0 = time.perf_counter()
        current_stage = int(getattr(args, "current_zoo_num", len(expected_order)) or len(expected_order))
        previous_payload, previous_path, previous_reason = _find_previous_gbdt_artifact(
            artifact_path,
            current_stage=current_stage,
            expected_order=expected_order,
            target_metric=target_metric,
            repr_set_name=repr_set_name,
            selector_mode=selector_mode,
            train_scope=train_scope,
        )
        if previous_payload is not None and previous_path is not None:
            old_order = [
                str(x)
                for x in previous_payload.get(
                    "model_abbr_order",
                    previous_payload.get("model_names", []),
                )
            ]
            feature_checks = {
                "feature_names": list(previous_payload.get("feature_names", [])) == autoforecast_feature_names(),
                "feature_source": str(previous_payload.get("feature_source", "")) == str(feature_source),
                "label_source": str(previous_payload.get("label_source", "")) == str(repr_forward_csv_path),
                "train_samples": int(previous_payload.get("train_samples", -1)) == int(x_train.shape[0]),
                "feature_dim": int(previous_payload.get("feature_dim", -1)) == int(x_train.shape[1]),
            }
            failed_feature_checks = [name for name, ok in feature_checks.items() if not ok]
            if failed_feature_checks:
                incremental_status = "rebuild_feature_contract_changed:" + ",".join(failed_feature_checks)
            else:
                previous_fingerprints = {
                    str(name): str(value)
                    for name, value in dict(previous_payload.get("target_fingerprints", {})).items()
                }
                previous_quality_fingerprints = {
                    str(name): str(value)
                    for name, value in dict(
                        previous_payload.get("quality_target_fingerprints", {})
                        if selector_mode == AUTOXPCR_SELECTOR_MODE
                        else previous_payload.get(
                            "quality_target_fingerprints",
                            previous_payload.get("target_fingerprints", {}),
                        )
                    ).items()
                }
                previous_fingerprints_complete = all(
                    name in previous_fingerprints for name in old_order
                )
                previous_quality_complete = all(
                    name in previous_quality_fingerprints for name in old_order
                )
                if not previous_fingerprints_complete or not previous_quality_complete:
                    if selector_mode == AUTOXPCR_SELECTOR_MODE:
                        incremental_status = "rebuild_previous_xpcr_fingerprints_missing"
                        previous_fingerprints = {}
                        previous_quality_fingerprints = {}
                    else:
                        try:
                            _log_step3(
                                "incremental reference lacks target fingerprints; "
                                f"rebuild labels for stage={previous_payload.get('stage', len(old_order))}, "
                                f"models={len(old_order)}",
                                selector_mode,
                            )
                            previous_metric_dicts, _previous_errors, previous_coverage = load_metric_dicts(
                                args=args,
                                current_zoo_abbr_order_list=old_order,
                                repr_forward_csv_path=repr_forward_csv_path,
                            )
                            if not bool(previous_coverage.get("metric_complete", False)):
                                raise ValueError(
                                    f"previous stage metric coverage incomplete: {previous_coverage.get('status', '')}"
                                )
                            previous_matrix = _ordered_metric_matrix(
                                previous_metric_dicts[target_metric],
                                old_order,
                            )
                            previous_finite_rows = np.isfinite(previous_matrix).all(axis=1)
                            _validate_legacy_target_reconstruction(
                                previous_payload,
                                previous_matrix,
                                previous_finite_rows,
                            )
                            previous_fingerprints = _target_fingerprints(
                                previous_matrix,
                                previous_finite_rows,
                                old_order,
                            )
                            previous_quality_fingerprints = dict(previous_fingerprints)
                        except Exception as exc:
                            incremental_status = (
                                "rebuild_previous_target_fingerprint_failed:"
                                f"{type(exc).__name__}"
                            )
                            previous_fingerprints = {}
                            previous_quality_fingerprints = {}

                old_regressors = list(previous_payload.get("regressors", []))
                old_statuses = list(previous_payload.get("regressor_status", []))
                for model_idx, model_name in enumerate(old_order):
                    if (
                        previous_quality_fingerprints.get(model_name)
                        != quality_target_fingerprints.get(model_name)
                    ):
                        label_changed_models.append(model_name)
                    if previous_fingerprints.get(model_name) == target_fingerprints.get(model_name):
                        old_status = (
                            old_statuses[model_idx]
                            if model_idx < len(old_statuses)
                            else type(old_regressors[model_idx]).__name__
                        )
                        reusable_regressors[model_idx] = (
                            old_regressors[model_idx],
                            old_status,
                        )
                    else:
                        if selector_mode == AUTOXPCR_SELECTOR_MODE:
                            xpcr_target_changed_models.append(model_name)
                incremental_source_artifact = str(previous_path)
                incremental_source_stage = int(previous_payload.get("stage", len(old_order)) or len(old_order))
                if reusable_regressors:
                    incremental_status = "reused_exact_target_regressors"
                elif not incremental_status.startswith("rebuild_"):
                    incremental_status = "compatible_artifact_but_targets_changed"
        else:
            incremental_status = f"no_reuse:{previous_reason}"

        _log_step3(
            "incremental GBDT plan: "
            f"status={incremental_status}, source_stage={incremental_source_stage or 'none'}, "
            f"reuse={len(reusable_regressors)}, train={len(expected_order) - len(reusable_regressors)}, "
            f"label_changed={label_changed_models}, "
            f"xpcr_target_changed={xpcr_target_changed_models}",
            selector_mode,
        )
        label_refresh_seconds += time.perf_counter() - incremental_plan_t0
        train_t0 = time.perf_counter()
        regressors, regressor_status = _fit_per_model_regressors(
            x_train,
            y_train,
            seed,
            progress_label=f"[{method.split('-')[0]} Step3][train]",
            reusable_regressors=reusable_regressors,
        )
    else:
        train_t0 = time.perf_counter()
        multi_output_regressor = _fit_torch_multioutput_regressor(
            x_train,
            y_train,
            learner=learner,
            seed=seed,
            hidden_dim=int(getattr(args, "autoforecast_hidden_dim", 64) or 64),
            epochs=int(getattr(args, "autoforecast_train_epochs", 120) or 120),
            learning_rate=float(getattr(args, "autoforecast_learning_rate", 1e-3) or 1e-3),
            batch_size=int(getattr(args, "autoforecast_batch_size", 256) or 256),
            progress_label=f"[{method.split('-')[0]} Step3][train]",
        )
        regressors = []
        regressor_status = [f"{learner}_multi_output"]
    selector_retrain_seconds = time.perf_counter() - train_t0
    reused_regressor_count = len(reusable_regressors)
    trained_regressor_count = len(expected_order) - reused_regressor_count
    _log_step3(
        f"selector training done: selector_retrain_seconds={selector_retrain_seconds:.3f}",
        selector_mode,
    )

    global_label_mean = np.nanmean(y_train, axis=0)
    inverse = 1.0 / np.maximum(global_label_mean, 1e-8)
    if not np.isfinite(inverse).all() or float(np.sum(inverse)) <= 0:
        inverse = np.ones_like(global_label_mean, dtype=np.float64)
    model_weights = {
        name: float(value)
        for name, value in zip(expected_order, inverse / np.max(inverse))
    }
    latest_model_abbr = expected_order[-1] if expected_order else ""
    _log_step3(
        f"read latest model profile runtime: latest_model={latest_model_abbr}",
        selector_mode,
    )
    incoming_profile_seconds, runtime_path, runtime_status = read_latest_model_runtime(
        args, latest_model_abbr
    )
    _log_step3(
        f"latest model profile runtime: seconds={_format_float(incoming_profile_seconds) or 'nan'}, "
        f"status={runtime_status}, path={runtime_path}",
        selector_mode,
    )
    timing_valid = bool(
        np.isfinite(float(incoming_profile_seconds))
        and float(incoming_profile_seconds) >= 0
    )
    incremental_measurement_status = _classify_incremental_measurement(
        selector_mode=selector_mode,
        incremental_source_stage=incremental_source_stage,
        timing_valid=timing_valid,
        feature_cache_status=feature_cache_status,
        reused_regressor_count=reused_regressor_count,
        trained_regressor_count=trained_regressor_count,
        model_count=len(expected_order),
        label_changed_models=label_changed_models,
        xpcr_target_changed_models=xpcr_target_changed_models,
    )
    _log_step3(
        f"incremental measurement status: {incremental_measurement_status}",
        selector_mode,
    )
    insert_total_seconds = (
        float(incoming_profile_seconds)
        + label_refresh_seconds
        + feature_refresh_seconds
        + resource_refresh_seconds
        + selector_retrain_seconds
        if timing_valid
        else float("nan")
    )

    payload = {
        "__repr_format__": AUTOFORECAST_REPR_FORMAT,
        "artifact_schema_version": artifact_schema_version,
        "selector_mode": selector_mode,
        "method": method,
        "method_version": method_version,
        "target_metric": target_metric,
        "quality_target_metric": target_metric,
        "learner": learner,
        "regressors": regressors,
        "multi_output_regressor": multi_output_regressor,
        "regressor_status": regressor_status,
        "feature_names": autoforecast_feature_names(),
        "feature_group_contract": [
            "observed_window_statistics",
            "shape_scale",
            "lightweight_frequency_autocorrelation",
        ],
        "disabled_feature_groups": [
            "dataset_domain_freq_term_metadata",
        ],
        "model_abbr_order": expected_order,
        "model_names": expected_order,
        "model_metric_weights": model_weights,
        "global_label_mean": global_label_mean.astype(float).tolist(),
        "target_fingerprints": target_fingerprints,
        "quality_target_fingerprints": quality_target_fingerprints,
        "xpcr_target_fingerprints": xpcr_target_fingerprints,
        "resource_score_fingerprint": resource_score_fingerprint,
        "incremental_status": incremental_status,
        "incremental_source_stage": incremental_source_stage,
        "incremental_source_artifact": incremental_source_artifact,
        "reused_regressor_count": reused_regressor_count,
        "trained_regressor_count": trained_regressor_count,
        "label_changed_models": label_changed_models,
        "xpcr_target_changed_models": xpcr_target_changed_models,
        "label_cleaning_scope": "per_model",
        "label_refresh_seconds": float(label_refresh_seconds),
        "feature_refresh_seconds": float(feature_refresh_seconds),
        "resource_refresh_seconds": float(resource_refresh_seconds),
        "feature_cache_status": feature_cache_status,
        "feature_cache_path": feature_cache_path,
        "insert_measurement_kind": (
            "incremental_cache_hit" if feature_cache_status == "hit" else "cold_feature_build"
        ),
        "incremental_measurement_status": incremental_measurement_status,
        "timing_valid": bool(timing_valid),
        "train_samples": int(x_train.shape[0]),
        "feature_dim": int(x_train.shape[1]),
        "advanced_baseline_train_scope": train_scope,
        "training_repr_forward_stem": training_repr_forward_stem,
        "repr_set_name": repr_set_name,
        "feature_source": feature_source,
        "label_source": repr_forward_csv_path,
        "feature_source_signature": feature_source_signature,
        "label_source_signature": label_source_signature,
        "resource_source_signature": resource_source_signature,
        "resource_source": (
            "forward_runtime_seconds"
            if selector_mode == AUTOXPCR_SELECTOR_MODE
            else "disabled"
        ),
        "resource_path": str(resource_info.get("runtime_path", "") or ""),
        "runtime_by_model_seconds": dict(
            resource_info.get("runtime_by_model_seconds", {})
        ),
        "resource_score_by_model": {
            name: float(value) for name, value in zip(expected_order, r_score)
        }
        if selector_mode == AUTOXPCR_SELECTOR_MODE
        else {},
        "xpcr_weights": {
            "P": AUTOXPCR_P_WEIGHT,
            "R": AUTOXPCR_R_WEIGHT,
            "C": AUTOXPCR_C_WEIGHT,
        }
        if selector_mode == AUTOXPCR_SELECTOR_MODE
        else {},
        "quality_normalization": (
            "per_sample_model_minmax"
            if selector_mode == AUTOXPCR_SELECTOR_MODE
            else "none"
        ),
        "resource_normalization": (
            "current_stage_model_minmax"
            if selector_mode == AUTOXPCR_SELECTOR_MODE
            else "none"
        ),
        "complexity_source": "disabled" if selector_mode == AUTOXPCR_SELECTOR_MODE else "not_applicable",
        "metric": target_metric,
        "repr_v": int(getattr(args, "repr_v", 7)),
        "lstm_sequence_contract": (
            "Current TSRouter Step2 samples do not expose consecutive-window task histories; "
            "the LSTM learner receives each Step2 window feature vector as a length-1 sequence."
        ),
        "stage": int(getattr(args, "current_zoo_num", len(expected_order)) or len(expected_order)),
        "zoo_total_num": int(getattr(args, "zoo_total_num", len(expected_order)) or len(expected_order)),
        **build_model_family_metadata(expected_order),
    }

    weight_payload = {
        "total_models": len(expected_order),
        "model_weights": model_weights,
        "weight_source": (
            "autoxpcr_global_inverse_composite"
            if selector_mode == AUTOXPCR_SELECTOR_MODE
            else f"autoforecast_global_inverse_{target_metric}"
        ),
        "selector_mode": selector_mode,
        "advanced_baseline_train_scope": train_scope,
        "model_abbr_order": expected_order,
        **build_model_family_metadata(expected_order),
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    weight_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    _log_step3(
        f"save artifacts: model={artifact_path}, weight={weight_artifact_path}",
        selector_mode,
    )
    atomic_pickle_dump(payload, str(artifact_path))
    atomic_pickle_dump(weight_payload, str(weight_artifact_path))

    manifest = {
        "schema_version": artifact_schema_version,
        "artifact_schema_version": artifact_schema_version,
        "selector_mode": selector_mode,
        "method": method,
        "method_version": method_version,
        "target_metric": target_metric,
        "quality_target_metric": target_metric,
        "learner": learner,
        "model_abbr_order": expected_order,
        "model_names": expected_order,
        "stage": payload["stage"],
        "zoo_total_num": payload["zoo_total_num"],
        "feature_source": feature_source,
        "label_source": repr_forward_csv_path,
        "advanced_baseline_train_scope": train_scope,
        "training_repr_forward_stem": training_repr_forward_stem,
        "feature_source_signature": feature_source_signature,
        "label_source_signature": label_source_signature,
        "resource_source_signature": resource_source_signature,
        "artifact_path": str(artifact_path),
        "weight_path": str(weight_artifact_path),
        "train_samples": int(x_train.shape[0]),
        "feature_dim": int(x_train.shape[1]),
        "target_fingerprints": target_fingerprints,
        "quality_target_fingerprints": quality_target_fingerprints,
        "xpcr_target_fingerprints": xpcr_target_fingerprints,
        "resource_score_fingerprint": resource_score_fingerprint,
        "incremental_status": incremental_status,
        "incremental_source_stage": incremental_source_stage,
        "incremental_source_artifact": incremental_source_artifact,
        "reused_regressor_count": reused_regressor_count,
        "trained_regressor_count": trained_regressor_count,
        "label_changed_models": label_changed_models,
        "xpcr_target_changed_models": xpcr_target_changed_models,
        "label_cleaning_scope": "per_model",
        "label_refresh_seconds": float(label_refresh_seconds),
        "feature_refresh_seconds": float(feature_refresh_seconds),
        "resource_refresh_seconds": float(resource_refresh_seconds),
        "feature_cache_status": feature_cache_status,
        "feature_cache_path": feature_cache_path,
        "insert_measurement_kind": (
            "incremental_cache_hit" if feature_cache_status == "hit" else "cold_feature_build"
        ),
        "incremental_measurement_status": incremental_measurement_status,
        "timing_valid": bool(timing_valid),
        "resource_source": (
            "forward_runtime_seconds"
            if selector_mode == AUTOXPCR_SELECTOR_MODE
            else "disabled"
        ),
        "resource_path": str(resource_info.get("runtime_path", "") or ""),
        "runtime_by_model_seconds": dict(
            resource_info.get("runtime_by_model_seconds", {})
        ),
        "resource_score_by_model": {
            name: float(value) for name, value in zip(expected_order, r_score)
        }
        if selector_mode == AUTOXPCR_SELECTOR_MODE
        else {},
        "xpcr_weights": {
            "P": AUTOXPCR_P_WEIGHT,
            "R": AUTOXPCR_R_WEIGHT,
            "C": AUTOXPCR_C_WEIGHT,
        }
        if selector_mode == AUTOXPCR_SELECTOR_MODE
        else {},
        "quality_normalization": (
            "per_sample_model_minmax"
            if selector_mode == AUTOXPCR_SELECTOR_MODE
            else "none"
        ),
        "resource_normalization": (
            "current_stage_model_minmax"
            if selector_mode == AUTOXPCR_SELECTOR_MODE
            else "none"
        ),
        "complexity_source": "disabled" if selector_mode == AUTOXPCR_SELECTOR_MODE else "not_applicable",
        "insert_timing_csv": str(
            autoforecast_csv_root(selector_mode) / "step3_insert_timing.csv"
        ),
        "assumptions": [
            "Only Step2-config-matched repr windows are used for training.",
            "Dataset/domain/frequency metadata features are disabled for v7.",
            "LSTM uses length-1 sequences until Step2 exposes true consecutive-window histories.",
            "AutoXPCR uses current-stage MinMax and may retrain old regressors after a zoo insert."
            if selector_mode == AUTOXPCR_SELECTOR_MODE
            else "AutoForecast raw per-model labels remain stable across ordinary inserts.",
        ],
        **build_model_family_metadata(expected_order),
    }
    _write_manifest(manifest_path, manifest)

    _write_autoforecast_insert_row(
        args=args,
        repr_set_name=repr_set_name,
        model_repr_path=str(artifact_path),
        weight_path=str(weight_artifact_path),
        model_names=expected_order,
        target_metric=target_metric,
        learner=learner,
        label_refresh_seconds=label_refresh_seconds,
        feature_refresh_seconds=feature_refresh_seconds,
        resource_refresh_seconds=resource_refresh_seconds,
        selector_retrain_seconds=selector_retrain_seconds,
        incoming_profile_seconds=incoming_profile_seconds,
        step2_coverage=step2_coverage,
        runtime_status=runtime_status,
        runtime_path=runtime_path,
        status="built",
        train_samples=int(x_train.shape[0]),
        feature_dim=int(x_train.shape[1]),
        feature_source=feature_source,
        label_source=repr_forward_csv_path,
        feature_cache_status=feature_cache_status,
        feature_cache_path=feature_cache_path,
        selector_mode=selector_mode,
        resource_source_signature=resource_source_signature,
        resource_score_fingerprint=resource_score_fingerprint,
        incremental_measurement_status=incremental_measurement_status,
        incremental_status=incremental_status,
        incremental_source_stage=incremental_source_stage,
        incremental_source_artifact=incremental_source_artifact,
        reused_regressor_count=reused_regressor_count,
        trained_regressor_count=trained_regressor_count,
        label_changed_regressor_count=len(label_changed_models),
        xpcr_target_changed_models=xpcr_target_changed_models,
        advanced_baseline_train_scope=train_scope,
        training_repr_forward_stem=training_repr_forward_stem,
    )
    _log_step3(
        f"saved {method} artifact -> {artifact_path}; "
        f"learner={learner}, "
        f"train_samples={x_train.shape[0]}, feature_dim={x_train.shape[1]}, "
        f"reused_regressors={reused_regressor_count}, trained_regressors={trained_regressor_count}, "
        f"insert_total_seconds={_format_float(insert_total_seconds) or 'nan'}",
        selector_mode,
    )


def _write_autoforecast_insert_row(
    *,
    args,
    repr_set_name: str,
    model_repr_path: str,
    weight_path: str,
    model_names: list[str],
    target_metric: str,
    learner: str,
    label_refresh_seconds: float,
    feature_refresh_seconds: float,
    resource_refresh_seconds: float = 0.0,
    selector_retrain_seconds: float,
    incoming_profile_seconds: float,
    step2_coverage: dict,
    runtime_status: str,
    runtime_path: str,
    status: str,
    train_samples: int,
    feature_dim: int,
    feature_source: str,
    label_source: str,
    feature_cache_status: str,
    feature_cache_path: str,
    selector_mode: str = AUTOFORECAST_SELECTOR_MODE,
    resource_source_signature: dict | None = None,
    resource_score_fingerprint: str = "",
    incremental_measurement_status: str = "not_measured",
    incremental_status: str = "",
    incremental_source_stage: int = 0,
    incremental_source_artifact: str = "",
    reused_regressor_count: int = 0,
    trained_regressor_count: int = 0,
    label_changed_regressor_count: int = 0,
    xpcr_target_changed_models: list[str] | None = None,
    advanced_baseline_train_scope: str | None = None,
    training_repr_forward_stem: str | None = None,
) -> None:
    selector_mode = _normalize_selector_mode(selector_mode)
    method_version = _method_version_for_mode(selector_mode)
    artifact_schema_version = _artifact_schema_for_mode(selector_mode)
    resource_source_signature = dict(resource_source_signature or {})
    xpcr_target_changed_models = list(xpcr_target_changed_models or [])
    train_scope = str(
        advanced_baseline_train_scope
        or get_advanced_baseline_train_scope(args)
    )
    training_stem = str(
        training_repr_forward_stem
        or _repr_forward_stem_for_train_scope(args, train_scope)
    )
    latest_model_abbr = str(model_names[-1]) if model_names else ""
    structure_refresh_seconds = 0.0
    timing_valid = bool(
        np.isfinite(float(incoming_profile_seconds))
        and float(incoming_profile_seconds) >= 0
    )
    insert_total_seconds = (
        float(incoming_profile_seconds)
        + float(label_refresh_seconds)
        + float(feature_refresh_seconds)
        + float(resource_refresh_seconds)
        + structure_refresh_seconds
        + float(selector_retrain_seconds)
        if timing_valid
        else float("nan")
    )
    row = {
        "row_key": Path(model_repr_path).stem,
        "status": status,
        "method": autoforecast_method_name(target_metric, selector_mode),
        "method_version": method_version,
        "artifact_schema_version": artifact_schema_version,
        "selector_mode": selector_mode,
        "stage": int(getattr(args, "current_zoo_num", len(model_names)) or len(model_names)),
        "zoo_total_num": int(getattr(args, "zoo_total_num", len(model_names)) or len(model_names)),
        "latest_model_abbr": latest_model_abbr,
        "latest_model_full_name": _latest_model_full_name(latest_model_abbr),
        "model_abbr_order": " ".join(str(x) for x in model_names),
        "model_count": len(model_names),
        "repr_set_name": repr_set_name,
        "repr_forward_stem": build_repr_forward_stem(args),
        "advanced_baseline_train_scope": train_scope,
        "training_repr_forward_stem": training_stem,
        "model_repr_name": Path(model_repr_path).stem,
        "repr_v": getattr(args, "repr_v", ""),
        "base_metrics": getattr(args, "base_metrics", ""),
        "target_metric": target_metric,
        "quality_target_metric": target_metric,
        "label_cleaning_scope": "per_model",
        "learner": normalize_autoforecast_learner(learner),
        "train_samples": int(train_samples),
        "feature_dim": int(feature_dim),
        "label_refresh_seconds": _format_float(label_refresh_seconds),
        "feature_refresh_seconds": _format_float(feature_refresh_seconds),
        "resource_refresh_seconds": _format_float(resource_refresh_seconds),
        "feature_cache_status": feature_cache_status,
        "feature_cache_path": feature_cache_path,
        "insert_measurement_kind": (
            "incremental_cache_hit" if feature_cache_status == "hit" else "cold_feature_build"
        ),
        "incremental_measurement_status": incremental_measurement_status,
        "structure_refresh_seconds": _format_float(structure_refresh_seconds),
        "selector_retrain_seconds": _format_float(selector_retrain_seconds),
        "incremental_status": incremental_status,
        "incremental_source_stage": int(incremental_source_stage or 0),
        "incremental_source_artifact": incremental_source_artifact,
        "reused_regressor_count": int(reused_regressor_count),
        "trained_regressor_count": int(trained_regressor_count),
        "label_changed_regressor_count": int(label_changed_regressor_count),
        "xpcr_target_changed_models": " ".join(xpcr_target_changed_models),
        "incoming_profile_seconds": _format_float(incoming_profile_seconds),
        "insert_total_seconds": _format_float(insert_total_seconds),
        "old_model_forwards": 0,
        "timing_valid": str(bool(timing_valid)).lower(),
        "feature_source": feature_source,
        "label_source": label_source,
        "feature_source_signature": json.dumps(
            _file_signature(feature_source),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "label_source_signature": json.dumps(
            _file_signature(label_source),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "resource_source_signature": json.dumps(
            resource_source_signature,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "resource_score_fingerprint": resource_score_fingerprint,
        "resource_source": (
            "forward_runtime_seconds"
            if selector_mode == AUTOXPCR_SELECTOR_MODE
            else "disabled"
        ),
        "quality_normalization": (
            "per_sample_model_minmax"
            if selector_mode == AUTOXPCR_SELECTOR_MODE
            else "none"
        ),
        "resource_normalization": (
            "current_stage_model_minmax"
            if selector_mode == AUTOXPCR_SELECTOR_MODE
            else "none"
        ),
        "xpcr_p_weight": AUTOXPCR_P_WEIGHT if selector_mode == AUTOXPCR_SELECTOR_MODE else "",
        "xpcr_r_weight": AUTOXPCR_R_WEIGHT if selector_mode == AUTOXPCR_SELECTOR_MODE else "",
        "xpcr_c_weight": AUTOXPCR_C_WEIGHT if selector_mode == AUTOXPCR_SELECTOR_MODE else "",
        "model_artifact_path": model_repr_path,
        "weight_artifact_path": weight_path,
        "step2_coverage_status": str(step2_coverage.get("status", "")),
        "step2_metric_complete": str(bool(step2_coverage.get("metric_complete", False))).lower(),
        "step2_runtime_complete": str(bool(step2_coverage.get("runtime_complete", False))).lower(),
        "step2_missing_by_metric": str(step2_coverage.get("step2_missing_by_metric", step2_coverage.get("missing_by_metric", ""))),
        "step2_runtime_status": runtime_status,
        "step2_runtime_path": runtime_path,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    csv_path = autoforecast_csv_root(selector_mode) / "step3_insert_timing.csv"
    _upsert_csv(csv_path, row, AUTOFORECAST_INSERT_FIELDS, key_fields=["row_key"])
    label = "AutoXPCR" if selector_mode == AUTOXPCR_SELECTOR_MODE else "AutoForecast"
    print(f"[{label} Step3][timing] saved -> {csv_path}", flush=True)

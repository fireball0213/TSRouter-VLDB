from __future__ import annotations

import csv
import gc
import importlib.util
import json
import os
import pickle
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config.model_zoo_config import Model_abbrev_map, build_model_family_metadata
from encoder.baseline.TS2Vec_encoder import TS2VecEncoder
from utils.io_lock import atomic_pickle_dump, file_lock
from utils.path_utils import (
    TSROUTER_SAMPLED_REPR_POOL_DIR,
    build_repr_eval_pool_forward_stem,
    build_repr_eval_pool_name,
    build_repr_forward_stem,
    build_selector_artifact_repr_set_name,
    get_advanced_baseline_train_scope,
    get_tsrouter_repr_forward_dir,
)
from utils.project_paths import BASELINE_ARTIFACT_ROOT, BASELINE_CSV_ROOT


SIMPLETS_REPR_FORMAT = "simplets_v1"
SIMPLETS_METHOD_VERSION = "simplets_v1"
SIMPLETS_SELECTOR_NAME = "SimpleTS_Select"
SIMPLETS_N_CLUSTERS = 3
SIMPLETS_ENCODER_CACHE_MAX_ENTRIES = 4
SIMPLETS_CLASSIFIER_POLICY_VERSION = "fixed_lightgbm_v1"
_SIMPLETS_ENCODER_CACHE: OrderedDict[tuple, TS2VecEncoder] = OrderedDict()


class ConstantClusterClassifier:
    def __init__(self, label: int, classes: list[int] | np.ndarray):
        self.label = int(label)
        self.classes_ = np.asarray(sorted({int(x) for x in classes} | {self.label}), dtype=np.int64)
        self.simplets_backend_ = "constant"

    def predict(self, x):
        return np.full((len(x),), self.label, dtype=np.int64)

    def predict_proba(self, x):
        probs = np.zeros((len(x), len(self.classes_)), dtype=np.float64)
        label_pos = int(np.where(self.classes_ == self.label)[0][0])
        probs[:, label_pos] = 1.0
        return probs


def is_simplets_v6(args) -> bool:
    raw = str(getattr(args, "repr_v", "") or "")
    return raw[:1] == "6"


def normalize_simplets_metric(value) -> str:
    raw = str(value or "MASE").strip()
    upper = raw.upper()
    if upper in {"M", "MASE"}:
        return "MASE"
    if upper in {"S", "SMAPE"}:
        return "sMAPE"
    if upper in {"C", "CRPS"}:
        return "CRPS"
    raise ValueError(f"SimpleTS supports base_metrics M/S/C, got {raw!r}")


def simplets_method_name(metric: str) -> str:
    metric = normalize_simplets_metric(metric)
    suffix = {"MASE": "M", "sMAPE": "S", "CRPS": "C"}[metric]
    return f"SimpleTS-{suffix}"


def simplets_csv_root() -> Path:
    return BASELINE_CSV_ROOT / "selectors" / SIMPLETS_SELECTOR_NAME


def simplets_artifact_root() -> Path:
    return BASELINE_ARTIFACT_ROOT / "selectors" / SIMPLETS_SELECTOR_NAME


def _log_step3(message: str) -> None:
    print(f"[SimpleTS Step3] {message}", flush=True)


def _format_float(value: float) -> str:
    try:
        value = float(value)
    except Exception:
        return ""
    return f"{value:.9f}" if np.isfinite(value) else ""


def _latest_model_full_name(latest_model_abbr: str) -> str:
    abbr_to_full = {abbr: full for full, abbr in Model_abbrev_map.items()}
    return abbr_to_full.get(str(latest_model_abbr), "")


def _coerce_windows_array(windows) -> np.ndarray:
    x = np.asarray(windows, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :, None]
    elif x.ndim == 2:
        x = x[:, :, None]
    elif x.ndim != 3:
        raise ValueError(f"SimpleTS windows expect 1D/2D/3D input, got shape={x.shape}")
    if x.shape[1] <= 0 or x.shape[2] <= 0:
        raise ValueError(f"SimpleTS windows must have positive T and C, got shape={x.shape}")
    return x


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
        source_hint = ""
        if str(getattr(args, "encoder_type", "") or "").strip().lower() == "train":
            source_hint = (
                " SimpleTS v6 trains its TS2Vec internally; pass the main-method Step1/2 "
                "source encoder (for example StatsRandom/Fourier), not Train/TS2Vec, "
                "unless a TrainTS2Vec Step1 source artifact intentionally exists."
            )
        raise FileNotFoundError(
            f"SimpleTS v6 missing Step1 source windows for scope={train_scope}: {path}.{source_hint}"
        )
    with path.open("rb") as f:
        payload = pickle.load(f)
    return _coerce_windows_array(payload), str(path)


def _ordered_metric_matrix(metric_dict: OrderedDict, model_order: list[str]) -> np.ndarray:
    missing = [name for name in model_order if name not in metric_dict]
    if missing:
        raise ValueError(f"SimpleTS label matrix missing models: {missing}")
    return np.stack(
        [np.asarray(metric_dict[name], dtype=np.float64) for name in model_order],
        axis=1,
    )


def _impute_metric_matrix(metric_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(metric_matrix, dtype=np.float64).copy()
    finite = np.isfinite(matrix)
    if not finite.any():
        raise ValueError("SimpleTS metric matrix has no finite values")
    global_fill = float(np.nanmedian(matrix[finite]))
    for col in range(matrix.shape[1]):
        col_vals = matrix[:, col]
        good = np.isfinite(col_vals)
        fill = float(np.nanmean(col_vals[good])) if good.any() else global_fill
        col_vals[~good] = fill
        matrix[:, col] = col_vals
    return matrix


def build_simplets_cluster_tables(
    metric_matrix: np.ndarray,
    model_names: list[str],
    *,
    n_clusters: int = SIMPLETS_N_CLUSTERS,
    seed: int = 2025,
) -> dict:
    clean_matrix = _impute_metric_matrix(metric_matrix)
    model_perf_vectors = clean_matrix.T
    model_perf_scaled = StandardScaler().fit_transform(model_perf_vectors)
    cluster_count = max(1, min(int(n_clusters), len(model_names)))
    if cluster_count == 1:
        model_cluster_ids = np.zeros(len(model_names), dtype=np.int64)
        cluster_centers = np.mean(model_perf_scaled, axis=0, keepdims=True)
    else:
        kmeans = KMeans(n_clusters=cluster_count, random_state=int(seed), n_init=10)
        model_cluster_ids = kmeans.fit_predict(model_perf_scaled).astype(np.int64)
        cluster_centers = kmeans.cluster_centers_.astype(np.float64)

    best_model_idx = np.argmin(clean_matrix, axis=1).astype(np.int64)
    sample_labels = model_cluster_ids[best_model_idx].astype(np.int64)
    model_mean_metric = np.mean(clean_matrix, axis=0)
    global_prior_order = np.argsort(model_mean_metric, kind="mergesort").astype(np.int64)
    cluster_winner_by_cluster: dict[int, int] = {}
    cluster_members: dict[int, list[int]] = {}
    for cluster_id in range(cluster_count):
        members = np.where(model_cluster_ids == cluster_id)[0].astype(np.int64)
        cluster_members[int(cluster_id)] = [int(x) for x in members.tolist()]
        if members.size == 0:
            continue
        member_means = model_mean_metric[members]
        winner = int(members[int(np.argmin(member_means))])
        cluster_winner_by_cluster[int(cluster_id)] = winner

    return {
        "clean_metric_matrix": clean_matrix,
        "model_perf_vectors": model_perf_vectors,
        "model_perf_scaled": model_perf_scaled,
        "model_cluster_ids": model_cluster_ids,
        "cluster_centers": cluster_centers,
        "best_model_idx_by_sample": best_model_idx,
        "sample_cluster_labels": sample_labels,
        "cluster_winner_by_cluster": cluster_winner_by_cluster,
        "cluster_members": cluster_members,
        "model_mean_metric": model_mean_metric,
        "global_prior_order": global_prior_order,
    }


def _windows_to_series_matrix(windows: np.ndarray, input_dim: int | None = None) -> np.ndarray:
    x = _coerce_windows_array(windows)
    if input_dim is not None:
        input_dim = max(1, int(input_dim))
        if x.shape[1] < input_dim:
            raise ValueError(
                f"SimpleTS source windows shorter than repr_input_dim: "
                f"window_len={x.shape[1]}, repr_input_dim={input_dim}"
            )
        # Step1 windows contain [context, forecast tail]. SimpleTS must learn
        # from the historical context only so Step3 and online Step4 agree.
        x = x[:, :input_dim, :]
    return np.nan_to_num(x.mean(axis=2), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _ts2vec_config(args, input_dim: int) -> dict:
    return {
        "input_dim": int(input_dim),
        "embedding_dim": int(getattr(args, "repr_output_dim", 256) or 256),
        "ts2vec_hidden_dim": max(64, int(getattr(args, "repr_output_dim", 256) or 256)),
        "ts2vec_depth": 4,
        "ts2vec_kernel_size": 3,
        "ts2vec_dropout": 0.1,
        "ts_l2norm": True,
    }


def _build_ts2vec_encoder(config: dict, device) -> TS2VecEncoder:
    return TS2VecEncoder(SimpleNamespace(**config), device=device)


def _instance_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    batch = z1.size(0)
    if batch <= 1:
        return z1.sum() * 0.0
    z = torch.cat([z1, z2], dim=0).transpose(0, 1)
    z = F.normalize(z, p=2, dim=2)
    sim = torch.matmul(z, z.transpose(1, 2))
    eye = torch.eye(2 * batch, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye[None, :, :], -1e9)
    log_prob = F.log_softmax(sim, dim=-1)
    idx = torch.arange(batch, device=z.device)
    return -0.5 * (
        log_prob[:, idx, idx + batch].mean()
        + log_prob[:, idx + batch, idx].mean()
    )


def _temporal_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    time_len = z1.size(1)
    if time_len <= 1:
        return z1.sum() * 0.0
    z = torch.cat([z1, z2], dim=1)
    z = F.normalize(z, p=2, dim=2)
    sim = torch.matmul(z, z.transpose(1, 2))
    eye = torch.eye(2 * time_len, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye[None, :, :], -1e9)
    log_prob = F.log_softmax(sim, dim=-1)
    idx = torch.arange(time_len, device=z.device)
    return -0.5 * (
        log_prob[:, idx, idx + time_len].mean()
        + log_prob[:, idx + time_len, idx].mean()
    )


def _hierarchical_ts2vec_loss(z1: torch.Tensor, z2: torch.Tensor, alpha: float) -> torch.Tensor:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    loss = z1.sum() * 0.0
    levels = 0
    while z1.size(1) > 1:
        loss = loss + alpha * _instance_contrastive_loss(z1, z2)
        loss = loss + (1.0 - alpha) * _temporal_contrastive_loss(z1, z2)
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
        levels += 1
    if z1.size(1) == 1:
        loss = loss + alpha * _instance_contrastive_loss(z1, z2)
        levels += 1
    return loss / max(levels, 1)


def _augment_series(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype) * 0.01
    keep = torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype) > 0.1
    return torch.where(keep, x + noise, torch.zeros_like(x))


def _is_cuda_batch_error(exc: RuntimeError) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "cuda out of memory",
            "out of memory",
            "cublas_status_alloc_failed",
            "cuda error: invalid configuration argument",
        )
    )


def _cleanup_cuda_after_oom() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


def _train_ts2vec_batch(
    encoder: TS2VecEncoder,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    generator: torch.Generator,
    alpha: float,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    z1 = encoder.encode_sequence(_augment_series(x, generator))
    z2 = encoder.encode_sequence(_augment_series(x, generator))
    loss = _hierarchical_ts2vec_loss(z1, z2, alpha=alpha)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def _simplets_ts2vec_checkpoint_path(args, repr_set_name: str) -> Path:
    root = simplets_artifact_root() / "encoders"
    alpha = float(np.clip(float(getattr(args, "train_top3_weight", 0.5) or 0.5), 0.0, 1.0))
    selector_repr_set_name = build_selector_artifact_repr_set_name(args)
    source_encoder = str(getattr(args, "repr_encoder", "source") or "source")
    train_scope = get_advanced_baseline_train_scope(args)
    scope_tag = "_abpool" if train_scope == "full_pool" else ""
    name = (
        f"{selector_repr_set_name}_ts2vec"
        f"_src{source_encoder}"
        f"{scope_tag}"
        f"_ep{int(getattr(args, 'train_encoder_epochs', 30) or 30)}"
        f"_lr{float(getattr(args, 'train_encoder_lr', 1e-3) or 1e-3):g}"
        f"_bs{int(getattr(args, 'train_encoder_batch_size', 256) or 256)}"
        f"_alpha{alpha:g}"
        f"_tau{float(getattr(args, 'train_encoder_temperature', 0.1) or 0.1):g}"
        ".pt"
    )
    return root / name


def _train_or_load_ts2vec(
    args,
    repr_set_name: str,
    windows: np.ndarray,
) -> tuple[TS2VecEncoder, dict, Path, float, str, dict]:
    requested_input_dim = max(1, int(getattr(args, "repr_input_dim", windows.shape[1]) or windows.shape[1]))
    train_scope = get_advanced_baseline_train_scope(args)
    series = _windows_to_series_matrix(windows, input_dim=requested_input_dim)
    config = _ts2vec_config(args, input_dim=requested_input_dim)
    bound_checkpoint = str(getattr(args, "simplets_ts2vec_checkpoint", "") or "").strip()
    path = (
        Path(bound_checkpoint).expanduser().resolve()
        if bound_checkpoint
        else _simplets_ts2vec_checkpoint_path(args, repr_set_name)
    )
    if bound_checkpoint and not path.is_file():
        raise FileNotFoundError(f"SimpleTS TS2Vec checkpoint does not exist: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = _build_ts2vec_encoder(config, device=device)
    requested_batch_size = max(1, int(getattr(args, "train_encoder_batch_size", 256) or 256))
    # The TS2Vec encoder is an immutable SimpleTS prerequisite shared by every
    # growing-zoo stage.  Reusing it is part of the method contract, not the
    # whole-stage skip_saved shortcut: a fresh Step3 timing run must not train
    # the same encoder again.
    if path.exists():
        try:
            try:
                ckpt = torch.load(path, map_location=device, weights_only=False)
            except TypeError:  # PyTorch versions before the weights_only option.
                ckpt = torch.load(path, map_location=device)
            if not isinstance(ckpt, dict):
                raise ValueError("checkpoint_not_mapping")
            checkpoint_config = dict(ckpt.get("config", {}))
            if checkpoint_config != config:
                raise ValueError("config_changed")
            if str(ckpt.get("method", "")) != "SimpleTS":
                raise ValueError("method_changed")
            if str(ckpt.get("method_version", "")) != SIMPLETS_METHOD_VERSION:
                raise ValueError("method_version_changed")
            if str(ckpt.get("repr_set_name", "")) != str(repr_set_name):
                raise ValueError("source_repr_changed")
            if str(ckpt.get("advanced_baseline_train_scope", "center") or "center") != str(train_scope):
                raise ValueError("train_scope_changed")
            state = ckpt.get("state_dict")
            if not isinstance(state, dict):
                raise ValueError("state_dict_missing")
            encoder.load_state_dict(state, strict=True)
            encoder.eval()
            train_batch_stats = dict(ckpt.get("train_batch_size", {}))
            train_batch_stats.setdefault("requested_batch_size", requested_batch_size)
            train_batch_stats.setdefault("runtime_batch_size", requested_batch_size)
            train_batch_stats.setdefault("min_batch_size", train_batch_stats["runtime_batch_size"])
            train_batch_stats.setdefault("fallback_count", 0)
            _log_step3(
                f"reuse compatible TS2Vec checkpoint (independent of skip_saved): {path}"
            )
            return encoder, config, path, 0.0, "reused", train_batch_stats
        except Exception as exc:
            if bound_checkpoint:
                raise ValueError(
                    f"SimpleTS TS2Vec checkpoint is incompatible: {path}; "
                    f"reason={type(exc).__name__}:{exc}"
                ) from exc
            _log_step3(
                "reject incompatible TS2Vec checkpoint and retrain once: "
                f"path={path}, reason={type(exc).__name__}:{exc}"
            )

    train_t0 = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = requested_batch_size
    min_batch_size = requested_batch_size
    batch_size_fallback_count = 0
    epochs = max(1, int(getattr(args, "train_encoder_epochs", 30) or 30))
    lr = float(getattr(args, "train_encoder_lr", 1e-3) or 1e-3)
    weight_decay = float(getattr(args, "train_encoder_weight_decay", 1e-4) or 1e-4)
    alpha = float(np.clip(float(getattr(args, "train_top3_weight", 0.5) or 0.5), 0.0, 1.0))
    tensor = torch.as_tensor(series, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(getattr(args, "repr_encoder_seed", 2025) or 2025))
    _log_step3(
        f"train TS2Vec: samples={series.shape[0]}, input_dim={series.shape[1]}, "
        f"embedding_dim={config['embedding_dim']}, epochs={epochs}, batch_size={batch_size}, alpha={alpha:g}"
    )
    encoder.train()
    for epoch in range(1, epochs + 1):
        epoch_t0 = time.perf_counter()
        order = torch.randperm(tensor.shape[0], generator=generator, device=device)
        loss_sum = 0.0
        count = 0
        start = 0
        while start < tensor.shape[0]:
            idx = order[start:start + batch_size]
            try:
                batch_loss = _train_ts2vec_batch(
                    encoder,
                    optimizer,
                    tensor[idx],
                    generator,
                    alpha,
                )
            except RuntimeError as exc:
                if not _is_cuda_batch_error(exc):
                    raise
                optimizer.zero_grad(set_to_none=True)
                if batch_size <= 1:
                    _cleanup_cuda_after_oom()
                    _log_step3(
                        f"TS2Vec CUDA OOM at batch_size=1, epoch={epoch}, "
                        f"sample_offset={start}; cannot reduce further."
                    )
                    raise
                next_batch_size = max(1, batch_size // 2)
                batch_size_fallback_count += 1
                min_batch_size = min(min_batch_size, next_batch_size)
                # Drop the failed batch frame before empty_cache(); its traceback
                # otherwise keeps the large activation tensors alive during retry.
                exc.__traceback__ = None
                _cleanup_cuda_after_oom()
                _log_step3(
                    f"TS2Vec CUDA OOM at batch_size={batch_size}, epoch={epoch}, "
                    f"sample_offset={start}; reducing to {next_batch_size} and retrying current batch "
                    f"(fallback_count={batch_size_fallback_count})"
                )
                batch_size = next_batch_size
                continue
            batch_count = int(idx.numel())
            loss_sum += batch_loss * batch_count
            count += batch_count
            start += batch_count
        _log_step3(
            f"TS2Vec epoch {epoch}/{epochs}: loss={loss_sum / max(count, 1):.6f}, "
            f"batch_size={batch_size}, epoch_seconds={time.perf_counter() - epoch_t0:.1f}s, "
            f"elapsed={time.perf_counter() - train_t0:.1f}s"
        )
    encoder.eval()
    train_batch_stats = {
        "requested_batch_size": int(requested_batch_size),
        "runtime_batch_size": int(batch_size),
        "min_batch_size": int(min_batch_size),
        "fallback_count": int(batch_size_fallback_count),
    }
    state = {key: value.detach().cpu() for key, value in encoder.state_dict().items()}
    torch.save(
        {
            "state_dict": state,
            "config": config,
            "repr_set_name": repr_set_name,
            "advanced_baseline_train_scope": train_scope,
            "method": "SimpleTS",
            "method_version": SIMPLETS_METHOD_VERSION,
            "train_batch_size": train_batch_stats,
        },
        path,
    )
    return encoder, config, path, time.perf_counter() - train_t0, "trained", train_batch_stats


def _embed_windows(encoder: TS2VecEncoder, windows: np.ndarray, batch_size: int) -> np.ndarray:
    series = _windows_to_series_matrix(windows, input_dim=int(encoder.input_len))
    device = encoder.device
    outs = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, series.shape[0], max(1, int(batch_size))):
            batch = torch.as_tensor(series[start:start + batch_size], dtype=torch.float32, device=device)
            outs.append(encoder.forward(batch).detach().cpu().numpy())
    return np.vstack(outs).astype(np.float32)


def _normalize_classifier_backend(value: object) -> str:
    raw = str(value or "lightgbm").strip().lower().replace("-", "_")
    aliases = {
        "": "lightgbm",
        "auto": "auto",
        "lgbm": "lightgbm",
        "light_gbm": "lightgbm",
        "lightgbm": "lightgbm",
        "rf": "random_forest",
        "randomforest": "random_forest",
        "random_forest": "random_forest",
        "hgb": "hist_gbdt",
        "hist": "hist_gbdt",
        "histgbdt": "hist_gbdt",
        "hist_gbdt": "hist_gbdt",
        "histgradientboosting": "hist_gbdt",
    }
    if raw not in aliases:
        raise ValueError(
            "Unknown SimpleTS classifier backend "
            f"{value!r}; expected auto/lightgbm/random_forest/hist_gbdt"
        )
    return aliases[raw]


def _requested_cluster_classifier_backend(args=None) -> str:
    return _normalize_classifier_backend(
        getattr(args, "simplets_classifier_backend", None)
        or os.environ.get("SIMPLETS_CLASSIFIER_BACKEND", "lightgbm")
    )


def _resolve_cluster_classifier_backend(requested: str) -> str:
    requested = _normalize_classifier_backend(requested)
    if requested == "auto":
        return "lightgbm" if importlib.util.find_spec("lightgbm") is not None else "random_forest"
    if requested == "lightgbm" and importlib.util.find_spec("lightgbm") is None:
        raise ModuleNotFoundError(
            "SIMPLETS_CLASSIFIER_BACKEND=lightgbm was requested, but lightgbm is not installed. "
            "Install lightgbm or use SIMPLETS_CLASSIFIER_BACKEND=random_forest."
        )
    return requested


def _cluster_classifier_runtime_diagnostics() -> dict:
    diagnostics = {
        "classifier_policy_version": SIMPLETS_CLASSIFIER_POLICY_VERSION,
        "numpy_version": str(np.__version__),
    }
    try:
        import sklearn

        diagnostics["sklearn_version"] = str(sklearn.__version__)
    except Exception as exc:  # pragma: no cover - defensive diagnostic only
        diagnostics["sklearn_version"] = f"unavailable:{type(exc).__name__}"
    try:
        from threadpoolctl import threadpool_info

        pools = []
        for info in threadpool_info():
            pools.append(
                {
                    "user_api": str(info.get("user_api", "")),
                    "internal_api": str(info.get("internal_api", "")),
                    "num_threads": int(info.get("num_threads", 0) or 0),
                    "prefix": str(info.get("prefix", "")),
                }
            )
        diagnostics["threadpools"] = pools
    except Exception as exc:  # pragma: no cover - optional dependency detail
        diagnostics["threadpools"] = [f"unavailable:{type(exc).__name__}"]
    return diagnostics


def _format_threadpool_summary(diagnostics: dict) -> str:
    pools = diagnostics.get("threadpools", [])
    if not isinstance(pools, list) or not pools:
        return "none"
    parts = []
    for item in pools[:4]:
        if isinstance(item, dict):
            parts.append(
                f"{item.get('user_api', '?')}/{item.get('internal_api', '?')}"
                f":threads={item.get('num_threads', '?')}"
            )
        else:
            parts.append(str(item))
    if len(pools) > 4:
        parts.append(f"+{len(pools) - 4} more")
    return "; ".join(parts)


def _classifier_backend_and_name(classifier) -> tuple[str, str]:
    final_estimator = classifier.steps[-1][1] if hasattr(classifier, "steps") and classifier.steps else classifier
    backend = (
        getattr(classifier, "simplets_backend_", None)
        or getattr(final_estimator, "simplets_backend_", None)
        or "unknown"
    )
    return str(backend), type(final_estimator).__name__


def _make_cluster_classifier(seed: int, backend: str):
    backend = _resolve_cluster_classifier_backend(backend)
    if backend == "lightgbm":
        from lightgbm import LGBMClassifier

        params = {
            "n_estimators": 64,
            "learning_rate": 0.08,
            "num_leaves": 31,
            "random_state": int(seed),
            "n_jobs": 1,
            "verbose": -1,
        }
        estimator = LGBMClassifier(**params)
    elif backend == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        params = {
            "n_estimators": 128,
            "max_depth": 16,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "random_state": int(seed),
            "n_jobs": 1,
        }
        estimator = RandomForestClassifier(**params)
    elif backend == "hist_gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier

        verbose = int(os.environ.get("SIMPLETS_CLASSIFIER_VERBOSE", "1") or "1")
        params = {
            "max_iter": 64,
            "learning_rate": 0.08,
            "random_state": int(seed),
            "verbose": verbose,
        }
        estimator = HistGradientBoostingClassifier(**params)
    else:  # pragma: no cover - guarded by _resolve_cluster_classifier_backend
        raise ValueError(f"Unsupported SimpleTS classifier backend: {backend}")

    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        estimator,
    )
    setattr(pipeline, "simplets_backend_", backend)
    setattr(pipeline, "simplets_params_", params)
    setattr(estimator, "simplets_backend_", backend)
    return pipeline, backend, params


def _fit_cluster_classifier(
    x: np.ndarray,
    labels: np.ndarray,
    seed: int,
    *,
    backend: str = "lightgbm",
    log_fn: Callable[[str], None] | None = None,
):
    def emit(message: str) -> None:
        if log_fn is not None:
            log_fn(message)

    requested_backend = _normalize_classifier_backend(backend)
    resolved_backend = _resolve_cluster_classifier_backend(requested_backend)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    filter_t0 = time.perf_counter()
    finite = np.isfinite(x).all(axis=1)
    x_train = np.asarray(x, dtype=np.float32)[finite]
    y_train = labels[finite]
    filter_seconds = time.perf_counter() - filter_t0
    unique = np.unique(y_train)
    label_values, label_counts_arr = np.unique(y_train, return_counts=True)
    label_counts = {str(int(k)): int(v) for k, v in zip(label_values, label_counts_arr)}
    diagnostics = _cluster_classifier_runtime_diagnostics()
    diagnostics.update(
        {
            "requested_backend": requested_backend,
            "resolved_backend": resolved_backend,
            "input_shape": [int(v) for v in np.asarray(x).shape],
            "train_shape": [int(v) for v in x_train.shape],
            "finite_rows": int(np.count_nonzero(finite)),
            "dropped_nonfinite_rows": int(labels.shape[0] - np.count_nonzero(finite)),
            "label_counts": label_counts,
            "finite_filter_seconds": float(filter_seconds),
        }
    )
    if unique.size <= 1:
        label = int(unique[0]) if unique.size else 0
        diagnostics.update(
            {
                "classifier_backend": "constant",
                "classifier_name": "ConstantClusterClassifier",
                "classifier_params": {},
                "fit_seconds": 0.0,
            }
        )
        emit(
            "classifier fit skipped: only one class after filtering; "
            f"label={label}, train_shape={tuple(x_train.shape)}, label_counts={label_counts}"
        )
        return ConstantClusterClassifier(label, classes=[label]), "constant", diagnostics

    clf, resolved_backend, params = _make_cluster_classifier(seed, resolved_backend)
    diagnostics["classifier_backend"] = resolved_backend
    diagnostics["classifier_params"] = params
    _, estimator_name = _classifier_backend_and_name(clf)
    diagnostics["classifier_name"] = estimator_name
    emit(
        "classifier fit start: "
        f"backend={resolved_backend}, estimator={estimator_name}, "
        f"train_shape={tuple(x_train.shape)}, label_counts={label_counts}, "
        f"sklearn={diagnostics.get('sklearn_version', 'unknown')}, "
        f"threadpools={_format_threadpool_summary(diagnostics)}"
    )
    fit_t0 = time.perf_counter()
    clf.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - fit_t0
    final_estimator = clf.steps[-1][1] if hasattr(clf, "steps") and clf.steps else clf
    diagnostics["fit_seconds"] = float(fit_seconds)
    diagnostics["classifier_name"] = type(final_estimator).__name__
    emit(
        "classifier fit done: "
        f"backend={resolved_backend}, estimator={type(final_estimator).__name__}, "
        f"fit_seconds={fit_seconds:.3f}, total_with_filter={fit_seconds + filter_seconds:.3f}"
    )
    return clf, type(final_estimator).__name__, diagnostics


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _write_tables(
    *,
    artifact_path: Path,
    model_names: list[str],
    metric_matrix: np.ndarray,
    cluster_info: dict,
    target_metric: str,
) -> dict[str, str]:
    paths = {
        "model_perf_matrix_csv": artifact_path.with_name(f"{artifact_path.stem}_simplets_model_perf_matrix.csv"),
        "model_clusters_csv": artifact_path.with_name(f"{artifact_path.stem}_simplets_model_clusters.csv"),
        "sample_labels_csv": artifact_path.with_name(f"{artifact_path.stem}_simplets_sample_labels.csv"),
        "cluster_winners_csv": artifact_path.with_name(f"{artifact_path.stem}_simplets_cluster_winners.csv"),
    }
    perf_rows = []
    for model_idx, model_name in enumerate(model_names):
        row = {"model": model_name, "metric": target_metric}
        for sample_idx, value in enumerate(metric_matrix[:, model_idx]):
            row[f"sample_{sample_idx}"] = float(value)
        perf_rows.append(row)
    _write_csv(paths["model_perf_matrix_csv"], perf_rows)

    cluster_rows = []
    for model_idx, model_name in enumerate(model_names):
        cluster_rows.append(
            {
                "model": model_name,
                "model_idx": model_idx,
                "cluster_id": int(cluster_info["model_cluster_ids"][model_idx]),
                f"mean_{target_metric}": float(cluster_info["model_mean_metric"][model_idx]),
            }
        )
    _write_csv(paths["model_clusters_csv"], cluster_rows)

    sample_rows = []
    for sample_idx, label in enumerate(cluster_info["sample_cluster_labels"]):
        best_idx = int(cluster_info["best_model_idx_by_sample"][sample_idx])
        sample_rows.append(
            {
                "sample_idx": sample_idx,
                "best_model_idx": best_idx,
                "best_model": model_names[best_idx],
                "cluster_label": int(label),
            }
        )
    _write_csv(paths["sample_labels_csv"], sample_rows)

    winner_rows = []
    for cluster_id, winner_idx in sorted(cluster_info["cluster_winner_by_cluster"].items()):
        members = cluster_info["cluster_members"].get(int(cluster_id), [])
        winner_rows.append(
            {
                "cluster_id": int(cluster_id),
                "winner_model_idx": int(winner_idx),
                "winner_model": model_names[int(winner_idx)],
                "member_model_indices": " ".join(str(int(x)) for x in members),
                "member_models": " ".join(model_names[int(x)] for x in members),
                f"winner_mean_{target_metric}": float(cluster_info["model_mean_metric"][int(winner_idx)]),
            }
        )
    _write_csv(paths["cluster_winners_csv"], winner_rows)
    return {key: str(path) for key, path in paths.items()}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


SIMPLETS_INSERT_FIELDS = [
    "row_key",
    "status",
    "method",
    "method_version",
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
    "n_clusters",
    "train_samples",
    "embedding_dim",
    "classifier_policy_version",
    "classifier_backend",
    "classifier_name",
    "ts2vec_train_seconds",
    "selector_retrain_excludes_ts2vec",
    "label_refresh_seconds",
    "structure_refresh_seconds",
    "selector_retrain_seconds",
    "incoming_profile_seconds",
    "insert_total_seconds",
    "old_model_forwards",
    "timing_valid",
    "ts2vec_checkpoint_status",
    "ts2vec_checkpoint_path",
    "feature_source",
    "label_source",
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


def build_simplets_step3(
    args,
    current_zoo_abbr_order_list: list[str],
    *,
    repr_set_name: str,
    weight_path: str,
    model_repr_path: str,
    load_metric_dicts: Callable,
    read_latest_model_runtime: Callable,
) -> None:
    target_metric = normalize_simplets_metric(getattr(args, "base_metrics", "M"))
    method = simplets_method_name(target_metric)
    expected_order = [str(x) for x in current_zoo_abbr_order_list]
    train_scope = get_advanced_baseline_train_scope(args)
    training_repr_forward_stem = _repr_forward_stem_for_train_scope(args, train_scope)
    selector_repr_set_name = build_selector_artifact_repr_set_name(args)
    classifier_backend_requested = _requested_cluster_classifier_backend(args)
    classifier_backend_expected = _resolve_cluster_classifier_backend(classifier_backend_requested)
    artifact_path = Path(model_repr_path)
    weight_artifact_path = Path(weight_path)
    manifest_path = artifact_path.with_name(f"{artifact_path.stem}_model_manifest.json")
    if bool(getattr(args, "skip_saved", False)) and artifact_path.exists() and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        manifest_source_repr = str(manifest.get("source_repr_set_name", "") or "")
        manifest_selector_repr = str(manifest.get("selector_repr_set_name", "") or "")
        manifest_train_scope = str(manifest.get("advanced_baseline_train_scope", "center") or "center")
        if (
            [str(x) for x in manifest.get("model_abbr_order", [])] == expected_order
            and manifest_source_repr == repr_set_name
            and manifest_selector_repr == selector_repr_set_name
            and manifest_train_scope == train_scope
            and str(manifest.get("classifier_policy_version", "")) == SIMPLETS_CLASSIFIER_POLICY_VERSION
            and str(manifest.get("classifier_backend", "")) == classifier_backend_expected
        ):
            _log_step3(f"skip-save ready: {artifact_path}")
            return

    _log_step3(
        f"source binding: Step1/2 repr={repr_set_name} ({getattr(args, 'repr_encoder', 'unknown')}), "
        f"Step3/4 selector repr={selector_repr_set_name} (internally trained TS2Vec), "
        f"train_scope={train_scope}"
    )

    label_t0 = time.perf_counter()
    repr_forward_csv_path = _repr_forward_csv_path_for_train_scope(args, train_scope)
    _log_step3(f"load Step2 per-sample labels: {repr_forward_csv_path}")
    metric_perf_dict_by_name, _metric_load_errors, step2_coverage = load_metric_dicts(
        args=args,
        current_zoo_abbr_order_list=current_zoo_abbr_order_list,
        repr_forward_csv_path=repr_forward_csv_path,
    )
    if not bool(step2_coverage.get("metric_complete", False)):
        label_refresh_seconds = time.perf_counter() - label_t0
        _write_simplets_insert_row(
            args=args,
            repr_set_name=repr_set_name,
            model_repr_path=str(artifact_path),
            weight_path=str(weight_artifact_path),
            model_names=expected_order,
            target_metric=target_metric,
            n_clusters=0,
            label_refresh_seconds=label_refresh_seconds,
            structure_refresh_seconds=0.0,
            selector_retrain_seconds=0.0,
            incoming_profile_seconds=float("nan"),
            step2_coverage=step2_coverage,
            runtime_status="skipped_step2_incomplete",
            runtime_path="",
            status="skipped_step2_incomplete",
            train_samples=0,
            embedding_dim=0,
            ts2vec_status="not_built",
            ts2vec_path="",
            feature_source="",
            label_source=repr_forward_csv_path,
            classifier_backend=classifier_backend_expected,
            classifier_name="not_built",
            ts2vec_train_seconds=0.0,
            advanced_baseline_train_scope=train_scope,
            training_repr_forward_stem=training_repr_forward_stem,
        )
        _log_step3("skip: Step2 metric coverage is incomplete.")
        return

    _log_step3(
        f"Step2 labels ready: metric={target_metric}, "
        f"samples={len(metric_perf_dict_by_name[target_metric]) and len(next(iter(metric_perf_dict_by_name[target_metric].values())))}"
    )
    _log_step3(f"load Step1 source windows: scope={train_scope}")
    windows, feature_source = _load_repr_windows(args, repr_set_name, train_scope)
    _log_step3(f"Step1 source windows ready: shape={tuple(windows.shape)}, source={feature_source}")
    metric_matrix = _ordered_metric_matrix(metric_perf_dict_by_name[target_metric], expected_order)
    if windows.shape[0] != metric_matrix.shape[0]:
        raise ValueError(
            "SimpleTS feature/label row mismatch: "
            f"windows={windows.shape[0]}, labels={metric_matrix.shape[0]}, "
            f"feature_source={feature_source}, label_source={repr_forward_csv_path}"
        )
    clean_metric_matrix = _impute_metric_matrix(metric_matrix)
    label_refresh_seconds = time.perf_counter() - label_t0

    structure_t0 = time.perf_counter()
    cluster_info = build_simplets_cluster_tables(
        clean_metric_matrix,
        expected_order,
        n_clusters=SIMPLETS_N_CLUSTERS,
        seed=int(getattr(args, "search_seed", 2025) or 2025),
    )
    structure_refresh_seconds = time.perf_counter() - structure_t0
    _log_step3(
        f"model clustering ready: models={len(expected_order)}, "
        f"clusters={len(cluster_info['cluster_winner_by_cluster'])}, "
        f"samples={clean_metric_matrix.shape[0]}"
    )

    (
        encoder,
        ts2vec_config,
        ts2vec_path,
        ts2vec_seconds,
        ts2vec_status,
        ts2vec_train_batch_size,
    ) = _train_or_load_ts2vec(
        args,
        repr_set_name,
        windows,
    )
    selector_t0 = time.perf_counter()
    _log_step3(
        f"encode classifier features: samples={windows.shape[0]}, "
        f"batch_size={int(ts2vec_train_batch_size['runtime_batch_size'])}"
    )
    embed_t0 = time.perf_counter()
    embeddings = _embed_windows(
        encoder,
        windows,
        batch_size=int(ts2vec_train_batch_size["runtime_batch_size"]),
    )
    _log_step3(
        f"classifier features ready: embeddings={tuple(embeddings.shape)}, "
        f"elapsed={time.perf_counter() - embed_t0:.3f}s"
    )
    _log_step3(
        f"train cluster classifier: samples={embeddings.shape[0]}, "
        f"classes={np.unique(cluster_info['sample_cluster_labels']).size}, "
        f"backend_request={classifier_backend_requested}, backend={classifier_backend_expected}"
    )
    classifier, classifier_status, classifier_diagnostics = _fit_cluster_classifier(
        embeddings,
        cluster_info["sample_cluster_labels"],
        seed=int(getattr(args, "search_seed", 2025) or 2025),
        backend=classifier_backend_expected,
        log_fn=_log_step3,
    )
    selector_retrain_seconds = time.perf_counter() - selector_t0
    _log_step3(
        f"classifier ready: status={classifier_status}, "
        f"backend={classifier_diagnostics.get('classifier_backend', classifier_backend_expected)}, "
        f"fit_seconds={float(classifier_diagnostics.get('fit_seconds', 0.0)):.3f}, "
        f"embeddings={tuple(embeddings.shape)}, selector_retrain_seconds={selector_retrain_seconds:.3f}"
    )

    latest_model_abbr = expected_order[-1] if expected_order else ""
    incoming_profile_seconds, runtime_path, runtime_status = read_latest_model_runtime(
        args,
        latest_model_abbr,
    )
    timing_valid = np.isfinite(float(incoming_profile_seconds))
    model_mean = np.asarray(cluster_info["model_mean_metric"], dtype=np.float64)
    inverse = 1.0 / np.maximum(model_mean, 1e-8)
    if not np.isfinite(inverse).all() or float(np.sum(inverse)) <= 0:
        inverse = np.ones_like(model_mean)
    model_weights = {
        name: float(value)
        for name, value in zip(expected_order, inverse / np.max(inverse))
    }
    table_paths = _write_tables(
        artifact_path=artifact_path,
        model_names=expected_order,
        metric_matrix=clean_metric_matrix,
        cluster_info=cluster_info,
        target_metric=target_metric,
    )
    payload = {
        "__repr_format__": SIMPLETS_REPR_FORMAT,
        "method": method,
        "method_version": SIMPLETS_METHOD_VERSION,
        "target_metric": target_metric,
        "n_clusters": SIMPLETS_N_CLUSTERS,
        "model_abbr_order": expected_order,
        "model_names": expected_order,
        "model_cluster_ids": cluster_info["model_cluster_ids"].astype(int).tolist(),
        "cluster_winner_by_cluster": {
            int(k): int(v) for k, v in cluster_info["cluster_winner_by_cluster"].items()
        },
        "cluster_members": cluster_info["cluster_members"],
        "sample_cluster_labels": cluster_info["sample_cluster_labels"].astype(int).tolist(),
        "best_model_idx_by_sample": cluster_info["best_model_idx_by_sample"].astype(int).tolist(),
        "global_prior_order": cluster_info["global_prior_order"].astype(int).tolist(),
        "model_mean_metric": cluster_info["model_mean_metric"].astype(float).tolist(),
        "model_metric_weights": model_weights,
        "classifier": classifier,
        "classifier_status": classifier_status,
        "classifier_policy_version": SIMPLETS_CLASSIFIER_POLICY_VERSION,
        "classifier_backend_requested": classifier_backend_requested,
        "classifier_backend": classifier_diagnostics.get("classifier_backend", classifier_backend_expected),
        "classifier_name": classifier_diagnostics.get("classifier_name", classifier_status),
        "classifier_params": classifier_diagnostics.get("classifier_params", {}),
        "classifier_diagnostics": classifier_diagnostics,
        "ts2vec_config": ts2vec_config,
        "ts2vec_state_dict": {
            key: value.detach().cpu()
            for key, value in encoder.state_dict().items()
        },
        "ts2vec_checkpoint_path": str(ts2vec_path),
        "ts2vec_checkpoint_status": ts2vec_status,
        "ts2vec_train_seconds": float(ts2vec_seconds),
        "ts2vec_train_batch_size": ts2vec_train_batch_size,
        "repr_set_name": repr_set_name,
        "source_repr_set_name": repr_set_name,
        "selector_repr_set_name": selector_repr_set_name,
        "advanced_baseline_train_scope": train_scope,
        "training_repr_forward_stem": training_repr_forward_stem,
        "selector_encoder": "TS2Vec",
        "feature_source": feature_source,
        "label_source": repr_forward_csv_path,
        "artifact_tables": table_paths,
        "train_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "repr_v": int(getattr(args, "repr_v", 6)),
        "stage": int(getattr(args, "current_zoo_num", len(expected_order)) or len(expected_order)),
        "zoo_total_num": int(getattr(args, "zoo_total_num", len(expected_order)) or len(expected_order)),
        **build_model_family_metadata(expected_order),
    }
    weight_payload = {
        "total_models": len(expected_order),
        "model_weights": model_weights,
        "weight_source": f"simplets_global_inverse_{target_metric}",
        "advanced_baseline_train_scope": train_scope,
        "model_abbr_order": expected_order,
        **build_model_family_metadata(expected_order),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    weight_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_pickle_dump(payload, str(artifact_path))
    atomic_pickle_dump(weight_payload, str(weight_artifact_path))

    manifest = {
        "schema_version": 1,
        "method": method,
        "method_version": SIMPLETS_METHOD_VERSION,
        "target_metric": target_metric,
        "model_abbr_order": expected_order,
        "stage": payload["stage"],
        "zoo_total_num": payload["zoo_total_num"],
        "n_clusters": SIMPLETS_N_CLUSTERS,
        "feature_source": feature_source,
        "source_repr_set_name": repr_set_name,
        "selector_repr_set_name": selector_repr_set_name,
        "advanced_baseline_train_scope": train_scope,
        "training_repr_forward_stem": training_repr_forward_stem,
        "selector_encoder": "TS2Vec",
        "classifier_policy_version": SIMPLETS_CLASSIFIER_POLICY_VERSION,
        "classifier_backend_requested": classifier_backend_requested,
        "classifier_backend": classifier_diagnostics.get("classifier_backend", classifier_backend_expected),
        "classifier_name": classifier_diagnostics.get("classifier_name", classifier_status),
        "classifier_params": classifier_diagnostics.get("classifier_params", {}),
        "label_source": repr_forward_csv_path,
        "artifact_path": str(artifact_path),
        "weight_path": str(weight_artifact_path),
        "ts2vec_checkpoint_path": str(ts2vec_path),
        "ts2vec_train_batch_size": ts2vec_train_batch_size,
        "insert_timing_csv": str(simplets_csv_root() / "step3_insert_timing.csv"),
        "artifact_tables": table_paths,
        "assumptions": [
            "Model clusters are fitted over model performance vectors across Step2 probe samples.",
            "Probe sample labels are the cluster ids of each sample's best model.",
            "SimpleTS route Rank1 is restricted to cluster winners; non-winners only fill the tail order.",
        ],
        **build_model_family_metadata(expected_order),
    }
    _write_manifest(manifest_path, manifest)
    _write_simplets_insert_row(
        args=args,
        repr_set_name=repr_set_name,
        model_repr_path=str(artifact_path),
        weight_path=str(weight_artifact_path),
        model_names=expected_order,
        target_metric=target_metric,
        n_clusters=SIMPLETS_N_CLUSTERS,
        label_refresh_seconds=label_refresh_seconds,
        structure_refresh_seconds=structure_refresh_seconds,
        selector_retrain_seconds=selector_retrain_seconds,
        incoming_profile_seconds=incoming_profile_seconds,
        step2_coverage=step2_coverage,
        runtime_status=runtime_status,
        runtime_path=runtime_path,
        status="built",
        train_samples=int(embeddings.shape[0]),
        embedding_dim=int(embeddings.shape[1]),
        ts2vec_status=ts2vec_status,
        ts2vec_path=str(ts2vec_path),
        feature_source=feature_source,
        label_source=repr_forward_csv_path,
        classifier_backend=str(classifier_diagnostics.get("classifier_backend", classifier_backend_expected)),
        classifier_name=str(classifier_diagnostics.get("classifier_name", classifier_status)),
        ts2vec_train_seconds=ts2vec_seconds,
        advanced_baseline_train_scope=train_scope,
        training_repr_forward_stem=training_repr_forward_stem,
    )
    insert_total = (
        float(incoming_profile_seconds)
        + label_refresh_seconds
        + structure_refresh_seconds
        + selector_retrain_seconds
        if timing_valid
        else float("nan")
    )
    _log_step3(
        f"saved {method} artifact -> {artifact_path}; insert_total_seconds={_format_float(insert_total) or 'nan'}"
    )


def _write_simplets_insert_row(
    *,
    args,
    repr_set_name: str,
    model_repr_path: str,
    weight_path: str,
    model_names: list[str],
    target_metric: str,
    n_clusters: int,
    label_refresh_seconds: float,
    structure_refresh_seconds: float,
    selector_retrain_seconds: float,
    incoming_profile_seconds: float,
    step2_coverage: dict,
    runtime_status: str,
    runtime_path: str,
    status: str,
    train_samples: int,
    embedding_dim: int,
    ts2vec_status: str,
    ts2vec_path: str,
    feature_source: str,
    label_source: str,
    classifier_backend: str = "",
    classifier_name: str = "",
    ts2vec_train_seconds: float = 0.0,
    advanced_baseline_train_scope: str | None = None,
    training_repr_forward_stem: str | None = None,
) -> None:
    train_scope = str(
        advanced_baseline_train_scope
        or get_advanced_baseline_train_scope(args)
    )
    training_stem = str(
        training_repr_forward_stem
        or _repr_forward_stem_for_train_scope(args, train_scope)
    )
    latest_model_abbr = str(model_names[-1]) if model_names else ""
    timing_valid = np.isfinite(float(incoming_profile_seconds))
    insert_total_seconds = (
        float(incoming_profile_seconds)
        + float(label_refresh_seconds)
        + float(structure_refresh_seconds)
        + float(selector_retrain_seconds)
        if timing_valid
        else float("nan")
    )
    row = {
        "row_key": Path(model_repr_path).stem,
        "status": status,
        "method": simplets_method_name(target_metric),
        "method_version": SIMPLETS_METHOD_VERSION,
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
        "n_clusters": int(n_clusters),
        "train_samples": int(train_samples),
        "embedding_dim": int(embedding_dim),
        "classifier_policy_version": SIMPLETS_CLASSIFIER_POLICY_VERSION,
        "classifier_backend": classifier_backend,
        "classifier_name": classifier_name,
        "ts2vec_train_seconds": _format_float(ts2vec_train_seconds),
        "selector_retrain_excludes_ts2vec": "true",
        "label_refresh_seconds": _format_float(label_refresh_seconds),
        "structure_refresh_seconds": _format_float(structure_refresh_seconds),
        "selector_retrain_seconds": _format_float(selector_retrain_seconds),
        "incoming_profile_seconds": _format_float(incoming_profile_seconds),
        "insert_total_seconds": _format_float(insert_total_seconds),
        "old_model_forwards": 0,
        "timing_valid": str(bool(timing_valid)).lower(),
        "ts2vec_checkpoint_status": ts2vec_status,
        "ts2vec_checkpoint_path": ts2vec_path,
        "feature_source": feature_source,
        "label_source": label_source,
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
    csv_path = simplets_csv_root() / "step3_insert_timing.csv"
    _upsert_csv(csv_path, row, SIMPLETS_INSERT_FIELDS, key_fields=["row_key"])
    print(f"[SimpleTS Step3][timing] saved -> {csv_path}", flush=True)


def _load_encoder_from_payload(payload: dict, device) -> TS2VecEncoder:
    config = dict(payload.get("ts2vec_config", {}))
    if not config:
        raise ValueError("SimpleTS payload missing ts2vec_config")
    encoder = _build_ts2vec_encoder(config, device=device)
    state = payload.get("ts2vec_state_dict")
    if state is None:
        ckpt_path = payload.get("ts2vec_checkpoint_path", "")
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"SimpleTS TS2Vec checkpoint missing: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("state_dict", ckpt)
    encoder.load_state_dict(state, strict=False)
    encoder.eval()
    return encoder


def _simplets_encoder_cache_key(payload: dict, device) -> tuple:
    checkpoint_path = str(payload.get("ts2vec_checkpoint_path", "") or "")
    checkpoint_signature: tuple = ("embedded_payload", id(payload))
    if checkpoint_path and os.path.exists(checkpoint_path):
        stat = os.stat(checkpoint_path)
        checkpoint_signature = (checkpoint_path, int(stat.st_size), int(stat.st_mtime_ns))
    config_signature = json.dumps(
        dict(payload.get("ts2vec_config", {})),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return str(torch.device(device)), checkpoint_signature, config_signature


def load_simplets_encoder_cached(payload: dict, device) -> tuple[TS2VecEncoder, bool]:
    """Reuse the immutable TS2Vec encoder across Step4 tasks in one process."""
    key = _simplets_encoder_cache_key(payload, device)
    cached = _SIMPLETS_ENCODER_CACHE.get(key)
    if cached is not None:
        _SIMPLETS_ENCODER_CACHE.move_to_end(key)
        return cached, True

    encoder = _load_encoder_from_payload(payload, device=device)
    _SIMPLETS_ENCODER_CACHE[key] = encoder
    _SIMPLETS_ENCODER_CACHE.move_to_end(key)
    while len(_SIMPLETS_ENCODER_CACHE) > SIMPLETS_ENCODER_CACHE_MAX_ENTRIES:
        _SIMPLETS_ENCODER_CACHE.popitem(last=False)
    return encoder, False


def _classifier_proba_with_timing(classifier, x: np.ndarray, n_clusters: int) -> tuple[np.ndarray, dict]:
    backend, classifier_name = _classifier_backend_and_name(classifier)
    timing = {
        "classifier_backend": backend,
        "classifier_name": classifier_name,
        "classifier_input_shape": [int(v) for v in np.asarray(x).shape],
        "predict_raw_ms": 0.0,
        "predict_postprocess_ms": 0.0,
        "predict_fallback_ms": 0.0,
    }
    if hasattr(classifier, "predict_proba"):
        raw_t0 = time.perf_counter()
        raw = np.asarray(classifier.predict_proba(x), dtype=np.float64)
        timing["predict_raw_ms"] = (time.perf_counter() - raw_t0) * 1000.0
        post_t0 = time.perf_counter()
        classes = getattr(classifier, "classes_", None)
        if classes is None and hasattr(classifier, "steps") and classifier.steps:
            classes = getattr(classifier.steps[-1][1], "classes_", None)
        if classes is None:
            classes = np.arange(raw.shape[1])
        classes = np.asarray(classes, dtype=np.int64)
        out = np.zeros((x.shape[0], int(n_clusters)), dtype=np.float64)
        for pos, cls in enumerate(classes):
            if 0 <= int(cls) < int(n_clusters):
                out[:, int(cls)] = raw[:, pos]
        row_sum = out.sum(axis=1, keepdims=True)
        bad = row_sum[:, 0] <= 0
        out[~bad] = out[~bad] / row_sum[~bad]
        if np.any(bad):
            fallback_t0 = time.perf_counter()
            pred = np.asarray(classifier.predict(x[bad]), dtype=np.int64)
            timing["predict_fallback_ms"] = (time.perf_counter() - fallback_t0) * 1000.0
            out[bad, :] = 0.0
            bad_rows = np.where(bad)[0]
            for row_idx, cluster_id in zip(bad_rows, pred):
                cluster_id = int(cluster_id)
                if 0 <= cluster_id < int(n_clusters):
                    out[row_idx, cluster_id] = 1.0
                else:
                    out[row_idx, 0] = 1.0
        timing["predict_postprocess_ms"] = (time.perf_counter() - post_t0) * 1000.0
        return out, timing
    raw_t0 = time.perf_counter()
    pred = np.asarray(classifier.predict(x), dtype=np.int64)
    timing["predict_raw_ms"] = (time.perf_counter() - raw_t0) * 1000.0
    post_t0 = time.perf_counter()
    out = np.zeros((x.shape[0], int(n_clusters)), dtype=np.float64)
    for row_idx, cluster_id in enumerate(pred):
        cluster_id = int(cluster_id)
        out[row_idx, cluster_id if 0 <= cluster_id < int(n_clusters) else 0] = 1.0
    timing["predict_postprocess_ms"] = (time.perf_counter() - post_t0) * 1000.0
    return out, timing


def _classifier_proba(classifier, x: np.ndarray, n_clusters: int) -> np.ndarray:
    out, _timing = _classifier_proba_with_timing(classifier, x, n_clusters)
    return out


def _simplets_sample_cluster_probabilities(
    payload: dict,
    samples,
    *,
    encoder: TS2VecEncoder | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    t_feature0 = time.perf_counter()
    x = _coerce_windows_array(samples)
    n, t, c = x.shape
    flat = x.transpose(0, 2, 1).reshape(n * c, t)
    if encoder is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = _load_encoder_from_payload(payload, device=device)
    else:
        device = encoder.device
    batch_size = 256
    outs = []
    with torch.no_grad():
        for start in range(0, flat.shape[0], batch_size):
            batch = torch.as_tensor(flat[start:start + batch_size], dtype=torch.float32, device=device)
            outs.append(encoder.forward(batch).detach().cpu().numpy())
    embeddings = np.vstack(outs).astype(np.float32)
    feature_ms = (time.perf_counter() - t_feature0) * 1000.0

    t_predict0 = time.perf_counter()
    n_clusters = int(payload.get("n_clusters", SIMPLETS_N_CLUSTERS))
    proba_flat, predict_detail = _classifier_proba_with_timing(
        payload["classifier"],
        embeddings,
        n_clusters=n_clusters,
    )
    proba_by_sample = proba_flat.reshape(n, c, n_clusters)
    predict_ms = (time.perf_counter() - t_predict0) * 1000.0

    return proba_by_sample, embeddings, {
        "feature_ms": feature_ms,
        "predict_ms": predict_ms,
        "n_clusters": n_clusters,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        **predict_detail,
    }


def _simplets_orders_from_probabilities(payload: dict, probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim < 2:
        raise ValueError(f"SimpleTS probabilities expect [..., clusters], got {probabilities.shape}")

    cluster_winners = {
        int(k): int(v)
        for k, v in dict(payload.get("cluster_winner_by_cluster", {})).items()
    }
    global_prior = [int(x) for x in payload.get("global_prior_order", [])]
    model_count = len(payload.get("model_abbr_order", payload.get("model_names", [])))
    if not global_prior:
        global_prior = list(range(model_count))
    flat_probabilities = probabilities.reshape(-1, probabilities.shape[-1])
    orders = []
    for row_probabilities in flat_probabilities:
        ordered = []
        for cluster_id in np.argsort(-row_probabilities, kind="mergesort"):
            winner = cluster_winners.get(int(cluster_id))
            if winner is not None and winner not in ordered:
                ordered.append(int(winner))
        for model_idx in global_prior:
            if int(model_idx) not in ordered:
                ordered.append(int(model_idx))
        orders.append(ordered[:model_count])
    return np.asarray(orders, dtype=np.int64).reshape(*probabilities.shape[:-1], model_count)


def predict_simplets_rank_tensor(
    payload: dict,
    samples,
    *,
    encoder: TS2VecEncoder | None = None,
) -> tuple[np.ndarray, dict]:
    proba_by_sample, embeddings, timing = _simplets_sample_cluster_probabilities(
        payload,
        samples,
        encoder=encoder,
    )
    proba = proba_by_sample.mean(axis=0)
    t_rank0 = time.perf_counter()
    order_arr = _simplets_orders_from_probabilities(payload, proba)
    rank_ms = (time.perf_counter() - t_rank0) * 1000.0
    c = int(proba.shape[0])
    return order_arr.T.reshape(-1, 1, c), {
        **timing,
        "rank_ms": rank_ms,
        "cluster_probability_shape": list(proba.shape),
    }


def predict_simplets_sample_rank_tensors(
    payload: dict,
    samples,
    *,
    encoder: TS2VecEncoder | None = None,
) -> tuple[np.ndarray, dict]:
    """Batch all task samples while preserving one independent rank per sample/channel."""
    proba_by_sample, embeddings, timing = _simplets_sample_cluster_probabilities(
        payload,
        samples,
        encoder=encoder,
    )
    t_rank0 = time.perf_counter()
    orders = _simplets_orders_from_probabilities(payload, proba_by_sample)  # (N,C,K)
    rank_tensors = orders.transpose(0, 2, 1).astype(np.int64, copy=False)  # (N,K,C)
    rank_ms = (time.perf_counter() - t_rank0) * 1000.0
    return rank_tensors, {
        **timing,
        "rank_ms": rank_ms,
        "cluster_probability_shape": list(proba_by_sample.shape),
        "sample_count": int(proba_by_sample.shape[0]),
    }

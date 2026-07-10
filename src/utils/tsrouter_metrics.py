from __future__ import annotations

import os
import pickle
import re
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd
from gluonts.time_feature import get_seasonality

from utils.gift_eval import evaluate_forecasts_fast, get_cached_seasonal_errors
from utils.project_paths import BASELINE_CSV_ROOT, TSFM_CSV_ROOT, TSROUTER_CSV_ROOT
from utils.path_utils import resolve_tsfm_result_path


TSROUTER_EXTRA_METRIC_COLUMNS = [
    "ENC_TOP1_ENRICH",
    "ENC_TOP3_ENRICH",
    "ENC_TOP1_SUBSET_RATE",
    "ENC_TOP3_SUBSET_RATE",
    "REGION_WEIGHTED_PURITY",
    "REGION_DIAG_RANK",
    "REGION_DELTA_RANK",
    "TEST_WINDOW_TOP1_ACC",
    "TEST_WINDOW_TOP3_HIT",
    "TEST_WINDOW_EVAL_N",
    "SINGLE_TOP1_ACC",
    "SINGLE_TOP3_HIT",
    "SINGLE_CHANNELS_EVAL",
    "TEST_WINDOW_CHANNEL_TOP1_ACC",
    "TEST_WINDOW_CHANNEL_TOP3_HIT",
    "TEST_WINDOW_CHANNEL_EVAL_N",
    "TEST_WINDOW_TASK_TOP1_ACC",
    "TEST_WINDOW_TASK_TOP3_HIT",
    "TEST_WINDOW_TASK_EVAL_N",
    "TEST_CHANNEL_TASK_TOP1_ACC",
    "TEST_CHANNEL_TASK_TOP3_HIT",
    "TEST_CHANNEL_TASK_EVAL_N",
]

COMPETENCE_REGION_SCHEMA_VERSION = 2
COMPETENCE_REGION_METRIC_COLUMNS = [
    "ENC_TOP1_SUBSET_RATE",
    "ENC_TOP3_SUBSET_RATE",
    "REGION_WEIGHTED_PURITY",
    "REGION_DIAG_RANK",
    "REGION_DELTA_RANK",
]
DEFAULT_PROCESS_METRICS_FIG_DIR = "results_csv/TSRouter/vldb/figures/competence_regions"
COMPETENCE_HEATMAP_FONT_SCALE = 2.0
COMPETENCE_HEATMAP_DIAGONAL_DARKEN = 0.86
COMPETENCE_HEATMAP_DIAGONAL_LINEWIDTH = 1.8

TSROUTER_ROUTE_DETAIL_COLUMNS = [
    "rank_consistency_instability_by_channel",
    "channel_model_rank",
]

TSROUTER_CORE_METRIC_COLUMNS = [
    c for c in TSROUTER_EXTRA_METRIC_COLUMNS
    if c not in {
        "ENC_TOP1_ENRICH",
        "ENC_TOP3_ENRICH",
        "SINGLE_CHANNELS_EVAL",
        "TEST_WINDOW_EVAL_N",
        "TEST_WINDOW_CHANNEL_EVAL_N",
        "TEST_WINDOW_TASK_EVAL_N",
        "TEST_CHANNEL_TASK_EVAL_N",
    }
]

TSROUTER_REQUIRED_PROCESS_METRIC_COLUMNS = [
    "ENC_TOP1_SUBSET_RATE",
    "ENC_TOP3_SUBSET_RATE",
    "SINGLE_TOP1_ACC",
    "SINGLE_TOP3_HIT",
    "TEST_WINDOW_CHANNEL_TOP1_ACC",
    "TEST_WINDOW_CHANNEL_TOP3_HIT",
    "TEST_WINDOW_TASK_TOP1_ACC",
    "TEST_WINDOW_TASK_TOP3_HIT",
    "TEST_CHANNEL_TASK_TOP1_ACC",
    "TEST_CHANNEL_TASK_TOP3_HIT",
]

_PER_CHANNEL_ERROR_MATRIX_CACHE: dict[tuple[Any, ...], tuple[np.ndarray, list[int], list[int]]] = {}
_PER_CHANNEL_METRIC_IMPL = "gluonts_per_channel_v1"
_PER_CHANNEL_FALLBACK_IMPL = "local_fallback_v2"
_PER_WINDOW_ERROR_MATRIX_CACHE: dict[tuple[Any, ...], tuple[np.ndarray, list[int], pd.DataFrame]] = {}
_PER_WINDOW_METRIC_IMPL = "local_per_window_v1"


class _PerChannelTestData(SimpleNamespace):
    def __len__(self) -> int:
        return len(self.label)


def _model_label(family: str, size_name: str, info: Mapping[str, Any]) -> str:
    return f"{family}_{size_name}(id={int(info['id'])})"


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def nan_metric_row() -> Dict[str, float]:
    return {c: float("nan") for c in TSROUTER_EXTRA_METRIC_COLUMNS}


def _fmt_float_token(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except Exception:
        return str(value)


def rank_decay_coef(args: Any | None = None) -> float:
    try:
        val = float(getattr(args, "rank_decay_coef", 1.0))
    except Exception:
        val = 1.0
    if not np.isfinite(val):
        val = 1.0
    return max(0.0, val)


def rank_position_scores(max_rank: int, decay_coef: float = 1.0) -> np.ndarray:
    max_rank = max(1, int(max_rank))
    decay_coef = max(0.0, float(decay_coef))
    pos = np.arange(max_rank, dtype=np.float32)
    return max_rank - decay_coef * pos


def aggregate_rank_orders(
    rank_orders: Iterable[Iterable[int]],
    *,
    num_models: int | None = None,
    decay_coef: float = 1.0,
    weights: Iterable[float] | None = None,
) -> list[int]:
    orders = [list(order) for order in rank_orders if order is not None]
    if not orders:
        return []
    max_rank = max(len(order) for order in orders)
    if num_models is None:
        model_set = {int(m) for order in orders for m in order if int(m) >= 0}
        num_models = (max(model_set) + 1) if model_set else max_rank
    num_models = int(num_models)
    score_template = rank_position_scores(max_rank, decay_coef=decay_coef)
    if weights is None:
        weights = [1.0] * len(orders)
    scores: dict[int, float] = {}
    for order, weight in zip(orders, weights):
        w = float(weight)
        for pos, mid in enumerate(order):
            mid = int(mid)
            if mid < 0 or pos >= score_template.size:
                continue
            scores[mid] = scores.get(mid, 0.0) + w * float(score_template[pos])
    all_ids = list(range(num_models))
    for mid in all_ids:
        scores.setdefault(mid, 0.0)
    return sorted(all_ids, key=lambda mid: (-scores.get(mid, 0.0), mid))


def encoder_enrichment_suffix(args: Any | None = None) -> str:
    if args is None:
        return ""
    parts = [
        f"w{_fmt_float_token(getattr(args, 'repr_weight_ratio', 0.0))}",
        f"d{getattr(args, 'repr_distance_metric', 'cos')}",
    ]
    if str(getattr(args, "repr_v", "")).startswith("5"):
        parts.extend([
            f"k{getattr(args, 'repr_v5_nearest_k', 10)}",
            f"rd{_fmt_float_token(rank_decay_coef(args))}",
            f"p{_fmt_float_token(getattr(args, 'repr_v5_distance_power', 1.0))}",
        ])
    else:
        parts.append(f"agg{getattr(args, 'model_repr_agg', 'min')}")
    return "_" + "_".join(str(p) for p in parts)


def encoder_enrichment_paths(model_repr_path: str, args: Any | None = None) -> tuple[str, str]:
    base_dir = TSROUTER_CSV_ROOT / "Model_zoo_repr"
    return str(base_dir / "encoder_enrichment_summary.csv"), str(base_dir / "encoder_enrichment_by_model.csv")


def competence_region_paths(model_repr_path: str, args: Any | None = None) -> tuple[str, str]:
    base_dir = TSROUTER_CSV_ROOT / "Model_zoo_repr"
    return str(base_dir / "competence_region_summary.csv"), str(base_dir / "competence_region_by_owner.csv")


def resolve_process_metrics_region_rule(args: Any | None = None) -> str:
    raw = str(getattr(args, "process_metrics_region_rule", "auto") or "auto").strip().lower()
    aliases = {
        "strict_1nn": "strict",
        "strict-1nn": "strict",
        "1nn": "strict",
        "effective_region": "effective",
    }
    raw = aliases.get(raw, raw)
    if raw == "auto":
        try:
            weight_ratio = float(getattr(args, "repr_weight_ratio", 0.0))
        except Exception:
            weight_ratio = 0.0
        return "effective" if abs(weight_ratio) > 1e-12 else "strict"
    if raw not in {"strict", "effective"}:
        raise ValueError(
            f"Unknown process_metrics_region_rule={raw!r}; use auto, strict, or effective"
        )
    return raw


def compute_competence_region_metrics(
    metrics_matrix: np.ndarray,
    model_names: Iterable[str],
    assignments: Iterable[int],
) -> tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Compute Table-5 region metrics from lower-is-better per-window scores.

    Heatmap rows are region owners, columns are evaluated models, and each
    value is the evaluated model's mean true rank inside that owner region.
    """
    names = [str(name) for name in model_names]
    arr = np.asarray(metrics_matrix, dtype=np.float64)
    owner = np.asarray(list(assignments), dtype=np.int64).reshape(-1)
    if arr.ndim != 2:
        raise ValueError(f"metrics_matrix must be 2D, got {arr.shape}")
    n_models, n_points = arr.shape
    if n_models != len(names):
        raise ValueError("metrics_matrix row count and model_names length mismatch")
    if owner.shape[0] != n_points:
        raise ValueError(
            f"assignment length mismatch: assignments={owner.shape[0]}, points={n_points}"
        )
    if n_points == 0:
        raise ValueError("competence-region metrics require at least one point")
    if np.any((owner < 0) | (owner >= n_models)):
        bad = np.unique(owner[(owner < 0) | (owner >= n_models)]).tolist()
        raise ValueError(f"assignments contain invalid owner ids: {bad[:8]}")

    # Stable double argsort gives one-based ranks and deterministic model-order
    # tie breaking, matching the Step3 lower-is-better contract.
    order = np.argsort(arr, axis=0, kind="stable")
    ranks = np.argsort(order, axis=0, kind="stable") + 1
    winner = order[0, :]
    point_ids = np.arange(n_points, dtype=np.int64)
    assigned_rank = ranks[owner, point_ids]

    heatmap = np.full((n_models, n_models), np.nan, dtype=np.float64)
    owner_rows: list[Dict[str, Any]] = []
    purity_numerator = 0
    for owner_id, owner_name in enumerate(names):
        idx = np.where(owner == owner_id)[0]
        if idx.size == 0:
            owner_rows.append(
                {
                    "region_owner": owner_name,
                    "owner_index": owner_id,
                    "region_size": 0,
                    "region_share": 0.0,
                    "majority_winner": "",
                    "region_purity": np.nan,
                    "owner_top1_rate": np.nan,
                    "owner_top3_rate": np.nan,
                    "owner_diag_rank": np.nan,
                }
            )
            continue
        heatmap[owner_id, :] = ranks[:, idx].mean(axis=1)
        counts = np.bincount(winner[idx], minlength=n_models)
        majority_id = int(np.argmax(counts))
        purity_numerator += int(counts[majority_id])
        owner_rows.append(
            {
                "region_owner": owner_name,
                "owner_index": owner_id,
                "region_size": int(idx.size),
                "region_share": float(idx.size / n_points),
                "majority_winner": names[majority_id],
                "region_purity": float(counts[majority_id] / idx.size),
                "owner_top1_rate": float(np.mean(assigned_rank[idx] <= 1)),
                "owner_top3_rate": float(np.mean(assigned_rank[idx] <= min(3, n_models))),
                "owner_diag_rank": float(heatmap[owner_id, owner_id]),
            }
        )

    valid_rows = np.where(np.isfinite(heatmap).any(axis=1))[0]
    diag_values = np.asarray([heatmap[i, i] for i in valid_rows], dtype=np.float64)
    offdiag_values = np.asarray(
        [
            heatmap[i, j]
            for i in valid_rows
            for j in range(n_models)
            if j != i and np.isfinite(heatmap[i, j])
        ],
        dtype=np.float64,
    )
    diag_rank = float(np.mean(diag_values)) if diag_values.size else float("nan")
    offdiag_rank = float(np.mean(offdiag_values)) if offdiag_values.size else float("nan")
    delta_rank = (
        float(offdiag_rank - diag_rank)
        if np.isfinite(offdiag_rank) and np.isfinite(diag_rank)
        else float("nan")
    )
    summary = {
        "ENC_TOP1_SUBSET_RATE": float(np.mean(assigned_rank <= 1)),
        "ENC_TOP3_SUBSET_RATE": float(np.mean(assigned_rank <= min(3, n_models))),
        "REGION_WEIGHTED_PURITY": float(purity_numerator / n_points),
        "REGION_DIAG_RANK": diag_rank,
        "REGION_DELTA_RANK": delta_rank,
        "REGION_OFFDIAG_RANK": offdiag_rank,
        "REGION_EVAL_N": int(n_points),
        "REGION_NONEMPTY_COUNT": int(valid_rows.size),
        "REGION_EMPTY_COUNT": int(n_models - valid_rows.size),
    }
    return (
        summary,
        pd.DataFrame(heatmap, index=names, columns=names),
        pd.DataFrame(owner_rows),
    )


def _safe_competence_figure_stem(
    model_repr_path: str,
    region_rule: str,
    configured_weight_ratio: float,
    distance_metric: str,
    model_repr_agg: str,
) -> str:
    raw_stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(str(model_repr_path)).stem).strip("-._")
    digest = hashlib.sha1(str(model_repr_path).encode("utf-8")).hexdigest()[:12]
    if len(raw_stem) > 150:
        raw_stem = f"{raw_stem[:135]}_{digest}"
    config_tag = (
        f"cfgw{configured_weight_ratio:g}-d{distance_metric}-agg{model_repr_agg}"
    )
    rule_tag = (
        f"{config_tag}-strict-1nn"
        if region_rule == "strict"
        else f"{config_tag}-effective-lambda{configured_weight_ratio:g}"
    )
    rule_tag = rule_tag.replace("-", "_").replace(".", "p")
    return f"{raw_stem}__{rule_tag}"


def _save_competence_heatmap(
    heatmap: pd.DataFrame,
    *,
    output_stem: Path,
) -> tuple[str, str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_stem.with_name(output_stem.name + "_owner_mean_rank.csv")
    png_path = output_stem.with_name(output_stem.name + "_owner_mean_rank_heatmap.png")
    pdf_path = output_stem.with_name(output_stem.name + "_owner_mean_rank_heatmap.pdf")
    heatmap.to_csv(csv_path, index=True, index_label="region_owner")

    data = heatmap.to_numpy(dtype=float)
    n_models = int(data.shape[0])
    axis_label_fontsize = 10 * COMPETENCE_HEATMAP_FONT_SCALE
    tick_fontsize = 8 * COMPETENCE_HEATMAP_FONT_SCALE
    colorbar_fontsize = 10 * COMPETENCE_HEATMAP_FONT_SCALE
    side = min(30.0, max(16.0, 0.96 * n_models + 6.0))
    fig, ax = plt.subplots(figsize=(side, side * 0.88))
    image = ax.imshow(
        data,
        cmap="viridis",
        vmin=1.0,
        vmax=float(max(1, n_models)),
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xlabel("Evaluated model", fontsize=axis_label_fontsize, fontweight="bold")
    ax.set_ylabel("Region owner", fontsize=axis_label_fontsize, fontweight="bold")
    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(heatmap.columns.tolist(), rotation=55, ha="right", fontsize=tick_fontsize)
    ax.set_yticklabels(heatmap.index.tolist(), fontsize=tick_fontsize)
    for i in range(n_models):
        diag_value = data[i, i]
        facecolor = "none"
        if np.isfinite(diag_value):
            base_rgba = image.cmap(image.norm(diag_value))
            facecolor = (
                base_rgba[0] * COMPETENCE_HEATMAP_DIAGONAL_DARKEN,
                base_rgba[1] * COMPETENCE_HEATMAP_DIAGONAL_DARKEN,
                base_rgba[2] * COMPETENCE_HEATMAP_DIAGONAL_DARKEN,
                base_rgba[3],
            )
        ax.add_patch(
            Rectangle(
                (i - 0.5, i - 0.5),
                1.0,
                1.0,
                facecolor=facecolor,
                edgecolor="black",
                linewidth=COMPETENCE_HEATMAP_DIAGONAL_LINEWIDTH,
            )
        )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean true rank (lower is better)", fontsize=colorbar_fontsize)
    colorbar.ax.tick_params(labelsize=colorbar_fontsize)
    fig.tight_layout()
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return str(csv_path), str(png_path), str(pdf_path)


def save_competence_region_report(
    *,
    model_repr_path: str,
    metrics_matrix: np.ndarray,
    model_names: Iterable[str],
    assignments: Iterable[int],
    region_rule: str,
    args: Any,
) -> Dict[str, Any]:
    region_rule = str(region_rule).strip().lower()
    if region_rule not in {"strict", "effective"}:
        raise ValueError(f"region_rule must be strict or effective, got {region_rule!r}")
    names = [str(name) for name in model_names]
    configured_weight_ratio = float(getattr(args, "repr_weight_ratio", 0.0))
    weight_ratio = configured_weight_ratio if region_rule == "effective" else 0.0
    summary, heatmap, by_owner = compute_competence_region_metrics(
        metrics_matrix,
        names,
        assignments,
    )
    stage = int(getattr(args, "current_zoo_num", len(names)) or len(names))
    figure_root = Path(
        str(getattr(args, "process_metrics_fig_dir", "") or DEFAULT_PROCESS_METRICS_FIG_DIR)
    )
    figure_stem = _safe_competence_figure_stem(
        model_repr_path,
        region_rule,
        configured_weight_ratio,
        str(getattr(args, "repr_distance_metric", "cos")),
        str(getattr(args, "model_repr_agg", "min")),
    )
    output_stem = figure_root / f"stage_{stage}" / figure_stem
    heatmap_csv, heatmap_png, heatmap_pdf = _save_competence_heatmap(
        heatmap,
        output_stem=output_stem,
    )

    summary_path, by_owner_path = competence_region_paths(model_repr_path, args=args)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    context = _encoder_enrichment_context(model_repr_path, args=args)
    common = {
        **context,
        "schema_version": COMPETENCE_REGION_SCHEMA_VERSION,
        "region_rule": region_rule,
        "region_weight_ratio": weight_ratio,
        "sample_mode": str(getattr(args, "sample_mode", "")),
        "heatmap_csv": heatmap_csv,
        "heatmap_png": heatmap_png,
        "heatmap_pdf": heatmap_pdf,
    }
    summary_row: Dict[str, Any] = {**common, **summary}
    _upsert_csv_rows(
        summary_path,
        pd.DataFrame([summary_row]),
        ["row_key", "region_rule"],
    )
    _migrate_legacy_competence_ca_columns(summary_path)
    by_owner_out = by_owner.copy()
    for key, value in common.items():
        by_owner_out[key] = value
    _upsert_csv_rows(
        by_owner_path,
        by_owner_out,
        ["row_key", "region_rule", "region_owner"],
    )
    print(
        f"[REGION_METRIC] rule={region_rule}, lambda={weight_ratio:g}, "
        f"PWW1={summary['ENC_TOP1_SUBSET_RATE']:.6g}, "
        f"PWW3={summary['ENC_TOP3_SUBSET_RATE']:.6g}, "
        f"WP={summary['REGION_WEIGHTED_PURITY']:.6g}, "
        f"DiagRank={summary['REGION_DIAG_RANK']:.6g}, "
        f"DeltaRank={summary['REGION_DELTA_RANK']:.6g}"
    )
    print(f"[REGION_METRIC] saved heatmap -> {heatmap_png}")
    return summary_row


def _legacy_encoder_enrichment_paths(model_repr_path: str, args: Any | None = None) -> tuple[str, str]:
    stem = str(model_repr_path).replace(".pkl", "")
    stem = stem + encoder_enrichment_suffix(args)
    return stem + "_encoder_enrichment_summary.csv", stem + "_encoder_enrichment_by_model.csv"


def _step3_subset_assign_candidates(model_repr_path: str, args: Any | None = None) -> list[str]:
    path = str(model_repr_path)
    if path.endswith(".pkl"):
        stem = path[:-4]
    else:
        stem = path
    unsuffixed = f"{stem}_subset_assign.pkl"
    suffix = encoder_enrichment_suffix(args)
    suffixed = f"{stem}{suffix}_subset_assign.pkl" if suffix else unsuffixed
    if str(getattr(args, "repr_v", ""))[:1] == "5":
        ordered = [suffixed, unsuffixed]
    else:
        ordered = [unsuffixed, suffixed]
    out: list[str] = []
    for candidate in ordered:
        if candidate not in out:
            out.append(candidate)
    return out


def _competence_metrics_from_mapping(row: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "ENC_TOP1_SUBSET_RATE": safe_float(
            row.get("ENC_TOP1_SUBSET_RATE", row.get("REGION_CA_TOP1", np.nan))
        ),
        "ENC_TOP3_SUBSET_RATE": safe_float(
            row.get("ENC_TOP3_SUBSET_RATE", row.get("REGION_CA_TOP3", np.nan))
        ),
        "REGION_WEIGHTED_PURITY": safe_float(row.get("REGION_WEIGHTED_PURITY", np.nan)),
        "REGION_DIAG_RANK": safe_float(row.get("REGION_DIAG_RANK", np.nan)),
        "REGION_DELTA_RANK": safe_float(row.get("REGION_DELTA_RANK", np.nan)),
    }


def _update_with_finite_metrics(out: Dict[str, float], metrics: Mapping[str, Any]) -> None:
    for key, value in metrics.items():
        parsed = safe_float(value)
        if np.isfinite(parsed):
            out[key] = parsed


def _load_subset_competence_metrics(
    model_repr_path: str,
    args: Any | None,
    region_rule: str,
) -> Dict[str, float]:
    for subset_path in _step3_subset_assign_candidates(model_repr_path, args=args):
        if not os.path.exists(subset_path):
            continue
        try:
            with open(subset_path, "rb") as f:
                payload = pickle.load(f)
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue
        competence = payload.get("competence_region")
        if not isinstance(competence, Mapping):
            continue
        reports = competence.get("reports")
        if not isinstance(reports, Mapping):
            continue
        rule = str(region_rule).strip().lower()
        selected = None
        for key, report in reports.items():
            if str(key).strip().lower() == rule:
                selected = report
                break
        if selected is None and str(competence.get("primary_rule", "")).strip().lower() == rule:
            selected = reports.get(competence.get("primary_rule"))
        if not isinstance(selected, Mapping):
            continue
        metrics = _competence_metrics_from_mapping(selected)
        if any(np.isfinite(safe_float(metrics.get(column, np.nan))) for column in COMPETENCE_REGION_METRIC_COLUMNS):
            return metrics
    return {}


def _encoder_enrichment_context(model_repr_path: str, args: Any | None = None) -> Dict[str, Any]:
    model_repr_name = Path(str(model_repr_path)).stem
    suffix = encoder_enrichment_suffix(args).lstrip("_")
    ctx: Dict[str, Any] = {
        "model_repr_name": model_repr_name,
        "encoder_enrichment_tag": suffix,
        "row_key": f"{model_repr_name}_{suffix}" if suffix else model_repr_name,
    }
    if args is not None:
        stage = int(getattr(args, "current_zoo_num", 0) or 0)
        total = int(getattr(args, "zoo_total_num", 0) or 0)
        ctx.update(
            {
                "zoo_tag": f"zoo{stage}-{total}" if stage and total else "",
                "stage": stage,
                "zoo_total_num": total,
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
            }
        )
        try:
            from utils.path_utils import build_repr_forward_stem, build_repr_set_name

            ctx["repr_set_name"] = build_repr_set_name(args)
            ctx["repr_forward_stem"] = build_repr_forward_stem(args)
        except Exception:
            pass
    else:
        m = re.match(r"^zoo(?P<stage>\d+)-(?P<total>\d+)_(?P<rest>.+)$", model_repr_name)
        if m:
            ctx["zoo_tag"] = f"zoo{m.group('stage')}-{m.group('total')}"
            ctx["stage"] = int(m.group("stage"))
            ctx["zoo_total_num"] = int(m.group("total"))
    return ctx


def _upsert_csv_rows(path: str, rows: pd.DataFrame, key_cols: list[str]) -> None:
    rows = rows.copy()
    if os.path.exists(path):
        try:
            existing = pd.read_csv(path)
        except Exception:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    if not existing.empty and all(col in existing.columns for col in key_cols):
        for _, row in rows[key_cols].drop_duplicates().iterrows():
            mask = pd.Series(True, index=existing.index)
            for col in key_cols:
                mask &= existing[col].astype(str).eq(str(row[col]))
            existing = existing.loc[~mask].copy()

    all_cols = list(dict.fromkeys(list(existing.columns) + list(rows.columns)))
    existing = existing.reindex(columns=all_cols)
    rows = rows.reindex(columns=all_cols)
    pd.concat([existing, rows], ignore_index=True).to_csv(path, index=False)


def _migrate_legacy_competence_ca_columns(path: str) -> None:
    """Rename the former CA columns to their equivalent PWW names in-place."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return
    changed = False
    for legacy, pww in (
        ("REGION_CA_TOP1", "ENC_TOP1_SUBSET_RATE"),
        ("REGION_CA_TOP3", "ENC_TOP3_SUBSET_RATE"),
    ):
        if legacy not in df.columns:
            continue
        legacy_values = pd.to_numeric(df[legacy], errors="coerce")
        if pww not in df.columns:
            df[pww] = legacy_values
        else:
            current = pd.to_numeric(df[pww], errors="coerce")
            df[pww] = current.where(current.notna(), legacy_values)
        df = df.drop(columns=[legacy])
        changed = True
    if changed:
        df.to_csv(path, index=False)


def compute_encoder_enrichment(
    metrics_matrix: np.ndarray,
    model_names: Iterable[str],
    selected_indices_dict: Mapping[str, Iterable[int]],
) -> tuple[Dict[str, float], pd.DataFrame]:
    """Compute advantage-subset enrichment over all repr-set centers.

    Lower metric is better. For each model m and K in {1, 3}, enrichment is:
        P(m in true TopK | point is in m's advantage subset)
        / P(m in true TopK | point is in full repr set)
    The summary is a selected-count weighted mean of per-model enrichment.
    """
    model_names = list(model_names)
    arr = np.asarray(metrics_matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"metrics_matrix must be 2D, got {arr.shape}")
    n_models, n_points = arr.shape
    if n_models != len(model_names):
        raise ValueError("metrics_matrix row count and model_names length mismatch")

    rank = np.argsort(arr, axis=0)
    rows = []
    eps = 1e-12
    for i, name in enumerate(model_names):
        raw_idx = selected_indices_dict.get(name, [])
        idx = np.asarray(list(raw_idx), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < n_points)]
        row: Dict[str, Any] = {"model_abbr": name, "selected_count": int(idx.size)}
        for k in (1, 3):
            kk = min(k, n_models)
            in_topk_global = np.any(rank[:kk, :] == i, axis=0)
            global_rate = float(np.mean(in_topk_global)) if n_points > 0 else np.nan
            if idx.size > 0:
                subset_rate = float(np.mean(in_topk_global[idx]))
            else:
                subset_rate = np.nan
            enrich = subset_rate / (global_rate + eps) if np.isfinite(global_rate) and global_rate > 0 else np.nan
            row[f"global_top{k}_rate"] = global_rate
            row[f"subset_top{k}_rate"] = subset_rate
            row[f"top{k}_enrichment"] = enrich
        rows.append(row)

    by_model = pd.DataFrame(rows)
    weights = pd.to_numeric(by_model["selected_count"], errors="coerce").fillna(0.0).to_numpy()
    summary: Dict[str, float] = {}
    for k, out_col in ((1, "ENC_TOP1_ENRICH"), (3, "ENC_TOP3_ENRICH")):
        vals = pd.to_numeric(by_model[f"top{k}_enrichment"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(vals) & (weights > 0)
        summary[out_col] = float(np.average(vals[mask], weights=weights[mask])) if np.any(mask) else float("nan")
    for k, out_col in ((1, "ENC_TOP1_SUBSET_RATE"), (3, "ENC_TOP3_SUBSET_RATE")):
        vals = pd.to_numeric(by_model[f"subset_top{k}_rate"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(vals) & (weights > 0)
        summary[out_col] = float(np.average(vals[mask], weights=weights[mask])) if np.any(mask) else float("nan")
    return summary, by_model


def save_encoder_enrichment_report(
    model_repr_path: str,
    metrics_matrix: np.ndarray,
    model_names: Iterable[str],
    selected_indices_dict: Mapping[str, Iterable[int]],
    extra: Mapping[str, Any] | None = None,
    args: Any | None = None,
) -> Dict[str, float]:
    summary, by_model = compute_encoder_enrichment(metrics_matrix, model_names, selected_indices_dict)
    summary_path, by_model_path = encoder_enrichment_paths(model_repr_path, args=args)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    context = _encoder_enrichment_context(model_repr_path, args=args)

    summary_row: Dict[str, Any] = dict(context)
    summary_row.update(dict(summary))
    if extra:
        summary_row.update(dict(extra))
    _upsert_csv_rows(summary_path, pd.DataFrame([summary_row]), ["row_key"])

    by_model_out = by_model.copy()
    by_model_context: Dict[str, Any] = dict(context)
    if extra:
        by_model_context.update(dict(extra))
    for key, value in by_model_context.items():
        by_model_out[key] = value
    _upsert_csv_rows(by_model_path, by_model_out, ["row_key", "model_abbr"])
    selected_total = int(pd.to_numeric(by_model.get("selected_count"), errors="coerce").fillna(0).sum())
    print(
        "[ENC_METRIC] "
        f"selected_total={selected_total}, "
        f"ENC_TOP1_ENRICH={summary.get('ENC_TOP1_ENRICH', np.nan):.6g}, "
        f"ENC_TOP3_ENRICH={summary.get('ENC_TOP3_ENRICH', np.nan):.6g}, "
        f"PWW1={summary.get('ENC_TOP1_SUBSET_RATE', np.nan):.6g}, "
        f"PWW3={summary.get('ENC_TOP3_SUBSET_RATE', np.nan):.6g}"
    )
    show_cols = [
        "model_abbr",
        "selected_count",
        "global_top1_rate",
        "subset_top1_rate",
        "top1_enrichment",
        "global_top3_rate",
        "subset_top3_rate",
        "top3_enrichment",
    ]
    table = by_model[[c for c in show_cols if c in by_model.columns]].copy()
    for col in table.columns:
        if col != "model_abbr":
            table[col] = pd.to_numeric(table[col], errors="coerce").round(4)
    print("[ENC_METRIC] by-model enrichment:")
    print(table.to_string(index=False))
    print(f"[ENC_METRIC] saved summary -> {summary_path}")
    print(f"[ENC_METRIC] saved by-model -> {by_model_path}")
    return summary


def load_encoder_enrichment_for_args(args: Any) -> Dict[str, float]:
    if str(getattr(args, "models", "")) != "TSRouter":
        return {}
    try:
        from utils.path_utils import get_repr_save_path

        _, _, model_repr_path, _ = get_repr_save_path(args)
    except Exception:
        return {}
    out = {
        "ENC_TOP1_ENRICH": np.nan,
        "ENC_TOP3_ENRICH": np.nan,
        "ENC_TOP1_SUBSET_RATE": np.nan,
        "ENC_TOP3_SUBSET_RATE": np.nan,
        "REGION_WEIGHTED_PURITY": np.nan,
        "REGION_DIAG_RANK": np.nan,
        "REGION_DELTA_RANK": np.nan,
    }
    context = _encoder_enrichment_context(model_repr_path, args=args)
    summary_path, _ = encoder_enrichment_paths(model_repr_path, args=args)
    legacy_paths = [
        _legacy_encoder_enrichment_paths(model_repr_path, args=args)[0],
        _legacy_encoder_enrichment_paths(model_repr_path)[0],
    ]
    if not os.path.exists(summary_path):
        summary_path = next((p for p in legacy_paths if os.path.exists(p)), summary_path)
    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            row = None
            if "row_key" in df.columns:
                matched = df[df["row_key"].astype(str).eq(str(context["row_key"]))].copy()
                if not matched.empty:
                    row = matched.iloc[-1]
            else:
                row = df.iloc[-1]
            if row is not None:
                _update_with_finite_metrics(
                    out,
                    {
                        "ENC_TOP1_ENRICH": row.get("ENC_TOP1_ENRICH", np.nan),
                        "ENC_TOP3_ENRICH": row.get("ENC_TOP3_ENRICH", np.nan),
                        "ENC_TOP1_SUBSET_RATE": row.get("ENC_TOP1_SUBSET_RATE", np.nan),
                        "ENC_TOP3_SUBSET_RATE": row.get("ENC_TOP3_SUBSET_RATE", np.nan),
                    },
                )
    rule = resolve_process_metrics_region_rule(args)
    competence_summary_path, _ = competence_region_paths(model_repr_path, args=args)
    if os.path.exists(competence_summary_path):
        try:
            competence_df = pd.read_csv(competence_summary_path)
        except Exception:
            competence_df = pd.DataFrame()
        if not competence_df.empty:
            matched = competence_df.copy()
            if "row_key" in matched.columns:
                matched = matched[matched["row_key"].astype(str).eq(str(context["row_key"]))]
            if "region_rule" in matched.columns:
                matched = matched[matched["region_rule"].astype(str).str.lower().eq(rule)]
            if not matched.empty:
                competence_row = matched.iloc[-1]
                # schema v1 originally named the rule-specific PWW values REGION_CA_*.
                # They use the same assignment and Top-K truth, so keep old reports
                # readable while exposing only the established PWW names.
                _update_with_finite_metrics(out, _competence_metrics_from_mapping(competence_row))
    if not all(
        np.isfinite(safe_float(out.get(column, np.nan)))
        for column in COMPETENCE_REGION_METRIC_COLUMNS
    ):
        _update_with_finite_metrics(
            out,
            _load_subset_competence_metrics(model_repr_path, args, rule),
        )
    if any(np.isfinite(safe_float(value)) for value in out.values()):
        return out
    return {}


def model_id_to_abbr(Model_sizes: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for family in Model_sizes.values():
        for info in family.values():
            out[int(info["id"])] = str(info["abbreviation"])
    return out


def _target_to_t_c(target: Any, channels: int, pred_len: int | None = None) -> np.ndarray:
    arr = np.asarray(target, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"Unsupported target shape: {arr.shape}")
    if pred_len is not None:
        if arr.shape[0] == pred_len:
            return arr
        if arr.shape[1] == pred_len:
            return arr.T
    if arr.shape[0] == channels and arr.shape[1] != channels:
        return arr.T
    if arr.shape[-1] == channels:
        return arr
    return arr.T


def _dataset_prediction_length(dataset: Any, first_quantile: np.ndarray) -> int:
    pred_len = getattr(dataset, "prediction_length", None)
    try:
        pred_len = int(pred_len)
    except Exception:
        pred_len = 0
    if pred_len > 0:
        return pred_len

    arr = np.asarray(first_quantile)
    if arr.ndim == 1:
        return int(arr.shape[0])
    if arr.ndim == 2:
        channels = int(getattr(dataset, "target_dim", 1) or 1)
        if arr.shape[0] == channels and arr.shape[1] != channels:
            return int(arr.shape[1])
        if arr.shape[1] == channels:
            return int(arr.shape[0])
        return int(max(arr.shape))
    return int(np.squeeze(arr).shape[0])


def _dataset_mase_lag(dataset: Any, history_len: int) -> int:
    lag = 1
    freq = getattr(dataset, "freq", None)
    if freq is not None:
        try:
            from gluonts.time_feature import get_seasonality

            lag = int(get_seasonality(str(freq)))
        except Exception:
            lag = 1
    lag = max(1, lag)
    if history_len <= lag:
        lag = 1
    return min(lag, max(1, history_len // 2)) if history_len > 1 else 1


def _entries_to_array(entries: Iterable[Any], channels: int, pred_len: int | None = None) -> np.ndarray:
    out = []
    for entry in entries:
        target = entry.get("target", entry) if isinstance(entry, dict) else entry
        out.append(_target_to_t_c(target, channels, pred_len))
    return np.stack(out, axis=0)


def _point_predictions(prediction_cache: Mapping[str, Any]) -> Dict[int, np.ndarray]:
    kind = prediction_cache.get("kind")
    data = prediction_cache.get("predictions", {})
    out: Dict[int, np.ndarray] = {}
    for model_id, arr in data.items():
        a = np.asarray(arr, dtype=np.float32)
        if kind == "distribution" or a.ndim == 4:
            a = np.median(a, axis=1)
        out[int(model_id)] = a
    return out


def _metric_col_name(rank_metric: str) -> str:
    return "sMAPE" if str(rank_metric).lower() == "smape" else str(rank_metric).upper()


def channel_orders_from_error_matrix(
    matrix: np.ndarray,
    model_ids: Iterable[int],
    channels: Iterable[int],
) -> Dict[int, list[int]]:
    arr = np.asarray(matrix, dtype=float)
    ids = np.asarray(list(model_ids), dtype=np.int64)
    chs = [int(ch) for ch in channels]
    out: Dict[int, list[int]] = {}
    for j, ch in enumerate(chs):
        values = arr[:, j]
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite per-channel metric values for channel={ch}")
        # stable sort keeps model id order only for exact ties, making tie behavior explicit.
        order_pos = np.argsort(values, kind="mergesort")
        out[int(ch)] = ids[order_pos].astype(int).tolist()
    return out


def save_channel_rank_orders(
    output_path: str | os.PathLike[str],
    dataset_name: str,
    rank_metric: str,
    orders: Mapping[int, Iterable[int]],
) -> None:
    os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
    rows = [
        {
            "dataset": dataset_name,
            "channel": int(ch),
            "rank_metric": str(rank_metric),
            "model_order": "[" + " ".join(str(int(x)) for x in order) + "]",
        }
        for ch, order in sorted(orders.items())
    ]
    new_df = pd.DataFrame(rows)
    if os.path.exists(output_path):
        try:
            old_df = pd.read_csv(output_path)
        except pd.errors.ParserError:
            old_df = pd.read_csv(output_path, engine="python", on_bad_lines="skip")
    else:
        old_df = pd.DataFrame(columns=["dataset", "channel", "rank_metric", "model_order"])
    if not old_df.empty and {"dataset", "rank_metric"}.issubset(set(old_df.columns)):
        keep = ~(
            old_df["dataset"].astype(str).eq(str(dataset_name))
            & old_df["rank_metric"].astype(str).str.lower().eq(str(rank_metric).lower())
        )
        old_df = old_df.loc[keep].copy()
    out = pd.concat([old_df, new_df], ignore_index=True)
    out.to_csv(output_path, index=False)


def _mase_by_channel(pred: np.ndarray, y_true: np.ndarray, past: np.ndarray) -> np.ndarray:
    n, _, c = y_true.shape
    out = np.full(c, np.nan, dtype=np.float64)
    for ch in range(c):
        num_sum = 0.0
        den_sum = 0.0
        for i in range(n):
            p = np.asarray(past[i, :, ch], dtype=np.float64)
            h0, h1 = p[:-1], p[1:]
            hist_mask = np.isfinite(h0) & np.isfinite(h1)
            diff = np.abs(h1[hist_mask] - h0[hist_mask])
            scale = float(np.mean(diff)) if diff.size else np.nan
            if not np.isfinite(scale) or scale <= 0:
                scale = 5e-2
            yp = np.asarray(pred[i, :, ch], dtype=np.float64)
            yt = np.asarray(y_true[i, :, ch], dtype=np.float64)
            obs_mask = np.isfinite(yp) & np.isfinite(yt)
            if not np.any(obs_mask):
                continue
            num_sum += float(np.sum(np.abs(yp[obs_mask] - yt[obs_mask])))
            den_sum += float(max(scale, 5e-2) * np.sum(obs_mask))
        out[ch] = float(num_sum / den_sum) if den_sum > 0 else np.nan
    return out


def _smape_by_channel(pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    out = np.full(y_true.shape[-1], np.nan, dtype=np.float64)
    for ch in range(y_true.shape[-1]):
        yp = pred[:, :, ch]
        yt = y_true[:, :, ch]
        mask = np.isfinite(yp) & np.isfinite(yt)
        if not np.any(mask):
            continue
        num = np.abs(yp[mask] - yt[mask])
        den = (np.abs(yp[mask]) + np.abs(yt[mask])) / 2.0
        out[ch] = float(np.mean(num / (den + 1e-8)))
    return out


def per_channel_results_path(model_output_dir: str | os.PathLike[str]) -> str:
    return str(Path(model_output_dir) / "per_channel_results.csv")


def per_window_results_path(model_output_dir: str | os.PathLike[str]) -> str:
    return str(Path(model_output_dir) / "per_window_results.csv")


def _forecast_quantile(fc: Any, q: float) -> np.ndarray:
    try:
        return np.asarray(fc.quantile(q), dtype=np.float32)
    except Exception:
        if hasattr(fc, "samples"):
            return np.quantile(np.asarray(fc.samples, dtype=np.float32), q, axis=0)
        raise


def _metric_result_value(res: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        if key not in res:
            continue
        value = res[key]
        arr = np.asarray(value, dtype=np.float64)
        if arr.size == 0:
            continue
        return float(arr.reshape(-1)[0])
    return float("nan")


def _entry_channel_offset(entry: Any, entry_idx: int, original_channels: int) -> int:
    item_id = ""
    if isinstance(entry, dict):
        item_id = str(entry.get("item_id", ""))
    m = re.search(r"_dim(\d+)$", item_id)
    if m:
        return int(m.group(1))
    if original_channels > 1:
        return int(entry_idx % original_channels)
    return 0


def _entry_item_id(entry: Any, entry_idx: int) -> str:
    if isinstance(entry, dict) and "item_id" in entry:
        return str(entry.get("item_id"))
    return str(entry_idx)


def _base_series_id(item_id: str) -> str:
    return re.sub(r"_dim\d+$", "", str(item_id))


def _safe_start_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def _read_metric_csv(csv_path: str | os.PathLike[str], dtype: dict[str, str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, dtype=dtype, low_memory=False)
    except pd.errors.EmptyDataError as e:
        raise FileNotFoundError(f"empty csv file {csv_path}") from e


def _canonical_id_series(values: pd.Series) -> pd.Series:
    out = values.astype(str).str.strip()
    out = out.replace({"nan": "", "NaN": "", "None": "", "<NA>": ""})
    return out.str.replace(r"^(-?\d+)\.0$", r"\1", regex=True)


def _canonical_time_series(values: pd.Series) -> pd.Series:
    out = values.astype(str).str.strip()
    out = out.replace({"nan": "", "NaN": "", "None": "", "<NA>": "", "NaT": ""})
    out = out.str.replace("T", " ", regex=False)
    out = out.str.replace(r" 00:00:00(\.0+)?([+-]00:00|Z)?$", "", regex=True)
    return out.fillna("")


def _canonical_window_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["series_id", "item_id"]:
        if col in out.columns:
            out[col] = _canonical_id_series(out[col])
    for col in ["forecast_start", "input_start"]:
        if col in out.columns:
            out[col] = _canonical_time_series(out[col])
    for col in ["channel", "window_id", "entry", "pred_len", "mase_lag"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out


def _dict_with_target(entry: Any, target: np.ndarray) -> Dict[str, Any]:
    if isinstance(entry, dict):
        out = dict(entry)
        out["target"] = np.asarray(target, dtype=np.float32)
        return out
    return {"target": np.asarray(target, dtype=np.float32)}


def _compute_per_channel_metric_rows_gluonts(
    forecast_list: list[Any],
    label_list: list[Any],
    input_list: list[Any],
    dataset: Any,
    dataset_name: str,
    model_name: str,
    pred_len: int,
    original_channels: int,
    quantile_levels: list[float],
) -> list[Dict[str, Any]]:
    from gluonts.ev.metrics import MASE, SMAPE, MeanWeightedSumQuantileLoss
    from gluonts.model.forecast import QuantileForecast
    from gluonts.time_feature import get_seasonality

    q_keys = [str(q) for q in quantile_levels]
    channel_payloads: Dict[int, Dict[str, Any]] = {}

    for entry_idx, (fc, label, inp) in enumerate(zip(forecast_list, label_list, input_list)):
        y_true_raw = label.get("target", label) if isinstance(label, dict) else label
        past_raw = inp.get("target", inp) if isinstance(inp, dict) else inp
        y_pred = _target_to_t_c(_forecast_quantile(fc, 0.5), channels=original_channels, pred_len=pred_len)
        y_true = _target_to_t_c(y_true_raw, channels=max(original_channels, y_pred.shape[1]), pred_len=pred_len)
        past = _target_to_t_c(past_raw, channels=max(original_channels, y_pred.shape[1]), pred_len=None)

        c_use = min(y_pred.shape[1], y_true.shape[1])
        if c_use <= 0:
            continue
        offset = _entry_channel_offset(inp if isinstance(inp, dict) else label, entry_idx, original_channels)
        forecast_start = label.get("start", None) if isinstance(label, dict) else None
        if forecast_start is None:
            forecast_start = getattr(fc, "start_date", None)

        quantile_arrays = [
            _target_to_t_c(_forecast_quantile(fc, q), channels=original_channels, pred_len=pred_len)
            for q in quantile_levels
        ]

        for local_c in range(c_use):
            global_c = offset + local_c if c_use == 1 and original_channels > 1 else local_c
            payload = channel_payloads.setdefault(
                int(global_c),
                {
                    "forecasts": [],
                    "labels": [],
                    "inputs": [],
                    "abs_y_sum": 0.0,
                    "valid_count": 0.0,
                    "mase_den": 0.0,
                },
            )
            q_stack = np.stack(
                [
                    np.asarray(q_arr[:, min(local_c, q_arr.shape[1] - 1)], dtype=np.float32)
                    for q_arr in quantile_arrays
                ],
                axis=0,
            )
            yt = np.asarray(y_true[:, local_c], dtype=np.float64)
            pc = min(local_c, past.shape[1] - 1)
            hist = np.asarray(past[:, pc], dtype=np.float64)
            obs_mask = np.isfinite(yt) & np.isfinite(np.asarray(y_pred[:, local_c], dtype=np.float64))
            lag = _dataset_mase_lag(dataset, int(hist.shape[0]))
            h0, h1 = hist[:-lag], hist[lag:]
            hist_mask = np.isfinite(h0) & np.isfinite(h1)
            diff = np.abs(h1[hist_mask] - h0[hist_mask])
            scale = float(np.mean(diff)) if diff.size else np.nan
            if not np.isfinite(scale) or scale <= 0:
                scale = 5e-2
            valid_count = float(np.sum(obs_mask))
            payload["abs_y_sum"] += float(np.sum(np.abs(yt[np.isfinite(yt)])))
            payload["valid_count"] += valid_count
            payload["mase_den"] += float(max(scale, 5e-2) * valid_count)
            payload["labels"].append(_dict_with_target(label, yt))
            payload["inputs"].append(_dict_with_target(inp, hist))
            payload["forecasts"].append(
                QuantileForecast(
                    forecast_arrays=q_stack,
                    forecast_keys=q_keys,
                    start_date=forecast_start,
                )
            )

    rows: list[Dict[str, Any]] = []
    metric_list = [MASE(), SMAPE(), MeanWeightedSumQuantileLoss(quantile_levels=quantile_levels)]
    seasonality = get_seasonality(dataset.freq)
    for ch in sorted(channel_payloads.keys()):
        payload = channel_payloads[ch]
        if not payload["forecasts"]:
            continue
        per_channel_test_data = _PerChannelTestData(
            input=payload["inputs"],
            label=payload["labels"],
        )
        res = evaluate_forecasts_fast(
            forecasts=payload["forecasts"],
            test_data=per_channel_test_data,
            metrics=metric_list,
            batch_size=1024,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=seasonality,
        )
        mase = _metric_result_value(res, "MASE[0.5]", "eval_metrics/MASE[0.5]", "MASE")
        smape = _metric_result_value(res, "sMAPE[0.5]", "eval_metrics/sMAPE[0.5]", "sMAPE")
        crps = _metric_result_value(
            res,
            "mean_weighted_sum_quantile_loss",
            "eval_metrics/mean_weighted_sum_quantile_loss",
            "MeanWeightedSumQuantileLoss",
        )
        mase_den = max(float(payload["mase_den"]), 1e-8)
        smape_den = max(float(payload["valid_count"]), 1.0)
        crps_den = max(float(payload["abs_y_sum"]), 1e-8)
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "channel": int(ch),
                "METRIC_IMPL": _PER_CHANNEL_METRIC_IMPL,
                "MASE": float(mase),
                "sMAPE": float(smape),
                "CRPS": float(crps),
                "MASE_NUM": float(mase * mase_den),
                "MASE_DEN": float(mase_den),
                "SMAPE_NUM": float(smape * smape_den),
                "SMAPE_DEN": float(smape_den),
                "CRPS_NUM": float(crps * crps_den),
                "CRPS_DEN": float(crps_den),
            }
        )
    return rows


def compute_per_channel_metric_rows(
    forecasts: Iterable[Any],
    dataset: Any,
    dataset_name: str,
    model_name: str,
    global_res: Mapping[str, Any] | None = None,
) -> list[Dict[str, Any]]:
    label_list = list(dataset.test_data.label)
    input_list = list(dataset.test_data.input)
    forecast_list = list(forecasts)
    if not forecast_list or not (len(label_list) == len(input_list) == len(forecast_list)):
        return []

    first_q = _forecast_quantile(forecast_list[0], 0.5)
    pred_len = _dataset_prediction_length(dataset, first_q)
    original_channels = int(getattr(dataset, "target_dim", 1) or 1)

    channel_values: Dict[int, Dict[str, float]] = {}
    quantile_levels = [0.1 * i for i in range(1, 10)]

    if original_channels == 1 and global_res is not None:
        mase = _metric_result_value(global_res, "MASE[0.5]", "eval_metrics/MASE[0.5]", "MASE")
        smape = _metric_result_value(global_res, "sMAPE[0.5]", "eval_metrics/sMAPE[0.5]", "sMAPE")
        crps = _metric_result_value(
            global_res,
            "mean_weighted_sum_quantile_loss",
            "eval_metrics/mean_weighted_sum_quantile_loss",
            "MeanWeightedSumQuantileLoss",
        )
        if np.isfinite([mase, smape, crps]).all():
            valid_count = 0.0
            abs_y_sum = 0.0
            for label in label_list:
                target = label.get("target", label) if isinstance(label, dict) else label
                target_arr = np.asarray(target, dtype=np.float64)
                valid_count += float(np.isfinite(target_arr).sum())
                abs_y_sum += float(np.nansum(np.abs(target_arr)))
            smape_den = max(valid_count, 1.0)
            crps_den = max(abs_y_sum, 1e-8)
            return [
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "channel": 0,
                    "METRIC_IMPL": _PER_CHANNEL_METRIC_IMPL,
                    "MASE": float(mase),
                    "sMAPE": float(smape),
                    "CRPS": float(crps),
                    "MASE_NUM": float(mase * smape_den),
                    "MASE_DEN": float(smape_den),
                    "SMAPE_NUM": float(smape * smape_den),
                    "SMAPE_DEN": float(smape_den),
                    "CRPS_NUM": float(crps * crps_den),
                    "CRPS_DEN": float(crps_den),
                }
            ]

    try:
        exact_rows = _compute_per_channel_metric_rows_gluonts(
            forecast_list=forecast_list,
            label_list=label_list,
            input_list=input_list,
            dataset=dataset,
            dataset_name=dataset_name,
            model_name=model_name,
            pred_len=pred_len,
            original_channels=original_channels,
            quantile_levels=quantile_levels,
        )
        if exact_rows:
            return exact_rows
    except Exception as e:
        print(f"per-channel GluonTS eval unavailable, fallback to local metrics: {e}")

    for entry_idx, (fc, label, inp) in enumerate(zip(forecast_list, label_list, input_list)):
        y_true_raw = label.get("target", label) if isinstance(label, dict) else label
        past_raw = inp.get("target", inp) if isinstance(inp, dict) else inp
        y_pred = _target_to_t_c(_forecast_quantile(fc, 0.5), channels=original_channels, pred_len=pred_len)
        y_true = _target_to_t_c(y_true_raw, channels=max(original_channels, y_pred.shape[1]), pred_len=pred_len)
        past = _target_to_t_c(past_raw, channels=max(original_channels, y_pred.shape[1]), pred_len=None)

        c_use = min(y_pred.shape[1], y_true.shape[1])
        if c_use <= 0:
            continue
        offset = _entry_channel_offset(inp if isinstance(inp, dict) else label, entry_idx, original_channels)
        for local_c in range(c_use):
            global_c = offset + local_c if c_use == 1 and original_channels > 1 else local_c
            store = channel_values.setdefault(
                global_c,
                {
                    "abs_err_sum": 0.0,
                    "mase_denom_sum": 0.0,
                    "smape_sum": 0.0,
                    "smape_n": 0.0,
                    "y_abs_sum": 0.0,
                    "q_loss_sum": 0.0,
                },
            )
            yp = np.asarray(y_pred[:, local_c], dtype=np.float64)
            yt = np.asarray(y_true[:, local_c], dtype=np.float64)
            pc = min(local_c, past.shape[1] - 1)
            hist = np.asarray(past[:, pc], dtype=np.float64)
            lag = _dataset_mase_lag(dataset, int(hist.shape[0]))
            h0, h1 = hist[:-lag], hist[lag:]
            hist_mask = np.isfinite(h0) & np.isfinite(h1)
            diff = np.abs(h1[hist_mask] - h0[hist_mask])

            obs_mask = np.isfinite(yp) & np.isfinite(yt)
            abs_err = np.abs(yp[obs_mask] - yt[obs_mask])
            store["abs_err_sum"] += float(np.sum(abs_err)) if abs_err.size else 0.0
            scale = float(np.mean(diff)) if diff.size else np.nan
            if not np.isfinite(scale) or scale <= 0:
                scale = 5e-2
            store["mase_denom_sum"] += float(max(scale, 5e-2) * abs_err.size)
            smape_terms = np.abs(yp[obs_mask] - yt[obs_mask]) / (
                (np.abs(yp[obs_mask]) + np.abs(yt[obs_mask])) / 2.0 + 1e-8
            )
            store["smape_sum"] += float(np.sum(smape_terms)) if smape_terms.size else 0.0
            store["smape_n"] += float(smape_terms.size)

            yt_valid = yt[np.isfinite(yt)]
            store["y_abs_sum"] += float(np.sum(np.abs(yt_valid))) if yt_valid.size else 0.0
            for q in quantile_levels:
                q_pred_full = _target_to_t_c(_forecast_quantile(fc, q), channels=original_channels, pred_len=pred_len)
                q_pred = np.asarray(q_pred_full[:, min(local_c, q_pred_full.shape[1] - 1)], dtype=np.float64)
                q_mask = np.isfinite(yt) & np.isfinite(q_pred)
                q_loss = (q - (yt[q_mask] < q_pred[q_mask])) * (yt[q_mask] - q_pred[q_mask])
                store["q_loss_sum"] += 2.0 * float(np.sum(q_loss)) if q_loss.size else 0.0

    rows = []
    for ch in sorted(channel_values.keys()):
        vals = channel_values[ch]
        mase_denom = max(vals["mase_denom_sum"], 5e-2)
        smape_denom = max(vals["smape_n"], 1.0)
        crps_denom = max(vals["y_abs_sum"] * len(quantile_levels), 1e-3)
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "channel": int(ch),
                "METRIC_IMPL": _PER_CHANNEL_FALLBACK_IMPL,
                "MASE": float(vals["abs_err_sum"] / mase_denom),
                "sMAPE": float(vals["smape_sum"] / smape_denom),
                "CRPS": float(vals["q_loss_sum"] / crps_denom),
                "MASE_NUM": float(vals["abs_err_sum"]),
                "MASE_DEN": float(mase_denom),
                "SMAPE_NUM": float(vals["smape_sum"]),
                "SMAPE_DEN": float(smape_denom),
                "CRPS_NUM": float(vals["q_loss_sum"]),
                "CRPS_DEN": float(crps_denom),
            }
        )
    return rows


def compute_per_window_metric_rows(
    forecasts: Iterable[Any],
    dataset: Any,
    dataset_name: str,
    model_name: str,
) -> list[Dict[str, Any]]:
    """Compute per test-window metrics with additive numerator/denominator stats.

    The rows intentionally use the same local stable denominators as the
    per-channel fallback path, so summing NUM/DEN over rows is an exact
    reconstruction of the corresponding grouped metric.
    """
    label_list = list(dataset.test_data.label)
    input_list = list(dataset.test_data.input)
    forecast_list = list(forecasts)
    if not forecast_list or not (len(label_list) == len(input_list) == len(forecast_list)):
        return []

    first_q = _forecast_quantile(forecast_list[0], 0.5)
    pred_len = _dataset_prediction_length(dataset, first_q)
    original_channels = int(getattr(dataset, "target_dim", 1) or 1)
    quantile_levels = [0.1 * i for i in range(1, 10)]
    seasonality = get_seasonality(dataset.freq)
    seasonal_errors = get_cached_seasonal_errors(
        dataset,
        inputs=input_list,
        seasonality=seasonality,
        mask_invalid_label=True,
    )

    rows: list[Dict[str, Any]] = []
    channel_window_counts: Dict[tuple[str, int], int] = {}

    for entry_idx, (fc, label, inp) in enumerate(zip(forecast_list, label_list, input_list)):
        y_true_raw = label.get("target", label) if isinstance(label, dict) else label
        past_raw = inp.get("target", inp) if isinstance(inp, dict) else inp
        y_pred = _target_to_t_c(_forecast_quantile(fc, 0.5), channels=original_channels, pred_len=pred_len)
        y_true = _target_to_t_c(y_true_raw, channels=max(original_channels, y_pred.shape[1]), pred_len=pred_len)
        past = _target_to_t_c(past_raw, channels=max(original_channels, y_pred.shape[1]), pred_len=None)

        c_use = min(y_pred.shape[1], y_true.shape[1])
        if c_use <= 0:
            continue

        item_id = _entry_item_id(inp if isinstance(inp, dict) else label, entry_idx)
        series_id = _base_series_id(item_id)
        input_start = _safe_start_text(inp.get("start") if isinstance(inp, dict) else None)
        forecast_start = _safe_start_text(label.get("start") if isinstance(label, dict) else None)
        if not forecast_start:
            forecast_start = _safe_start_text(getattr(fc, "start_date", None))
        offset = _entry_channel_offset(inp if isinstance(inp, dict) else label, entry_idx, original_channels)

        for local_c in range(c_use):
            global_c = offset + local_c if c_use == 1 and original_channels > 1 else local_c
            global_c = int(global_c)
            window_count_key = (series_id, global_c)
            channel_window_counts[window_count_key] = channel_window_counts.get(window_count_key, 0) + 1
            window_id = channel_window_counts[window_count_key] - 1

            yp = np.asarray(y_pred[:, local_c], dtype=np.float64)
            yt = np.asarray(y_true[:, local_c], dtype=np.float64)
            pc = min(local_c, past.shape[1] - 1)
            hist = np.asarray(past[:, pc], dtype=np.float64)

            obs_mask = np.isfinite(yp) & np.isfinite(yt)
            valid_count = float(np.sum(obs_mask))
            if valid_count <= 0:
                continue

            lag = _dataset_mase_lag(dataset, int(hist.shape[0]))
            scale_values = np.ma.asarray(seasonal_errors[entry_idx]).filled(np.nan).reshape(-1)
            scale = float(scale_values[min(pc, scale_values.size - 1)]) if scale_values.size else np.nan
            if not np.isfinite(scale) or scale <= 0:
                scale = 5e-2

            abs_err = np.abs(yp[obs_mask] - yt[obs_mask])
            mase_num = float(np.sum(abs_err))
            mase_den = float(max(scale, 5e-2) * valid_count)

            smape_terms = np.abs(yp[obs_mask] - yt[obs_mask]) / (
                (np.abs(yp[obs_mask]) + np.abs(yt[obs_mask])) / 2.0 + 1e-8
            )
            smape_num = float(np.sum(smape_terms)) if smape_terms.size else 0.0
            smape_den = max(float(smape_terms.size), 1.0)

            yt_valid = yt[np.isfinite(yt)]
            y_abs_sum = float(np.sum(np.abs(yt_valid))) if yt_valid.size else 0.0
            crps_num = 0.0
            for q in quantile_levels:
                q_pred_full = _target_to_t_c(_forecast_quantile(fc, q), channels=original_channels, pred_len=pred_len)
                q_pred = np.asarray(q_pred_full[:, min(local_c, q_pred_full.shape[1] - 1)], dtype=np.float64)
                q_mask = np.isfinite(yt) & np.isfinite(q_pred)
                q_loss = (q - (yt[q_mask] < q_pred[q_mask])) * (yt[q_mask] - q_pred[q_mask])
                crps_num += 2.0 * float(np.sum(q_loss)) if q_loss.size else 0.0
            crps_den = max(y_abs_sum * len(quantile_levels), 1e-3)

            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "entry": int(entry_idx),
                    "series_id": series_id,
                    "item_id": item_id,
                    "forecast_start": forecast_start,
                    "input_start": input_start,
                    "channel": global_c,
                    "window_id": int(window_id),
                    "pred_len": int(pred_len),
                    "mase_lag": int(lag),
                    "METRIC_IMPL": _PER_WINDOW_METRIC_IMPL,
                    "MASE": float(mase_num / max(mase_den, 5e-2)),
                    "sMAPE": float(smape_num / smape_den),
                    "CRPS": float(crps_num / crps_den),
                    "MASE_NUM": float(mase_num),
                    "MASE_DEN": float(max(mase_den, 5e-2)),
                    "SMAPE_NUM": float(smape_num),
                    "SMAPE_DEN": float(smape_den),
                    "CRPS_NUM": float(crps_num),
                    "CRPS_DEN": float(crps_den),
                }
            )
    return rows


def _save_metric_rows_common(
    csv_path: str,
    rows: list[Dict[str, Any]],
    key_cols: Mapping[str, Any],
    skip_saved: bool,
    label: str,
    required_key_cols: list[str],
) -> tuple[bool, str]:
    if not rows:
        return False, f"no {label} rows computed"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new_df = pd.DataFrame(rows)
    metric_cols = [c for c in ["MASE", "sMAPE", "CRPS"] if c in new_df.columns]
    if metric_cols:
        metric_values = new_df[metric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(metric_values)
        if not finite_mask.all():
            bad_count = int((~finite_mask.all(axis=1)).sum())
            print(f"{label} metrics contain NaN/Inf; skip saving invalid rows_count={bad_count}")
            return False, f"new rows contain NaN/Inf count={bad_count}"
        negative = metric_values < -1e-12
        if negative.any():
            bad_count = int(negative.any(axis=1).sum())
            preview_cols = [c for c in [*required_key_cols, *metric_cols] if c in new_df.columns]
            bad_preview = new_df.loc[negative.any(axis=1), preview_cols].head(5).to_dict("records")
            print(f"{label} metrics contain negative values; skip saving rows_count={bad_count}, preview={bad_preview}")
            return False, f"new rows contain negative metrics count={bad_count}"
    stat_cols = [c for c in ["MASE_NUM", "MASE_DEN", "SMAPE_NUM", "SMAPE_DEN", "CRPS_NUM", "CRPS_DEN"] if c in new_df.columns]
    if stat_cols:
        stat_values = new_df[stat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(stat_values).all():
            bad_count = int((~np.isfinite(stat_values).all(axis=1)).sum())
            print(f"{label} metric stats contain NaN/Inf; skip saving rows_count={bad_count}")
            return False, f"new rows contain NaN/Inf metric stats count={bad_count}"
        for col in ["MASE_DEN", "SMAPE_DEN", "CRPS_DEN"]:
            if col in new_df.columns:
                vals = pd.to_numeric(new_df[col], errors="coerce").to_numpy(dtype=float)
                if (vals <= 0).any():
                    return False, f"new rows contain non-positive {col} count={int((vals <= 0).sum())}"
    if os.path.exists(csv_path):
        old_df = _read_metric_csv(csv_path)
    else:
        old_df = pd.DataFrame(columns=list(new_df.columns))

    if skip_saved and not old_df.empty:
        mask = np.ones(len(old_df), dtype=bool)
        for col, value in key_cols.items():
            if col not in old_df.columns:
                mask = np.zeros(len(old_df), dtype=bool)
                break
            mask &= old_df[col].astype(str).eq(str(value)).to_numpy()
        if mask.any() and metric_cols:
            old_vals = old_df.loc[mask, metric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            has_all_columns = set(new_df.columns).issubset(set(old_df.columns))
            old_ok = old_vals.size > 0 and np.isfinite(old_vals).all()
            old_ok = old_ok and bool((old_vals >= -1e-12).all())
            if has_all_columns and old_ok:
                return True, "already complete"

    keep = old_df.copy()
    if not keep.empty:
        mask = np.ones(len(keep), dtype=bool)
        for col, value in key_cols.items():
            if col in keep.columns:
                mask &= keep[col].astype(str).eq(str(value)).to_numpy()
            else:
                mask &= False
        keep = keep.loc[~mask].copy()
    out = new_df.copy() if keep.empty else pd.concat([keep, new_df], ignore_index=True)
    print(f"👉 {label}TSRouter runtime message: {csv_path}")
    out.to_csv(csv_path, index=False)
    return True, "saved"


def save_per_channel_metric_rows(
    csv_path: str,
    rows: list[Dict[str, Any]],
    key_cols: Mapping[str, Any],
    skip_saved: bool = False,
) -> tuple[bool, str]:
    saved, msg = _save_metric_rows_common(
        csv_path=csv_path,
        rows=rows,
        key_cols=key_cols,
        skip_saved=skip_saved,
        label="per-channel",
        required_key_cols=["dataset", "model", "channel"],
    )
    if not saved:
        return saved, msg
    _PER_CHANNEL_ERROR_MATRIX_CACHE.clear()
    return saved, msg


def save_per_window_metric_rows(
    csv_path: str,
    rows: list[Dict[str, Any]],
    key_cols: Mapping[str, Any],
    skip_saved: bool = False,
) -> tuple[bool, str]:
    saved, msg = _save_metric_rows_common(
        csv_path=csv_path,
        rows=rows,
        key_cols=key_cols,
        skip_saved=skip_saved,
        label="per-window",
        required_key_cols=["dataset", "model", "series_id", "forecast_start", "channel", "window_id"],
    )
    if saved:
        _PER_WINDOW_ERROR_MATRIX_CACHE.clear()
    return saved, msg


def load_per_channel_error_matrix(
    model_sizes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    dataset_name: str,
    model_cl_name: str,
    rank_metric: str = "MASE",
    results_dir: str | os.PathLike[str] = TSFM_CSV_ROOT,
    require_complete: bool = True,
) -> tuple[np.ndarray, list[int], list[int]]:
    metric_col = _metric_col_name(rank_metric)
    model_sig = tuple(
        (str(family), str(size_name), int(info["id"]))
        for family, size_dict in model_sizes.items()
        for size_name, info in size_dict.items()
    )
    cache_key = (
        str(results_dir),
        str(model_cl_name),
        str(dataset_name),
        str(metric_col),
        bool(require_complete),
        model_sig,
    )
    cached = _PER_CHANNEL_ERROR_MATRIX_CACHE.get(cache_key)
    if cached is not None:
        matrix, model_ids, channels = cached
        return matrix.copy(), list(model_ids), list(channels)

    rows = []
    model_ids = []
    channels: list[int] | None = None
    expected_model_ids = []
    problems = []
    for family, size_dict in model_sizes.items():
        for size_name, info in size_dict.items():
            model_id = int(info["id"])
            expected_model_ids.append(model_id)
            model_label = _model_label(family, size_name, info)
            model_folder = f"{family}_{size_name}"
            csv_path = resolve_tsfm_result_path(
                results_dir,
                model_folder,
                model_cl_name,
                "per_channel_results.csv",
            )
            if not csv_path.exists():
                problems.append(f"{model_label}: missing file {csv_path}")
                continue
            try:
                df = _read_metric_csv(csv_path)
            except FileNotFoundError as e:
                problems.append(f"{model_label}: {e}")
                continue
            if metric_col not in df.columns:
                problems.append(f"{model_label}: missing metric column {metric_col} in {csv_path}")
                continue
            if "METRIC_IMPL" not in df.columns:
                problems.append(f"{model_label}: missing METRIC_IMPL in {csv_path}")
                continue
            sub = df[df["dataset"].astype(str) == str(dataset_name)].copy()
            if sub.empty:
                problems.append(f"{model_label}: missing dataset row {dataset_name}")
                continue
            if not sub["METRIC_IMPL"].astype(str).eq(_PER_CHANNEL_METRIC_IMPL).all():
                problems.append(f"{model_label}: outdated per-channel metric implementation")
                continue
            sub["channel"] = pd.to_numeric(sub["channel"], errors="coerce").astype("Int64")
            sub = sub.dropna(subset=["channel"]).sort_values("channel")
            vals = pd.to_numeric(sub[metric_col], errors="coerce").to_numpy(dtype=float)
            chs = [int(x) for x in sub["channel"].tolist()]
            if channels is None:
                channels = chs
            if chs != channels:
                problems.append(
                    f"{model_label}: channel mismatch expected={channels}, got={chs}"
                )
                continue
            bad_pos = np.where(~np.isfinite(vals))[0]
            if bad_pos.size > 0:
                bad_channels = [chs[int(i)] for i in bad_pos[:10]]
                suffix = "" if bad_pos.size <= 10 else f"...(+{bad_pos.size - 10})"
                problems.append(
                    f"{model_label}: non-finite {metric_col} at channels={bad_channels}{suffix}"
                )
                continue
            bad_pos = np.where(vals < -1e-12)[0]
            if bad_pos.size > 0:
                bad_channels = [chs[int(i)] for i in bad_pos[:10]]
                suffix = "" if bad_pos.size <= 10 else f"...(+{bad_pos.size - 10})"
                problems.append(
                    f"{model_label}: negative {metric_col} at channels={bad_channels}{suffix}"
                )
                continue
            rows.append(vals)
            model_ids.append(model_id)
    if not rows or channels is None:
        raise FileNotFoundError(
            f"per-channel metric cache missing for dataset={dataset_name}, metric={metric_col}, "
            f"model_cl_name={model_cl_name}; problems={problems[:20]}"
        )
    if require_complete:
        missing = sorted(set(expected_model_ids) - set(model_ids))
        if missing:
            missing_labels = []
            for family, size_dict in model_sizes.items():
                for size_name, info in size_dict.items():
                    if int(info["id"]) in missing:
                        missing_labels.append(_model_label(family, size_name, info))
            raise FileNotFoundError(
                f"per-channel metric cache incomplete for dataset={dataset_name}, metric={metric_col}, "
                f"model_cl_name={model_cl_name}, missing_models={missing_labels}, "
                f"problems={problems[:20]}"
            )
    matrix = np.stack(rows, axis=0)
    _PER_CHANNEL_ERROR_MATRIX_CACHE[cache_key] = (matrix.copy(), list(model_ids), list(channels))
    return matrix, model_ids, channels


def build_channel_rank_cache(
    model_sizes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    dataset_names: Iterable[str],
    model_cl_name: str,
    rank_metric: str = "MASE",
    output_path: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    rows = []
    for ds in dataset_names:
        try:
            matrix, model_ids, channels = load_per_channel_error_matrix(
                model_sizes, ds, model_cl_name=model_cl_name, rank_metric=rank_metric
            )
        except FileNotFoundError as e:
            print(f"TSRouter runtime message: {ds}, metric={rank_metric}: {e}")
            continue
        orders = channel_orders_from_error_matrix(matrix, model_ids, channels)
        for ch, order in sorted(orders.items()):
            rows.append(
                {
                    "dataset": ds,
                    "channel": int(ch),
                    "rank_metric": str(rank_metric),
                    "model_order": "[" + " ".join(str(int(x)) for x in order) + "]",
                }
            )
    out = pd.DataFrame(rows)
    if output_path is not None:
        os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
        out.to_csv(output_path, index=False)
    return out


def load_per_window_error_matrix(
    model_sizes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    dataset_name: str,
    model_cl_name: str,
    rank_metric: str = "MASE",
    results_dir: str | os.PathLike[str] = TSFM_CSV_ROOT,
    require_complete: bool = True,
) -> tuple[np.ndarray, list[int], pd.DataFrame]:
    metric_col = _metric_col_name(rank_metric)
    model_sig = tuple(
        (str(family), str(size_name), int(info["id"]))
        for family, size_dict in model_sizes.items()
        for size_name, info in size_dict.items()
    )
    cache_key = (
        str(results_dir),
        str(model_cl_name),
        str(dataset_name),
        str(metric_col),
        bool(require_complete),
        model_sig,
    )
    cached = _PER_WINDOW_ERROR_MATRIX_CACHE.get(cache_key)
    if cached is not None:
        matrix, model_ids, window_df = cached
        return matrix.copy(), list(model_ids), window_df.copy()

    key_cols = ["series_id", "channel", "window_id"]
    rows = []
    model_ids = []
    window_df: pd.DataFrame | None = None
    expected_model_ids = []
    problems = []
    for family, size_dict in model_sizes.items():
        for size_name, info in size_dict.items():
            model_id = int(info["id"])
            expected_model_ids.append(model_id)
            model_label = _model_label(family, size_name, info)
            model_folder = f"{family}_{size_name}"
            csv_path = resolve_tsfm_result_path(
                results_dir,
                model_folder,
                model_cl_name,
                "per_window_results.csv",
            )
            if not csv_path.exists():
                problems.append(f"{model_label}: missing file {csv_path}")
                continue
            try:
                df = _read_metric_csv(
                    csv_path,
                    dtype={
                        "series_id": "string",
                        "item_id": "string",
                        "forecast_start": "string",
                        "input_start": "string",
                    },
                )
            except FileNotFoundError as e:
                problems.append(f"{model_label}: {e}")
                continue
            meta_cols = ["entry", "item_id", "forecast_start", "input_start", "pred_len", "mase_lag"]
            missing_cols = [c for c in [*key_cols, *meta_cols, "dataset", "model", "METRIC_IMPL", metric_col] if c not in df.columns]
            if missing_cols:
                problems.append(f"{model_label}: missing columns {missing_cols} in {csv_path}")
                continue
            sub = df[df["dataset"].astype(str) == str(dataset_name)].copy()
            if sub.empty:
                problems.append(f"{model_label}: missing dataset row {dataset_name}")
                continue
            if not sub["METRIC_IMPL"].astype(str).eq(_PER_WINDOW_METRIC_IMPL).all():
                problems.append(f"{model_label}: outdated per-window metric implementation")
                continue
            sub = _canonical_window_key_columns(sub)
            sub = sub.dropna(subset=["channel", "window_id"]).copy()
            sub = sub.sort_values(key_cols, kind="mergesort")
            duplicate_keys = sub.duplicated(key_cols, keep=False)
            if bool(duplicate_keys.any()):
                problems.append(
                    f"{model_label}: duplicate window keys for dataset={dataset_name}, "
                    f"rows={int(duplicate_keys.sum())}"
                )
                continue
            vals = pd.to_numeric(sub[metric_col], errors="coerce").to_numpy(dtype=float)
            keys = sub[key_cols].astype(str).reset_index(drop=True)
            if window_df is None:
                window_df = sub[key_cols + meta_cols].reset_index(drop=True)
            else:
                prev_keys = window_df[key_cols].astype(str).reset_index(drop=True)
                if not keys.equals(prev_keys):
                    problems.append(
                        f"{model_label}: window key mismatch for dataset={dataset_name}, "
                        f"expected_n={len(prev_keys)}, got_n={len(keys)}"
                    )
                    continue
            bad_pos = np.where(~np.isfinite(vals))[0]
            if bad_pos.size > 0:
                problems.append(f"{model_label}: non-finite {metric_col} rows={int(bad_pos.size)}")
                continue
            bad_pos = np.where(vals < -1e-12)[0]
            if bad_pos.size > 0:
                problems.append(f"{model_label}: negative {metric_col} rows={int(bad_pos.size)}")
                continue
            rows.append(vals)
            model_ids.append(model_id)
    if not rows or window_df is None:
        raise FileNotFoundError(
            f"per-window metric cache missing for dataset={dataset_name}, metric={metric_col}, "
            f"model_cl_name={model_cl_name}; problems={problems[:20]}"
        )
    if require_complete:
        missing = sorted(set(expected_model_ids) - set(model_ids))
        if missing:
            missing_labels = []
            for family, size_dict in model_sizes.items():
                for size_name, info in size_dict.items():
                    if int(info["id"]) in missing:
                        missing_labels.append(_model_label(family, size_name, info))
            raise FileNotFoundError(
                f"per-window metric cache incomplete for dataset={dataset_name}, metric={metric_col}, "
                f"model_cl_name={model_cl_name}, missing_models={missing_labels}, "
                f"problems={problems[:20]}"
            )
    matrix = np.stack(rows, axis=0)
    _PER_WINDOW_ERROR_MATRIX_CACHE[cache_key] = (matrix.copy(), list(model_ids), window_df.copy())
    return matrix, model_ids, window_df


def real_window_rank_cache_path(
    current_zoo_num: int,
    zoo_total_num: int,
    rank_metric: str,
    model_cl_name: str,
) -> str:
    return str(
        BASELINE_CSV_ROOT
        / "selectors"
        / "Real_Window_Select"
        / f"zoo{current_zoo_num}-{zoo_total_num}_real_window_{rank_metric}_{model_cl_name}_rank.csv"
    )


def build_window_rank_cache(
    model_sizes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    dataset_names: Iterable[str],
    model_cl_name: str,
    rank_metric: str = "MASE",
    output_path: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    rows = []
    dataset_list = list(dataset_names)
    total = len(dataset_list)
    for idx, ds in enumerate(dataset_list, start=1):
        print(f"[Real_Window_Select] {idx}/{total} load dataset={ds}, metric={rank_metric}", flush=True)
        try:
            matrix, model_ids, window_df = load_per_window_error_matrix(
                model_sizes, ds, model_cl_name=model_cl_name, rank_metric=rank_metric
            )
        except FileNotFoundError as e:
            print(f"⚠️ [Real_Window_Select] skip rank cache dataset={ds}, metric={rank_metric}: {e}")
            continue
        print(
            f"[Real_Window_Select] {idx}/{total} rank dataset={ds}, "
            f"windows={matrix.shape[1]}, models={matrix.shape[0]}",
            flush=True,
        )
        order_pos = np.argsort(matrix, axis=0, kind="mergesort")
        for col_idx in range(order_pos.shape[1]):
            meta = window_df.iloc[col_idx].to_dict()
            order = [int(model_ids[int(pos)]) for pos in order_pos[:, col_idx]]
            rows.append(
                {
                    "dataset": ds,
                    "series_id": meta.get("series_id", ""),
                    "forecast_start": meta.get("forecast_start", ""),
                    "channel": int(meta.get("channel", 0)),
                    "window_id": int(meta.get("window_id", col_idx)),
                    "rank_metric": str(rank_metric),
                    "model_order": "[" + " ".join(str(int(x)) for x in order) + "]",
                }
            )
    out = pd.DataFrame(rows)
    if output_path is not None:
        os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
        out.to_csv(output_path, index=False)
    return out


def parse_order_string(value: Any) -> list[int]:
    if value is None:
        return []
    try:
        if isinstance(value, float) and not np.isfinite(value):
            return []
    except Exception:
        pass
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(x) for x in value]
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return [int(x) for x in s.replace(",", " ").split() if x.strip()]


def normalize_rank_truth_cl(value: Any, default: str = "cl512") -> str:
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return str(default)
    match = re.search(r"cl[_-]?(\d+)", text)
    if match:
        return f"cl{int(match.group(1))}"
    match = re.search(r"\d+", text)
    if match:
        return f"cl{int(match.group(0))}"
    return str(default)


def resolve_rank_truth_cl(args: Any | None, default: str = "cl512") -> str:
    if args is None:
        return str(default)
    for attr in ("rank_truth_cl", "resolved_eval_cl"):
        raw = getattr(args, attr, None)
        if raw not in (None, ""):
            return normalize_rank_truth_cl(raw, default=default)
    for attr in ("repr_input_dim", "search_context_len", "context_len"):
        raw = getattr(args, attr, None)
        try:
            val = int(raw)
        except Exception:
            continue
        if val > 0:
            return f"cl{val}"
    for attr in ("model_cl_name", "TSFM_results_dir"):
        raw = getattr(args, attr, None)
        if raw not in (None, ""):
            return normalize_rank_truth_cl(raw, default=default)
    return str(default)


def _rank_order_for_channel(arr: np.ndarray, sample_idx: int, channel_idx: int) -> list[int]:
    if arr.ndim != 3:
        return []
    if not (0 <= sample_idx < arr.shape[0]) or not (0 <= channel_idx < arr.shape[2]):
        return []
    return [
        int(x) for x in arr[int(sample_idx), :, int(channel_idx)].reshape(-1).tolist()
        if int(x) >= 0
    ]


def compute_window_channel_task_process_metrics(
    *,
    task_sample_rankings: Any = None,
    channel_orders: Mapping[int, list[int]] | None = None,
    task_order: Iterable[int] | None = None,
    selected_model_list_2d: Any = None,
    model_order: Any = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    channel_orders = channel_orders or {}
    task_order_list = [int(x) for x in list(task_order or [])]
    task_top1 = int(task_order_list[0]) if task_order_list else None

    if task_sample_rankings is not None:
        arr = np.asarray(task_sample_rankings, dtype=np.int64)
        if arr.ndim == 3 and arr.shape[0] > 0 and arr.shape[1] > 0 and arr.shape[2] > 0:
            wc_top1: list[bool] = []
            wc_top3: list[bool] = []
            wr_top1: list[bool] = []
            wr_top3: list[bool] = []
            for sample_idx in range(arr.shape[0]):
                for channel_idx in range(arr.shape[2]):
                    pred_order = _rank_order_for_channel(arr, sample_idx, channel_idx)
                    if not pred_order:
                        continue
                    ch_order = [int(x) for x in channel_orders.get(int(channel_idx), [])]
                    if ch_order:
                        ch_top1 = int(ch_order[0])
                        pred_top1 = int(pred_order[0])
                        wc_top1.append(pred_top1 == ch_top1)
                        wc_top3.append(pred_top1 in set(ch_order[:3]))
                    if task_top1 is not None:
                        pred_top1 = int(pred_order[0])
                        wr_top1.append(pred_top1 == task_top1)
                        wr_top3.append(pred_top1 in set(task_order_list[:3]))
            if wc_top1:
                metrics["TEST_WINDOW_CHANNEL_TOP1_ACC"] = float(np.mean(wc_top1))
                metrics["TEST_WINDOW_CHANNEL_TOP3_HIT"] = float(np.mean(wc_top3))
                metrics["TEST_WINDOW_CHANNEL_EVAL_N"] = float(len(wc_top1))
            if wr_top1:
                metrics["TEST_WINDOW_TASK_TOP1_ACC"] = float(np.mean(wr_top1))
                metrics["TEST_WINDOW_TASK_TOP3_HIT"] = float(np.mean(wr_top3))
                metrics["TEST_WINDOW_TASK_EVAL_N"] = float(len(wr_top1))

    cr_pred_orders: list[list[int]] = []
    if selected_model_list_2d is not None:
        arr2d = np.asarray(selected_model_list_2d, dtype=np.int64)
        if arr2d.ndim == 2 and arr2d.shape[0] > 0 and arr2d.shape[1] > 0:
            for channel_idx in range(arr2d.shape[1]):
                pred_order = [
                    int(x) for x in arr2d[:, int(channel_idx)].reshape(-1).tolist()
                    if int(x) >= 0
                ]
                if pred_order:
                    cr_pred_orders.append(pred_order)
    if not cr_pred_orders and model_order is not None:
        pred_order = [int(x) for x in parse_order_string(model_order) if int(x) >= 0]
        if pred_order:
            channel_count = len(channel_orders) if channel_orders else 1
            cr_pred_orders = [pred_order for _ in range(max(1, int(channel_count)))]
    if cr_pred_orders and task_top1 is not None:
        cr_top1 = [int(order[0]) == task_top1 for order in cr_pred_orders if order]
        cr_top3 = [int(order[0]) in set(task_order_list[:3]) for order in cr_pred_orders if order]
        if cr_top1:
            metrics["TEST_CHANNEL_TASK_TOP1_ACC"] = float(np.mean(cr_top1))
            metrics["TEST_CHANNEL_TASK_TOP3_HIT"] = float(np.mean(cr_top3))
            metrics["TEST_CHANNEL_TASK_EVAL_N"] = float(len(cr_top1))
    return metrics


def load_channel_rank_orders(
    csv_path: str | os.PathLike[str],
    dataset_name: str,
    rank_metric: str = "MASE",
) -> Dict[int, list[int]]:
    if not os.path.exists(csv_path):
        return {}
    try:
        df = _read_metric_csv(csv_path)
    except pd.errors.ParserError:
        try:
            df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
        except Exception:
            return {}
    except FileNotFoundError:
        return {}
    if df.empty or "dataset" not in df.columns or "model_order" not in df.columns:
        return {}
    sub = df[df["dataset"].astype(str) == str(dataset_name)].copy()
    if "rank_metric" in sub.columns:
        sub = sub[sub["rank_metric"].astype(str).str.lower() == str(rank_metric).lower()]
    out: Dict[int, list[int]] = {}
    for _, row in sub.iterrows():
        try:
            out[int(row["channel"])] = parse_order_string(row["model_order"])
        except Exception:
            continue
    return out


def real_channel_rank_cache_path(
    current_zoo_num: int,
    zoo_total_num: int,
    rank_metric: str,
    model_cl_name: str,
) -> str:
    return str(
        BASELINE_CSV_ROOT
        / "selectors"
        / "Real_Channel_Select"
        / f"zoo{current_zoo_num}-{zoo_total_num}_real_channel_{rank_metric}_{model_cl_name}_rank.csv"
    )


def compute_single_series_recommendation_metrics_from_orders(
    channel_orders: Mapping[int, list[int]],
    selected_model_list_2d: Any = None,
    selected_models_per_channel: Any = None,
    model_order: Iterable[int] | None = None,
) -> Dict[str, float]:
    if not channel_orders:
        return {}
    channels = sorted(channel_orders.keys())
    pred_top3_by_channel: list[list[int]] | None = None
    if selected_model_list_2d is not None:
        arr2d = np.asarray(selected_model_list_2d, dtype=np.int64)
        if arr2d.ndim == 2:
            pred_top3_by_channel = []
            for cpos in range(len(channels)):
                if cpos < arr2d.shape[1]:
                    vals = [int(x) for x in arr2d[:3, cpos].tolist() if int(x) >= 0]
                else:
                    vals = []
                pred_top3_by_channel.append(vals)
    if selected_models_per_channel is not None:
        pred_top1 = np.asarray(selected_models_per_channel, dtype=np.int64).reshape(-1)
    elif selected_model_list_2d is not None:
        pred_top1 = np.asarray(selected_model_list_2d, dtype=np.int64)[0].reshape(-1)
    elif model_order is not None and len(list(model_order)) > 0:
        model_order_list = [int(x) for x in list(model_order)]
        pred_top1 = np.full(len(channels), int(model_order_list[0]), dtype=np.int64)
        pred_top3_by_channel = [model_order_list[:3] for _ in channels]
    else:
        return {}
    if pred_top1.size < len(channels):
        pred_top1 = np.resize(pred_top1, len(channels))
    pred_top1 = pred_top1[: len(channels)]
    if pred_top3_by_channel is None:
        pred_top3_by_channel = [[int(pred_top1[pos])] for pos in range(len(channels))]
    top1_hits = []
    top3_hits = []
    for pos, ch in enumerate(channels):
        order = [int(x) for x in channel_orders[ch]]
        if not order:
            continue
        pred = int(pred_top1[pos])
        top1_hits.append(pred == order[0])
        top3_hits.append(pred in set(order[:3]))
    if not top1_hits:
        return {}
    return {
        "SINGLE_TOP1_ACC": float(np.mean(top1_hits)),
        "SINGLE_TOP3_HIT": float(np.mean(top3_hits)),
        "SINGLE_CHANNELS_EVAL": float(len(top1_hits)),
    }


def compute_single_series_recommendation_metrics(
    prediction_cache: Mapping[str, Any] | None,
    label_entries: Iterable[Any] | None,
    input_entries: Iterable[Any] | None,
    selected_model_list_2d: Any = None,
    selected_models_per_channel: Any = None,
    model_order: Iterable[int] | None = None,
    channels: int | None = None,
    rank_metric: str = "MASE",
) -> Dict[str, float]:
    if prediction_cache is None or label_entries is None or input_entries is None:
        return {}

    preds = _point_predictions(prediction_cache)
    if not preds:
        return {}
    first = next(iter(preds.values()))
    if first.ndim != 3:
        return {}
    n, pred_len, inferred_c = first.shape
    channels = int(channels or inferred_c)

    y_true = _entries_to_array(label_entries, channels=channels, pred_len=pred_len)
    past = _entries_to_array(input_entries, channels=channels, pred_len=None)
    if y_true.shape[0] != n:
        n_use = min(y_true.shape[0], n)
        y_true = y_true[:n_use]
        past = past[:n_use]
        preds = {k: v[:n_use] for k, v in preds.items()}

    metric_by_model = {}
    metric_name = str(rank_metric).upper()
    for model_id, pred in preds.items():
        pred = np.asarray(pred, dtype=np.float32)
        if pred.shape[-1] != channels:
            continue
        if metric_name == "SMAPE":
            metric_by_model[model_id] = _smape_by_channel(pred, y_true)
        else:
            metric_by_model[model_id] = _mase_by_channel(pred, y_true, past)
    if not metric_by_model:
        return {}

    model_ids = sorted(metric_by_model.keys())
    metric_matrix = np.stack([metric_by_model[mid] for mid in model_ids], axis=0)
    true_order_pos = np.argsort(metric_matrix, axis=0)
    true_order_ids = np.asarray(model_ids, dtype=np.int64)[true_order_pos]

    if selected_models_per_channel is not None:
        pred_top1 = np.asarray(selected_models_per_channel, dtype=np.int64).reshape(-1)
        pred_top3_by_channel = None
    elif selected_model_list_2d is not None:
        pred_arr_2d = np.asarray(selected_model_list_2d, dtype=np.int64)
        pred_top1 = pred_arr_2d[0].reshape(-1)
        pred_top3_by_channel = [
            [int(x) for x in pred_arr_2d[:3, c].tolist() if int(x) >= 0]
            for c in range(min(channels, pred_arr_2d.shape[1]))
        ]
    elif model_order is not None and len(list(model_order)) > 0:
        model_order_list = [int(x) for x in list(model_order)]
        pred_top1 = np.full(channels, int(model_order_list[0]), dtype=np.int64)
        pred_top3_by_channel = [model_order_list[:3] for _ in range(channels)]
    else:
        return {}

    if pred_top1.size < channels:
        pred_top1 = np.resize(pred_top1, channels)
    pred_top1 = pred_top1[:channels]
    if pred_top3_by_channel is None:
        pred_top3_by_channel = [[int(pred_top1[c])] for c in range(channels)]
    elif len(pred_top3_by_channel) < channels:
        pred_top3_by_channel = pred_top3_by_channel + [[int(pred_top1[c])] for c in range(len(pred_top3_by_channel), channels)]

    top1 = true_order_ids[0, :]
    top1_acc = np.mean(pred_top1 == top1)
    top3_hit = np.mean([int(pred_top1[c]) in set(true_order_ids[:3, c].tolist()) for c in range(channels)])
    return {
        "SINGLE_TOP1_ACC": float(top1_acc),
        "SINGLE_TOP3_HIT": float(top3_hit),
        "SINGLE_CHANNELS_EVAL": float(channels),
    }


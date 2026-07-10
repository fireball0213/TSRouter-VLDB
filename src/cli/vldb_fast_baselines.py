from __future__ import annotations

import ast
import copy
import hashlib
import json
import pickle
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tabulate import tabulate

from config.dataset_config import ALL_DATASETS, ALL_Fast_DATASETS
from config.model_zoo_config import All_sorted_model_names, Model_abbrev_map, Model_zoo_details
from utils.path_utils import (
    auto_cl_tsfm_comparison_dir,
    build_repr_eval_pool_forward_stem,
    build_repr_forward_stem,
    get_auto_cl_mode,
    get_auto_cl_profile_by_name,
    get_auto_cl_profiles,
    get_fixed_cl_profile,
    get_zoo_repr_prefix,
    get_repr_save_path,
    get_tsrouter_repr_forward_dir,
    get_tsrouter_selector_result_dir,
    get_tsrouter_selector_stage_result_dir,
    materialize_compatible_tsrouter_result,
    normalize_advanced_baseline_train_scope,
    normalize_route_family_mode,
    normalize_repr_scale_protocol,
    normalize_encoder_variant_args,
    normalize_auto_cl_args,
    resolve_tsfm_csv_path,
    tsfm_csv_glob_display,
)
from utils.project_paths import BASELINE_CSV_ROOT, CACHE_ROOT, TSFM_CSV_ROOT, TSROUTER_CSV_ROOT, TSROUTER_VLDB_TABLE_ROOT
from utils.tsrouter_metrics import TSROUTER_CORE_METRIC_COLUMNS
from selector.TSRouter_Select.task_probe_select import (
    task_probe_sample_timing_csv_candidates,
    task_probe_select_root,
    task_probe_select_cache_path,
    task_probe_select_selection_for_stage,
)


FAST_BASELINE_RANDOM_SEEDS = [2025, 2026, 2027, 2028, 2029]
VLDB_STAGE_SUMMARY_CACHE_VERSION = 6
VLDB_STAGE_SUMMARY_CACHE_ROOT = CACHE_ROOT / "vldb_results_stage_summaries"
VLDB_STAGE_METHOD_SUMMARY_CACHE_VERSION = 6
VLDB_STAGE_METHOD_SUMMARY_CACHE_ROOT = CACHE_ROOT / "vldb_results_stage_method_summaries"
VLDB_RESULTS_FIGURE_ROOT = TSROUTER_VLDB_TABLE_ROOT.parent / "figures"
VLDB_RESULTS_FIGURE_FONT_SERIF = ["Times New Roman", "Times", "DejaVu Serif"]
FIGURE_MASE_XLIM = (1.48, 1.97)
FIGURE1_LEGEND_FONTSIZE = 16.0
FIGURE1_E2E_P95_YLIM = (0.0, 580.0)
FIGURE1_E2E_P95_Y_BREAK = (300.0, 500.0)
FIGURE1_Y_BREAK_GAP = 32.0
FIGURE1_LEGEND_ORDER = [
    "TSFM",
    "Task-probe",
    "AutoForecast",
    "AutoXPCR",
    "SimpleTS",
    "TSRouter",
    "TSRouter-fast",
]
FIGURE1_LEGEND_LOC = "upper right"
FIGURE1_REFERENCE_LINESTYLE = "--"
FIGURE1_REFERENCE_LINEWIDTH = 1.4
FIGURE1_REFERENCE_LINE_ALPHA = 0.78
FIGURE1_INSERT_COST_COL = "insert_total_mean_stage4_last_s"
FIGURE1_INSERT_COST_VALUE_COL = "Total_mean_stage4_last(s)"
FIGURE1_INSERT_COST_COMPONENTS = ("IncomingProfile", "Retrain_total", "IndexRefresh")
FIGURE1_INSERT_REFRESH_COMPONENTS = ("Retrain_total", "IndexRefresh")
FIGURE1_INSERT_COST_MARKER_AREA_RANGE = (145.0, 620.0)
FIGURE1_TSFM_LABEL_QUADRANTS = {
    "Kai.50": "upper_right",
    "Moi.S": "upper_right",
    "PTS.FM": "upper_left",
    "Moi2.S": "upper_left",
    "TFM.25": "lower_right",
    "Flo.r1": "lower_right",
    "Chr.2": "lower_left",
}
FIGURE2_MASE_YLIM = (1.45, 2.0)
FIGURE2_OVERLAP_ATOL = 0.002
FIGURE2_OVERLAP_OFFSET_POINTS = 1.0
FIGURE2_TSROUTER_DISPLAY_OFFSET_POINTS = 0.0
FIGURE2_FAMILY_TSROUTER_DISPLAY_OFFSET_POINTS = -4.8
FIGURE2_FAMILY_TSROUTER_RAW_TAIL_N = 3
FIGURE2_BASE_MARKERSIZE = 7.0
FIGURE2_STATIC_MARKERSIZE = 5.8
FIGURE2_TSROUTER_MARKERSIZE = 12.0
FIGURE_TASK_PROBE_COLOR = "#cc79a7"
FIGURE_TASK_PROBE_MARKER = "o"
FIGURE2_SPECIAL_MARKERS = {
    "TSRouter": "*",
    "AutoForecast": "D",
    "AutoXPCR": "X",
    "SimpleTS": "P",
    "Task-probe": FIGURE_TASK_PROBE_MARKER,
}
FIGURE4_GROWTH_STAGES = (5, 10, 15, 20, 40, 80)
FIGURE4_STAGE_X_POSITIONS = {
    5: 0.0,
    10: 1.0,
    15: 2.0,
    20: 3.0,
    40: 4.25,
    80: 5.75,
}
FIGURE4_OBSERVED_STAGES = (5, 10, 15, 20)
FIGURE4_INSERT_STAGE_WINDOWS = {
    5: (1, 5),
    10: (6, 10),
    15: (11, 15),
    20: (16, 20),
}
FIGURE4_INSERT_UPPER_ENVELOPE_LOG_GROWTH = 0.12
FIGURE4_FAST_INSERT_DISPLAY_SCALE = 0.88
FIGURE1_SELECTOR_STYLES = {
    "TSRouter": {"color": "#0057b8", "marker": "*", "size": 300, "edgecolor": "black"},
    "TSRouter-fast": {"color": "#d1495b", "marker": "*", "size": 300, "edgecolor": "black"},
    "AutoForecast": {"color": "#9467bd", "marker": "D", "size": 205, "edgecolor": "black"},
    "AutoXPCR": {"color": "#17a2a4", "marker": "X", "size": 215, "edgecolor": "black"},
    "SimpleTS": {"color": "#2ca02c", "marker": "P", "size": 205, "edgecolor": "black"},
    "Task-probe": {
        "color": FIGURE_TASK_PROBE_COLOR,
        "marker": FIGURE_TASK_PROBE_MARKER,
        "size": 150,
        "edgecolor": FIGURE_TASK_PROBE_COLOR,
    },
}
FIGURE2_FAMILY_REPRESENTATIVES = [
    ("Moirai", "Moi.L"),
    ("Chronos", "Chr.bM"),
    ("Kairos", "Kai.50"),
    ("Toto2", "Toto2.B"),
]
VLDB_RESULTS_AUTOFORECAST_LEARNER = "GBDT"


# The single paper-facing TSRouter configuration used by --vldb_results.
# Keep every value as a one-item list so the shape matches the search grids in
# check_selector.py while making it explicit that this is one main-method row.
VLDB_RESULTS_MAIN_PARAM_GRID = {
    "repr_encoder": ["StatsRandomFourier"],
    "repr_input_dim": [512],
    "repr_output_dim": [256],
    "repr_sub_pred_len": [480],
    "zoo_repr_set": ["c-e-n-h-w-s"],
    # "zoo_repr_set": ['all_sources'],
    "repr_size": [3000],
    # "repr_v": [5],
    "repr_v": [4],
    "base_metrics": ["C"],
    # "repr_weight_ratio": [0.0],
    "repr_weight_ratio": [0.5],
    "sample_repr_num": [20],
    "repr_data_seed": [2029],
    "repr_encoder_seed": [2025],
    "forward_seed": [2025],
    "search_seed": [2025],
    "repr_sample_qc_mode": ["strict"],
    "repr_scale_protocol": ["standard"],
    "task_sample_version": [2],
    # "repr_anchor_window_sample_strategy": ["even"],
    "repr_anchor_window_sample_strategy": ["first"],
    # "task_window_sample_strategy": ["random"],
    "task_window_sample_strategy": ["even"],
    "sample_repr_ratio": [0.0],
    "task_rank_top3_instability_threshold": [-1.0],
    # "task_rank_top3_instability_threshold": [3],
    "task_channel_fuse_limit": ["all"],
    "route_family_mode": ["default"],
    "sample_mode": ["cluster_nearest"],
    "model_repr_mode": ["all"],
    "subset_top_k": [0],
    "subset_perf_scale": [1.0],
    "repr_v5_nearest_k": [10],
    "repr_v5_distance_power": [1.0],
    "rank_decay_coef": [1.0],
    "ensemble_size": [1],
    "ensemble_agg": ["median"],
    "restrict_top_model_num": [1],
}

VLDB_RESULTS_AUTOCL_PARAM_GRID = {
    **VLDB_RESULTS_MAIN_PARAM_GRID,
    "auto_cl_mode": ["v1"],
    "enable_context_len_adaptive_repr": [True],
    "repr_input_dim": [2048],
    "repr_output_dim": [512],
    "repr_sub_pred_len": [720],
    "repr_source_exact_length": [3000],
}


FAST_BASELINE_METHODS_V0 = [
    "TSRouter-main",
    "TSRouter-fast",
    "AutoForecast",
    "AutoXPCR",
    "SimpleTS",
    "Profile-probe-M",
    "Task-probe-M",
    "Task-probe-C",
    "Random",
    "Recent",
    "Current_best-M",
    "Current_best-C",
]

FAST_BASELINE_METHODS_AUTOCL = [
    "TSRouter-autocl",
    "TSRouter-fast",
    "AutoForecast",
    "AutoXPCR",
    "SimpleTS",
    "Profile-probe-M",
    "Task-probe-M",
    "Task-probe-C",
    "Random",
    "Recent",
    "Current_best-M",
    "Current_best-C",
]

PROFILE_PROBE_METHODS = {"Profile-probe-M"}
FIGURE2_STATIC_METHODS = ["Random", "Profile-probe", "Recent", "Current-best"]
FIGURE2_TASK_ADAPTIVE_METHODS = [
    "Task-probe",
    "AutoForecast",
    "AutoXPCR",
    "SimpleTS",
    "TSRouter",
]
FIGURE3_OVERHEAD_ORDER = ["Route", "E2E", "Insert"]
FIGURE3_COMPONENT_COLORS = {
    # Insert: light/dark blue.
    "IncomingProfile": "#9ecae1",
    "Retrain_total": "#3182bd",
    "IndexRefresh": "#084594",
    # Route: light/dark orange.
    "Sample": "#fdd0a2",
    "Sample_to_route": "#e6550d",
    # E2E: green, intentionally separate from Insert and Route.
    "Total": "#74c476",
}
FIGURE3_COMPONENT_ALIASES = {
    "Step2InsertRuntime": "IncomingProfile",
}
FIGURE3_COMPONENT_LEGEND_LABELS = {
    "IncomingProfile": "Incoming profile",
    "Retrain_total": "Relabel + Retrain",
    "IndexRefresh": "Index refresh",
    "Sample": "Request-window sampling",
    "Sample_to_route": "Core-route",
    "Total": "E2E total (route + forecast)",
}
FIGURE3_LEGEND_GROUPS = [
    (
        "Request Routing",
        ("Sample", "Sample_to_route"),
        "#d94801",
    ),
    (
        "End-to-End Latency",
        ("Total",),
        "#238b45",
    ),
    (
        "Model Insertion",
        ("IncomingProfile", "Retrain_total", "IndexRefresh"),
        "#2171b5",
    ),
]
FIGURE3_LEGEND_ANCHOR_Y = 0.985
FIGURE3_Y_AXIS_TOP = 270.0
FIGURE3_P95_LEFT_YLABEL = "P95 latency at the 20-model zoo (s)"
FIGURE3_INSERT_RIGHT_YLABEL = "Mean INSERT time per arrival (s)"
FIGURE3_LEGEND_FONTSIZE = 11.2
FIGURE3_LEGEND_TITLE_FONTSIZE = 13.0


def _summary_auto_cl_mode(args) -> str:
    return get_auto_cl_mode(args)


def _summary_auto_cl_enabled(args) -> bool:
    return _summary_auto_cl_mode(args) != "v0"


def _summary_auto_cl_profiles(args) -> tuple[dict, ...]:
    return get_auto_cl_profiles(_summary_auto_cl_mode(args))


def _summary_auto_cl_profile_token(args) -> str:
    mode = _summary_auto_cl_mode(args)
    return f"autocl{mode}_p{len(_summary_auto_cl_profiles(args))}"


def _summary_auto_cl_long_profile(args) -> dict:
    profiles = _summary_auto_cl_profiles(args)
    for profile in profiles:
        if str(profile.get("profile_key", "")) == "long":
            return dict(profile)
    if profiles:
        return dict(profiles[-1])
    raise ValueError(f"auto_cl_mode={_summary_auto_cl_mode(args)} has no profiles")


def _summary_methods(args) -> list[str]:
    return (
        list(FAST_BASELINE_METHODS_AUTOCL)
        if _summary_auto_cl_enabled(args)
        else list(FAST_BASELINE_METHODS_V0)
    )


def _summary_file_token(args) -> str:
    return f"autocl{_summary_auto_cl_mode(args)}_" if _summary_auto_cl_enabled(args) else ""


def _safe_cache_token(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-_.")
    return text or "none"


def _json_safe_value(value):
    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(val) for val in value]
    if isinstance(value, np.ndarray):
        return _json_safe_value(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stage_summary_profile_signature(args) -> list[dict[str, object]]:
    if not _summary_auto_cl_enabled(args):
        return []
    keys = [
        "adaptive_profile",
        "profile_key",
        "repr_input_dim",
        "repr_output_dim",
        "repr_sub_pred_len",
        "repr_source_exact_length",
        "tsfm_results_dir",
    ]
    return [
        {key: profile.get(key) for key in keys if key in profile}
        for profile in _summary_auto_cl_profiles(args)
    ]


def _stage_summary_cache_key(
    args,
    ordered_model_names: list[str],
    stage: int,
) -> dict[str, object]:
    auto_cl = _summary_auto_cl_enabled(args)
    main_params = _resolve_main_params(args=args, auto_cl=auto_cl)
    rank_path, forward_path, sample_paths, task_probe_cache_stem = _task_probe_source_paths(
        args,
        [int(stage)],
    )
    return _json_safe_value(
        {
            "cache_version": VLDB_STAGE_SUMMARY_CACHE_VERSION,
            "summary_mode": _summary_auto_cl_mode(args),
            "main_method": "TSRouter-autocl" if auto_cl else "TSRouter-main",
            "main_params": main_params,
            "stage": int(stage),
            "zoo_total_num": int(getattr(args, "zoo_total_num", stage)),
            "quick_test": bool(getattr(args, "quick_test", False)),
            "GE_released": bool(getattr(args, "GE_released", False)),
            "TSFM_results_dir": str(getattr(args, "TSFM_results_dir", "cl_512")),
            "rank_base": str(getattr(args, "rank_base", "")),
            "route_family_mode": normalize_route_family_mode(
                getattr(args, "route_family_mode", "default")
            ),
            "ordered_model_names": [str(name) for name in ordered_model_names[: int(stage)]],
            "auto_cl_profiles": _stage_summary_profile_signature(args),
            "task_probe_root": task_probe_select_root(_args_for_stage(args, int(stage), auto_cl=auto_cl)).as_posix(),
            "task_probe_rank_summary": rank_path.as_posix(),
            "task_probe_forward_summary": forward_path.as_posix(),
            "task_probe_sample_timing": [path.as_posix() for path in sample_paths],
            "task_probe_cache_stem": str(task_probe_cache_stem),
        }
    )


def _stage_summary_cache_digest(key: dict[str, object]) -> str:
    blob = json.dumps(key, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _stage_summary_main_token(key: dict[str, object]) -> str:
    params = dict(key.get("main_params", {}) or {})
    parts = [
        params.get("repr_encoder", "repr"),
        f"{params.get('repr_input_dim', 'in')}to{params.get('repr_output_dim', 'out')}",
        f"pl{params.get('repr_sub_pred_len', 'pl')}",
        params.get("repr_scale_protocol", "scale"),
        params.get("zoo_repr_set", "zoo"),
        f"x{params.get('repr_size', 'size')}",
        f"{params.get('repr_anchor_window_sample_strategy', 'anchor')}-{params.get('task_window_sample_strategy', 'task')}",
        f"ss{params.get('search_seed', 'seed')}",
    ]
    token = _safe_cache_token("_".join(str(part) for part in parts))
    return token[:120].rstrip("-_.") or "main"


def _stage_summary_cache_path(
    args,
    ordered_model_names: list[str],
    stage: int,
) -> tuple[Path, dict[str, object]]:
    key = _stage_summary_cache_key(args, ordered_model_names, int(stage))
    digest = _stage_summary_cache_digest(key)
    target_stage = int(key.get("zoo_total_num", stage))
    mode = _safe_cache_token(key.get("summary_mode", "v0"))
    tsfm = _safe_cache_token(key.get("TSFM_results_dir", "cl_512"))
    main_token = _stage_summary_main_token(key)
    name = (
        f"stage_z{int(stage):02d}-z{target_stage:02d}_"
        f"{mode}_{tsfm}_{main_token}_{digest}.json"
    )
    return VLDB_STAGE_SUMMARY_CACHE_ROOT / name, key


def _extract_csv_paths_from_text(text: object) -> list[Path]:
    raw = str(text or "")
    if not raw or raw.startswith("baseline_df_all:"):
        return []
    paths: list[Path] = []
    for token in re.split(r"[\s,;]+", raw):
        if ".csv" not in token:
            continue
        end = token.find(".csv") + len(".csv")
        path_text = token[:end].strip("'\"")
        if "=" in path_text:
            path_text = path_text.rsplit("=", maxsplit=1)[-1]
        if path_text:
            paths.append(Path(path_text))
    return paths


def _tsfm_stage_dependency_paths(
    args,
    current_model_names: list[str],
    season_naive_df: pd.DataFrame | None,
) -> list[Path]:
    paths: list[Path] = []
    tsfm_dirs = [str(getattr(args, "TSFM_results_dir", "cl_512"))]
    if _summary_auto_cl_enabled(args):
        tsfm_dirs.extend(
            str(profile.get("tsfm_results_dir", ""))
            for profile in _summary_auto_cl_profiles(args)
        )
    for tsfm_dir in dict.fromkeys(token for token in tsfm_dirs if token):
        for model_name in current_model_names:
            paths.append(
                resolve_tsfm_csv_path(
                    _model_key_for_abbr(model_name),
                    tsfm_dir,
                    "all_results.csv",
                )
            )
    if (
        season_naive_df is not None
        and "source_file" in season_naive_df.columns
    ):
        paths.extend(
            Path(str(path))
            for path in season_naive_df["source_file"].dropna().astype(str).unique().tolist()
            if str(path).strip()
        )
    return paths


def _stage_summary_cache_dependency_paths(
    args,
    baseline_df_all: pd.DataFrame,
    ordered_model_names: list[str],
    stage: int,
    season_naive_df: pd.DataFrame | None,
    rows: list[dict[str, object]],
    checks: list[dict[str, object]],
) -> list[Path]:
    paths: list[Path] = []
    current_model_names = [str(name) for name in ordered_model_names[: int(stage)]]
    paths.extend(_tsfm_stage_dependency_paths(args, current_model_names, season_naive_df))

    if "source_file" in baseline_df_all.columns and "model" in baseline_df_all.columns:
        sub = baseline_df_all[
            baseline_df_all["model"].astype(str).isin(current_model_names)
        ]
        paths.extend(
            Path(str(path))
            for path in sub["source_file"].dropna().astype(str).unique().tolist()
            if str(path).strip()
        )
    rank_path, forward_path, sample_paths, _cache_stem = _task_probe_source_paths(
        args,
        [int(stage)],
    )
    paths.extend([rank_path, forward_path, *sample_paths])
    for row in rows:
        for key in ["_source", "Source", "Note"]:
            paths.extend(_extract_csv_paths_from_text(row.get(key, "")))
    for row in checks:
        for key in ["Source", "Note"]:
            paths.extend(_extract_csv_paths_from_text(row.get(key, "")))

    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if not path:
            continue
        resolved = Path(path).expanduser().resolve(strict=False)
        token = resolved.as_posix()
        if token not in seen:
            out.append(resolved)
            seen.add(token)
    return out


def _stage_summary_file_stat(path: Path) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve(strict=False)
    try:
        stat = resolved.stat()
    except FileNotFoundError:
        return {"path": resolved.as_posix(), "exists": False}
    return {
        "path": resolved.as_posix(),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _stage_summary_source_stats(paths: Iterable[Path]) -> list[dict[str, object]]:
    return [_stage_summary_file_stat(path) for path in paths]


def _stage_summary_cache_sources_current(
    source_stats: list[dict[str, object]],
) -> tuple[bool, str]:
    for saved in source_stats:
        path = Path(str(saved.get("path", "")))
        current = _stage_summary_file_stat(path)
        if bool(saved.get("exists", False)) != bool(current.get("exists", False)):
            return False, current["path"]
        if not bool(saved.get("exists", False)):
            continue
        if (
            int(saved.get("size", -1)) != int(current.get("size", -2))
            or int(saved.get("mtime_ns", -1)) != int(current.get("mtime_ns", -2))
        ):
            return False, current["path"]
    return True, ""


def _stage_method_summary_params(args, method: str) -> dict[str, object]:
    method = str(method)
    if method == "TSRouter-autocl":
        return _resolve_main_params(args=args, auto_cl=True)
    if method in PROFILE_PROBE_METHODS:
        params = _resolve_main_params(
            args=args,
            auto_cl=_summary_auto_cl_enabled(args),
        )
        params.update(_method_param_overrides(method))
        return params
    if method in {"TSRouter-main", "TSRouter-fast", "AutoForecast", "AutoXPCR", "SimpleTS"}:
        params = _resolve_main_params(args=args, auto_cl=False)
        params.update(_method_param_overrides(method))
        if method in {"AutoForecast", "AutoXPCR", "SimpleTS"}:
            params["advanced_baseline_train_scope"] = normalize_advanced_baseline_train_scope(
                getattr(args, "advanced_baseline_train_scope", "center")
            )
        return params
    if method in {"Task-probe-M", "Task-probe-C"}:
        params = _resolve_main_params(
            args=args,
            auto_cl=_summary_auto_cl_enabled(args),
        )
        params["task_probe_metric"] = "MASE" if method.endswith("-M") else "CRPS"
        return params
    if method == "Random":
        return {"random_seeds": list(FAST_BASELINE_RANDOM_SEEDS)}
    if method == "Recent":
        return {"selection": "latest_stage_model"}
    if method == "Current_best-M":
        return {"selection_metric": "MASE"}
    if method == "Current_best-C":
        return {"selection_metric": "CRPS"}
    return {}


def _stage_method_summary_cache_key(
    args,
    ordered_model_names: list[str],
    stage: int,
    method: str,
) -> dict[str, object]:
    method = str(method)
    current_model_names = [str(name) for name in ordered_model_names[: int(stage)]]
    key: dict[str, object] = {
        "cache_version": VLDB_STAGE_METHOD_SUMMARY_CACHE_VERSION,
        "summary_mode": _summary_auto_cl_mode(args),
        "method": method,
        "method_params": _stage_method_summary_params(args, method),
        "stage": int(stage),
        "zoo_total_num": int(getattr(args, "zoo_total_num", stage)),
        "quick_test": bool(getattr(args, "quick_test", False)),
        "GE_released": bool(getattr(args, "GE_released", False)),
        "TSFM_results_dir": str(getattr(args, "TSFM_results_dir", "cl_512")),
        "rank_base": str(getattr(args, "rank_base", "")),
        "route_family_mode": normalize_route_family_mode(
            getattr(args, "route_family_mode", "default")
        ),
        "ordered_model_names": current_model_names,
        "auto_cl_profiles": _stage_summary_profile_signature(args),
    }
    selector_methods = {
        "TSRouter-main",
        "TSRouter-autocl",
        "TSRouter-fast",
        "AutoForecast",
        "AutoXPCR",
        "SimpleTS",
        *PROFILE_PROBE_METHODS,
    }
    if method in selector_methods:
        auto_cl = method == "TSRouter-autocl" or (
            method in PROFILE_PROBE_METHODS and _summary_auto_cl_enabled(args)
        )
        key["selector_result"] = _main_selector_candidate_path(
            args,
            int(stage),
            auto_cl=auto_cl,
            param_overrides=_method_param_overrides(method),
        ).resolve(strict=False).as_posix()
    if method in {"Task-probe-M", "Task-probe-C"}:
        rank_path, forward_path, sample_paths, cache_stem = _task_probe_source_paths(
            args,
            [int(stage)],
        )
        key["task_probe_root"] = task_probe_select_root(
            _args_for_stage(
                args,
                int(stage),
                auto_cl=_summary_auto_cl_enabled(args),
            )
        ).resolve(strict=False).as_posix()
        key["task_probe_rank_summary"] = rank_path.as_posix()
        key["task_probe_forward_summary"] = forward_path.as_posix()
        key["task_probe_sample_timing"] = [path.as_posix() for path in sample_paths]
        key["task_probe_cache_stem"] = str(cache_stem)
        if _summary_auto_cl_enabled(args):
            key["selector_result"] = _main_selector_candidate_path(
                args,
                int(stage),
                auto_cl=True,
            ).resolve(strict=False).as_posix()
    return _json_safe_value(key)


def _stage_method_summary_cache_path(
    args,
    ordered_model_names: list[str],
    stage: int,
    method: str,
) -> tuple[Path, dict[str, object]]:
    key = _stage_method_summary_cache_key(args, ordered_model_names, int(stage), method)
    digest = _stage_summary_cache_digest(key)
    target_stage = int(key.get("zoo_total_num", stage))
    mode = _safe_cache_token(key.get("summary_mode", "v0"))
    tsfm = _safe_cache_token(key.get("TSFM_results_dir", "cl_512"))
    method_token = _safe_cache_token(method)
    name = (
        f"stage_z{int(stage):02d}-z{target_stage:02d}_"
        f"{mode}_{tsfm}_{method_token}_{digest}.json"
    )
    return VLDB_STAGE_METHOD_SUMMARY_CACHE_ROOT / method_token / name, key


def _dedupe_resolved_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if not path:
            continue
        resolved = Path(path).expanduser().resolve(strict=False)
        token = resolved.as_posix()
        if token not in seen:
            out.append(resolved)
            seen.add(token)
    return out


def _stage_method_summary_cache_dependency_paths(
    args,
    baseline_df: pd.DataFrame,
    current_model_names: list[str],
    stage: int,
    method: str,
    season_naive_df: pd.DataFrame | None,
    row: dict[str, object],
    check: dict[str, object],
) -> list[Path]:
    method = str(method)
    paths: list[Path] = []
    paths.extend(_tsfm_stage_dependency_paths(args, current_model_names, season_naive_df))
    if "source_file" in baseline_df.columns and "model" in baseline_df.columns:
        sub = baseline_df[
            baseline_df["model"].astype(str).isin([str(m) for m in current_model_names])
        ]
        paths.extend(
            Path(str(path))
            for path in sub["source_file"].dropna().astype(str).unique().tolist()
            if str(path).strip()
        )
    selector_methods = {
        "TSRouter-main",
        "TSRouter-autocl",
        "TSRouter-fast",
        "AutoForecast",
        "AutoXPCR",
        "SimpleTS",
        *PROFILE_PROBE_METHODS,
    }
    if method in selector_methods:
        auto_cl = method == "TSRouter-autocl" or (
            method in PROFILE_PROBE_METHODS and _summary_auto_cl_enabled(args)
        )
        paths.append(
            _main_selector_candidate_path(
                args,
                int(stage),
                auto_cl=auto_cl,
                param_overrides=_method_param_overrides(method),
            )
        )
    if method in {"Task-probe-M", "Task-probe-C"}:
        rank_path, forward_path, sample_paths, _cache_stem = _task_probe_source_paths(
            args,
            [int(stage)],
        )
        paths.extend([rank_path, forward_path, *sample_paths])
        if _summary_auto_cl_enabled(args):
            paths.append(_main_selector_candidate_path(args, int(stage), auto_cl=True))
    for source_row in [row, check]:
        for key in ["_source", "Source", "Note"]:
            paths.extend(_extract_csv_paths_from_text(source_row.get(key, "")))
    return _dedupe_resolved_paths(paths)


def _read_stage_method_summary_cache(
    args,
    ordered_model_names: list[str],
    stage: int,
    method: str,
) -> tuple[dict[str, object], dict[str, object]] | None:
    path, key = _stage_method_summary_cache_path(
        args,
        ordered_model_names,
        int(stage),
        method,
    )
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            f"[vldb_results][cache] miss stage=z{int(stage)} method={method} "
            f"reason=read_error:{type(exc).__name__} path={path.as_posix()}",
            flush=True,
        )
        return None
    if (
        payload.get("cache_version") != VLDB_STAGE_METHOD_SUMMARY_CACHE_VERSION
        or payload.get("key") != key
    ):
        print(
            f"[vldb_results][cache] miss stage=z{int(stage)} method={method} "
            f"reason=key_mismatch path={path.as_posix()}",
            flush=True,
        )
        return None
    ok, stale_source = _stage_summary_cache_sources_current(
        list(payload.get("source_stats", []) or [])
    )
    if not ok:
        print(
            f"[vldb_results][cache] miss stage=z{int(stage)} method={method} "
            f"reason=source_changed source={stale_source} path={path.as_posix()}",
            flush=True,
        )
        return None
    row = dict(payload.get("row", {}) or {})
    check = dict(payload.get("check", {}) or {})
    if not row or not check:
        print(
            f"[vldb_results][cache] miss stage=z{int(stage)} method={method} "
            f"reason=empty_payload path={path.as_posix()}",
            flush=True,
        )
        return None
    return row, check


def _write_stage_method_summary_cache(
    args,
    baseline_df: pd.DataFrame,
    current_model_names: list[str],
    ordered_model_names: list[str],
    stage: int,
    method: str,
    season_naive_df: pd.DataFrame | None,
    row: dict[str, object],
    check: dict[str, object],
) -> None:
    path, key = _stage_method_summary_cache_path(
        args,
        ordered_model_names,
        int(stage),
        method,
    )
    dependency_paths = _stage_method_summary_cache_dependency_paths(
        args=args,
        baseline_df=baseline_df,
        current_model_names=current_model_names,
        stage=int(stage),
        method=method,
        season_naive_df=season_naive_df,
        row=row,
        check=check,
    )
    payload = {
        "cache_version": VLDB_STAGE_METHOD_SUMMARY_CACHE_VERSION,
        "key": key,
        "source_stats": _stage_summary_source_stats(dependency_paths),
        "row": _json_safe_value(row),
        "check": _json_safe_value(check),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(path)
    except Exception as exc:
        print(
            f"[vldb_results][cache] write_failed stage=z{int(stage)} "
            f"method={method} error={type(exc).__name__}: {exc} "
            f"path={path.as_posix()}",
            flush=True,
        )
        return
    print(
        f"[vldb_results][cache] saved stage=z{int(stage)} method={method} "
        f"deps={len(dependency_paths)} path={path.as_posix()}",
        flush=True,
    )


def _read_stage_summary_cache(
    args,
    baseline_df_all: pd.DataFrame,
    ordered_model_names: list[str],
    stage: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]] | None:
    del baseline_df_all
    path, key = _stage_summary_cache_path(args, ordered_model_names, int(stage))
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            f"[vldb_results][cache] miss stage=z{int(stage)} reason=read_error:{type(exc).__name__} "
            f"path={path.as_posix()}",
            flush=True,
        )
        return None
    if payload.get("cache_version") != VLDB_STAGE_SUMMARY_CACHE_VERSION or payload.get("key") != key:
        print(
            f"[vldb_results][cache] miss stage=z{int(stage)} reason=key_mismatch "
            f"path={path.as_posix()}",
            flush=True,
        )
        return None
    ok, stale_source = _stage_summary_cache_sources_current(
        list(payload.get("source_stats", []) or [])
    )
    if not ok:
        print(
            f"[vldb_results][cache] miss stage=z{int(stage)} reason=source_changed "
            f"source={stale_source} path={path.as_posix()}",
            flush=True,
        )
        return None
    rows = list(payload.get("rows", []) or [])
    checks = list(payload.get("checks", []) or [])
    return rows, checks


def _legacy_stage_summary_cache_by_method(
    args,
    baseline_df_all: pd.DataFrame,
    baseline_df: pd.DataFrame,
    current_model_names: list[str],
    ordered_model_names: list[str],
    stage: int,
    season_naive_df: pd.DataFrame | None,
) -> dict[str, tuple[dict[str, object], dict[str, object]]]:
    del baseline_df_all
    path, key = _stage_summary_cache_path(args, ordered_model_names, int(stage))
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if (
        payload.get("cache_version") != VLDB_STAGE_SUMMARY_CACHE_VERSION
        or payload.get("key") != key
    ):
        return {}
    rows = list(payload.get("rows", []) or [])
    checks = list(payload.get("checks", []) or [])
    saved_stats_by_path = {
        Path(str(saved.get("path", ""))).expanduser().resolve(strict=False).as_posix(): saved
        for saved in list(payload.get("source_stats", []) or [])
        if str(saved.get("path", "")).strip()
    }
    checks_by_method = {
        str(check.get("Method", "")): dict(check)
        for check in checks
        if str(check.get("Method", ""))
    }
    out: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    for row in rows:
        method = str(row.get("Method", ""))
        if not method or method in out:
            continue
        check = checks_by_method.get(method)
        if check is None:
            continue
        dependency_paths = _stage_method_summary_cache_dependency_paths(
            args=args,
            baseline_df=baseline_df,
            current_model_names=current_model_names,
            stage=int(stage),
            method=method,
            season_naive_df=season_naive_df,
            row=dict(row),
            check=check,
        )
        saved_dependency_stats: list[dict[str, object]] = []
        missing_saved_dependency = ""
        for dependency_path in dependency_paths:
            token = dependency_path.resolve(strict=False).as_posix()
            saved = saved_stats_by_path.get(token)
            if saved is None:
                missing_saved_dependency = token
                break
            saved_dependency_stats.append(saved)
        if missing_saved_dependency:
            print(
                f"[vldb_results][cache] legacy_skip stage=z{int(stage)} method={method} "
                f"reason=dependency_not_recorded source={missing_saved_dependency}",
                flush=True,
            )
            continue
        current, stale_source = _stage_summary_cache_sources_current(
            saved_dependency_stats
        )
        if not current:
            print(
                f"[vldb_results][cache] legacy_skip stage=z{int(stage)} method={method} "
                f"reason=source_changed source={stale_source}",
                flush=True,
            )
            continue
        out[method] = (dict(row), check)
    return out


def _write_stage_summary_cache(
    args,
    baseline_df_all: pd.DataFrame,
    ordered_model_names: list[str],
    stage: int,
    season_naive_df: pd.DataFrame | None,
    rows: list[dict[str, object]],
    checks: list[dict[str, object]],
) -> None:
    path, key = _stage_summary_cache_path(args, ordered_model_names, int(stage))
    dependency_paths = _stage_summary_cache_dependency_paths(
        args=args,
        baseline_df_all=baseline_df_all,
        ordered_model_names=ordered_model_names,
        stage=int(stage),
        season_naive_df=season_naive_df,
        rows=rows,
        checks=checks,
    )
    payload = {
        "cache_version": VLDB_STAGE_SUMMARY_CACHE_VERSION,
        "key": key,
        "source_stats": _stage_summary_source_stats(dependency_paths),
        "rows": _json_safe_value(rows),
        "checks": _json_safe_value(checks),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(path)
    except Exception as exc:
        print(
            f"[vldb_results][cache] write_failed stage=z{int(stage)} "
            f"error={type(exc).__name__}: {exc} path={path.as_posix()}",
            flush=True,
        )
        return
    print(
        f"[vldb_results][cache] saved stage=z{int(stage)} deps={len(dependency_paths)} "
        f"path={path.as_posix()}",
        flush=True,
    )


TABLE1_DISPLAY_COLUMNS = [
    "Method",
    "MASE",
    "Regret-M",
    "Regret-M P90",
    "Rank-M",
    "MASE-hit1/3",
    "CRPS",
    "Rank-C",
    "CRPS-hit1/3",
    "sMAPE",
    "Count-1",
    "Count-2",
    "PWW1/3↑",
    "TWW1/3↑",
    "TCC1/3↑",
    "TWC1/3↑",
    "TWR1/3↑",
    "TCR1/3↑",
    "Route P50/95(s)",
    "Core-route P50/P95(s)",
    "E2E P50/95(s)",
    "Route throughput(req/min)",
    "N_VALID_EXPECTED_DS",
]

TABLE1_OPTIONAL_PROCESS_PREFIXES = (
    "PWW",
    "TWW",
    "TCC",
    "TWC",
    "TWR",
    "TCR",
)


SOURCE_CHECK_COLUMNS = [
    "Stage",
    "Method",
    "SourceKind",
    "Source",
    "Rows",
    "ValidExpected",
    "Expected",
    "Complete",
    "MetricComplete",
    "RouteComplete",
    "E2EComplete",
    "RuntimeComplete",
    "RouteValidN",
    "E2EValidN",
    "SelectedModel",
    "MissingDatasets",
    "MetricMissingDatasets",
    "AutoCLMode",
    "RequiredProfiles",
    "ObservedProfiles",
    "ProfileComplete",
    "ResolvedCLComplete",
    "RankTruthCLComplete",
    "TSFMForwardCLComplete",
    "ProfileDatasetCounts",
    "LegacyCLDerivedN",
    "EvalCLFallbackN",
    "Note",
]


ROUTE_BREAKDOWN_METRICS = [
    ("TSRouter-sample_seconds", "sample_seconds"),
    ("TSRouter-sample_to_route_seconds", "sample_to_route_seconds"),
    ("TSRouter-route_final_seconds", "route_final_seconds"),
    ("TSRouter-fast-sample_seconds", "sample_seconds"),
    ("TSRouter-fast-sample_to_route_seconds", "sample_to_route_seconds"),
    ("TSRouter-fast-route_final_seconds", "route_final_seconds"),
    ("AutoForecast-sample_seconds", "sample_seconds"),
    ("AutoForecast-sample_to_route_seconds", "sample_to_route_seconds"),
    ("AutoForecast-route_final_seconds", "route_final_seconds"),
    ("AutoXPCR-sample_seconds", "sample_seconds"),
    ("AutoXPCR-sample_to_route_seconds", "sample_to_route_seconds"),
    ("AutoXPCR-route_final_seconds", "route_final_seconds"),
    ("SimpleTS-sample_seconds", "sample_seconds"),
    ("SimpleTS-sample_to_route_seconds", "sample_to_route_seconds"),
    ("SimpleTS-route_final_seconds", "route_final_seconds"),
]

AUTOCL_ROUTE_BREAKDOWN_METRICS = [
    ("TSRouter-autocl-sample_seconds", "sample_seconds"),
    ("TSRouter-autocl-sample_to_route_seconds", "sample_to_route_seconds"),
    ("TSRouter-autocl-route_final_seconds", "route_final_seconds"),
    ("TSRouter-fast-sample_seconds", "sample_seconds"),
    ("TSRouter-fast-sample_to_route_seconds", "sample_to_route_seconds"),
    ("TSRouter-fast-route_final_seconds", "route_final_seconds"),
    ("AutoForecast-sample_seconds", "sample_seconds"),
    ("AutoForecast-sample_to_route_seconds", "sample_to_route_seconds"),
    ("AutoForecast-route_final_seconds", "route_final_seconds"),
    ("AutoXPCR-sample_seconds", "sample_seconds"),
    ("AutoXPCR-sample_to_route_seconds", "sample_to_route_seconds"),
    ("AutoXPCR-route_final_seconds", "route_final_seconds"),
    ("SimpleTS-sample_seconds", "sample_seconds"),
    ("SimpleTS-sample_to_route_seconds", "sample_to_route_seconds"),
    ("SimpleTS-route_final_seconds", "route_final_seconds"),
]


TASK_PROBE_ROUTE_BREAKDOWN_METRICS = [
    ("Task-probe-sample_seconds", "sample_seconds"),
    ("Task-probe-forward_seconds", "task_probe_forward_seconds"),
    ("Task-probe-rank_seconds", "task_probe_rank_seconds"),
    ("Task-probe-route_final_seconds", "route_final_seconds"),
]

ROUTE_MEAN_METRICS = (
    ROUTE_BREAKDOWN_METRICS
    + TASK_PROBE_ROUTE_BREAKDOWN_METRICS
)

TSROUTER_INSERT_METRICS = [
    ("TSRouter-Step2InsertRuntime(s)", "insert_runtime_seconds"),
    ("TSRouter-IndexRefresh(s)", "index_refresh_seconds"),
    ("TSRouter-InsertTotal(s)", "insert_total_seconds"),
]

TSROUTER_FAST_INSERT_METRICS = [
    ("TSRouter-fast-Step2InsertRuntime(s)", "insert_runtime_seconds"),
    ("TSRouter-fast-IndexRefresh(s)", "index_refresh_seconds"),
    ("TSRouter-fast-InsertTotal(s)", "insert_total_seconds"),
]

AUTOFORECAST_INSERT_METRICS = [
    ("AutoForecast-IncomingProfile(s)", "incoming_profile_seconds"),
    ("AutoForecast-LabelRefresh(s)", "label_refresh_seconds"),
    ("AutoForecast-FeatureRefresh(s)", "feature_refresh_seconds"),
    ("AutoForecast-Retrain(s)", "selector_retrain_seconds"),
    ("AutoForecast-InsertTotal(s)", "insert_total_seconds"),
]

AUTOXPCR_INSERT_METRICS = [
    ("AutoXPCR-IncomingProfile(s)", "incoming_profile_seconds"),
    ("AutoXPCR-LabelRefresh(s)", "label_refresh_seconds"),
    ("AutoXPCR-FeatureRefresh(s)", "feature_refresh_seconds"),
    ("AutoXPCR-ResourceRefresh(s)", "resource_refresh_seconds"),
    ("AutoXPCR-Retrain(s)", "selector_retrain_seconds"),
    ("AutoXPCR-InsertTotal(s)", "insert_total_seconds"),
]

SIMPLETS_INSERT_METRICS = [
    ("SimpleTS-IncomingProfile(s)", "incoming_profile_seconds"),
    ("SimpleTS-LabelRefresh(s)", "label_refresh_seconds"),
    ("SimpleTS-StructureRefresh(s)", "structure_refresh_seconds"),
    ("SimpleTS-Retrain(s)", "selector_retrain_seconds"),
    ("SimpleTS-InsertTotal(s)", "insert_total_seconds"),
]

AUTOCL_INSERT_METRICS = [
    ("TSRouter-autocl-Step2InsertRuntime(s)", "insert_runtime_seconds"),
    ("TSRouter-autocl-IndexRefresh(s)", "index_refresh_seconds"),
    ("TSRouter-autocl-InsertTotal(s)", "insert_total_seconds"),
]


def _expected_datasets(quick_test: bool) -> set[str]:
    return set(ALL_Fast_DATASETS if quick_test else ALL_DATASETS)


def _stage_list(args) -> list[int]:
    main_ensemble_size = int(_resolve_main_params().get("ensemble_size", getattr(args, "ensemble_size", 1)))
    start = max(3, main_ensemble_size + 1)
    end = int(getattr(args, "zoo_total_num", start))
    return list(range(start, end + 1))


def _model_id_maps(ordered_model_names: Iterable[str]) -> tuple[dict[str, int], dict[int, str]]:
    ordered = [str(m) for m in ordered_model_names]
    abbr_to_id = {abbr: idx for idx, abbr in enumerate(ordered)}
    id_to_abbr = {idx: abbr for idx, abbr in enumerate(ordered)}
    return abbr_to_id, id_to_abbr


def _to_numeric_or_nan(value) -> float:
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(val) or not np.isfinite(float(val)):
        return np.nan
    return float(val)


def _timing_stats(values: Iterable[object]) -> dict[str, object]:
    """Return exact aggregate/quantile statistics for one timing population."""
    series = pd.to_numeric(pd.Series(list(values), dtype=object), errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    series = series[series.ge(0)]
    if series.empty:
        return {
            "total": np.nan,
            "mean": np.nan,
            "p50": np.nan,
            "p95": np.nan,
            "n": 0,
        }
    return {
        "total": float(series.sum()),
        "mean": float(series.mean()),
        "p50": float(series.quantile(0.50)),
        "p95": float(series.quantile(0.95)),
        "n": int(len(series)),
    }


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t", "on"}


def _round2(value) -> float:
    val = _to_numeric_or_nan(value)
    return round(val, 2) if np.isfinite(val) else np.nan


def _format_pair(left, right, decimals: int = 3) -> str:
    left_val = _to_numeric_or_nan(left)
    right_val = _to_numeric_or_nan(right)
    if not np.isfinite(left_val) and not np.isfinite(right_val):
        return ""
    ltxt = "" if not np.isfinite(left_val) else f"{left_val:.{int(decimals)}f}"
    rtxt = "" if not np.isfinite(right_val) else f"{right_val:.{int(decimals)}f}"
    return f"{ltxt}/{rtxt}"


def _normalize_by_season_naive(df: pd.DataFrame, season_naive_df: pd.DataFrame | None) -> pd.DataFrame:
    if season_naive_df is None or season_naive_df.empty or df is None or df.empty:
        return df
    if not {"dataset", "MASE", "CRPS"}.issubset(season_naive_df.columns):
        return df
    ref = season_naive_df[["dataset", "MASE", "CRPS"]].copy()
    ref = ref.dropna(subset=["dataset"]).drop_duplicates("dataset", keep="last")
    ref = ref.rename(columns={"MASE": "_sn_MASE", "CRPS": "_sn_CRPS"})
    out = df.merge(ref, on="dataset", how="left")
    eps = 1e-12
    for metric, ref_col in [("MASE", "_sn_MASE"), ("CRPS", "_sn_CRPS")]:
        if metric in out.columns:
            out[metric] = pd.to_numeric(out[metric], errors="coerce") / (
                pd.to_numeric(out[ref_col], errors="coerce") + eps
            )
    return out.drop(columns=["_sn_MASE", "_sn_CRPS"], errors="ignore")


def _stage_baseline(
    args,
    baseline_df_all: pd.DataFrame,
    ordered_model_names: list[str],
    stage: int,
    season_naive_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[str], set[str]]:
    expected = _expected_datasets(bool(getattr(args, "quick_test", False)))
    current_models = [str(m) for m in ordered_model_names[: int(stage)]]
    base = baseline_df_all[baseline_df_all["model"].astype(str).isin(current_models)].copy()
    if getattr(args, "GE_released", False):
        base = _normalize_by_season_naive(base, season_naive_df)
    if "dataset" in base.columns:
        base["dataset"] = base["dataset"].astype(str)
        base = base[base["dataset"].isin(expected)].copy()
        base = base.drop_duplicates(["dataset", "model"], keep="last")
    for metric in ["MASE", "sMAPE", "CRPS", "forward_runtime_seconds", "non_eval_runtime_seconds", "runtime_seconds"]:
        if metric in base.columns:
            base[metric] = pd.to_numeric(base[metric], errors="coerce")
    return base, current_models, expected


def _summary_baseline_df(
    args,
    baseline_df_all: pd.DataFrame,
    ordered_model_names: list[str],
) -> pd.DataFrame:
    if not _summary_auto_cl_enabled(args):
        return baseline_df_all
    comparison_dir = auto_cl_tsfm_comparison_dir(args)
    args.TSFM_results_dir = comparison_dir
    print(
        f"[vldb_results][baseline] loading {comparison_dir} TSFM rows for "
        f"{len(ordered_model_names)} models",
        flush=True,
    )
    rows: list[pd.DataFrame] = []
    for index, model_name in enumerate(ordered_model_names, start=1):
        model_key = _model_key_for_abbr(model_name)
        path = resolve_tsfm_csv_path(model_key, comparison_dir, "all_results.csv")
        try:
            model_df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            print(
                f"[vldb_results][baseline][{index}/{len(ordered_model_names)}] "
                f"missing model={model_name} path={path.as_posix()} error={exc}",
                flush=True,
            )
            continue
        if model_df.empty or "dataset" not in model_df.columns:
            print(
                f"[vldb_results][baseline][{index}/{len(ordered_model_names)}] "
                f"invalid model={model_name} path={path.as_posix()}",
                flush=True,
            )
            continue
        model_df = model_df.copy()
        model_df["model"] = str(model_name)
        model_df["source_file"] = path.as_posix()
        rows.append(model_df)
        print(
            f"[vldb_results][baseline][{index}/{len(ordered_model_names)}] "
            f"loaded model={model_name} rows={len(model_df)}",
            flush=True,
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _single_value_grid_params(grid: dict, grid_name: str) -> dict:
    params = {}
    bad = []
    for key, values in grid.items():
        if not isinstance(values, (list, tuple)) or len(values) != 1:
            bad.append(key)
            continue
        params[key] = values[0]
    if bad:
        raise ValueError(
            f"{grid_name} must define one value per key for --vldb_results; "
            f"multi-valued keys: {bad}"
        )
    return params


def _resolve_main_params(args=None, auto_cl: bool = False) -> dict:
    params = _single_value_grid_params(
        VLDB_RESULTS_MAIN_PARAM_GRID,
        "VLDB_RESULTS_MAIN_PARAM_GRID",
    )
    if auto_cl and args is None:
        params.update(
            _single_value_grid_params(
                VLDB_RESULTS_AUTOCL_PARAM_GRID,
                "VLDB_RESULTS_AUTOCL_PARAM_GRID",
            )
        )
    if auto_cl and args is not None:
        mode = _summary_auto_cl_mode(args)
        if mode != "v0":
            long_profile = _summary_auto_cl_long_profile(args)
            params["auto_cl_mode"] = mode
            params["enable_context_len_adaptive_repr"] = True
            params["repr_input_dim"] = int(long_profile["repr_input_dim"])
            params["repr_output_dim"] = int(long_profile["repr_output_dim"])
            params["repr_sub_pred_len"] = int(long_profile["repr_sub_pred_len"])
            params["repr_source_exact_length"] = int(long_profile["repr_source_exact_length"])
    if not auto_cl and args is not None:
        fixed_profile = get_fixed_cl_profile(
            getattr(args, "TSFM_results_dir", "cl_512")
        )
        if fixed_profile is not None:
            for key in [
                "repr_input_dim",
                "repr_output_dim",
                "repr_sub_pred_len",
                "repr_source_exact_length",
            ]:
                params[key] = int(fixed_profile[key])
    if args is not None:
        params["route_family_mode"] = normalize_route_family_mode(
            getattr(args, "route_family_mode", "default")
        )
        raw_fb = getattr(args, "task_rank_top3_instability_threshold", None)
        if raw_fb is not None:
            try:
                fb_threshold = float(raw_fb)
            except Exception:
                fb_threshold = -1.0
            if abs(fb_threshold) < 1e-12:
                fb_threshold = 0.0
            if fb_threshold >= 0:
                params["task_rank_top3_instability_threshold"] = fb_threshold
    return params


def _route_fast_param_overrides() -> dict[str, object]:
    return {"route_efficiency_mode": True}


def _autoforecast_param_overrides() -> dict[str, object]:
    return {
        "repr_v": 7,
        "route_efficiency_mode": False,
        "autoforecast_learner": VLDB_RESULTS_AUTOFORECAST_LEARNER,
    }


def _autoxpcr_param_overrides() -> dict[str, object]:
    return {
        "repr_v": 7,
        "route_efficiency_mode": True,
        "autoforecast_learner": VLDB_RESULTS_AUTOFORECAST_LEARNER,
    }


def _simplets_param_overrides() -> dict[str, object]:
    return {
        "repr_encoder": "TS2Vec",
        "repr_v": 6,
        "route_efficiency_mode": False,
    }


def _profile_probe_param_overrides(method: str) -> dict[str, object]:
    if method not in PROFILE_PROBE_METHODS:
        return {}
    return {
        "base_metrics": "M" if method.endswith("-M") else "C",
        "task_rank_top3_instability_threshold": 0.0,
    }


def _method_param_overrides(method: str) -> dict[str, object]:
    if method == "TSRouter-fast":
        return _route_fast_param_overrides()
    if method == "AutoForecast":
        return _autoforecast_param_overrides()
    if method == "AutoXPCR":
        return _autoxpcr_param_overrides()
    if method == "SimpleTS":
        return _simplets_param_overrides()
    if method in PROFILE_PROBE_METHODS:
        return _profile_probe_param_overrides(method)
    return {}


def vldb_results_param_grid(args=None) -> dict:
    auto_cl = bool(args is not None and _summary_auto_cl_enabled(args))
    return {
        key: [value]
        for key, value in _resolve_main_params(args=args, auto_cl=auto_cl).items()
    }


def _args_for_stage(
    args,
    stage: int,
    *,
    auto_cl: bool = False,
    param_overrides: dict[str, object] | None = None,
):
    out = copy.deepcopy(args)
    out.current_zoo_num = int(stage)
    out.zoo_total_num = int(getattr(args, "zoo_total_num", stage))
    main_params = _resolve_main_params(args=args, auto_cl=auto_cl)
    for key, value in main_params.items():
        setattr(out, key, value)
    for key, value in (param_overrides or {}).items():
        setattr(out, key, value)
    if not auto_cl:
        out.auto_cl_mode = "v0"
        out.enable_context_len_adaptive_repr = False
    if "repr_encoder" in main_params:
        # The paper grid is authoritative. Stale variant fields from the
        # caller must not rewrite StatsRandomFourier back to another encoder.
        out.encoder_type = None
        out.encoder_structure = None
    normalize_encoder_variant_args(out)
    normalize_auto_cl_args(out)
    return out


def _main_selector_candidate_path(
    args,
    stage: int,
    *,
    auto_cl: bool = False,
    param_overrides: dict[str, object] | None = None,
) -> Path:
    args_stage = _args_for_stage(
        args,
        stage,
        auto_cl=auto_cl,
        param_overrides=param_overrides,
    )
    _, _, _, save_name = get_repr_save_path(args_stage)
    stage_path = Path(get_tsrouter_selector_stage_result_dir(args_stage)) / save_name
    root_path = Path(get_tsrouter_selector_result_dir(args_stage)) / save_name
    return stage_path if stage_path.exists() or not root_path.exists() else root_path


def _main_selector_result_path(
    args,
    stage: int,
    *,
    auto_cl: bool = False,
    param_overrides: dict[str, object] | None = None,
) -> Path:
    args_stage = _args_for_stage(
        args,
        stage,
        auto_cl=auto_cl,
        param_overrides=param_overrides,
    )
    target = _main_selector_candidate_path(
        args,
        stage,
        auto_cl=auto_cl,
        param_overrides=param_overrides,
    )
    if target.exists():
        return target
    expected_model_order = [
        str(Model_abbrev_map.get(model_name, model_name))
        for model_name in All_sorted_model_names[: int(stage)]
    ]
    reused = materialize_compatible_tsrouter_result(
        args_stage,
        str(target),
        expected_model_order,
    )
    if reused is not None:
        return target
    return target


def _main_repr_forward_path(args, stage: int) -> Path:
    args_stage = _args_for_stage(args, stage)
    return Path(get_tsrouter_repr_forward_dir(args_stage)) / f"{build_repr_forward_stem(args_stage)}_per_sample_results.csv"


def _parse_model_order_ids(value, abbr_to_id: dict[str, int], k: int | None = None) -> list[int]:
    def parse_one(token) -> int | None:
        try:
            val = float(token)
            if np.isfinite(val):
                return int(val)
        except Exception:
            text = str(token).strip().strip("'\"")
            if text in abbr_to_id:
                return int(abbr_to_id[text])
        return None

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        raw_vals = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            raw_vals = list(parsed) if isinstance(parsed, (list, tuple, np.ndarray)) else []
        except Exception:
            raw_vals = re.findall(r"-?\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_.-]*", text.strip("[]"))
    out: list[int] = []
    for raw in raw_vals:
        parsed = parse_one(raw)
        if parsed is None:
            continue
        out.append(int(parsed))
        if k is not None and len(out) >= int(k):
            break
    return out


def _selected_id_from_row(row: pd.Series, abbr_to_id: dict[str, int]) -> int | None:
    ids = _parse_model_order_ids(row.get("model_order", None), abbr_to_id, k=1)
    if ids:
        return ids[0]
    selected = _to_numeric_or_nan(row.get("selected_model_id", np.nan))
    if np.isfinite(selected):
        return int(selected)
    model = str(row.get("_selected_tsfm_model", row.get("model", "")))
    if model in abbr_to_id:
        return int(abbr_to_id[model])
    return None


def _forward_seconds_for_row(row: pd.Series) -> float:
    if "forward_runtime_seconds" in row.index:
        val = _to_numeric_or_nan(row.get("forward_runtime_seconds"))
        if np.isfinite(val):
            return val
    return np.nan


def _first_selected_forward_seconds(
    row: pd.Series,
    baseline_lookup: dict[tuple[str, str], pd.Series],
    id_to_abbr: dict[int, str],
    abbr_to_id: dict[str, int],
) -> float:
    exact = _to_numeric_or_nan(row.get("_selected_forward_runtime_seconds", np.nan))
    if np.isfinite(exact):
        return exact
    mid = _selected_id_from_row(row, abbr_to_id)
    if mid is None:
        return np.nan
    model = id_to_abbr.get(int(mid))
    if model is None:
        return np.nan
    rec = baseline_lookup.get((str(row.get("dataset", "")), str(model)))
    if rec is None:
        return np.nan
    return _forward_seconds_for_row(rec)


def _attach_auto_cl_selected_forward(
    selected_df: pd.DataFrame,
    ordered_model_names: list[str],
) -> pd.DataFrame:
    if selected_df is None or selected_df.empty:
        return selected_df
    abbr_to_id, id_to_abbr = _model_id_maps(ordered_model_names)
    csv_cache: dict[str, pd.DataFrame] = {}
    work = selected_df.copy()
    runtimes: list[float] = []
    complete: list[bool] = []
    sources: list[str] = []
    for _, rec in work.iterrows():
        resolved_cl = str(rec.get("resolved_eval_cl", "") or "")
        selected_id = _selected_id_from_row(rec, abbr_to_id)
        selected_model = id_to_abbr.get(selected_id) if selected_id is not None else None
        if not resolved_cl or not selected_model:
            runtimes.append(np.nan)
            complete.append(False)
            sources.append("")
            continue
        model_key = _model_key_for_abbr(selected_model)
        path = resolve_tsfm_csv_path(model_key, resolved_cl, "all_results.csv")
        token = path.as_posix()
        if token not in csv_cache:
            try:
                csv_cache[token] = pd.read_csv(path, low_memory=False)
            except Exception:
                csv_cache[token] = pd.DataFrame()
        model_df = csv_cache[token]
        if (
            model_df.empty
            or "dataset" not in model_df.columns
            or "forward_runtime_seconds" not in model_df.columns
        ):
            runtimes.append(np.nan)
            complete.append(False)
            sources.append(token)
            continue
        match = model_df[
            model_df["dataset"].astype(str).eq(str(rec.get("dataset", "")))
        ]
        runtime = (
            _to_numeric_or_nan(match.iloc[-1].get("forward_runtime_seconds"))
            if not match.empty
            else np.nan
        )
        runtimes.append(runtime)
        complete.append(bool(np.isfinite(runtime) and runtime >= 0))
        sources.append(token)
    work["_selected_forward_runtime_seconds"] = runtimes
    work["_tsfm_forward_cl_ok"] = complete
    work["_selected_forward_runtime_source"] = sources
    return work


def _metric_rank_and_hit(
    selected_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    metric: str,
    abbr_to_id: dict[str, int],
) -> tuple[float, float, float]:
    if selected_df is None or selected_df.empty or metric not in selected_df.columns or metric not in baseline_df.columns:
        return np.nan, np.nan, np.nan
    base = baseline_df[["dataset", "model", metric]].copy()
    base[metric] = pd.to_numeric(base[metric], errors="coerce")
    base = base.dropna(subset=["dataset", "model", metric])
    if base.empty:
        return np.nan, np.nan, np.nan
    if "model_id" not in base.columns:
        base["model_id"] = base["model"].astype(str).map(abbr_to_id)
    ranks = []
    hit1 = []
    hit3 = []
    eps_rel = 1e-3
    grouped = {str(ds): group.copy() for ds, group in base.groupby("dataset")}
    for _, rec in selected_df.iterrows():
        ds = str(rec.get("dataset", ""))
        group = grouped.get(ds)
        if group is None or group.empty:
            continue
        selector_val = _to_numeric_or_nan(rec.get(metric))
        if not np.isfinite(selector_val):
            continue
        vals = pd.to_numeric(group[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            continue
        rank_pos = int((vals < selector_val * (1.0 - eps_rel)).sum()) + 1
        ranks.append(rank_pos)
        mid = _selected_id_from_row(rec, abbr_to_id)
        if mid is None:
            continue
        model_rows = group[group["model_id"].eq(int(mid))]
        if model_rows.empty:
            continue
        selected_model_val = _to_numeric_or_nan(model_rows.iloc[-1].get(metric))
        if not np.isfinite(selected_model_val):
            continue
        selected_model_rank = int((vals < selected_model_val * (1.0 - eps_rel)).sum()) + 1
        hit1.append(selected_model_rank <= 1)
        hit3.append(selected_model_rank <= 3)
    rank_mean = float(np.mean(ranks)) if ranks else np.nan
    hit1_mean = float(np.mean(hit1)) if hit1 else np.nan
    hit3_mean = float(np.mean(hit3)) if hit3 else np.nan
    return rank_mean, hit1_mean, hit3_mean


def _mase_regret_stats(
    selected_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> tuple[float, float]:
    """Return mean and P90 MASE regret against the per-dataset best TSFM."""
    if (
        selected_df is None
        or selected_df.empty
        or baseline_df is None
        or baseline_df.empty
        or "dataset" not in selected_df.columns
        or "MASE" not in selected_df.columns
        or "dataset" not in baseline_df.columns
        or "MASE" not in baseline_df.columns
    ):
        return np.nan, np.nan
    base = baseline_df[["dataset", "MASE"]].copy()
    base["dataset"] = base["dataset"].astype(str)
    base["MASE"] = pd.to_numeric(base["MASE"], errors="coerce")
    best = (
        base.dropna(subset=["dataset", "MASE"])
        .groupby("dataset", sort=False)["MASE"]
        .min()
        .rename("_best_tsfm_mase")
        .reset_index()
    )
    if best.empty:
        return np.nan, np.nan
    selected = selected_df[["dataset", "MASE"]].copy()
    selected["dataset"] = selected["dataset"].astype(str)
    selected["MASE"] = pd.to_numeric(selected["MASE"], errors="coerce")
    merged = selected.merge(best, on="dataset", how="left")
    regret = (
        pd.to_numeric(merged["MASE"], errors="coerce")
        - pd.to_numeric(merged["_best_tsfm_mase"], errors="coerce")
    )
    regret = regret.replace([np.inf, -np.inf], np.nan).dropna()
    if regret.empty:
        return np.nan, np.nan
    return float(regret.mean()), float(regret.quantile(0.90))


def _selected_counts(selected_df: pd.DataFrame, abbr_to_id: dict[str, int], id_to_abbr: dict[int, str]) -> tuple[str, str]:
    ids = []
    if selected_df is not None and not selected_df.empty:
        for _, rec in selected_df.iterrows():
            mid = _selected_id_from_row(rec, abbr_to_id)
            if mid is not None:
                ids.append(int(mid))
    if not ids:
        return "", ""
    counts = pd.Series(ids).value_counts().sort_values(ascending=False)
    total = float(counts.sum())

    def fmt(mid: int, count: int) -> str:
        return f"{id_to_abbr.get(int(mid), str(int(mid)))} {float(count) * 100.0 / max(total, 1.0):.1f}%"

    pairs = list(counts.items())[:2]
    first = fmt(int(pairs[0][0]), int(pairs[0][1])) if len(pairs) >= 1 else ""
    second = fmt(int(pairs[1][0]), int(pairs[1][1])) if len(pairs) >= 2 else ""
    return first, second


def _empty_summary(method: str, note: str) -> dict[str, object]:
    row = {col: np.nan for col in TABLE1_DISPLAY_COLUMNS if col != "Method"}
    row["Method"] = method
    row["N_VALID_EXPECTED_DS"] = f"0 {note}".strip()
    row["_note"] = note
    row["_valid_expected_ds"] = 0
    row["_route_valid_n"] = 0
    row["_e2e_valid_n"] = 0
    row["_timing_component_stats"] = {}
    return row


def _summarize_selection(
    method: str,
    selected_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    ordered_model_names: list[str],
    expected: set[str],
    route_from_selector: bool,
) -> dict[str, object]:
    if selected_df is None or selected_df.empty:
        return _empty_summary(method, "missing result rows")

    abbr_to_id, id_to_abbr = _model_id_maps(ordered_model_names)
    work = selected_df.copy()
    if "dataset" not in work.columns:
        return _empty_summary(method, "missing dataset column")
    work["dataset"] = work["dataset"].astype(str)
    work = work[work["dataset"].isin(expected)].copy()
    if work.empty:
        return _empty_summary(method, "no expected dataset rows")
    for metric in [
        "MASE",
        "sMAPE",
        "CRPS",
        "sample_seconds",
        "sample_to_route_seconds",
        "route_final_seconds",
        "task_probe_forward_seconds",
        "task_probe_rank_seconds",
        "forward_runtime_seconds",
        "non_eval_runtime_seconds",
        "runtime_seconds",
    ]:
        if metric in work.columns:
            work[metric] = pd.to_numeric(work[metric], errors="coerce")

    row: dict[str, object] = {"Method": method}
    for metric in ["MASE", "CRPS", "sMAPE"]:
        row[metric] = float(pd.to_numeric(work.get(metric, pd.Series(dtype=float)), errors="coerce").mean())
    regret_mean, regret_p90 = _mase_regret_stats(work, baseline_df)
    row["Regret-M"] = regret_mean
    row["Regret-M P90"] = regret_p90
    for _display_name, metric in ROUTE_MEAN_METRICS:
        values = pd.to_numeric(work.get(metric, pd.Series(dtype=float)), errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        row[f"_mean_{metric}"] = float(values.mean()) if not values.dropna().empty else np.nan

    rank_m, hit_m1, hit_m3 = _metric_rank_and_hit(work, baseline_df, "MASE", abbr_to_id)
    rank_c, hit_c1, hit_c3 = _metric_rank_and_hit(work, baseline_df, "CRPS", abbr_to_id)
    row["Rank-M"] = rank_m
    row["Rank-C"] = rank_c
    row["MASE-hit1"] = hit_m1
    row["MASE-hit3"] = hit_m3
    row["CRPS-hit1"] = hit_c1
    row["CRPS-hit3"] = hit_c3
    row["MASE-hit1/3"] = _format_pair(hit_m1, hit_m3)
    row["CRPS-hit1/3"] = _format_pair(hit_c1, hit_c3)
    row["Count-1"], row["Count-2"] = _selected_counts(work, abbr_to_id, id_to_abbr)
    row["TWW1/3↑"] = _format_pair(
        work["TEST_WINDOW_TOP1_ACC"].mean() if "TEST_WINDOW_TOP1_ACC" in work.columns else np.nan,
        work["TEST_WINDOW_TOP3_HIT"].mean() if "TEST_WINDOW_TOP3_HIT" in work.columns else np.nan,
    )
    row["PWW1/3↑"] = _format_pair(
        work["ENC_TOP1_SUBSET_RATE"].mean() if "ENC_TOP1_SUBSET_RATE" in work.columns else np.nan,
        work["ENC_TOP3_SUBSET_RATE"].mean() if "ENC_TOP3_SUBSET_RATE" in work.columns else np.nan,
    )
    row["TCC1/3↑"] = _format_pair(
        work["SINGLE_TOP1_ACC"].mean() if "SINGLE_TOP1_ACC" in work.columns else np.nan,
        work["SINGLE_TOP3_HIT"].mean() if "SINGLE_TOP3_HIT" in work.columns else np.nan,
    )
    row["TWC1/3↑"] = _format_pair(
        work["TEST_WINDOW_CHANNEL_TOP1_ACC"].mean() if "TEST_WINDOW_CHANNEL_TOP1_ACC" in work.columns else np.nan,
        work["TEST_WINDOW_CHANNEL_TOP3_HIT"].mean() if "TEST_WINDOW_CHANNEL_TOP3_HIT" in work.columns else np.nan,
    )
    row["TWR1/3↑"] = _format_pair(
        work["TEST_WINDOW_TASK_TOP1_ACC"].mean() if "TEST_WINDOW_TASK_TOP1_ACC" in work.columns else np.nan,
        work["TEST_WINDOW_TASK_TOP3_HIT"].mean() if "TEST_WINDOW_TASK_TOP3_HIT" in work.columns else np.nan,
    )
    row["TCR1/3↑"] = _format_pair(
        work["TEST_CHANNEL_TASK_TOP1_ACC"].mean() if "TEST_CHANNEL_TASK_TOP1_ACC" in work.columns else np.nan,
        work["TEST_CHANNEL_TASK_TOP3_HIT"].mean() if "TEST_CHANNEL_TASK_TOP3_HIT" in work.columns else np.nan,
    )

    route_seconds = pd.Series(dtype=float)
    core_route_seconds = pd.Series(dtype=float)
    if route_from_selector and {
        "sample_seconds",
        "task_probe_forward_seconds",
        "task_probe_rank_seconds",
        "route_final_seconds",
    }.issubset(work.columns):
        sample = pd.to_numeric(work["sample_seconds"], errors="coerce")
        forward = pd.to_numeric(
            work["task_probe_forward_seconds"], errors="coerce"
        )
        rank = pd.to_numeric(work["task_probe_rank_seconds"], errors="coerce")
        route_final = pd.to_numeric(work["route_final_seconds"], errors="coerce")
        valid_route = (
            np.isfinite(sample)
            & np.isfinite(forward)
            & np.isfinite(rank)
            & np.isfinite(route_final)
            & sample.ge(0)
            & forward.ge(0)
            & rank.ge(0)
            & route_final.ge(0)
            & (route_final - sample - forward - rank).abs().le(1e-6)
        )
        core_route_seconds = (forward + rank).loc[valid_route].dropna()
        route_seconds = route_final.loc[valid_route].dropna()
    elif route_from_selector and {
        "sample_seconds",
        "sample_to_route_seconds",
        "route_final_seconds",
    }.issubset(work.columns):
        sample = pd.to_numeric(work["sample_seconds"], errors="coerce")
        sample_to_route = pd.to_numeric(
            work["sample_to_route_seconds"], errors="coerce"
        )
        route_final = pd.to_numeric(work["route_final_seconds"], errors="coerce")
        valid_route = (
            np.isfinite(sample)
            & np.isfinite(sample_to_route)
            & np.isfinite(route_final)
            & sample.ge(0)
            & sample_to_route.ge(0)
            & route_final.ge(0)
            & (route_final - sample - sample_to_route).abs().le(1e-6)
        )
        core_route_seconds = sample_to_route.loc[valid_route].dropna()
        route_seconds = route_final.loc[valid_route].dropna()
    elif route_from_selector and "route_final_seconds" in work.columns:
        route_seconds = pd.to_numeric(
            work["route_final_seconds"], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        route_seconds = route_seconds[route_seconds.ge(0)].dropna()
    row["_route_valid_n"] = int(len(route_seconds))
    row["Route P50(s)"] = _round2(route_seconds.quantile(0.50)) if not route_seconds.empty else np.nan
    row["Route P95(s)"] = _round2(route_seconds.quantile(0.95)) if not route_seconds.empty else np.nan
    row["_core_route_valid_n"] = int(len(core_route_seconds))
    row["Core-route P50(s)"] = (
        _round2(core_route_seconds.quantile(0.50))
        if not core_route_seconds.empty
        else np.nan
    )
    row["Core-route P95(s)"] = (
        _round2(core_route_seconds.quantile(0.95))
        if not core_route_seconds.empty
        else np.nan
    )
    if len(route_seconds) == len(expected) and float(route_seconds.sum()) > 0:
        row["Route throughput(req/min)"] = _round2(
            float(len(route_seconds)) * 60.0 / float(route_seconds.sum())
        )
    else:
        row["Route throughput(req/min)"] = np.nan

    baseline_lookup = {
        (str(rec["dataset"]), str(rec["model"])): rec
        for _, rec in baseline_df.drop_duplicates(["dataset", "model"], keep="last").iterrows()
        if pd.notna(rec.get("dataset")) and pd.notna(rec.get("model"))
    }
    selected_forward_vals = []
    selected_route_vals = []
    e2e_vals = []
    for _, rec in work.iterrows():
        forward_s = (
            _first_selected_forward_seconds(rec, baseline_lookup, id_to_abbr, abbr_to_id)
            if route_from_selector
            else _forward_seconds_for_row(rec)
        )
        if not np.isfinite(forward_s):
            continue
        selected_forward_vals.append(float(forward_s))
        route_s = _to_numeric_or_nan(rec.get("route_final_seconds", np.nan)) if route_from_selector else 0.0
        if route_from_selector and not np.isfinite(route_s):
            continue
        selected_route_vals.append(float(route_s))
        e2e_vals.append(float(route_s) + float(forward_s))
    forward_series = pd.Series(selected_forward_vals, dtype=float)
    route_series = pd.Series(selected_route_vals, dtype=float)
    e2e_series = pd.Series(e2e_vals, dtype=float)

    component_stats: dict[str, dict[str, object]] = {}
    if {
        "sample_seconds",
        "task_probe_forward_seconds",
        "task_probe_rank_seconds",
        "route_final_seconds",
    }.issubset(work.columns):
        sample = pd.to_numeric(work["sample_seconds"], errors="coerce")
        forward = pd.to_numeric(work["task_probe_forward_seconds"], errors="coerce")
        rank = pd.to_numeric(work["task_probe_rank_seconds"], errors="coerce")
        route_final = pd.to_numeric(work["route_final_seconds"], errors="coerce")
        sample_to_route = forward + rank
        valid_components = (
            np.isfinite(sample)
            & np.isfinite(sample_to_route)
            & np.isfinite(route_final)
            & sample.ge(0)
            & sample_to_route.ge(0)
            & route_final.ge(0)
            & (route_final - sample - sample_to_route).abs().le(1e-6)
        )
        component_stats["sample_seconds"] = _timing_stats(sample.loc[valid_components])
        component_stats["sample_to_route_seconds"] = _timing_stats(
            sample_to_route.loc[valid_components]
        )
        component_stats["route_total_seconds"] = _timing_stats(
            route_final.loc[valid_components]
        )
    elif {
        "sample_seconds",
        "sample_to_route_seconds",
        "route_final_seconds",
    }.issubset(work.columns):
        sample = pd.to_numeric(work["sample_seconds"], errors="coerce")
        sample_to_route = pd.to_numeric(
            work["sample_to_route_seconds"], errors="coerce"
        )
        route_final = pd.to_numeric(work["route_final_seconds"], errors="coerce")
        valid_components = (
            np.isfinite(sample)
            & np.isfinite(sample_to_route)
            & np.isfinite(route_final)
            & sample.ge(0)
            & sample_to_route.ge(0)
            & route_final.ge(0)
            & (route_final - sample - sample_to_route).abs().le(1e-6)
        )
        component_stats["sample_seconds"] = _timing_stats(sample.loc[valid_components])
        component_stats["sample_to_route_seconds"] = _timing_stats(
            sample_to_route.loc[valid_components]
        )
        component_stats["route_total_seconds"] = _timing_stats(
            route_final.loc[valid_components]
        )
    component_stats["e2e_total_seconds"] = _timing_stats(e2e_series)
    row["_timing_component_stats"] = component_stats
    row["_selected_forward_valid_n"] = int(len(forward_series))
    row["_total_selected_forward_time_s"] = float(forward_series.sum()) if not forward_series.empty else np.nan
    row["_selected_forward_p95_s"] = float(forward_series.quantile(0.95)) if not forward_series.empty else np.nan
    row["_total_route_time_s"] = float(route_series.sum()) if not route_series.empty else np.nan
    row["_selector_route_p95_s"] = float(route_series.quantile(0.95)) if not route_series.empty else np.nan
    row["_total_e2e_time_s"] = float(e2e_series.sum()) if not e2e_series.empty else np.nan
    row["_e2e_valid_n"] = int(len(e2e_series))
    row["E2E P50(s)"] = _round2(e2e_series.quantile(0.50)) if not e2e_series.empty else np.nan
    row["E2E P95(s)"] = _round2(e2e_series.quantile(0.95)) if not e2e_series.empty else np.nan

    valid_unique = len(set(work["dataset"].dropna().astype(str)) & expected)
    row["N_VALID_EXPECTED_DS"] = int(valid_unique)
    row["_valid_expected_ds"] = int(valid_unique)
    for metric in TSROUTER_CORE_METRIC_COLUMNS:
        if metric in work.columns and metric not in row:
            row[metric] = float(pd.to_numeric(work[metric], errors="coerce").mean())
    return row


def _select_rows_for_model(
    method: str,
    model_name: str,
    baseline_df: pd.DataFrame,
    ordered_model_names: list[str],
) -> pd.DataFrame:
    abbr_to_id, _ = _model_id_maps(ordered_model_names)
    sub = baseline_df[baseline_df["model"].astype(str).eq(str(model_name))].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.drop_duplicates("dataset", keep="last")
    sub["_selected_tsfm_model"] = str(model_name)
    sub["selected_model_id"] = abbr_to_id.get(str(model_name), np.nan)
    sub["model_order"] = sub["selected_model_id"].map(lambda x: [int(x)] if pd.notna(x) else [])
    sub["model"] = method
    return sub


def _random_selection(
    baseline_df: pd.DataFrame,
    current_model_names: list[str],
    ordered_model_names: list[str],
    expected: set[str],
) -> pd.DataFrame:
    abbr_to_id, _ = _model_id_maps(ordered_model_names)
    lookup = {
        (str(rec["dataset"]), str(rec["model"])): rec
        for _, rec in baseline_df.drop_duplicates(["dataset", "model"], keep="last").iterrows()
    }
    rows = []
    datasets = sorted(expected & set(baseline_df["dataset"].dropna().astype(str).unique()))
    for seed in FAST_BASELINE_RANDOM_SEEDS:
        rng = np.random.default_rng(int(seed))
        for ds in datasets:
            selected_model = str(rng.choice(current_model_names))
            rec = lookup.get((ds, selected_model))
            if rec is None:
                continue
            row = rec.copy()
            row["_seed"] = int(seed)
            row["_selected_tsfm_model"] = selected_model
            row["selected_model_id"] = abbr_to_id.get(selected_model, np.nan)
            row["model_order"] = [int(row["selected_model_id"])] if pd.notna(row["selected_model_id"]) else []
            row["model"] = "Random"
            rows.append(row)
    return pd.DataFrame(rows)


def _recent_model(current_model_names: list[str]) -> str | None:
    return str(current_model_names[-1]) if current_model_names else None


def _model_key_for_abbr(model_name: str) -> str:
    target = str(model_name)
    for key in All_sorted_model_names:
        if str(Model_abbrev_map.get(key, key)) == target:
            return str(key)
    return target


def _tsfm_result_source_for_model(args, model_name: str | None) -> str:
    if not model_name:
        return "baseline_df_all: missing selected TSFM"
    model_key = _model_key_for_abbr(str(model_name))
    tsfm_dir = str(getattr(args, "TSFM_results_dir", "cl_512"))
    return resolve_tsfm_csv_path(model_key, tsfm_dir, "all_results.csv").as_posix()


def _tsfm_stage_source(args, current_model_names: list[str]) -> str:
    tsfm_dir = str(getattr(args, "TSFM_results_dir", "cl_512"))
    if not current_model_names:
        return f"baseline_df_all: results_csv/TSFM/{tsfm_dir}/*/all_results.csv (no active models)"
    return (
        f"baseline_df_all: results_csv/TSFM/{tsfm_dir}/<{len(current_model_names)} stage models>/all_results.csv; "
        f"models={','.join(current_model_names[:3])}{'...' if len(current_model_names) > 3 else ''}"
    )


def _current_best_model(
    baseline_df: pd.DataFrame,
    current_model_names: list[str],
    metric: str,
) -> tuple[str | None, str]:
    if baseline_df is None or baseline_df.empty:
        return None, "missing TSFM baseline rows"
    if metric not in baseline_df.columns:
        return None, f"missing TSFM metric column: {metric}"
    sub = baseline_df[baseline_df["model"].astype(str).isin([str(m) for m in current_model_names])].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    vals = sub.groupby("model")[metric].mean(numeric_only=True).replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return None, f"no valid TSFM {metric} rows"
    order = {str(model): idx for idx, model in enumerate(current_model_names)}
    best = sorted(vals.items(), key=lambda item: (float(item[1]), order.get(str(item[0]), 10**9)))[0]
    return str(best[0]), f"test-set current best by {metric}: model={best[0]} mean_{metric}={float(best[1]):.6g}"


def _probe_static_best_model(
    args,
    stage: int,
    current_model_names: list[str],
    metric: str,
) -> tuple[str | None, str]:
    path = _main_repr_forward_path(args, stage)
    if not path.exists():
        return None, f"missing repr_forward: {path.as_posix()}"
    scores: dict[str, float] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = [part.strip() for part in line.rstrip("\n").split(",")]
                if len(parts) < 3 or parts[1] != metric:
                    continue
                model = Model_abbrev_map.get(parts[0], parts[0])
                if model not in current_model_names:
                    continue
                vals = pd.to_numeric(pd.Series(parts[2:]), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if vals.empty:
                    continue
                scores[str(model)] = float(vals.mean())
    except Exception as exc:
        return None, f"cannot read repr_forward: {exc}"
    if not scores:
        return None, f"no {metric} rows in repr_forward"
    order = {model: idx for idx, model in enumerate(current_model_names)}
    best = sorted(scores.items(), key=lambda item: (item[1], order.get(item[0], 10**9)))[0][0]
    return best, f"{path.as_posix()} mean_{metric}={scores[best]:.6g}"


def _selector_stage_summary_row(
    args,
    stage: int,
    *,
    method: str,
    baseline_df: pd.DataFrame,
    ordered_model_names: list[str],
    expected: set[str],
    auto_cl: bool = False,
    param_overrides: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    selected_df, note = _main_selector_selection(
        args,
        stage,
        auto_cl=auto_cl,
        param_overrides=param_overrides,
    )
    if auto_cl:
        selected_df = _attach_auto_cl_selected_forward(
            selected_df,
            ordered_model_names,
        )
    row = _summarize_selection(
        method,
        selected_df,
        baseline_df,
        ordered_model_names,
        expected,
        route_from_selector=True,
    )
    row["_source"] = note
    check = _source_check_row(
        stage=stage,
        method=method,
        source_kind="selector-result",
        source=note,
        selected_df=selected_df,
        expected=expected,
        summary_row=row,
        note=note if selected_df.empty else "",
        auto_cl_mode=_summary_auto_cl_mode(args) if auto_cl else "v0",
    )
    return selected_df, row, check


def _main_selector_selection(
    args,
    stage: int,
    *,
    auto_cl: bool = False,
    param_overrides: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, str]:
    path = _main_selector_result_path(
        args,
        stage,
        auto_cl=auto_cl,
        param_overrides=param_overrides,
    )
    display_path = path.resolve()
    if not path.exists():
        return pd.DataFrame(), f"missing selector result: {display_path.as_posix()}"
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return pd.DataFrame(), f"cannot read selector result: {display_path.as_posix()}: {exc}"
    return df, display_path.as_posix()


def _auto_cl_source_audit(
    selected_df: pd.DataFrame,
    expected: set[str],
    auto_cl_mode: str = "v1",
) -> dict[str, object]:
    required_profiles = [
        str(profile["adaptive_profile"]) for profile in get_auto_cl_profiles(auto_cl_mode)
    ]
    empty = {
        "AutoCLMode": auto_cl_mode,
        "RequiredProfiles": ",".join(required_profiles),
        "ObservedProfiles": "",
        "ProfileComplete": False,
        "ResolvedCLComplete": False,
        "RankTruthCLComplete": False,
        "TSFMForwardCLComplete": False,
        "ProfileDatasetCounts": "",
        "LegacyCLDerivedN": 0,
        "EvalCLFallbackN": 0,
    }
    if (
        selected_df is None
        or selected_df.empty
        or "dataset" not in selected_df.columns
    ):
        return empty
    work = selected_df[
        selected_df["dataset"].astype(str).isin(expected)
    ].copy()
    work["_file_order"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values("_file_order").drop_duplicates("dataset", keep="last")
    observed = sorted(
        {
            str(value)
            for value in work.get("adaptive_profile", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            if str(value)
        }
    )
    counts = (
        work.get("adaptive_profile", pd.Series(dtype=str))
        .astype(str)
        .value_counts()
        .to_dict()
    )
    profile_ok = set(observed) == set(required_profiles)
    resolved_ok = True
    rank_ok = True
    legacy_derived_n = 0
    for _, rec in work.iterrows():
        profile = get_auto_cl_profile_by_name(
            str(rec.get("adaptive_profile", "") or ""),
            auto_cl_mode,
        )
        if profile is None:
            legacy_derived_n += 1
            resolved_ok = False
            rank_ok = False
            continue
        expected_cl = str(profile["tsfm_results_dir"])
        expected_dims = {
            "repr_input_dim": int(profile["repr_input_dim"]),
            "repr_output_dim": int(profile["repr_output_dim"]),
            "repr_sub_pred_len": int(profile["repr_sub_pred_len"]),
            "repr_source_exact_length": int(profile["repr_source_exact_length"]),
        }
        if str(rec.get("resolved_eval_cl", "") or "") != expected_cl:
            resolved_ok = False
        if str(rec.get("rank_truth_cl", "") or "") != expected_cl:
            rank_ok = False
        for col, expected_value in expected_dims.items():
            actual = _to_numeric_or_nan(rec.get(col, np.nan))
            if not np.isfinite(actual) or int(actual) != expected_value:
                profile_ok = False
    fallback_text = (
        work.get("eval_cl_fallback_used", pd.Series(False, index=work.index))
        .astype(str)
        .str.strip()
        .str.lower()
    )
    fallback_n = int(fallback_text.isin({"true", "1", "yes", "y", "t"}).sum())
    auto_mode_ok = (
        "auto_cl_mode" in work.columns
        and work["auto_cl_mode"].astype(str).eq(auto_cl_mode).all()
    )
    forward_ok = (
        "_tsfm_forward_cl_ok" in work.columns
        and work["_tsfm_forward_cl_ok"].map(_truthy).all()
    )
    return {
        "AutoCLMode": auto_cl_mode if auto_mode_ok else "invalid",
        "RequiredProfiles": ",".join(required_profiles),
        "ObservedProfiles": ",".join(observed),
        "ProfileComplete": bool(profile_ok),
        "ResolvedCLComplete": bool(resolved_ok),
        "RankTruthCLComplete": bool(rank_ok),
        "TSFMForwardCLComplete": bool(forward_ok),
        "ProfileDatasetCounts": ",".join(
            f"{profile}:{int(counts.get(profile, 0))}"
            for profile in required_profiles
        ),
        "LegacyCLDerivedN": int(legacy_derived_n),
        "EvalCLFallbackN": int(fallback_n),
    }


def _source_check_row(
    stage: int,
    method: str,
    source_kind: str,
    source: str,
    selected_df: pd.DataFrame,
    expected: set[str],
    summary_row: dict[str, object],
    selected_model: str = "",
    note: str = "",
    auto_cl_mode: str = "v0",
) -> dict[str, object]:
    rows = int(len(selected_df)) if selected_df is not None else 0
    valid = int(summary_row.get("_valid_expected_ds", 0) or 0)
    expected_n = int(len(expected))
    selected_work = selected_df.copy() if selected_df is not None else pd.DataFrame()
    if "dataset" in selected_work.columns:
        selected_work["dataset"] = selected_work["dataset"].astype(str)
        present_datasets = set(selected_work["dataset"].dropna().astype(str)) & expected
    else:
        present_datasets = set()
    missing_datasets = sorted(expected - present_datasets)
    metric_missing: dict[str, list[str]] = {}
    for metric in ["MASE", "CRPS"]:
        if metric not in selected_work.columns or "dataset" not in selected_work.columns:
            metric_missing[metric] = sorted(expected)
            continue
        metric_vals = pd.to_numeric(selected_work[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid_metric_datasets = set(selected_work.loc[metric_vals.notna(), "dataset"].astype(str)) & expected
        missing = sorted(expected - valid_metric_datasets)
        if missing:
            metric_missing[metric] = missing
    metric_complete = not metric_missing
    route_valid_n = int(summary_row.get("_route_valid_n", 0) or 0)
    e2e_valid_n = int(summary_row.get("_e2e_valid_n", 0) or 0)
    route_complete = route_valid_n >= expected_n if source_kind == "selector-result" else True
    e2e_complete = e2e_valid_n >= expected_n
    runtime_complete = bool(route_complete and e2e_complete)
    row_note = str(summary_row.get("_note", "") or note or "")
    if valid != expected_n and not row_note:
        row_note = "partial expected dataset coverage"
    auto_cl_method = (
        "autocl" in method.lower()
        or (
            "auto_cl_mode" in selected_work.columns
            and selected_work["auto_cl_mode"].astype(str).ne("v0").any()
        )
    )
    auto_audit = (
        _auto_cl_source_audit(selected_work, expected, auto_cl_mode)
        if auto_cl_method
        else {
            "AutoCLMode": "v0",
            "RequiredProfiles": "",
            "ObservedProfiles": "",
            "ProfileComplete": True,
            "ResolvedCLComplete": True,
            "RankTruthCLComplete": True,
            "TSFMForwardCLComplete": True,
            "ProfileDatasetCounts": "",
            "LegacyCLDerivedN": 0,
            "EvalCLFallbackN": 0,
        }
    )
    paper_complete = (
        bool(auto_audit["ProfileComplete"])
        and bool(auto_audit["ResolvedCLComplete"])
        and bool(auto_audit["RankTruthCLComplete"])
        and bool(auto_audit["TSFMForwardCLComplete"])
        and int(auto_audit["LegacyCLDerivedN"]) == 0
        and int(auto_audit["EvalCLFallbackN"]) == 0
    )
    out = {
        "Stage": int(stage),
        "Method": method,
        "SourceKind": source_kind,
        "Source": str(source),
        "Rows": rows,
        "ValidExpected": valid,
        "Expected": expected_n,
        "Complete": bool(
            valid == expected_n
            and metric_complete
            and runtime_complete
            and paper_complete
        ),
        "MetricComplete": bool(metric_complete),
        "RouteComplete": bool(route_complete),
        "E2EComplete": bool(e2e_complete),
        "RuntimeComplete": bool(runtime_complete),
        "RouteValidN": route_valid_n,
        "E2EValidN": e2e_valid_n,
        "SelectedModel": str(selected_model or summary_row.get("_selected_model", "") or ""),
        "MissingDatasets": ",".join(missing_datasets),
        "MetricMissingDatasets": ";".join(
            f"{metric}:{','.join(datasets)}" for metric, datasets in metric_missing.items()
        ),
        **auto_audit,
        "Note": row_note,
    }
    return out


def _format_stage_ranges(stages: Iterable[int]) -> str:
    ordered = sorted({int(stage) for stage in stages})
    if not ordered:
        return ""
    ranges: list[str] = []
    start = ordered[0]
    end = ordered[0]
    for stage in ordered[1:]:
        if stage == end + 1:
            end = stage
            continue
        ranges.append(f"z{start}" if start == end else f"z{start}-z{end}")
        start = end = stage
    ranges.append(f"z{start}" if start == end else f"z{start}-z{end}")
    return ",".join(ranges)


def _method_success_dir(args, method: str) -> str:
    if method == "TSRouter-main":
        return f"{Path(get_tsrouter_selector_result_dir(args)).resolve().as_posix()}/"
    if method in {"TSRouter-fast", "AutoForecast", "AutoXPCR", "SimpleTS", *PROFILE_PROBE_METHODS}:
        method_args = _args_for_stage(
            args,
            int(getattr(args, "zoo_total_num", 0) or 0),
            auto_cl=(method in PROFILE_PROBE_METHODS and _summary_auto_cl_enabled(args)),
            param_overrides=_method_param_overrides(method),
        )
        return f"{Path(get_tsrouter_selector_result_dir(method_args)).resolve().as_posix()}/"
    if method == "TSRouter-autocl":
        auto_args = _args_for_stage(
            args,
            int(getattr(args, "zoo_total_num", 0) or 0),
            auto_cl=True,
        )
        return (
            f"{Path(get_tsrouter_selector_result_dir(auto_args)).resolve().as_posix()}/"
        )
    if method.startswith("Task-probe"):
        return f"{task_probe_select_root(args).resolve().as_posix()}/"
    tsfm_dir = str(getattr(args, "TSFM_results_dir", "cl_512"))
    return tsfm_csv_glob_display(tsfm_dir, root=Path(TSFM_CSV_ROOT).resolve())


def _missing_file_from_check(row: dict[str, object]) -> str:
    source = str(row.get("Source", "") or "")
    for prefix in ["missing selector result: ", "cannot read selector result: "]:
        if source.startswith(prefix):
            value = source[len(prefix):]
            return value.split(": ", maxsplit=1)[0] if prefix.startswith("cannot read") else value
    source_path = source.split("; ", maxsplit=1)[0]
    if source_path.endswith(".csv") and not Path(source_path).exists():
        return source_path
    return ""


def _check_issue_summary(row: dict[str, object]) -> str:
    issues = []
    if not bool(row.get("MetricComplete", False)):
        issues.append("metric")
    if not bool(row.get("RouteComplete", False)):
        issues.append(f"route={row.get('RouteValidN', 0)}/{row.get('Expected', 0)}")
    if not bool(row.get("E2EComplete", False)):
        issues.append(f"e2e={row.get('E2EValidN', 0)}/{row.get('Expected', 0)}")
    if int(row.get("ValidExpected", 0) or 0) != int(row.get("Expected", 0) or 0):
        issues.append(f"datasets={row.get('ValidExpected', 0)}/{row.get('Expected', 0)}")
    note = str(row.get("Note", "") or "")
    if note and note not in issues:
        issues.append(note)
    return ",".join(issues) if issues else "incomplete"


def _print_source_checks_by_method(args, rows: list[dict[str, object]]) -> None:
    by_method: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_method.setdefault(str(row.get("Method", "")), []).append(row)
    summary_methods = _summary_methods(args)
    ordered_methods = summary_methods + sorted(
        set(by_method) - set(summary_methods)
    )
    for method in ordered_methods:
        method_rows = sorted(by_method.get(method, []), key=lambda row: int(row.get("Stage", 0)))
        if not method_rows:
            continue
        complete_rows = [row for row in method_rows if bool(row.get("Complete", False))]
        incomplete_rows = [row for row in method_rows if not bool(row.get("Complete", False))]
        all_stages = _format_stage_ranges(int(row["Stage"]) for row in method_rows)
        if not incomplete_rows:
            print(
                f"✅ [vldb-check] method={method} stages={all_stages} "
                f"check_dir={_method_success_dir(args, method)}"
            )
            continue

        print(
            f"⚠️ [vldb-check] method={method} "
            f"failed_stages={_format_stage_ranges(int(row['Stage']) for row in incomplete_rows)}"
        )
        if complete_rows:
            print(
                f"  [ok-dir] stages={_format_stage_ranges(int(row['Stage']) for row in complete_rows)} "
                f"path={_method_success_dir(args, method)}"
            )
        for row in incomplete_rows:
            stage = int(row["Stage"])
            missing_file = _missing_file_from_check(row)
            issue = _check_issue_summary(row)
            if missing_file:
                print(f"  [missing-file] stage=z{stage} path={missing_file}")
            else:
                print(f"  [failed-source] stage=z{stage} source={row['Source']} issue={issue}")


def _build_stage_method_summary(
    args,
    main_args,
    baseline_df: pd.DataFrame,
    current_model_names: list[str],
    ordered_model_names: list[str],
    expected: set[str],
    stage: int,
    method: str,
    auto_cl: bool,
    main_df_for_task_probe: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    main_method = "TSRouter-autocl" if auto_cl else "TSRouter-main"
    if method == main_method:
        return _selector_stage_summary_row(
            args,
            stage,
            method=main_method,
            baseline_df=baseline_df,
            ordered_model_names=ordered_model_names,
            expected=expected,
            auto_cl=auto_cl,
        )

    if method in {"TSRouter-fast", "AutoForecast", "AutoXPCR", "SimpleTS"}:
        return _selector_stage_summary_row(
            args,
            stage,
            method=method,
            baseline_df=baseline_df,
            ordered_model_names=ordered_model_names,
            expected=expected,
            auto_cl=False,
            param_overrides=_method_param_overrides(method),
        )

    if method in PROFILE_PROBE_METHODS:
        return _selector_stage_summary_row(
            args,
            stage,
            method=method,
            baseline_df=baseline_df,
            ordered_model_names=ordered_model_names,
            expected=expected,
            auto_cl=auto_cl,
            param_overrides=_method_param_overrides(method),
        )

    if method in {"Task-probe-M", "Task-probe-C"}:
        metric = "MASE" if method.endswith("-M") else "CRPS"
        selector_df = main_df_for_task_probe
        if auto_cl and selector_df is None:
            selector_df, _selector_note = _main_selector_selection(
                args,
                stage,
                auto_cl=True,
            )
        task_probe_df, task_probe_note = task_probe_select_selection_for_stage(
            args=main_args,
            stage=stage,
            metric=metric,
            baseline_df=baseline_df,
            ordered_model_names=ordered_model_names,
            expected=expected,
            selector_df=selector_df if auto_cl else None,
        )
        task_probe_row = _summarize_selection(
            method,
            task_probe_df,
            baseline_df,
            ordered_model_names,
            expected,
            route_from_selector=True,
        )
        task_probe_row["_source"] = task_probe_note
        task_probe_check = _source_check_row(
            stage=stage,
            method=method,
            source_kind="selector-result",
            source=task_probe_note,
            selected_df=task_probe_df,
            expected=expected,
            summary_row=task_probe_row,
            note=task_probe_note if task_probe_df.empty else "",
            auto_cl_mode=_summary_auto_cl_mode(args),
        )
        return task_probe_df, task_probe_row, task_probe_check

    if method == "Random":
        random_df = _random_selection(
            baseline_df,
            current_model_names,
            ordered_model_names,
            expected,
        )
        random_row = _summarize_selection(
            "Random",
            random_df,
            baseline_df,
            ordered_model_names,
            expected,
            route_from_selector=False,
        )
        random_row["_source"] = (
            f"{_tsfm_stage_source(args, current_model_names)}; "
            f"random_seeds={','.join(map(str, FAST_BASELINE_RANDOM_SEEDS))}"
        )
        random_check = _source_check_row(
            stage=stage,
            method="Random",
            source_kind="fast-baseline-runtime",
            source=random_row["_source"],
            selected_df=random_df,
            expected=expected,
            summary_row=random_row,
            auto_cl_mode=_summary_auto_cl_mode(args),
        )
        return random_df, random_row, random_check

    if method == "Recent":
        recent = _recent_model(current_model_names)
        recent_df = (
            _select_rows_for_model("Recent", recent, baseline_df, ordered_model_names)
            if recent
            else pd.DataFrame()
        )
        recent_row = _summarize_selection(
            "Recent",
            recent_df,
            baseline_df,
            ordered_model_names,
            expected,
            route_from_selector=False,
        )
        recent_row["_source"] = _tsfm_result_source_for_model(args, recent)
        recent_row["_selected_model"] = recent or ""
        recent_check = _source_check_row(
            stage=stage,
            method="Recent",
            source_kind="fast-baseline-runtime",
            source=recent_row["_source"],
            selected_df=recent_df,
            expected=expected,
            summary_row=recent_row,
            selected_model=recent or "",
            auto_cl_mode=_summary_auto_cl_mode(args),
        )
        return recent_df, recent_row, recent_check

    if method in {"Current_best-M", "Current_best-C"}:
        metric = "MASE" if method.endswith("-M") else "CRPS"
        best_model, note = _current_best_model(baseline_df, current_model_names, metric)
        selected = (
            _select_rows_for_model(method, best_model, baseline_df, ordered_model_names)
            if best_model
            else pd.DataFrame()
        )
        row = _summarize_selection(
            method,
            selected,
            baseline_df,
            ordered_model_names,
            expected,
            route_from_selector=False,
        )
        row["_source"] = f"{_tsfm_result_source_for_model(args, best_model)}; {note}"
        row["_selected_model"] = best_model or ""
        check = _source_check_row(
            stage=stage,
            method=method,
            source_kind="fast-baseline-runtime",
            source=row["_source"],
            selected_df=selected,
            expected=expected,
            summary_row=row,
            selected_model=best_model or "",
            note=note,
            auto_cl_mode=_summary_auto_cl_mode(args),
        )
        return selected, row, check

    empty_df = pd.DataFrame()
    row = _empty_summary(method, f"unsupported summary method: {method}")
    check = _source_check_row(
        stage=stage,
        method=method,
        source_kind="unsupported",
        source="",
        selected_df=empty_df,
        expected=expected,
        summary_row=row,
        note=row.get("_note", ""),
        auto_cl_mode=_summary_auto_cl_mode(args),
    )
    return empty_df, row, check


def _stage_summaries(
    args,
    baseline_df_all: pd.DataFrame,
    ordered_model_names: list[str],
    stage: int,
    season_naive_df: pd.DataFrame | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    stage_total_t0 = time.perf_counter()
    phase_stats: list[tuple[str, float, int | None]] = []
    phase_t0 = time.perf_counter()
    baseline_df, current_model_names, expected = _stage_baseline(
        args=args,
        baseline_df_all=baseline_df_all,
        ordered_model_names=ordered_model_names,
        stage=stage,
        season_naive_df=season_naive_df,
    )
    phase_stats.append(("baseline", time.perf_counter() - phase_t0, len(baseline_df)))
    if baseline_df.empty:
        summary_methods = _summary_methods(args)
        rows = [
            _empty_summary(method, "missing TSFM baseline rows")
            for method in summary_methods
        ]
        checks = [
            _source_check_row(
                stage=stage,
                method=method,
                source_kind="tsfm-baseline",
                source=_tsfm_stage_source(args, current_model_names),
                selected_df=pd.DataFrame(),
                expected=expected,
                summary_row=row,
                note="missing TSFM baseline rows",
                auto_cl_mode=_summary_auto_cl_mode(args),
            )
            for method, row in zip(summary_methods, rows)
        ]
        detail = " ".join(
            f"{name}={elapsed:.2f}s(n={count})" if count is not None else f"{name}={elapsed:.2f}s"
            for name, elapsed, count in phase_stats
        )
        print(
            f"[vldb_results][detail] stage=z{int(stage)}-"
            f"{int(getattr(args, 'zoo_total_num', stage))} {detail}; "
            f"total={time.perf_counter() - stage_total_t0:.2f}s; baseline empty",
            flush=True,
        )
        return rows, checks

    rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    auto_cl = _summary_auto_cl_enabled(args)
    main_method = "TSRouter-autocl" if auto_cl else "TSRouter-main"
    main_args = _args_for_stage(args, stage, auto_cl=auto_cl)
    main_df_for_task_probe: pd.DataFrame | None = None
    legacy_by_method: dict[str, tuple[dict[str, object], dict[str, object]]] | None = None

    def get_legacy_by_method() -> dict[str, tuple[dict[str, object], dict[str, object]]]:
        nonlocal legacy_by_method
        if legacy_by_method is None:
            legacy_by_method = _legacy_stage_summary_cache_by_method(
                args=args,
                baseline_df_all=baseline_df_all,
                baseline_df=baseline_df,
                current_model_names=current_model_names,
                ordered_model_names=ordered_model_names,
                stage=int(stage),
                season_naive_df=season_naive_df,
            )
        return legacy_by_method

    for method in _summary_methods(args):
        phase_t0 = time.perf_counter()
        cached = _read_stage_method_summary_cache(
            args=args,
            ordered_model_names=ordered_model_names,
            stage=int(stage),
            method=method,
        )
        selected_df: pd.DataFrame | None = None
        cache_label = method
        if cached is None:
            legacy = get_legacy_by_method().get(method)
            if legacy is not None:
                cached = legacy
                cache_label = f"{method}[stage-cache]"
                _write_stage_method_summary_cache(
                    args=args,
                    baseline_df=baseline_df,
                    current_model_names=current_model_names,
                    ordered_model_names=ordered_model_names,
                    stage=int(stage),
                    method=method,
                    season_naive_df=season_naive_df,
                    row=cached[0],
                    check=cached[1],
                )
        if cached is not None:
            row, check = cached
            rows.append(row)
            checks.append(check)
            try:
                cached_count = int(check.get("Rows", 0))
            except Exception:
                cached_count = None
            phase_stats.append((cache_label, time.perf_counter() - phase_t0, cached_count))
            continue

        selected_df, row, check = _build_stage_method_summary(
            args=args,
            main_args=main_args,
            baseline_df=baseline_df,
            current_model_names=current_model_names,
            ordered_model_names=ordered_model_names,
            expected=expected,
            stage=int(stage),
            method=method,
            auto_cl=auto_cl,
            main_df_for_task_probe=main_df_for_task_probe,
        )
        if method == main_method:
            main_df_for_task_probe = selected_df
        phase_stats.append((method, time.perf_counter() - phase_t0, len(selected_df)))
        rows.append(row)
        checks.append(check)
        _write_stage_method_summary_cache(
            args=args,
            baseline_df=baseline_df,
            current_model_names=current_model_names,
            ordered_model_names=ordered_model_names,
            stage=int(stage),
            method=method,
            season_naive_df=season_naive_df,
            row=row,
            check=check,
        )
    detail = " ".join(
        f"{name}={elapsed:.2f}s(n={count})" if count is not None else f"{name}={elapsed:.2f}s"
        for name, elapsed, count in phase_stats
    )
    print(
        f"[vldb_results][detail] stage=z{int(stage)}-"
        f"{int(getattr(args, 'zoo_total_num', stage))} {detail}; "
        f"total={time.perf_counter() - stage_total_t0:.2f}s",
        flush=True,
    )
    return rows, checks


def _table1_from_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    pair_specs = [
        ("Route P50/95(s)", "Route P50(s)", "Route P95(s)"),
        ("Core-route P50/P95(s)", "Core-route P50(s)", "Core-route P95(s)"),
        ("E2E P50/95(s)", "E2E P50(s)", "E2E P95(s)"),
    ]
    for display_col, left_col, right_col in pair_specs:
        if left_col in table.columns or right_col in table.columns:
            table[display_col] = table.apply(
                lambda row, l=left_col, r=right_col: _format_pair(row.get(l, np.nan), row.get(r, np.nan), decimals=2),
                axis=1,
            )
    for col in TABLE1_DISPLAY_COLUMNS:
        if col not in table.columns:
            table[col] = np.nan if col != "Method" else ""
    return table[TABLE1_DISPLAY_COLUMNS].copy()


def _best_mask(values: pd.Series, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(False, index=values.index)
    best = float(valid.max() if higher_is_better else valid.min())
    return pd.Series(
        np.isclose(numeric.to_numpy(dtype=float), best, rtol=1e-9, atol=1e-12, equal_nan=False),
        index=values.index,
    )


def _mark_numeric_best(values: pd.Series, higher_is_better: bool, decimals: int = 3) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    best = _best_mask(numeric, higher_is_better)
    return pd.Series(
        [
            "" if not np.isfinite(value) else f"{'★' if bool(best.loc[idx]) else ''}{float(value):.{decimals}f}"
            for idx, value in numeric.items()
        ],
        index=values.index,
        dtype=object,
    )


def _split_pair(value) -> tuple[float, float]:
    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    parts = text.split("/", maxsplit=1)
    if len(parts) != 2:
        return np.nan, np.nan
    return _to_numeric_or_nan(parts[0]), _to_numeric_or_nan(parts[1])


def _mark_pair_best(values: pd.Series, higher_is_better: bool, decimals: int) -> pd.Series:
    pairs = values.map(_split_pair)
    left = pairs.map(lambda pair: pair[0])
    right = pairs.map(lambda pair: pair[1])
    left_best = _best_mask(left, higher_is_better)
    right_best = _best_mask(right, higher_is_better)
    marked = []
    for idx, (left_val, right_val) in pairs.items():
        left_text = (
            ""
            if not np.isfinite(left_val)
            else f"{'★' if bool(left_best.loc[idx]) else ''}{left_val:.{decimals}f}"
        )
        right_text = (
            ""
            if not np.isfinite(right_val)
            else f"{'★' if bool(right_best.loc[idx]) else ''}{right_val:.{decimals}f}"
        )
        marked.append(f"{left_text}/{right_text}" if left_text or right_text else "")
    return pd.Series(marked, index=values.index, dtype=object)


def _table1_mark_best(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    numeric_specs = {
        "MASE": (False, 3),
        "Regret-M": (False, 3),
        "Regret-M P90": (False, 3),
        "Rank-M": (False, 3),
        "CRPS": (False, 3),
        "Rank-C": (False, 3),
        "sMAPE": (False, 3),
        "Route throughput(req/min)": (True, 2),
    }
    pair_specs = {
        "MASE-hit1/3": (True, 3),
        "CRPS-hit1/3": (True, 3),
        "PWW1/3↑": (True, 3),
        "TWW1/3↑": (True, 3),
        "TCC1/3↑": (True, 3),
        "TWC1/3↑": (True, 3),
        "TWR1/3↑": (True, 3),
        "TCR1/3↑": (True, 3),
        "Route P50/95(s)": (False, 2),
        "Core-route P50/P95(s)": (False, 2),
        "E2E P50/95(s)": (False, 2),
    }
    for col, (higher_is_better, decimals) in numeric_specs.items():
        if col in display.columns:
            display[col] = _mark_numeric_best(display[col], higher_is_better, decimals)
    for col, (higher_is_better, decimals) in pair_specs.items():
        if col in display.columns:
            display[col] = _mark_pair_best(display[col], higher_is_better, decimals)
    return display


def _table2_mark_best(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for col in display.columns:
        display[col] = _mark_numeric_best(display[col], higher_is_better=(col == "Win_vs_Best_TSFM"))
    return display


def _table2_from_stage_rows(
    args,
    metric: str,
    stage_rows: dict[int, list[dict[str, object]]],
) -> pd.DataFrame:
    stages = sorted(stage_rows.keys())
    zcols = [f"z{stage}-{int(getattr(args, 'zoo_total_num', stage))}" for stage in stages]
    summary_methods = _summary_methods(args)
    table = pd.DataFrame(index=summary_methods, columns=zcols, dtype=float)
    for stage, zcol in zip(stages, zcols):
        by_method = {str(row.get("Method", "")): row for row in stage_rows.get(stage, [])}
        for method in summary_methods:
            table.loc[method, zcol] = _to_numeric_or_nan(by_method.get(method, {}).get(metric, np.nan))
    wins = pd.Series(0, index=table.index, dtype=float)
    reference_method = "Current_best-M" if metric == "MASE" else "Current_best-C"
    for stage, zcol in zip(stages, zcols):
        reference = _to_numeric_or_nan(table.loc[reference_method, zcol])
        if not np.isfinite(reference):
            continue
        vals = pd.to_numeric(table[zcol], errors="coerce")
        wins += vals.le(reference).fillna(False).astype(float)
    table["Win_vs_Best_TSFM"] = wins
    return table


def _vldb_figure_output_dir(args) -> Path:
    raw = str(getattr(args, "vldb_figure_dir", "") or "").strip()
    out_dir = Path(raw) if raw else VLDB_RESULTS_FIGURE_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _vldb_stage_figure_output_dir(base_dir: Path, latest_stage: int) -> Path:
    stage_dir = base_dir / f"stage{int(latest_stage):02d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def _figure_pdf_path(out_path: Path) -> Path:
    return out_path.with_suffix(".pdf")


def _save_figure_outputs(fig, out_path: Path, *, dpi: int = 240) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = _figure_pdf_path(out_path)
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
    fig.savefig(out_path, dpi=dpi, **save_kwargs)
    fig.savefig(pdf_path, **save_kwargs)
    print(f"[vldb_results][figures] write {out_path.as_posix()}", flush=True)
    print(f"[vldb_results][figures] write {pdf_path.as_posix()}", flush=True)
    return out_path


def _record_figure_output(paths: dict[str, Path], key: str, path: Path) -> None:
    paths[key] = path
    pdf_path = _figure_pdf_path(path)
    if pdf_path.exists():
        paths[f"{key}_pdf"] = pdf_path


def _model_release_month_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for family, variants in Model_zoo_details.items():
        for variant, info in variants.items():
            full_name = f"{family}_{variant}"
            abbrev = str(info.get("abbreviation", Model_abbrev_map.get(full_name, full_name)))
            release_date = str(info.get("release_date", "") or "").strip()
            release_month = release_date[:7] if len(release_date) >= 7 else release_date
            out[full_name] = release_month
            out[abbrev] = release_month
    return out


def _family_display_name(family: str) -> str:
    display = {
        "moirai": "Moirai",
        "chronos": "Chronos-bolt",
        "kairos": "Kairos",
        "toto2": "Toto2",
        "moirai2": "Moirai2",
        "flowstate": "FlowState",
        "timesfm": "TimesFM2.5",
        "chronos2": "Chronos2",
        "patchtst": "PatchTST-FM",
        "timemoe": "TimeMoE",
        "toto": "Toto",
        "tirex": "TiRex",
    }
    return display.get(str(family), str(family))


def _model_family_size_metadata() -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for family, variants in Model_zoo_details.items():
        for size_rank, (variant, info) in enumerate(variants.items()):
            full_name = f"{family}_{variant}"
            abbrev = str(info.get("abbreviation", Model_abbrev_map.get(full_name, full_name)))
            payload = {
                "family": str(family),
                "family_display": _family_display_name(str(family)),
                "size_variant": str(variant),
                "size_rank": int(size_rank),
            }
            out[full_name] = payload
            out[abbrev] = payload
    return out


def _figure2_family_representatives(
    ordered_model_names: list[str],
) -> dict[str, dict[str, object]]:
    metadata_by_model = _model_family_size_metadata()
    release_month_by_model = _model_release_month_map()
    explicit_by_family: dict[str, str] = {}
    for _display_name, abbrev in FIGURE2_FAMILY_REPRESENTATIVES:
        meta = metadata_by_model.get(abbrev, {})
        family = str(meta.get("family", ""))
        if family:
            explicit_by_family[family] = abbrev

    family_models: dict[str, list[tuple[int, str, str]]] = {}
    for stage, token in enumerate(ordered_model_names, start=1):
        abbrev = str(Model_abbrev_map.get(str(token), str(token)))
        meta = metadata_by_model.get(str(token), metadata_by_model.get(abbrev, {}))
        family = str(meta.get("family", abbrev))
        family_display = str(meta.get("family_display", _family_display_name(family)))
        family_models.setdefault(family, []).append((int(stage), abbrev, family_display))

    selected: list[tuple[int, str, str, str]] = []
    for family, items in family_models.items():
        explicit_abbrev = explicit_by_family.get(family)
        if explicit_abbrev:
            match = next((item for item in items if item[1] == explicit_abbrev), None)
            if match is not None:
                stage, abbrev, family_display = match
            else:
                stage, abbrev, family_display = items[-1]
        elif len(items) == 1:
            stage, abbrev, family_display = items[0]
        else:
            stage, abbrev, family_display = items[-1]
        selected.append((stage, family, family_display, abbrev))

    selected = sorted(selected, key=lambda item: item[0])
    return {
        abbrev: {
            "x_pos": int(x_pos),
            "stage": int(stage),
            "family": family,
            "family_display": family_display,
            "representative_abbrev": abbrev,
            "release_month": release_month_by_model.get(abbrev, ""),
        }
        for x_pos, (stage, family, family_display, abbrev) in enumerate(selected)
    }


def _stage_model_label(ordered_model_names: list[str], stage: int) -> tuple[str, str, str]:
    if 1 <= int(stage) <= len(ordered_model_names):
        token = str(ordered_model_names[int(stage) - 1])
    else:
        token = f"z{int(stage)}"
    abbrev = str(Model_abbrev_map.get(token, token))
    release_month = _model_release_month_map().get(token, _model_release_month_map().get(abbrev, ""))
    label = f"{abbrev}\n{release_month}" if release_month else abbrev
    return token, abbrev, label


def _figure_method_label(method: str) -> str:
    if method in {"TSRouter-main", "TSRouter-autocl"}:
        return "TSRouter"
    if method == "Task-probe-M":
        return "Task-probe"
    if method == "Profile-probe-M":
        return "Profile-probe"
    if method == "Current_best-M":
        return "Current-best"
    return str(method)


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _selector_forward_e2e_summary(
    selected_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    ordered_model_names: list[str],
    expected: set[str],
    *,
    route_from_selector: bool,
) -> dict[str, object]:
    out = {
        "total_forward_time_s": np.nan,
        "total_e2e_time_s": np.nan,
        "mean_forward_time_s": np.nan,
        "mean_e2e_request_time_s": np.nan,
        "forward_p95_s": np.nan,
        "route_p95_s": np.nan,
        "e2e_p95_s": np.nan,
        "forward_valid_n": 0,
        "e2e_valid_n": 0,
    }
    if selected_df is None or selected_df.empty or "dataset" not in selected_df.columns:
        return out

    abbr_to_id, id_to_abbr = _model_id_maps(ordered_model_names)
    baseline_lookup = {
        (str(rec["dataset"]), str(rec["model"])): rec
        for _, rec in baseline_df.drop_duplicates(["dataset", "model"], keep="last").iterrows()
        if pd.notna(rec.get("dataset")) and pd.notna(rec.get("model"))
    }
    work = selected_df.copy()
    work["dataset"] = work["dataset"].astype(str)
    work = work[work["dataset"].isin(expected)].copy()
    work = work.drop_duplicates("dataset", keep="last")

    forward_vals: list[float] = []
    route_vals: list[float] = []
    e2e_vals: list[float] = []
    for _, rec in work.iterrows():
        forward_s = (
            _first_selected_forward_seconds(rec, baseline_lookup, id_to_abbr, abbr_to_id)
            if route_from_selector
            else _forward_seconds_for_row(rec)
        )
        if np.isfinite(forward_s) and forward_s >= 0:
            forward_vals.append(float(forward_s))
        else:
            continue

        route_s = _to_numeric_or_nan(rec.get("route_final_seconds", 0.0)) if route_from_selector else 0.0
        if route_from_selector and (not np.isfinite(route_s) or route_s < 0):
            continue
        route_vals.append(float(route_s))
        e2e_vals.append(float(route_s) + float(forward_s))

    forward_series = pd.Series(forward_vals, dtype=float)
    route_series = pd.Series(route_vals, dtype=float)
    e2e_series = pd.Series(e2e_vals, dtype=float)
    if not forward_series.empty:
        out["total_forward_time_s"] = float(forward_series.sum())
        out["mean_forward_time_s"] = float(forward_series.mean())
        out["forward_p95_s"] = float(forward_series.quantile(0.95))
        out["forward_valid_n"] = int(len(forward_series))
    if not route_series.empty:
        out["route_p95_s"] = float(route_series.quantile(0.95))
    if not e2e_series.empty:
        out["total_e2e_time_s"] = float(e2e_series.sum())
        out["mean_e2e_request_time_s"] = float(e2e_series.mean())
        out["e2e_p95_s"] = float(e2e_series.quantile(0.95))
        out["e2e_valid_n"] = int(len(e2e_series))
    return out


def _figure1_tsfm_points(
    baseline_df: pd.DataFrame,
    current_model_names: list[str],
    expected: set[str],
    stage: int,
    args,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    release_month_by_model = _model_release_month_map()
    for model_name in current_model_names:
        sub = baseline_df[baseline_df["model"].astype(str).eq(str(model_name))].copy()
        if sub.empty:
            continue
        mase = _to_numeric_or_nan(_numeric_series(sub, "MASE").mean())
        forward = _numeric_series(sub, "forward_runtime_seconds")
        timing_source = "forward_runtime_seconds"
        if forward.empty:
            forward = _numeric_series(sub, "runtime_seconds")
            timing_source = "runtime_seconds_fallback"
        source = ""
        if "source_file" in sub.columns:
            values = sub["source_file"].dropna().astype(str).unique().tolist()
            source = ";".join(values[:3])
        if not source:
            source = _tsfm_result_source_for_model(args, str(model_name))
        release_month = release_month_by_model.get(str(model_name), "")
        rows.append(
            {
                "figure": "Figure1",
                "method": str(model_name),
                "display_label": str(model_name),
                "method_type": "TSFM",
                "stage": int(stage),
                "model": str(model_name),
                "model_abbrev": str(model_name),
                "release_month": release_month,
                "MASE": mase,
                "total_forward_time_s": float(forward.sum()) if not forward.empty else np.nan,
                "total_e2e_time_s": float(forward.sum()) if not forward.empty else np.nan,
                "mean_forward_time_s": float(forward.mean()) if not forward.empty else np.nan,
                "mean_e2e_request_time_s": float(forward.mean()) if not forward.empty else np.nan,
                "forward_p95_s": float(forward.quantile(0.95)) if not forward.empty else np.nan,
                "route_p95_s": 0.0 if not forward.empty else np.nan,
                "e2e_p95_s": float(forward.quantile(0.95)) if not forward.empty else np.nan,
                "timing_valid_n": int(len(forward)),
                "expected_n": int(len(expected)),
                "source": source,
                "timing_source": timing_source,
            }
        )
    return rows


def _figure1_point_from_selector_summary(
    summary: dict[str, object],
    timing: dict[str, object] | None,
    *,
    method: str,
    display_label: str,
    stage: int,
    expected_n: int,
    source: str,
    timing_source: str,
) -> dict[str, object] | None:
    timing = dict(timing or {})

    def timing_value(key: str, summary_key: str) -> float:
        val = _to_numeric_or_nan(timing.get(key, np.nan))
        if np.isfinite(val):
            return val
        return _to_numeric_or_nan(summary.get(summary_key, np.nan))

    mase = _to_numeric_or_nan(summary.get("MASE", np.nan))
    if not np.isfinite(mase):
        return None
    total_forward = timing_value("total_forward_time_s", "_total_selected_forward_time_s")
    total_e2e = timing_value("total_e2e_time_s", "_total_e2e_time_s")
    mean_forward = timing_value("mean_forward_time_s", "_mean_selected_forward_time_s")
    mean_e2e_request = timing_value("mean_e2e_request_time_s", "_mean_e2e_time_s")
    forward_p95 = timing_value("forward_p95_s", "_selected_forward_p95_s")
    route_p95 = timing_value("route_p95_s", "_selector_route_p95_s")
    e2e_p95 = timing_value("e2e_p95_s", "E2E P95(s)")
    if not np.isfinite(route_p95) and "Route P50/95(s)" in summary:
        _route_p50, route_p95 = _split_pair(summary.get("Route P50/95(s)"))
    if not np.isfinite(e2e_p95) and "E2E P50/95(s)" in summary:
        _e2e_p50, e2e_p95 = _split_pair(summary.get("E2E P50/95(s)"))
    forward_valid_n = int(
        _to_numeric_or_nan(timing.get("forward_valid_n", np.nan))
        if np.isfinite(_to_numeric_or_nan(timing.get("forward_valid_n", np.nan)))
        else max(
            _to_numeric_or_nan(summary.get("_selected_forward_valid_n", 0)),
            0,
        )
    )
    timing_valid_n = int(
        _to_numeric_or_nan(timing.get("e2e_valid_n", np.nan))
        if np.isfinite(_to_numeric_or_nan(timing.get("e2e_valid_n", np.nan)))
        else max(
            _to_numeric_or_nan(summary.get("_e2e_valid_n", 0)),
            0,
        )
    )
    if not np.isfinite(mean_forward) and np.isfinite(total_forward) and forward_valid_n > 0:
        mean_forward = float(total_forward) / float(forward_valid_n)
    if not np.isfinite(mean_e2e_request) and np.isfinite(total_e2e) and timing_valid_n > 0:
        mean_e2e_request = float(total_e2e) / float(timing_valid_n)
    return {
        "figure": "Figure1",
        "method": method,
        "display_label": display_label,
        "method_type": "Selector",
        "stage": int(stage),
        "model": display_label,
        "model_abbrev": display_label,
        "release_month": "",
        "MASE": mase,
        "total_forward_time_s": total_forward,
        "total_e2e_time_s": total_e2e,
        "mean_forward_time_s": mean_forward,
        "mean_e2e_request_time_s": mean_e2e_request,
        "forward_p95_s": forward_p95,
        "route_p95_s": route_p95,
        "e2e_p95_s": e2e_p95,
        "forward_timing_valid_n": forward_valid_n,
        "timing_valid_n": timing_valid_n,
        "expected_n": int(expected_n),
        "source": source,
        "timing_source": timing_source,
    }


def _figure1_selector_point(
    args,
    baseline_df: pd.DataFrame,
    ordered_model_names: list[str],
    expected: set[str],
    stage: int,
    *,
    method: str,
    display_label: str,
    route_efficiency_mode: bool,
    auto_cl: bool,
    param_overrides: dict[str, object] | None = None,
    fallback_summary: dict[str, object] | None = None,
) -> dict[str, object] | None:
    selector_args = copy.deepcopy(args)
    selector_args.route_efficiency_mode = bool(route_efficiency_mode)
    selected_df, source = _main_selector_selection(
        selector_args,
        stage,
        auto_cl=auto_cl,
        param_overrides=param_overrides,
    )
    if auto_cl:
        selected_df = _attach_auto_cl_selected_forward(selected_df, ordered_model_names)
    summary = _summarize_selection(
        method,
        selected_df,
        baseline_df,
        ordered_model_names,
        expected,
        route_from_selector=True,
    )
    timing = _selector_forward_e2e_summary(
        selected_df,
        baseline_df,
        ordered_model_names,
        expected,
        route_from_selector=True,
    )
    if fallback_summary and not np.isfinite(_to_numeric_or_nan(summary.get("MASE", np.nan))):
        summary = dict(fallback_summary)
        source = str(summary.get("_source", source) or source)

    point = _figure1_point_from_selector_summary(
        summary,
        timing,
        method=method,
        display_label=display_label,
        stage=stage,
        expected_n=len(expected),
        source=source,
        timing_source="selected_tsfm_forward+route_final_seconds",
    )
    if point is None:
        print(f"[vldb_results][figures] skip Figure1 point {display_label}: {source}", flush=True)
    return point


def _figure1_task_probe_point(
    args,
    baseline_df: pd.DataFrame,
    ordered_model_names: list[str],
    expected: set[str],
    stage: int,
    *,
    auto_cl: bool,
    fallback_summary: dict[str, object] | None = None,
) -> dict[str, object] | None:
    method = "Task-probe-M"
    display_label = "Task-probe"
    summary = dict(fallback_summary or {})
    timing: dict[str, object] = {}
    source = str(summary.get("_source", "") or "")
    if not np.isfinite(_to_numeric_or_nan(summary.get("MASE", np.nan))):
        main_args = _args_for_stage(args, stage, auto_cl=auto_cl)
        selector_df = None
        if auto_cl:
            selector_df, _selector_source = _main_selector_selection(
                args,
                stage,
                auto_cl=True,
            )
        task_probe_df, source = task_probe_select_selection_for_stage(
            args=main_args,
            stage=stage,
            metric="MASE",
            baseline_df=baseline_df,
            ordered_model_names=ordered_model_names,
            expected=expected,
            selector_df=selector_df,
        )
        summary = _summarize_selection(
            method,
            task_probe_df,
            baseline_df,
            ordered_model_names,
            expected,
            route_from_selector=True,
        )
        timing = _selector_forward_e2e_summary(
            task_probe_df,
            baseline_df,
            ordered_model_names,
            expected,
            route_from_selector=True,
        )
    point = _figure1_point_from_selector_summary(
        summary,
        timing,
        method=method,
        display_label=display_label,
        stage=stage,
        expected_n=len(expected),
        source=source,
        timing_source="task_probe_selected_tsfm_forward+route_final_seconds",
    )
    if point is None:
        print(
            f"[vldb_results][figures] skip Figure1 point {display_label}: {source}",
            flush=True,
        )
    return point


def _build_figure1_points(
    args,
    baseline_df_all: pd.DataFrame,
    ordered_model_names: list[str],
    latest_stage: int,
    season_naive_df: pd.DataFrame | None,
    latest_stage_rows: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    baseline_df, current_model_names, expected = _stage_baseline(
        args=args,
        baseline_df_all=baseline_df_all,
        ordered_model_names=ordered_model_names,
        stage=latest_stage,
        season_naive_df=season_naive_df,
    )
    rows = _figure1_tsfm_points(baseline_df, current_model_names, expected, latest_stage, args)
    auto_cl = _summary_auto_cl_enabled(args)
    main_method = "TSRouter-autocl" if auto_cl else "TSRouter-main"
    rows_by_method = {
        str(row.get("Method", "")): row
        for row in (latest_stage_rows or [])
    }
    main_point = _figure1_selector_point(
        args,
        baseline_df,
        ordered_model_names,
        expected,
        latest_stage,
        method=main_method,
        display_label="TSRouter",
        route_efficiency_mode=False,
        auto_cl=auto_cl,
        fallback_summary=rows_by_method.get(main_method),
    )
    if main_point is not None:
        rows.append(main_point)
    if not auto_cl:
        fast_point = _figure1_selector_point(
            args,
            baseline_df,
            ordered_model_names,
            expected,
            latest_stage,
            method="TSRouter-fast",
            display_label="TSRouter-fast",
            route_efficiency_mode=True,
            auto_cl=False,
            param_overrides=_method_param_overrides("TSRouter-fast"),
            fallback_summary=rows_by_method.get("TSRouter-fast"),
        )
        if fast_point is not None:
            rows.append(fast_point)
    af_point = _figure1_selector_point(
        args,
        baseline_df,
        ordered_model_names,
        expected,
        latest_stage,
        method="AutoForecast",
        display_label="AutoForecast",
        route_efficiency_mode=False,
        auto_cl=False,
        param_overrides=_method_param_overrides("AutoForecast"),
        fallback_summary=rows_by_method.get("AutoForecast"),
    )
    if af_point is not None:
        rows.append(af_point)
    xpcr_point = _figure1_selector_point(
        args,
        baseline_df,
        ordered_model_names,
        expected,
        latest_stage,
        method="AutoXPCR",
        display_label="AutoXPCR",
        route_efficiency_mode=True,
        auto_cl=False,
        param_overrides=_method_param_overrides("AutoXPCR"),
        fallback_summary=rows_by_method.get("AutoXPCR"),
    )
    if xpcr_point is not None:
        rows.append(xpcr_point)
    simplets_point = _figure1_selector_point(
        args,
        baseline_df,
        ordered_model_names,
        expected,
        latest_stage,
        method="SimpleTS",
        display_label="SimpleTS",
        route_efficiency_mode=False,
        auto_cl=False,
        param_overrides=_method_param_overrides("SimpleTS"),
        fallback_summary=rows_by_method.get("SimpleTS"),
    )
    if simplets_point is not None:
        rows.append(simplets_point)
    task_probe_point = _figure1_task_probe_point(
        args,
        baseline_df,
        ordered_model_names,
        expected,
        latest_stage,
        auto_cl=auto_cl,
        fallback_summary=rows_by_method.get("Task-probe-M"),
    )
    if task_probe_point is not None:
        rows.append(task_probe_point)
    return pd.DataFrame(rows)


def _with_figure1_display_flags(points: pd.DataFrame) -> pd.DataFrame:
    if points is None or points.empty:
        return points
    out = points.copy()
    out["figure1_displayed"] = True
    tsfm_mask = out["method_type"].astype(str).eq("TSFM")
    if not tsfm_mask.any():
        return out

    metadata = _model_family_size_metadata()
    out.loc[tsfm_mask, "figure1_displayed"] = False
    keep_indices: set[int] = set()
    family_groups: dict[str, list[tuple[int, float]]] = {}
    unknown_indices: list[int] = []
    for idx, row in out[tsfm_mask].iterrows():
        token = str(row.get("model_abbrev", row.get("model", "")))
        meta = metadata.get(token)
        if meta is None:
            unknown_indices.append(idx)
            continue
        rank = _to_numeric_or_nan(meta.get("size_rank", np.nan))
        family = str(meta.get("family", token))
        family_groups.setdefault(family, []).append((idx, rank))

    keep_indices.update(unknown_indices)
    for _family, items in family_groups.items():
        finite = [(idx, rank) for idx, rank in items if np.isfinite(rank)]
        if not finite:
            keep_indices.update(idx for idx, _rank in items)
            continue
        ranks = [rank for _idx, rank in finite]
        min_rank = min(ranks)
        max_rank = max(ranks)
        for idx, rank in finite:
            if rank == min_rank or rank == max_rank:
                keep_indices.add(idx)
    if keep_indices:
        out.loc[list(keep_indices), "figure1_displayed"] = True
    return out


def _figure1_insert_label(method: object) -> str:
    label = str(method)
    if label.startswith("TSRouter") and label != "TSRouter-fast":
        return "TSRouter"
    if label.startswith("Task-probe"):
        return "Task-probe"
    return label


def _figure1_insert_cost_by_label(table5: pd.DataFrame) -> dict[str, float]:
    required = {
        "Method",
        "Overhead",
        "Component",
        FIGURE1_INSERT_COST_VALUE_COL,
    }
    if table5 is None or table5.empty or not required.issubset(table5.columns):
        return {}
    work = table5[
        table5["Overhead"].astype(str).eq("Insert")
        & table5["Component"].astype(str).isin(FIGURE1_INSERT_COST_COMPONENTS)
    ].copy()
    if work.empty:
        return {}
    work["Component"] = (
        work["Component"].astype(str).replace(FIGURE3_COMPONENT_ALIASES)
    )
    work["_insert_value_s"] = pd.to_numeric(
        work[FIGURE1_INSERT_COST_VALUE_COL],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["_insert_value_s"])
    if work.empty:
        return {}

    out: dict[str, float] = {}
    for method, group in work.groupby("Method", sort=False):
        label = _figure1_insert_label(method)
        if label in {"TSFM", "Task-probe"}:
            continue
        component_totals = (
            group.groupby("Component", sort=False)["_insert_value_s"].sum()
        )
        incoming = _to_numeric_or_nan(component_totals.get("IncomingProfile", np.nan))
        refresh_values = [
            _to_numeric_or_nan(component_totals.get(component, np.nan))
            for component in FIGURE1_INSERT_REFRESH_COMPONENTS
        ]
        refresh_values = [value for value in refresh_values if np.isfinite(value)]
        if not np.isfinite(incoming) or not refresh_values:
            continue
        out[label] = float(incoming + sum(refresh_values))
    return out


def _with_figure1_insert_costs(
    points: pd.DataFrame,
    table5: pd.DataFrame,
) -> pd.DataFrame:
    if points is None or points.empty:
        return points
    out = points.copy()
    out[FIGURE1_INSERT_COST_COL] = np.nan
    cost_by_label = _figure1_insert_cost_by_label(table5)
    if not cost_by_label:
        return out
    label_col = "display_label" if "display_label" in out.columns else "method"
    out[FIGURE1_INSERT_COST_COL] = (
        out[label_col].astype(str).map(cost_by_label)
    )
    return out


def _figure1_insert_cost_marker_area(
    value: object,
    *,
    min_cost: float,
    max_cost: float,
) -> float:
    lo, hi = FIGURE1_INSERT_COST_MARKER_AREA_RANGE
    cost = _to_numeric_or_nan(value)
    if not np.isfinite(cost):
        return float((lo + hi) / 2.0)
    if not np.isfinite(min_cost) or not np.isfinite(max_cost) or max_cost <= min_cost:
        return float((lo + hi) / 2.0)
    ratio = (float(cost) - float(min_cost)) / (float(max_cost) - float(min_cost))
    ratio = min(1.0, max(0.0, ratio))
    return float(lo + ratio * (hi - lo))


def _load_matplotlib_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": VLDB_RESULTS_FIGURE_FONT_SERIF,
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    import matplotlib.pyplot as plt

    return plt


def _figure1_transform_y_values(
    values,
    *,
    y_break: tuple[float, float] | None,
    break_gap: float = FIGURE1_Y_BREAK_GAP,
) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if y_break is None:
        return arr
    low, high = sorted((float(y_break[0]), float(y_break[1])))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return arr
    gap = max(0.0, float(break_gap))
    out = arr.copy()
    in_break = (out > low) & (out < high)
    above = out >= high
    out[in_break] = low + (out[in_break] - low) * gap / (high - low)
    out[above] = out[above] - (high - low) + gap
    return out


def _figure1_y_break_ticks(
    y_lim: tuple[float, float],
    *,
    y_break: tuple[float, float],
    break_gap: float = FIGURE1_Y_BREAK_GAP,
) -> tuple[list[float], list[str]]:
    low, high = sorted((float(y_break[0]), float(y_break[1])))
    bottom, top = (float(y_lim[0]), float(y_lim[1]))
    lower_ticks = [
        float(tick)
        for tick in np.arange(np.ceil(bottom / 100.0) * 100.0, low + 1e-9, 100.0)
    ]
    upper_ticks = [
        float(tick)
        for tick in np.arange(high, top + 1e-9, 50.0)
    ]
    ticks = lower_ticks + [tick for tick in upper_ticks if tick > low]
    positions = _figure1_transform_y_values(
        ticks,
        y_break=(low, high),
        break_gap=break_gap,
    ).tolist()
    labels = [f"{int(tick)}" if float(tick).is_integer() else f"{tick:g}" for tick in ticks]
    return positions, labels


def _figure1_add_y_break_marks(
    ax,
    *,
    y_break: tuple[float, float],
    break_gap: float = FIGURE1_Y_BREAK_GAP,
) -> None:
    low, high = sorted((float(y_break[0]), float(y_break[1])))
    y0, y1 = _figure1_transform_y_values(
        [low, high],
        y_break=(low, high),
        break_gap=break_gap,
    )
    transform = ax.get_yaxis_transform()
    y_delta = max(4.0, float(break_gap) * 0.18)
    for y in (float(y0), float(y1)):
        for x in (0.0, 1.0):
            ax.plot(
                [x - 0.010, x + 0.010],
                [y - y_delta, y + y_delta],
                transform=transform,
                color="#202020",
                linewidth=1.15,
                clip_on=False,
                zorder=6,
            )


def _figure1_router_label_candidates(label: str, base_candidates):
    if str(label) == "AutoXPCR":
        return [(0, 28), (0, 42), (10, 44), (-10, 44), *base_candidates]
    if str(label) == "TSRouter-fast":
        return [
            (8, 8),
            (-8, 8),
            (0, 13),
            (10, 13),
            (-10, 13),
            (12, 0),
            (-12, 0),
            *[(-dx, dy) for dx, dy in base_candidates],
        ]
    if str(label) == "TSRouter":
        return [(-dx, dy) for dx, dy in base_candidates]
    return base_candidates


def _plot_figure1_one(
    points: pd.DataFrame,
    *,
    y_col: str,
    y_label: str,
    title: str,
    out_path: Path,
    y_lim: tuple[float, float] | None = None,
    y_break: tuple[float, float] | None = None,
    insert_cost_col: str | None = None,
) -> Path | None:
    plot_df = points.copy()
    for col in ["MASE", y_col]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    insert_cost_mode = bool(insert_cost_col and insert_cost_col in plot_df.columns)
    if insert_cost_mode:
        plot_df[str(insert_cost_col)] = pd.to_numeric(
            plot_df[str(insert_cost_col)],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
    y_plot_col = "_figure1_y_plot"
    plot_df[y_plot_col] = _figure1_transform_y_values(
        plot_df[y_col],
        y_break=y_break,
    )
    plot_df = plot_df.dropna(subset=["MASE", y_col]).copy()
    if "figure1_displayed" in plot_df.columns:
        plot_df = plot_df[plot_df["figure1_displayed"].map(_truthy)].copy()
    if plot_df.empty:
        print(f"[vldb_results][figures] skip {out_path.name}: no finite points", flush=True)
        return None
    plt = _load_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(12.0, 4.4))
    tsfm_df = plot_df[plot_df["method_type"].astype(str).eq("TSFM")].copy()
    router_df = plot_df[~plot_df["method_type"].astype(str).eq("TSFM")].copy()
    insert_size_values = pd.Series(dtype=float)
    if insert_cost_mode:
        label_col = "display_label" if "display_label" in router_df.columns else "method"
        insert_size_values = pd.to_numeric(
            router_df[
                ~router_df[label_col].astype(str).eq("Task-probe")
            ][str(insert_cost_col)],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).dropna()
    insert_cost_min = (
        float(insert_size_values.min()) if not insert_size_values.empty else np.nan
    )
    insert_cost_max = (
        float(insert_size_values.max()) if not insert_size_values.empty else np.nan
    )

    def _box_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        width = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
        height = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
        return width * height

    x_min, x_max = FIGURE_MASE_XLIM
    x_span = max(float(x_max - x_min), 1e-9)
    y_all = pd.to_numeric(plot_df[y_plot_col], errors="coerce").dropna()
    if y_lim is not None:
        transformed_lim = _figure1_transform_y_values(
            [float(y_lim[0]), float(y_lim[1])],
            y_break=y_break,
        )
        y_min, y_max = (float(transformed_lim[0]), float(transformed_lim[1]))
    else:
        y_min = float(y_all.min())
        y_max = float(y_all.max())
        y_span = max(y_max - y_min, 1e-9)
        y_pad = 0.06 * y_span
        y_min -= y_pad
        y_max += y_pad
    y_span = max(y_max - y_min, 1e-9)
    point_coords = [
        (
            point_idx,
            (float(point["MASE"]) - x_min) / x_span,
            (float(point[y_plot_col]) - y_min) / y_span,
        )
        for point_idx, point in plot_df.iterrows()
    ]

    def _label_offsets(
        df: pd.DataFrame,
        *,
        candidates_for_label,
        initial_boxes: list[tuple[float, float, float, float]] | None = None,
        font_size: float,
        bold: bool,
        leader_distance: int,
    ) -> tuple[
        dict[object, tuple[int, int, str, str, bool]],
        list[tuple[float, float, float, float]],
    ]:
        if df.empty:
            return {}, list(initial_boxes or [])
        placed = list(initial_boxes or [])
        offsets: dict[object, tuple[int, int, str, str, bool]] = {}
        for idx, row in df.sort_values([y_plot_col, "MASE"], ascending=[False, True]).iterrows():
            label = str(row.get("display_label", ""))
            point_x = (float(row["MASE"]) - x_min) / x_span
            point_y = (float(row[y_plot_col]) - y_min) / y_span
            width_per_char = 0.0082 if bold else 0.0075
            text_width = min(0.165, max(0.046, width_per_char * len(label)))
            text_height = 0.060 if font_size >= 11 else 0.052
            best: tuple[float, tuple[int, int], tuple[float, float, float, float]] | None = None
            for dx, dy in candidates_for_label(label):
                label_x = point_x + dx * 0.00155
                label_y = point_y + dy * 0.00245
                if dx == 0:
                    left, right = label_x - text_width / 2.0, label_x + text_width / 2.0
                elif dx > 0:
                    left, right = label_x, label_x + text_width
                else:
                    left, right = label_x - text_width, label_x
                if dy > 0:
                    bottom, top = label_y, label_y + text_height
                elif dy < 0:
                    bottom, top = label_y - text_height, label_y
                else:
                    bottom, top = label_y - text_height / 2.0, label_y + text_height / 2.0
                box = (left, right, bottom, top)
                score_box = (left - 0.014, right + 0.014, bottom - 0.012, top + 0.012)
                overlap_penalty = sum(_box_overlap(score_box, other) for other in placed) * 12000.0
                point_penalty = 0.0
                for other_idx, other_x, other_y in point_coords:
                    if other_idx == idx:
                        continue
                    if left - 0.010 <= other_x <= right + 0.010 and bottom - 0.010 <= other_y <= top + 0.010:
                        point_penalty += 18.0
                edge_penalty = 0.0
                if point_y < 0.14 and dy < 0:
                    edge_penalty += 22.0
                if point_y > 0.86 and dy > 0:
                    edge_penalty += 22.0
                if point_x < 0.12 and dx < 0:
                    edge_penalty += 8.0
                if point_x > 0.88 and dx > 0:
                    edge_penalty += 8.0
                horizontal_overflow = max(0.0, -left) + max(0.0, right - 1.0)
                vertical_overflow = max(0.0, -bottom) + max(0.0, top - 1.0)
                bounds_penalty = horizontal_overflow * 90.0 + vertical_overflow * 5000.0
                distance_penalty = 0.008 * (abs(dx) + abs(dy))
                score = overlap_penalty + point_penalty + edge_penalty + bounds_penalty + distance_penalty
                if best is None or score < best[0]:
                    best = (score, (dx, dy), score_box)
            assert best is not None
            dx, dy = best[1]
            offsets[idx] = (
                dx,
                dy,
                "center" if dx == 0 else ("left" if dx > 0 else "right"),
                "bottom" if dy > 0 else ("top" if dy < 0 else "center"),
                (dy == 0 and abs(dx) >= 38)
                or abs(dx) + abs(dy) >= leader_distance,
            )
            placed.append(best[2])
        return offsets, placed

    selector_styles = FIGURE1_SELECTOR_STYLES
    router_label_col = "display_label" if "display_label" in router_df.columns else "method"
    router_labels = (
        router_df[router_label_col].astype(str)
        if router_label_col in router_df.columns
        else pd.Series(index=router_df.index, dtype=str)
    )
    fast_rows = router_df[router_labels.eq("TSRouter-fast")]
    if not fast_rows.empty:
        ax.axhline(
            y=float(fast_rows.iloc[0][y_plot_col]),
            color=selector_styles["TSRouter-fast"]["color"],
            linestyle=FIGURE1_REFERENCE_LINESTYLE,
            linewidth=FIGURE1_REFERENCE_LINEWIDTH,
            alpha=FIGURE1_REFERENCE_LINE_ALPHA,
            zorder=1,
        )
    main_rows = router_df[router_labels.eq("TSRouter")]
    if not main_rows.empty:
        ax.axvline(
            x=float(main_rows.iloc[0]["MASE"]),
            color=selector_styles["TSRouter"]["color"],
            linestyle=FIGURE1_REFERENCE_LINESTYLE,
            linewidth=FIGURE1_REFERENCE_LINEWIDTH,
            alpha=FIGURE1_REFERENCE_LINE_ALPHA,
            zorder=1,
        )
    router_candidates = [
        (10, 10),
        (10, 27),
        (10, 44),
        (-10, 10),
        (-10, 27),
        (-10, 44),
        (10, -10),
        (10, -27),
        (-10, -10),
        (-10, -27),
        (42, 0),
        (-42, 0),
    ]

    router_offsets, router_label_boxes = _label_offsets(
        router_df,
        candidates_for_label=lambda label: _figure1_router_label_candidates(
            label,
            router_candidates,
        ),
        font_size=11.5,
        bold=True,
        leader_distance=30,
    )
    for _, row in router_df.iterrows():
        label = str(row.get("display_label", row.get("method", "")))
        style = selector_styles.get(
            label,
            {"color": "#111111", "marker": "D", "size": 180, "edgecolor": "black"},
        )
        marker = style["marker"]
        marker_size = style["size"]
        marker_edgecolor = style.get("edgecolor", "black")
        marker_linewidth = 1.0
        hollow_marker = False
        if insert_cost_mode:
            if label == "Task-probe":
                marker = "s"
                marker_edgecolor = style["color"]
                marker_linewidth = 1.7
                hollow_marker = True
            else:
                marker_size = _figure1_insert_cost_marker_area(
                    row.get(str(insert_cost_col), np.nan),
                    min_cost=insert_cost_min,
                    max_cost=insert_cost_max,
                )
        scatter_kwargs = {
            "s": marker_size,
            "marker": marker,
            "linewidth": marker_linewidth,
            "label": label,
            "zorder": 4,
        }
        if hollow_marker:
            scatter_kwargs.update(
                {
                    "facecolors": "none",
                    "edgecolors": marker_edgecolor,
                }
            )
        else:
            scatter_kwargs.update(
                {
                    "color": style["color"],
                    "edgecolor": marker_edgecolor,
                }
            )
        ax.scatter(
            [row["MASE"]],
            [row[y_plot_col]],
            **scatter_kwargs,
        )
        dx, dy, ha, va, leader = router_offsets.get(
            row.name, (10, 10, "left", "bottom", False)
        )
        if label == "TSRouter-fast":
            leader = False
        annotate_kwargs = {
            "xytext": (dx, dy),
            "textcoords": "offset points",
            "fontsize": 11.5,
            "fontweight": "bold",
            "ha": ha,
            "va": va,
            "clip_on": False,
            "zorder": 5,
        }
        if leader:
            annotate_kwargs["arrowprops"] = {
                "arrowstyle": "-",
                "connectionstyle": "arc3,rad=0.0",
                "color": style["color"],
                "lw": 0.75,
                "alpha": 0.75,
                "shrinkA": 1,
                "shrinkB": 5,
            }
        ax.annotate(
            label,
            (row["MASE"], row[y_plot_col]),
            **annotate_kwargs,
        )

    # Place TSFM labels after router labels. Router label boxes are reserved so
    # the later TSFM placement cannot cover paper-facing method names. Explicit
    # close quadrants keep dense low-latency labels readable without leaders.
    if not tsfm_df.empty:
        tsfm_scatter_kwargs = {
            "s": 106,
            "alpha": 0.82,
            "linewidth": 0.9,
            "label": "TSFM",
            "zorder": 2,
        }
        if insert_cost_mode:
            tsfm_scatter_kwargs.update(
                {
                    "s": 118,
                    "facecolors": "none",
                    "edgecolors": "#8a8f98",
                    "linewidth": 1.35,
                    "alpha": 0.92,
                }
            )
        else:
            tsfm_scatter_kwargs.update(
                {
                    "color": "#8a8f98",
                    "edgecolor": "white",
                }
            )
        ax.scatter(
            tsfm_df["MASE"],
            tsfm_df[y_plot_col],
            **tsfm_scatter_kwargs,
        )
        close_quadrants = {
            "upper_right": [(7, 7), (10, 9), (13, 11)],
            "upper_left": [(-7, 7), (-10, 9), (-13, 11)],
            "lower_right": [(7, -7), (10, -9), (13, -11)],
            "lower_left": [(-7, -7), (-10, -9), (-13, -11)],
        }
        def _tsfm_candidates(label: str):
            preferred = FIGURE1_TSFM_LABEL_QUADRANTS.get(label)
            if preferred is not None:
                return close_quadrants[preferred]
            return [
                close_quadrants[quadrant][distance]
                for distance in range(3)
                for quadrant in [
                    "upper_right",
                    "upper_left",
                    "lower_right",
                    "lower_left",
                ]
            ]

        offsets, _all_label_boxes = _label_offsets(
            tsfm_df,
            candidates_for_label=_tsfm_candidates,
            initial_boxes=router_label_boxes,
            font_size=9.8,
            bold=False,
            leader_distance=10_000,
        )
        for _, row in tsfm_df.iterrows():
            dx, dy, ha, va, _leader = offsets.get(
                row.name, (16, 12, "left", "bottom", False)
            )
            annotate_kwargs = {
                "xytext": (dx, dy),
                "textcoords": "offset points",
                "fontsize": 9.8,
                "alpha": 0.90,
                "color": "#3f444a",
                "ha": ha,
                "va": va,
                "clip_on": False,
                "zorder": 3,
            }
            ax.annotate(
                str(row.get("display_label", "")),
                (row["MASE"], row[y_plot_col]),
                **annotate_kwargs,
            )

    ax.set_xlabel("MASE (lower is better)", fontsize=13.5)
    ax.set_ylabel(y_label, fontsize=13.5)
    if title:
        ax.set_title(title, fontsize=14)
    ax.set_xlim(*FIGURE_MASE_XLIM)
    y_values = pd.to_numeric(plot_df[y_plot_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if y_lim is not None:
        transformed_lim = _figure1_transform_y_values(
            [float(y_lim[0]), float(y_lim[1])],
            y_break=y_break,
        )
        ax.set_ylim(float(transformed_lim[0]), float(transformed_lim[1]))
        if y_break is not None:
            tick_positions, tick_labels = _figure1_y_break_ticks(
                y_lim,
                y_break=y_break,
            )
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)
            _figure1_add_y_break_marks(ax, y_break=y_break)
    elif not y_values.empty:
        y_min = float(y_values.min())
        y_max = float(y_values.max())
        y_span = max(y_max - y_min, 1e-9)
        y_bottom = y_min - 0.05 * y_span
        if y_min >= 0:
            y_bottom = max(0.0, y_bottom)
        ax.set_ylim(y_bottom, y_max + 0.14 * y_span)
    ax.tick_params(axis="both", labelsize=11.5)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.32)
    handles, labels = ax.get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    legend_order = [
        label
        for label in FIGURE1_LEGEND_ORDER
        if label in handle_by_label
    ]
    legend_handles = [handle_by_label[label] for label in legend_order]
    legend_kwargs = {
        "loc": FIGURE1_LEGEND_LOC,
        "frameon": True,
        "fancybox": False,
        "framealpha": 0.94,
        "facecolor": "white",
        "edgecolor": "#9a9a9a",
        "fontsize": FIGURE1_LEGEND_FONTSIZE,
        "markerscale": 0.72,
        "borderpad": 0.34,
        "labelspacing": 0.28,
        "handletextpad": 0.46,
        "borderaxespad": 0.36,
        "scatterpoints": 1,
    }
    if insert_cost_mode:
        from matplotlib.lines import Line2D

        proxy_marker_size = 9.0
        proxy_marker_sizes = {
            "AutoForecast": 7.8,
            "TSRouter": 10.4,
            "TSRouter-fast": 10.4,
        }
        legend_handles = []
        for label in legend_order:
            label_marker_size = proxy_marker_sizes.get(label, proxy_marker_size)
            if label == "TSFM":
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle="None",
                        marker="o",
                        markersize=label_marker_size,
                        markerfacecolor="none",
                        markeredgecolor="#8a8f98",
                        markeredgewidth=1.35,
                    )
                )
            elif label == "Task-probe":
                style = selector_styles["Task-probe"]
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle="None",
                        marker="s",
                        markersize=label_marker_size,
                        markerfacecolor="none",
                        markeredgecolor=style["color"],
                        markeredgewidth=1.55,
                    )
                )
            else:
                style = selector_styles.get(
                    label,
                    {"color": "#111111", "marker": "D", "edgecolor": "black"},
                )
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle="None",
                        marker=style["marker"],
                        markersize=label_marker_size,
                        markerfacecolor=style["color"],
                        markeredgecolor=style.get("edgecolor", "black"),
                        markeredgewidth=1.0,
                    )
                )
        legend_kwargs.update(
            {
                "title": "Marker area = Insert cost\n(smaller) = (easier)",
                "title_fontsize": 11.2,
                "markerscale": 1.0,
            }
        )
    legend = ax.legend(
        legend_handles,
        legend_order,
        **legend_kwargs,
    )
    if insert_cost_mode:
        legend.get_title().set_fontweight("bold")
    fig.tight_layout()
    _save_figure_outputs(fig, out_path, dpi=240)
    plt.close(fig)
    return out_path


def _plot_vldb_figure1(
    args,
    baseline_df_all: pd.DataFrame,
    ordered_model_names: list[str],
    latest_stage: int,
    season_naive_df: pd.DataFrame | None,
    out_dir: Path,
    latest_stage_rows: list[dict[str, object]] | None = None,
    table5: pd.DataFrame | None = None,
) -> dict[str, Path]:
    points = _build_figure1_points(
        args=args,
        baseline_df_all=baseline_df_all,
        ordered_model_names=ordered_model_names,
        latest_stage=latest_stage,
        season_naive_df=season_naive_df,
        latest_stage_rows=latest_stage_rows,
    )
    if points.empty:
        print("[vldb_results][figures] skip Figure1: no points", flush=True)
        return {}
    points = _with_figure1_display_flags(points)
    data_path = out_dir / "figure1_quality_time_points.csv"
    points.to_csv(data_path, index=False)
    print(f"[vldb_results][figures] write {data_path.as_posix()}", flush=True)

    paths: dict[str, Path] = {"figure1_data": data_path}
    path = _plot_figure1_one(
        points,
        y_col="mean_e2e_request_time_s",
        y_label="Mean E2E latency per request (s, lower is better)",
        title="",
        out_path=out_dir / "figure1_mase_vs_total_forward_time.png",
    )
    if path is not None:
        _record_figure_output(paths, "figure1_mean_e2e_request", path)
    path = _plot_figure1_one(
        points,
        y_col="e2e_p95_s",
        y_label="E2E P95 latency (s, lower is better)",
        title="",
        out_path=out_dir / "figure1_mase_vs_e2e_p95.png",
        y_lim=FIGURE1_E2E_P95_YLIM,
        y_break=FIGURE1_E2E_P95_Y_BREAK,
    )
    if path is not None:
        _record_figure_output(paths, "figure1_e2e_p95", path)
    insert_points = _with_figure1_insert_costs(
        points,
        table5 if table5 is not None else pd.DataFrame(),
    )
    if FIGURE1_INSERT_COST_COL in insert_points.columns and insert_points[
        FIGURE1_INSERT_COST_COL
    ].notna().any():
        insert_data_path = out_dir / "figure1_mase_vs_e2e_p95_insert_cost_points.csv"
        insert_points.to_csv(insert_data_path, index=False)
        print(f"[vldb_results][figures] write {insert_data_path.as_posix()}", flush=True)
        paths["figure1_e2e_p95_insert_cost_data"] = insert_data_path
        path = _plot_figure1_one(
            insert_points,
            y_col="e2e_p95_s",
            y_label="E2E P95 latency (s, lower is better)",
            title="",
            out_path=out_dir / "figure1_mase_vs_e2e_p95_insert_cost.png",
            y_lim=FIGURE1_E2E_P95_YLIM,
            y_break=FIGURE1_E2E_P95_Y_BREAK,
            insert_cost_col=FIGURE1_INSERT_COST_COL,
        )
        if path is not None:
            _record_figure_output(paths, "figure1_e2e_p95_insert_cost", path)
    else:
        print(
            "[vldb_results][figures] skip Figure1 insert-cost variant: no Table5 insert costs",
            flush=True,
        )
    return paths


def _table2_stage_columns(table2_mase_raw: pd.DataFrame) -> list[str]:
    return [
        str(col)
        for col in table2_mase_raw.columns
        if re.fullmatch(r"z\d+-\d+", str(col))
    ]


def _build_figure2_points(
    args,
    table2_mase_raw: pd.DataFrame,
    ordered_model_names: list[str],
    *,
    family_representative_only: bool = False,
) -> pd.DataFrame:
    if table2_mase_raw is None or table2_mase_raw.empty:
        return pd.DataFrame()
    main_method = "TSRouter-autocl" if _summary_auto_cl_enabled(args) else "TSRouter-main"
    methods = [
        main_method,
        "AutoForecast",
        "AutoXPCR",
        "SimpleTS",
        "Random",
        "Profile-probe-M",
        "Recent",
        "Task-probe-M",
        "Current_best-M",
    ]
    stage_cols = _table2_stage_columns(table2_mase_raw)
    if not stage_cols:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    release_month_by_model = _model_release_month_map()
    metadata_by_model = _model_family_size_metadata()
    representative_by_abbrev = (
        _figure2_family_representatives(ordered_model_names)
        if family_representative_only
        else {}
    )
    for method in methods:
        if method not in table2_mase_raw.index:
            continue
        display_method = _figure_method_label(method)
        for x_pos, col in enumerate(stage_cols):
            match = re.fullmatch(r"z(\d+)-(\d+)", str(col))
            stage = int(match.group(1)) if match else x_pos + 1
            latest_model, latest_abbrev, x_label = _stage_model_label(ordered_model_names, stage)
            representative_meta: dict[str, object] = {}
            if family_representative_only:
                representative_meta = representative_by_abbrev.get(latest_abbrev, {})
                if not representative_meta:
                    continue
                x_pos = int(representative_meta["x_pos"])
                release_month = str(representative_meta.get("release_month", "") or "")
                family_label = str(representative_meta["family_display"])
                x_label = f"{family_label}\n({release_month})" if release_month else family_label
            meta = metadata_by_model.get(latest_model, metadata_by_model.get(latest_abbrev, {}))
            family = str(representative_meta.get("family", meta.get("family", "")))
            family_display = str(
                representative_meta.get("family_display", meta.get("family_display", ""))
            )
            rows.append(
                {
                    "figure": "Figure2",
                    "variant": "family_representative" if family_representative_only else "all_stages",
                    "method": method,
                    "display_method": display_method,
                    "stage": stage,
                    "stage_col": str(col),
                    "x_pos": int(x_pos),
                    "latest_model": latest_model,
                    "latest_model_abbrev": latest_abbrev,
                    "family": family,
                    "family_display": family_display,
                    "size_variant": str(meta.get("size_variant", "")),
                    "representative_abbrev": str(
                        representative_meta.get("representative_abbrev", latest_abbrev)
                    ),
                    "release_month": release_month_by_model.get(latest_model, release_month_by_model.get(latest_abbrev, "")),
                    "x_label": x_label,
                    "MASE": _to_numeric_or_nan(table2_mase_raw.loc[method, col]),
                }
            )
    return pd.DataFrame(rows)


def _figure2_method_group(display_method: str) -> str:
    method = str(display_method)
    if method in FIGURE2_STATIC_METHODS:
        return "static"
    if method in FIGURE2_TASK_ADAPTIVE_METHODS:
        return "task-adaptive"
    return ""


def _figure2_line_style(display_method: str, color: str | None) -> dict[str, object]:
    method = str(display_method)
    group = _figure2_method_group(method)
    is_tsrouter = method == "TSRouter"
    is_current_best = method == "Current-best"
    marker = FIGURE2_SPECIAL_MARKERS.get(
        method,
        "o",
    )
    marker_size = (
        FIGURE2_TSROUTER_MARKERSIZE
        if is_tsrouter
        else (FIGURE2_STATIC_MARKERSIZE if group == "static" else FIGURE2_BASE_MARKERSIZE)
    )
    return {
        "marker": marker,
        "markersize": marker_size,
        "linewidth": 4.0 if is_tsrouter else 2.2,
        "color": color,
        "linestyle": "--" if is_current_best else "-",
        "markerfacecolor": color if group != "static" else "none",
        "markeredgecolor": "black" if is_tsrouter else color,
        "markeredgewidth": 1.6 if is_tsrouter else (1.2 if group == "static" else 1.0),
        "alpha": 0.98 if is_tsrouter else 0.86,
        "zorder": 4 if is_tsrouter else 2,
    }


def _figure2_display_offsets(
    plot_df: pd.DataFrame,
    *,
    overlap_atol: float = FIGURE2_OVERLAP_ATOL,
    spacing_points: float = FIGURE2_OVERLAP_OFFSET_POINTS,
) -> dict[str, float]:
    methods = list(dict.fromkeys(plot_df.get("display_method", pd.Series(dtype=str)).astype(str)))
    offsets = {method: 0.0 for method in methods}
    curves: dict[str, pd.Series] = {}
    for method, group in plot_df.groupby("display_method", sort=False):
        values = pd.to_numeric(group["MASE"], errors="coerce")
        curves[str(method)] = pd.Series(
            values.to_numpy(dtype=float),
            index=pd.to_numeric(group["x_pos"], errors="coerce").to_numpy(dtype=float),
        ).groupby(level=0).last()

    adjacency = {method: set() for method in methods}
    for left_idx, left in enumerate(methods):
        for right in methods[left_idx + 1 :]:
            joined = pd.concat(
                [curves.get(left, pd.Series(dtype=float)), curves.get(right, pd.Series(dtype=float))],
                axis=1,
                join="inner",
            ).dropna()
            if joined.empty:
                continue
            if np.isclose(
                joined.iloc[:, 0].to_numpy(dtype=float),
                joined.iloc[:, 1].to_numpy(dtype=float),
                rtol=0.0,
                atol=float(overlap_atol),
            ).any():
                adjacency[left].add(right)
                adjacency[right].add(left)

    visited: set[str] = set()
    for method in methods:
        if method in visited:
            continue
        stack = [method]
        component: list[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacency[current] - visited)
        if len(component) <= 1:
            continue
        ordered = [name for name in methods if name in component]
        if "TSRouter" in ordered:
            offsets["TSRouter"] = float(FIGURE2_TSROUTER_DISPLAY_OFFSET_POINTS)
            others = [name for name in ordered if name != "TSRouter"]
            for rank, name in enumerate(others):
                distance = float(rank // 2 + 1) * float(spacing_points)
                offsets[name] = -distance if rank % 2 == 0 else distance
            continue
        center = (len(ordered) - 1) / 2.0
        for rank, name in enumerate(ordered):
            offsets[name] = (float(rank) - center) * float(spacing_points)
    return offsets


def _figure2_y_with_point_offsets(
    ax,
    x_values,
    y_values,
    offset_points,
) -> np.ndarray:
    y_arr = pd.to_numeric(pd.Series(y_values), errors="coerce").to_numpy(dtype=float)
    offsets = pd.to_numeric(pd.Series(offset_points), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(offsets) != len(y_arr):
        offsets = np.resize(offsets, len(y_arr))
    if not np.isfinite(offsets).any() or np.allclose(offsets, 0.0, equal_nan=True):
        return y_arr
    x_arr = pd.to_numeric(pd.Series(x_values), errors="coerce").to_numpy(dtype=float)
    out = y_arr.copy()
    finite = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(offsets)
    if not finite.any():
        return out
    coords = np.column_stack([x_arr[finite], y_arr[finite]])
    display_coords = ax.transData.transform(coords)
    display_coords[:, 1] += offsets[finite] * ax.figure.dpi / 72.0
    out[finite] = ax.transData.inverted().transform(display_coords)[:, 1]
    return out


def _render_figure2_plot(
    plot_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    xlabel: str,
    figsize: tuple[float, float] = (15.0, 6.0),
    x_tick_fontsize: float = 9.5,
    x_tick_date_fontsize: float | None = None,
) -> Path | None:
    if plot_df.empty:
        return None

    plt = _load_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=figsize)
    color_by_method = {
        "TSRouter": "#0057b8",
        "AutoForecast": "#9467bd",
        "AutoXPCR": "#17a2a4",
        "SimpleTS": "#2ca02c",
        "Random": "#8a8f98",
        "Profile-probe": "#9c755f",
        "Recent": "#f28e2b",
        "Task-probe-M": FIGURE_TASK_PROBE_COLOR,
        "Task-probe": FIGURE_TASK_PROBE_COLOR,
        "Current_best-M": "#f28e2b",
        "Current-best": "#e15759",
    }
    x_labels = (
        plot_df[["x_pos", "x_label"]]
        .drop_duplicates()
        .sort_values("x_pos")
    )
    if "display_offset_points" in plot_df.columns:
        display_offsets = {
            str(method): float(
                pd.to_numeric(group["display_offset_points"], errors="coerce")
                .dropna()
                .iloc[0]
            )
            for method, group in plot_df.groupby("display_method", sort=False)
            if not pd.to_numeric(
                group["display_offset_points"], errors="coerce"
            ).dropna().empty
        }
    else:
        display_offsets = _figure2_display_offsets(plot_df)
    nonzero_offsets = {
        method: offset
        for method, offset in display_offsets.items()
        if not np.isclose(offset, 0.0)
    }
    if nonzero_offsets:
        print(
            "[vldb_results][figures] Figure2 display-only overlap offsets(points): "
            + ", ".join(
                f"{method}={offset:+.1f}"
                for method, offset in nonzero_offsets.items()
            ),
            flush=True,
        )
    ax.set_xticks(x_labels["x_pos"].tolist())
    x_positions = pd.to_numeric(x_labels["x_pos"], errors="coerce").dropna()
    if not x_positions.empty:
        ax.set_xlim(float(x_positions.min()) - 0.35, float(x_positions.max()) + 0.35)
    ax.set_ylim(*FIGURE2_MASE_YLIM)
    fig.canvas.draw()

    for display_method, group in plot_df.groupby("display_method", sort=False):
        group = group.sort_values("x_pos")
        y = pd.to_numeric(group["MASE"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if y.notna().sum() == 0:
            print(f"[vldb_results][figures] Figure2 skip line {display_method}: all NaN", flush=True)
            continue
        color = color_by_method.get(str(display_method), None)
        line_style = _figure2_line_style(str(display_method), color)
        if "display_offset_points" in group.columns:
            point_offsets = pd.to_numeric(
                group["display_offset_points"], errors="coerce"
            ).fillna(0.0)
        else:
            point_offsets = pd.Series(
                float(display_offsets.get(str(display_method), 0.0)),
                index=group.index,
            )
        y_plot = _figure2_y_with_point_offsets(
            ax,
            group["x_pos"],
            y,
            point_offsets,
        )
        ax.plot(
            group["x_pos"],
            y_plot,
            label=str(display_method),
            **line_style,
        )
    if x_tick_date_fontsize is None:
        ax.set_xticklabels(x_labels["x_label"].tolist(), fontsize=x_tick_fontsize)
        x_labelpad = 4
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        tick_transform = ax.get_xaxis_transform()
        for _, label_row in x_labels.iterrows():
            raw_label = str(label_row["x_label"])
            name, _, date = raw_label.partition("\n")
            date = date.strip()
            if date and not (date.startswith("(") and date.endswith(")")):
                date = f"({date})"
            ax.text(
                float(label_row["x_pos"]),
                -0.052,
                name,
                transform=tick_transform,
                ha="center",
                va="top",
                fontsize=x_tick_fontsize,
                clip_on=False,
            )
            if date:
                ax.text(
                    float(label_row["x_pos"]),
                    -0.112,
                    date,
                    transform=tick_transform,
                    ha="center",
                    va="top",
                    fontsize=x_tick_date_fontsize,
                    clip_on=False,
                )
        x_labelpad = 44
    ax.set_xlabel(xlabel, fontsize=13, labelpad=x_labelpad)
    ax.set_ylabel("MASE (lower is better)", fontsize=13)
    if title:
        ax.set_title(title, fontsize=15)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.32)
    handles, labels = ax.get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    if handle_by_label:
        from matplotlib.lines import Line2D

        header_handle = Line2D([], [], linestyle="none", marker="")
        blank_handle = Line2D([], [], linestyle="none", marker="")
        static_entries = [
            (handle_by_label[label], label)
            for label in FIGURE2_STATIC_METHODS
            if label in handle_by_label
        ]
        adaptive_entries = [
            (handle_by_label[label], label)
            for label in FIGURE2_TASK_ADAPTIVE_METHODS
            if label in handle_by_label
        ]
        column_height = max(len(static_entries), len(adaptive_entries)) + 1

        def legend_column(header: str, entries: list[tuple[object, str]]):
            column = [(header_handle, header), *entries]
            column.extend(
                (blank_handle, "\u200b")
                for _ in range(column_height - len(column))
            )
            return column

        legend_entries = legend_column("Static", static_entries) + legend_column(
            "Task-adaptive",
            adaptive_entries,
        )
        legend = ax.legend(
            [handle for handle, _label in legend_entries],
            [label for _handle, label in legend_entries],
            loc="upper right",
            frameon=True,
            framealpha=0.95,
            edgecolor="#b8b8b8",
            fontsize=10.5,
            ncol=2,
            columnspacing=1.5,
            handlelength=2.0,
        )
        legend.get_texts()[0].set_fontweight("bold")
        legend.get_texts()[column_height].set_fontweight("bold")
    fig.tight_layout()
    if x_tick_date_fontsize is not None:
        fig.subplots_adjust(bottom=0.24)
    _save_figure_outputs(fig, out_path, dpi=240)
    plt.close(fig)
    return out_path


def _apply_figure2_family_tsrouter_offsets(plot_df: pd.DataFrame) -> pd.DataFrame:
    if plot_df.empty or "display_method" not in plot_df.columns:
        return plot_df
    out = plot_df.copy()
    if "display_offset_points" not in out.columns:
        out["display_offset_points"] = 0.0
    x_positions = (
        pd.to_numeric(out.get("x_pos", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .sort_values()
        .drop_duplicates()
        .tolist()
    )
    raw_tail = set(x_positions[-int(FIGURE2_FAMILY_TSROUTER_RAW_TAIL_N) :])
    tsrouter = out["display_method"].astype(str).eq("TSRouter")
    x_numeric = pd.to_numeric(out.get("x_pos", pd.Series(dtype=float)), errors="coerce")
    out.loc[tsrouter, "display_offset_points"] = float(
        FIGURE2_FAMILY_TSROUTER_DISPLAY_OFFSET_POINTS
    )
    if raw_tail:
        out.loc[tsrouter & x_numeric.isin(raw_tail), "display_offset_points"] = 0.0
    return out


def _plot_vldb_figure2(
    args,
    table2_mase_raw: pd.DataFrame,
    ordered_model_names: list[str],
    out_dir: Path,
) -> dict[str, Path]:
    if table2_mase_raw is None or table2_mase_raw.empty:
        print("[vldb_results][figures] skip Figure2: empty Table2-MASE", flush=True)
        return {}
    if not _table2_stage_columns(table2_mase_raw):
        print("[vldb_results][figures] skip Figure2: no stage columns", flush=True)
        return {}

    paths: dict[str, Path] = {}
    plot_df = _build_figure2_points(
        args,
        table2_mase_raw,
        ordered_model_names,
        family_representative_only=False,
    )
    if plot_df.empty:
        print("[vldb_results][figures] skip Figure2: requested methods missing", flush=True)
        return {}
    display_offsets = _figure2_display_offsets(plot_df)
    plot_df["display_offset_points"] = (
        plot_df["display_method"].astype(str).map(display_offsets).fillna(0.0)
    )
    data_path = out_dir / "figure2_table2_mase_by_stage_points.csv"
    plot_df.to_csv(data_path, index=False)
    print(f"[vldb_results][figures] write {data_path.as_posix()}", flush=True)
    paths["figure2_data"] = data_path
    out_path = _render_figure2_plot(
        plot_df,
        out_path=out_dir / "figure2_table2_mase_by_stage.png",
        title="",
        xlabel="Growing zoo stage by released TSFM",
        figsize=(15.0, 6.0),
        x_tick_fontsize=9.5,
    )
    if out_path is not None:
        _record_figure_output(paths, "figure2_mase_by_stage", out_path)

    family_df = _build_figure2_points(
        args,
        table2_mase_raw,
        ordered_model_names,
        family_representative_only=True,
    )
    if family_df.empty:
        print("[vldb_results][figures] skip Figure2 family representative: no matching representative stages", flush=True)
        return paths
    family_df["display_offset_points"] = (
        family_df["display_method"]
        .astype(str)
        .map(display_offsets)
        .fillna(0.0)
    )
    family_df = _apply_figure2_family_tsrouter_offsets(family_df)
    family_data_path = out_dir / "figure2_table2_mase_by_family_representative_points.csv"
    family_df.to_csv(family_data_path, index=False)
    print(f"[vldb_results][figures] write {family_data_path.as_posix()}", flush=True)
    paths["figure2_family_representative_data"] = family_data_path
    family_path = _render_figure2_plot(
        family_df,
        out_path=out_dir / "figure2_table2_mase_by_family_representative.png",
        title="",
        xlabel="Growing zoo by released representative TSFM family",
        figsize=(12.5, 5.0),
        x_tick_fontsize=10.2,
        x_tick_date_fontsize=8.8,
    )
    if family_path is not None:
        _record_figure_output(paths, "figure2_family_representative", family_path)
    return paths


def _figure3_method_order(table5: pd.DataFrame) -> list[str]:
    if table5 is None or table5.empty or "Method" not in table5.columns:
        return []
    available = list(dict.fromkeys(table5["Method"].dropna().astype(str)))
    main_tsrouter = next(
        (
            method
            for method in available
            if method.startswith("TSRouter") and method != "TSRouter-fast"
        ),
        "TSRouter",
    )
    requested = [
        "Task-probe",
        "AutoForecast",
        "AutoXPCR",
        "SimpleTS",
        main_tsrouter,
        "TSRouter-fast",
    ]
    return [method for method in requested if method in available]


def _build_figure3_points(
    table5: pd.DataFrame,
    *,
    statistic: str,
) -> pd.DataFrame:
    value_cols = {
        "p95": "P95_mean_stage4_last(s)",
        "total_mean": "Total_mean_stage4_last(s)",
    }
    if statistic not in value_cols:
        raise ValueError(f"unsupported Figure3 statistic: {statistic}")
    value_col = value_cols[statistic]
    required = {"Method", "Overhead", "Component", value_col}
    if table5 is None or table5.empty or not required.issubset(table5.columns):
        return pd.DataFrame()

    method_order = _figure3_method_order(table5)
    method_rank = {method: rank for rank, method in enumerate(method_order)}
    overhead_rank = {
        overhead: rank for rank, overhead in enumerate(FIGURE3_OVERHEAD_ORDER)
    }
    work = table5[
        table5["Method"].astype(str).isin(method_order)
        & table5["Overhead"].astype(str).isin(FIGURE3_OVERHEAD_ORDER)
    ].copy()
    if work.empty:
        return pd.DataFrame()
    work["Component"] = (
        work["Component"].astype(str).replace(FIGURE3_COMPONENT_ALIASES)
    )
    work["figure"] = "Figure3"
    work["statistic"] = statistic
    work["source_column"] = value_col
    if statistic == "p95":
        work.loc[
            work["Overhead"].astype(str).eq("Insert"),
            "source_column",
        ] = "Total_mean_stage4_last(s)"
    work["value_s"] = [
        _to_numeric_or_nan(row.get(str(row["source_column"]), np.nan))
        for _, row in work.iterrows()
    ]
    work["value_s"] = pd.to_numeric(work["value_s"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    work["value_available"] = work["value_s"].notna()
    work["method_rank"] = work["Method"].astype(str).map(method_rank)
    work["overhead_rank"] = work["Overhead"].astype(str).map(overhead_rank)
    work["component_rank"] = work.groupby(
        ["Method", "Overhead"], sort=False
    ).cumcount()
    return work[
        [
            "figure",
            "statistic",
            "source_column",
            "Method",
            "method_rank",
            "Overhead",
            "overhead_rank",
            "Component",
            "component_rank",
            "value_s",
            "value_available",
        ]
    ].sort_values(["method_rank", "overhead_rank", "component_rank"])


def _equal_gap_left_edges(widths: Iterable[float], available_width: float = 1.0) -> list[float]:
    widths = [max(0.0, float(width)) for width in widths]
    if not widths:
        return []
    gap = max((float(available_width) - sum(widths)) / float(len(widths) + 1), 0.0)
    left_edges: list[float] = []
    cursor = gap
    for width in widths:
        left_edges.append(cursor)
        cursor += width + gap
    return left_edges


def _figure3_legend_component_set(points: pd.DataFrame) -> set[str]:
    if points is None or points.empty or "Component" not in points.columns:
        return set()
    return set(points["Component"].dropna().astype(str))


def _figure3_dynamic_axis_top(max_value: float) -> float:
    value = float(max_value) if np.isfinite(max_value) else 0.0
    if value <= 0.0:
        return 1.0
    return value * 1.14


def _figure3_segment_value_label(value: float) -> str:
    value = float(value)
    if abs(value) >= 10.0:
        return f"{value:.0f}"
    return f"{value:.2f}"


def _render_figure3_plot(
    points: pd.DataFrame,
    *,
    statistic: str,
    out_path: Path,
) -> Path | None:
    if points is None or points.empty:
        return None
    plt = _load_matplotlib_pyplot()
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(11.2, 4.2))
    insert_ax = ax.twinx()
    ax.set_zorder(2)
    insert_ax.set_zorder(1)
    ax.patch.set_visible(False)
    insert_ax.patch.set_visible(False)
    method_rows = (
        points[["Method", "method_rank"]]
        .drop_duplicates()
        .sort_values("method_rank")
    )
    methods = method_rows["Method"].astype(str).tolist()
    group_step = 1.12
    method_positions = {
        method: rank * group_step for rank, method in enumerate(methods)
    }
    bar_width = 0.30
    bar_step = bar_width
    overhead_offsets = {
        overhead: (rank - 1) * bar_step
        for rank, overhead in enumerate(FIGURE3_OVERHEAD_ORDER)
    }
    missing_bars: list[tuple[object, float]] = []
    segment_labels: list[tuple[object, float, float, float, str]] = []
    max_left_total = 0.0
    max_insert_total = 0.0
    for method_rank, method in enumerate(methods):
        method_x = method_rank * group_step
        method_points = points[points["Method"].astype(str).eq(method)]
        for overhead in FIGURE3_OVERHEAD_ORDER:
            bar_x = method_x + overhead_offsets[overhead]
            target_ax = insert_ax if overhead == "Insert" else ax
            bar_points = method_points[
                method_points["Overhead"].astype(str).eq(overhead)
            ].sort_values("component_rank")
            finite = bar_points.dropna(subset=["value_s"])
            if finite.empty:
                missing_bars.append((target_ax, bar_x))
                continue
            bottom = 0.0
            for _, row in finite.iterrows():
                component = str(row["Component"])
                value = max(0.0, float(row["value_s"]))
                target_ax.bar(
                    bar_x,
                    value,
                    width=bar_width,
                    bottom=bottom,
                    color=FIGURE3_COMPONENT_COLORS.get(component, "#bab0ab"),
                    edgecolor="white",
                    linewidth=0.75,
                    zorder=3,
                )
                if value > 0.0:
                    segment_labels.append((target_ax, bar_x, bottom, value, component))
                bottom += value
            if overhead == "Insert":
                max_insert_total = max(max_insert_total, bottom)
            else:
                max_left_total = max(max_left_total, bottom)
            if np.isclose(bottom, 0.0):
                target_ax.hlines(
                    0.0,
                    bar_x - bar_width * 0.42,
                    bar_x + bar_width * 0.42,
                    color="#4d4d4d",
                    linewidth=2.0,
                    zorder=4,
                )

    left_y_top = _figure3_dynamic_axis_top(max_left_total)
    insert_y_top = _figure3_dynamic_axis_top(max_insert_total)
    ax.set_ylim(0.0, left_y_top)
    insert_ax.set_ylim(0.0, insert_y_top)
    y_top_by_axis = {ax: left_y_top, insert_ax: insert_y_top}
    for target_ax, bar_x, bottom, value, component in segment_labels:
        axis_top = y_top_by_axis[target_ax]
        ax.text(
            bar_x,
            bottom + value / 2.0,
            _figure3_segment_value_label(value),
            transform=target_ax.transData,
            ha="center",
            va="center",
            fontsize=8.7 if value / axis_top < 0.035 else 9.3,
            color=("white" if component == "IndexRefresh" else "#202020"),
            bbox=(
                {
                    "facecolor": FIGURE3_COMPONENT_COLORS.get(component, "#bab0ab"),
                    "edgecolor": "none",
                    "alpha": 0.88,
                    "pad": 0.15,
                }
                if value / axis_top < 0.035
                else None
            ),
            clip_on=False,
            zorder=1200,
        )
    for target_ax, bar_x in missing_bars:
        axis_top = y_top_by_axis[target_ax]
        ax.text(
            bar_x,
            0.015 * axis_top,
            "N/A",
            transform=target_ax.transData,
            rotation=0,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#202020",
            zorder=1200,
        )
    ax.set_xticks([method_positions[method] for method in methods])
    ax.set_xticklabels(methods, fontsize=13.0)
    for tick_label, method in zip(ax.get_xticklabels(), methods):
        if method.startswith("TSRouter"):
            tick_label.set_fontweight("bold")
    ax.tick_params(axis="x", pad=8, length=0)
    ax.tick_params(axis="y", labelsize=12.2)
    insert_ax.tick_params(axis="y", labelsize=12.2, colors="#2171b5", pad=7)
    right_edge = method_positions[methods[-1]] if methods else 0.0
    ax.set_xlim(-0.72, right_edge + 0.72)
    if statistic == "p95":
        ylabel = FIGURE3_P95_LEFT_YLABEL
    else:
        ylabel = "Total-mean latency across zoo stages (s)"
    ax.set_ylabel(ylabel, fontsize=11.8, labelpad=8, fontfamily="serif", fontstyle="normal")
    insert_ax.set_ylabel(
        FIGURE3_INSERT_RIGHT_YLABEL,
        fontsize=11.8,
        color="#2171b5",
        labelpad=10,
        fontfamily="serif",
        fontstyle="normal",
    )
    ax.grid(True, axis="y", linestyle="--", linewidth=0.55, alpha=0.34, zorder=0)
    insert_ax.grid(False)
    fig.subplots_adjust(top=0.955, bottom=0.175, left=0.150, right=0.875)
    present_component_set = _figure3_legend_component_set(points)
    legend_artists = []
    for title, components, title_color in FIGURE3_LEGEND_GROUPS:
        legend_handles = []
        seen_labels: set[str] = set()
        for component in components:
            if component not in present_component_set:
                continue
            legend_label = FIGURE3_COMPONENT_LEGEND_LABELS.get(component, component)
            if legend_label in seen_labels:
                continue
            seen_labels.add(legend_label)
            legend_handles.append(
                Patch(
                    facecolor=FIGURE3_COMPONENT_COLORS.get(component, "#bab0ab"),
                    edgecolor="white",
                    label=legend_label,
                )
            )
        if not legend_handles:
            continue
        legend = ax.legend(
            handles=legend_handles,
            title=title,
            loc="upper left",
            bbox_to_anchor=(0.0, FIGURE3_LEGEND_ANCHOR_Y),
            frameon=True,
            fancybox=False,
            framealpha=0.94,
            fontsize=FIGURE3_LEGEND_FONTSIZE,
            title_fontsize=FIGURE3_LEGEND_TITLE_FONTSIZE,
            labelspacing=0.32,
            borderpad=0.45,
            handlelength=1.65,
            handletextpad=0.55,
            borderaxespad=0.0,
        )
        legend.get_frame().set_edgecolor(title_color)
        legend.get_frame().set_linewidth(1.15)
        legend.get_title().set_color(title_color)
        legend.get_title().set_fontweight("bold")
        legend.set_zorder(1500)
        ax.add_artist(legend)
        legend_artists.append(legend)
    if legend_artists:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_width_px = max(float(ax.get_window_extent(renderer=renderer).width), 1.0)
        legend_widths = [
            float(legend.get_window_extent(renderer=renderer).width) / ax_width_px
            for legend in legend_artists
        ]
        for legend, left_edge in zip(
            legend_artists,
            _equal_gap_left_edges(legend_widths),
        ):
            legend.set_bbox_to_anchor(
                (left_edge, FIGURE3_LEGEND_ANCHOR_Y),
                transform=ax.transAxes,
            )
        fig.canvas.draw()
    _save_figure_outputs(fig, out_path, dpi=240)
    plt.close(fig)
    return out_path


def _stage_overhead_observed_values(
    table5: pd.DataFrame,
    *,
    method: str,
    overhead: str,
    statistic: str = "p95",
) -> dict[int, float]:
    overhead_totals = dict(table5.attrs.get("overhead_totals", {}) or {})
    by_stage = dict(overhead_totals.get(str(method), {}) or {})
    observed: dict[int, float] = {}
    for stage, stage_stats in by_stage.items():
        stats = dict(dict(stage_stats or {}).get(str(overhead), {}) or {})
        value = _table5_stat_value(stats, statistic)
        if np.isfinite(value) and value >= 0:
            observed[int(stage)] = float(value)
    return dict(sorted(observed.items()))


def _stage_overhead_linear_fit(
    observed: dict[int, float],
    *,
    fit_limit: int = 20,
) -> tuple[float, float]:
    fit_items = [(stage, value) for stage, value in observed.items() if int(stage) <= int(fit_limit)]
    if len(fit_items) < 2:
        fit_items = list(observed.items())
    if len(fit_items) < 2:
        return np.nan, np.nan
    fit_x = np.asarray([stage for stage, _value in fit_items], dtype=float)
    fit_y = np.asarray([value for _stage, value in fit_items], dtype=float)
    slope, _unconstrained_intercept = np.polyfit(fit_x, fit_y, 1)
    slope = max(0.0, float(slope))
    intercept = float(fit_y.mean() - slope * fit_x.mean())
    return slope, intercept


def _figure4_display_x(stage: int) -> float:
    stage = int(stage)
    if stage in FIGURE4_STAGE_X_POSITIONS:
        return float(FIGURE4_STAGE_X_POSITIONS[stage])
    return float(stage)


def _figure4_display_adjustment(panel: str, display_method: str) -> tuple[float, str]:
    if str(panel) == "Insert" and str(display_method) == "TSRouter-fast":
        return (
            float(FIGURE4_FAST_INSERT_DISPLAY_SCALE),
            f"tsrouter_fast_insert_x{FIGURE4_FAST_INSERT_DISPLAY_SCALE:.2f}",
        )
    return 1.0, ""


def _figure4_predicted_value(
    *,
    panel: str,
    display_method: str,
    stage: int,
    observed_points: dict[int, float],
    slope: float,
    intercept: float,
) -> tuple[float, str]:
    linear = (
        max(0.0, float(slope * int(stage) + intercept))
        if np.isfinite(slope) and np.isfinite(intercept)
        else np.nan
    )
    if str(panel) == "Insert" and not str(display_method).startswith("TSRouter"):
        values = [float(value) for value in observed_points.values() if np.isfinite(value)]
        if values:
            peak = max(values)
            growth = 1.0 + float(FIGURE4_INSERT_UPPER_ENVELOPE_LOG_GROWTH) * max(
                0.0,
                float(np.log2(max(float(stage) / 20.0, 1.0))),
            )
            envelope = max(0.0, float(peak * growth))
            candidates = [envelope]
            if np.isfinite(linear):
                candidates.append(float(linear))
            return max(candidates), "upper_envelope_log_growth"
    return linear, "linear_fit"


def _build_stage_overhead_growth_points(
    table5: pd.DataFrame,
    *,
    target_stages: Iterable[int] = FIGURE4_GROWTH_STAGES,
    route_statistic: str = "p95",
    insert_statistic: str = "p95",
    figure: str = "Figure4",
) -> pd.DataFrame:
    if table5 is None or table5.empty:
        return pd.DataFrame()
    overhead_totals = dict(table5.attrs.get("overhead_totals", {}) or {})
    if not overhead_totals:
        return pd.DataFrame()
    main_route_method = next(
        (
            method
            for method in ["TSRouter", "TSRouter-autocl"]
            if method in overhead_totals
        ),
        None,
    )
    route_specs = []
    if main_route_method is not None:
        route_specs.append((main_route_method, "TSRouter"))
    if "Task-probe" in overhead_totals:
        route_specs.append(("Task-probe", "Task-probe"))
    insert_specs = [
        (method, "TSRouter" if str(method) == "TSRouter-autocl" else str(method))
        for method in overhead_totals.keys()
    ]
    target_values = [int(stage) for stage in target_stages]
    observed_targets = [
        stage for stage in FIGURE4_OBSERVED_STAGES if stage in target_values
    ]
    rows: list[dict[str, object]] = []

    def add_series(panel: str, source_method: str, display_method: str, overhead: str) -> None:
        statistic = insert_statistic if str(panel) == "Insert" else route_statistic
        statistic_label = "Mean" if str(statistic).lower() == "mean" else str(statistic).upper()
        observed = _stage_overhead_observed_values(
            table5,
            method=source_method,
            overhead=overhead,
            statistic=statistic,
        )
        if not observed:
            return
        observed_points: dict[int, float] = {}
        point_meta: dict[int, dict[str, object]] = {}
        adjustment_scale, adjustment_label = _figure4_display_adjustment(
            panel,
            display_method,
        )
        for stage in observed_targets:
            if panel == "Insert":
                start_stage, end_stage = FIGURE4_INSERT_STAGE_WINDOWS[int(stage)]
                source_items = [
                    (src_stage, value)
                    for src_stage, value in observed.items()
                    if int(start_stage) <= int(src_stage) <= int(end_stage)
                ]
                values = [value for _src_stage, value in source_items]
                if not values:
                    continue
                value = float(np.mean(values))
                point_meta[int(stage)] = {
                    "Aggregation": "stage_window_mean",
                    "SourceStageRange": f"{int(start_stage)}-{int(end_stage)}",
                    "SourceStages": ",".join(
                        str(int(src_stage)) for src_stage, _value in source_items
                    ),
                    "SourceN": int(len(source_items)),
                }
            else:
                value = observed.get(int(stage), np.nan)
                if not np.isfinite(value):
                    continue
                point_meta[int(stage)] = {
                    "Aggregation": "exact_stage",
                    "SourceStageRange": str(int(stage)),
                    "SourceStages": str(int(stage)),
                    "SourceN": 1,
                }
            observed_points[int(stage)] = float(value) * float(adjustment_scale)
        if not observed_points:
            return
        slope, intercept = _stage_overhead_linear_fit(observed_points)
        fit_input_stages = ",".join(str(stage) for stage in sorted(observed_points))
        fit_input_values = ",".join(
            f"{observed_points[stage]:.8g}" for stage in sorted(observed_points)
        )
        for stage in observed_targets:
            if stage not in observed_points:
                continue
            meta = point_meta[int(stage)]
            rows.append(
                {
                    "figure": figure,
                    "panel": panel,
                    "Method": display_method,
                    "source_method": source_method,
                    "Overhead": overhead,
                    "Statistic": statistic_label,
                    "Stage": int(stage),
                    "DisplayX": _figure4_display_x(int(stage)),
                    "value_s": float(observed_points[int(stage)]),
                    "segment": "observed",
                    "Aggregation": meta["Aggregation"],
                    "PredictionMode": "",
                    "DisplayAdjustment": adjustment_label,
                    "SourceStageRange": meta["SourceStageRange"],
                    "SourceStages": meta["SourceStages"],
                    "SourceN": meta["SourceN"],
                    "FitInputStages": fit_input_stages,
                    "FitInputValues(s)": fit_input_values,
                    "LinearSlope(s/stage)": slope,
                    "LinearIntercept(s)": intercept,
                }
            )
        if not np.isfinite(slope) or not np.isfinite(intercept):
            return
        for stage in target_values:
            if int(stage) <= max(observed_targets):
                continue
            value, prediction_mode = _figure4_predicted_value(
                panel=panel,
                display_method=display_method,
                stage=int(stage),
                observed_points=observed_points,
                slope=slope,
                intercept=intercept,
            )
            if not np.isfinite(value):
                continue
            rows.append(
                {
                    "figure": figure,
                    "panel": panel,
                    "Method": display_method,
                    "source_method": source_method,
                    "Overhead": overhead,
                    "Statistic": statistic_label,
                    "Stage": int(stage),
                    "DisplayX": _figure4_display_x(int(stage)),
                    "value_s": float(value),
                    "segment": "predicted",
                    "Aggregation": prediction_mode,
                    "PredictionMode": prediction_mode,
                    "DisplayAdjustment": adjustment_label,
                    "SourceStageRange": "",
                    "SourceStages": "",
                    "SourceN": 0,
                    "FitInputStages": fit_input_stages,
                    "FitInputValues(s)": fit_input_values,
                    "LinearSlope(s/stage)": slope,
                    "LinearIntercept(s)": intercept,
                }
            )

    for source_method, display_method in route_specs:
        add_series("Route", source_method, display_method, "Route")
    for source_method, display_method in insert_specs:
        add_series("Insert", source_method, display_method, "Insert")
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    panel_rank = {"Route": 0, "Insert": 1}
    method_order = {
        "Task-probe": 0,
        "AutoForecast": 1,
        "AutoXPCR": 2,
        "SimpleTS": 3,
        "TSRouter": 4,
        "TSRouter-fast": 5,
    }
    out["_panel_rank"] = out["panel"].astype(str).map(panel_rank).fillna(99)
    out["_method_rank"] = out["Method"].astype(str).map(method_order).fillna(98)
    out["_stage_rank"] = out["Stage"].astype(int).map(
        {stage: rank for rank, stage in enumerate(target_values)}
    )
    return out.sort_values(["_panel_rank", "_method_rank", "_stage_rank"]).drop(
        columns=["_panel_rank", "_method_rank", "_stage_rank"]
    )


def _build_overhead_growth_figure_table(table5: pd.DataFrame) -> pd.DataFrame:
    """Build Table6.3, the exact preprocessed data consumed by Figure4."""
    return _build_stage_overhead_growth_points(table5)


def _build_figure4_fused_points(
    table5: pd.DataFrame,
    fallback_points: pd.DataFrame,
) -> pd.DataFrame:
    points = _build_stage_overhead_growth_points(
        table5,
        route_statistic="p95",
        insert_statistic="mean",
        figure="Figure4Fused",
    )
    if not points.empty:
        return points
    out = _figure4_points_with_display_x(fallback_points)
    if out.empty:
        return out
    out["figure"] = "Figure4Fused"
    out.loc[out["panel"].astype(str).eq("Route"), "Statistic"] = "P95"
    out.loc[out["panel"].astype(str).eq("Insert"), "Statistic"] = "Mean"
    return out


def _figure4_line_style(method: str) -> dict[str, object]:
    label = "TSRouter" if str(method) == "TSRouter-autocl" else str(method)
    style = dict(FIGURE1_SELECTOR_STYLES.get(label, {}))
    color = style.get("color", "#4d4d4d")
    marker = style.get("marker", FIGURE2_SPECIAL_MARKERS.get(label, "o"))
    is_tsrouter = label.startswith("TSRouter")
    return {
        "marker": marker,
        "markersize": FIGURE2_TSROUTER_MARKERSIZE if is_tsrouter else FIGURE2_BASE_MARKERSIZE,
        "linewidth": 3.0 if is_tsrouter else 2.2,
        "color": color,
        "linestyle": "-",
        "markerfacecolor": color,
        "markeredgecolor": style.get("edgecolor", color),
        "markeredgewidth": 1.6 if is_tsrouter else 1.0,
        "alpha": 0.98 if is_tsrouter else 0.88,
        "zorder": 4 if is_tsrouter else 2,
    }


def _figure4_points_with_display_x(points: pd.DataFrame) -> pd.DataFrame:
    out = points.copy()
    if "DisplayX" not in out.columns:
        stage_values = pd.to_numeric(
            out.get("Stage", pd.Series(dtype=float)),
            errors="coerce",
        )
        out["DisplayX"] = [
            _figure4_display_x(int(stage)) if np.isfinite(stage) else np.nan
            for stage in stage_values
        ]
    out["DisplayX"] = pd.to_numeric(out["DisplayX"], errors="coerce")
    return out


def _render_figure4_fused_plot(points: pd.DataFrame, out_path: Path) -> Path | None:
    if points is None or points.empty:
        return None
    plt = _load_matplotlib_pyplot()
    from matplotlib.lines import Line2D

    plot_df = _figure4_points_with_display_x(points)
    fig, route_ax = plt.subplots(figsize=(4.35, 2.7))
    insert_ax = route_ax.twinx()
    panel_linestyles = {"Route": "-", "Insert": "--"}
    panel_linewidth_scale = {"Route": 1.0, "Insert": 0.9}
    method_order = ["Task-probe", "AutoForecast", "AutoXPCR", "SimpleTS", "TSRouter", "TSRouter-fast"]
    present_methods = [
        method
        for method in method_order
        if method in set(plot_df["Method"].dropna().astype(str))
    ]
    for panel in ["Route", "Insert"]:
        panel_df = plot_df[plot_df["panel"].astype(str).eq(panel)].copy()
        target_ax = route_ax if panel == "Route" else insert_ax
        for method, group in panel_df.groupby("Method", sort=False):
            group = group.sort_values("DisplayX")
            values = pd.to_numeric(group["value_s"], errors="coerce").replace(
                [np.inf, -np.inf],
                np.nan,
            )
            group = group.loc[values.notna()].copy()
            if group.empty:
                continue
            style = _figure4_line_style(str(method))
            style["linestyle"] = panel_linestyles[panel]
            style["linewidth"] = float(style["linewidth"]) * panel_linewidth_scale[panel]
            if panel == "Insert":
                style["alpha"] = min(float(style.get("alpha", 0.9)), 0.82)
            target_ax.plot(
                group["DisplayX"],
                pd.to_numeric(group["value_s"], errors="coerce"),
                **style,
            )
    tick_stages = list(FIGURE4_GROWTH_STAGES)
    tick_x = [_figure4_display_x(stage) for stage in tick_stages]
    route_ax.axvline(_figure4_display_x(20), color="#6b6b6b", linestyle=":", linewidth=0.75, alpha=0.75)
    route_ax.set_xticks(tick_x)
    route_ax.set_xticklabels([str(stage) for stage in tick_stages])
    route_ax.set_xlim(min(tick_x) - 0.12, max(tick_x) + 0.18)
    route_ax.set_xlabel("Zoo stage (models)", fontsize=7.8, labelpad=1.0)
    route_ax.set_ylabel("Route P95 latency (s)", fontsize=7.6, labelpad=2.0)
    insert_ax.set_ylabel("Mean INSERT latency (s)", fontsize=7.6, labelpad=2.0)
    route_ax.set_title("Route and insert latency", fontsize=8.6, pad=3.0)
    route_ax.grid(True, axis="y", linestyle="--", linewidth=0.45, alpha=0.32)
    route_ax.tick_params(axis="both", labelsize=7.0, pad=1.5)
    insert_ax.tick_params(axis="y", labelsize=7.0, pad=1.5)
    fig.subplots_adjust(left=0.14, right=0.86, bottom=0.23, top=0.86)

    method_handles = []
    for method in present_methods:
        style = _figure4_line_style(method)
        method_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker=style["marker"],
                markersize=style["markersize"],
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["markeredgecolor"],
                markeredgewidth=style["markeredgewidth"],
                color=style["color"],
                label=method,
            )
        )
    panel_handles = [
        Line2D([0], [0], color="#303030", linewidth=1.6, linestyle="-", label="Route"),
        Line2D([0], [0], color="#303030", linewidth=1.6, linestyle="--", label="Insert"),
    ]
    fig.legend(
        handles=panel_handles + method_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        fontsize=5.9,
        handlelength=1.4,
        columnspacing=0.95,
        handletextpad=0.42,
    )
    _save_figure_outputs(fig, out_path, dpi=240)
    plt.close(fig)
    return out_path


def _plot_vldb_stage_overhead_growth(
    table5: pd.DataFrame,
    out_dir: Path,
    *,
    table6_3: pd.DataFrame | None = None,
) -> dict[str, Path]:
    points = (
        table6_3.copy()
        if table6_3 is not None
        else _build_overhead_growth_figure_table(table5)
    )
    if points.empty:
        print("[vldb_results][figures] skip stage overhead growth: no Table5 overhead totals", flush=True)
        return {}
    points = _figure4_points_with_display_x(points)
    data_path = out_dir / "figure_stage_overhead_growth_p95_points.csv"
    points.to_csv(data_path, index=False)
    print(f"[vldb_results][figures] write {data_path.as_posix()}", flush=True)

    plt = _load_matplotlib_pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.35), sharex=True)
    panel_titles = {
        "Route": "Route latency by stage",
        "Insert": "Insert latency by stage",
    }
    panel_ylabels = {
        "Route": "Route P95 latency (s)",
        "Insert": "Insert P95 latency (s)",
    }
    for ax, panel in zip(axes, ["Route", "Insert"]):
        panel_df = points[points["panel"].astype(str).eq(panel)].copy()
        for method, group in panel_df.groupby("Method", sort=False):
            group = group.sort_values("Stage")
            values = pd.to_numeric(group["value_s"], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            group = group.loc[values.notna()].copy()
            if group.empty:
                continue
            ax.plot(
                pd.to_numeric(group["DisplayX"], errors="coerce"),
                pd.to_numeric(group["value_s"], errors="coerce"),
                label=str(method),
                **_figure4_line_style(str(method)),
            )
        ax.axvline(
            _figure4_display_x(20),
            color="#6b6b6b",
            linestyle=":",
            linewidth=0.9,
            alpha=0.75,
        )
        ax.set_title(panel_titles[panel], fontsize=12.5)
        ax.set_ylabel(panel_ylabels[panel], fontsize=11.2)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.55, alpha=0.34)
        ax.tick_params(axis="both", labelsize=10.4)
        tick_x = [_figure4_display_x(stage) for stage in FIGURE4_GROWTH_STAGES]
        ax.set_xticks(tick_x)
        ax.set_xticklabels([str(stage) for stage in FIGURE4_GROWTH_STAGES])
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_xlim(min(tick_x) - 0.12, max(tick_x) + 0.18)
        legend_loc = "upper left" if panel == "Route" else "upper right"
        ax.legend(frameon=False, fontsize=8.8, loc=legend_loc)
    axes[-1].set_xlabel("Zoo stage (models)", fontsize=11.2)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.10, top=0.94, hspace=0.36)
    out_path = out_dir / "figure_stage_overhead_growth_p95.png"
    _save_figure_outputs(fig, out_path, dpi=240)
    plt.close(fig)
    paths: dict[str, Path] = {"figure4_stage_overhead_growth_data": data_path}
    _record_figure_output(paths, "figure4_stage_overhead_growth", out_path)
    fused_points = _figure4_points_with_display_x(
        _build_figure4_fused_points(table5, points)
    )
    fused_data_path = out_dir / "figure_stage_overhead_growth_fused_points.csv"
    fused_points.to_csv(fused_data_path, index=False)
    print(f"[vldb_results][figures] write {fused_data_path.as_posix()}", flush=True)
    paths["figure4_stage_overhead_growth_fused_data"] = fused_data_path
    fused_path = _render_figure4_fused_plot(
        fused_points,
        out_dir / "figure_stage_overhead_growth_route_p95_insert_mean_fused.png",
    )
    if fused_path is not None:
        _record_figure_output(paths, "figure4_stage_overhead_growth_fused", fused_path)
    return paths


def _plot_vldb_figure3(table5: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    if table5 is None or table5.empty:
        print("[vldb_results][figures] skip Figure3: empty Table5", flush=True)
        return {}
    point_frames = [
        _build_figure3_points(table5, statistic=statistic)
        for statistic in ["p95", "total_mean"]
    ]
    point_frames = [frame for frame in point_frames if not frame.empty]
    if not point_frames:
        print("[vldb_results][figures] skip Figure3: Table5 summary columns missing", flush=True)
        return {}
    points = pd.concat(point_frames, ignore_index=True)
    data_path = out_dir / "figure3_table5_overhead_breakdown_points.csv"
    points.to_csv(data_path, index=False)
    print(f"[vldb_results][figures] write {data_path.as_posix()}", flush=True)
    paths: dict[str, Path] = {"figure3_data": data_path}
    for statistic, filename, key in [
        ("p95", "figure3_table5_overhead_breakdown_p95.png", "figure3_p95"),
        (
            "total_mean",
            "figure3_table5_overhead_breakdown_total_mean.png",
            "figure3_total_mean",
        ),
    ]:
        path = _render_figure3_plot(
            points[points["statistic"].astype(str).eq(statistic)].copy(),
            statistic=statistic,
            out_path=out_dir / filename,
        )
        if path is not None:
            _record_figure_output(paths, key, path)
    return paths


def _write_vldb_figures(
    args,
    baseline_df_all: pd.DataFrame,
    ordered_model_names: list[str],
    latest_stage: int,
    table2_mase_raw: pd.DataFrame,
    table5: pd.DataFrame,
    season_naive_df: pd.DataFrame | None,
    latest_stage_rows: list[dict[str, object]] | None = None,
    table6_3: pd.DataFrame | None = None,
) -> dict[str, Path]:
    try:
        out_dir = _vldb_stage_figure_output_dir(
            _vldb_figure_output_dir(args),
            latest_stage,
        )
        paths: dict[str, Path] = {}
        paths.update(
            _plot_vldb_figure1(
                args=args,
                baseline_df_all=baseline_df_all,
                ordered_model_names=ordered_model_names,
                latest_stage=latest_stage,
                season_naive_df=season_naive_df,
                out_dir=out_dir,
                latest_stage_rows=latest_stage_rows,
                table5=table5,
            )
        )
        paths.update(
            _plot_vldb_figure2(
                args=args,
                table2_mase_raw=table2_mase_raw,
                ordered_model_names=ordered_model_names,
                out_dir=out_dir,
            )
        )
        paths.update(_plot_vldb_figure3(table5=table5, out_dir=out_dir))
        paths.update(
            _plot_vldb_stage_overhead_growth(
                table5=table5,
                out_dir=out_dir,
                table6_3=table6_3,
            )
        )
        return paths
    except Exception as exc:
        print(f"[vldb_results][figures] failed: {type(exc).__name__}: {exc}", flush=True)
        return {}


def _write_table(table: pd.DataFrame, filename: str, index: bool = False) -> Path:
    TSROUTER_VLDB_TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    path = TSROUTER_VLDB_TABLE_ROOT / filename
    table.to_csv(path, index=index, index_label="Method" if index else None)
    return path


def _latest_by_file_order(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    for col in key_cols:
        if col not in out.columns:
            out[col] = ""
    if out.empty:
        return out
    out["_file_order"] = np.arange(len(out), dtype=np.int64)
    return out.sort_values("_file_order").drop_duplicates(key_cols, keep="last").drop(columns=["_file_order"])


def _format_breakdown_value(value):
    numeric = _to_numeric_or_nan(value)
    if np.isfinite(numeric):
        return round(float(numeric), 4)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _build_insert_breakdown_table(
    args,
    stages: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, str]:
    source = TSROUTER_CSV_ROOT / "Model_zoo_repr" / "step3_insert_timing.csv"
    autoforecast_source = (
        BASELINE_CSV_ROOT
        / "selectors"
        / "AutoForecast_Select"
        / "step3_insert_timing.csv"
    )
    autoxpcr_source = (
        BASELINE_CSV_ROOT
        / "selectors"
        / "AutoXPCR_Select"
        / "step3_insert_timing.csv"
    )
    simplets_source = (
        BASELINE_CSV_ROOT
        / "selectors"
        / "SimpleTS_Select"
        / "step3_insert_timing.csv"
    )
    simplets_retrain_corrections: list[str] = []
    full_pool_forward_by_model: dict[str, float] = {}
    full_pool_forward_source = ""
    full_pool_forward_issue = ""

    def simplets_retrain_without_ts2vec(table: pd.DataFrame) -> pd.DataFrame:
        """Normalize legacy SimpleTS rows whose retrain timer enclosed TS2Vec training."""
        if table.empty or "selector_retrain_seconds" not in table.columns:
            return table
        out = table.copy()
        if "ts2vec_train_seconds" not in out.columns:
            out["ts2vec_train_seconds"] = np.nan
        if "selector_retrain_excludes_ts2vec" not in out.columns:
            out["selector_retrain_excludes_ts2vec"] = ""
        for idx, rec in out.iterrows():
            if _truthy(rec.get("selector_retrain_excludes_ts2vec", False)):
                continue
            status = str(rec.get("ts2vec_checkpoint_status", "") or "").strip().lower()
            if status != "trained":
                continue
            train_seconds = _to_numeric_or_nan(rec.get("ts2vec_train_seconds", np.nan))
            artifact_path = Path(str(rec.get("model_artifact_path", "") or ""))
            if not np.isfinite(train_seconds) and artifact_path.is_file():
                try:
                    with artifact_path.open("rb") as artifact_file:
                        payload = pickle.load(artifact_file)
                    if isinstance(payload, dict):
                        train_seconds = _to_numeric_or_nan(
                            payload.get("ts2vec_train_seconds", np.nan)
                        )
                except Exception:
                    train_seconds = np.nan
            retrain_seconds = _to_numeric_or_nan(
                rec.get("selector_retrain_seconds", np.nan)
            )
            if (
                not np.isfinite(train_seconds)
                or train_seconds <= 0
                or not np.isfinite(retrain_seconds)
            ):
                continue
            corrected_retrain = max(0.0, retrain_seconds - train_seconds)
            out.at[idx, "selector_retrain_seconds"] = corrected_retrain
            insert_total = _to_numeric_or_nan(rec.get("insert_total_seconds", np.nan))
            if np.isfinite(insert_total):
                out.at[idx, "insert_total_seconds"] = max(
                    0.0, insert_total - train_seconds
                )
            out.at[idx, "ts2vec_train_seconds"] = train_seconds
            out.at[idx, "selector_retrain_excludes_ts2vec"] = "true"
            simplets_retrain_corrections.append(
                f"z{int(_to_numeric_or_nan(rec.get('stage', 0)))}:{train_seconds:.6g}s"
            )
        return out
    auto_cl = _summary_auto_cl_enabled(args)
    auto_cl_mode = _summary_auto_cl_mode(args)
    output_metrics = (
        AUTOCL_INSERT_METRICS + TSROUTER_FAST_INSERT_METRICS + AUTOFORECAST_INSERT_METRICS + AUTOXPCR_INSERT_METRICS + SIMPLETS_INSERT_METRICS
        if auto_cl
        else TSROUTER_INSERT_METRICS + TSROUTER_FAST_INSERT_METRICS + AUTOFORECAST_INSERT_METRICS + AUTOXPCR_INSERT_METRICS + SIMPLETS_INSERT_METRICS
    )
    main_params = _resolve_main_params(args=args, auto_cl=auto_cl)
    expected_advanced_scope = normalize_advanced_baseline_train_scope(
        getattr(args, "advanced_baseline_train_scope", "center")
    )
    if expected_advanced_scope == "full_pool":
        pool_args = _args_for_stage(
            args,
            int(getattr(args, "zoo_total_num", 0) or 0),
            auto_cl=False,
        )
        pool_path = Path(get_tsrouter_repr_forward_dir(pool_args)) / (
            build_repr_eval_pool_forward_stem(pool_args) + "_all_results.csv"
        )
        full_pool_forward_source = pool_path.as_posix()
        if not pool_path.exists():
            full_pool_forward_issue = "missing full_pool forward CSV"
        else:
            try:
                pool_df = pd.read_csv(pool_path, low_memory=False)
                required_pool_columns = {"model", "forward_runtime_seconds"}
                missing_pool_columns = sorted(
                    required_pool_columns.difference(pool_df.columns)
                )
                if pool_df.empty:
                    full_pool_forward_issue = "empty full_pool forward CSV"
                elif missing_pool_columns:
                    full_pool_forward_issue = (
                        "full_pool forward CSV missing columns="
                        + ",".join(missing_pool_columns)
                    )
                else:
                    pool_df = pool_df.copy()
                    pool_df["_file_order"] = np.arange(
                        len(pool_df), dtype=np.int64
                    )
                    pool_df["_model_abbr"] = pool_df["model"].astype(str).map(
                        lambda name: Model_abbrev_map.get(name, name)
                    )
                    pool_df["_dataset_key"] = (
                        pool_df["dataset"].astype(str)
                        if "dataset" in pool_df.columns
                        else "__full_pool_profile__"
                    )
                    pool_df = pool_df.sort_values("_file_order").drop_duplicates(
                        ["_model_abbr", "_dataset_key"], keep="last"
                    )
                    pool_df["_forward_seconds"] = pd.to_numeric(
                        pool_df["forward_runtime_seconds"], errors="coerce"
                    )
                    for model_abbr, group in pool_df.groupby(
                        "_model_abbr", sort=False
                    ):
                        values = group["_forward_seconds"]
                        if (
                            not values.empty
                            and values.notna().all()
                            and np.isfinite(values.to_numpy(dtype=float)).all()
                            and values.ge(0).all()
                        ):
                            full_pool_forward_by_model[str(model_abbr)] = float(
                                values.sum()
                            )
                    if not full_pool_forward_by_model:
                        full_pool_forward_issue = (
                            "full_pool forward CSV has no valid model runtimes"
                        )
            except Exception as exc:
                full_pool_forward_issue = (
                    f"full_pool forward CSV read error: {type(exc).__name__}: {exc}"
                )

    full_pool_missing_models: set[str] = set()

    def with_expected_incoming_profile(
        table: pd.DataFrame,
        metric_specs: list[tuple[str, str]],
    ) -> pd.DataFrame:
        if expected_advanced_scope != "full_pool" or table.empty:
            return table
        out = table.copy()
        component_columns = [
            source_col
            for _label, source_col in metric_specs
            if source_col not in {"incoming_profile_seconds", "insert_total_seconds"}
        ]
        for idx, rec in out.iterrows():
            model_abbr = str(
                rec.get("latest_model_abbr", "")
                or Model_abbrev_map.get(
                    str(rec.get("latest_model_full_name", "") or ""), ""
                )
            )
            incoming_seconds = full_pool_forward_by_model.get(model_abbr, np.nan)
            if not np.isfinite(incoming_seconds):
                if model_abbr:
                    full_pool_missing_models.add(model_abbr)
                out.at[idx, "incoming_profile_seconds"] = np.nan
                out.at[idx, "insert_total_seconds"] = np.nan
                out.at[idx, "timing_valid"] = False
                continue
            component_values = [
                _to_numeric_or_nan(rec.get(column, np.nan))
                for column in component_columns
            ]
            out.at[idx, "incoming_profile_seconds"] = float(incoming_seconds)
            out.at[idx, "step2_runtime_path"] = full_pool_forward_source
            out.at[idx, "step2_runtime_status"] = "ok_full_pool_forward"
            out.at[idx, "insert_total_seconds"] = (
                float(incoming_seconds + sum(component_values))
                if component_values
                and all(np.isfinite(value) and value >= 0 for value in component_values)
                else np.nan
            )
        return out
    expected_zoo_prefix = get_zoo_repr_prefix(main_params.get("zoo_repr_set", ""))
    expected_scale_protocol = normalize_repr_scale_protocol(
        main_params.get("repr_scale_protocol", "raw")
    )
    expected_scale_tag = "raw" if expected_scale_protocol == "raw" else "std"

    def row_complete(rec: pd.Series, stage: int) -> bool:
        status = str(rec.get("status", "") or "").strip().lower()
        if status.startswith("skipped") or status in {"failed", "error"}:
            return False
        if "incomplete" in status and status != "built_step2_runtime_incomplete":
            return False
        model_count = _to_numeric_or_nan(rec.get("model_count", np.nan))
        if np.isfinite(model_count) and int(model_count) < int(stage):
            return False
        required = ["insert_runtime_seconds", "index_refresh_seconds", "insert_total_seconds"]
        return all(np.isfinite(_to_numeric_or_nan(rec.get(col, np.nan))) for col in required)

    def selector_insert_row_validation(
        rec: pd.Series,
        metric_specs: list[tuple[str, str]],
    ) -> tuple[bool, str]:
        if rec is None or rec.empty:
            return False, "timing row missing"
        status = str(rec.get("status", "") or "").strip().lower()
        if (
            status.startswith("skipped")
            or status in {"failed", "error"}
            or "incomplete" in status
        ):
            return False, f"status={status or 'missing'}"
        timing_valid = str(rec.get("timing_valid", "") or "").strip()
        if timing_valid and not _truthy(timing_valid):
            return False, f"timing_valid={timing_valid}"
        invalid_fields = []
        for _label, source_col in metric_specs:
            value = _to_numeric_or_nan(rec.get(source_col, np.nan))
            if not np.isfinite(value) or value < 0:
                invalid_fields.append(source_col)
        if invalid_fields:
            return False, "invalid timing fields=" + ",".join(invalid_fields)
        return True, ""

    def autoforecast_insert_row_validation(rec: pd.Series) -> tuple[bool, str]:
        complete, reason = selector_insert_row_validation(
            rec,
            AUTOFORECAST_INSERT_METRICS,
        )
        if not complete:
            return complete, reason
        schema = _to_numeric_or_nan(rec.get("artifact_schema_version", np.nan))
        if np.isfinite(schema) and int(schema) != 3:
            return False, f"artifact_schema_version={int(schema)} (expected 3)"
        selector_mode = str(
            rec.get("selector_mode", "") or ""
        ).strip().lower()
        if selector_mode and selector_mode != "autoforecast":
            return False, f"selector_mode={selector_mode}"
        return True, ""

    def autoxpcr_insert_row_validation(rec: pd.Series) -> tuple[bool, str]:
        complete, reason = selector_insert_row_validation(
            rec,
            AUTOXPCR_INSERT_METRICS,
        )
        if not complete:
            return complete, reason
        schema = _to_numeric_or_nan(rec.get("artifact_schema_version", np.nan))
        if np.isfinite(schema) and int(schema) != 1:
            return False, f"artifact_schema_version={int(schema)} (expected 1)"
        selector_mode = str(
            rec.get("selector_mode", "") or ""
        ).strip().lower()
        if selector_mode and selector_mode != "autoxpcr":
            return False, f"selector_mode={selector_mode}"
        return True, ""

    def has_value(value) -> bool:
        text = str(value if value is not None else "").strip()
        return bool(text and text.lower() not in {"nan", "none"})

    def column_matches(rec: pd.Series, key: str, expected) -> bool:
        if key not in rec.index or not has_value(rec.get(key, "")):
            return True
        actual = rec.get(key)
        if isinstance(expected, bool):
            return _truthy(actual) == expected
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            actual_num = _to_numeric_or_nan(actual)
            return bool(np.isfinite(actual_num) and np.isclose(actual_num, float(expected)))
        return str(actual).strip().lower() == str(expected).strip().lower()

    def advanced_baseline_scope_matches(rec: pd.Series, expected: dict) -> bool:
        expected_scope = normalize_advanced_baseline_train_scope(
            expected.get(
                "advanced_baseline_train_scope",
                getattr(args, "advanced_baseline_train_scope", "center"),
            )
        )
        if (
            "advanced_baseline_train_scope" not in rec.index
            or not has_value(rec.get("advanced_baseline_train_scope", ""))
        ):
            return expected_scope == "center"
        try:
            actual_scope = normalize_advanced_baseline_train_scope(
                rec.get("advanced_baseline_train_scope", "center")
            )
        except ValueError:
            return False
        return actual_scope == expected_scope

    def repr_scale_matches(rec: pd.Series, repr_name: str) -> bool:
        if "repr_scale_protocol" in rec.index and has_value(rec.get("repr_scale_protocol", "")):
            try:
                return (
                    normalize_repr_scale_protocol(rec.get("repr_scale_protocol", ""))
                    == expected_scale_protocol
                )
            except ValueError:
                return False
        return f"_{expected_scale_tag}_" in repr_name

    def zoo_repr_set_matches(rec: pd.Series, repr_name: str) -> bool:
        if not expected_zoo_prefix:
            return True
        if "zoo_repr_set" in rec.index and has_value(rec.get("zoo_repr_set", "")):
            try:
                return get_zoo_repr_prefix(rec.get("zoo_repr_set", "")) == expected_zoo_prefix
            except ValueError:
                return False
        return expected_zoo_prefix in repr_name

    def main_config_row(rec: pd.Series, *, route_efficiency_mode: bool = False) -> bool:
        repr_name = str(
            rec.get("repr_set_name", "")
            or rec.get("repr_forward_stem", "")
            or rec.get("row_key", "")
        )
        expected_encoder = str(main_params.get("repr_encoder", "") or "")
        if not column_matches(rec, "repr_encoder", expected_encoder):
            return False
        if expected_encoder and "repr_encoder" not in rec.index and expected_encoder not in repr_name:
            return False
        if not repr_scale_matches(rec, repr_name):
            return False
        if not zoo_repr_set_matches(rec, repr_name):
            return False
        for seed_col, token_prefix in [
            ("repr_data_seed", "sd"),
            ("repr_encoder_seed", "se"),
        ]:
            expected_seed = main_params.get(seed_col)
            if expected_seed is None:
                continue
            if not column_matches(rec, seed_col, int(expected_seed)):
                return False
            if seed_col not in rec.index and f"_{token_prefix}{int(expected_seed)}" not in repr_name:
                return False
        for col in [
            "repr_size",
            "repr_v",
            "repr_weight_ratio",
            "subset_top_k",
            "subset_perf_scale",
            "repr_v5_nearest_k",
            "rank_decay_coef",
            "repr_v5_distance_power",
            "sample_repr_num",
            "sample_repr_ratio",
            "task_rank_top3_instability_threshold",
        ]:
            if col in main_params and not column_matches(rec, col, main_params[col]):
                return False
        for col in [
            "base_metrics",
            "model_repr_mode",
            "sample_mode",
            "repr_sample_qc_mode",
            "repr_anchor_window_sample_strategy",
            "task_window_sample_strategy",
            "task_channel_fuse_limit",
            "ensemble_agg",
        ]:
            if col in main_params and not column_matches(rec, col, main_params[col]):
                return False
        if _truthy(rec.get("route_efficiency_mode", False)) != bool(route_efficiency_mode):
            return False
        return True

    def v7_selector_config_row(
        rec: pd.Series,
        *,
        expected: dict,
        selector_mode: str,
    ) -> bool:
        for col in ["repr_v", "base_metrics", "route_efficiency_mode"]:
            if col in expected and not column_matches(rec, col, expected[col]):
                return False
        recorded_mode = str(rec.get("selector_mode", "") or "").strip().lower()
        if recorded_mode and recorded_mode != selector_mode:
            return False
        if not advanced_baseline_scope_matches(rec, expected):
            return False
        if "learner" in rec.index and has_value(rec.get("learner", "")):
            if str(rec.get("learner", "")).strip().upper() != VLDB_RESULTS_AUTOFORECAST_LEARNER:
                return False
        elif "autoforecast_learner" in rec.index and has_value(rec.get("autoforecast_learner", "")):
            if str(rec.get("autoforecast_learner", "")).strip().upper() != VLDB_RESULTS_AUTOFORECAST_LEARNER:
                return False
        return True

    def autoforecast_config_row(rec: pd.Series) -> bool:
        return v7_selector_config_row(
            rec,
            expected={**main_params, **_autoforecast_param_overrides()},
            selector_mode="autoforecast",
        )

    def autoxpcr_config_row(rec: pd.Series) -> bool:
        return v7_selector_config_row(
            rec,
            expected={**main_params, **_autoxpcr_param_overrides()},
            selector_mode="autoxpcr",
        )

    def simplets_config_row(rec: pd.Series) -> bool:
        expected = {
            **main_params,
            **_simplets_param_overrides(),
        }
        for col in ["repr_v", "base_metrics"]:
            if col in expected and not column_matches(rec, col, expected[col]):
                return False
        if not advanced_baseline_scope_matches(rec, expected):
            return False
        target_metric = str(rec.get("target_metric", "") or "").strip().upper()
        metric_code = str(expected.get("base_metrics", "") or "").strip().upper()
        expected_metric = {"M": "MASE", "S": "SMAPE", "C": "CRPS"}.get(metric_code, metric_code)
        if target_metric and target_metric.upper() != expected_metric.upper():
            return False
        return True

    def with_legacy_feature_refresh_alias(table: pd.DataFrame) -> pd.DataFrame:
        """Support older selector timing CSVs that used structure_refresh_seconds."""
        if table.empty or "structure_refresh_seconds" not in table.columns:
            return table
        out = table.copy()
        if "feature_refresh_seconds" not in out.columns:
            out["feature_refresh_seconds"] = out["structure_refresh_seconds"]
            return out
        feature_values = pd.to_numeric(
            out["feature_refresh_seconds"],
            errors="coerce",
        )
        fallback_values = pd.to_numeric(
            out["structure_refresh_seconds"],
            errors="coerce",
        )
        fill_mask = feature_values.isna() & fallback_values.notna()
        if fill_mask.any():
            out.loc[fill_mask, "feature_refresh_seconds"] = out.loc[
                fill_mask,
                "structure_refresh_seconds",
            ]
        return out

    def empty_table(reason: str) -> pd.DataFrame:
        table = pd.DataFrame(
            [
                {"Metric": label}
                for label, _source_col in output_metrics
            ]
        )
        table.attrs["missing_reasons"] = [
            f"Table3 cannot build stage columns: {reason}; source={source.as_posix()}"
        ]
        return table

    if not source.exists():
        reason = "missing step3_insert_timing.csv"
        return empty_table(reason), pd.DataFrame(), source, reason
    try:
        raw = pd.read_csv(source, low_memory=False)
    except Exception as exc:
        reason = f"read_error:{exc}"
        return empty_table(reason), pd.DataFrame(), source, reason
    if raw.empty:
        reason = "empty step3_insert_timing.csv"
        return empty_table(reason), pd.DataFrame(), source, reason
    if "stage" not in raw.columns:
        reason = "missing stage column"
        return empty_table(reason), pd.DataFrame(), source, reason

    df = raw.copy()
    df["_stage_num"] = pd.to_numeric(df["stage"], errors="coerce")
    df = df.dropna(subset=["_stage_num"]).copy()
    if df.empty:
        reason = "no numeric stage rows"
        return empty_table(reason), pd.DataFrame(), source, reason
    df["_stage_num"] = df["_stage_num"].astype(int)
    df["_auto_mode"] = (
        df.get("auto_cl_mode", pd.Series("v0", index=df.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"": "v0", "none": "v0"})
    )
    v0_all_latest = _latest_by_file_order(
        df[df["_auto_mode"].eq("v0")].copy(), ["_stage_num"]
    ).sort_values("_stage_num")
    v0_latest = _latest_by_file_order(
        df[
            df["_auto_mode"].eq("v0")
            & df.apply(lambda rec: main_config_row(rec, route_efficiency_mode=False), axis=1)
        ].copy(),
        ["_stage_num"],
    ).sort_values("_stage_num")
    if v0_latest.empty:
        v0_latest = v0_all_latest
    v0_fast_latest = _latest_by_file_order(
        df[
            df["_auto_mode"].eq("v0")
            & df.apply(lambda rec: main_config_row(rec, route_efficiency_mode=True), axis=1)
        ].copy(),
        ["_stage_num"],
    ).sort_values("_stage_num")
    auto_latest = _latest_by_file_order(
        df[
            df["_auto_mode"].eq(auto_cl_mode)
            & df.apply(lambda rec: main_config_row(rec, route_efficiency_mode=False), axis=1)
        ].copy(),
        ["_stage_num", "adaptive_profile"],
    ).sort_values(["_stage_num", "adaptive_profile"])
    source_issues: dict[str, str] = {}
    af_latest = pd.DataFrame()
    if autoforecast_source.exists():
        try:
            af_raw = pd.read_csv(autoforecast_source, low_memory=False)
            if not af_raw.empty and "stage" in af_raw.columns:
                af_df = with_legacy_feature_refresh_alias(af_raw.copy())
                af_df = with_expected_incoming_profile(
                    af_df, AUTOFORECAST_INSERT_METRICS
                )
                af_df["_stage_num"] = pd.to_numeric(af_df["stage"], errors="coerce")
                af_df = af_df.dropna(subset=["_stage_num"]).copy()
                af_df["_stage_num"] = af_df["_stage_num"].astype(int)
                af_latest = _latest_by_file_order(
                    af_df[af_df.apply(autoforecast_config_row, axis=1)].copy(),
                    ["_stage_num"],
                ).sort_values("_stage_num")
                if af_latest.empty:
                    source_issues["AutoForecast"] = (
                        "no rows match the expected repr_v/base_metrics/learner configuration"
                    )
            elif af_raw.empty:
                source_issues["AutoForecast"] = "timing CSV is empty"
            else:
                source_issues["AutoForecast"] = "timing CSV has no stage column"
        except Exception as exc:
            af_latest = pd.DataFrame()
            source_issues["AutoForecast"] = (
                f"timing CSV read error: {type(exc).__name__}: {exc}"
            )
    else:
        source_issues["AutoForecast"] = (
            f"timing CSV missing: {autoforecast_source.as_posix()}"
        )
    xpcr_latest = pd.DataFrame()
    if autoxpcr_source.exists():
        try:
            xpcr_raw = pd.read_csv(autoxpcr_source, low_memory=False)
            if not xpcr_raw.empty and "stage" in xpcr_raw.columns:
                xpcr_df = with_legacy_feature_refresh_alias(xpcr_raw.copy())
                xpcr_df = with_expected_incoming_profile(
                    xpcr_df, AUTOXPCR_INSERT_METRICS
                )
                xpcr_df["_stage_num"] = pd.to_numeric(
                    xpcr_df["stage"], errors="coerce"
                )
                xpcr_df = xpcr_df.dropna(subset=["_stage_num"]).copy()
                xpcr_df["_stage_num"] = xpcr_df["_stage_num"].astype(int)
                xpcr_latest = _latest_by_file_order(
                    xpcr_df[xpcr_df.apply(autoxpcr_config_row, axis=1)].copy(),
                    ["_stage_num"],
                ).sort_values("_stage_num")
                if xpcr_latest.empty:
                    source_issues["AutoXPCR"] = (
                        "no rows match the expected repr_v/base_metrics/learner configuration"
                    )
            elif xpcr_raw.empty:
                source_issues["AutoXPCR"] = "timing CSV is empty"
            else:
                source_issues["AutoXPCR"] = "timing CSV has no stage column"
        except Exception as exc:
            xpcr_latest = pd.DataFrame()
            source_issues["AutoXPCR"] = (
                f"timing CSV read error: {type(exc).__name__}: {exc}"
            )
    else:
        source_issues["AutoXPCR"] = (
            f"timing CSV missing: {autoxpcr_source.as_posix()}"
        )
    st_latest = pd.DataFrame()
    if simplets_source.exists():
        try:
            st_raw = pd.read_csv(simplets_source, low_memory=False)
            if not st_raw.empty and "stage" in st_raw.columns:
                st_df = simplets_retrain_without_ts2vec(st_raw.copy())
                st_df = with_expected_incoming_profile(
                    st_df, SIMPLETS_INSERT_METRICS
                )
                st_df["_stage_num"] = pd.to_numeric(st_df["stage"], errors="coerce")
                st_df = st_df.dropna(subset=["_stage_num"]).copy()
                st_df["_stage_num"] = st_df["_stage_num"].astype(int)
                st_latest = _latest_by_file_order(
                    st_df[st_df.apply(simplets_config_row, axis=1)].copy(),
                    ["_stage_num"],
                ).sort_values("_stage_num")
                if st_latest.empty:
                    source_issues["SimpleTS"] = (
                        "no rows match the expected repr_v/base_metrics configuration"
                    )
            elif st_raw.empty:
                source_issues["SimpleTS"] = "timing CSV is empty"
            else:
                source_issues["SimpleTS"] = "timing CSV has no stage column"
        except Exception as exc:
            st_latest = pd.DataFrame()
            source_issues["SimpleTS"] = (
                f"timing CSV read error: {type(exc).__name__}: {exc}"
            )
    else:
        source_issues["SimpleTS"] = (
            f"timing CSV missing: {simplets_source.as_posix()}"
        )

    af_status_counts = {
        str(status): int(count)
        for status, count in af_latest.get(
            "incremental_measurement_status", pd.Series(dtype=str)
        )
        .fillna("")
        .astype(str)
        .value_counts()
        .items()
        if str(status)
    }
    xpcr_status_counts = {
        str(status): int(count)
        for status, count in xpcr_latest.get(
            "incremental_measurement_status", pd.Series(dtype=str)
        )
        .fillna("")
        .astype(str)
        .value_counts()
        .items()
        if str(status)
    }

    rows = [{"Metric": label} for label, _ in output_metrics]
    used_cols: list[str] = []
    incomplete_stages: list[int] = []
    incomplete_auto_stages: list[int] = []
    issue_stages: dict[str, dict[str, list[int]]] = {}

    def record_issue(method: str, stage: int, reason: str) -> None:
        issue_stages.setdefault(str(method), {}).setdefault(
            str(reason or "timing row incomplete"), []
        ).append(int(stage))

    seen_cols: set[str] = set()
    explicit_stages = (
        sorted({int(stage) for stage in stages})
        if stages is not None
        else []
    )
    stage_values = (
        explicit_stages
        if explicit_stages
        else sorted(
            set(_stage_list(args))
            | set(v0_latest.get("_stage_num", pd.Series(dtype=int)).tolist())
            | set(v0_fast_latest.get("_stage_num", pd.Series(dtype=int)).tolist())
            | set(auto_latest.get("_stage_num", pd.Series(dtype=int)).tolist())
            | set(af_latest.get("_stage_num", pd.Series(dtype=int)).tolist())
            | set(xpcr_latest.get("_stage_num", pd.Series(dtype=int)).tolist())
            | set(st_latest.get("_stage_num", pd.Series(dtype=int)).tolist())
        )
    )
    profile_rows: list[dict[str, object]] = []
    required_profiles = {
        str(profile["adaptive_profile"]) for profile in _summary_auto_cl_profiles(args)
    }
    rows_by_metric = {str(row["Metric"]): row for row in rows}
    stage_columns: dict[int, str] = {}

    def fill_insert_metrics(
        metric_specs: list[tuple[str, str]],
        rec: pd.Series,
        col: str,
        complete: bool,
        *,
        aggregate_df: pd.DataFrame | None = None,
    ) -> None:
        for label, source_col in metric_specs:
            target = rows_by_metric.get(label)
            if target is None:
                continue
            if not complete:
                target[col] = ""
                continue
            if aggregate_df is not None:
                vals = pd.to_numeric(
                    aggregate_df.get(source_col, pd.Series(dtype=float)),
                    errors="coerce",
                )
                target[col] = _format_breakdown_value(vals.sum())
            else:
                target[col] = _format_breakdown_value(rec.get(source_col, ""))

    for stage in stage_values:
        v0_sub = v0_latest[v0_latest["_stage_num"].eq(int(stage))]
        v0_fast_sub = v0_fast_latest[v0_fast_latest["_stage_num"].eq(int(stage))]
        auto_sub = auto_latest[auto_latest["_stage_num"].eq(int(stage))]
        af_sub = af_latest[af_latest["_stage_num"].eq(int(stage))] if not af_latest.empty else pd.DataFrame()
        xpcr_sub = xpcr_latest[xpcr_latest["_stage_num"].eq(int(stage))] if not xpcr_latest.empty else pd.DataFrame()
        st_sub = st_latest[st_latest["_stage_num"].eq(int(stage))] if not st_latest.empty else pd.DataFrame()
        rec = (
            auto_sub.iloc[-1]
            if auto_cl and not auto_sub.empty
            else (
                v0_sub.iloc[-1]
                if not auto_cl and not v0_sub.empty
                else pd.Series(dtype=object)
            )
        )
        model = str(
            rec.get("latest_model_abbr", "")
            or rec.get("latest_model_full_name", "")
            or f"stage{stage}"
        )
        col = model
        if col in seen_cols:
            col = f"{model}_z{stage}"
        seen_cols.add(col)
        used_cols.append(col)
        stage_columns[int(stage)] = col
        v0_complete = not v0_sub.empty and row_complete(v0_sub.iloc[-1], stage)
        if not v0_complete:
            incomplete_stages.append(stage)
            if not auto_cl:
                record_issue(
                    "TSRouter",
                    stage,
                    "timing row missing or insert_runtime/index_refresh/insert_total is invalid",
                )
        if not auto_cl:
            fill_insert_metrics(
                TSROUTER_INSERT_METRICS,
                v0_sub.iloc[-1] if not v0_sub.empty else pd.Series(dtype=object),
                col,
                v0_complete,
            )
        fast_complete = not v0_fast_sub.empty and row_complete(v0_fast_sub.iloc[-1], stage)
        if not fast_complete:
            record_issue(
                "TSRouter-fast",
                stage,
                "timing row missing or insert_runtime/index_refresh/insert_total is invalid",
            )
        fill_insert_metrics(
            TSROUTER_FAST_INSERT_METRICS,
            v0_fast_sub.iloc[-1] if not v0_fast_sub.empty else pd.Series(dtype=object),
            col,
            fast_complete,
        )
        if af_sub.empty:
            af_complete = False
            af_reason = source_issues.get(
                "AutoForecast", "no timing row for this stage"
            )
        else:
            af_complete, af_reason = autoforecast_insert_row_validation(
                af_sub.iloc[-1]
            )
        if not af_complete:
            record_issue("AutoForecast", stage, af_reason)
        fill_insert_metrics(
            AUTOFORECAST_INSERT_METRICS,
            af_sub.iloc[-1] if not af_sub.empty else pd.Series(dtype=object),
            col,
            af_complete,
        )
        if xpcr_sub.empty:
            xpcr_complete = False
            xpcr_reason = source_issues.get(
                "AutoXPCR", "no timing row for this stage"
            )
        else:
            xpcr_complete, xpcr_reason = autoxpcr_insert_row_validation(
                xpcr_sub.iloc[-1]
            )
        if not xpcr_complete:
            record_issue("AutoXPCR", stage, xpcr_reason)
        fill_insert_metrics(
            AUTOXPCR_INSERT_METRICS,
            xpcr_sub.iloc[-1] if not xpcr_sub.empty else pd.Series(dtype=object),
            col,
            xpcr_complete,
        )
        if st_sub.empty:
            st_complete = False
            st_reason = source_issues.get("SimpleTS", "no timing row for this stage")
        else:
            st_complete, st_reason = selector_insert_row_validation(
                st_sub.iloc[-1],
                SIMPLETS_INSERT_METRICS,
            )
        if not st_complete:
            record_issue("SimpleTS", stage, st_reason)
        fill_insert_metrics(
            SIMPLETS_INSERT_METRICS,
            st_sub.iloc[-1] if not st_sub.empty else pd.Series(dtype=object),
            col,
            st_complete,
        )

        observed_profiles = {
            str(value)
            for value in auto_sub.get(
                "adaptive_profile", pd.Series(dtype=str)
            ).dropna()
        }
        profile_mapping_ok = True
        for _, profile_rec in auto_sub.iterrows():
            profile_cfg = get_auto_cl_profile_by_name(
                str(profile_rec.get("adaptive_profile", "") or ""),
                auto_cl_mode,
            )
            if profile_cfg is None:
                profile_mapping_ok = False
                continue
            for dim_col in [
                "repr_input_dim",
                "repr_output_dim",
                "repr_sub_pred_len",
                "repr_source_exact_length",
            ]:
                actual = _to_numeric_or_nan(
                    profile_rec.get(dim_col, np.nan)
                )
                if (
                    not np.isfinite(actual)
                    or int(actual) != int(profile_cfg[dim_col])
                ):
                    profile_mapping_ok = False
        model_orders = {
            str(value).strip()
            for value in auto_sub.get("model_abbr_order", pd.Series(dtype=str))
            .dropna()
            .astype(str)
        }
        latest_models = {
            str(value).strip()
            for value in auto_sub.get(
                "latest_model_abbr", pd.Series(dtype=str)
            )
            .dropna()
            .astype(str)
        }
        auto_complete = (
            observed_profiles == required_profiles
            and profile_mapping_ok
            and len(model_orders) == 1
            and len(latest_models) == 1
            and all(
                row_complete(profile_rec, stage)
                for _, profile_rec in auto_sub.iterrows()
            )
        )
        if not auto_complete:
            incomplete_auto_stages.append(stage)
            if auto_cl:
                record_issue(
                    "TSRouter-autocl",
                    stage,
                    "required adaptive profiles or insert timing fields are incomplete",
                )
        if auto_cl:
            fill_insert_metrics(
                AUTOCL_INSERT_METRICS,
                pd.Series(dtype=object),
                col,
                auto_complete,
                aggregate_df=auto_sub,
            )
        for _, profile_rec in auto_sub.iterrows():
            profile_rows.append(
                {
                    "Stage": int(stage),
                    "adaptive_profile": str(
                        profile_rec.get("adaptive_profile", "") or ""
                    ),
                    "resolved_eval_cl": str(
                        next(
                            (
                                profile["tsfm_results_dir"]
                                for profile in _summary_auto_cl_profiles(args)
                                if str(profile["adaptive_profile"])
                                == str(profile_rec.get("adaptive_profile", "") or "")
                            ),
                            "",
                        )
                    ),
                    "Complete": bool(row_complete(profile_rec, stage)),
                    "insert_runtime_seconds": _to_numeric_or_nan(
                        profile_rec.get("insert_runtime_seconds", np.nan)
                    ),
                    "index_refresh_seconds": _to_numeric_or_nan(
                        profile_rec.get("index_refresh_seconds", np.nan)
                    ),
                    "insert_total_seconds": _to_numeric_or_nan(
                        profile_rec.get("insert_total_seconds", np.nan)
                    ),
                    "row_key": str(profile_rec.get("row_key", "") or ""),
                }
            )
    table = pd.DataFrame(rows)
    table.attrs["stage_columns"] = stage_columns
    missing_reasons: list[str] = []
    for method, reasons in issue_stages.items():
        for reason, missing_stages in reasons.items():
            missing_reasons.append(
                f"{method}: stages {_format_stage_ranges(missing_stages)} missing; {reason}"
            )
    table.attrs["missing_reasons"] = missing_reasons
    note = (
        f"rows={len(raw)}, report_stages={_format_stage_ranges(stage_values)}, "
        f"v0_stages={len(v0_latest)}, "
        f"v0_fast_stages={len(v0_fast_latest)}, "
        f"autoforecast_stages={len(af_latest)}, "
        f"autoforecast_status_counts={json.dumps(af_status_counts, sort_keys=True)}, "
        f"autoxpcr_stages={len(xpcr_latest)}, "
        f"autoxpcr_status_counts={json.dumps(xpcr_status_counts, sort_keys=True)}, "
        f"simplets_stages={len(st_latest)}, "
        f"advanced_baseline_train_scope={expected_advanced_scope}, "
        f"full_pool_forward_source={full_pool_forward_source or 'n/a'}, "
        f"full_pool_forward_models={len(full_pool_forward_by_model)}, "
        f"full_pool_forward_issue={full_pool_forward_issue or 'none'}, "
        f"auto_profile_rows={len(auto_latest)}, columns={','.join(used_cols)}, "
        f"autoforecast_source={autoforecast_source.as_posix()}, "
        f"autoxpcr_source={autoxpcr_source.as_posix()}, "
        f"simplets_source={simplets_source.as_posix()}"
    )
    if incomplete_stages and not auto_cl:
        note += f", incomplete_v0_stages={_format_stage_ranges(incomplete_stages)}"
    if incomplete_auto_stages and auto_cl:
        note += (
            f", incomplete_auto_stages="
            f"{_format_stage_ranges(incomplete_auto_stages)}"
        )
    if simplets_retrain_corrections:
        note += ", simplets_ts2vec_retrain_subtracted=" + "|".join(
            simplets_retrain_corrections
        )
    if full_pool_missing_models:
        note += ", full_pool_missing_models=" + " ".join(
            sorted(full_pool_missing_models)
        )
    return (
        table,
        pd.DataFrame(profile_rows) if auto_cl else pd.DataFrame(),
        source,
        note,
    )


def _build_route_breakdown_table(args, stage_rows: dict[int, list[dict[str, object]]]) -> pd.DataFrame:
    stages = sorted(stage_rows)
    auto_cl = _summary_auto_cl_enabled(args)
    metrics = (
        AUTOCL_ROUTE_BREAKDOWN_METRICS + TASK_PROBE_ROUTE_BREAKDOWN_METRICS
        if auto_cl
        else ROUTE_BREAKDOWN_METRICS + TASK_PROBE_ROUTE_BREAKDOWN_METRICS
    )
    rows = [{"Metric": display_name} for display_name, _source_col in metrics]
    for stage in stages:
        zcol = f"z{stage}-{int(getattr(args, 'zoo_total_num', stage))}"
        main_row = next(
            (row for row in stage_rows.get(stage, []) if str(row.get("Method", "")) == "TSRouter-main"),
            {},
        )
        fast_row = next(
            (row for row in stage_rows.get(stage, []) if str(row.get("Method", "")) == "TSRouter-fast"),
            {},
        )
        autoforecast_row = next(
            (row for row in stage_rows.get(stage, []) if str(row.get("Method", "")) == "AutoForecast"),
            {},
        )
        autoxpcr_row = next(
            (row for row in stage_rows.get(stage, []) if str(row.get("Method", "")) == "AutoXPCR"),
            {},
        )
        simplets_row = next(
            (row for row in stage_rows.get(stage, []) if str(row.get("Method", "")) == "SimpleTS"),
            {},
        )
        task_probe_row = next(
            (row for row in stage_rows.get(stage, []) if str(row.get("Method", "")) == "Task-probe-M"),
            {},
        )
        auto_row = next(
            (
                row
                for row in stage_rows.get(stage, [])
                if str(row.get("Method", "")) == "TSRouter-autocl"
            ),
            {},
        )
        for row, (display_name, source_col) in zip(rows, metrics):
            if display_name.startswith("Task-probe-"):
                source_row = task_probe_row
            elif display_name.startswith("TSRouter-fast-"):
                source_row = fast_row
            elif display_name.startswith("AutoForecast-"):
                source_row = autoforecast_row
            elif display_name.startswith("AutoXPCR-"):
                source_row = autoxpcr_row
            elif display_name.startswith("SimpleTS-"):
                source_row = simplets_row
            elif display_name.startswith("TSRouter-autocl-"):
                source_row = auto_row
            else:
                source_row = main_row
            row[zcol] = _format_breakdown_value(source_row.get(f"_mean_{source_col}", np.nan))
    return pd.DataFrame(rows)


def _table5_method_specs(args) -> list[dict[str, object]]:
    auto_cl = _summary_auto_cl_enabled(args)
    main_summary_method = "TSRouter-autocl" if auto_cl else "TSRouter-main"
    main_display = "TSRouter-autocl" if auto_cl else "TSRouter"
    shared_incoming_profile_metric = f"{main_display}-Step2InsertRuntime(s)"
    return [
        {
            "summary_method": main_summary_method,
            "display_method": main_display,
            "insert_components": [
                ("IncomingProfile", [shared_incoming_profile_metric]),
                ("IndexRefresh", [f"{main_display}-IndexRefresh(s)"]),
            ],
        },
        {
            "summary_method": "TSRouter-fast",
            "display_method": "TSRouter-fast",
            "insert_components": [
                ("IncomingProfile", [shared_incoming_profile_metric]),
                ("IndexRefresh", [f"{main_display}-IndexRefresh(s)"]),
            ],
        },
        {
            "summary_method": "AutoForecast",
            "display_method": "AutoForecast",
            "insert_components": [
                ("IncomingProfile", ["AutoForecast-IncomingProfile(s)"]),
                (
                    "Retrain_total",
                    [
                        "AutoForecast-LabelRefresh(s)",
                        "AutoForecast-FeatureRefresh(s)",
                        "AutoForecast-Retrain(s)",
                    ],
                ),
            ],
        },
        {
            "summary_method": "AutoXPCR",
            "display_method": "AutoXPCR",
            "insert_components": [
                ("IncomingProfile", ["AutoXPCR-IncomingProfile(s)"]),
                (
                    "Retrain_total",
                    [
                        "AutoXPCR-LabelRefresh(s)",
                        "AutoXPCR-FeatureRefresh(s)",
                        "AutoXPCR-ResourceRefresh(s)",
                        "AutoXPCR-Retrain(s)",
                    ],
                ),
            ],
        },
        {
            "summary_method": "SimpleTS",
            "display_method": "SimpleTS",
            "insert_components": [
                ("IncomingProfile", ["SimpleTS-IncomingProfile(s)"]),
                (
                    "Retrain_total",
                    [
                        "SimpleTS-LabelRefresh(s)",
                        "SimpleTS-StructureRefresh(s)",
                        "SimpleTS-Retrain(s)",
                    ],
                ),
            ],
        },
        {
            "summary_method": "Task-probe-M",
            "display_method": "Task-probe",
            "insert_components": [
                ("IncomingProfile", []),
                ("Retrain_total", []),
            ],
            "insert_unavailable": True,
        },
    ]


def _table5_stat_value(stats: dict[str, object], key: str) -> float:
    value = _to_numeric_or_nan(stats.get(key, np.nan))
    return float(value) if np.isfinite(value) else np.nan


def _build_combined_overhead_table(
    args,
    table3: pd.DataFrame,
    stage_rows: dict[int, list[dict[str, object]]],
) -> pd.DataFrame:
    """Build Table5: method-major Insert/Route/E2E overhead by stage."""
    stages = sorted(int(stage) for stage in stage_rows)
    target_zoo = int(getattr(args, "zoo_total_num", stages[-1] if stages else 0))
    stage_output_cols = {stage: f"z{stage}-{target_zoo}" for stage in stages}
    table3_stage_cols = {
        int(stage): str(column)
        for stage, column in dict(table3.attrs.get("stage_columns", {}) or {}).items()
    }
    table3_values = (
        table3.set_index("Metric", drop=False) if not table3.empty and "Metric" in table3.columns else pd.DataFrame()
    )

    def insert_component_stats(
        metric_labels: list[str],
        stage: int,
        *,
        unavailable: bool,
    ) -> dict[str, object]:
        if unavailable or not metric_labels:
            return {
                "total": np.nan,
                "mean": np.nan,
                "p50": np.nan,
                "p95": np.nan,
                "n": 0,
            }
        source_col = table3_stage_cols.get(int(stage), "")
        values: list[float] = []
        for metric in metric_labels:
            if (
                table3_values.empty
                or metric not in table3_values.index
                or source_col not in table3_values.columns
            ):
                return {
                    "total": np.nan,
                    "mean": np.nan,
                    "p50": np.nan,
                    "p95": np.nan,
                    "n": 0,
                }
            value = _to_numeric_or_nan(table3_values.loc[metric, source_col])
            if not np.isfinite(value) or value < 0:
                return {
                    "total": np.nan,
                    "mean": np.nan,
                    "p50": np.nan,
                    "p95": np.nan,
                    "n": 0,
                }
            values.append(float(value))
        total = float(sum(values))
        # Insert CSV rows are one aggregate measurement for a stage/config.
        return {
            "total": total,
            "mean": total,
            "p50": total,
            "p95": total,
            "n": 1,
        }

    def summary_for_stage(stage: int, method: str) -> dict[str, object]:
        return next(
            (
                row
                for row in stage_rows.get(int(stage), [])
                if str(row.get("Method", "")) == str(method)
            ),
            {},
        )

    rows: list[dict[str, object]] = []
    cell_stats: dict[tuple[str, str, str, int], dict[str, object]] = {}
    overhead_totals: dict[str, dict[int, dict[str, dict[str, object]]]] = {}

    for spec in _table5_method_specs(args):
        summary_method = str(spec["summary_method"])
        display_method = str(spec["display_method"])
        insert_unavailable = bool(spec.get("insert_unavailable", False))
        insert_components = list(spec["insert_components"])
        insert_total_metric_labels = [
            str(metric)
            for _component, metric_labels in insert_components
            for metric in list(metric_labels)
        ]
        method_rows: list[tuple[str, str, dict[int, dict[str, object]]]] = []
        for component, metric_labels in insert_components:
            stats_by_stage = {
                stage: insert_component_stats(
                    list(metric_labels),
                    stage,
                    unavailable=insert_unavailable,
                )
                for stage in stages
            }
            method_rows.append(("Insert", str(component), stats_by_stage))

        for source_key, component in [
            ("sample_seconds", "Sample"),
            ("sample_to_route_seconds", "Sample_to_route"),
        ]:
            stats_by_stage = {}
            for stage in stages:
                summary = summary_for_stage(stage, summary_method)
                component_map = dict(summary.get("_timing_component_stats", {}) or {})
                stats_by_stage[stage] = dict(component_map.get(source_key, {}) or {})
            method_rows.append(("Route", component, stats_by_stage))

        e2e_by_stage = {}
        for stage in stages:
            summary = summary_for_stage(stage, summary_method)
            component_map = dict(summary.get("_timing_component_stats", {}) or {})
            e2e_by_stage[stage] = dict(component_map.get("e2e_total_seconds", {}) or {})
        method_rows.append(("E2E", "Total", e2e_by_stage))

        overhead_totals[display_method] = {}
        for stage in stages:
            summary = summary_for_stage(stage, summary_method)
            component_map = dict(summary.get("_timing_component_stats", {}) or {})
            route_total_stats = dict(
                component_map.get("route_total_seconds", {}) or {}
            )
            e2e_total_stats = dict(
                component_map.get("e2e_total_seconds", {}) or {}
            )
            # Table4 is mean latency over expected datasets.  Table6.1 must
            # preserve that per-request total-latency semantic rather than sum
            # unrelated benchmark datasets into one artificial duration.
            route_total_stats["total"] = route_total_stats.get("mean", np.nan)
            e2e_total_stats["total"] = e2e_total_stats.get("mean", np.nan)
            overhead_totals[display_method][stage] = {
                "Insert": insert_component_stats(
                    insert_total_metric_labels,
                    stage,
                    unavailable=insert_unavailable,
                ),
                "Route": route_total_stats,
                "E2E": e2e_total_stats,
            }

        for overhead, component, stats_by_stage in method_rows:
            rec: dict[str, object] = {
                "Method": display_method,
                "Overhead": overhead,
                "Component": component,
            }
            for stage in stages:
                stats = dict(stats_by_stage.get(stage, {}) or {})
                cell_stats[(display_method, overhead, component, stage)] = stats
                rec[stage_output_cols[stage]] = _format_breakdown_value(
                    stats.get("mean", np.nan)
                )
            mean_stages = [stage for stage in stages if stage >= 4]
            if overhead == "Insert":
                # Insert has one measured maintenance value per stage.  Its
                # Table5 quantiles are therefore computed across stage values,
                # unlike Route/E2E whose quantiles are first computed across
                # test requests within each stage.
                stage_values = [
                    _table5_stat_value(
                        dict(stats_by_stage.get(stage, {}) or {}),
                        "mean",
                    )
                    for stage in mean_stages
                ]
                stage_stats = _timing_stats(stage_values)
                for stat_key, output_col in [
                    ("p50", "P50_mean_stage4_last(s)"),
                    ("p95", "P95_mean_stage4_last(s)"),
                    ("mean", "Total_mean_stage4_last(s)"),
                ]:
                    rec[output_col] = _format_breakdown_value(
                        stage_stats.get(stat_key, np.nan)
                    )
            else:
                for stat_key, output_col in [
                    ("p50", "P50_mean_stage4_last(s)"),
                    ("p95", "P95_mean_stage4_last(s)"),
                    ("mean", "Total_mean_stage4_last(s)"),
                ]:
                    values = [
                        _table5_stat_value(
                            dict(stats_by_stage.get(stage, {}) or {}),
                            stat_key,
                        )
                        for stage in mean_stages
                    ]
                    values = [value for value in values if np.isfinite(value)]
                    rec[output_col] = _format_breakdown_value(
                        float(np.mean(values)) if values else np.nan
                )
            rows.append(rec)

    table = pd.DataFrame(rows)
    table.attrs["cell_stats"] = cell_stats
    table.attrs["overhead_totals"] = overhead_totals
    table.attrs["stage_columns"] = stage_output_cols
    return table


def _build_overhead_growth_table(
    table5: pd.DataFrame,
    *,
    statistic: str,
    target_stages: Iterable[int] = (5, 10, 15, 20, 40, 80),
) -> pd.DataFrame:
    """Build Table6.1/6.2 with observed points and all-stage linear estimates."""
    if statistic not in {"total", "p95"}:
        raise ValueError(f"unsupported Table6 statistic: {statistic}")
    overhead_totals = dict(table5.attrs.get("overhead_totals", {}) or {})
    targets = [int(stage) for stage in target_stages]
    rows: list[dict[str, object]] = []
    for method, by_stage_raw in overhead_totals.items():
        by_stage = {int(stage): dict(value or {}) for stage, value in dict(by_stage_raw).items()}
        for overhead in ["Insert", "Route", "E2E"]:
            observed = {
                stage: _table5_stat_value(
                    dict(by_stage.get(stage, {}).get(overhead, {}) or {}), statistic
                )
                for stage in sorted(by_stage)
            }
            observed = {
                stage: value for stage, value in observed.items() if np.isfinite(value)
            }
            fit_stages = sorted(observed)
            slope = np.nan
            intercept = np.nan
            if len(fit_stages) >= 2:
                fit_x = np.asarray(fit_stages, dtype=float)
                fit_y = np.asarray(
                    [observed[stage] for stage in fit_stages], dtype=float
                )
                slope, _unconstrained_intercept = np.polyfit(
                    fit_x,
                    fit_y,
                    1,
                )
                # These are growth curves for non-negative durations.  A noisy
                # negative OLS slope is projected onto the monotone linear
                # constraint instead of producing negative future runtimes.
                slope = max(0.0, float(slope))
                intercept = float(fit_y.mean() - slope * fit_x.mean())
            rec: dict[str, object] = {
                "Method": method,
                "Overhead": overhead,
                "Statistic": "Total" if statistic == "total" else "P95",
            }
            extrapolated: list[int] = []
            for stage in targets:
                if stage in observed:
                    value = observed[stage]
                elif np.isfinite(slope) and np.isfinite(intercept):
                    value = max(0.0, float(slope * stage + intercept))
                    extrapolated.append(stage)
                else:
                    value = np.nan
                rec[f"z{stage}(s)"] = _format_breakdown_value(value)
            rec["ObservedStages"] = ",".join(str(stage) for stage in fit_stages)
            rec["ExtrapolatedStages"] = ",".join(str(stage) for stage in extrapolated)
            rec["LinearSlope(s/stage)"] = _format_breakdown_value(slope)
            rec["LinearIntercept(s)"] = _format_breakdown_value(intercept)
            rec["FitConstraint"] = "slope>=0"
            rec["FitN"] = int(len(fit_stages))
            rows.append(rec)
    return pd.DataFrame(rows)


def _build_route_breakdown_by_profile(
    args,
    stages: list[int],
) -> pd.DataFrame:
    if not _summary_auto_cl_enabled(args):
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for stage in stages:
        auto_df, source = _main_selector_selection(args, stage, auto_cl=True)
        if auto_df.empty:
            continue
        for profile in _summary_auto_cl_profiles(args):
            profile_name = str(profile["adaptive_profile"])
            sub = auto_df[
                auto_df.get("adaptive_profile", pd.Series(dtype=str))
                .astype(str)
                .eq(profile_name)
            ].copy()
            if sub.empty:
                continue
            for col in [
                "sample_seconds",
                "sample_to_route_seconds",
                "route_final_seconds",
            ]:
                sub[col] = pd.to_numeric(
                    sub.get(col, pd.Series(dtype=float)), errors="coerce"
                )
            valid = (
                np.isfinite(sub["sample_seconds"])
                & np.isfinite(sub["sample_to_route_seconds"])
                & np.isfinite(sub["route_final_seconds"])
                & sub["sample_seconds"].ge(0)
                & sub["sample_to_route_seconds"].ge(0)
                & sub["route_final_seconds"].ge(0)
                & (
                    sub["route_final_seconds"]
                    - sub["sample_seconds"]
                    - sub["sample_to_route_seconds"]
                )
                .abs()
                .le(1e-6)
            )
            good = sub.loc[valid]
            rows.append(
                {
                    "Stage": int(stage),
                    "adaptive_profile": profile_name,
                    "resolved_eval_cl": str(profile["tsfm_results_dir"]),
                    "DatasetN": int(len(good)),
                    "sample_seconds_mean": (
                        float(good["sample_seconds"].mean()) if not good.empty else np.nan
                    ),
                    "sample_to_route_seconds_mean": (
                        float(good["sample_to_route_seconds"].mean())
                        if not good.empty
                        else np.nan
                    ),
                    "route_final_seconds_mean": (
                        float(good["route_final_seconds"].mean())
                        if not good.empty
                        else np.nan
                    ),
                    "Source": source,
                }
            )
    return pd.DataFrame(rows)


def _task_probe_source_paths(args, stages: list[int]) -> tuple[Path, Path, list[Path], str]:
    stage = int(stages[-1]) if stages else int(getattr(args, "zoo_total_num", 0) or 0)
    task_args = _args_for_stage(args, stage)
    root = task_probe_select_root(task_args).resolve()
    cache_path = task_probe_select_cache_path(task_args)
    cache_paths = []
    if _summary_auto_cl_enabled(args):
        for profile in _summary_auto_cl_profiles(args):
            profile_args = _args_for_stage(args, stage, auto_cl=True)
            profile_args.repr_input_dim = int(profile["repr_input_dim"])
            profile_args.repr_output_dim = int(profile["repr_output_dim"])
            profile_args.repr_sub_pred_len = int(profile["repr_sub_pred_len"])
            profile_args.repr_source_exact_length = int(
                profile["repr_source_exact_length"]
            )
            normalize_auto_cl_args(profile_args)
            cache_paths.append(task_probe_select_cache_path(profile_args))
    else:
        cache_paths.append(cache_path)
    timing_paths: list[Path] = []
    for candidate_cache in cache_paths:
        timing_paths.extend(
            path.resolve()
            for path in task_probe_sample_timing_csv_candidates(candidate_cache)
        )
    timing_paths = list(dict.fromkeys(timing_paths))
    return (
        root / "rank_summary.csv",
        root / "forward_summary.csv",
        timing_paths,
        ",".join(path.stem for path in cache_paths),
    )


def _print_task_probe_rank_sample_check(args, stages: list[int], expected: set[str]) -> None:
    rank_path, _forward_path, sample_paths, cache_stem = _task_probe_source_paths(args, stages)
    sample_path_text = ",".join(path.as_posix() for path in sample_paths)
    if not rank_path.exists():
        print(
            f"⚠️ [task-probe-check] rank_summary missing; "
            f"path={rank_path.as_posix()} cache={cache_stem} sample_timing={sample_path_text}"
        )
        return
    try:
        rank_df = pd.read_csv(rank_path, low_memory=False)
    except Exception as exc:
        print(f"⚠️ [task-probe-check] rank_summary read_error={type(exc).__name__}: {exc}; path={rank_path.as_posix()}")
        return
    if "sample_seconds" not in rank_df.columns:
        print(
            f"⚠️ [task-probe-check] rank_summary has no sample_seconds column; "
            f"path={rank_path.as_posix()} cache={cache_stem} sample_timing={sample_path_text}"
        )
        return
    cache_stems = {token for token in str(cache_stem).split(",") if token}
    sub = rank_df[
        rank_df.get("cache_stem", pd.Series(dtype=str))
        .astype(str)
        .isin(cache_stems)
    ].copy()
    if "task_probe_eval_protocol" in sub.columns:
        sub = sub[sub["task_probe_eval_protocol"].astype(str).eq("within_window_half_v1")].copy()
    valid = pd.to_numeric(sub.get("sample_seconds", pd.Series(dtype=float)), errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    dataset_count = 0
    if "dataset" in sub.columns:
        dataset_count = len(set(sub.loc[valid.notna(), "dataset"].dropna().astype(str)) & set(expected))
    if valid.dropna().empty:
        print(
            f"⚠️ [task-probe-check] rank_summary sample_seconds column exists but has no finite rows; "
            f"path={rank_path.as_posix()} cache={cache_stem} sample_timing={sample_path_text}"
        )
    else:
        print(
            f"✅ [task-probe-check] rank_summary sample_seconds rows={len(valid.dropna())} "
            f"datasets={dataset_count}/{len(expected)} path={rank_path.as_posix()} cache={cache_stem}"
        )


def _print_vldb_table_source_manifest(
    args,
    stages: list[int],
    table3_source: Path,
    path_check: Path,
) -> None:
    rank_path, forward_path, sample_paths, cache_stem = _task_probe_source_paths(args, stages)
    tsfm_dir = str(getattr(args, "TSFM_results_dir", "cl_512"))
    selector_root = Path(get_tsrouter_selector_result_dir(args)).resolve()
    auto_selector_root = None
    if _summary_auto_cl_enabled(args):
        auto_args = _args_for_stage(
            args,
            int(stages[-1]) if stages else int(getattr(args, "zoo_total_num", 0)),
            auto_cl=True,
        )
        auto_selector_root = Path(get_tsrouter_selector_result_dir(auto_args)).resolve()
    print("\n" + "=" * 88)
    print("VLDB Results Source Manifest")
    print("=" * 88)
    print(f"[source-check] detail_csv={path_check.as_posix()}")
    print(f"[Table1/Table2] TSFM metrics/runtime: {tsfm_csv_glob_display(tsfm_dir, root=Path(TSFM_CSV_ROOT).resolve())}all_results.csv")
    if _summary_auto_cl_enabled(args):
        print(
            f"[Table1/Table2/Table4] TSRouter-autocl selector rows: "
            f"{auto_selector_root.as_posix()}/stage<stage>/*{_summary_auto_cl_profile_token(args)}*_all_results.csv"
        )
    else:
        print(
            f"[Table1/Table2/Table4] TSRouter-main selector rows: "
            f"{selector_root.as_posix()}/stage<stage>/*_all_results.csv"
        )
    fast_args = _args_for_stage(
        args,
        int(stages[-1]) if stages else int(getattr(args, "zoo_total_num", 0)),
        auto_cl=False,
        param_overrides=_method_param_overrides("TSRouter-fast"),
    )
    af_args = _args_for_stage(
        args,
        int(stages[-1]) if stages else int(getattr(args, "zoo_total_num", 0)),
        auto_cl=False,
        param_overrides=_method_param_overrides("AutoForecast"),
    )
    xpcr_args = _args_for_stage(
        args,
        int(stages[-1]) if stages else int(getattr(args, "zoo_total_num", 0)),
        auto_cl=False,
        param_overrides=_method_param_overrides("AutoXPCR"),
    )
    st_args = _args_for_stage(
        args,
        int(stages[-1]) if stages else int(getattr(args, "zoo_total_num", 0)),
        auto_cl=False,
        param_overrides=_method_param_overrides("SimpleTS"),
    )
    profile_probe_args = {
        method: _args_for_stage(
            args,
            int(stages[-1]) if stages else int(getattr(args, "zoo_total_num", 0)),
            auto_cl=_summary_auto_cl_enabled(args),
            param_overrides=_method_param_overrides(method),
        )
        for method in sorted(PROFILE_PROBE_METHODS)
    }
    print(
        f"[Table1/Table2/Table4] TSRouter-fast selector rows: "
        f"{Path(get_tsrouter_selector_result_dir(fast_args)).resolve().as_posix()}/stage<stage>/*_all_results.csv"
    )
    print(
        f"[Table1/Table2/Table4] AutoForecast selector rows: "
        f"{Path(get_tsrouter_selector_result_dir(af_args)).resolve().as_posix()}/stage<stage>/*_all_results.csv"
    )
    print(
        f"[Table1/Table2/Table4] AutoXPCR selector rows: "
        f"{Path(get_tsrouter_selector_result_dir(xpcr_args)).resolve().as_posix()}/stage<stage>/*_all_results.csv"
    )
    print(
        f"[Table1/Table2/Table4] SimpleTS selector rows: "
        f"{Path(get_tsrouter_selector_result_dir(st_args)).resolve().as_posix()}/stage<stage>/*_all_results.csv"
    )
    for method in ["Profile-probe-M"]:
        print(
            f"[Table1/Table2] {method} selector rows: "
            f"{Path(get_tsrouter_selector_result_dir(profile_probe_args[method])).resolve().as_posix()}"
            "/stage<stage>/*_fb0*_all_results.csv"
        )
    print(f"[Table1/Table2/Table4] Task-probe rank: {rank_path.as_posix()} cache={cache_stem}")
    print(f"[Table1/Table2/Table4] Task-probe forward: {forward_path.as_posix()}")
    print(
        "[Table1/Table2/Table4] Task-probe sample timing: "
        + ", ".join(path.as_posix() for path in sample_paths)
    )
    print(f"[Table3] TSRouter insert timing: {table3_source.as_posix()}")
    print(
        "[Table3] AutoForecast insert timing: "
        f"{(BASELINE_CSV_ROOT / 'selectors' / 'AutoForecast_Select' / 'step3_insert_timing.csv').as_posix()}"
    )
    print(
        "[Table3] AutoXPCR insert timing: "
        f"{(BASELINE_CSV_ROOT / 'selectors' / 'AutoXPCR_Select' / 'step3_insert_timing.csv').as_posix()}"
    )
    print(
        "[Table3] SimpleTS insert timing: "
        f"{(BASELINE_CSV_ROOT / 'selectors' / 'SimpleTS_Select' / 'step3_insert_timing.csv').as_posix()}"
    )


def _print_table4_source_check(args, source_checks: list[dict[str, object]]) -> None:
    main_method = (
        "TSRouter-autocl"
        if _summary_auto_cl_enabled(args)
        else "TSRouter-main"
    )
    main_checks = [
        row for row in source_checks
        if str(row.get("Method", "")) == main_method
    ]
    selector_root = Path(get_tsrouter_selector_result_dir(args)).resolve()
    checked_stages = _format_stage_ranges(int(row["Stage"]) for row in main_checks)
    print(
        f"[check-dir] stage_parent={selector_root.as_posix()}/ "
        f"stage_subdirs={checked_stages or 'none'} fallback_root={selector_root.as_posix()}/"
    )
    if not main_checks:
        print(f"[main-result-status] no {main_method} source checks")
        return
    complete_stages = [int(row["Stage"]) for row in main_checks if bool(row.get("Complete", False))]
    incomplete_stages = [int(row["Stage"]) for row in main_checks if not bool(row.get("Complete", False))]
    if incomplete_stages:
        print(
            f"[main-result-status] failed_stages={_format_stage_ranges(incomplete_stages)}; "
            f"exact missing files are listed in the {main_method} "
            "[vldb-check] block above"
        )
    else:
        print(f"[main-result-status] complete_stages={_format_stage_ranges(complete_stages)}")


def _table_value_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null", "/"}


def _table1_optional_process_column(column: object) -> bool:
    text = str(column)
    return any(text.startswith(prefix) for prefix in TABLE1_OPTIONAL_PROCESS_PREFIXES)


def _compact_items(items: Iterable[object], limit: int = 8) -> str:
    values = list(dict.fromkeys(str(item) for item in items))
    if len(values) <= limit:
        return ",".join(values)
    return f"{','.join(values[:limit])},...(+{len(values) - limit})"


def _source_check_reason(
    check: dict[str, object] | None,
    *,
    metric: str = "",
    runtime: bool = False,
) -> str:
    if not check:
        return "source check row missing"
    rows = _to_numeric_or_nan(check.get("Rows", np.nan))
    note = str(check.get("Note", "") or "").strip()
    source = str(check.get("Source", "") or "").strip()
    if np.isfinite(rows) and rows <= 0:
        return note or source or "source contains no result rows"
    if metric and not _truthy(check.get("MetricComplete", False)):
        return f"{metric} dataset coverage incomplete"
    if runtime and not _truthy(check.get("RuntimeComplete", False)):
        return "runtime timing coverage incomplete"
    if not _truthy(check.get("Complete", False)):
        return note or "source/dataset/runtime coverage incomplete"
    return note or "required table fields are missing"


def _source_checks_by_stage_method(
    source_checks: list[dict[str, object]],
) -> dict[tuple[int, str], dict[str, object]]:
    return {
        (int(row.get("Stage", 0) or 0), str(row.get("Method", ""))): row
        for row in source_checks
        if str(row.get("Method", ""))
    }


def _table1_missing_reasons(
    table: pd.DataFrame,
    source_checks: list[dict[str, object]],
    latest_stage: int,
) -> list[str]:
    checks = _source_checks_by_stage_method(source_checks)
    warnings: list[str] = []
    for _, row in table.iterrows():
        method = str(row.get("Method", "") or "")
        missing = [
            col
            for col in table.columns
            if (
                col != "Method"
                and not _table1_optional_process_column(col)
                and _table_value_missing(row.get(col))
            )
        ]
        if not missing:
            continue
        runtime = any("P50/95" in col or "P50/P95" in col or "throughput" in col for col in missing)
        metric = "MASE/CRPS" if any(
            col in {"MASE", "Regret-M", "Regret-M P90", "Rank-M", "MASE-hit1/3", "CRPS", "Rank-C", "CRPS-hit1/3", "sMAPE"}
            for col in missing
        ) else ""
        reason = _source_check_reason(
            checks.get((int(latest_stage), method)),
            metric=metric,
            runtime=runtime,
        )
        warnings.append(
            f"{method}: missing columns {_compact_items(missing)}; {reason}"
        )
    return warnings


def _table2_missing_reasons(
    table: pd.DataFrame,
    source_checks: list[dict[str, object]],
    metric: str,
) -> list[str]:
    checks = _source_checks_by_stage_method(source_checks)
    stage_cols = [col for col in table.columns if re.fullmatch(r"z\d+-\d+", str(col))]
    warnings: list[str] = []
    for method, row in table.iterrows():
        missing_cols = [col for col in stage_cols if _table_value_missing(row.get(col))]
        if not missing_cols:
            continue
        reasons = []
        for col in missing_cols:
            match = re.fullmatch(r"z(\d+)-\d+", str(col))
            stage = int(match.group(1)) if match else 0
            reasons.append(
                _source_check_reason(
                    checks.get((stage, str(method))),
                    metric=metric,
                )
            )
        warnings.append(
            f"{method}: missing {metric} at {_compact_items(missing_cols)}; "
            f"{_compact_items(reasons, limit=3)}"
        )
    return warnings


def _table3_missing_reasons(table: pd.DataFrame) -> list[str]:
    warnings = list(table.attrs.get("missing_reasons", []) or [])
    if len(table.columns) <= 1 and not warnings:
        warnings.append("no stage columns were produced")
    return warnings


def _table4_missing_reasons(
    args,
    table: pd.DataFrame,
    source_checks: list[dict[str, object]],
) -> list[str]:
    checks = _source_checks_by_stage_method(source_checks)
    stage_cols = [col for col in table.columns if re.fullmatch(r"z\d+-\d+", str(col))]
    missing_by_method: dict[str, set[str]] = {}
    source_method_by_display = {
        "TSRouter-autocl": "TSRouter-autocl",
        "TSRouter-fast": "TSRouter-fast",
        "AutoForecast": "AutoForecast",
        "AutoXPCR": "AutoXPCR",
        "SimpleTS": "SimpleTS",
        "Task-probe": "Task-probe-M",
        "TSRouter": "TSRouter-autocl" if _summary_auto_cl_enabled(args) else "TSRouter-main",
    }
    display_methods = sorted(source_method_by_display, key=len, reverse=True)
    for _, row in table.iterrows():
        metric_name = str(row.get("Metric", "") or "")
        display_method = next(
            (name for name in display_methods if metric_name.startswith(f"{name}-")),
            "",
        )
        if not display_method:
            continue
        for col in stage_cols:
            if _table_value_missing(row.get(col)):
                missing_by_method.setdefault(display_method, set()).add(str(col))
    warnings: list[str] = []
    for display_method, missing_cols in missing_by_method.items():
        ordered_cols = [col for col in stage_cols if col in missing_cols]
        reasons = []
        source_method = source_method_by_display[display_method]
        for col in ordered_cols:
            match = re.fullmatch(r"z(\d+)-\d+", col)
            stage = int(match.group(1)) if match else 0
            reasons.append(
                _source_check_reason(
                    checks.get((stage, source_method)),
                    runtime=True,
                )
            )
        warnings.append(
            f"{display_method}: missing route breakdown at {_compact_items(ordered_cols)}; "
            f"{_compact_items(reasons, limit=3)}"
        )
    return warnings


def _print_table_missing_reasons(reasons: Iterable[str]) -> None:
    for reason in dict.fromkeys(str(reason) for reason in reasons if str(reason).strip()):
        print(f"⚠ {reason}")


def run_vldb_results_fast_baselines(
    args,
    baseline_df_all: pd.DataFrame,
    ordered_model_names: list[str],
    season_naive_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    baseline_df_all = _summary_baseline_df(
        args, baseline_df_all, ordered_model_names
    )
    stages = _stage_list(args)
    if not stages:
        print("[vldb_results] no stages to summarize")
        return {}

    stage_rows: dict[int, list[dict[str, object]]] = {}
    source_checks: list[dict[str, object]] = []
    summary_mode = _summary_auto_cl_mode(args)
    print(
        f"[vldb_results] mode={summary_mode} stages={stages} "
        f"TSFM_baseline={getattr(args, 'TSFM_results_dir', 'cl_512')}",
        flush=True,
    )
    for index, stage in enumerate(stages, start=1):
        stage_started = time.perf_counter()
        rows, checks = _stage_summaries(
            args=args,
            baseline_df_all=baseline_df_all,
            ordered_model_names=ordered_model_names,
            stage=int(stage),
            season_naive_df=season_naive_df,
        )
        stage_rows[int(stage)] = rows
        source_checks.extend(checks)
        print(
            f"[vldb_results][stage {index}/{len(stages)}] "
            f"done z{int(stage)} rows={len(rows)} "
            f"elapsed={time.perf_counter() - stage_started:.2f}s",
            flush=True,
        )

    print("[vldb_results][tables] start building Table1-6/source_check", flush=True)
    tables_started = time.perf_counter()
    table_phase_started = time.perf_counter()
    latest_stage = int(stages[-1])
    table1_raw = _table1_from_rows(stage_rows[latest_stage])
    table2_mase_raw = _table2_from_stage_rows(args, "MASE", stage_rows)
    table2_crps_raw = _table2_from_stage_rows(args, "CRPS", stage_rows)
    table1 = _table1_mark_best(table1_raw)
    table2_mase = _table2_mark_best(table2_mase_raw)
    table2_crps = _table2_mark_best(table2_crps_raw)
    print(
        f"[vldb_results][tables] table1/table2 built "
        f"elapsed={time.perf_counter() - table_phase_started:.2f}s",
        flush=True,
    )
    table_phase_started = time.perf_counter()
    table3, table3_profile, table3_source, table3_note = (
        _build_insert_breakdown_table(args, stages=stages)
    )
    print(
        f"[vldb_results][tables] table3 built "
        f"elapsed={time.perf_counter() - table_phase_started:.2f}s",
        flush=True,
    )
    table_phase_started = time.perf_counter()
    table4 = _build_route_breakdown_table(args, stage_rows)
    table4_profile = _build_route_breakdown_by_profile(args, stages)
    print(
        f"[vldb_results][tables] table4 built "
        f"elapsed={time.perf_counter() - table_phase_started:.2f}s",
        flush=True,
    )
    table_phase_started = time.perf_counter()
    table5 = _build_combined_overhead_table(args, table3, stage_rows)
    table6_1 = _build_overhead_growth_table(table5, statistic="total")
    table6_2 = _build_overhead_growth_table(table5, statistic="p95")
    table6_3 = _build_overhead_growth_figure_table(table5)
    print(
        f"[vldb_results][tables] table5/table6 built "
        f"elapsed={time.perf_counter() - table_phase_started:.2f}s",
        flush=True,
    )
    table_phase_started = time.perf_counter()
    check_table = pd.DataFrame(source_checks)
    for col in SOURCE_CHECK_COLUMNS:
        if col not in check_table.columns:
            check_table[col] = ""
    check_table = check_table[SOURCE_CHECK_COLUMNS]

    file_token = _summary_file_token(args)
    path1 = _write_table(
        table1, f"vldb_results_{file_token}table1_latest_stage.csv", index=False
    )
    path2m = _write_table(
        table2_mase,
        f"vldb_results_{file_token}table2_mase_by_stage.csv",
        index=True,
    )
    path2c = _write_table(
        table2_crps,
        f"vldb_results_{file_token}table2_crps_by_stage.csv",
        index=True,
    )
    path1_raw = _write_table(
        table1_raw,
        f"vldb_results_{file_token}table1_latest_stage_raw.csv",
        index=False,
    )
    path2m_raw = _write_table(
        table2_mase_raw,
        f"vldb_results_{file_token}table2_mase_by_stage_raw.csv",
        index=True,
    )
    path2c_raw = _write_table(
        table2_crps_raw,
        f"vldb_results_{file_token}table2_crps_by_stage_raw.csv",
        index=True,
    )
    path3 = _write_table(
        table3,
        f"vldb_results_{file_token}table3_insert_breakdown_by_stage.csv",
        index=False,
    )
    path3_profile = (
        _write_table(
            table3_profile,
            f"vldb_results_{file_token}table3_insert_breakdown_by_profile.csv",
            index=False,
        )
        if summary_mode != "v0"
        else None
    )
    path4 = _write_table(
        table4,
        f"vldb_results_{file_token}table4_route_breakdown_by_stage.csv",
        index=False,
    )
    path5 = _write_table(
        table5,
        f"vldb_results_{file_token}table5_combined_overhead_by_stage.csv",
        index=False,
    )
    path6_1 = _write_table(
        table6_1,
        f"vldb_results_{file_token}table6_1_total_overhead_growth.csv",
        index=False,
    )
    path6_2 = _write_table(
        table6_2,
        f"vldb_results_{file_token}table6_2_p95_overhead_growth.csv",
        index=False,
    )
    path6_3 = _write_table(
        table6_3,
        f"vldb_results_{file_token}table6_3_figure4_p95_growth_points.csv",
        index=False,
    )
    path4_profile = (
        _write_table(
            table4_profile,
            f"vldb_results_{file_token}table4_route_breakdown_by_profile.csv",
            index=False,
        )
        if summary_mode != "v0"
        else None
    )
    path_check = _write_table(
        check_table,
        f"vldb_results_{file_token}source_check.csv",
        index=False,
    )
    figure_paths = _write_vldb_figures(
        args=args,
        baseline_df_all=baseline_df_all,
        ordered_model_names=ordered_model_names,
        latest_stage=latest_stage,
        table2_mase_raw=table2_mase_raw,
        table5=table5,
        season_naive_df=season_naive_df,
        latest_stage_rows=stage_rows.get(latest_stage, []),
        table6_3=table6_3,
    )
    print(
        f"[vldb_results][tables] wrote tables/source_check "
        f"elapsed={time.perf_counter() - table_phase_started:.2f}s "
        f"total={time.perf_counter() - tables_started:.2f}s",
        flush=True,
    )
    expected = _expected_datasets(bool(getattr(args, "quick_test", False)))

    print("\n" + "=" * 88)
    print("VLDB Results Source Check: method/file/runtime coverage")
    print("=" * 88)
    print(f"[write] {path_check.as_posix()}")
    _print_source_checks_by_method(args, source_checks)
    _print_vldb_table_source_manifest(args, stages, table3_source, path_check)
    _print_task_probe_rank_sample_check(args, stages, expected)

    print("\n" + "=" * 88)
    print(
        f"VLDB Results Table1: mode={summary_mode}, latest-stage main vs "
        f"fast baselines (stage={latest_stage})"
    )
    print("=" * 88)
    _print_table_missing_reasons(
        _table1_missing_reasons(table1_raw, source_checks, latest_stage)
    )
    print("[main-param-grid] src/cli/vldb_fast_baselines.py::VLDB_RESULTS_MAIN_PARAM_GRID")
    if summary_mode != "v0":
        print(f"[auto-cl-profile-overlay] mode={summary_mode}")
    print(f"[tsfm-comparison-baseline] {getattr(args, 'TSFM_results_dir', '')}")
    print(f"[write] display={path1.as_posix()} raw={path1_raw.as_posix()}")
    print("[display] ★ marks the best performance/efficiency value in each column")
    print(tabulate(table1, headers="keys", tablefmt="plain", floatfmt=".3f", numalign="decimal", stralign="left", showindex=False))

    print("\n" + "=" * 88)
    print("VLDB Results Table2-MASE: all-stage main vs fast baselines")
    print("=" * 88)
    _print_table_missing_reasons(
        _table2_missing_reasons(table2_mase_raw, source_checks, "MASE")
    )
    print("[reference] Win_vs_Best_TSFM compares every method with Current_best-M")
    print(f"[write] display={path2m.as_posix()} raw={path2m_raw.as_posix()}")
    print(tabulate(table2_mase, headers="keys", tablefmt="plain", floatfmt=".3f", numalign="decimal", stralign="left"))

    print("\n" + "=" * 88)
    print("VLDB Results Table2-CRPS: all-stage main vs fast baselines")
    print("=" * 88)
    _print_table_missing_reasons(
        _table2_missing_reasons(table2_crps_raw, source_checks, "CRPS")
    )
    print("[reference] Win_vs_Best_TSFM compares every method with Current_best-C")
    print(f"[write] display={path2c.as_posix()} raw={path2c_raw.as_posix()}")
    print(tabulate(table2_crps, headers="keys", tablefmt="plain", floatfmt=".3f", numalign="decimal", stralign="left"))

    print("\n" + "=" * 88)
    print("VLDB Results Table3: TSRouter insert breakdown by stage")
    print("=" * 88)
    _print_table_missing_reasons(_table3_missing_reasons(table3))
    print(f"[source] {table3_source.as_posix()} ({table3_note})")
    profile_text = (
        f" profile_audit={path3_profile.as_posix()}"
        if path3_profile is not None
        else ""
    )
    print(f"[write] aggregate={path3.as_posix()}{profile_text}")
    print(tabulate(table3, headers="keys", tablefmt="plain", floatfmt=".4f", numalign="decimal", stralign="left", showindex=False))

    print("\n" + "=" * 88)
    print("VLDB Results Table4: TSRouter route breakdown mean seconds by stage")
    print("=" * 88)
    _print_table_missing_reasons(
        _table4_missing_reasons(args, table4, source_checks)
    )
    print("[meaning] each cell is mean seconds over expected datasets for that stage")
    print(
        "[source] TSRouter rows use VLDB_RESULTS_MAIN_PARAM_GRID selector result rows; "
        "Task-probe sample uses rank_summary/sample-timing CSV and forward uses forward_summary summed by stage model list"
    )
    _print_table4_source_check(args, source_checks)
    profile_text = (
        f" profile_audit={path4_profile.as_posix()}"
        if path4_profile is not None
        else ""
    )
    print(f"[write] aggregate={path4.as_posix()}{profile_text}")
    print(tabulate(table4, headers="keys", tablefmt="plain", floatfmt=".4f", numalign="decimal", stralign="left", showindex=False))

    print("\n" + "=" * 88)
    print("VLDB Results Table5: combined Insert/Route/E2E overhead by method and stage")
    print("=" * 88)
    print(
        "[meaning] Insert cells are one measured insert total; Route cells exactly "
        "match Table4 per-dataset means; E2E cells are per-dataset mean total latency. "
        "The last three columns average per-stage P50/P95/mean-total over stage4 "
        "through the last real stage"
    )
    print(
        "[insert] Advanced baselines use their own train-scope profile-forward "
        "measurement; full_pool uses the latest model's pool forward time. "
        "TSRouter-fast reuses the complete TSRouter "
        "insert breakdown; non-TSRouter Retrain_total merges label, structure/feature, "
        "resource (when present), and retrain; Task-probe Insert is N/A"
    )
    print(f"[write] {path5.as_posix()}")
    print(tabulate(table5, headers="keys", tablefmt="plain", floatfmt=".4f", numalign="decimal", stralign="left", showindex=False))

    print("\n" + "=" * 88)
    print("VLDB Results Table6.1: total overhead growth curve")
    print("=" * 88)
    print(
        "[meaning] exact stage points use observed data; missing points use one linear "
        "fit over every finite observed stage for that method/overhead; Route/E2E "
        "totals preserve Table5 per-dataset mean-latency semantics"
    )
    print(f"[write] {path6_1.as_posix()}")
    print(tabulate(table6_1, headers="keys", tablefmt="plain", floatfmt=".4f", numalign="decimal", stralign="left", showindex=False))

    print("\n" + "=" * 88)
    print("VLDB Results Table6.2: P95 overhead growth curve")
    print("=" * 88)
    print(f"[write] {path6_2.as_posix()}")
    print(tabulate(table6_2, headers="keys", tablefmt="plain", floatfmt=".4f", numalign="decimal", stralign="left", showindex=False))

    print("\n" + "=" * 88)
    print("VLDB Results Table6.3: Figure4 P95 growth points")
    print("=" * 88)
    print(
        "[meaning] Figure4 uses these six x-axis points only. Route uses exact "
        "stage 5/10/15/20 points; Insert uses window means for 1-5, 6-10, "
        "11-15, and 16-20. Route z40/z80 use linear fits; non-TSRouter Insert "
        "z40/z80 use an upper-envelope log-growth fit; the x-axis DisplayX is "
        "compressed for plotting"
    )
    print(f"[write] {path6_3.as_posix()}")
    print(tabulate(table6_3, headers="keys", tablefmt="plain", floatfmt=".4f", numalign="decimal", stralign="left", showindex=False))

    if figure_paths:
        print("\n" + "=" * 88)
        print("VLDB Results Figures")
        print("=" * 88)
        for name, path in figure_paths.items():
            print(f"[write] {name}={path.as_posix()}")

    return {
        "table1": table1_raw,
        "table1_display": table1,
        "table2_mase": table2_mase_raw,
        "table2_mase_display": table2_mase,
        "table2_crps": table2_crps_raw,
        "table2_crps_display": table2_crps,
        "table3_insert_breakdown": table3,
        "table3_insert_breakdown_profile": table3_profile,
        "table4_route_breakdown": table4,
        "table4_route_breakdown_profile": table4_profile,
        "table5_combined_overhead": table5,
        "table6_1_total_overhead_growth": table6_1,
        "table6_2_p95_overhead_growth": table6_2,
        "table6_3_figure4_p95_growth_points": table6_3,
        "source_check": check_table,
        "figure_paths": pd.DataFrame(
            [{"figure": name, "path": path.as_posix()} for name, path in figure_paths.items()]
        ),
    }

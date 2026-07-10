import copy
import glob
import pickle
import ast
import json
from pathlib import Path
from itertools import product
import os
import sys

if __package__ in {None, ""}:
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

import numpy as np
import pandas as pd
from tabulate import tabulate

try:
    from analysis.tsfm_context_summary import (
        DEFAULT_CONTEXT_LENS,
        DEFAULT_TRADEOFF_METRICS,
        default_figure_dir,
        default_output_csv_dir,
        parse_context_lens,
        summarize_tsfm_context_lens,
    )
except ImportError:
    DEFAULT_CONTEXT_LENS = ("64", "128", "256", "512")
    DEFAULT_TRADEOFF_METRICS = ("MASE", "CRPS")

    def default_figure_dir():
        return "results_csv/TSRouter/vldb/figures"

    def default_output_csv_dir():
        return "results_csv/TSRouter/vldb/context_summary"

    def parse_context_lens(value):
        return [int(x) for x in str(value).split(",") if str(x).strip()]

    def summarize_tsfm_context_lens(*args, **kwargs):
        return None

try:
    from analysis.channel_failure_analysis import run_channel_failure_analysis_for_check_selector
except ImportError:
    def run_channel_failure_analysis_for_check_selector(*args, **kwargs):
        raise RuntimeError("Supplementary channel analysis is not included in the public method package.")

from cli.vldb_fast_baselines import run_vldb_results_fast_baselines, vldb_results_param_grid
from config.model_zoo_config import Model_zoo_details, Model_abbrev_map, All_model_names, All_sorted_model_names
from selector.selector_config import Selector_zoo_details
from utils.check_tools import (
    check_results_file,
    standardize_model_names,
    calculate_order_metrics,
)
from utils.tsrouter_metrics import (
    COMPETENCE_REGION_METRIC_COLUMNS,
    TSROUTER_EXTRA_METRIC_COLUMNS,
    TSROUTER_CORE_METRIC_COLUMNS,
    load_encoder_enrichment_for_args,
)
from utils.path_utils import (
    TSROUTER_MODEL_REPR_DIR,
    get_tsrouter_repr_forward_dir,
    get_tsrouter_selector_result_dir,
    get_tsrouter_selector_stage_result_dir,
    materialize_compatible_tsrouter_result,
    resolve_tsfm_artifact_path,
    resolve_tsfm_csv_path,
    resolve_tsfm_result_path,
)
from utils.project_paths import (
    PROJECT_ROOT,
    TSFM_CSV_ROOT,
    TSROUTER_REPR_FORWARD_CSV_ROOT,
    rel,
)

from config.dataset_config import ALL_Fast_DATASETS, ALL_DATASETS


import re
from collections import defaultdict

SELECTOR_VALID_EXPECTED_DATASETS_COL = "N_VALID_EXPECTED_DS"
SELECTOR_ROUTE_P50_COL = "Route P50(s)"
SELECTOR_ROUTE_P95_COL = "Route P95(s)"
SELECTOR_ROUTE_P50P95_COL = "Route P50/95(s)"
SELECTOR_CORE_ROUTE_P50_COL = "Core-route P50(s)"
SELECTOR_CORE_ROUTE_P95_COL = "Core-route P95(s)"
SELECTOR_CORE_ROUTE_P50P95_COL = "Core-route P50/P95(s)"
SELECTOR_E2E_P50_COL = "E2E P50(s)"
SELECTOR_E2E_P95_COL = "E2E P95(s)"
SELECTOR_E2E_P50P95_COL = "E2E P50/95(s)"
SELECTOR_ROUTE_THROUGHPUT_COL = "Route TP(req/min)"
SELECTOR_EFFICIENCY_METRIC_COLUMNS = [
    SELECTOR_ROUTE_P50_COL,
    SELECTOR_ROUTE_P95_COL,
    SELECTOR_CORE_ROUTE_P50_COL,
    SELECTOR_CORE_ROUTE_P95_COL,
    SELECTOR_E2E_P50_COL,
    SELECTOR_E2E_P95_COL,
    SELECTOR_ROUTE_THROUGHPUT_COL,
]
SELECTOR_SUMMARY_METRICS = [
    "Rank",
    "MASE",
    "Regret-M",
    "Regret-M P90",
    "Rank-M",
    "MASE-hit1",
    "MASE-hit3",
    "CRPS",
    "Rank-C",
    "CRPS-hit1",
    "CRPS-hit3",
    "sMAPE",
    "Count-1",
    "Count-2",
    *SELECTOR_EFFICIENCY_METRIC_COLUMNS,
    SELECTOR_VALID_EXPECTED_DATASETS_COL,
] + TSROUTER_CORE_METRIC_COLUMNS


def count_valid_expected_datasets(df: pd.DataFrame, quick_test: bool = False) -> int:
    if df is None or "dataset" not in df.columns:
        return 0
    expected = set(ALL_Fast_DATASETS if quick_test else ALL_DATASETS)
    done = set(df["dataset"].dropna().astype(str).unique())
    return int(len(done & expected))


def print_results_file_one_line(file_path: Path, df: pd.DataFrame, quick_test: bool = False) -> None:
    if df is None:
        return
    df = harmonize_metrics_schema(df.copy())
    expected = set(ALL_Fast_DATASETS if quick_test else ALL_DATASETS)
    done = set(df["dataset"].dropna().astype(str).unique()) if "dataset" in df.columns else set()
    missing = sorted(expected - done)
    if missing:
        status = f"TSRouter runtime message: {len(missing)}TSRouter runtime message: "
    else:
        status = f"TSRouter runtime message: {len(expected)}TSRouter runtime message: "

    metric_parts = []
    for metric in ["sMAPE", "MASE", "CRPS"]:
        if metric in df.columns:
            val = pd.to_numeric(df[metric], errors="coerce").mean()
            if pd.notna(val):
                metric_parts.append(f"{metric}: {float(val):.4f}")
    metric_text = " ".join(metric_parts)
    print(f"TSRouter runtime message: {file_path} {status} df shape: {df.shape}TSRouter runtime message: {metric_text}")


def _format_metric_pair_for_display(first, third, decimals: int = 3) -> str:
    first_val = pd.to_numeric(pd.Series([first]), errors="coerce").iloc[0]
    third_val = pd.to_numeric(pd.Series([third]), errors="coerce").iloc[0]
    if pd.isna(first_val) and pd.isna(third_val):
        return ""
    left = "" if pd.isna(first_val) else f"{float(first_val):.{int(decimals)}f}"
    right = "" if pd.isna(third_val) else f"{float(third_val):.{int(decimals)}f}"
    return f"{left}/{right}"


def build_selector_display_summary(df_summary: pd.DataFrame) -> pd.DataFrame:
    out = df_summary.copy()
    pair_specs = [
        ("MASE-hit1/3", "MASE-hit1", "MASE-hit3", 3),
        ("CRPS-hit1/3", "CRPS-hit1", "CRPS-hit3", 3),
        ("PWW1/3↑", "ENC_TOP1_SUBSET_RATE", "ENC_TOP3_SUBSET_RATE", 3),
        ("TWW1/3↑", "TEST_WINDOW_TOP1_ACC", "TEST_WINDOW_TOP3_HIT", 3),
        ("TCC1/3↑", "SINGLE_TOP1_ACC", "SINGLE_TOP3_HIT", 3),
        ("TWC1/3↑", "TEST_WINDOW_CHANNEL_TOP1_ACC", "TEST_WINDOW_CHANNEL_TOP3_HIT", 3),
        ("TWR1/3↑", "TEST_WINDOW_TASK_TOP1_ACC", "TEST_WINDOW_TASK_TOP3_HIT", 3),
        ("TCR1/3↑", "TEST_CHANNEL_TASK_TOP1_ACC", "TEST_CHANNEL_TASK_TOP3_HIT", 3),
        (SELECTOR_ROUTE_P50P95_COL, SELECTOR_ROUTE_P50_COL, SELECTOR_ROUTE_P95_COL, 2),
        (SELECTOR_CORE_ROUTE_P50P95_COL, SELECTOR_CORE_ROUTE_P50_COL, SELECTOR_CORE_ROUTE_P95_COL, 2),
        (SELECTOR_E2E_P50P95_COL, SELECTOR_E2E_P50_COL, SELECTOR_E2E_P95_COL, 2),
    ]
    for display_row, first_row, third_row, decimals in pair_specs:
        if first_row in out.index and third_row in out.index:
            out.loc[display_row] = [
                _format_metric_pair_for_display(out.loc[first_row, col], out.loc[third_row, col], decimals=decimals)
                for col in out.columns
            ]

    drop_rows = {
        "Rank",
        "MASE-hit1",
        "MASE-hit3",
        "CRPS-hit1",
        "CRPS-hit3",
        "ENC_TOP1_ENRICH",
        "ENC_TOP3_ENRICH",
        "ENC_TOP1_SUBSET_RATE",
        "ENC_TOP3_SUBSET_RATE",
        "TEST_WINDOW_TOP1_ACC",
        "TEST_WINDOW_TOP3_HIT",
        "SINGLE_TOP1_ACC",
        "SINGLE_TOP3_HIT",
        "TEST_WINDOW_CHANNEL_TOP1_ACC",
        "TEST_WINDOW_CHANNEL_TOP3_HIT",
        "TEST_WINDOW_TASK_TOP1_ACC",
        "TEST_WINDOW_TASK_TOP3_HIT",
        "TEST_CHANNEL_TASK_TOP1_ACC",
        "TEST_CHANNEL_TASK_TOP3_HIT",
        SELECTOR_ROUTE_P50_COL,
        SELECTOR_ROUTE_P95_COL,
        SELECTOR_CORE_ROUTE_P50_COL,
        SELECTOR_CORE_ROUTE_P95_COL,
        SELECTOR_E2E_P50_COL,
        SELECTOR_E2E_P95_COL,
    }
    out = out.drop(index=[row for row in drop_rows if row in out.index], errors="ignore")
    preferred = [
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
        "REGION_WEIGHTED_PURITY",
        "REGION_DIAG_RANK",
        "REGION_DELTA_RANK",
        "TWW1/3↑",
        "TCC1/3↑",
        "TWC1/3↑",
        "TWR1/3↑",
        "TCR1/3↑",
        SELECTOR_ROUTE_P50P95_COL,
        SELECTOR_CORE_ROUTE_P50P95_COL,
        SELECTOR_E2E_P50P95_COL,
        SELECTOR_ROUTE_THROUGHPUT_COL,
        SELECTOR_VALID_EXPECTED_DATASETS_COL,
    ]
    row_order = [row for row in preferred if row in out.index] + [row for row in out.index if row not in preferred]
    return out.reindex(row_order)

def _extract_zcol_from_name(model_col_name: str) -> str | None:
    'TSRouter runtime message.'
    m = re.search(r"_z(\d+-\d+)$", str(model_col_name))
    if not m:
        return None
    return f"z{m.group(1)}"

def _strip_zsuffix(model_col_name: str) -> str:
    'TSRouter runtime message.'
    s = str(model_col_name)
    return re.sub(r"_z\d+-\d+$", "", s)

def _z_sort_key(z: str):
    'TSRouter runtime message.'
    m = re.match(r"z(\d+)-(\d+)$", z)
    if not m:
        return (10**9, 10**9)
    return (int(m.group(1)), int(m.group(2)))


def process_results(
    file_path,
    model_name,
    common_datasets,
    verbose=False,
    quick_test=False,
    process_metric_overrides: dict | None = None,
):
    'TSRouter runtime message.'
    if not file_path.exists():
        print(f"TSRouter runtime message: {file_path}")
        return None

    df = check_results_file(file_path, verbose, quick_test)
    if df is None:
        return None
    df = harmonize_metrics_schema(df)
    missing_core = [
        col for col in TSROUTER_CORE_METRIC_COLUMNS
        if col not in df.columns or not _has_valid_numeric(df[col])
    ]
    if missing_core and str(file_path.name).endswith("_GE_fast.csv"):
        fallback_path = file_path.with_name(file_path.name.replace("_GE_fast.csv", ".csv"))
        if fallback_path.exists():
            fb = check_results_file(fallback_path, False, quick_test)
            if fb is not None and not fb.empty:
                fb = harmonize_metrics_schema(fb)
                cols = ["dataset"] + [col for col in TSROUTER_CORE_METRIC_COLUMNS if col in fb.columns]
                if len(cols) > 1:
                    fill = fb[cols].drop_duplicates(subset=["dataset"], keep="last")
                    df = df.drop(columns=[col for col in cols[1:] if col in df.columns], errors="ignore")
                    df = df.merge(fill, on="dataset", how="left")
                    print(f"[process-metrics] filled {model_name} core metrics from {fallback_path}")
    # Summary-time strict/effective selection must win over values persisted by
    # either the GE-fast result or its compatibility fallback.
    if process_metric_overrides is not None:
        for column in COMPETENCE_REGION_METRIC_COLUMNS:
            value = pd.to_numeric(
                pd.Series([process_metric_overrides.get(column, np.nan)]),
                errors="coerce",
            ).iloc[0]
            df[column] = (
                float(value)
                if pd.notna(value) and np.isfinite(float(value))
                else np.nan
            )
    df["model"] = model_name
    df[['ds_key', 'ds_freq', 'term']] = df['dataset'].str.extract(r'^(.*?)/([^/]+)/([^/]+)$')
    if 'model_order' in df.columns:
        df['model_order'] = df['model_order'].apply(
            lambda x: x.tolist() if hasattr(x, 'tolist') else
            [int(i) for i in x.strip('[]').split()] if isinstance(x, str) else x
        )
    df_return=df[df['dataset'].isin(common_datasets)].copy()
    return df_return

def harmonize_metrics_schema(df: pd.DataFrame) -> pd.DataFrame:
    'TSRouter runtime message.'
    df = df.copy()
    if "MASE" not in df.columns and "eval_metrics/MASE[0.5]" in df.columns:
        df["MASE"] = pd.to_numeric(df["eval_metrics/MASE[0.5]"], errors="coerce")
    if "sMAPE" not in df.columns and "eval_metrics/sMAPE[0.5]" in df.columns:
        df["sMAPE"] = pd.to_numeric(df["eval_metrics/sMAPE[0.5]"], errors="coerce")
    if "CRPS" not in df.columns and "eval_metrics/mean_weighted_sum_quantile_loss" in df.columns:
        df["CRPS"] = pd.to_numeric(df["eval_metrics/mean_weighted_sum_quantile_loss"], errors="coerce")
    for col in TSROUTER_EXTRA_METRIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_by_season_naive(df: pd.DataFrame, season_naive_df: pd.DataFrame) -> pd.DataFrame:
    'TSRouter runtime message.'
    if season_naive_df is None or season_naive_df.empty:
        return df

    ref = harmonize_metrics_schema(season_naive_df)
    ref = ref[["dataset", "MASE", "CRPS"]].dropna(subset=["dataset"]).drop_duplicates(subset=["dataset"], keep="first")
    ref = ref.rename(columns={"MASE": "_sn_MASE", "CRPS": "_sn_CRPS"})

    out = df.merge(ref, on="dataset", how="left")
    eps = 1e-12
    if "MASE" in out.columns:
        out["MASE"] = out["MASE"] / (out["_sn_MASE"] + eps)
    if "CRPS" in out.columns:
        out["CRPS"] = out["CRPS"] / (out["_sn_CRPS"] + eps)

    if "eval_metrics/MASE[0.5]" in out.columns:
        out["eval_metrics/MASE[0.5]"] = pd.to_numeric(out["eval_metrics/MASE[0.5]"], errors="coerce") / (out["_sn_MASE"] + eps)
    if "eval_metrics/mean_weighted_sum_quantile_loss" in out.columns:
        out["eval_metrics/mean_weighted_sum_quantile_loss"] = (
            pd.to_numeric(out["eval_metrics/mean_weighted_sum_quantile_loss"], errors="coerce") / (out["_sn_CRPS"] + eps)
        )

    out = out.drop(columns=["_sn_MASE", "_sn_CRPS"], errors="ignore")
    return out


def compute_tsfm_best_metrics_for_summary(
    baseline_df: pd.DataFrame,
    ordered_model_names: list[str],
    rank_base: str = "MASE",
) -> dict[str, float]:
    'TSRouter runtime message.'
    if baseline_df is None or baseline_df.empty:
        return {}

    base = baseline_df[baseline_df["model"].isin(ordered_model_names)].copy()
    if base.empty:
        return {}

    best = {}
    for metric in ["sMAPE", "MASE", "CRPS"]:
        if metric not in base.columns:
            continue
        vals = (
            base.groupby("model")[metric]
            .mean(numeric_only=True)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if not vals.empty:
            best[metric] = float(vals.min())

    for metric, rank_key in [("MASE", "Rank-M"), ("CRPS", "Rank-C")]:
        if metric not in base.columns:
            continue
        rank_df = base[["dataset", "model", metric]].dropna().copy()
        if rank_df.empty:
            continue
        rank_df[metric] = pd.to_numeric(rank_df[metric], errors="coerce")
        rank_df = rank_df.dropna(subset=[metric])
        if rank_df.empty:
            continue
        rank_df["_rank"] = rank_df.groupby("dataset")[metric].rank(method="min", ascending=True)
        rank_vals = rank_df.groupby("model")["_rank"].mean().replace([np.inf, -np.inf], np.nan).dropna()
        if not rank_vals.empty:
            best[rank_key] = float(rank_vals.min())
        hit1_vals = rank_df.assign(_hit=(rank_df["_rank"] <= 1.0)).groupby("model")["_hit"].mean()
        hit3_vals = rank_df.assign(_hit=(rank_df["_rank"] <= 3.0)).groupby("model")["_hit"].mean()
        hit1_vals = hit1_vals.replace([np.inf, -np.inf], np.nan).dropna()
        hit3_vals = hit3_vals.replace([np.inf, -np.inf], np.nan).dropna()
        if not hit1_vals.empty:
            best[f"{metric}-hit1"] = float(hit1_vals.max())
        if not hit3_vals.empty:
            best[f"{metric}-hit3"] = float(hit3_vals.max())

    if rank_base in base.columns:
        rank_df = base[["dataset", "model", rank_base]].dropna().copy()
        if not rank_df.empty:
            rank_df["Rank"] = rank_df.groupby("dataset")[rank_base].rank(method="min", ascending=True)
            rank_vals = rank_df.groupby("model")["Rank"].mean().replace([np.inf, -np.inf], np.nan).dropna()
            if not rank_vals.empty:
                best["Rank"] = float(rank_vals.min())

    return best


def caculate_combined_rank(
    combined_df: pd.DataFrame,
    zoo_model_name: str,
    verbose: bool = False,
    first_col_prefix: str | None = None,
    rank_base: str = "MASE",
    include_selector_in_rank: bool = False,
):
    'TSRouter runtime message.'
    df = combined_df.copy()

    if rank_base not in df.columns:
        raise ValueError(f"rank_base='{rank_base}TSRouter runtime message: ")

               
    ranked_df = df.copy()
    rank_col = "RANK"

                             
    if zoo_model_name is None or include_selector_in_rank:
        ranked_df[rank_col] = ranked_df.groupby("dataset")[rank_base].rank(
            method="min", ascending=True
        )
    else:
                                                            
        special_mask = ranked_df["model"] == zoo_model_name
        special_rows = ranked_df[special_mask].copy()
        other_rows = ranked_df[~special_mask].copy()

                                      
        other_rows[rank_col] = other_rows.groupby("dataset")[rank_base].rank(
            method="min", ascending=True
        )

        EPS_REL = 1e-3                      
        final_dfs = []
        for dataset, group in other_rows.groupby("dataset"):
            dataset_special = special_rows[special_rows["dataset"] == dataset].copy()

            if not dataset_special.empty:
                special_val = dataset_special[rank_base].values[0]
                                                                              
                better_mask = group[rank_base] < special_val * (1.0 - EPS_REL)
                rank_pos = int(better_mask.sum()) + 1
                dataset_special[rank_col] = rank_pos

                                                                                      
                if (
                        verbose
                        and str(zoo_model_name).startswith("Real-")
                        and not str(zoo_model_name).startswith("Real-Channel")
                        and rank_pos != 1
                        and args.ensemble_size==1
                ):
                    debug_df = pd.concat([group, dataset_special], ignore_index=True)
                    keep_cols = ["dataset", "model", rank_base, rank_col]
                    keep_cols_exist = [c for c in keep_cols if c in debug_df.columns]
                    debug_df = debug_df[keep_cols_exist].copy()
                    debug_df = debug_df.sort_values(by=rank_base, ascending=True)

                    print(f"\n⚠️ [DEBUG-Real-RankStep] dataset = {dataset}")
                    print(
                        tabulate(
                            debug_df,
                            headers="keys",
                            tablefmt="plain",
                            floatfmt=".6f",
                            numalign="decimal",
                            stralign="left",
                        )
                    )

                if (
                        verbose
                        and False
                        and str(zoo_model_name).startswith("Real-Channel")
                        and rank_pos != 1
                ):
                    better_rows = group.loc[better_mask].copy()
                    best_row = group.sort_values(by=rank_base, ascending=True).iloc[0]
                    print(
                        f"\n⚠️ [DEBUG-RealChannel-RankStep] dataset={dataset}, "
                        f"rank_base={rank_base}, rank_pos={rank_pos}, "
                        f"real_channel_{rank_base}={float(special_val):.6f}, "
                        f"best_tsfm={best_row['model']}, "
                        f"best_tsfm_{rank_base}={float(best_row[rank_base]):.6f}, "
                        f"better_tsfm_count={len(better_rows)}"
                    )
                    debug_df = pd.concat([better_rows, dataset_special], ignore_index=True)
                    keep_cols = ["dataset", "model", rank_base, rank_col]
                    keep_cols_exist = [c for c in keep_cols if c in debug_df.columns]
                    debug_df = debug_df[keep_cols_exist].sort_values(by=rank_base, ascending=True)
                    print(
                        tabulate(
                            debug_df,
                            headers="keys",
                            tablefmt="plain",
                            floatfmt=".6f",
                            numalign="decimal",
                            stralign="left",
                        )
                    )

                final_dfs.append(pd.concat([group, dataset_special]))
            else:
                final_dfs.append(group)

        if not final_dfs and not special_rows.empty:
                                           
            special_rows[rank_col] = 1
            ranked_df = special_rows
        else:
            ranked_df = pd.concat(final_dfs)

    df = ranked_df.sort_index()

                          
    metrics_to_show = ["sMAPE", "MASE", "CRPS", "RANK"] + TSROUTER_CORE_METRIC_COLUMNS
    metrics_exist = [m for m in metrics_to_show if m in df.columns]
    if metrics_exist:
        global_avg = df.groupby("model", sort=False)[metrics_exist].mean().T
        global_avg = global_avg.reindex(metrics_exist).round(4)

        if verbose:
            n_ds = df["dataset"].nunique()
            print(f"TSRouter runtime message: {n_ds}, "
                  f"rank_base={rank_base}TSRouter runtime message: {include_selector_in_rank}")

            data = global_avg if isinstance(global_avg, pd.DataFrame) else global_avg.to_frame().T
            cols = list(data.columns)
            if first_col_prefix:
                first_cols = [c for c in cols if str(c).startswith(first_col_prefix)]
                other_cols = [c for c in cols if c not in first_cols]
                cols = first_cols + other_cols
                data = data[cols]

            data_print = data.copy()
            data_print.columns = [Model_abbrev_map.get(str(c), str(c)) for c in data_print.columns]

            data_print = data_print.reset_index().rename(columns={"index": "Metrics"})
            print(
                tabulate(
                    data_print,
                    headers="keys",
                    tablefmt="plain",
                    floatfmt=".3f",
                    numalign="decimal",
                    stralign="left",
                )
            )

                                         
    filtered = df[df["model"] == zoo_model_name]

    table = pd.DataFrame(index=[], columns=[zoo_model_name])

                              
    table.loc["Rank", zoo_model_name] = filtered["RANK"].mean()

                                            
    for m in SELECTOR_SUMMARY_METRICS[1:]:
        if m in filtered.columns:
            table.loc[m, zoo_model_name] = filtered[m].mean()

    table = table.round(2)

    rank_summary = {"RANK": table}
    return rank_summary



def add_order_metrics(
    baseline_subset: pd.DataFrame,
    subset_df: pd.DataFrame,
    model_name: str,
    rank_summary_all: dict,
    add_index: int = 0,
    verbose: bool = True,
    df_real: pd.DataFrame | None = None,
    k_order=None,
    rank_base: str = "MASE",
    include_selector_in_rank: bool = False,
):

    combined_df = pd.concat([baseline_subset, subset_df], ignore_index=True)
    rank_summary = caculate_combined_rank(combined_df, zoo_model_name=model_name, verbose=verbose,rank_base=rank_base,include_selector_in_rank=include_selector_in_rank,)

    for rank_type in rank_summary_all:
        if rank_type in rank_summary and model_name not in rank_summary_all[rank_type].columns:
            incoming = rank_summary[rank_type][model_name]
            wanted_index = list(dict.fromkeys(SELECTOR_SUMMARY_METRICS + list(rank_summary_all[rank_type].index) + list(incoming.index)))
            rank_summary_all[rank_type] = rank_summary_all[rank_type].reindex(wanted_index)
            incoming = incoming.reindex(wanted_index)
            rank_summary_all[rank_type].insert(add_index, model_name, incoming)

                                        
            skip_order_for_this_model = str(model_name).startswith("All_")

                           
            if (
                not skip_order_for_this_model
                and df_real is not None
                and "model_order" in subset_df.columns
                and "model_order" in df_real.columns
            ):
                metrics = calculate_order_metrics(df_real, subset_df, k_order)
                for metric_name, value in metrics.items():
                    if metric_name not in rank_summary_all[rank_type].index:
                        rank_summary_all[rank_type].loc[metric_name] = np.nan
                    rank_summary_all[rank_type].loc[metric_name, model_name] = value
        elif model_name in rank_summary_all[rank_type].columns:
            print(f"TSRouter runtime message: {model_name}TSRouter runtime message: ")

    return rank_summary_all

def parse_seed_list(seed_str: str):
    'TSRouter runtime message.'
    seeds = []
    for part in seed_str.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            seeds.append(int(part))
        except ValueError:
            print(f"TSRouter runtime message: {part}TSRouter runtime message: ")
    return seeds

                              
def make_selector_path_builder(
    results_dir: Path,
    current_zoo_num: int,
    zoo_total_num: int,
    ensemble_size: int,
    default_ensemble_agg: str,          # ✅ rename
    default_real_metric: str,
    ge_released: bool = False,
    ge_fast_eval: bool = False,
    sample_repr_num: int = 20,
):
    def build(
        selector_name: str,
        seed: int | None = None,
        real_order_metric: str | None = None,
        ensemble_agg: str | None = None,           
            ensemble_size_override: int | None = None,                            
    ) -> Path:
        cfg = Selector_zoo_details[selector_name]
        tpl = cfg["csv_name_tpl"]

        fname = tpl.format(
            current_zoo_num=current_zoo_num,
            zoo_total_num=zoo_total_num,
            ensemble_size=ensemble_size_override if ensemble_size_override is not None else ensemble_size,
            ensemble_agg=ensemble_agg if ensemble_agg is not None else default_ensemble_agg,
            real_order_metric=real_order_metric or default_real_metric,
            search_seed=seed if seed is not None else 0,
            sample_repr_num=sample_repr_num,
        )
        if ge_released and fname.endswith(".csv"):
            if ge_fast_eval:
                fname = fname[:-4] + "_GE_fast.csv"
            else:
                fname = fname[:-4] + "_GE.csv"
        return results_dir / selector_name / fname

    return build

def resolve_model_result_csv(args,results_dir: Path, model_name: str, context_len: int) -> Path:
    'TSRouter runtime message.'
    base_dir = f"{args.TSFM_results_dir}"
    candidates = []
    if args.GE_released:
        candidates.extend([
            resolve_tsfm_result_path(results_dir, model_name, base_dir, "GE_all_results.csv"),
            resolve_tsfm_result_path(results_dir, model_name, base_dir, "all_results.csv"),
        ])
    else:
        candidates.extend([
            resolve_tsfm_result_path(results_dir, model_name, base_dir, "all_results.csv"),
            resolve_tsfm_result_path(results_dir, model_name, base_dir, "GE_all_results.csv"),
        ])
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def resolve_tsrouter_selector_result_path(args, save_name: str) -> Path:
    stage_path = Path(get_tsrouter_selector_stage_result_dir(args)) / save_name
    root_path = Path(get_tsrouter_selector_result_dir(args)) / save_name
    if stage_path.exists() or not root_path.exists():
        return stage_path
    return root_path


# ---------------------------------------------------------------------------
# Legacy --vldb-table implementation.
#
# The old VLDB table CLI is intentionally disabled in build_check_selector_parser()
# and in the main control flow below. These helpers are kept temporarily as
# unreachable history until the new check_selector summary path fully replaces
# the paper-table implementation.
# ---------------------------------------------------------------------------
def _vldb_expected_datasets(args) -> set[str]:
    return set(ALL_Fast_DATASETS if args.quick_test else ALL_DATASETS)


def _model_id_maps() -> tuple[dict[str, int], dict[int, str]]:
    abbr_to_id: dict[str, int] = {}
    id_to_abbr: dict[int, str] = {}
    for idx, full_name in enumerate(All_sorted_model_names):
        abbr = str(Model_abbrev_map.get(full_name, full_name))
        abbr_to_id[abbr] = idx
        id_to_abbr[idx] = abbr
    for family, sizes in Model_zoo_details.items():
        for size, details in sizes.items():
            full_name = f"{family}_{size}"
            abbr = details.get("abbreviation", Model_abbrev_map.get(full_name, full_name))
            if "id" in details:
                mid = int(details["id"])
                abbr_to_id[str(abbr)] = mid
                id_to_abbr[mid] = str(abbr)
    return abbr_to_id, id_to_abbr


def _parse_first_model_id(model_order) -> float:
    vals = _parse_model_order_ids(model_order, 1)
    return float(vals[0]) if vals else np.nan


def _parse_model_order_ids(model_order, k: int | None = None) -> list[float]:
    abbr_to_id, _ = _model_id_maps()

    def parse_one(value) -> float | None:
        try:
            val = float(value)
            return val if np.isfinite(val) else None
        except Exception:
            text = str(value).strip().strip("'\"")
            if text in abbr_to_id:
                return float(abbr_to_id[text])
            return None

    if model_order is None or (isinstance(model_order, float) and np.isnan(model_order)):
        return []
    vals = []
    if isinstance(model_order, (list, tuple, np.ndarray)):
        raw_vals = list(model_order)
    else:
        text = str(model_order).strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            raw_vals = list(parsed) if isinstance(parsed, (list, tuple, np.ndarray)) else []
        except Exception:
            body = text.strip("[]")
            raw_vals = re.findall(r"-?\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_.-]*", body)
    for value in raw_vals:
        parsed = parse_one(value)
        if parsed is None:
            continue
        vals.append(float(parsed))
        if k is not None and len(vals) >= int(k):
            break
    return vals


def _row_selected_model_ids(row: pd.Series) -> list[float]:
    ids = _parse_model_order_ids(row.get("model_order", None), None)
    if ids:
        return ids
    selected = pd.to_numeric(pd.Series([row.get("selected_model_id", np.nan)]), errors="coerce").iloc[0]
    if pd.notna(selected) and np.isfinite(float(selected)):
        return [float(selected)]
    abbr_to_id, _ = _model_id_maps()
    mid = abbr_to_id.get(str(row.get("model", "")), np.nan)
    if pd.notna(mid) and np.isfinite(float(mid)):
        return [float(mid)]
    return []


def compute_selector_recommendation_summary(
    baseline_df: pd.DataFrame,
    selector_df: pd.DataFrame,
) -> dict[str, object]:
    if baseline_df is None or baseline_df.empty or selector_df is None or selector_df.empty:
        return {}

    abbr_to_id, id_to_abbr = _model_id_maps()
    selected_by_dataset: dict[str, int] = {}
    selected_ids = []
    for _, rec in selector_df.iterrows():
        dataset = str(rec.get("dataset", ""))
        ids = _row_selected_model_ids(rec)
        if not dataset or not ids:
            continue
        mid = int(float(ids[0]))
        selected_by_dataset[dataset] = mid
        selected_ids.append(mid)

    out: dict[str, object] = {}
    if selected_ids:
        counts = pd.Series(selected_ids).value_counts().sort_values(ascending=False)
        total = float(counts.sum())

        def fmt_count(mid: int, count: int) -> str:
            abbr = id_to_abbr.get(int(mid), str(int(mid)))
            return f"{abbr} {float(count) * 100.0 / max(total, 1.0):.1f}%"

        tops = list(counts.items())[:2]
        out["Count-1"] = fmt_count(int(tops[0][0]), int(tops[0][1])) if len(tops) >= 1 else ""
        out["Count-2"] = fmt_count(int(tops[1][0]), int(tops[1][1])) if len(tops) >= 2 else ""

    base = baseline_df.copy()
    if "model_id" not in base.columns:
        base["model_id"] = base["model"].map(lambda x: abbr_to_id.get(str(x), np.nan))
    base = base.dropna(subset=["dataset", "model_id"]).copy()
    if base.empty:
        return out
    base["model_id"] = base["model_id"].astype(int)

    for metric in ["MASE", "CRPS"]:
        if metric not in base.columns:
            continue
        metric_df = base[["dataset", "model_id", metric]].copy()
        metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
        metric_df = metric_df.dropna(subset=[metric])
        if metric_df.empty:
            continue
        metric_df["_rank"] = metric_df.groupby("dataset")[metric].rank(method="min", ascending=True)

        rank_lookup: dict[str, dict[int, float]] = {}
        for ds, group in metric_df.groupby("dataset"):
            rank_lookup[str(ds)] = {
                int(rec["model_id"]): float(rec["_rank"])
                for _, rec in group.iterrows()
            }

        hit1 = []
        hit3 = []
        for dataset, mid in selected_by_dataset.items():
            ds_ranks = rank_lookup.get(str(dataset))
            if not ds_ranks or int(mid) not in ds_ranks:
                continue
            rank_val = float(ds_ranks[int(mid)])
            hit1.append(rank_val <= 1.0)
            hit3.append(rank_val <= 3.0)

        if hit1:
            out[f"{metric}-hit1"] = float(np.mean(hit1))
            out[f"{metric}-hit3"] = float(np.mean(hit3))

    return out


def compute_selector_metric_rank_summary(
    baseline_df: pd.DataFrame,
    selector_df: pd.DataFrame,
) -> dict[str, float]:
    if baseline_df is None or baseline_df.empty or selector_df is None or selector_df.empty:
        return {}
    out: dict[str, float] = {}
    eps_rel = 1e-3
    selector_by_dataset = selector_df.drop_duplicates(subset=["dataset"], keep="last").set_index("dataset")
    for metric, rank_col in [("MASE", "Rank-M"), ("CRPS", "Rank-C")]:
        if metric not in baseline_df.columns or metric not in selector_df.columns:
            continue
        ranks = []
        for dataset, group in baseline_df.groupby("dataset"):
            if dataset not in selector_by_dataset.index:
                continue
            selector_val = pd.to_numeric(pd.Series([selector_by_dataset.at[dataset, metric]]), errors="coerce").iloc[0]
            if pd.isna(selector_val) or not np.isfinite(float(selector_val)):
                continue
            vals = pd.to_numeric(group[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if vals.empty:
                continue
            rank_pos = int((vals < float(selector_val) * (1.0 - eps_rel)).sum()) + 1
            ranks.append(rank_pos)
        if ranks:
            out[rank_col] = float(np.mean(ranks))
    if "MASE" in baseline_df.columns and "MASE" in selector_df.columns:
        best_mase = (
            baseline_df[["dataset", "MASE"]]
            .assign(MASE=lambda df: pd.to_numeric(df["MASE"], errors="coerce"))
            .dropna(subset=["dataset", "MASE"])
            .groupby("dataset", sort=False)["MASE"]
            .min()
            .rename("_best_tsfm_mase")
            .reset_index()
        )
        selected_mase = selector_df[["dataset", "MASE"]].copy()
        selected_mase["MASE"] = pd.to_numeric(selected_mase["MASE"], errors="coerce")
        regret_df = selected_mase.merge(best_mase, on="dataset", how="left")
        regret = (
            pd.to_numeric(regret_df["MASE"], errors="coerce")
            - pd.to_numeric(regret_df["_best_tsfm_mase"], errors="coerce")
        )
        regret = regret.replace([np.inf, -np.inf], np.nan).dropna()
        if not regret.empty:
            out["Regret-M"] = float(regret.mean())
            out["Regret-M P90"] = float(regret.quantile(0.90))
    return out


def _has_valid_numeric(series: pd.Series) -> bool:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return bool(vals.notna().any())


def _vldb_best_tsfm_by_dataset(baseline_df: pd.DataFrame, rank_base: str) -> pd.DataFrame:
    if baseline_df is None or baseline_df.empty:
        return pd.DataFrame(columns=["dataset"])
    abbr_to_id, _ = _model_id_maps()
    rows = []
    for ds, group in baseline_df.groupby("dataset"):
        row = {"dataset": ds}
        for metric in ["MASE", "sMAPE", "CRPS"]:
            if metric not in group.columns:
                continue
            g = group[["model", metric]].copy()
            g[metric] = pd.to_numeric(g[metric], errors="coerce")
            g = g.dropna(subset=[metric])
            if g.empty:
                continue
            best = g.sort_values(metric, ascending=True).iloc[0]
            row[f"Best_TSFM_{metric}"] = float(best[metric])
            row[f"Best_TSFM_{metric}_model"] = str(best["model"])
            row[f"Best_TSFM_{metric}_model_id"] = abbr_to_id.get(str(best["model"]), np.nan)
            top3 = g.sort_values(metric, ascending=True).head(3)
            row[f"Best_TSFM_{metric}_top3_model_ids"] = [
                abbr_to_id.get(str(model), np.nan) for model in top3["model"].tolist()
            ]
        rows.append(row)
    return pd.DataFrame(rows)


def _vldb_row_selected_model_ids(row: pd.Series) -> list[float]:
    return _row_selected_model_ids(row)


def _vldb_top3_ids(value) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        raw = list(value)
    else:
        raw = _parse_model_order_ids(value, None)
    out = []
    for item in raw:
        parsed = pd.to_numeric(pd.Series([item]), errors="coerce").iloc[0]
        if pd.notna(parsed) and np.isfinite(float(parsed)):
            out.append(float(parsed))
    return out[:3]


def _vldb_add_quality_refs(df: pd.DataFrame, best_df: pd.DataFrame, rank_base: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = harmonize_metrics_schema(df.copy())
    if best_df is None or best_df.empty:
        return out
    out = out.merge(best_df, on="dataset", how="left")
    for metric in ["MASE", "sMAPE", "CRPS"]:
        if metric in out.columns and f"Best_TSFM_{metric}" in out.columns:
            out[f"win_{metric}"] = (
                pd.to_numeric(out[metric], errors="coerce")
                <= pd.to_numeric(out[f"Best_TSFM_{metric}"], errors="coerce") * (1.0 + 1e-3)
            )
            out[f"regret_{metric}"] = (
                pd.to_numeric(out[metric], errors="coerce")
                - pd.to_numeric(out[f"Best_TSFM_{metric}"], errors="coerce")
            )
    hit_col = f"Top1Hit_{rank_base}"
    top3_col = f"Top3Hit_{rank_base}"
    if f"Best_TSFM_{rank_base}_model_id" in out.columns:
        best_ids = pd.to_numeric(out[f"Best_TSFM_{rank_base}_model_id"], errors="coerce")
        selected_lists = [_vldb_row_selected_model_ids(row) for _, row in out.iterrows()]
        out[hit_col] = [
            bool(ids and np.isfinite(best_id) and float(ids[0]) == float(best_id))
            for ids, best_id in zip(selected_lists, best_ids)
        ]
        if f"Best_TSFM_{rank_base}_top3_model_ids" in out.columns:
            top3_lists = out[f"Best_TSFM_{rank_base}_top3_model_ids"].map(_vldb_top3_ids)
            out[top3_col] = [
                bool(
                    ids
                    and (
                        (len(ids) >= 3 and np.isfinite(best_id) and float(best_id) in ids[:3])
                        or (len(ids) < 3 and any(float(ids[0]) == float(top_id) for top_id in top3_ids))
                    )
                )
                for ids, best_id, top3_ids in zip(selected_lists, best_ids, top3_lists)
            ]
        else:
            out[top3_col] = [
                bool(np.isfinite(best_id) and float(best_id) in ids[:3])
                for ids, best_id in zip(selected_lists, best_ids)
            ]
    elif hit_col in out.columns and _has_valid_numeric(out[hit_col]):
        out[hit_col] = pd.to_numeric(out[hit_col], errors="coerce")
    elif top3_col in out.columns and _has_valid_numeric(out[top3_col]):
        out[top3_col] = pd.to_numeric(out[top3_col], errors="coerce")
    if top3_col not in out.columns and "model_order" in out.columns and f"Best_TSFM_{rank_base}_model_id" in out.columns:
        best_ids = pd.to_numeric(out[f"Best_TSFM_{rank_base}_model_id"], errors="coerce")
        out[top3_col] = [
            bool(np.isfinite(best_id) and float(best_id) in _parse_model_order_ids(order, 3))
            for order, best_id in zip(out["model_order"], best_ids)
        ]
    return out


def _vldb_mean(df: pd.DataFrame, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return np.nan
    vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.mean()) if not vals.empty else np.nan


def _vldb_bool_mean(df: pd.DataFrame, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return np.nan
    vals = df[col].dropna()
    if vals.empty:
        return np.nan
    if vals.dtype == bool:
        return float(vals.mean())
    return float(vals.astype(float).mean())


def _vldb_selected_model_count_columns(method: str, df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"Selected_Top1": "", "Selected_Top2": ""}
    _, id_to_abbr = _model_id_maps()
    counts: dict[int, int] = {}
    for _, row in df.iterrows():
        ids = _vldb_row_selected_model_ids(row)
        if not ids:
            continue
        selected_id = pd.to_numeric(pd.Series([row.get("selected_model_id", np.nan)]), errors="coerce").iloc[0]
        count_all_ids = (
            str(method) == "Random"
            and (pd.isna(selected_id) or not np.isfinite(float(selected_id)))
            and len(ids) > 1
        )
        for mid in (ids if count_all_ids else ids[:1]):
            if not np.isfinite(float(mid)):
                continue
            key = int(mid)
            counts[key] = counts.get(key, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:2]

    total = max(int(sum(counts.values())), 1)

    def fmt(item: tuple[int, int] | None) -> str:
        if item is None:
            return ""
        mid, count = item
        pct = 100.0 * float(count) / float(total)
        pct_text = f"{pct:.1f}%" if abs(pct - round(pct)) >= 0.05 else f"{int(round(pct))}%"
        return f"{id_to_abbr.get(int(mid), str(int(mid)))}:{pct_text}"

    return {
        "Selected_Top1": fmt(top[0]) if len(top) >= 1 else "",
        "Selected_Top2": fmt(top[1]) if len(top) >= 2 else "",
    }


def _vldb_quantile(series: pd.Series, q: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.quantile(q)) if not vals.empty else np.nan


def _vldb_p95(series: pd.Series) -> float:
    return _vldb_quantile(series, 0.95)


def _vldb_p90(series: pd.Series) -> float:
    return _vldb_quantile(series, 0.90)


def _vldb_format_ms_triplet(values: list[float]) -> str:
    rounded = []
    for value in values:
        if not np.isfinite(value):
            rounded.append("nan")
        else:
            rounded.append(str(int(round(float(value)))))
    if not rounded:
        return "nan"
    if len(set(rounded)) == 1:
        return rounded[0]
    return "/".join(rounded)


def _vldb_bool_col(df: pd.DataFrame, col: str, default: bool = True) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return df[col].astype(str).str.lower().eq("true")


def _vldb_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _vldb_latest_by_file_order(df: pd.DataFrame, subset: str | list[str]) -> pd.DataFrame:
    """Keep the last appended row for duplicate VLDB records."""
    if df is None or df.empty:
        return pd.DataFrame()
    keys = [subset] if isinstance(subset, str) else list(subset)
    keys = [key for key in keys if key in df.columns]
    if not keys:
        return df.copy()
    return df.drop_duplicates(subset=keys, keep="last").copy()


def _vldb_main_insert_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "insert_id" not in df.columns:
        return pd.DataFrame()
    insert_id = df["insert_id"].astype(str)
    main = insert_id.str.contains(r"stage\d+_to_\d+_main(?:_|_insert$)", regex=True)
    main = main & ~insert_id.str.contains("table1_main|schema_check", case=False, regex=True)
    return df[main].copy()


def _vldb_dataset_to_filename(dataset: str) -> str:
    parts = str(dataset).split("/")
    if len(parts) >= 3:
        return "_".join([parts[0], parts[1], parts[2]])
    return str(dataset).replace("/", "_")


def _vldb_dataset_runtime_candidates(dataset: str) -> list[str]:
    text = str(dataset)
    candidates = [text]
    if "/" not in text:
        for ds in list(ALL_DATASETS) + list(ALL_Fast_DATASETS):
            if _vldb_dataset_to_filename(str(ds)) == text:
                candidates.append(str(ds))
        parts = text.rsplit("_", 2)
        if len(parts) == 3:
            candidates.append(f"{parts[0]}/{parts[1]}/{parts[2]}")
    return list(dict.fromkeys(candidates))


def _vldb_model_key_from_token(token) -> str | None:
    if token is None or (isinstance(token, float) and np.isnan(token)):
        return None
    text = str(token).strip().strip("'\"")
    if not text:
        return None
    try:
        mid = int(float(text))
        if 0 <= mid < len(All_sorted_model_names):
            return str(All_sorted_model_names[mid])
    except Exception:
        pass
    if text in All_sorted_model_names:
        return text
    abbr_to_key = {str(v): str(k) for k, v in Model_abbrev_map.items()}
    return abbr_to_key.get(text)


def _vldb_first_selected_model_key(order_value) -> str | None:
    if order_value is None or (isinstance(order_value, float) and np.isnan(order_value)):
        return None
    if isinstance(order_value, (list, tuple, np.ndarray)):
        raw_vals = list(order_value)
    else:
        text = str(order_value).strip()
        raw_vals = []
        if text:
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, np.ndarray)):
                    raw_vals = list(parsed)
            except Exception:
                raw_vals = []
        if not raw_vals:
            raw_vals = re.findall(r"[A-Za-z0-9_.-]+", text)
    for token in raw_vals:
        model_key = _vldb_model_key_from_token(token)
        if model_key:
            return model_key
    return None


def _vldb_tsfm_forward_ms(model_key: str, dataset: str, tsfm_results_dir: str = "cl_512") -> float:
    val, _source = _vldb_tsfm_forward_ms_with_source(model_key, dataset, tsfm_results_dir=tsfm_results_dir)
    return val


def _vldb_tsfm_forward_ms_with_source(model_key: str, dataset: str, tsfm_results_dir: str = "cl_512") -> tuple[float, str]:
    dataset_candidates = set(_vldb_dataset_runtime_candidates(dataset))
    result_path = resolve_tsfm_csv_path(str(model_key), tsfm_results_dir, "all_results.csv")
    if result_path.exists():
        try:
            df = pd.read_csv(
                result_path,
                usecols=lambda c: c in {
                    "dataset",
                    "forward_runtime_seconds",
                },
                low_memory=False,
            )
            sub = df[df["dataset"].astype(str).isin(dataset_candidates)] if "dataset" in df.columns else pd.DataFrame()
            if not sub.empty:
                row = sub.iloc[-1]
                val = pd.to_numeric(pd.Series([row.get("forward_runtime_seconds")]), errors="coerce").iloc[0]
                if pd.notna(val) and np.isfinite(float(val)):
                    return float(val) * 1000.0, f"{result_path}:forward_runtime_seconds"
        except Exception:
            pass
    for candidate in _vldb_dataset_runtime_candidates(dataset):
        meta_path = resolve_tsfm_artifact_path(
            str(model_key),
            tsfm_results_dir,
            "meta",
            f"{_vldb_dataset_to_filename(candidate)}_meta.json",
        )
        if not meta_path.exists():
            continue
        try:
            perf = json.loads(meta_path.read_text(encoding="utf-8")).get("performance", {}) or {}
            val = pd.to_numeric(pd.Series([perf.get("forward_runtime_seconds")]), errors="coerce").iloc[0]
            if pd.notna(val) and np.isfinite(float(val)):
                return float(val) * 1000.0, f"{meta_path}:performance.forward_runtime_seconds"
        except Exception:
            continue
    return np.nan, "missing_runtime"


def _selector_round2(value) -> float:
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(val) or not np.isfinite(float(val)):
        return np.nan
    return round(float(val), 2)


def _selector_expected_datasets(quick_test: bool = False) -> set[str]:
    return set(ALL_Fast_DATASETS if quick_test else ALL_DATASETS)


def _selector_first_model_key(row: pd.Series) -> str | None:
    ids = _row_selected_model_ids(row)
    if ids:
        return _vldb_model_key_from_token(ids[0])
    if "selected_model_order" in row.index:
        return _vldb_first_selected_model_key(row.get("selected_model_order"))
    return None


def _selector_runtime_summary_from_result(
    df: pd.DataFrame,
    quick_test: bool = False,
    tsfm_results_dir: str = "cl_512",
) -> dict[str, object]:
    expected = _selector_expected_datasets(quick_test)
    expected_n = int(len(expected))
    out: dict[str, object] = {
        SELECTOR_ROUTE_P50_COL: np.nan,
        SELECTOR_ROUTE_P95_COL: np.nan,
        SELECTOR_CORE_ROUTE_P50_COL: np.nan,
        SELECTOR_CORE_ROUTE_P95_COL: np.nan,
        SELECTOR_E2E_P50_COL: np.nan,
        SELECTOR_E2E_P95_COL: np.nan,
        SELECTOR_ROUTE_THROUGHPUT_COL: np.nan,
        "_selector_efficiency_note": "",
        "_selector_route_valid_n": 0,
        "_selector_e2e_valid_n": 0,
    }
    notes: list[str] = []
    if df is None or df.empty or "dataset" not in df.columns:
        notes.extend([f"Route-0/{expected_n}", f"E2E-0/{expected_n}"])
        out["_selector_efficiency_note"] = " ".join(dict.fromkeys(notes))
        return out

    work = df.copy()
    work["dataset"] = work["dataset"].astype(str)
    work = work[work["dataset"].isin(expected)].copy()
    work = _vldb_latest_by_file_order(work, "dataset")

    route_valid_mask = pd.Series(False, index=work.index)
    route_seconds = pd.Series(dtype=float)
    core_route_seconds = pd.Series(dtype=float)
    if "route_final_seconds" not in work.columns:
        pass
    else:
        route_vals = pd.to_numeric(work["route_final_seconds"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        route_valid_mask = route_vals.notna() & route_vals.ge(0)
        route_seconds = route_vals[route_valid_mask].astype(float)
        route_n = int(route_seconds.shape[0])
        out["_selector_route_valid_n"] = route_n
        if route_n > 0:
            out[SELECTOR_ROUTE_P50_COL] = _selector_round2(route_seconds.quantile(0.50))
            out[SELECTOR_ROUTE_P95_COL] = _selector_round2(route_seconds.quantile(0.95))
            route_sum = float(route_seconds.sum())
            if route_sum > 0:
                out[SELECTOR_ROUTE_THROUGHPUT_COL] = _selector_round2(float(expected_n) * 60.0 / route_sum)
        if route_n != expected_n:
            notes.append(f"Route-{route_n}/{expected_n}")
    if {
        "sample_seconds",
        "task_probe_forward_seconds",
        "task_probe_rank_seconds",
        "route_final_seconds",
    }.issubset(work.columns):
        sample = pd.to_numeric(work["sample_seconds"], errors="coerce")
        forward = pd.to_numeric(work["task_probe_forward_seconds"], errors="coerce")
        rank = pd.to_numeric(work["task_probe_rank_seconds"], errors="coerce")
        route_vals = pd.to_numeric(work["route_final_seconds"], errors="coerce")
        core_vals = forward + rank
        valid_core = (
            np.isfinite(sample)
            & np.isfinite(core_vals)
            & np.isfinite(route_vals)
            & sample.ge(0)
            & core_vals.ge(0)
            & route_vals.ge(0)
            & (route_vals - sample - core_vals).abs().le(1e-6)
        )
        core_route_seconds = core_vals.loc[valid_core].dropna().astype(float)
    elif {
        "sample_seconds",
        "sample_to_route_seconds",
        "route_final_seconds",
    }.issubset(work.columns):
        sample = pd.to_numeric(work["sample_seconds"], errors="coerce")
        core_vals = pd.to_numeric(work["sample_to_route_seconds"], errors="coerce")
        route_vals = pd.to_numeric(work["route_final_seconds"], errors="coerce")
        valid_core = (
            np.isfinite(sample)
            & np.isfinite(core_vals)
            & np.isfinite(route_vals)
            & sample.ge(0)
            & core_vals.ge(0)
            & route_vals.ge(0)
            & (route_vals - sample - core_vals).abs().le(1e-6)
        )
        core_route_seconds = core_vals.loc[valid_core].dropna().astype(float)
    elif "sample_to_route_seconds" in work.columns:
        core_vals = pd.to_numeric(work["sample_to_route_seconds"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        core_route_seconds = core_vals[core_vals.ge(0)].dropna().astype(float)
    if not core_route_seconds.empty:
        out[SELECTOR_CORE_ROUTE_P50_COL] = _selector_round2(core_route_seconds.quantile(0.50))
        out[SELECTOR_CORE_ROUTE_P95_COL] = _selector_round2(core_route_seconds.quantile(0.95))

    e2e_values: list[float] = []
    has_order_source = any(col in work.columns for col in ["model_order", "selected_model_id", "selected_model_order"])
    if "route_final_seconds" not in work.columns:
        pass
    elif not has_order_source:
        notes.append("E2E-no_order")
    else:
        for idx, rec in work.loc[route_valid_mask].iterrows():
            model_key = _selector_first_model_key(rec)
            if not model_key:
                continue
            forward_ms = _vldb_tsfm_forward_ms(str(model_key), str(rec.get("dataset", "")), tsfm_results_dir=tsfm_results_dir)
            if not np.isfinite(forward_ms):
                continue
            e2e_values.append(float(pd.to_numeric(pd.Series([rec.get("route_final_seconds")]), errors="coerce").iloc[0]) + float(forward_ms) / 1000.0)
    e2e_n = int(len(e2e_values))
    out["_selector_e2e_valid_n"] = e2e_n
    if e2e_values:
        e2e_series = pd.Series(e2e_values, dtype=float)
        out[SELECTOR_E2E_P50_COL] = _selector_round2(e2e_series.quantile(0.50))
        out[SELECTOR_E2E_P95_COL] = _selector_round2(e2e_series.quantile(0.95))
    if e2e_n != expected_n:
        notes.append(f"E2E-{e2e_n}/{expected_n}")

    out["_selector_efficiency_note"] = " ".join(dict.fromkeys(notes))
    return out


def _selector_valid_ds_with_efficiency_note(valid_count: int, note: str) -> int | str:
    try:
        base = str(int(valid_count))
    except Exception:
        base = str(valid_count)
    note = str(note or "").strip()
    return f"{base} {note}" if note else int(valid_count)


def _vldb_full_zoo_forward_efficiency(
    current_zoo_num: int,
    expected: set[str],
    tsfm_results_dir: str = "cl_512",
) -> dict:
    rows = []
    missing = []
    model_keys = list(All_sorted_model_names[: int(current_zoo_num)])
    for dataset in sorted(expected):
        vals = []
        bad = []
        for model_key in model_keys:
            val = _vldb_tsfm_forward_ms(model_key, dataset, tsfm_results_dir=tsfm_results_dir)
            if np.isfinite(val):
                vals.append(float(val))
            else:
                bad.append(model_key)
        if bad:
            missing.append(f"{dataset}:{len(bad)}")
            continue
        full_forward_ms = float(sum(vals))
        rows.append(full_forward_ms)
    route = pd.Series(rows, dtype=float)
    qs = [_vldb_quantile(route, q) for q in [0.50, 0.90, 0.95]]
    p95_route_s = qs[-1] / 1000.0 if np.isfinite(qs[-1]) else np.nan
    reasons = []
    if len(rows) != len(expected):
        reasons.append(f"partial Eff-N={len(rows)}/{len(expected)}")
    else:
        reasons.append("full-zoo forward")
    if missing:
        reasons.append(f"missing_runtime_head={';'.join(missing[:3])}")
    return {
        "ROUTE_ms_P50_P90_P95": _vldb_format_ms_triplet(qs),
        "E2E_ms_P50_P90_P95": _vldb_format_ms_triplet(qs),
        "Route_Throughput": 60.0 / p95_route_s if np.isfinite(p95_route_s) and p95_route_s > 0 else np.nan,
        "Efficiency_valid_DS": int(len(rows)),
        "Skipped_parts": ", ".join(reasons),
        "_efficiency_diag": f"full_zoo_forward_models={len(model_keys)}, Eff-N={len(rows)}/{len(expected)}",
    }


def _vldb_tsfm_e2e_efficiency(
    model_key: str,
    expected: set[str],
    tsfm_results_dir: str = "cl_512",
) -> dict:
    vals = []
    missing = []
    for dataset in sorted(expected):
        val = _vldb_tsfm_forward_ms(str(model_key), str(dataset), tsfm_results_dir=tsfm_results_dir)
        if np.isfinite(val):
            vals.append(float(val))
        else:
            missing.append(str(dataset))
    series = pd.Series(vals, dtype=float)
    e2e_qs = [_vldb_quantile(series, q) for q in [0.50, 0.90, 0.95]]
    reasons = ["TSFM direct forward; Route=N/A"]
    if len(vals) != len(expected):
        reasons.append(f"partial E2E-N={len(vals)}/{len(expected)}")
    if missing:
        reasons.append(f"missing_head={','.join(missing[:3])}")
    return {
        "ROUTE_ms_P50_P90_P95": "N/A",
        "E2E_ms_P50_P90_P95": _vldb_format_ms_triplet(e2e_qs),
        "Route_Throughput": "N/A",
        "Efficiency_valid_DS": int(len(vals)),
        "Skipped_parts": ", ".join(reasons),
        "_efficiency_diag": f"tsfm_forward={model_key}, E2E-N={len(vals)}/{len(expected)}",
    }


def run_vldb_efficiency_presummary(args) -> pd.DataFrame:
    rows: list[dict] = []
    tsfm_results_dir = getattr(args, "TSFM_results_dir", "cl_512")
    expected = _vldb_expected_datasets(args)
    for model_key in All_sorted_model_names[: int(getattr(args, "zoo_total_num", len(All_sorted_model_names)))]:
        for dataset in sorted(expected):
            ms, source = _vldb_tsfm_forward_ms_with_source(model_key, dataset, tsfm_results_dir=tsfm_results_dir)
            rows.append({
                "source_type": "tsfm_full_task_forward",
                "stage": "",
                "dataset": dataset,
                "model_key": model_key,
                "model_id": "",
                "runtime_ms": ms,
                "runtime_role": "selected_tsfm_e2e",
                "source_file": source.split(":", 1)[0],
                "runtime_source": source,
                "status": "OK" if np.isfinite(ms) else "missing_runtime",
            })

    task_probe_path = Path("results_csv") / "TSRouter" / "vldb" / "logs" / "task_probe_sample_forward_log.csv"
    if task_probe_path.exists():
        try:
            df = pd.read_csv(task_probe_path, low_memory=False)
            value_col = next((c for c in ["sample_forward_ms", "candidate_forward_ms", "forward_ms"] if c in df.columns), "")
            for _, rec in df.iterrows():
                rows.append({
                    "source_type": "task_probe_sample_forward",
                    "stage": rec.get("stage", ""),
                    "dataset": rec.get("dataset", ""),
                    "model_key": rec.get("model_key", ""),
                    "model_id": rec.get("model_id", ""),
                    "runtime_ms": pd.to_numeric(pd.Series([rec.get(value_col, np.nan)]), errors="coerce").iloc[0] if value_col else np.nan,
                    "runtime_role": "task_probe_route_candidate_forward",
                    "sample_repr_num": rec.get("sample_repr_num", ""),
                    "task_window_sample_strategy": rec.get("task_window_sample_strategy", ""),
                    "sample_repr_ratio": rec.get("sample_repr_ratio", ""),
                    "task_sample_version": rec.get("task_sample_version", ""),
                    "search_seed": rec.get("search_seed", ""),
                    "repr_scale_protocol": rec.get("repr_scale_protocol", ""),
                    "source_file": str(task_probe_path),
                    "runtime_source": f"{task_probe_path}:{value_col}",
                    "status": rec.get("status", ""),
                })
        except Exception as e:
            rows.append({"source_type": "task_probe_sample_forward", "source_file": str(task_probe_path), "status": f"read_error:{type(e).__name__}"})

    repr_roots = [TSROUTER_REPR_FORWARD_CSV_ROOT]
    seen_repr_paths: set[Path] = set()
    for repr_root in repr_roots:
        if not repr_root.exists():
            continue
        for path in sorted(repr_root.glob("*_all_results.csv")):
            if path in seen_repr_paths:
                continue
            seen_repr_paths.add(path)
            is_pool = "_pool" in path.name
            source_type = "profile_pool_forward" if is_pool else "tsrouter_center_forward"
            runtime_role = "profile_pool_insert_forward" if is_pool else "tsrouter_insert_center_forward"
            try:
                df = pd.read_csv(path, low_memory=False)
            except Exception as e:
                rows.append({"source_type": source_type, "source_file": str(path), "status": f"read_error:{type(e).__name__}"})
                continue
            for _, rec in df.iterrows():
                ms = np.nan
                runtime_source_col = ""
                for col in ["insert_runtime_seconds", "forward_runtime_seconds", "non_eval_runtime_seconds", "runtime_seconds"]:
                    val = pd.to_numeric(pd.Series([rec.get(col)]), errors="coerce").iloc[0]
                    if pd.notna(val) and np.isfinite(float(val)):
                        ms = float(val) * 1000.0
                        runtime_source_col = col
                        break
                rows.append({
                    "source_type": source_type,
                    "stage": "",
                    "dataset": rec.get("dataset", ""),
                    "model_key": rec.get("model", ""),
                    "runtime_ms": ms,
                    "runtime_role": runtime_role,
                    "source_file": str(path),
                    "runtime_source": f"{path}:{runtime_source_col or 'missing_runtime'}",
                    "status": "OK" if np.isfinite(ms) else "missing_runtime",
                })

    route_path = Path("results_csv") / "TSRouter" / "vldb" / "logs" / "route_latency_log.csv"
    if route_path.exists():
        try:
            df = pd.read_csv(route_path, low_memory=False)
            if "timing_level" in df.columns:
                df = df[df["timing_level"].astype(str).eq("selector_dataset_internal")].copy()
            if "stage" in df.columns:
                df = df[pd.to_numeric(df["stage"], errors="coerce").le(float(getattr(args, "zoo_total_num", 10)))]
            for _, rec in df.iterrows():
                method = str(rec.get("method", ""))
                is_tsrouter = method in {"TSRouter", "TSRouter"}
                fast_sample = str(rec.get("vldb_fast_sample", "")).lower() == "true"
                roles = []
                if is_tsrouter and not fast_sample:
                    roles.append(("tsrouter_task_sample_repr", "task_sampling_ms"))
                roles.append(("route_overhead", "route_overhead_ms"))
                for role, col in roles:
                    source_type = "tsrouter_task_sample_repr" if role == "tsrouter_task_sample_repr" else "route_latency_log"
                    rows.append({
                        "source_type": source_type,
                        "stage": rec.get("stage", ""),
                        "dataset": rec.get("dataset", ""),
                        "method": method,
                        "route_id": rec.get("route_id", ""),
                        "runtime_ms": pd.to_numeric(pd.Series([rec.get(col, np.nan)]), errors="coerce").iloc[0],
                        "runtime_role": role,
                        "source_file": str(route_path),
                        "runtime_source": f"{route_path}:{col}",
                        "status": rec.get("timing_valid", ""),
                    })
        except Exception as e:
            rows.append({"source_type": "route_latency_log", "source_file": str(route_path), "status": f"read_error:{type(e).__name__}"})

    out = pd.DataFrame(rows)
    if not out.empty:
        out["runtime_ms"] = pd.to_numeric(out.get("runtime_ms", pd.Series(index=out.index, dtype=float)), errors="coerce")
        ok_status = out.get("status", pd.Series(index=out.index, dtype=str)).astype(str).str.lower().isin(
            {"ok", "success", "executed", "true"}
        )
        ok_numeric = out["runtime_ms"].notna() & np.isfinite(out["runtime_ms"]) & out["runtime_ms"].ge(0)
        out = out[ok_status & ok_numeric].copy()
        sort_cols = [c for c in ["timestamp_utc", "source_file"] if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols)
        subset = [
            c for c in [
                "source_type",
                "runtime_role",
                "stage",
                "dataset",
                "model_id",
                "model_key",
                "method",
                "route_id",
                "sample_repr_num",
                "task_window_sample_strategy",
                "sample_repr_ratio",
                "task_sample_version",
                "search_seed",
                "repr_scale_protocol",
            ]
            if c in out.columns
        ]
        if subset:
            out = out.drop_duplicates(subset=subset, keep="last").copy()
    out_dir = Path("results_csv") / "TSRouter" / "vldb" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "vldb_efficiency_presummary.csv"
    out.to_csv(path, index=False)
    print(f"[write] {str(path).replace(chr(92), '/')}")
    check_table = print_vldb_efficiency_presummary_check(out, args)
    try:
        setattr(args, "_vldb_presummary_check_table", check_table)
    except Exception:
        pass
    return out


def print_vldb_efficiency_presummary_check(out: pd.DataFrame, args) -> None:
    expected = _vldb_expected_datasets(args)
    stage = int(getattr(args, "zoo_total_num", 10))
    task_n = int(getattr(args, "sample_repr_num", 20))
    checks: list[dict] = []
    if out is None or out.empty:
        checks.append({"Check": "presummary", "Have": 0, "Expected": 1, "OK": False, "Note": "empty usable presummary"})
    else:
        tsfm = out[out.get("source_type", pd.Series(dtype=str)).astype(str).eq("tsfm_full_task_forward")]
        checks.append({
            "Check": "TSFM full-task forward",
            "Have": int(len(tsfm)),
            "Expected": int(len(expected) * stage),
            "OK": int(len(tsfm)) >= int(len(expected) * stage),
            "Note": "dataset x model runtime for TSFM E2E",
        })
        tp = out[
            out.get("source_type", pd.Series(dtype=str)).astype(str).eq("task_probe_sample_forward")
            & pd.to_numeric(out.get("stage", pd.Series(dtype=float)), errors="coerce").eq(float(stage))
            & pd.to_numeric(out.get("sample_repr_num", pd.Series(dtype=float)), errors="coerce").eq(float(task_n))
        ]
        checks.append({
            "Check": f"Task-Probe sample-forward taskN={task_n}",
            "Have": int(len(tp)),
            "Expected": int(len(expected) * stage),
            "OK": int(len(tp)) >= int(len(expected) * stage),
            "Note": "dataset x current-stage model candidate forward",
        })
        sample = out[
            out.get("source_type", pd.Series(dtype=str)).astype(str).eq("tsrouter_task_sample_repr")
            | out.get("runtime_role", pd.Series(dtype=str)).astype(str).eq("tsrouter_task_sample_repr")
        ]
        sample_n = int(len(sample))
        checks.append({
            "Check": "TSRouter task-sample repr rows",
            "Have": sample_n,
            "Expected": int(len(expected)),
            "OK": sample_n >= int(len(expected)),
            "Note": "usable presummary rows from task_sampling_ms",
        })
        checks.append({
            "Check": "TSRouter real task-sample build",
            "Have": int(sample["dataset"].astype(str).nunique()) if "dataset" in sample.columns else 0,
            "Expected": int(len(expected)),
            "OK": int(sample["dataset"].astype(str).nunique()) >= int(len(expected)) if "dataset" in sample.columns else False,
            "Note": "task_sampling_ms with vldb_fast_sample=false",
        })
        for source_type, label in [
            ("tsrouter_center_forward", "TSRouter center forward for insert"),
            ("profile_pool_forward", "Profile pool forward for insert"),
        ]:
            sub = out[out.get("source_type", pd.Series(dtype=str)).astype(str).eq(source_type)]
            checks.append({
                "Check": label,
                "Have": int(len(sub)),
                "Expected": int(stage),
                "OK": int(len(sub)) >= int(stage),
                "Note": "rows must be regenerated if zero; rerun Step2 with timing columns",
            })
    table = pd.DataFrame(checks)
    print("\n" + "=" * 88)
    print("VLDB Efficiency Presummary Quality Check")
    print("=" * 88)
    print(tabulate(table, headers="keys", tablefmt="plain", showindex=False))
    return table


def _vldb_task_probe_efficiency_from_detail(
    current_zoo_num: int,
    method_name: str,
    expected_n: int | None,
) -> dict | None:
    detail_path = Path("results_csv") / "TSRouter" / "vldb" / "logs" / "baseline_forward_detail_log.csv"
    if not detail_path.exists() or not str(method_name).startswith("Task-Probe Forward"):
        return None
    try:
        detail = pd.read_csv(detail_path)
    except Exception:
        return None
    required = {"dataset", "model_id", "model_key", "candidate_forward_ms", "internal_eval_ms", "sample_select_ms", "score_MASE"}
    if detail.empty or not required.issubset(set(detail.columns)):
        return None
    match = re.search(r"taskN=(\d+)", str(method_name))
    if match and "sample_repr_num" in detail.columns:
        detail = detail[pd.to_numeric(detail["sample_repr_num"], errors="coerce").eq(float(match.group(1)))]
    if "stage" in detail.columns:
        detail = detail[pd.to_numeric(detail["stage"], errors="coerce").eq(float(current_zoo_num))]
    detail = detail[pd.to_numeric(detail["model_id"], errors="coerce").lt(int(current_zoo_num))].copy()
    if detail.empty:
        return None
    detail = _vldb_latest_by_file_order(detail, ["dataset", "model_id"])
    rows = []
    for dataset, group in detail.groupby("dataset"):
        forward_ms = pd.to_numeric(group["candidate_forward_ms"], errors="coerce")
        eval_ms = pd.to_numeric(group["internal_eval_ms"], errors="coerce").fillna(0.0)
        sample_ms = pd.to_numeric(group["sample_select_ms"], errors="coerce").fillna(0.0)
        if "task_sample_build_ms" in group.columns:
            build_vals = pd.to_numeric(group["task_sample_build_ms"], errors="coerce").dropna()
            if build_vals.empty:
                continue
            task_sample_build_ms = float(build_vals.max())
        else:
            task_sample_build_ms = 0.0
        score = pd.to_numeric(group["score_MASE"], errors="coerce")
        if forward_ms.isna().any() or score.dropna().empty:
            continue
        best_idx = score.idxmin()
        selected_key = str(group.loc[best_idx, "model_key"])
        selected_forecast_ms = _vldb_tsfm_forward_ms(selected_key, str(dataset))
        if not np.isfinite(selected_forecast_ms):
            continue
        route_ms = float(task_sample_build_ms + forward_ms.sum() + eval_ms.sum() + sample_ms.sum())
        rows.append((route_ms, route_ms + float(selected_forecast_ms)))
    if not rows:
        return None
    route_series = pd.Series([r for r, _ in rows], dtype=float)
    e2e_series = pd.Series([e for _, e in rows], dtype=float)
    route_qs = [_vldb_quantile(route_series, q) for q in [0.50, 0.90, 0.95]]
    e2e_qs = [_vldb_quantile(e2e_series, q) for q in [0.50, 0.90, 0.95]]
    p95_route_s = route_qs[-1] / 1000.0 if np.isfinite(route_qs[-1]) else np.nan
    reasons = [f"derived from stage detail by model_id<{int(current_zoo_num)}"]
    if "task_sample_build_ms" not in detail.columns:
        reasons.append("legacy detail lacks task_sample_build_ms")
    if expected_n is not None and len(rows) != int(expected_n):
        reasons.append(f"partial Eff-N={len(rows)}/{int(expected_n)}")
    return {
        "ROUTE_ms_P50_P90_P95": _vldb_format_ms_triplet(route_qs),
        "E2E_ms_P50_P90_P95": _vldb_format_ms_triplet(e2e_qs),
        "Route_Throughput": 60.0 / p95_route_s if np.isfinite(p95_route_s) and p95_route_s > 0 else np.nan,
        "Efficiency_valid_DS": int(len(rows)),
        "Skipped_parts": ", ".join(reasons),
        "_efficiency_diag": f"task_probe_detail={detail_path}, rows={len(rows)}",
    }


def _vldb_route_efficiency(
    results_dir: Path,
    method_name: str,
    current_zoo_num: int,
    route_id: str | list[str] | tuple[str, ...] | None = None,
    expected_n: int | None = None,
    ignore_route_valid_flags: bool = False,
) -> dict:
    log_path = Path("results_csv") / "TSRouter" / "vldb" / "logs" / "route_latency_log.csv"
    out = {
        "ROUTE_ms_P50_P90_P95": "",
        "E2E_ms_P50_P90_P95": "",
        "Route_Throughput": np.nan,
        "Efficiency_valid_DS": 0,
        "Skipped_parts": "",
        "_efficiency_diag": "",
    }
    is_task_probe = str(method_name).startswith("Task-Probe Forward")
    if not log_path.exists():
        out["Skipped_parts"] = "missing route_latency_log"
        out["_efficiency_diag"] = out["Skipped_parts"]
        return out
    try:
        log_df = pd.read_csv(log_path)
    except Exception:
        out["Skipped_parts"] = "cannot read route_latency_log"
        out["_efficiency_diag"] = out["Skipped_parts"]
        return out
    if log_df.empty or "timing_level" not in log_df.columns:
        out["Skipped_parts"] = "empty route_latency_log"
        out["_efficiency_diag"] = out["Skipped_parts"]
        return out
    df = log_df[log_df["timing_level"].astype(str).eq("selector_dataset_internal")].copy()
    if "stage" in df.columns:
        df = df[pd.to_numeric(df["stage"], errors="coerce").eq(int(current_zoo_num))]
    if "method" in df.columns and method_name:
        method_candidates = {str(method_name)}
        if str(method_name) in {"TSRouter", "TSRouter-fast"}:
            method_candidates.update({"TSRouter", "TSRouter"})
        if str(method_name).startswith("Task-Probe Forward"):
            method_candidates.update({"Task-Probe Forward"})
        df = df[df["method"].astype(str).isin(method_candidates)]
    candidate_ids = []
    if isinstance(route_id, (list, tuple)):
        candidate_ids = [str(x) for x in route_id if x]
    elif route_id:
        candidate_ids = [str(route_id)]
    requested = candidate_ids[0] if candidate_ids else ""
    used_route_id = ""
    if candidate_ids and "route_id" in df.columns:
        matched = pd.DataFrame()
        for rid in candidate_ids:
            sub = df[df["route_id"].astype(str).eq(rid)].copy()
            if not sub.empty:
                matched = sub
                used_route_id = rid
                break
        df = matched
    if df.empty:
        out["Skipped_parts"] = "no matched route rows"
        out["_efficiency_diag"] = f"no matched route rows: {requested or 'stage aggregate'}"
        return out
    raw_row_count = len(df)
    if "dataset" in df.columns:
        df = _vldb_latest_by_file_order(df, "dataset")
    route_ms = pd.to_numeric(df.get("route_overhead_ms", pd.Series(index=df.index, dtype=float)), errors="coerce")
    forward_ms = pd.to_numeric(df.get("selected_forecast_ms", pd.Series(index=df.index, dtype=float)), errors="coerce")
    row_valid = _vldb_bool_col(df, "timing_valid", True)
    route_valid = _vldb_bool_col(df, "task_sampling_timing_valid", True)
    forward_valid = _vldb_bool_col(df, "selected_forecast_timing_valid", True)
    proxy_forward = pd.Series(False, index=df.index)
    if "forward_mode" in df.columns:
        proxy_forward = df["forward_mode"].fillna("").astype(str).str.lower().str.contains("proxy")
        forward_valid = forward_valid & ~proxy_forward
        if is_task_probe:
            real_task_probe_forward = df["forward_mode"].fillna("").astype(str).str.lower().str.contains("task_probe_sample_forward_log")
            route_valid = route_valid & real_task_probe_forward
    derived_forward_ms = pd.Series(np.nan, index=df.index, dtype=float)
    if {"selected_model_order", "dataset"}.issubset(df.columns):
        for idx, rec in df.iterrows():
            model_key = _vldb_first_selected_model_key(rec.get("selected_model_order"))
            if not model_key:
                continue
            val = _vldb_tsfm_forward_ms(model_key, str(rec.get("dataset", "")))
            if np.isfinite(val):
                derived_forward_ms.at[idx] = float(val)
    runtime_fill_mask = derived_forward_ms.notna()
    effective_forward_ms = forward_ms.copy()
    effective_forward_ms[runtime_fill_mask] = derived_forward_ms[runtime_fill_mask]
    effective_forward_valid = runtime_fill_mask | (forward_valid & forward_ms.notna() & ~proxy_forward)
    if ignore_route_valid_flags:
        route_mask = route_ms.notna()
        if is_task_probe:
            route_mask = route_mask & route_valid
    else:
        route_mask = row_valid & route_valid & route_ms.notna()
    e2e_mask = route_mask & effective_forward_valid & effective_forward_ms.notna()
    valid = df[route_mask].copy()
    valid_route_ms = route_ms[route_mask]
    valid_e2e_ms = route_ms[e2e_mask] + effective_forward_ms[e2e_mask]
    route_qs = []
    e2e_qs = []
    for q_label, q in [("P50", 0.50), ("P90", 0.90), ("P95", 0.95)]:
        route_q = _vldb_quantile(valid_route_ms, q)
        e2e_q = _vldb_quantile(valid_e2e_ms, q)
        route_qs.append(route_q)
        e2e_qs.append(e2e_q)
    out["ROUTE_ms_P50_P90_P95"] = _vldb_format_ms_triplet(route_qs)
    out["E2E_ms_P50_P90_P95"] = _vldb_format_ms_triplet(e2e_qs)
    p95_route_s = route_qs[-1] / 1000.0 if np.isfinite(route_qs[-1]) else np.nan
    out["Route_Throughput"] = 60.0 / p95_route_s if np.isfinite(p95_route_s) and p95_route_s > 0 else np.nan
    out["Efficiency_valid_DS"] = int(len(valid))
    reasons = []
    if expected_n is not None and len(valid) != int(expected_n):
        reasons.append(f"partial Eff-N={len(valid)}/{int(expected_n)}")
    elif len(valid) == 0:
        reasons.append("no valid route timing")
    if expected_n is not None and len(valid_e2e_ms) != int(expected_n):
        reasons.append(f"partial E2E-N={len(valid_e2e_ms)}/{int(expected_n)}")
    method_values = set(df.get("method", pd.Series(dtype=str)).dropna().astype(str).unique())
    is_tsrouter = bool(method_values & {"TSRouter", "TSRouter"})
    if is_tsrouter:
        for flag, label in [
            ("vldb_fast_sample", "sample cache"),
            ("vldb_fast_forward", "forward cache"),
        ]:
            if flag in df.columns and df[flag].astype(str).str.lower().eq("true").any():
                reasons.append(label)
    if runtime_fill_mask.any():
        reasons.append("selected forward from TSFM runtime")
    for flag, label in [
        ("timing_valid", "timing invalid"),
        ("task_sampling_timing_valid", "sample invalid"),
        ("selected_forecast_timing_valid", "forward invalid"),
    ]:
        if flag in df.columns and df[flag].astype(str).str.lower().eq("false").any():
            if flag == "selected_forecast_timing_valid" and len(valid_e2e_ms) == len(valid):
                continue
            reasons.append(label)
    if "forward_mode" in df.columns:
        forward_modes = " ".join(df["forward_mode"].dropna().astype(str).unique()).lower()
        proxy_unrecovered = proxy_forward & ~runtime_fill_mask
        if "proxy" in forward_modes and proxy_unrecovered.any():
            reasons.append("proxy timing")
        if is_task_probe and not forward_modes.count("task_probe_sample_forward_log"):
            reasons.append("missing real task_probe_sample_forward_log route timing")
    if not reasons:
        reasons.append("OK")
    out["Skipped_parts"] = ", ".join(dict.fromkeys(reasons))
    legacy_note = f", using legacy route_id={used_route_id}" if requested and used_route_id and used_route_id != requested else ""
    out["_efficiency_diag"] = (
        f"route_id={used_route_id or requested or 'stage aggregate'}, "
        f"rows={len(df)}/{raw_row_count}, Eff-N={len(valid)}{legacy_note}"
    )
    return out


def _vldb_generated_baseline_configs(current_zoo_num: int, zoo_total_num: int, sample_repr_num: int = 20) -> list[dict]:
    return [
        {
            "label": "Random",
            "method_type": "static",
            "method": "Random",
            "stage": int(current_zoo_num),
            "glob": f"results_csv/baselines/vldb/Random/zoo{current_zoo_num}-{zoo_total_num}_random_mean*_all_results.csv",
            "route_id": f"stage{current_zoo_num}_random_route",
        },
        {
            "label": "Recent",
            "method_type": "static",
            "method": "Recent",
            "stage": int(current_zoo_num),
            "glob": f"results_csv/baselines/vldb/Recent/zoo{current_zoo_num}-{zoo_total_num}_recent_all_results.csv",
            "route_id": f"stage{current_zoo_num}_recent_route",
        },
        {
            "label": "Zoo-Oracle Static Best",
            "method_type": "oracle",
            "method": "Static Best",
            "stage": int(current_zoo_num),
            "glob": f"results_csv/baselines/vldb/Static_Best/zoo{current_zoo_num}-{zoo_total_num}_static_best_all_results.csv",
            "route_id": f"stage{current_zoo_num}_static_best_route",
            "oracle": True,
        },
        {
            "label": f"Task-Probe Forward (taskN={int(sample_repr_num)})",
            "method_type": "dynamic-select",
            "method": f"Task-Probe Forward (taskN={int(sample_repr_num)})",
            "stage": int(current_zoo_num),
            "glob": f"results_csv/baselines/vldb/Task_Probe_Forward/zoo{current_zoo_num}-{zoo_total_num}_task_probe_forward_task{int(sample_repr_num)}_all_results.csv",
            "route_id": [
                f"stage{current_zoo_num}_task_probe_forward_task{int(sample_repr_num)}_route",
                f"stage{current_zoo_num}_task_probe_forward_route",
            ],
        },
    ]


def _vldb_load_generated_baseline(config: dict, expected: set[str], args) -> pd.DataFrame:
    candidates = []
    glob_specs = config.get("glob", [])
    if isinstance(glob_specs, str):
        glob_specs = [glob_specs]
    stage = int(config.get("stage", getattr(args, "zoo_total_num", 0)))
    exact_prefix = f"zoo{stage}-{int(getattr(args, 'zoo_total_num', 0))}_"
    for priority, glob_spec in enumerate(glob_specs):
        for path in sorted(Path(".").glob(str(glob_spec))):
            if not path.name.startswith(exact_prefix):
                continue
            raw = check_results_file(path, args.verbose, args.quick_test)
            if raw is None or raw.empty:
                continue
            raw = harmonize_metrics_schema(raw)
            raw["model"] = str(config["label"])
            raw["source_file"] = str(path)
            if args.quick_test:
                raw = raw[raw["dataset"].isin(expected)].copy()
            qualified = int(len(set(raw.get("dataset", pd.Series(dtype=str)).dropna().astype(str)) & expected))
            complete = int(qualified == len(expected))
            try:
                mtime = float(path.stat().st_mtime)
            except Exception:
                mtime = 0.0
            candidates.append((complete, qualified, -priority, mtime, path, raw))
    if not candidates:
        return pd.DataFrame()
    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    out = candidates[0][5].copy()
    if "dataset" in out.columns:
        out = _vldb_latest_by_file_order(out, "dataset")
    return out


def _vldb_missing_generated_baseline_row(config: dict, expected: set[str]) -> dict:
    label = str(config.get("label", "Generated Baseline"))
    return {
        "Method": label,
        "Method_Type": str(config.get("method_type", "baseline")),
        "Files": 0,
        "Qualified_DS": 0,
        "Expected_DS": int(len(expected)),
        "Complete": False,
        "Mean_MASE": np.nan,
        "Rank": np.nan,
        "Top-1 hit": np.nan,
        "Top-3 hit": np.nan,
        "Selected_Top1": "",
        "Selected_Top2": "",
        "Mean_sMAPE": np.nan,
        "win-rate-sMAPE": np.nan,
        "Mean_CRPS": np.nan,
        "ROUTE_ms_P50_P90_P95": "N/A",
        "E2E_ms_P50_P90_P95": "N/A",
        "Route_Throughput": "N/A",
        "Efficiency_valid_DS": 0,
        "Skipped_parts": f"missing generated file: {config.get('glob', '')}",
        "_efficiency_diag": f"missing generated file: {config.get('glob', '')}",
    }


def _vldb_oracle_efficiency_na() -> dict:
    return {
        "ROUTE_ms_P50_P90_P95": "N/A",
        "E2E_ms_P50_P90_P95": "N/A",
        "Route_Throughput": "N/A",
        "Efficiency_valid_DS": "N/A",
        "Skipped_parts": "oracle quality only",
        "_efficiency_diag": "oracle quality only; latency/throughput/insert cost are N/A",
    }


def _vldb_apply_grid_value(args_v, key: str, value):
    from utils.path_utils import infer_type_structure_from_encoder_name, normalize_encoder_variant_args

    if key in {
        "ablation_name",
        "vldb_route_id",
        "task_channel_fuse_limit",
        "route_family_mode",
    }:
        setattr(args_v, key, value)
        return
    if key in {"route_efficiency_mode"}:
        setattr(args_v, key, _vldb_truthy(value))
        return
    if key in {
        "repr_weight_ratio",
        "minus_ratio",
        "subset_perf_scale",
        "rank_decay_coef",
        "context_len_adaptive_threshold",
        "sample_repr_ratio",
        "task_rank_top3_instability_threshold",
    }:
        value = float(value)
    elif key in {
        "repr_v",
        "sample_repr_num",
        "task_sample_version",
        "restrict_top_model_num",
        "ensemble_size",
        "subset_top_k",
        "repr_size",
        "repr_data_seed",
        "repr_encoder_seed",
        "forward_seed",
        "search_seed",
        "repr_input_dim",
        "repr_output_dim",
        "repr_sub_pred_len",
        "repr_v5_nearest_k",
        "pred_len_adaptive_threshold",
        "short_repr_input_dim",
        "long_repr_input_dim",
        "short_repr_sub_pred_len",
        "long_repr_sub_pred_len",
        "train_encoder_epochs",
        "train_encoder_batch_size",
    }:
        value = int(value)
    setattr(args_v, key, value)
    if key == "repr_encoder":
        encoder_type, encoder_structure = infer_type_structure_from_encoder_name(str(value))
        if encoder_type is not None:
            args_v.encoder_type = encoder_type
            args_v.encoder_structure = encoder_structure
            normalize_encoder_variant_args(args_v)
    elif key in {"encoder_type", "encoder_structure"}:
        normalize_encoder_variant_args(args_v)


def _bind_simplets_ts2vec_summary_checkpoint(args_v, *, allow_missing: bool = False) -> bool:
    """Bind checkpoint provenance before reconstructing a Step4 result path."""
    if str(getattr(args_v, "repr_encoder", "") or "") != "SimpleTS2Vec":
        return True
    from encoder.simplets_checkpoint import resolve_simplets_ts2vec_checkpoint

    try:
        resolve_simplets_ts2vec_checkpoint(args_v)
    except FileNotFoundError as exc:
        if not allow_missing:
            raise
        args_v._simplets_ts2vec_checkpoint_missing_reason = str(exc)
        return False
    return True


def _vldb_grid_tsrouter_tasks(args, current_zoo_num: int, param_grid: dict) -> list[tuple[str, Path, dict]]:
    from utils.path_utils import get_repr_save_path

    keys = list(param_grid.keys())
    tasks = []
    for values in product(*[param_grid[k] for k in keys]):
        args_v = copy.deepcopy(args)
        args_v.current_zoo_num = current_zoo_num
        for key, value in zip(keys, values):
            _vldb_apply_grid_value(args_v, key, value)
        from utils.path_utils import normalize_encoder_variant_args
        normalize_encoder_variant_args(args_v)
        _bind_simplets_ts2vec_summary_checkpoint(args_v)
        if args_v.fix_context_len:
            args_v.context_len = int(args_v.repr_input_dim)
        _, _, _, save_name = get_repr_save_path(args_v)
        path = resolve_tsrouter_selector_result_path(args_v, save_name)
        params = dict(zip(keys, values))
        label = _vldb_config_label(args_v, params)
        effective_route_id = _vldb_route_id_for_args(current_zoo_num, args_v, params)
        params["_vldb_effective_route_id"] = effective_route_id
        candidates = [effective_route_id]
        if (
            str(params.get("ablation_name", "main")) == "main"
            and str(params.get("repr_encoder", getattr(args_v, "repr_encoder", ""))) == "StatsRandomFourier"
            and str(params.get("repr_sample_qc_mode", getattr(args_v, "repr_sample_qc_mode", "strict"))) == "strict"
            and str(params.get("repr_scale_protocol", getattr(args_v, "repr_scale_protocol", "std"))) in {"std", "standard"}
            and not _vldb_truthy(params.get("route_efficiency_mode", False))
            and str(params.get("route_family_mode", "default")) == "default"
        ):
            candidates.append(f"stage{current_zoo_num}_main_route")
        params["_vldb_route_id_candidates"] = candidates
        tasks.append((label, path, params))
    return tasks


def _vldb_config_label(args_v, params: dict) -> str:
    if _vldb_truthy(params.get("route_efficiency_mode", getattr(args_v, "route_efficiency_mode", False))):
        route_family_mode = str(
            params.get("route_family_mode", getattr(args_v, "route_family_mode", "default"))
            or "default"
        ).strip().lower()
        if route_family_mode == "bigger_size":
            return "TSRouter-fast-rfbigger"
        if route_family_mode == "smaller_size":
            return "TSRouter-fast-rfsmaller"
        return "TSRouter-fast"
    if params.get("ablation_name"):
        name = str(params["ablation_name"])
        if name and name != "main":
            return name
    label_parts = [
        str(params.get("repr_encoder", getattr(args_v, "repr_encoder", "TSRouter"))),
        f"v{params.get('repr_v', getattr(args_v, 'repr_v', ''))}{params.get('base_metrics', getattr(args_v, 'base_metrics', ''))}",
        f"task{params.get('sample_repr_num', getattr(args_v, 'sample_repr_num', ''))}",
        f"res{params.get('restrict_top_model_num', getattr(args_v, 'restrict_top_model_num', ''))}",
    ]
    if "repr_anchor_window_sample_strategy" in params:
        label_parts.append(f"aws{params['repr_anchor_window_sample_strategy']}")
    if "task_window_sample_strategy" in params:
        label_parts.append(f"ws{params['task_window_sample_strategy']}")
    if "sample_repr_ratio" in params:
        label_parts.append(f"sr{params['sample_repr_ratio']}")
    if "task_rank_top3_instability_threshold" in params:
        fallback_token = _vldb_fallback_param_token(params["task_rank_top3_instability_threshold"])
        if fallback_token is not None:
            label_parts.append(fallback_token)
    if "task_channel_fuse_limit" in params:
        raw = str(params["task_channel_fuse_limit"])
        if raw.strip().lower() not in {"", "all", "none"}:
            label_parts.append(f"cf{raw}")
    route_family_mode = str(
        params.get("route_family_mode", getattr(args_v, "route_family_mode", "default"))
        or "default"
    ).strip().lower()
    if route_family_mode != "default":
        label_parts.append("rfbigger" if route_family_mode == "bigger_size" else "rfsmaller")
    return "-".join([p for p in label_parts if p])


def _vldb_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower() or "config"


def _vldb_param_token(value) -> str:
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    text = str(value)
    try:
        f = float(text)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return text


def _vldb_fallback_param_token(value) -> str | None:
    try:
        threshold = float(value)
    except Exception:
        return None
    if not np.isfinite(threshold) or threshold < 0:
        return None
    if abs(threshold) < 1e-12:
        threshold = 0.0
    return f"fb{_vldb_param_token(threshold)}"


def _vldb_route_suffix_for_args(args_v, params: dict) -> str:
    prefix = str(params.get("ablation_name") or params.get("route_suffix") or "main")
    fallback_token = _vldb_fallback_param_token(
        params.get(
            "task_rank_top3_instability_threshold",
            getattr(args_v, "task_rank_top3_instability_threshold", -1),
        )
    )
    parts = [
        prefix,
        f"enc{params.get('repr_encoder', getattr(args_v, 'repr_encoder', ''))}",
        f"qc{params.get('repr_sample_qc_mode', getattr(args_v, 'repr_sample_qc_mode', 'strict'))}",
        f"scale{params.get('repr_scale_protocol', getattr(args_v, 'repr_scale_protocol', 'std'))}",
        f"zoo{params.get('zoo_repr_set', getattr(args_v, 'zoo_repr_set', ''))}",
        f"n{_vldb_param_token(params.get('repr_size', getattr(args_v, 'repr_size', '')))}",
        f"v{_vldb_param_token(params.get('repr_v', getattr(args_v, 'repr_v', '')))}{params.get('base_metrics', getattr(args_v, 'base_metrics', ''))}",
        f"task{_vldb_param_token(params.get('sample_repr_num', getattr(args_v, 'sample_repr_num', '')))}",
        f"aws{params.get('repr_anchor_window_sample_strategy', getattr(args_v, 'repr_anchor_window_sample_strategy', 'even'))}",
        f"ws{params.get('task_window_sample_strategy', getattr(args_v, 'task_window_sample_strategy', 'legacy'))}",
        f"sr{_vldb_param_token(params.get('sample_repr_ratio', getattr(args_v, 'sample_repr_ratio', 0)))}",
    ]
    if fallback_token is not None:
        parts.append(fallback_token)
    channel_fuse = str(params.get("task_channel_fuse_limit", getattr(args_v, "task_channel_fuse_limit", "all")) or "all")
    if channel_fuse.strip().lower() not in {"", "all", "none"}:
        parts.append(f"cf{channel_fuse}")
    if _vldb_truthy(params.get("route_efficiency_mode", getattr(args_v, "route_efficiency_mode", False))):
        parts.append("rfast")
    route_family_mode = str(
        params.get("route_family_mode", getattr(args_v, "route_family_mode", "default"))
        or "default"
    ).strip().lower()
    if route_family_mode != "default":
        parts.append("rfbigger" if route_family_mode == "bigger_size" else "rfsmaller")
    return _vldb_slug("_".join(str(p) for p in parts if str(p) != ""))


def _vldb_route_id_for_args(current_zoo_num: int, args_v, params: dict) -> str:
    if params.get("vldb_route_id"):
        return str(params["vldb_route_id"])
    return f"stage{current_zoo_num}_{_vldb_route_suffix_for_args(args_v, params)}_route"


def _vldb_route_id_for_config(current_zoo_num: int, label: str, params: dict) -> str | None:
    if params.get("vldb_route_id"):
        return str(params["vldb_route_id"])
    if params.get("_vldb_effective_route_id"):
        return str(params["_vldb_effective_route_id"])
    return f"stage{current_zoo_num}_{_vldb_slug(label)}_route"


def _vldb_route_id_candidates_for_config(current_zoo_num: int, label: str, params: dict) -> list[str]:
    if params.get("_vldb_route_id_candidates"):
        return [str(x) for x in params["_vldb_route_id_candidates"]]
    rid = _vldb_route_id_for_config(current_zoo_num, label, params)
    return [str(rid)] if rid else []


VLDB_TABLE_COLUMNS = {
    "table1": [
        ("Method_Type", "Type"),
        ("Method", "Method"),
        ("Files", "Files"),
        ("Qualified_DS", "N"),
        ("Expected_DS", "Expected"),
        ("Complete", "Complete"),
        ("Mean_MASE", "MASE"),
        ("Regret_MASE", "Regret-M"),
        ("Regret_MASE_P90", "Regret-M P90"),
        ("Rank", "Rank"),
        ("Top-1 hit", "Top1-hit"),
        ("Top-3 hit", "Top3-hit"),
        ("Selected_Top1", "Selected Top1(%)"),
        ("Selected_Top2", "Selected Top2(%)"),
        ("Mean_sMAPE", "sMAPE"),
        ("win-rate-sMAPE", "Win-sMAPE"),
        ("Mean_CRPS", "CRPS"),
        ("ROUTE_ms_P50_P90_P95", "Route P50/90/95(ms)"),
        ("E2E_ms_P50_P90_P95", "E2E P50/90/95(ms)"),
        ("Route_Throughput", "ROUTE-Throughput(req/min)"),
        ("Efficiency_valid_DS", "Eff-N"),
    ],
    "table1_ablation": [
        ("Config", "Config"),
        ("Ablation", "Ablation"),
        ("Files", "Files"),
        ("Qualified_DS", "N"),
        ("Expected_DS", "Expected"),
        ("Complete", "Complete"),
        ("Seed_N", "Seed-N"),
        ("ReprSeeds", "ReprSeeds"),
        ("Mean_MASE", "MASE"),
        ("Regret_MASE", "Regret-M"),
        ("Regret_MASE_P90", "Regret-M P90"),
        ("Top-1 hit", "Top1-hit"),
        ("Top-3 hit", "Top3-hit"),
        ("Mean_sMAPE", "sMAPE"),
        ("win-rate-sMAPE", "Win-sMAPE"),
        ("Mean_CRPS", "CRPS"),
        ("ROUTE_ms_P50_P90_P95", "Route P50/90/95(ms)"),
        ("E2E_ms_P50_P90_P95", "E2E P50/90/95(ms)"),
        ("Route_Throughput", "ROUTE-Throughput(req/min)"),
        ("Efficiency_valid_DS", "Eff-N"),
    ],
}

VLDB_PARAM_COLUMN_ALIASES = {
    "repr_encoder": "Enc",
    "zoo_repr_set": "ZooSet",
    "repr_size": "Nrepr",
    "repr_v": "v",
    "base_metrics": "Metric",
    "sample_repr_num": "TaskN",
    "repr_sample_qc_mode": "QC",
    "repr_scale_protocol": "Scale",
    "repr_anchor_window_sample_strategy": "AnchorWinSample",
    "task_window_sample_strategy": "WinSample",
    "sample_repr_ratio": "SampleRatio",
    "task_rank_top3_instability_threshold": "Fallback",
    "task_channel_fuse_limit": "ChanFuse",
    "route_family_mode": "RouteFamily",
    "encoder_type": "EncType",
    "encoder_structure": "EncArch",
    "repr_data_seed": "ReprSeed",
    "repr_encoder_seed": "EncSeed",
}


def _vldb_display_table(table: pd.DataFrame, table_key: str, param_cols: list[str] | None = None) -> pd.DataFrame:
    if table is None or table.empty:
        return table
    pairs = list(VLDB_TABLE_COLUMNS.get(table_key, []))
    if table_key.endswith("_ablation") and "Config" in table.columns:
        vals = {str(v) for v in table["Config"].dropna().unique()}
        if len(vals) <= 1:
            pairs = [pair for pair in pairs if pair[0] != "Config"]
    param_pairs = [
        (col, VLDB_PARAM_COLUMN_ALIASES.get(col, col))
        for col in (param_cols or [])
        if col in table.columns
    ]
    if table_key.endswith("_ablation"):
        if pairs and pairs[0][0] == "Config":
            pairs = pairs[:1] + param_pairs + pairs[1:]
        else:
            pairs = param_pairs + pairs
    else:
        pairs.extend(param_pairs)
    cols = [src for src, _ in pairs if src in table.columns]
    out = table[cols].copy()
    rename = {src: dst for src, dst in pairs if src in out.columns}
    return out.rename(columns=rename)


def _vldb_varying_param_cols(rows: list[dict], param_keys: list[str]) -> list[str]:
    varying = []
    for key in param_keys:
        vals = {
            str(row.get(key, ""))
            for row in rows
            if key in row and str(row.get(key, "")) != ""
        }
        if len(vals) > 1:
            varying.append(key)
    return varying


VLDB_TABLE1_METHOD_ORDER = [
    "Random",
    "Recent",
    "Zoo-Oracle Static Best",
    "Task-Probe Forward",
    "TSRouter",
    "TSRouter-fast",
]


def _vldb_table1_method_group(row: dict) -> tuple[int, int, str]:
    method = str(row.get("Method", ""))
    method_type = str(row.get("Method_Type", ""))
    if method_type == "TSFM":
        return (0, int(row.get("_tsfm_order", 0) or 0), method)
    if method in {"Random", "Recent", "Zoo-Oracle Static Best"}:
        return (1, {"Random": 0, "Recent": 1, "Zoo-Oracle Static Best": 2}[method], method)
    if method.startswith("Task-Probe Forward"):
        return (2, 0, method)
    if method == "TSRouter":
        return (3, 0, method)
    if method == "TSRouter-fast":
        return (3, 1, method)
    return (4, 0, method)


VLDB_TABLE1_METHOD_ORDER_LEGACY = [
    "Random",
    "Recent",
    "Zoo-Oracle Static Best",
    "TSRouter",
    "TSRouter-fast",
]


def _vldb_sort_table1_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=_vldb_table1_method_group)


def _vldb_table1_main_grid() -> dict:
    return {
        "ablation_name": ["main"],
        "repr_encoder": [
            "StatsRandomFourier",
            # "TrainFourier",
        ],
        "repr_input_dim": [512],
        "repr_output_dim": [256],
        "repr_sub_pred_len": [480],
        "zoo_repr_set": ["c-e-n-h-w-s"],
        "repr_size": [3000],
        "repr_v": [5],
        "base_metrics": [
            # "S",
            "C",
        ],
        "repr_weight_ratio": [0],
        "sample_repr_num": [20],
        "repr_data_seed": [2029],
        "repr_encoder_seed": [2025],
        "repr_scale_protocol": ["std"],
        "repr_sample_qc_mode": ["strict"],
        # "task_sample_strategy": ["latest_random", "time_coverage"],
        # "repr_anchor_protocol": ["window", "task_sample"],
        # "task_window_sample_strategy": ["legacy", "even", "random", "first", "last"],
        # "sample_repr_ratio": [0, 0.1, 0.3, 0.5],
        # "task_rank_top3_instability_threshold": [0, 0.5, 0.7, 0.9],
        "task_channel_fuse_limit": ["all"],
        "route_family_mode": ["default"],
        "route_efficiency_mode": [False, True],
    }


def _vldb_table1_ablation_grid() -> dict:
    return {
        "ablation_name": [
            "main",
            # "sample-ratio-0.3",
            # "fallback-0.7",
        ],
        "repr_encoder": [
            "StatsNone",
            "None",
            "StatsRandomFourier",
            # "StatsRandomPatch",
            # "StatsRandomInception",
            # "StatsRandomTCN",
            # "RandomFourier",
            # "StatsRandomMLP",
            # "RandomMLP",
            # "StatsRandomConv",
            # "RandomConv",
            # "TrainFourier",
        ],
        "repr_input_dim": [512],
        "repr_output_dim": [256],
        "repr_sub_pred_len": [480],
        "zoo_repr_set": ["c-e-n-h-w-s","c"],
        # "repr_size": [3000],
        "repr_size": [3000,1500],
        # "repr_v": [5,0],
        "repr_v": [5,4,0],
        "base_metrics": ["C"],
        "repr_weight_ratio": [0],
        # "sample_repr_num": [20,1],
        "sample_repr_num": [20],
        # "repr_data_seed": [2025, 2026, 2027, 2028 ,2029],
        "repr_data_seed": [2029],
        # "repr_encoder_seed": [2025, 2026, 2027, 2028 ,2029],
        "repr_encoder_seed": [2025],
        "repr_scale_protocol": ["std"],
        # "repr_sample_qc_mode": ['strict', "off"],
        "repr_sample_qc_mode": ['strict', "off"],
        # Edit these lists to expand TSRouter ablations.
        # "task_window_sample_strategy": ["legacy", "even", "first", "last"],
        # "sample_repr_ratio": [0, 0.1, 0.3, 0.5],
        # "task_rank_top3_instability_threshold": [0, 0.5, 0.7, 0.9],
        # "task_channel_fuse_limit": ["all",1],
        "task_channel_fuse_limit": ["all"],
        "route_family_mode": ["default", "bigger_size", "smaller_size"],
    }


def _vldb_ablation_description(params: dict) -> str:
    parts = []
    ablation_name = str(params.get("ablation_name", "main") or "main")
    repr_encoder = str(params.get("repr_encoder", ""))
    qc_mode = str(params.get("repr_sample_qc_mode", ""))
    repr_size = str(params.get("repr_size", ""))
    repr_v = str(params.get("repr_v", ""))
    task_n = str(params.get("sample_repr_num", ""))
    channel_fuse = str(params.get("task_channel_fuse_limit", "all"))
    route_family_mode = str(params.get("route_family_mode", "default") or "default")
    if ablation_name and ablation_name != "main":
        parts.append(ablation_name)
    if repr_encoder.lower() in {"none", "statsnone"}:
        parts.append("remove encoder")
    elif repr_encoder and repr_encoder not in {"StatsRandomFourier"}:
        parts.append(f"encoder={repr_encoder}")
    if qc_mode.lower() == "off":
        parts.append("QC")
    if repr_size == "1500":
        parts.append("reduced profile set, 50%")
    fusion_removed = []
    if channel_fuse == "1":
        fusion_removed.append("channel-level")
    if route_family_mode != "default":
        parts.append(f"route family={route_family_mode}")
    if task_n == "1":
        fusion_removed.append("window-level")
    if repr_v == "4":
        fusion_removed.append("representation-level")
    if {"channel-level", "window-level", "representation-level"}.issubset(set(fusion_removed)):
        parts.append("remove all rank-fusion")
    else:
        for item in fusion_removed:
            parts.append(f"remove {item} fusion")
    return "; ".join(parts) if parts else "main configuration"

def _vldb_table2_ablation_grid() -> dict:
    return {
        "ablation_name": [
            "main",
            # "sample-ratio-0.3",
            # "fallback-0.7",
        ],
        "repr_encoder": [
            "StatsRandomFourier",
            # "TrainFourier",
        ],
        "repr_input_dim": [512],
        "repr_output_dim": [256],
        "repr_sub_pred_len": [480],
        "zoo_repr_set": ["c-e-n-h-w-s"],
        "repr_size": [3000],
        "repr_v": [5],
        "base_metrics": ["C"],
        "repr_weight_ratio": [0],
        "sample_repr_num": [20],
        # "repr_data_seed": [2025, 2026, 2027, 2028 ,2029],
        "repr_data_seed": [2029],
        # "repr_encoder_seed": [2025, 2026, 2027, 2028 ,2029],
        "repr_encoder_seed": [2025],
        "repr_scale_protocol": ["std"],
        "repr_sample_qc_mode": ['strict'],
        # Edit these lists to expand TSRouter ablations.
        # "task_window_sample_strategy": ["legacy", "even", "first", "last"],
        # "sample_repr_ratio": [0, 0.1, 0.3, 0.5],
        # "task_rank_top3_instability_threshold": [0, 0.5, 0.7, 0.9],
        "task_channel_fuse_limit": ["all"],
        "route_family_mode": ["default"],
    }


def _vldb_summarize_row(
    method: str,
    df: pd.DataFrame,
    expected: set[str],
    best_df: pd.DataFrame,
    rank_base: str,
    baseline_df: pd.DataFrame | None = None,
) -> dict:
    df_ref = _vldb_add_quality_refs(df, best_df, rank_base)
    done = set(df_ref["dataset"].dropna().astype(str)) if "dataset" in df_ref.columns else set()
    row = {
        "Method": method,
        "Method_Type": "",
        "Files": int(df_ref["source_file"].nunique()) if "source_file" in df_ref.columns and not df_ref.empty else 0,
        "Qualified_DS": int(len(done & expected)),
        "Expected_DS": int(len(expected)),
        "Complete": int(len(done & expected)) == int(len(expected)),
        "Mean_MASE": _vldb_mean(df_ref, "MASE"),
        "Rank": _vldb_rank_value_against_tsfm(df_ref, baseline_df, rank_base) if baseline_df is not None else np.nan,
        "win-rate-MASE": _vldb_bool_mean(df_ref, "win_MASE"),
        "Regret_MASE": _vldb_mean(df_ref, "regret_MASE"),
        "Regret_MASE_P90": _vldb_p90(df_ref.get("regret_MASE", pd.Series(dtype=float))),
        "Mean_sMAPE": _vldb_mean(df_ref, "sMAPE"),
        "win-rate-sMAPE": _vldb_bool_mean(df_ref, "win_sMAPE"),
        "Mean_CRPS": _vldb_mean(df_ref, "CRPS"),
        "Top-1 hit": _vldb_bool_mean(df_ref, f"Top1Hit_{rank_base}"),
        "Top-3 hit": _vldb_bool_mean(df_ref, f"Top3Hit_{rank_base}"),
        **_vldb_selected_model_count_columns(method, df_ref),
    }
    return row


def _vldb_selector_quality_summary_row(path: Path, current_zoo_num: int) -> pd.Series | None:
    summary_path = Path("results_csv") / "TSRouter" / "vldb" / "tables" / "selector_quality_summary.csv"
    if not summary_path.exists():
        return None
    try:
        df = pd.read_csv(summary_path)
    except Exception:
        return None
    if df.empty or "source_csv" not in df.columns:
        return None
    if "stage" in df.columns:
        df = df[pd.to_numeric(df["stage"], errors="coerce").eq(int(current_zoo_num))]
    target = str(path).replace("\\", "/")
    source = df["source_csv"].astype(str).str.replace("\\", "/", regex=False)
    matched = df[source.eq(target)].copy()
    if matched.empty:
        target_name = Path(target).name
        target_key = re.sub(rf"^zoo{int(current_zoo_num)}-\d+_", f"zoo{int(current_zoo_num)}-*_", target_name)
        source_key = source.map(
            lambda x: re.sub(rf"^zoo{int(current_zoo_num)}-\d+_", f"zoo{int(current_zoo_num)}-*_", Path(str(x)).name)
        )
        matched = df[source_key.eq(target_key)].copy()
    if matched.empty:
        return None
    return matched.iloc[-1]


def _vldb_row_from_selector_quality_summary(
    label: str,
    summary: pd.Series | None,
    expected: set[str],
    rank_base: str,
) -> dict | None:
    if summary is None:
        return None

    def pick(*cols, default=np.nan):
        for col in cols:
            if col in summary.index and pd.notna(summary[col]):
                return summary[col]
        return default

    n = int(pd.to_numeric(pd.Series([pick("n_datasets", default=0)]), errors="coerce").fillna(0).iloc[0])
    metric = str(rank_base)
    return {
        "Method": label,
        "Files": 1,
        "Qualified_DS": n,
        "Expected_DS": int(len(expected)),
        "Complete": n == int(len(expected)),
        "Mean_MASE": pick("Mean_MASE", "MASE"),
        "win-rate-MASE": pick("Mean_Win_vs_Best_TSFM_MASE", "Win_vs_Best_TSFM_MASE"),
        "Regret_MASE": pick("Mean_Regret_MASE", "Regret_MASE"),
        "Regret_MASE_P90": pick("P90_Regret_MASE", "Regret_MASE_P90"),
        "Mean_sMAPE": pick("Mean_sMAPE", "sMAPE"),
        "win-rate-sMAPE": pick("Mean_Win_vs_Best_TSFM_sMAPE", "Win_vs_Best_TSFM_sMAPE"),
        "Mean_CRPS": pick("Mean_CRPS", "CRPS"),
        "Top-1 hit": pick(f"Mean_Win_vs_Best_TSFM_{metric}", f"Win_vs_Best_TSFM_{metric}", "Mean_Win_vs_Best_TSFM_MASE", "Win_vs_Best_TSFM_MASE"),
        "Top-3 hit": pick("Mean_SINGLE_TOP3_HIT", "SINGLE_TOP3_HIT", f"Mean_Win_vs_Best_TSFM_{metric}", f"Win_vs_Best_TSFM_{metric}"),
    }


def _vldb_value_from_selector_quality_summary(summary: pd.Series | None, value_kind: str, metric: str) -> tuple[float, int]:
    if summary is None:
        return np.nan, 0

    def pick(*cols):
        for col in cols:
            if col in summary.index and pd.notna(summary[col]):
                val = pd.to_numeric(pd.Series([summary[col]]), errors="coerce").iloc[0]
                if pd.notna(val):
                    return float(val)
        return np.nan

    n = int(pd.to_numeric(pd.Series([summary.get("n_datasets", 0)]), errors="coerce").fillna(0).iloc[0])
    if value_kind == "Rank":
        value = pick(f"Mean_Rank_by_{metric}", "Rank")
    elif value_kind == "Top1-hit":
        value = pick(f"Mean_Win_vs_Best_TSFM_{metric}", f"Win_vs_Best_TSFM_{metric}", "Mean_Win_vs_Best_TSFM_MASE", "Win_vs_Best_TSFM_MASE")
    elif value_kind == "Top3-hit":
        value = pick("Mean_SINGLE_TOP3_HIT", "SINGLE_TOP3_HIT", f"Mean_Win_vs_Best_TSFM_{metric}", f"Win_vs_Best_TSFM_{metric}")
    else:
        value = pick(f"Mean_{metric}", metric)
    return value, n


def run_vldb_table1(args, baseline_df_all: pd.DataFrame, ordered_model_names: list[str], results_dir: Path, season_naive_df=None):
    param_grid_vldb_main = _vldb_table1_main_grid()
    current_zoo_num = args.zoo_total_num
    current_model_names = ordered_model_names[:current_zoo_num]
    expected = _vldb_expected_datasets(args)
    baseline_df = baseline_df_all[baseline_df_all["model"].isin(current_model_names)].copy()
    if getattr(args, "GE_released", False):
        baseline_df = normalize_by_season_naive(baseline_df, season_naive_df)
    if args.quick_test:
        baseline_df = baseline_df[baseline_df["dataset"].isin(expected)].copy()
    best_df = _vldb_best_tsfm_by_dataset(baseline_df, args.rank_base)

    print("\n" + "=" * 88)
    print("VLDB Table1: Fixed-Zoo Main Table")
    print("=" * 88)
    print(f"[config] param_grid_vldb_main combinations={np.prod([len(v) for v in param_grid_vldb_main.values()])}")
    print(f"[config] current_zoo_num={current_zoo_num}, expected_ds={len(expected)}, rank_base={args.rank_base}")

    rows = []
    for model_name in current_model_names:
        model_df = baseline_df[baseline_df["model"] == model_name].copy()
        model_df["source_file"] = resolve_tsfm_csv_path(
            model_name,
            args.TSFM_results_dir,
            "all_results.csv",
        ).as_posix()
        row = _vldb_summarize_row(model_name, model_df, expected, best_df, args.rank_base, baseline_df=baseline_df)
        row["Method_Type"] = "TSFM"
        row["_tsfm_order"] = len(rows)
        row.update(_vldb_tsfm_e2e_efficiency(model_name, expected, tsfm_results_dir=args.TSFM_results_dir))
        abbr_to_id, _ = _model_id_maps()
        if "Best_TSFM_MASE_model_id" in best_df.columns:
            tmp_cols = ["dataset", "Best_TSFM_MASE_model_id"]
            if "Best_TSFM_MASE_top3_model_ids" in best_df.columns:
                tmp_cols.append("Best_TSFM_MASE_top3_model_ids")
            tmp = model_df.merge(best_df[tmp_cols], on="dataset", how="left")
            model_id = abbr_to_id.get(model_name, -999)
            row["Top-1 hit"] = float((pd.to_numeric(tmp["Best_TSFM_MASE_model_id"], errors="coerce") == model_id).mean())
            if "Best_TSFM_MASE_top3_model_ids" in tmp.columns:
                row["Top-3 hit"] = float(tmp["Best_TSFM_MASE_top3_model_ids"].map(lambda ids: model_id in _vldb_top3_ids(ids)).mean())
            else:
                row["Top-3 hit"] = row["Top-1 hit"]
        rows.append(row)
        print(f"[TSFM] {model_name}: qualified={row['Qualified_DS']}/{row['Expected_DS']}, MASE={row['Mean_MASE']:.4f}")

    tsrouter_by_method: dict[str, dict[str, list]] = {}
    for label, path, params in _vldb_grid_tsrouter_tasks(args, current_zoo_num, param_grid_vldb_main):
        display_method = "TSRouter-fast" if _vldb_truthy(params.get("route_efficiency_mode", False)) else "TSRouter"
        bucket = tsrouter_by_method.setdefault(display_method, {"frames": [], "route_ids": [], "route_id_candidates": []})
        route_id = _vldb_route_id_for_config(current_zoo_num, label, params)
        print(f"\n[{display_method}] {label}")
        print(f"  params={params}")
        print(f"  route_id={route_id}")
        print(f"  file={path}")
        if not path.exists():
            summary = _vldb_selector_quality_summary_row(path, current_zoo_num)
            if summary is None:
                print("  missing file")
                continue
            row_from_summary = _vldb_row_from_selector_quality_summary(label, summary, expected, args.rank_base)
            if row_from_summary is not None:
                raw = pd.DataFrame([{
                    "dataset": f"__summary__{current_zoo_num}",
                    args.rank_base: row_from_summary.get(f"Mean_{args.rank_base}", row_from_summary.get("Mean_MASE", np.nan)),
                    "source_file": str(summary.get("source_csv", path)),
                    "_vldb_summary_row": row_from_summary,
                }])
                print(f"  using selector_quality_summary fallback: N={row_from_summary['Qualified_DS']}/{row_from_summary['Expected_DS']}")
                bucket["frames"].append(raw)
        else:
            raw = check_results_file(path, args.verbose, args.quick_test)
            if raw is None or raw.empty:
                print("  empty or invalid file")
                continue
            raw = harmonize_metrics_schema(raw)
            raw["model"] = display_method
            raw["source_file"] = str(path)
            if args.quick_test:
                raw = raw[raw["dataset"].isin(expected)].copy()
            done = set(raw["dataset"].dropna().astype(str))
            missing = sorted(expected - done)
            print(f"  rows={len(raw)}, valid_expected={len(done & expected)}/{len(expected)}")
            if missing:
                print(f"  missing_head={missing[:8]}")
            bucket["frames"].append(raw)
        bucket["route_ids"].append(route_id)
        bucket["route_id_candidates"].append(_vldb_route_id_candidates_for_config(current_zoo_num, label, params))

    for display_method in ["TSRouter", "TSRouter-fast"]:
        bucket = tsrouter_by_method.get(display_method, {"frames": [], "route_ids": [], "route_id_candidates": []})
        frames = bucket.get("frames", [])
        route_candidates = bucket.get("route_id_candidates", [])
        if frames:
            zc_df = pd.concat(frames, ignore_index=True)
            summary_rows = [r for r in zc_df.get("_vldb_summary_row", pd.Series(dtype=object)).dropna().tolist() if isinstance(r, dict)]
            if summary_rows:
                row = dict(summary_rows[0])
                row["Method"] = display_method
            else:
                zc_df = zc_df.sort_values(args.rank_base, ascending=True).drop_duplicates(subset=["dataset"], keep="first")
                row = _vldb_summarize_row(display_method, zc_df, expected, best_df, args.rank_base, baseline_df=baseline_df)
            row["Method_Type"] = "task-adaptive"
            route_id = route_candidates[0] if len(route_candidates) == 1 else None
            eff = _vldb_route_efficiency(
                results_dir,
                display_method,
                current_zoo_num,
                route_id=route_id,
                expected_n=len(expected),
            )
            row.update(eff)
            rows.append(row)
            print(
                f"\n[{display_method}-selected] qualified={row['Qualified_DS']}/{row['Expected_DS']}, "
                f"files={row['Files']}, efficiency={eff.get('_efficiency_diag')}, skipped={row.get('Skipped_parts') or 'none/unknown'}"
            )
        else:
            rows.append({
                "Method": display_method,
                "Method_Type": "task-adaptive",
                "Files": 0,
                "Qualified_DS": 0,
                "Expected_DS": len(expected),
                "Complete": False,
                "Mean_MASE": np.nan,
                "win-rate-MASE": np.nan,
                "Top-1 hit": np.nan,
                "Top-3 hit": np.nan,
                "Mean_sMAPE": np.nan,
                "win-rate-sMAPE": np.nan,
                "Mean_CRPS": np.nan,
                **_vldb_route_efficiency(results_dir, display_method, current_zoo_num),
            })

    loaded_generated_labels = set()
    print("\n[VLDB generated baselines]")
    for config in _vldb_generated_baseline_configs(current_zoo_num, args.zoo_total_num, getattr(args, "sample_repr_num", 20)):
        label = str(config["label"])
        raw = _vldb_load_generated_baseline(config, expected, args)
        print(f"  {label}: glob={config['glob']}")
        if raw is None or raw.empty:
            print("    missing generated file")
            if label.startswith("Task-Probe Forward"):
                rows.append(_vldb_missing_generated_baseline_row(config, expected))
            continue
        row = _vldb_summarize_row(label, raw, expected, best_df, args.rank_base, baseline_df=baseline_df)
        row["Method_Type"] = str(config.get("method_type", "baseline"))
        if bool(config.get("oracle", False)):
            if label == "Zoo-Oracle Static Best":
                row.update(_vldb_full_zoo_forward_efficiency(current_zoo_num, expected, tsfm_results_dir=getattr(args, "TSFM_results_dir", "cl_512")))
            else:
                row.update(_vldb_oracle_efficiency_na())
        else:
            eff = _vldb_route_efficiency(
                results_dir,
                str(config.get("method", label)),
                current_zoo_num,
                route_id=config.get("route_id"),
                expected_n=len(expected),
            )
            row.update(eff)
        loaded_generated_labels.add(label)
        rows.append(row)
        print(
            f"    qualified={row['Qualified_DS']}/{row['Expected_DS']}, "
            f"MASE={row['Mean_MASE']:.4f}, efficiency={row.get('_efficiency_diag', '')}"
        )

    table = pd.DataFrame(_vldb_sort_table1_rows(rows))
    table = _vldb_display_table(table, "table1")
    print("\n" + "=" * 88)
    print("VLDB Table1 Final")
    print("=" * 88)
    print(tabulate(table, headers="keys", tablefmt="plain", floatfmt=".4f", numalign="decimal", stralign="left", showindex=False))
    return table


def _vldb_mean_ms_triplet(values: pd.Series) -> str:
    parsed = [_vldb_parse_ms_triplet(v) for v in values.dropna().tolist() if str(v).strip()]
    if not parsed:
        return "nan/nan/nan"
    arr = np.asarray(parsed, dtype=float)
    means = []
    for idx in range(3):
        col = arr[:, idx]
        col = col[np.isfinite(col)]
        means.append(float(col.mean()) if col.size else np.nan)
    return _vldb_format_ms_triplet(means)


def _vldb_aggregate_table1_ablation_over_repr_seed(
    table: pd.DataFrame,
    param_keys: list[str],
) -> tuple[pd.DataFrame, list[str], bool]:
    if table is None or table.empty or "repr_data_seed" not in table.columns:
        return table, param_keys, False
    seeds = sorted({str(v) for v in table["repr_data_seed"].dropna().astype(str).unique() if str(v) != ""})
    if len(seeds) <= 1:
        return table, param_keys, False

    group_keys = [
        key
        for key in ["Config", "Ablation", *param_keys]
        if key in table.columns and key != "repr_data_seed"
    ]
    if not group_keys:
        return table, param_keys, False

    numeric_mean_cols = [
        "Mean_MASE",
        "Rank",
        "win-rate-MASE",
        "Regret_MASE",
        "Regret_MASE_P90",
        "Mean_sMAPE",
        "win-rate-sMAPE",
        "Mean_CRPS",
        "Top-1 hit",
        "Top-3 hit",
        "Route_Throughput",
        "Efficiency_valid_DS",
    ]
    triplet_cols = ["ROUTE_ms_P50_P90_P95", "E2E_ms_P50_P90_P95"]
    rows = []
    for _, group in table.groupby(group_keys, dropna=False, sort=False):
        out = {key: group.iloc[0].get(key, "") for key in group_keys}
        unique_seeds = sorted({
            str(v)
            for v in group.get("repr_data_seed", pd.Series(dtype=object)).dropna().astype(str).unique()
            if str(v) != ""
        })
        out["Seed_N"] = int(len(unique_seeds))
        out["ReprSeeds"] = ",".join(unique_seeds)
        out["Files"] = int(pd.to_numeric(group.get("Files", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        out["Qualified_DS"] = float(pd.to_numeric(group.get("Qualified_DS", pd.Series(dtype=float)), errors="coerce").mean())
        expected_vals = pd.to_numeric(group.get("Expected_DS", pd.Series(dtype=float)), errors="coerce").dropna()
        out["Expected_DS"] = int(expected_vals.iloc[0]) if not expected_vals.empty else np.nan
        complete = group.get("Complete", pd.Series(dtype=object)).map(lambda v: str(v).lower() in {"true", "1"})
        out["Complete"] = bool(complete.all()) if len(complete) else False
        for col in numeric_mean_cols:
            if col in group.columns:
                out[col] = pd.to_numeric(group[col], errors="coerce").mean()
        for col in triplet_cols:
            if col in group.columns:
                out[col] = _vldb_mean_ms_triplet(group[col])
        for col in ["Skipped_parts", "_efficiency_diag"]:
            if col in group.columns:
                vals = [str(v) for v in group[col].dropna().astype(str).unique() if str(v)]
                out[col] = " | ".join(vals[:3])
        rows.append(out)

    next_param_keys = [key for key in param_keys if key != "repr_data_seed"]
    return pd.DataFrame(rows), next_param_keys, True


def run_vldb_table1_ablation(args, baseline_df_all: pd.DataFrame, ordered_model_names: list[str], results_dir: Path, season_naive_df=None):
    param_grid_vldb_ablation = _vldb_table1_ablation_grid()
    current_zoo_num = args.zoo_total_num
    current_model_names = ordered_model_names[:current_zoo_num]
    expected = _vldb_expected_datasets(args)
    baseline_df = baseline_df_all[baseline_df_all["model"].isin(current_model_names)].copy()
    if getattr(args, "GE_released", False):
        baseline_df = normalize_by_season_naive(baseline_df, season_naive_df)
    if args.quick_test:
        baseline_df = baseline_df[baseline_df["dataset"].isin(expected)].copy()
    best_df = _vldb_best_tsfm_by_dataset(baseline_df, args.rank_base)

    print("\n" + "=" * 88)
    print("VLDB Table1 TSRouter Ablation")
    print("=" * 88)
    print(f"[config] param_grid_vldb_ablation combinations={np.prod([len(v) for v in param_grid_vldb_ablation.values()])}")
    print(f"[config] current_zoo_num={current_zoo_num}, expected_ds={len(expected)}, rank_base={args.rank_base}")

    rows = []
    for label, path, params in _vldb_grid_tsrouter_tasks(args, current_zoo_num, param_grid_vldb_ablation):
        route_id = _vldb_route_id_for_config(current_zoo_num, label, params)
        print(f"\n[TSRouter-ablation] {label}")
        print(f"  params={params}")
        print(f"  route_id={route_id or 'stage aggregate'}")
        print(f"  file={path}")
        if not path.exists():
            eff = _vldb_route_efficiency(
                results_dir,
                "TSRouter",
                current_zoo_num,
                route_id=route_id,
                expected_n=len(expected),
            )
            summary = _vldb_selector_quality_summary_row(path, current_zoo_num)
            row = _vldb_row_from_selector_quality_summary(label, summary, expected, args.rank_base)
            if row is None:
                print("  missing file")
                continue
            else:
                print(f"  using selector_quality_summary fallback: N={row['Qualified_DS']}/{row['Expected_DS']}")
                row["Config"] = row.pop("Method")
            row.update(eff)
            print(f"  efficiency={eff.get('_efficiency_diag')}; skipped={eff.get('Skipped_parts')}")
        else:
            raw = check_results_file(path, args.verbose, args.quick_test)
            if raw is None or raw.empty:
                print("  empty or invalid file")
                eff = _vldb_route_efficiency(
                    results_dir,
                    "TSRouter",
                    current_zoo_num,
                    route_id=route_id,
                    expected_n=len(expected),
                )
                summary = _vldb_selector_quality_summary_row(path, current_zoo_num)
                row = _vldb_row_from_selector_quality_summary(label, summary, expected, args.rank_base)
                if row is None:
                    continue
                else:
                    row["Config"] = row.pop("Method")
                row.update(eff)
                print(f"  efficiency={eff.get('_efficiency_diag')}; skipped={eff.get('Skipped_parts')}")
            else:
                raw = harmonize_metrics_schema(raw)
                raw["model"] = label
                raw["source_file"] = str(path)
                if args.quick_test:
                    raw = raw[raw["dataset"].isin(expected)].copy()
                row = _vldb_summarize_row(label, raw, expected, best_df, args.rank_base)
                row["Config"] = row.pop("Method")
                eff = _vldb_route_efficiency(
                    results_dir,
                    "TSRouter",
                    current_zoo_num,
                    route_id=route_id,
                    expected_n=len(expected),
                )
                row.update(eff)
                done = set(raw["dataset"].dropna().astype(str))
                missing = sorted(expected - done)
                print(f"  rows={len(raw)}, valid_expected={len(done & expected)}/{len(expected)}")
                print(f"  efficiency={eff.get('_efficiency_diag')}; skipped={eff.get('Skipped_parts')}")
                if missing:
                    print(f"  missing_head={missing[:8]}")
        for key in [
            "repr_encoder",
            "zoo_repr_set",
            "repr_size",
            "repr_v",
            "base_metrics",
            "sample_repr_num",
            "repr_data_seed",
            "repr_encoder_seed",
            "repr_sample_qc_mode",
            "repr_scale_protocol",
            "repr_anchor_window_sample_strategy",
            "task_window_sample_strategy",
            "sample_repr_ratio",
            "task_rank_top3_instability_threshold",
            "task_channel_fuse_limit",
            "route_family_mode",
        ]:
            if key in params:
                row[key] = params[key]
        row["Ablation"] = _vldb_ablation_description(params)
        rows.append(row)

    table = pd.DataFrame(rows)
    if not table.empty:
        result_mask = (
            pd.to_numeric(table.get("Files", pd.Series(index=table.index, dtype=float)), errors="coerce").fillna(0).gt(0)
            | pd.to_numeric(table.get("Qualified_DS", pd.Series(index=table.index, dtype=float)), errors="coerce").fillna(0).gt(0)
        )
        table = table[result_mask].copy()
    param_keys = [
        "repr_encoder",
        "zoo_repr_set",
        "repr_size",
        "repr_v",
        "base_metrics",
        "sample_repr_num",
        "repr_data_seed",
        "repr_encoder_seed",
        "repr_sample_qc_mode",
        "repr_scale_protocol",
        "repr_anchor_window_sample_strategy",
        "task_window_sample_strategy",
        "sample_repr_ratio",
        "task_rank_top3_instability_threshold",
        "task_channel_fuse_limit",
        "route_family_mode",
    ]
    table, display_param_keys, seed_aggregated = _vldb_aggregate_table1_ablation_over_repr_seed(table, param_keys)
    varying_source = table.to_dict("records") if isinstance(table, pd.DataFrame) and not table.empty else rows
    if seed_aggregated:
        print("[aggregate] VLDB Table1 ablation averaged over repr_data_seed while keeping other params fixed.")
    table = _vldb_display_table(table, "table1_ablation", _vldb_varying_param_cols(varying_source, display_param_keys))
    print("\n" + "=" * 88)
    print("VLDB Table1 TSRouter Ablation Final")
    print("=" * 88)
    print(tabulate(table, headers="keys", tablefmt="plain", floatfmt=".4f", numalign="decimal", stralign="left", showindex=False))
    return table


def _vldb_growing_zoo_stages(args) -> list[int]:
    start = max(3, int(getattr(args, "ensemble_size", 1)) + 1)
    end = int(getattr(args, "zoo_total_num", start))
    return list(range(start, end + 1))


def _vldb_quality_context_for_stage(
    args,
    current_zoo_num: int,
    baseline_df_all: pd.DataFrame,
    ordered_model_names: list[str],
    season_naive_df=None,
) -> tuple[set[str], pd.DataFrame, pd.DataFrame]:
    current_model_names = ordered_model_names[:current_zoo_num]
    expected = _vldb_expected_datasets(args)
    baseline_df = baseline_df_all[baseline_df_all["model"].isin(current_model_names)].copy()
    if getattr(args, "GE_released", False):
        baseline_df = normalize_by_season_naive(baseline_df, season_naive_df)
    if args.quick_test:
        baseline_df = baseline_df[baseline_df["dataset"].isin(expected)].copy()
    best_df = _vldb_best_tsfm_by_dataset(baseline_df, args.rank_base)
    return expected, best_df, baseline_df


def _vldb_rank_value_against_tsfm(raw: pd.DataFrame, baseline_df: pd.DataFrame, rank_base: str) -> float:
    if raw is None or raw.empty or baseline_df is None or baseline_df.empty:
        return np.nan
    if "dataset" not in raw.columns or rank_base not in raw.columns:
        return np.nan
    base = baseline_df[["dataset", "model", rank_base]].copy()
    base[rank_base] = pd.to_numeric(base[rank_base], errors="coerce")
    zc = raw[["dataset", rank_base]].copy()
    zc[rank_base] = pd.to_numeric(zc[rank_base], errors="coerce")
    zc["model"] = "__TSRouter__"
    all_df = pd.concat([base, zc], ignore_index=True)
    all_df = all_df.dropna(subset=["dataset", "model", rank_base])
    if all_df.empty:
        return np.nan
    all_df["Rank"] = all_df.groupby("dataset")[rank_base].rank(method="min", ascending=True)
    vals = all_df[all_df["model"] == "__TSRouter__"]["Rank"].replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.mean()) if not vals.empty else np.nan


def _vldb_oracle_best_value(best_df: pd.DataFrame, value_kind: str) -> float:
    if value_kind == "MASE":
        return _vldb_mean(best_df, "Best_TSFM_MASE")
    if value_kind == "Rank":
        return 1.0
    if value_kind in {"Win-MASE", "Top1-hit", "Top3-hit"}:
        return 1.0
    return np.nan


def _vldb_current_best_fixed_tsfm(baseline_df: pd.DataFrame, best_df: pd.DataFrame, value_kind: str, rank_base: str) -> tuple[float, str]:
    if baseline_df is None or baseline_df.empty or "model" not in baseline_df.columns or "MASE" not in baseline_df.columns:
        return np.nan, ""
    df = baseline_df.copy()
    df["MASE"] = pd.to_numeric(df["MASE"], errors="coerce")
    model_means = df.groupby("model")["MASE"].mean().replace([np.inf, -np.inf], np.nan).dropna()
    if model_means.empty:
        return np.nan, ""
    best_model = str(model_means.sort_values(ascending=True).index[0])
    best_model_df = df[df["model"].astype(str).eq(best_model)].copy()
    if value_kind == "MASE":
        return float(model_means.loc[best_model]), best_model
    if value_kind == "Rank":
        if rank_base not in df.columns:
            return np.nan, best_model
        rank_df = df[["dataset", "model", rank_base]].copy()
        rank_df[rank_base] = pd.to_numeric(rank_df[rank_base], errors="coerce")
        rank_df = rank_df.dropna(subset=["dataset", "model", rank_base])
        if rank_df.empty:
            return np.nan, best_model
        rank_df["Rank"] = rank_df.groupby("dataset")[rank_base].rank(method="min", ascending=True)
        vals = rank_df[rank_df["model"].astype(str).eq(best_model)]["Rank"].dropna()
        return (float(vals.mean()) if not vals.empty else np.nan), best_model
    if value_kind == "Win-MASE":
        if best_df is None or best_df.empty or "Best_TSFM_MASE" not in best_df.columns:
            return np.nan, best_model
        tmp = best_model_df[["dataset", "MASE"]].merge(best_df[["dataset", "Best_TSFM_MASE"]], on="dataset", how="left")
        wins = pd.to_numeric(tmp["MASE"], errors="coerce") <= pd.to_numeric(tmp["Best_TSFM_MASE"], errors="coerce") * (1.0 + 1e-3)
        return _vldb_bool_mean(pd.DataFrame({"win": wins}), "win"), best_model
    if value_kind in {"Top1-hit", "Top3-hit"}:
        best_col = f"Best_TSFM_{rank_base}_model_id"
        if best_df is None or best_df.empty or best_col not in best_df.columns:
            return np.nan, best_model
        abbr_to_id, _ = _model_id_maps()
        best_id = abbr_to_id.get(best_model, np.nan)
        tmp_cols = ["dataset", best_col]
        top3_col = f"Best_TSFM_{rank_base}_top3_model_ids"
        if value_kind == "Top3-hit" and top3_col in best_df.columns:
            tmp_cols.append(top3_col)
        tmp = best_model_df[["dataset"]].merge(best_df[tmp_cols], on="dataset", how="left")
        if value_kind == "Top3-hit" and top3_col in tmp.columns:
            hits = tmp[top3_col].map(lambda ids: best_id in _vldb_top3_ids(ids))
        else:
            hits = pd.to_numeric(tmp[best_col], errors="coerce").eq(best_id)
        return _vldb_bool_mean(pd.DataFrame({"hit": hits}), "hit"), best_model
    return np.nan, best_model


def _vldb_value_better_or_equal(value: float, current_best: float, value_kind: str) -> bool:
    if not np.isfinite(value) or not np.isfinite(current_best):
        return False
    if value_kind in {"MASE", "Rank"}:
        return value <= current_best * (1.0 + 1e-3)
    if value_kind in {"Win-MASE", "Top1-hit", "Top3-hit"}:
        return value >= current_best - 1e-3
    return False


def _vldb_table2_one_grid(
    args,
    baseline_df_all: pd.DataFrame,
    ordered_model_names: list[str],
    results_dir: Path,
    param_grid: dict,
    title: str,
    table_key: str,
    value_kind: str,
    season_naive_df=None,
):
    stages = _vldb_growing_zoo_stages(args)
    param_keys = [k for k in param_grid.keys() if not k.startswith("_") and k not in {"ablation_name", "vldb_route_id"}]
    metric = args.rank_base

    print("\n" + "=" * 88)
    print(f"{title} ({value_kind})")
    print("=" * 88)
    print(f"[config] combinations={np.prod([len(v) for v in param_grid.values()])}, stages={stages}, value={value_kind}")

    rows = {}
    oracle_best_row = {"Config": "Oracle_Best", "Complete-Zoo": 0, "Win-Zoo": "ref", "Support": ""}
    current_best_row = {"Config": "Current_Best", "Complete-Zoo": 0, "Win-Zoo": "ref", "Support": ""}
    for current_zoo_num in stages:
        expected, best_df, baseline_df = _vldb_quality_context_for_stage(
            args=args,
            current_zoo_num=current_zoo_num,
            baseline_df_all=baseline_df_all,
            ordered_model_names=ordered_model_names,
            season_naive_df=season_naive_df,
        )
        zcol = f"z{current_zoo_num}-{args.zoo_total_num}"
        oracle_best = _vldb_oracle_best_value(best_df, value_kind)
        current_best, current_best_model = _vldb_current_best_fixed_tsfm(baseline_df, best_df, value_kind, metric)
        support_n = int(len(set(best_df["dataset"].dropna().astype(str)) & expected)) if "dataset" in best_df.columns else 0
        oracle_best_row[zcol] = oracle_best
        oracle_best_row[f"{zcol}_N"] = support_n
        current_best_row[zcol] = current_best
        current_best_row[f"{zcol}_N"] = support_n
        current_best_row[f"{zcol}_model"] = current_best_model
        if support_n == len(expected):
            oracle_best_row["Complete-Zoo"] += 1
            current_best_row["Complete-Zoo"] += 1
        for label, path, params in _vldb_grid_tsrouter_tasks(args, current_zoo_num, param_grid):
            row_key = _vldb_route_suffix_for_args(args, params)
            display_label = "TSRouter" if table_key == "table2" and label == "main" else label
            row = rows.setdefault(
                row_key,
                {
                    "Config": display_label,
                    "Complete-Zoo": 0,
                    "Win-Zoo": 0,
                    "Support": "",
                },
            )
            for key in param_keys:
                if key in params:
                    row[key] = params[key]

            print(f"[Growing-Zoo] stage={current_zoo_num}, config={label}")
            print(f"  route_id={_vldb_route_id_for_config(current_zoo_num, label, params)}")
            print(f"  file={path}")
            if not path.exists():
                summary = _vldb_selector_quality_summary_row(path, current_zoo_num)
                value, support_n = _vldb_value_from_selector_quality_summary(summary, value_kind, metric)
                row[zcol] = value
                row[f"{zcol}_N"] = support_n
                if support_n:
                    if support_n == len(expected):
                        row["Complete-Zoo"] += 1
                    if _vldb_value_better_or_equal(value, current_best, value_kind):
                        row["Win-Zoo"] += 1
                    print(f"  using selector_quality_summary fallback: qualified={support_n}/{len(expected)}, {value_kind}={value}")
                else:
                    print("  missing file")
                continue
            raw = check_results_file(path, args.verbose, args.quick_test)
            if raw is None or raw.empty:
                summary = _vldb_selector_quality_summary_row(path, current_zoo_num)
                value, support_n = _vldb_value_from_selector_quality_summary(summary, value_kind, metric)
                row[zcol] = value
                row[f"{zcol}_N"] = support_n
                if support_n:
                    if support_n == len(expected):
                        row["Complete-Zoo"] += 1
                    if _vldb_value_better_or_equal(value, current_best, value_kind):
                        row["Win-Zoo"] += 1
                    print(f"  using selector_quality_summary fallback: qualified={support_n}/{len(expected)}, {value_kind}={value}")
                else:
                    print("  empty or invalid file")
                continue
            raw = harmonize_metrics_schema(raw)
            raw["model"] = label
            raw["source_file"] = str(path)
            if args.quick_test:
                raw = raw[raw["dataset"].isin(expected)].copy()
            summary = _vldb_summarize_row(label, raw, expected, best_df, metric)
            if value_kind == "Rank":
                value = _vldb_rank_value_against_tsfm(raw, baseline_df, metric)
            elif value_kind == "Win-MASE":
                value = summary.get("win-rate-MASE", np.nan)
            elif value_kind == "Top1-hit":
                value = summary.get("Top-1 hit", np.nan)
            elif value_kind == "Top3-hit":
                value = summary.get("Top-3 hit", np.nan)
            else:
                value = summary.get(f"Mean_{metric}", np.nan)
            row[zcol] = value
            row[f"{zcol}_N"] = summary["Qualified_DS"]
            if summary["Complete"]:
                row["Complete-Zoo"] += 1
            if _vldb_value_better_or_equal(value, current_best, value_kind):
                row["Win-Zoo"] += 1
            print(
                f"  qualified={summary['Qualified_DS']}/{summary['Expected_DS']}, "
                f"{value_kind}={value}, Current_Best={current_best} ({current_best_model}), Oracle_Best={oracle_best}"
            )

        for config in _vldb_generated_baseline_configs(
            current_zoo_num,
            args.zoo_total_num,
            getattr(args, "sample_repr_num", 20),
        ):
            label = str(config["label"])
            if table_key.startswith("table2") and label == "Zoo-Oracle Static Best":
                continue
            row_key = f"baseline::{label}"
            row = rows.setdefault(
                row_key,
                {
                    "Config": label,
                    "Complete-Zoo": 0,
                    "Win-Zoo": 0,
                    "Support": "",
                    "Type": config.get("method_type", ""),
                },
            )
            raw = _vldb_load_generated_baseline(config, expected, args)
            print(f"[Growing-Zoo Baseline] stage={current_zoo_num}, method={label}",end=" ")
            if raw.empty:
                row[zcol] = np.nan
                row[f"{zcol}_N"] = 0
                print("  missing baseline artifact")
                continue
            summary = _vldb_summarize_row(label, raw, expected, best_df, metric)
            if value_kind == "Rank":
                value = _vldb_rank_value_against_tsfm(raw, baseline_df, metric)
            elif value_kind == "Win-MASE":
                value = summary.get("win-rate-MASE", np.nan)
            elif value_kind == "Top1-hit":
                value = summary.get("Top-1 hit", np.nan)
            elif value_kind == "Top3-hit":
                value = summary.get("Top-3 hit", np.nan)
            else:
                value = summary.get(f"Mean_{metric}", np.nan)
            row[zcol] = value
            row[f"{zcol}_N"] = summary["Qualified_DS"]
            if summary["Complete"]:
                row["Complete-Zoo"] += 1
            if _vldb_value_better_or_equal(value, current_best, value_kind):
                row["Win-Zoo"] += 1
            print(
                f"  qualified={summary['Qualified_DS']}/{summary['Expected_DS']}, "
                f"{value_kind}={value}, source={raw.get('source_file', pd.Series([''])).iloc[-1]}"
            )

    table = pd.DataFrame(rows.values())
    if table_key == "table2":
        rows_with_current = [oracle_best_row, current_best_row] + list(rows.values())
        table = pd.DataFrame(rows_with_current)
    if table.empty:
        print("(empty table)")
        return table
    zcols = [f"z{stage}-{args.zoo_total_num}" for stage in stages]
    ncols = [f"{zcol}_N" for zcol in zcols if f"{zcol}_N" in table.columns]
    if table_key.endswith("_ablation") and ncols:
        support_sum = table[ncols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        table = table[support_sum.gt(0)].copy()
        if table.empty:
            print("(empty table after dropping ablation configs with no result support)")
            return table
    table["Support"] = table[ncols].apply(
        lambda r: ",".join(f"{c[:-2]}:{int(v)}" for c, v in r.items() if pd.notna(v)),
        axis=1,
    ) if ncols else ""
    varying_source = list(rows.values())
    varying = _vldb_varying_param_cols(varying_source, param_keys)
    include_config = not (table_key.endswith("_ablation") and "Config" in table.columns and table["Config"].dropna().astype(str).nunique() <= 1)
    base_cols = (["Config"] if include_config else []) + varying + zcols + ["Win-Zoo", "Complete-Zoo", "Support"]
    table = table[[c for c in base_cols if c in table.columns]].copy()
    rename = {c: VLDB_PARAM_COLUMN_ALIASES.get(c, c) for c in varying}
    if table_key == "table2":
        table = table.rename(columns={"Config": "Method", **rename})
    else:
        table = table.rename(columns=rename)
    print("\n" + "=" * 88)
    print(f"{title} ({value_kind}) Final")
    print("=" * 88)
    print(tabulate(table, headers="keys", tablefmt="plain", floatfmt=".4f", numalign="decimal", stralign="left", showindex=False))
    return table


def run_vldb_table2(args, baseline_df_all: pd.DataFrame, ordered_model_names: list[str], results_dir: Path, season_naive_df=None):
    tables = []
    for value_kind in ["MASE", "Rank", "Top1-hit", "Top3-hit"]:
        table = _vldb_table2_one_grid(
            args=args,
            baseline_df_all=baseline_df_all,
            ordered_model_names=ordered_model_names,
            results_dir=results_dir,
            param_grid=_vldb_table1_main_grid(),
            title="VLDB Table2 Growing-Zoo TSRouter Main",
            table_key="table2",
            value_kind=value_kind,
            season_naive_df=season_naive_df,
        )
        tables.append((f"VLDB Table2 Growing-Zoo TSRouter Main ({value_kind}) Final", table))
    return tables


def run_vldb_table2_ablation(args, baseline_df_all: pd.DataFrame, ordered_model_names: list[str], results_dir: Path, season_naive_df=None):
    tables = []
    for value_kind in ["MASE", "Rank", "Top1-hit", "Top3-hit"]:
        table = _vldb_table2_one_grid(
            args=args,
            baseline_df_all=baseline_df_all,
            ordered_model_names=ordered_model_names,
            results_dir=results_dir,
            param_grid=_vldb_table2_ablation_grid(),
            title="VLDB Table2 Growing-Zoo TSRouter Ablation",
            table_key="table2_ablation",
            value_kind=value_kind,
            season_naive_df=season_naive_df,
        )
        tables.append((f"VLDB Table2 Growing-Zoo TSRouter Ablation ({value_kind}) Final", table))
    return tables


def _vldb_route_dataset_rows_for_ids(current_zoo_num: int, route_ids: list[str]) -> tuple[pd.DataFrame, str, str]:
    log_path = Path("results_csv") / "TSRouter" / "vldb" / "logs" / "route_latency_log.csv"
    if not log_path.exists():
        return pd.DataFrame(), "", "missing route_latency_log"
    try:
        log_df = pd.read_csv(log_path)
    except Exception as e:
        return pd.DataFrame(), "", f"cannot read route_latency_log: {e}"
    df = log_df[log_df.get("timing_level", "").astype(str).eq("selector_dataset_internal")].copy()
    if "stage" in df.columns:
        df = df[pd.to_numeric(df["stage"], errors="coerce").eq(int(current_zoo_num))]
    if "method" in df.columns:
        df = df[df["method"].astype(str).eq("TSRouter")]
    for route_id in route_ids:
        sub = df[df.get("route_id", "").astype(str).eq(str(route_id))].copy()
        if not sub.empty:
            if "dataset" in sub.columns:
                sub = _vldb_latest_by_file_order(sub, "dataset")
            return sub, str(route_id), "OK"
    return pd.DataFrame(), route_ids[0] if route_ids else "", "no matched route rows"


def _vldb_valid_route_forward_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    route_ms = pd.to_numeric(out.get("route_overhead_ms", pd.Series(index=out.index, dtype=float)), errors="coerce")
    forward_ms = pd.to_numeric(out.get("selected_forecast_ms", pd.Series(index=out.index, dtype=float)), errors="coerce")
    forward_valid = _vldb_bool_col(out, "selected_forecast_timing_valid", True)
    proxy_forward = pd.Series(False, index=out.index)
    if "forward_mode" in out.columns:
        proxy_forward = out["forward_mode"].fillna("").astype(str).str.lower().str.contains("proxy")
        forward_valid = forward_valid & ~proxy_forward
    if {"selected_model_order", "dataset"}.issubset(out.columns):
        for idx, rec in out.iterrows():
            if pd.notna(forward_ms.get(idx)) and bool(forward_valid.get(idx, False)):
                continue
            model_key = _vldb_first_selected_model_key(rec.get("selected_model_order"))
            if not model_key:
                continue
            val = _vldb_tsfm_forward_ms(model_key, str(rec.get("dataset", "")))
            if np.isfinite(val):
                out.at[idx, "selected_forecast_ms"] = float(val)
                forward_ms.at[idx] = float(val)
                forward_valid.at[idx] = True
    valid_mask = (
        _vldb_bool_col(out, "task_sampling_timing_valid", True)
        & route_ms.notna()
    )
    out = out[valid_mask].copy()
    out["route_forward_e2e_ms"] = route_ms[valid_mask] + forward_ms[valid_mask]
    return out


def _vldb_main_route_id_candidates(args, current_zoo_num: int, route_efficiency_mode: bool = False) -> list[str]:
    tasks = _vldb_grid_tsrouter_tasks(args, current_zoo_num, _vldb_table1_main_grid())
    if not tasks:
        return []
    wanted = bool(route_efficiency_mode)
    filtered = [task for task in tasks if _vldb_truthy(task[2].get("route_efficiency_mode", False)) == wanted]
    label, _, params = (filtered[0] if filtered else tasks[0])
    return _vldb_route_id_candidates_for_config(current_zoo_num, label, params)


def _vldb_component_quantiles_ms_triplet(df: pd.DataFrame, col: str) -> str:
    vals = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce")
    return _vldb_format_ms_triplet([
        _vldb_quantile(vals, 0.50),
        _vldb_quantile(vals, 0.90),
        _vldb_quantile(vals, 0.95),
    ])


def run_vldb_table31_latency_breakdown(args, results_dir: Path):
    current_zoo_num = args.zoo_total_num
    route_ids = _vldb_main_route_id_candidates(args, current_zoo_num)
    rows, used_route_id, note = _vldb_route_dataset_rows_for_ids(current_zoo_num, route_ids)
    valid = _vldb_valid_route_forward_rows(rows)
    row = {
        "Method": "TSRouter",
        "Stage": current_zoo_num,
        "Eff-N": int(len(valid)),
        "Issue": "OK" if not valid.empty and len(valid) == len(_vldb_expected_datasets(args)) else note,
    }
    component_cols = [
        ("TaskEmb", "task_embedding_ms"),
        ("IndexLookup", "index_lookup_ms"),
        ("CandRank", "rank_ms"),
        ("SelectedForecast", "selected_forecast_ms"),
    ]
    for label, col in component_cols:
        row[f"{label} P50/90/95(ms)"] = _vldb_component_quantiles_ms_triplet(valid, col)
    row["CandForward P50/90/95(ms)"] = "0/0/0" if not valid.empty else "nan/nan/nan"
    table = pd.DataFrame([row])
    print("\n" + "=" * 88)
    print("VLDB Table3.1 Routing Latency Breakdown (TSRouter: main StatsRandomFourier strict std) Final")
    print("=" * 88)
    print(tabulate(table, headers="keys", tablefmt="plain", floatfmt=".4f", numalign="decimal", stralign="left", showindex=False))
    return table


def _vldb_insert_cost_rows_for_method(method: str, df: pd.DataFrame) -> list[dict]:
    metrics = [
        ("NewForward(s)", "new_forward_s"),
        ("ProfileMerge(s)", "profile_merge_s"),
        ("IndexRefresh(s)", "index_refresh_s"),
        ("TotalInsert(s)", None),
        ("OldForwards", "old_model_forwards"),
        ("Valid", "_valid_insert"),
    ]
    rows = []
    for metric_name, source_col in metrics:
        row = {"Method": method, "Metric": metric_name}
        for _, rec in df.iterrows():
            stage_val = pd.to_numeric(pd.Series([rec.get("to_stage", np.nan)]), errors="coerce").iloc[0]
            col = f"stage{int(stage_val)}" if pd.notna(stage_val) else str(rec.get("new_stable_model_id", "stage?"))
            index_valid = bool(rec.get("_index_refresh_internal_valid", rec.get("_valid_insert", False)))
            if metric_name == "TotalInsert(s)":
                if not index_valid:
                    row[col] = np.nan
                    continue
                vals = [
                    pd.to_numeric(pd.Series([rec.get("new_forward_s")]), errors="coerce").iloc[0],
                    pd.to_numeric(pd.Series([rec.get("profile_merge_s")]), errors="coerce").iloc[0],
                    pd.to_numeric(pd.Series([rec.get("index_refresh_s")]), errors="coerce").iloc[0],
                ]
                row[col] = float(np.nansum(vals))
            elif metric_name == "IndexRefresh(s)" and not index_valid:
                row[col] = np.nan
            elif metric_name == "Valid":
                row[col] = bool(rec.get(source_col)) and index_valid
            else:
                row[col] = rec.get(source_col, np.nan)
        rows.append(row)
    return rows


def _vldb_profile_pool_forward_ms_for_model(model_key: str) -> float:
    pre_path = Path("results_csv") / "TSRouter" / "vldb" / "tables" / "vldb_efficiency_presummary.csv"
    if not pre_path.exists():
        return np.nan
    try:
        pre = pd.read_csv(pre_path, low_memory=False)
    except Exception:
        return np.nan
    if pre.empty or "source_type" not in pre.columns:
        return np.nan
    sub = pre[pre["source_type"].astype(str).eq("profile_pool_forward")].copy()
    if "model_key" in sub.columns and str(model_key):
        sub = sub[sub["model_key"].astype(str).eq(str(model_key))]
    vals = pd.to_numeric(sub.get("runtime_ms", pd.Series(dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.sum()) if not vals.empty else np.nan

def run_vldb_table32_insert_cost(args, results_dir: Path):
    log_path = Path("results_csv") / "TSRouter" / "vldb" / "logs" / "insert_log.csv"
    if not log_path.exists():
        table = pd.DataFrame([{"Metric": "Issue", "Value": "missing insert_log"}])
        print("\n" + "=" * 88)
        print("VLDB Table3.2 Insert Cost Breakdown Final")
        print("=" * 88)
        print(tabulate(table, headers="keys", tablefmt="plain", showindex=False))
        return table
    df = pd.read_csv(log_path)
    if df.empty:
        table = pd.DataFrame([{"Metric": "Issue", "Value": "empty insert_log"}])
        print("\n" + "=" * 88)
        print("VLDB Table3.2 Insert Cost Breakdown Final")
        print("=" * 88)
        print(tabulate(table, headers="keys", tablefmt="plain", showindex=False))
        return table
    df = df[df["status"].astype(str).eq("executed")].copy()
    df = df[pd.to_numeric(df["from_stage"], errors="coerce").ge(3)]
    df = df[pd.to_numeric(df["to_stage"], errors="coerce").le(int(args.zoo_total_num))]
    if "insert_id" in df.columns:
        main = _vldb_main_insert_rows(df)
        if not main.empty:
            df = main
    timing_note = df.get("timing_note", pd.Series(index=df.index, dtype=str)).fillna("").astype(str).str.lower()
    df["_index_refresh_internal_valid"] = _vldb_bool_col(df, "index_refresh_timing_valid", False) & ~timing_note.str.contains("outer legacy-command runtime")
    df["_valid_insert"] = _vldb_bool_col(df, "new_forward_timing_valid", False) & df["_index_refresh_internal_valid"]
    df = _vldb_latest_by_file_order(df, "to_stage")
    df = df.sort_values("to_stage")
    rows = _vldb_insert_cost_rows_for_method("TSRouter", df)
    baseline_log_path = Path("results_csv") / "TSRouter" / "vldb" / "logs" / "baseline_insert_log.csv"
    if baseline_log_path.exists():
        try:
            bdf = pd.read_csv(baseline_log_path)
        except Exception:
            bdf = pd.DataFrame()
        if not bdf.empty:
            final_bdf = bdf.copy()
            if "to_stage" in final_bdf.columns:
                final_bdf = final_bdf[pd.to_numeric(final_bdf["to_stage"], errors="coerce").eq(int(args.zoo_total_num))]
            if "method" in final_bdf.columns:
                final_bdf = _vldb_latest_by_file_order(final_bdf, "method")
            for _, rec in final_bdf.iterrows():
                method = str(rec.get("method", "Baseline"))
                timing_note = str(rec.get("timing_note", ""))
                invalid_tokens = ["proxy", "single_label", "fallback"]
                timing_valid = (
                    str(rec.get("timing_valid", "")).lower() == "true"
                    and not any(token in timing_note.lower() for token in invalid_tokens)
                )
                invalid_note = (
                    timing_note
                    if timing_valid
                    else f"{timing_note}; not paper-final: real insert operation was not timed"
                )
                detail_cols = [
                    ("IncomingProfile(s)", "incoming_profile_ms"),
                    ("StaticRankRefresh(s)", "static_rank_refresh_ms"),
                    ("Relabel(s)", "relabel_ms"),
                    ("Retrain(s)", "retrain_ms"),
                    ("TotalInsert(s)", "total_insert_ms"),
                    ("OldForwards", "old_model_forwards"),
                ]
                for metric_name, col in detail_cols:
                    value = rec.get(col, np.nan)
                    if metric_name != "TotalInsert(s)" and (pd.isna(value) or str(value) == ""):
                        continue
                    if metric_name != "OldForwards" and not timing_valid:
                        value = np.nan
                    if metric_name.endswith("(s)"):
                        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                        value = float(numeric_value) / 1000.0 if pd.notna(numeric_value) and np.isfinite(float(numeric_value)) else np.nan
                    stage_val = pd.to_numeric(pd.Series([rec.get("to_stage", np.nan)]), errors="coerce").iloc[0]
                    stage_col = f"stage{int(stage_val)}" if pd.notna(stage_val) else "Value"
                    rows.append({
                        "Method": method,
                        "Metric": metric_name,
                        stage_col: value,
                        "Valid": "true" if timing_valid else "false",
                        "Note": invalid_note if metric_name == "TotalInsert(s)" else "",
                    })
    table = pd.DataFrame(rows).fillna("")
    stage_cols = sorted(
        [col for col in table.columns if str(col).startswith("stage")],
        key=lambda x: int(str(x).replace("stage", "")) if str(x).replace("stage", "").isdigit() else 10**9,
    )
    front_cols = [col for col in ["Method", "Metric"] if col in table.columns]
    tail_cols = [col for col in ["Valid", "Note"] if col in table.columns]
    other_cols = [col for col in table.columns if col not in set(front_cols + stage_cols + tail_cols)]
    table = table[front_cols + stage_cols + other_cols + tail_cols]
    print("\n" + "=" * 88)
    print("VLDB Table3.2 Insert Cost Breakdown Final")
    print("=" * 88)
    print(tabulate(table, headers="keys", tablefmt="plain", floatfmt=".4f", numalign="decimal", stralign="left", showindex=False))
    return table


def _vldb_write_table_csv(table: pd.DataFrame, filename: str) -> str:
    out_dir = Path("results_csv") / "TSRouter" / "vldb" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    table.to_csv(path, index=False)
    return str(path).replace("\\", "/")


def _vldb_parse_ms_triplet(text) -> tuple[float, float, float]:
    raw = str(text).strip()
    if not raw or raw.lower() in {"nan", "n/a", "na"}:
        return (np.nan, np.nan, np.nan)
    parts = raw.split("/")
    if len(parts) == 1:
        parts = parts * 3
    vals = []
    for part in parts[:3]:
        val = pd.to_numeric(pd.Series([part]), errors="coerce").iloc[0]
        vals.append(float(val) if pd.notna(val) and np.isfinite(float(val)) else np.nan)
    while len(vals) < 3:
        vals.append(np.nan)
    return tuple(vals[:3])


def _vldb_route_metric_values(eff: dict) -> dict[str, float]:
    p50, p90, p95 = _vldb_parse_ms_triplet(eff.get("ROUTE_ms_P50_P90_P95", ""))
    throughput = pd.to_numeric(pd.Series([eff.get("Route_Throughput", np.nan)]), errors="coerce").iloc[0]
    return {
        "Route-P50(s)": p50 / 1000.0 if np.isfinite(p50) else np.nan,
        "Route-P90(s)": p90 / 1000.0 if np.isfinite(p90) else np.nan,
        "Route-P95(s)": p95 / 1000.0 if np.isfinite(p95) else np.nan,
        "ROUTE-Throughput(req/min)": float(throughput) if pd.notna(throughput) and np.isfinite(float(throughput)) else np.nan,
    }


def _vldb_tsrouter_task_sample_efficiency(stage: int, expected_n: int | None = None) -> dict | None:
    log_path = Path("results_csv") / "TSRouter" / "vldb" / "logs" / "route_latency_log.csv"
    if not log_path.exists():
        return None
    try:
        log_df = pd.read_csv(log_path, low_memory=False)
    except Exception:
        return None
    required = {"timing_level", "stage", "method", "dataset", "task_sampling_ms"}
    if log_df.empty or not required.issubset(set(log_df.columns)):
        return None
    df = log_df[log_df["timing_level"].astype(str).eq("selector_dataset_internal")].copy()
    df = df[pd.to_numeric(df["stage"], errors="coerce").eq(float(stage))]
    df = df[df["method"].astype(str).isin({"TSRouter", "TSRouter"})].copy()
    if "vldb_fast_sample" in df.columns:
        df = df[~df["vldb_fast_sample"].astype(str).str.lower().eq("true")].copy()
    fresh = pd.Series(True, index=df.index)
    if "task_sampling_timing_valid" in df.columns:
        fresh = fresh & df["task_sampling_timing_valid"].astype(str).str.lower().eq("true")
    if "task_sampling_ms" in df.columns:
        sample_ms = pd.to_numeric(df["task_sampling_ms"], errors="coerce")
        fresh = fresh & sample_ms.notna() & np.isfinite(sample_ms) & sample_ms.ge(0)
    df = df[fresh].copy()
    if df.empty:
        return None
    df = _vldb_latest_by_file_order(df, "dataset")
    vals = pd.to_numeric(df["task_sampling_ms"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return None
    route_qs = [_vldb_quantile(vals, q) for q in [0.50, 0.90, 0.95]]
    p95_route_s = route_qs[-1] / 1000.0 if np.isfinite(route_qs[-1]) else np.nan
    reasons = ["fallback to TSRouter non-skip task_sampling_ms"]
    if expected_n is not None and len(vals) != int(expected_n):
        reasons.append(f"partial Eff-N={len(vals)}/{int(expected_n)}")
    route_ids = ",".join(df.get("route_id", pd.Series(dtype=str)).dropna().astype(str).unique()[:3])
    return {
        "ROUTE_ms_P50_P90_P95": _vldb_format_ms_triplet(route_qs),
        "E2E_ms_P50_P90_P95": _vldb_format_ms_triplet(route_qs),
        "Route_Throughput": 60.0 / p95_route_s if np.isfinite(p95_route_s) and p95_route_s > 0 else np.nan,
        "Efficiency_valid_DS": int(len(vals)),
        "Skipped_parts": ", ".join(reasons),
        "_efficiency_diag": f"tsrouter_sample_fallback route_ids={route_ids}, Eff-N={len(vals)}",
    }


def _vldb_route_eff_is_missing_or_stale(eff: dict) -> bool:
    p50, p90, p95 = _vldb_parse_ms_triplet(eff.get("ROUTE_ms_P50_P90_P95", ""))
    if not np.isfinite(p95) or p95 <= 0:
        return True
    note = " ".join(str(eff.get(key, "")) for key in ["Skipped_parts", "_efficiency_diag"]).lower()
    return any(token in note for token in ["proxy", "single_label", "fallback"])


def _vldb_table4_method_eff(args, results_dir: Path, method: str, stage: int, expected: set[str]) -> dict:
    task_n = int(getattr(args, "sample_repr_num", 20))
    if method == "TSRouter":
        return _vldb_route_efficiency(
            results_dir,
            "TSRouter",
            stage,
            route_id=_vldb_main_route_id_candidates(args, stage, route_efficiency_mode=False),
            expected_n=len(expected),
            ignore_route_valid_flags=True,
        )
    if method == "TSRouter-fast":
        return _vldb_route_efficiency(
            results_dir,
            "TSRouter-fast",
            stage,
            route_id=_vldb_main_route_id_candidates(args, stage, route_efficiency_mode=True),
            expected_n=len(expected),
            ignore_route_valid_flags=True,
        )
    if method == "Task-Probe Forward":
        return _vldb_route_efficiency(
            results_dir,
            f"Task-Probe Forward (taskN={task_n})",
            stage,
            route_id=[
                f"stage{stage}_task_probe_forward_task{task_n}_route",
                f"stage{stage}_task_probe_forward_route",
            ],
            expected_n=len(expected),
            ignore_route_valid_flags=True,
        )
    if method == "Full-Zoo Forward":
        return _vldb_full_zoo_forward_efficiency(stage, expected, tsfm_results_dir=getattr(args, "TSFM_results_dir", "cl_512"))
    return {}


def _vldb_linear_extrapolate(stage_values: dict[int, float], target_stage: int) -> float:
    points = [
        (float(stage), float(value))
        for stage, value in sorted(stage_values.items())
        if np.isfinite(float(value))
    ]
    if len(points) >= 2:
        xs = np.asarray([p[0] for p in points], dtype=float)
        ys = np.asarray([p[1] for p in points], dtype=float)
        slope, intercept = np.polyfit(xs, ys, 1)
        return max(0.0, float(slope * float(target_stage) + intercept))
    if len(points) == 1:
        return max(0.0, float(points[0][1]))
    return np.nan


def run_vldb_table4_latency_scalability_maintenance(args, results_dir: Path):
    stages = _vldb_growing_zoo_stages(args)
    expected = _vldb_expected_datasets(args)
    task_n = int(getattr(args, "sample_repr_num", 20))
    method_labels = {
        "TSRouter": "TSRouter",
        "TSRouter-fast": "TSRouter-fast",
        "Task-Probe Forward": f"Task-Probe Forward (taskN={task_n})",
        "Full-Zoo Forward": "Full-Zoo Forward",
    }
    metrics_by_method: dict[str, dict[str, dict[int, float]]] = {
        method: {
            "Route-P50(s)": {},
            "Route-P90(s)": {},
            "Route-P95(s)": {},
            "ROUTE-Throughput(req/min)": {},
        }
        for method in method_labels
    }
    effn_by_method: dict[str, dict[int, object]] = {method: {} for method in method_labels}
    note_by_method: dict[str, list[str]] = {method: [] for method in method_labels}
    for stage in stages:
        for method in method_labels:
            eff = _vldb_table4_method_eff(args, results_dir, method, stage, expected)
            values = _vldb_route_metric_values(eff)
            for metric, value in values.items():
                metrics_by_method[method][metric][int(stage)] = value
            effn_by_method[method][int(stage)] = eff.get("Efficiency_valid_DS", "")
            note = str(eff.get("Skipped_parts", "") or "")
            if note and note not in note_by_method[method]:
                note_by_method[method].append(note)
    rows = []
    pred_stages = [20, 40, 80]
    for method, display_method in method_labels.items():
        for metric, stage_values in metrics_by_method[method].items():
            row = {
                "Method": display_method,
                "Metric": metric,
                "Fit": "observed+linear",
            }
            for stage in stages:
                row[f"stage{int(stage)}"] = stage_values.get(int(stage), np.nan)
            for pred_stage in pred_stages:
                row[f"pred_stage{pred_stage}"] = _vldb_linear_extrapolate(stage_values, pred_stage)
            row["Eff-N_by_stage"] = "; ".join(f"{stage}:{effn_by_method[method].get(int(stage), '')}" for stage in stages)
            row["Note"] = " | ".join(note_by_method[method][:3])
            rows.append(row)
    table = pd.DataFrame(rows)
    value_cols = [c for c in table.columns if str(c).startswith("stage") or str(c).startswith("pred_stage")]
    for col in value_cols:
        table[col] = pd.to_numeric(table[col], errors="coerce").round(2)
    csv_path = _vldb_write_table_csv(table, "vldb_table4_latency_scalability_maintenance.csv")
    print("\n" + "=" * 88)
    print("VLDB Table4 Latency, Scalability, and Maintenance Plot Data Final")
    print("=" * 88)
    print(f"[write] {csv_path}")
    print(tabulate(table, headers="keys", tablefmt="plain", floatfmt=".2f", numalign="decimal", stralign="left", showindex=False))
    return table


def _vldb_quantile_triplet_from_values(vals: pd.Series, scale: float = 1.0) -> str:
    vals = pd.to_numeric(vals, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna() * float(scale)
    return _vldb_format_ms_triplet([_vldb_quantile(vals, q) for q in [0.50, 0.90, 0.95]])


def run_vldb_table5_maintenance_component_summary(args, results_dir: Path):
    rows = []
    max_stage = int(getattr(args, "zoo_total_num", 10))
    insert_path = Path("results_csv") / "TSRouter" / "vldb" / "logs" / "insert_log.csv"
    baseline_insert_path = Path("results_csv") / "TSRouter" / "vldb" / "logs" / "baseline_insert_log.csv"

    if insert_path.exists():
        idf = pd.read_csv(insert_path)
        if not idf.empty:
            idf = idf[pd.to_numeric(idf.get("to_stage", pd.Series(dtype=float)), errors="coerce").le(max_stage)].copy()
            idf = idf[pd.to_numeric(idf.get("from_stage", pd.Series(dtype=float)), errors="coerce").ge(3)].copy()
            if "insert_id" in idf.columns:
                main = _vldb_main_insert_rows(idf)
                if not main.empty:
                    idf = main
            if "to_stage" in idf.columns:
                idf = _vldb_latest_by_file_order(idf, "to_stage")
            timing_note = idf.get("timing_note", pd.Series(index=idf.index, dtype=str)).fillna("").astype(str).str.lower()
            index_internal_valid = _vldb_bool_col(idf, "index_refresh_timing_valid", False) & ~timing_note.str.contains("outer legacy-command runtime")
            total_s = (
                pd.to_numeric(idf.get("new_forward_s", pd.Series(index=idf.index, dtype=float)), errors="coerce").fillna(0.0)
                + pd.to_numeric(idf.get("profile_merge_s", pd.Series(index=idf.index, dtype=float)), errors="coerce").fillna(0.0)
                + pd.to_numeric(idf.get("index_refresh_s", pd.Series(index=idf.index, dtype=float)), errors="coerce").fillna(0.0)
            )
            valid = _vldb_bool_col(idf, "new_forward_timing_valid", False) & index_internal_valid
            total_s = total_s.where(valid, np.nan)
            rows.append({
                "Method": "TSRouter",
                "MaintainedProfile": "Yes",
                "RetrainOnInsert": "No",
                "OnlineCandidateForwards": 0,
                "OldModelForwardsOnInsert": int(pd.to_numeric(idf.get("old_model_forwards", pd.Series(dtype=float)), errors="coerce").fillna(0).max()) if not idf.empty else np.nan,
                "InsertCost P50/90/95(s)": _vldb_quantile_triplet_from_values(total_s, scale=1.0),
                "Valid-N": int(valid.sum()),
                "Stages": f"{int(pd.to_numeric(idf['from_stage'], errors='coerce').min())}->{int(pd.to_numeric(idf['to_stage'], errors='coerce').max())}" if "from_stage" in idf.columns and "to_stage" in idf.columns and not idf.empty else "",
                "SourceFiles": "results_csv/TSRouter/vldb/logs/insert_log.csv",
                "Missing": "" if int(valid.sum()) else "no valid internal Step3 index-refresh timing rows",
            })
    else:
        rows.append({
            "Method": "TSRouter",
            "MaintainedProfile": "Yes",
            "RetrainOnInsert": "No",
            "OnlineCandidateForwards": 0,
            "OldModelForwardsOnInsert": 0,
            "InsertCost P50/90/95(s)": "nan/nan/nan",
            "Valid-N": 0,
            "Stages": "",
            "SourceFiles": "results_csv/TSRouter/vldb/logs/insert_log.csv",
            "Missing": "missing insert_log.csv",
        })

    baseline_methods = {
        "Task-Probe Forward": ("No", "No", f"{int(getattr(args, 'sample_repr_num', 20))} x M"),
        "Probe-Static Best": ("No", "No", 0),
    }
    if baseline_insert_path.exists():
        bdf = pd.read_csv(baseline_insert_path)
        for method, (maintained, retrain, online_forwards) in baseline_methods.items():
            sub = bdf[bdf.get("method", pd.Series(dtype=str)).astype(str).eq(method)].copy()
            sub = sub[pd.to_numeric(sub.get("to_stage", pd.Series(dtype=float)), errors="coerce").le(max_stage)]
            if sub.empty:
                rows.append({
                    "Method": method,
                    "MaintainedProfile": maintained,
                    "RetrainOnInsert": retrain,
                    "OnlineCandidateForwards": online_forwards,
                    "OldModelForwardsOnInsert": np.nan,
                    "InsertCost P50/90/95(s)": "nan/nan/nan",
                    "Valid-N": 0,
                    "Stages": "",
                    "SourceFiles": "results_csv/TSRouter/vldb/logs/baseline_insert_log.csv",
                    "Missing": "no matching baseline insert rows",
                })
                continue
            if "to_stage" in sub.columns:
                sub = _vldb_latest_by_file_order(sub, "to_stage")
            note = " ".join(sub.get("timing_note", pd.Series(dtype=str)).dropna().astype(str).unique()).lower()
            base_valid = _vldb_bool_col(sub, "timing_valid", False)
            paper_invalid = any(token in note for token in ["proxy", "single_label", "fallback", "oracle"])
            cost_s = pd.to_numeric(sub.get("total_insert_ms", pd.Series(index=sub.index, dtype=float)), errors="coerce") / 1000.0
            rows.append({
                "Method": method,
                "MaintainedProfile": maintained,
                "RetrainOnInsert": retrain,
                "OnlineCandidateForwards": online_forwards,
                "OldModelForwardsOnInsert": int(pd.to_numeric(sub.get("old_model_forwards", pd.Series(dtype=float)), errors="coerce").fillna(0).max()) if not sub.empty else np.nan,
                "InsertCost P50/90/95(s)": _vldb_quantile_triplet_from_values(cost_s, scale=1.0),
                "Valid-N": int(base_valid.sum()) if not paper_invalid else 0,
                "Stages": f"{int(pd.to_numeric(sub['from_stage'], errors='coerce').min())}->{int(pd.to_numeric(sub['to_stage'], errors='coerce').max())}" if "from_stage" in sub.columns and "to_stage" in sub.columns and not sub.empty else "",
                "SourceFiles": "results_csv/TSRouter/vldb/logs/baseline_insert_log.csv",
                "Missing": "not paper-final: proxy/single_label/fallback/oracle timing note" if paper_invalid else "",
            })
    else:
        for method, (maintained, retrain, online_forwards) in baseline_methods.items():
            rows.append({
                "Method": method,
                "MaintainedProfile": maintained,
                "RetrainOnInsert": retrain,
                "OnlineCandidateForwards": online_forwards,
                "OldModelForwardsOnInsert": np.nan,
                "InsertCost P50/90/95(s)": "nan/nan/nan",
                "Valid-N": 0,
                "Stages": "",
                "SourceFiles": "results_csv/TSRouter/vldb/logs/baseline_insert_log.csv",
                "Missing": "missing baseline_insert_log.csv",
            })

    rows.append({
        "Method": "Full rebuild profile",
        "MaintainedProfile": "No",
        "RetrainOnInsert": "Optional",
        "OnlineCandidateForwards": 0,
        "OldModelForwardsOnInsert": "all old models",
        "InsertCost P50/90/95(s)": "nan/nan/nan",
        "Valid-N": 0,
        "Stages": "",
        "SourceFiles": "not found",
        "Missing": "no full-rebuild profile timing artifact in current downloaded results",
    })
    table = pd.DataFrame(rows)
    csv_path = _vldb_write_table_csv(table, "vldb_table5_maintenance_component_summary.csv")
    print("\n" + "=" * 88)
    print("VLDB Table5 Maintenance and Component Summary on Instance-Level Zoo Final")
    print("=" * 88)
    print(f"[write] {csv_path}")
    print(tabulate(table, headers="keys", tablefmt="plain", floatfmt=".4f", numalign="decimal", stralign="left", showindex=False))
    return table


def run_vldb_table6_winner_distribution(
    args,
    baseline_df_all: pd.DataFrame,
    ordered_model_names: list[str],
    season_naive_df=None,
):
    metric = "MASE"
    stages = _vldb_growing_zoo_stages(args)
    final_models = ordered_model_names[: int(getattr(args, "zoo_total_num", len(ordered_model_names)))]
    expected = _vldb_expected_datasets(args)
    rows = {model: {"Model": model, "ActiveStages": 0, "TotalWins": 0} for model in final_models}
    support_notes = []

    for stage in stages:
        active_models = final_models[: int(stage)]
        zcol = f"z{int(stage)}-{int(getattr(args, 'zoo_total_num', stage))}"
        stage_df = baseline_df_all[baseline_df_all["model"].isin(active_models)].copy()
        if getattr(args, "GE_released", False):
            stage_df = normalize_by_season_naive(stage_df, season_naive_df)
        if "dataset" in stage_df.columns:
            stage_df = stage_df[stage_df["dataset"].isin(expected)].copy()
        if metric not in stage_df.columns or stage_df.empty:
            for model in final_models:
                rows[model][zcol] = np.nan if model in active_models else 0
            support_notes.append(f"{zcol}:0/{len(expected)}")
            continue

        order_map = {model: idx for idx, model in enumerate(active_models)}
        work = stage_df[["dataset", "model", metric]].copy()
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        work = work.dropna(subset=["dataset", "model", metric])
        work["_model_order"] = work["model"].map(order_map).fillna(10**9)
        winners = (
            work.sort_values(["dataset", metric, "_model_order"], ascending=[True, True, True])
            .drop_duplicates("dataset", keep="first")
        )
        winners = winners[winners["dataset"].isin(expected)].copy()
        counts = winners["model"].value_counts().to_dict()
        support_n = int(len(set(winners["dataset"].dropna().astype(str)) & expected))
        support_notes.append(f"{zcol}:{support_n}/{len(expected)}")
        for model in final_models:
            count = int(counts.get(model, 0)) if model in active_models else 0
            rows[model][zcol] = count
            if model in active_models:
                rows[model]["ActiveStages"] += 1
                rows[model]["TotalWins"] += count

    table = pd.DataFrame(rows.values())
    zcols = [f"z{int(stage)}-{int(getattr(args, 'zoo_total_num', stage))}" for stage in stages]
    table = table[["Model", *zcols, "TotalWins", "ActiveStages"]]
    csv_path = _vldb_write_table_csv(table, "vldb_table6_mase_winner_distribution.csv")
    print("\n" + "=" * 88)
    print("VLDB Table6 MASE Winner Distribution by Stage Final")
    print("=" * 88)
    print(f"[config] metric=MASE, stages={stages}, final_zoo_total_num={getattr(args, 'zoo_total_num', '')}")
    print(f"[support] {', '.join(support_notes)}")
    print(f"[write] {csv_path}")
    print(tabulate(table, headers="keys", tablefmt="plain", floatfmt=".4f", numalign="decimal", stralign="left", showindex=False))
    return table


import argparse

from cli.run_model_zoo import build_parser, prepare_args




def build_check_selector_parser():
                                    
    base = build_parser(add_help=False)

    parser = argparse.ArgumentParser(
        parents=[base],
        add_help=False,
        description="check selector",
    )

                                           
    parser.add_argument("--results_dir", type=str, default=rel(TSFM_CSV_ROOT), help='TSRouter runtime message.')
    parser.add_argument(
        "--TSFM_context_len_list",
        type=str,
        default=",".join(DEFAULT_CONTEXT_LENS),
        help='TSRouter runtime message.',
    )
    parser.add_argument(
        "--TSFM_summary_output_dir",
        type=str,
        default=default_output_csv_dir(),
        help='TSRouter runtime message.',
    )
    parser.add_argument(
        "--TSFM_summary_figure_dir",
        type=str,
        default=default_figure_dir(),
        help='TSRouter runtime message.',
    )
    parser.add_argument(
        "--TSFM_tradeoff_metrics",
        type=str,
        default=",".join(DEFAULT_TRADEOFF_METRICS),
        help='TSRouter runtime message.',
    )
    parser.add_argument(
        "--TSFM_plot_exclude_models",
        type=str,
        default="",
        help='TSRouter runtime message.',
    )
    parser.add_argument("--skip_TSFM_context_summary", action="store_true", help='TSRouter runtime message.')
    parser.add_argument("--skip_TSFM_tradeoff_plot", action="store_true", help='TSRouter runtime message.')
    parser.add_argument("--TSFM_context_summary_only", action="store_true", help='TSRouter runtime message.')
    parser.add_argument("--rank_base", type=str, default="MASE", choices=["MASE", "sMAPE", "CRPS"])
    parser.add_argument(
        "--summary_process_metrics_region_rule",
        dest="process_metrics_region_rule",
        choices=["auto", "strict", "effective"],
        default=argparse.SUPPRESS,
        help=(
            'TSRouter runtime message.'
            'TSRouter runtime message.'
        ),
    )
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument("--random_seeds", type=str, default="2025,2026,2027,2028,2029,2030,2031,2032,2033,2034", help='TSRouter runtime message.')
    parser.add_argument("--season_naive_path", type=str, default=f"{rel(TSFM_CSV_ROOT)}/naive/all_results.csv", help='TSRouter runtime message.')
    parser.add_argument("--enable_rebuttal_analysis", action="store_true", help='TSRouter runtime message.')
    parser.add_argument("--enable_rebuttal_subset_analysis", action="store_true", help='TSRouter runtime message.')
    parser.add_argument(
        "--vldb_results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help='TSRouter runtime message.',
    )
    parser.add_argument(
        "--vldb_figure_dir",
        type=str,
        default="results_csv/TSRouter/vldb/figures",
        help='TSRouter runtime message.',
    )
    parser.add_argument(
        "--channel_failure_analysis",
        "--enable_channel_failure_analysis",
        action="store_true",
        help='TSRouter runtime message.',
    )
    parser.add_argument(
        "--channel_failure_stages",
        type=str,
        default="latest",
        help='TSRouter runtime message.',
    )
    parser.add_argument(
        "--channel_failure_rank_metric",
        type=str,
        default="",
        choices=["", "MASE", "sMAPE", "CRPS"],
        help='TSRouter runtime message.',
    )
    parser.add_argument(
        "--channel_failure_model_cl_name",
        type=str,
        default="",
        help='TSRouter runtime message.',
    )
    parser.add_argument(
        "--channel_failure_output_dir",
        type=str,
        default="results_csv/analysis/channel_failure",
        help='TSRouter runtime message.',
    )
    parser.add_argument(
        "--channel_failure_force_meta",
        action="store_true",
        help='TSRouter runtime message.',
    )
    parser.add_argument(
        "--channel_failure_force_rank",
        action="store_true",
        help='TSRouter runtime message.',
    )
    parser.add_argument("--analysis_fig_root", type=str, default="figs", help='TSRouter runtime message.')
    parser.add_argument("--top1_check_mode", action="store_true", default=True, help='TSRouter runtime message.')
    parser.add_argument("--top1_check_print_limit", type=int, default=97, help='TSRouter runtime message.')
    parser.add_argument("--table4_repr_token", type=str, default="c-l-e", help='TSRouter runtime message.')
    parser.add_argument("--table4_repr_size", type=int, default=3000, help='TSRouter runtime message.')
    # --sample_repr_num is inherited from run_model_zoo.build_parser().
    # Deprecated old paper-table entry. New summary work is tracked in
    # docs/tsrouter_check_selector_summary_tracking_2026_06_03.md.
                                                                                                                          
    # parser.add_argument(
    #     "--vldb-table",
    #     nargs="?",
    #     const="all",
    #     default="",
                                                 
    # )
    parser.add_argument(
        "--table4_top_cluster_list",
        type=str,
        default="15,20,25,30,35,40,45,50,60,70,80,90,100",
        help='TSRouter runtime message.',
    )
    return parser



if __name__ == "__main__":
    parser = build_check_selector_parser()
    args = parser.parse_args()
    args = prepare_args(args)
    from utils.path_utils import (
        auto_cl_tsfm_comparison_dir,
        get_auto_cl_mode,
        get_auto_cl_profiles,
        get_fixed_cl_profile,
    )

    summary_auto_cl_mode = get_auto_cl_mode(args)
    summary_fixed_cl_profile = None
    if summary_auto_cl_mode != "v0":
        comparison_dir = auto_cl_tsfm_comparison_dir(summary_auto_cl_mode)
        if str(getattr(args, "TSFM_results_dir", "")) != comparison_dir:
            print(
                f"[summary] auto_cl_mode={summary_auto_cl_mode} forces TSFM comparison baseline "
                f"to {comparison_dir}"
            )
        args.TSFM_results_dir = comparison_dir
    else:
        summary_fixed_cl_profile = get_fixed_cl_profile(
            getattr(args, "TSFM_results_dir", "cl_512")
        )
        print(
            f"[summary] auto_cl_mode=v0 uses TSFM comparison baseline "
            f"{getattr(args, 'TSFM_results_dir', 'cl_512')}"
        )
        if summary_fixed_cl_profile is not None:
            print(
                "[summary] fixed-cl selector profile "
                f"repr={summary_fixed_cl_profile['repr_input_dim']}to"
                f"{summary_fixed_cl_profile['repr_output_dim']} "
                f"pl={summary_fixed_cl_profile['repr_sub_pred_len']} "
                f"source_len={summary_fixed_cl_profile['repr_source_exact_length']}"
            )

    base_metrics_dict = {'C': 'CRPS', 'M': "MASE", 'S': "sMAPE"}
    print(
        "[summary] Step3 process metrics: "
        f"region_rule={getattr(args, 'process_metrics_region_rule', 'auto')}; "
        "metric source follows each configuration's base_metrics"
    )

                                                                                          

    results_dir = Path(args.results_dir)
    args.zoo_total_num = sum(len(sizes) for sizes in Model_zoo_details.values())

    if not getattr(args, "skip_TSFM_context_summary", False):
        summarize_tsfm_context_lens(
            context_lens=parse_context_lens(args.TSFM_context_len_list),
            results_root=results_dir,
            output_csv_dir=args.TSFM_summary_output_dir,
            figure_dir=args.TSFM_summary_figure_dir,
            quick_test=args.quick_test,
            verbose=args.verbose,
            print_tables=True,
            make_plot=not args.skip_TSFM_tradeoff_plot,
            tradeoff_metrics=args.TSFM_tradeoff_metrics,
            plot_exclude_models=args.TSFM_plot_exclude_models,
        )
        if getattr(args, "TSFM_context_summary_only", False):
            raise SystemExit(0)

    # =========================
                     
    # =========================

                            
    baseline_data = []
    for model_name in All_sorted_model_names:
        context_tag = args.context_len if args.fix_context_len else "original"
        file_path = resolve_model_result_csv(args,results_dir, model_name, context_tag)
        if file_path.exists():
            df = check_results_file(file_path, False, args.quick_test)
            if df is not None:
                df = harmonize_metrics_schema(df)
                df["model"] = model_name
                baseline_data.append(df)
        else:
            if args.verbose:
                print(f"TSRouter runtime message: {file_path}\n")

    season_naive_df = None
    season_naive_path = Path(args.season_naive_path)
    if season_naive_path.exists():
        sn = check_results_file(season_naive_path, False, args.quick_test)
        if sn is not None:
            sn = harmonize_metrics_schema(sn)
            sn["model"] = "Season_Naive"
            season_naive_df = sn
            if args.verbose:
                print(f"TSRouter runtime message: {season_naive_path}")
    else:
        if args.verbose:
            print(f"TSRouter runtime message: {season_naive_path}")

    baseline_df_all = standardize_model_names(baseline_data)

                                        
    model_release_list = []
    for family, sizes in Model_zoo_details.items():
        for size, details in sizes.items():
            full_name = f"{family}_{size}"
            abbrev = details.get("abbreviation", Model_abbrev_map.get(full_name, full_name))
            rel = details.get("release_date", "2026-01-01")
            model_release_list.append((rel, abbrev))

             
    model_release_list = sorted(model_release_list, key=lambda x: x[0])
    ordered_model_names = [Model_abbrev_map.get(m, m) for m in All_sorted_model_names]
    args.zoo_total_num = len(ordered_model_names)

    # =========================
                     
    # =========================
    rank_summary_all = {"RANK": pd.DataFrame()}
    random_seeds = parse_seed_list(args.random_seeds)
    selector_records = []                    

    if args.real_world_mode:
        current_zoo_nums = range(args.ensemble_size + 1, args.zoo_total_num + 1)
    else:                                         
        current_zoo_nums = [args.zoo_total_num]

    k_order = [1,3]                            

    # =========================
                            
    # rw_store[row_name][zcol] = {"Rank": x, "sMAPE": y, "MASE": z}
    # =========================
    rw_store = defaultdict(dict)
    rw_zcols = set()                        
    rw_main_rows = set()                        
    rw_best_tsfm_by_z = {}

    for current_zoo_num in current_zoo_nums:
                                      
        allowed_datasets = None
        if args.quick_test:
            allowed_datasets = set(ALL_Fast_DATASETS)

        current_model_names = ordered_model_names[:current_zoo_num]
        baseline_df = baseline_df_all[baseline_df_all["model"].isin(current_model_names)].copy()
        if getattr(args, "GE_released", False):
            baseline_df = normalize_by_season_naive(baseline_df, season_naive_df)
        baseline_datasets = set(baseline_df["dataset"].unique())
        current_zcol = f"z{current_zoo_num}-{args.zoo_total_num}"
        if args.real_world_mode:
            rw_best_tsfm_by_z[current_zcol] = compute_tsfm_best_metrics_for_summary(
                baseline_df=baseline_df,
                ordered_model_names=current_model_names,
                rank_base=args.rank_base,
            )

        if allowed_datasets is not None:
            baseline_datasets &= allowed_datasets

                               
        build_sel_path = make_selector_path_builder(
            results_dir=results_dir,
            current_zoo_num=current_zoo_num,
            zoo_total_num=args.zoo_total_num,
            ensemble_size=args.ensemble_size,
            default_ensemble_agg=args.ensemble_agg,
            default_real_metric=args.real_order_metric,
            ge_released=args.GE_released,
            ge_fast_eval=args.GE_fast_eval,
            sample_repr_num=getattr(args, "sample_repr_num", 20),
        )

        if args.verbose:
            print(
                f"\n{'=' * 60}TSRouter runtime message: {current_zoo_num}-{args.zoo_total_num}, "
                f"ensemble_size={args.ensemble_size}, rank_base={args.rank_base})\n{'=' * 60}"
            )

                                                                  
        real_model_name = f"Real-{args.real_order_metric}_z{current_zoo_num}-{args.zoo_total_num}"

        real_path = build_sel_path(
            selector_name="Real_Select",
            seed=0,
            real_order_metric=args.real_order_metric,
        )
        real_channel_path = build_sel_path(
            selector_name="Real_Channel_Select",
            seed=0,
            real_order_metric=args.real_order_metric,
        )

        real_raw = None
        real_datasets = set()

        if real_path.exists():
            real_raw = check_results_file(real_path, False, args.quick_test)
            if real_raw is not None:
                real_datasets = set(real_raw["dataset"].unique())
                if allowed_datasets is not None:
                    real_datasets &= allowed_datasets
                if args.verbose:
                    print(f"TSRouter runtime message: {real_path}")
        else:
            if args.verbose:
                print(f"TSRouter runtime message: {real_path}TSRouter runtime message: ")

                                             
        selector_tasks = []
        summary_process_metrics_by_task: dict[tuple[str, str], dict] = {}

        # 1) All_Select
        for agg in ["mean", "median"]:
            all_path = build_sel_path("All_Select", ensemble_agg=agg)
            selector_tasks.append((
                "All_Select",
                None,
                all_path,
                f"All_{agg}_z{current_zoo_num}-{args.zoo_total_num}",          
            ))




        # 3) Real_Select
        if real_raw is not None:
            selector_tasks.append(("Real_Select",None,real_path,real_model_name,))
        if real_channel_path.exists():
            selector_tasks.append((
                "Real_Channel_Select",
                None,
                real_channel_path,
                f"Real-Channel-{args.real_order_metric}_z{current_zoo_num}-{args.zoo_total_num}",
            ))

                                      
        # for seed in random_seeds:
        #     rand_path = build_sel_path("Random_Select", seed=seed)
        #     selector_tasks.append(("Random_Select",seed,rand_path,f"Random_s{seed}_z{current_zoo_num}-{args.zoo_total_num}",))



        for en_size in []:#1,3,5
            for ensemble_agg in ["mean", "median"]:
                # 2) Recent_Select
                recent_path = build_sel_path("Recent_Select",ensemble_size_override=en_size,ensemble_agg=ensemble_agg)
                selector_tasks.append(("Recent_Select", None, recent_path, f"Recent_{en_size}_{ensemble_agg}_z{current_zoo_num}-{args.zoo_total_num}",))

                # 5) Current_best_sMAPE
                current_best_smape_Rank_path = build_sel_path("Current_best_sMAPE_Rank",ensemble_size_override=en_size,ensemble_agg=ensemble_agg)
                selector_tasks.append(("Current_best_sMAPE_Rank", None, current_best_smape_Rank_path,
                                       f"Current_best_sMAPE_Rank_{en_size}_{ensemble_agg}_z{current_zoo_num}-{args.zoo_total_num}"))

                current_best_smape_path = build_sel_path("Current_best_sMAPE",ensemble_size_override=en_size,ensemble_agg=ensemble_agg)
                selector_tasks.append(
                    ("Current_best_sMAPE", None, current_best_smape_path, f"Current_best_sMAPE_{en_size}_{ensemble_agg}_z{current_zoo_num}-{args.zoo_total_num}"))
                # 6) Current_best_MASE
                current_best_mase_path = build_sel_path("Current_best_MASE",ensemble_size_override=en_size,ensemble_agg=ensemble_agg)
                selector_tasks.append(
                    ("Current_best_MASE", None, current_best_mase_path, f"Current_best_MASE_{en_size}_{ensemble_agg}_z{current_zoo_num}-{args.zoo_total_num}"))
                # 7) Current_best_CRPS
                current_best_crps_path = build_sel_path("Current_best_CRPS",ensemble_size_override=en_size,ensemble_agg=ensemble_agg)
                selector_tasks.append(
                    ("Current_best_CRPS", None, current_best_crps_path, f"Current_best_CRPS_{en_size}_{ensemble_agg}_z{current_zoo_num}-{args.zoo_total_num}"))

                # 8) LogME_Select
                logme_path = build_sel_path("LogME_Select",ensemble_size_override=en_size,ensemble_agg=ensemble_agg)
                selector_tasks.append(("LogME_Select",None,logme_path,f"LogME_{en_size}_{ensemble_agg}_z{current_zoo_num}-{args.zoo_total_num}",))




        # 5) TSRouter
        def add_zoo_cast_tasks(selector_tasks: list, args, param_grid: dict):
            'TSRouter runtime message.'
            from utils.path_utils import get_repr_save_path

                                           
            keys = list(param_grid.keys())
            values_lists = [param_grid[k] for k in keys]
            missing_simplets_checkpoint_warnings = set()

            for values in product(*values_lists):
                args_v = copy.deepcopy(args)
                args_v.current_zoo_num = current_zoo_num
                                                                           
                                                                   
                # args_v.repr_encoder="Chronos_512to384"
                # args_v.repr_encoder="SimMTM_36to128"
                                
                for k, v in zip(keys, values):
                                                 
                    if k in ["repr_weight_ratio", "minus_ratio", "repr_weight_ratio", "subset_perf_scale",
                             "rank_decay_coef", "context_len_adaptive_threshold", "sample_repr_ratio",
                             "task_rank_top3_instability_threshold"]:
                        v = float(v)
                    elif k in ["repr_v", "sample_repr_num","task_sample_version",  "restrict_top_model_num", "ensemble_size", "subset_top_k","repr_size", "repr_data_seed",
                               "repr_encoder_seed", "forward_seed", "search_seed", "repr_input_dim", "repr_output_dim", "repr_sub_pred_len", "repr_v5_nearest_k",
                               "pred_len_adaptive_threshold", "short_repr_input_dim", "long_repr_input_dim", "short_repr_sub_pred_len", "long_repr_sub_pred_len",
                               "train_encoder_epochs", "train_encoder_batch_size",
                               ]:
                        v = int(v)
                    setattr(args_v, k, v)
                    if k == "repr_encoder":
                        from utils.path_utils import infer_type_structure_from_encoder_name
                        encoder_type, encoder_structure = infer_type_structure_from_encoder_name(str(v))
                        if encoder_type is not None:
                            args_v.encoder_type = encoder_type
                            args_v.encoder_structure = encoder_structure
                    elif k in {"encoder_type", "encoder_structure"}:
                        setattr(args_v, "repr_encoder", "")
                from utils.path_utils import normalize_encoder_variant_args
                normalize_encoder_variant_args(args_v)
                from utils.path_utils import normalize_auto_cl_args
                normalize_auto_cl_args(args_v)
                simplets_checkpoint_bound = _bind_simplets_ts2vec_summary_checkpoint(
                    args_v,
                    allow_missing=True,
                )
                                                     
                if False and "seed" in keys:
                    if "repr_data_seed" not in keys:
                        args_v.repr_data_seed = int(args_v.seed)
                    if "repr_encoder_seed" not in keys:
                        args_v.repr_encoder_seed = int(args_v.seed)
                    if "forward_seed" not in keys:
                        args_v.forward_seed = int(args_v.seed)
                    if "search_seed" not in keys:
                        args_v.search_seed = int(args_v.seed)

                if args_v.fix_context_len:
                    args_v.context_len = int(args_v.repr_input_dim)
                _,_,_,tsrouter_save_name = get_repr_save_path(args_v)
                tsrouter_path = resolve_tsrouter_selector_result_path(args_v, tsrouter_save_name)
                if not tsrouter_path.exists() and simplets_checkpoint_bound:
                    materialize_compatible_tsrouter_result(
                        args_v,
                        str(tsrouter_path),
                        current_model_names,
                        verbose=bool(args.verbose),
                    )
                elif not simplets_checkpoint_bound and bool(args.verbose):
                    reason = str(
                        getattr(
                            args_v,
                            "_simplets_ts2vec_checkpoint_missing_reason",
                            "missing SimpleTS2Vec checkpoint",
                        )
                    )
                    if reason not in missing_simplets_checkpoint_warnings:
                        missing_simplets_checkpoint_warnings.add(reason)
                        print(
                            'TSRouter runtime message.'
                            f"TSRouter runtime message: {reason}"
                        )

                                           
                tag_parts = []
                repr_v_head = str(getattr(args_v, "repr_v", 0))[0]
                for k, v in zip(keys, values):
                                                
                    if repr_v_head == "5" and k in {"repr_weight_ratio", "model_repr_agg"}:
                        continue
                for k, v in zip(keys, values):
                    if k in {"enable_context_len_adaptive_repr", "enable_pred_len_adaptive_repr"}:
                        continue
                    if k in {
                        "simplets_ts2vec_checkpoint",
                        "simplets_ts2vec_source_repr_encoder",
                        "simplets_ts2vec_checkpoint_fingerprint",
                    }:
                        # Provenance selects the saved result but is not a
                        # tunable method parameter or a report display label.
                        continue
                    if k == "repr_sample_qc_mode":
                        qc_mode = str(v).strip().lower()
                        if qc_mode in {"", "strict"}:
                            continue
                        tag_parts.append("noqc" if qc_mode == "off" else f"qc{qc_mode}")
                        continue
                    if k == "task_sample_strategy" and str(v).strip().lower() == "latest_random":
                        continue
                    if k == "repr_anchor_window_sample_strategy" and str(v).strip().lower() == "even":
                        continue
                    if k == "task_window_sample_strategy" and str(v).strip().lower() == "legacy":
                        continue
                    if k == "sample_repr_ratio" and float(v) <= 0:
                        continue
                    if k == "task_rank_top3_instability_threshold" and float(v) < 0:
                        continue
                    if k == "repr_anchor_protocol" and str(v).strip().lower() == "window":
                        continue
                    if k == "route_family_mode":
                        route_family_mode = str(v).strip().lower()
                        if route_family_mode == "default":
                            continue
                        tag_parts.append(
                            "rfbigger" if route_family_mode == "bigger_size" else "rfsmaller"
                        )
                        continue
                    if k == "repr_input_dim" and bool(getattr(args_v, "enable_context_len_adaptive_repr", False)):
                        tag_parts.append("inauto")
                        continue
                    if k == "repr_sub_pred_len" and bool(getattr(args_v, "enable_pred_len_adaptive_repr", False)):
                        tag_parts.append("plauto")
                        continue
                                       
                    short = {
                        "ensemble_size": "top",
                        "repr_size": "rs",
                        "repr_v": "v",
                        "subset_top_k": "sub",
                        "sample_repr_num": "task",
                        "task_sample_version": "tv",
                        "task_sample_strategy": "ts",
                        "repr_anchor_window_sample_strategy": "aws",
                        "task_window_sample_strategy": "ws",
                        "sample_repr_ratio": "sr",
                        "task_rank_top3_instability_threshold": "fb",
                        "task_channel_fuse_limit": "cf",
                        "route_family_mode": "rf",
                        "repr_anchor_protocol": "ra",
                        "restrict_top_model_num": "res",
                        "repr_weight_ratio": "w",
                        "minus_ratio": "mr",
                        "zoo_repr_set": "",
                        "base_metrics":"",
                        "repr_distance_metric":"dis",
                        "model_repr_agg":"agg",
                        "subset_perf_scale":"sps",
                        "model_repr_mode":"mode",
                        "ensemble_agg":"eagg",
                        "repr_encoder":"r",
                        "encoder_type":"et",
                        "encoder_structure":"es",
                        "train_rank_metric":"trm",
                        "train_encoder_epochs":"tep",
                        "train_encoder_batch_size":"tbs",
                        "train_encoder_lr":"tlr",
                        "train_top3_weight":"tt3",
                        "cluster_method":"cm",
                        "repr_data_seed": "sd",
                        "repr_encoder_seed": "se",
                        "forward_seed": "sf",
                        "context_len":"cl",
                        "search_seed": "ss",
                        "repr_input_dim": "in",
                        "repr_output_dim": "out",
                        "repr_sub_pred_len": "pl",
                        "repr_scale_protocol": "",
                        "repr_v5_nearest_k": "v5k",
                        "repr_v5_distance_power": "v5p",
                        "rank_decay_coef": "rd",
                        "sample_mode":"",
                        "enable_search_ensemble":"en",
                        "enable_context_len_adaptive_repr":"adcl",
                        "enable_pred_len_adaptive_repr":"adpl",
                        "context_len_adaptive_threshold":"cth",
                        "pred_len_adaptive_threshold":"pth",
                        "short_repr_input_dim":"sin",
                        "long_repr_input_dim":"lin",
                        "short_repr_sub_pred_len":"spl",
                        "long_repr_sub_pred_len":"lpl",
                    }.get(k, k)

                    if isinstance(v, float):
                        tag_parts.append(f"{short}{v:g}")
                    else:
                        tag_parts.append(f"{short}{v}")
                if bool(getattr(args_v, "enable_context_len_adaptive_repr", False)) and "repr_input_dim" not in keys:
                    tag_parts.append("inauto")
                if bool(getattr(args_v, "enable_pred_len_adaptive_repr", False)) and "repr_sub_pred_len" not in keys:
                    tag_parts.append("plauto")

                model_col_name = "_".join(tag_parts) + f"_z{current_zoo_num}-{args.zoo_total_num}"
                if get_auto_cl_mode(args_v) != "v0" and getattr(args, "vldb_results", True):
                    model_col_name = "TSRouter-autocl"

                summary_args = copy.deepcopy(args_v)
                summary_args.models = "TSRouter"
                summary_args.process_metrics_region_rule = str(
                    getattr(args, "process_metrics_region_rule", "auto")
                )
                summary_metrics = load_encoder_enrichment_for_args(summary_args)
                task_key = (str(tsrouter_path), model_col_name)
                summary_metrics_complete = all(
                    np.isfinite(float(summary_metrics.get(column, np.nan)))
                    for column in COMPETENCE_REGION_METRIC_COLUMNS
                )
                summary_process_metrics_by_task[task_key] = {
                    column: (
                        float(summary_metrics[column])
                        if np.isfinite(float(summary_metrics.get(column, np.nan)))
                        else np.nan
                    )
                    for column in COMPETENCE_REGION_METRIC_COLUMNS
                }
                if args.verbose and not summary_metrics_complete:
                    print(
                        "⚠️ Step3 competence-region summary unavailable: "
                        f"rule={summary_args.process_metrics_region_rule}, "
                        f"config={model_col_name}"
                    )

                selector_tasks.append(("TSRouter", None, tsrouter_path, model_col_name))


        param_grid91 = {
            # "repr_encoder": [
            #     "RandomMLP", "RandomPatch", "RandomConv", "RandomInception", "RandomTCN", "RandomFourier",
            #     "StatsRandomMLP", "StatsRandomPatch", "StatsRandomConv", "StatsRandomInception", "StatsRandomTCN", "StatsRandomFourier",
            #     "RandomStatsMLP", "RandomStatsPatch", "RandomStatsConv", "RandomStatsInception", "RandomStatsTCN", "RandomStatsFourier",
            # ],
            "repr_encoder": [
                # "RandomInception", "RandomMLP", "StatsRandomInception",  "StatsRandomFourier","RandomFourier",
                "StatsRandomFourier",
            ],
            "repr_input_dim": [512],
            # "repr_input_dim": [96],
            # "repr_output_dim": [128, 256, 512],
            "repr_output_dim": [256],
            # "repr_sub_pred_len": [96,192,512],
            # "repr_sub_pred_len": [48,480],
            "repr_sub_pred_len": [480],
            "repr_sample_qc_mode": ['strict',"off"],
            # "repr_sample_qc_mode": ['strict'],
            # "repr_sample_qc_mode": ["off"],
            # "repr_scale_protocol":['std','raw'],
            "repr_scale_protocol":['std'],
            # "zoo_repr_set": ['c-l-e'],
            "zoo_repr_set": ['c-e-n-h-w-s'],
            # "zoo_repr_set": ['c-l-e', 'c-l-e-n', 'c-l-e-n-h', 'c-l-e-n-h-w', 'c-l-e-n-h-w-s', 'c-e', 'c-e-n', 'c-e-n-h', 'c-e-n-h-w', 'c-e-n-h-w-s','c-l-e-n-h-w-s-f-m-t',
            #                  'e', 'e-n', 'e-n-h', 'e-n-h-w', 'e-n-h-w-s', 'c-l-e-n-w', 'c-l-e-n-s', 'c-l-e-n-m', 'c-l-e-n-t', 'c-l-e-n-f'],
            # "zoo_repr_set": ['c-l-e', 'c-l-e-n', 'c-l-e-n-h', 'c-l-e-n-w', 'c-l-e-n-s', 'c-l-e-n-m', 'c-l-e-n-t', 'c-l-e-n-f'],
            "repr_size": [1000,3000],  #
            # "repr_size": [3000],  #
            # "repr_size": [1000],  #
            # "repr_v": [5],
            "repr_v": [5],
            # "base_metrics": ["S"],
            # "base_metrics": ["C"],
            # "base_metrics": ["S", "C","M"],
            # "enable_context_len_adaptive_repr": [True,False],
            # "enable_context_len_adaptive_repr": [True],
            # "enable_pred_len_adaptive_repr": [True,False],
            # "enable_pred_len_adaptive_repr": [True],
            "base_metrics": ["S", "C"],
            # "repr_weight_ratio": [0,0.5,1],
            # "repr_weight_ratio": [0,1],
            "repr_weight_ratio": [0],
            # "ensemble_agg": ["median"],
            # "sample_repr_num": [ 20],
            # "task_sample_version":[2,1],
            # "task_sample_version":[2],
            # "repr_v5_nearest_k": [3,5,10],
            # "repr_v5_nearest_k": [10],
            # "repr_v5_distance_power": [1,3,5],
            # "rank_decay_coef":[1],
            # "restrict_top_model_num": [1,3, 9],
            # "ensemble_size": [1,3,5],
            # "ensemble_size": [1],
            # "sample_mode": ["cluster"],

            "repr_data_seed":[2025,2026,2027,2028,2029,2030],
            # "repr_data_seed":[2029],
            "repr_encoder_seed":[2025,2026,2027,2028,2029,2030],
            # "repr_encoder_seed":[2025],

        }
            
        param_grid92 = {
            "repr_encoder": [
                "StatsRandomFourier",
            ],
            # "sample_mode": ["random", "cluster_nearest"],
            "repr_input_dim": [512],
            "repr_output_dim": [256],
            "repr_sub_pred_len": [480],
            # "zoo_repr_set": ['f', 'e', 'h', 'n', 's', 't', 'w', 'c', 'l', 'm', 'o'],
            # "zoo_repr_set": ['c-e-n-h-w-s','e-n-h-w-s'],
            "zoo_repr_set": ['c-e-n-h-w-s'],
            # "zoo_repr_set": ['c-l-e'],
            "repr_size": [3000],  #
            # "repr_size": [3000,1500],  #
            # "repr_size": [1000],  #
            # "repr_v": [0,4,5],
            # "repr_v": [5],
            "repr_v": [4],
            # "repr_v": [7],
            "base_metrics": [
                # "S",
                # "M",
                "C"
            ],
            # "enable_context_len_adaptive_repr": [True,False],
            # "enable_context_len_adaptive_repr": [True],
            # "enable_pred_len_adaptive_repr": [True,False],
            # "enable_pred_len_adaptive_repr": [True],
            # "repr_weight_ratio": [0,0.5],
            "repr_weight_ratio": [0.5],
            # "repr_weight_ratio": [0,0.5,1],
            # "repr_weight_ratio": [0.2,0.5,0.7],
            # "sample_repr_num": [ 30,20,10,5,1],
            "sample_repr_num": [20],
            "repr_data_seed": [2029],
            "repr_encoder_seed": [2025],
            "repr_sample_qc_mode": ["strict"],
            # "repr_sample_qc_mode": ['strict',"off"],
            "repr_scale_protocol": ['std', ],
            # Protocol controls for the PWW->TCC breakpoint experiments.
            # Defaults keep old filenames; switch to "time_coverage"/"task_sample" to inspect aligned protocols.
            # "task_sample_strategy": ["latest_random","time_coverage"],
            # "repr_anchor_protocol": ["window","task_sample"],
            # "repr_anchor_window_sample_strategy": ["even", "random", "first", "last"],
            # "repr_anchor_window_sample_strategy": ["first","random", ],
            "repr_anchor_window_sample_strategy": ["first"],
            # "repr_anchor_window_sample_strategy": ["random", ],
            # "repr_anchor_window_sample_strategy": ["last"],

            # "task_window_sample_strategy": ["even", "random", "first", "last"],
            # "task_window_sample_strategy": ["even", "first"],
            "task_window_sample_strategy": ["even"],
            # "task_window_sample_strategy": ["first"],
            # "task_window_sample_strategy": ["legacy", "even", "first", "last"],
            # "sample_repr_ratio": [0,0.1],
            "sample_repr_ratio": [0],
            # "sample_repr_ratio": [0,0.1, 0.3, 0.5],
            # "task_rank_top3_instability_threshold": [0,0.5 ,0.7 ,0.9],
            # "task_rank_top3_instability_threshold": [-1,0,0.5,1,2,3],
            # "task_rank_top3_instability_threshold": [-1,2,3],
            "task_rank_top3_instability_threshold": [-1],
            # "repr_v5_nearest_k": [1,3, 5, 10],
            # "task_channel_fuse_limit": ["all", 1],
            "task_channel_fuse_limit": ["all"],
            # "route_family_mode": ["default", "bigger_size", "smaller_size"],
            # "route_efficiency_mode":["True","False"],
            # "autoforecast_learner":["LSTM","GBDT","MLP"]
        }
             
        param_grid922 = {
            "repr_encoder": [
                "StatsRandomFourier",
                #  "TrainFourier",
                # "SimpleTS2Vec",
                # "StatsNone",
                # "None",
                #  "StatsRandomFourier",
                #  "RandomFourier",
                # "StatsRandomMLP",
                #  "RandomMLP",
                #  "StatsRandomConv",
                #  "RandomConv",
            ],
            # "sample_mode": ["random", "cluster_nearest"],
            "repr_input_dim": [512],
            "repr_output_dim": [256],
            "repr_sub_pred_len": [480],
            # "zoo_repr_set": ['f', 'e', 'h', 'n', 's', 't', 'w', 'c', 'l', 'm', 'o'],
            # "zoo_repr_set": ['c-e-n-h-w-s','e-n-h-w-s'],
            "zoo_repr_set": ['c-e-n-h-w-s'],
            # "zoo_repr_set": ['c-l-e'],
            "repr_size": [3000],  #
            # "repr_size": [3000,1500],  #
            # "repr_size": [1000],  #
            # "repr_v": [0,4,5],
            # "repr_v": [5],
            "repr_v": [4],
            # "repr_v": [7],
            "base_metrics": [
                # "S",
                "M",
                "C"
            ],
            # "enable_context_len_adaptive_repr": [True,False],
            # "enable_context_len_adaptive_repr": [True],
            # "enable_pred_len_adaptive_repr": [True,False],
            # "enable_pred_len_adaptive_repr": [True],
            # "repr_weight_ratio": [0,0.5],
            # "repr_weight_ratio": [0.5],
            "repr_weight_ratio": [0,0.5,1],
            # "repr_weight_ratio": [0.2,0.5,0.7],
            # "sample_repr_num": [ 30,20,10,5,1],
            "sample_repr_num": [20],
            "repr_data_seed": [2029],
            "repr_encoder_seed": [2025],
            # "repr_sample_qc_mode": ["strict"],
            "repr_sample_qc_mode": ['strict',"off"],
            "repr_scale_protocol": ['std', ],
            # Protocol controls for the PWW->TCC breakpoint experiments.
            # Defaults keep old filenames; switch to "time_coverage"/"task_sample" to inspect aligned protocols.
            # "task_sample_strategy": ["latest_random","time_coverage"],
            # "repr_anchor_protocol": ["window","task_sample"],
            # "repr_anchor_window_sample_strategy": ["even", "random", "first", "last"],
            "repr_anchor_window_sample_strategy": ["first","random", ],
            # "repr_anchor_window_sample_strategy": ["first"],
            # "repr_anchor_window_sample_strategy": ["random", ],
            # "repr_anchor_window_sample_strategy": ["last"],

            # "task_window_sample_strategy": ["even", "random", "first", "last"],
            "task_window_sample_strategy": ["even", "first"],
            # "task_window_sample_strategy": ["even"],
            # "task_window_sample_strategy": ["first"],
            # "task_window_sample_strategy": ["legacy", "even", "first", "last"],
            # "sample_repr_ratio": [0,0.1],
            "sample_repr_ratio": [0],
            # "sample_repr_ratio": [0,0.1, 0.3, 0.5],
            # "task_rank_top3_instability_threshold": [0,0.5 ,0.7 ,0.9],
            # "task_rank_top3_instability_threshold": [-1,0,0.5,1,2,3],
            "task_rank_top3_instability_threshold": [-1,2,3],
            # "task_rank_top3_instability_threshold": [-1],
            # "repr_v5_nearest_k": [1,3, 5, 10],
            # "task_channel_fuse_limit": ["all", 1],
            "task_channel_fuse_limit": ["all"],
            # "route_family_mode": ["default", "bigger_size", "smaller_size"],
            # "route_efficiency_mode":["True","False"],
            # "autoforecast_learner":["LSTM","GBDT","MLP"]
        }
                  
        param_grid93 = {
            "repr_encoder": [
               "StatsRandomFourier",
               #  "TrainFourier",
                "SimpleTS2Vec",
                # "StatsNone",
                "None",
               #  "RandomFourier",
                "StatsRandomMLP",
               #  "RandomMLP",
            ],
            "sample_mode": ["random", "cluster_nearest"],
            "repr_input_dim": [512],
            "repr_output_dim": [256],
            "repr_sub_pred_len": [480],
            "zoo_repr_set": ['c-e-n-h-w-s'],
            "repr_size": [3000],  #
            "repr_v": [4],
            "base_metrics": [
                # "S",
                # "M",
                "C"
                ],

            "repr_weight_ratio": [0.5],
            "sample_repr_num": [ 20],
            "repr_data_seed":[2029],
            "repr_encoder_seed":[2025],
            "repr_sample_qc_mode": ["strict"],
            "repr_scale_protocol": ['std',],
            "repr_anchor_window_sample_strategy": ["first"],
            "task_window_sample_strategy": ["even"],
            "sample_repr_ratio": [0],
            "task_rank_top3_instability_threshold": [-1],
            "task_channel_fuse_limit": ["all"],

        }
             
        param_grid94 = {
            "repr_encoder": [
                "StatsRandomFourier",
            ],
            # "sample_mode": ["random", "cluster_nearest"],
            "repr_input_dim": [512],
            "repr_output_dim": [256],
            "repr_sub_pred_len": [480],
            # "zoo_repr_set": ['f', 'e', 'h', 'n', 's', 't', 'w', 'c', 'l', 'm', 'o'],
            "zoo_repr_set": ['c-e-n-h-w-s','e-n-h-w-s'],
            # "zoo_repr_set": ['c-e-n-h-w-s'],
            # "zoo_repr_set": ['c-l-e'],
            # "repr_size": [3000],  #
            "repr_size": [3000,1500,5000],  #
            # "repr_size": [1000],  #
            # "repr_v": [0,4,5],
            # "repr_v": [5],
            "repr_v": [4],
            # "repr_v": [7],
            "base_metrics": [
                # "S",
                # "M",
                "C"
            ],
            # "enable_context_len_adaptive_repr": [True,False],
            # "enable_context_len_adaptive_repr": [True],
            # "enable_pred_len_adaptive_repr": [True,False],
            # "enable_pred_len_adaptive_repr": [True],
            "repr_weight_ratio": [0,0.5],
            # "repr_weight_ratio": [0.5],
            # "repr_weight_ratio": [0,0.5,1],
            # "repr_weight_ratio": [0.2,0.5,0.7],
            "sample_repr_num": [ 30,20,10,5,1],
            # "sample_repr_num": [20],
            "repr_data_seed": [2029],
            "repr_encoder_seed": [2025],
            "repr_sample_qc_mode": ["strict"],
            # "repr_sample_qc_mode": ['strict',"off"],
            "repr_scale_protocol": ['std', ],
            # Protocol controls for the PWW->TCC breakpoint experiments.
            # Defaults keep old filenames; switch to "time_coverage"/"task_sample" to inspect aligned protocols.
            # "task_sample_strategy": ["latest_random","time_coverage"],
            # "repr_anchor_protocol": ["window","task_sample"],
            # "repr_anchor_window_sample_strategy": ["even", "random", "first", "last"],
            # "repr_anchor_window_sample_strategy": ["even", "first", "last"],
            # "repr_anchor_window_sample_strategy": ["last"],
            "repr_anchor_window_sample_strategy": ["first"],
            # "task_window_sample_strategy": ["even", "random", "first", "last"],
            # "task_window_sample_strategy": ["even", "random", "first"],
            "task_window_sample_strategy": ["even"],
            # "task_window_sample_strategy": ["random"],
            # "task_window_sample_strategy": ["legacy", "even", "first", "last"],
            # "sample_repr_ratio": [0,0.1],
            "sample_repr_ratio": [0],
            # "sample_repr_ratio": [0,0.1, 0.3, 0.5],
            # "task_rank_top3_instability_threshold": [0,0.5 ,0.7 ,0.9],
            # "task_rank_top3_instability_threshold": [-1,0,0.5,1,2,3],
            # "task_rank_top3_instability_threshold": [-1,2,3],
            "task_rank_top3_instability_threshold": [-1],
            # "repr_v5_nearest_k": [1,3, 5, 10],
            "task_channel_fuse_limit": ["all", 1],
            # "task_channel_fuse_limit": ["all"],
            # "route_family_mode": ["default", "bigger_size", "smaller_size"],
            # "route_efficiency_mode":["True","False"],
            # "autoforecast_learner":["LSTM","GBDT","MLP"]
        }
        if getattr(args, "vldb_results", True):
            param_grid92 = vldb_results_param_grid(args)
        elif summary_auto_cl_mode != "v0":
            profiles = get_auto_cl_profiles(summary_auto_cl_mode)
            long_profile = next(
                (
                    profile
                    for profile in profiles
                    if str(profile.get("profile_key", "")) == "long"
                ),
                profiles[-1],
            )
            param_grid92["auto_cl_mode"] = [summary_auto_cl_mode]
            param_grid92["enable_context_len_adaptive_repr"] = [True]
            param_grid92["repr_input_dim"] = [int(long_profile["repr_input_dim"])]
            param_grid92["repr_output_dim"] = [int(long_profile["repr_output_dim"])]
            param_grid92["repr_sub_pred_len"] = [int(long_profile["repr_sub_pred_len"])]
            param_grid92["repr_source_exact_length"] = [
                int(long_profile["repr_source_exact_length"])
            ]
        elif summary_fixed_cl_profile is not None:
            param_grid92["repr_input_dim"] = [
                int(summary_fixed_cl_profile["repr_input_dim"])
            ]
            param_grid92["repr_output_dim"] = [
                int(summary_fixed_cl_profile["repr_output_dim"])
            ]
            param_grid92["repr_sub_pred_len"] = [
                int(summary_fixed_cl_profile["repr_sub_pred_len"])
            ]
            param_grid92["repr_source_exact_length"] = [
                int(summary_fixed_cl_profile["repr_source_exact_length"])
            ]

        add_zoo_cast_tasks(
            selector_tasks=selector_tasks,
            args=args,
            param_grid=param_grid94
            ,
        )

                                                
        for selector_name, seed, sel_path, model_col_name in selector_tasks:
            if not sel_path.exists():
                if args.verbose:
                    print(f"⚠️ {selector_name}TSRouter runtime message: {sel_path}")
                continue

            sel_raw = check_results_file(sel_path, False, args.quick_test)
            if sel_raw is None:
                continue
            if selector_name == "TSRouter":
                print_results_file_one_line(sel_path, sel_raw, quick_test=args.quick_test)
            valid_expected_datasets = count_valid_expected_datasets(sel_raw, quick_test=args.quick_test)
            sel_datasets = set(sel_raw["dataset"].unique())

                                    
            if real_raw is not None:
                common_datasets = baseline_datasets & sel_datasets & real_datasets
            else:
                common_datasets = baseline_datasets & sel_datasets

            # if allowed_datasets is not None:
            #     common_datasets &= allowed_datasets
            if not common_datasets:
                if args.verbose:
                    print(f"⚠️ {model_col_name}TSRouter runtime message: ")
                continue

            baseline_subset = baseline_df[baseline_df["dataset"].isin(common_datasets)].copy()

            subset_df = process_results(
                sel_path,
                model_col_name,
                common_datasets,
                verbose=False,
                quick_test=args.quick_test,
                process_metric_overrides=summary_process_metrics_by_task.get(
                    (str(sel_path), model_col_name)
                ),
            )
            if getattr(args, "GE_released", False):
                subset_df = normalize_by_season_naive(subset_df, season_naive_df)
            if subset_df is None or subset_df.empty:
                if args.verbose:
                    print(f"⚠️ {model_col_name}TSRouter runtime message: ")
                continue

                                         
            df_real = None
            if real_raw is not None:
                df_real = process_results(
                    real_path, real_model_name, common_datasets, verbose=False,quick_test=args.quick_test
                )

            selector_rank_base = args.real_order_metric if selector_name in {"Real_Select", "Real_Channel_Select"} else args.rank_base
            rank_summary_all = add_order_metrics(
                baseline_subset=baseline_subset,
                subset_df=subset_df,
                model_name=model_col_name,
                rank_summary_all=rank_summary_all,
                add_index=0,
                k_order=k_order,
                df_real=df_real,
                rank_base=selector_rank_base,
                include_selector_in_rank=False,                            
                verbose=False,
            )

            if "RANK" in rank_summary_all and model_col_name in rank_summary_all["RANK"].columns:
                selector_efficiency_summary = _selector_runtime_summary_from_result(
                    sel_raw,
                    quick_test=args.quick_test,
                    tsfm_results_dir=getattr(args, "TSFM_results_dir", "cl_512"),
                )
                efficiency_note = str(selector_efficiency_summary.pop("_selector_efficiency_note", "") or "")
                rank_summary_all["RANK"].loc[SELECTOR_VALID_EXPECTED_DATASETS_COL, model_col_name] = (
                    _selector_valid_ds_with_efficiency_note(valid_expected_datasets, efficiency_note)
                )
                selector_quality_summary = {}
                selector_quality_summary.update(compute_selector_metric_rank_summary(
                    baseline_df=baseline_subset,
                    selector_df=subset_df,
                ))
                selector_quality_summary.update(compute_selector_recommendation_summary(
                    baseline_df=baseline_subset,
                    selector_df=subset_df,
                ))
                for metric_name, value in selector_quality_summary.items():
                    rank_summary_all["RANK"].loc[metric_name, model_col_name] = value
                for metric_name in SELECTOR_EFFICIENCY_METRIC_COLUMNS:
                    rank_summary_all["RANK"].loc[metric_name, model_col_name] = selector_efficiency_summary.get(metric_name, np.nan)

            selector_records.append({
                "selector_name": selector_name,
                "model_col_name": model_col_name,
                "df": subset_df.copy(),
                "current_zoo_num": int(current_zoo_num),
                "valid_expected_datasets": valid_expected_datasets,
            })

            # =========================
                                                                 
            # =========================
            if args.real_world_mode:
                                                                      
                combined_df = pd.concat([baseline_subset, subset_df], ignore_index=True)
                one_summary = caculate_combined_rank(
                    combined_df,
                    zoo_model_name=model_col_name,
                    verbose=False,
                    rank_base=selector_rank_base,
                    include_selector_in_rank=False,
                )["RANK"]                                              

                           
                col = one_summary[model_col_name]

                zcol = _extract_zcol_from_name(model_col_name)
                if zcol is None:
                    zcol = f"z{current_zoo_num}-{args.zoo_total_num}"
                rw_zcols.add(zcol)

                                                 
                if selector_name == "TSRouter":
                    row_name = _strip_zsuffix(model_col_name)  # e.g. 'w1_sub10_rs1000_c-m-t-v_v8_M'
                    rw_main_rows.add(row_name)
                elif selector_name == "Real_Select":
                    row_name = "Real"
                elif selector_name == "Real_Channel_Select":
                    row_name = "Real_Channel"
                elif selector_name == "Recent_Select":
                    row_name = "Recent"
                elif selector_name == "All_Select":
                # model_col_name: "All_mean_z9-9" or "All_median_z9-9"
                    if model_col_name.startswith("All_mean"):
                        row_name = "All_mean"
                    elif model_col_name.startswith("All_median"):
                        row_name = "All_median"
                    else:
                        row_name = "All"

                elif selector_name == "Current_best_sMAPE":
                    row_name = "Current_best_sMAPE"
                elif selector_name == "Current_best_MASE":
                    row_name = "Current_best_MASE"
                elif selector_name == "Current_best_CRPS":
                    row_name = "Current_best_CRPS"
                elif selector_name == "Random_Select":
                    row_name = f"Random_s{seed}"
                elif selector_name == "LogME_Select":
                    if model_col_name.startswith("LogME_1"):
                        row_name ="LogME_1"
                    elif model_col_name.startswith("LogME_3"):
                        row_name ="LogME_3"
                    elif model_col_name.startswith("LogME_5"):
                        row_name = "LogME_5"
                    elif model_col_name.startswith("LogME_7"):
                        row_name = "LogME_7"
                    else:
                        row_name = "LogME"
                else:
                    row_name = selector_name      

                           
                rw_store[row_name][zcol] = {
                    "Rank": float(col.get("Rank", np.nan)),
                    "sMAPE": float(col.get("sMAPE", np.nan)),
                    "MASE": float(col.get("MASE", np.nan)),
                }
                for extra_metric in TSROUTER_CORE_METRIC_COLUMNS:
                    if extra_metric in col.index:
                        rw_store[row_name][zcol][extra_metric] = float(col.get(extra_metric, np.nan))

    # =========================
              
    # =========================
    # =========================
                                                                 
    # =========================
    base_use = baseline_df_all[baseline_df_all["model"].isin(ordered_model_names)].copy()
    if getattr(args, "GE_released", False):
        base_use = normalize_by_season_naive(base_use, season_naive_df)
    top_summary_tsfm_best = compute_tsfm_best_metrics_for_summary(
        baseline_df=base_use,
        ordered_model_names=ordered_model_names,
        rank_base=args.rank_base,
    )
    tsfms = [m for m in ordered_model_names if m in base_use["model"].unique().tolist()]

    if args.enable_rebuttal_analysis:
        try:
            from analysis.rebuttal_selector_analysis import run_rebuttal_analyses
        except ImportError as exc:
            raise RuntimeError("Supplementary selector analysis is not included in the public method package.") from exc
        run_rebuttal_analyses(
            baseline_df_all=baseline_df_all,
            selector_records=selector_records,
            ordered_model_names=ordered_model_names,
            rank_base=args.rank_base,
            figs_root=args.analysis_fig_root,
            top1_check_mode=args.top1_check_mode,
            top1_check_print_limit=args.top1_check_print_limit,
        )


        # =========================
                                           
        # =========================
        def _pick_tsrouter_record(k: int):
            rows = [r for r in selector_records if r["selector_name"] == "TSRouter"]
            if not rows:
                return None
            for r in rows:
                name = str(r.get("model_col_name", "")).lower()
                if f"top{k}" in name:
                    print("[]k=",k,name)
                    return r
            return rows[0] if k == 1 else None

        def _compute_domain_rank_table(
            baseline_df: pd.DataFrame,
            selector_df: pd.DataFrame,
            selector_name: str,
            rank_base: str,
            value_metric: str,
            models: list[str],
        ) -> pd.DataFrame:
                                                 
                                                    
            if value_metric not in baseline_df.columns or value_metric not in selector_df.columns:
                raise KeyError(
                    f"value_metric={value_metric} not found in baseline/selector columns"
                )
            b = baseline_df[["dataset", "domain", "model", rank_base]].copy()
            s = selector_df[["dataset", "domain", rank_base]].copy()
            s["model"] = selector_name
            all_df = pd.concat([b, s], ignore_index=True)
            all_df["Rank"] = all_df.groupby("dataset")[rank_base].rank(method="min", ascending=True)
            metric_df = pd.concat(
                [
                    baseline_df[["dataset", "domain", "model", value_metric]],
                    selector_df[["dataset", "domain", value_metric]].assign(model=selector_name),
                ],
                ignore_index=True,
            )
            tbl = (
                metric_df.groupby(["domain", "model"])[value_metric]
                .mean()
                .reset_index()
                .merge(
                    all_df.groupby(["domain", "model"])["Rank"].mean().reset_index(),
                    on=["domain", "model"],
                    how="outer",
                )
            )
            piv_m = tbl.pivot(index="domain", columns="model", values=value_metric)
            piv_r = tbl.pivot(index="domain", columns="model", values="Rank")

            dom_cnt = baseline_df[["dataset", "domain"]].drop_duplicates().groupby("domain").size().to_dict()
            rows = []
            for d in sorted(dom_cnt.keys()):
                dn = f"{str(d).title()}  (n={int(dom_cnt[d])})"
                r1 = {"Domain": dn, "Metric": value_metric}
                r2 = {"Domain": "", "Metric": "Rank"}
                for m in models + [selector_name]:
                    r1[m] = float(piv_m.loc[d, m]) if (d in piv_m.index and m in piv_m.columns and pd.notna(piv_m.loc[d, m])) else np.nan
                    r2[m] = float(piv_r.loc[d, m]) if (d in piv_r.index and m in piv_r.columns and pd.notna(piv_r.loc[d, m])) else np.nan
                rows.extend([r1, r2])

            # ALL
            all_metric = metric_df.groupby("model")[value_metric].mean().to_dict()
            all_rank = all_df.groupby("model")["Rank"].mean().to_dict()
            all_label = f"ALL Domains  (n={int(sum(dom_cnt.values()))})"
            r1 = {"Domain": all_label, "Metric": value_metric}
            r2 = {"Domain": "", "Metric": "Rank"}
            for m in models + [selector_name]:
                r1[m] = float(all_metric.get(m, np.nan))
                r2[m] = float(all_rank.get(m, np.nan))
            rows.extend([r1, r2])
            return pd.DataFrame(rows)

        def _fmt_row_max_bold(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy().astype(object)
            for i in range(len(out)):
                vals = pd.to_numeric(df.iloc[i], errors="coerce")
                if vals.notna().sum() == 0:
                    continue
                vmax = vals.max()
                for j, col in enumerate(out.columns):
                    v = vals.get(col, np.nan)
                    if pd.isna(v):
                        out.iat[i, j] = ""
                    elif np.isclose(v, vmax, atol=1e-12):
                        out.iat[i, j] = f"**{int(v) if float(v).is_integer() else f'{v:.3f}'}**"
                    else:
                        out.iat[i, j] = f"{int(v) if float(v).is_integer() else f'{v:.3f}'}"
            return out

        def _fmt_row_min_bold_second_code(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
            out = df.copy().astype(object)
            for i in range(len(df)):
                vals = pd.to_numeric(df.loc[df.index[i], metric_cols], errors="coerce")
                valid = vals.dropna().sort_values()
                if valid.empty:
                    continue
                vmin = valid.iloc[0]
                v2 = valid.iloc[1] if len(valid) > 1 else np.nan
                for col in metric_cols:
                    j = out.columns.get_loc(col)
                    v = vals.get(col, np.nan)
                    if pd.isna(v):
                        out.iat[i, j] = ""
                    elif np.isclose(v, vmin, atol=1e-12):
                        out.iat[i, j] = f"**{v:.3f}**"
                    elif pd.notna(v2) and np.isclose(v, v2, atol=1e-12):
                        out.iat[i, j] = f"`{v:.3f}`"
                    else:
                        out.iat[i, j] = f"{v:.3f}"
            return out




                                            
        domain_counts_map = (
            base_use[["dataset", "domain"]]
            .drop_duplicates()
            .groupby("domain")
            .size()
            .to_dict()
        )
        metric_candidates = ["sMAPE", "MASE", "CRPS"]
        metrics_for_comp = [m for m in metric_candidates if m in base_use.columns]
        if args.rank_base in base_use.columns and args.rank_base not in metrics_for_comp:
            metrics_for_comp.append(args.rank_base)
        comp_tbl_by_metric = {}
        for metric_name in metrics_for_comp:
            winner = base_use.loc[
                base_use.groupby("dataset")[metric_name].idxmin()
            ][["dataset", "domain", "model"]]
            comp_tbl = winner.groupby(["domain", "model"]).size().unstack(fill_value=0)
            comp_tbl = comp_tbl.reindex(columns=tsfms, fill_value=0)
            comp_tbl.loc["ALL"] = comp_tbl.sum(axis=0)
                                    
            renamed_index = {}
            for dom in comp_tbl.index:
                if dom == "ALL":
                    renamed_index[dom] = f"**ALL Domains  (n={int(sum(domain_counts_map.values()))})**"
                else:
                    renamed_index[dom] = f"{dom}  (n={int(domain_counts_map.get(dom, 0))})"
            comp_tbl = comp_tbl.rename(index=renamed_index)
            comp_tbl_by_metric[metric_name] = comp_tbl
            print(f"\n## Rebuttal-Table-1 ({metric_name}): Complementarity (Best-distribution by Domain + ALL)")
            print(_fmt_row_max_bold(comp_tbl).to_markdown())

                                                               
        comp_tbl = comp_tbl_by_metric.get(args.rank_base)
        if comp_tbl is None and len(comp_tbl_by_metric) > 0:
            comp_tbl = comp_tbl_by_metric[next(iter(comp_tbl_by_metric.keys()))]
        if comp_tbl is None:
            print("⚠️ [rebuttal] Table-1 skipped: no metric columns found in baseline_df_all.")

                                                    
        zc1 = _pick_tsrouter_record(1)
        if zc1 is not None:
            zc1_df = zc1["df"].copy()
            for metric_name in [m for m in ["sMAPE", "MASE", "CRPS"] if m in base_use.columns and m in zc1_df.columns]:
                zc = zc1_df[["dataset", metric_name]].rename(columns={metric_name: "z"})
                comp_tbl_metric = comp_tbl_by_metric.get(metric_name)
                best_dist_all = (
                    comp_tbl_metric.loc[[idx for idx in comp_tbl_metric.index if str(idx).startswith("ALL Domains")][0]].to_dict()
                    if comp_tbl_metric is not None and any(str(idx).startswith("ALL Domains") for idx in comp_tbl_metric.index)
                    else {m: np.nan for m in tsfms}
                )
                rows = {
                    "TSFM BestCount (ALL Domains)": best_dist_all,
                    "TSRouter Win vs TSFM": {},
                    "TSRouter Tie vs TSFM": {},
                    "TSRouter Loss vs TSFM": {},
                    "Outperform-or-Tie Rate": {},
                }

                forced_tie = {}
                if args.top1_check_mode:
                    piv = base_use.pivot_table(index="dataset", columns="model", values=metric_name, aggfunc="mean")
                    zoo_s = zc.set_index("dataset")["z"]
                    for ds in sorted(set(piv.index).intersection(set(zoo_s.index))):
                        row = piv.loc[ds]
                        row = row[[m for m in tsfms if m in row.index]].dropna()
                        if row.empty:
                            continue
                        zv = float(zoo_s.loc[ds])
                        diff = row - zv
                        exact = diff[np.abs(diff) <= 1e-12].index.tolist()
                        if len(exact) >= 1:
                            forced_tie[ds] = exact[0]
                        else:
                            lower = diff[diff < 0]
                            forced_tie[ds] = lower.abs().idxmin() if len(lower) > 0 else diff.abs().idxmin()
                                                
                        # print(f\"[top1_check] ds={ds}, zv={zv}, forced_tie={forced_tie[ds]}\")

                n_eval = 0
                for m in tsfms:
                    b = base_use[base_use["model"] == m][["dataset", metric_name]].rename(columns={metric_name: "b"})
                    j = zc.merge(b, on="dataset", how="inner").dropna()
                    n_eval = max(n_eval, len(j))
                    if args.top1_check_mode:
                        tie = j["dataset"].map(lambda ds: forced_tie.get(ds, None) == m).values
                        win = (j["z"] < j["b"]).values & (~tie)
                        loss = (j["z"] > j["b"]).values & (~tie)
                    else:
                        tol = 1e-3 * np.maximum(np.abs(j["b"]), 1e-12)
                        diff = j["z"] - j["b"]
                        tie = (np.abs(diff) <= tol).values
                        win = (diff < -tol).values
                        loss = (diff > tol).values
                    rows["TSRouter Win vs TSFM"][m] = int(win.sum())
                    rows["TSRouter Tie vs TSFM"][m] = int(tie.sum())
                    rows["TSRouter Loss vs TSFM"][m] = int(loss.sum())
                    rows["Outperform-or-Tie Rate"][
                        m] = f"{((rows['TSRouter Win vs TSFM'][m] + rows['TSRouter Tie vs TSFM'][m]) * 100.0 / max(n_eval, 1)):.2f}%"

                step2_tbl1 = pd.DataFrame(rows).T[tsfms]
                if "TSFM BestCount (ALL Domains)" in step2_tbl1.index:
                    step2_tbl1.loc[["TSFM BestCount (ALL Domains)"]] = _fmt_row_max_bold(step2_tbl1.loc[["TSFM BestCount (ALL Domains)"]])
                print(f"\n## Rebuttal-Table-2.1 ({metric_name}): ALL + Win/Tie/Loss + EdgeRate")
                print(step2_tbl1.to_markdown())

                                                                                  
                step2_tbl2 = _compute_domain_rank_table(
                    baseline_df=base_use,
                    selector_df=zc1_df,
                    selector_name="TSRouter-Top1",
                    rank_base=metric_name,
                    value_metric=metric_name,
                    models=tsfms,
                )
                                           
                step2_tbl2_display = step2_tbl2.copy()
                metric_cols = [c for c in tsfms + ["TSRouter-Top1"] if c in step2_tbl2_display.columns]
                step2_tbl2_fmt = _fmt_row_min_bold_second_code(step2_tbl2_display, metric_cols=metric_cols)
                print(f"\n## Rebuttal-Table-2.2 ({metric_name}): Domain-wise {metric_name}/Rank (TSFMs + TSRouter-Top1)")
                print(step2_tbl2_fmt.to_markdown(index=False))

                                                    
        strong_targets = [
            ("Recent_3_mean_z9-9", "Recent_3_mean_z9-9"),
            ("Recent_5_mean_z9-9", "Recent_5_mean_z9-9"),
            ("Current_best_sMAPE_Rank_3_mean_z9-9", "Current_best_sMAPE_Rank_3_mean_z9-9"),
            ("Current_best_sMAPE_Rank_5_mean_z9-9", "Current_best_sMAPE_Rank_5_mean_z9-9"),
        ]
        zc3 = _pick_tsrouter_record(3)
        zc5 = _pick_tsrouter_record(5)
        strong_cols = [x[1] for x in strong_targets] + ["TSRouter-Top3", "TSRouter-Top5"]
        strong_rename = {
            "Recent_3_mean_z9-9": "Recent_3 En.",
            "Recent_5_mean_z9-9": "Recent_5 En.",
            "Current_best_sMAPE_Rank_3_mean_z9-9": "Current_best_3 En.",
            "Current_best_sMAPE_Rank_5_mean_z9-9": "Current_best_5 En.",
            "TSRouter-Top3": "TSRouter-Top3 En.",
            "TSRouter-Top5": "TSRouter-Top5 En.",
        }

        dom_cnt = base_use[["dataset", "domain"]].drop_duplicates().groupby("domain").size().to_dict()

        def _find_selector_df_by_name(target_name: str):
            for r in selector_records:
                if str(r.get("model_col_name", "")) == target_name:
                    return r["df"].copy()
            return None


        for strong_metric in [m for m in ["MASE", "CRPS"] if m in base_use.columns]:
            strong_rows = []
            for d in sorted(dom_cnt.keys()) + ["ALL"]:
                dom_name = str(d).title() if d != "ALL" else "**ALL Domains**"
                row_m = {"Domain": f"{dom_name}  (n={int(dom_cnt.get(d, sum(dom_cnt.values())))})", "Metric": strong_metric}
                row_r = {"Domain": "", "Metric": "Rank"}
                for raw_name, col_name in strong_targets:
                    sdf = _find_selector_df_by_name(raw_name)
                    if sdf is None:
                        row_m[col_name] = np.nan
                        row_r[col_name] = np.nan
                        continue
                    b = base_use if d == "ALL" else base_use[base_use["domain"] == d]
                    s = sdf if d == "ALL" else sdf[sdf["domain"] == d]
                    metric_col = strong_metric if strong_metric in s.columns else args.rank_base
                    row_m[col_name] = float(s[metric_col].mean()) if not s.empty else np.nan
                    comb = pd.concat([b[["dataset", "model", strong_metric]], s.assign(model=col_name)[["dataset", "model", strong_metric]]], ignore_index=True)
                    if comb.empty:
                        row_r[col_name] = np.nan
                    else:
                        comb["Rank"] = comb.groupby("dataset")[strong_metric].rank(method="min", ascending=True)
                        row_r[col_name] = float(comb[comb["model"] == col_name]["Rank"].mean())

                for zc_df, zc_name in [(zc3["df"].copy() if zc3 else None, "TSRouter-Top3"), (zc5["df"].copy() if zc5 else None, "TSRouter-Top5")]:
                    if zc_df is None:
                        row_m[zc_name] = np.nan
                        row_r[zc_name] = np.nan
                        continue
                    b = base_use if d == "ALL" else base_use[base_use["domain"] == d]
                    s = zc_df if d == "ALL" else zc_df[zc_df["domain"] == d]
                    metric_col = strong_metric if strong_metric in s.columns else args.rank_base
                    row_m[zc_name] = float(s[metric_col].mean()) if not s.empty else np.nan
                    comb = pd.concat([b[["dataset", "model", strong_metric]], s.assign(model=zc_name)[["dataset", "model", strong_metric]]], ignore_index=True)
                    if comb.empty:
                        row_r[zc_name] = np.nan
                    else:
                        comb["Rank"] = comb.groupby("dataset")[strong_metric].rank(method="min", ascending=True)
                        row_r[zc_name] = float(comb[comb["model"] == zc_name]["Rank"].mean())

                strong_rows.extend([row_m, row_r])

            step3_tbl = pd.DataFrame(strong_rows)
            step3_tbl = step3_tbl[["Domain", "Metric"] + strong_cols]
            step3_tbl = step3_tbl.rename(columns=strong_rename)
            metric_cols = [c for c in step3_tbl.columns if c not in ["Domain", "Metric"]]
            step3_tbl_fmt = _fmt_row_min_bold_second_code(step3_tbl, metric_cols=metric_cols)
            print(f"\n## Rebuttal-Table-3 ({strong_metric}): Strong-baseline Domain-wise {strong_metric}/Rank")
            print(step3_tbl_fmt.to_markdown(index=False))

                                       
    if args.enable_rebuttal_subset_analysis:
        # Step4
        try:
            try:
                from analysis.a_utils import (
                    load_meta, load_per_sample_metrics, get_metric_matrix,
                    rank_matrix_lower_better, build_expert_clusters_def2_center_sample_best,
                )
            except ImportError as exc:
                raise RuntimeError("Supplementary subset analysis is not included in the public method package.") from exc

            t4_tok = str(args.table4_repr_token)
            t4_rs = int(args.table4_repr_size)
            meta_pat = os.path.join(args.save_repr_data_path, f"*_{t4_tok}*{t4_rs}_*_meta.pkl")
            per_pat = os.path.join(get_tsrouter_repr_forward_dir(args), f"*_{t4_tok}*{t4_rs}_*_per_sample_results.csv")
            meta_candidates = sorted(glob.glob(meta_pat), key=os.path.getmtime, reverse=True)
            per_sample_candidates = sorted(glob.glob(per_pat), key=os.path.getmtime, reverse=True)
            if len(meta_candidates) == 0 or len(per_sample_candidates) == 0:
                raise FileNotFoundError(f"TSRouter runtime message: {meta_pat} / {per_pat}")
            meta = load_meta(meta_candidates[0])
            cluster_labels = np.asarray(meta["cluster_labels"], dtype=np.int32)
            dist2 = np.asarray(meta["distance2_to_center"], dtype=np.float32)
            df_ps, models_ps = load_per_sample_metrics(per_sample_candidates[0])
            metric_arr, models_ps = get_metric_matrix(df_ps, model_list=models_ps, metric=args.rank_base)
            models_ps = [Model_abbrev_map.get(m, m) for m in models_ps]
            n = int(min(len(cluster_labels), metric_arr.shape[0]))
            cluster_labels, dist2, metric_arr = cluster_labels[:n], dist2[:n], metric_arr[:n, :]

            counts = pd.Series(cluster_labels).value_counts().sort_values(ascending=False)
            m2j = {m: j for j, m in enumerate(models_ps)}
            table4_scan_rows = []
            cluster_top_list = sorted(set(int(x.strip()) for x in str(args.table4_top_cluster_list).split(',') if x.strip()))
            cluster_top_list = [x for x in cluster_top_list if x > 0] or [100]
            for top_cluster_n in cluster_top_list:
                keep_clusters = set(counts.head(top_cluster_n).index.tolist())
                keep_mask = np.isin(cluster_labels, np.array(list(keep_clusters), dtype=np.int32))
                cl_sub, d2_sub, arr_sub = cluster_labels[keep_mask], dist2[keep_mask], metric_arr[keep_mask, :]
                expert2idx = build_expert_clusters_def2_center_sample_best(cl_sub, d2_sub, arr_sub, models_ps)
                ranks = rank_matrix_lower_better(arr_sub)

                rows, ratio_by_k = [], {1: [], 3: [], 5: []}
                for m in tsfms:
                    if m not in m2j:
                        continue
                    j = m2j[m]
                    idx_owner = np.asarray(expert2idx.get(m, np.zeros((0,), dtype=np.int64)), dtype=np.int64)
                    row = {"Model": m}
                    for k in [1, 3, 5]:
                        g = float(np.mean(ranks[:, j] <= k)) if ranks.shape[0] > 0 else np.nan
                        o = float(np.mean(ranks[idx_owner, j] <= k)) if idx_owner.size > 0 else np.nan
                        row[f"Global@Top{k}"] = g
                        row[f"Owner@Top{k}"] = o
                        if np.isfinite(g) and np.isfinite(o) and g > 0:
                            ratio_by_k[k].append(o / g)
                    rows.append(row)

                if len(rows) == 0:
                    continue
                t4 = pd.DataFrame(rows).set_index("Model")
                for k in [1, 3, 5]:
                    gcol, ocol, bcol = f"Global@Top{k}", f"Owner@Top{k}", f"Boost↑@Top{k}"
                    t4[bcol] = np.where(t4[gcol] > 0, (t4[ocol] / t4[gcol] - 1.0) * 100.0, np.nan)
                t4.loc["AVG(9 TSFMs)"] = [float(np.nanmean(t4[c].astype(float).values)) for c in t4.columns]
                t4 = t4[[f"Global@Top{k}" for k in [1, 3, 5]] + [f"Owner@Top{k}" for k in [1, 3, 5]] + [f"Boost↑@Top{k}" for k in [1, 3, 5]]]
                t4_show = t4.copy()
                for c in t4_show.columns:
                    if c.startswith("Boost↑"):
                        t4_show[c] = t4_show[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}%")
                    else:
                        t4_show[c] = t4_show[c].map(lambda x: "" if pd.isna(x) else f"{100.0 * x:.2f}%")
                print(f"\n## Rebuttal-Table-4 ({args.rank_base}, TopClusters={top_cluster_n}): Owner Top-k Coverage vs Global Top-k")
                print(t4_show.to_markdown())

                table4_scan_rows.append({
                    "TopClusters": int(top_cluster_n),
                    "MeanRatio@Top1": float(np.mean(ratio_by_k[1])) if ratio_by_k[1] else np.nan,
                    "MeanRatio@Top3": float(np.mean(ratio_by_k[3])) if ratio_by_k[3] else np.nan,
                    "MeanRatio@Top5": float(np.mean(ratio_by_k[5])) if ratio_by_k[5] else np.nan,
                })
            if table4_scan_rows:
                scan = pd.DataFrame(table4_scan_rows).sort_values("MeanRatio@Top1", ascending=False)
                print("\n## Rebuttal-Table-4 Sweep Summary: sorted by MeanRatio@Top1")
                print(scan.to_markdown(index=False, floatfmt='.4f'))
        except Exception as e:
            print(f"\n⚠️ Rebuttal-Table-4 skipped: {e}")

        # Step5
        token_order = ["f", "e", "h", "n", "s", "t", "w"]
        target_domain_token = set(token_order)
        full_token = "-".join(token_order)
        keep_tokens = [full_token] + ["-".join([x for x in token_order if x != d]) for d in token_order]
        domain_col_order = [
            ("econ/fin", "Econ/Fin"), ("energy", "Energy"), ("healthcare", "Healthcare"),
            ("nature", "Nature"), ("sales", "Sales"), ("transport", "Transport"), ("web/cloudops", "Web/CloudOps"),
        ]
        table5_csv_pattern = os.path.join(
            get_tsrouter_selector_result_dir(args),
            "zoo9-9_RandomMLP_96to256_pl96_*_sd2025_se2025_{}[*]3000_kmeans-n_sf2025-v4S_repr-all_sub0_1.0_w{}_min_task5_ss2025_top{}-median_res1*all_results.csv",
        )


        def _repr_row_name(tok: str):
            if tok == full_token:
                return "Full (f-e-h-n-s-t-w)"
            miss = sorted(list(target_domain_token - set(tok.split("-"))))
            return f"w/o {miss[0]}" if len(miss) == 1 else f"Custom ({tok})"


        def _norm_domain(v: str):
            s = str(v).strip().lower().replace(" ", "")
            return {
                "econ_fin": "econ/fin", "econ-fin": "econ/fin", "webcloudops": "web/cloudops",
                "web/cloud_ops": "web/cloudops", "web/cloud-ops": "web/cloudops",
            }.get(s, s)


        table5_ablation_rows = []
        for top_k in [1, 3, 5]:
            for wv in [0.0, 1.0]:
                rows5 = []
                for tok in keep_tokens:
                    w_token = f"{wv:.1f}"
                    pat = table5_csv_pattern.format(tok, w_token, top_k)
                    hits = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
                    if len(hits) == 0:
                        legacy = os.path.join(get_tsrouter_selector_result_dir(args),
                                              f"zoo9-9_RandomMLP_96to256_pl96_*_sd2025_se2025_{tok}_x3000*_w{w_token}*_top{top_k}*all_results.csv")
                        hits = sorted(glob.glob(legacy), key=os.path.getmtime, reverse=True)
                    if len(hits) == 0:
                        continue
                    dfv = check_results_file(Path(hits[0]), verbose=False, quick_test=args.quick_test)
                    if dfv is None or dfv.empty:
                        continue
                    dfv = dfv.copy()
                    dfv["domain_norm"] = dfv["domain"].map(_norm_domain)
                    row = {"Config": _repr_row_name(tok)}
                    vals = []
                    for dk, dd in domain_col_order:
                        ds = dfv[dfv["domain_norm"] == dk]
                        val = float(ds[args.rank_base].mean()) if not ds.empty else np.nan
                        row[dd] = val
                        vals.append(val)
                    row["DomainMean"] = float(np.nanmean(np.array(vals, dtype=float))) if vals else np.nan
                    rows5.append(row)
                if not rows5:
                    continue
                tbl5 = pd.DataFrame(rows5).drop_duplicates(subset=["Config"], keep="first")
                row_order = ["Full (f-e-h-n-s-t-w)"] + [f"w/o {d}" for d in token_order]
                tbl5["__ord"] = tbl5["Config"].map(lambda x: row_order.index(x) if x in row_order else 999)
                tbl5 = tbl5.sort_values("__ord").drop(columns=["__ord"])
                print(f"\n## Rebuttal-Table-5 (top{top_k}, w={wv:g}, metric={args.rank_base}): domain-wise zoo_repr_set ablation")
                print(tbl5.to_markdown(index=False, floatfmt=".4f"))
                tmp = tbl5[["Config", "DomainMean"]].copy()
                tmp["TopK"] = int(top_k)
                tmp["Weight"] = float(wv)
                table5_ablation_rows.append(tmp)
        if table5_ablation_rows:
            ablation_long = pd.concat(table5_ablation_rows, ignore_index=True)
            ablation_long["Cell"] = ablation_long.apply(lambda r: f"Top{int(r['TopK'])}_w{r['Weight']:g}", axis=1)
            ablation_tbl = ablation_long.pivot_table(index="Config", columns="Cell", values="DomainMean", aggfunc="mean")
            ablation_tbl["OverallMean"] = ablation_tbl.mean(axis=1, skipna=True)
            print("\n## Rebuttal-Table-5 Summary: repr-source ablation (DomainMean aggregation)")
            print(ablation_tbl.to_markdown(floatfmt=".4f"))


        # Step6
        def _load_subset_members(model_repr_path: str):
            if any(ch in model_repr_path for ch in "*?[]"):
                hits = sorted(glob.glob(model_repr_path), key=os.path.getmtime, reverse=True)
                if not hits:
                    return None, None
                model_repr_path = hits[0]
            subset_path = model_repr_path.replace('.pkl', '_subset_assign.pkl')
            if os.path.exists(subset_path):
                with open(subset_path, 'rb') as f:
                    obj = pickle.load(f)
                sid = obj.get('selected_indices_dict', {}) if isinstance(obj, dict) else {}
                return {k: set(np.asarray(v, dtype=np.int64).tolist()) for k, v in sid.items()}, 'index'
            if not os.path.exists(model_repr_path):
                return None, None
            with open(model_repr_path, 'rb') as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                out = {}
                for k, arr in obj.items():
                    a = np.asarray(arr)
                    if a.ndim == 1:
                        a = a.reshape(1, -1)
                    if a.ndim == 2:
                        out[k] = set(tuple(np.round(r, 6).tolist()) for r in a)
                return out, 'signature'
            return None, None


        t6_paths = {
            'BASE': os.path.join(TSROUTER_MODEL_REPR_DIR, 'zoo9-9_RandomMLP_96to256_pl96_*_sd2025_se2025_c-l-e_x3000_kmeans-n_sf2025-v4S_repr-all_sub0_1.0.pkl'),
            'ALT1': os.path.join(TSROUTER_MODEL_REPR_DIR, 'zoo9-9_RandomMLP_96to256_pl96_*_sd2025_se2025_c-e_x3000_kmeans-n_sf2025-v4S_repr-all_sub0_1.0.pkl'),
            'ALT2': os.path.join(TSROUTER_MODEL_REPR_DIR, 'zoo9-9_RandomMLP_96to256_pl96_*_sd2025_se2025_l-e_x3000_kmeans-n_sf2025-v4S_repr-all_sub0_1.0.pkl'),
            'ALT3': os.path.join(TSROUTER_MODEL_REPR_DIR, 'zoo9-9_RandomMLP_96to256_pl96_*_sd2025_se2025_c-l-e_x3000_kmeans-n_sf2025-v4C_repr-all_sub0_1.0.pkl'),
        }
        base_members, base_mode = _load_subset_members(t6_paths['BASE'])
        if base_members is not None:
            alt_cache = {k: _load_subset_members(v)[0] for k, v in t6_paths.items() if k != 'BASE'}
            rows6 = []
            for m in tsfms:
                b = base_members.get(m, set())
                row = {'Model': m, 'BASE_size': len(b)}
                for tag in ['ALT1', 'ALT2', 'ALT3']:
                    mem = alt_cache.get(tag)
                    if mem is None:
                        row[f'{tag}_size'] = np.nan
                        row[f'{tag}_overlap%'] = np.nan
                    else:
                        a = mem.get(m, set())
                        row[f'{tag}_size'] = len(a)
                        row[f'{tag}_overlap%'] = (len(b & a) / max(len(b), 1)) * 100.0
                rows6.append(row)
            t6 = pd.DataFrame(rows6).set_index('Model')
            for c in t6.columns:
                if c.endswith('overlap%'):
                    t6[c] = t6[c].map(lambda x: '' if pd.isna(x) else f"{x:.2f}%")
            print(f"\n## Rebuttal-Table-6: Advantage-subset size & overlap (base_mode={base_mode})")
            print(t6.to_markdown())
        else:
            print(f"\n⚠️ Rebuttal-Table-6 skipped: base file missing or unreadable -> {t6_paths['BASE']}")

    if args.real_world_mode:
        zcols_sorted = sorted(list(rw_zcols), key=_z_sort_key)

        # =========================
              
        # =========================
        SHOW_Z9_RANK_GT = True
        Z9_RANK_TH = 3.7

        SHOW_RANDOM_SEEDS = False                                                          
        ADD_RANDOM_STD = False

        # =========================
                                                   
        # =========================
        FIXED_BASE_ROWS = [
            "Real", "Recent", "All_mean", "All_median",
            "Current_best_sMAPE", "Current_best_MASE", "Current_best_CRPS",
            "LogME_1_Select","LogME_3_Select","LogME_5_Select","LogME_7_Select",
        ]

                                                             
        random_seed_rows = sorted([r for r in rw_store.keys() if str(r).startswith("Random_s")])

                                     
        main_rows_sorted = sorted([
            r for r in rw_store.keys()
            if (r not in FIXED_BASE_ROWS) and (not str(r).startswith("Random_s"))
        ])

                              
        tail_rows = [r for r in FIXED_BASE_ROWS if r in rw_store]

                                                 
        final_rows_for_build = main_rows_sorted + random_seed_rows + tail_rows


        def build_metric_table(metric: str, rows: list[str]) -> pd.DataFrame:
            out = pd.DataFrame(index=rows, columns=zcols_sorted, dtype=float)
            for r in rows:
                for z in zcols_sorted:
                    out.loc[r, z] = rw_store.get(r, {}).get(z, {}).get(metric, np.nan)
            return out


                                                           
        rank_tbl = build_metric_table("Rank", final_rows_for_build)
        smape_tbl = build_metric_table("sMAPE", final_rows_for_build)
        mase_tbl = build_metric_table("MASE", final_rows_for_build)

                                                                                          
        if len(random_seed_rows) > 0:
            rank_tbl.loc["Random_mean", zcols_sorted] = rank_tbl.loc[random_seed_rows, zcols_sorted].mean(axis=0, skipna=True).values
            smape_tbl.loc["Random_mean", zcols_sorted] = smape_tbl.loc[random_seed_rows, zcols_sorted].mean(axis=0, skipna=True).values
            mase_tbl.loc["Random_mean", zcols_sorted] = mase_tbl.loc[random_seed_rows, zcols_sorted].mean(axis=0, skipna=True).values

            if ADD_RANDOM_STD:
                rank_tbl.loc["Random_std", zcols_sorted] = rank_tbl.loc[random_seed_rows, zcols_sorted].std(axis=0, ddof=0, skipna=True).values
                smape_tbl.loc["Random_std", zcols_sorted] = smape_tbl.loc[random_seed_rows, zcols_sorted].std(axis=0, ddof=0, skipna=True).values
                mase_tbl.loc["Random_std", zcols_sorted] = mase_tbl.loc[random_seed_rows, zcols_sorted].std(axis=0, ddof=0, skipna=True).values

                              
            if not SHOW_RANDOM_SEEDS:
                rank_tbl = rank_tbl.drop(index=random_seed_rows)
                smape_tbl = smape_tbl.drop(index=random_seed_rows)
                mase_tbl = mase_tbl.drop(index=random_seed_rows)

        else:
            print('TSRouter runtime message.')

                                                     
        def _wins_vs_best_tsfm(metric_tbl: pd.DataFrame, metric: str) -> pd.Series:
            wins_by_row = pd.Series(0, index=metric_tbl.index, dtype=float)
            missing_z = []
            for z in zcols_sorted:
                best = rw_best_tsfm_by_z.get(z, {}).get(metric, np.nan)
                if not np.isfinite(best):
                    missing_z.append(z)
                    continue
                for r in metric_tbl.index:
                    a = metric_tbl.loc[r, z]
                    if np.isnan(a):
                        continue
                    if a <= best:                                                  
                        wins_by_row.loc[r] += 1
            if missing_z:
                print(f"⚠️ [Real-World] {metric}TSRouter runtime message: {missing_z[:8]}")
            return wins_by_row

        rank_win_count = _wins_vs_best_tsfm(rank_tbl, "Rank")
        smape_win_count = _wins_vs_best_tsfm(smape_tbl, "sMAPE")
        mase_win_count = _wins_vs_best_tsfm(mase_tbl, "MASE")

                                                   
        prefer_z9 = "z9-9"
        z_last = prefer_z9 if prefer_z9 in zcols_sorted else (zcols_sorted[-1] if len(zcols_sorted) > 0 else None)
        if z_last is None:
            print('TSRouter runtime message.')

                                                    
        rank_tbl = rank_tbl.copy()
        smape_tbl = smape_tbl.copy()
        mase_tbl = mase_tbl.copy()

        rank_tbl["Win_vs_Best_TSFM"] = rank_win_count
        smape_tbl["Win_vs_Best_TSFM"] = smape_win_count
        mase_tbl["Win_vs_Best_TSFM"] = mase_win_count

                                                        
        if (not SHOW_Z9_RANK_GT) and (z_last is not None) and (z_last in rank_tbl.columns):
                                               
            protect = {
                "Real", "Recent","All_mean", "All_median",
                "Current_best_sMAPE", "Current_best_MASE", "Current_best_CRPS",
                "LogME_1_Select","LogME_3_Select","LogME_5_Select","LogME_7_Select",
                "Random_mean", "Random_std",
            }

            mask_bad = rank_tbl[z_last].apply(lambda x: (np.isfinite(x) and x > Z9_RANK_TH))
            drop_rows = [r for r in rank_tbl.index if (r not in protect) and bool(mask_bad.loc[r])]
            if drop_rows:
                rank_tbl = rank_tbl.drop(index=drop_rows)
                smape_tbl = smape_tbl.drop(index=drop_rows)
                mase_tbl = mase_tbl.drop(index=drop_rows)

                                                                                
                                                     
        if z_last is not None and z_last in rank_tbl.columns:
            rank_tbl["_z9_sort_tmp"] = rank_tbl[z_last].fillna(1e9)
            sort_cols = ["Win_vs_Best_TSFM", "_z9_sort_tmp"]
            asc = [False, True]
        else:
                                    
            rank_tbl["_mean_rank_tmp"] = rank_tbl[zcols_sorted].mean(axis=1, skipna=True).fillna(1e9)
            sort_cols = ["Win_vs_Best_TSFM", "_mean_rank_tmp"]
            asc = [False, True]

        rank_tbl_sorted = rank_tbl.sort_values(sort_cols, ascending=asc)

               
        for tmp in ["_z9_sort_tmp", "_mean_rank_tmp"]:
            if tmp in rank_tbl_sorted.columns:
                rank_tbl_sorted = rank_tbl_sorted.drop(columns=[tmp])

        row_order = rank_tbl_sorted.index.tolist()

                                                
        keep_cols = zcols_sorted + ["Win_vs_Best_TSFM"]
        rank_tbl_sorted = rank_tbl_sorted.loc[row_order, keep_cols]
        smape_tbl_sorted = smape_tbl.loc[row_order, keep_cols]
        mase_tbl_sorted = mase_tbl.loc[row_order, keep_cols]

                                 
        print("\n" + "=" * 60 + f"TSRouter runtime message: {z_last}↑）\n" + "=" * 60)
        print(tabulate(rank_tbl_sorted, headers="keys", tablefmt="plain",
                       floatfmt=".3f", numalign="decimal", stralign="left"))

        print("\n" + "=" * 60 + 'TSRouter runtime message.' + "=" * 60)
        print(tabulate(smape_tbl_sorted, headers="keys", tablefmt="plain",
                       floatfmt=".3f", numalign="decimal", stralign="left"))

        print("\n" + "=" * 60 + 'TSRouter runtime message.' + "=" * 60)
        print(tabulate(mase_tbl_sorted, headers="keys", tablefmt="plain",
                       floatfmt=".3f", numalign="decimal", stralign="left"))

        if getattr(args, "vldb_results", True):
            run_vldb_results_fast_baselines(
                args=args,
                baseline_df_all=baseline_df_all,
                ordered_model_names=ordered_model_names,
                season_naive_df=season_naive_df,
            )
        if getattr(args, "channel_failure_analysis", False):
            run_channel_failure_analysis_for_check_selector(args=args, print_tables=True)

        raise SystemExit(0)


    else:
        for rank_type, df_summary in rank_summary_all.items():
                                                      
            random_cols = [
                col for col in df_summary.columns if col.startswith("Rt") and not col.endswith("m")
            ]
            cols_to_show = [col for col in df_summary.columns if col not in random_cols]
            preferred_metric_order = [
                "MASE",
                "Regret-M",
                "Regret-M P90",
                "Rank-M",
                "MASE-hit1",
                "MASE-hit3",
                "CRPS",
                "Rank-C",
                "CRPS-hit1",
                "CRPS-hit3",
                "sMAPE",
                "Count-1",
                "Count-2",
                *SELECTOR_EFFICIENCY_METRIC_COLUMNS,
                SELECTOR_VALID_EXPECTED_DATASETS_COL,
            ] + TSROUTER_CORE_METRIC_COLUMNS
            row_order = (
                [r for r in preferred_metric_order if r in df_summary.index]
                + [r for r in df_summary.index if r not in preferred_metric_order and r != "Rank"]
            )
            df_summary_to_print = df_summary.reindex(row_order)[cols_to_show]

                                                 
            df_summary_display = build_selector_display_summary(df_summary_to_print)
            df_summary_transposed = df_summary_display.T
            df_summary_transposed.index.name = "Metric"

            print('TSRouter runtime message.')
            print(
                tabulate(
                    df_summary_transposed,
                    headers="keys",
                    tablefmt="plain",
                    floatfmt=".3f",
                    numalign="decimal",
                    stralign="left",
                    showindex=True,
                )
            )
                                                              
            try:
                from cli.auto_search_tsrouter import analyze_param_effects
            except ImportError:
                from auto_search_tsrouter import analyze_param_effects
            analyze_param_effects(
                df_summary_to_print=df_summary_to_print,
                # focus_rank_col="Rank",
                # focus_secondary_cols=["sMAPE", "MASE"],
                # focus_rank_col="Rank",
                # focus_secondary_cols=[ "MASE","sMAPE"],
                focus_rank_col="MASE",
                focus_secondary_cols=["Rank-M", "CRPS", "Rank-C", "sMAPE"],
                topn=100,
                verbose=True,
                best_tsfm_metrics=top_summary_tsfm_best,
            )

        if getattr(args, "vldb_results", True):
            run_vldb_results_fast_baselines(
                args=args,
                baseline_df_all=baseline_df_all,
                ordered_model_names=ordered_model_names,
                season_naive_df=season_naive_df,
            )
        if getattr(args, "channel_failure_analysis", False):
            run_channel_failure_analysis_for_check_selector(args=args, print_tables=True)

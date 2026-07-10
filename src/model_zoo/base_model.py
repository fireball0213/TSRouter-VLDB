import os
import csv
import gc
import json
import time
import warnings
import random
import ast
import pickle
import re
import shutil
import threading
from pathlib import Path
from dotenv import load_dotenv
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from datetime import datetime
from pandas.tseries.frequencies import to_offset

from gluonts.model import evaluate_forecasts as gluonts_evaluate_forecasts
from gluonts.time_feature import get_seasonality
from gluonts.time_feature import norm_freq_str
from gluonts.ev.metrics import (
    MSE,
    MAE,
    MASE,
    MAPE,
    SMAPE,
    MSIS,
    RMSE,
    NRMSE,
    ND,
    MeanWeightedSumQuantileLoss,
)

from utils.data import Dataset, FastEvalDatasetCacheLoader
from utils.data import M4_PRED_LENGTH_MAP, PRED_LENGTH_MAP
from utils.data import Dataset
from utils.debug import debug_check_input_nan, debug_print_test_input, debug_forecasts, debug_forecast_brief, debug_dataset_brief, debug_predictor_brief

from config.model_zoo_config import (
    All_sorted_model_names,
    Model_abbrev_map,
    MULTIVAR_TSFM_PREFIXES,
    get_model_default_batch_size,
)
from selector.selector_config import Selector_zoo_details
from utils.io_lock import file_lock
from utils.gift_eval import evaluate_forecasts_fast, get_cached_seasonal_errors
from utils.tsrouter_metrics import (
    TSROUTER_EXTRA_METRIC_COLUMNS,
    TSROUTER_CORE_METRIC_COLUMNS,
    TSROUTER_REQUIRED_PROCESS_METRIC_COLUMNS,
    TSROUTER_ROUTE_DETAIL_COLUMNS,
    build_channel_rank_cache,
    channel_orders_from_error_matrix,
    compute_single_series_recommendation_metrics,
    compute_single_series_recommendation_metrics_from_orders,
    compute_window_channel_task_process_metrics,
    compute_per_channel_metric_rows,
    compute_per_window_metric_rows,
    load_encoder_enrichment_for_args,
    load_channel_rank_orders,
    load_per_channel_error_matrix,
    nan_metric_row,
    parse_order_string,
    per_channel_results_path,
    per_window_results_path,
    real_channel_rank_cache_path,
    resolve_rank_truth_cl,
    save_channel_rank_orders,
    save_per_channel_metric_rows,
    save_per_window_metric_rows,
)
from utils.path_utils import (
    auto_cl_enabled,
    get_auto_cl_mode,
    get_auto_cl_profile_by_name,
    get_gift_eval_task_repr_cache_path,
    get_tsrouter_repr_forward_dir,
    get_tsrouter_selector_stage_result_dir,
    materialize_compatible_tsrouter_result,
    normalize_route_family_mode,
    route_efficiency_mode_enabled,
    resolve_tsfm_artifact_path,
    resolve_tsfm_csv_path,
    tsfm_artifact_dir,
    tsfm_csv_dir,
)
from utils.project_paths import (
    BASELINE_CSV_ROOT,
    SAMPLED_REPR_POOL_CACHE_ROOT,
    TSFM_CSV_ROOT,
    TSROUTER_REPR_FORWARD_ARTIFACT_ROOT,
)
from utils.project_paths import CHANNEL_META_PATH, DATASET_PROPERTIES_PATH

warnings.filterwarnings("ignore")
load_dotenv()          

                                                       

                           
dataset_properties_map = json.load(open(DATASET_PROPERTIES_PATH, encoding="utf-8"))

        
GE_QUANTILES = [0.1 * i for i in range(1, 10)]
DEFAULT_METRICS = [
    MASE(),
    SMAPE(),
    MeanWeightedSumQuantileLoss(quantile_levels=GE_QUANTILES),
]
GE_RELEASED_METRICS = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=GE_QUANTILES),
]

        
TIMING_COLUMNS = [
    "runtime_seconds",
    "sample_seconds",
    "sample_to_route_seconds",
    "route_final_seconds",
    "insert_runtime_seconds",
    "repr_data_load_seconds",
    "forward_runtime_seconds",
    "eval_runtime_seconds",
    "per_sample_metric_seconds",
    "per_sample_metric_save_seconds",
    "metric_read_seconds",
    "non_eval_runtime_seconds",
    "eval_skipped",
    "evaluation_mode",
    "cli_batch_size",
    "batch_size_source",
    "requested_batch_size",
    "runtime_batch_size",
    "min_batch_size",
    "batch_size_fallback_count",
    "requested_context_length",
    "runtime_context_length",
    "min_context_length",
    "context_length_fallback_count",
    "memory_use_mb",
    "gpu_memory_source",
    "nvml_process_memory_mb",
    "nvml_process_peak_memory_mb",
    "torch_memory_allocated_mb",
    "torch_memory_reserved_mb",
    "torch_memory_peak_allocated_mb",
    "torch_memory_peak_reserved_mb",
]

TSROUTER_STEP4_PROVENANCE_COLUMNS = [
    "auto_cl_mode",
    "route_family_mode",
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
    "task_sample_cache_hit",
    "sample_timing_source",
    "eval_cl_fallback_used",
    "adaptive_task_term_fallback_used",
]

pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

TASK_PROBE_SAMPLE_FORWARD_COLUMNS = [
    "timestamp_utc",
    "stage",
    "dataset",
    "sample_repr_num",
    "task_window_sample_strategy",
    "sample_repr_ratio",
    "task_sample_version",
    "search_seed",
    "repr_scale_protocol",
    "model_id",
    "model_key",
    "model_abbr",
    "selected_windows",
    "selected_entry_indices",
    "model_load_ms",
    "forward_ms",
    "sample_forward_ms",
    "evaluate_ms",
    "metric_read_ms",
    "status",
    "error",
    "step4_task_cache_path",
]

TASK_PROBE_SAMPLE_ERROR_COLUMNS = [
    "timestamp_utc",
    "stage",
    "dataset",
    "sample_repr_num",
    "task_window_sample_strategy",
    "sample_repr_ratio",
    "task_sample_version",
    "search_seed",
    "repr_scale_protocol",
    "model_id",
    "model_key",
    "model_abbr",
    "local_entry",
    "step4_entry_idx",
    "series_id",
    "item_id",
    "forecast_start",
    "input_start",
    "channel",
    "window_id",
    "pred_len",
    "mase_lag",
    "MASE",
    "sMAPE",
    "CRPS",
    "MASE_NUM",
    "MASE_DEN",
    "SMAPE_NUM",
    "SMAPE_DEN",
    "CRPS_NUM",
    "CRPS_DEN",
    "step4_task_cache_path",
]


class _GpuMemoryMonitor:
    def __init__(self, poll_interval_seconds: float = 0.2):
        self.poll_interval_seconds = max(0.05, float(poll_interval_seconds))
        self.pid = os.getpid()
        self.peak_mb = float("nan")
        self.current_mb = float("nan")
        self.source = "nvml_unavailable"
        self._stop = threading.Event()
        self._thread = None
        self._nvml = None
        self._handle = None

    def _init_nvml(self) -> bool:
        if not torch.cuda.is_available():
            self.source = "cuda_unavailable"
            return False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            current_device = int(torch.cuda.current_device())
            visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
            token = None
            if visible and visible not in {"-1", "none", "None"}:
                tokens = [part.strip() for part in visible.split(",") if part.strip()]
                if current_device < len(tokens):
                    token = tokens[current_device]
            if token and token.startswith("GPU-"):
                self._handle = pynvml.nvmlDeviceGetHandleByUUID(token.encode("utf-8"))
            else:
                index = int(token) if token is not None else current_device
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.source = "nvml_process_peak"
            return True
        except Exception as exc:
            self.source = f"nvml_unavailable:{type(exc).__name__}"
            return False

    def _sample_mb(self) -> float:
        pynvml = self._nvml
        if pynvml is None or self._handle is None:
            return float("nan")
        process_lists = []
        for fn_name in [
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
        ]:
            fn = getattr(pynvml, fn_name, None)
            if fn is None:
                continue
            try:
                process_lists.append(fn(self._handle))
                break
            except Exception:
                continue
        for fn_name in [
            "nvmlDeviceGetGraphicsRunningProcesses_v3",
            "nvmlDeviceGetGraphicsRunningProcesses_v2",
            "nvmlDeviceGetGraphicsRunningProcesses",
        ]:
            fn = getattr(pynvml, fn_name, None)
            if fn is None:
                continue
            try:
                process_lists.append(fn(self._handle))
                break
            except Exception:
                continue
        used = 0.0
        found = False
        for processes in process_lists:
            for proc in processes or []:
                if int(getattr(proc, "pid", -1)) != self.pid:
                    continue
                value = getattr(proc, "usedGpuMemory", None)
                try:
                    value = float(value)
                except Exception:
                    continue
                if value < 0:
                    continue
                used += value / 1024 ** 2
                found = True
        return used if found else float("nan")

    def _poll(self) -> None:
        while not self._stop.is_set():
            value = self._sample_mb()
            if np.isfinite(float(value)):
                self.current_mb = float(value)
                if not np.isfinite(float(self.peak_mb)) or float(value) > float(self.peak_mb):
                    self.peak_mb = float(value)
            self._stop.wait(self.poll_interval_seconds)

    def start(self):
        if not self._init_nvml():
            return self
        first = self._sample_mb()
        if np.isfinite(float(first)):
            self.current_mb = float(first)
            self.peak_mb = float(first)
        self._thread = threading.Thread(target=self._poll, name="tsfm-gpu-memory-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.poll_interval_seconds * 2))
        final = self._sample_mb()
        if np.isfinite(float(final)):
            self.current_mb = float(final)
            if not np.isfinite(float(self.peak_mb)) or float(final) > float(self.peak_mb):
                self.peak_mb = float(final)
        try:
            if self._nvml is not None:
                self._nvml.nvmlShutdown()
        except Exception:
            pass
        return {
            "nvml_process_memory_mb": self.current_mb,
            "nvml_process_peak_memory_mb": self.peak_mb,
            "gpu_memory_source": self.source,
        }


class BaseModel:
    _ge_fast_dataset_shared_cache: dict[tuple, dict] = {}
    _stage_rank_truth_logged_keys: set[tuple[str, int]] = set()

    def _resolve_initial_batch_size(self) -> int:
        cli_batch_size = int(getattr(self.args, "batch_size", 0) or 0)
        if cli_batch_size <= 0:
            cli_batch_size = 1
        self._cli_batch_size = cli_batch_size
        self._initial_batch_size_source = "cli_batch_size"
        if bool(getattr(self.args, "use_model_default_batch_size", True)):
            configured = get_model_default_batch_size(self.model_name, fallback=None)
            if configured is not None and int(configured) > 0:
                self._initial_batch_size_source = "model_zoo_config"
                return int(configured)
        return cli_batch_size

    def __init__(self, model_name, args, output_dir=None):
        self.args = args
        self.model_name = model_name
        if output_dir is None:
            self.output_dir = args.output_dir
        self.artifact_output_dir = None
        self.batch_size = self._resolve_initial_batch_size()
        if int(getattr(self, "_cli_batch_size", self.batch_size)) != int(self.batch_size):
            print(
                f"[batch-size] {self.model_name}: initial={self.batch_size} "
                f"source={self._initial_batch_size_source}, cli_fallback={self._cli_batch_size}"
            )

        self.per_sample_csv_file_path = None
        self.get_save_path()
        if self.per_sample_csv_file_path:
            print('Save Path (main + per-sample): ',self.csv_file_path)
        else:
            print('Save Path (main):',self.csv_file_path)

        self.done_datasets = []
                                                                 
                                        
        if self.args.clean_saved:
                                                             
            if os.path.exists(self.csv_file_path):
                with file_lock(self.csv_file_path + ".lock"):
                    with open(self.csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(self.csv_header)
                print(f"[clean_saved] cleaned main results: {self.csv_file_path}")

                                              
                                                                                                                            
            if self.args.run_mode == "zoo":
                per_channel_path = per_channel_results_path(self.output_dir)
                if os.path.exists(per_channel_path):
                    with file_lock(per_channel_path + ".lock"):
                        if os.path.exists(per_channel_path):
                            os.remove(per_channel_path)
                    print(f"[clean_saved] removed per-channel results: {per_channel_path}")

                                                   
                per_channel_tmp_path = per_channel_path + ".tmp"
                if os.path.exists(per_channel_tmp_path):
                    os.remove(per_channel_tmp_path)
                    print(f"[clean_saved] removed per-channel tmp: {per_channel_tmp_path}")

                self._invalidate_per_channel_status_cache()
                per_window_path = per_window_results_path(self.output_dir)
                if os.path.exists(per_window_path):
                    with file_lock(per_window_path + ".lock"):
                        if os.path.exists(per_window_path):
                            os.remove(per_window_path)
                    print(f"[clean_saved] removed per-window results: {per_window_path}")

                per_window_tmp_path = per_window_path + ".tmp"
                if os.path.exists(per_window_tmp_path):
                    os.remove(per_window_tmp_path)
                    print(f"[clean_saved] removed per-window tmp: {per_window_tmp_path}")

                self._invalidate_per_window_status_cache()
        if self.args.skip_saved:
            if os.path.exists(self.csv_file_path):
                df_res = pd.read_csv(self.csv_file_path)
                if "dataset" in df_res.columns:
                    self.done_datasets = df_res["dataset"].values

                    print(f"Existing main result rows: {len(self.done_datasets)} ,",end=' ')
            else:
                print(f"TSRouter runtime message: {self.csv_file_path}")


    # ==============================================================
                  
    # ==============================================================
    def _reuse_same_stage_tsrouter_selector_results(self) -> None:
        if self.args.run_mode != "select" or self.model_name != "TSRouter":
            return
        if not bool(getattr(self.args, "skip_saved", False)) or bool(getattr(self.args, "clean_saved", False)):
            return
        if os.path.exists(self.csv_file_path):
            return
        abbr_order = [
            str(details.get("abbreviation", ""))
            for details in sorted(
                [
                    details
                    for variants in getattr(self, "Model_sizes", {}).values()
                    for details in variants.values()
                    if details.get("abbreviation", "")
                ],
                key=lambda x: int(x.get("id", 0)),
            )
        ]
        src = materialize_compatible_tsrouter_result(
            self.args,
            self.csv_file_path,
            abbr_order,
            verbose=True,
        )
        if src is None:
            return
        print(
            f"[Step4][same-stage-reuse] copied main result from {src} -> {self.csv_file_path}; "
            f"stage={getattr(self.args, 'current_zoo_num', '')}, model_abbr_order={abbr_order}"
        )
        src_dir = os.path.dirname(src)
        for sidecar in ["per_channel_results.csv", "per_window_results.csv"]:
            src_sidecar = os.path.join(src_dir, sidecar)
            dst_sidecar = os.path.join(self.output_dir, sidecar)
            if os.path.exists(src_sidecar) and not os.path.exists(dst_sidecar):
                shutil.copy2(src_sidecar, dst_sidecar)
                print(f"[Step4][same-stage-reuse] copied {sidecar}: {src_sidecar} -> {dst_sidecar}")

    def get_save_path(self):

        os.makedirs(self.output_dir, exist_ok=True)
        route_detail_columns = (
            TSROUTER_ROUTE_DETAIL_COLUMNS
            if self.args.run_mode == "select" and self.model_name == "TSRouter"
            else []
        )
        provenance_columns = (
            TSROUTER_STEP4_PROVENANCE_COLUMNS
            if self.args.run_mode == "select" and self.model_name == "TSRouter"
            else []
        )

        if getattr(self.args, "GE_released", False):
            self.csv_header = [
                "dataset",
                "model",
                "eval_metrics/MSE[mean]",
                "eval_metrics/MSE[0.5]",
                "eval_metrics/MAE[0.5]",
                "eval_metrics/MASE[0.5]",
                "eval_metrics/MAPE[0.5]",
                "eval_metrics/sMAPE[0.5]",
                "eval_metrics/MSIS",
                "eval_metrics/RMSE[mean]",
                "eval_metrics/NRMSE[mean]",
                "eval_metrics/ND[0.5]",
                "eval_metrics/mean_weighted_sum_quantile_loss",
                "domain",
                "num_variates",
                *TIMING_COLUMNS,
                *TSROUTER_EXTRA_METRIC_COLUMNS,
                *route_detail_columns,
                *provenance_columns,
            ]
        else:
            self.csv_header = [
                "dataset",
                "model",
                "MASE",
                "sMAPE",
                "CRPS",
                "domain",
                "num_variates",
                "model_order",
                *TIMING_COLUMNS,
                *TSROUTER_EXTRA_METRIC_COLUMNS,
                *route_detail_columns,
                *provenance_columns,
            ]

        if self.args.run_mode == "zoo":
            if self.args.fix_context_len:
                self.model_cl_name = f"cl_{self.args.context_len}"
            else:
                self.model_cl_name = "cl_original"
            tsfm_output_root = str(getattr(self.args, "output_dir", self.output_dir))
            self.output_dir = str(
                tsfm_csv_dir(
                    self.model_name,
                    self.model_cl_name,
                    root=tsfm_output_root,
                    create=True,
                )
            )
            self.artifact_output_dir = str(
                tsfm_artifact_dir(self.model_name, self.model_cl_name, create=True)
            )
            os.makedirs(os.path.join(self.artifact_output_dir, "npy"), exist_ok=True)
            os.makedirs(os.path.join(self.artifact_output_dir, "meta"), exist_ok=True)

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)

            if getattr(self.args, "GE_released", False):
                self.csv_file_path = os.path.join(self.output_dir, f"GE_all_results.csv")
            else:
                self.csv_file_path = os.path.join(self.output_dir, "all_results.csv")

        elif self.args.run_mode == "select":
            if self.args.models == "TSRouter":
                from utils.path_utils import get_repr_save_path
                _,_,_,tsrouter_save_name = get_repr_save_path(self.args)
                self.output_dir = get_tsrouter_selector_stage_result_dir(self.args)
                os.makedirs(self.output_dir, exist_ok=True)
                self.csv_file_path = os.path.join(self.output_dir, tsrouter_save_name)
                self._reuse_same_stage_tsrouter_selector_results()
            else:
                cfg = Selector_zoo_details.get(self.model_name, None)
                if cfg is None:
                    raise ValueError(f"TSRouter runtime message: {self.model_name}TSRouter runtime message: ")

                tpl = cfg["csv_name_tpl"]

                                        
                                                  
                                                              
                filename = tpl.format(
                    current_zoo_num=self.args.current_zoo_num,
                    zoo_total_num=self.args.zoo_total_num,
                    ensemble_size=self.args.ensemble_size,
                    ensemble_agg=self.args.ensemble_agg,
                    search_seed=getattr(self.args, "search_seed", None),
                    real_order_metric=getattr(self.args, "real_order_metric", None),
                    sample_repr_num=getattr(self.args, "sample_repr_num", None),
                )
                if getattr(self.args, "GE_released", False) and filename.endswith(".csv"):
                    if getattr(self.args, "GE_fast_eval", False):
                        filename = filename[:-4] + "_GE_fast.csv"
                    else:
                        filename = filename[:-4] + "_GE.csv"
                self.csv_file_path = os.path.join(self.output_dir, filename)

        elif self.args.run_mode == "zoo_repr_set_forward":
            repr_forward_dir = get_tsrouter_repr_forward_dir(self.args)
            os.makedirs(repr_forward_dir, exist_ok=True)
            self.artifact_output_dir = str(TSROUTER_REPR_FORWARD_ARTIFACT_ROOT)
            os.makedirs(os.path.join(self.artifact_output_dir, "npy"), exist_ok=True)
            os.makedirs(os.path.join(self.artifact_output_dir, "meta"), exist_ok=True)
            from utils.path_utils import (
                build_repr_eval_pool_forward_stem,
                build_repr_forward_all_results_stem,
                build_repr_forward_stem,
            )
            repr_set_name = build_repr_forward_stem(self.args)
            repr_all_results_name = build_repr_forward_all_results_stem(self.args)
            self.csv_file_path = os.path.join(repr_forward_dir, repr_all_results_name + "_all_results.csv")
            self.per_sample_csv_file_path = os.path.join(repr_forward_dir, repr_set_name + "_per_sample_results.csv")
            pool_stem = build_repr_eval_pool_forward_stem(self.args)
            self.pool_csv_file_path = os.path.join(repr_forward_dir, pool_stem + "_all_results.csv")
            self.pool_per_sample_csv_file_path = os.path.join(repr_forward_dir, pool_stem + "_per_sample_results.csv")

                                                               
            if not os.path.exists(self.per_sample_csv_file_path):
                with file_lock(self.per_sample_csv_file_path + ".lock"):
                    if not os.path.exists(self.per_sample_csv_file_path):
                        with open(self.per_sample_csv_file_path, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(["model", "metric"])                

            if not os.path.exists(self.pool_per_sample_csv_file_path):
                with file_lock(self.pool_per_sample_csv_file_path + ".lock"):
                    if not os.path.exists(self.pool_per_sample_csv_file_path):
                        with open(self.pool_per_sample_csv_file_path, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(["model", "metric"])
            if not os.path.exists(self.pool_csv_file_path):
                with file_lock(self.pool_csv_file_path + ".lock"):
                    if not os.path.exists(self.pool_csv_file_path):
                        with open(self.pool_csv_file_path, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(self.csv_header)

        elif self.args.run_mode == "zoo_task_sample_forward":
            if self.args.fix_context_len:
                self.model_cl_name = f"cl_{self.args.context_len}"
            else:
                self.model_cl_name = "cl_original"
            tsfm_output_root = str(getattr(self.args, "output_dir", self.output_dir))
            self.output_dir = str(
                tsfm_csv_dir(
                    self.model_name,
                    self.model_cl_name,
                    root=tsfm_output_root,
                    create=True,
                )
            )
            default_log = os.path.join(
                "results_csv",
                "TSRouter",
                "vldb",
                "logs",
                "task_probe_sample_forward_log.csv",
            )
            self.csv_file_path = str(getattr(self.args, "task_probe_sample_forward_log", "") or default_log)
            self.csv_header = TASK_PROBE_SAMPLE_FORWARD_COLUMNS

        elif self.args.run_mode == "zoo_task_probe_select":
            if self.args.fix_context_len:
                self.model_cl_name = f"cl_{self.args.context_len}"
            else:
                self.model_cl_name = "cl_original"
            os.makedirs(self.output_dir, exist_ok=True)
            self.csv_file_path = str(
                getattr(self.args, "task_probe_select_result_csv", "")
                or os.path.join(self.output_dir, "task_probe_select_windows.csv")
            )
            self.csv_header = TASK_PROBE_SAMPLE_ERROR_COLUMNS

        else:
            raise ValueError(f"Unknown run_mode={self.args.run_mode}; supported: zoo / select / zoo_repr_set_forward / zoo_task_sample_forward / zoo_task_probe_select")

        csv_parent = os.path.dirname(str(self.csv_file_path))
        if csv_parent:
            os.makedirs(csv_parent, exist_ok=True)
        if not os.path.exists(self.csv_file_path):
            with file_lock(self.csv_file_path + ".lock"):
                if not os.path.exists(self.csv_file_path):
                    with open(self.csv_file_path, "w", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(self.csv_header)

    def _row_exists(self, csv_path: str, key_cols: dict) -> bool:
        if not os.path.exists(csv_path):
            return False
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return False
        for col, value in key_cols.items():
            if col not in df.columns:
                return False
            df = df[df[col].astype(str) == str(value)]
            if df.empty:
                return False
        return not df.empty

    def _model_name_aliases(self) -> set[str]:
        aliases = {str(self.model_name)}
        abbr = Model_abbrev_map.get(str(self.model_name))
        if abbr:
            aliases.add(str(abbr))
        return aliases

    def _csv_model_done(self, csv_path: str | None) -> bool:
        if not csv_path or not os.path.exists(csv_path):
            return False
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c == "model")
        except Exception:
            return False
        if "model" not in df.columns:
            return False
        present = set(df["model"].dropna().astype(str).tolist())
        return bool(self._model_name_aliases() & present)

    def _model_abbrev(self, model_name: str | None = None) -> str:
        name = str(model_name or self.model_name)
        return str(Model_abbrev_map.get(name, name))

    def _remove_matching_csv_rows(self, csv_path: str, key_cols: dict) -> int:
        if not os.path.exists(csv_path):
            return 0

        with open(csv_path, "r", newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            return 0

        header = rows[0]
        col_idx = {}
        for col, value in key_cols.items():
            if col not in header:
                return 0
            col_idx[col] = header.index(col)

        kept = [header]
        removed = 0
        for row in rows[1:]:
            matched = True
            for col, value in key_cols.items():
                idx = col_idx[col]
                if idx >= len(row) or str(row[idx]) != str(value):
                    matched = False
                    break
            if matched:
                removed += 1
            else:
                kept.append(row)

        if removed > 0:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(kept)
        return removed

    def _ensure_csv_columns(self, csv_path: str, required_columns: list[str]) -> None:
        if not os.path.exists(csv_path):
            return
        with open(csv_path, "r", newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            return
        header = rows[0]
        ordered_header = list(required_columns) + [c for c in header if c not in required_columns]
        if header == ordered_header:
            return
        old_idx = {c: i for i, c in enumerate(header)}
        new_header = ordered_header
        new_rows = [new_header]
        for row in rows[1:]:
            new_rows.append([
                row[old_idx[col]] if col in old_idx and old_idx[col] < len(row) else ""
                for col in new_header
            ])
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(new_rows)

    def _aggregate_saved_zoo_runtime(self) -> dict:
        summary = {
            "dataset_num": 0,
            "total_time_s": np.nan,
            "average_time_s": np.nan,
            "total_forward_time_s": np.nan,
            "average_forward_time_s": np.nan,
            "forward_dataset_num": 0,
            "total_eval_time_s": np.nan,
            "average_eval_time_s": np.nan,
            "eval_dataset_num": 0,
            "total_metric_read_time_s": np.nan,
            "average_metric_read_time_s": np.nan,
            "average_memory_MB": np.nan,
            "max_memory_MB": np.nan,
            "min_batch_size": np.nan,
            "batch_size_fallback_count": 0,
            "min_context_length": np.nan,
            "context_length_fallback_count": 0,
        }
        if not os.path.exists(self.csv_file_path):
            return summary
        try:
            with file_lock(self.csv_file_path + ".lock"):
                saved = pd.read_csv(self.csv_file_path, low_memory=False)
        except Exception as exc:
            print(f"[runtime-summary] failed to read {self.csv_file_path}: {type(exc).__name__}: {exc}")
            return summary
        if saved.empty or "dataset" not in saved.columns:
            return summary
        if "model" in saved.columns:
            saved = saved[saved["model"].astype(str).eq(str(self.model_name))]
        saved = saved[saved["dataset"].notna()].drop_duplicates(subset=["dataset"], keep="last")
        summary["dataset_num"] = int(saved["dataset"].nunique())

        def _finite_values(column: str) -> pd.Series:
            if column not in saved.columns:
                return pd.Series(dtype=float)
            return (
                pd.to_numeric(saved[column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

        for column, total_key, average_key, count_key in [
            ("runtime_seconds", "total_time_s", "average_time_s", None),
            ("forward_runtime_seconds", "total_forward_time_s", "average_forward_time_s", "forward_dataset_num"),
            ("eval_runtime_seconds", "total_eval_time_s", "average_eval_time_s", "eval_dataset_num"),
            ("metric_read_seconds", "total_metric_read_time_s", "average_metric_read_time_s", None),
        ]:
            values = _finite_values(column)
            if values.empty:
                continue
            summary[total_key] = float(values.sum())
            summary[average_key] = float(values.mean())
            if count_key is not None:
                summary[count_key] = int(len(values))

        memory_values = _finite_values("memory_use_mb")
        if not memory_values.empty:
            summary["average_memory_MB"] = float(memory_values.mean())
            summary["max_memory_MB"] = float(memory_values.max())
        min_batch_values = _finite_values("min_batch_size")
        if not min_batch_values.empty:
            summary["min_batch_size"] = int(min_batch_values.min())
        fallback_values = _finite_values("batch_size_fallback_count")
        if not fallback_values.empty:
            summary["batch_size_fallback_count"] = int(fallback_values.sum())
        min_context_values = _finite_values("min_context_length")
        if not min_context_values.empty:
            summary["min_context_length"] = int(min_context_values.min())
        context_fallback_values = _finite_values("context_length_fallback_count")
        if not context_fallback_values.empty:
            summary["context_length_fallback_count"] = int(context_fallback_values.sum())
        return summary

    def _update_matching_csv_cells(self, csv_path: str, key_cols: dict, update_cols: dict) -> int:
        if not os.path.exists(csv_path):
            return 0
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return 0
        mask = pd.Series(True, index=df.index)
        for col, value in key_cols.items():
            if col not in df.columns:
                return 0
            mask &= df[col].astype(str).eq(str(value))
        if not mask.any():
            return 0
        changed = 0
        for col, value in update_cols.items():
            if col not in df.columns:
                df[col] = np.nan
            try:
                is_valid = np.isfinite(float(value))
            except Exception:
                is_valid = value is not None and str(value) != ""
            if not is_valid:
                continue
            before = df.loc[mask, col].copy()
            df.loc[mask, col] = value
            if not before.astype(str).eq(str(value)).all():
                changed += int(mask.sum())
        if changed > 0:
            df.to_csv(csv_path, index=False)
        return changed

    def _reset_dataset_timing(self) -> None:
        self._last_forward_ms = np.nan
        self._last_evaluate_ms = np.nan
        self._last_metric_read_ms = 0.0
        self._last_repr_data_load_ms = np.nan
        self._last_per_sample_metric_ms = np.nan
        self._last_per_sample_metric_save_ms = np.nan
        self._last_eval_skipped = False
        self._last_evaluation_mode = "gluonts_evaluate"
        self._last_requested_batch_size = int(getattr(self, "batch_size", getattr(self.args, "batch_size", 0)) or 0)
        self._last_runtime_batch_size = self._last_requested_batch_size
        self._last_min_batch_size = self._last_requested_batch_size
        self._last_batch_size_fallback_count = 0
        fixed_context_length = bool(getattr(self.args, "fix_context_len", False))
        self._last_requested_context_length = (
            int(getattr(self.args, "context_len", 0) or 0)
            if fixed_context_length
            else np.nan
        )
        self._last_runtime_context_length = self._last_requested_context_length
        self._last_min_context_length = self._last_requested_context_length
        self._last_context_length_fallback_count = 0

    def _evaluate_forecasts(self, forecasts, dataset, metric_list):
        eval_batch_size = max(1, int(getattr(self.args, "eval_batch_size", 1024) or 1024))
        seasonality = get_seasonality(dataset.freq)
        test_data = dataset.test_data
        if bool(getattr(self.args, "fast_gluonts_eval", True)):
            try:
                seasonal_errors = get_cached_seasonal_errors(
                    dataset,
                    test_data=test_data,
                    seasonality=seasonality,
                    mask_invalid_label=True,
                )
                result = evaluate_forecasts_fast(
                    forecasts=forecasts,
                    test_data=test_data,
                    metrics=metric_list,
                    batch_size=eval_batch_size,
                    axis=None,
                    mask_invalid_label=True,
                    allow_nan_forecast=False,
                    seasonality=seasonality,
                    seasonal_errors=seasonal_errors,
                )
                self._active_evaluation_mode = "gluonts_evaluate_rolling"
                return result
            except Exception as exc:
                print(
                    f"[eval] rolling seasonal-error reuse unavailable; "
                    f"falling back to GluonTS evaluate_forecasts: {type(exc).__name__}: {exc}"
                )

        self._active_evaluation_mode = "gluonts_evaluate"
        return gluonts_evaluate_forecasts(
            forecasts=forecasts,
            test_data=test_data,
            metrics=metric_list,
            batch_size=eval_batch_size,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=seasonality,
        )

    def _timing_row_values(self, elapsed: float) -> dict:
        if bool(getattr(self, "_force_nan_timing", False)):
            return {
                "runtime_seconds": np.nan,
                "sample_seconds": np.nan,
                "sample_to_route_seconds": np.nan,
                "route_final_seconds": np.nan,
                "insert_runtime_seconds": np.nan,
                "repr_data_load_seconds": np.nan,
                "forward_runtime_seconds": np.nan,
                "eval_runtime_seconds": np.nan,
                "per_sample_metric_seconds": np.nan,
                "per_sample_metric_save_seconds": np.nan,
                "metric_read_seconds": np.nan,
                "non_eval_runtime_seconds": np.nan,
                "eval_skipped": "true",
                "evaluation_mode": str(getattr(self, "_forced_evaluation_mode", "pool_artifact_replay")),
                "cli_batch_size": getattr(self, "_cli_batch_size", np.nan),
                "batch_size_source": getattr(self, "_initial_batch_size_source", ""),
                "requested_batch_size": getattr(self, "_last_requested_batch_size", np.nan),
                "runtime_batch_size": getattr(self, "_last_runtime_batch_size", np.nan),
                "min_batch_size": getattr(self, "_last_min_batch_size", np.nan),
                "batch_size_fallback_count": getattr(self, "_last_batch_size_fallback_count", 0),
                "requested_context_length": getattr(self, "_last_requested_context_length", np.nan),
                "runtime_context_length": getattr(self, "_last_runtime_context_length", np.nan),
                "min_context_length": getattr(self, "_last_min_context_length", np.nan),
                "context_length_fallback_count": getattr(self, "_last_context_length_fallback_count", 0),
            }
        forward_ms = getattr(self, "_last_forward_ms", np.nan)
        eval_ms = getattr(self, "_last_evaluate_ms", np.nan)
        metric_read_ms = getattr(self, "_last_metric_read_ms", 0.0)
        repr_data_load_ms = getattr(self, "_last_repr_data_load_ms", np.nan)
        per_sample_metric_ms = getattr(self, "_last_per_sample_metric_ms", np.nan)
        per_sample_metric_save_ms = getattr(self, "_last_per_sample_metric_save_ms", np.nan)
        if not np.isfinite(float(forward_ms)) and np.isfinite(float(elapsed)):
            eval_part = float(eval_ms) / 1000.0 if np.isfinite(float(eval_ms)) else 0.0
            metric_part = float(metric_read_ms) / 1000.0 if np.isfinite(float(metric_read_ms)) else 0.0
            forward_ms = max(0.0, float(elapsed) - eval_part - metric_part) * 1000.0
        forward_s = float(forward_ms) / 1000.0 if np.isfinite(float(forward_ms)) else np.nan
        eval_s = float(eval_ms) / 1000.0 if np.isfinite(float(eval_ms)) else np.nan
        metric_read_s = float(metric_read_ms) / 1000.0 if np.isfinite(float(metric_read_ms)) else 0.0
        repr_data_load_s = (
            float(repr_data_load_ms) / 1000.0
            if np.isfinite(float(repr_data_load_ms))
            else np.nan
        )
        per_sample_metric_s = (
            float(per_sample_metric_ms) / 1000.0
            if np.isfinite(float(per_sample_metric_ms))
            else np.nan
        )
        per_sample_metric_save_s = (
            float(per_sample_metric_save_ms) / 1000.0
            if np.isfinite(float(per_sample_metric_save_ms))
            else np.nan
        )
        non_eval_s = (
            (forward_s if np.isfinite(forward_s) else 0.0)
            + (metric_read_s if np.isfinite(metric_read_s) else 0.0)
        )
        insert_parts = [
            repr_data_load_s,
            forward_s,
            eval_s,
            metric_read_s,
            per_sample_metric_s,
            per_sample_metric_save_s,
        ]
        insert_finite = [float(v) for v in insert_parts if np.isfinite(float(v))]
        insert_s = float(np.sum(insert_finite)) if insert_finite else np.nan
        sample_s = np.nan
        sample_to_route_s = np.nan
        route_final_s = np.nan
        if self.args.run_mode == "select" and self.model_name == "TSRouter":
            extra = getattr(self, "_last_selector_extra", {}) or {}
            route_timing = extra.get("step4_route_timing", {}) if isinstance(extra, dict) else {}
            try:
                sample_s = float(route_timing.get("sample_seconds", np.nan))
            except Exception:
                sample_s = np.nan
            try:
                sample_to_route_s = float(route_timing.get("sample_to_route_seconds", np.nan))
            except Exception:
                sample_to_route_s = np.nan
            try:
                route_final_s = float(route_timing.get("route_final_seconds", np.nan))
            except Exception:
                route_final_s = np.nan
            route_values = [sample_s, sample_to_route_s, route_final_s]
            if any(not np.isfinite(value) for value in route_values):
                raise ValueError(
                    f"[Step4 timing] missing/non-finite route timing: sample={sample_s}, "
                    f"sample_to_route={sample_to_route_s}, route_final={route_final_s}"
                )
            if any(value < 0 for value in route_values):
                raise ValueError(
                    f"[Step4 timing] negative route timing: sample={sample_s}, "
                    f"sample_to_route={sample_to_route_s}, route_final={route_final_s}"
                )
            if abs(route_final_s - sample_s - sample_to_route_s) > 1e-6:
                raise ValueError(
                    f"[Step4 timing] invalid route formula: sample={sample_s}, "
                    f"sample_to_route={sample_to_route_s}, route_final={route_final_s}"
                )
        return {
            "runtime_seconds": elapsed,
            "sample_seconds": sample_s,
            "sample_to_route_seconds": sample_to_route_s,
            "route_final_seconds": route_final_s,
            "insert_runtime_seconds": insert_s,
            "repr_data_load_seconds": repr_data_load_s,
            "forward_runtime_seconds": forward_s,
            "eval_runtime_seconds": eval_s,
            "per_sample_metric_seconds": per_sample_metric_s,
            "per_sample_metric_save_seconds": per_sample_metric_save_s,
            "metric_read_seconds": metric_read_s,
            "non_eval_runtime_seconds": non_eval_s,
            "eval_skipped": "true" if bool(getattr(self, "_last_eval_skipped", False)) else "false",
            "evaluation_mode": str(getattr(self, "_last_evaluation_mode", "gluonts_evaluate")),
            "cli_batch_size": getattr(self, "_cli_batch_size", np.nan),
            "batch_size_source": getattr(self, "_initial_batch_size_source", ""),
            "requested_batch_size": getattr(self, "_last_requested_batch_size", np.nan),
            "runtime_batch_size": getattr(self, "_last_runtime_batch_size", np.nan),
            "min_batch_size": getattr(self, "_last_min_batch_size", np.nan),
            "batch_size_fallback_count": getattr(self, "_last_batch_size_fallback_count", 0),
            "requested_context_length": getattr(self, "_last_requested_context_length", np.nan),
            "runtime_context_length": getattr(self, "_last_runtime_context_length", np.nan),
            "min_context_length": getattr(self, "_last_min_context_length", np.nan),
            "context_length_fallback_count": getattr(self, "_last_context_length_fallback_count", 0),
        }

    def _resolved_step4_eval_cl(self) -> str:
        extra = getattr(self, "_last_selector_extra", {}) or {}
        search_args = extra.get("search_args") if isinstance(extra, dict) else None
        resolved = ""
        if search_args is not None:
            resolved = str(getattr(search_args, "resolved_eval_cl", "") or "")
        if not resolved and isinstance(extra, dict):
            resolved = str(extra.get("resolved_eval_cl", "") or "")
        if resolved:
            return resolved
        if self.args.run_mode == "select" and self.model_name == "TSRouter" and auto_cl_enabled(self.args):
            raise ValueError("[AutoCL Step4] resolved_eval_cl is missing from selector output")
        return str(getattr(self.args, "TSFM_results_dir", "cl_512"))

    def _format_timing_seconds(self, value) -> str:
        try:
            v = float(value)
        except Exception:
            return "nan"
        return f"{v:.3f}s" if np.isfinite(v) else "nan"

    def _print_repr_insert_timing(self, ds_config: str, timing_values: dict, memory_used: float) -> None:
        print(
            f"[TIMING] {self.model_name} {ds_config}: "
            f"repr_load={self._format_timing_seconds(timing_values.get('repr_data_load_seconds'))}, "
            f"forward={self._format_timing_seconds(timing_values.get('forward_runtime_seconds'))}, "
            f"eval={self._format_timing_seconds(timing_values.get('eval_runtime_seconds'))}, "
            f"metric_read={self._format_timing_seconds(timing_values.get('metric_read_seconds'))}, "
            f"per_sample_metric={self._format_timing_seconds(timing_values.get('per_sample_metric_seconds'))}, "
            f"per_sample_save={self._format_timing_seconds(timing_values.get('per_sample_metric_save_seconds'))}, "
            f"insert={self._format_timing_seconds(timing_values.get('insert_runtime_seconds'))}, "
            f"wall={self._format_timing_seconds(timing_values.get('runtime_seconds'))}, "
            f"memory={float(memory_used):.0f}MB",
            flush=True,
        )

    def _begin_gpu_memory_tracking(self):
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        monitor = _GpuMemoryMonitor(
            poll_interval_seconds=float(getattr(self.args, "gpu_memory_poll_interval_seconds", 0.1) or 0.1)
        ).start()
        self._last_memory_snapshot = {}
        return monitor

    def _finish_gpu_memory_tracking(self, monitor=None) -> dict:
        snapshot = {}
        if monitor is not None:
            try:
                snapshot.update(monitor.stop())
            except Exception as exc:
                snapshot["gpu_memory_source"] = f"monitor_error:{type(exc).__name__}"
        if torch.cuda.is_available():
            try:
                snapshot["torch_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 ** 2
                snapshot["torch_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 ** 2
                snapshot["torch_memory_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
                snapshot["torch_memory_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024 ** 2
            except Exception:
                pass
        candidates = [
            snapshot.get("nvml_process_peak_memory_mb"),
            snapshot.get("torch_memory_peak_reserved_mb"),
            snapshot.get("torch_memory_peak_allocated_mb"),
            snapshot.get("torch_memory_reserved_mb"),
            snapshot.get("torch_memory_allocated_mb"),
        ]
        finite = []
        for value in candidates:
            try:
                value = float(value)
            except Exception:
                continue
            if np.isfinite(value) and value >= 0:
                finite.append(value)
        snapshot["memory_use_mb"] = float(finite[0]) if finite else 0.0
        try:
            nvml_peak = float(snapshot.get("nvml_process_peak_memory_mb", np.nan))
        except Exception:
            nvml_peak = np.nan
        if not snapshot.get("gpu_memory_source"):
            snapshot["gpu_memory_source"] = "torch_peak_reserved"
        elif not np.isfinite(nvml_peak):
            snapshot["gpu_memory_source"] = "torch_peak_reserved"
        self._last_memory_snapshot = snapshot
        return snapshot

    def _read_saved_tsfm_metric_for_dataset(self, ds_config: str):
        metric_path = Path(self.csv_file_path)
        if not metric_path.exists():
            return None, 0.0, f"missing saved metric file {metric_path}"
        t0 = time.perf_counter()
        try:
            df = pd.read_csv(metric_path)
        except Exception as e:
            return None, (time.perf_counter() - t0) * 1000.0, f"cannot read {metric_path}: {e}"
        if "dataset" not in df.columns or "model" not in df.columns:
            return None, (time.perf_counter() - t0) * 1000.0, "saved metric file missing dataset/model columns"
        sub = df[
            df["dataset"].astype(str).eq(str(ds_config))
            & df["model"].astype(str).isin(self._model_name_aliases())
        ]
        if sub.empty:
            return None, (time.perf_counter() - t0) * 1000.0, f"missing saved metric row for dataset={ds_config}"
        row = sub.iloc[-1]
        metric_cols = (
            ["eval_metrics/MASE[0.5]", "eval_metrics/sMAPE[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss"]
            if getattr(self.args, "GE_released", False)
            else ["MASE", "sMAPE", "CRPS"]
        )
        if not set(metric_cols).issubset(set(df.columns)):
            return None, (time.perf_counter() - t0) * 1000.0, f"saved metric row missing columns {metric_cols}"
        vals = pd.to_numeric(row[metric_cols], errors="coerce").to_numpy(dtype=float)
        if vals.size == 0 or not np.isfinite(vals).all():
            return None, (time.perf_counter() - t0) * 1000.0, "saved metric row has NaN/Inf metrics"
        res = self._build_fast_eval_res(row)
        return res, (time.perf_counter() - t0) * 1000.0, f"saved metrics from {metric_path}"

    def _print_process_metrics(self, extra_metrics: dict) -> None:
        def _finite(value) -> bool:
            try:
                return bool(np.isfinite(float(value)))
            except Exception:
                return False

        def _pair(label: str, top1_col: str, top3_col: str, n_col: str | None = None) -> str:
            top1 = extra_metrics.get(top1_col, np.nan)
            top3 = extra_metrics.get(top3_col, np.nan)
            if _finite(top1) and _finite(top3):
                text = f"{label}1/3: {float(top1):.4f}/{float(top3):.4f}"
            else:
                text = f"{label}1/3: NA"
            if n_col:
                n_val = extra_metrics.get(n_col, np.nan)
                if _finite(n_val):
                    text += f" n={int(float(n_val))}"
            return text

        def _rank_group_text(value) -> str:
            if value is None:
                return "NA"
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if not isinstance(value, (list, tuple)) or len(value) == 0:
                return "NA"
            def _as_int_list(seq) -> list[int]:
                if isinstance(seq, np.ndarray):
                    seq = seq.tolist()
                return [int(x) for x in list(seq)]
            first = value[0]
            if isinstance(first, np.ndarray):
                first = first.tolist()
            if isinstance(first, (list, tuple)):
                groups = [_as_int_list(group) for group in value]
                return str(groups) if groups else "NA"
            return str(_as_int_list(value))

        row1_source = (
            f"forward_window_top3={_rank_group_text(extra_metrics.get('_PROCESS_FORWARD_WINDOW_TOP3'))}; "
            f"repr_window_top3={_rank_group_text(extra_metrics.get('_PROCESS_REPR_WINDOW_TOP3'))}"
        )
        row2_source = (
            f"real_channel_top3={_rank_group_text(extra_metrics.get('_PROCESS_REAL_CHANNEL_TOP3'))}; "
            f"pred_channel_top3={_rank_group_text(extra_metrics.get('_PROCESS_PRED_CHANNEL_TOP3'))}"
        )
        row3_source = (
            f"real_task_top3={_rank_group_text(extra_metrics.get('_PROCESS_REAL_TASK_TOP3'))}; "
            f"pred_task_top3={_rank_group_text(extra_metrics.get('_PROCESS_PRED_TASK_TOP3'))}"
        )

        rows = [
            [
                _pair("PWW", "ENC_TOP1_SUBSET_RATE", "ENC_TOP3_SUBSET_RATE"),
                _pair("TWW", "TEST_WINDOW_TOP1_ACC", "TEST_WINDOW_TOP3_HIT", "TEST_WINDOW_EVAL_N"),
                "",
                row1_source,
            ],
            [
                "",
                _pair("TWC", "TEST_WINDOW_CHANNEL_TOP1_ACC", "TEST_WINDOW_CHANNEL_TOP3_HIT", "TEST_WINDOW_CHANNEL_EVAL_N"),
                _pair("TCC", "SINGLE_TOP1_ACC", "SINGLE_TOP3_HIT", "SINGLE_CHANNELS_EVAL"),
                row2_source,
            ],
            [
                "",
                _pair("TWR", "TEST_WINDOW_TASK_TOP1_ACC", "TEST_WINDOW_TASK_TOP3_HIT", "TEST_WINDOW_TASK_EVAL_N"),
                _pair("TCR", "TEST_CHANNEL_TASK_TOP1_ACC", "TEST_CHANNEL_TASK_TOP3_HIT", "TEST_CHANNEL_TASK_EVAL_N"),
                row3_source,
            ],
        ]
        if not any(cell and "NA" not in cell for row in rows for cell in row[:3]):
            return
        headers = ["PWW(valid-window)", "Window-level", "Channel-level", "Rank source"]
        widths = [
            max(len(headers[idx]), *(len(row[idx]) for row in rows))
            for idx in range(len(headers))
        ]
        print("[PROCESS_METRIC] 1/3 process metrics")
        print("  " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
        print("  " + "-+-".join("-" * width for width in widths))
        for row in rows:
            print("  " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(row))))

    def _per_channel_result_exists(self, ds_config: str) -> bool:
        ok, _ = self._per_channel_result_status(ds_config)
        return ok

    def _load_per_channel_results_df(self) -> pd.DataFrame | None:
        csv_path = per_channel_results_path(self.output_dir)
        cached_path = getattr(self, "_per_channel_status_cache_path", None)
        cached_df = getattr(self, "_per_channel_status_cache_df", None)
        if cached_path == csv_path and cached_df is not None:
            return cached_df
        if not os.path.exists(csv_path):
            self._per_channel_status_cache_path = csv_path
            self._per_channel_status_cache_df = None
            return None
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            self._per_channel_status_cache_path = csv_path
            self._per_channel_status_cache_df = None
            return None
        self._per_channel_status_cache_path = csv_path
        self._per_channel_status_cache_df = df
        return df

    def _invalidate_per_channel_status_cache(self) -> None:
        self._per_channel_status_cache_path = None
        self._per_channel_status_cache_df = None

    def _load_per_window_results_df(self) -> pd.DataFrame | None:
        csv_path = per_window_results_path(self.output_dir)
        cached_path = getattr(self, "_per_window_status_cache_path", None)
        cached_df = getattr(self, "_per_window_status_cache_df", None)
        if cached_path == csv_path and cached_df is not None:
            return cached_df
        if not os.path.exists(csv_path):
            self._per_window_status_cache_path = csv_path
            self._per_window_status_cache_df = None
            return None
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            self._per_window_status_cache_path = csv_path
            self._per_window_status_cache_df = None
            return None
        self._per_window_status_cache_path = csv_path
        self._per_window_status_cache_df = df
        return df

    def _invalidate_per_window_status_cache(self) -> None:
        self._per_window_status_cache_path = None
        self._per_window_status_cache_df = None

    def _per_window_result_exists(self, ds_config: str) -> bool:
        ok, _ = self._per_window_result_status(ds_config)
        return ok

    def _per_window_result_status(self, ds_config: str) -> tuple[bool, str]:
        if (
            self.args.run_mode != "zoo"
            or not bool(getattr(self.args, "enable_process_metrics", True))
            or not bool(getattr(self.args, "enable_per_window_metrics", False))
        ):
            return True, "per-window metrics disabled or not zoo mode"
        csv_path = per_window_results_path(self.output_dir)
        if not os.path.exists(csv_path):
            return False, f"missing file {csv_path}"
        df = self._load_per_window_results_df()
        if df is None:
            return False, f"failed to read {csv_path}"
        required = {
            "dataset",
            "model",
            "series_id",
            "forecast_start",
            "channel",
            "window_id",
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
        }
        if not required.issubset(set(df.columns)):
            missing = sorted(required - set(df.columns))
            return False, f"missing columns {missing}"
        sub = df[
            (df["dataset"].astype(str) == str(ds_config))
            & (df["model"].astype(str).isin(self._model_name_aliases()))
        ]
        if sub.empty:
            return False, "missing dataset/model rows"
        impl = sub["METRIC_IMPL"].astype(str)
        if not impl.eq("local_per_window_v1").all():
            return False, "outdated per-window metric implementation"
        metric_cols = ["MASE", "sMAPE", "CRPS"]
        vals = sub[metric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if vals.size == 0:
            return False, "empty metric values"
        if not np.isfinite(vals).all():
            bad_count = int((~np.isfinite(vals).all(axis=1)).sum())
            return False, f"NaN/Inf metric rows={bad_count}"
        if (vals < -1e-12).any():
            bad_count = int((vals < -1e-12).any(axis=1).sum())
            return False, f"negative metric rows={bad_count}"
        stat_cols = ["MASE_NUM", "MASE_DEN", "SMAPE_NUM", "SMAPE_DEN", "CRPS_NUM", "CRPS_DEN"]
        stat_vals = sub[stat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if stat_vals.size == 0 or not np.isfinite(stat_vals).all():
            bad_count = int((~np.isfinite(stat_vals).all(axis=1)).sum())
            return False, f"NaN/Inf metric stats rows={bad_count}"
        for col in ["MASE_DEN", "SMAPE_DEN", "CRPS_DEN"]:
            vals_col = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            if (vals_col <= 0).any():
                return False, f"non-positive {col} rows={int((vals_col <= 0).sum())}"
        return True, f"complete rows={len(sub)}"

    def _per_channel_result_status(self, ds_config: str) -> tuple[bool, str]:
        if self.args.run_mode != "zoo" or not bool(getattr(self.args, "enable_process_metrics", True)):
            return True, "process metrics disabled or not zoo mode"
        csv_path = per_channel_results_path(self.output_dir)
        if not os.path.exists(csv_path):
            return False, f"missing file {csv_path}"
        df = self._load_per_channel_results_df()
        if df is None:
            return False, f"failed to read {csv_path}"
        required = {
            "dataset",
            "model",
            "channel",
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
        }
        if not required.issubset(set(df.columns)):
            missing = sorted(required - set(df.columns))
            return False, f"missing columns {missing}"
        sub = df[
            (df["dataset"].astype(str) == str(ds_config))
            & (df["model"].astype(str).isin(self._model_name_aliases()))
        ]
        if sub.empty:
            return False, "missing dataset/model rows"
        impl = sub["METRIC_IMPL"].astype(str)
        if not impl.eq("gluonts_per_channel_v1").all():
            return False, "outdated per-channel metric implementation"
        metric_cols = ["MASE", "sMAPE", "CRPS"]
        vals = sub[metric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if vals.size == 0:
            return False, "empty metric values"
        if not np.isfinite(vals).all():
            bad_count = int((~np.isfinite(vals).all(axis=1)).sum())
            return False, f"NaN/Inf metric rows={bad_count}"
        if (vals < -1e-12).any():
            bad_count = int((vals < -1e-12).any(axis=1).sum())
            return False, f"negative metric rows={bad_count}"
        stat_cols = ["MASE_NUM", "MASE_DEN", "SMAPE_NUM", "SMAPE_DEN", "CRPS_NUM", "CRPS_DEN"]
        stat_vals = sub[stat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if stat_vals.size == 0 or not np.isfinite(stat_vals).all():
            bad_count = int((~np.isfinite(stat_vals).all(axis=1)).sum())
            return False, f"NaN/Inf metric stats rows={bad_count}"
        for col in ["MASE_DEN", "SMAPE_DEN", "CRPS_DEN"]:
            vals_col = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            if (vals_col <= 0).any():
                return False, f"non-positive {col} rows={int((vals_col <= 0).sum())}"
        return True, f"complete rows={len(sub)}"

    def _dataset_result_complete(self, ds_config: str) -> bool:
        if not os.path.exists(self.csv_file_path):
            return False
        try:
            df = pd.read_csv(self.csv_file_path)
        except Exception:
            return False
        if df.empty or "dataset" not in df.columns or "model" not in df.columns:
            return False
        sub = df[
            (df["dataset"].astype(str) == str(ds_config))
            & (df["model"].astype(str).isin(self._model_name_aliases()))
        ]
        if sub.empty:
            return False
        if getattr(self.args, "GE_released", False):
            metric_cols = [
                "eval_metrics/MASE[0.5]",
                "eval_metrics/sMAPE[0.5]",
                "eval_metrics/mean_weighted_sum_quantile_loss",
            ]
        else:
            metric_cols = ["MASE", "sMAPE", "CRPS"]
        if not set(metric_cols).issubset(set(sub.columns)):
            return False
        vals = sub.iloc[-1][metric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if vals.size == 0 or not np.isfinite(vals).all():
            return False
        if self.args.run_mode == "select" and self.model_name == "TSRouter":
            timing_cols = ["sample_seconds", "sample_to_route_seconds", "route_final_seconds", "eval_runtime_seconds"]
            if not set(timing_cols).issubset(set(sub.columns)):
                return False
            latest = sub.iloc[-1]
            sample_val = pd.to_numeric(pd.Series([latest.get("sample_seconds")]), errors="coerce").iloc[0]
            sample_to_route_val = pd.to_numeric(pd.Series([latest.get("sample_to_route_seconds")]), errors="coerce").iloc[0]
            route_final_val = pd.to_numeric(pd.Series([latest.get("route_final_seconds")]), errors="coerce").iloc[0]
            eval_val = pd.to_numeric(pd.Series([latest.get("eval_runtime_seconds")]), errors="coerce").iloc[0]
            required_vals = [sample_val, sample_to_route_val, route_final_val, eval_val]
            if any(pd.isna(v) or not np.isfinite(float(v)) for v in required_vals):
                return False
            if any(float(v) < 0 for v in required_vals):
                return False
            if abs(float(route_final_val) - (float(sample_val) + float(sample_to_route_val))) > 1e-6:
                return False
            if not set(TSROUTER_ROUTE_DETAIL_COLUMNS).issubset(set(sub.columns)):
                return False
            expected_family_mode = normalize_route_family_mode(
                getattr(self.args, "route_family_mode", "default")
            )
            if "route_family_mode" in sub.columns:
                try:
                    saved_family_mode = normalize_route_family_mode(
                        latest.get("route_family_mode", "default")
                    )
                except ValueError:
                    return False
                if saved_family_mode != expected_family_mode:
                    return False
            elif expected_family_mode != "default":
                return False
            try:
                consistency_by_channel = json.loads(str(latest["rank_consistency_instability_by_channel"]))
                channel_model_rank = json.loads(str(latest["channel_model_rank"]))
            except Exception:
                return False
            if not isinstance(consistency_by_channel, list) or not isinstance(channel_model_rank, list):
                return False
            if not channel_model_rank or not all(isinstance(order, list) and order for order in channel_model_rank):
                return False
            try:
                rank_values = np.asarray(channel_model_rank, dtype=np.int64)
            except Exception:
                return False
            if rank_values.ndim != 2 or rank_values.shape[0] != len(channel_model_rank):
                return False
            if consistency_by_channel:
                try:
                    consistency_values = np.asarray(consistency_by_channel, dtype=float)
                except Exception:
                    return False
                if (
                    consistency_values.ndim != 1
                    or consistency_values.size != rank_values.shape[0]
                    or not np.isfinite(consistency_values).all()
                ):
                    return False
            elif int(getattr(self.args, "task_sample_version", 1)) == 2:
                return False
            if auto_cl_enabled(self.args):
                required_provenance = set(TSROUTER_STEP4_PROVENANCE_COLUMNS) - {
                    "route_family_mode"
                }
                if not required_provenance.issubset(set(sub.columns)):
                    return False
                profile = str(latest.get("adaptive_profile", "") or "")
                profile_cfg = get_auto_cl_profile_by_name(profile, self.args)
                if profile_cfg is None:
                    return False
                if str(latest.get("auto_cl_mode", "") or "") != get_auto_cl_mode(self.args):
                    return False
                expected_numeric = {
                    "repr_input_dim": int(profile_cfg["repr_input_dim"]),
                    "repr_output_dim": int(profile_cfg["repr_output_dim"]),
                    "repr_sub_pred_len": int(profile_cfg["repr_sub_pred_len"]),
                    "repr_source_exact_length": int(profile_cfg["repr_source_exact_length"]),
                }
                for column, expected in expected_numeric.items():
                    value = pd.to_numeric(pd.Series([latest.get(column)]), errors="coerce").iloc[0]
                    if pd.isna(value) or int(value) != int(expected):
                        return False
                pred_len = pd.to_numeric(
                    pd.Series([latest.get("adaptive_pred_len")]),
                    errors="coerce",
                ).iloc[0]
                if pd.isna(pred_len) or not np.isfinite(float(pred_len)) or float(pred_len) <= 0:
                    return False
                context_avg = pd.to_numeric(
                    pd.Series([latest.get("adaptive_context_len_avg")]),
                    errors="coerce",
                ).iloc[0]
                term_fallback_used = str(
                    latest.get("adaptive_task_term_fallback_used", "")
                ).strip().lower() in {"true", "1", "yes", "y", "t"}
                if (
                    (pd.isna(context_avg) or not np.isfinite(float(context_avg)) or float(context_avg) < 0)
                    and not term_fallback_used
                ):
                    return False
                resolved_eval_cl = str(latest.get("resolved_eval_cl", "") or "")
                rank_truth_cl = str(latest.get("rank_truth_cl", "") or "")
                if resolved_eval_cl != str(profile_cfg["tsfm_results_dir"]):
                    return False
                if rank_truth_cl != resolved_eval_cl:
                    return False
                cache_path = str(latest.get("task_sample_cache_path", "") or "")
                cache_match = re.search(r"(?:^|[\\/])cl(\d+)_", cache_path)
                if cache_match is None or int(cache_match.group(1)) != int(profile_cfg["repr_input_dim"]):
                    return False
                if not str(latest.get("sample_timing_source", "") or "").strip():
                    return False
                cache_hit_text = str(latest.get("task_sample_cache_hit", "")).strip().lower()
                if cache_hit_text not in {"true", "false", "1", "0", "yes", "no", "y", "n", "t", "f"}:
                    return False
                fallback_used = str(latest.get("eval_cl_fallback_used", "")).strip().lower()
                if fallback_used not in {"false", "0", "no", "n", "f"}:
                    return False
        if bool(getattr(self.args, "vldb_skip_evaluate", False)):
            timing_cols = ["forward_runtime_seconds", "eval_runtime_seconds", "evaluation_mode"]
            if not set(timing_cols).issubset(set(sub.columns)):
                return False
            latest = sub.iloc[-1]
            forward_val = pd.to_numeric(pd.Series([latest.get("forward_runtime_seconds")]), errors="coerce").iloc[0]
            eval_val = pd.to_numeric(pd.Series([latest.get("eval_runtime_seconds")]), errors="coerce").iloc[0]
            if pd.isna(forward_val) or not np.isfinite(float(forward_val)):
                return False
            if pd.isna(eval_val) or not np.isfinite(float(eval_val)):
                return False
        if self.model_name == "Real_Channel_Select":
            return bool((vals >= -1e-12).all())
        return True

    def _repr_per_sample_result_complete(self, csv_path: str | None) -> bool:
        if not csv_path or not os.path.exists(csv_path):
            return False
        try:
            with open(csv_path, "r", newline="") as f:
                rows = list(csv.reader(f))
        except Exception:
            return False
        if not rows:
            return False
        metrics = set()
        sample_lengths = []
        for row in rows:
            if len(row) < 2:
                continue
            if row[0] == "model" and row[1] == "metric":
                continue
            if str(row[0]) not in self._model_name_aliases():
                continue
            metric_name = str(row[1])
            if metric_name in {"MASE", "sMAPE", "CRPS"}:
                metrics.add(metric_name)
                sample_lengths.append(max(0, len(row) - 2))
        if not {"MASE", "sMAPE", "CRPS"}.issubset(metrics):
            return False
                                                              
        return bool(sample_lengths) and min(sample_lengths) > 0

    def _repr_forward_artifact_stem(self, csv_path: str | None = None, dataset_name: str | None = None) -> str:
        if self.args.run_mode == "zoo_repr_set_forward":
            path = csv_path
            main_csv_path = getattr(self, "csv_file_path", None)
            if path is None or (
                main_csv_path
                and os.path.abspath(str(path)) == os.path.abspath(str(main_csv_path))
            ):
                path = getattr(self, "per_sample_csv_file_path", None) or path
            path = path or self.csv_file_path
            base = os.path.basename(str(path))
            if base.endswith("_per_sample_results.csv"):
                return base[:-len("_per_sample_results.csv")]
            if base.endswith("_all_results.csv"):
                return base[:-len("_all_results.csv")]
        return str(dataset_name or "repr")

    def _repr_forward_artifact_model_tag(self, model_name: str | None = None) -> str:
        raw = str(model_name or self.model_name)
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)

    def _repr_forward_samples_path(
        self,
        csv_path: str | None = None,
        dataset_name: str | None = None,
        model_name: str | None = None,
    ) -> str:
        root = self.artifact_output_dir or self.output_dir
        stem = self._repr_forward_artifact_stem(csv_path, dataset_name)
        model_tag = self._repr_forward_artifact_model_tag(model_name)
        return os.path.join(root, "npy", f"{stem}__{model_tag}_samples.npy")

    def _repr_forward_meta_path(
        self,
        csv_path: str | None = None,
        dataset_name: str | None = None,
        model_name: str | None = None,
    ) -> str:
        root = self.artifact_output_dir or self.output_dir
        stem = self._repr_forward_artifact_stem(csv_path, dataset_name)
        model_tag = self._repr_forward_artifact_model_tag(model_name)
        return os.path.join(root, "meta", f"{stem}__{model_tag}_meta.json")

    def _repr_main_result_complete_at(self, csv_path: str, ds_config: str, *, require_timing: bool = False) -> bool:
        if not csv_path or not os.path.exists(csv_path):
            return False
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return False
        if df.empty or "dataset" not in df.columns or "model" not in df.columns:
            return False
        sub = df[
            df["dataset"].astype(str).eq(str(ds_config))
            & df["model"].astype(str).isin(self._model_name_aliases())
        ]
        if sub.empty:
            return False
        metric_cols = (
            ["eval_metrics/MASE[0.5]", "eval_metrics/sMAPE[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss"]
            if getattr(self.args, "GE_released", False)
            else ["MASE", "sMAPE", "CRPS"]
        )
        if not set(metric_cols).issubset(set(sub.columns)):
            return False
        latest = sub.iloc[-1]
        vals = latest[metric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if vals.size == 0 or not np.isfinite(vals).all():
            return False
        if require_timing and not self._repr_forward_timing_complete(latest):
            return False
        return True

    def _repr_forward_timing_complete(self, row) -> bool:
        required = [
            "runtime_seconds",
            "insert_runtime_seconds",
            "repr_data_load_seconds",
            "forward_runtime_seconds",
            "eval_runtime_seconds",
            "per_sample_metric_seconds",
            "per_sample_metric_save_seconds",
            "non_eval_runtime_seconds",
        ]
        try:
            if str(row.get("evaluation_mode", "")).strip() == "pool_artifact_replay":
                return False
            vals = [
                pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
                for col in required
            ]
            return all(pd.notna(v) and np.isfinite(float(v)) for v in vals)
        except Exception:
            return False

    def _repr_forward_result_complete_at(
        self,
        *,
        csv_path: str | None,
        per_sample_csv_path: str | None,
        ds_config: str,
        require_timing: bool,
        require_samples: bool = False,
    ) -> bool:
        if not self._repr_main_result_complete_at(str(csv_path or ""), ds_config, require_timing=require_timing):
            return False
        if not self._repr_per_sample_result_complete(per_sample_csv_path):
            return False
        samples_path = self._repr_forward_samples_path(csv_path=csv_path, dataset_name=ds_config)
        samples_ok, _ = self._repr_forward_samples_complete_at(samples_path)
        if require_samples and not samples_ok:
            return False
        return True

    def _repr_forward_samples_complete_at(self, samples_path: str | None) -> tuple[bool, str]:
        if not samples_path or not os.path.exists(str(samples_path)):
            return False, "missing"
        try:
            arr = np.load(str(samples_path), mmap_mode="r")
            shape = tuple(int(v) for v in arr.shape)
        except Exception as exc:
            return False, f"unreadable:{type(exc).__name__}:{exc}"
        if len(shape) < 2:
            return False, f"invalid_shape:{shape}"
        if any(dim <= 0 for dim in shape):
            return False, f"invalid_shape:{shape}"
        return True, f"shape={shape}"

    def _repr_forward_result_status_at(
        self,
        *,
        csv_path: str | None,
        per_sample_csv_path: str | None,
        ds_config: str,
        require_samples: bool,
    ) -> dict:
        csv_exists = bool(csv_path and os.path.exists(str(csv_path)))
        main_metrics = self._repr_main_result_complete_at(str(csv_path or ""), ds_config, require_timing=False)
        timing = self._repr_main_result_complete_at(str(csv_path or ""), ds_config, require_timing=True)
        per_sample = self._repr_per_sample_result_complete(per_sample_csv_path)
        samples_path = self._repr_forward_samples_path(csv_path=csv_path, dataset_name=ds_config)
        samples, samples_reason = self._repr_forward_samples_complete_at(samples_path)
        return {
            "csv": csv_exists,
            "metrics": bool(main_metrics),
            "per_sample": bool(per_sample),
            "timing": bool(timing),
            "samples": bool(samples),
            "samples_reason": samples_reason,
            "samples_required": bool(require_samples),
            "samples_path": samples_path,
            "complete_with_timing": bool(main_metrics and per_sample and timing and ((not require_samples) or samples)),
            "complete_without_timing": bool(main_metrics and per_sample and ((not require_samples) or samples)),
        }

    def _format_repr_forward_status(self, status: dict, *, timing_override: str | None = None) -> str:
        sample_text = str(status.get("samples"))
        sample_reason = str(status.get("samples_reason", "") or "")
        if sample_reason and (status.get("samples_required", False) or not status.get("samples", False)):
            sample_text += f"[{sample_reason}]"
        if not status.get("samples_required", False):
            sample_text += "(not_required)"
        timing_text = timing_override if timing_override is not None else str(status.get("timing"))
        return (
            f"csv={status.get('csv')}, metrics={status.get('metrics')}, "
            f"per_sample={status.get('per_sample')}, timing={timing_text}, "
            f"samples={sample_text}"
        )

    def _repr_center_done_for_skip(
        self,
        *,
        center_done_with_timing: bool,
        center_done_without_timing: bool,
        skip_step2_cluster_forward: bool,
    ) -> bool:
        if bool(getattr(self.args, "skip_saved", False)) and bool(center_done_with_timing):
            return True
        if route_efficiency_mode_enabled(self.args):
            return False
        return bool(skip_step2_cluster_forward and center_done_without_timing)

    def _repr_forward_sources_fresh(
        self,
        *,
        result_paths: list[str | None],
        source_paths: list[str | None],
    ) -> tuple[bool, str]:
        existing_sources = [str(p) for p in source_paths if p and os.path.exists(str(p))]
        if not existing_sources:
            return True, "no_source_files"
        existing_results = [str(p) for p in result_paths if p and os.path.exists(str(p))]
        if not existing_results:
            return False, "missing_result_files"
        source_mtime = max(os.path.getmtime(path) for path in existing_sources)
        result_mtime = min(os.path.getmtime(path) for path in existing_results)
        if result_mtime + 1e-6 < source_mtime:
            newest_source = max(existing_sources, key=os.path.getmtime)
            oldest_result = min(existing_results, key=os.path.getmtime)
            return (
                False,
                "source_newer_than_result:"
                f"source={os.path.basename(newest_source)},result={os.path.basename(oldest_result)}",
            )
        return True, "fresh"

    def _repr_center_forward_fresh_for_anchor(self) -> tuple[bool, str]:
        from utils.path_utils import build_repr_set_name

        repr_set_name = build_repr_set_name(self.args)
        anchor_root = str(getattr(self.args, "save_repr_data_path", ""))
        center_ds_config = f"{repr_set_name}_freqH"
        return self._repr_forward_sources_fresh(
            result_paths=[
                self.csv_file_path,
                self.per_sample_csv_file_path,
                self._repr_forward_samples_path(
                    csv_path=self.csv_file_path,
                    dataset_name=center_ds_config,
                ),
                self._repr_forward_meta_path(
                    csv_path=self.csv_file_path,
                    dataset_name=center_ds_config,
                ),
            ],
            source_paths=[
                os.path.join(anchor_root, repr_set_name + ".pkl"),
                os.path.join(anchor_root, repr_set_name + "_meta.pkl"),
            ],
        )

    def _repr_pool_forward_fresh_for_pool(self) -> tuple[bool, str]:
        from utils.path_utils import build_repr_eval_pool_name

        pool_name = build_repr_eval_pool_name(self.args)
        pool_root = str(SAMPLED_REPR_POOL_CACHE_ROOT)
        pool_ds_config = f"{pool_name}_freqH"
        pool_csv = getattr(self, "pool_csv_file_path", None)
        pool_sample_path = (
            self._repr_forward_samples_path(csv_path=pool_csv, dataset_name=pool_ds_config)
            if pool_csv
            else None
        )
        pool_meta_path = (
            self._repr_forward_meta_path(csv_path=pool_csv, dataset_name=pool_ds_config)
            if pool_csv
            else None
        )
        return self._repr_forward_sources_fresh(
            result_paths=[
                pool_csv,
                getattr(self, "pool_per_sample_csv_file_path", None),
                pool_sample_path,
                pool_meta_path,
            ],
            source_paths=[
                os.path.join(pool_root, pool_name + ".pkl"),
                os.path.join(pool_root, pool_name + "_meta.pkl"),
            ],
        )

    def _load_repr_anchor_center_idx(self) -> tuple[np.ndarray | None, str]:
        from utils.path_utils import build_repr_eval_pool_name, build_repr_set_name

        repr_set_name = build_repr_set_name(self.args)
        meta_path = os.path.join(str(getattr(self.args, "save_repr_data_path", "")), repr_set_name + "_meta.pkl")
        if not os.path.exists(meta_path):
            return None, f"missing anchor meta: {meta_path}"
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
        except Exception as e:
            return None, f"cannot read anchor meta: {e}"
        if str(meta.get("sample_mode", getattr(self.args, "sample_mode", ""))) != "cluster_nearest":
            return None, "only cluster_nearest can replay center forecasts from pool rows"
        expected_pool = build_repr_eval_pool_name(self.args)
        if str(meta.get("pool_name", expected_pool)) != expected_pool:
            return None, f"anchor meta pool mismatch: meta={meta.get('pool_name')} expected={expected_pool}"
        idx = meta.get("center_member_idx_in_pool")
        if idx is None:
            return None, "anchor meta missing center_member_idx_in_pool"
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        if idx.size == 0 or np.any(idx < 0):
            return None, "anchor center indices are empty or invalid"
        return idx, meta_path

    def _write_repr_per_sample_from_pool(self, center_idx: np.ndarray) -> bool:
        pool_csv = getattr(self, "pool_per_sample_csv_file_path", None)
        center_csv = self.per_sample_csv_file_path
        if not pool_csv or not os.path.exists(pool_csv) or not center_csv:
            return False
        idx = np.asarray(center_idx, dtype=np.int64).reshape(-1)
        aliases = self._model_name_aliases()
        with open(pool_csv, "r", newline="") as f:
            rows = list(csv.reader(f))
        out_rows = []
        for row in rows:
            if len(row) < 2:
                continue
            if row[0] == "model" and row[1] == "metric":
                continue
            if str(row[0]) not in aliases or str(row[1]) not in {"MASE", "sMAPE", "CRPS"}:
                continue
            values = row[2:]
            if not values:
                return False
            picked = [values[int(i)] if 0 <= int(i) < len(values) else "" for i in idx]
            if any(v == "" for v in picked):
                return False
            out_rows.append([self.model_name, row[1], *picked])
        if len(out_rows) < 3:
            return False
        with file_lock(center_csv + ".lock"):
            if not os.path.exists(center_csv) or os.path.getsize(center_csv) == 0:
                with open(center_csv, "w", newline="") as f:
                    csv.writer(f).writerow(["model", "metric"])
            self._ensure_csv_columns(center_csv, ["model", "metric"])
            self._remove_matching_csv_rows(center_csv, {"model": self.model_name})
            with open(center_csv, "a", newline="") as f:
                csv.writer(f).writerows(out_rows)
        print(f"✅ [Step2:cluster-forward-skip] center per-sample copied from pool → {center_csv}")
        return True

    def _forecasts_from_saved_repr_samples(self, samples: np.ndarray, center_idx: np.ndarray, dataset):
        from gluonts.model.forecast import QuantileForecast

        idx = np.asarray(center_idx, dtype=np.int64).reshape(-1)
        if idx.size == 0 or np.max(idx) >= samples.shape[0]:
            raise ValueError(f"center indices out of range for saved pool samples: max={np.max(idx) if idx.size else None}, n={samples.shape[0]}")
        selected_samples = np.asarray(samples[idx], dtype=np.float32)
        labels = list(dataset.test_data.label)
        if len(labels) != selected_samples.shape[0]:
            raise ValueError(f"center label/sample mismatch: labels={len(labels)}, samples={selected_samples.shape[0]}")
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        q_keys = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
        forecasts = []
        for sample_arr, label in zip(selected_samples, labels):
            arr = np.asarray(sample_arr, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            if arr.ndim == 2 and arr.shape[0] == len(quantiles):
                q_stack = arr
            elif arr.ndim == 1:
                q_stack = np.stack([arr for _ in quantiles], axis=0)
            else:
                q_stack = np.quantile(arr, quantiles, axis=0).astype(np.float32)
                if q_stack.ndim == 3 and q_stack.shape[-1] == 1:
                    q_stack = q_stack[..., 0]
            q_stack = np.asarray(q_stack, dtype=np.float32)
            if q_stack.ndim != 2:
                q_stack = q_stack.reshape((len(quantiles), -1))
            forecast_start = label.get("start", None) if isinstance(label, dict) else None
            if forecast_start is not None and not isinstance(forecast_start, pd.Period):
                forecast_start = pd.Period(forecast_start, freq=getattr(dataset, "freq", "H"))
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=q_stack,
                    forecast_keys=q_keys,
                    start_date=forecast_start,
                )
            )
        return forecasts, selected_samples

    def _try_replay_repr_center_from_pool(self, fixed_model_order=None) -> bool:
        if not bool(getattr(self.args, "skip_step2_cluster_forward", False)):
            return False
        if self.args.run_mode != "zoo_repr_set_forward":
            return False
        if str(getattr(self.args, "sample_mode", "")) != "cluster_nearest":
            print("[Step2:cluster-forward-skip] disabled: sample_mode is not cluster_nearest")
            return False

        from utils.path_utils import build_repr_eval_pool_name
        from selector.TSRouter_Select.sampled_repr_set import ReprDatasetAdapter

        center_idx, idx_note = self._load_repr_anchor_center_idx()
        if center_idx is None:
            print(f"[Step2:cluster-forward-skip] fallback to center forward: {idx_note}")
            return False

        pool_name = build_repr_eval_pool_name(self.args)
        pool_pkl = os.path.join(str(SAMPLED_REPR_POOL_CACHE_ROOT), pool_name + ".pkl")
        pool_meta = os.path.join(str(SAMPLED_REPR_POOL_CACHE_ROOT), pool_name + "_meta.pkl")
        pool_dataset_name = f"{pool_name}_freqH"
        pool_samples_path = self._repr_forward_samples_path(
            csv_path=getattr(self, "pool_csv_file_path", None),
            dataset_name=pool_dataset_name,
        )
        pool_status = self._repr_forward_result_status_at(
            csv_path=getattr(self, "pool_csv_file_path", None),
            per_sample_csv_path=getattr(self, "pool_per_sample_csv_file_path", None),
            ds_config=pool_dataset_name,
            require_samples=True,
        )
        pool_fresh, pool_fresh_reason = self._repr_pool_forward_fresh_for_pool()
        checks = {
            "pool_pkl": os.path.exists(pool_pkl),
            "pool_meta": os.path.exists(pool_meta),
            "pool_metrics": bool(pool_status["metrics"]),
            "pool_per_sample": bool(pool_status["per_sample"]),
            "pool_samples": bool(pool_status["samples"]),
            "pool_fresh": bool(pool_fresh),
        }
        print(f"[Step2:cluster-forward-skip] pool replay check: {checks}; status=({self._format_repr_forward_status(pool_status)})")
        if not all(checks.values()):
            print(
                "[Step2:cluster-forward-skip] fallback to center forward: "
                f"incomplete/stale pool artifacts {checks}, freshness={pool_fresh_reason}"
            )
            return False

        old_override = getattr(self.args, "repr_set_file_stem_override", "")
        old_save_repr_data_path = getattr(self.args, "save_repr_data_path", "")
        old_adapter_role = getattr(self.args, "repr_dataset_adapter_role", "")
        try:
            self.args.repr_dataset_adapter_role = "replay_labels"
            dataset = ReprDatasetAdapter(self.args, freq="H")
            try:
                pool_samples = np.load(pool_samples_path, mmap_mode="r")
            except Exception as exc:
                print(
                    "[Step2:cluster-forward-skip] fallback to center forward: "
                    f"cannot load pool samples {pool_samples_path}: {type(exc).__name__}: {exc}"
                )
                return False
            forecasts, selected_samples = self._forecasts_from_saved_repr_samples(pool_samples, center_idx, dataset)
            print(
                f"[Step2:cluster-forward-skip] GluonTS replay evaluate center metrics: "
                f"n={len(forecasts)}, source_pool_samples={pool_samples_path}; no TSFM forward"
            )
            metric_list = GE_RELEASED_METRICS if getattr(self.args, "GE_released", False) else DEFAULT_METRICS
            res = self._evaluate_forecasts(forecasts, dataset, metric_list)
            if not self._write_repr_per_sample_from_pool(center_idx):
                print("[Step2:cluster-forward-skip] fallback requested but failed to copy center per-sample from pool")
                return False
            old_force = getattr(self, "_force_nan_timing", False)
            old_mode = getattr(self, "_forced_evaluation_mode", None)
            old_skip_per = getattr(self, "_skip_repr_per_sample_save", False)
            old_samples_override = getattr(self, "_saved_forecast_samples_override", None)
            self._force_nan_timing = True
            self._forced_evaluation_mode = "pool_artifact_replay"
            self._skip_repr_per_sample_save = True
            self._saved_forecast_samples_override = selected_samples
            try:
                self.save_results(
                    res,
                    forecasts,
                    dataset.name,
                    dataset.name,
                    "repr",
                    np.nan,
                    np.nan,
                    dataset,
                    fixed_model_order,
                )
            finally:
                self._force_nan_timing = old_force
                if old_mode is None and hasattr(self, "_forced_evaluation_mode"):
                    delattr(self, "_forced_evaluation_mode")
                else:
                    self._forced_evaluation_mode = old_mode
                self._skip_repr_per_sample_save = old_skip_per
                if old_samples_override is None and hasattr(self, "_saved_forecast_samples_override"):
                    delattr(self, "_saved_forecast_samples_override")
                else:
                    self._saved_forecast_samples_override = old_samples_override
            print(
                f"[Step2:cluster-forward-skip] replayed center results from pool artifacts: "
                f"indices={center_idx.size}, pool_samples={pool_samples_path}"
            )
            return True
        finally:
            self.args.repr_set_file_stem_override = old_override
            self.args.save_repr_data_path = old_save_repr_data_path
            self.args.repr_dataset_adapter_role = old_adapter_role

    def _process_metrics_result_complete(self, ds_config: str) -> bool:
        if not bool(getattr(self.args, "enable_process_metrics", True)):
            return True
        if self.args.run_mode != "select" or self.model_name != "TSRouter":
            return True
        if not os.path.exists(self.csv_file_path):
            return False
        try:
            df = pd.read_csv(self.csv_file_path)
        except Exception:
            return False
        if df.empty or "dataset" not in df.columns or "model" not in df.columns:
            return False
        sub = df[
            (df["dataset"].astype(str) == str(ds_config))
            & (df["model"].astype(str).isin(self._model_name_aliases()))
        ]
        if sub.empty:
            return False
        required_columns = list(TSROUTER_REQUIRED_PROCESS_METRIC_COLUMNS)
        step3_metrics = load_encoder_enrichment_for_args(self.args)
        competence_columns = [
            "REGION_WEIGHTED_PURITY",
            "REGION_DIAG_RANK",
            "REGION_DELTA_RANK",
        ]
        if all(np.isfinite(float(step3_metrics.get(col, np.nan))) for col in competence_columns):
            required_columns.extend(competence_columns)
        existing = set(sub.columns)
        if not set(required_columns).issubset(existing):
            return False
        vals = sub.iloc[-1][required_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return vals.size > 0 and np.isfinite(vals).all()


    def get_predictor(self, dataset, batch_size):
        'TSRouter runtime message.'
        raise NotImplementedError('TSRouter runtime message.')

    def _build_ds_meta(self, ds_name, term):
        'TSRouter runtime message.'
        if "/" in ds_name:
            ds_key_raw, ds_freq = ds_name.split("/")
            ds_key = pretty_names.get(ds_key_raw.lower(), ds_key_raw.lower())
        else:
            ds_key_raw = ds_name
            ds_key = pretty_names.get(ds_key_raw.lower(), ds_key_raw.lower())
            ds_freq = dataset_properties_map[ds_key]["frequency"]

        ds_config = f"{ds_key}/{ds_freq}/{term}"
        dataset_name = f"{ds_key}_{ds_freq}_{term}"
        return ds_key, ds_freq, ds_config, dataset_name

    def _decide_univariate(self, ds_name, term):
        'TSRouter runtime message.'
        prefix = self.model_name.split("_")[0].lower()
        if (prefix in MULTIVAR_TSFM_PREFIXES or self.args.run_mode == "select"):
            to_univariate = False
        else:
                                                               
            to_univariate = (
                False
                if Dataset(name=ds_name, term=term, to_univariate=False).target_dim == 1
                else True
            )
        return to_univariate

    def _resolve_model_name_by_id(self, model_id: int) -> str | None:
        model_sizes = getattr(self, "Model_sizes", None)
        if not isinstance(model_sizes, dict):
            return None
        for family, variants in model_sizes.items():
            for variant, info in variants.items():
                if int(info.get("id", -1)) == int(model_id):
                    return f"{family}_{variant}"
        return None

    def _build_fast_eval_res(self, row: pd.Series) -> dict:
        def _safe_one(col: str) -> float:
            return float(pd.to_numeric(row.get(col, np.nan), errors="coerce"))

        def _safe_any(cols: list[str]) -> float:
            for col in cols:
                val = _safe_one(col)
                if not np.isnan(val):
                    return val
            return np.nan

        out = {
                                    
            "MASE[0.5]": np.array([_safe_any(["eval_metrics/MASE[0.5]","MASE",])]),
            "sMAPE[0.5]": np.array([_safe_any(["eval_metrics/sMAPE[0.5]","sMAPE",])]),
            "mean_weighted_sum_quantile_loss": np.array([_safe_any(["eval_metrics/mean_weighted_sum_quantile_loss","CRPS",])]),
        }
                          
        out["MSE[mean]"] = np.array([_safe_one("eval_metrics/MSE[mean]")])
        out["MSE[0.5]"] = np.array([_safe_one("eval_metrics/MSE[0.5]")])
        out["MAE[0.5]"] = np.array([_safe_one("eval_metrics/MAE[0.5]")])
        out["MAPE[0.5]"] = np.array([_safe_one("eval_metrics/MAPE[0.5]")])
        out["MSIS"] = np.array([_safe_one("eval_metrics/MSIS")])
        out["RMSE[mean]"] = np.array([_safe_one("eval_metrics/RMSE[mean]")])
        out["NRMSE[mean]"] = np.array([_safe_one("eval_metrics/NRMSE[mean]")])
        out["ND[0.5]"] = np.array([_safe_one("eval_metrics/ND[0.5]")])
        return out

    def _read_selected_model_saved_metric_with_runtime(self, ds_config: str, model_order) -> tuple[dict, float, float, str]:
        if model_order is None or len(model_order) == 0:
            raise ValueError("VLDB fast-eval: model_order is empty")
        selected_id = int(model_order[0])
        selected_model_name = self._resolve_model_name_by_id(selected_id)
        if selected_model_name is None:
            raise ValueError(f"VLDB fast-eval: cannot resolve selected model id={selected_id}")
        metric_dir = self._resolved_step4_eval_cl()
        metric_csv_path = resolve_tsfm_csv_path(selected_model_name, metric_dir, "all_results.csv")
        t0 = time.perf_counter()
        if not metric_csv_path.exists():
            raise FileNotFoundError(f"VLDB fast-eval: missing saved metric file {metric_csv_path}")
        metric_df = pd.read_csv(metric_csv_path)
        match = metric_df[metric_df["dataset"].astype(str) == str(ds_config)]
        if match.empty:
            raise ValueError(f"VLDB fast-eval: {metric_csv_path} missing dataset={ds_config}")
        row = match.iloc[-1]
        res = self._build_fast_eval_res(row)
        runtime_ms = float("nan")
        runtime_source = f"missing_{metric_dir}_forward_runtime"
        if "forward_runtime_seconds" in row.index:
            val = pd.to_numeric(pd.Series([row.get("forward_runtime_seconds")]), errors="coerce").iloc[0]
            if pd.notna(val) and np.isfinite(float(val)):
                runtime_ms = float(val) * 1000.0
                runtime_source = f"{metric_csv_path}:forward_runtime_seconds"
        return res, (time.perf_counter() - t0) * 1000.0, runtime_ms, runtime_source

    def _read_selected_model_saved_metric(self, ds_config: str, model_order) -> tuple[dict, float]:
        res, metric_read_ms, _, _ = self._read_selected_model_saved_metric_with_runtime(ds_config, model_order)
        return res, metric_read_ms

    def _set_route_selected_runtime_from_metric(
        self,
        model_order,
        selected_forward_ms: float,
        runtime_source: str,
    ) -> None:
        extra = getattr(self, "_last_selector_extra", {}) or {}
        if not isinstance(extra, dict):
            return
        route_row = dict(extra.get("vldb_route_latency_row", {}) or {})
        if not route_row:
            return
        route_overhead = pd.to_numeric(pd.Series([route_row.get("route_overhead_ms")]), errors="coerce").iloc[0]
        route_overhead = float(route_overhead) if pd.notna(route_overhead) and np.isfinite(float(route_overhead)) else 0.0
        if np.isfinite(float(selected_forward_ms)):
            route_row["selected_forecast_ms"] = f"{float(selected_forward_ms):.3f}"
            route_row["selected_forecast_timing_valid"] = "true"
            route_row["end_to_end_ms"] = f"{route_overhead + float(selected_forward_ms):.3f}"
        else:
            route_row["selected_forecast_ms"] = ""
            route_row["selected_forecast_timing_valid"] = "false"
            route_row["end_to_end_ms"] = f"{route_overhead:.3f}"
        route_row["selected_model_order"] = " ".join(map(str, model_order or []))
        note = str(route_row.get("timing_note", "") or "")
        route_row["timing_note"] = (
            f"{note} selected_forecast_source={runtime_source}; "
            f"E2E uses ROUTE plus selected TSFM full-dataset runtime from "
            f"{self._resolved_step4_eval_cl()} all_results.csv."
        ).strip()
        extra["vldb_route_latency_row"] = route_row
        self._last_selector_extra = extra

    def _use_ge_fast_eval_for_select(self) -> bool:
        if not bool(getattr(self.args, "GE_fast_eval", False)):
            return False
        if int(getattr(self.args, "ensemble_size", 1)) != 1:
            return False
        model_name = str(getattr(self.args, "models", ""))
        if model_name in {"TSRouter", "Real_Channel_Select"}:
            return True
        return (
            model_name in {"Real_Select", "Current_best_sMAPE_Rank"}
            and int(getattr(self.args, "restrict_top_model_num", 1)) == 1
        )

    def _use_ge_fast_dataset_cache_for_tsrouter(self) -> bool:
        return (
            self.args.run_mode == "select"
            and str(getattr(self.args, "models", "")) == "TSRouter"
            and self._use_ge_fast_eval_for_select()
        )

    def _finalize_vldb_route_latency(
        self,
        evaluate_ms: float = 0.0,
        metric_read_ms: float = 0.0,
        evaluation_mode: str = "gluonts_evaluate",
    ) -> None:
        if not str(getattr(self.args, "vldb_route_latency_log", "") or ""):
            return
        extra = getattr(self, "_last_selector_extra", {}) or {}
        if not isinstance(extra, dict):
            return
        route_row = dict(extra.get("vldb_route_latency_row", {}) or {})
        if not route_row:
            return
        route_overhead = float(route_row.get("route_overhead_ms", 0.0) or 0.0)
        selected_forecast = float(route_row.get("selected_forecast_ms", 0.0) or 0.0)
        route_row["evaluate_ms"] = f"{float(evaluate_ms):.3f}"
        route_row["metric_read_ms"] = f"{float(metric_read_ms):.3f}"
        route_row["fast_eval_enabled"] = "true" if (bool(getattr(self.args, "GE_fast_eval", False)) or bool(getattr(self.args, "vldb_skip_evaluate", False))) else "false"
        route_row["vldb_fast_eval"] = "true" if bool(getattr(self.args, "vldb_skip_evaluate", False)) else "false"
        route_row["evaluate_timing_valid"] = "false" if bool(getattr(self.args, "vldb_skip_evaluate", False)) else "true"
        route_row["evaluation_mode"] = str(evaluation_mode)
        route_row["end_to_end_ms"] = f"{route_overhead + selected_forecast:.3f}"
        note = str(route_row.get("timing_note", "") or "")
        if evaluation_mode == "saved_metric_fast_eval":
            note = (note + " " if note else "") + "evaluate_forecasts was skipped; metrics were read from saved TSFM results."
        else:
            note = (note + " " if note else "") + "evaluate_forecasts runtime is recorded separately in evaluate_ms."
        route_row["timing_note"] = note
        try:
            from selector.TSRouter_Select.tsrouter_select import _append_vldb_route_latency_row
            _append_vldb_route_latency_row(self.args, route_row)
        except Exception as e:
            print(f"⚠️ VLDB route latency sidecar not written: {e}")

    def _build_real_channel_fast_eval_res(self, ds_config: str, extra: dict) -> dict:
        model_sizes = getattr(self, "Model_sizes", None)
        if not isinstance(model_sizes, dict):
            raise ValueError("Real_Channel GE_fast_eval requires Model_sizes")
        model_cl_name = self._resolved_step4_eval_cl()
        selected = np.asarray(extra.get("selected_models_per_channel"), dtype=np.int64).reshape(-1)
        if selected.size == 0:
            raise ValueError("Real_Channel GE_fast_eval: selected_models_per_channel is empty")

        model_id_to_path = {}
        for family, size_dict in model_sizes.items():
            for size_name, info in size_dict.items():
                model_id_to_path[int(info["id"])] = resolve_tsfm_csv_path(
                    f"{family}_{size_name}",
                    model_cl_name,
                    "per_channel_results.csv",
                )

        df_cache: dict[int, pd.DataFrame] = {}

        def _row_for_selected(mid: int, ch: int) -> pd.Series:
            if mid not in model_id_to_path:
                raise FileNotFoundError(f"Real_Channel GE_fast_eval unknown model_id={mid}")
            if mid not in df_cache:
                csv_path = model_id_to_path[mid]
                if not csv_path.exists():
                    raise FileNotFoundError(f"Real_Channel GE_fast_eval missing per-channel file: {csv_path}")
                df_cache[mid] = pd.read_csv(csv_path)
            df = df_cache[mid]
            sub = df[
                df["dataset"].astype(str).eq(str(ds_config))
                & pd.to_numeric(df["channel"], errors="coerce").eq(int(ch))
            ]
            if sub.empty:
                raise FileNotFoundError(
                    f"Real_Channel GE_fast_eval missing row: dataset={ds_config}, model_id={mid}, channel={ch}"
                )
            return sub.iloc[-1]

        stat_cols = {
            "MASE": ("MASE_NUM", "MASE_DEN"),
            "sMAPE": ("SMAPE_NUM", "SMAPE_DEN"),
            "CRPS": ("CRPS_NUM", "CRPS_DEN"),
        }
        metric_values = {}
        channels_ref = None
        for metric in ["MASE", "sMAPE", "CRPS"]:
            matrix, model_ids, channels = load_per_channel_error_matrix(
                model_sizes=model_sizes,
                dataset_name=ds_config,
                model_cl_name=model_cl_name,
                rank_metric=metric,
                require_complete=True,
            )
            if channels_ref is None:
                channels_ref = list(channels)
            elif list(channels) != list(channels_ref):
                raise FileNotFoundError(
                    f"Real_Channel GE_fast_eval channel mismatch for dataset={ds_config}, metric={metric}: "
                    f"expected={channels_ref}, got={channels}"
                )
            id_to_row = {int(mid): i for i, mid in enumerate(model_ids)}
            vals = []
            num_col, den_col = stat_cols[metric]
            num_sum = 0.0
            den_sum = 0.0
            for pos, ch in enumerate(channels):
                mid = int(selected[pos]) if pos < selected.size else int(selected[-1])
                if mid not in id_to_row:
                    vals.append(np.nan)
                else:
                    vals.append(float(matrix[id_to_row[mid], pos]))
                row = _row_for_selected(mid, int(ch))
                if num_col not in row.index or den_col not in row.index:
                    raise FileNotFoundError(
                        f"Real_Channel GE_fast_eval requires numerator/denominator columns in per-channel results; "
                        f"missing {num_col}/{den_col} for dataset={ds_config}. "
                        "Please rerun zoo baselines after the per-channel metric fix."
                    )
                num_val = float(pd.to_numeric(row.get(num_col), errors="coerce"))
                den_val = float(pd.to_numeric(row.get(den_col), errors="coerce"))
                if not np.isfinite(num_val) or not np.isfinite(den_val) or den_val <= 0:
                    raise FileNotFoundError(
                        f"Real_Channel GE_fast_eval invalid metric stats for dataset={ds_config}, "
                        f"metric={metric}, model_id={mid}, channel={ch}, num={num_val}, den={den_val}"
                    )
                num_sum += num_val
                den_sum += den_val
            if vals and not np.isfinite(vals).all():
                bad = [
                    {
                        "channel": int(channels[i]),
                        "model_id": int(selected[i] if i < selected.size else selected[-1]),
                    }
                    for i, v in enumerate(vals)
                    if not np.isfinite(v)
                ]
                raise FileNotFoundError(
                    f"Real_Channel GE_fast_eval non-finite selected metrics for dataset={ds_config}, "
                    f"metric={metric}, bad={bad[:20]}"
                )
            metric_values[metric] = float(num_sum / den_sum) if den_sum > 0 else float("nan")
        bad_metric_values = {k: v for k, v in metric_values.items() if not np.isfinite(v)}
        if bad_metric_values:
            raise FileNotFoundError(
                f"Real_Channel GE_fast_eval invalid aggregate metrics for dataset={ds_config}: {bad_metric_values}"
            )
        print(
            f"[Real_Channel][GE_fast_eval] dataset={ds_config}, "
            f"model_cl={model_cl_name}, channels={len(channels_ref or [])}, "
            f"selected_head={selected[:10].tolist()}, "
            f"MASE={metric_values['MASE']:.4f}, sMAPE={metric_values['sMAPE']:.4f}, CRPS={metric_values['CRPS']:.4f}"
        )
        return {
            "MASE[0.5]": np.array([metric_values["MASE"]]),
            "sMAPE[0.5]": np.array([metric_values["sMAPE"]]),
            "mean_weighted_sum_quantile_loss": np.array([metric_values["CRPS"]]),
            "MSE[mean]": np.array([np.nan]),
            "MSE[0.5]": np.array([np.nan]),
            "MAE[0.5]": np.array([np.nan]),
            "MAPE[0.5]": np.array([np.nan]),
            "MSIS": np.array([np.nan]),
            "RMSE[mean]": np.array([np.nan]),
            "NRMSE[mean]": np.array([np.nan]),
            "ND[0.5]": np.array([np.nan]),
        }

    def _preload_ge_fast_dataset_cache(self):
        if getattr(self, "_ge_fast_dataset_loader", None) is None:
            self._ge_fast_dataset_loader = FastEvalDatasetCacheLoader(
                all_datasets=self.args.all_datasets,
                med_long_datasets=self.args.med_long_datasets,
                build_ds_meta_fn=self._build_ds_meta,
                decide_univariate_fn=self._decide_univariate,
                cache_only=bool(getattr(self.args, "route_cache_only", False)),
                metadata_path=CHANNEL_META_PATH,
            )
        cache_key = (
            tuple(map(str, getattr(self.args, "all_datasets", []))),
            tuple(str(getattr(self.args, "med_long_datasets", "")).split()),
            bool(getattr(self.args, "route_cache_only", False)),
        )
        shared_cache = BaseModel._ge_fast_dataset_shared_cache.get(cache_key)
        if shared_cache is not None:
            self._ge_fast_dataset_loader._cache = shared_cache
            return
        if getattr(self._ge_fast_dataset_loader, "_cache", None) is not None:
            BaseModel._ge_fast_dataset_shared_cache[cache_key] = self._ge_fast_dataset_loader._cache
            return
        cache = self._ge_fast_dataset_loader.preload()
        BaseModel._ge_fast_dataset_shared_cache[cache_key] = cache
        print(f"TSRouter runtime message: {len(cache)}TSRouter runtime message: ")

    def _build_fast_eval_dataset_stub(self, ds_config: str):
        if getattr(self, "_ge_fast_dataset_loader", None) is None:
            self._preload_ge_fast_dataset_cache()
        return self._ge_fast_dataset_loader.build_stub(ds_config)

    def _log_step4_task_lengths(self, ds_config: str, dataset) -> None:
        try:
            pred_len = int(getattr(dataset, "prediction_length"))
            input_obj = getattr(getattr(dataset, "test_data", None), "input", None)
            lengths = []
            if input_obj is not None:
                for entry in input_obj:
                    target = entry.get("target") if isinstance(entry, dict) else None
                    if target is None:
                        continue
                    arr = np.asarray(target)
                    if arr.ndim == 1:
                        lengths.append(int(arr.shape[0]))
                    elif arr.ndim >= 2:
                        lengths.append(int(arr.shape[-1]))
            if lengths:
                vals = np.asarray(lengths, dtype=float)
                print(
                    f"[Step4TaskLen] "
                    f"context_len_avg={vals.mean():.2f}, context_len_min={int(vals.min())}, "
                    f"context_len_max={int(vals.max())}, windows={len(lengths)}, pred_len={pred_len}"
                )
            else:
                print(f"[Step4TaskLen] context_len_avg=NA, pred_len={pred_len}")
        except Exception as e:
            print(f"[Step4TaskLen] failed_to_log={e}")



    def _parse_model_order_str(self, s):
        'TSRouter runtime message.'
        if isinstance(s, (list, tuple, np.ndarray)):
            return [int(x) for x in s]

        s = str(s).strip()

        if s.startswith("[") and s.endswith("]"):
            body = s[1:-1].strip()
            if not body:
                return []
            return [int(x) for x in body.replace(",", " ").split()]

        return None

    def _preload_real_order_cache_for_ge_fast(self):
        'TSRouter runtime message.'
        if hasattr(self, "_ge_fast_real_order_cache") and self._ge_fast_real_order_cache is not None:
            return

        current_zoo_num = getattr(self.args, "current_zoo_num", 9)
        zoo_total_num = getattr(self.args, "zoo_total_num", 9)

        ensemble_size = int(getattr(self.args, "ensemble_size", 1))
        ensemble_agg = str(getattr(self.args, "ensemble_agg", "median"))
        metrics_to_load = ["MASE", "sMAPE", "CRPS"]
        real_paths = {
            metric: BASELINE_CSV_ROOT
            / "selectors"
            / "Real_Select"
            / f"zoo{current_zoo_num}-{zoo_total_num}_top{ensemble_size}-{ensemble_agg}_real_{metric}_all_results.csv"
            for metric in metrics_to_load
        }

        cache = {metric: {} for metric in metrics_to_load}

        for metric, csv_path in real_paths.items():
            if not csv_path.exists():
                print(f"TSRouter runtime message: {csv_path}")
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"TSRouter runtime message: {csv_path}TSRouter runtime message: {e}")
                continue

            if "dataset" not in df.columns or "model_order" not in df.columns:
                print(f"⚠️ GE_fast_eval preload: {csv_path}TSRouter runtime message: ")
                continue

            metric_cache = {}
            for _, row in df.iterrows():
                ds = str(row["dataset"])
                order = self._parse_model_order_str(row["model_order"])
                metric_cache[ds] = order

            cache[metric] = metric_cache
            print(f"✅ GE_fast_eval preload: {metric}TSRouter runtime message: {len(metric_cache)}TSRouter runtime message: {csv_path}")

        self._ge_fast_real_order_cache = cache

    def _load_real_orders_for_dataset(self, ds_config: str) -> dict:
        'TSRouter runtime message.'
        cache = getattr(self, "_ge_fast_real_order_cache", None)
        if cache is None:
            self._preload_real_order_cache_for_ge_fast()
            cache = getattr(self, "_ge_fast_real_order_cache", {})

        return {
            "MASE": cache.get("MASE", {}).get(str(ds_config)),
            "sMAPE": cache.get("sMAPE", {}).get(str(ds_config)),
            "CRPS": cache.get("CRPS", {}).get(str(ds_config)),
        }
    def _make_forecasts(self, dataset, dataset_name, ds_config, fixed_model_order, debug_mode, cached_search_input=None):
        'TSRouter runtime message.'
        model_order = None
        batch_size = self.batch_size
        requested_batch_size = int(batch_size)
        min_batch_size = int(batch_size)
        batch_size_fallback_count = 0
        configured_context_length = int(getattr(self.args, "context_len", 0) or 0)
        has_fixed_context_length = (
            bool(getattr(self.args, "fix_context_len", False))
            and configured_context_length > 0
        )
        requested_context_length = configured_context_length if has_fixed_context_length else np.nan
        runtime_context_length = requested_context_length
        min_context_length = requested_context_length
        context_length_fallback_count = 0
        original_context_length = getattr(self.args, "context_len", None)
        self._last_selector_extra = {}
        self._last_selector_prediction_cache = None
        self._reset_dataset_timing()

        oom_retry_wait_seconds = max(0, int(getattr(self.args, "oom_retry_wait_seconds", 30)))
        oom_retry_max_retries = max(0, int(getattr(self.args, "oom_retry_max_retries", 100)))
        oom_retry_count_at_bs1 = 0
        allow_context_len_fallback = (
            bool(getattr(self.args, "allow_context_len_fallback", False))
            and has_fixed_context_length
        )

        def _cleanup_cuda_after_oom():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except RuntimeError:
                    pass

        def _schedule_cuda_oom_retry(reason, detail=None):
            nonlocal batch_size, oom_retry_count_at_bs1
            nonlocal min_batch_size, batch_size_fallback_count
            nonlocal runtime_context_length, min_context_length, context_length_fallback_count

            if batch_size > 1:
                next_batch_size = max(1, batch_size // 2)
                if detail:
                    print(f"⚠️ {detail}\nReducing to {next_batch_size}")
                else:
                    print(f"⚠️ {reason} at batch_size {batch_size}, reducing to {next_batch_size}")
                batch_size = next_batch_size
                min_batch_size = min(int(min_batch_size), int(batch_size))
                batch_size_fallback_count += 1
                oom_retry_count_at_bs1 = 0
                return 0

            if allow_context_len_fallback:
                current_context_length = max(1, int(getattr(self.args, "context_len", 1) or 1))
                next_context_length = max(1, current_context_length // 2)
                if next_context_length < current_context_length:
                    self.args.context_len = int(next_context_length)
                    runtime_context_length = int(next_context_length)
                    min_context_length = min(int(min_context_length), int(next_context_length))
                    context_length_fallback_count += 1
                    oom_retry_count_at_bs1 = 0
                    fallback_message = (
                        f"context_len fallback {current_context_length} -> {next_context_length} "
                        f"(count={context_length_fallback_count}), retrying immediately"
                    )
                    if detail:
                        print(f"⚠️ {detail}\n{fallback_message}")
                    else:
                        print(f"⚠️ {reason} at batch_size 1; {fallback_message}")
                    return 0
                print(
                    f"❌ {reason} at batch_size 1 and context_len {current_context_length}; "
                    "context_len fallback is enabled but cannot reduce context_len further, raising."
                )
                return None

            if oom_retry_count_at_bs1 >= oom_retry_max_retries:
                print(
                    f"❌ {reason} at batch_size 1; "
                    f"failed after {oom_retry_max_retries} retries, raising."
                )
                return None

            oom_retry_count_at_bs1 += 1
            if detail:
                print(
                    f"⚠️ {detail}\n"
                    f"Waiting {oom_retry_wait_seconds}s before retry "
                    f"{oom_retry_count_at_bs1}/{oom_retry_max_retries} at batch_size 1"
                )
            else:
                print(
                    f"⚠️ {reason} at batch_size 1; waiting {oom_retry_wait_seconds}s "
                    f"before retry {oom_retry_count_at_bs1}/{oom_retry_max_retries}"
                )
            return oom_retry_wait_seconds

        def _record_batch_size_state() -> None:
            self._last_requested_batch_size = int(requested_batch_size)
            self._last_runtime_batch_size = int(batch_size)
            self._last_min_batch_size = int(min_batch_size)
            self._last_batch_size_fallback_count = int(batch_size_fallback_count)
            self._last_requested_context_length = requested_context_length
            self._last_runtime_context_length = runtime_context_length
            self._last_min_context_length = min_context_length
            self._last_context_length_fallback_count = int(context_length_fallback_count)

        if self.args.run_mode in {"zoo", "zoo_repr_set_forward", "zoo_task_sample_forward", "zoo_task_probe_select"}:

            # if debug_mode:
            #     debug_dataset_brief(dataset, tag=f"{ds_config}|before_predictor")

            try:
                while True:
                    retry_delay_seconds = None
                    try:
                        t_forward0 = time.perf_counter()
                        t_load0 = time.perf_counter()
                        predictor = self.get_predictor(dataset, batch_size)
                        self._last_model_load_ms = (time.perf_counter() - t_load0) * 1000.0

                        if debug_mode:
                            # debug_predictor_brief(predictor, tag=f"{ds_config}|predictor_ready")
                                    
                            # debug_print_test_input(dataset)
                            debug_check_input_nan(dataset.test_data.input)

                        t_predict0 = time.perf_counter()
                        forecasts = list(
                            tqdm(
                                predictor.predict(dataset.test_data.input),
                                total=len(dataset.test_data.input),
                                desc=f"Predicting {ds_config}",
                            )
                        )
                        self._last_predict_ms = (time.perf_counter() - t_predict0) * 1000.0
                        self._last_forward_ms = (time.perf_counter() - t_forward0) * 1000.0
                        # if debug_mode:
                        #     debug_forecast_brief(forecasts, tag=f"{ds_config}|after_predict")

                        break

                    except torch.cuda.OutOfMemoryError:
                        retry_delay_seconds = _schedule_cuda_oom_retry("OOM")
                        if retry_delay_seconds is None:
                            _cleanup_cuda_after_oom()
                            raise

                    except RuntimeError as e:
                        err_msg = str(e).lower()

                        cuda_batch_related_error = (
                                "cuda error: invalid configuration argument" in err_msg
                                or "out of memory" in err_msg
                                or "cublas_status_alloc_failed" in err_msg
                                or "cuda out of memory" in err_msg
                        )

                        if cuda_batch_related_error:
                            retry_delay_seconds = _schedule_cuda_oom_retry(
                                "CUDA runtime error",
                                detail=f"CUDA runtime error at batch_size {batch_size}: {e}",
                            )
                            if retry_delay_seconds is None:
                                _cleanup_cuda_after_oom()
                                raise
                        else:
                            raise

                    _cleanup_cuda_after_oom()
                    if retry_delay_seconds and retry_delay_seconds > 0:
                        time.sleep(retry_delay_seconds)
            finally:
                if original_context_length is not None:
                    self.args.context_len = original_context_length
            _record_batch_size_state()

        elif self.args.run_mode == "select":
            predictor = self.get_predictor(dataset, batch_size)
            use_ge_fast = self._use_ge_fast_eval_for_select()
            if use_ge_fast:
                if (
                        str(getattr(self.args, "models", "")) == "TSRouter"
                        and cached_search_input is not None
                        and hasattr(self, "get_model_order_from_search_input")
                        and not str(getattr(self.args, "vldb_route_latency_log", "") or "")
                ):
                    model_order = self.get_model_order_from_search_input(
                        dataset_name,
                        cached_search_input,
                        dataset=dataset,
                        test_data_input=getattr(getattr(dataset, "test_data", None), "input", None),
                    )
                    self._last_selector_extra = getattr(self, "last_selector_extra", {})
                else:
                    if dataset is None:
                        raise ValueError('TSRouter runtime message.')
                    predictor = self.get_predictor(dataset, batch_size)
                    sel_out = predictor.select_strategy(dataset_name, dataset.test_data.input)
                    if isinstance(sel_out, tuple) and len(sel_out) >= 2:
                        model_order = sel_out[0]
                        if len(sel_out) >= 3:
                            self._last_selector_extra = sel_out[2]
                    else:
                        raise ValueError('TSRouter runtime message.')
                if model_order is None or len(model_order) == 0:
                    raise ValueError('TSRouter runtime message.')
                if (
                    str(getattr(self.args, "models", "")) == "Real_Channel_Select"
                    or (
                        str(getattr(self.args, "models", "")) == "TSRouter"
                        and int(getattr(self.args, "restrict_top_model_num", 1)) != 1
                    )
                ):
                    res = self._build_real_channel_fast_eval_res(ds_config, getattr(self, "_last_selector_extra", {}) or {})
                    forecasts = []
                    self._last_forward_ms = 0.0
                    self._last_evaluate_ms = 0.0
                    self._last_metric_read_ms = 0.0
                    self._last_eval_skipped = True
                    self._last_evaluation_mode = "saved_metric_fast_eval"
                    self._finalize_vldb_route_latency(
                        evaluate_ms=0.0,
                        metric_read_ms=0.0,
                        evaluation_mode="saved_metric_fast_eval",
                    )
                    _record_batch_size_state()
                    return res, forecasts, model_order

                selected_id = int(model_order[0])

                if str(getattr(self.args, "models", "")) == "TSRouter":
                    print(
                        f"TSRouter runtime message: {selected_id}TSRouter runtime message: {model_order}"
                    )
                else:
                    real_orders = self._load_real_orders_for_dataset(ds_config)
                    real_order_mase = real_orders.get("MASE")
                    real_order_smape = real_orders.get("sMAPE")
                    real_order_crps = real_orders.get("CRPS")

                    print(
                        f"TSRouter runtime message: {selected_id}TSRouter runtime message: {model_order} | "
                        f"TSRouter runtime message: {real_order_mase} | "
                        f"TSRouter runtime message: {real_order_smape} | "
                        f"TSRouter runtime message: {real_order_crps}"
                    )

                selected_model_name = self._resolve_model_name_by_id(selected_id)
                if selected_model_name is None:
                    raise ValueError(f"TSRouter runtime message: {selected_id}TSRouter runtime message: ")

                ge_cl_dir = self._resolved_step4_eval_cl()
                ge_csv_path = resolve_tsfm_csv_path(selected_model_name, ge_cl_dir, "all_results.csv")
                t_metric0 = time.perf_counter()
                if not ge_csv_path.exists():
                    raise FileNotFoundError(f"TSRouter runtime message: {ge_csv_path}")
                ge_df = pd.read_csv(ge_csv_path)
                match = ge_df[ge_df["dataset"].astype(str) == str(ds_config)]
                if match.empty:
                    raise ValueError(f"GE_fast_eval: {selected_model_name}TSRouter runtime message: {ge_csv_path}TSRouter runtime message: {ds_config}")
                res = self._build_fast_eval_res(match.iloc[-1])
                metric_read_ms = (time.perf_counter() - t_metric0) * 1000.0
                forecasts = []
                self._last_forward_ms = 0.0
                self._last_evaluate_ms = 0.0
                self._last_metric_read_ms = metric_read_ms
                self._last_eval_skipped = True
                self._last_evaluation_mode = "saved_metric_fast_eval"
                self._finalize_vldb_route_latency(
                    evaluate_ms=0.0,
                    metric_read_ms=metric_read_ms,
                    evaluation_mode="saved_metric_fast_eval",
                )
                _record_batch_size_state()
                return res, forecasts, model_order
            else:
                if dataset is None:
                    raise ValueError('TSRouter runtime message.')
                if (
                    str(getattr(self.args, "models", "")) == "Task_Probe_Forward_Select"
                    and bool(getattr(self.args, "vldb_skip_evaluate", False))
                    and int(getattr(self.args, "ensemble_size", 1)) == 1
                ):
                    t_select0 = time.perf_counter()
                    predictor = self.get_predictor(dataset, batch_size)
                    sel_out = predictor.select_strategy(dataset_name, dataset.test_data.input)
                    model_order, ensemble_size, selector_extra = predictor._parse_select_output(sel_out)
                    predictor.last_selector_extra = {"model_order": model_order, **(selector_extra or {})}
                    self._last_selector_extra = getattr(predictor, "last_selector_extra", {})
                    res, metric_read_ms, selected_forward_ms, runtime_source = self._read_selected_model_saved_metric_with_runtime(
                        ds_config,
                        model_order,
                    )
                    self._set_route_selected_runtime_from_metric(model_order, selected_forward_ms, runtime_source)
                    self._last_forward_ms = (time.perf_counter() - t_select0) * 1000.0
                    self._last_evaluate_ms = 0.0
                    self._last_metric_read_ms = metric_read_ms
                    self._last_eval_skipped = True
                    self._last_evaluation_mode = "saved_metric_fast_eval"
                    self._finalize_vldb_route_latency(
                        evaluate_ms=0.0,
                        metric_read_ms=metric_read_ms,
                        evaluation_mode="saved_metric_fast_eval",
                    )
                    _record_batch_size_state()
                    return res, [], model_order
                t_forward0 = time.perf_counter()
                predictor = self.get_predictor(dataset, batch_size)
                forecast_iter, model_order = predictor.predict(
                    dataset.test_data.input, dataset_name, fixed_model_order
                )
                self._last_selector_extra = getattr(predictor, "last_selector_extra", {})
                self._last_selector_prediction_cache = getattr(predictor, "last_prediction_cache", None)
                forecasts = list(forecast_iter)
                self._last_forward_ms = (time.perf_counter() - t_forward0) * 1000.0
                if bool(getattr(self.args, "vldb_skip_evaluate", False)):
                    res, metric_read_ms = self._read_selected_model_saved_metric(ds_config, model_order)
                    self._last_evaluate_ms = 0.0
                    self._last_metric_read_ms = metric_read_ms
                    self._last_eval_skipped = True
                    self._last_evaluation_mode = "saved_metric_fast_eval"
                    self._finalize_vldb_route_latency(
                        evaluate_ms=0.0,
                        metric_read_ms=metric_read_ms,
                        evaluation_mode="saved_metric_fast_eval",
                    )
                    _record_batch_size_state()
                    return res, forecasts, model_order

        else:
            raise ValueError(f"Unknown run_mode={self.args.run_mode}; supported: zoo / select / zoo_repr_set_forward / zoo_task_sample_forward / zoo_task_probe_select")

        _record_batch_size_state()

        if debug_mode:
            debug_forecasts(forecasts)

                                  
        if self.args.run_mode == "zoo_task_sample_forward":
            self._last_evaluate_ms = 0.0
            self._last_metric_read_ms = 0.0
            self._last_eval_skipped = True
            self._last_evaluation_mode = "task_sample_forward_only"
            res = {
                "MASE[0.5]": np.array([np.nan]),
                "sMAPE[0.5]": np.array([np.nan]),
                "mean_weighted_sum_quantile_loss": np.array([np.nan]),
            }
            return res, forecasts, model_order

        if self.args.run_mode == "zoo" and bool(getattr(self.args, "vldb_skip_evaluate", False)):
            res_fast, metric_read_ms, fast_eval_note = self._read_saved_tsfm_metric_for_dataset(ds_config)
            if res_fast is not None:
                self._last_evaluate_ms = 0.0
                self._last_metric_read_ms = metric_read_ms
                self._last_eval_skipped = True
                self._last_evaluation_mode = "saved_metric_fast_eval"
                print(
                    f"[TIMING] {self.model_name} {ds_config}: "
                    f"forward={float(self._last_forward_ms) / 1000.0:.3f}s, "
                    f"eval=skipped, metric_read={metric_read_ms / 1000.0:.3f}s | {fast_eval_note}"
                )
                return res_fast, forecasts, model_order
            print(
                f"[TIMING] {self.model_name} {ds_config}: saved metrics unavailable for fast-eval "
                f"({fast_eval_note}); falling back to evaluate_forecasts"
            )

        metric_list = GE_RELEASED_METRICS if getattr(self.args, "GE_released", False) else DEFAULT_METRICS
        t_eval0 = time.perf_counter()
        res = self._evaluate_forecasts(forecasts, dataset, metric_list)
        evaluate_ms = (time.perf_counter() - t_eval0) * 1000.0
        self._last_evaluate_ms = evaluate_ms
        self._last_metric_read_ms = 0.0
        self._last_eval_skipped = False
        self._last_evaluation_mode = str(getattr(self, "_active_evaluation_mode", "gluonts_evaluate"))
        if self.args.run_mode != "zoo_repr_set_forward":
            print(
                f"[TIMING] {self.model_name} {ds_config}: "
                f"forward={float(self._last_forward_ms) / 1000.0:.3f}s, "
                f"eval={evaluate_ms / 1000.0:.3f}s, metric_read=0.000s, "
                f"eval_backend={self._last_evaluation_mode}, "
                f"eval_batch={max(1, int(getattr(self.args, 'eval_batch_size', 1024) or 1024))}"
            )
        self._finalize_vldb_route_latency(
            evaluate_ms=evaluate_ms,
            metric_read_ms=0.0,
            evaluation_mode=self._last_evaluation_mode,
        )

        return res, forecasts, model_order

    # ==============================================================
    # Task-Probe sample-window forward mode
    # ==============================================================

    def _append_task_probe_sample_rows(self, path: str, rows: list[dict], columns: list[str]) -> None:
        if not rows:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        exists = os.path.exists(path)
        with file_lock(path + ".lock"):
            exists = os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
                if not exists:
                    writer.writeheader()
                for row in rows:
                    writer.writerow({c: row.get(c, "") for c in columns})

    def _task_probe_sample_config_row(self) -> dict:
        return {
            "sample_repr_num": int(getattr(self.args, "sample_repr_num", 0) or 0),
            "task_window_sample_strategy": str(getattr(self.args, "task_window_sample_strategy", "legacy") or "legacy"),
            "sample_repr_ratio": str(getattr(self.args, "sample_repr_ratio", 0) or 0),
            "task_sample_version": str(getattr(self.args, "task_sample_version", "")),
            "search_seed": str(getattr(self.args, "search_seed", "")),
            "repr_scale_protocol": str(getattr(self.args, "repr_scale_protocol", "")),
        }

    def _task_probe_model_id(self) -> int:
        try:
            return int(All_sorted_model_names.index(str(self.model_name)))
        except ValueError:
            return -1

    def _task_probe_sample_forward_complete(self, ds_config: str) -> bool:
        if not os.path.exists(self.csv_file_path):
            return False
        try:
            df = pd.read_csv(self.csv_file_path, low_memory=False)
        except Exception:
            try:
                df = pd.read_csv(self.csv_file_path, engine="python", on_bad_lines="skip")
            except Exception:
                return False
        required = {"stage", "dataset", "model_id", "sample_forward_ms", "selected_windows", "status"}
        if df.empty or not required.issubset(set(df.columns)):
            return False
        sub = df[
            pd.to_numeric(df["stage"], errors="coerce").eq(float(getattr(self.args, "current_zoo_num", 0) or 0))
            & df["dataset"].astype(str).eq(str(ds_config))
            & pd.to_numeric(df["model_id"], errors="coerce").eq(float(self._task_probe_model_id()))
        ].copy()
        cfg = self._task_probe_sample_config_row()
        for col, expected in cfg.items():
            if col not in sub.columns:
                return False
            sub = sub[sub[col].astype(str).str.lower().eq(str(expected).lower())]
        if sub.empty:
            return False
        sub = sub[sub["status"].astype(str).str.lower().isin({"success", "ok", "executed", "cache_hit"})]
        if sub.empty:
            return False
        latest = sub.iloc[-1]
        fwd = pd.to_numeric(pd.Series([latest.get("sample_forward_ms")]), errors="coerce").iloc[0]
        n = pd.to_numeric(pd.Series([latest.get("selected_windows")]), errors="coerce").iloc[0]
        return pd.notna(fwd) and np.isfinite(float(fwd)) and pd.notna(n) and float(n) > 0

    def _task_probe_prediction_length(self, ds_key: str, ds_freq: str, term: str) -> int:
        freq = norm_freq_str(to_offset(ds_freq).name)
        if freq.endswith("E"):
            freq = freq[:-1]
        base = M4_PRED_LENGTH_MAP[freq] if "m4" in str(ds_key).lower() else PRED_LENGTH_MAP[freq]
        multiplier = {"short": 1, "medium": 10, "long": 15}.get(str(term).lower(), 1)
        return int(base) * int(multiplier)

    def _task_probe_load_cache(self) -> tuple[dict, dict, str]:
        cache_path = get_gift_eval_task_repr_cache_path(
            self.args,
            search_context_len=int(getattr(self.args, "repr_input_dim", getattr(self.args, "context_len", 512))),
        )
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        meta_path = f"{cache_path}.meta.json"
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                meta = loaded
        if not isinstance(cache, dict):
            raise ValueError(f"Task-Probe sample cache must be a dict: {cache_path}")
        return cache, meta, cache_path

    def _task_probe_sample_dataset_from_cache(self, ds_config: str, ds_key: str, ds_freq: str, term: str, cache: dict, meta_all: dict):
        arr = cache.get(ds_config)
        if arr is None:
            arr = cache.get(str(ds_config).replace("_", "/", 2))
        if arr is None:
            raise FileNotFoundError(f"missing Step4 sample cache row for dataset={ds_config}")
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Step4 sample cache expects (K,T,C), got shape={arr.shape} for {ds_config}")
        meta = meta_all.get(ds_config) or meta_all.get(str(ds_config).replace("_", "/", 2)) or {}
        entry_indices = meta.get("entry_indices", []) if isinstance(meta, dict) else []
        if not isinstance(entry_indices, list):
            entry_indices = []
        k, _t, c = arr.shape
        pred_len = self._task_probe_prediction_length(ds_key, ds_freq, term)
        prefix = self.model_name.split("_")[0].lower()
        use_multivar = prefix in MULTIVAR_TSFM_PREFIXES
        inputs = []
        step4_ids = []
        start = pd.Period("2000-01-01 00:00:00", freq=str(ds_freq))
        for sample_idx in range(k):
            original_idx = entry_indices[sample_idx] if sample_idx < len(entry_indices) else sample_idx
            if use_multivar or c == 1:
                target = arr[sample_idx].T if c > 1 else arr[sample_idx, :, 0]
                inputs.append({"target": target, "start": start, "item_id": f"{ds_config}#{sample_idx}"})
                step4_ids.append(int(original_idx))
            else:
                for ch in range(c):
                    target = arr[sample_idx, :, ch]
                    inputs.append({"target": target, "start": start, "item_id": f"{ds_config}#{sample_idx}_dim{ch}"})
                    step4_ids.append(int(original_idx))
        return SimpleNamespace(
            name=ds_config,
            freq=str(ds_freq),
            prediction_length=pred_len,
            target_dim=int(c if use_multivar else 1),
            past_feat_dynamic_real_dim=0,
            windows=len(inputs),
            test_data=SimpleNamespace(input=inputs, label=[]),
            step4_entry_indices=step4_ids,
            step4_cache_shape=list(arr.shape),
        )

    def _task_probe_sample_error_rows_from_saved_per_window(self, ds_config: str, sample_dataset, cache_path: str) -> list[dict]:
        per_window_path = per_window_results_path(self.output_dir)
        if not os.path.exists(per_window_path):
            return []
        try:
            df = pd.read_csv(per_window_path)
        except Exception:
            return []
        if "dataset" not in df.columns:
            return []
        sub = df[df["dataset"].astype(str).eq(str(ds_config))].copy()
        if sub.empty:
            return []
        keys = [str(x) for x in getattr(sample_dataset, "step4_entry_indices", [])]
        key_set = set(keys)
        original_c = 1
        try:
            original_c = int(getattr(sample_dataset, "step4_cache_shape", [0, 0, 1])[2])
        except Exception:
            original_c = 1
        target_dim = int(getattr(sample_dataset, "target_dim", 1) or 1)
        prefer_entry = not (target_dim == 1 and original_c > 1)
        if prefer_entry and "entry" in sub.columns:
            sub = sub[sub["entry"].astype(str).isin(key_set)]
            entry_col = "entry"
        elif "window_id" in sub.columns:
            sub = sub[sub["window_id"].astype(str).isin(key_set)]
            entry_col = "window_id"
        elif "entry" in sub.columns:
            sub = sub[sub["entry"].astype(str).isin(key_set)]
            entry_col = "entry"
        else:
            return []
        rows = []
        base = {
            "timestamp_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "stage": int(getattr(self.args, "current_zoo_num", 0) or 0),
            "dataset": ds_config,
            "model_id": self._task_probe_model_id(),
            "model_key": self.model_name,
            "model_abbr": Model_abbrev_map.get(self.model_name, self.model_name),
            "step4_task_cache_path": cache_path,
            **self._task_probe_sample_config_row(),
        }
        for local_entry, (_, row) in enumerate(sub.iterrows()):
            out = dict(base)
            key_value = row.get(entry_col, "")
            out.update(
                {
                    "local_entry": local_entry,
                    "step4_entry_idx": key_value,
                    "window_id": row.get("window_id", ""),
                    "MASE": row.get("MASE", ""),
                    "sMAPE": row.get("sMAPE", ""),
                    "CRPS": row.get("CRPS", ""),
                }
            )
            for col in TASK_PROBE_SAMPLE_ERROR_COLUMNS:
                if col in row and col not in out:
                    out[col] = row.get(col, "")
            rows.append(out)
        return rows

    def _run_task_probe_sample_forward(self) -> None:
        cache, meta_all, cache_path = self._task_probe_load_cache()
        planned_configs = []
        for ds_name in self.args.all_datasets:
            for term in ["short", "medium", "long"]:
                if (term in ["medium", "long"]) and (ds_name not in self.args.med_long_datasets.split()):
                    continue
                planned_configs.append((*self._build_ds_meta(ds_name, term), ds_name, term))
        only_dataset_config = str(getattr(self.args, "only_dataset_config", "") or "").strip()
        if only_dataset_config:
            wanted = {token.strip() for chunk in only_dataset_config.split(",") for token in chunk.split() if token.strip()}
            planned_configs = [
                cfg for cfg in planned_configs
                if cfg[2] in wanted or cfg[3] in wanted or cfg[2].replace("/", "_") in wanted
            ]
        print(
            f"[TaskProbeSampleForward] model={self.model_name}, stage={getattr(self.args, 'current_zoo_num', '')}, "
            f"datasets={len(planned_configs)}, cache={cache_path}"
        )
        if bool(getattr(self.args, "skip_saved", False)):
            complete = [
                cfg for cfg in planned_configs
                if self._task_probe_sample_forward_complete(cfg[2])
            ]
            if len(complete) == len(planned_configs):
                print(
                    f"[TaskProbeSampleForward][skip_saved] model={self.model_name} all planned sample-forward rows complete; skip"
                )
                return
            if complete:
                print(
                    f"[TaskProbeSampleForward][skip_saved] model={self.model_name} complete={len(complete)}/{len(planned_configs)}; "
                    "remaining datasets will run"
                )
        if bool(getattr(self.args, "dry_run", False)):
            for ds_key, ds_freq, ds_config, _dataset_name, _ds_name, term in planned_configs[:20]:
                arr = cache.get(ds_config)
                meta = meta_all.get(ds_config, {})
                entry_indices = meta.get("entry_indices", []) if isinstance(meta, dict) else []
                print(
                    f"[dry-run] model={self.model_name}, dataset={ds_config}, "
                    f"shape={getattr(arr, 'shape', None)}, sample_n={len(entry_indices)}, pred_len={self._task_probe_prediction_length(ds_key, ds_freq, term)}"
                )
            return
        error_log = str(
            getattr(self.args, "task_probe_sample_error_log", "")
            or os.path.join("results_csv", "TSRouter", "vldb", "logs", "task_probe_sample_error_log.csv")
        )
        for ds_key, ds_freq, ds_config, dataset_name, _ds_name, term in planned_configs:
            if bool(getattr(self.args, "skip_saved", False)) and self._task_probe_sample_forward_complete(ds_config):
                print(f"[TaskProbeSampleForward][skip_saved] dataset={ds_config}, model={self.model_name}")
                continue
            timing_row = {
                "timestamp_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                "stage": int(getattr(self.args, "current_zoo_num", 0) or 0),
                "dataset": ds_config,
                "model_id": self._task_probe_model_id(),
                "model_key": self.model_name,
                "model_abbr": Model_abbrev_map.get(self.model_name, self.model_name),
                "status": "success",
                "error": "",
                "step4_task_cache_path": cache_path,
                **self._task_probe_sample_config_row(),
            }
            try:
                sample_dataset = self._task_probe_sample_dataset_from_cache(ds_config, ds_key, ds_freq, term, cache, meta_all)
                print(
                    f"[TaskProbeSampleForward] dataset={ds_config}, model={self.model_name}, "
                    f"input_n={len(sample_dataset.test_data.input)}, cache_shape={sample_dataset.step4_cache_shape}"
                )
                _res, _forecasts, _model_order = self._make_forecasts(
                    dataset=sample_dataset,
                    dataset_name=dataset_name,
                    ds_config=ds_config,
                    fixed_model_order=None,
                    debug_mode=bool(getattr(self.args, "debug_mode", False)),
                )
                total_ms = float(getattr(self, "_last_forward_ms", 0.0) or 0.0)
                load_ms = float(getattr(self, "_last_model_load_ms", 0.0) or 0.0)
                predict_ms = float(getattr(self, "_last_predict_ms", max(0.0, total_ms - load_ms)) or 0.0)
                timing_row.update(
                    {
                        "selected_windows": len(sample_dataset.test_data.input),
                        "selected_entry_indices": " ".join(map(str, getattr(sample_dataset, "step4_entry_indices", []))),
                        "model_load_ms": f"{load_ms:.3f}",
                        "forward_ms": f"{predict_ms:.3f}",
                        "sample_forward_ms": f"{total_ms:.3f}",
                        "evaluate_ms": "0.000",
                        "metric_read_ms": "0.000",
                    }
                )
                error_rows = self._task_probe_sample_error_rows_from_saved_per_window(ds_config, sample_dataset, cache_path)
                print(
                    f"[TaskProbeSampleForward][scores] dataset={ds_config}, model={self.model_name}, "
                    f"rows={len(error_rows)}, source=per_window_results.csv"
                )
                self._append_task_probe_sample_rows(error_log, error_rows, TASK_PROBE_SAMPLE_ERROR_COLUMNS)
            except Exception as e:
                timing_row.update(
                    {
                        "status": "failed",
                        "error": f"{type(e).__name__}: {e}"[:1000],
                        "selected_windows": "",
                        "selected_entry_indices": "",
                        "model_load_ms": "",
                        "forward_ms": "",
                        "sample_forward_ms": "",
                        "evaluate_ms": "0.000",
                        "metric_read_ms": "0.000",
                    }
                )
                print(f"[TaskProbeSampleForward] failed dataset={ds_config}, model={self.model_name}: {timing_row['error']}")
            self._append_task_probe_sample_rows(self.csv_file_path, [timing_row], TASK_PROBE_SAMPLE_FORWARD_COLUMNS)

    # ==============================================================
           
    # ==============================================================

    def run(self):
        if self.args.run_mode == "zoo_task_sample_forward":
            self._run_task_probe_sample_forward()
            return

        self.all_data_configs = []

        if self.model_name == "Random_Select":
            fixed_model_order = list(range(self.args.current_zoo_num))                          
            random.shuffle(fixed_model_order)
        else:
            fixed_model_order = None

        if self.args.run_mode == "zoo_repr_set_forward":
            from utils.path_utils import build_repr_eval_pool_name, build_repr_set_name

            center_ds_config = f"{build_repr_set_name(self.args)}_freqH"
            self.done_models = []
            if self.args.skip_saved:
                if os.path.exists(self.csv_file_path):
                    df_res = pd.read_csv(self.csv_file_path)
                    if "model" in df_res.columns:
                        if "dataset" in df_res.columns:
                            df_res = df_res[df_res["dataset"].astype(str).eq(center_ds_config)]
                        self.done_models = df_res["model"].drop_duplicates().tolist()
                        done_abbrs = [self._model_abbrev(name) for name in self.done_models]
                        print(
                            f"Done {len(done_abbrs)} models: "
                            f"{done_abbrs}"
                        )

            center_status = self._repr_forward_result_status_at(
                csv_path=self.csv_file_path,
                per_sample_csv_path=self.per_sample_csv_file_path,
                ds_config=center_ds_config,
                require_samples=False,
            )
            center_done_with_timing = bool(center_status["complete_with_timing"])
            center_done_without_timing = bool(center_status["complete_without_timing"])
            skip_step2_cluster_forward = bool(getattr(self.args, "skip_step2_cluster_forward", False))
            center_fresh, center_fresh_reason = self._repr_center_forward_fresh_for_anchor()
            if not center_fresh:
                center_done_with_timing = False
                center_done_without_timing = False
            center_done_for_skip = self._repr_center_done_for_skip(
                center_done_with_timing=center_done_with_timing,
                center_done_without_timing=center_done_without_timing,
                skip_step2_cluster_forward=skip_step2_cluster_forward,
            )

            pool_done = True
            pool_status = None
            if bool(getattr(self.args, "enable_process_metrics", True)):
                pool_name_for_skip = build_repr_eval_pool_name(self.args)
                pool_ds_config_for_skip = f"{pool_name_for_skip}_freqH"
                pool_pkl_for_skip = os.path.join(str(SAMPLED_REPR_POOL_CACHE_ROOT), pool_name_for_skip + ".pkl")
                if os.path.exists(pool_pkl_for_skip):
                    pool_status = self._repr_forward_result_status_at(
                        csv_path=getattr(self, "pool_csv_file_path", None),
                        per_sample_csv_path=getattr(self, "pool_per_sample_csv_file_path", None),
                        ds_config=pool_ds_config_for_skip,
                        require_samples=True,
                    )
                    pool_fresh, pool_fresh_reason = self._repr_pool_forward_fresh_for_pool()
                    pool_done = bool(pool_status["complete_with_timing"] and pool_fresh)
                else:
                    pool_done = True
                    pool_fresh = True
                    pool_fresh_reason = "missing_pool_pkl_not_required"
                    print(f"TSRouter runtime message: {pool_pkl_for_skip}TSRouter runtime message: ")
            else:
                pool_fresh = True
                pool_fresh_reason = "process_metrics_disabled"

            if getattr(self.args, "skip_saved", False):
                center_timing_override = (
                    "skipped_by_skip_step2"
                    if skip_step2_cluster_forward and center_done_without_timing
                    else None
                )
                pool_status_text = (
                    self._format_repr_forward_status(pool_status)
                    if pool_status is not None
                    else "disabled_or_missing_pool_pkl"
                )
                print(
                    f"[skip_saved][repr] {self.model_name}: "
                    f"aliases={sorted(self._model_name_aliases())}, "
                    f"center=({self._format_repr_forward_status(center_status, timing_override=center_timing_override)}), "
                    f"pool=({pool_status_text}), "
                    f"skip_step2_cluster_forward={skip_step2_cluster_forward}, "
                    f"center_fresh={center_fresh}({center_fresh_reason}), "
                    f"pool_fresh={pool_fresh}({pool_fresh_reason})"
                )
            if (
                getattr(self.args, "skip_saved", False)
                and center_done_for_skip
                and pool_done
            ):
                print("✅ ", end=" ")
                return

            if (
                    self.args.run_mode == "select"
                    and bool(getattr(self.args, "GE_fast_eval", False))
                    and str(getattr(self.args, "models", "")) == "Real_Select"
                    and int(getattr(self.args, "ensemble_size", 1)) == 1
                    and int(getattr(self.args, "restrict_top_model_num", 1)) == 1
            ):
                print('preload cache...')
                self._preload_real_order_cache_for_ge_fast()
                self._preload_ge_fast_dataset_cache()

            from selector.TSRouter_Select.sampled_repr_set import ReprDatasetAdapter
            center_replayed_from_pool = False
            if not center_done_for_skip:
                if route_efficiency_mode_enabled(self.args):
                    print(
                        "[Step2:cluster-forward-skip] disabled for center timing: "
                        "route_efficiency_mode=True requires real forward_runtime_seconds"
                    )
                else:
                    center_replayed_from_pool = self._try_replay_repr_center_from_pool(fixed_model_order=fixed_model_order)
                if center_replayed_from_pool:
                    center_done_for_skip = True

            if center_done_for_skip:
                print(f"⏩ {self.model_name} center forward already complete; only checking incremental branches")
            else:
                print(f"🚀 Running {self.model_name}")
                print(f"\n==== [Step2:center] {self.model_name} ====", flush=True)
                t_repr_load0 = time.perf_counter()
                dataset = ReprDatasetAdapter(self.args, freq="H")                                                             
                repr_data_load_ms = (time.perf_counter() - t_repr_load0) * 1000.0
                memory_monitor = self._begin_gpu_memory_tracking()
                start_time = time.time()
                try:
                    res, forecasts, model_order = self._make_forecasts(
                        dataset=dataset,
                        dataset_name=dataset.name,
                        ds_config=dataset.name,
                        fixed_model_order=fixed_model_order,
                        debug_mode=self.args.debug_mode,
                    )
                except Exception:
                    self._finish_gpu_memory_tracking(memory_monitor)
                    raise

                end_time = time.time()
                memory_snapshot = self._finish_gpu_memory_tracking(memory_monitor)
                self._last_repr_data_load_ms = repr_data_load_ms

                elapsed = end_time - start_time
                memory_used = float(memory_snapshot.get("memory_use_mb", 0.0) or 0.0)
                ds_key = 'repr'                                        
                if self.args.save_pred:
                    self.save_results(res, forecasts, dataset.name, dataset.name, ds_key, elapsed, memory_used, dataset, model_order)

            if bool(getattr(self.args, "enable_process_metrics", True)):
                from utils.path_utils import build_repr_eval_pool_name

                pool_name = build_repr_eval_pool_name(self.args)
                pool_pkl = os.path.join(str(SAMPLED_REPR_POOL_CACHE_ROOT), pool_name + ".pkl")
                if not os.path.exists(pool_pkl):
                    print(f"TSRouter runtime message: {pool_pkl}TSRouter runtime message: ")
                else:
                    pool_done = (
                        self._repr_forward_result_complete_at(
                            csv_path=self.pool_csv_file_path,
                            per_sample_csv_path=self.pool_per_sample_csv_file_path,
                            ds_config=f"{pool_name}_freqH",
                            require_timing=True,
                            require_samples=True,
                        )
                        if getattr(self.args, "skip_saved", False)
                        else False
                    )
                    pool_fresh, pool_fresh_reason = self._repr_pool_forward_fresh_for_pool()
                    pool_done = bool(pool_done and pool_fresh)
                    if pool_done:
                        print(f"{self.model_name} [pool]. ✅  forward_stem={os.path.basename(self.pool_per_sample_csv_file_path).replace('_per_sample_results.csv', '')}")
                    else:
                        if getattr(self.args, "skip_saved", False) and not pool_fresh:
                            print(f"[pool][skip_saved] stale pool forward rejected: {pool_fresh_reason}")
                        old_csv = self.csv_file_path
                        old_per = self.per_sample_csv_file_path
                        old_override = getattr(self.args, "repr_set_file_stem_override", "")
                        old_save_repr_data_path = getattr(self.args, "save_repr_data_path", "")
                        self.csv_file_path = self.pool_csv_file_path
                        self.per_sample_csv_file_path = self.pool_per_sample_csv_file_path
                        self.args.repr_set_file_stem_override = pool_name
                        self.args.save_repr_data_path = str(SAMPLED_REPR_POOL_CACHE_ROOT)
                        print(f"\n==== [Step2:pool] {self.model_name} ====", flush=True)
                        print(f"[pool] input={pool_name}.pkl")
                        t_repr_load0 = time.perf_counter()
                        dataset_pool = ReprDatasetAdapter(self.args, freq="H")
                        repr_data_load_ms = (time.perf_counter() - t_repr_load0) * 1000.0
                        memory_monitor = self._begin_gpu_memory_tracking()
                        start_time = time.time()
                        try:
                            res_pool, forecasts_pool, model_order_pool = self._make_forecasts(
                                dataset=dataset_pool,
                                dataset_name=dataset_pool.name,
                                ds_config=dataset_pool.name,
                                fixed_model_order=fixed_model_order,
                                debug_mode=self.args.debug_mode,
                            )
                        except Exception:
                            self._finish_gpu_memory_tracking(memory_monitor)
                            raise
                        end_time = time.time()
                        memory_snapshot = self._finish_gpu_memory_tracking(memory_monitor)
                        self._last_repr_data_load_ms = repr_data_load_ms
                        memory_used_pool = float(memory_snapshot.get("memory_use_mb", 0.0) or 0.0)
                        if self.args.save_pred:
                            self.save_results(
                                res_pool, forecasts_pool,
                                dataset_pool.name, dataset_pool.name, "repr",
                                end_time - start_time, memory_used_pool,
                                dataset_pool, model_order_pool,
                            )
                        self.args.repr_set_file_stem_override = old_override
                        self.args.save_repr_data_path = old_save_repr_data_path
                        self.csv_file_path = old_csv
                        self.per_sample_csv_file_path = old_per

        else:
            use_ge_fast_selector_cache = self._use_ge_fast_dataset_cache_for_tsrouter()

            planned_configs = []
            for ds_name in self.args.all_datasets:
                for term in ["short", "medium", "long"]:
                    if (term in ["medium", "long"]) and (ds_name not in self.args.med_long_datasets.split()):
                        continue
                    planned_configs.append((*self._build_ds_meta(ds_name, term), ds_name, term))
            only_dataset_config = str(getattr(self.args, "only_dataset_config", "") or "").strip()
            if only_dataset_config:
                wanted = {
                    token.strip()
                    for chunk in only_dataset_config.split(",")
                    for token in chunk.split()
                    if token.strip()
                }
                planned_configs = [
                    cfg for cfg in planned_configs
                    if cfg[2] in wanted or cfg[3] in wanted or cfg[2].replace("/", "_") in wanted
                ]
                print(
                    f"[dataset-filter] only_dataset_config={only_dataset_config}, "
                    f"matched={len(planned_configs)}"
                )

            if getattr(self.args, "skip_saved", False):
                main_done_map = {
                    ds_config: self._dataset_result_complete(ds_config)
                    for _, _, ds_config, _, _, _ in planned_configs
                }
                channel_status_map = {
                    ds_config: self._per_channel_result_status(ds_config)
                    for _, _, ds_config, _, _, _ in planned_configs
                }
                channel_done_map = {k: v[0] for k, v in channel_status_map.items()}
                window_status_map = {
                    ds_config: self._per_window_result_status(ds_config)
                    for _, _, ds_config, _, _, _ in planned_configs
                }
                window_done_map = {k: v[0] for k, v in window_status_map.items()}
                process_done_map = {
                    ds_config: self._process_metrics_result_complete(ds_config)
                    for _, _, ds_config, _, _, _ in planned_configs
                }
                mix_route_done_map = {}
                mix_route_enabled_for_skip = False
                if self.args.run_mode == "select" and self.model_name == "TSRouter":
                    try:
                        from selector.TSRouter_Select.task_probe_select import (
                            task_probe_select_dataset_rank_complete,
                            task_probe_select_enabled,
                        )

                        mix_route_enabled_for_skip = task_probe_select_enabled(self.args)
                        if mix_route_enabled_for_skip:
                            mix_route_done_map = {
                                ds_config: task_probe_select_dataset_rank_complete(
                                    self.args,
                                    ds_config,
                                    selector_result_path=self.csv_file_path,
                                )
                                for _, _, ds_config, _, _, _ in planned_configs
                            }
                    except Exception as exc:
                        mix_route_enabled_for_skip = True
                        mix_route_done_map = {
                            ds_config: False
                            for _, _, ds_config, _, _, _ in planned_configs
                        }
                        print(f"[TaskProbeSelect][skip_saved] rank status check failed: {type(exc).__name__}: {exc}")
                fully_done = [
                    ds
                    for ds in main_done_map
                    if main_done_map[ds]
                    and channel_done_map.get(ds, True)
                    and window_done_map.get(ds, True)
                    and process_done_map.get(ds, True)
                    and mix_route_done_map.get(ds, True)
                ]
                skip_status = (
                    f"[skip_saved][{self.model_name}] planned={len(planned_configs)}, "
                    f"main_done={sum(main_done_map.values())}, "
                    f"channel_done={sum(channel_done_map.values())}, "
                    f"window_done={sum(window_done_map.values())}, "
                    f"process_done={sum(process_done_map.values())}, "
                )
                if mix_route_enabled_for_skip:
                    skip_status += f"mix_route_done={sum(mix_route_done_map.values())}, "
                skip_status += f"fully_skippable={len(fully_done)}"
                print(skip_status)
                if len(fully_done) == len(planned_configs):
                    print(f"⏩ [skip_saved][{self.model_name}] all planned datasets complete; skip without preload")
                    return
                missing_channel = [ds for ds in main_done_map if main_done_map[ds] and not channel_done_map.get(ds, True)]
                if missing_channel:
                    preview = ", ".join(missing_channel[:8])
                    suffix = "" if len(missing_channel) <= 8 else f"; ... +{len(missing_channel) - 8} more"
                    print(
                        f"[skip_saved][{self.model_name}] per-channel todo={len(missing_channel)}: "
                        f"{preview}{suffix}"
                    )
                missing_window = [ds for ds in main_done_map if main_done_map[ds] and not window_done_map.get(ds, True)]
                if missing_window:
                    preview = ", ".join(missing_window[:8])
                    suffix = "" if len(missing_window) <= 8 else f"; ... +{len(missing_window) - 8} more"
                    print(
                        f"[skip_saved][{self.model_name}] per-window todo={len(missing_window)}: "
                        f"{preview}{suffix}"
                    )
                missing_mix_route = [ds for ds in main_done_map if main_done_map[ds] and not mix_route_done_map.get(ds, True)]
                if missing_mix_route:
                    preview = ", ".join(missing_mix_route[:8])
                    suffix = "" if len(missing_mix_route) <= 8 else f"; ... +{len(missing_mix_route) - 8} more"
                    print(
                        f"[TaskProbeSelect][skip_saved] rank todo={len(missing_mix_route)}: "
                        f"{preview}{suffix}"
                    )

            print(f"🚀 Running {self.model_name}", )
            if self.args.run_mode == "select" and self.model_name == "TSRouter":
                self._preload_stage_rank_truth()
            if use_ge_fast_selector_cache:
                self._preload_ge_fast_dataset_cache()

            for ds_key, ds_freq, ds_config, dataset_name, ds_name, term in planned_configs:

                    self.all_data_configs.append(ds_config)

                    main_result_done = ds_config in self.done_datasets and self._dataset_result_complete(ds_config)
                    channel_result_done, channel_reason = self._per_channel_result_status(ds_config)
                    window_result_done, window_reason = self._per_window_result_status(ds_config)
                    process_metrics_done = self._process_metrics_result_complete(ds_config)
                    mix_route_result_done = True
                    if self.args.run_mode == "select" and self.model_name == "TSRouter":
                        try:
                            from selector.TSRouter_Select.task_probe_select import (
                                task_probe_select_dataset_rank_complete,
                                task_probe_select_enabled,
                            )

                            if task_probe_select_enabled(self.args):
                                mix_route_result_done = task_probe_select_dataset_rank_complete(
                                    self.args,
                                    ds_config,
                                    selector_result_path=self.csv_file_path,
                                )
                        except Exception as exc:
                            mix_route_result_done = False
                            if getattr(self.args, "skip_saved", False):
                                print(
                                    f"\n[TaskProbeSelect] Dataset: [{ds_config}] "
                                    f"rank status check failed: {type(exc).__name__}: {exc}"
                                )
                    if main_result_done and channel_result_done and window_result_done and process_metrics_done and mix_route_result_done and getattr(self.args, "skip_saved", False):
                        # print(f"{ds_config}.", end=" ✅  ")
                        continue
                    elif main_result_done and not channel_result_done and getattr(self.args, "skip_saved", False):
                        print(
                            f"\n[per-channel-metrics] Dataset: [{ds_config}] "
                        )
                    elif main_result_done and not window_result_done and getattr(self.args, "skip_saved", False):
                        print(
                            f"\n[per-window-metrics] Dataset: [{ds_config}] "
                        )
                    elif main_result_done and not process_metrics_done and getattr(self.args, "skip_saved", False):
                        print(
                            f"\n[main-metrics] Dataset: [{ds_config}] "
                        )
                    elif main_result_done and channel_result_done and window_result_done and process_metrics_done and not mix_route_result_done and getattr(self.args, "skip_saved", False):
                        print(
                            f"\n[TaskProbeSelect] Dataset: [{ds_config}] "
                            "main selector result exists; generating missing Task-probe Select rank"
                        )
                    else:
                        print(f"\n🚀 Dataset: [{ds_config}]",
                              f"Model: {self.model_name}",
                              'GPU:', os.environ.get('CUDA_VISIBLE_DEVICES', 'None'),
                              'Batch_size:', self.batch_size,
                              'num_workers:', self.args.num_workers
                              )
                    cached_search_input = None
                    dataset = None
                    if self.args.run_mode == "select" and self.model_name == "TSRouter":
                        try:
                            force_fresh_task_repr = bool(getattr(self.args, "vldb_force_fresh_task_repr", False))
                            adaptive_repr = bool(getattr(self.args, "enable_context_len_adaptive_repr", False)) or bool(getattr(self.args, "enable_pred_len_adaptive_repr", False))
                            searcher = None
                            search_args = None
                            if force_fresh_task_repr:
                                print(f"[VLDB latency] force fresh task repr; pre-cache lookup disabled: {ds_config}")
                            elif adaptive_repr and use_ge_fast_selector_cache:
                                dataset = self._build_fast_eval_dataset_stub(ds_config)
                                search_args, _, _, _ = self._adaptive_search_args(
                                    dataset=dataset,
                                    dataset_name=dataset_name,
                                    test_data_input=dataset.test_data.input,
                                )
                                searcher = self._get_searcher(search_args)
                                cached_search_input = self.get_cached_task_repr(
                                    ds_config,
                                    searcher=searcher,
                                    search_args=search_args,
                                )
                            elif not adaptive_repr:
                                cached_search_input = self.get_cached_task_repr(ds_config)
                            task_cache_path = str(getattr(self, "_task_repr_cache_path", "") or "")
                            if cached_search_input is not None:
                                sample_seconds = None
                                sample_timing_path = ""
                                if hasattr(self, "get_cached_task_sample_seconds"):
                                    if adaptive_repr and searcher is not None and search_args is not None:
                                        sample_seconds, sample_timing_path = self.get_cached_task_sample_seconds(
                                            ds_config,
                                            searcher=searcher,
                                            search_args=search_args,
                                        )
                                    else:
                                        sample_seconds, sample_timing_path = self.get_cached_task_sample_seconds(ds_config)
                                if sample_seconds is None:
                                    print(
                                        f"TSRouter runtime message: "
                                        f"{ds_config}, cache_pkl={task_cache_path}, timing_csv={sample_timing_path}"
                                    )
                                    cached_search_input = None
                                else:
                                    print(
                                        f"TSRouter runtime message: "
                                        f"{ds_config}, cache_pkl={task_cache_path}, sample={float(sample_seconds):.6f}s"
                                    )
                            elif force_fresh_task_repr:
                                print(f"TSRouter runtime message: {ds_config}")
                            elif adaptive_repr and not use_ge_fast_selector_cache:
                                print(f"[AdaptiveRepr] defer task cache lookup until dataset lengths are available: {ds_config}")
                            else:
                                print(f"TSRouter runtime message: {ds_config}, cache_pkl={task_cache_path}")
                        except FileNotFoundError:
                            raise
                        except Exception:
                            cached_search_input = None

                    can_skip_dataset_load = (
                            self.args.run_mode == "select"
                            and self.model_name == "TSRouter"
                            and cached_search_input is not None
                            and self._use_ge_fast_eval_for_select()
                    )


                    if dataset is None and can_skip_dataset_load:
                        dataset = self._build_fast_eval_dataset_stub(ds_config)
                    elif dataset is None:
                                                       
                        to_univariate = self._decide_univariate(ds_name, term)
                        dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
                    if self.args.run_mode == "select":
                        self._log_step4_task_lengths(ds_config, dataset)
                    memory_monitor = self._begin_gpu_memory_tracking()
                    start_time = time.time()

                    try:
                        res, forecasts, model_order = self._make_forecasts(
                            dataset=dataset,
                            dataset_name=dataset_name,
                            ds_config=ds_config,
                            fixed_model_order=fixed_model_order,
                            debug_mode=self.args.debug_mode,
                            cached_search_input=cached_search_input,
                        )
                    except FileNotFoundError as e:
                        self._finish_gpu_memory_tracking(memory_monitor)
                        if self.args.run_mode == "select" and self.model_name == "Real_Channel_Select":
                            print(f"⚠️ [Real_Channel_Select] skip dataset={ds_config}: {e}")
                            row_key = {"dataset": ds_config, "model": self.model_name}
                            if self._row_exists(self.csv_file_path, row_key):
                                removed = self._remove_matching_csv_rows(self.csv_file_path, row_key)
                                if removed:
                                    print(
                                        f"⚠️ [Real_Channel_Select] removed stale invalid rows: "
                                        f"path={self.csv_file_path}, dataset={ds_config}, rows={removed}"
                                    )
                            continue
                        raise
                    except Exception:
                        self._finish_gpu_memory_tracking(memory_monitor)
                        raise

                                                                         
                    end_time = time.time()
                    memory_snapshot = self._finish_gpu_memory_tracking(memory_monitor)
                    elapsed = end_time - start_time
                    memory_used = float(memory_snapshot.get("memory_use_mb", 0.0) or 0.0)

                    print(f"time cost 🧭 {elapsed:.2f}s",
                          f"memory-use {memory_used:.0f} MB", end=' ')

                    if self.args.run_mode == "select" and self.model_name == "TSRouter":
                        from selector.TSRouter_Select.task_probe_select import (
                            run_task_probe_select_for_dataset,
                            task_probe_select_enabled,
                        )

                        if task_probe_select_enabled(self.args):
                            run_task_probe_select_for_dataset(
                                router_model=self,
                                dataset=dataset,
                                ds_key=ds_key,
                                ds_freq=ds_freq,
                                ds_config=ds_config,
                                dataset_name=dataset_name,
                                ds_name=ds_name,
                                term=term,
                                model_order=model_order,
                            )

                    if self.args.save_pred:
                        self.save_results(res, forecasts, ds_config, dataset_name, ds_key, elapsed, memory_used, dataset, model_order)

                                                                         
        planned_num_ds = len(self.all_data_configs)
        if (
            self.args.run_mode == "zoo"
            and self.args.save_pred
            and os.path.exists(self.csv_file_path)
        ):
            saved_runtime = self._aggregate_saved_zoo_runtime()
            num_ds = int(saved_runtime["dataset_num"])
            total_time = saved_runtime["total_time_s"]
            average_time = saved_runtime["average_time_s"]
            total_forward_time = saved_runtime["total_forward_time_s"]
            average_forward_time = saved_runtime["average_forward_time_s"]
            total_eval_time = saved_runtime["total_eval_time_s"]
            average_eval_time = saved_runtime["average_eval_time_s"]
            total_metric_read_time = saved_runtime["total_metric_read_time_s"]
            average_metric_read_time = saved_runtime["average_metric_read_time_s"]
            average_memory = saved_runtime["average_memory_MB"]
            max_memory = saved_runtime["max_memory_MB"]
            min_runtime_batch_size = saved_runtime["min_batch_size"]
            total_batch_size_fallback_count = saved_runtime["batch_size_fallback_count"]
            min_runtime_context_length = saved_runtime["min_context_length"]
            total_context_length_fallback_count = saved_runtime["context_length_fallback_count"]

            print(
                f"\n[runtime-summary] source={self.csv_file_path}, "
                f"saved_datasets={num_ds}, planned_datasets={planned_num_ds}, "
                f"forward={float(total_forward_time):.2f}s "
                f"({int(saved_runtime['forward_dataset_num'])}/{num_ds}), "
                f"eval={float(total_eval_time):.2f}s "
                f"({int(saved_runtime['eval_dataset_num'])}/{num_ds}), "
                f"wall={float(total_time):.2f}s, "
                f"batch_fallbacks={int(total_batch_size_fallback_count)}, "
                f"context_fallbacks={int(total_context_length_fallback_count)}, "
                f"min_context={min_runtime_context_length}\n"
            )

                              
            time_save_path = TSFM_CSV_ROOT / "logs" / "runtime-TSFM.csv"
            time_save_path.parent.mkdir(parents=True, exist_ok=True)
            time_save_filename = str(time_save_path)

            if self.args.fix_context_len:
                context_tag = self.args.context_len
            else:
                context_tag = "original"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file_exists = os.path.isfile(time_save_filename)

            row = {
                "model_name": self.model_name,
                "context_length": context_tag,
                "dataset_num": num_ds,
                "planned_dataset_num": planned_num_ds,
                "forward_dataset_num": int(saved_runtime["forward_dataset_num"]),
                "eval_dataset_num": int(saved_runtime["eval_dataset_num"]),
                "total_forward_time_s": round(float(total_forward_time), 6),
                "average_forward_time_s": round(float(average_forward_time), 6),
                "total_eval_time_s": round(float(total_eval_time), 6),
                "average_eval_time_s": round(float(average_eval_time), 6),
                "total_metric_read_time_s": round(float(total_metric_read_time), 6),
                "average_metric_read_time_s": round(float(average_metric_read_time), 6),
                # Compatibility-only wall time. E2E readers must use forward timing.
                "total_time_s": round(float(total_time), 6),
                "average_time_s": round(float(average_time), 6),
                "average_memory_MB": round(float(average_memory), 0),
                "max_memory_MB": round(float(max_memory), 0),
                "cli_batch_size": int(getattr(self, "_cli_batch_size", getattr(self.args, "batch_size", self.batch_size))),
                "batch_size_source": str(getattr(self, "_initial_batch_size_source", "")),
                "requested_batch_size": int(self.batch_size),
                "runtime_batch_size": int(min_runtime_batch_size if np.isfinite(min_runtime_batch_size) else self.batch_size),
                "min_batch_size": int(min_runtime_batch_size if np.isfinite(min_runtime_batch_size) else self.batch_size),
                "batch_size_fallback_count": int(total_batch_size_fallback_count),
                "requested_context_length": context_tag,
                "runtime_context_length": int(
                    min_runtime_context_length
                    if np.isfinite(min_runtime_context_length)
                    else self.args.context_len
                ) if self.args.fix_context_len else "",
                "min_context_length": int(
                    min_runtime_context_length
                    if np.isfinite(min_runtime_context_length)
                    else self.args.context_len
                ) if self.args.fix_context_len else "",
                "context_length_fallback_count": int(total_context_length_fallback_count),
                "memory_source": f"{self.csv_file_path}:memory_use_mb" if np.isfinite(average_memory) else "",
                "runtime_source": str(self.csv_file_path),
                "timestamp": timestamp,          
            }

                  
            with file_lock(time_save_filename + ".lock"):
                file_exists = os.path.isfile(time_save_filename)
                fieldnames = list(row.keys())
                if file_exists:
                    self._ensure_csv_columns(time_save_filename, fieldnames)
                    try:
                        with open(time_save_filename, "r", newline="") as csvfile:
                            reader = csv.reader(csvfile)
                            header = next(reader, fieldnames)
                        fieldnames = list(header)
                    except Exception:
                        fieldnames = list(row.keys())
                with open(time_save_filename, "a", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()              
                    writer.writerow({col: row.get(col, "") for col in fieldnames})

    # ==============================================================
            
    # ==============================================================
    def _save_forward_repr_per_sample_metrics(self, forecasts, dataset, res=None):
        'TSRouter runtime message.'
        metric_t0 = time.perf_counter()
        save_ms = 0.0
                                    
        def _is_abnormal_metric(v: float, *, name: str, thresh: float) -> bool:
            if not np.isfinite(v):
                return True
            return abs(float(v)) > thresh
        label_list = list(dataset.test_data.label)
        input_list = list(dataset.test_data.input)                     
        if not (len(label_list) == len(forecasts) == len(input_list)):
            raise ValueError(
                f"TSRouter runtime message: {len(label_list)} forecasts={len(forecasts)} inputs={len(input_list)}"
            )

        quantile_levels = [0.1 * i for i in range(1, 10)]
        seasonality = get_seasonality(dataset.freq)                             

                                                   
        mase_all, smape_all, crps_all = [], [], []

                                                          
        global_y_sum = 0.0
        global_q_loss_sums = {a: 0.0 for a in quantile_levels}

        abn_total = 0

        for sample_idx, (fc, label, inp) in enumerate(zip(forecasts, label_list, input_list)):
                                          
            y_true = label["target"] if isinstance(label, dict) else label
            y_true = np.asarray(y_true)

                                                
            past = inp["target"] if isinstance(inp, dict) and "target" in inp else None
            if past is None:
                                                       
                past = y_true
            past = np.asarray(past)

                                                                         
            y_pred = np.asarray(fc.quantile(0.5), dtype=np.float32)

            # =========================
                                                   
            # =========================

                                                                                        
            if getattr(self.args, "run_mode", "") == "zoo_repr_set_forward":
                m = int(getattr(self.args, "mase_lag", 1))                                
            else:
                m = int(seasonality) if seasonality is not None else 1
            m = max(1, m)

            past_len = past.shape[0]
            m = min(m, max(1, past_len // 2))
            if past_len <= m:
                m = 1

            diff = past[m:] - past[:-m]
            denom = float(np.mean(np.abs(diff))) if diff.size > 0 else 0.0

                                                                 
                                       
            denom_floor = float(getattr(self.args, "mase_denom_floor", 5e-2))
            denom = max(denom, denom_floor)

            mase = float(np.mean(np.abs(y_pred - y_true)) / denom)

                                                                
            mase_all.append(float(mase))

            # =========================
                                            
            # =========================
            numerator = np.abs(y_pred - y_true)
            denominator = (np.abs(y_pred) + np.abs(y_true)) / 2.0
            smape = np.mean(numerator / (denominator + 1e-8))
            smape_all.append(float(smape))

            # =========================
                                    
            # =========================
            crps_y_floor = float(getattr(self.args, "crps_y_floor", 1e-3))
            y_abs_sum = float(np.sum(np.abs(y_true)))
            y_abs_sum = max(y_abs_sum, crps_y_floor)

            global_y_sum += float(np.sum(np.abs(y_true)))

            crps_series = 0.0
            for a in quantile_levels:
                q = np.asarray(fc.quantile(a), dtype=np.float32)
                if not np.isfinite(q).all():
                    print(f"⚠️ [NaN/Inf quantile] model={self.model_name} q={a} -> nan_to_num")
                    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

                q_loss = (a - (y_true < q)) * (y_true - q)

                crps_series += 2.0 * float(np.sum(q_loss)) / y_abs_sum
                global_q_loss_sums[a] += float(np.sum(q_loss))

            crps_series /= len(quantile_levels)
            crps_all.append(float(crps_series))

                                                  
            mase_th = getattr(self.args, "abn_mase_th", 50.0)
            smape_th = getattr(self.args, "abn_smape_th", 5.0)                   
            crps_th = getattr(self.args, "abn_crps_th", 50.0)

            abn = (
                    _is_abnormal_metric(mase, name="MASE", thresh=mase_th) or
                    _is_abnormal_metric(smape, name="sMAPE", thresh=smape_th) or
                    _is_abnormal_metric(crps_series, name="CRPS", thresh=crps_th)
            )

            if abn:
                abn_total += 1

                                                                
                action = getattr(self.args, "abn_action", "keep")  # keep / clip
                if action == "clip":
                    mase = float(np.clip(mase, 0, mase_th))
                    smape = float(np.clip(smape, 0, smape_th))
                    crps_series = float(np.clip(crps_series, 0, crps_th))
                    mase_all[-1] = mase
                    smape_all[-1] = smape
                    crps_all[-1] = crps_series

        if abn_total > 0:
            total_samples = max(1, len(label_list))
            abn_ratio = float(abn_total) / float(total_samples)
            print(
                f"⚠️ [Forward Repr Abnormal] model={self.model_name}: "
                f"{abn_total}/{total_samples} abnormal per-sample metrics detected "
                f"(ratio={abn_ratio:.2%}); details suppressed."
            )

                                                           
        avg_mase_series = float(np.mean(mase_all)) if len(mase_all) > 0 else float("nan")
        avg_smape_series = float(np.mean(smape_all))
        avg_crps_series = float(np.mean(crps_all))

                                                                    
        global_y_sum = global_y_sum + 1e-8
        mwsql_global = 0.0
        for a in quantile_levels:
            mwsql_global += 2.0 * global_q_loss_sums[a] / global_y_sum
        mwsql_global /= len(quantile_levels)

                          
        compute_ms = (time.perf_counter() - metric_t0) * 1000.0
        save_t0 = time.perf_counter()
        with file_lock(self.per_sample_csv_file_path + ".lock"):
            if not getattr(self.args, "skip_saved", False):
                self._remove_matching_csv_rows(
                    self.per_sample_csv_file_path, {"model": self.model_name}
                )
            complete = (
                self._repr_per_sample_result_complete(self.per_sample_csv_file_path)
                if getattr(self.args, "skip_saved", False)
                else False
            )
            if not complete:
                if getattr(self.args, "skip_saved", False):
                    self._remove_matching_csv_rows(
                        self.per_sample_csv_file_path, {"model": self.model_name}
                    )
                with open(self.per_sample_csv_file_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([self.model_name, "MASE"] + mase_all)
                    writer.writerow([self.model_name, "sMAPE"] + smape_all)
                    writer.writerow([self.model_name, "CRPS"] + crps_all)
            else:
                print(f"TSRouter runtime message: {self.model_name}")
        save_ms = (time.perf_counter() - save_t0) * 1000.0
        self._last_per_sample_metric_save_ms = save_ms
        self._last_per_sample_metric_ms = compute_ms

        print(f"TSRouter runtime message: {self.per_sample_csv_file_path}")
        print(
            f"TSRouter runtime message: "
            f"MASE: {avg_mase_series:.4f}, sMAPE: {avg_smape_series:.4f}, CRPS(MWSQL): {avg_crps_series:.4f}"
        )
                                                                            

                               
        if res is not None:
            try:
                mase_gluonts = float(res["MASE[0.5]"][0])
                smape_gluonts = float(res["sMAPE[0.5]"][0])
                crps_gluonts = float(res["mean_weighted_sum_quantile_loss"][0])
                print(
                    f"TSRouter runtime message: "
                    f"MASE: {mase_gluonts:.4f}, sMAPE: {smape_gluonts:.4f}, MWSQL: {crps_gluonts:.4f}"
                )
            except Exception:
                pass

    def _ensure_real_channel_rank_cache(self, ds_config: str) -> str | None:
        model_sizes = getattr(self, "Model_sizes", None)
        if not isinstance(model_sizes, dict):
            return None
        rank_metric = str(getattr(self.args, "sgl_rank_metric", "MASE"))
        model_cl_name = getattr(self, "model_cl_name", getattr(self.args, "TSFM_results_dir", "cl_GE"))
        path = real_channel_rank_cache_path(
            current_zoo_num=int(getattr(self.args, "current_zoo_num", 0)),
            zoo_total_num=int(getattr(self.args, "zoo_total_num", 0)),
            rank_metric=rank_metric,
            model_cl_name=model_cl_name,
        )
        try:
            matrix, model_ids, channels = load_per_channel_error_matrix(
                model_sizes=model_sizes,
                dataset_name=ds_config,
                model_cl_name=model_cl_name,
                rank_metric=rank_metric,
                require_complete=True,
            )
            orders = channel_orders_from_error_matrix(matrix, model_ids, channels)
            save_channel_rank_orders(path, ds_config, rank_metric, orders)
        except Exception as e:
            if str(getattr(self.args, "models", self.model_name)) != "Real_Select":
                print(f"⚠️ Real-channel rank cache unavailable for dataset={ds_config}: {e}")
            return None
        return path

    def _preload_stage_rank_truth(self) -> dict | None:
        if not bool(getattr(self.args, "enable_process_metrics", True)):
            return None
        if self.args.run_mode != "select" or self.model_name != "TSRouter":
            return None
        try:
            from selector.baselines.stage_rank_truth import ensure_stage_rank_truth

            path, df = ensure_stage_rank_truth(
                args=self.args,
                model_sizes=getattr(self, "Model_sizes", {}) or {},
                force=bool(getattr(self.args, "rank_truth_force", False)),
            )
        except Exception as e:
            print(f"[WARN] stage rank truth preload unavailable: {e}")
            self._stage_rank_truth_df = None
            self._stage_rank_truth_path = None
            return None
        self._stage_rank_truth_df = df
        self._stage_rank_truth_path = path
        log_key = (str(path), int(len(df)))
        if log_key not in BaseModel._stage_rank_truth_logged_keys:
            print(
                f"[OK] Rank truth preload: rows={len(df)}, source={path}"
            )
            BaseModel._stage_rank_truth_logged_keys.add(log_key)
        return {"path": str(path), "rows": int(len(df))}

    def _load_stage_rank_truth_orders(
        self,
        ds_config: str,
        search_args=None,
    ) -> tuple[dict[int, list[int]], list[int], str]:
        if getattr(self, "_stage_rank_truth_df", None) is None:
            self._preload_stage_rank_truth()
        rank_truth_cl = resolve_rank_truth_cl(search_args or self.args)
        from selector.baselines.stage_rank_truth import load_stage_rank_truth_orders

        channel_orders, task_order, cl_token, _path = load_stage_rank_truth_orders(
            args=self.args,
            ds_config=ds_config,
            cl_token=rank_truth_cl,
            stage_df=getattr(self, "_stage_rank_truth_df", None),
            model_sizes=getattr(self, "Model_sizes", {}) or {},
            ensure=True,
        )
        return channel_orders, task_order, cl_token

    def _build_tsrouter_route_detail_row(self) -> dict:
        details = {column: "" for column in TSROUTER_ROUTE_DETAIL_COLUMNS}
        if self.args.run_mode != "select" or self.model_name != "TSRouter":
            return details
        extra = getattr(self, "_last_selector_extra", {}) or {}

        consistency = extra.get("rank_consistency_instability")
        if consistency is None:
            consistency_by_channel = []
        else:
            consistency_by_channel = [
                round(float(value), 6)
                for value in np.asarray(consistency, dtype=float).reshape(-1).tolist()
            ]

        channel_rank = extra.get("selected_model_list_2d")
        if channel_rank is None:
            channel_model_rank = []
        else:
            rank_array = np.asarray(channel_rank, dtype=np.int64)
            channel_model_rank = rank_array.T.tolist() if rank_array.ndim == 2 else []

        details["rank_consistency_instability_by_channel"] = json.dumps(
            consistency_by_channel,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        details["channel_model_rank"] = json.dumps(
            channel_model_rank,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return details

    def _build_tsrouter_step4_provenance_row(self) -> dict:
        values = {column: "" for column in TSROUTER_STEP4_PROVENANCE_COLUMNS}
        if self.args.run_mode != "select" or self.model_name != "TSRouter":
            return values

        extra = getattr(self, "_last_selector_extra", {}) or {}
        search_args = extra.get("search_args") if isinstance(extra, dict) else None
        concrete_args = search_args or self.args
        route_timing = extra.get("step4_route_timing", {}) if isinstance(extra, dict) else {}
        context_avg = extra.get(
            "adaptive_context_len_avg",
            getattr(concrete_args, "adaptive_context_len_avg", np.nan),
        )
        values.update(
            {
                "auto_cl_mode": get_auto_cl_mode(concrete_args),
                "route_family_mode": normalize_route_family_mode(
                    extra.get(
                        "route_family_mode",
                        getattr(concrete_args, "route_family_mode", "default"),
                    )
                ),
                "adaptive_profile": str(
                    extra.get(
                        "adaptive_profile",
                        getattr(concrete_args, "adaptive_profile", "default"),
                    )
                ),
                "adaptive_context_len_avg": (
                    np.nan if context_avg is None else context_avg
                ),
                "adaptive_pred_len": extra.get(
                    "adaptive_pred_len",
                    getattr(concrete_args, "adaptive_pred_len", getattr(self.args, "test_pred_len", np.nan)),
                ),
                "resolved_eval_cl": str(
                    extra.get(
                        "resolved_eval_cl",
                        getattr(concrete_args, "resolved_eval_cl", ""),
                    )
                ),
                "rank_truth_cl": str(
                    extra.get(
                        "rank_truth_cl",
                        getattr(concrete_args, "rank_truth_cl", ""),
                    )
                ),
                "repr_input_dim": int(getattr(concrete_args, "repr_input_dim", 0) or 0),
                "repr_output_dim": int(getattr(concrete_args, "repr_output_dim", 0) or 0),
                "repr_sub_pred_len": int(getattr(concrete_args, "repr_sub_pred_len", 0) or 0),
                "repr_source_exact_length": int(
                    getattr(concrete_args, "repr_source_exact_length", 0) or 0
                ),
                "task_sample_cache_path": str(
                    extra.get(
                        "task_sample_cache_path",
                        route_timing.get("task_sample_cache_path", ""),
                    )
                    or ""
                ),
                "task_sample_cache_hit": (
                    "true"
                    if bool(
                        extra.get(
                            "task_sample_cache_hit",
                            route_timing.get("cache_hit", False),
                        )
                    )
                    else "false"
                ),
                "sample_timing_source": str(
                    extra.get(
                        "sample_timing_source",
                        route_timing.get("sample_timing_source", ""),
                    )
                    or ""
                ),
                "eval_cl_fallback_used": (
                    "true" if bool(extra.get("eval_cl_fallback_used", False)) else "false"
                ),
                "adaptive_task_term_fallback_used": (
                    "true"
                    if bool(
                        extra.get(
                            "adaptive_task_term_fallback_used",
                            getattr(concrete_args, "adaptive_task_term_fallback_used", False),
                        )
                    )
                    else "false"
                ),
            }
        )
        return values

    def _build_tsrouter_extra_metric_row(self, dataset=None, model_order=None, ds_config: str | None = None) -> dict:
        metrics = nan_metric_row()
        if not bool(getattr(self.args, "enable_process_metrics", True)):
            return metrics
        extra = getattr(self, "_last_selector_extra", {}) or {}
        adaptive_search_args = extra.get("search_args") if isinstance(extra, dict) else None
        metrics.update(load_encoder_enrichment_for_args(adaptive_search_args or self.args))

        if self.args.run_mode != "select":
            return metrics

        rank_metric = str(getattr(self.args, "sgl_rank_metric", "MASE"))
        if ds_config is None or self.model_name != "TSRouter":
            return metrics

        channel_orders: dict[int, list[int]] = {}
        task_order: list[int] = []
        rank_truth_cl = resolve_rank_truth_cl(adaptive_search_args or self.args)
        try:
            channel_orders, task_order, rank_truth_cl = self._load_stage_rank_truth_orders(
                ds_config,
                search_args=adaptive_search_args or self.args,
            )
            extra["rank_truth_cl"] = str(
                getattr(adaptive_search_args, "resolved_eval_cl", "")
                or getattr(adaptive_search_args, "rank_truth_cl", "")
                or rank_truth_cl
            )
            if adaptive_search_args is not None:
                adaptive_search_args.rank_truth_cl = extra["rank_truth_cl"]
            self._last_selector_extra = extra
        except Exception as e:
            print(f"[WARN] Rank truth unavailable for dataset={ds_config}: {e}")

        model_order_list: list[int] = []
        try:
            raw_model_order = model_order if model_order is not None else extra.get("model_order")
            if raw_model_order is not None:
                model_order_list = [int(x) for x in list(raw_model_order)]
        except Exception:
            model_order_list = []
        if task_order:
            metrics["_PROCESS_REAL_TASK_TOP3"] = [[int(x) for x in task_order[:3]]]
        if model_order_list:
            metrics["_PROCESS_PRED_TASK_TOP3"] = [model_order_list[:3]]

        if ds_config is not None:
            try:
                from selector.TSRouter_Select.task_probe_select import (
                    compute_task_probe_window_hit_metrics,
                    task_probe_select_enabled,
                )

                if task_probe_select_enabled(self.args):
                    test_window_metrics = compute_task_probe_window_hit_metrics(
                        args=self.args,
                        ds_config=ds_config,
                        model_sizes=getattr(self, "Model_sizes", {}) or {},
                        selector_extra=extra,
                        model_order=model_order,
                    )
                    metrics.update(test_window_metrics)
            except Exception as e:
                print(f"⚠️ Task-probe window metrics unavailable for dataset={ds_config}: {e}")

        cross_metrics = compute_window_channel_task_process_metrics(
            task_sample_rankings=extra.get("task_sample_rankings"),
            channel_orders=channel_orders,
            task_order=task_order,
            selected_model_list_2d=extra.get("selected_model_list_2d"),
            model_order=model_order,
        )
        metrics.update(cross_metrics)
        if not cross_metrics and not channel_orders and not task_order:
            print(f"[WARN] Rank process metrics unavailable: no channel/task truth for dataset={ds_config}, cl={rank_truth_cl}")

        if channel_orders:
            channels = sorted(channel_orders.keys())
            real_channel_top3 = [
                [int(x) for x in channel_orders[ch][:3]]
                for ch in channels
                if channel_orders[ch]
            ]
            pred_channel_top3: list[list[int]] = []
            if extra.get("selected_model_list_2d") is not None:
                pred_arr_2d = np.asarray(extra.get("selected_model_list_2d"), dtype=np.int64)
                for cpos in range(len(channels)):
                    vals = []
                    if pred_arr_2d.ndim == 2 and cpos < pred_arr_2d.shape[1]:
                        vals = [int(x) for x in pred_arr_2d[:3, cpos].tolist() if int(x) >= 0]
                    pred_channel_top3.append(vals)
            elif model_order_list:
                pred_channel_top3 = [model_order_list[:3] for _ in channels]
            if real_channel_top3:
                metrics["_PROCESS_REAL_CHANNEL_TOP3"] = real_channel_top3
            if pred_channel_top3:
                metrics["_PROCESS_PRED_CHANNEL_TOP3"] = pred_channel_top3

            single_metrics = compute_single_series_recommendation_metrics_from_orders(
                channel_orders=channel_orders,
                selected_model_list_2d=extra.get("selected_model_list_2d"),
                selected_models_per_channel=extra.get("selected_models_per_channel"),
                model_order=model_order,
            )
            metrics.update(single_metrics)
        else:
            print(f"[WARN] TCC single-channel metrics unavailable: no channel truth for dataset={ds_config}")
        return metrics

    def save_results(self, res, forecasts, ds_config, dataset_name, ds_key, elapsed, memory_used, dataset=None, model_order=None):
        if self.args.run_mode == "zoo_repr_set_forward":
            self._last_per_sample_metric_ms = np.nan
            self._last_per_sample_metric_save_ms = np.nan
            if not bool(getattr(self, "_skip_repr_per_sample_save", False)):
                self._save_forward_repr_per_sample_metrics(forecasts, dataset, res=res)

        extra_metrics = self._build_tsrouter_extra_metric_row(dataset=dataset, model_order=model_order, ds_config=ds_config)
        extra_values = [extra_metrics.get(c, np.nan) for c in TSROUTER_EXTRA_METRIC_COLUMNS]
        route_detail_columns = [column for column in TSROUTER_ROUTE_DETAIL_COLUMNS if column in self.csv_header]
        route_details = self._build_tsrouter_route_detail_row()
        route_detail_values = [route_details.get(c, "") for c in route_detail_columns]
        provenance_columns = [
            column for column in TSROUTER_STEP4_PROVENANCE_COLUMNS
            if column in self.csv_header
        ]
        provenance = self._build_tsrouter_step4_provenance_row()
        provenance_values = [provenance.get(c, "") for c in provenance_columns]
        timing_values = self._timing_row_values(elapsed)
        if self.args.run_mode == "select" and self.model_name == "TSRouter":
            route_values = [
                timing_values.get("sample_seconds", np.nan),
                timing_values.get("sample_to_route_seconds", np.nan),
                timing_values.get("route_final_seconds", np.nan),
            ]
            if any(not np.isfinite(float(value)) or float(value) < 0 for value in route_values):
                raise ValueError(
                    f"[Step4 timing] main result requires finite non-negative route timing: "
                    f"sample={route_values[0]}, sample_to_route={route_values[1]}, route_final={route_values[2]}"
                )
            if abs(float(route_values[2]) - float(route_values[0]) - float(route_values[1])) > 1e-6:
                raise ValueError(
                    f"[Step4 timing] route_final_seconds formula mismatch: "
                    f"sample={route_values[0]}, sample_to_route={route_values[1]}, route_final={route_values[2]}"
                )
        memory_snapshot = dict(getattr(self, "_last_memory_snapshot", {}) or {})
        memory_snapshot.setdefault("memory_use_mb", memory_used)
        memory_snapshot.setdefault("gpu_memory_source", "legacy_memory_argument")
        for column in [
            "memory_use_mb",
            "gpu_memory_source",
            "nvml_process_memory_mb",
            "nvml_process_peak_memory_mb",
            "torch_memory_allocated_mb",
            "torch_memory_reserved_mb",
            "torch_memory_peak_allocated_mb",
            "torch_memory_peak_reserved_mb",
        ]:
            timing_values[column] = memory_snapshot.get(column, np.nan)
        timing_row = [timing_values.get(c, np.nan) for c in TIMING_COLUMNS]
        if self.args.run_mode == "zoo_repr_set_forward":
            self._print_repr_insert_timing(ds_config, timing_values, memory_used)

        if getattr(self.args, "GE_released", False):
            row = [
                ds_config,
                self.model_name,
                res["MSE[mean]"][0],
                res["MSE[0.5]"][0],
                res["MAE[0.5]"][0],
                res["MASE[0.5]"][0],
                res["MAPE[0.5]"][0],
                res["sMAPE[0.5]"][0],
                res["MSIS"][0],
                res["RMSE[mean]"][0],
                res["NRMSE[mean]"][0],
                res["ND[0.5]"][0],
                res["mean_weighted_sum_quantile_loss"][0],
                dataset_properties_map[ds_key]["domain"],
                dataset_properties_map[ds_key]["num_variates"],
                *timing_row,
                *extra_values,
                *route_detail_values,
                *provenance_values,
            ]
        else:
            formatted_model_order = '[' + " ".join(map(str, model_order)) + ']' if model_order is not None else ""
            row = [
                ds_config,
                self.model_name,
                res["MASE[0.5]"][0],
                res["sMAPE[0.5]"][0],
                res["mean_weighted_sum_quantile_loss"][0],
                dataset_properties_map[ds_key]["domain"],
                dataset_properties_map[ds_key]["num_variates"],
                formatted_model_order,
                *timing_row,
                *extra_values,
                *route_detail_values,
                *provenance_values,
            ]

        row_key = {"dataset": ds_config, "model": self.model_name}
        with file_lock(self.csv_file_path + ".lock"):
            self._ensure_csv_columns(self.csv_file_path, self.csv_header)
            if not getattr(self.args, "skip_saved", False):
                self._remove_matching_csv_rows(self.csv_file_path, row_key)
            elif self._row_exists(self.csv_file_path, row_key) and not self._dataset_result_complete(ds_config):
                removed = self._remove_matching_csv_rows(self.csv_file_path, row_key)
                if removed:
                    print(
                        f"⚠️ removed incomplete/NaN result rows before rewriting. "
                        # f"path={self.csv_file_path}, dataset={ds_config}, model={self.model_name}, rows={removed}"
                    )
            if self._row_exists(self.csv_file_path, row_key):
                updated = self._update_matching_csv_cells(
                    self.csv_file_path,
                    row_key,
                    {
                        **{c: timing_values.get(c, np.nan) for c in TIMING_COLUMNS},
                        **{c: extra_metrics.get(c, np.nan) for c in TSROUTER_EXTRA_METRIC_COLUMNS},
                        **{c: route_details.get(c, "") for c in route_detail_columns},
                        **{c: provenance.get(c, "") for c in provenance_columns},
                    },
                )
                msg = f"TSRouter runtime message: "
                if updated:
                    msg += f" | updated process metrics cells={updated}"
                print(msg)
            else:
                with open(self.csv_file_path, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(row)

        self._print_process_metrics(extra_metrics)

        if (
            self.args.run_mode == "zoo"
            and bool(getattr(self.args, "enable_process_metrics", True))
            and dataset is not None
            and forecasts
            and not self._per_channel_result_exists(ds_config)
        ):
            try:
                per_channel_rows = compute_per_channel_metric_rows(
                    forecasts=forecasts,
                    dataset=dataset,
                    dataset_name=ds_config,
                    model_name=self.model_name,
                    global_res=res,
                )
                saved_channel, channel_msg = save_per_channel_metric_rows(
                    per_channel_results_path(self.output_dir),
                    per_channel_rows,
                    key_cols={"dataset": ds_config, "model": self.model_name},
                                          
                                                         
                                                                 
                    skip_saved=False,
                )
                if saved_channel:
                    self._invalidate_per_channel_status_cache()
                if not saved_channel:
                    print(
                        f"⚠️ per-channel metrics not saved for {self.model_name} {ds_config}: "
                        f"{channel_msg}"
                    )
            except Exception as e:
                print(f"⚠️ per-channel metrics skipped for {self.model_name} {ds_config}: {e}")

        if (
            self.args.run_mode == "zoo"
            and bool(getattr(self.args, "enable_process_metrics", True))
            and bool(getattr(self.args, "enable_per_window_metrics", False))
            and dataset is not None
            and forecasts
            and not self._per_window_result_exists(ds_config)
        ):
            try:
                per_window_rows = compute_per_window_metric_rows(
                    forecasts=forecasts,
                    dataset=dataset,
                    dataset_name=ds_config,
                    model_name=self.model_name,
                )
                saved_window, window_msg = save_per_window_metric_rows(
                    per_window_results_path(self.output_dir),
                    per_window_rows,
                    key_cols={"dataset": ds_config, "model": self.model_name},
                    skip_saved=False,
                )
                if saved_window:
                    self._invalidate_per_window_status_cache()
                if not saved_window:
                    print(
                        f"⚠️ per-window metrics not saved for {self.model_name} {ds_config}: "
                        f"{window_msg}"
                    )
            except Exception as e:
                print(f"⚠️ per-window metrics skipped for {self.model_name} {ds_config}: {e}")

        if res is not None:
            print(
                f"👉 Saved metrics:[",
                f"MASE: {res['MASE[0.5]'][0]:.2f}",
                f"sMAPE: {res['sMAPE[0.5]'][0]:.2f}",
                f"CRPS: {res['mean_weighted_sum_quantile_loss'][0]:.2f}"
                f"] to {self.csv_file_path}")
        else:
            print(f"{self.model_name} No evaluation results.")

                         
        if self.args.run_mode in {"zoo", "zoo_repr_set_forward"} and forecasts:
            artifact_output_dir = self.artifact_output_dir or self.output_dir
                                                                               
            samples_override = getattr(self, "_saved_forecast_samples_override", None)
            if samples_override is not None:
                samples = np.asarray(samples_override)
            elif hasattr(forecasts[0], "samples"):
                arrs = [fc.samples for fc in forecasts]
                samples = np.stack(arrs, axis=0)
            elif hasattr(forecasts[0], "forecast_array"):
                arrs = [fc.forecast_array for fc in forecasts]
                samples = np.stack(arrs, axis=0)
            else:
                print(f"forecasts[0] attributes: {dir(forecasts[0])}")
                raise ValueError("forecasts[0] does not have 'samples' or 'forecast_array' attribute")

            if self.args.run_mode == "zoo_repr_set_forward":
                samples_path = self._repr_forward_samples_path(dataset_name=dataset_name)
            else:
                samples_path = os.path.join(artifact_output_dir, "npy", f"{dataset_name}_samples.npy")

            np.save(samples_path, samples)

                             
            meta = {
                "model": self.model_name,
                "dataset": ds_config,
                "artifact_samples_path": samples_path,
                "performance": {
                    "runtime_seconds": elapsed,
                    "sample_seconds": timing_values.get("sample_seconds", np.nan),
                    "sample_to_route_seconds": timing_values.get("sample_to_route_seconds", np.nan),
                    "route_final_seconds": timing_values.get("route_final_seconds", np.nan),
                    "insert_runtime_seconds": timing_values.get("insert_runtime_seconds", np.nan),
                    "repr_data_load_seconds": timing_values.get("repr_data_load_seconds", np.nan),
                    "forward_runtime_seconds": timing_values.get("forward_runtime_seconds", np.nan),
                    "eval_runtime_seconds": timing_values.get("eval_runtime_seconds", np.nan),
                    "per_sample_metric_seconds": timing_values.get("per_sample_metric_seconds", np.nan),
                    "per_sample_metric_save_seconds": timing_values.get("per_sample_metric_save_seconds", np.nan),
                    "metric_read_seconds": timing_values.get("metric_read_seconds", np.nan),
                    "non_eval_runtime_seconds": timing_values.get("non_eval_runtime_seconds", np.nan),
                    "eval_skipped": timing_values.get("eval_skipped", "false"),
                    "evaluation_mode": timing_values.get("evaluation_mode", ""),
                    "memory_use_mb": timing_values.get("memory_use_mb", memory_used),
                    "gpu_memory_source": timing_values.get("gpu_memory_source", ""),
                    "nvml_process_memory_mb": timing_values.get("nvml_process_memory_mb", np.nan),
                    "nvml_process_peak_memory_mb": timing_values.get("nvml_process_peak_memory_mb", np.nan),
                    "torch_memory_allocated_mb": timing_values.get("torch_memory_allocated_mb", np.nan),
                    "torch_memory_reserved_mb": timing_values.get("torch_memory_reserved_mb", np.nan),
                    "torch_memory_peak_allocated_mb": timing_values.get("torch_memory_peak_allocated_mb", np.nan),
                    "torch_memory_peak_reserved_mb": timing_values.get("torch_memory_peak_reserved_mb", np.nan),
                    "batch_size": timing_values.get("runtime_batch_size", self.batch_size),
                    "cli_batch_size": timing_values.get("cli_batch_size", getattr(self, "_cli_batch_size", self.batch_size)),
                    "batch_size_source": timing_values.get("batch_size_source", ""),
                    "requested_batch_size": timing_values.get("requested_batch_size", self.batch_size),
                    "min_batch_size": timing_values.get("min_batch_size", self.batch_size),
                    "batch_size_fallback_count": timing_values.get("batch_size_fallback_count", 0),
                    "requested_context_length": timing_values.get("requested_context_length", getattr(self.args, "context_len", "")),
                    "runtime_context_length": timing_values.get("runtime_context_length", getattr(self.args, "context_len", "")),
                    "min_context_length": timing_values.get("min_context_length", getattr(self.args, "context_len", "")),
                    "context_length_fallback_count": timing_values.get("context_length_fallback_count", 0),
                },
                "entries": [
                    {
                        "item_id": getattr(fc, "item_id", ""),
                        "start_date": str(getattr(fc, "start_date", ""))
                    }
                    for fc in forecasts
                ]
            }
            if self.args.run_mode == "zoo_repr_set_forward":
                meta_path = self._repr_forward_meta_path(dataset_name=dataset_name)
            else:
                meta_path = os.path.join(artifact_output_dir, "meta", f"{dataset_name}_meta.json")
            with open(meta_path, "w") as fp:
                json.dump(meta, fp)
            print(f"TSRouter runtime message: {artifact_output_dir}")

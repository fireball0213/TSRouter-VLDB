# run_model_zoo.py
import argparse
import os
import re
import json
import pickle

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import random
import torch
from torch.backends import cudnn
import sys
import hashlib

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import importlib
import warnings

warnings.filterwarnings("ignore")
from utils.path_utils import (
    TSROUTER_MODEL_REPR_DIR,
    TSROUTER_REPR_FORWARD_DIR,
    TSROUTER_SAMPLED_REPR_DIR,
    TSROUTER_SELECTOR_RESULT_DIR,
    TSROUTER_TASK_REPR_DIR,
    TSROUTER_TRAINED_ENCODER_DIR,
    ensure_tsrouter_dirs,
    get_gift_eval_task_repr_cache_path,
    get_auto_cl_mode,
    make_train_bootstrap_args,
    normalize_advanced_baseline_train_scope,
    normalize_auto_cl_args,
    normalize_route_family_mode,
    normalize_repr_scale_protocol,
    normalize_encoder_variant_args,
)
from utils.project_paths import DATASET_PROPERTIES_PATH, TSFM_CSV_ROOT, resolve_checkpoint_path


def set_seed(seed):
    np.random.seed(seed=seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _rng_state_digest() -> str:
    """Return a compact digest of the current RNG state for run audits."""
    try:
        np_state = np.random.get_state()
        np_key = np_state[1].tobytes()
    except Exception:
        np_key = b""

    try:
        torch_state = torch.get_rng_state().cpu().numpy().tobytes()
    except Exception:
        torch_state = b""

    payload = np_key + b"||" + torch_state
    return hashlib.md5(payload).hexdigest()[:12]


PHASE_RANDOM_DEFAULTS = {
    "repr_data_seed": 2029,
    "repr_encoder_seed": 2025,
    "forward_seed": 2025,
    "search_seed": 2025,
}


def _phase_seed_default(args, attr_name: str) -> int:
    """Default deterministic value used when a phase override is omitted."""
    return int(PHASE_RANDOM_DEFAULTS.get(attr_name, 2025))


def _resolve_phase_seed(args, attr_name: str) -> int:
    value = getattr(args, attr_name, None)
    if value is None:
        value = _phase_seed_default(args, attr_name)
    return int(value)


def set_seed_for_phase(args, phase: str):
    'TSRouter runtime message.'
    seed_map = {
        "repr_data": _resolve_phase_seed(args, "repr_data_seed"),
        "repr_encoder": _resolve_phase_seed(args, "repr_encoder_seed"),
        "forward": _resolve_phase_seed(args, "forward_seed"),
        "search": _resolve_phase_seed(args, "search_seed"),
    }
    if phase not in seed_map:
        raise ValueError(f"Unknown deterministic phase: {phase}")
    phase_seed = seed_map[phase]
    set_seed(phase_seed)
    print(f"[DeterminismPhase] phase={phase} rng={_rng_state_digest()}")


from config.dataset_config import Med_long_Fast_datasets, Short_Fast_datasets,Med_long_datasets, Short_datasets
from config.model_zoo_config import All_sorted_model_names, Model_zoo_details
from selector.selector_config import Selector_zoo_details
from utils.check_tools import filter_models_by_key


def _normalize_autoforecast_learner(value: str) -> str:
    raw = str(value or "LSTM").strip().upper()
    if raw == "GDBT":
        raw = "GBDT"
    if raw in {"HGBDT", "HISTGBDT", "HISTGRADIENTBOOSTING"}:
        raw = "GBDT"
    if raw not in {"LSTM", "GBDT", "MLP"}:
        raise argparse.ArgumentTypeError(
            f"AutoForecast learner must be one of LSTM/GBDT/GDBT/MLP, got {value!r}"
        )
    return raw


def _normalize_advanced_baseline_train_scope(value: str) -> str:
    try:
        return normalize_advanced_baseline_train_scope(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    def str2bool(v):
        'TSRouter runtime message.'
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in ("true", "1", "yes", "y", "t"):
            return True
        if s in ("false", "0", "no", "n", "f"):
            return False
        raise argparse.ArgumentTypeError(f"TSRouter runtime message: {v}")

    parser = argparse.ArgumentParser(description='TSRouter runtime message.')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--use_model_default_batch_size",
        type=str2bool,
        default=True,
        help="zoo/model forward uses per-model initial batch sizes from model_zoo_config when available",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--gpu_memory_poll_interval_seconds",
        type=float,
        default=0.1,
        help="NVML GPU process-memory polling interval used for runtime memory_use_mb",
    )
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--source_data', type=str, default=None, help='dataset type')
    parser.add_argument('--target_data', type=str, default=None, help='dataset type')
    parser.add_argument('--root_path', type=str, default=None, help='root path of the data file')
    parser.add_argument('--data_path', type=str, default=None, help='data file')
    parser.add_argument('--target', type=str, default='OT', help='name of target column')
    parser.add_argument('--scale', type=bool, default=True, help='scale the time series with sklearn.StandardScale()')
    parser.add_argument('--output_dir', type=str, default=str(TSFM_CSV_ROOT.relative_to(TSFM_CSV_ROOT.parents[1])), help='output dir')

    # =========================
                       
    # =========================
    parser.add_argument("--run_mode", type=str, default="zoo", help='TSRouter runtime message.')

    # =========================
                         
    # =========================
    parser.add_argument("--context_len", type=int, default=512, help='TSRouter runtime message.')
    parser.add_argument("--fix_context_len", action="store_true", help='TSRouter runtime message.')
    parser.add_argument(
        "--allow_context_len_fallback",
        "--allow_cl_fallback",
        dest="allow_context_len_fallback",
        type=str2bool,
        default=False,
        help='TSRouter runtime message.',
    )
    parser.add_argument("--save_pred", default=True, help='TSRouter runtime message.')
    parser.add_argument("--skip_saved", action="store_true", help='TSRouter runtime message.')
    parser.add_argument("--clean_saved", action="store_true", help='TSRouter runtime message.')
    parser.add_argument("--debug_mode", action="store_true", help='TSRouter runtime message.')
    parser.add_argument("--quick_test", action="store_true", help='TSRouter runtime message.')
    parser.add_argument(
        "--only_dataset_config",
        type=str,
        default="",
        help="Only run matching dataset configs/names, e.g. bitbrains_rnd/5T/long or bitbrains_rnd_5T_long; comma/space separated.",
    )
    parser.add_argument("--dry-run", action="store_true", help='TSRouter runtime message.')
    parser.add_argument('--zoo_total_num', type=int, default=4, help='TSRouter runtime message.')

    # =========================
                             
    # =========================
    parser.add_argument("--select_mode", type=str, default="Recent", help='TSRouter runtime message.')
    parser.add_argument("--random_times", type=int, default=10, help='TSRouter runtime message.')
    parser.add_argument("--ensemble_size", type=int, default=1, help='TSRouter runtime message.')
    parser.add_argument("--ensemble_agg",type=str,default="median",help='TSRouter runtime message.')
    parser.add_argument("--selector_use_sample_dist",type=str2bool,default=True,help='TSRouter runtime message.',)
    parser.add_argument("--selector_samples_per_model",type=int,default=100,help='TSRouter runtime message.',)
    parser.add_argument("--selector_point_from", type=str, default="median",choices=["median", "mean"], help='TSRouter runtime message.',)
    parser.add_argument(
        "--enable_search_ensemble",
        type=str2bool,
        # default=True,
        default=False,
        help='TSRouter runtime message.',
    )

    # =========================
                          
    # =========================
    parser.add_argument("--real_world_mode", action="store_true", default=False, help='TSRouter runtime message.')
    parser.add_argument("--select_date", type=str, default="2027-01-01", help='TSRouter runtime message.')
    parser.add_argument("--current_zoo_num", type=int, default=0, help='TSRouter runtime message.')
    parser.add_argument("--only_zoo_stage", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--real_order_metric", type=str, default="MASE", help='TSRouter runtime message.')
    parser.add_argument("--sgl_rank_metric", type=str, default="MASE", choices=["MASE", "sMAPE", "CRPS"], help="TCC/TWC real channel-rank metrics use this TSFM metric; default is fixed to MASE.")

    # =========================
                     
    # =========================
    parser.add_argument("--GE_released",type=str2bool,default=False,help='TSRouter runtime message.',)
    parser.add_argument("--GE_fast_eval",type=str2bool,default=False,help='TSRouter runtime message.',)
    parser.add_argument("--route-cache-only", dest="route_cache_only", type=str2bool, default=False, help=argparse.SUPPRESS)
    parser.add_argument(
        "--fast_gluonts_eval",
        type=str2bool,
        default=True,
        help='TSRouter runtime message.',
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1024,
        help='TSRouter runtime message.',
    )
    parser.add_argument("--TSFM_results_dir",type=str,default="cl_512",help='TSRouter runtime message.',)
    parser.add_argument("--enable_process_metrics", type=str2bool, default=True, help='TSRouter runtime message.')
    parser.add_argument(
        "--process_metrics_region_rule",
        type=str,
        default="auto",
        choices=["auto", "strict", "effective"],
        help=(
            'TSRouter runtime message.'
            'TSRouter runtime message.'
            'TSRouter runtime message.'
        ),
    )
    parser.add_argument(
        "--process_metrics_fig_dir",
        type=str,
        default="results_csv/TSRouter/vldb/figures/competence_regions",
        help='TSRouter runtime message.',
    )
    parser.add_argument("--rank_truth_cls", type=str, default="", help='TSRouter runtime message.')
    parser.add_argument("--rank_truth_output_dir", type=str, default="", help='TSRouter runtime message.')
    parser.add_argument("--rank_truth_force", type=str2bool, default=False, help='TSRouter runtime message.')
    parser.add_argument("--enable_per_window_metrics", type=str2bool, default=False, help='TSRouter runtime message.')
    parser.add_argument(
        "--skip-step2-cluster-forward",
        "--skip_step2_cluster_forward",
        dest="skip_step2_cluster_forward",
        type=str2bool,
        default=False,
        help="Step2 repr forward: when cluster_nearest anchor can be reconstructed from complete pool artifacts, skip center TSFM forward and replay saved pool predictions.",
    )
    # =========================
                                               
    # =========================
    parser.add_argument(
        "--models", type=str, default="all_zoo",
        help=(
            'TSRouter runtime message.'
            'TSRouter runtime message.'
            'TSRouter runtime message.'
        ),
    )
    parser.add_argument(
        "--size_mode", type=str, default="all_size",
        help=(
            'TSRouter runtime message.'
            'TSRouter runtime message.'
            'TSRouter runtime message.'
        ),
    )


    # ==========================================================
                                                                
    # ==========================================================

                
    parser.add_argument("--repr_encoder", type=str, default="RandomMLP", help='TSRouter runtime message.')
    parser.add_argument("--repr_input_dim", type=int, default=96, help='TSRouter runtime message.')
    parser.add_argument("--repr_output_dim", type=int, default=256, help='TSRouter runtime message.')
    parser.add_argument("--repr_sub_pred_len", type=int, default=48, help='TSRouter runtime message.')
    parser.add_argument(
        "--allow_missing_repr_sources",
        type=str2bool,
        default=False,
        help=(
            "Step1: skip unavailable repr source domains instead of failing. "
            "Used by fixed long-CL v0 runs where some domains have no sufficiently long series."
        ),
    )
    parser.add_argument("--encoder_type", type=str, default=None, choices=[None, "Random", "StatsRandom", "RandomStats", "Train", "SimpleTS", "None"], help="Encoder variant; SimpleTS loads a trained SimpleTS TS2Vec checkpoint")
    parser.add_argument("--encoder_structure", type=str, default=None, choices=[None, "MLP", "Patch", "Conv", "Inception", "TCN", "Fourier", "TS2Vec", "None"], help="Random encoder base structure")
    parser.add_argument("--trained_encoder_dir", type=str, default=TSROUTER_TRAINED_ENCODER_DIR, help="Directory for Train encoder checkpoints")
    parser.add_argument("--simplets_ts2vec_checkpoint", type=str, default="", help="SimpleTS-trained TS2Vec checkpoint used by encoder_type=SimpleTS; empty auto-discovers one compatible checkpoint")
    parser.add_argument("--simplets_ts2vec_source_repr_encoder", type=str, default="StatsRandomFourier", help="Step1 source encoder identity recorded by the SimpleTS checkpoint")
    parser.add_argument("--train_bootstrap_encoder_type", type=str, default="Random", choices=["Random", "StatsRandom", "RandomStats"], help="Non-train encoder_type used to bootstrap Step1/2 for Train")
    parser.add_argument("--train_rank_metric", type=str, default="MASE", choices=["MASE", "sMAPE", "CRPS", "M", "S", "C"], help="Metric used to build rank labels for Train encoder")
    parser.add_argument("--train_encoder_epochs", type=int, default=30, help="Train encoder epochs")
    parser.add_argument("--train_encoder_batch_size", type=int, default=256, help="Train encoder batch size")
    parser.add_argument("--train_encoder_lr", type=float, default=1e-3, help="Train encoder learning rate")
    parser.add_argument("--train_encoder_weight_decay", type=float, default=1e-4, help="Train encoder weight decay")
    parser.add_argument("--train_encoder_temperature", type=float, default=0.1, help="Train encoder contrastive temperature")
    parser.add_argument("--train_encoder_val_ratio", type=float, default=0.2, help="Train encoder validation ratio")
    parser.add_argument("--train_top3_weight", type=float, default=0.5, help="Positive-pair weight for rank Top3 overlap")
    parser.add_argument("--train_encoder_early_stop_patience", type=int, default=10, help="Stop Train encoder after this many non-improving validation epochs; 0 disables")
    parser.add_argument("--train_encoder_early_stop_min_delta", type=float, default=0.001, help="Minimum validation PWW improvement for Train encoder early stopping")
    parser.add_argument("--train_encoder_loss", type=str, default="supcon", choices=["supcon"], help="Train encoder loss")
    parser.add_argument(
        "--random_stats_fusion",
        type=str,
        default=None,
        choices=[None, "none", "early", "late"],
        help='TSRouter runtime message.',
    )
    parser.add_argument("--repr_data_seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--repr_encoder_seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--forward_seed", type=int, default=None, help=argparse.SUPPRESS)
                      
    # parser.add_argument("--encoder_model_path", type=str, default="checkpoints/encoders/SimMTM_36to128.pth")
    # parser.add_argument("--scaler_path", type=str, default="checkpoints/encoders/scaler_SimMTM_36to128.pkl")


                    
    parser.add_argument("--save_repr_selection", action="store_true", default=False, help='TSRouter runtime message.')
    parser.add_argument("--repr_v", type=int, default=4, help='TSRouter runtime message.')
    parser.add_argument("--repr_v5_nearest_k", type=int, default=10, help='TSRouter runtime message.')
    parser.add_argument("--repr_v5_distance_power", type=float, default=1.0, help='TSRouter runtime message.')
    parser.add_argument("--rank_decay_coef", type=float, default=1.0, help='TSRouter runtime message.')
    parser.add_argument("--save_repr_data_path", type=str, default=TSROUTER_SAMPLED_REPR_DIR, help='TSRouter runtime message.')
    parser.add_argument("--repr_forward_dir", type=str, default=TSROUTER_REPR_FORWARD_DIR, help='TSRouter runtime message.')
    parser.add_argument("--gift_eval_task_repr_dir", type=str, default=TSROUTER_TASK_REPR_DIR, help='TSRouter runtime message.')
    parser.add_argument(
        "--zoo_repr_set",
        type=str,
        nargs="+",
        default=["c", "l", "e"],
        help='TSRouter runtime message.',
    )
    parser.add_argument("--sample_mode", type=str, default="cluster_nearest", help='TSRouter runtime message.')
    parser.add_argument("--repr_size", type=int, default=3000, help='TSRouter runtime message.')
    parser.add_argument(
        "--repr_sample_qc_mode",
        type=str,
        default="strict",
        choices=["strict", "off"],
        help="Step1 repr sampling quality control: strict keeps existing filters; off disables window/std/MASE-denom filters.",
    )
                                                                                                 
                                                                                             

                                             
                                            

                              
    parser.add_argument("--save_model_zoo_repr", action="store_true", default=False, help='TSRouter runtime message.')
    parser.add_argument("--base_metrics", type=str, default="S", help='TSRouter runtime message.')
    parser.add_argument("--save_model_repr_path", type=str, default=TSROUTER_MODEL_REPR_DIR, help='TSRouter runtime message.')
    parser.add_argument('--model_repr_mode', type=str, default="all", help='TSRouter runtime message.')
    parser.add_argument("--subset_top_k", type=int, default=0, help='TSRouter runtime message.')
    parser.add_argument('--subset_perf_scale', type=float, default=1.0, help='TSRouter runtime message.')
    parser.add_argument(
        "--autoforecast_learner",
        "--autoforecast-learner",
        type=_normalize_autoforecast_learner,
        default="LSTM",
        help="AutoForecast v7 meta-regressor architecture: LSTM / GBDT(GDBT) / MLP.",
    )
    parser.add_argument("--autoforecast_hidden_dim", type=int, default=64, help="AutoForecast v7 LSTM/MLP hidden dimension")
    parser.add_argument("--autoforecast_train_epochs", type=int, default=120, help="AutoForecast v7 LSTM/MLP training epochs")
    parser.add_argument("--autoforecast_learning_rate", type=float, default=0.001, help="AutoForecast v7 LSTM/MLP learning rate")
    parser.add_argument("--autoforecast_batch_size", type=int, default=256, help="AutoForecast v7 LSTM/MLP training batch size")
    parser.add_argument(
        "--advanced_baseline_train_scope",
        "--advanced-baseline-train-scope",
        type=_normalize_advanced_baseline_train_scope,
        default="center",
        choices=["center", "full_pool"],
        help=(
            "Step3 training sample scope for AutoForecast/AutoXPCR/SimpleTS: "
            "center uses Step1 cluster-center anchors; full_pool uses the complete candidate pool."
        ),
    )


                                
    parser.add_argument("--search_context_len", type=int, default=36, help='TSRouter runtime message.')
    parser.add_argument("--sample_repr_num", type=int, default=20, help='TSRouter runtime message.')
    parser.add_argument("--task_sample_version", type=int, default=2, choices=[1, 2],
                        help='TSRouter runtime message.')
    parser.add_argument(
        "--task_sample_strategy",
        type=str,
        default="latest_random",
        choices=["latest_random", "time_coverage"],
        help="Step4 task sampling: latest_random keeps the old random-entry latest-window protocol; time_coverage spreads samples over early-to-late windows.",
    )
    parser.add_argument(
        "--task_window_sample_strategy",
        type=str,
        default="legacy",
        choices=["legacy", "even", "random", "first", "last"],
        help="Step4 task window index sampling: legacy keeps old per-sample random permutation; otherwise choose entries by this strategy.",
    )
    parser.add_argument(
        "--repr_anchor_window_sample_strategy",
        type=str,
        default=None,
        choices=["legacy", "even", "random", "first", "last"],
        help=(
            "Step1/2 repr-anchor window sampling. Defaults to task_window_sample_strategy for "
            "backward compatibility; pass this explicitly when Step1 anchors and Step4 task "
            "windows use different strategies."
        ),
    )
    parser.add_argument(
        "--sample_repr_ratio",
        type=float,
        default=0.0,
        help="Step4 task sample ratio. If >0, effective samples=max(sample_repr_num, ceil(valid_windows*ratio)).",
    )
    parser.add_argument(
        "--task_rank_top3_instability_threshold",
        type=float,
        default=-1.0,
        help="Rank-consistency fallback threshold. -1 disables fallback and omits the filename tag; 0 forces all channels to use the model-weight prior and writes _fb0; values >0 fallback only channels whose Rank1 count score exceeds the threshold.",
    )
    parser.add_argument(
        "--task_channel_fuse_limit",
        type=str,
        default="all",
        help="Step4 channel-rank fusion channel cap. all keeps current behavior; an integer uses only the first N channels in task-level fusion.",
    )
    parser.add_argument(
        "--route_family_mode",
        "--route-family-mode",
        "--route-famlily-mode",
        dest="route_family_mode",
        type=normalize_route_family_mode,
        default="default",
        choices=["default", "bigger_size", "smaller_size"],
        help=(
            "Step4 family routing: default keeps every size independent; bigger_size/smaller_size "
            "merge every family signal into its largest/smallest available stage model."
        ),
    )
    parser.add_argument(
        "--repr_anchor_protocol",
        type=str,
        default="window",
        choices=["window", "task_sample"],
        help="Step1 repr sampling protocol: window keeps independent windows; task_sample samples groups using the Step4 task_sample_strategy.",
    )
    parser.add_argument("--search_seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--repeat_id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tsrouter_selector_result_dir", type=str, default=TSROUTER_SELECTOR_RESULT_DIR, help='TSRouter runtime message.')
    parser.add_argument(
        "--repr_scale_protocol",
        type=normalize_repr_scale_protocol,
        default="raw",
        choices=["raw", "standard"],
        help='TSRouter runtime message.',
    )
    parser.add_argument("--restrict_top_model_num", type=int, default=1, help='TSRouter runtime message.')
    parser.add_argument("--minus_ratio", type=float, default=0, help='TSRouter runtime message.')
    parser.add_argument(
        "--repr_weight_ratio",
        type=float,
        default=1.0,
        help='TSRouter runtime message.',
    )
    parser.add_argument(
        "--route_efficiency_mode",
        "--enable_route_efficiency_mode",
        type=str2bool,
        default=False,
        help="TSRouter Step3: build route prior weights from saved Step2 runtime instead of metric-rank weights; VLDB summaries display this mode as TSRouter-fast.",
    )
    parser.add_argument("--auto_cl_mode", type=str, default="v0", choices=["v0", "v1", "v2", "v3"], help="TSRouter auto context-length profile mode. v0 keeps legacy fixed-cl naming; v1/v2/v3 use fixed auto-cl profile configs.")
    parser.add_argument("--enable_context_len_adaptive_repr", type=str2bool, default=False, help="Deprecated alias for --auto_cl_mode v1")
    parser.add_argument("--enable_pred_len_adaptive_repr", type=str2bool, default=False, help="Step4: choose short/long model repr files by downstream prediction length")
    parser.add_argument("--context_len_adaptive_threshold", type=float, default=256.0, help="Step4: avg context_len < threshold uses the short repr_input_dim")
    parser.add_argument("--pred_len_adaptive_threshold", type=int, default=96, help="Step4: pred_len > threshold uses the long repr_sub_pred_len")
    parser.add_argument("--short_repr_input_dim", type=int, default=96, help="adaptive short profile repr_input_dim")
    parser.add_argument("--middle_repr_input_dim", type=int, default=512, help="adaptive middle profile repr_input_dim")
    parser.add_argument("--long_repr_input_dim", type=int, default=2048, help="adaptive long profile repr_input_dim")
    parser.add_argument("--short_repr_output_dim", type=int, default=128, help="auto_cl v1 short profile repr_output_dim")
    parser.add_argument("--middle_repr_output_dim", type=int, default=256, help="auto_cl v1 middle profile repr_output_dim")
    parser.add_argument("--long_repr_output_dim", type=int, default=512, help="auto_cl v1 long profile repr_output_dim")
    parser.add_argument("--short_repr_sub_pred_len", type=int, default=48, help="adaptive short profile repr_sub_pred_len")
    parser.add_argument("--middle_repr_sub_pred_len", type=int, default=480, help="adaptive middle profile repr_sub_pred_len")
    parser.add_argument("--long_repr_sub_pred_len", type=int, default=720, help="adaptive long profile repr_sub_pred_len")
    parser.add_argument("--long_repr_source_len", type=int, default=3000, help="Step1 auto_cl exact long source length")
    parser.add_argument("--middle_repr_source_len", type=int, default=992, help="Step1 auto_cl exact middle source length")
    parser.add_argument("--short_repr_source_len", type=int, default=144, help="Step1 auto_cl exact short source length")
    parser.add_argument('--err_rate', type=float, default=0.0, help='TSRouter runtime message.')
    parser.add_argument('--repr_distance_metric', type=str,default="cos",help='TSRouter runtime message.')
    parser.add_argument('--model_repr_agg', type=str,default="min",help='TSRouter runtime message.')
    parser.add_argument(
        "--mix-route",
        "--mix_route",
        dest="mix_route",
        nargs="?",
        const=True,
        default=False,
        type=str2bool,
        help="TSRouter Step4: after TSRouter rank, run Task-probe Select on cached task-sample windows.",
    )
    parser.add_argument(
        "--mix-route-model-num",
        "--mix_route_model_num",
        "--task_probe_select_model_num",
        dest="mix_route_model_num",
        type=int,
        default=0,
        help="Task-probe Select candidate count. 0 means all models in the current stage zoo.",
    )
    parser.add_argument(
        "--task-probe-select-output-dir",
        "--task_probe_select_output_dir",
        default="",
        help="Task-probe Select output root. Default: results_csv/baselines/selectors/Task_probe_Select.",
    )

    # =========================
                                               
    # =========================
    parser.add_argument("--vldb_route_latency_log", type=str, default="", help="VLDB route latency sidecar CSV path")
    parser.add_argument("--task_probe_sample_forward_log", type=str, default="", help="Task-Probe sample-window TSFM forward timing sidecar CSV path")
    parser.add_argument("--task_probe_sample_error_log", type=str, default="", help="Task-Probe sample-window error sidecar CSV path")
    parser.add_argument("--vldb_route_id", type=str, default="", help="VLDB route run id for latency sidecar")
    parser.add_argument("--vldb_route_stage", type=int, default=-1, help="VLDB stage for latency sidecar")
    parser.add_argument("--vldb_route_profile_id", type=str, default="", help="VLDB profile id for latency sidecar")
    parser.add_argument(
        "--vldb_skip_evaluate",
        type=str2bool,
        default=False,
        help="VLDB latency mode: keep forecast construction but replace evaluate_forecasts with saved metrics when available; records forward/eval timings separately",
    )
    parser.add_argument(
        "--vldb_fast_sample",
        type=str2bool,
        default=False,
        help="VLDB latency mode: reuse saved Step4 task samples instead of rebuilding online samples",
    )
    parser.add_argument(
        "--vldb_fast_forward",
        type=str2bool,
        default=True,
        help="VLDB latency mode: use saved TSFM predictions instead of charging selected-model forward runtime",
    )
    parser.add_argument(
        "--vldb_force_fresh_task_repr",
        type=str2bool,
        default=False,
        help="VLDB latency mode: ignore cached Step4 task repr and rebuild online samples",
    )

    # Deprecated compatibility parameter.
    parser.add_argument('--analysis_keep_clusters', type=int, default=0, help='deprecated no-op; Step1 no longer writes partial analysis pool files')

    # =========================
                                
    # =========================
    parser.add_argument("--dec_mode", action="store_true", default=False, help='TSRouter runtime message.')
    parser.add_argument("--decomp_method", type=str, default="MA2", help='TSRouter runtime message.')
    parser.add_argument("--dec_save_mode", type=str, default="dec", help='TSRouter runtime message.')
    parser.add_argument("--kernel_size", type=int, default=25, help="moving average window size")
    parser.add_argument("--period", type=int, default=96, help="STL seasonal period")
    parser.add_argument("--resid", type=str, default="None", help='TSRouter runtime message.')
    parser.add_argument("--trend_dec_times", type=int, default=1, help='TSRouter runtime message.')

    # =========================
               
    # =========================
    parser.add_argument("--noise_scale", type=float, default=0.0, help='TSRouter runtime message.', )
    parser.add_argument("--noise_mode", type=str, default="relative", help="relative / absolute")
    parser.add_argument("--noise_seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--strict_phase_seed", action="store_true", default=True, help=argparse.SUPPRESS)

    return parser

def prepare_args(args):
    'TSRouter runtime message.'
    # Use a fixed bootstrap value before phase-specific deterministic resets.
    set_seed(0)
    if getattr(args, "repeat_id", None) is not None and args.search_seed is None:
        args.search_seed = int(args.repeat_id)
    if args.repr_data_seed is None:
        args.repr_data_seed = _phase_seed_default(args, "repr_data_seed")
    if args.repr_encoder_seed is None:
        args.repr_encoder_seed = _phase_seed_default(args, "repr_encoder_seed")
    if args.forward_seed is None:
        args.forward_seed = _phase_seed_default(args, "forward_seed")
    if args.search_seed is None:
        args.search_seed = _phase_seed_default(args, "search_seed")
    args.repr_scale_protocol = normalize_repr_scale_protocol(getattr(args, "repr_scale_protocol", "raw"))
    args.route_family_mode = normalize_route_family_mode(
        getattr(args, "route_family_mode", "default")
    )
    args.autoforecast_learner = _normalize_autoforecast_learner(
        getattr(args, "autoforecast_learner", "LSTM")
    )
    args.advanced_baseline_train_scope = normalize_advanced_baseline_train_scope(
        getattr(args, "advanced_baseline_train_scope", "center")
    )
    normalize_auto_cl_args(args)
    normalize_encoder_variant_args(args)
    if str(getattr(args, "repr_encoder", "")) == "SimpleTS2Vec":
        from encoder.simplets_checkpoint import resolve_simplets_ts2vec_checkpoint

        path, _ = resolve_simplets_ts2vec_checkpoint(args)
        print(
            "[SimpleTS2Vec] bound main-method Step1-4 encoder: "
            f"path={path}, source_repr={args.simplets_ts2vec_source_repr_set_name}, "
            f"fingerprint={args.simplets_ts2vec_checkpoint_fingerprint[:12]}"
        )
    ensure_tsrouter_dirs(args)

    if args.quick_test:
        print('TSRouter runtime message.')
        args.all_datasets = sorted(set(Short_Fast_datasets.split() + Med_long_Fast_datasets.split()))
        args.med_long_datasets = Med_long_Fast_datasets
    else:
        args.all_datasets = sorted(set(Short_datasets.split() + Med_long_datasets.split()))
        args.med_long_datasets = Med_long_datasets

                        
    args.zoo_total_num = sum(len(sizes) for sizes in Model_zoo_details.values())
    return args


def _dry_run_task_sample_forward(args) -> None:
    cache_path = get_gift_eval_task_repr_cache_path(
        args,
        search_context_len=int(getattr(args, "repr_input_dim", getattr(args, "context_len", 512))),
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
    planned = []
    med_long = set(str(args.med_long_datasets).split())
    for ds_name in args.all_datasets:
        for term in ["short", "medium", "long"]:
            if term in {"medium", "long"} and ds_name not in med_long:
                continue
            if "/" in ds_name:
                ds_key_raw, ds_freq = ds_name.split("/")
                ds_key = {
                    "saugeenday": "saugeen",
                    "temperature_rain_with_missing": "temperature_rain",
                    "kdd_cup_2018_with_missing": "kdd_cup_2018",
                    "car_parts_with_missing": "car_parts",
                }.get(ds_key_raw.lower(), ds_key_raw.lower())
            else:
                ds_key = {
                    "saugeenday": "saugeen",
                    "temperature_rain_with_missing": "temperature_rain",
                    "kdd_cup_2018_with_missing": "kdd_cup_2018",
                    "car_parts_with_missing": "car_parts",
                }.get(ds_name.lower(), ds_name.lower())
                with open(DATASET_PROPERTIES_PATH, encoding="utf-8") as f:
                    dataset_properties = json.load(f)
                ds_freq = dataset_properties[ds_key]["frequency"]
            ds_config = f"{ds_key}/{ds_freq}/{term}"
            dataset_name = f"{ds_key}_{ds_freq}_{term}"
            planned.append((ds_config, dataset_name))
    only_dataset_config = str(getattr(args, "only_dataset_config", "") or "").strip()
    if only_dataset_config:
        wanted = {token.strip() for chunk in only_dataset_config.split(",") for token in chunk.split() if token.strip()}
        planned = [cfg for cfg in planned if cfg[0] in wanted or cfg[1] in wanted or cfg[0].replace("/", "_") in wanted]
    stage = int(getattr(args, "current_zoo_num", 0) or 0)
    models = All_sorted_model_names[:stage] if stage > 0 else All_sorted_model_names
    print(f"[TaskProbeSampleForward][dry-run] stage={stage}, models={len(models)}, datasets={len(planned)}, cache={cache_path}")
    for ds_config, _dataset_name in planned[:20]:
        arr = cache.get(ds_config)
        item_meta = meta.get(ds_config, {}) if isinstance(meta, dict) else {}
        entry_indices = item_meta.get("entry_indices", []) if isinstance(item_meta, dict) else []
        print(
            f"[dry-run] dataset={ds_config}, shape={getattr(arr, 'shape', None)}, "
            f"sample_n={len(entry_indices)}, meta={'yes' if item_meta else 'no'}"
        )

def main():

    parser = build_parser(add_help=True)
    args = parser.parse_args()
    args = prepare_args(args)
    if (
        args.run_mode == "select"
        and str(getattr(args, "models", "")) == "TSRouter"
        and get_auto_cl_mode(args) != "v0"
        and not bool(getattr(args, "save_model_zoo_repr", False))
    ):
        from selector.TSRouter_Select.auto_cl_step4 import preflight_auto_cl_step4

        preflight_auto_cl_step4(args)
    if args.run_mode == "zoo_task_sample_forward" and bool(getattr(args, "dry_run", False)):
        if args.strict_phase_seed:
            set_seed_for_phase(args, "forward")
        _dry_run_task_sample_forward(args)
        return

    # ==========================================================
                                 
    # ==========================================================
    if args.save_repr_selection:
        if str(getattr(args, "encoder_type", "")).lower() == "train":
            print("[TrainEncoder] Step1 bootstrap uses non-train encoder; Train artifacts are rebuilt after Step2 in Step3.")
            args = make_train_bootstrap_args(args)
        if args.strict_phase_seed:
            set_seed_for_phase(args, "repr_data")
        print('TSRouter runtime message.')
        from selector.TSRouter_Select.sampled_repr_set import save_sampled_repr_set

        save_sampled_repr_set(args)
        print('TSRouter runtime message.')
        return


    # ==========================================================
                      
    # ==========================================================
    if args.run_mode in {"zoo", "zoo_repr_set_forward", "zoo_task_sample_forward"}:
        if str(getattr(args, "encoder_type", "")).lower() == "train":
            print("[TrainEncoder] Step2 bootstrap forward uses non-train encoder; Train center CSV is derived from pool forward after training.")
            args = make_train_bootstrap_args(args)
        if args.strict_phase_seed:
            set_seed_for_phase(args, "forward")
        else:
            # Non-strict compatibility path.
            set_seed(int(args.forward_seed))
            print(f"[DeterminismPhase] phase=forward(non-strict) rng={_rng_state_digest()}")
        if args.models == "all_zoo":
            families = list(Model_zoo_details.keys())
        else:
            requested = [m.strip() for m in re.split(r"[,\s]+", str(args.models)) if m.strip()]
            families = [m for m in requested if m in Model_zoo_details]
            missing = set(requested) - set(families)
            if missing:
                print(f"TSRouter runtime message: {missing} \n ")

        print('TSRouter runtime message.', families)
        for family in families:
            sizes_dict = Model_zoo_details[family]

            if not sizes_dict:
                print(f"TSRouter runtime message: {family}TSRouter runtime message: ")
                continue
            if args.size_mode == "all_size":
                sizes = list(sizes_dict.keys())
            elif args.size_mode == "first_size":
                sizes = [next(iter(sizes_dict.keys()))]
            else:
                all_sizes = [s.strip() for s in args.size_mode.split(",")]
                sizes = [s for s in all_sizes if s in sizes_dict]
                if len(all_sizes) - len(sizes) > 0:
                    raise ValueError(f"TSRouter runtime message: {args.size_mode}TSRouter runtime message: {family}TSRouter runtime message: ")
            print(f"TSRouter runtime message: {family}TSRouter runtime message: {sizes}")

            if not sizes:
                sizes = [None]

            for size in sizes:
                variant_cfg = sizes_dict[size]
                model_key = f"{family}_{size}"
                if args.run_mode == "zoo_task_sample_forward":
                    stage = int(getattr(args, "current_zoo_num", 0) or 0)
                    if stage > 0 and model_key not in set(All_sorted_model_names[:stage]):
                        continue

                                 
                ModelModule = importlib.import_module(variant_cfg["model_module"])
                ModelClass = getattr(ModelModule, variant_cfg["model_class"])

                model = ModelClass(
                    args,
                    module_name=variant_cfg["module_name"],
                    model_name=model_key,
                    model_local_path=str(resolve_checkpoint_path(variant_cfg["model_local_path"])),
                )

                if args.debug_mode:
                    model.run()
                else:
                    try:
                        model.run()
                    except Exception as e:
                        if args.debug_mode:
                            print(f"TSRouter runtime message: {e}")
                        else:
                            raise e
                print(f"TSRouter runtime message: {family} [{size}] ===\n")


    elif args.run_mode == "select":
        args.zoo_total_num = sum(len(sizes) for sizes in Model_zoo_details.values())

        if args.real_world_mode:
                               
            all_models = [
                details
                for family in Model_zoo_details.values()
                for details in family.values()
            ]
                        
            sorted_models = sorted(all_models, key=lambda x: x["release_date"])
            all_zoo_release_list = [model["release_date"] for model in sorted_models]

            assert args.ensemble_size + 1 <= len(all_zoo_release_list), "ensemble_size must < current_zoo_num)"
            # Growing-zoo Step3/Step4 both start from stage3 by contract.
            start_zoo_num = max(3, int(args.ensemble_size) + 1)
            only_zoo_stage = int(getattr(args, "only_zoo_stage", 0) or 0)
            if only_zoo_stage:
                if only_zoo_stage < start_zoo_num or only_zoo_stage > len(all_zoo_release_list):
                    raise ValueError(
                        f"only_zoo_stage must be in [{start_zoo_num}, {len(all_zoo_release_list)}], "
                        f"got {only_zoo_stage}"
                    )
                zoo_stages = [only_zoo_stage]
            else:
                zoo_stages = range(start_zoo_num, len(all_zoo_release_list) + 1)
            for current_zoo_num in zoo_stages:
            # for current_zoo_num in range(8, len(all_zoo_release_list) + 1):
                current_zoo_release_list = all_zoo_release_list[args.ensemble_size:current_zoo_num]
                args.select_date = current_zoo_release_list[-1]
                print(f"TSRouter runtime message: {args.select_date}TSRouter runtime message: "
                      f"{current_zoo_num} / {len(all_zoo_release_list)}, ensemble_size={args.ensemble_size}")
                run_select(args)
        else:
                                                           
            run_select(args)

    else:
        raise ValueError('TSRouter runtime message.')



def run_select(args):

                                                         
    Model_zoo_current, sorted_filtered_models = filter_models_by_key(Model_zoo_details, args.select_date, select_key="release_date")
    args.current_zoo_num = sum(len(sizes) for sizes in Model_zoo_current.values())          
                                                                                          

    # ==========================================================
                           
    # ==========================================================
    if args.save_model_zoo_repr:
        if args.strict_phase_seed:
            set_seed_for_phase(args, "repr_encoder")
        from selector.TSRouter_Select.model_zoo_repr import get_model_zoo_repr
        current_zoo_abbr_order_list = [model["abbreviation"] for model in sorted_filtered_models]
        print("current_zoo_order:",current_zoo_abbr_order_list)
        is_simplets_v6 = str(getattr(args, "repr_v", "") or "")[:1] == "6"
        if str(getattr(args, "encoder_type", "")).lower() == "train" and not is_simplets_v6:
            from encoder.trained_encoder import ensure_trained_encoder_artifacts
            ensure_trained_encoder_artifacts(args, current_zoo_abbr_order_list)
        elif str(getattr(args, "encoder_type", "")).lower() == "train" and is_simplets_v6:
            print(
                "[SimpleTS v6] skip generic Train encoder bootstrap: "
                "--encoder_type/--encoder_structure identify the existing Step1/2 source; "
                "the Step3/4 TS2Vec is trained and loaded internally."
            )
        get_model_zoo_repr(args, current_zoo_abbr_order_list)
        print('TSRouter runtime message.')
        return

    select_name = args.models

    cfg = Selector_zoo_details.get(select_name, None)
    if cfg is None:
        raise ValueError(f"TSRouter runtime message: {select_name}TSRouter runtime message: ")

               
    module = importlib.import_module(cfg["model_module"])
    SelectorClass = getattr(module, cfg["model_class"])
    if args.strict_phase_seed and select_name == "TSRouter":
        set_seed_for_phase(args, "search")
    model = SelectorClass(args, model_name=select_name, Model_zoo_current=Model_zoo_current, )

    model.run()
    
if __name__ == "__main__":
    main()


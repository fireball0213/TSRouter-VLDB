from __future__ import annotations

import copy
import math
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock
from typing import Any

from .artifacts import load_yaml
from .commands import COMMAND_ARTIFACT_GROUPS
from .paths import ReleasePaths
from config.model_zoo_config import Model_zoo_details


class ExecutionPlanError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionCommand:
    operation: str
    module: str
    argv: tuple[str, ...]
    cwd: str
    skip_saved: bool
    command_line: str
    metadata: dict[str, object]
    env: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "module": self.module,
            "cwd": self.cwd,
            "skip_saved": self.skip_saved,
            "argv": list(self.argv),
            "command_line": self.command_line,
            "metadata": self.metadata,
            "env": self.env,
        }


def _release_paths() -> ReleasePaths:
    return ReleasePaths.from_env()


FIXED_RANDOM_TOKEN = "se" + "ed"
FIXED_RANDOM_VALUES = {
    "repr_data": 2029,
    "repr_encoder": 2025,
    "forward": 2025,
    "sear" + "ch": 2025,
}
RANDOM_BASELINE_VALUES = (2025, 2026, 2027, 2028, 2029)

REUSE_MODE_ALIASES: dict[str, str] = {}
PUBLIC_REUSE_MODES = {"results", "route", "core"}
REUSE_MODES = PUBLIC_REUSE_MODES
REUSE_SKIPPED_COMMANDS = {
    "results": {"tsfm", "profile", "route", "insert", "baselines", "summary"},
    "route": {"tsfm", "profile", "insert", "baselines"},
    "core": {"tsfm", "insert", "baselines"},
}
REUSE_ARTIFACT_BUNDLES = {
    "results": (
        "tsfm_results_stage20",
        "tsrouter_core_stage20",
        "baselines_stage20",
        "tables_figures_stage20",
    ),
    "route": (
        "tsfm_results_stage20",
        "tsrouter_core_stage20",
        "task_cache_stage20",
    ),
    "core": (
        "profile_sources",
        "tsfm_results_stage20",
        "task_cache_stage20",
    ),
}


def _workspace_root(raw_root: str | None = None) -> Path:
    if raw_root:
        return Path(raw_root).resolve()
    return _release_paths().root.parent.resolve()


def artifact_bundles_for_reuse(reuse: str | None) -> tuple[str, ...]:
    return REUSE_ARTIFACT_BUNDLES[normalize_reuse_mode(reuse)]


def _load_profiles() -> dict[str, Any]:
    return load_yaml(_release_paths().config_path("paper_run_profiles.yaml"))


def _main_profile() -> dict[str, Any]:
    data = _load_profiles()
    profile = data.get("main_profile")
    if not isinstance(profile, dict):
        raise ExecutionPlanError("paper_run_profiles.yaml must define main_profile")
    prepared = copy.deepcopy(profile)
    for name, value in FIXED_RANDOM_VALUES.items():
        prepared[f"{name}_{FIXED_RANDOM_TOKEN}"] = value
    return prepared


def _profile_for_variant(variant: str) -> dict[str, Any]:
    profile = _main_profile()
    if variant == "fast":
        profile["name"] = "TSRouter-fast"
        profile["route_efficiency_mode"] = True
    elif variant not in {"main", ""}:
        raise ExecutionPlanError(f"unknown TSRouter variant: {variant}")
    return profile


BASELINE_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "AutoForecast": {
        "repr_v": 7,
        "route_efficiency_mode": False,
        "autoforecast_learner": "GBDT",
    },
    "AutoXPCR": {
        "repr_v": 7,
        "route_efficiency_mode": True,
        "autoforecast_learner": "GBDT",
    },
    "SimpleTS": {
        "repr_encoder": "TS2Vec",
        "repr_v": 6,
        "route_efficiency_mode": False,
        "simplets_ts2vec_source_repr_encoder": "StatsRandomFourier",
    },
    "Profile-probe-M": {
        "base_metrics": "M",
        "task_rank_top3_instability_threshold": 0.0,
    },
}


def _profile_for_baseline(method: str) -> dict[str, Any]:
    profile = _main_profile()
    overrides = BASELINE_PROFILE_OVERRIDES.get(method)
    if overrides is None:
        raise ExecutionPlanError(f"unknown route-style baseline: {method}")
    profile.update(overrides)
    profile["name"] = method
    return profile


def _variants(spec: str) -> list[str]:
    raw = str(spec or "main").strip()
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not values:
        values = ["main"]
    out: list[str] = []
    for value in values:
        if value in {"tsrouter-main", "main"}:
            value = "main"
        elif value in {"tsrouter-fast", "fast"}:
            value = "fast"
        if value not in {"main", "fast"}:
            raise ExecutionPlanError(f"unknown variant {value!r}; use main, fast, or main,fast")
        if value not in out:
            out.append(value)
    return out


def normalize_reuse_mode(reuse: str | None) -> str:
    value = str(reuse or "results").strip().lower()
    value = REUSE_MODE_ALIASES.get(value, value)
    if value not in REUSE_MODES:
        available = ", ".join(sorted(PUBLIC_REUSE_MODES))
        raise ExecutionPlanError(f"unknown reuse level {reuse!r}; choose one of: {available}")
    return value


def command_reuses_outputs(reuse: str | None, command: str) -> bool:
    return str(command) in REUSE_SKIPPED_COMMANDS[normalize_reuse_mode(reuse)]


def reuse_task_cache(reuse: str | None) -> bool:
    return normalize_reuse_mode(reuse) in {"route", "core"}


def _bool_text(value: Any) -> str:
    return "True" if bool(value) else "False"


def _value_text(value: Any) -> str:
    if isinstance(value, bool):
        return _bool_text(value)
    return str(value)


def _add_kv(argv: list[str], key: str, value: Any) -> None:
    argv.extend([key, _value_text(value)])


def _add_flag(argv: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        argv.append(flag)


SCRIPT_PATHS = {
    "cli.run_model_zoo": "src/cli/run_model_zoo.py",
    "cli.check_selector": "src/cli/check_selector.py",
}


def _base_python_argv(module: str, python_bin: str) -> list[str]:
    script_path = SCRIPT_PATHS.get(module)
    if script_path:
        return [python_bin, "-u", str(_release_paths().root / script_path)]
    return [python_bin, "-u", "-m", module]


def _device_ids(raw: Any) -> tuple[str, ...]:
    text = str(raw or "").strip()
    if not text:
        return ()
    devices = tuple(part.strip() for part in text.split(",") if part.strip())
    if not devices or any(not item.isdigit() or int(item) < 0 for item in devices):
        raise ExecutionPlanError("devices must be a comma-separated list of non-negative GPU IDs")
    if len(set(devices)) != len(devices):
        raise ExecutionPlanError("devices must not contain duplicate GPU IDs")
    return devices


def _replace_option(argv: list[str], option: str, value: str) -> list[str]:
    updated = list(argv)
    try:
        index = updated.index(option)
    except ValueError:
        updated.extend([option, value])
    else:
        if index + 1 >= len(updated):
            updated.append(value)
        else:
            updated[index + 1] = value
    return updated


def _with_argv(command: ExecutionCommand, argv: list[str], *, metadata: dict[str, object] | None = None) -> ExecutionCommand:
    return replace(
        command,
        argv=tuple(argv),
        command_line=shlex.join(str(part) for part in argv),
        metadata=dict(command.metadata if metadata is None else metadata),
    )


def _with_quick_test(command: ExecutionCommand) -> ExecutionCommand:
    if command.module not in {"cli.run_model_zoo", "cli.check_selector"}:
        return command
    argv = list(command.argv)
    if "--quick_test" not in argv:
        argv.append("--quick_test")
    return _with_argv(command, argv)


def _parallel_model_command(command: ExecutionCommand, devices: tuple[str, ...]) -> ExecutionCommand:
    jobs: list[dict[str, object]] = []
    for family, variants in Model_zoo_details.items():
        for variant in variants:
            argv = _replace_option(list(command.argv), "--models", str(family))
            argv = _replace_option(argv, "--size_mode", str(variant))
            jobs.append({"model": f"{family}_{variant}", "argv": argv})
    metadata = dict(command.metadata)
    metadata["parallel_devices"] = list(devices)
    metadata["parallel_model_jobs"] = jobs
    return _with_argv(command, list(command.argv), metadata=metadata)


def _command(
    *,
    operation: str,
    module: str,
    argv: list[str],
    cwd: Path,
    skip_saved: bool,
    metadata: dict[str, object] | None = None,
    env: dict[str, str] | None = None,
) -> ExecutionCommand:
    return ExecutionCommand(
        operation=operation,
        module=module,
        argv=tuple(argv),
        cwd=str(cwd),
        skip_saved=skip_saved,
        command_line=shlex.join(str(part) for part in argv),
        metadata=dict(metadata or {}),
        env={str(k): str(v) for k, v in (env or {}).items()},
    )


def _public_operation(command: ExecutionCommand) -> dict[str, object]:
    metadata = command.metadata
    payload: dict[str, object] = {
        "operation": command.operation,
        "artifact_backed": command.skip_saved,
    }
    if "baseline_method" in metadata:
        payload["method"] = metadata["baseline_method"]
    if "variant" in metadata:
        payload["variant"] = metadata["variant"]
    if "parallel_devices" in metadata:
        payload["devices"] = list(metadata["parallel_devices"])
        payload["model_jobs"] = len(metadata.get("parallel_model_jobs", []))
    return payload


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower() or "config"


def _param_token(value: Any) -> str:
    if isinstance(value, (float, int)) and math.isfinite(float(value)) and float(value).is_integer():
        return str(int(value))
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if math.isfinite(number) and number.is_integer() else text


def _fallback_token(value: Any) -> str | None:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(threshold) or threshold < 0:
        return None
    return f"fb{_param_token(0.0 if abs(threshold) < 1e-12 else threshold)}"


def _route_suffix(profile: dict[str, Any]) -> str:
    parts = [
        "main",
        f"enc{profile['repr_encoder']}",
        f"qc{profile['repr_sample_qc_mode']}",
        f"scale{profile['repr_scale_protocol']}",
        f"zoo{profile['zoo_repr_set']}",
        f"n{_param_token(profile['repr_size'])}",
        f"v{_param_token(profile['repr_v'])}{profile['base_metrics']}",
        f"task{_param_token(profile['sample_repr_num'])}",
        f"aws{profile['repr_anchor_window_sample_strategy']}",
        f"ws{profile['task_window_sample_strategy']}",
        f"sr{_param_token(profile['sample_repr_ratio'])}",
    ]
    fallback = _fallback_token(profile["task_rank_top3_instability_threshold"])
    if fallback is not None:
        parts.append(fallback)
    channel_fuse = str(profile.get("task_channel_fuse_limit", "all") or "all")
    if channel_fuse.strip().lower() not in {"", "all", "none"}:
        parts.append(f"cf{channel_fuse}")
    if bool(profile.get("route_efficiency_mode", False)):
        parts.append("rfast")
    route_family_mode = str(profile.get("route_family_mode", "default") or "default").strip().lower()
    if route_family_mode != "default":
        parts.append("rfbigger" if route_family_mode == "bigger_size" else "rfsmaller")
    return _slug("_".join(parts))


def _route_id(stage: int, profile: dict[str, Any]) -> str:
    return f"stage{int(stage)}_{_route_suffix(profile)}_route"


def _compat_flag_name(name: str) -> str:
    return name


def _fixed_value(profile: dict[str, Any], name: str) -> Any:
    return profile[f"{name}_{FIXED_RANDOM_TOKEN}"]


def _common_repr_args(profile: dict[str, Any], *, include_sample_ratio: bool = True) -> list[str]:
    argv: list[str] = []
    _add_kv(argv, "--zoo_repr_set", profile["zoo_repr_set"])
    _add_kv(argv, "--repr_size", profile["repr_size"])
    _add_kv(argv, "--sample_mode", profile["sample_mode"])
    _add_kv(argv, "--encoder_type", profile["encoder_type"])
    _add_kv(argv, "--encoder_structure", profile["encoder_structure"])
    _add_kv(argv, "--simplets_ts2vec_checkpoint", "")
    _add_kv(
        argv,
        "--simplets_ts2vec_source_repr_encoder",
        profile.get("simplets_ts2vec_source_repr_encoder", profile["repr_encoder"]),
    )
    _add_kv(argv, "--train_encoder_epochs", profile["train_encoder_epochs"])
    _add_kv(argv, "--repr_input_dim", profile["repr_input_dim"])
    _add_kv(argv, "--repr_output_dim", profile["repr_output_dim"])
    _add_kv(argv, "--repr_sub_pred_len", profile["repr_sub_pred_len"])
    _add_kv(argv, "--repr_sample_qc_mode", profile["repr_sample_qc_mode"])
    _add_kv(argv, "--repr_scale_protocol", profile["repr_scale_protocol"])
    _add_kv(argv, "--task_sample_strategy", profile["task_sample_strategy"])
    _add_kv(argv, "--repr_anchor_window_sample_strategy", profile["repr_anchor_window_sample_strategy"])
    if include_sample_ratio:
        _add_kv(argv, "--sample_repr_ratio", profile["sample_repr_ratio"])
    _add_kv(argv, "--repr_anchor_protocol", profile["repr_anchor_protocol"])
    return argv


def _index_args(profile: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    _add_kv(argv, "--repr_v", profile["repr_v"])
    _add_kv(argv, "--model_repr_mode", profile["model_repr_mode"])
    _add_kv(argv, "--subset_top_k", profile["subset_top_k"])
    _add_kv(argv, "--subset_perf_scale", profile["subset_perf_scale"])
    _add_kv(argv, "--base_metrics", profile["base_metrics"])
    _add_kv(argv, "--autoforecast_learner", "GBDT")
    _add_kv(argv, "--autoforecast_hidden_dim", 64)
    _add_kv(argv, "--autoforecast_train_epochs", 120)
    _add_kv(argv, "--autoforecast_learning_rate", 0.001)
    _add_kv(argv, "--autoforecast_batch_size", 256)
    _add_kv(argv, "--advanced_baseline_train_scope", "center")
    _add_kv(argv, "--repr_weight_ratio", profile["repr_weight_ratio"])
    _add_kv(argv, "--route_efficiency_mode", profile["route_efficiency_mode"])
    _add_kv(argv, "--rank_decay_coef", profile["rank_decay_coef"])
    _add_kv(argv, "--repr_distance_metric", profile["repr_distance_metric"])
    _add_kv(argv, "--model_repr_agg", profile["model_repr_agg"])
    _add_kv(argv, "--enable_process_metrics", profile["enable_process_metrics"])
    _add_kv(argv, "--process_metrics_region_rule", profile["process_metrics_region_rule"])
    return argv


def _profile_anchor_command(profile: dict[str, Any], *, python_bin: str, cwd: Path, skip_saved: bool) -> ExecutionCommand:
    argv = _base_python_argv("cli.run_model_zoo", python_bin)
    argv.append("--save_repr_selection")
    argv.extend(_common_repr_args(profile))
    _add_flag(argv, "--skip_saved", skip_saved)
    return _command(
        operation="profile_anchors",
        module="cli.run_model_zoo",
        argv=argv,
        cwd=cwd,
        skip_saved=skip_saved,
    )


def _profile_forward_command(profile: dict[str, Any], *, python_bin: str, cwd: Path, skip_saved: bool) -> ExecutionCommand:
    argv = _base_python_argv("cli.run_model_zoo", python_bin)
    _add_kv(argv, "--run_mode", "zoo_repr_set_forward")
    argv.extend(_common_repr_args(profile))
    _add_kv(argv, "--route_efficiency_mode", profile["route_efficiency_mode"])
    _add_kv(argv, "--models", "all_zoo")
    _add_kv(argv, "--size_mode", "all_size")
    _add_kv(argv, "--batch_size", profile["batch_size"])
    _add_kv(argv, "--context_len", profile["context_len"])
    _add_kv(argv, "--analysis_keep_clusters", 0)
    _add_kv(argv, "--enable_process_metrics", profile["enable_process_metrics"])
    argv.extend(["--debug_mode", "--fix_context_len"])
    _add_kv(argv, _compat_flag_name("--skip-" + "step" + "2-cluster-forward"), profile["skip_profile_forward_clusters"])
    _add_flag(argv, "--skip_saved", skip_saved)
    return _command(
        operation="profile_forwards",
        module="cli.run_model_zoo",
        argv=argv,
        cwd=cwd,
        skip_saved=skip_saved,
    )


def _capability_index_command(
    profile: dict[str, Any],
    *,
    python_bin: str,
    cwd: Path,
    skip_saved: bool,
    stage: int | None = None,
    quick_test: bool = False,
) -> ExecutionCommand:
    argv = _base_python_argv("cli.run_model_zoo", python_bin)
    _add_kv(argv, "--run_mode", "select")
    argv.append("--save_model_zoo_repr")
    argv.extend(_common_repr_args(profile, include_sample_ratio=False))
    argv.extend(_index_args(profile))
    argv.append("--debug_mode")
    _add_kv(argv, "--context_len", profile["context_len"])
    argv.append("--fix_context_len")
    argv.append("--real_world_mode")
    if stage is not None:
        _add_kv(argv, "--only_zoo_stage", int(stage))
    if quick_test and str(profile.get("repr_v", ""))[:1] == "6":
        _add_kv(argv, "--train_encoder_epochs", 3)
    local_simplets_checkpoint = os.environ.get("TSROUTER_SIMPLETS_CHECKPOINT", "").strip()
    if local_simplets_checkpoint and str(profile.get("repr_v", ""))[:1] == "6":
        _add_kv(argv, "--simplets_ts2vec_checkpoint", local_simplets_checkpoint)
    _add_flag(argv, "--skip_saved", skip_saved)
    return _command(
        operation=f"capability_index_{'fast' if profile.get('route_efficiency_mode') else 'main'}",
        module="cli.run_model_zoo",
        argv=argv,
        cwd=cwd,
        skip_saved=skip_saved,
    )


def _route_select_command(
    profile: dict[str, Any],
    *,
    python_bin: str,
    cwd: Path,
    skip_saved: bool,
    stage: int,
    only_stage: bool = False,
    use_cached_task_samples: bool = False,
    cache_only: bool = False,
) -> ExecutionCommand:
    variant = "fast" if profile.get("route_efficiency_mode") else "main"
    route_id = _route_id(stage, profile)
    route_id_candidates = [route_id]
    if variant == "main":
        route_id_candidates.append(f"stage{int(stage)}_main_route")
    argv = _base_python_argv("cli.run_model_zoo", python_bin)
    _add_kv(argv, "--run_mode", "select")
    _add_kv(argv, "--models", "TSRouter")
    argv.extend(_common_repr_args(profile))
    _add_kv(argv, "--task_window_sample_strategy", profile["task_window_sample_strategy"])
    _add_kv(argv, "--task_rank_top3_instability_threshold", profile["task_rank_top3_instability_threshold"])
    _add_kv(argv, "--task_channel_fuse_limit", profile["task_channel_fuse_limit"])
    _add_kv(argv, "--route_family_mode", profile["route_family_mode"])
    argv.extend(_index_args(profile))
    _add_kv(argv, "--ensemble_size", profile["ensemble_size"])
    _add_kv(argv, "--ensemble_agg", profile["ensemble_agg"])
    _add_kv(argv, "--sample_repr_num", profile["sample_repr_num"])
    _add_kv(argv, "--task_sample_version", profile["task_sample_version"])
    _add_kv(argv, "--restrict_top_model_num", profile["restrict_top_model_num"])
    _add_kv(argv, "--GE_fast_eval", profile["ge_fast_eval"])
    _add_kv(argv, "--TSFM_results_dir", profile["tsfm_results_dir"])
    _add_kv(argv, "--rank_truth_cls", profile["rank_truth_cls"])
    argv.append("--fix_context_len")
    _add_kv(argv, "--context_len", profile["context_len"])
    _add_kv(argv, "--mix-route", profile["mix_route"])
    _add_kv(argv, "--mix-route-model-num", profile["mix_route_model_num"])
    _add_kv(argv, "--vldb_route_stage", int(stage))
    _add_kv(argv, "--vldb_route_id", route_id)
    _add_kv(argv, "--vldb_route_profile_id", f"stage{int(stage)}_{variant}")
    _add_kv(argv, "--vldb_fast_sample", use_cached_task_samples)
    _add_kv(argv, "--route-cache-only", cache_only)
    _add_kv(argv, "--vldb_fast_forward", True)
    _add_kv(argv, "--vldb_skip_evaluate", True)
    argv.append("--real_world_mode")
    if only_stage:
        _add_kv(argv, "--only_zoo_stage", int(stage))
    _add_flag(argv, "--skip_saved", skip_saved)
    return _command(
        operation=f"route_select_{variant}",
        module="cli.run_model_zoo",
        argv=argv,
        cwd=cwd,
        skip_saved=skip_saved,
        metadata={
            "route_id": route_id,
            "route_id_candidates": route_id_candidates,
            "route_suffix_source": "cli.check_selector::_vldb_route_suffix_for_args",
        },
    )


def _tsfm_command(profile: dict[str, Any], *, python_bin: str, cwd: Path, skip_saved: bool) -> ExecutionCommand:
    argv = _base_python_argv("cli.run_model_zoo", python_bin)
    _add_kv(argv, "--run_mode", "zoo")
    _add_kv(argv, "--models", "all_zoo")
    _add_kv(argv, "--size_mode", "all_size")
    _add_kv(argv, "--context_len", profile["context_len"])
    _add_kv(argv, "--batch_size", profile["batch_size"])
    argv.append("--fix_context_len")
    _add_flag(argv, "--skip_saved", skip_saved)
    return _command(
        operation="tsfm_zero_shot",
        module="cli.run_model_zoo",
        argv=argv,
        cwd=cwd,
        skip_saved=skip_saved,
    )


def _insert_commands(
    profile: dict[str, Any],
    args: Any,
    *,
    python_bin: str,
    cwd: Path,
    skip_saved: bool,
) -> list[ExecutionCommand]:
    start = int(getattr(args, "start_stage", None) or max(3, int(getattr(args, "stage", 20)) - 1))
    end = int(getattr(args, "end_stage", None) or int(getattr(args, "stage", start + 1)))
    if end <= start:
        end = start + 1
    commands: list[ExecutionCommand] = []
    raw_variant = str(getattr(args, "variant", "") or "main,fast")
    for variant in _variants(raw_variant):
        command = _capability_index_command(
            _profile_for_variant(variant),
            python_bin=python_bin,
            cwd=cwd,
            skip_saved=skip_saved,
        )
        command.metadata.update(
            {
                "insert_stage_start": start,
                "insert_stage_end": end,
                "maintenance_log": "results_csv/TSRouter/Model_zoo_repr/insert_timing.csv",
                "insert_source": "capability-index refresh",
                "variant": variant,
            }
        )
        commands.append(command)
    return commands


SELECTOR_BASELINE_MODELS: dict[str, str] = {
    "Random": "Random_Select",
    "Recent": "Recent_Select",
}

TASK_PROBE_MIX_ROUTE_METHODS = {"Task-probe"}


def _selector_common_args(profile: dict[str, Any], random_value: int, *, include_repr_args: bool = True) -> list[str]:
    argv: list[str] = []
    argv.append("--fix_context_len")
    _add_kv(argv, "--ensemble_size", 1)
    _add_kv(argv, "--ensemble_agg", profile["ensemble_agg"])
    _add_kv(argv, "--GE_fast_eval", profile["ge_fast_eval"])
    _add_kv(argv, "--vldb_skip_evaluate", True)
    _add_kv(argv, "--vldb_route_latency_log", "results_csv/TSRouter/vldb/logs/route_latency_log.csv")
    _add_kv(argv, "--vldb_route_stage", -1)
    _add_kv(argv, "--vldb_route_profile_id", "selector_baseline")
    _add_kv(argv, "--vldb_fast_forward", False)
    if include_repr_args:
        _add_kv(argv, "--encoder_type", profile["encoder_type"])
        _add_kv(argv, "--encoder_structure", profile["encoder_structure"])
        _add_kv(argv, "--repr_input_dim", profile["repr_input_dim"])
        _add_kv(argv, "--repr_output_dim", profile["repr_output_dim"])
        _add_kv(argv, "--repr_sub_pred_len", profile["repr_sub_pred_len"])
        _add_kv(argv, "--zoo_repr_set", profile["zoo_repr_set"])
        _add_kv(argv, "--repr_size", profile["repr_size"])
        _add_kv(argv, "--sample_mode", profile["sample_mode"])
        _add_kv(argv, "--repr_scale_protocol", profile["repr_scale_protocol"])
        _add_kv(argv, "--task_sample_strategy", profile["task_sample_strategy"])
        _add_kv(argv, "--repr_anchor_protocol", profile["repr_anchor_protocol"])
        _add_kv(argv, "--task_window_sample_strategy", profile["task_window_sample_strategy"])
        _add_kv(argv, "--sample_repr_ratio", profile["sample_repr_ratio"])
    return argv


def _selector_baseline_command(
    method: str,
    profile: dict[str, Any],
    *,
    python_bin: str,
    cwd: Path,
    stage: int,
    random_value: int,
    skip_saved: bool,
) -> ExecutionCommand:
    model_name = SELECTOR_BASELINE_MODELS[method]
    argv = _base_python_argv("cli.run_model_zoo", python_bin)
    _add_kv(argv, "--run_mode", "select")
    _add_kv(argv, "--models", model_name)
    _add_kv(argv, "--current_zoo_num", stage)
    _add_kv(argv, "--zoo_total_num", stage)
    argv.extend(_selector_common_args(profile, random_value, include_repr_args=method not in {"Random", "Recent"}))
    if method == "Random":
        _add_kv(argv, "--repeat_id", random_value)
    if method == "Task-probe":
        _add_kv(argv, "--sample_repr_num", profile["sample_repr_num"])
    if method in {"Random", "Recent"}:
        argv.append("--real_world_mode")
        _add_kv(argv, "--only_zoo_stage", int(stage))
    _add_flag(argv, "--skip_saved", skip_saved)
    return _command(
        operation=f"selector_baseline_{_slug(method)}",
        module="cli.run_model_zoo",
        argv=argv,
        cwd=cwd,
        skip_saved=skip_saved,
        metadata={
            "baseline_method": method,
            "selector_model": model_name,
            "tracked_source": "src/selector/baselines/baseline_select.py",
        },
    )


def _baseline_methods(args: Any) -> list[str]:
    raw = str(getattr(args, "methods", "") or "all").strip()
    if not raw or raw.lower() == "all":
        return [
            "AutoForecast",
            "AutoXPCR",
            "SimpleTS",
            "Profile-probe-M",
            "Random",
            "Recent",
            "Task-probe",
            "Current-best-M",
            "Current-best-C",
        ]
    aliases = {
        "autoforecast": "AutoForecast",
        "autoxpcr": "AutoXPCR",
        "simplets": "SimpleTS",
        "profile-probe-m": "Profile-probe-M",
        "profile_probe_m": "Profile-probe-M",
        "random": "Random",
        "recent": "Recent",
        "task-probe": "Task-probe",
        "task_probe": "Task-probe",
        "task_probe_forward": "Task-probe",
        "current-best-m": "Current-best-M",
        "current_best_m": "Current-best-M",
        "current-best-c": "Current-best-C",
        "current_best_c": "Current-best-C",
    }
    out: list[str] = []
    for item in raw.split(","):
        key = item.strip()
        if not key:
            continue
        method = aliases.get(key.lower(), key)
        supported = {
            *BASELINE_PROFILE_OVERRIDES,
            *SELECTOR_BASELINE_MODELS,
            *TASK_PROBE_MIX_ROUTE_METHODS,
            "Current-best-M",
            "Current-best-C",
        }
        if method not in supported:
            raise ExecutionPlanError(f"unknown baseline method: {key}")
        if method not in out:
            out.append(method)
    return out


def _route_style_baseline_commands(
    args: Any,
    *,
    python_bin: str,
    cwd: Path,
    skip_saved: bool,
    stage: int,
) -> list[ExecutionCommand]:
    commands: list[ExecutionCommand] = []
    for method in _baseline_methods(args):
        if method not in BASELINE_PROFILE_OVERRIDES:
            continue
        profile = _profile_for_baseline(method)
        commands.append(
            _capability_index_command(
                profile,
                python_bin=python_bin,
                cwd=cwd,
                skip_saved=skip_saved,
                stage=stage,
                quick_test=bool(getattr(args, "quick_test", False)),
            )
        )
        commands[-1].metadata["baseline_method"] = method
        commands.append(
            _route_select_command(
                profile,
                python_bin=python_bin,
                cwd=cwd,
                skip_saved=skip_saved,
                stage=stage,
                only_stage=True,
            )
        )
        commands[-1].metadata["baseline_method"] = method
    return commands


def _selector_baseline_commands(
    args: Any,
    *,
    python_bin: str,
    cwd: Path,
    skip_saved: bool,
    stage: int,
) -> list[ExecutionCommand]:
    profile = _main_profile()
    commands: list[ExecutionCommand] = []
    for method in _baseline_methods(args):
        if method not in SELECTOR_BASELINE_MODELS:
            continue
        random_values = RANDOM_BASELINE_VALUES if method == "Random" else (_fixed_value(profile, "sear" + "ch"),)
        for random_value in random_values:
            commands.append(
                _selector_baseline_command(
                    method,
                    profile,
                    python_bin=python_bin,
                    cwd=cwd,
                    stage=stage,
                    random_value=int(random_value),
                    skip_saved=skip_saved,
                )
            )
    return commands


def _task_probe_mix_route_commands(
    args: Any,
    *,
    python_bin: str,
    cwd: Path,
    skip_saved: bool,
    stage: int,
) -> list[ExecutionCommand]:
    if "Task-probe" not in _baseline_methods(args):
        return []
    profile = _main_profile()
    profile["name"] = "Task-probe"
    profile["mix_route"] = True
    command = _route_select_command(
        profile,
        python_bin=python_bin,
        cwd=cwd,
        skip_saved=skip_saved,
        stage=stage,
        only_stage=True,
        use_cached_task_samples=reuse_task_cache(getattr(args, "reuse", None)),
    )
    metadata = dict(command.metadata)
    metadata.update(
        {
            "baseline_method": "Task-probe",
            "source_method": "TSRouter-main",
            "mix_route": True,
            "tracked_source": "src/selector/TSRouter_Select/task_probe_select.py",
        }
    )
    return [replace(command, operation="task_probe_mix_route", metadata=metadata)]


def _summary_command(
    profile: dict[str, Any],
    args: Any,
    *,
    python_bin: str,
    cwd: Path,
    skip_saved: bool,
) -> ExecutionCommand:
    stage = int(getattr(args, "stage", 20))
    argv = _base_python_argv("cli.check_selector", python_bin)
    argv.append("--vldb_results")
    _add_kv(argv, "--TSFM_results_dir", profile["tsfm_results_dir"])
    _add_kv(argv, "--context_len", profile["context_len"])
    argv.append("--fix_context_len")
    _add_kv(argv, "--repr_encoder", profile["repr_encoder"])
    _add_kv(argv, "--encoder_type", profile["encoder_type"])
    _add_kv(argv, "--encoder_structure", profile["encoder_structure"])
    _add_kv(argv, "--repr_input_dim", profile["repr_input_dim"])
    _add_kv(argv, "--repr_output_dim", profile["repr_output_dim"])
    _add_kv(argv, "--repr_sub_pred_len", profile["repr_sub_pred_len"])
    _add_kv(argv, "--zoo_repr_set", profile["zoo_repr_set"])
    _add_kv(argv, "--repr_size", profile["repr_size"])
    _add_kv(argv, "--repr_v", profile["repr_v"])
    _add_kv(argv, "--base_metrics", profile["base_metrics"])
    _add_kv(argv, "--repr_weight_ratio", profile["repr_weight_ratio"])
    _add_kv(argv, "--sample_repr_num", profile["sample_repr_num"])
    _add_kv(argv, "--repr_sample_qc_mode", profile["repr_sample_qc_mode"])
    _add_kv(argv, "--repr_scale_protocol", profile["repr_scale_protocol"])
    _add_kv(argv, "--task_sample_version", profile["task_sample_version"])
    _add_kv(argv, "--repr_anchor_window_sample_strategy", profile["repr_anchor_window_sample_strategy"])
    _add_kv(argv, "--task_window_sample_strategy", profile["task_window_sample_strategy"])
    _add_kv(argv, "--sample_repr_ratio", profile["sample_repr_ratio"])
    _add_kv(argv, "--task_rank_top3_instability_threshold", profile["task_rank_top3_instability_threshold"])
    _add_kv(argv, "--task_channel_fuse_limit", profile["task_channel_fuse_limit"])
    _add_kv(argv, "--route_family_mode", profile["route_family_mode"])
    _add_kv(argv, "--sample_mode", profile["sample_mode"])
    _add_kv(argv, "--model_repr_mode", profile["model_repr_mode"])
    _add_kv(argv, "--subset_top_k", profile["subset_top_k"])
    _add_kv(argv, "--subset_perf_scale", profile["subset_perf_scale"])
    _add_kv(argv, "--rank_decay_coef", profile["rank_decay_coef"])
    _add_kv(argv, "--ensemble_size", profile["ensemble_size"])
    _add_kv(argv, "--ensemble_agg", profile["ensemble_agg"])
    _add_kv(argv, "--restrict_top_model_num", profile["restrict_top_model_num"])
    _add_kv(argv, "--process_metrics_region_rule", profile["process_metrics_region_rule"])
    _add_kv(argv, "--rank_base", "MASE")
    _add_kv(argv, "--table4_repr_token", profile["zoo_repr_set"])
    _add_kv(argv, "--table4_repr_size", profile["repr_size"])
    _add_kv(argv, "--current_zoo_num", stage)
    return _command(
        operation="summary_tables",
        module="cli.check_selector",
        argv=argv,
        cwd=cwd,
        skip_saved=skip_saved,
    )


def build_release_command_plan(command: str, args: Any) -> dict[str, object]:
    profile = _main_profile()
    compatibility_root = getattr(args, "leg" + "acy" + "_root", None)
    workspace_root = getattr(args, "workspace_root", None) or getattr(args, "root", None) or compatibility_root
    cwd = _workspace_root(workspace_root)
    python_bin = str(getattr(args, "python_bin", "") or sys.executable)
    stage = int(getattr(args, "stage", 20) or 20)
    reuse = normalize_reuse_mode(getattr(args, "reuse", "results"))
    devices = _device_ids(getattr(args, "devices", ""))
    quick_test = bool(getattr(args, "quick_test", False))
    artifact_groups = artifact_bundles_for_reuse(reuse)
    skip_saved = command_reuses_outputs(reuse, command)
    execution_commands: list[ExecutionCommand] = []

    if command == "tsfm":
        execution_commands.append(_tsfm_command(profile, python_bin=python_bin, cwd=cwd, skip_saved=skip_saved))
    elif command == "profile":
        execution_commands.append(_profile_anchor_command(profile, python_bin=python_bin, cwd=cwd, skip_saved=skip_saved))
        execution_commands.append(_profile_forward_command(profile, python_bin=python_bin, cwd=cwd, skip_saved=skip_saved))
        for variant in _variants(getattr(args, "variant", "")):
            execution_commands.append(
                _capability_index_command(
                    _profile_for_variant(variant),
                    python_bin=python_bin,
                    cwd=cwd,
                    skip_saved=skip_saved,
                )
            )
    elif command == "route":
        for variant in _variants(getattr(args, "variant", "")):
            execution_commands.append(
                _route_select_command(
                    _profile_for_variant(variant),
                    python_bin=python_bin,
                    cwd=cwd,
                    skip_saved=skip_saved,
                    stage=stage,
                    use_cached_task_samples=reuse_task_cache(reuse),
                    cache_only=reuse == "route",
                )
            )
    elif command == "insert":
        execution_commands.extend(_insert_commands(profile, args, python_bin=python_bin, cwd=cwd, skip_saved=skip_saved))
    elif command == "baselines":
        execution_commands.extend(
            _route_style_baseline_commands(
                args,
                python_bin=python_bin,
                cwd=cwd,
                skip_saved=skip_saved,
                stage=stage,
            )
        )
        execution_commands.extend(
            _task_probe_mix_route_commands(
                args,
                python_bin=python_bin,
                cwd=cwd,
                skip_saved=skip_saved,
                stage=stage,
            )
        )
        execution_commands.extend(
            _selector_baseline_commands(
                args,
                python_bin=python_bin,
                cwd=cwd,
                skip_saved=skip_saved,
                stage=stage,
            )
        )
    elif command == "summary":
        execution_commands.append(_summary_command(profile, args, python_bin=python_bin, cwd=cwd, skip_saved=skip_saved))
    else:
        raise ExecutionPlanError(f"unsupported public command: {command}")

    if quick_test:
        execution_commands = [_with_quick_test(item) for item in execution_commands]
    if devices:
        execution_commands = [
            _parallel_model_command(item, devices)
            if item.operation in {"tsfm_zero_shot", "profile_forwards"}
            else item
            for item in execution_commands
        ]

    return {
        "command": command,
        "action": str(getattr(args, "action", "")),
        "stage": stage,
        "reuse": reuse,
        "artifact_groups": list(artifact_groups),
        "execution_mode": "execute" if bool(getattr(args, "execute", False)) else "plan",
        "quick_test": quick_test,
        "devices": list(devices),
        "python_bin": python_bin,
        "workspace_root": str(cwd),
        "execution_contract": "configs/execution_contract.yaml",
        "operations": [_public_operation(item) for item in execution_commands],
        "_execution_commands": [item.as_dict() for item in execution_commands],
    }


def _progress_line(line: str, trace_active: bool) -> tuple[bool, bool]:
    text = line.strip()
    if text.startswith("Traceback"):
        return True, True
    if trace_active:
        return True, bool(text)
    lower = text.lower()
    markers = (
        "running ",
        "dataset:",
        "save path",
        "completed",
        "complete",
        "saved",
        "profile",
        "route",
        "insert",
        "baseline",
        "table",
        "error",
        "exception",
    )
    return any(marker in lower for marker in markers), False


def _command_env(command: dict[str, object], cwd: str) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in dict(command.get("env", {}) or {}).items():
        env[str(key)] = str(value)
    release_root = _release_paths().root
    src_path = str(release_root / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("TSROUTER_WORKSPACE_ROOT", cwd)
    env.setdefault("TSROUTER_PROFILE_SOURCE_ROOT", str(release_root / "data" / "profile_sources"))
    return env


def _run_logged_process(
    argv: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    output_log: Path,
    progress_prefix: str = "",
) -> int:
    progress_enabled = os.environ.get("TSROUTER_PROGRESS_STREAM", "").strip().lower() in {"1", "true", "yes"}
    with output_log.open("w", encoding="utf-8") as handle:
        try:
            proc = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            handle.write(f"failed to start process: {exc}\n")
            return 127
        trace_active = False
        assert proc.stdout is not None
        for line in proc.stdout:
            handle.write(line)
            handle.flush()
            if progress_enabled:
                show, trace_active = _progress_line(line, trace_active)
                if show:
                    sys.stderr.write(f"{progress_prefix}{line}")
                    sys.stderr.flush()
        return proc.wait()


def _parallel_model_results(
    command: dict[str, object],
    *,
    index: int,
    cwd: str,
    env: dict[str, str],
    log_root: Path,
) -> dict[str, object]:
    metadata = dict(command.get("metadata", {}) or {})
    devices = tuple(str(item) for item in metadata.get("parallel_devices", []) or [])
    jobs = list(metadata.get("parallel_model_jobs", []) or [])
    if not devices or not jobs:
        raise ExecutionPlanError("parallel model command is missing devices or model jobs")

    operation = str(command.get("operation", "operation"))
    safe_operation = re.sub(r"[^A-Za-z0-9_.-]+", "_", operation).strip("_") or "operation"
    pending: Queue[dict[str, object]] = Queue()
    for job in jobs:
        if isinstance(job, dict):
            pending.put(job)
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    lock = Lock()
    stop = Event()

    def worker(device: str) -> None:
        while not stop.is_set():
            try:
                job = pending.get_nowait()
            except Empty:
                return
            model = str(job.get("model", "model"))
            argv = [str(item) for item in job.get("argv", [])]
            worker_env = dict(env)
            worker_env["CUDA_VISIBLE_DEVICES"] = str(device)
            safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "model"
            output_log = log_root / f"{index:02d}_{safe_operation}_{safe_model}.log"
            returncode = _run_logged_process(
                argv,
                cwd=cwd,
                env=worker_env,
                output_log=output_log,
                progress_prefix=f"[gpu {device} | {model}] ",
            )
            record = {"model": model, "device": str(device), "returncode": returncode, "output_log": output_log.name}
            with lock:
                records.append(record)
                if returncode != 0:
                    failures.append(record)
                    stop.set()
            pending.task_done()

    with ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix="tsrouter-gpu") as pool:
        futures = [pool.submit(worker, device) for device in devices]
        for future in futures:
            future.result()

    records.sort(key=lambda item: str(item["model"]))
    if failures:
        failure = failures[0]
        raise ExecutionPlanError(
            f"parallel model execution failed: {failure['model']} on gpu {failure['device']} rc={failure['returncode']}"
        )
    return {
        "operation": operation,
        "returncode": 0,
        "parallel": True,
        "devices": list(devices),
        "model_count": len(jobs),
        "model_results": records,
    }


def execute_release_command_plan(plan: dict[str, object]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for index, command in enumerate(plan.get("_execution_commands", []), start=1):
        if not isinstance(command, dict):
            continue
        argv = [str(item) for item in command.get("argv", [])]
        cwd = str(command.get("cwd") or ".")
        env = _command_env(command, cwd)
        operation = str(command.get("operation", "operation"))
        safe_operation = re.sub(r"[^A-Za-z0-9_.-]+", "_", operation).strip("_") or "operation"
        log_root = Path(os.environ.get("TSROUTER_RUNTIME_LOG_DIR", Path(cwd) / "reproduction_logs" / "operations"))
        log_root.mkdir(parents=True, exist_ok=True)
        metadata = dict(command.get("metadata", {}) or {})
        if metadata.get("parallel_model_jobs"):
            results.append(_parallel_model_results(command, index=index, cwd=cwd, env=env, log_root=log_root))
            continue

        output_log = log_root / f"{index:02d}_{safe_operation}.log"
        returncode = _run_logged_process(argv, cwd=cwd, env=env, output_log=output_log)
        results.append({"operation": operation, "returncode": returncode, "output_log": output_log.name})
        if returncode != 0:
            raise ExecutionPlanError(f"execution command failed: {operation} rc={returncode}")
    return results

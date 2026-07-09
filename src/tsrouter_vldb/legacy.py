from __future__ import annotations

import copy
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .artifacts import load_yaml
from .commands import COMMAND_ARTIFACT_GROUPS
from .paths import ReleasePaths


class LegacyPlanError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendCommand:
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


def _legacy_root(raw_root: str | None = None) -> Path:
    if raw_root:
        return Path(raw_root).resolve()
    return _release_paths().root.parent.resolve()


def _load_profiles() -> dict[str, Any]:
    return load_yaml(_release_paths().config_path("paper_run_profiles.yaml"))


def _main_profile() -> dict[str, Any]:
    data = _load_profiles()
    profile = data.get("main_profile")
    if not isinstance(profile, dict):
        raise LegacyPlanError("paper_run_profiles.yaml must define main_profile")
    return copy.deepcopy(profile)


def _profile_for_variant(variant: str) -> dict[str, Any]:
    profile = _main_profile()
    if variant == "fast":
        profile["name"] = "TSRouter-fast"
        profile["route_efficiency_mode"] = True
    elif variant not in {"main", ""}:
        raise LegacyPlanError(f"unknown TSRouter variant: {variant}")
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
    "Profile-probe-C": {
        "base_metrics": "C",
        "task_rank_top3_instability_threshold": 0.0,
    },
}


def _profile_for_baseline(method: str) -> dict[str, Any]:
    profile = _main_profile()
    overrides = BASELINE_PROFILE_OVERRIDES.get(method)
    if overrides is None:
        raise LegacyPlanError(f"unknown route-style baseline: {method}")
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
            raise LegacyPlanError(f"unknown variant {value!r}; use main, fast, or main,fast")
        if value not in out:
            out.append(value)
    return out


def _reuse_skip_saved(reuse: str, artifact_groups: tuple[str, ...] = ()) -> bool:
    value = str(reuse or "all").strip().lower()
    if value == "all":
        return True
    if value in {"", "none", "false", "no"}:
        return False
    requested = {item.strip().replace("-", "_") for item in value.split(",") if item.strip()}
    groups = {item.strip().replace("-", "_") for item in artifact_groups}
    return bool(groups) and groups.issubset(requested)


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


BACKEND_SCRIPT_PATHS = {
    "cli.run_model_zoo": "src/cli/run_model_zoo.py",
    "cli.check_selector": "src/cli/check_selector.py",
}


def _base_python_argv(module: str, python_bin: str) -> list[str]:
    script_path = BACKEND_SCRIPT_PATHS.get(module)
    if script_path:
        return [python_bin, script_path]
    return [python_bin, "-m", module]


def _command(
    *,
    operation: str,
    module: str,
    argv: list[str],
    cwd: Path,
    skip_saved: bool,
    metadata: dict[str, object] | None = None,
    env: dict[str, str] | None = None,
) -> BackendCommand:
    return BackendCommand(
        operation=operation,
        module=module,
        argv=tuple(argv),
        cwd=str(cwd),
        skip_saved=skip_saved,
        command_line=shlex.join(str(part) for part in argv),
        metadata=dict(metadata or {}),
        env={str(k): str(v) for k, v in (env or {}).items()},
    )


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


def _common_repr_args(profile: dict[str, Any], *, include_sample_ratio: bool = True) -> list[str]:
    argv: list[str] = []
    for key in ("repr_data_seed", "repr_encoder_seed", "forward_seed", "search_seed"):
        _add_kv(argv, f"--{key}", profile[key])
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


def _step1_command(profile: dict[str, Any], *, python_bin: str, cwd: Path, skip_saved: bool) -> BackendCommand:
    argv = _base_python_argv("cli.run_model_zoo", python_bin)
    argv.append("--save_repr_selection")
    argv.extend(_common_repr_args(profile))
    argv.append("--strict_phase_seed")
    _add_flag(argv, "--skip_saved", skip_saved)
    return _command(
        operation="profile_anchors",
        module="cli.run_model_zoo",
        argv=argv,
        cwd=cwd,
        skip_saved=skip_saved,
    )


def _step2_command(profile: dict[str, Any], *, python_bin: str, cwd: Path, skip_saved: bool) -> BackendCommand:
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
    argv.extend(["--debug_mode", "--strict_phase_seed", "--fix_context_len"])
    _add_kv(argv, "--skip-step2-cluster-forward", profile["skip_step2_cluster_forward"])
    _add_flag(argv, "--skip_saved", skip_saved)
    return _command(
        operation="profile_forwards",
        module="cli.run_model_zoo",
        argv=argv,
        cwd=cwd,
        skip_saved=skip_saved,
    )


def _step3_command(profile: dict[str, Any], *, python_bin: str, cwd: Path, skip_saved: bool) -> BackendCommand:
    argv = _base_python_argv("cli.run_model_zoo", python_bin)
    _add_kv(argv, "--run_mode", "select")
    argv.append("--save_model_zoo_repr")
    argv.extend(_common_repr_args(profile, include_sample_ratio=False))
    argv.extend(_index_args(profile))
    argv.extend(["--debug_mode", "--strict_phase_seed"])
    _add_kv(argv, "--context_len", profile["context_len"])
    argv.append("--fix_context_len")
    argv.append("--real_world_mode")
    _add_flag(argv, "--skip_saved", skip_saved)
    return _command(
        operation=f"capability_index_{'fast' if profile.get('route_efficiency_mode') else 'main'}",
        module="cli.run_model_zoo",
        argv=argv,
        cwd=cwd,
        skip_saved=skip_saved,
    )


def _step4_command(profile: dict[str, Any], *, python_bin: str, cwd: Path, skip_saved: bool, stage: int) -> BackendCommand:
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
    argv.extend(["--strict_phase_seed", "--fix_context_len"])
    _add_kv(argv, "--context_len", profile["context_len"])
    _add_kv(argv, "--mix-route", profile["mix_route"])
    _add_kv(argv, "--mix-route-model-num", profile["mix_route_model_num"])
    _add_kv(argv, "--vldb_route_stage", int(stage))
    _add_kv(argv, "--vldb_route_id", route_id)
    _add_kv(argv, "--vldb_route_profile_id", f"stage{int(stage)}_{variant}")
    _add_kv(argv, "--vldb_fast_sample", skip_saved)
    _add_kv(argv, "--vldb_fast_forward", True)
    _add_kv(argv, "--vldb_skip_evaluate", True)
    argv.append("--real_world_mode")
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


def _tsfm_command(profile: dict[str, Any], *, python_bin: str, cwd: Path, skip_saved: bool) -> BackendCommand:
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
) -> list[BackendCommand]:
    start = int(getattr(args, "start_stage", None) or max(3, int(getattr(args, "stage", 20)) - 1))
    end = int(getattr(args, "end_stage", None) or int(getattr(args, "stage", start + 1)))
    if end <= start:
        end = start + 1
    commands: list[BackendCommand] = []
    raw_variant = str(getattr(args, "variant", "") or "main,fast")
    for variant in _variants(raw_variant):
        command = _step3_command(
            _profile_for_variant(variant),
            python_bin=python_bin,
            cwd=cwd,
            skip_saved=skip_saved,
        )
        command.metadata.update(
            {
                "insert_stage_start": start,
                "insert_stage_end": end,
                "insert_timing_csv": "results_csv/TSRouter/Model_zoo_repr/step3_insert_timing.csv",
                "insert_source": "cli.run_model_zoo capability-index refresh",
                "variant": variant,
            }
        )
        commands.append(command)
    return commands


SELECTOR_BASELINE_MODELS: dict[str, str] = {
    "Random": "Random_Select",
    "Recent": "Recent_Select",
    "MetaFeature-GBDT": "MetaFeature_GBDT_Select",
    "MetaFeature-MLP": "MetaFeature_MLP_Select",
    "Task-Oracle": "Task_Oracle_Best_Select",
}

TASK_PROBE_MIX_ROUTE_METHODS = {"Task-probe"}


def _selector_common_args(profile: dict[str, Any], seed: int, *, include_repr_args: bool = True) -> list[str]:
    argv: list[str] = []
    argv.append("--fix_context_len")
    _add_kv(argv, "--ensemble_size", 1)
    _add_kv(argv, "--ensemble_agg", profile["ensemble_agg"])
    _add_kv(argv, "--search_seed", seed)
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
        _add_kv(argv, "--repr_data_seed", profile["repr_data_seed"])
        _add_kv(argv, "--repr_encoder_seed", profile["repr_encoder_seed"])
        _add_kv(argv, "--forward_seed", profile["forward_seed"])
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
    seed: int,
    skip_saved: bool,
) -> BackendCommand:
    model_name = SELECTOR_BASELINE_MODELS[method]
    argv = _base_python_argv("cli.run_model_zoo", python_bin)
    _add_kv(argv, "--run_mode", "select")
    _add_kv(argv, "--models", model_name)
    _add_kv(argv, "--current_zoo_num", stage)
    _add_kv(argv, "--zoo_total_num", stage)
    argv.extend(_selector_common_args(profile, seed, include_repr_args=method not in {"Random", "Recent"}))
    if method == "Random":
        _add_kv(argv, "--seed", seed)
    if method == "Task-probe":
        _add_kv(argv, "--sample_repr_num", profile["sample_repr_num"])
    if method == "Task-Oracle":
        _add_kv(argv, "--real_order_metric", "MASE")
    if method in {"Random", "Recent", "MetaFeature-GBDT", "MetaFeature-MLP", "Task-Oracle"}:
        argv.append("--real_world_mode")
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
            "seed": seed,
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
            "Profile-probe-C",
            "Random",
            "Recent",
            "Task-probe",
            "MetaFeature-GBDT",
            "MetaFeature-MLP",
            "Task-Oracle",
            "Current-best-M",
            "Current-best-C",
        ]
    aliases = {
        "autoforecast": "AutoForecast",
        "autoxpcr": "AutoXPCR",
        "simplets": "SimpleTS",
        "profile-probe-m": "Profile-probe-M",
        "profile_probe_m": "Profile-probe-M",
        "profile-probe-c": "Profile-probe-C",
        "profile_probe_c": "Profile-probe-C",
        "random": "Random",
        "recent": "Recent",
        "task-probe": "Task-probe",
        "task_probe": "Task-probe",
        "task_probe_forward": "Task-probe",
        "metafeature-gbdt": "MetaFeature-GBDT",
        "meta_gbdt": "MetaFeature-GBDT",
        "metafeature_gbdt": "MetaFeature-GBDT",
        "metafeature-mlp": "MetaFeature-MLP",
        "meta_mlp": "MetaFeature-MLP",
        "metafeature_mlp": "MetaFeature-MLP",
        "task-oracle": "Task-Oracle",
        "task_oracle": "Task-Oracle",
        "task_oracle_best": "Task-Oracle",
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
            raise LegacyPlanError(f"unknown baseline method: {key}")
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
) -> list[BackendCommand]:
    commands: list[BackendCommand] = []
    for method in _baseline_methods(args):
        if method not in BASELINE_PROFILE_OVERRIDES:
            continue
        profile = _profile_for_baseline(method)
        commands.append(
            _step3_command(
                profile,
                python_bin=python_bin,
                cwd=cwd,
                skip_saved=skip_saved,
            )
        )
        commands[-1].metadata["baseline_method"] = method
        commands.append(
            _step4_command(
                profile,
                python_bin=python_bin,
                cwd=cwd,
                skip_saved=skip_saved,
                stage=stage,
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
) -> list[BackendCommand]:
    profile = _main_profile()
    commands: list[BackendCommand] = []
    for method in _baseline_methods(args):
        if method not in SELECTOR_BASELINE_MODELS:
            continue
        seeds = [2025, 2026, 2027, 2028, 2029] if method == "Random" else [profile["search_seed"]]
        for seed in seeds:
            commands.append(
                _selector_baseline_command(
                    method,
                    profile,
                    python_bin=python_bin,
                    cwd=cwd,
                    stage=stage,
                    seed=int(seed),
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
) -> list[BackendCommand]:
    if "Task-probe" not in _baseline_methods(args):
        return []
    profile = _main_profile()
    profile["name"] = "Task-probe"
    profile["mix_route"] = True
    command = _step4_command(
        profile,
        python_bin=python_bin,
        cwd=cwd,
        skip_saved=skip_saved,
        stage=stage,
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


def _summary_command(profile: dict[str, Any], args: Any, *, python_bin: str, cwd: Path) -> BackendCommand:
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
    _add_kv(argv, "--repr_data_seed", profile["repr_data_seed"])
    _add_kv(argv, "--repr_encoder_seed", profile["repr_encoder_seed"])
    _add_kv(argv, "--forward_seed", profile["forward_seed"])
    _add_kv(argv, "--search_seed", profile["search_seed"])
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
        skip_saved=True,
    )


def build_release_command_plan(command: str, args: Any) -> dict[str, object]:
    profile = _main_profile()
    cwd = _legacy_root(getattr(args, "legacy_root", None))
    python_bin = str(getattr(args, "python_bin", "") or sys.executable)
    stage = int(getattr(args, "stage", 20) or 20)
    reuse = str(getattr(args, "reuse", "all") or "all")
    artifact_groups = tuple(COMMAND_ARTIFACT_GROUPS.get(command, ()))
    skip_saved = _reuse_skip_saved(reuse, artifact_groups)
    backend_commands: list[BackendCommand] = []

    if command == "tsfm":
        backend_commands.append(_tsfm_command(profile, python_bin=python_bin, cwd=cwd, skip_saved=skip_saved))
    elif command == "profile":
        backend_commands.append(_step1_command(profile, python_bin=python_bin, cwd=cwd, skip_saved=skip_saved))
        backend_commands.append(_step2_command(profile, python_bin=python_bin, cwd=cwd, skip_saved=skip_saved))
        for variant in _variants(getattr(args, "variant", "")):
            backend_commands.append(
                _step3_command(
                    _profile_for_variant(variant),
                    python_bin=python_bin,
                    cwd=cwd,
                    skip_saved=skip_saved,
                )
            )
    elif command == "route":
        for variant in _variants(getattr(args, "variant", "")):
            backend_commands.append(
                _step4_command(
                    _profile_for_variant(variant),
                    python_bin=python_bin,
                    cwd=cwd,
                    skip_saved=skip_saved,
                    stage=stage,
                )
            )
    elif command == "insert":
        backend_commands.extend(_insert_commands(profile, args, python_bin=python_bin, cwd=cwd, skip_saved=skip_saved))
    elif command == "baselines":
        backend_commands.extend(
            _route_style_baseline_commands(
                args,
                python_bin=python_bin,
                cwd=cwd,
                skip_saved=skip_saved,
                stage=stage,
            )
        )
        backend_commands.extend(
            _task_probe_mix_route_commands(
                args,
                python_bin=python_bin,
                cwd=cwd,
                skip_saved=skip_saved,
                stage=stage,
            )
        )
        backend_commands.extend(
            _selector_baseline_commands(
                args,
                python_bin=python_bin,
                cwd=cwd,
                skip_saved=skip_saved,
                stage=stage,
            )
        )
    elif command == "summary":
        backend_commands.append(_summary_command(profile, args, python_bin=python_bin, cwd=cwd))
    else:
        raise LegacyPlanError(f"unsupported public command: {command}")

    return {
        "command": command,
        "action": str(getattr(args, "action", "")),
        "stage": stage,
        "reuse": reuse,
        "artifact_groups": list(artifact_groups),
        "execution_mode": "execute" if bool(getattr(args, "execute", False)) else "plan",
        "python_bin": python_bin,
        "legacy_root": str(cwd),
        "backend_source": "configs/legacy_run_contract.yaml",
        "backend_commands": [item.as_dict() for item in backend_commands],
    }


def execute_release_command_plan(plan: dict[str, object]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for command in plan.get("backend_commands", []):
        if not isinstance(command, dict):
            continue
        argv = [str(item) for item in command.get("argv", [])]
        cwd = str(command.get("cwd") or ".")
        env = os.environ.copy()
        for key, value in dict(command.get("env", {}) or {}).items():
            env[str(key)] = str(value)
        src_path = str(Path(cwd) / "src")
        env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        proc = subprocess.run(argv, cwd=cwd, env=env, check=False)
        results.append(
            {
                "operation": command.get("operation", ""),
                "returncode": proc.returncode,
                "command_line": command.get("command_line", ""),
            }
        )
        if proc.returncode != 0:
            raise LegacyPlanError(f"backend command failed: {command.get('operation')} rc={proc.returncode}")
    return results

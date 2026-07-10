import os
import sys
import glob
import json
import pickle
import shutil
from typing import Iterable, Iterator, List, Dict, Tuple
from pathlib import Path
import re

from utils.project_paths import (
    BASELINE_ARTIFACT_ROOT,
    GE_TEST_SAMPLE_CACHE_ROOT,
    REPR_DATA_SOURCE_ROOT,
    SAMPLED_REPR_POOL_CACHE_ROOT,
    TSROUTER_ANCHOR_ROOT,
    TSROUTER_MODEL_REPR_ROOT,
    TSROUTER_REPR_FORWARD_CSV_ROOT,
    TSROUTER_SELECTOR_CSV_ROOT,
    TSROUTER_TRAINED_ENCODER_ROOT,
    TSFM_ARTIFACT_ROOT,
    TSFM_CSV_ROOT,
    rel,
)


TSROUTER_ROOT_DIR = "results_artifacts/TSRouter"
TSROUTER_REPR_DATA_SOURCE_DIR = rel(REPR_DATA_SOURCE_ROOT)
TSROUTER_SAMPLED_REPR_DIR = rel(TSROUTER_ANCHOR_ROOT)
TSROUTER_SAMPLED_REPR_POOL_DIR = rel(SAMPLED_REPR_POOL_CACHE_ROOT)
TSROUTER_MODEL_REPR_DIR = rel(TSROUTER_MODEL_REPR_ROOT)
TSROUTER_TASK_REPR_DIR = rel(GE_TEST_SAMPLE_CACHE_ROOT)
TSROUTER_REPR_FORWARD_DIR = rel(TSROUTER_REPR_FORWARD_CSV_ROOT)
TSROUTER_SELECTOR_RESULT_DIR = rel(TSROUTER_SELECTOR_CSV_ROOT)
TSROUTER_TRAINED_ENCODER_DIR = rel(TSROUTER_TRAINED_ENCODER_ROOT)
AUTOFORECAST_SELECTOR_ARTIFACT_DIR = rel(BASELINE_ARTIFACT_ROOT / "selectors" / "AutoForecast_Select")
AUTOXPCR_SELECTOR_ARTIFACT_DIR = rel(BASELINE_ARTIFACT_ROOT / "selectors" / "AutoXPCR_Select")
SIMPLETS_SELECTOR_ARTIFACT_DIR = rel(BASELINE_ARTIFACT_ROOT / "selectors" / "SimpleTS_Select")


RANDOM_ENCODER_STRUCTURES = ["MLP", "Patch", "Conv", "Inception", "TCN", "Fourier", "TS2Vec", "None"]
ENCODER_TYPES = ["Random", "StatsRandom", "RandomStats", "Train", "SimpleTS", "None"]
ADVANCED_BASELINE_TRAIN_SCOPES = {"center", "full_pool"}


def _path_with_parts(base: Path, parts: tuple[object, ...]) -> Path:
    out = base
    for part in parts:
        if part in (None, ""):
            continue
        out = out / str(part)
    return out


def tsfm_result_dir(
    root: str | os.PathLike[str],
    model_key: str,
    model_cl_name: str,
    *,
    layout: str = "new",
) -> Path:
    root_path = Path(root)
    if str(layout).lower() in {"old", "legacy"}:
        return root_path / str(model_key) / str(model_cl_name)
    return root_path / str(model_cl_name) / str(model_key)


def tsfm_result_path_candidates(
    root: str | os.PathLike[str],
    model_key: str,
    model_cl_name: str,
    *relative_parts: object,
) -> list[Path]:
    return [
        _path_with_parts(tsfm_result_dir(root, model_key, model_cl_name, layout="new"), relative_parts),
        _path_with_parts(tsfm_result_dir(root, model_key, model_cl_name, layout="old"), relative_parts),
    ]


def resolve_tsfm_result_path(
    root: str | os.PathLike[str],
    model_key: str,
    model_cl_name: str,
    *relative_parts: object,
) -> Path:
    candidates = tsfm_result_path_candidates(root, model_key, model_cl_name, *relative_parts)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def tsfm_csv_dir(
    model_key: str,
    model_cl_name: str,
    *,
    root: str | os.PathLike[str] = TSFM_CSV_ROOT,
    create: bool = False,
) -> Path:
    path = tsfm_result_dir(root, model_key, model_cl_name, layout="new")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def tsfm_artifact_dir(
    model_key: str,
    model_cl_name: str,
    *,
    root: str | os.PathLike[str] = TSFM_ARTIFACT_ROOT,
    create: bool = False,
) -> Path:
    path = tsfm_result_dir(root, model_key, model_cl_name, layout="new")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_tsfm_csv_path(model_key: str, model_cl_name: str, *relative_parts: object) -> Path:
    return resolve_tsfm_result_path(TSFM_CSV_ROOT, model_key, model_cl_name, *relative_parts)


def resolve_tsfm_artifact_path(model_key: str, model_cl_name: str, *relative_parts: object) -> Path:
    return resolve_tsfm_result_path(TSFM_ARTIFACT_ROOT, model_key, model_cl_name, *relative_parts)


def tsfm_csv_glob_display(model_cl_name: str, *, root: str | os.PathLike[str] = TSFM_CSV_ROOT) -> str:
    return f"{Path(root).as_posix()}/{model_cl_name}/*/"


AUTO_CL_DISABLED_MODES = {"", "0", "false", "no", "off", "none", "v0"}
AUTO_CL_V1_PROFILES = (
    {
        "adaptive_profile": "acl_len144",
        "profile_key": "short",
        "repr_input_dim": 96,
        "repr_output_dim": 128,
        "repr_sub_pred_len": 48,
        "repr_source_exact_length": 144,
        "tsfm_context_len": 96,
        "tsfm_results_dir": "cl_96",
    },
    {
        "adaptive_profile": "acl_len992",
        "profile_key": "middle",
        "repr_input_dim": 512,
        "repr_output_dim": 256,
        "repr_sub_pred_len": 480,
        "repr_source_exact_length": 992,
        "tsfm_context_len": 512,
        "tsfm_results_dir": "cl_512",
    },
    {
        "adaptive_profile": "acl_len3000",
        "profile_key": "long",
        "repr_input_dim": 2048,
        "repr_output_dim": 512,
        "repr_sub_pred_len": 720,
        "repr_source_exact_length": 3000,
        "tsfm_context_len": 2048,
        "tsfm_results_dir": "cl_2048",
    },
)
AUTO_CL_V2_PROFILES = (
    {
        "adaptive_profile": "acl_len108",
        "profile_key": "short",
        "repr_input_dim": 96,
        "repr_output_dim": 128,
        "repr_sub_pred_len": 12,
        "repr_source_exact_length": 108,
        "tsfm_context_len": 96,
        "tsfm_results_dir": "cl_96",
    },
    {
        "adaptive_profile": "acl_len542",
        "profile_key": "middle",
        "repr_input_dim": 512,
        "repr_output_dim": 256,
        "repr_sub_pred_len": 30,
        "repr_source_exact_length": 542,
        "tsfm_context_len": 512,
        "tsfm_results_dir": "cl_512",
    },
    {
        "adaptive_profile": "acl_len2768",
        "profile_key": "long",
        "repr_input_dim": 2048,
        "repr_output_dim": 512,
        "repr_sub_pred_len": 720,
        "repr_source_exact_length": 2768,
        "tsfm_context_len": 2048,
        "tsfm_results_dir": "cl_2048",
    },
)
AUTO_CL_V3_PROFILES = (
    {
        "adaptive_profile": "acl_len108",
        "profile_key": "short",
        "repr_input_dim": 96,
        "repr_output_dim": 128,
        "repr_sub_pred_len": 12,
        "repr_source_exact_length": 108,
        "tsfm_context_len": 96,
        "tsfm_results_dir": "cl_96",
    },
    {
        "adaptive_profile": "acl_len992",
        "profile_key": "middle",
        "repr_input_dim": 512,
        "repr_output_dim": 256,
        "repr_sub_pred_len": 480,
        "repr_source_exact_length": 992,
        "tsfm_context_len": 512,
        "tsfm_results_dir": "cl_512",
    },
)
AUTO_CL_PROFILES_BY_MODE = {
    "v1": AUTO_CL_V1_PROFILES,
    "v2": AUTO_CL_V2_PROFILES,
    "v3": AUTO_CL_V3_PROFILES,
}

FIXED_CL_V0_PROFILES = (
    {
        "repr_input_dim": 96,
        "repr_output_dim": 128,
        "repr_sub_pred_len": 48,
        "repr_source_exact_length": 144,
        "tsfm_context_len": 96,
        "tsfm_results_dir": "cl_96",
    },
    {
        "repr_input_dim": 512,
        "repr_output_dim": 256,
        "repr_sub_pred_len": 480,
        "repr_source_exact_length": 992,
        "tsfm_context_len": 512,
        "tsfm_results_dir": "cl_512",
    },
    {
        "repr_input_dim": 2048,
        "repr_output_dim": 512,
        "repr_sub_pred_len": 480,
        "repr_source_exact_length": 2528,
        "tsfm_context_len": 2048,
        "tsfm_results_dir": "cl_2048",
    },
)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t", "on"}


def normalize_auto_cl_mode_value(value, legacy_enabled: bool = False) -> str:
    raw = str(value if value is not None else "").strip().lower()
    if raw in AUTO_CL_DISABLED_MODES:
        return "v1" if legacy_enabled else "v0"
    if raw in {"1", "true", "yes", "y", "t", "on"}:
        return "v1"
    if raw in AUTO_CL_PROFILES_BY_MODE:
        return raw
    raise ValueError(
        f"Unknown auto_cl_mode={value!r}; use v0, "
        f"{', '.join(sorted(AUTO_CL_PROFILES_BY_MODE))}"
    )


def get_auto_cl_mode(args) -> str:
    legacy_enabled = _as_bool(getattr(args, "enable_context_len_adaptive_repr", False))
    return normalize_auto_cl_mode_value(getattr(args, "auto_cl_mode", "v0"), legacy_enabled=legacy_enabled)


def auto_cl_enabled(args) -> bool:
    return get_auto_cl_mode(args) != "v0"


def get_auto_cl_profiles(mode_or_args="v1") -> tuple[dict, ...]:
    if hasattr(mode_or_args, "__dict__"):
        mode = get_auto_cl_mode(mode_or_args)
    else:
        mode = normalize_auto_cl_mode_value(mode_or_args)
    profiles = AUTO_CL_PROFILES_BY_MODE.get(mode)
    if profiles is None:
        return ()
    return tuple(dict(profile) for profile in profiles)


def auto_cl_tsfm_comparison_dir(mode_or_args="v1") -> str:
    profiles = get_auto_cl_profiles(mode_or_args)
    if not profiles:
        return "cl_512"
    return str(profiles[-1]["tsfm_results_dir"])


def auto_cl_rank_truth_cls(mode_or_args="v1") -> str:
    tokens = []
    for profile in get_auto_cl_profiles(mode_or_args):
        token = str(profile["tsfm_results_dir"]).replace("_", "")
        if token not in tokens:
            tokens.append(token)
    return " ".join(tokens)


def get_auto_cl_profile_for_dims(
    mode_or_args,
    repr_input_dim: int,
    repr_sub_pred_len: int,
) -> dict | None:
    for profile in get_auto_cl_profiles(mode_or_args):
        if (
            int(profile["repr_input_dim"]) == int(repr_input_dim)
            and int(profile["repr_sub_pred_len"]) == int(repr_sub_pred_len)
        ):
            return dict(profile)
    return None


def get_auto_cl_profile_by_name(
    adaptive_profile: str,
    mode_or_args=None,
) -> dict | None:
    target = str(adaptive_profile or "").strip()
    if mode_or_args is None:
        modes = sorted(AUTO_CL_PROFILES_BY_MODE)
    else:
        mode = get_auto_cl_mode(mode_or_args) if hasattr(mode_or_args, "__dict__") else normalize_auto_cl_mode_value(mode_or_args)
        modes = [mode]
    for mode in modes:
        for profile in get_auto_cl_profiles(mode):
            if str(profile["adaptive_profile"]) == target:
                return dict(profile)
    return None


def get_auto_cl_v1_profile_for_dims(repr_input_dim: int, repr_sub_pred_len: int) -> dict | None:
    return get_auto_cl_profile_for_dims("v1", repr_input_dim, repr_sub_pred_len)


def get_auto_cl_v1_profile_by_name(adaptive_profile: str) -> dict | None:
    return get_auto_cl_profile_by_name(adaptive_profile, "v1")


def get_fixed_cl_profile(tsfm_results_dir: str) -> dict | None:
    target = str(tsfm_results_dir or "").strip().lower().replace("-", "_")
    target = re.sub(r"^cl_?", "", target)
    if not target.isdigit():
        return None
    context_len = int(target)
    for profile in FIXED_CL_V0_PROFILES:
        if int(profile["tsfm_context_len"]) == context_len:
            return dict(profile)
    return None


def resolve_auto_cl_v1_profile(
    context_len_avg: float | None,
    task_term: str | None = None,
) -> tuple[dict, bool, float]:
    return resolve_auto_cl_profile(context_len_avg, task_term, mode_or_args="v1")


def resolve_auto_cl_profile(
    context_len_avg: float | None,
    task_term: str | None = None,
    mode_or_args="v1",
) -> tuple[dict, bool, float]:
    """
    Resolve the fixed auto-cl profile for the requested mode.

    The returned boolean records whether task-term fallback was needed because
    no usable context-length average was available.
    """
    profiles = get_auto_cl_profiles(mode_or_args)
    if len(profiles) not in {2, 3}:
        mode = get_auto_cl_mode(mode_or_args) if hasattr(mode_or_args, "__dict__") else normalize_auto_cl_mode_value(mode_or_args)
        raise ValueError(f"auto_cl_mode={mode} must define two or three profiles")
    fallback_used = context_len_avg is None
    if fallback_used:
        term = str(task_term or "").strip().lower()
        if term == "long":
            effective_context_len = 512.0
        elif term == "medium":
            effective_context_len = 128.0
        else:
            effective_context_len = 96.0
    else:
        effective_context_len = float(context_len_avg)

    if len(profiles) == 2:
        profile = profiles[1] if effective_context_len >= 128 else profiles[0]
    elif effective_context_len >= 512:
        profile = profiles[2]
    elif effective_context_len >= 128:
        profile = profiles[1]
    else:
        profile = profiles[0]
    return dict(profile), fallback_used, effective_context_len


def get_auto_cl_profile(args) -> dict | None:
    mode = get_auto_cl_mode(args)
    if mode == "v0":
        return None
    profile = get_auto_cl_profile_for_dims(
        mode,
        int(getattr(args, "repr_input_dim", 0)),
        int(getattr(args, "repr_sub_pred_len", 0)),
    )
    if profile is None:
        return None
    source_len = getattr(args, "repr_source_exact_length", None)
    if source_len is not None:
        profile["repr_source_exact_length"] = int(source_len)
        profile["adaptive_profile"] = f"acl_len{int(source_len)}"
    return profile


def get_auto_cl_profile_name(args) -> str:
    profile = get_auto_cl_profile(args)
    if profile is None:
        return "default"
    return str(profile["adaptive_profile"])


def normalize_auto_cl_args(args):
    """Normalize the legacy bool into auto_cl_mode and fix concrete profile dims."""
    mode = get_auto_cl_mode(args)
    args.auto_cl_mode = mode
    args.enable_context_len_adaptive_repr = mode != "v0"
    if mode == "v0":
        return args

    profiles = get_auto_cl_profiles(mode)
    profiles_by_key = {
        str(profile["profile_key"]): profile
        for profile in profiles
    }
    for profile_key, prefix in [("short", "short"), ("middle", "middle"), ("long", "long")]:
        profile = profiles_by_key.get(profile_key)
        if profile is None:
            profile = profiles[-1]
        setattr(args, f"{prefix}_repr_input_dim", int(profile["repr_input_dim"]))
        setattr(args, f"{prefix}_repr_sub_pred_len", int(profile["repr_sub_pred_len"]))
        setattr(args, f"{prefix}_repr_output_dim", int(profile["repr_output_dim"]))
        setattr(args, f"{prefix}_repr_source_len", int(profile["repr_source_exact_length"]))

    current_input_dim = int(getattr(args, "repr_input_dim", 0))
    profile = next(
        (
            dict(profile_cfg)
            for profile_cfg in profiles
            if int(profile_cfg["repr_input_dim"]) == current_input_dim
        ),
        None,
    )
    if profile is None:
        profile = get_auto_cl_profile(args)
    if profile is not None:
        args.repr_sub_pred_len = int(profile["repr_sub_pred_len"])
        args.repr_output_dim = int(profile["repr_output_dim"])
        args.repr_source_exact_length = int(profile["repr_source_exact_length"])
    return args


def route_efficiency_mode_enabled(args) -> bool:
    raw = getattr(args, "route_efficiency_mode", False)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"true", "1", "yes", "y", "t"}


def normalize_advanced_baseline_train_scope(value) -> str:
    raw = str(value or "center").strip().lower().replace("-", "_")
    aliases = {
        "anchor": "center",
        "anchors": "center",
        "cluster_center": "center",
        "cluster_centers": "center",
        "centers": "center",
        "pool": "full_pool",
        "full": "full_pool",
        "fullpool": "full_pool",
        "full_pool": "full_pool",
        "all_pool": "full_pool",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in ADVANCED_BASELINE_TRAIN_SCOPES:
        raise ValueError(
            "advanced_baseline_train_scope must be one of "
            f"{sorted(ADVANCED_BASELINE_TRAIN_SCOPES)}, got {value!r}"
        )
    return normalized


def get_advanced_baseline_train_scope(args) -> str:
    return normalize_advanced_baseline_train_scope(
        getattr(args, "advanced_baseline_train_scope", "center")
    )


def get_advanced_baseline_train_scope_tag(args) -> str:
    scope = get_advanced_baseline_train_scope(args)
    return "_abpool" if scope == "full_pool" else ""


ROUTE_FAMILY_MODES = ("default", "bigger_size", "smaller_size")


def normalize_route_family_mode(value) -> str:
    mode = str(value or "default").strip().lower().replace("-", "_")
    if mode not in ROUTE_FAMILY_MODES:
        raise ValueError(
            f"Unknown route_family_mode={value!r}; use {', '.join(ROUTE_FAMILY_MODES)}"
        )
    return mode


def get_route_family_tag(args) -> str:
    mode = normalize_route_family_mode(getattr(args, "route_family_mode", "default"))
    return {
        "default": "",
        "bigger_size": "_rfbigger",
        "smaller_size": "_rfsmaller",
    }[mode]


def encoder_name_from_type_structure(encoder_type: str, encoder_structure: str) -> str:
    encoder_type = str(encoder_type or "Random").strip()
    structure = str(encoder_structure or "MLP").strip()
    valid_structures = {s.lower(): s for s in RANDOM_ENCODER_STRUCTURES}
    structure_key = structure.lower()
    if structure_key not in valid_structures:
        raise ValueError(f"Unknown encoder_structure={encoder_structure!r}; use {RANDOM_ENCODER_STRUCTURES}")
    structure = valid_structures[structure_key]
    valid_types = {t.lower(): t for t in ENCODER_TYPES}
    type_key = encoder_type.lower()
    if type_key not in valid_types:
        raise ValueError(f"Unknown encoder_type={encoder_type!r}; use {ENCODER_TYPES}")
    encoder_type = valid_types[type_key]
    if encoder_type == "None":
        return "None"
    if encoder_type == "SimpleTS":
        if structure != "TS2Vec":
            raise ValueError("encoder_type=SimpleTS only supports encoder_structure=TS2Vec")
        return "SimpleTS2Vec"
    if structure == "TS2Vec":
        return "TrainTS2Vec" if encoder_type == "Train" else "TS2Vec"
    if structure == "None" and encoder_type == "StatsRandom":
        return "StatsNone"
    if structure == "None" or encoder_type == "None":
        return "None"
    if encoder_type == "Random":
        return f"Random{structure}"
    if encoder_type == "StatsRandom":
        return f"StatsRandom{structure}"
    if encoder_type == "RandomStats":
        return f"RandomStats{structure}"
    return f"Train{structure}"


def infer_type_structure_from_encoder_name(repr_encoder: str) -> tuple[str, str]:
    name = str(repr_encoder or "RandomMLP").strip()
    if name.lower() == "none":
        return "None", "None"
    if name == "StatsNone":
        return "StatsRandom", "None"
    if name == "TS2Vec":
        return "Random", "TS2Vec"
    if name == "TrainTS2Vec":
        return "Train", "TS2Vec"
    if name == "SimpleTS2Vec":
        return "SimpleTS", "TS2Vec"
    for structure in sorted(RANDOM_ENCODER_STRUCTURES, key=len, reverse=True):
        if name == f"Random{structure}":
            return "Random", structure
        if name == f"StatsRandom{structure}":
            return "StatsRandom", structure
        if name == f"RandomStats{structure}":
            return "RandomStats", structure
        if name == f"Train{structure}":
            return "Train", structure
    return None, None


def normalize_encoder_variant_args(args):
    """Fill encoder_type/encoder_structure and set repr_encoder to the canonical effective name."""
    encoder_type = getattr(args, "encoder_type", None)
    encoder_structure = getattr(args, "encoder_structure", None)
    if encoder_type is None and encoder_structure is None:
        encoder_type, encoder_structure = infer_type_structure_from_encoder_name(getattr(args, "repr_encoder", "RandomMLP"))
        if encoder_type is None:
            return args
    elif encoder_type is None:
        encoder_type = "Random"
    elif encoder_structure is None:
        _, encoder_structure = infer_type_structure_from_encoder_name(getattr(args, "repr_encoder", "RandomMLP"))
    encoder_name = encoder_name_from_type_structure(encoder_type, encoder_structure)
    if encoder_name == "None":
        encoder_type = "None"
        encoder_structure = "None"
        input_dim = int(getattr(args, "repr_input_dim", 512))
        args.repr_output_dim = input_dim
    elif encoder_structure == "None":
        input_dim = int(getattr(args, "repr_input_dim", 512))
        args.repr_output_dim = input_dim
    args.encoder_type = encoder_type
    args.encoder_structure = encoder_structure
    args.repr_encoder = encoder_name
    return args


def make_train_bootstrap_args(args):
    import copy
    boot = copy.copy(args)
    if str(getattr(args, "encoder_structure", "")).lower() == "none":
        boot.encoder_type = "None"
        boot.encoder_structure = "None"
        boot.repr_encoder = "None"
        boot.repr_output_dim = int(getattr(args, "repr_input_dim", 512))
        return boot
    boot.encoder_type = str(getattr(args, "train_bootstrap_encoder_type", "Random"))
    boot.encoder_structure = str(getattr(args, "encoder_structure", "MLP"))
    boot.repr_encoder = encoder_name_from_type_structure(boot.encoder_type, boot.encoder_structure)
    return boot


def normalize_repr_scale_protocol(value) -> str:
    protocol = str(value or "raw").strip().lower()
    aliases = {
        "none": "raw",
        "origin": "raw",
        "original": "raw",
        "std": "standard",
        "zscore": "standard",
        "standardized": "standard",
        "standardscale": "standard",
        "standardscaler": "standard",
    }
    protocol = aliases.get(protocol, protocol)
    if protocol not in {"raw", "standard"}:
        raise ValueError(f"Unknown repr_scale_protocol={value!r}; use raw or standard")
    return protocol


def get_repr_scale_protocol(args) -> str:
    return normalize_repr_scale_protocol(getattr(args, "repr_scale_protocol", "raw"))


def get_repr_scale_tag(args) -> str:
    return "raw" if get_repr_scale_protocol(args) == "raw" else "std"


_WINDOW_SAMPLE_STRATEGIES = {"legacy", "even", "random", "first", "last"}


def normalize_window_sample_strategy(value, *, default: str = "legacy") -> str:
    strategy = str(value if value not in (None, "") else default).strip().lower()
    if strategy not in _WINDOW_SAMPLE_STRATEGIES:
        allowed = ", ".join(sorted(_WINDOW_SAMPLE_STRATEGIES))
        raise ValueError(f"Unknown window sample strategy={value!r}; use one of: {allowed}")
    return strategy


def get_effective_task_window_sample_strategy(args) -> tuple[str, float, bool]:
    """Return effective Step4 window-sampling strategy, ratio, and whether to add a filename tag."""
    strategy = normalize_window_sample_strategy(
        getattr(args, "task_window_sample_strategy", "legacy"),
        default="legacy",
    )
    ratio = float(getattr(args, "sample_repr_ratio", 0.0) or 0.0)
    if ratio > 0 and strategy == "legacy":
        strategy = "even"
    should_tag = strategy != "legacy" or ratio > 0
    return strategy, ratio, should_tag


def get_repr_anchor_window_sample_strategy(args) -> str:
    """
    Step1/2 anchor-window sampling strategy.

    New callers should pass ``repr_anchor_window_sample_strategy`` so Step1/2
    artifact identity is decoupled from Step4 task-window sampling.  For
    backward compatibility, older callers that only pass
    ``task_window_sample_strategy`` still drive Step1 the old way.
    """
    raw = getattr(args, "repr_anchor_window_sample_strategy", None)
    if raw in (None, ""):
        if hasattr(args, "task_window_sample_strategy"):
            raw = getattr(args, "task_window_sample_strategy", "legacy")
        else:
            raw = "even"
    return normalize_window_sample_strategy(raw, default="even")


def get_effective_repr_anchor_window_sample_strategy(args) -> tuple[str, float, bool]:
    """Return effective Step1/2 window strategy, ratio, and whether to tag window-mode stems."""
    strategy = get_repr_anchor_window_sample_strategy(args)
    ratio = float(getattr(args, "sample_repr_ratio", 0.0) or 0.0)
    if ratio > 0 and strategy == "legacy":
        strategy = "even"
    should_tag = strategy != "even" or ratio > 0
    return strategy, ratio, should_tag


def build_repr_anchor_sample_tag(args) -> str:
    """
    Filename tag for Step1/2 anchor sampling choices.

    In the default window protocol, ``even`` intentionally keeps the legacy
    untagged name.  Other Step1 window strategies can change anchor values and
    must be visible in Step1/2/3 stems.
    """
    repr_anchor_protocol = str(getattr(args, "repr_anchor_protocol", "window")).lower()
    task_strategy = str(getattr(args, "task_sample_strategy", "latest_random")).lower()
    window_strategy, ratio, tag_window_strategy = get_effective_repr_anchor_window_sample_strategy(args)

    if repr_anchor_protocol != "window":
        sample_tag = (
            f"_ra{repr_anchor_protocol}-{task_strategy}"
            f"_n{int(getattr(args, 'sample_repr_num', 0))}"
        )
        if window_strategy != "legacy" or ratio > 0:
            sample_tag += f"_ws{window_strategy}"
        if ratio > 0:
            sample_tag += f"_sr{ratio:g}"
        return sample_tag

    if not tag_window_strategy:
        return ""
    sample_tag = f"_aws{window_strategy}"
    if ratio > 0:
        sample_tag += f"_sr{ratio:g}"
    return sample_tag


def get_task_channel_fuse_tag(args) -> str:
    value = str(getattr(args, "task_channel_fuse_limit", "all") or "all").strip().lower()
    if value in {"", "all", "none"}:
        return ""
    try:
        n = int(value)
    except ValueError as exc:
        raise ValueError(f"Unknown task_channel_fuse_limit={value!r}; use all or a positive integer") from exc
    if n <= 0:
        raise ValueError(f"task_channel_fuse_limit must be positive when not all, got {value!r}")
    return f"_cf{n}"


def get_tsrouter_repr_forward_dir(args=None) -> str:
    return str(getattr(args, "repr_forward_dir", TSROUTER_REPR_FORWARD_DIR))


def get_tsrouter_selector_result_dir(args=None) -> str:
    return str(getattr(args, "tsrouter_selector_result_dir", TSROUTER_SELECTOR_RESULT_DIR))


def get_tsrouter_selector_stage_result_dir(args=None) -> str:
    root = get_tsrouter_selector_result_dir(args)
    try:
        stage = int(getattr(args, "current_zoo_num", 0) or 0)
    except Exception:
        stage = 0
    if stage <= 0:
        return root
    return os.path.join(root, f"stage{stage}")


def ensure_tsrouter_dirs(args=None) -> None:
    for path in [
        str(getattr(args, "save_repr_data_path", TSROUTER_SAMPLED_REPR_DIR)),
        TSROUTER_REPR_DATA_SOURCE_DIR,
        str(getattr(args, "save_model_repr_path", TSROUTER_MODEL_REPR_DIR)),
        get_tsrouter_repr_forward_dir(args),
        str(getattr(args, "gift_eval_task_repr_dir", TSROUTER_TASK_REPR_DIR)),
        get_tsrouter_selector_result_dir(args),
        get_tsrouter_selector_stage_result_dir(args),
        str(getattr(args, "trained_encoder_dir", TSROUTER_TRAINED_ENCODER_DIR)),
    ]:
        os.makedirs(path, exist_ok=True)


def candidate_sample_dirs(primary: str | None = None) -> List[str]:
    dirs: List[str] = []
    for d in [
        primary,
        TSROUTER_REPR_DATA_SOURCE_DIR,
        TSROUTER_SAMPLED_REPR_DIR,
        TSROUTER_SAMPLED_REPR_POOL_DIR,
    ]:
        if d and d not in dirs:
            dirs.append(d)
    return dirs




_ZOO_MODEL_ALIAS = {
    "c": "chronos",
    "chronos": "chronos",

    "m": "moirai",
    "moirai": "moirai",

    "t": "timesfm",
    "timesfm": "timesfm",

    "v": "visionts",
    "visionts": "visionts",

    "l": "lotsa",
    "lotsa": "lotsa",

    "e": "domain_energy",
    "energy": "domain_energy",
    "domain_energy": "domain_energy",

    "f": "domain_econ_fin",
    "econ_fin": "domain_econ_fin",
    "domain_econ_fin": "domain_econ_fin",

    "h": "domain_healthcare",
    "healthcare": "domain_healthcare",
    "domain_healthcare": "domain_healthcare",

    "n": "domain_nature",
    "nature": "domain_nature",
    "domain_nature": "domain_nature",

    "s": "domain_sales",
    "sales": "domain_sales",
    "domain_sales": "domain_sales",

    "w": "domain_web_cloudops",
    "web": "domain_web_cloudops",
    "web_cloudops": "domain_web_cloudops",
    "domain_web_cloudops": "domain_web_cloudops",

    "tr": "domain_transport",
    "transport": "domain_transport",
    "domain_transport": "domain_transport",

    "o": "domain_oracle",
    "oracle": "domain_oracle",
    "domain_oracle": "domain_oracle",

    "os": "oracle_sample",
    "oracle_sample": "oracle_sample",
    "sample_oracle": "oracle_sample",
}

_ALL_REPR_SOURCE_TOKENS = {"all", "all_source", "all_sources"}
_BASE_REPR_SOURCE_NAMES = ["chronos", "moirai", "timesfm", "visionts", "lotsa"]


def _discover_domain_aliases(sample_dir: str = TSROUTER_SAMPLED_REPR_DIR) -> Dict[str, str]:
    'TSRouter runtime message.'
                                  
    domain_to_name: Dict[str, str] = {}
    candidate_dirs: List[str] = []
    for base_dir in candidate_sample_dirs(sample_dir):
        candidate_dirs.extend([base_dir, os.path.join(base_dir, "domain_sample")])
    for cur_dir in candidate_dirs:
        if not os.path.isdir(cur_dir):
            continue
        for fn in os.listdir(cur_dir):
            if not fn.endswith(".npy"):
                continue
            m = re.match(r"^([A-Za-z0-9_-]+)_num\d+.*_len\d+.*\.npy$", fn)
            if m is None:
                continue
            domain_raw = m.group(1).lower()
            canonical = f"domain_{domain_raw}"
            domain_to_name[domain_raw] = canonical

    if not domain_to_name:
        return {}

    out: Dict[str, str] = {}
              
    for dom, canonical in domain_to_name.items():
        out[canonical] = canonical
        out[dom] = canonical


               
                    
                  
    if "econ_fin" in domain_to_name:
        out["f"] = domain_to_name["econ_fin"]
    if "energy" in domain_to_name:
        out["e"] = domain_to_name["energy"]

                           
    first_char_count: Dict[str, int] = {}
    for dom in domain_to_name.keys():
        c = dom[0]
        first_char_count[c] = first_char_count.get(c, 0) + 1

    for dom, canonical in domain_to_name.items():
        c = dom[0]
        if first_char_count.get(c, 0) == 1 and c not in _ZOO_MODEL_ALIAS:
            out[c] = canonical

    return out


def _all_available_repr_source_names(domain_aliases: Dict[str, str]) -> List[str]:
    names = list(_BASE_REPR_SOURCE_NAMES)
    discovered_domains = sorted(
        {
            canonical
            for canonical in domain_aliases.values()
            if str(canonical).startswith("domain_")
        }
    )
    for name in discovered_domains:
        if name not in names:
            names.append(name)
    return names


def _preferred_alias_for_name(name: str, alias_map: Dict[str, str]) -> str:
    'TSRouter runtime message.'
              
    fixed = {
        "chronos": "c",
        "moirai": "m",
        "timesfm": "t",
        "visionts": "v",
        "lotsa": "l",
        "domain_energy": "e",
        "domain_econ_fin": "f",
        "domain_healthcare": "h",
        "domain_nature": "n",
        "domain_sales": "s",
        "domain_web_cloudops": "w",
        "domain_transport": "tr",
        "domain_oracle": "o",
        "oracle_sample": "os",
    }
    if name in fixed:
        return fixed[name]

                         
    one_char = [k for k, v in alias_map.items() if v == name and len(k) == 1 and k.isalpha()]
    if one_char:
        return sorted(one_char)[0]

                                 
    if name.startswith("domain_"):
        tail = name[len("domain_"):]
        return tail if tail else "d"
    return name[0]

def parse_zoo_repr_set(zoo_repr_set):
    'TSRouter runtime message.'
    if zoo_repr_set is None:
        return []

                                
    if isinstance(zoo_repr_set, str):
        tokens = [zoo_repr_set]
    else:
        tokens = list(zoo_repr_set)

    domain_aliases = _discover_domain_aliases()
    alias_map = dict(_ZOO_MODEL_ALIAS)
    alias_map.update(domain_aliases)
    all_source_names = _all_available_repr_source_names(domain_aliases)

    parsed: List[str] = []
    for token in tokens:
                                              
        subtokens = str(token).lower().split("-")

        for sub in subtokens:
            sub = sub.strip()
            if not sub:
                continue
            if sub in _ALL_REPR_SOURCE_TOKENS:
                for name in all_source_names:
                    if name not in parsed:
                        parsed.append(name)
                continue
            if sub not in alias_map:
                raise ValueError(
                    f"Unknown zoo_repr_set token: {sub} (from '{token}'). "
                    f"Supported: {sorted(set(alias_map.keys()) | _ALL_REPR_SOURCE_TOKENS)}"
                )
            name = alias_map[sub]
                     
            if name not in parsed:
                parsed.append(name)

    return parsed

def get_zoo_repr_prefix(zoo_repr_set) -> str:
    'TSRouter runtime message.'
    if zoo_repr_set is None:
        return "none"

    if isinstance(zoo_repr_set, str):
        tokens = [zoo_repr_set]
    else:
        tokens = list(zoo_repr_set)

    domain_aliases = _discover_domain_aliases()
    alias_map = dict(_ZOO_MODEL_ALIAS)
    alias_map.update(domain_aliases)
    all_source_names = _all_available_repr_source_names(domain_aliases)

    ordered_names: List[str] = []
    ordered_alias: List[str] = []
    for token in tokens:
        subtokens = str(token).lower().split("-")
        for sub in subtokens:
            sub = sub.strip()
            if not sub:
                continue
            if sub in _ALL_REPR_SOURCE_TOKENS:
                for name in all_source_names:
                    if name not in ordered_names:
                        ordered_names.append(name)
                        ordered_alias.append(_preferred_alias_for_name(name, alias_map))
                continue
            if sub not in alias_map:
                raise ValueError(
                    f"Unknown zoo_repr_set token: {sub} (from '{token}'). "
                    f"Supported: {sorted(set(alias_map.keys()) | _ALL_REPR_SOURCE_TOKENS)}"
                )
            name = alias_map[sub]
            if name not in ordered_names:
                ordered_names.append(name)
                ordered_alias.append(_preferred_alias_for_name(name, alias_map))

    return "-".join(ordered_alias) if ordered_alias else "none"


def _phase_seed_default(args) -> int:
    return 2025


def _resolve_seed(args, attr_name: str) -> int:
    value = getattr(args, attr_name, None)
    if value is None:
        value = _phase_seed_default(args)
    return int(value)


def get_repr_seed_tag(args) -> str:
    repr_data_seed = _resolve_seed(args, "repr_data_seed")
    repr_encoder_seed = _resolve_seed(args, "repr_encoder_seed")
    return f"sd{repr_data_seed}_se{repr_encoder_seed}"


def get_selector_seed_tag(args) -> str:
    tag=f"ss{_resolve_seed(args, 'search_seed')}"
    if args.enable_search_ensemble:
        tag+='_en5'
    return tag

def get_forward_seed_tag(args) -> str:
    return f"sf{_resolve_seed(args, 'forward_seed')}"


def build_repr_source_size_tag(prefix: str, size: int) -> str:
    """Return the Windows-safe source/size token used in repr artifact names."""
    return f"{prefix}_x{int(size)}"


def get_auto_cl_artifact_tag(args) -> str:
    if not auto_cl_enabled(args):
        return ""
    source_len = getattr(args, "repr_source_exact_length", None)
    if source_len is None:
        profile = get_auto_cl_profile(args)
        if profile is not None:
            source_len = int(profile["repr_source_exact_length"])
        else:
            repr_input_dim = int(getattr(args, "repr_input_dim", 0))
            repr_sub_pred_len = int(getattr(args, "repr_sub_pred_len", 0))
            long_input_dim = int(getattr(args, "long_repr_input_dim", 2048))
            long_pred_len = int(getattr(args, "long_repr_sub_pred_len", 720))
            short_input_dim = int(getattr(args, "short_repr_input_dim", 96))
            short_pred_len = int(getattr(args, "short_repr_sub_pred_len", 48))
            if repr_input_dim == long_input_dim and repr_sub_pred_len == long_pred_len:
                source_len = int(getattr(args, "long_repr_source_len", 3000))
            elif repr_input_dim == short_input_dim and repr_sub_pred_len == short_pred_len:
                source_len = int(getattr(args, "short_repr_source_len", 144))
            else:
                source_len = int(getattr(args, "middle_repr_source_len", 992))
    return f"_acl_len{int(source_len)}"


def build_repr_set_name(args) -> str:
    'TSRouter runtime message.'
    prefix = get_zoo_repr_prefix(getattr(args, "zoo_repr_set", None))

    size = int(getattr(args, "repr_size", 0))
    source_size_tag = build_repr_source_size_tag(prefix, size)

    seed_tag = get_repr_seed_tag(args)
    scale_tag = get_repr_scale_tag(args)
    dim_tag = f"_{int(getattr(args, 'repr_input_dim', 0))}to{int(getattr(args, 'repr_output_dim', 0))}_pl{int(getattr(args, 'repr_sub_pred_len', 0))}"
    checkpoint_tag = ""
    if str(getattr(args, "repr_encoder", "")) == "SimpleTS2Vec":
        fingerprint = str(getattr(args, "simplets_ts2vec_checkpoint_fingerprint", "") or "")
        if fingerprint:
            checkpoint_tag = f"_ck{fingerprint[:12]}"
    name = f"{args.repr_encoder}{checkpoint_tag}{dim_tag}{get_auto_cl_artifact_tag(args)}_{scale_tag}_{seed_tag}_{source_size_tag}"
    if str(getattr(args, "repr_sample_qc_mode", "strict")).lower() == "off":
        name += "_noqc"
    name += build_repr_anchor_sample_tag(args)
    # if dec_mode:
    #     # name += f"_cm{cluster_mode}_mc{int(max_cluster)}"
    #     name += f"_{args.decomp_method}_{args.dec_save_mode}"

    if args.sample_mode=="cluster":
        name += f"_kmeans"
    if args.sample_mode=="cluster_nearest":
        name += f"_kmeans-n"

    return name


def build_selector_display_repr_set_name(args) -> str:
    """
    Selector result filenames can summarize adaptive search profiles without
    pretending there is one concrete repr_input_dim/repr_sub_pred_len.
    Step1/2 source artifacts still use build_repr_set_name(args). SimpleTS v6
    exposes its internally trained TS2Vec in Step3/4/result names.
    """
    is_simplets = str(getattr(args, "repr_v", "") or "")[:1] == "6"
    if not (auto_cl_enabled(args) or bool(getattr(args, "enable_pred_len_adaptive_repr", False))):
        concrete_name = build_repr_set_name(args)
        return _replace_repr_encoder_prefix(concrete_name, args, "TS2Vec") if is_simplets else concrete_name

    prefix = get_zoo_repr_prefix(getattr(args, "zoo_repr_set", None))
    size = int(getattr(args, "repr_size", 0))
    source_size_tag = build_repr_source_size_tag(prefix, size)
    seed_tag = get_repr_seed_tag(args)
    scale_tag = get_repr_scale_tag(args)
    encoder = "TS2Vec" if is_simplets else getattr(args, "repr_encoder", "repr")
    if str(getattr(args, "repr_encoder", "")) == "SimpleTS2Vec":
        fingerprint = str(getattr(args, "simplets_ts2vec_checkpoint_fingerprint", "") or "")
        if fingerprint:
            encoder = f"{encoder}_ck{fingerprint[:12]}"
    output_dim = int(getattr(args, "repr_output_dim", 0))

    if auto_cl_enabled(args):
        mode = get_auto_cl_mode(args)
        dim_base = f"autocl{mode}_p{len(get_auto_cl_profiles(mode))}"
    else:
        dim_base = f"{int(getattr(args, 'repr_input_dim', 0))}to{output_dim}"

    if auto_cl_enabled(args):
        dim_tag = f"_{dim_base}"
    elif bool(getattr(args, "enable_pred_len_adaptive_repr", False)):
        dim_tag = f"_{dim_base}_autopl"
    else:
        dim_tag = f"_{dim_base}_pl{int(getattr(args, 'repr_sub_pred_len', 0))}"

    name = f"{encoder}{dim_tag}_{scale_tag}_{seed_tag}_{source_size_tag}"
    if str(getattr(args, "repr_sample_qc_mode", "strict")).lower() == "off":
        name += "_noqc"
    name += build_repr_anchor_sample_tag(args)
    if args.sample_mode == "cluster":
        name += "_kmeans"
    if args.sample_mode == "cluster_nearest":
        name += "_kmeans-n"
    return name


def build_repr_eval_pool_name(args) -> str:
    """
    Name for the full candidate pool behind clustered or random anchors.

    This file stores the pre-cluster candidate windows used to produce the
    anchors. It intentionally omits encoder and anchor-construction tags because
    the raw windows are shared by different encoder/cluster diagnostics with
    the same source set, repr size, and repr-data seed.
    """
    prefix = get_zoo_repr_prefix(getattr(args, "zoo_repr_set", None))
    size = int(getattr(args, "repr_size", 0))
    source_size_tag = build_repr_source_size_tag(prefix, size)
    repr_data_seed = _resolve_seed(args, "repr_data_seed")
    input_dim = int(getattr(args, "repr_input_dim", 0))
    pred_len = int(getattr(args, "repr_sub_pred_len", 0))
    name = f"{source_size_tag}_in{input_dim}_pl{pred_len}{get_auto_cl_artifact_tag(args)}_{get_repr_scale_tag(args)}_sd{repr_data_seed}"
    name += build_repr_anchor_sample_tag(args)
    return f"{name}_pool"

def build_repr_forward_stem(args) -> str:
    'TSRouter runtime message.'
    repr_set_name = build_repr_set_name(args)
    return f"{repr_set_name}_{get_forward_seed_tag(args)}"


def build_repr_forward_all_results_stem(args) -> str:
    """
    Step2 all-results prefix.

    auto_cl keeps one summary CSV across its profile set, while
    per-sample files still use build_repr_forward_stem(args) so Step3 can load
    the concrete profile artifacts.
    """
    if not auto_cl_enabled(args):
        return build_repr_forward_stem(args)

    prefix = get_zoo_repr_prefix(getattr(args, "zoo_repr_set", None))
    size = int(getattr(args, "repr_size", 0))
    source_size_tag = build_repr_source_size_tag(prefix, size)
    seed_tag = get_repr_seed_tag(args)
    scale_tag = get_repr_scale_tag(args)
    encoder = getattr(args, "repr_encoder", "repr")
    mode = get_auto_cl_mode(args)
    name = f"{encoder}_auto_cl_{mode}_{scale_tag}_{seed_tag}_{source_size_tag}"
    if str(getattr(args, "repr_sample_qc_mode", "strict")).lower() == "off":
        name += "_noqc"
    name += build_repr_anchor_sample_tag(args)
    if args.sample_mode == "cluster":
        name += "_kmeans"
    if args.sample_mode == "cluster_nearest":
        name += "_kmeans-n"
    return f"{name}_{get_forward_seed_tag(args)}"


def build_repr_eval_pool_forward_stem(args) -> str:
    pool_name = build_repr_eval_pool_name(args)
    return f"{pool_name}_{get_forward_seed_tag(args)}"


def build_trained_encoder_name(args) -> str:
    prefix = get_zoo_repr_prefix(getattr(args, "zoo_repr_set", None))
    source_size_tag = build_repr_source_size_tag(prefix, int(getattr(args, "repr_size", 0)))
    metric = str(getattr(args, "train_rank_metric", getattr(args, "sgl_rank_metric", "MASE"))).upper()
    loss = str(getattr(args, "train_encoder_loss", "supcon")).lower()
    dim_tag = f"{int(getattr(args, 'repr_input_dim', 0))}to{int(getattr(args, 'repr_output_dim', 0))}_pl{int(getattr(args, 'repr_sub_pred_len', 0))}"
    seed_tag = get_repr_seed_tag(args) + f"_{get_forward_seed_tag(args)}"
    train_tag = (
        f"ep{int(getattr(args, 'train_encoder_epochs', 30))}"
        f"_lr{float(getattr(args, 'train_encoder_lr', 1e-3)):g}"
        f"_bs{int(getattr(args, 'train_encoder_batch_size', 256))}"
        f"_top3{float(getattr(args, 'train_top3_weight', 0.5)):g}"
        f"_tau{float(getattr(args, 'train_encoder_temperature', 0.1)):g}"
    )
    patience = int(getattr(args, "train_encoder_early_stop_patience", 0) or 0)
    min_delta = float(getattr(args, "train_encoder_early_stop_min_delta", 0.0) or 0.0)
    if patience > 0:
        train_tag += f"_esp{patience}_esd{min_delta:g}"
    return (
        f"{getattr(args, 'repr_encoder', 'Train')}_{dim_tag}{get_auto_cl_artifact_tag(args)}_{get_repr_scale_tag(args)}"
        f"_{seed_tag}_{source_size_tag}_{getattr(args, 'sample_mode', 'sample')}"
        f"_{metric}_{loss}_{train_tag}"
    )


def get_trained_encoder_path(args) -> str:
    root = str(getattr(args, "trained_encoder_dir", TSROUTER_TRAINED_ENCODER_DIR))
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, build_trained_encoder_name(args) + ".pt")


def _is_autoforecast_repr(args) -> bool:
    return str(getattr(args, "repr_v", "") or "")[:1] == "7"


def _is_autoxpcr_repr(args) -> bool:
    return _is_autoforecast_repr(args) and route_efficiency_mode_enabled(args)


def _is_simplets_repr(args) -> bool:
    return str(getattr(args, "repr_v", "") or "")[:1] == "6"


def _replace_repr_encoder_prefix(name: str, args, replacement: str) -> str:
    source_encoder = str(getattr(args, "repr_encoder", "") or "")
    source_prefix = f"{source_encoder}_"
    if source_encoder and str(name).startswith(source_prefix):
        return f"{replacement}_{str(name)[len(source_prefix):]}"
    return str(name)


def build_selector_artifact_repr_set_name(args) -> str:
    """Concrete Step3/4 repr identity, decoupled from the Step1/2 source encoder."""
    concrete_name = build_repr_set_name(args)
    if _is_simplets_repr(args):
        return _replace_repr_encoder_prefix(concrete_name, args, "TS2Vec")
    return concrete_name


def _autoforecast_learner_tag(args) -> str:
    raw = str(getattr(args, "autoforecast_learner", "LSTM") or "LSTM").strip().upper()
    if raw == "GDBT":
        raw = "GBDT"
    if raw in {"HGBDT", "HISTGBDT", "HISTGRADIENTBOOSTING"}:
        raw = "GBDT"
    if raw not in {"LSTM", "GBDT", "MLP"}:
        raw = "LSTM"
    return f"af{raw.lower()}"


def build_model_repr_name(args) -> str:
    'TSRouter runtime message.'
    if _is_simplets_repr(args):
        forward_stem = f"{build_selector_artifact_repr_set_name(args)}_{get_forward_seed_tag(args)}"
    else:
        forward_stem = build_repr_forward_stem(args)
    route_fast_tag = (
        "_rfast"
        if route_efficiency_mode_enabled(args) and not _is_simplets_repr(args)
        else ""
    )
    name = (
        f"zoo{args.current_zoo_num}-{args.zoo_total_num}_{forward_stem}"
        f"-v{args.repr_v}{args.base_metrics}_repr-{args.model_repr_mode}"
        f"_sub{args.subset_top_k}_{args.subset_perf_scale}{route_fast_tag}"
    )
    if _is_autoforecast_repr(args):
        name += f"_{_autoforecast_learner_tag(args)}"
    if _is_autoforecast_repr(args) or _is_simplets_repr(args):
        name += get_advanced_baseline_train_scope_tag(args)
    return name


def build_tsrouter_result_filename(args, model_repr_name: str | None = None) -> str:
    'TSRouter runtime message.'
    if model_repr_name is None:
        model_repr_name = build_model_repr_name(args)
    if auto_cl_enabled(args) or bool(getattr(args, "enable_pred_len_adaptive_repr", False)):
        concrete_repr_set_name = build_selector_artifact_repr_set_name(args)
        display_repr_set_name = build_selector_display_repr_set_name(args)
        model_repr_name = str(model_repr_name).replace(concrete_repr_set_name, display_repr_set_name, 1)

    task_strategy = str(getattr(args, "task_sample_strategy", "latest_random")).lower()
    task_strategy_tag = "" if task_strategy == "latest_random" else f"_ts{task_strategy}"
    window_strategy, sample_ratio, tag_window_strategy = get_effective_task_window_sample_strategy(args)
    window_strategy_tag = f"_ws{window_strategy}" if tag_window_strategy else ""
    sample_ratio_tag = "" if sample_ratio <= 0 else f"_sr{sample_ratio:g}"
    raw_instability_threshold = getattr(args, "task_rank_top3_instability_threshold", -1.0)
    if raw_instability_threshold is None or str(raw_instability_threshold).strip() == "":
        instability_threshold = -1.0
    else:
        instability_threshold = float(raw_instability_threshold)
    if abs(instability_threshold) < 1e-12:
        instability_threshold = 0.0
    instability_tag = "" if instability_threshold < 0 else f"_fb{instability_threshold:g}"
    channel_fuse_tag = get_task_channel_fuse_tag(args)
    route_family_tag = get_route_family_tag(args)
    repr_v_head = str(getattr(args, "repr_v", 0))[0]
    if repr_v_head == "5":
        stem = model_repr_name + (
            f"_w{args.repr_weight_ratio}_k{args.repr_v5_nearest_k}_rd{getattr(args, 'rank_decay_coef', 1.0):g}_v5w{args.repr_v5_distance_power:g}"
            f"_task{args.sample_repr_num}_v{args.task_sample_version}{task_strategy_tag}{window_strategy_tag}{sample_ratio_tag}{instability_tag}{channel_fuse_tag}{route_family_tag}_{get_selector_seed_tag(args)}_"
            f"top{args.ensemble_size}-{args.ensemble_agg}_res{args.restrict_top_model_num}"
        )
    else:
        stem = model_repr_name + (
            f"_w{args.repr_weight_ratio}_{args.model_repr_agg}"
            f"_task{args.sample_repr_num}_v{args.task_sample_version}{task_strategy_tag}{window_strategy_tag}{sample_ratio_tag}{instability_tag}{channel_fuse_tag}{route_family_tag}_{get_selector_seed_tag(args)}_"
            f"top{args.ensemble_size}-{args.ensemble_agg}_res{args.restrict_top_model_num}"
        )

    if getattr(args, "GE_released", False):
        if getattr(args, "GE_fast_eval", False):
            return stem + "_GE_fast_all_results.csv"
        return stem + "_GE_all_results.csv"
    return stem + "_all_results.csv"


def same_stage_tsrouter_result_candidates(args, target_path: str) -> list[str]:
    """
    Find Step4 result CSVs with the same stage and identical parameter suffix,
    allowing zoo_total_num and numerically equivalent weight tokens to differ.

    For example, ``_w0_`` and ``_w0.0_`` represent the same
    ``repr_weight_ratio`` and are compatible read candidates.
    """
    target_path = str(target_path)
    target_base = os.path.basename(target_path)
    match = re.match(r"^zoo(\d+)-(\d+)_(.+)$", target_base)
    if match is None:
        return []
    stage = match.group(1)
    suffix = match.group(3)
    weight_token_re = re.compile(
        r"_w(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)_"
    )

    def canonical_weight_token(text: str) -> str:
        def replace(match_obj: re.Match) -> str:
            value = float(match_obj.group("value"))
            if value == 0:
                value = 0.0
            return f"_w{value:g}_"

        return weight_token_re.sub(replace, str(text), count=1)

    def canonical_auto_cl_token(text: str) -> str:
        return re.sub(
            r"autocl(v\d+)_to\d+_pl\d+",
            r"autocl\1_p3",
            str(text),
            count=1,
        )

    canonical_suffix = canonical_auto_cl_token(canonical_weight_token(suffix))
    weight_match = weight_token_re.search(suffix)
    suffix_patterns = [suffix]
    if weight_match is not None:
        weight_glob = suffix[:weight_match.start()] + "_w*_" + suffix[weight_match.end():]
        if weight_glob not in suffix_patterns:
            suffix_patterns.append(weight_glob)
    for pattern in list(suffix_patterns):
        profile_token = re.search(r"autocl(v\d+)_p3", pattern)
        if profile_token is not None:
            legacy_pattern = pattern.replace(
                profile_token.group(0),
                f"autocl{profile_token.group(1)}_to*_pl*",
                1,
            )
            if legacy_pattern not in suffix_patterns:
                suffix_patterns.append(legacy_pattern)
    roots: list[str] = []
    for root in [
        os.path.dirname(target_path),
        os.path.join(get_tsrouter_selector_result_dir(args), f"stage{stage}"),
        get_tsrouter_selector_result_dir(args),
    ]:
        if root and root not in roots:
            roots.append(root)
    candidates: list[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for suffix_pattern in suffix_patterns:
            pattern = f"zoo{stage}-*_{suffix_pattern}"
            for path in glob.glob(os.path.join(root, pattern)):
                norm = os.path.normpath(path)
                if os.path.normcase(norm) == os.path.normcase(os.path.normpath(target_path)):
                    continue
                candidate_match = re.match(rf"^zoo{re.escape(stage)}-(\d+)_(.+)$", os.path.basename(norm))
                if candidate_match is None:
                    continue
                candidate_suffix = canonical_auto_cl_token(
                    canonical_weight_token(candidate_match.group(2))
                )
                if candidate_suffix != canonical_suffix:
                    continue
                if os.path.isfile(norm):
                    candidates.append(norm)
    return sorted(
        set(candidates),
        key=lambda p: (os.path.getmtime(p), p),
        reverse=True,
    )

def _model_order_from_step3_repr_path(model_repr_path: Path) -> tuple[list[str], str]:
    """Read the model-id order attached to a Step3 representation artifact."""
    manifest_path = model_repr_path.with_name(f"{model_repr_path.stem}_model_manifest.json")
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        for key in ("model_abbr_order", "model_names"):
            value = payload.get(key) if isinstance(payload, dict) else None
            if isinstance(value, list) and value:
                return [str(item) for item in value], f"manifest:{manifest_path}"

    subset_paths = sorted(model_repr_path.parent.glob(f"{model_repr_path.stem}*_subset_assign.pkl"))
    for subset_path in subset_paths:
        try:
            with subset_path.open("rb") as file_obj:
                payload = pickle.load(file_obj)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for key in ("model_abbr_order", "model_names"):
            value = payload.get(key)
            if isinstance(value, list) and value:
                return [str(item) for item in value], f"subset:{subset_path}"

    if model_repr_path.is_file():
        try:
            with model_repr_path.open("rb") as file_obj:
                payload = pickle.load(file_obj)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            for key in ("model_abbr_order", "model_names"):
                value = payload.get(key)
                if isinstance(value, list) and value:
                    return [str(item) for item in value], f"model_repr:{model_repr_path}"
            metadata_keys = {
                "total_models",
                "model_weights",
                "model_metric_weights",
                "metric",
                "centers",
                "center_repr",
            }
            model_keys = [
                str(key)
                for key in payload
                if not str(key).startswith("__") and str(key) not in metadata_keys
            ]
            if model_keys:
                return model_keys, f"model_repr:{model_repr_path}"
    return [], ""


def _step3_repr_paths_for_result_candidate(args, candidate_path: str) -> list[Path]:
    """Map a Step4 result filename back to possible Step3 representation files."""
    candidate_name = Path(candidate_path).name
    match = re.match(r"^zoo(?P<stage>\d+)-(?P<total>\d+)_(?P<suffix>.+)$", candidate_name)
    if match is None:
        return []

    try:
        _, _, target_model_repr_path, _ = get_repr_save_path(args)
    except Exception:
        return []
    target_model_repr = Path(target_model_repr_path)
    stage = match.group("stage")
    total = match.group("total")
    candidate_repr_name = re.sub(
        rf"^zoo{re.escape(stage)}-\d+_",
        f"zoo{stage}-{total}_",
        target_model_repr.name,
        count=1,
    )

    roots: list[Path] = []
    for root in (
        target_model_repr.parent,
        Path(str(getattr(args, "save_model_repr_path", TSROUTER_MODEL_REPR_DIR))) / f"stage{stage}",
    ):
        if root not in roots:
            roots.append(root)

    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = os.path.normcase(os.path.normpath(str(path)))
        if key not in seen:
            seen.add(key)
            candidates.append(path)

    for root in roots:
        exact = root / candidate_repr_name
        exact_manifest = exact.with_name(f"{exact.stem}_model_manifest.json")
        if exact.is_file() or exact_manifest.is_file():
            add(exact)
        if not root.is_dir():
            continue
        for manifest_path in root.glob(f"zoo{stage}-{total}_*_model_manifest.json"):
            repr_stem = manifest_path.name[: -len("_model_manifest.json")]
            if candidate_name.startswith(f"{repr_stem}_"):
                add(manifest_path.with_name(f"{repr_stem}.pkl"))
        for repr_path in root.glob(f"zoo{stage}-{total}_*.pkl"):
            if repr_path.stem.endswith("_subset_assign"):
                continue
            if candidate_name.startswith(f"{repr_path.stem}_"):
                add(repr_path)
    return candidates


def compatible_same_stage_tsrouter_result_candidates(
    args,
    target_path: str,
    expected_model_order: Iterable[str],
    *,
    verbose: bool = False,
) -> list[str]:
    """Return same-stage results whose persisted model-id order is identical."""
    expected = [str(item) for item in expected_model_order]
    compatible: list[str] = []
    for candidate in same_stage_tsrouter_result_candidates(args, target_path):
        model_order: list[str] = []
        order_source = ""
        for model_repr_path in _step3_repr_paths_for_result_candidate(args, candidate):
            model_order, order_source = _model_order_from_step3_repr_path(model_repr_path)
            if model_order:
                break
        if model_order == expected:
            compatible.append(candidate)
        elif verbose:
            reason = (
                f"model order mismatch: saved={model_order}, expected={expected}"
                if model_order
                else "no persisted Step3 model order found"
            )
            print(f"[same-stage-reuse] rejected {candidate}: {reason}; source={order_source or 'none'}")
    return compatible


def materialize_compatible_tsrouter_result(
    args,
    target_path: str,
    expected_model_order: Iterable[str],
    *,
    verbose: bool = False,
) -> str | None:
    """Copy a model-compatible same-stage Step4 CSV to its canonical zoo path."""
    target = os.path.normpath(str(target_path))
    if os.path.isfile(target):
        return target
    expected = [str(item) for item in expected_model_order]
    candidates = compatible_same_stage_tsrouter_result_candidates(
        args,
        target,
        expected,
        verbose=verbose,
    )
    if not candidates:
        return None
    source = candidates[0]
    target_dir = os.path.dirname(target)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    shutil.copy2(source, target)
    print(
        f"[same-stage-reuse] copied model-compatible result: {source} -> {target}; "
        f"model_abbr_order={expected}"
    )
    return source


def build_gift_eval_task_repr_cache_name(args, search_context_len: int | None = None) -> str:
    'TSRouter runtime message.'
    if search_context_len is None:
        search_context_len = int(getattr(args, "repr_input_dim", 0))
    strategy = str(getattr(args, "task_sample_strategy", "latest_random")).lower()
    strategy_tag = "" if strategy == "latest_random" else f"_ts{strategy}"
    window_strategy, ratio, tag_window_strategy = get_effective_task_window_sample_strategy(args)
    window_strategy_tag = f"_ws{window_strategy}" if tag_window_strategy else ""
    ratio_tag = "" if ratio <= 0 else f"_sr{ratio:g}"
    return (
        f"cl{int(search_context_len)}"
        f"_n{int(getattr(args, 'sample_repr_num', 0))}"
        f"_{get_repr_scale_tag(args)}"
        f"{strategy_tag}{window_strategy_tag}{ratio_tag}"
        f"_ss{_resolve_seed(args, 'search_seed')}"
        f".pkl"
    )


def get_gift_eval_task_repr_cache_path(args, search_context_len: int | None = None) -> str:
    cache_dir = str(getattr(args, "gift_eval_task_repr_dir", TSROUTER_TASK_REPR_DIR))
    os.makedirs(cache_dir, exist_ok=True)
    save_path=os.path.join(cache_dir, build_gift_eval_task_repr_cache_name(args, search_context_len))
    return save_path

def get_repr_save_path(args):
    repr_set_name = build_repr_set_name(args)
    Model_repr_name = build_model_repr_name(args)
    weight_filename = f"weight_{Model_repr_name}.pkl"
    tsrouter_save_name = build_tsrouter_result_filename(args, model_repr_name=Model_repr_name)
    stage = int(getattr(args, "current_zoo_num", 0) or 0)
    if _is_autoxpcr_repr(args):
        save_root = AUTOXPCR_SELECTOR_ARTIFACT_DIR
    elif _is_autoforecast_repr(args):
        save_root = AUTOFORECAST_SELECTOR_ARTIFACT_DIR
    elif _is_simplets_repr(args):
        save_root = SIMPLETS_SELECTOR_ARTIFACT_DIR
    else:
        save_root = str(getattr(args, "save_model_repr_path", TSROUTER_MODEL_REPR_DIR))
    save_dir = os.path.join(save_root, f"stage{stage}") if stage > 0 else save_root
    weight_path = os.path.join(save_dir, weight_filename)
    model_repr_path = os.path.join(save_dir, f'{Model_repr_name}.pkl')
    return repr_set_name, weight_path, model_repr_path, tsrouter_save_name





# from run_model_zoo import build_parser, prepare_args
#
# if __name__ == "__main__":
#     parser = build_parser(add_help=False)
#     args = parser.parse_args()
#     args = prepare_args(args)

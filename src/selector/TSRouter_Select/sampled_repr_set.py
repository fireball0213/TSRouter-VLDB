'TSRouter runtime message.'

import os
import re
import sys
import glob
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from gluonts.dataset.common import ListDataset

from gluonts.dataset import DataEntry
from gluonts.transform import Transformation

                                     
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PARENT_PARENT_DIR = os.path.abspath(os.path.join(PARENT_DIR, ".."))
sys.path.append(PARENT_DIR)
sys.path.append(PARENT_PARENT_DIR)

from utils.decomposition import decomposition_method
from utils.tools import convert_tsf_to_dataframe
from utils.path_utils import (
    TSROUTER_REPR_DATA_SOURCE_DIR,
    TSROUTER_SAMPLED_REPR_DIR,
    TSROUTER_SAMPLED_REPR_POOL_DIR,
    auto_cl_enabled as is_auto_cl_enabled,
    build_repr_eval_pool_name,
    build_repr_set_name,
    candidate_sample_dirs,
    get_auto_cl_mode,
    get_auto_cl_profile,
    get_auto_cl_profile_name,
    get_auto_cl_profiles,
    get_repr_anchor_window_sample_strategy,
    get_repr_scale_protocol,
    parse_zoo_repr_set,
)
from utils.io_lock import atomic_pickle_dump
from encoder.base_encoder import EncoderFactory
                                                      

                        
MODEL_DATASET_PATH = {
    "chronos": os.path.join(TSROUTER_REPR_DATA_SOURCE_DIR, "chronos", "chronos_profile_source.tsf"),
    "moirai": os.path.join(TSROUTER_REPR_DATA_SOURCE_DIR, "c65.tsf"),
    "timesfm": os.path.join(TSROUTER_REPR_DATA_SOURCE_DIR, "c68.tsf"),
    "visionts": os.path.join(TSROUTER_REPR_DATA_SOURCE_DIR, "time_series_shape_5000_84.npy"),
    "lotsa": os.path.join(TSROUTER_REPR_DATA_SOURCE_DIR, "lotsa_verified_sample_5k.npy"),
}


def _resolve_source_path(path: str) -> str:
    if os.path.exists(path):
        return path
    norm_path = os.path.normpath(path)
    known_roots = [
        TSROUTER_REPR_DATA_SOURCE_DIR,
        "results_artifacts/caches/Sampled_repr_pool",
        TSROUTER_SAMPLED_REPR_DIR,
    ]
    candidate_rels: List[str] = [os.path.basename(norm_path)]
    for root in known_roots:
        try:
            rel = os.path.relpath(norm_path, os.path.normpath(root))
        except ValueError:
            continue
        if rel and not rel.startswith(".."):
            candidate_rels.insert(0, rel)
            break

    for rel in candidate_rels:
        for base_dir in candidate_sample_dirs(TSROUTER_REPR_DATA_SOURCE_DIR):
            for candidate in [
                os.path.join(base_dir, rel),
                os.path.join(base_dir, "domain_sample", rel),
            ]:
                if os.path.exists(candidate):
                    return candidate
    return path


def _inject_domain_repr_paths(
    path_map: dict,
    sample_dir: str = TSROUTER_REPR_DATA_SOURCE_DIR,
) -> dict:
    'TSRouter runtime message.'
    candidate_dirs = []
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
            dom = m.group(1).lower()
            key = f"domain_{dom}"
            path_map[key] = os.path.join(cur_dir, fn)
    return path_map


MODEL_DATASET_PATH = _inject_domain_repr_paths(MODEL_DATASET_PATH)


def _domain_source_prefix(model_name: str) -> str:
    domain = str(model_name)
    if domain.startswith("domain_"):
        domain = domain[len("domain_"):]
    if domain == "oracle":
        return "oracle"
    return domain


def _repr_source_len_from_path(path: str) -> int | None:
    m = re.search(r"_len(\d+)(k)?(?:_|\.npy$)", os.path.basename(str(path)).lower())
    if m is None:
        return None
    value = int(m.group(1))
    if m.group(2) == "k":
        value *= 1000
    return value


def _npy_source_usable(path: str, min_source_length: int | None = None) -> bool:
    try:
        arr = np.load(path, mmap_mode="r")
        shape = tuple(arr.shape)
        dtype_itemsize = int(arr.dtype.itemsize)
        offset = int(getattr(arr, "offset", 0) or 0)
        expected_bytes = offset + int(arr.size) * dtype_itemsize
        actual_bytes = os.path.getsize(path)
        del arr
    except Exception as exc:
        print(f"TSRouter runtime message: {path} ({exc})")
        return False
    if actual_bytes < expected_bytes:
        print(
            f"TSRouter runtime message: {path} "
            f"size={actual_bytes}, expected>={expected_bytes}, shape={shape}"
        )
        return False
    if len(shape) != 2:
        print(f"TSRouter runtime message: {path} shape={shape}")
        return False
    if min_source_length is not None and shape[1] < int(min_source_length):
        return False
    return True


def _find_domain_repr_path(
    model_name: str,
    repr_data_seed: int | None,
    repr_scale_protocol: str,
    sample_dir: str = TSROUTER_REPR_DATA_SOURCE_DIR,
    min_source_length: int | None = None,
    exact_source_length: int | None = None,
) -> str | None:
    prefix = _domain_source_prefix(model_name)
    direct_candidates = [
        os.path.join(sample_dir, "moirai_timesfm", f"domain_{prefix}.npy"),
        os.path.join(sample_dir, "moirai_timesfm", f"{prefix}.npy"),
        os.path.join(sample_dir, f"domain_{prefix}.npy"),
        os.path.join(sample_dir, f"{prefix}.npy"),
    ]
    for direct in direct_candidates:
        if os.path.exists(direct) and _npy_source_usable(direct, min_source_length=min_source_length):
            return direct
    protocol_tag = "raw" if str(repr_scale_protocol or "raw").lower() == "raw" else "std"
    candidate_dirs = []
    for base_dir in candidate_sample_dirs(sample_dir):
        candidate_dirs.extend([base_dir, os.path.join(base_dir, "domain_sample")])

    patterns: list[tuple[str, bool]] = []
    if repr_data_seed is not None:
        patterns.append((f"{prefix}_num*_len*_sd{int(repr_data_seed)}_{protocol_tag}.npy", False))
    else:
        patterns.append((f"{prefix}_num*_len*_{protocol_tag}.npy", False))
        patterns.append((f"{prefix}_num*_len*.npy", True))

    for pat, legacy_only in patterns:
        hits = []
        for cur_dir in candidate_dirs:
            hits.extend(glob.glob(os.path.join(cur_dir, pat)))
        if legacy_only:
            hits = [p for p in hits if "_sd" not in os.path.basename(p)]
        if exact_source_length is not None:
            hits = [
                p for p in hits
                if _repr_source_len_from_path(p) == int(exact_source_length)
                and f"_len{int(exact_source_length)}" in os.path.basename(p).lower()
            ]
        elif min_source_length is not None:
            hits = [
                p for p in hits
                if (_repr_source_len_from_path(p) is None or _repr_source_len_from_path(p) >= int(min_source_length))
            ]
        hits = [p for p in hits if _npy_source_usable(p, min_source_length=min_source_length)]
        if hits:
            return sorted(
                hits,
                key=lambda p: (
                    _repr_source_len_from_path(p) if _repr_source_len_from_path(p) is not None else 10**12,
                    -os.path.getmtime(p),
                ),
            )[0]
    if repr_data_seed is not None:
        legacy_or_mismatch = []
        for cur_dir in candidate_dirs:
            legacy_or_mismatch.extend(glob.glob(os.path.join(cur_dir, f"{prefix}_num*_len*.npy")))
        legacy_or_mismatch = [
            p for p in legacy_or_mismatch
            if os.path.exists(p)
            and os.path.basename(p) not in {
                os.path.basename(h)
                for h in glob.glob(os.path.join(os.path.dirname(p), f"{prefix}_num*_len*_sd{int(repr_data_seed)}_{protocol_tag}.npy"))
            }
        ]
        if legacy_or_mismatch:
            preview = ", ".join(os.path.basename(p) for p in sorted(legacy_or_mismatch)[:5])
            print(
                f"[DomainRepr][strict-seed] reject legacy/mismatched sources for {model_name}: "
                f"need sd{int(repr_data_seed)}_{protocol_tag}; examples={preview}"
            )
    return None


def _domain_rebuild_command(
    model_name: str,
    repr_data_seed: int = 2025,
    repr_scale_protocol: str = "raw",
    source_seq_len: int = 1000,
    adaptive_source_len: bool = False,
) -> str:
    def adaptive_source_args() -> tuple[str, int, int, int]:
        for mode in ["v3", "v2", "v1"]:
            profiles = get_auto_cl_profiles(mode)
            if any(int(profile["repr_source_exact_length"]) == int(source_seq_len) for profile in profiles):
                by_key = {str(profile["profile_key"]): profile for profile in profiles}
                return (
                    mode,
                    int(by_key["long"]["repr_source_exact_length"]),
                    int(by_key["middle"]["repr_source_exact_length"]),
                    int(by_key["short"]["repr_source_exact_length"]),
                )
        return "v1", 3000, 992, 144

    domain = str(model_name)
    if domain.startswith("domain_"):
        domain = domain[len("domain_"):]
    if domain == "oracle":
        if adaptive_source_len:
            mode, long_len, middle_len, short_len = adaptive_source_args()
            return (
                "PYTHONPATH=src python -m selector.TSRouter_Select.build_domain_repr_set "
                f"--out_dir {TSROUTER_REPR_DATA_SOURCE_DIR} --build_oracle "
                f"--n_samples_per_domain 10000 --repr_data_seed {int(repr_data_seed)} "
                f"--repr_scale_protocol {repr_scale_protocol} --auto_cl_mode {mode} "
                "--enable_context_len_adaptive_repr true "
                f"--long_repr_source_len {long_len} --middle_repr_source_len {middle_len} --short_repr_source_len {short_len}"
            )
        return (
            "PYTHONPATH=src python -m selector.TSRouter_Select.build_domain_repr_set "
            f"--out_dir {TSROUTER_REPR_DATA_SOURCE_DIR} --build_oracle "
            f"--n_samples_per_domain 10000 --seq_len {int(source_seq_len)} --repr_data_seed {int(repr_data_seed)} "
            f"--repr_scale_protocol {repr_scale_protocol}"
        )
    alias = {
        "econ_fin": "econ_fin",
        "web_cloudops": "web_cloudops",
    }.get(domain, domain)
    if adaptive_source_len:
        mode, long_len, middle_len, short_len = adaptive_source_args()
        return (
            "PYTHONPATH=src python -m selector.TSRouter_Select.build_domain_repr_set "
            f"--out_dir {TSROUTER_REPR_DATA_SOURCE_DIR} --only_domains {alias} "
            f"--n_samples_per_domain 10000 --repr_data_seed {int(repr_data_seed)} "
            f"--repr_scale_protocol {repr_scale_protocol} --auto_cl_mode {mode} "
            "--enable_context_len_adaptive_repr true "
            f"--long_repr_source_len {long_len} --middle_repr_source_len {middle_len} --short_repr_source_len {short_len}"
        )
    return (
        "PYTHONPATH=src python -m selector.TSRouter_Select.build_domain_repr_set "
        f"--out_dir {TSROUTER_REPR_DATA_SOURCE_DIR} --only_domains {alias} "
        f"--n_samples_per_domain 10000 --seq_len {int(source_seq_len)} --repr_data_seed {int(repr_data_seed)} "
        f"--repr_scale_protocol {repr_scale_protocol}"
    )


def _source_seq_len_hint(min_source_length: int | None = None) -> int:
    if min_source_length is None:
        return 1000
    value = int(min_source_length)
    if value <= 144:
        return 144
    if value <= 1000:
        return 1000
    if value <= 3000:
        return 3000
    return value


def _adaptive_exact_source_length(args, sub_seq_length: int) -> int | None:
    if not is_auto_cl_enabled(args):
        return None
    profile = get_auto_cl_profile(args)
    if profile is not None:
        return int(profile["repr_source_exact_length"])
    repr_input_dim = int(getattr(args, "repr_input_dim", 0))
    repr_sub_pred_len = int(getattr(args, "repr_sub_pred_len", 0))
    long_input_dim = int(getattr(args, "long_repr_input_dim", 2048))
    long_pred_len = int(getattr(args, "long_repr_sub_pred_len", 720))
    short_input_dim = int(getattr(args, "short_repr_input_dim", 96))
    short_pred_len = int(getattr(args, "short_repr_sub_pred_len", 48))
    long_source_len = int(getattr(args, "long_repr_source_len", 3000))
    middle_source_len = int(getattr(args, "middle_repr_source_len", 992))
    short_source_len = int(getattr(args, "short_repr_source_len", 144))

    if repr_input_dim == long_input_dim and repr_sub_pred_len == long_pred_len:
        return long_source_len
    if repr_input_dim == short_input_dim and repr_sub_pred_len == short_pred_len:
        return short_source_len
    if int(sub_seq_length) <= middle_source_len:
        return middle_source_len
    return _source_seq_len_hint(sub_seq_length)


def _load_npy_checked(
    data_file: str,
    model_name: str,
    repr_data_seed: int | None = None,
    repr_scale_protocol: str = "raw",
) -> np.ndarray:
    try:
        arr = np.load(data_file)
    except ValueError as e:
        raise ValueError(
            f"TSRouter runtime message: {model_name}TSRouter runtime message: {data_file}\n"
            f"TSRouter runtime message: {e}\n"
            'TSRouter runtime message.'
            'TSRouter runtime message.'
            f"  {_domain_rebuild_command(model_name, repr_data_seed or 2025, repr_scale_protocol)}"
        ) from e
    except OSError as e:
        raise OSError(
            f"TSRouter runtime message: {model_name}TSRouter runtime message: {data_file}\n"
            f"TSRouter runtime message: {e}\n"
            f"TSRouter runtime message: {_domain_rebuild_command(model_name, repr_data_seed or 2025, repr_scale_protocol)}"
        ) from e

    if arr.ndim != 2:
        raise ValueError(
            f"[DomainRepr] {data_file}TSRouter runtime message: {arr.shape}。\n"
            f"TSRouter runtime message: {_domain_rebuild_command(model_name, repr_data_seed or 2025, repr_scale_protocol)}"
        )
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(
            f"[DomainRepr] {data_file}TSRouter runtime message: {arr.shape}。\n"
            f"TSRouter runtime message: {_domain_rebuild_command(model_name, repr_data_seed or 2025, repr_scale_protocol)}"
        )
    return arr


def _series_length_summary(data: List[np.ndarray]) -> str:
    if not data:
        return "empty"
    lengths = np.asarray([np.asarray(x).shape[0] for x in data], dtype=np.int64)
    return (
        f"n={len(data)}, first={int(lengths[0])}, min={int(lengths.min())}, "
        f"p50={int(np.median(lengths))}, max={int(lengths.max())}"
    )

                                                        
def resolve_dataset_source_file(
    model_name: str,
    repr_scale_protocol: str = "raw",
    repr_data_seed: int | None = None,
    min_source_length: int | None = None,
    exact_source_length: int | None = None,
) -> str:
    """Resolve the concrete Step0/source file used by one Step1 source."""
    if model_name not in MODEL_DATASET_PATH and not str(model_name).startswith("domain_"):
        raise ValueError(f"{model_name} not in MODEL_DATASET_PATH.")

    if str(model_name).startswith("domain_"):
        data_file = _find_domain_repr_path(
            model_name,
            repr_data_seed,
            repr_scale_protocol,
            min_source_length=min_source_length,
            exact_source_length=exact_source_length,
        )
        if data_file is None:
            rebuild_source_len = int(exact_source_length) if exact_source_length is not None else _source_seq_len_hint(min_source_length)
            raise FileNotFoundError(
                f"[DomainRepr] missing source for {model_name}, repr_data_seed={repr_data_seed}, "
                f"repr_scale_protocol={repr_scale_protocol}, min_source_length={min_source_length}, "
                f"exact_source_length={exact_source_length}. Try:\n"
                f"  {_domain_rebuild_command(model_name, repr_data_seed or 2025, repr_scale_protocol, rebuild_source_len, adaptive_source_len=exact_source_length is not None)}"
            )
        return os.path.abspath(str(data_file))

    return os.path.abspath(str(_resolve_source_path(MODEL_DATASET_PATH[model_name])))


def load_dataset(
    model_name: str,
    repr_scale_protocol: str = "raw",
    repr_data_seed: int | None = None,
    min_source_length: int | None = None,
    exact_source_length: int | None = None,
) -> List[np.ndarray]:
    'TSRouter runtime message.'
    data_file = resolve_dataset_source_file(
        model_name,
        repr_scale_protocol=repr_scale_protocol,
        repr_data_seed=repr_data_seed,
        min_source_length=min_source_length,
        exact_source_length=exact_source_length,
    )

    if not os.path.exists(data_file):
        if str(model_name).startswith("domain_"):
            hint = (
                "Regenerate it with:\n"
                f"  {_domain_rebuild_command(model_name, repr_data_seed or 2025, repr_scale_protocol, _source_seq_len_hint(min_source_length))}"
            )
        else:
            hint = (
                "Copy the pre-sampled repr source file from the old ZooCast "
                f"Sampled_repr_set/Sampled_repr_pool location into {TSROUTER_REPR_DATA_SOURCE_DIR}/."
            )
        raise FileNotFoundError(
            f"[ReprSource] missing source for {model_name}: {data_file}\n"
            f"Expected repr source pool root: {TSROUTER_REPR_DATA_SOURCE_DIR}/\n"
            f"{hint}"
        )

                                           
    if model_name in ["chronos", "moirai", "timesfm"]:
        df_raw, freq, *_ = convert_tsf_to_dataframe(data_file)
                                      
        df_raw = df_raw.drop("series_name", axis=1)
        df_data = df_raw.values[:, 1]
        df_data = [np.array(series) for series in df_data]

                                    
    elif model_name in ["visionts", "lotsa"] or str(model_name).startswith("domain_"):
        df_raw = _load_npy_checked(
            data_file,
            model_name,
            repr_data_seed=repr_data_seed,
            repr_scale_protocol=repr_scale_protocol,
        )
        df_data = list(df_raw)

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    protocol = str(repr_scale_protocol or "raw").lower()
    if protocol == "raw":
        return [np.asarray(sequence, dtype=np.float32).reshape(-1) for sequence in df_data]
    if protocol != "standard":
        raise ValueError(f"Unknown repr_scale_protocol={repr_scale_protocol!r}")

                                  
    scaler = StandardScaler()
    normalized_data: List[np.ndarray] = []
    for sequence in df_data:
        sequence = sequence.reshape(-1, 1)                      # (T,) -> (T, 1)
        normalized_sequence = scaler.fit_transform(sequence)         
        normalized_sequence = normalized_sequence.reshape(-1)             
        normalized_data.append(normalized_sequence)

    return normalized_data


def _scale_window_by_protocol(
    x: np.ndarray,
    protocol: str,
    *,
    context_len: int | None = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if str(protocol or "raw").lower() == "raw":
        return x

    fit_len = int(context_len or x.shape[0])
    fit_len = max(1, min(fit_len, x.shape[0]))
    context = x[:fit_len]
    mean = float(np.mean(context))
    std = float(np.std(context))
    if std < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mean) / std).astype(np.float32)

                                                                                          
                                                                            
def save_pkl_array(
    args,
    file_stem: str,
    data,
    *,
    squeeze_last_dim: bool = False,
    output_dir: Optional[str] = None,
) -> str:
    'TSRouter runtime message.'
                                          
    if isinstance(data, dict):
        data_to_save = data
        shape_info = "{dict}"
    elif torch.is_tensor(data):
        if squeeze_last_dim:
            data = data.squeeze(-1)
        data_to_save = data.detach().cpu().numpy()
        shape_info = str(getattr(data_to_save, "shape", None))
    else:
        data_to_save = np.array(data)
        shape_info = str(getattr(data_to_save, "shape", None))

            
    root_dir = output_dir or args.save_repr_data_path
    save_path = os.path.join(root_dir, f"{file_stem}.pkl")
    os.makedirs(root_dir, exist_ok=True)

    print("Save object shape:", shape_info, " saving to:", save_path)

    atomic_pickle_dump(data_to_save, save_path)

    return save_path


def _pool_meta_matches_current_args(
    path: str,
    args,
    *,
    pool_name: str,
    exact_source_length: int | None,
) -> Tuple[bool, str]:
    if not os.path.exists(path):
        return False, "missing"
    try:
        meta = _load_pickle(path)
    except Exception as exc:
        return False, f"unreadable:{exc}"
    if not isinstance(meta, dict):
        return False, "not_dict"

    expected = {
        "meta_role": "repr_candidate_pool_sidecar",
        "pool_name": str(pool_name),
        "repr_input_dim": int(getattr(args, "repr_input_dim", 0)),
        "repr_sub_pred_len": int(getattr(args, "repr_sub_pred_len", 0)),
        "repr_scale_protocol": get_repr_scale_protocol(args),
        "repr_data_seed": int(getattr(args, "repr_data_seed", 2025)),
        "repr_anchor_protocol": str(getattr(args, "repr_anchor_protocol", "window")),
        "task_sample_strategy": str(getattr(args, "task_sample_strategy", "latest_random")),
        "task_window_sample_strategy": get_repr_anchor_window_sample_strategy(args),
        "repr_anchor_window_sample_strategy": get_repr_anchor_window_sample_strategy(args),
        "sample_repr_ratio": float(getattr(args, "sample_repr_ratio", 0.0)),
        "anchor_sample_num": int(getattr(args, "sample_repr_num", 20)),
    }
    if exact_source_length is not None:
        expected["repr_source_exact_length"] = int(exact_source_length)

    for key, expected_value in expected.items():
        actual_value = meta.get(key)
        try:
            if isinstance(expected_value, bool):
                matched = bool(actual_value) == expected_value
            elif isinstance(expected_value, int) and not isinstance(expected_value, bool):
                matched = int(actual_value) == expected_value
            elif isinstance(expected_value, float):
                matched = np.isclose(float(actual_value), expected_value, rtol=0.0, atol=1e-12)
            else:
                matched = str(actual_value) == str(expected_value)
        except (TypeError, ValueError):
            matched = False
        if not matched:
            return False, f"metadata_changed:{key}"

    seq_shape = meta.get("seq_shape")
    expected_width = int(getattr(args, "repr_input_dim", 0)) + int(getattr(args, "repr_sub_pred_len", 0))
    if not isinstance(seq_shape, (tuple, list)) or len(seq_shape) < 2:
        return False, "seq_shape_missing"
    try:
        if int(seq_shape[1]) != expected_width:
            return False, "metadata_changed:seq_shape"
    except (TypeError, ValueError):
        return False, "seq_shape_invalid"

    per_source = meta.get("per_source")
    if not isinstance(per_source, list) or not per_source:
        return False, "per_source_missing"
    return True, "matched"


def _anchor_rows_match_pool_members(
    *,
    anchor_path: str,
    anchor_meta_path: str,
    pool_path: str,
    sample_mode: str,
) -> Tuple[bool, str]:
    # Only real-window anchor modes should be byte-lineage checked against pool rows.
    if str(sample_mode) not in {"random", "cluster_nearest"}:
        return True, "not_pool_member_anchor_mode"
    for label, path in (
        ("anchor", anchor_path),
        ("anchor_meta", anchor_meta_path),
        ("pool", pool_path),
    ):
        if not os.path.exists(path):
            return False, f"{label}_missing"
    try:
        anchor = np.asarray(_load_pickle(anchor_path), dtype=np.float32)
        anchor_meta = _load_pickle(anchor_meta_path)
        pool = np.asarray(_load_pickle(pool_path), dtype=np.float32)
    except Exception as exc:
        return False, f"unreadable:{exc}"
    if not isinstance(anchor_meta, dict):
        return False, "anchor_meta_not_dict"
    member_idx = anchor_meta.get("center_member_idx_in_pool")
    if member_idx is None:
        return False, "center_member_idx_missing"
    member_idx = np.asarray(member_idx, dtype=np.int64).reshape(-1)
    if anchor.ndim != 2 or pool.ndim != 2:
        return False, f"invalid_shape:anchor={anchor.shape},pool={pool.shape}"
    if anchor.shape[0] != member_idx.size:
        return False, f"row_count_mismatch:anchor={anchor.shape[0]},member_idx={member_idx.size}"
    if anchor.shape[1] != pool.shape[1]:
        return False, f"width_mismatch:anchor={anchor.shape[1]},pool={pool.shape[1]}"
    if np.any((member_idx < 0) | (member_idx >= pool.shape[0])):
        return False, "center_member_idx_out_of_range"

    mapped = pool[member_idx]
    row_mismatch = ~np.all(np.isclose(anchor, mapped, rtol=0.0, atol=1e-6), axis=1)
    mismatch_count = int(np.sum(row_mismatch))
    if mismatch_count:
        max_abs = float(np.max(np.abs(anchor[row_mismatch] - mapped[row_mismatch])))
        return False, f"anchor_pool_rows_mismatch:mismatches={mismatch_count},max_abs={max_abs:.6g}"
    return True, "matched"


def _qc_subseq(
    x: np.ndarray,
    *,
    max_abs: float = 1e6,
    max_std: float = 1e4,
    min_std: float = 1e-6,
    clip_value: Optional[float] = None,
    fill_nan: str = "interp",   # "interp" / "zero"
) -> Optional[np.ndarray]:
    'TSRouter runtime message.'
    x = np.asarray(x, dtype=np.float32)

                   
    if not np.isfinite(x).all():
        if fill_nan == "zero":
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        elif fill_nan == "interp":
                             
            bad = ~np.isfinite(x)
            if bad.all():
                return None
            idx = np.arange(len(x))
            x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
        else:
            return None

                  
    if np.max(np.abs(x)) > max_abs:
        if clip_value is not None:
            x = np.clip(x, -clip_value, clip_value)
        else:
            return None

                  
    s = float(np.std(x))
    if s > max_std:
        if clip_value is not None:
            x = np.clip(x, -clip_value, clip_value)
        else:
            return None
    if s < min_std:
                                    
        return None

    return x.astype(np.float32)

def _qc_mase_denom(
    x: np.ndarray,
    *,
    m: int,
    denom_floor: float = 1e-3,
    min_denom: float = 5e-3,
) -> bool:
    'TSRouter runtime message.'
    x = np.asarray(x, dtype=np.float32)
    T = x.shape[0]
    m = int(max(1, min(m, max(1, T // 2))))
    if T <= m:
        m = 1
    diff = x[m:] - x[:-m]
    denom = float(np.mean(np.abs(diff))) if diff.size > 0 else 0.0
    denom = max(denom, float(denom_floor))
    return denom >= float(min_denom)


def _time_coverage_start(max_start: int, sample_index: int, sample_num: int, rng: np.random.RandomState) -> int:
    max_start = int(max_start)
    if max_start <= 0:
        return 0
    sample_num = max(1, int(sample_num))
    sample_index = int(sample_index) % sample_num
    if sample_num == 1:
        return int(max_start)
    base = round(max_start * sample_index / (sample_num - 1))
    stride = max(1, max_start // max(sample_num - 1, 1))
    jitter = int(rng.randint(-stride // 4, stride // 4 + 1)) if stride > 1 else 0
    return int(np.clip(base + jitter, 0, max_start))


def _sample_start_for_strategy(
    max_start: int,
    rng: np.random.RandomState,
    *,
    strategy: str,
    sample_index: int,
    sample_num: int,
) -> int:
    strategy = str(strategy or "latest_random").lower()
    if strategy == "time_coverage":
        return _time_coverage_start(max_start, sample_index, sample_num, rng)
    return int(rng.randint(0, int(max_start) + 1))


def _anchor_sample_indices(anchor_sample_num: int, ratio: float, n_series: int, strategy: str, rng: np.random.RandomState) -> list[int]:
    target_n = max(1, int(anchor_sample_num or 1))
    ratio = float(ratio or 0.0)
    if ratio > 0:
        target_n = max(target_n, int(np.ceil(max(1, int(n_series)) * ratio)))
    strategy = str(strategy or "legacy").lower()
    if strategy == "legacy":
        return list(range(target_n))
    if strategy == "first":
        return list(range(target_n))
    if strategy == "last":
        return list(range(max(0, target_n - 1), -1, -1))
    if strategy == "random":
        return rng.randint(0, target_n, size=target_n).astype(int).tolist()
    if strategy == "even":
        if target_n == 1:
            return [0]
        return np.linspace(0, target_n - 1, target_n).round().astype(int).tolist()
    return list(range(target_n))


def _exact_row_duplicate_stats(rows: np.ndarray) -> dict:
    arr = np.ascontiguousarray(np.asarray(rows))
    if arr.ndim != 2 or arr.shape[0] == 0:
        return {
            "total_rows": int(arr.shape[0]) if arr.ndim >= 1 else 0,
            "unique_rows": 0,
            "duplicate_rows": 0,
            "duplicate_rate": 0.0,
            "repeated_groups": 0,
            "max_repeat": 0,
        }
    row_bytes = arr.dtype.itemsize * arr.shape[1]
    row_view = arr.view(np.dtype((np.void, row_bytes))).reshape(-1)
    _, counts = np.unique(row_view, return_counts=True)
    duplicate_rows = int(arr.shape[0] - counts.size)
    return {
        "total_rows": int(arr.shape[0]),
        "unique_rows": int(counts.size),
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": float(duplicate_rows / arr.shape[0]),
        "repeated_groups": int(np.sum(counts > 1)),
        "max_repeat": int(counts.max()),
    }


def _cluster_candidate_pool(
    candidates: np.ndarray,
    *,
    mode: str,
    n_samples: int,
    repr_model,
    device: Optional[torch.device],
    scaler: Optional[object],
    embed_len: Optional[int],
    batch_size: int,
    seed: int,
    rng: np.random.RandomState,
    qc_enabled: bool,
    source_label: str,
    pool_accumulator: Optional[dict],
    analysis_source: Optional[str],
    analysis_source_file: Optional[str],
    candidate_mul: int,
    max_candidates: int,
) -> np.ndarray:
    """Select anchors from a candidate pool with random sampling or KMeans."""
    duplicate_stats = _exact_row_duplicate_stats(candidates)
    print(
        f"[pool-build][duplicates] source={source_label}, "
        f"unique={duplicate_stats['unique_rows']}/{duplicate_stats['total_rows']}, "
        f"duplicate_rows={duplicate_stats['duplicate_rows']} "
        f"({duplicate_stats['duplicate_rate']:.2%}), "
        f"repeated_groups={duplicate_stats['repeated_groups']}, "
        f"max_repeat={duplicate_stats['max_repeat']}",
        flush=True,
    )
    embed_len_eff = int(embed_len or candidates.shape[1])
    candidates_embed = candidates[:, :embed_len_eff]

    if scaler is not None:
        candidates_embed = scaler.transform(candidates_embed)

    repr_model.eval()
    feats_list = []
    with torch.no_grad():
        for st in range(0, candidates_embed.shape[0], batch_size):
            ed = min(st + batch_size, candidates_embed.shape[0])
            x = candidates_embed[st:ed]
            x = torch.from_numpy(x).float().to(device).unsqueeze(-1)
            feats = repr_model.encode(x).detach().cpu().numpy()
            feats_list.append(feats)
    feats = np.concatenate(feats_list, axis=0)

    M = feats.shape[0]
    k = min(n_samples, M)
    representative = (
        "random_real_window"
        if mode == "random"
        else ("mean_sequence" if mode == "cluster" else "nearest_real_window")
    )
    if mode == "random":
        # Anchor membership must depend only on the saved pool and seed, not on
        # how many RNG calls were needed while constructing that pool. This
        # keeps fresh builds and rebuild-from-pool byte-for-byte consistent.
        anchor_rng = np.random.RandomState(int(seed) ^ 0x5EED5EED)
        center_member_idx = anchor_rng.choice(M, size=k, replace=False).astype(np.int32)
        centers = feats[center_member_idx]
        labels = np.empty((M,), dtype=np.int32)
        # Only build diagnostic Voronoi labels; anchor selection itself is
        # encoder-independent and never calls KMeans.
        chunk_size = max(1, min(4096, M))
        center_norm = np.sum(centers * centers, axis=1)[None, :]
        for st in range(0, M, chunk_size):
            ed = min(st + chunk_size, M)
            query = feats[st:ed]
            dist2 = (
                np.sum(query * query, axis=1)[:, None]
                + center_norm
                - 2.0 * np.matmul(query, centers.T)
            )
            labels[st:ed] = np.argmin(dist2, axis=1).astype(np.int32)
        print(
            f"[random-anchor] source={source_label}, method=random_without_replacement, "
            f"n_anchors={k}, candidates={M}, seed={seed}",
            flush=True,
        )
    elif k <= 1:
        print(
            f"[cluster] source={source_label}, method=single_cluster, n_clusters={k}, "
            f"embedding_dim={feats.shape[1]}, representative={representative}, seed={seed}",
            flush=True,
        )
        labels = np.zeros((M,), dtype=np.int32)
        centers = feats.mean(axis=0, keepdims=True)
    else:
        print(
            f"[cluster] source={source_label}, method=KMeans, n_clusters={k}, "
            f"embedding_dim={feats.shape[1]}, representative={representative}, seed={seed}",
            flush=True,
        )
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = kmeans.fit_predict(feats).astype(np.int32)
        centers = kmeans.cluster_centers_

    if mode != "random":
        center_member_idx = np.full((k,), -1, dtype=np.int32)
        for c in range(k):
            cluster_idx = np.where(labels == c)[0]
            if cluster_idx.size == 0:
                continue
            center = centers[c]
            d = np.sum((feats[cluster_idx] - center) ** 2, axis=1)
            center_member_idx[c] = int(cluster_idx[int(np.argmin(d))])
        if mode == "cluster_nearest" and np.any(center_member_idx < 0):
            missing_pos = np.where(center_member_idx < 0)[0]
            used = set(int(v) for v in center_member_idx[center_member_idx >= 0])
            fill_order = [int(v) for v in rng.permutation(M) if int(v) not in used]
            if len(fill_order) < len(missing_pos):
                fill_order.extend(int(v) for v in rng.permutation(M))
            for pos, ridx in zip(missing_pos, fill_order):
                center_member_idx[int(pos)] = int(ridx)
            print(
                f"⚠️ [cluster_nearest] source={source_label}, empty_clusters={len(missing_pos)}; "
                "filled center_member_idx from real pool rows",
                flush=True,
            )

    if pool_accumulator is not None:
        src = analysis_source if analysis_source is not None else "unknown"
        cluster_offset = int(pool_accumulator.get("_cluster_offset", 0))
        candidate_offset = int(pool_accumulator.get("_candidate_offset", 0))
        pool_accumulator["_cluster_offset"] = cluster_offset + int(k)
        pool_accumulator["_candidate_offset"] = candidate_offset + int(M)
        pool_accumulator["seq"].append(candidates)
        pool_accumulator["emb"].append(feats)
        pool_accumulator["cluster"].append((labels + cluster_offset).astype(np.int32))
        pool_accumulator["source"].append(np.array([src] * candidates.shape[0], dtype=object))
        pool_accumulator["center_member_idx"].append((center_member_idx + candidate_offset).astype(np.int32))
        pool_accumulator["per_source"].append({
            "source": src,
            "source_file": str(analysis_source_file or ""),
            "k_clusters": int(k),
            "cluster_offset": int(cluster_offset),
            "candidate_offset": int(candidate_offset),
            "n_candidates": int(M),
            "candidate_mul": int(candidate_mul),
            "max_candidates": int(max_candidates),
            "seed": int(seed),
            "duplicate_stats": duplicate_stats,
        })

    chosen_sequences: List[np.ndarray] = []
    if mode == "random":
        chosen_sequences = [
            candidates[int(idx)].astype(np.float32)
            for idx in center_member_idx
        ]
    elif mode == "cluster":
        for c in range(k):
            idxs = np.where(labels == c)[0]
            if idxs.size == 0:
                continue
            rep = candidates[idxs].mean(axis=0)
            if qc_enabled:
                rep = _qc_subseq(rep, clip_value=None)
                if rep is None:
                    continue
            else:
                rep = np.nan_to_num(rep, nan=0.0, posinf=0.0, neginf=0.0)
            chosen_sequences.append(rep.astype(np.float32))
    else:
        chosen_sequences = [
            candidates[int(idx)].astype(np.float32)
            for idx in center_member_idx
        ]

    if len(chosen_sequences) < n_samples:
        need = n_samples - len(chosen_sequences)
        rest_idx = rng.permutation(M)
        for ridx in rest_idx:
            if need <= 0:
                break
            chosen_sequences.append(candidates[int(ridx)].astype(np.float32))
            need -= 1

    chosen_sequences = chosen_sequences[:n_samples]
    return np.stack(chosen_sequences, axis=0).astype(np.float32)


def _build_anchor_sidecar(
    args,
    *,
    repr_set_name: str,
    selected_data,
    zoo_repr_set: List[str],
    n_list: List[int],
    source_files: Optional[List[str]] = None,
    pool_name: Optional[str] = None,
    center_member_idx_all: Optional[np.ndarray] = None,
    per_source_cluster: Optional[List[dict]] = None,
) -> dict:
    selected_shape = tuple(np.asarray(selected_data, dtype=np.float32).shape)
    source_files = list(source_files or [""] * len(zoo_repr_set))
    if len(source_files) < len(zoo_repr_set):
        source_files.extend([""] * (len(zoo_repr_set) - len(source_files)))
    center_source = [
        str(source)
        for source, count in zip(zoo_repr_set, n_list)
        for _ in range(int(count))
    ]
    center_source_file = [
        str(source_file)
        for source_file, count in zip(source_files, n_list)
        for _ in range(int(count))
    ]
    selected_count = int(selected_shape[0]) if selected_shape else 0
    if len(center_source) != selected_count:
        center_source = (center_source + ["unknown"] * selected_count)[:selected_count]
        center_source_file = (center_source_file + [""] * selected_count)[:selected_count]
    meta = {
        "repr_set_name": repr_set_name,
        "meta_role": "repr_anchor_sidecar",
        "seq_shape": selected_shape,
        "zoo_repr_set": list(zoo_repr_set),
        "per_source_anchor_count": {
            str(name): int(n)
            for name, n in zip(zoo_repr_set, n_list)
        },
        "source_file_by_source": {
            str(name): str(source_file)
            for name, source_file in zip(zoo_repr_set, source_files)
        },
        "center_source": np.asarray(center_source, dtype=object),
        "center_source_file": np.asarray(center_source_file, dtype=object),
        "sample_mode": str(getattr(args, "sample_mode", "")),
        "repr_encoder": str(getattr(args, "repr_encoder", "")),
        "encoder_type": getattr(args, "encoder_type", None),
        "encoder_structure": getattr(args, "encoder_structure", None),
        "simplets_ts2vec_checkpoint": str(getattr(args, "simplets_ts2vec_checkpoint", "") or ""),
        "simplets_ts2vec_checkpoint_fingerprint": str(
            getattr(args, "simplets_ts2vec_checkpoint_fingerprint", "") or ""
        ),
        "simplets_ts2vec_source_repr_set_name": str(
            getattr(args, "simplets_ts2vec_source_repr_set_name", "") or ""
        ),
        "auto_cl_mode": get_auto_cl_mode(args),
        "adaptive_profile": get_auto_cl_profile_name(args),
        "repr_input_dim": int(getattr(args, "repr_input_dim", 0)),
        "repr_output_dim": int(getattr(args, "repr_output_dim", 0)),
        "repr_sub_pred_len": int(getattr(args, "repr_sub_pred_len", 0)),
        "repr_source_exact_length": (
            int(getattr(args, "repr_source_exact_length"))
            if getattr(args, "repr_source_exact_length", None) is not None
            else None
        ),
        "repr_scale_protocol": get_repr_scale_protocol(args),
        "repr_data_seed": int(getattr(args, "repr_data_seed", 2025)),
        "repr_encoder_seed": int(getattr(args, "repr_encoder_seed", 2025)),
        "repr_sample_qc_mode": str(getattr(args, "repr_sample_qc_mode", "strict")),
        "repr_anchor_protocol": str(getattr(args, "repr_anchor_protocol", "window")),
        "task_sample_strategy": str(getattr(args, "task_sample_strategy", "latest_random")),
        "task_window_sample_strategy": get_repr_anchor_window_sample_strategy(args),
        "repr_anchor_window_sample_strategy": get_repr_anchor_window_sample_strategy(args),
        "step4_task_window_sample_strategy": str(getattr(args, "task_window_sample_strategy", "legacy")),
        "sample_repr_ratio": float(getattr(args, "sample_repr_ratio", 0.0)),
        "anchor_sample_num": int(getattr(args, "sample_repr_num", 20)),
    }
    if pool_name is not None:
        meta.update({
            "pool_name": pool_name,
            "pool_path": os.path.join(TSROUTER_SAMPLED_REPR_POOL_DIR, f"{pool_name}.pkl"),
            "pool_meta_path": os.path.join(TSROUTER_SAMPLED_REPR_POOL_DIR, f"{pool_name}_meta.pkl"),
        })
    if center_member_idx_all is not None:
        meta["center_member_idx_in_pool"] = np.asarray(center_member_idx_all, dtype=np.int64)
    if per_source_cluster is not None:
        meta["per_source_cluster"] = per_source_cluster
    return meta


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _meta_matches_exact_source_length(path: str, exact_source_length: int | None) -> bool:
    if exact_source_length is None:
        return True
    if not os.path.exists(path):
        return False
    try:
        meta = _load_pickle(path)
    except Exception as exc:
        print(f"[Step1:cache][WARN] cannot read meta for source-length check: {path} ({exc})")
        return False
    if not isinstance(meta, dict):
        return False
    value = meta.get("repr_source_exact_length")
    try:
        return int(value) == int(exact_source_length)
    except (TypeError, ValueError):
        return False


def _pool_per_source_segments(
    pool_meta: dict,
    pool_seq: np.ndarray,
    fallback_sources: List[str],
    fallback_n_list: List[int],
) -> List[dict]:
    per_source = pool_meta.get("per_source") if isinstance(pool_meta, dict) else None
    source_file_by_source = dict(pool_meta.get("source_file_by_source", {}) or {}) if isinstance(pool_meta, dict) else {}
    if isinstance(per_source, list) and per_source:
        segments: List[dict] = []
        for i, item in enumerate(per_source):
            if not isinstance(item, dict):
                continue
            offset = int(item.get("candidate_offset", item.get("offset", 0)))
            n_candidates = int(item.get("n_candidates", item.get("count", 0)))
            if n_candidates <= 0:
                continue
            source = str(item.get("source", fallback_sources[i] if i < len(fallback_sources) else f"source{i}"))
            segments.append({
                "source": source,
                "source_file": str(item.get("source_file", source_file_by_source.get(source, ""))),
                "candidate_offset": offset,
                "n_candidates": n_candidates,
                "k_clusters": int(item.get("k_clusters", fallback_n_list[i] if i < len(fallback_n_list) else n_candidates)),
                "seed": int(item.get("seed", 2025 + i * 1000)),
            })
        if segments:
            return segments

    total = int(pool_seq.shape[0])
    if not fallback_sources:
        fallback_sources = ["pool"]
        fallback_n_list = [min(total, int(np.sum(fallback_n_list)) if fallback_n_list else total)]
    base = total // len(fallback_sources)
    rem = total % len(fallback_sources)
    offset = 0
    segments = []
    for i, source in enumerate(fallback_sources):
        n_candidates = base + (1 if i < rem else 0)
        segments.append({
            "source": source,
            "source_file": "",
            "candidate_offset": offset,
            "n_candidates": n_candidates,
            "k_clusters": int(fallback_n_list[i] if i < len(fallback_n_list) else n_candidates),
            "seed": 2025 + i * 1000,
        })
        offset += n_candidates
    return segments


def _rebuild_anchor_from_saved_pool(
    args,
    *,
    repr_set_name: str,
    pool_name: str,
    zoo_repr_set: List[str],
    n_list: List[int],
    sub_seq_length: int,
    encoder,
    device: torch.device,
    scaler,
    embed_len: int,
    batch_size: int = 1024,
) -> Tuple[np.ndarray, dict]:
    pool_path = os.path.join(TSROUTER_SAMPLED_REPR_POOL_DIR, f"{pool_name}.pkl")
    pool_meta_path = os.path.join(TSROUTER_SAMPLED_REPR_POOL_DIR, f"{pool_name}_meta.pkl")
    pool_seq = np.asarray(_load_pickle(pool_path), dtype=np.float32)
    if pool_seq.ndim != 2:
        raise ValueError(f"pool data should be 2-D, got shape={pool_seq.shape}: {pool_path}")
    if int(pool_seq.shape[1]) != int(sub_seq_length):
        raise ValueError(
            f"pool seq length mismatch: pool={pool_seq.shape[1]}, expected={sub_seq_length}, path={pool_path}"
        )
    pool_meta = _load_pickle(pool_meta_path) if os.path.exists(pool_meta_path) else {}
    segments = _pool_per_source_segments(pool_meta, pool_seq, zoo_repr_set, n_list)
    cluster_acc = {
        "seq": [],
        "emb": [],
        "cluster": [],
        "source": [],
        "center_member_idx": [],
        "per_source": [],
        "_cluster_offset": 0,
        "_candidate_offset": 0,
    }
    selected_parts: List[np.ndarray] = []
    for i, item in enumerate(segments):
        start = int(item["candidate_offset"])
        end = start + int(item["n_candidates"])
        if start < 0 or end > pool_seq.shape[0] or end <= start:
            raise ValueError(f"invalid pool segment in {pool_meta_path}: {item}")
        k = int(item.get("k_clusters", 0))
        if k <= 0:
            continue
        seed = int(item.get("seed", int(getattr(args, "repr_data_seed", 2025)) + i * 1000))
        source = str(item.get("source", f"source{i}"))
        cluster_acc["_candidate_offset"] = start
        selected_parts.append(
            _cluster_candidate_pool(
                pool_seq[start:end],
                mode=str(getattr(args, "sample_mode", "")),
                n_samples=k,
                repr_model=encoder,
                device=device,
                scaler=scaler,
                embed_len=embed_len,
                batch_size=batch_size,
                seed=seed,
                rng=np.random.RandomState(seed),
                qc_enabled=str(getattr(args, "repr_sample_qc_mode", "strict")).lower() != "off",
                source_label=source,
                pool_accumulator=cluster_acc,
                analysis_source=source,
                analysis_source_file=str(item.get("source_file", "")),
                candidate_mul=int(item.get("candidate_mul", 10)),
                max_candidates=int(item.get("max_candidates", pool_seq.shape[0])),
            )
        )
    if not selected_parts:
        raise ValueError(f"saved pool has no usable segments: {pool_path}")
    selected = np.concatenate(selected_parts, axis=0).astype(np.float32)
    center_member_idx_all = np.concatenate(cluster_acc["center_member_idx"], axis=0).astype(np.int64)
    anchor_meta = _build_anchor_sidecar(
        args,
        repr_set_name=repr_set_name,
        selected_data=selected,
        zoo_repr_set=[str(item["source"]) for item in segments],
        n_list=[int(item.get("k_clusters", 0)) for item in segments],
        source_files=[str(item.get("source_file", "")) for item in segments],
        pool_name=pool_name,
        center_member_idx_all=center_member_idx_all,
        per_source_cluster=cluster_acc["per_source"],
    )
    print(f"TSRouter runtime message: {pool_name}")
    return selected, anchor_meta


def sample_sub_sequences(
    data: List[np.ndarray],
    sub_seq_length: int,
    n_samples: int,
    *,
    mode: str = "random",            # ✅ "random" / "cluster"
                                             
    repr_model=None,                                            
    device: Optional[torch.device] = None,
    scaler: Optional[object] = None,                                              
    embed_len: Optional[int] = None,
    candidate_mul: int = 10,
    max_candidates: int = 50000,
    batch_size: int = 1024,
    seed: int = 42,
    qc_mode: str = "strict",
    analysis_source: Optional[str] = None,
    analysis_source_file: Optional[str] = None,
    pool_accumulator: Optional[dict] = None,
    repr_scale_protocol: str = "raw",
    repr_anchor_protocol: str = "window",
    task_sample_strategy: str = "latest_random",
    task_window_sample_strategy: str = "legacy",
    sample_repr_ratio: float = 0.0,
    anchor_sample_num: int = 20,
) -> np.ndarray:
    'TSRouter runtime message.'

                                                                
    rng = np.random.RandomState(seed)
    qc_mode = str(qc_mode or "strict").lower()
    qc_enabled = qc_mode != "off"
    repr_anchor_protocol = str(repr_anchor_protocol or "window").lower()
    task_sample_strategy = str(task_sample_strategy or "latest_random").lower()
    anchor_sample_num = max(1, int(anchor_sample_num or 1))
    task_window_sample_strategy = str(task_window_sample_strategy or "legacy").lower()
    sample_repr_ratio = float(sample_repr_ratio or 0.0)
    planned_anchor_indices = _anchor_sample_indices(
        anchor_sample_num,
        sample_repr_ratio,
        len(data),
        task_window_sample_strategy,
        rng,
    )
    if repr_anchor_protocol == "task_sample" and (sample_repr_ratio > 0 or task_window_sample_strategy != "legacy"):
        print(
            f"[repr-task-sample] strategy={task_window_sample_strategy}, ratio={sample_repr_ratio:g}, "
            f"anchor_samples={len(planned_anchor_indices)}, base_n={anchor_sample_num}",
            flush=True,
        )

    def _pick_window(series: np.ndarray, sample_index: int = 0) -> np.ndarray:
        max_start = int(series.shape[0] - sub_seq_length)
        start_idx = _sample_start_for_strategy(
            max_start,
            rng,
            strategy=task_sample_strategy,
            sample_index=sample_index,
            sample_num=len(planned_anchor_indices),
        )
        return series[start_idx:start_idx + sub_seq_length]
    if mode == "random" and pool_accumulator is None:
        sub_sequences: List[np.ndarray] = []
        while len(sub_sequences) < n_samples:
            instance_idx = rng.randint(0, len(data))
            series = data[instance_idx]

            if series.shape[0] < sub_seq_length:
                continue

            sample_indices = planned_anchor_indices if repr_anchor_protocol == "task_sample" else [0]
            for sample_index in sample_indices:
                if len(sub_sequences) >= n_samples:
                    break
                sub_seq = _pick_window(series, sample_index)
                sub_seq = _scale_window_by_protocol(sub_seq, repr_scale_protocol, context_len=embed_len)
                if qc_enabled:
                    sub_seq = _qc_subseq(sub_seq, clip_value=None)
                    if sub_seq is None:
                        continue
                else:
                    sub_seq = np.nan_to_num(sub_seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                sub_sequences.append(sub_seq)

        if len(sub_sequences) < n_samples:
            print(f"TSRouter runtime message: {len(sub_sequences)}/{n_samples}")

        return np.array(sub_sequences)

    # ===================== pool-backed random / cluster / cluster_nearest =====================
             
    if n_samples <= 0:
        return np.zeros((0, sub_seq_length), dtype=np.float32)

                  
    valid_series_idx = [i for i, s in enumerate(data) if s.shape[0] >= sub_seq_length]
    if len(valid_series_idx) == 0:
        raise ValueError(f"TSRouter runtime message: {sub_seq_length}")

                
    n_candidates = min(int(n_samples * candidate_mul), max_candidates)
    n_candidates = max(n_candidates, n_samples)

    candidates: List[np.ndarray] = []
    max_tries = n_candidates * 20
    tries = 0

    mase_denom_floor = 1e-3
    analysis_min_mase_denom = 5e-3

    while len(candidates) < n_candidates and tries < max_tries:
        tries += 1
        sid = valid_series_idx[rng.randint(0, len(valid_series_idx))]
        series = data[sid]

        sample_indices = planned_anchor_indices if repr_anchor_protocol == "task_sample" else [0]
        for sample_index in sample_indices:
            if len(candidates) >= n_candidates:
                break
            sub_seq = _pick_window(series, sample_index)
            sub_seq = _scale_window_by_protocol(sub_seq, repr_scale_protocol, context_len=embed_len)

            if qc_enabled:
                sub_seq = _qc_subseq(sub_seq, clip_value=None)
                if sub_seq is None:
                    continue

                if not _qc_mase_denom(sub_seq, m=embed_len, denom_floor=mase_denom_floor, min_denom=analysis_min_mase_denom):
                    continue
            else:
                sub_seq = np.nan_to_num(sub_seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            candidates.append(sub_seq.astype(np.float32))

                           
    if len(candidates) < n_samples:
        print(f"⚠️ [{mode}TSRouter runtime message: {len(candidates)}<{n_samples}TSRouter runtime message: ")

                                       
        if len(candidates) > 0:
            need = int(n_samples - len(candidates))
            dup_ids = rng.choice(len(candidates), size=need, replace=True)
            for did in dup_ids:
                candidates.append(candidates[int(did)].copy())

                                        
        if len(candidates) == 0:
            relax_tries = max(1000, n_samples * 50)
            relax_cnt = 0
            while len(candidates) < n_samples and relax_cnt < relax_tries:
                relax_cnt += 1
                sid = valid_series_idx[rng.randint(0, len(valid_series_idx))]
                series = data[sid]
                max_start = series.shape[0] - sub_seq_length
                start_idx = rng.randint(0, max_start + 1)
                sub_seq = series[start_idx:start_idx + sub_seq_length]
                sub_seq = _scale_window_by_protocol(sub_seq, repr_scale_protocol, context_len=embed_len)
                                          
                sub_seq = np.nan_to_num(sub_seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                candidates.append(sub_seq)

                                    
            if len(candidates) < n_samples:
                pad = np.zeros((sub_seq_length,), dtype=np.float32)
                while len(candidates) < n_samples:
                    candidates.append(pad.copy())

    candidates = np.stack(candidates, axis=0).astype(np.float32)  # (M, T)
    source_label = analysis_source if analysis_source is not None else "unknown"
    print(
        f"[pool-build] source={source_label}, mode={mode}, "
        f"candidates={candidates.shape[0]}, target_centers={n_samples}, "
        f"seq_len={sub_seq_length}, qc_mode={qc_mode}",
        flush=True,
    )
    return _cluster_candidate_pool(
        candidates,
        mode=mode,
        n_samples=n_samples,
        repr_model=repr_model,
        device=device,
        scaler=scaler,
        embed_len=embed_len,
        batch_size=batch_size,
        seed=seed,
        rng=rng,
        qc_enabled=qc_enabled,
        source_label=source_label,
        pool_accumulator=pool_accumulator,
        analysis_source=analysis_source,
        analysis_source_file=analysis_source_file,
        candidate_mul=candidate_mul,
        max_candidates=max_candidates,
    )




                                                          

def save_sampled_repr_set(args) -> None:
    'TSRouter runtime message.'
                     
    dec_mode=False


    repr_scale_protocol = get_repr_scale_protocol(args)
    dec_save_mode = args.dec_save_mode
    sample_mode = str(getattr(args, "sample_mode", ""))
    pool_backed_mode = sample_mode in {"random", "cluster", "cluster_nearest"}
    os.makedirs(args.save_repr_data_path, exist_ok=True)
    repr_set_name = build_repr_set_name(args)
    pool_name = build_repr_eval_pool_name(args) if pool_backed_mode else ""
    pre_skip_sub_seq_length = int(getattr(args, "repr_input_dim", 0)) + int(getattr(args, "repr_sub_pred_len", 0))
    pre_skip_exact_source_length = _adaptive_exact_source_length(args, pre_skip_sub_seq_length)
    setattr(args, "repr_source_exact_length", pre_skip_exact_source_length)

    def _pool_expected_paths() -> List[str]:
        if not pool_backed_mode:
            return []
        return [
            os.path.join(TSROUTER_SAMPLED_REPR_POOL_DIR, f"{pool_name}.pkl"),
            os.path.join(TSROUTER_SAMPLED_REPR_POOL_DIR, f"{pool_name}_meta.pkl"),
        ]

    def _cluster_expected_paths() -> List[str]:
        paths: List[str] = []
        if dec_mode:
            if dec_save_mode in ("dec", "full"):
                paths.append(os.path.join(args.save_repr_data_path, f"{repr_set_name}.pkl"))
            elif dec_save_mode == "separate":
                paths.extend([
                    os.path.join(args.save_repr_data_path, f"{repr_set_name}_trend.pkl"),
                    os.path.join(args.save_repr_data_path, f"{repr_set_name}_seasonal.pkl"),
                ])
            else:
                raise ValueError(f"Unknown dec_save_mode: {dec_save_mode}")
        else:
            paths.append(os.path.join(args.save_repr_data_path, f"{repr_set_name}.pkl"))
            paths.append(os.path.join(args.save_repr_data_path, f"{repr_set_name}_meta.pkl"))
        return paths

    pool_expected = _pool_expected_paths()
    cluster_expected = _cluster_expected_paths()
    pool_missing = [p for p in pool_expected if not os.path.exists(p)]
    cluster_missing = [p for p in cluster_expected if not os.path.exists(p)]
    if pool_backed_mode:
        anchor_meta_path = os.path.join(args.save_repr_data_path, f"{repr_set_name}_meta.pkl")
        pool_path = os.path.join(TSROUTER_SAMPLED_REPR_POOL_DIR, f"{pool_name}.pkl")
        pool_meta_path = os.path.join(TSROUTER_SAMPLED_REPR_POOL_DIR, f"{pool_name}_meta.pkl")
        if pool_meta_path not in pool_missing:
            pool_meta_ok, pool_meta_reason = _pool_meta_matches_current_args(
                pool_meta_path,
                args,
                pool_name=pool_name,
                exact_source_length=pre_skip_exact_source_length,
            )
            if not pool_meta_ok:
                pool_missing.append(pool_meta_path)
                print(
                    f"[INFO][Step1:pool] rebuild candidate pool: {pool_name}, "
                    f"reason={pool_meta_reason}"
                )
        if anchor_meta_path not in cluster_missing and os.path.exists(anchor_meta_path):
            try:
                anchor_meta_check = _load_pickle(anchor_meta_path)
            except Exception:
                anchor_meta_check = {}
            center_idx_check = (
                anchor_meta_check.get("center_member_idx_in_pool")
                if isinstance(anchor_meta_check, dict)
                else None
            )
            pool_link_ok = (
                isinstance(anchor_meta_check, dict)
                and str(anchor_meta_check.get("pool_name", "")) == str(pool_name)
                and str(anchor_meta_check.get("sample_mode", "")) == sample_mode
                and center_idx_check is not None
                and np.asarray(center_idx_check).reshape(-1).size == int(getattr(args, "repr_size", 0))
            )
            if not pool_link_ok:
                cluster_missing.append(anchor_meta_path)
                print(
                    f"[INFO][Step1:anchor] rebuild pool-linked {sample_mode} anchors: "
                    f"{repr_set_name}"
                )
        if not cluster_missing and not pool_missing:
            anchor_data_path = os.path.join(args.save_repr_data_path, f"{repr_set_name}.pkl")
            rows_ok, rows_reason = _anchor_rows_match_pool_members(
                anchor_path=anchor_data_path,
                anchor_meta_path=anchor_meta_path,
                pool_path=pool_path,
                sample_mode=sample_mode,
            )
            if not rows_ok:
                cluster_missing.append(anchor_meta_path)
                print(
                    f"[INFO][Step1:anchor] rebuild pool-linked {sample_mode} anchors: "
                    f"{repr_set_name}, reason={rows_reason}"
                )
    if pre_skip_exact_source_length is not None:
        if pool_backed_mode:
            pool_meta_path = os.path.join(TSROUTER_SAMPLED_REPR_POOL_DIR, f"{pool_name}_meta.pkl")
            if pool_meta_path not in pool_missing and not _meta_matches_exact_source_length(pool_meta_path, pre_skip_exact_source_length):
                pool_missing.append(pool_meta_path)
                print(
                    f"[INFO][Step1:pool] source bucket changed or old meta lacks "
                    f"repr_source_exact_length={pre_skip_exact_source_length}: {pool_name}"
                )
        cluster_meta_path = os.path.join(args.save_repr_data_path, f"{repr_set_name}_meta.pkl")
        if cluster_meta_path not in cluster_missing and not _meta_matches_exact_source_length(cluster_meta_path, pre_skip_exact_source_length):
            cluster_missing.append(cluster_meta_path)
            print(
                f"[INFO][Step1:cluster] source bucket changed or old meta lacks "
                f"repr_source_exact_length={pre_skip_exact_source_length}: {repr_set_name}"
            )
    pool_complete = (not pool_expected) or not pool_missing
    cluster_complete = not cluster_missing

    if args.skip_saved:
        if pool_backed_mode:
            if pool_complete:
                print(f"TSRouter runtime message: {pool_name}")
            else:
                print(f"TSRouter runtime message: {len(pool_missing)}/{len(pool_expected)}TSRouter runtime message: {pool_name}")
            if cluster_complete:
                print(f"TSRouter runtime message: {repr_set_name}")
            else:
                print(f"TSRouter runtime message: {len(cluster_missing)}/{len(cluster_expected)}TSRouter runtime message: {repr_set_name}")
            if pool_complete and cluster_complete:
                return
        elif cluster_complete:
            print(f"TSRouter runtime message: {repr_set_name}")
            return
        else:
            print(f"TSRouter runtime message: {len(cluster_missing)}/{len(cluster_expected)}TSRouter runtime message: {repr_set_name}")

                                                          
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, scaler, configs = EncoderFactory.build_encoder(args, device=device)
    encoder = encoder.to(device)
    encoder.eval()

    sub_context_len = configs.input_dim
    sub_pred_len = configs.sub_pred_len
    sub_seq_length = sub_context_len + sub_pred_len

    zoo_repr_set = parse_zoo_repr_set(args.zoo_repr_set)
    auto_cl_enabled = is_auto_cl_enabled(args)
    allow_missing_sources = auto_cl_enabled or bool(
        getattr(args, "allow_missing_repr_sources", False)
    )
    exact_source_length = _adaptive_exact_source_length(args, sub_seq_length)
    setattr(args, "repr_source_exact_length", exact_source_length)
    if auto_cl_enabled:
        print(
            f"[repr-source] auto_cl profile expects exact source length="
            f"{exact_source_length} for sub_seq_length={sub_seq_length}"
        )

    repr_size = int(args.repr_size)
    base_n = repr_size // max(1, len(zoo_repr_set))
    remainder = repr_size % max(1, len(zoo_repr_set))
    planned_n_list = [base_n + (1 if i < remainder else 0) for i in range(len(zoo_repr_set))]

    if pool_backed_mode and pool_complete and not cluster_complete:
        selected_data, anchor_meta = _rebuild_anchor_from_saved_pool(
            args,
            repr_set_name=repr_set_name,
            pool_name=pool_name,
            zoo_repr_set=zoo_repr_set,
            n_list=planned_n_list,
            sub_seq_length=sub_seq_length,
            encoder=encoder,
            device=device,
            scaler=scaler,
            embed_len=sub_context_len,
            batch_size=1024,
        )
        anchor_path = save_pkl_array(args, file_stem=repr_set_name, data=selected_data, squeeze_last_dim=False)
        anchor_meta_path = save_pkl_array(args, file_stem=f"{repr_set_name}_meta", data=anchor_meta)
        rows_ok, rows_reason = _anchor_rows_match_pool_members(
            anchor_path=anchor_path,
            anchor_meta_path=anchor_meta_path,
            pool_path=os.path.join(TSROUTER_SAMPLED_REPR_POOL_DIR, f"{pool_name}.pkl"),
            sample_mode=sample_mode,
        )
        if not rows_ok:
            raise RuntimeError(
                f"Step1 rebuilt pool-linked {sample_mode} anchors but lineage is still invalid: "
                f"{repr_set_name}, reason={rows_reason}"
            )
        print(
            f"[INFO][Step1:anchor] rebuilt pool-linked {sample_mode} anchors verified: "
            f"{repr_set_name}, reason={rows_reason}"
        )
        return

    print("zoo_repr_set(raw):", args.zoo_repr_set, "scale_protocol:", repr_scale_protocol, end=" ")
    print('TSRouter runtime message.', zoo_repr_set)

    loaded_sources: List[tuple[str, List[np.ndarray], str]] = []
    for model_name in zoo_repr_set:
        try:
            source_file = resolve_dataset_source_file(
                model_name,
                repr_scale_protocol=repr_scale_protocol,
                repr_data_seed=int(getattr(args, "repr_data_seed", 2025)),
                min_source_length=sub_seq_length,
                exact_source_length=exact_source_length,
            )
            data = load_dataset(
                model_name,
                repr_scale_protocol=repr_scale_protocol,
                repr_data_seed=int(getattr(args, "repr_data_seed", 2025)),
                min_source_length=sub_seq_length,
                exact_source_length=exact_source_length,
            )
            valid_count = sum(1 for seq in data if np.asarray(seq).shape[0] >= sub_seq_length)
            if valid_count <= 0:
                raise ValueError(f"no series length >= {sub_seq_length}")
            loaded_sources.append((model_name, data, source_file))
            print(f"[repr-source] {model_name}: {_series_length_summary(data)}, usable_for_{sub_seq_length}={valid_count}")
        except Exception as exc:
            if allow_missing_sources:
                print(f"[repr-source][skip-missing] {model_name}: {exc}")
                continue
            raise

    if not loaded_sources:
        raise ValueError(
            f"No usable repr sources for sub_seq_length={sub_seq_length}, "
            f"repr_scale_protocol={repr_scale_protocol}, "
            f"allow_missing_sources={allow_missing_sources}"
        )
    if len(loaded_sources) != len(zoo_repr_set):
        print(
            f"[repr-source] using {len(loaded_sources)}/{len(zoo_repr_set)} available sources: "
            f"{[name for name, _, _ in loaded_sources]}"
        )
    zoo_repr_set = [name for name, _, _ in loaded_sources]
    loaded_data_by_source = {name: data for name, data, _ in loaded_sources}
    source_file_by_source = {name: source_file for name, _, source_file in loaded_sources}

                                     
    num_sets = len(zoo_repr_set)
    base_n = repr_size // num_sets
    remainder = repr_size % num_sets
    n_list = [base_n + (1 if i < remainder else 0) for i in range(num_sets)]
    assert sum(n_list) == repr_size

    pool_acc = None
    if pool_backed_mode:
        pool_acc = {
            "seq": [],
            "emb": [],
            "cluster": [],
            "source": [],
            "center_member_idx": [],
            "per_source": [],
            "_cluster_offset": 0,
            "_candidate_offset": 0,
        }

                                              
    datasets_periods = [24] * len(zoo_repr_set)

    if dec_mode:
        all_trend = []
        all_seasonal = []
        selected_data: List[np.ndarray] = []

        for i, model_name in enumerate(zoo_repr_set):
            print(f"Sample {n_list[i]}  Processing dataset: {model_name}", end=" ")

                          
            args.period = datasets_periods[i]
            decomp = decomposition_method(args.decomp_method, args)

                          
            data = loaded_data_by_source[model_name]
            print(f"Loaded {_series_length_summary(data)}", end=" ")

                      
            sub_sequences = sample_sub_sequences(
                data, sub_seq_length, n_list[i],
                mode=args.sample_mode,
                repr_model=encoder,
                device=device,
                scaler=scaler,
                embed_len=sub_context_len,
                candidate_mul=10,
                batch_size=1024,
                seed=int(getattr(args, "repr_data_seed", 2025)) + i * 1000,
                qc_mode=getattr(args, "repr_sample_qc_mode", "strict"),
                analysis_source=model_name,
                analysis_source_file=source_file_by_source[model_name],
                pool_accumulator=pool_acc,
                repr_scale_protocol=repr_scale_protocol,
                repr_anchor_protocol=getattr(args, "repr_anchor_protocol", "window"),
                task_sample_strategy=getattr(args, "task_sample_strategy", "latest_random"),
                task_window_sample_strategy=get_repr_anchor_window_sample_strategy(args),
                sample_repr_ratio=float(getattr(args, "sample_repr_ratio", 0.0)),
                anchor_sample_num=int(getattr(args, "sample_repr_num", 20)),
            )
            # print(f"Sampled sub-sequences shape: {sub_sequences.shape}")

            selected_data.extend(sub_sequences)

                                              
            sub_sequences_x = np.expand_dims(sub_sequences, axis=2)
            trend_x, seasonal_x, residual_x = decomp(sub_sequences_x)

            all_trend.append(trend_x)
                                               
            all_seasonal.append(seasonal_x + residual_x)

                       
        trend_x = torch.cat(all_trend, dim=0)
        seasonal_x = torch.cat(all_seasonal, dim=0)



                    
        if dec_save_mode == "dec":
                                        
            selected_dec_data = torch.cat([trend_x, seasonal_x], dim=0)
            save_pkl_array(args,file_stem=repr_set_name,data=selected_dec_data,
                squeeze_last_dim=True,  # (N,T,1) -> (N,T)
            )


        elif dec_save_mode == "full":
                                          
            selected_data_x = torch.tensor(
                np.expand_dims(selected_data, axis=2),
                dtype=torch.float32,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            selected_dec_data = torch.cat([trend_x, seasonal_x, selected_data_x], dim=0)
            save_pkl_array(args,file_stem=repr_set_name,data=selected_dec_data,
                squeeze_last_dim=True,  # (N,T,1) -> (N,T)
            )

        elif dec_save_mode == "separate":
                                    
            components = {
                f"{repr_set_name}_trend": trend_x,
                f"{repr_set_name}_seasonal": seasonal_x,
            }
            for name, data_tensor in components.items():
                save_pkl_array(args,file_stem=name,data=data_tensor,
                    squeeze_last_dim=True,  # (N,T,1) -> (N,T)
                )

        else:
            raise ValueError(f"Unknown Dec_save_mode: {dec_save_mode}")

                                                                  
    else:
        selected_data: List[np.ndarray] = []

        for i, model_name in enumerate(zoo_repr_set):
            print(f"Processing dataset: {model_name}", end=" ")

            data = loaded_data_by_source[model_name]

            print(f"Loaded {model_name}, {_series_length_summary(data)}", end=" ")

            sub_sequences = sample_sub_sequences(
                data, sub_seq_length, n_list[i],
                mode=args.sample_mode,
                repr_model=encoder,
                device=device,
                scaler=scaler,
                embed_len=sub_context_len,
                candidate_mul=10,
                batch_size=1024,
                seed=int(getattr(args, "repr_data_seed", 2025)) + i * 1000,
                qc_mode=getattr(args, "repr_sample_qc_mode", "strict"),
                analysis_source=model_name,
                analysis_source_file=source_file_by_source[model_name],
                pool_accumulator=pool_acc,
                repr_scale_protocol=repr_scale_protocol,
                repr_anchor_protocol=getattr(args, "repr_anchor_protocol", "window"),
                task_sample_strategy=getattr(args, "task_sample_strategy", "latest_random"),
                task_window_sample_strategy=get_repr_anchor_window_sample_strategy(args),
                sample_repr_ratio=float(getattr(args, "sample_repr_ratio", 0.0)),
                anchor_sample_num=int(getattr(args, "sample_repr_num", 20)),
            )
            # print(f"Sampled sub-sequences shape: {sub_sequences.shape}")

            selected_data.extend(sub_sequences)

        write_anchor = not (
            bool(getattr(args, "skip_saved", False))
            and cluster_complete
            and pool_complete
        )
        anchor_meta = None
        anchor_path = os.path.join(args.save_repr_data_path, f"{repr_set_name}.pkl")
        anchor_meta_path = os.path.join(args.save_repr_data_path, f"{repr_set_name}_meta.pkl")
        if write_anchor:
                           
            anchor_path = save_pkl_array(args,file_stem=repr_set_name,data=selected_data,squeeze_last_dim=False,)
            anchor_meta = _build_anchor_sidecar(
                args,
                repr_set_name=repr_set_name,
                selected_data=selected_data,
                zoo_repr_set=zoo_repr_set,
                n_list=n_list,
                source_files=[source_file_by_source[name] for name in zoo_repr_set],
            )

                                                    
        if pool_acc is not None and len(pool_acc["seq"]) > 0:
            pool_name = build_repr_eval_pool_name(args)
            pool_seq_all = np.concatenate(pool_acc["seq"], axis=0).astype(np.float32)
            pool_source_all = np.concatenate(pool_acc["source"], axis=0)
            center_member_idx_all = np.concatenate(pool_acc["center_member_idx"], axis=0).astype(np.int32)
            save_pkl_array(
                args,
                file_stem=pool_name,
                data=pool_seq_all,
                squeeze_last_dim=False,
                output_dir=TSROUTER_SAMPLED_REPR_POOL_DIR,
            )
            pool_meta = {
                "meta_role": "repr_candidate_pool_sidecar",
                "pool_name": pool_name,
                "seq_shape": pool_seq_all.shape,
                "source_unique": sorted(list(set(pool_source_all.tolist()))),
                "source": pool_source_all,
                "source_file_by_source": {
                    str(name): str(source_file_by_source.get(name, ""))
                    for name in zoo_repr_set
                },
                "per_source": pool_acc["per_source"],
                "candidate_mul": 10,
                "max_candidates": 50000,
                "auto_cl_mode": get_auto_cl_mode(args),
                "adaptive_profile": get_auto_cl_profile_name(args),
                "repr_input_dim": int(getattr(args, "repr_input_dim", 0)),
                "repr_output_dim": int(getattr(args, "repr_output_dim", 0)),
                "repr_sub_pred_len": int(getattr(args, "repr_sub_pred_len", 0)),
                "repr_source_exact_length": (
                    int(getattr(args, "repr_source_exact_length"))
                    if getattr(args, "repr_source_exact_length", None) is not None
                    else None
                ),
                "repr_scale_protocol": get_repr_scale_protocol(args),
                "repr_data_seed": int(getattr(args, "repr_data_seed", 2025)),
                "repr_anchor_protocol": str(getattr(args, "repr_anchor_protocol", "window")),
                "task_sample_strategy": str(getattr(args, "task_sample_strategy", "latest_random")),
                "task_window_sample_strategy": get_repr_anchor_window_sample_strategy(args),
                "repr_anchor_window_sample_strategy": get_repr_anchor_window_sample_strategy(args),
                "step4_task_window_sample_strategy": str(getattr(args, "task_window_sample_strategy", "legacy")),
                "sample_repr_ratio": float(getattr(args, "sample_repr_ratio", 0.0)),
                "anchor_sample_num": int(getattr(args, "sample_repr_num", 20)),
                "cluster_info_location": os.path.join(args.save_repr_data_path, f"{repr_set_name}_meta.pkl"),
            }
            if anchor_meta is not None:
                anchor_meta.update({
                    "pool_name": pool_name,
                    "pool_path": os.path.join(TSROUTER_SAMPLED_REPR_POOL_DIR, f"{pool_name}.pkl"),
                    "pool_meta_path": os.path.join(TSROUTER_SAMPLED_REPR_POOL_DIR, f"{pool_name}_meta.pkl"),
                    "center_member_idx_in_pool": center_member_idx_all,
                    "per_source_cluster": pool_acc["per_source"],
                })
            save_pkl_array(
                args,
                file_stem=f"{pool_name}_meta",
                data=pool_meta,
                output_dir=TSROUTER_SAMPLED_REPR_POOL_DIR,
            )
        if anchor_meta is not None:
            anchor_meta_path = save_pkl_array(args, file_stem=f"{repr_set_name}_meta", data=anchor_meta)
            if pool_acc is not None and len(pool_acc["seq"]) > 0:
                rows_ok, rows_reason = _anchor_rows_match_pool_members(
                    anchor_path=anchor_path,
                    anchor_meta_path=anchor_meta_path,
                    pool_path=os.path.join(TSROUTER_SAMPLED_REPR_POOL_DIR, f"{pool_name}.pkl"),
                    sample_mode=sample_mode,
                )
                if not rows_ok:
                    raise RuntimeError(
                        f"Step1 wrote pool-linked {sample_mode} anchors but lineage is invalid: "
                        f"{repr_set_name}, reason={rows_reason}"
                    )
                print(
                    f"[INFO][Step1:anchor] pool-linked {sample_mode} anchors verified: "
                    f"{repr_set_name}, reason={rows_reason}"
                )


                             
class ReprDatasetAdapter:
    def __init__(
        self,
        args,
        freq='H',
    ):
        'TSRouter runtime message.'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder, scaler, configs = EncoderFactory.build_encoder(args, device=device)

        input_length = configs.input_dim
        prediction_length = configs.sub_pred_len

                                   
        root_path = args.save_repr_data_path,
        repr_set_name = build_repr_set_name(args)

                                                                                        
        override_stem = getattr(args, "repr_set_file_stem_override", "")
        suffix = getattr(args, "repr_set_suffix", "")
        if override_stem:
            file_stem = override_stem
        elif suffix:
            file_stem = f"{repr_set_name}_{suffix}"
        else:
            file_stem = repr_set_name

        file_path = os.path.join(root_path[0], file_stem + ".pkl")
        adapter_role = str(getattr(args, "repr_dataset_adapter_role", "") or "")
        if adapter_role == "replay_labels":
            print("[Step2:cluster-forward-skip] loading anchor labels for GluonTS replay:", file_path)
        else:
            print("Loading repr data from:", file_path)
        with open(file_path, "rb") as f:
            repr_data = pickle.load(f)



        if isinstance(repr_data, list):
            repr_data = np.array(repr_data, dtype=np.float32)  # (n, 84)

        assert repr_data.shape[1] == input_length + prediction_length, \
            f"TSRouter runtime message: {input_length})+prediction({prediction_length})={input_length+prediction_length}"

        self.freq = freq
        self.prediction_length = prediction_length
        self.target_dim = 1
        self.windows = 1

        input_data = repr_data[:, :input_length]
        label_data = repr_data[:, input_length:]

        n = repr_data.shape[0]
        start_time = pd.Timestamp("2000-01-01 00:00:00")

                                      
        input_dataset = ListDataset(
            [
                {
                    "item_id": f"repr_item_{i}",
                    "start": start_time + pd.Timedelta(hours=i),
                    "target": input_data[i],
                    "freq": freq,
                }
                for i in range(n)
            ],
            freq=freq
        )

        label_dataset = ListDataset(
            [
                {
                    "item_id": f"repr_item_{i}",
                    "start": start_time + pd.Timedelta(hours=i + input_length),
                    "target": label_data[i],
                    "freq": freq,
                }
                for i in range(n)
            ],
            freq=freq
        )

                                    
        class TestDataWrapper:
            def __init__(self, input_dataset, label_dataset):
                self.input = input_dataset
                self.label = label_dataset

        self.test_data = TestDataWrapper(input_dataset, label_dataset)
        self.prediction_length = prediction_length
        self.freq = freq
                                                             
        self.name = file_stem + "_freq" + self.freq
        self.past_feat_dynamic_real_dim=0

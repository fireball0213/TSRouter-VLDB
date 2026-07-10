from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import torch

from utils.path_utils import (
    SIMPLETS_SELECTOR_ARTIFACT_DIR,
    build_repr_set_name,
    normalize_advanced_baseline_train_scope,
)


SIMPLETS_TS2VEC_METHOD = "SimpleTS"
SIMPLETS_TS2VEC_METHOD_VERSION = "simplets_v1"


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only option.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a mapping")
    return payload


def _source_repr_set_name(args) -> str:
    source_encoder = str(
        getattr(args, "simplets_ts2vec_source_repr_encoder", "StatsRandomFourier")
        or "StatsRandomFourier"
    ).strip()
    source_args = copy.copy(args)
    source_args.repr_encoder = source_encoder
    # A SimpleTS checkpoint is trained from the main-method Step1 anchors.  The
    # source identity therefore uses the original encoder name, not
    # SimpleTS2Vec and not the repr_v=6 selector display name.
    return build_repr_set_name(source_args)


def _compatibility_error(payload: dict[str, Any], args) -> str | None:
    if str(payload.get("method", "")) != SIMPLETS_TS2VEC_METHOD:
        return f"method={payload.get('method')!r}"
    if str(payload.get("method_version", "")) != SIMPLETS_TS2VEC_METHOD_VERSION:
        return f"method_version={payload.get('method_version')!r}"
    config = payload.get("config")
    if not isinstance(config, dict):
        return "config missing"
    expected_input = int(getattr(args, "repr_input_dim", 0) or 0)
    expected_output = int(getattr(args, "repr_output_dim", 0) or 0)
    if int(config.get("input_dim", -1)) != expected_input:
        return f"input_dim={config.get('input_dim')!r}, expected={expected_input}"
    if int(config.get("embedding_dim", -1)) != expected_output:
        return f"embedding_dim={config.get('embedding_dim')!r}, expected={expected_output}"
    expected_source = _source_repr_set_name(args)
    actual_source = str(payload.get("repr_set_name", ""))
    if actual_source != expected_source:
        return f"repr_set_name={actual_source!r}, expected={expected_source!r}"
    expected_scope = normalize_advanced_baseline_train_scope(
        getattr(args, "advanced_baseline_train_scope", "center")
    )
    actual_scope = normalize_advanced_baseline_train_scope(
        payload.get("advanced_baseline_train_scope", "center")
    )
    if actual_scope != expected_scope:
        return f"advanced_baseline_train_scope={actual_scope!r}, expected={expected_scope!r}"
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        return "state_dict missing"
    return None


def _checkpoint_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_simplets_ts2vec_checkpoint(args) -> tuple[Path, dict[str, Any]]:
    """Resolve and validate the SimpleTS encoder reused by main-method Step1-4."""
    bound_path = str(getattr(args, "simplets_ts2vec_checkpoint", "") or "").strip()
    if bound_path:
        candidates = [Path(bound_path).expanduser().resolve()]
        explicit = True
    else:
        root = Path(SIMPLETS_SELECTOR_ARTIFACT_DIR) / "encoders"
        candidates = sorted(root.glob("*.pt")) if root.exists() else []
        explicit = False

    compatible: list[tuple[Path, dict[str, Any]]] = []
    rejected: list[str] = []
    for path in candidates:
        if not path.is_file():
            rejected.append(f"{path}: file missing")
            continue
        try:
            payload = _torch_load(path)
            reason = _compatibility_error(payload, args)
        except Exception as exc:
            rejected.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        if reason is None:
            compatible.append((path.resolve(), payload))
        else:
            rejected.append(f"{path}: {reason}")

    if not compatible:
        detail = "; ".join(rejected[:5]) or "no checkpoint files found"
        mode = "explicit" if explicit else "auto-discovery"
        raise FileNotFoundError(
            "No compatible SimpleTS TS2Vec checkpoint for main-method encoder "
            f"({mode}). Expected source={_source_repr_set_name(args)!r}, "
            f"input_dim={int(getattr(args, 'repr_input_dim', 0))}, "
            f"embedding_dim={int(getattr(args, 'repr_output_dim', 0))}, "
            f"train_scope={normalize_advanced_baseline_train_scope(getattr(args, 'advanced_baseline_train_scope', 'center'))}. "
            f"Details: {detail}. Run SimpleTS repr_v=6 Step3 first or pass "
            "--simplets_ts2vec_checkpoint explicitly."
        )
    if len(compatible) > 1:
        paths = ", ".join(str(path) for path, _ in compatible)
        raise RuntimeError(
            "Multiple compatible SimpleTS TS2Vec checkpoints found; bind one with "
            f"--simplets_ts2vec_checkpoint. Candidates: {paths}"
        )

    path, payload = compatible[0]
    fingerprint = _checkpoint_fingerprint(path)
    args.simplets_ts2vec_checkpoint = str(path)
    args.simplets_ts2vec_checkpoint_fingerprint = fingerprint
    args.simplets_ts2vec_source_repr_set_name = str(payload.get("repr_set_name", ""))
    args.simplets_ts2vec_train_scope = normalize_advanced_baseline_train_scope(
        payload.get("advanced_baseline_train_scope", "center")
    )
    args._simplets_ts2vec_config = dict(payload["config"])
    return path, payload

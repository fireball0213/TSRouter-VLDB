from __future__ import annotations

import csv
import os
import pickle
import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from encoder.base_encoder import EncoderFactory
from selector.TSRouter_Select.model_zoo_repr import load_metrics_matrix
from utils.io_lock import atomic_pickle_dump
from utils.path_utils import (
    TSROUTER_SAMPLED_REPR_POOL_DIR,
    build_repr_eval_pool_forward_stem,
    build_repr_eval_pool_name,
    build_repr_forward_stem,
    build_repr_set_name,
    get_trained_encoder_path,
    get_tsrouter_repr_forward_dir,
    make_train_bootstrap_args,
)


def _metric_name(args) -> str:
    raw = str(getattr(args, "train_rank_metric", None) or getattr(args, "sgl_rank_metric", None) or "MASE")
    return {"M": "MASE", "S": "sMAPE", "SMAPE": "sMAPE", "C": "CRPS"}.get(raw.upper(), raw)


def _load_rank_targets(args, current_zoo_abbr_order_list) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    boot = make_train_bootstrap_args(args)
    csv_path = os.path.join(get_tsrouter_repr_forward_dir(boot), build_repr_forward_stem(boot) + "_per_sample_results.csv")
    metric = _metric_name(args)
    metric_dict = load_metrics_matrix(csv_path, current_zoo_abbr_order_list, metric=metric)
    model_names = list(metric_dict.keys())
    metrics_matrix = np.stack([metric_dict[m] for m in model_names]).astype(np.float32)
    ranks = np.argsort(metrics_matrix, axis=0).T.astype(np.int64)
    winners = ranks[:, 0].astype(np.int64)
    top3 = ranks[:, : min(3, ranks.shape[1])].astype(np.int64)
    return model_names, ranks, winners, top3


def _load_bootstrap_repr_data(args) -> np.ndarray:
    boot = make_train_bootstrap_args(args)
    path = os.path.join(boot.save_repr_data_path, build_repr_set_name(boot) + ".pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bootstrap repr data missing for Train encoder: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.asarray(data, dtype=np.float32)


def _build_trainable_encoder(args, device):
    train_args = copy.copy(args)
    train_args._allow_missing_trained_encoder = True
    train_args._suppress_trained_encoder_load = True
    encoder, scaler, configs = EncoderFactory.build_encoder(train_args, device=device)
    encoder = encoder.to(device)
    return encoder, scaler, configs


def _embed_array(encoder, array: np.ndarray, batch_size: int, device) -> np.ndarray:
    encoder.eval()
    outs = []
    with torch.no_grad():
        for st in range(0, array.shape[0], batch_size):
            x = torch.from_numpy(array[st: st + batch_size]).float().to(device).unsqueeze(-1)
            z = encoder.forward(x) if hasattr(encoder, "forward") else encoder.encode(x).to(device)
            outs.append(z.detach().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def _supcon_rank_loss(z: torch.Tensor, winners: torch.Tensor, top3: torch.Tensor, temperature: float, top3_weight: float) -> torch.Tensor:
    z = F.normalize(z, p=2, dim=1)
    sim = z @ z.T / float(temperature)
    eye = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    same_winner = winners[:, None].eq(winners[None, :]) & ~eye
    top3_overlap = (top3[:, None, :, None] == top3[None, :, None, :]).any(dim=(2, 3)) & ~eye
    weights = same_winner.float() + float(top3_weight) * (top3_overlap & ~same_winner).float()
    denom_mask = ~eye
    exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True).values.detach()) * denom_mask.float()
    numer = (exp_sim * weights).sum(dim=1)
    denom = exp_sim.sum(dim=1).clamp_min(1e-12)
    valid = numer > 0
    if not torch.any(valid):
        return z.sum() * 0.0
    return -torch.log((numer[valid] / denom[valid]).clamp_min(1e-12)).mean()


def _nearest_eval(train_z: np.ndarray, val_z: np.ndarray, train_winner: np.ndarray, val_winner: np.ndarray, val_top3: np.ndarray) -> dict:
    train_z = train_z / np.maximum(np.linalg.norm(train_z, axis=1, keepdims=True), 1e-12)
    val_z = val_z / np.maximum(np.linalg.norm(val_z, axis=1, keepdims=True), 1e-12)
    nn = np.argmax(val_z @ train_z.T, axis=1)
    pred = train_winner[nn]
    sub1 = float(np.mean(pred == val_winner)) if val_winner.size else float("nan")
    sub3 = float(np.mean([p in set(row.tolist()) for p, row in zip(pred, val_top3)])) if val_winner.size else float("nan")
    return {"SUB1": sub1, "SUB3": sub3}


def _history_path(ckpt_path: str) -> str:
    return os.path.splitext(ckpt_path)[0] + "_history.csv"


def _score_is_better(scores: dict, best: dict, min_delta: float) -> bool:
    cur_sub1 = float(scores.get("SUB1", float("nan")))
    cur_sub3 = float(scores.get("SUB3", float("nan")))
    best_sub1 = float(best.get("SUB1", -1.0))
    best_sub3 = float(best.get("SUB3", -1.0))
    if cur_sub1 > best_sub1 + min_delta:
        return True
    if abs(cur_sub1 - best_sub1) <= min_delta and cur_sub3 > best_sub3 + min_delta:
        return True
    return False


def _write_history_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["epoch", "loss", "val_SUB1", "val_SUB3", "best_epoch", "is_best"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_encoder_if_needed(args, current_zoo_abbr_order_list) -> str:
    path = get_trained_encoder_path(args)
    if os.path.exists(path) and bool(getattr(args, "skip_saved", False)):
        print(f"[TrainEncoder] checkpoint exists, skip training: {path}")
        return path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repr_data = _load_bootstrap_repr_data(args)
    model_names, ranks, winners, top3 = _load_rank_targets(args, current_zoo_abbr_order_list)

    encoder, scaler, configs = _build_trainable_encoder(args, device)
    input_dim = int(configs.input_dim)
    x_all = repr_data[:, :input_dim].astype(np.float32)
    if scaler is not None:
        x_all = scaler.transform(x_all).astype(np.float32)

    rng = np.random.RandomState(int(getattr(args, "repr_encoder_seed", 2025)))
    idx = np.arange(x_all.shape[0])
    rng.shuffle(idx)
    val_ratio = float(getattr(args, "train_encoder_val_ratio", 0.2))
    n_val = max(1, int(round(len(idx) * val_ratio))) if len(idx) > 5 else max(1, len(idx) // 5)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:] if n_val < len(idx) else idx

    batch_size = int(getattr(args, "train_encoder_batch_size", 256))
    epochs = int(getattr(args, "train_encoder_epochs", 30))
    lr = float(getattr(args, "train_encoder_lr", 1e-3))
    temperature = float(getattr(args, "train_encoder_temperature", 0.1))
    top3_weight = float(getattr(args, "train_top3_weight", 0.5))
    early_stop_patience = max(0, int(getattr(args, "train_encoder_early_stop_patience", 0) or 0))
    early_stop_min_delta = max(0.0, float(getattr(args, "train_encoder_early_stop_min_delta", 0.0) or 0.0))
    optimizer = torch.optim.AdamW([p for p in encoder.parameters() if p.requires_grad], lr=lr, weight_decay=float(getattr(args, "train_encoder_weight_decay", 1e-4)))

    x_train = torch.from_numpy(x_all[train_idx]).float()
    winner_train = torch.from_numpy(winners[train_idx]).long()
    top3_train = torch.from_numpy(top3[train_idx]).long()
    best = {"SUB1": -1.0, "SUB3": -1.0, "epoch": -1, "state_dict": None}
    history_rows: list[dict] = []
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        encoder.train()
        order = rng.permutation(len(train_idx))
        losses = []
        for st in range(0, len(order), batch_size):
            b = order[st: st + batch_size]
            xb = x_train[b].to(device).unsqueeze(-1)
            wb = winner_train[b].to(device)
            t3b = top3_train[b].to(device)
            optimizer.zero_grad(set_to_none=True)
            z = encoder.forward(xb)
            loss = _supcon_rank_loss(z, wb, t3b, temperature=temperature, top3_weight=top3_weight)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        train_z = _embed_array(encoder, x_all[train_idx], batch_size, device)
        val_z = _embed_array(encoder, x_all[val_idx], batch_size, device)
        scores = _nearest_eval(train_z, val_z, winners[train_idx], winners[val_idx], top3[val_idx])
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        improved = _score_is_better(scores, best, early_stop_min_delta)
        if improved:
            best = {
                **scores,
                "epoch": epoch,
                "loss": mean_loss,
                "state_dict": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        history_rows.append({
            "epoch": int(epoch),
            "loss": mean_loss,
            "val_SUB1": float(scores["SUB1"]),
            "val_SUB3": float(scores["SUB3"]),
            "best_epoch": int(best["epoch"]),
            "is_best": int(improved),
        })
        best_mark = " *" if improved else ""
        print(f"[TrainEncoder] epoch={epoch}/{epochs} loss={mean_loss:.5f} val_SUB1={scores['SUB1']:.4f} val_SUB3={scores['SUB3']:.4f}{best_mark}")
        if early_stop_patience > 0 and stale_epochs >= early_stop_patience:
            print(
                f"[TrainEncoder] early stop at epoch={epoch}; "
                f"best_epoch={best['epoch']} best_SUB1={best['SUB1']:.4f} best_SUB3={best['SUB3']:.4f}"
            )
            break

    payload = {
        "state_dict": best["state_dict"],
        "meta": {
            "model_names": model_names,
            "metric": _metric_name(args),
            "best_epoch": best["epoch"],
            "best_SUB1": best["SUB1"],
            "best_SUB3": best["SUB3"],
            "repr_encoder_seed": int(getattr(args, "repr_encoder_seed", 2025)),
            "repr_data_seed": int(getattr(args, "repr_data_seed", 2025)),
            "forward_seed": int(getattr(args, "forward_seed", 2025)),
            "encoder_structure": str(getattr(args, "encoder_structure", "")),
            "train_encoder_early_stop_patience": early_stop_patience,
            "train_encoder_early_stop_min_delta": early_stop_min_delta,
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    history_csv = _history_path(path)
    _write_history_csv(history_csv, history_rows)
    print(f"[TrainEncoder] saved checkpoint: {path}")
    print(f"[TrainEncoder] wrote history CSV: {history_csv}")
    return path


def _write_center_forward_from_pool(args, center_member_idx: np.ndarray) -> str:
    boot = make_train_bootstrap_args(args)
    pool_csv = os.path.join(get_tsrouter_repr_forward_dir(boot), build_repr_eval_pool_forward_stem(boot) + "_per_sample_results.csv")
    center_csv = os.path.join(get_tsrouter_repr_forward_dir(args), build_repr_forward_stem(args) + "_per_sample_results.csv")
    if not os.path.exists(pool_csv):
        raise FileNotFoundError(f"Pool forward CSV missing: {pool_csv}")
    os.makedirs(os.path.dirname(center_csv), exist_ok=True)
    idx = np.asarray(center_member_idx, dtype=np.int64)
    with open(pool_csv, "r", newline="") as f:
        rows = list(csv.reader(f))
    out = []
    for r, row in enumerate(rows):
        if r == 0 and len(row) >= 2 and row[0] == "model" and row[1] == "metric":
            out.append(["model", "metric"])
            continue
        values = row[2:]
        picked = [values[int(i)] if 0 <= int(i) < len(values) else "" for i in idx]
        out.append(row[:2] + picked)
    with open(center_csv, "w", newline="") as f:
        csv.writer(f).writerows(out)
    print(f"[TrainEncoder] wrote Train center forward CSV from pool: {center_csv}")
    return center_csv


def rebuild_train_repr_set_from_pool(args) -> None:
    boot = make_train_bootstrap_args(args)
    pool_name = build_repr_eval_pool_name(boot)
    pool_path = os.path.join(TSROUTER_SAMPLED_REPR_POOL_DIR, pool_name + ".pkl")
    if not os.path.exists(pool_path):
        raise FileNotFoundError(f"Pool repr data missing for Train recluster: {pool_path}")
    with open(pool_path, "rb") as f:
        pool_seq = np.asarray(pickle.load(f), dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, scaler, configs = EncoderFactory.build_encoder(args, device=device)
    input_dim = int(configs.input_dim)
    x = pool_seq[:, :input_dim].astype(np.float32)
    if scaler is not None:
        x = scaler.transform(x).astype(np.float32)
    emb = _embed_array(encoder, x, int(getattr(args, "batch_size", 128)), device)

    n_clusters = min(int(getattr(args, "repr_size", 3000)), emb.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=int(getattr(args, "repr_data_seed", 2025)), n_init="auto")
    labels = kmeans.fit_predict(emb)
    centers = kmeans.cluster_centers_
    center_member_idx = np.full(n_clusters, -1, dtype=np.int64)
    for c in range(n_clusters):
        ids = np.where(labels == c)[0]
        if ids.size == 0:
            continue
        d = np.sum((emb[ids] - centers[c]) ** 2, axis=1)
        center_member_idx[c] = int(ids[int(np.argmin(d))])
    valid = center_member_idx >= 0
    center_member_idx = center_member_idx[valid]
    selected = pool_seq[center_member_idx].astype(np.float32)

    repr_set_name = build_repr_set_name(args)
    os.makedirs(args.save_repr_data_path, exist_ok=True)
    atomic_pickle_dump(selected, os.path.join(args.save_repr_data_path, repr_set_name + ".pkl"))
    pool_meta_path = os.path.join(TSROUTER_SAMPLED_REPR_POOL_DIR, pool_name + "_meta.pkl")
    if not os.path.exists(pool_meta_path):
        atomic_pickle_dump(
            {
                "meta_role": "repr_candidate_pool_sidecar",
                "pool_name": pool_name,
                "seq_shape": tuple(pool_seq.shape),
                "source_unique": [],
                "per_source": [],
                "note": "minimal sidecar created by Train recluster; no encoder embeddings stored",
            },
            pool_meta_path,
        )
    anchor_meta = {
        "repr_set_name": repr_set_name,
        "meta_role": "repr_anchor_sidecar",
        "pool_name": pool_name,
        "pool_path": pool_path,
        "pool_meta_path": pool_meta_path,
        "seq_shape": tuple(selected.shape),
        "center_member_idx_in_pool": center_member_idx.astype(np.int64),
        "encoder_type": "Train",
        "repr_encoder": str(getattr(args, "repr_encoder", "Train")),
        "sample_mode": str(getattr(args, "sample_mode", "cluster_nearest")),
    }
    atomic_pickle_dump(anchor_meta, os.path.join(args.save_repr_data_path, repr_set_name + "_meta.pkl"))
    _write_center_forward_from_pool(args, center_member_idx)
    print(f"[TrainEncoder] rebuilt Train repr set: {repr_set_name}, centers={selected.shape[0]}")


def ensure_trained_encoder_artifacts(args, current_zoo_abbr_order_list) -> None:
    train_encoder_if_needed(args, current_zoo_abbr_order_list)
    repr_path = os.path.join(args.save_repr_data_path, build_repr_set_name(args) + ".pkl")
    center_csv = os.path.join(get_tsrouter_repr_forward_dir(args), build_repr_forward_stem(args) + "_per_sample_results.csv")
    if bool(getattr(args, "skip_saved", False)) and os.path.exists(repr_path) and os.path.exists(center_csv):
        print(f"[TrainEncoder] Train repr artifacts exist, skip recluster: {build_repr_set_name(args)}")
        return
    rebuild_train_repr_set_from_pool(args)

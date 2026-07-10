from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_stats_features(x_2d: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    """
    Fixed statistical features for rank-oriented random encoders.

    Args:
        x_2d: (B, L) aligned time windows.
        normalize: row-wise layer normalization over the statistic vector.

    Returns:
        stats: (B, 16)
    """
    x = torch.nan_to_num(x_2d.float(), nan=0.0, posinf=0.0, neginf=0.0)
    bsz, length = x.shape
    if length <= 1:
        diff = torch.zeros(bsz, 1, device=x.device, dtype=x.dtype)
    else:
        diff = x[:, 1:] - x[:, :-1]

    q25 = torch.quantile(x, 0.25, dim=1)
    q50 = torch.quantile(x, 0.50, dim=1)
    q75 = torch.quantile(x, 0.75, dim=1)
    iqr = q75 - q25

    t = torch.linspace(-0.5, 0.5, steps=length, device=x.device, dtype=x.dtype)
    t = t.unsqueeze(0)
    t_var = torch.mean(t * t).clamp_min(1e-6)
    slope = torch.mean((x - x.mean(dim=1, keepdim=True)) * t, dim=1) / t_var

    if diff.shape[1] <= 1:
        diff_std = torch.zeros(bsz, device=x.device, dtype=x.dtype)
    else:
        diff_std = diff.std(dim=1, unbiased=False)

    sign = torch.sign(diff)
    if sign.shape[1] <= 1:
        sign_change_rate = torch.zeros(bsz, device=x.device, dtype=x.dtype)
    else:
        sign_change_rate = (sign[:, 1:] * sign[:, :-1] < 0).float().mean(dim=1)

    stats = torch.stack(
        [
            x.mean(dim=1),
            x.std(dim=1, unbiased=False),
            x.min(dim=1).values,
            x.max(dim=1).values,
            q25,
            q50,
            q75,
            iqr,
            x[:, 0],
            x[:, -1],
            slope,
            diff.abs().mean(dim=1),
            diff_std,
            diff.abs().max(dim=1).values,
            sign_change_rate,
            (x * x).mean(dim=1),
        ],
        dim=1,
    )
    stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
    if normalize:
        stats = F.layer_norm(stats, (stats.shape[1],))
    return stats


def stats_feature_dim() -> int:
    return 16

from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.base_encoder import BaseEncoder


class _SamePadConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (int(kernel_size) - 1) * int(dilation) // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                int(channels),
                int(channels),
                kernel_size=int(kernel_size),
                padding=padding,
                dilation=int(dilation),
            ),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(int(channels), int(channels), kernel_size=1),
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=int(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if out.shape[-1] != x.shape[-1]:
            out = out[..., -x.shape[-1]:]
        return self.norm(x + out)


class TS2VecEncoder(BaseEncoder):
    """
    Lightweight TS2Vec-style encoder used by SimpleTS.

    The module exposes sequence embeddings for hierarchical contrastive training
    and pooled embeddings for TSRouter lookup/classification. It intentionally
    keeps the dependency surface local instead of importing an external TS2Vec
    package.
    """

    def __init__(self, configs, device="cuda", **kwargs):
        super().__init__()
        self.configs = configs
        self.device = torch.device(device)
        self.input_len = int(getattr(configs, "input_dim", 512))
        self.embed_dim = int(getattr(configs, "embedding_dim", 256))
        hidden_dim = int(getattr(configs, "ts2vec_hidden_dim", max(64, self.embed_dim)))
        depth = int(getattr(configs, "ts2vec_depth", 4))
        kernel_size = int(getattr(configs, "ts2vec_kernel_size", 3))
        dropout = float(getattr(configs, "ts2vec_dropout", 0.1))
        self.normalize = bool(getattr(configs, "ts_l2norm", True))

        self.input_proj = nn.Conv1d(1, hidden_dim, kernel_size=1)
        blocks = []
        for layer_idx in range(max(1, depth)):
            dilation = 2 ** layer_idx
            blocks.append(_SamePadConvBlock(hidden_dim, kernel_size, dilation, dropout))
        self.encoder = nn.Sequential(*blocks)
        self.output_proj = nn.Conv1d(hidden_dim, self.embed_dim, kernel_size=1)
        self.to(self.device)

    @property
    def embedding_dim(self) -> int:
        return int(self.embed_dim)

    def _to_tensor(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(series_data, np.ndarray):
            return torch.from_numpy(series_data).float()
        return series_data.float()

    def _fix_length(self, x_2d: torch.Tensor) -> torch.Tensor:
        batch, length = x_2d.shape
        if length == self.input_len:
            return x_2d
        if length > self.input_len:
            return x_2d[:, -self.input_len:]
        pad = torch.zeros(
            batch,
            self.input_len - length,
            device=x_2d.device,
            dtype=x_2d.dtype,
        )
        return torch.cat([pad, x_2d], dim=1)

    def _prepare_series(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        x = self._to_tensor(series_data)
        if x.dim() == 3:
            x = x.mean(dim=-1)
        elif x.dim() != 2:
            raise ValueError(f"[TS2VecEncoder] Expected 2D/3D input, got shape={tuple(x.shape)}")
        x = self._fix_length(x.to(self.device))
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (x - mean) / std

    def encode_sequence(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        x = self._prepare_series(series_data).unsqueeze(1)
        z = self.output_proj(self.encoder(self.input_proj(x))).transpose(1, 2)
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        z = self.encode_sequence(series_data)
        emb = torch.max(z, dim=1).values
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb

    @torch.no_grad()
    def encode(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        return self.forward(series_data).detach().cpu()


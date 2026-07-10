from __future__ import annotations

import math
from typing import Iterable, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.base_encoder import BaseEncoder
from encoder.baseline.random_stats_features import compute_stats_features, stats_feature_dim


def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    return nn.GELU()


def _as_int_list(value, default: Iterable[int]) -> list[int]:
    if value is None:
        return [int(v) for v in default]
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    return [int(v) for v in value]


class _RandomTSBaseEncoder(BaseEncoder):
    def __init__(self, configs, device="cuda", **kwargs):
        super().__init__()
        self.configs = configs
        self.device = torch.device(device)
        self.input_len = int(getattr(configs, "input_dim", 512))
        self.embed_dim = int(getattr(configs, "embedding_dim", 256))
        self.normalize = bool(getattr(configs, "ts_l2norm", True))
        self.instance_norm = bool(getattr(configs, "ts_instance_norm", False))
        self.stats_fusion = (getattr(configs, "random_stats_fusion", "none") or "none").lower()
        self.stats_normalize = bool(getattr(configs, "random_stats_normalize", True))
        self.trainable = str(getattr(configs, "encoder_type", "")).lower() == "train" or bool(getattr(configs, "encoder_trainable", False))
        self._stats_dim = stats_feature_dim()
        self._early_channels = 1 + self._stats_dim if self.stats_fusion == "early" else 1

    @property
    def embedding_dim(self) -> int:
        return int(self.embed_dim)

    def _to_tensor(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(series_data, np.ndarray):
            x = torch.from_numpy(series_data).float()
        else:
            x = series_data.float()
        return x

    def _prepare_series_and_stats(self, series_data: Union[np.ndarray, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._prepare_series(series_data)
        stats = compute_stats_features(x, normalize=self.stats_normalize)
        return x, stats

    def _as_conv_input(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        x_ch = x.unsqueeze(1)
        if self.stats_fusion != "early":
            return x_ch
        stats_ch = stats.unsqueeze(-1).expand(-1, -1, x.shape[1])
        return torch.cat([x_ch, stats_ch], dim=1)

    def _init_late_fusion_layers(self):
        if self.stats_fusion == "late":
            self.stats_proj = nn.Linear(self._stats_dim, self.embed_dim)
            self.late_fuse = nn.Linear(self.embed_dim * 2, self.embed_dim)
        else:
            self.stats_proj = None
            self.late_fuse = None

    def _apply_late_fusion(self, emb: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        if self.stats_fusion != "late":
            return emb
        stats_emb = self.stats_proj(stats)
        return self.late_fuse(torch.cat([emb, stats_emb], dim=1))

    def _fix_length(self, x_2d: torch.Tensor) -> torch.Tensor:
        batch, length = x_2d.shape
        if length == self.input_len:
            return x_2d
        if length > self.input_len:
            return x_2d[:, -self.input_len:]
        pad = torch.zeros(batch, self.input_len - length, device=x_2d.device, dtype=x_2d.dtype)
        return torch.cat([pad, x_2d], dim=1)

    def _prepare_series(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        x = self._to_tensor(series_data)
        if x.dim() == 3:
            x = x.mean(dim=-1)
        elif x.dim() != 2:
            raise ValueError(f"[{self.__class__.__name__}] Expected 2D/3D input, got shape={tuple(x.shape)}")

        x = self._fix_length(x.to(self.device))
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.instance_norm:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            x = (x - mean) / std
        return x

    def _finish(self, emb: torch.Tensor) -> torch.Tensor:
        emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb if self.trainable else emb.cpu()

    @torch.no_grad()
    def encode(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        return self.forward(series_data).cpu()

    def _freeze_random_weights(self):
        if self.trainable:
            self.to(self.device)
            self.train()
            return
        for p in self.parameters():
            p.requires_grad_(False)
        self.to(self.device)
        self.eval()


class RandomPatchEncoder(_RandomTSBaseEncoder):
    """
    Random Patch encoder for fixed-length time windows.

    The series is split into overlapping patches, projected to random token
    vectors, mixed token-wise, then pooled by mean/std/max/last before the final
    random projection. This keeps RandomMLP's no-training baseline property but
    gives the encoder local temporal receptive fields.
    """

    def __init__(self, configs, device="cuda", **kwargs):
        super().__init__(configs, device=device, **kwargs)
        patch_len = int(getattr(configs, "patch_len", kwargs.get("patch_len", 16)))
        patch_stride = int(getattr(configs, "patch_stride", kwargs.get("patch_stride", 8)))
        self.patch_len = max(1, min(patch_len, self.input_len))
        self.patch_stride = max(1, patch_stride)
        self.hidden_dim = int(getattr(configs, "patch_hidden_dim", kwargs.get("patch_hidden_dim", 256)))
        depth = int(getattr(configs, "patch_depth", kwargs.get("patch_depth", 2)))
        dropout = float(getattr(configs, "patch_dropout", kwargs.get("patch_dropout", 0.0)))
        act_name = getattr(configs, "patch_activation", kwargs.get("patch_activation", "gelu"))

        self.n_patches = 1 + max(0, self.input_len - self.patch_len) // self.patch_stride
        self.patch_proj = nn.Linear(self.patch_len * self._early_channels, self.hidden_dim)
        blocks = []
        for _ in range(max(1, depth)):
            blocks.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            blocks.append(_activation(act_name))
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
        self.token_mixer = nn.Sequential(*blocks)
        pos = torch.randn(1, self.n_patches, self.hidden_dim) / math.sqrt(max(1, self.hidden_dim))
        self.register_buffer("pos_embed", pos)
        self.out = nn.Linear(self.hidden_dim * 4, self.embed_dim)
        self._init_late_fusion_layers()
        self._freeze_random_weights()

    def forward(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        x, stats = self._prepare_series_and_stats(series_data)
        x_ch = self._as_conv_input(x, stats)
        patches = x_ch.unfold(dimension=2, size=self.patch_len, step=self.patch_stride)
        patches = patches.permute(0, 2, 1, 3).reshape(patches.shape[0], patches.shape[2], -1)
        if patches.shape[1] != self.n_patches:
            patches = patches[:, : self.n_patches, :]
        tok = self.patch_proj(patches) + self.pos_embed[:, : patches.shape[1], :]
        tok = self.token_mixer(tok)
        pooled = torch.cat(
            [
                tok.mean(dim=1),
                tok.std(dim=1, unbiased=False),
                tok.max(dim=1).values,
                tok[:, -1, :],
            ],
            dim=1,
        )
        emb = self.out(pooled)
        emb = self._apply_late_fusion(emb, stats)
        return self._finish(emb)


class RandomConvEncoder(_RandomTSBaseEncoder):
    """
    Random dilated Conv1d encoder.

    Dilated convolutions provide local-to-medium temporal receptive fields while
    keeping the encoder lightweight and fully random.
    """

    def __init__(self, configs, device="cuda", **kwargs):
        super().__init__(configs, device=device, **kwargs)
        channels = int(getattr(configs, "conv_channels", kwargs.get("conv_channels", 128)))
        kernel_size = int(getattr(configs, "conv_kernel_size", kwargs.get("conv_kernel_size", 5)))
        dilations = _as_int_list(getattr(configs, "conv_dilations", None), default=[1, 2, 4, 8])
        dropout = float(getattr(configs, "conv_dropout", kwargs.get("conv_dropout", 0.0)))
        act_name = getattr(configs, "conv_activation", kwargs.get("conv_activation", "gelu"))

        layers = []
        in_ch = self._early_channels
        for dilation in dilations:
            padding = (kernel_size // 2) * int(dilation)
            layers.append(
                nn.Conv1d(
                    in_ch,
                    channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=int(dilation),
                )
            )
            layers.append(nn.GroupNorm(num_groups=1, num_channels=channels, affine=False))
            layers.append(_activation(act_name))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = channels
        self.conv = nn.Sequential(*layers)
        self.out = nn.Linear(channels * 4, self.embed_dim)
        self._init_late_fusion_layers()
        self._freeze_random_weights()

    def forward(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        x, stats = self._prepare_series_and_stats(series_data)
        x = self._as_conv_input(x, stats)
        feat = self.conv(x)
        pooled = torch.cat(
            [
                feat.mean(dim=-1),
                feat.std(dim=-1, unbiased=False),
                feat.max(dim=-1).values,
                feat[:, :, -1],
            ],
            dim=1,
        )
        emb = self.out(pooled)
        emb = self._apply_late_fusion(emb, stats)
        return self._finish(emb)


class RandomInceptionEncoder(_RandomTSBaseEncoder):
    """
    Random multi-scale Conv1d filter-bank encoder.

    Parallel kernels behave like random shape/motif detectors over different
    temporal scales. Global pooling turns their responses into a compact task
    representation.
    """

    def __init__(self, configs, device="cuda", **kwargs):
        super().__init__(configs, device=device, **kwargs)
        branch_channels = int(getattr(configs, "inception_branch_channels", kwargs.get("inception_branch_channels", 48)))
        kernels = _as_int_list(getattr(configs, "inception_kernels", None), default=[3, 5, 9, 17])
        dropout = float(getattr(configs, "inception_dropout", kwargs.get("inception_dropout", 0.0)))
        act_name = getattr(configs, "inception_activation", kwargs.get("inception_activation", "gelu"))

        self.branches = nn.ModuleList()
        for kernel in kernels:
            kernel = max(1, int(kernel))
            padding = kernel // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(self._early_channels, branch_channels, kernel_size=kernel, padding=padding),
                    nn.GroupNorm(num_groups=1, num_channels=branch_channels, affine=False),
                    _activation(act_name),
                )
            )
        total_channels = branch_channels * len(self.branches)
        self.mix = nn.Sequential(
            nn.Conv1d(total_channels, total_channels, kernel_size=1),
            nn.GroupNorm(num_groups=1, num_channels=total_channels, affine=False),
            _activation(act_name),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.out = nn.Linear(total_channels * 4, self.embed_dim)
        self._init_late_fusion_layers()
        self._freeze_random_weights()

    def forward(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        x, stats = self._prepare_series_and_stats(series_data)
        x = self._as_conv_input(x, stats)
        branches = [branch(x) for branch in self.branches]
        feat = self.mix(torch.cat(branches, dim=1))
        pooled = torch.cat(
            [
                feat.mean(dim=-1),
                feat.std(dim=-1, unbiased=False),
                feat.max(dim=-1).values,
                feat.min(dim=-1).values,
            ],
            dim=1,
        )
        emb = self.out(pooled)
        emb = self._apply_late_fusion(emb, stats)
        return self._finish(emb)


class _RandomTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, act_name: str):
        super().__init__()
        padding = (kernel_size // 2) * int(dilation)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=int(dilation)),
            nn.GroupNorm(num_groups=1, num_channels=channels, affine=False),
            _activation(act_name),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=int(dilation)),
            nn.GroupNorm(num_groups=1, num_channels=channels, affine=False),
        )
        self.act = _activation(act_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if y.shape[-1] != x.shape[-1]:
            y = y[..., -x.shape[-1]:]
        return self.act(x + y)


class RandomTCNEncoder(_RandomTSBaseEncoder):
    """
    Random temporal convolutional network with residual dilated blocks.

    Compared with RandomConv, the residual stack keeps multi-layer random
    filters stable and gives a wider effective receptive field.
    """

    def __init__(self, configs, device="cuda", **kwargs):
        super().__init__(configs, device=device, **kwargs)
        channels = int(getattr(configs, "tcn_channels", kwargs.get("tcn_channels", 128)))
        kernel_size = int(getattr(configs, "tcn_kernel_size", kwargs.get("tcn_kernel_size", 3)))
        dilations = _as_int_list(getattr(configs, "tcn_dilations", None), default=[1, 2, 4, 8, 16])
        dropout = float(getattr(configs, "tcn_dropout", kwargs.get("tcn_dropout", 0.0)))
        act_name = getattr(configs, "tcn_activation", kwargs.get("tcn_activation", "gelu"))

        self.input_proj = nn.Conv1d(self._early_channels, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                _RandomTCNBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    act_name=act_name,
                )
                for dilation in dilations
            ]
        )
        self.out = nn.Linear(channels * 4, self.embed_dim)
        self._init_late_fusion_layers()
        self._freeze_random_weights()

    def forward(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        x, stats = self._prepare_series_and_stats(series_data)
        x = self._as_conv_input(x, stats)
        feat = self.blocks(self.input_proj(x))
        pooled = torch.cat(
            [
                feat.mean(dim=-1),
                feat.std(dim=-1, unbiased=False),
                feat.max(dim=-1).values,
                feat[:, :, -1],
            ],
            dim=1,
        )
        emb = self.out(pooled)
        emb = self._apply_late_fusion(emb, stats)
        return self._finish(emb)


class RandomFourierEncoder(_RandomTSBaseEncoder):
    """
    Random Fourier/Wavelet encoder.

    It mixes low-frequency FFT magnitude, random Fourier projections, and
    Haar-like multi-scale difference energies before a random projection.
    """

    def __init__(self, configs, device="cuda", **kwargs):
        super().__init__(configs, device=device, **kwargs)
        self.freq_bins = int(getattr(configs, "fourier_bins", kwargs.get("fourier_bins", 64)))
        self.random_features = int(getattr(configs, "fourier_random_features", kwargs.get("fourier_random_features", 128)))
        self.wavelet_scales = _as_int_list(getattr(configs, "wavelet_scales", None), default=[2, 4, 8, 16, 32])
        hidden_dim = int(getattr(configs, "fourier_hidden_dim", kwargs.get("fourier_hidden_dim", 256)))
        dropout = float(getattr(configs, "fourier_dropout", kwargs.get("fourier_dropout", 0.0)))
        act_name = getattr(configs, "fourier_activation", kwargs.get("fourier_activation", "gelu"))

        freq_count = min(self.freq_bins, self.input_len // 2 + 1)
        rand_weight = torch.randn(self.input_len, self.random_features) / math.sqrt(max(1, self.input_len))
        rand_phase = torch.rand(self.random_features) * (2.0 * math.pi)
        self.register_buffer("rand_weight", rand_weight)
        self.register_buffer("rand_phase", rand_phase)

        base_dim = freq_count + self.random_features * 2 + len(self.wavelet_scales) * 2
        if self.stats_fusion == "early":
            base_dim += self._stats_dim
        self.freq_count = freq_count
        self.net = nn.Sequential(
            nn.Linear(base_dim, hidden_dim),
            _activation(act_name),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, self.embed_dim),
        )
        self._init_late_fusion_layers()
        self._freeze_random_weights()

    def _wavelet_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for scale in self.wavelet_scales:
            scale = int(max(1, min(scale, x.shape[1] - 1)))
            d = x[:, scale:] - x[:, :-scale]
            feats.append(d.abs().mean(dim=1))
            feats.append(d.std(dim=1, unbiased=False))
        return torch.stack(feats, dim=1)

    def forward(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        x, stats = self._prepare_series_and_stats(series_data)
        fft_mag = torch.fft.rfft(x, dim=1).abs()
        fft_mag = torch.log1p(fft_mag[:, : self.freq_count])

        proj = x @ self.rand_weight + self.rand_phase
        rand_fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)
        wavelet = self._wavelet_features(x)

        feat = torch.cat([fft_mag, rand_fourier, wavelet], dim=1)
        if self.stats_fusion == "early":
            feat = torch.cat([feat, stats], dim=1)
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        emb = self.net(feat)
        emb = self._apply_late_fusion(emb, stats)
        return self._finish(emb)

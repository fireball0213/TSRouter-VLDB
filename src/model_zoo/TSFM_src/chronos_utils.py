# model_zoo/chronos_utils.py
     
                                                      
                                                     

from typing import Any, List

import torch
from torch.utils.data import Dataset

from chronos.chronos import MeanScaleUniformBins

__all__ = [
    "SeriesDataset",
    "identity_collate",
]

                                                                                       

          
_orig_input_transform = MeanScaleUniformBins._input_transform
_orig_append_eos = MeanScaleUniformBins._append_eos_token
_orig_output_transform = MeanScaleUniformBins.output_transform


def _patched_input_transform(self, context: torch.Tensor, scale=None):
    if self.boundaries.device != context.device:
        self.boundaries = self.boundaries.to(context.device)
    return _orig_input_transform(self, context, scale)


def _patched_append_eos(self, token_ids: torch.Tensor, attention_mask: torch.Tensor):
    device = token_ids.device
    batch_size = token_ids.shape[0]

    eos_tokens = torch.full(
        (batch_size, 1),
        fill_value=self.config.eos_token_id,
        device=device,
    )
    eos_mask = torch.full(
        (batch_size, 1),
        fill_value=True,
        device=device,
    )

    attention_mask = attention_mask.to(device)

    token_ids = torch.concat((token_ids, eos_tokens), dim=1)
    attention_mask = torch.concat((attention_mask, eos_mask), dim=1)
    return token_ids, attention_mask


def _patched_output_transform(self, samples: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    target_device = samples.device if samples.device.type != "cpu" else scale.device

    if self.centers.device != target_device:
        self.centers = self.centers.to(target_device)
    if scale.device != target_device:
        scale = scale.to(target_device)

    return _orig_output_transform(self, samples.to(target_device), scale)


                                
MeanScaleUniformBins._input_transform = _patched_input_transform
MeanScaleUniformBins._append_eos_token = _patched_append_eos
MeanScaleUniformBins.output_transform = _patched_output_transform

                                                                                   


def identity_collate(batch: List[Any]) -> List[Any]:
    return batch


class SeriesDataset(Dataset):

    def __init__(self, raw):
        self.raw = list(raw)

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx: int):
        return self.raw[idx]

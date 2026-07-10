import os
import sys
from pathlib import Path
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import cuda

from gluonts.itertools import batcher
from gluonts.model.forecast import QuantileForecast

from model_zoo.base_model import BaseModel


_TSFMSRC_DIR = Path(__file__).resolve().parent / "TSFM_src"
if _TSFMSRC_DIR.exists():
    _p = str(_TSFMSRC_DIR)
    if _p not in sys.path:
        sys.path.insert(0, _p)                   
else:
    raise FileNotFoundError(f"[Moirai] TSFM_src directory not found: {_TSFMSRC_DIR}")

def _get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if cuda.is_available() else "cpu")
    return torch.device(device)


                                                     

class Moirai2Model(BaseModel):

    def __init__(self, args, module_name, model_name, model_local_path):
        self.args = args
        self.module_name = module_name
        self.model_name = model_name
        self.model_local_path = model_local_path
        self.output_dir = os.path.join(self.args.output_dir, self.model_name)

        super().__init__(self.model_name, args, self.output_dir)

    def get_predictor(self, dataset, batch_size):
                             
        if self.args.fix_context_len:
            context_length = self.args.context_len
        else:
            context_length = 1680                

        print(
            f"[Moirai2] context_len={context_length}, "
            f"batch_size={batch_size}, "
            f"freq_used=False, impute_missing=False"
        )

                                              
        prediction_length = dataset.prediction_length
        target_dim = 1
        # target_dim = getattr(dataset, "target_dim", 1)
        feat_dynamic_real_dim = getattr(dataset, "feat_dynamic_real_dim", 0)
        past_feat_dynamic_real_dim = getattr(dataset, "past_feat_dynamic_real_dim", 0)


                                           
        predictor = Moirai2QuantilePredictor(
            model_path=self.model_local_path,
            prediction_length=prediction_length,
            context_length=context_length,
            target_dim=target_dim,
            feat_dynamic_real_dim=feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
            batch_size=batch_size,
            device="auto",
            quantile_levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        )
        return predictor


class Moirai2QuantilePredictor:

    def __init__(
        self,
        model_path: str,
        prediction_length: int,
        context_length: int,
        target_dim: int,
        feat_dynamic_real_dim: int,
        past_feat_dynamic_real_dim: int,
        batch_size: int,
        device: str,
        quantile_levels: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    ):
        self.model_path = model_path
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.target_dim = target_dim
        self.feat_dynamic_real_dim = feat_dynamic_real_dim
        self.past_feat_dynamic_real_dim = past_feat_dynamic_real_dim
        self.batch_size = batch_size
        self.device =_get_device(device)
        self.quantile_levels = quantile_levels

        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

        self.model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(self.model_path,
            local_files_only=True),
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            # context_length=1680,
            target_dim=self.target_dim,
            feat_dynamic_real_dim=self.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=self.past_feat_dynamic_real_dim,
        ).to(self.device)

        self.model.eval()

    # =========================================================
                                                        
    # =========================================================
    # def _format_target(self, raw_target) -> np.ndarray:
    #     """
                            
    #     - 1D: (T,)
                             
                                                    
                
    #     - float32 numpy: (context_length, target_dim)
    #     """
                                            
    #     if isinstance(raw_target, torch.Tensor):
    #         arr = raw_target.detach().cpu().numpy()
    #     elif isinstance(raw_target, (list, tuple)):
                                                 
                                            
    #         try:
    #             elems = [np.asarray(x, dtype=np.float32) for x in raw_target]
    #             arr = np.stack(elems, axis=0)  # (C, T)
    #         except Exception:
                                    
    #             arr = np.asarray(raw_target, dtype=np.float32)
    #     else:
    #         arr = np.asarray(raw_target, dtype=np.float32)
    #
                           
    #     if arr.dtype == np.object_:
    #         arr = arr.astype(np.float32)
    #
                                           
    #     if arr.ndim == 1:
    #         # (T,) -> (T,1)
    #         arr = arr[:, None]
    #     elif arr.ndim == 2:
                                              
                                 
    #         if arr.shape[1] == self.target_dim:
    #             # (T,C) OK
    #             pass
    #         elif arr.shape[0] == self.target_dim:
    #             # (C,T) -> (T,C)
    #             arr = arr.T
    #         else:
                                              
    #             raise ValueError(
    #                 f"[Moirai2] target shape mismatch: got {arr.shape}, target_dim={self.target_dim}. "
    #                 f"Cannot infer (C,T) vs (T,C)."
    #             )
    #     else:
    #         raise ValueError(f"[Moirai2] unsupported target ndim={arr.ndim}, shape={arr.shape}")
    #
                                                            
    #     if np.isnan(arr).any():
    #         for c in range(arr.shape[1]):
    #             x = arr[:, c]
    #             nan = np.isnan(x)
    #             if nan.all():
    #                 x[:] = 0.0
    #             else:
    #                 # ffill
    #                 last = np.nan
    #                 for i in range(len(x)):
    #                     if np.isnan(x[i]):
    #                         if not np.isnan(last):
    #                             x[i] = last
    #                     else:
    #                         last = x[i]
    #                 # bfill for leading NaN
    #                 if np.isnan(x[0]):
    #                     first_valid = x[~np.isnan(x)][0]
    #                     x[np.isnan(x)] = first_valid
    #             arr[:, c] = x
    #
                                                                     
    #     T = arr.shape[0]
    #     if T >= self.context_length:
    #         arr = arr[-self.context_length:, :]
    #     else:
    #         pad_len = self.context_length - T
                                                       
    #         if T > 0:
    #             pad_block = np.repeat(arr[0:1, :], repeats=pad_len, axis=0)
    #         else:
    #             pad_block = np.zeros((pad_len, arr.shape[1]), dtype=np.float32)
    #         arr = np.concatenate([pad_block, arr], axis=0)
    #
                                                    
    #     if arr.shape != (self.context_length, self.target_dim):
    #         raise ValueError(
    #             f"[Moirai2] formatted target shape error: got {arr.shape}, "
    #             f"expected {(self.context_length, self.target_dim)}"
    #         )
    #
    #     return arr

    def predict(self, test_data_input: List[dict]) -> List[QuantileForecast]:

        forecast_quantiles = []
        for batch in batcher(test_data_input, batch_size=self.batch_size):
            past_target = [entry["target"] for entry in batch]
            # past_target = []
            # for entry in batch:
            #     try:
            #         past_target.append(self._format_target(entry["target"]))
            #     except Exception as e:
                                               
            #         rt = entry.get("target", None)
            #         shape = getattr(rt, "shape", None)
            #         print(
            #             f"[Moirai2] format_target failed. "
            #             f"type={type(rt)}, shape={shape}, target_dim={self.target_dim}, "
            #             f"context_length={self.context_length}"
            #         )
            #         raise
            forecasts = self.model.predict(past_target)  # (B, Q, H, C)
            forecast_quantiles.append(forecasts)

        forecast_quantiles = np.concatenate(forecast_quantiles, axis=0)

        quantile_forecasts: List[QuantileForecast] = []
        q_keys = list(map(str, self.quantile_levels))

        for item, ts in zip(forecast_quantiles, test_data_input):
                                                              
            forecast_start_date = ts["start"] + len(ts["target"])
            quantile_forecasts.append(
                QuantileForecast(
                    item_id=ts.get("item_id"),
                    forecast_arrays=item,
                    start_date=forecast_start_date,
                    forecast_keys=q_keys,
                )
            )

        return quantile_forecasts

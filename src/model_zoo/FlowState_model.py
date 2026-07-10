# -*- coding: utf-8 -*-
"""
FlowState model integration.
"""

import os
import sys
from pathlib import Path
import logging
from typing import List

import numpy as np
import torch
from torch import cuda
from tqdm.auto import tqdm

from gluonts.model.forecast import QuantileForecast
from gluonts.itertools import batcher

from model_zoo.base_model import BaseModel, pretty_names, dataset_properties_map
from utils.missing import fill_missing

                                                              
_TSFMSRC_DIR = Path(__file__).resolve().parent / "TSFM_src"
if _TSFMSRC_DIR.exists():
    _p = str(_TSFMSRC_DIR)
    if _p not in sys.path:
        sys.path.insert(0, _p)                   
else:
    raise FileNotFoundError(f"[FlowState] TSFM_src directory not found: {_TSFMSRC_DIR}")

from tsfm_public import FlowStateForPrediction
from tsfm_public.models.flowstate.utils.utils import get_fixed_factor



class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter: str):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()


gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(
    WarningFilter("The mean prediction is not stored in the forecast data")
)



class FlowStateModel(BaseModel):

    def __init__(self, args, module_name, model_name, model_local_path):
        self.args = args
        self.module_name = module_name
        self.model_name = model_name
        self.model_local_path = model_local_path
        self.output_dir = os.path.join(self.args.output_dir, self.model_name)

        super().__init__(self.model_name, args, self.output_dir)

    def get_predictor(self, dataset, batch_size):
        ds_name_raw = dataset.name.split("/")[0].lower()
        ds_key = pretty_names.get(ds_name_raw, ds_name_raw)
        domain = dataset_properties_map.get(ds_key, {}).get("domain", None)

        predictor = FlowStatePredictor(
            config=self.args,
            batch_size=batch_size,
            model_path=self.model_local_path,
            prediction_length=dataset.prediction_length,
            ds_freq=dataset.freq,
            domain=domain
        )
        return predictor



class FlowStatePredictor:

    def __init__(
        self,
        config,
        batch_size: int,
        model_path: str,
        prediction_length: int,
        ds_freq: str,
        domain: str = None,
        *args,
        **kwargs,
    ):
        self.config = config
        self.batch_size = batch_size
        self.model_path = model_path
        self.prediction_length = prediction_length
        self.ds_freq = ds_freq
        self.domain = domain
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")

        self.impute_missing = False

        try:
            self.model = FlowStateForPrediction.from_pretrained(model_path).to(self.device)
        except Exception as e:
            print(f"Warning: Failed to load from {model_path} ({e}). Trying ibm-research/flowstate")
            self.model = FlowStateForPrediction.from_pretrained("ibm-research/flowstate").to(self.device)
        
        self.model.eval()
        if getattr(self.config, "fix_context_len", False):
            self.context_length = self.config.context_len
        else:
            self.context_length = self.model.config.context_length

        self.quantiles = self.model.config.quantiles
        
        try:
            self.scale_factor = get_fixed_factor(self.ds_freq, self.domain)
        except Exception as e:
            print(f"Warning: Failed to get scale_factor for {self.ds_freq}/{self.domain} ({e}). Using default 1.0")
            self.scale_factor = 1.0

        print(
            f"[FlowState] context_len={self.context_length}, "
            f"batch_size={self.batch_size}, "
            f"prediction_length={self.prediction_length}, "
            f"freq={self.ds_freq}, "
            f"domain={self.domain}, "
            f"scale_factor={self.scale_factor}, "
            f"impute_missing={self.impute_missing}"
        )

    def predict(self, test_data_input: List[dict], batch_size: int = None) -> List:
        if batch_size is None:
            batch_size = self.batch_size

        forecasts = []
        
        for batch in tqdm(
            batcher(test_data_input, batch_size=batch_size),
            total=(len(test_data_input) + batch_size - 1) // batch_size,
            desc="FlowState Predict"
        ):
            context_batch = []
            for entry in batch:
                raw = np.array(entry["target"], dtype=float)
                if raw.ndim == 1:
                    raw = raw[None, :] # (1, T)
                if self.impute_missing:
                    raw = fill_missing(raw)
                if raw.shape[1] > self.context_length:
                    arr = raw[:, -self.context_length:]
                else:
                    arr = raw
                
                context_batch.append(torch.from_numpy(arr).float()) # (C, T_trunc)

            max_len = max(x.shape[1] for x in context_batch)
            n_ch = context_batch[0].shape[0]
            
            padded_batch = torch.zeros((len(context_batch), n_ch, max_len))
            for i, x in enumerate(context_batch):
                padded_batch[i, :, -x.shape[1]:] = x
            
            # (B, C, T) -> (T, B, C)
            input_tensor = padded_batch.permute(2, 0, 1).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    past_values=input_tensor,
                    scale_factor=self.scale_factor,
                    prediction_length=self.prediction_length,
                    batch_first=False
                )
            
            # (B, num_quantiles, pred_len, n_ch)
            pred_quantiles = outputs.quantile_outputs
            
            for i, ts in enumerate(batch):
                forecast_start_date = ts["start"] + len(ts["target"])
                
                # pred_i: (Q, H, C)
                pred_i = pred_quantiles[i].cpu().numpy()
                if n_ch == 1:
                    pred_i = pred_i.squeeze(-1)
                
                forecasts.append(
                    QuantileForecast(
                        forecast_arrays=pred_i,
                        forecast_keys=list(map(str, self.quantiles)),
                        start_date=forecast_start_date,
                        item_id=ts.get("item_id")
                    )
                )

        return forecasts

# -*- coding: utf-8 -*-

import os
import logging
from typing import List

import numpy as np
import torch
from torch import cuda
from tqdm.auto import tqdm

from gluonts.model.forecast import QuantileForecast
from gluonts.itertools import batcher

from model_zoo.base_model import BaseModel
from utils.missing import fill_missing

          
from tirex import load_model, ForecastModel
from tirex.models.tirex import TiRexZero


                                                            

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


                                                                

class TiRexModel(BaseModel):

    def __init__(self, args, module_name, model_name, model_local_path):
        self.args = args
        self.module_name = module_name
        self.model_name = model_name
        self.model_local_path = model_local_path
        self.output_dir = os.path.join(self.args.output_dir, self.model_name)

        super().__init__(self.model_name, args, self.output_dir)

    def get_predictor(self, dataset, batch_size):
        predictor = TiRexPredictor(
            config=self.args,
            batch_size=batch_size,
            model_path=self.model_local_path,
            prediction_length=dataset.prediction_length,
            ds_freq=dataset.freq,
        )
        return predictor


                                                                   

class TiRexPredictor:

                         
    QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(
        self,
        config,
        batch_size: int,
        model_path: str,
        prediction_length: int,
        ds_freq: str,
        *args,
        **kwargs,
    ):
                               
        self.config = config
        self.batch_size = batch_size
        self.model_path = model_path
        self.prediction_length = prediction_length
        self.ds_freq = ds_freq
        self.impute_missing=False

              
        self.device = "cuda" if cuda.is_available() else "cpu"

                                    
                                                          
                                   
        local_ckpt_path = os.path.join(model_path, "model.ckpt")

        if os.path.exists(local_ckpt_path):
                                                               
            print(f"[TiRex] Loading model from local path: {local_ckpt_path}")
            self.model: ForecastModel = TiRexZero.from_pretrained(
                local_ckpt_path,
                device=self.device,
                backend="torch",
            )
        elif os.path.exists(model_path) and model_path.endswith(".ckpt"):
                            
            print(f"[TiRex] Loading model from checkpoint: {model_path}")
            self.model: ForecastModel = TiRexZero.from_pretrained(
                model_path,
                device=self.device,
                backend="torch",
            )
        else:
                                   
            print(f"[TiRex] Loading model from HuggingFace: NX-AI/TiRex-1.1-gifteval")
            self.model: ForecastModel = load_model(
                "NX-AI/TiRex-1.1-gifteval",
                device=self.device,
                backend="torch",
            )

                                       
                                          
        if getattr(self.config, "fix_context_len", False):
            self.context_length = self.config.context_len
        else:
            self.context_length = 2048                       

                              
        self.quantiles = self.QUANTILES

        context_info = (
            self.context_length
            if getattr(self.config, "fix_context_len", False)
            else "full_history"
        )
        print(
            f"[TiRex] context_len={context_info}, "
            f"batch_size={self.batch_size}, "
            f"freq_in={self.ds_freq}, "
            f"impute_missing={self.impute_missing}"
        )

    def predict(self, test_data_input: List[dict], batch_size: int = None) -> List[QuantileForecast]:
        if batch_size is None:
            batch_size = self.batch_size

        forecasts: List[QuantileForecast] = []
        idx_offset = 0

        for batch in tqdm(
                batcher(test_data_input, batch_size=self.batch_size),
                total=(len(test_data_input) + self.batch_size - 1) // self.batch_size,
                desc="TiRex Predict"):

            # --------------------------------------------------
                                          
            # --------------------------------------------------
            context_list = []
            for entry in batch:
                raw = np.array(entry["target"], dtype=np.float32)

                       
                if self.impute_missing:
                    raw = fill_missing(
                        raw,
                        all_nan_strategy_1d="linspace",
                        interp_kind_1d="nearest",
                        add_noise_1d=True,
                        noise_ratio_1d=0.01,
                    )

                            
                if getattr(self.config, "fix_context_len", False):
                    arr = raw[-self.context_length:]
                else:
                    arr = raw

                context_list.append(arr)

            # --------------------------------------------------
                              
                                                               
                                   
                                                              
            #   - means: [B, H]
            # --------------------------------------------------
            quantiles_output, means_output = self.model.forecast(
                context=context_list,
                prediction_length=self.prediction_length,
                batch_size=self.batch_size,
                output_type="numpy",               
            )

                                            
                                                      
            batch_forecast = np.transpose(quantiles_output, (0, 2, 1))  # [B, 9, H]

            # --------------------------------------------------
                                              
            # --------------------------------------------------
            for i_in_batch, (arr, ts) in enumerate(zip(batch_forecast, batch)):
                if np.isnan(arr).any():
                    global_idx = idx_offset + i_in_batch
                    raise ValueError(f"TSRouter runtime message: {global_idx}TSRouter runtime message: ")

                forecast_start_date = ts["start"] + len(ts["target"])

                forecasts.append(
                    QuantileForecast(
                        forecast_arrays=arr,
                        forecast_keys=list(map(str, self.quantiles)),
                        start_date=forecast_start_date,
                    )
                )

            idx_offset += len(batch)

                    
            del context_list, quantiles_output, means_output, batch_forecast
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return forecasts


# -*- coding: utf-8 -*-

import os
import logging
from typing import List, Optional

import numpy as np
import torch
from torch import cuda

from chronos import BaseChronosPipeline, Chronos2Pipeline
from gluonts.model.forecast import QuantileForecast

from model_zoo.base_model import BaseModel


logger = logging.getLogger("Chronos-2 Predictor")
logger.setLevel(logging.INFO)


                                                                               

class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter: str):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()


gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(WarningFilter("The mean prediction is not stored in the forecast data"))


                                                     

class Chronos2Model(BaseModel):
    def __init__(self, args, module_name, model_name, model_local_path):
        self.module_name = module_name
        self.model_name = model_name
        self.args = args
        self.model_local_path = model_local_path
        self.output_dir = os.path.join(self.args.output_dir, self.model_name)
        super().__init__(self.model_name, args, self.output_dir)

    def get_predictor(self, dataset, batch_size):
                                                         
        predictor = Chronos2Predictor(
            config=self.args,
            model_path=self.model_local_path,
            prediction_length=dataset.prediction_length,
            batch_size=batch_size,
            quantile_levels=[0.1 * i for i in range(1, 10)],
                                                                 
            predict_batches_jointly=getattr(self.args, "chronos2_joint", False),
            local_files_only=True,
            device_map=torch.device("cuda" if cuda.is_available() else "cpu"),
        )
        return predictor


                                                           

class Chronos2Predictor:
    def __init__(
        self,
        config,
        model_path: str,
        prediction_length: int,
        batch_size: int,
        quantile_levels: Optional[List[float]] = None,
        predict_batches_jointly: bool = False,
        **kwargs
    ):
        self.config = config
        self.model_path = model_path
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.quantile_levels = quantile_levels or [0.1 * i for i in range(1, 10)]
        self.predict_batches_jointly = predict_batches_jointly

                                                      
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")

                                                    
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_path,
            **kwargs,
        )
        assert isinstance(
            self.pipeline, Chronos2Pipeline
        ), 'TSRouter runtime message.'

        context_info = (
            self.config.context_len
            if getattr(self.config, "fix_context_len", False)
            else "full_history"
        )
        print(
            f"[Chronos-2] context_len={context_info}, "
            f"batch_size={self.batch_size}, "
            f"prediction_length={self.prediction_length}, "
            f"quantiles={len(self.quantile_levels)}, "
            f"predict_batches_jointly={self.predict_batches_jointly}"
        )

    # ------------------------------------------------------
                                          
                                  
    # ------------------------------------------------------
    def _normalize_target(self, raw_target) -> np.ndarray:
        arr = np.asarray(raw_target)

        if arr.ndim == 1:
            return arr  # (T,)

        if arr.ndim != 2:
            raise ValueError(f"[Chronos-2] Unsupported target ndim={arr.ndim}, shape={arr.shape}")

                          
                           
                                             
        if arr.shape[0] > arr.shape[1] and arr.shape[1] <= 64:
            arr = arr.T  # (V,T)

                                         
        return arr

    def _pack_model_items(self, items: List[dict]) -> List[dict]:
        packed = []
        for item in items:
            target = self._normalize_target(item["target"])

                                         
            if getattr(self.config, "fix_context_len", False):
                cl = int(self.config.context_len)
                if target.ndim == 1:
                    target = target[-cl:]  # (T,)
                else:
                    target = target[:, -cl:]                           

            packed.append({"target": target})
        return packed

    def predict(self, test_data_input: List[dict]) -> List[QuantileForecast]:
        pipeline = self.pipeline

        if self.predict_batches_jointly:
            logger.info(
                "Note: Using cross learning mode. "
                "Please ensure that different rolling windows of the same time series "
                "are not in `test_data_input` to avoid leakage due to in-context learning."
            )

        input_data = self._pack_model_items(test_data_input)

                                       
        first_target = input_data[0]["target"]
        is_univariate_data = (np.asarray(first_target).ndim == 1)

                                                   

                                                                              
        quantiles, _ = pipeline.predict_quantiles(
            inputs=input_data,
            prediction_length=self.prediction_length,
            batch_size=self.batch_size,
            quantile_levels=self.quantile_levels,
            predict_batches_jointly=self.predict_batches_jointly,
        )

        quantiles = torch.stack(quantiles)
                                                              
                                                                          
        quantiles = quantiles.permute(0, 3, 1, 2).cpu().numpy()  # [B, Q, V, H]
        quantiles = np.transpose(quantiles, (0, 1, 3, 2))  # [B, Q, H, V]

                                         
        is_univariate_data = (np.asarray(input_data[0]["target"]).ndim == 1)
        if is_univariate_data:
            quantiles = quantiles.squeeze(-1)

        forecast_outputs = quantiles


                                                     
        forecasts: List[QuantileForecast] = []
        q_keys = list(map(str, self.quantile_levels))

        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])

            forecast = QuantileForecast(
                forecast_arrays=item,
                forecast_keys=q_keys,
                start_date=forecast_start_date,
            )

                                                            
            # if isinstance(item, np.ndarray) and item.ndim == 3:  # (Q,H,V)
            #     forecast = QuantileForecastVFirst(
            #         forecast_arrays=item,
            #         forecast_keys=q_keys,
            #         start_date=forecast_start_date,
            #     )
            # else:  # (Q,H)
            #     forecast = QuantileForecast(
            #         forecast_arrays=item,
            #         forecast_keys=q_keys,
            #         start_date=forecast_start_date,
            #     )

            forecasts.append(forecast)

        return forecasts


class QuantileForecastVFirst(QuantileForecast):

    def quantile(self, q: str) -> np.ndarray:
        arr = super().quantile(q)
                             
        # multivariate: (H, V) -> (V, H)
        if arr.ndim == 2:
            return arr.T
        return arr

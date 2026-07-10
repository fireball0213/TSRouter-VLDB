
import os
import logging
import warnings
from typing import List

import numpy as np
import torch
from torch import cuda
from tqdm.auto import tqdm

from gluonts.model.forecast import QuantileForecast,SampleForecast
from gluonts.itertools import batcher

from model_zoo.base_model import BaseModel
from utils.missing import fill_missing
warnings.filterwarnings('ignore')

                                                    

class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()


gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(
    WarningFilter("The mean prediction is not stored in the forecast data")
)

#
# class KairosModel(BaseModel):
#     """
             
                                                               
                                                          
#     """
#
#     def __init__(self, args, module_name, model_name, model_local_path):
#         self.args = args
#         self.module_name = module_name
#         self.model_name = model_name
#         self.model_local_path = model_local_path
#         self.output_dir = os.path.join(self.args.output_dir, self.model_name)
#
#         super().__init__(self.model_name, args, self.output_dir)
#
#     def get_predictor(self, dataset, batch_size):
#         """
                              
#             - prediction_length
                     
#
                                                                          
#         """
#         predictor = KairosModelPredictor(
#             config=self.args,
#             batch_size=batch_size,
#             model_path=self.model_local_path,
#             prediction_length=dataset.prediction_length,
#             ds_freq=dataset.freq,
                                               
#         )
#         return predictor
#
#
                                                                        
#
# class KairosModelPredictor:
#     """
           
                                     
                       
                           
#         * pred_len (prediction_length)
                                                  
                               
                      
                                
                    
                                          
                                
                           
                                                    
#     """
#
#     def __init__(
#             self,
#             config,
#             batch_size: int,
#             model_path: str,
#             prediction_length: int,
#             ds_freq: str,
#             target_dim: int = 1,
#             past_feat_dynamic_real_dim: int = 0,
#             *args,
#             **kwargs,
#     ):
                                 
                                                                                          
#         self.batch_size = batch_size
                                                        
#         self.prediction_length = prediction_length
                                                               
#         self.target_dim = target_dim
#         self.past_feat_dynamic_real_dim = past_feat_dynamic_real_dim
#
                          
#         self.device = device = "cuda" if cuda.is_available() else "cpu"
#
                               
                                                                                                                                                       
#         from model_zoo.TSFM_src.Kairos.tsfm.model.kairos import AutoModel
#         # model_path = self.model_local_path if os.path.exists(self.model_local_path) else "mldi-lab/Kairos_10m"
#         model_path = self.model_path
#         self.model = AutoModel.from_pretrained(
#                         model_path,
#                         trust_remote_code=True,
#                         local_files_only=True
#                     ).to(self.device)
#         self.model.eval()
                                         
#         if getattr(self.config, "fix_context_len", False):
#             self.context_length = self.config.context_len
#         else:
                                                                                                 
#
                                                       
                                                                        
                                                                                      
#         #
               
#         #   self.freq = some_freq_mapping(self.ds_freq)
#         #   self.quantiles = [0.1 * i for i in range(1, 10)]
#         #
#         self.freq = ds_freq
#         self.quantiles = None
#
#         context_info = (
#             self.context_length
#             if getattr(self.config, "fix_context_len", False)
#             else "full_history"
#         )
#         print(
#             f"[NewModel] context_len={context_info}, "
#             f"freq_in={self.ds_freq}, "
                                                     
#         )
#
#     # =========================================================================
                      
#     # =========================================================================
#     def predict(self, test_data_input: List[dict], batch_size: int = None) -> List:
#         """
               
#         ----------
#         test_data_input:
                                                                          
#         batch_size:
                                                    
#
              
#         ----------
                                         
                                    
                                         
#         """
#         if batch_size is None:
#             batch_size = self.batch_size
#
                                                                                             
#
                                                     
#
               
#         for batch in tqdm(
#                 batcher(test_data_input, batch_size=self.batch_size),
#                 total=len(test_data_input) // self.batch_size,
#                 desc="New Model Predict"):
#
#             # --------------------------------------------------
                                      
                                
                                                
                                                           
#             # --------------------------------------------------
#
#             max_len = 0
#             context_batch = []
#             for entry in batch:
#                 raw = np.array(entry["target"], dtype=float)
#
                           
#                 raw = fill_missing(
#                     raw,
#                     all_nan_strategy_1d="linspace",
#                     interp_kind_1d="nearest",
#                     add_noise_1d=True,
#                     noise_ratio_1d=0.01,
#                 )
#
                                
#                 if getattr(self.config, "fix_context_len", False):
#                     arr = raw[-self.context_length:]
#                 else:
#                     arr = raw
#
#                 if len(arr) > max_len:
#                     max_len = len(arr)
#
#                 context_batch.append(arr)
#
                                
#             padded_batch = []
#             for arr in context_batch:
#                 pad_len = max_len - len(arr)
#                 if pad_len > 0:
#                     pad = np.full((pad_len,), np.nan, dtype=float)
#                     arr_ = np.concatenate([pad, arr])
#                     padded_batch.append(arr_)
#                 else:
#                     padded_batch.append(arr)
#
#             past_target = torch.tensor(padded_batch, device=self.device, dtype=torch.float32)
#             # --------------------------------------------------
                                    
                                                                    
                                                                    
                                             
#             # --------------------------------------------------
#
                       
#             #   context_tensor = torch.tensor(context_batch, device=self.device, dtype=torch.float32)
#             #   model_outputs = self.model(...)
#
                                                                           
#
#             with torch.no_grad():
#                 model_outputs = self.model(
#                     past_target=past_target,
#                     prediction_length=self.prediction_length,
#                     generation=True,
#                     preserve_positivity=True,
#                     average_with_flipped_input=True,
#                 )
#
#             # [B, num_samples, H]
#             samples = model_outputs["prediction_outputs"].detach().cpu().numpy()
#
#             # --------------------------------------------------
                                                      
                                                                    
                                                                  
                                                                      
                                              
#             # --------------------------------------------------
#
                                     
#             for i, ts in enumerate(batch):
#                 forecast_start_date = ts["start"] + len(ts["target"])
#
                                                                  
#                 pred_i = samples[i]  # np.ndarray
#
                                    
#                 if np.isnan(pred_i).any():
#                     raise ValueError(f"[NaN DEBUG] FlowState prediction contains NaN at index {i}")
#
                                                     
                             
#                 # forecasts.append(
#                 #     QuantileForecast(
#                 #         forecast_arrays=pred_i,
#                 #         forecast_keys=list(map(str, self.quantiles)),
#                 #         start_date=forecast_start_date,
#                 #     )
#                 # )
#                 #
                            
#                 forecasts.append(
#                     SampleForecast(
#                         samples=pred_i,
#                         start_date=forecast_start_date,
#                     )
#                 )
#
#         return forecasts
class KairosModel(BaseModel):


    def __init__(self, args, module_name, model_name, model_local_path):
        self.args = args
        self.module_name = module_name
        self.model_name = model_name
        self.model_local_path = model_local_path
        self.output_dir = os.path.join(self.args.output_dir, self.model_name)

        super().__init__(self.model_name, args, self.output_dir)

    def get_predictor(self, dataset, batch_size):

        device = "cuda" if cuda.is_available() else "cpu"

        # print(f"[Kairos] Initializing ultra-minimal 10M model on device {device}...")

        # try:
        from model_zoo.TSFM_src.Kairos.tsfm.model.kairos import AutoModel
        # model_path = self.model_local_path if os.path.exists(self.model_local_path) else "mldi-lab/Kairos_10m"
        model_path = self.model_local_path

        print(f"[Kairos] Loading weights from: {model_path}")
        kairos_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        ).to(device)
        kairos_model.eval()

        # except Exception as e:
        #     raise RuntimeError(f"Failed to load Kairos model: {e}")

        return KairosPredictor(
            self.args,
            kairos_model,
            batch_size,
            prediction_length=dataset.prediction_length,
            ds_freq=dataset.freq,
            device=device,
        )

def pad_or_truncate(sequence, max_length=2048, pad_value=np.nan):
    """
    Pads or truncates a sequence on the left to a specified max_length.

    Args:
        sequence (list or np.ndarray): The input sequence.
        max_length (int): The target length.
        pad_value (int or float): The value to use for padding, defaults to np.nan.

    Returns:
        np.ndarray: A NumPy array of length max_length.
    """
    seq_np = np.array(sequence)
    current_length = len(seq_np)

    if current_length < max_length:
        # If the current length is less than the target, calculate the required padding
        padding_size = max_length - current_length
        # Use np.pad to add padding to the left
        # (padding_size, 0) means pad `padding_size` elements at the beginning of the first (and only) axis
        return np.pad(seq_np, (padding_size, 0), 'constant', constant_values=pad_value)
    else:
        # If the current length is greater than or equal to the target, truncate to the last max_length elements
        return seq_np[-max_length:]

class KairosPredictor:
    def __init__(
            self,
            args,
            kairos_model,
            batch_size,
            prediction_length: int,
            ds_freq: str,
            device: str = "cpu",
    ):
        self.args = args
        self.model = kairos_model
        self.batch_size = batch_size
        self.prediction_length = prediction_length
        self.device = device

        self.model_max_past_len = 2048
        self.impute_missing = False

        context_info = (
            self.args.context_len
            if getattr(self.args, "fix_context_len", False)
            else "full_history"
        )
        print(
            f"[Kairos] context_len={context_info}, "
            f"batch_size={self.batch_size}, "
            f"freq_in={ds_freq}, "
            f"impute_missing={self.impute_missing}"
        )

    def prepare_past_target(self, raw_target) -> torch.Tensor:
        target = np.asarray(raw_target, dtype=np.float32)

                                       
        if self.impute_missing:
            target = fill_missing(target,
                all_nan_strategy_1d="linspace",
                interp_kind_1d="nearest",
                add_noise_1d=True,
                noise_ratio_1d=0.01,
            )
                 
        context_len = self.args.context_len if getattr(self.args, "fix_context_len", False) else None
        if context_len:
            target = target[-context_len:]

                                   
        target_fixed = pad_or_truncate(target, max_length=self.model_max_past_len, pad_value=np.nan)

        return torch.tensor(target_fixed, dtype=torch.float32)

    def predict(self, test_data_input) :
        self.model.eval()
        model = self.model

        # Generate forecast samples
        forecast_outputs = []
        with torch.no_grad():
            for batch in tqdm(batcher(test_data_input, batch_size=self.batch_size),
                              total=(len(test_data_input) + self.batch_size - 1) // self.batch_size,
                              desc="Kairos Inference"):
                context = [self.prepare_past_target(entry["target"]) for entry in batch]
                forecast_outputs.append(
                    model(
                        past_target=torch.stack(context).to(self.device),
                        prediction_length=self.prediction_length,
                        generation=True,
                        infer_is_positive=True,
                        force_flip_invariance=True,
                    )["prediction_outputs"].detach().cpu().numpy()
                )
        forecast_outputs = np.concatenate(forecast_outputs)


        # Convert forecast samples into gluonts Forecast objects
        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                SampleForecast(samples=item, start_date=forecast_start_date)
            )

        return forecasts

#
                                                      
# class KairosPredictor:
#     def __init__(self, args, kairos_model, batch_size, prediction_length, device="cpu"):
#         self.args = args
#         self.model = kairos_model
#         self.batch_size = batch_size
#         self.prediction_length = prediction_length
#         self.device = device
#         self.quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#
#     def predict(self, test_data_input: List[dict], batch_size: int = None) -> List[QuantileForecast]:
#         batch_size = batch_size or self.batch_size
#         forecasts = []
#
#         context_len = self.args.context_len if self.args.fix_context_len else None
#
                                  
#         for batch in tqdm(batcher(test_data_input, batch_size=batch_size),
#                           total=(len(test_data_input) + batch_size - 1) // batch_size,
#                           desc="Kairos Inference"):
#
#             context_list = []
                                           
#             for entry in batch:
                                              
#                 raw = fill_missing(
#                     np.array(entry["target"], dtype="float32"),
#                     all_nan_strategy_1d="linspace",
#                     interp_kind_1d="nearest",
#                     add_noise_1d=True,
#                     noise_ratio_1d=0.01,
#                 )
#                 if context_len:
#                     raw = raw[-context_len:]
#                 context_list.append(raw)
#
                         
#             try:
#                 batch_quantiles = self._run_model_inference(context_list)
#             except Exception as e:
#                 print(f"[Kairos] Batch failed, skipping: {e}")
#                 continue
#
                                          
#             for i, entry in enumerate(batch):
#                 forecast_start_date = entry["start"] + len(entry["target"])
#
                                   
                                                     
#                 pred_i = batch_quantiles[i]
#
                                                            
#                 if pred_i.shape[0] != len(self.quantile_levels):
#                     pred_i = pred_i.T
#
                                          
#                 h = self.prediction_length
#                 if pred_i.shape[1] > h:
#                     pred_i = pred_i[:, :h]
#                 elif pred_i.shape[1] < h:
                            
#                     last_vals = pred_i[:, -1:]
#                     padding = np.tile(last_vals, (1, h - pred_i.shape[1]))
#                     pred_i = np.concatenate([pred_i, padding], axis=1)
#
#                 forecasts.append(
#                     QuantileForecast(
#                         forecast_arrays=pred_i,
#                         forecast_keys=[str(q) for q in self.quantile_levels],
#                         start_date=forecast_start_date,
#                     )
#                 )
#
                        
#             if "cuda" in str(self.device):
#                 torch.cuda.empty_cache()
#
#         return forecasts
#
#     def _run_model_inference(self, context_list: List[np.ndarray]) -> np.ndarray:
                                             
#         batch_size = len(context_list)
#         max_len = max(len(ctx) for ctx in context_list)
#
                                
#         padded_context = np.zeros((batch_size, max_len), dtype="float32")
#         for i, ctx in enumerate(context_list):
#             padded_context[i, -len(ctx):] = ctx
#
#         past_target = torch.from_numpy(padded_context).to(self.device)
#
#         with torch.no_grad():
#             output = self.model(
#                 past_target=past_target,
#                 prediction_length=self.prediction_length,
#                 generation=True,
#                 preserve_positivity=True,
#                 average_with_flipped_input=True,
#             )
#
                                        
                               
#             quantiles = output.get("prediction_outputs", None)
#             if quantiles is None:
#                 raise ValueError("Model output does not contain 'prediction_outputs'")
#
#             return quantiles.cpu().numpy()

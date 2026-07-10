import os
import logging
from typing import List

import numpy as np
import torch
from torch import cuda
from tqdm.auto import tqdm
from gluonts.itertools import batcher
from gluonts.model.forecast import SampleForecast
from transformers import AutoModelForCausalLM

from model_zoo.base_model import BaseModel
from utils.missing import fill_missing


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


class TimeMOEModel(BaseModel):
    def __init__(self, args, module_name, model_name, model_local_path):
        self.args = args
        self.module_name = module_name
        self.model_name = model_name
        self.model_local_path = model_local_path
        self.output_dir = os.path.join(self.args.output_dir, self.model_name)

        super().__init__(self.model_name, args, self.output_dir)

    def get_predictor(self, dataset, batch_size):
        return TimeMOEPredictor(
            config=self.args,
            batch_size=batch_size,
            model_path=self.model_local_path,
            prediction_length=dataset.prediction_length,
            ds_freq=getattr(dataset, "freq", None),
            target_dim=getattr(dataset, "target_dim", 1),
            dataset_name=getattr(dataset, "name", "unknown"),
        )


class TimeMOEPredictor:

    MAX_TOTAL_LEN = 4096

    def __init__(
        self,
        config,
        batch_size: int,
        model_path: str,
        prediction_length: int,
        ds_freq: str = None,
        target_dim: int = 1,
        dataset_name: str = "unknown",
        *args,
        **kwargs,
    ):
        self.config = config
        self.batch_size = batch_size
        self.model_path = model_path
        self.prediction_length = prediction_length
        self.ds_freq = ds_freq
        self.target_dim = target_dim
        self.dataset_name = dataset_name
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")

        self.impute_missing = False

        max_ctx = self.MAX_TOTAL_LEN - int(self.prediction_length)
        if max_ctx <= 0:
            raise ValueError(
                f"prediction_length={self.prediction_length} is too large for TimeMOE max total length {self.MAX_TOTAL_LEN}"
            )

                                             
        if getattr(self.config, "fix_context_len", False):
            requested_context = int(getattr(self.config, "context_len", 512))
        else:
            requested_context = self._official_context_length(self.prediction_length)

        if requested_context <= 0:
            requested_context = 512

        self.context_length = min(int(requested_context), max_ctx)

                                                  
        self.norm_mode = "zscore"

                            
        self.use_flash_attn = False
        self.infer_dtype = torch.float32

        model_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
        }

        use_flash_attn = bool(getattr(self.config, "use_flash_attn", True))
        if use_flash_attn and self.device.type == "cuda":
            self.use_flash_attn = True
            self.infer_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            model_kwargs["attn_implementation"] = "flash_attention_2"
            model_kwargs["dtype"] = self.infer_dtype

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs,
            )
        except Exception as e:
            if model_kwargs.get("attn_implementation") == "flash_attention_2":
                print(f"TSRouter runtime message: {e}")
                self.use_flash_attn = False
                self.infer_dtype = torch.float32
                model_kwargs.pop("attn_implementation", None)
                model_kwargs.pop("dtype", None)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_kwargs,
                )
            else:
                raise

        self.model.to(self.device)
        self.model.eval()
        self.model_dtype = next(self.model.parameters()).dtype

        print(
            f"[TimeMOE] context_len={self.context_length}, "
            f"batch_size={self.batch_size}, "
            f"prediction_length={self.prediction_length}, "
            f"freq={self.ds_freq}, "
            f"target_dim={self.target_dim}, "
            f"norm_mode={self.norm_mode}, "
            f"impute_missing={self.impute_missing}, "
            f"use_flash_attn={self.use_flash_attn}, "
            f"model_dtype={self.model_dtype}, "
            f"local_only=True"
        )

        self._debug_structure_once = False
        self._debug_batch_once = False
        self._debug_model_once = False
        self._warn_split_multivar_once = False

    @staticmethod
    def _official_context_length(prediction_length: int) -> int:
        if prediction_length == 96:
            return 512
        elif prediction_length == 192:
            return 1024
        elif prediction_length == 336:
            return 2048
        elif prediction_length == 720:
            return 3072
        else:
            return prediction_length * 4

    def _debug_dataset_structure(self, batch: List[dict]):
        if self._debug_structure_once:
            return

        # print(f"[TimeMOE-debug] dataset_name={self.dataset_name}")
        # print(
        #     f"[TimeMOE-debug] ds_freq={self.ds_freq}, "
        #     f"target_dim={self.target_dim}, "
        #     f"batch_size={len(batch)}"
        # )

        for i, entry in enumerate(batch[:5]):
            arr = np.asarray(entry["target"])
            item_id = entry.get("item_id", None)
            # print(
            #     f"[TimeMOE-debug] sample={i}, raw_shape={arr.shape}, raw_ndim={arr.ndim}, "
            #     f"dtype={arr.dtype}, item_id={item_id}"
            # )

        self._debug_structure_once = True

    def _to_univariate_series(self, raw_target, item_id=None) -> np.ndarray:
        raw = np.asarray(raw_target, dtype=np.float32)

        if raw.ndim == 1:
            series = raw
            if self.target_dim > 1 and not self._warn_split_multivar_once:
                # print(
                                                                                
                                                 
                # )
                self._warn_split_multivar_once = True

        elif raw.ndim == 2:
            if raw.shape[0] == 1:
                series = raw[0]
            elif raw.shape[1] == 1:
                series = raw[:, 0]
            else:
                raise ValueError(
                    f"TSRouter runtime message: {raw.shape}, item_id={item_id}。"
                    f"TSRouter runtime message: "
                    f"TSRouter runtime message: "
                )
        else:
            raise ValueError(
                f"TSRouter runtime message: {raw.ndim}TSRouter runtime message: {raw.shape}, item_id={item_id}"
            )

        if self.impute_missing:
            series = fill_missing(
                series,
                all_nan_strategy_1d="linspace",
                interp_kind_1d="nearest",
                add_noise_1d=True,
                noise_ratio_1d=0.01,
            )

        return series.astype(np.float32)

                                                
    def _normalize_batch(self, arr_2d: np.ndarray, valid_mask_2d: np.ndarray):
        mask = valid_mask_2d.astype(np.float32)

        valid_count = np.sum(mask, axis=1, keepdims=True)
        valid_count = np.maximum(valid_count, 1.0)

        mu = np.sum(arr_2d * mask, axis=1, keepdims=True) / valid_count

                                 
        var_num = np.sum(((arr_2d - mu) ** 2) * mask, axis=1, keepdims=True)
        var_denom = np.maximum(valid_count - 1.0, 1.0)
        sigma = np.sqrt(var_num / var_denom)

        mu = np.where(np.isfinite(mu), mu, 0.0)
        sigma = np.where(np.isfinite(sigma) & (sigma >= 1e-8), sigma, 1.0)

        normed = (arr_2d - mu) / sigma
        normed = np.where(mask > 0, normed, 0.0)

        return normed.astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32)

    def _forecast_batch(
            self,
            norm_context_batch: np.ndarray,
            attention_mask: np.ndarray,
    ) -> np.ndarray:
        x = torch.as_tensor(
            norm_context_batch,
            dtype=self.model_dtype,
            device=self.device,
        )
        attn_mask = torch.as_tensor(
            attention_mask,
            dtype=torch.long,
            device=self.device,
        )

        has_padding = not bool(torch.all(attn_mask == 1).item())

        # if not self._debug_model_once:
        #     print(f"[TimeMOE-debug] norm_context_batch.shape={norm_context_batch.shape}")
        #     print(f"[TimeMOE-debug] attention_mask.shape={attention_mask.shape}")
        #     print(f"[TimeMOE-debug] x.shape={tuple(x.shape)}, x.dtype={x.dtype}")
        #     print(f"[TimeMOE-debug] attn_mask.shape={tuple(attn_mask.shape)}, attn_mask.dtype={attn_mask.dtype}")
        #     print(f"[TimeMOE-debug] model_dtype={self.model_dtype}, use_flash_attn={self.use_flash_attn}")
        #     print(f"[TimeMOE-debug] has_padding={has_padding}")

        gen_kwargs = dict(
            inputs=x,
            max_new_tokens=self.prediction_length,
            do_sample=False,
            use_cache=False,
        )

                                            
        if has_padding:
            gen_kwargs["attention_mask"] = attn_mask

        with torch.no_grad():
            if self.device.type == "cuda" and self.model_dtype in (torch.float16, torch.bfloat16):
                with torch.autocast(device_type="cuda", dtype=self.model_dtype):
                    output = self.model.generate(**gen_kwargs)
            else:
                output = self.model.generate(**gen_kwargs)

        if isinstance(output, (tuple, list)):
            output = output[0]

        if isinstance(output, torch.Tensor):
            if not self._debug_model_once:
                print(f"[TimeMOE-debug] raw output.shape={tuple(output.shape)}, output.dtype={output.dtype}")
            output = output.detach().float().cpu().numpy()
        else:
            output = np.asarray(output, dtype=np.float32)
            if not self._debug_model_once:
                print(f"[TimeMOE-debug] raw output.shape={output.shape}, output.dtype={output.dtype}")

        if output.ndim != 2:
            output = np.asarray(output).reshape(output.shape[0], -1)

        pred_batch = output[:, -self.prediction_length:]

        if not self._debug_model_once:
            print(f"[TimeMOE-debug] pred_batch.shape={pred_batch.shape}")
            self._debug_model_once = True

        return pred_batch.astype(np.float32)

    def predict(self, test_data_input: List[dict], batch_size: int = None) -> List[SampleForecast]:
        if batch_size is None:
            batch_size = self.batch_size

        forecasts: List[SampleForecast] = []

        for batch in tqdm(
            batcher(test_data_input, batch_size=batch_size),
            total=(len(test_data_input) + batch_size - 1) // batch_size,
            desc="TimeMOE Predict",
        ):
            self._debug_dataset_structure(batch)

            series_list = []
            start_list = []
            raw_len_list = []

            for entry in batch:
                item_id = entry.get("item_id", None)
                series = self._to_univariate_series(entry["target"], item_id=item_id)

                raw_len = len(series)
                raw_len_list.append(raw_len)
                start_list.append(entry["start"])

                                          
                ctx = series[-self.context_length:]
                series_list.append(ctx)

                                        
            padded_batch = np.full(
                (len(series_list), self.context_length),
                np.nan,
                dtype=np.float32,
            )

            for i, seq in enumerate(series_list):
                L = len(seq)
                padded_batch[i, -L:] = seq

            valid_mask = ~np.isnan(padded_batch)
            filled_batch = np.where(valid_mask, padded_batch, 0.0)

            if not self._debug_batch_once:
                eff_len_list = [len(x) for x in series_list[:10]]
                # print(f"[TimeMOE-debug] fixed_context_len={self.context_length}")
                # print(f"[TimeMOE-debug] truncated_len_list[:10]={eff_len_list}")
                # print(f"[TimeMOE-debug] padded_batch.shape={padded_batch.shape}")
                # print(f"[TimeMOE-debug] valid_mask.shape={valid_mask.shape}")
                self._debug_batch_once = True

            norm_ctx_batch, mu_batch, sigma_batch = self._normalize_batch(
                filled_batch,
                valid_mask.astype(np.int64),
            )

            pred_norm_batch = self._forecast_batch(
                norm_ctx_batch,
                valid_mask.astype(np.int64),
            )

                                    
            pred_batch = pred_norm_batch * sigma_batch + mu_batch

            for i in range(len(batch)):
                pred = pred_batch[i]
                forecast_start_date = start_list[i] + raw_len_list[i]

                forecasts.append(
                    SampleForecast(
                        samples=pred[np.newaxis, :],
                        start_date=forecast_start_date,
                    )
                )

        return forecasts

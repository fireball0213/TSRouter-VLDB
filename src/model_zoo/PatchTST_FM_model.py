# -*- coding: utf-8 -*-
"""PatchTST-FM model integration for the local TSRouter model zoo."""

import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import cuda
from tqdm.auto import tqdm

from gluonts.itertools import batcher
from gluonts.model.forecast import QuantileForecast

from model_zoo.base_model import BaseModel
from utils.missing import fill_missing


GE_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

_TSFMSRC_DIR = Path(__file__).resolve().parent / "TSFM_src"
if _TSFMSRC_DIR.exists():
    _p = str(_TSFMSRC_DIR)
    if _p not in sys.path:
        sys.path.insert(0, _p)
else:
    raise FileNotFoundError(f"[PatchTST-FM] TSFM_src directory not found: {_TSFMSRC_DIR}")


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


def _target_time_length(raw_target) -> int:
    arr = np.asarray(raw_target)
    if arr.ndim <= 1:
        return int(arr.shape[0])
    return int(arr.shape[-1] if arr.shape[0] <= arr.shape[-1] else arr.shape[0])


def _import_patchtst_fm_class():
    try:
        from tsfm_public.models.patchtst_fm import PatchTSTFMForPrediction
    except Exception as exc:
        raise ImportError(
            "[PatchTST-FM] Missing local granite-tsfm PatchTST-FM source. "
            "Copy the GitHub folder "
            "ibm-granite/granite-tsfm@patchtst-fm-gift:"
            "tsfm_public/models/patchtst_fm into "
            "src/model_zoo/TSFM_src/tsfm_public/models/patchtst_fm."
        ) from exc
    return PatchTSTFMForPrediction


def _suppress_patchtst_fm_info_logs() -> None:
    logging.getLogger("tsfm_public.models.patchtst_fm.modeling_patchtst_fm").setLevel(logging.WARNING)


class PatchTSTFMModel(BaseModel):
    def __init__(self, args, module_name, model_name, model_local_path):
        self.args = args
        self.module_name = module_name
        self.model_name = model_name
        self.model_local_path = model_local_path
        self.output_dir = os.path.join(self.args.output_dir, self.model_name)
        super().__init__(self.model_name, args, self.output_dir)

    def get_predictor(self, dataset, batch_size):
        PatchTSTFMForPrediction = _import_patchtst_fm_class()
        _suppress_patchtst_fm_info_logs()
        device = torch.device("cuda" if cuda.is_available() else "cpu")

        model = PatchTSTFMForPrediction.from_pretrained(
            self.model_local_path,
            local_files_only=True,
        ).to(device)
        model.eval()

        return PatchTSTFMPredictor(
            config=self.args,
            model=model,
            batch_size=batch_size,
            prediction_length=dataset.prediction_length,
            ds_freq=dataset.freq,
            device=device,
        )


class PatchTSTFMPredictor:
    def __init__(
        self,
        config,
        model,
        batch_size: int,
        prediction_length: int,
        ds_freq: str,
        device: torch.device,
    ):
        self.config = config
        self.model = model
        self.batch_size = batch_size
        self.prediction_length = int(prediction_length)
        self.device = device
        self.ds_freq = ds_freq
        self.quantiles = GE_QUANTILES
        self.impute_missing = True
        self.model_context_length = int(getattr(self.model.config, "context_length", 8192))

        context_info = (
            int(self.config.context_len)
            if getattr(self.config, "fix_context_len", False)
            else f"full_history(model_max={self.model_context_length})"
        )
        if (
            getattr(self.config, "fix_context_len", False)
            and int(self.config.context_len) + self.prediction_length > self.model_context_length
        ):
            print(
                "[PatchTST-FM] warning: context_len + prediction_length exceeds "
                f"model context_length={self.model_context_length}; the model will internally "
                "fit/downsample to its own context budget."
            )
        print(
            f"[PatchTST-FM] context_len={context_info}, "
            f"batch_size={self.batch_size}, "
            f"prediction_length={self.prediction_length}, "
            f"freq_used=False, "
            f"impute_missing={self.impute_missing}, "
            f"quantiles={self.quantiles}"
        )

    def _format_target(self, raw_target) -> np.ndarray:
        arr = np.asarray(raw_target, dtype=np.float32)
        if arr.ndim not in (1, 2):
            raise ValueError(f"[PatchTST-FM] unsupported target ndim={arr.ndim}, shape={arr.shape}")

        if getattr(self.config, "fix_context_len", False):
            cl = int(self.config.context_len)
            if arr.ndim == 1:
                arr = arr[-cl:]
            elif arr.shape[0] <= arr.shape[1]:
                arr = arr[:, -cl:]
            else:
                arr = arr[-cl:, :]

        if self.impute_missing:
            if arr.ndim == 1:
                arr = fill_missing(
                    arr,
                    all_nan_strategy_1d="zero",
                    interp_kind_1d="nearest",
                ).astype(np.float32)
            else:
                channel_first = arr if arr.shape[0] <= arr.shape[1] else arr.T
                arr = fill_missing(channel_first, interp_kind_2d="nearest").astype(np.float32)

        if arr.ndim == 2 and arr.shape[0] <= arr.shape[1]:
            arr = arr.T

        return np.ascontiguousarray(arr, dtype=np.float32)

    @torch.no_grad()
    def predict(self, test_data_input: List[dict], batch_size: int = None) -> List[QuantileForecast]:
        batch_size = int(batch_size or self.batch_size)
        forecasts: List[QuantileForecast] = []

        for batch in tqdm(
            batcher(test_data_input, batch_size=batch_size),
            total=(len(test_data_input) + batch_size - 1) // batch_size,
            desc="PatchTST-FM Predict",
        ):
            target = [
                torch.from_numpy(self._format_target(entry["target"])).float().to(self.device)
                for entry in batch
            ]

            model_outputs = self.model(
                past_values=target,
                prediction_length=self.prediction_length,
                quantile_levels=self.quantiles,
            )

            pred_quantiles = []
            for item in model_outputs.quantile_outputs:
                item_np = item.detach().cpu().numpy()
                if item_np.shape[-1] == 1:
                    item_np = item_np.squeeze(-1)
                if not np.isfinite(item_np).all():
                    raise ValueError("[PatchTST-FM] prediction contains NaN/Inf")
                pred_quantiles.append(item_np)

            for item, ts in zip(pred_quantiles, batch):
                forecast_start_date = ts["start"] + _target_time_length(ts["target"])
                forecasts.append(
                    QuantileForecast(
                        forecast_arrays=item,
                        forecast_keys=list(map(str, self.quantiles)),
                        start_date=forecast_start_date,
                        item_id=ts.get("item_id"),
                    )
                )

        return forecasts

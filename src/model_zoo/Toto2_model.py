# -*- coding: utf-8 -*-
"""Toto 2.0 model integration for the local TSRouter model zoo."""

import logging
import os
import sys
from pathlib import Path
from typing import List

import torch
from torch import cuda

from model_zoo.base_model import BaseModel


GE_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

_TSFMSRC_DIR = Path(__file__).resolve().parent / "TSFM_src"
if _TSFMSRC_DIR.exists():
    _p = str(_TSFMSRC_DIR)
    if _p not in sys.path:
        sys.path.insert(0, _p)
else:
    raise FileNotFoundError(f"[Toto2] TSFM_src directory not found: {_TSFMSRC_DIR}")


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


def _import_toto2_classes():
    try:
        from toto2 import (
            Toto2GluonTSModel,
            Toto2GluonTSModelConfig,
            Toto2Model as Toto2CoreModel,
        )
    except Exception as exc:
        raise ImportError(
            "[Toto2] Missing local Toto 2.0 source or dependencies. "
            "Copy the GitHub folder DataDog/toto:toto2/toto2 into "
            "src/model_zoo/TSFM_src/toto2, and install Toto2 dependencies "
            "such as dd-unit-scaling and gluonts>=0.16 when needed."
        ) from exc
    return Toto2CoreModel, Toto2GluonTSModel, Toto2GluonTSModelConfig


class Toto2Model(BaseModel):
    def __init__(self, args, module_name, model_name, model_local_path):
        self.args = args
        self.module_name = module_name
        self.model_name = model_name
        self.model_local_path = model_local_path
        self.output_dir = os.path.join(self.args.output_dir, self.model_name)
        super().__init__(self.model_name, args, self.output_dir)

    def get_predictor(self, dataset, batch_size):
        return Toto2ModelPredictorWrapper(
            config=self.args,
            batch_size=batch_size,
            model_path=self.model_local_path,
            prediction_length=dataset.prediction_length,
            ds_freq=dataset.freq,
            target_dim=getattr(dataset, "target_dim", 1),
            past_feat_dynamic_real_dim=getattr(dataset, "past_feat_dynamic_real_dim", 0),
            feat_dynamic_real_dim=getattr(dataset, "feat_dynamic_real_dim", 0),
        )


class Toto2ModelPredictorWrapper:
    def __init__(
        self,
        config,
        batch_size: int,
        model_path: str,
        prediction_length: int,
        ds_freq: str,
        target_dim: int = 1,
        past_feat_dynamic_real_dim: int = 0,
        feat_dynamic_real_dim: int = 0,
        *args,
        **kwargs,
    ):
        self.config = config
        self.batch_size = batch_size
        self.model_path = model_path
        self.prediction_length = int(prediction_length)
        self.ds_freq = ds_freq
        self.target_dim = int(target_dim)
        self.past_feat_dynamic_real_dim = int(past_feat_dynamic_real_dim)
        self.feat_dynamic_real_dim = int(feat_dynamic_real_dim)
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.context_length = (
            int(self.config.context_len)
            if getattr(self.config, "fix_context_len", False)
            else 4096
        )
        self.quantiles = GE_QUANTILES
        self.impute_missing = True

        Toto2CoreModel, Toto2GluonTSModel, Toto2GluonTSModelConfig = _import_toto2_classes()

        core_model = Toto2CoreModel.from_pretrained(
            self.model_path,
            map_location=self.device,
        ).to(self.device)
        core_model.eval()

        gts_config = Toto2GluonTSModelConfig(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            target_dim=self.target_dim,
            past_feat_dynamic_real_dim=self.past_feat_dynamic_real_dim,
            feat_dynamic_real_dim=self.feat_dynamic_real_dim,
            has_missing_values=True,
            quantiles=self.quantiles,
            imputation_internal="ffill",
        )
        self.gts_model = Toto2GluonTSModel(core_model, gts_config).to(self.device).eval()
        self.predictor = self.gts_model.create_predictor(
            batch_size=self.batch_size,
            device=self.device,
        )

        print(
            f"[Toto2] context_len={self.context_length}, "
            f"batch_size={self.batch_size}, "
            f"prediction_length={self.prediction_length}, "
            f"target_dim={self.target_dim}, "
            f"freq_used=False, "
            f"impute_missing={self.impute_missing}, "
            f"quantiles={self.quantiles}"
        )

    def predict(self, test_data_input: List[dict], batch_size: int = None):
        if batch_size is not None and int(batch_size) != int(self.batch_size):
            self.predictor = self.gts_model.create_predictor(
                batch_size=int(batch_size),
                device=self.device,
            )
            self.batch_size = int(batch_size)
        deterministic_enabled = torch.are_deterministic_algorithms_enabled()
        if deterministic_enabled and str(self.device).startswith("cuda"):
            torch.use_deterministic_algorithms(False)

        def _iter_forecasts():
            try:
                yield from self.predictor.predict(test_data_input)
            finally:
                if deterministic_enabled:
                    try:
                        torch.use_deterministic_algorithms(True, warn_only=True)
                    except TypeError:
                        torch.use_deterministic_algorithms(True)

        return _iter_forecasts()

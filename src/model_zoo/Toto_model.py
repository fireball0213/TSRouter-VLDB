# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import logging
from typing import List

import numpy as np
import torch
from torch import cuda
from tqdm.auto import tqdm

from gluonts.model.forecast import QuantileForecast, SampleForecast
from gluonts.itertools import batcher

from model_zoo.base_model import BaseModel

_TSFMSRC_DIR = Path(__file__).resolve().parent / "TSFM_src"
if _TSFMSRC_DIR.exists():
    _p = str(_TSFMSRC_DIR)
    if _p not in sys.path:
        sys.path.insert(0, _p)                   
else:
    raise FileNotFoundError(f"[Toto] TSFM_src directory not found: {_TSFMSRC_DIR}")

from toto.inference.gluonts_predictor import Multivariate, TotoPredictor
from toto.model.toto import Toto


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



class TotoModel(BaseModel):

    def __init__(self, args, module_name, model_name, model_local_path):
        self.args = args
        self.module_name = module_name
        self.model_name = model_name
        self.model_local_path = model_local_path
        self.output_dir = os.path.join(self.args.output_dir, self.model_name)

        super().__init__(self.model_name, args, self.output_dir)

    def get_predictor(self, dataset, batch_size):
        predictor = TotoModelPredictorWrapper(
            config=self.args,
            batch_size=batch_size,
            model_path=self.model_local_path,
            prediction_length=dataset.prediction_length,
            ds_freq=dataset.freq,
            target_dim=getattr(dataset, "target_dim", 1),
        )
        return predictor


class TotoModelPredictorWrapper:
    def __init__(
        self,
        config,
        batch_size: int,
        model_path: str,
        prediction_length: int,
        ds_freq: str,
        target_dim: int = 1,
        past_feat_dynamic_real_dim: int = 0,
        *args,
        **kwargs,
    ):
        self.config = config             
        self.batch_size = batch_size
        self.model_path = model_path       
        self.prediction_length = prediction_length
        self.ds_freq = ds_freq             
        self.target_dim = target_dim
        self.past_feat_dynamic_real_dim = past_feat_dynamic_real_dim

        self.device = torch.device("cuda" if cuda.is_available() else "cpu")
        self.impute_missing = False

        if Toto is None:
             raise ImportError("Toto library is not installed.")
             
        self.model = Toto.from_pretrained(self.model_path).to(self.device)
        self.model = self.model.eval()

        if getattr(self.config, "fix_context_len", False):
            self.context_length = self.config.context_len
        else:
            self.context_length = 4096

        self.freq = ds_freq
        self.quantiles = None

        context_info = (
            self.context_length
            if getattr(self.config, "fix_context_len", False)
            else "full_history"
        )
        print(
            f"[Toto] context_len={context_info}, "
            f"batch_size={self.batch_size}, "
            f"freq_used=False, "
            f"impute_missing={self.impute_missing}"
        )

        # ============================================================
               
        # ============================================================

    def predict(self, test_data_input: List[dict], batch_size: int = None) -> List:
        if batch_size is None:
            batch_size = self.batch_size

        num_samples = 100

        # ============================================================
                                  
        # ============================================================
        debug_on = bool(
            getattr(self.config, "debug_mode", False)
            or getattr(self.config, "debug", False)
            or getattr(self.config, "verbose", False)
        )

        import pandas as pd
        import calendar
        from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime

        # ============================================================
                                                                   
             
                                                              
                                                                  
             
                              
                                                                            
                            
        # ============================================================
        _orig_to_timestamp = pd.PeriodIndex.to_timestamp

        def _manual_periodindex_to_datetime64ms(pidx: "pd.PeriodIndex", freq=None, how="start"):
            f = freq or getattr(pidx, "freqstr", None) or getattr(pidx, "freq", None)
            if hasattr(f, "freqstr"):
                f = f.freqstr
            f = str(f)

            def _last_day(y, m):
                return calendar.monthrange(y, m)[1]

            out = []
            for p in pidx:
                if f.startswith(("A", "Y")):
                    y = p.year
                    if how == "end":
                        out.append(np.datetime64(f"{y:04d}-12-31T00:00:00.000", "ms"))
                    else:
                        out.append(np.datetime64(f"{y:04d}-01-01T00:00:00.000", "ms"))

                elif f.startswith("Q"):
                    y = p.year
                    q = p.quarter
                    sm = (q - 1) * 3 + 1
                    em = sm + 2
                    if how == "end":
                        d = _last_day(y, em)
                        out.append(np.datetime64(f"{y:04d}-{em:02d}-{d:02d}T00:00:00.000", "ms"))
                    else:
                        out.append(np.datetime64(f"{y:04d}-{sm:02d}-01T00:00:00.000", "ms"))

                elif f.startswith("M"):
                    y = p.year
                    m = p.month
                    if how == "end":
                        d = _last_day(y, m)
                        out.append(np.datetime64(f"{y:04d}-{m:02d}-{d:02d}T00:00:00.000", "ms"))
                    else:
                        out.append(np.datetime64(f"{y:04d}-{m:02d}-01T00:00:00.000", "ms"))
                else:
                                                            
                    s = str(p)
                    if len(s) == 4 and s.isdigit():  # YYYY
                        s = f"{s}-01-01"
                    elif len(s) == 7:  # YYYY-MM
                        s = f"{s}-01"
                    out.append(np.datetime64(f"{s}T00:00:00.000", "ms"))

            return np.array(out, dtype="datetime64[ms]")

                                                    
        _oob_count = {"n": 0}
        _oob_print_limit = 2                           

        def _safe_to_timestamp(self, freq=None, how="start"):
            try:
                return _orig_to_timestamp(self, freq=freq, how=how)
            except OutOfBoundsDatetime as e:
                _oob_count["n"] += 1

                                                
                if debug_on and _oob_count["n"] <= _oob_print_limit:
                    try:
                        freqstr = getattr(self, "freqstr", None)
                        pmin = self.min()
                        pmax = self.max()
                    except Exception:
                        freqstr, pmin, pmax = None, None, None

                    print(
                        f"[Toto][OOB] #{_oob_count['n']} PeriodIndex.to_timestamp overflow! "
                        f"freq_arg={freq}, how={how}, self.freqstr={freqstr}, "
                        f"min_period={pmin}, max_period={pmax}, len={len(self)}"
                    )
                    print(f"[Toto][OOB] pandas err: {e}")

                arr_ms = _manual_periodindex_to_datetime64ms(self, freq=freq, how=how)

                                        
                if debug_on and _oob_count["n"] == 1:
                    print(f"[Toto][OOB] Fallback -> numpy datetime64[ms], head={arr_ms[:3]}")

                return arr_ms

                  
        pd.PeriodIndex.to_timestamp = _safe_to_timestamp

        try:
            input_list = list(test_data_input)

            # ============================================================
                                                 
            # ============================================================
            if debug_on and len(input_list) > 0:
                first = input_list[0]
                _st = first.get("start", None)
                _fq = getattr(_st, "freqstr", None)
                max_target_len = max(len(e.get("target", [])) for e in input_list)
                print(
                    f"[Toto][Stat] freq={_fq}, "
                    f"num_series={len(input_list)}, "
                    f"max_target_len={max_target_len}, "
                    f"pred_len={self.prediction_length}, "
                    f"context_len={self.context_length}"
                )

            modified_input = []
            shifts = {}

            # ============================================================
                                      
                                                 
                                                      
            # ============================================================
            safe_start_year = 2000

            for i, entry in enumerate(input_list):
                start = entry["start"]

                if debug_on and i < 3:
                    print(
                        f"[Toto][Input] i={i} original_start={start} "
                        f"freq={getattr(start, 'freqstr', None)} "
                        f"target_len={len(entry.get('target', []))}"
                    )

                try:
                    new_start = pd.Period(f"{safe_start_year}-01-01", freq=start.freq)
                    shifts[i] = start
                    new_entry = entry.copy()
                    new_entry["start"] = new_start
                    modified_input.append(new_entry)
                except Exception as e:
                    print(f"[Toto] Warning: Failed to shift date for item {i}: {e}")
                    modified_input.append(entry)

            predictor = TotoPredictor.create_for_eval(
                model=self.model,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                mode=Multivariate(batch_size=batch_size),
                samples_per_batch=num_samples,
            )

            forecasts = list(
                predictor.predict(
                    modified_input,
                    use_kv_cache=True,
                    num_samples=num_samples,
                )
            )

                                        
            for i, original_start in shifts.items():
                if i < len(forecasts):
                    target_len = len(input_list[i]["target"])
                    forecasts[i].start_date = original_start + target_len

            if debug_on and _oob_count["n"] > 0:
                print(f"[Toto][OOB] Total overflow-fallback calls in this predict(): {_oob_count['n']}")

            return forecasts

        finally:
            # ============================================================
                                                   
            # ============================================================
            pd.PeriodIndex.to_timestamp = _orig_to_timestamp

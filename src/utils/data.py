# Copyright (c) 2023, Salesforce, Inc.
# Copyright (c) 2025 fireball0213, LAMDA, Nanjing University
# SPDX-License-Identifier: Apache-2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
from functools import cached_property
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple
from gluonts.model.forecast import SampleForecast, QuantileForecast
import json
import numpy as np
import pandas as pd
import datasets
from huggingface_hub import snapshot_download
from dataclasses import dataclass
from types import SimpleNamespace
from dotenv import load_dotenv
from gluonts.dataset import DataEntry
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import TestData, TrainingDataset, split
from gluonts.itertools import Map
from gluonts.time_feature import norm_freq_str
from utils.path_utils import resolve_tsfm_result_path
from gluonts.transform import Transformation
from pandas.tseries.frequencies import to_offset
import pyarrow.compute as pc
from toolz import compose
from utils.project_paths import CHANNEL_META_PATH, PROJECT_ROOT

                                                    

                
TEST_SPLIT = 0.1
GIFT_EVAL_REPOSITORY = "Salesforce/GiftEval"

                 
MAX_WINDOW = 20

               
M4_PRED_LENGTH_MAP = {
    "A": 6,
    "Q": 8,
    "M": 18,
    "W": 13,
    "D": 14,
    "H": 48,
    # new version fix:
    "h": 48,
    "Y": 6,

}

                
PRED_LENGTH_MAP = {
    "M": 12,
    "W": 8,
    "D": 30,
    "H": 48,
    "T": 48,
    "S": 60,
    # new version fix:
    "h": 48,
    "s": 60,
    "min": 48,
}

               
TFB_PRED_LENGTH_MAP = {
    "A": 6,
    "H": 48,
    "Q": 8,
    "D": 14,
    "M": 18,
    "W": 13,
    "U": 8,
    "T": 8,
    # new version fix:
    "min": 8,
    "us": 8,
    "Y": 6,
    "h": 48,
}


class Term(Enum):
    'TSRouter runtime message.'
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def multiplier(self) -> int:
        if self == Term.SHORT:
            return 1
        elif self == Term.MEDIUM:
            return 10
        elif self == Term.LONG:
            return 15


def itemize_start(data_entry: DataEntry) -> DataEntry:
    data_entry["start"] = data_entry["start"].item()
    return data_entry




class MultivariateToUnivariate(Transformation):
    'TSRouter runtime message.'
    def __init__(self, field):
        self.field = field

    def __call__(
            self, data_it: Iterable[DataEntry], is_train: bool = False
    ) -> Iterator:
        for data_entry in data_it:
            item_id = data_entry["item_id"]
            val_ls = list(data_entry[self.field])
            for id, val in enumerate(val_ls):
                univariate_entry = data_entry.copy()
                univariate_entry[self.field] = val
                univariate_entry["item_id"] = item_id + "_dim" + str(id)
                yield univariate_entry


class Dataset:
    'TSRouter runtime message.'
    def __init__(
            self,
            name: str,
            term: Term | str = Term.SHORT,
            to_univariate: bool = False,
            storage_env_var: str = "Project_Path",
    ):
        load_dotenv()
        storage_path = self._resolve_storage_path(name, storage_env_var)
        self.hf_dataset = datasets.load_from_disk(str(storage_path / name)).with_format(
            "numpy"
        )
        process = ProcessDataEntry(
            self.freq,
            one_dim_target=self.target_dim == 1,
        )

        self.gluonts_dataset = Map(compose(process, itemize_start), self.hf_dataset)
        if to_univariate:
            self.gluonts_dataset = MultivariateToUnivariate("target").apply(
                self.gluonts_dataset
            )

        self.term = Term(term)
        self.name = name

    @staticmethod
    def _resolve_storage_path(name: str, storage_env_var: str) -> Path:
        explicit_root = os.getenv("TSROUTER_GIFTEVAL_ROOT")
        if explicit_root:
            storage_path = Path(explicit_root).expanduser().resolve()
        else:
            workspace_root = os.getenv(storage_env_var)
            if workspace_root:
                storage_path = Path(workspace_root).expanduser().resolve() / "Dataset"
            else:
                storage_path = (PROJECT_ROOT / "data" / "gifteval").resolve()

        dataset_path = storage_path / name
        if dataset_path.exists():
            return storage_path

        try:
            snapshot_download(
                repo_id=GIFT_EVAL_REPOSITORY,
                repo_type="dataset",
                local_dir=str(storage_path),
                allow_patterns=[f"{name}/**"],
            )
        except Exception as exc:
            raise FileNotFoundError(
                f"GIFT-Eval dataset '{name}' is unavailable at {dataset_path}. "
                "Set TSROUTER_GIFTEVAL_ROOT to a local GIFT-Eval directory or enable access to "
                f"https://huggingface.co/datasets/{GIFT_EVAL_REPOSITORY}."
            ) from exc

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"GIFT-Eval download completed without dataset directory: {dataset_path}."
            )
        return storage_path

    @cached_property
    def prediction_length(self) -> int:
        'TSRouter runtime message.'
        freq = norm_freq_str(to_offset(self.freq).name)
        if freq.endswith("E"):
            freq = freq[:-1]
        pred_len = (
            M4_PRED_LENGTH_MAP[freq] if "m4" in self.name else PRED_LENGTH_MAP[freq]
        )
        return self.term.multiplier * pred_len

    @cached_property
    def freq(self) -> str:
        'TSRouter runtime message.'
        return self.hf_dataset[0]["freq"]

    @cached_property
    def target_dim(self) -> int:
        'TSRouter runtime message.'
        target = self.hf_dataset[0]["target"]
        return target.shape[0] if target.ndim > 1 else 1

    @cached_property
    def past_feat_dynamic_real_dim(self) -> int:
        'TSRouter runtime message.'
        first = self.hf_dataset[0]
        if "past_feat_dynamic_real" not in first:
            return 0

        past_feat_dynamic_real = first["past_feat_dynamic_real"]
        return past_feat_dynamic_real.shape[0] if past_feat_dynamic_real.ndim > 1 else 1

    @cached_property
    def windows(self) -> int:
        'TSRouter runtime message.'
        if "m4" in self.name:
            return 1
        w = math.ceil(TEST_SPLIT * self._min_series_length / self.prediction_length)
        return min(max(1, w), MAX_WINDOW)

    @cached_property
    def _min_series_length(self) -> int:
        'TSRouter runtime message.'
        column = self.hf_dataset.data.column("target")
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(pc.list_slice(column, 0, 1))
            )
        else:
            lengths = pc.list_value_length(column)
        return int(min(lengths.to_numpy()))

    @cached_property
    def sum_series_length(self) -> int:
        'TSRouter runtime message.'
        column = self.hf_dataset.data.column("target")
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(pc.list_flatten(column))
        else:
            lengths = pc.list_value_length(column)
        return int(sum(lengths.to_numpy()))

                                          

    @property
    def training_dataset(self) -> TrainingDataset:
        'TSRouter runtime message.'
        training_dataset, _ = split(
            self.gluonts_dataset,
            offset=-self.prediction_length * (self.windows + 1),
        )
        return training_dataset

    @property
    def validation_dataset(self) -> TrainingDataset:
        'TSRouter runtime message.'
        validation_dataset, _ = split(
            self.gluonts_dataset,
            offset=-self.prediction_length * self.windows,
        )
        return validation_dataset

    @property
    def test_data(self) -> TestData:
        'TSRouter runtime message.'
        _, test_template = split(
            self.gluonts_dataset,
            offset=-self.prediction_length * self.windows,
        )
        test_data = test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )
        return test_data

                                                                                           

def gluonts_to_numpy(input_dataset: List[SampleForecast]):
    'TSRouter runtime message.'
    data_list: List[np.ndarray] = []
    for forecast in input_dataset:
                                               
        data_list.append(forecast.samples.T)

    data_array = np.stack(data_list, axis=0)  # (N, T, C)
                                               
    return data_array

def load_forecasts_from_npy(
    samples_path: str,
    meta_path: str,
    freq: str,
) -> List[SampleForecast]:
    'TSRouter runtime message.'
    samples = np.load(samples_path)  # (N_series, N_samples, pred_len, C)

    with open(meta_path, "r") as fp:
        meta = json.load(fp)

    if len(meta) != samples.shape[0]:
        raise ValueError('TSRouter runtime message.')

    forecasts: List[SampleForecast] = []
    for idx, info in enumerate(meta):
        item_id = info["item_id"]
        start_date = pd.Period(info["start_date"], freq=freq)
        sample_arr = samples[idx]  # (num_samples, pred_len, C)

        sf = SampleForecast(
            samples=sample_arr,
            start_date=start_date,
            item_id=item_id,
        )
        forecasts.append(sf)

    return forecasts


def _reshape_saved_samples_to_4d(
    samples: np.ndarray,
    channels: int,
    windows: int,
    pred_len: int,
) -> np.ndarray:
    'TSRouter runtime message.'
    if samples.ndim == 4:
                           
        return samples

    if samples.ndim == 3:
                                   
                            
                                                   
        n_mul_c, n_samples, t = samples.shape
        if t != pred_len:
            raise ValueError(
                f"TSRouter runtime message: {samples.shape}, pred_len={pred_len}"
            )
        if channels <= 0 or windows <= 0:
            raise ValueError(f"TSRouter runtime message: {channels}, windows={windows}")

                                                   
                                                        
        restored = samples.reshape(-1, channels, windows, n_samples, pred_len)
                                               
        restored = restored.transpose(0, 2, 3, 4, 1)
                                             
        restored = restored.reshape(-1, n_samples, pred_len, channels)
        return restored

    raise ValueError(f"TSRouter runtime message: {samples.shape}")


def load_gluonts_pred(base_path: str, model_name: str, model_cl_name: str, dataset_name: str, pred_len, channels, windows, verobse=False):
    'TSRouter runtime message.'
             
    samples_path = resolve_tsfm_result_path(
        base_path,
        model_name,
        model_cl_name,
        "npy",
        f"{dataset_name}_samples.npy",
    )
    meta_path = resolve_tsfm_result_path(
        base_path,
        model_name,
        model_cl_name,
        "meta",
        f"{dataset_name}_meta.json",
    )

               
    samples = np.load(samples_path)
    if verobse:
        print(f"Load {model_name} pred: {samples.shape}", end=" ")

              
    with open(meta_path, "r") as fp:
        meta = json.load(fp)

    entries = meta.get("entries", None)
    performance = meta.get("performance", None) or {}
    runtime_seconds = (
        performance.get("forward_runtime_seconds", None)
        or performance.get("non_eval_runtime_seconds", None)
        or performance.get("runtime_seconds", None)
    )

                                
    samples_4d = _reshape_saved_samples_to_4d(
        samples=samples,
        channels=channels,
        windows=windows,
        pred_len=pred_len,
    )

                          
    median_forecast = np.median(samples_4d, axis=1)

    if verobse:
        print('median_forecast:', median_forecast.shape, end=" ")

    if median_forecast.ndim == 2:
        median_forecast = median_forecast[..., None]
    elif median_forecast.ndim != 3:
        raise ValueError(f"TSRouter runtime message: {median_forecast.shape}")
    if verobse:
        print('to ', median_forecast.shape, end=" ")
                                                  

             
    freq = dataset_name.split("_")[-2]

                             
    pred_dataset: List[SampleForecast] = []
    n_series = median_forecast.shape[0]

    for idx in range(n_series):
        if entries is None or idx >= len(entries):
            raise ValueError(f"TSRouter runtime message: {idx}。")

        info = entries[idx]
        item_id = str(info["item_id"])
        start = pd.Period(info["start_date"], freq=freq)

        target_array = median_forecast[idx]  # (pred_len, C)
        if target_array.ndim != 2:
            raise ValueError(f"TSRouter runtime message: {target_array.shape}")

                           
        target = target_array.T

        forecast = SampleForecast(
            samples=target,
            start_date=start,
            item_id=item_id,
        )
        pred_dataset.append(forecast)

    if verobse and pred_dataset:
        print(
            f"TSRouter runtime message: {len(pred_dataset)}，"
            f"target shape: {pred_dataset[0].samples.shape}"
        )

    return pred_dataset, samples, runtime_seconds


def load_gluonts_pred_distribution(
    base_path: str,
    model_name: str,
    model_cl_name: str,
    dataset_name: str,
    pred_len: int,
    channels: int,
    windows: int,
    verbose: bool = False,
):
    'TSRouter runtime message.'
    samples_path = resolve_tsfm_result_path(
        base_path,
        model_name,
        model_cl_name,
        "npy",
        f"{dataset_name}_samples.npy",
    )
    meta_path = resolve_tsfm_result_path(
        base_path,
        model_name,
        model_cl_name,
        "meta",
        f"{dataset_name}_meta.json",
    )

    samples = np.load(samples_path)
    with open(meta_path, "r") as fp:
        meta = json.load(fp)

    entries = meta.get("entries", None)
    performance = meta.get("performance", None) or {}
    runtime_seconds = (
        performance.get("forward_runtime_seconds", None)
        or performance.get("non_eval_runtime_seconds", None)
        or performance.get("runtime_seconds", None)
    )

    samples_4d = _reshape_saved_samples_to_4d(
        samples=samples,
        channels=channels,
        windows=windows,
        pred_len=pred_len,
    )  # (N, S, T, C)

    freq = dataset_name.split("_")[-2]
    pred_dataset: List[SampleForecast] = []
    n_series = samples_4d.shape[0]

    if entries is None or len(entries) < n_series:
        raise ValueError(
            f"TSRouter runtime message: {0 if entries is None else len(entries)}, n_series={n_series}"
        )

    for idx in range(n_series):
        info = entries[idx]
        item_id = str(info["item_id"])
        start = pd.Period(info["start_date"], freq=freq)

        sample_arr = samples_4d[idx]  # (S, T, C)
        if sample_arr.ndim != 3:
            raise ValueError(f"TSRouter runtime message: {sample_arr.shape}")

        if sample_arr.shape[-1] == 1:
                                         
            sample_arr = sample_arr[:, :, 0]

        forecast = SampleForecast(
            samples=sample_arr.astype(np.float32),
            start_date=start,
            item_id=item_id,
        )
        pred_dataset.append(forecast)

    if verbose and pred_dataset:
        print(
            f"[load_gluonts_pred_distribution] {model_name} -> "
            f"n={len(pred_dataset)}, sample_shape={pred_dataset[0].samples.shape}"
        )

    return pred_dataset, samples_4d, runtime_seconds


def numpy_to_gluonts(data_array, template_dataset):
    'TSRouter runtime message.'
    forecasts: List[SampleForecast] = []
    N, T, C = data_array.shape

    if len(template_dataset) != N:
        raise ValueError(
            f"TSRouter runtime message: {len(template_dataset)}TSRouter runtime message: {N}TSRouter runtime message: "
        )

    for i in range(N):
        base_sample = template_dataset[i]
        item_id = base_sample.item_id
        start_date = base_sample.start_date

        # data_array[i]: (T, C)
        sample_array = data_array[i]

                                                                                     
        if C > 1:
            sample_array = sample_array.reshape(1, T, C)
        else:
            sample_array = sample_array.reshape(1, T)

        forecast = SampleForecast(
            samples=sample_array.astype(np.float32),
            start_date=start_date,
            item_id=item_id,
        )
        forecasts.append(forecast)

                                                              
    return forecasts


def numpy_samples_to_gluonts(samples_array: np.ndarray, template_dataset):
    'TSRouter runtime message.'
    forecasts: List[SampleForecast] = []
    if samples_array.ndim != 4:
        raise ValueError(f"TSRouter runtime message: {samples_array.shape}")

    N, S, T, C = samples_array.shape
    if len(template_dataset) != N:
        raise ValueError(
            f"TSRouter runtime message: {len(template_dataset)}TSRouter runtime message: {N}TSRouter runtime message: "
        )

    for i in range(N):
        base_sample = template_dataset[i]
        item_id = base_sample.item_id
        start_date = base_sample.start_date

        arr = samples_array[i]  # (S, T, C)
        if C == 1:
            arr = arr[:, :, 0]  # (S, T)

        forecasts.append(
            SampleForecast(
                samples=arr.astype(np.float32),
                start_date=start_date,
                item_id=item_id,
            )
        )

    return forecasts




@dataclass
class FastEvalDatasetStub:
    prediction_length: int
    target_dim: int
    windows: int
    freq: str
    test_data: object


class FastEvalDatasetCacheLoader:
    'TSRouter runtime message.'

    def __init__(
        self,
        all_datasets,
        med_long_datasets,
        build_ds_meta_fn,
        decide_univariate_fn,
        *,
        cache_only: bool = False,
        metadata_path: Path = CHANNEL_META_PATH,
    ):
        self.all_datasets = list(all_datasets)
        self.med_long_datasets = set(str(med_long_datasets).split())
        self.build_ds_meta = build_ds_meta_fn
        self.decide_univariate = decide_univariate_fn
        self.cache_only = bool(cache_only)
        self.metadata_path = Path(metadata_path)
        self._cache = None

    def _preload_metadata_stubs(self):
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"route metadata is unavailable: {self.metadata_path}")
        frame = pd.read_csv(self.metadata_path)
        required = {"dataset", "pl", "target_dim", "windows_total"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"route metadata is missing columns: {', '.join(missing)}")

        cache = {}
        for ds_name in self.all_datasets:
            for term in ("short", "medium", "long"):
                if term in {"medium", "long"} and ds_name not in self.med_long_datasets:
                    continue
                _, _, ds_config, _ = self.build_ds_meta(ds_name, term)
                rows = frame[frame["dataset"].astype(str).eq(str(ds_config))]
                if rows.empty:
                    raise ValueError(f"route metadata has no entry for {ds_config}")
                row = rows.iloc[0]
                cache[ds_config] = {
                    "prediction_length": int(row["pl"]),
                    "target_dim": int(row["target_dim"]),
                    "windows": int(row["windows_total"]),
                    "freq": str(row.get("actual_freq", row.get("freq", ""))),
                    "search_input": [],
                }
        return cache

    def preload(self):
        if self._cache is not None:
            return self._cache
        if self.cache_only:
            self._cache = self._preload_metadata_stubs()
            return self._cache
        cache = {}
        for ds_name in self.all_datasets:
            for term in ["short", "medium", "long"]:
                if term in {"medium", "long"} and ds_name not in self.med_long_datasets:
                    continue
                _, _, ds_config, _ = self.build_ds_meta(ds_name, term)
                to_univariate = self.decide_univariate(ds_name, term)
                ds = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
                search_input = list(ds.test_data.input)
                cache[ds_config] = {
                    "prediction_length": int(ds.prediction_length),
                    "target_dim": int(ds.target_dim),
                    "windows": int(ds.windows),
                    "freq": str(ds.freq),
                    "search_input": search_input,
                }
        self._cache = cache
        return self._cache

    def build_stub(self, ds_config: str) -> FastEvalDatasetStub:
        cache = self.preload()
        item = cache.get(str(ds_config))
        if item is None:
            raise ValueError(f"TSRouter runtime message: {ds_config}")
        return FastEvalDatasetStub(
            prediction_length=item["prediction_length"],
            target_dim=item["target_dim"],
            windows=item["windows"],
            freq=item["freq"],
            test_data=SimpleNamespace(input=item["search_input"]),
        )

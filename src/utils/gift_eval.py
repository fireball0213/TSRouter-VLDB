from __future__ import annotations

import logging
from collections import ChainMap
from dataclasses import dataclass, field
from itertools import chain, groupby
from typing import Any, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from toolz import first, valmap
from tqdm.auto import tqdm

from gluonts.dataset import DataEntry
from gluonts.dataset.split import TestData
from gluonts.ev.ts_stats import seasonal_error
from gluonts.itertools import batcher, prod
from gluonts.model import Forecast


logger = logging.getLogger(__name__)


@dataclass
class CachedBatchForecast:
    """Batch forecast mapping that avoids rebuilding the same quantile arrays."""

    forecasts: List[Forecast]
    allow_nan: bool = False
    _cache: dict[str, np.ndarray] = field(default_factory=dict)

    def __getitem__(self, name):
        key = str(name)
        if key not in self._cache:
            values = [forecast[name].T for forecast in self.forecasts]
            result = np.stack(values, axis=0)
            if np.isnan(result).any():
                if not self.allow_nan:
                    raise ValueError("Forecast contains NaN values")
                logger.warning("Forecast contains NaN values. Metrics may be incorrect.")
            self._cache[key] = result
        return self._cache[key]


def _entry_group_key(index: int, entry: DataEntry) -> tuple:
    item_id = entry.get("item_id")
    if item_id is None:
        return ("__entry__", index)
    target = np.asarray(entry["target"])
    return (
        str(item_id),
        str(entry.get("start", "")),
        target.ndim,
        tuple(target.shape[:-1]),
    )


def _seasonal_errors_for_group(
    entries: list[DataEntry],
    seasonality: int,
    mask_invalid_label: bool,
) -> list[np.ndarray]:
    if len(entries) == 1 or not mask_invalid_label:
        return [
            seasonal_error(
                np.ma.masked_invalid(np.asarray(entry["target"]))
                if mask_invalid_label
                else np.asarray(entry["target"]),
                seasonality=seasonality,
                time_axis=-1,
            )
            for entry in entries
        ]

    targets = [np.asarray(entry["target"]) for entry in entries]
    prefix_shapes = {target.shape[:-1] for target in targets}
    if len(prefix_shapes) != 1:
        return [
            seasonal_error(
                np.ma.masked_invalid(target),
                seasonality=seasonality,
                time_axis=-1,
            )
            for target in targets
        ]

    indexed_targets = sorted(
        enumerate(targets),
        key=lambda pair: int(pair[1].shape[-1]),
    )
    longest = np.asarray(indexed_targets[-1][1], dtype=np.float64)
    longest_length = int(longest.shape[-1])
    lag = int(seasonality) if int(seasonality) <= longest_length else 1
    if any(
        (int(seasonality) if int(target.shape[-1]) >= int(seasonality) else 1) != lag
        for target in targets
    ):
        return [
            seasonal_error(
                np.ma.masked_invalid(target),
                seasonality=seasonality,
                time_axis=-1,
            )
            for target in targets
        ]

    # GIFT-Eval test windows for one item are growing prefixes. Accumulate only
    # the newly exposed differences, keeping memory bounded for multivariate data.
    running_sum = np.zeros(longest.shape[:-1], dtype=np.float64)
    running_count = np.zeros(longest.shape[:-1], dtype=np.int64)
    processed = 0
    chunk_size = 65536
    values: list[np.ndarray | None] = [None] * len(targets)
    for original_index, target in indexed_targets:
        target_length = int(target.shape[-1])
        endpoint = target_length - lag
        if endpoint <= 0:
            values[original_index] = seasonal_error(
                np.ma.masked_invalid(target),
                seasonality=seasonality,
                time_axis=-1,
            )
            continue

        while processed < endpoint:
            stop = min(endpoint, processed + chunk_size)
            left = longest[..., processed:stop]
            right = longest[..., processed + lag:stop + lag]
            valid = np.isfinite(left) & np.isfinite(right)
            running_sum += np.sum(
                np.where(valid, np.abs(right - left), 0.0),
                axis=-1,
                dtype=np.float64,
            )
            running_count += np.sum(valid, axis=-1, dtype=np.int64)
            processed = stop

        mean = np.divide(
            running_sum,
            running_count,
            out=np.full(np.shape(running_sum), np.nan, dtype=np.float64),
            where=running_count > 0,
        )
        values[original_index] = np.ma.masked_invalid(
            np.expand_dims(mean.copy(), axis=-1)
        )
    return [value for value in values if value is not None]


def compute_seasonal_errors_fast(
    inputs: Iterable[DataEntry],
    *,
    seasonality: int,
    mask_invalid_label: bool = True,
) -> list[np.ndarray]:
    """Compute rolling-window seasonal errors once per underlying series."""

    indexed_inputs = enumerate(inputs)
    output: list[np.ndarray] = []
    for _, group in groupby(
        indexed_inputs,
        key=lambda pair: _entry_group_key(pair[0], pair[1]),
    ):
        entries = [entry for _, entry in group]
        output.extend(
            _seasonal_errors_for_group(
                entries,
                seasonality=int(seasonality),
                mask_invalid_label=mask_invalid_label,
            )
        )
    return output


def get_cached_seasonal_errors(
    dataset: Any,
    *,
    test_data: TestData | None = None,
    inputs: Iterable[DataEntry] | None = None,
    seasonality: int,
    mask_invalid_label: bool = True,
) -> list[np.ndarray]:
    """Return per-window seasonal errors cached on a dataset instance."""

    cache_key = (
        str(getattr(dataset, "name", "")),
        str(getattr(dataset, "term", "")),
        int(getattr(dataset, "prediction_length", 0) or 0),
        int(getattr(dataset, "windows", 0) or 0),
        int(getattr(dataset, "target_dim", 0) or 0),
        int(seasonality),
        bool(mask_invalid_label),
    )
    cache = getattr(dataset, "_gift_eval_seasonal_error_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(dataset, "_gift_eval_seasonal_error_cache", cache)
    if cache_key not in cache:
        if inputs is None:
            if test_data is None:
                test_data = dataset.test_data
            inputs = test_data.input
        cache[cache_key] = compute_seasonal_errors_fast(
            inputs,
            seasonality=int(seasonality),
            mask_invalid_label=mask_invalid_label,
        )
    return cache[cache_key]


def evaluate_forecasts_fast(
    forecasts: Iterable[Forecast],
    *,
    test_data: TestData,
    metrics,
    axis: Optional[Union[int, tuple]] = None,
    batch_size: int = 100,
    mask_invalid_label: bool = True,
    allow_nan_forecast: bool = False,
    seasonality: Optional[int] = None,
    seasonal_errors: Optional[Iterable[np.ndarray]] = None,
) -> pd.DataFrame:
    """GluonTS-compatible evaluation with rolling seasonal-error reuse."""

    if seasonality is None:
        raise ValueError("fast evaluation requires an explicit seasonality")

    label_iter = iter(test_data.label)
    try:
        first_label = next(label_iter)
    except StopIteration:
        raise ValueError("cannot evaluate an empty test dataset")

    label_ndim = np.asarray(first_label["target"]).ndim
    assert label_ndim in [1, 2]

    if axis is None:
        axis = tuple(range(label_ndim + 1))
    if isinstance(axis, int):
        axis = (axis,)
    assert all(ax in range(3) for ax in axis)

    evaluators = {}
    for metric in metrics:
        evaluator = metric(axis=axis)
        evaluators[evaluator.name] = evaluator

    if seasonal_errors is None:
        seasonal_errors = compute_seasonal_errors_fast(
            test_data.input,
            seasonality=int(seasonality),
            mask_invalid_label=mask_invalid_label,
        )

    index_data = []
    label_batches = batcher(
        chain((first_label,), label_iter),
        batch_size=batch_size,
    )
    forecast_batches = batcher(forecasts, batch_size=batch_size)
    seasonal_batches = batcher(seasonal_errors, batch_size=batch_size)

    progress = tqdm()
    for label_batch, forecast_batch, seasonal_batch in zip(
        label_batches,
        forecast_batches,
        seasonal_batches,
    ):
        if 0 not in axis:
            index_data.extend(
                [
                    (forecast.item_id, forecast.start_date)
                    for forecast in forecast_batch
                ]
            )

        label_target = np.stack(
            [label["target"] for label in label_batch],
            axis=0,
        )
        if mask_invalid_label:
            label_target = np.ma.masked_invalid(label_target)

        data_batch = ChainMap(
            {
                "label": label_target,
                "seasonal_error": np.ma.stack(seasonal_batch, axis=0),
            },
            CachedBatchForecast(
                forecast_batch,
                allow_nan=allow_nan_forecast,
            ),
        )
        for evaluator in evaluators.values():
            evaluator.update(data_batch)
        progress.update(len(forecast_batch))
    progress.close()

    metric_values = {
        metric_name: evaluator.get()
        for metric_name, evaluator in evaluators.items()
    }
    if index_data:
        metric_values["__index_0"] = index_data

    index0 = metric_values.pop("__index_0", None)
    metric_shape = metric_values[first(metric_values)].shape
    if metric_shape == ():
        index = [None]
    else:
        index_arrays = np.unravel_index(
            range(prod(metric_shape)),
            metric_shape,
        )
        if index0 is not None:
            index0_repeated = np.take(index0, indices=index_arrays[0], axis=0)
            index_arrays = (*zip(*index0_repeated), *index_arrays[1:])
        index = pd.MultiIndex.from_arrays(index_arrays)

    flattened_metrics = valmap(np.ravel, metric_values)
    return pd.DataFrame(flattened_metrics, index=index)

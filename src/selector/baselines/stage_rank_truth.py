from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from utils.project_paths import BASELINE_CSV_ROOT, CHANNEL_META_PATH
from utils.path_utils import auto_cl_enabled, resolve_tsfm_csv_path
from utils.tsrouter_metrics import (
    _PER_CHANNEL_METRIC_IMPL,
    channel_orders_from_error_matrix,
    normalize_rank_truth_cl,
    parse_order_string,
    resolve_rank_truth_cl,
)


RANK_TRUTH_DIRNAME = "Rank_Truth_Select"
RANK_TRUTH_SCHEMA_VERSION = "stage_rank_truth_v1"
BASE_CHANNEL_META = CHANNEL_META_PATH


def rank_truth_root(args: Any | None = None) -> Path:
    raw = str(getattr(args, "rank_truth_output_dir", "") or "").strip() if args is not None else ""
    if raw:
        return Path(raw)
    return BASELINE_CSV_ROOT / "selectors" / RANK_TRUTH_DIRNAME


def _metric_col_name(metric: str) -> str:
    return "sMAPE" if str(metric).lower() == "smape" else str(metric).upper()


def _cl_sort_key(cl_token: str) -> int:
    match = re.search(r"\d+", str(cl_token))
    return int(match.group(0)) if match else 0


def required_rank_truth_cls(args: Any | None = None) -> list[str]:
    raw = str(getattr(args, "rank_truth_cls", "") or "").strip() if args is not None else ""
    cls: set[str] = set()
    if raw:
        for token in re.split(r"[,\s]+", raw):
            if token.strip():
                cls.add(normalize_rank_truth_cl(token))

    cls.add(resolve_rank_truth_cl(args))
    if args is not None and auto_cl_enabled(args):
        for attr in ("repr_input_dim", "short_repr_input_dim", "middle_repr_input_dim", "long_repr_input_dim"):
            raw_val = getattr(args, attr, None)
            try:
                val = int(raw_val)
            except Exception:
                continue
            if val > 0:
                cls.add(f"cl{val}")
    return sorted(cls, key=_cl_sort_key)


def rank_truth_stage_path(args: Any, rank_metric: str | None = None) -> Path:
    stage = int(getattr(args, "current_zoo_num", 0) or 0)
    total = int(getattr(args, "zoo_total_num", stage) or stage)
    metric = _metric_col_name(rank_metric or getattr(args, "sgl_rank_metric", "MASE"))
    return rank_truth_root(args) / f"zoo{stage}-{total}_rank_truth_{metric}.csv"


def _model_records(model_sizes: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for family, size_dict in (model_sizes or {}).items():
        for size_name, info in size_dict.items():
            records.append(
                {
                    "model_id": int(info["id"]),
                    "model_key": f"{family}_{size_name}",
                    "model_abbr": str(info.get("abbreviation", f"{family}_{size_name}")),
                }
            )
    return sorted(records, key=lambda rec: int(rec["model_id"]))


def _order_text(order: Iterable[int]) -> str:
    return "[" + " ".join(str(int(x)) for x in order) + "]"


def _expected_channel_rows(meta_df: pd.DataFrame, *, quick_test: bool = False) -> pd.DataFrame:
    df = meta_df.copy()
    if "load_status" in df.columns:
        df = df[df["load_status"].astype(str).str.lower().eq("ok")].copy()
    if quick_test:
        from config.dataset_config import ALL_Fast_DATASETS

        df = df[df["dataset"].astype(str).isin(set(ALL_Fast_DATASETS))].copy()
    df = df.dropna(subset=["dataset", "channel"]).copy()
    df["channel"] = pd.to_numeric(df["channel"], errors="coerce")
    df = df.dropna(subset=["channel"]).copy()
    df["channel"] = df["channel"].astype(int)
    return df


def _read_channel_meta(path: str | os.PathLike[str] = BASE_CHANNEL_META) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"channel_meta.csv not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty or not {"dataset", "channel"}.issubset(df.columns):
        raise FileNotFoundError(f"invalid channel_meta.csv: {path}")
    return df


def _build_task_orders_for_cl(
    *,
    model_records: list[dict[str, Any]],
    dataset_names: Iterable[str],
    cl_token: str,
    rank_metric: str,
) -> tuple[dict[str, list[int]], list[str]]:
    metric_col = _metric_col_name(rank_metric)
    per_model: dict[int, dict[str, float]] = {}
    problems: list[str] = []
    for rec in model_records:
        csv_path = resolve_tsfm_csv_path(
            str(rec["model_key"]),
            f"cl_{_cl_sort_key(cl_token)}",
            "all_results.csv",
        )
        if not csv_path.exists():
            problems.append(f"{rec['model_abbr']}: missing {csv_path}")
            continue
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as exc:
            problems.append(f"{rec['model_abbr']}: read_error {csv_path}: {exc}")
            continue
        if "dataset" not in df.columns or metric_col not in df.columns:
            problems.append(f"{rec['model_abbr']}: missing dataset/{metric_col} in {csv_path}")
            continue
        sub = df[["dataset", metric_col]].copy()
        sub["dataset"] = sub["dataset"].astype(str)
        sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
        sub = sub.dropna(subset=["dataset", metric_col])
        sub = sub.drop_duplicates("dataset", keep="last")
        per_model[int(rec["model_id"])] = {
            str(row["dataset"]): float(row[metric_col])
            for _, row in sub.iterrows()
            if np.isfinite(float(row[metric_col]))
        }

    orders: dict[str, list[int]] = {}
    for ds in sorted(set(str(x) for x in dataset_names)):
        rows: list[tuple[int, float]] = []
        missing: list[str] = []
        for rec in model_records:
            mid = int(rec["model_id"])
            values = per_model.get(mid)
            if values is None:
                missing.append(str(rec["model_abbr"]))
                continue
            value = values.get(str(ds))
            if value is None or not np.isfinite(float(value)):
                missing.append(str(rec["model_abbr"]))
                continue
            rows.append((mid, float(value)))
        if missing:
            problems.append(f"{ds}: missing task {metric_col} for {missing[:8]}")
            continue
        orders[ds] = [mid for mid, _ in sorted(rows, key=lambda item: (item[1], item[0]))]
    return orders, problems


def _build_channel_orders_for_cl(
    *,
    model_sizes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    dataset_names: Iterable[str],
    cl_token: str,
    rank_metric: str,
) -> tuple[dict[str, dict[int, list[int]]], list[str]]:
    metric_col = _metric_col_name(rank_metric)
    orders_by_dataset: dict[str, dict[int, list[int]]] = {}
    problems: list[str] = []
    model_cl_name = f"cl_{_cl_sort_key(cl_token)}"
    model_records = _model_records(model_sizes)
    per_model: dict[int, dict[str, pd.DataFrame]] = {}
    for rec in model_records:
        csv_path = resolve_tsfm_csv_path(
            str(rec["model_key"]),
            model_cl_name,
            "per_channel_results.csv",
        )
        if not csv_path.exists():
            problems.append(f"{rec['model_abbr']}: missing {csv_path}")
            continue
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as exc:
            problems.append(f"{rec['model_abbr']}: read_error {csv_path}: {exc}")
            continue
        required = {"dataset", "channel", "METRIC_IMPL", metric_col}
        if not required.issubset(df.columns):
            problems.append(
                f"{rec['model_abbr']}: missing columns in {csv_path}: {sorted(required - set(df.columns))}"
            )
            continue
        sub = df[["dataset", "channel", "METRIC_IMPL", metric_col]].copy()
        sub["dataset"] = sub["dataset"].astype(str)
        sub["channel"] = pd.to_numeric(sub["channel"], errors="coerce")
        sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
        sub = sub.dropna(subset=["dataset", "channel", metric_col]).copy()
        sub["channel"] = sub["channel"].astype(int)
        per_model[int(rec["model_id"])] = {
            str(dataset): group.sort_values("channel", kind="mergesort").drop_duplicates("channel", keep="last")
            for dataset, group in sub.groupby("dataset", sort=False)
        }

    for ds in sorted(set(str(x) for x in dataset_names)):
        rows: list[np.ndarray] = []
        model_ids: list[int] = []
        channels: list[int] | None = None
        missing: list[str] = []
        for rec in model_records:
            mid = int(rec["model_id"])
            by_dataset = per_model.get(mid)
            group = by_dataset.get(ds) if by_dataset is not None else None
            if group is None or group.empty:
                missing.append(str(rec["model_abbr"]))
                continue
            if not group["METRIC_IMPL"].astype(str).eq(_PER_CHANNEL_METRIC_IMPL).all():
                missing.append(f"{rec['model_abbr']}: outdated METRIC_IMPL")
                continue
            chs = [int(x) for x in group["channel"].tolist()]
            vals = pd.to_numeric(group[metric_col], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                missing.append(f"{rec['model_abbr']}: non-finite {metric_col}")
                continue
            if (vals < -1e-12).any():
                missing.append(f"{rec['model_abbr']}: negative {metric_col}")
                continue
            if channels is None:
                channels = chs
            elif chs != channels:
                missing.append(f"{rec['model_abbr']}: channel mismatch")
                continue
            rows.append(vals)
            model_ids.append(mid)
        if missing:
            problems.append(f"{ds}: missing/incomplete channel {metric_col} for {missing[:8]}")
            continue
        if not rows or channels is None:
            problems.append(f"{ds}: empty channel rank inputs")
            continue
        orders_by_dataset[ds] = channel_orders_from_error_matrix(
            np.stack(rows, axis=0),
            model_ids,
            channels,
        )
    return orders_by_dataset, problems


def _file_complete(
    *,
    path: Path,
    required_cls: list[str],
    meta_df: pd.DataFrame,
    expected_model_count: int,
    quick_test: bool = False,
) -> tuple[bool, str, pd.DataFrame | None]:
    if not path.exists():
        return False, f"missing {path}", None
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return False, f"read_error {path}: {exc}", None
    expected_rows = _expected_channel_rows(meta_df, quick_test=quick_test)
    if df.empty or "dataset" not in df.columns or "channel" not in df.columns:
        return False, "missing dataset/channel columns", df
    got_keys = {
        (str(row["dataset"]), int(row["channel"]))
        for _, row in _expected_channel_rows(df, quick_test=quick_test).iterrows()
    }
    expected_keys = {
        (str(row["dataset"]), int(row["channel"]))
        for _, row in expected_rows.iterrows()
    }
    missing_keys = expected_keys - got_keys
    if missing_keys:
        return False, f"missing channel rows={len(missing_keys)}", df

    for cl_token in required_cls:
        channel_col = f"real_channel_rank_model_ids_{cl_token}"
        task_col = f"real_task_rank_model_ids_{cl_token}"
        required = {channel_col, task_col}
        if not required.issubset(df.columns):
            return False, f"missing rank columns for {cl_token}: {sorted(required - set(df.columns))}", df
        check = df[df[["dataset", "channel"]].notna().all(axis=1)].copy()
        for col in [channel_col, task_col]:
            lengths = check[col].map(lambda value: len(parse_order_string(value)))
            if lengths.empty or int(lengths.min()) < int(expected_model_count):
                return False, f"incomplete {col}: min_len={int(lengths.min()) if not lengths.empty else 0}", df
    return True, "complete", df


def ensure_stage_rank_truth(
    *,
    args: Any,
    model_sizes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    force: bool = False,
) -> tuple[Path, pd.DataFrame]:
    total_t0 = time.perf_counter()
    rank_metric = _metric_col_name(getattr(args, "sgl_rank_metric", "MASE"))
    required_cls = required_rank_truth_cls(args)
    path = rank_truth_stage_path(args, rank_metric=rank_metric)
    root = path.parent
    root.mkdir(parents=True, exist_ok=True)
    meta_df = _read_channel_meta()
    quick_test = bool(getattr(args, "quick_test", False))
    expected_rows = _expected_channel_rows(meta_df, quick_test=quick_test)
    model_records = _model_records(model_sizes)
    if not model_records:
        raise ValueError("[RankTruth] empty model_sizes; cannot build rank truth")
    expected_model_count = len(model_records)
    stage = int(getattr(args, "current_zoo_num", expected_model_count) or expected_model_count)
    total = int(getattr(args, "zoo_total_num", stage) or stage)

    print(
        f"[RankTruth] check stage=z{stage}-{total}, metric={rank_metric}, "
        f"cls={required_cls}, path={path}",
        flush=True,
    )
    check_t0 = time.perf_counter()
    complete, reason, df_existing = _file_complete(
        path=path,
        required_cls=required_cls,
        meta_df=meta_df,
        expected_model_count=expected_model_count,
        quick_test=quick_test,
    )
    check_elapsed = time.perf_counter() - check_t0
    if complete and not force:
        print(
            f"[RankTruth] complete existing rows={len(df_existing) if df_existing is not None else 'unknown'} "
            f"check_s={check_elapsed:.2f}",
            flush=True,
        )
        return path, df_existing if df_existing is not None else pd.read_csv(path, low_memory=False)
    if force:
        print(f"[RankTruth] force rebuild requested: {path} check_s={check_elapsed:.2f}", flush=True)
    else:
        print(f"[RankTruth] rebuild required: {reason}; check_s={check_elapsed:.2f}", flush=True)

    out = expected_rows.copy()
    out.insert(0, "rank_truth_schema_version", RANK_TRUTH_SCHEMA_VERSION)
    out.insert(1, "stage", stage)
    out.insert(2, "zoo_total_num", total)
    out.insert(3, "rank_metric", rank_metric)
    dataset_names = sorted(out["dataset"].dropna().astype(str).unique().tolist())

    for cl_token in required_cls:
        cl_t0 = time.perf_counter()
        print(f"[RankTruth] build cl={cl_token}: datasets={len(dataset_names)}, models={expected_model_count}", flush=True)
        channel_t0 = time.perf_counter()
        channel_orders, channel_problems = _build_channel_orders_for_cl(
            model_sizes=model_sizes,
            dataset_names=dataset_names,
            cl_token=cl_token,
            rank_metric=rank_metric,
        )
        channel_elapsed = time.perf_counter() - channel_t0
        task_t0 = time.perf_counter()
        task_orders, task_problems = _build_task_orders_for_cl(
            model_records=model_records,
            dataset_names=dataset_names,
            cl_token=cl_token,
            rank_metric=rank_metric,
        )
        task_elapsed = time.perf_counter() - task_t0
        if channel_problems:
            print(f"[WARN] [RankTruth] cl={cl_token} channel problems={len(channel_problems)}; first={channel_problems[:3]}", flush=True)
        if task_problems:
            print(f"[WARN] [RankTruth] cl={cl_token} task problems={len(task_problems)}; first={task_problems[:3]}", flush=True)
        print(
            f"[RankTruth] done cl={cl_token}: channel_s={channel_elapsed:.2f} "
            f"task_s={task_elapsed:.2f} total_s={time.perf_counter() - cl_t0:.2f}",
            flush=True,
        )

        channel_col = f"real_channel_rank_model_ids_{cl_token}"
        task_col = f"real_task_rank_model_ids_{cl_token}"
        out[f"rank_truth_cl_{cl_token}"] = cl_token
        out[channel_col] = [
            _order_text(channel_orders.get(str(row["dataset"]), {}).get(int(row["channel"]), []))
            for _, row in out.iterrows()
        ]
        out[f"real_channel_rank_n_models_{cl_token}"] = out[channel_col].map(lambda value: len(parse_order_string(value)))
        out[f"real_channel_rank_source_{cl_token}"] = f"results_csv/TSFM/cl_{_cl_sort_key(cl_token)}/*/per_channel_results.csv"
        out[task_col] = [
            _order_text(task_orders.get(str(row["dataset"]), []))
            for _, row in out.iterrows()
        ]
        out[f"real_task_rank_n_models_{cl_token}"] = out[task_col].map(lambda value: len(parse_order_string(value)))
        out[f"real_task_rank_source_{cl_token}"] = f"results_csv/TSFM/cl_{_cl_sort_key(cl_token)}/*/all_results.csv"

    out.to_csv(path, index=False)
    complete, reason, df_new = _file_complete(
        path=path,
        required_cls=required_cls,
        meta_df=meta_df,
        expected_model_count=expected_model_count,
        quick_test=quick_test,
    )
    if not complete:
        raise RuntimeError(f"[RankTruth] rebuilt file is incomplete: {reason}")
    print(
        f"[RankTruth] saved complete stage file rows={len(df_new) if df_new is not None else len(out)} "
        f"elapsed={time.perf_counter() - total_t0:.2f}s",
        flush=True,
    )
    return path, df_new if df_new is not None else out


def load_stage_rank_truth_orders(
    *,
    args: Any,
    ds_config: str,
    cl_token: str,
    stage_df: pd.DataFrame | None = None,
    model_sizes: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    ensure: bool = True,
) -> tuple[dict[int, list[int]], list[int], str, Path]:
    cl_token = normalize_rank_truth_cl(cl_token)
    path = rank_truth_stage_path(args, rank_metric=getattr(args, "sgl_rank_metric", "MASE"))
    if stage_df is None:
        if ensure:
            if model_sizes is None:
                raise ValueError("[RankTruth] model_sizes is required when ensure=True and no stage_df is provided")
            path, stage_df = ensure_stage_rank_truth(args=args, model_sizes=model_sizes)
        else:
            stage_df = pd.read_csv(path, low_memory=False)
    if stage_df is None or stage_df.empty:
        return {}, [], cl_token, path
    channel_col = f"real_channel_rank_model_ids_{cl_token}"
    task_col = f"real_task_rank_model_ids_{cl_token}"
    if channel_col not in stage_df.columns or task_col not in stage_df.columns:
        raise FileNotFoundError(f"[RankTruth] missing columns for {cl_token} in {path}")
    sub = stage_df[stage_df["dataset"].astype(str).eq(str(ds_config))].copy()
    channel_orders: dict[int, list[int]] = {}
    for _, row in sub.iterrows():
        try:
            channel = int(pd.to_numeric(pd.Series([row.get("channel")]), errors="coerce").iloc[0])
        except Exception:
            continue
        order = parse_order_string(row.get(channel_col, ""))
        if order:
            channel_orders[channel] = order
    task_order: list[int] = []
    for _, row in sub.iterrows():
        task_order = parse_order_string(row.get(task_col, ""))
        if task_order:
            break
    return channel_orders, task_order, cl_token, path


def _main() -> None:
    import argparse

    from config.model_zoo_config import Model_zoo_details
    from utils.check_tools import filter_models_by_key

    parser = argparse.ArgumentParser(description="Build or check TSRouter stage/cl rank truth files.")
    parser.add_argument("--select_date", type=str, required=True, help="Model zoo cutoff date, e.g. 2026-06-01")
    parser.add_argument("--sgl_rank_metric", type=str, default="MASE", choices=["MASE", "sMAPE", "CRPS"])
    parser.add_argument("--rank_truth_cls", type=str, default="cl512", help="Space/comma separated cl list, e.g. 'cl96 cl512 cl2048'")
    parser.add_argument("--rank_truth_output_dir", type=str, default="")
    parser.add_argument("--zoo_total_num", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    ns = parser.parse_args()

    model_sizes, _ = filter_models_by_key(Model_zoo_details, ns.select_date, select_key="release_date")
    ns.current_zoo_num = sum(len(sizes) for sizes in model_sizes.values())
    if int(ns.zoo_total_num or 0) <= 0:
        ns.zoo_total_num = sum(len(sizes) for sizes in Model_zoo_details.values())
    ensure_stage_rank_truth(args=ns, model_sizes=model_sizes, force=bool(ns.force))


if __name__ == "__main__":
    _main()

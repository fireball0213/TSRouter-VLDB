#!/usr/bin/env python
from __future__ import annotations

import argparse
import fnmatch
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required for the server migration snapshot check.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration file must contain a mapping: {path}")
    return data


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def unique_files(values: Iterable[Path]) -> list[Path]:
    return sorted({path.resolve() for path in values if path.is_file()}, key=lambda item: item.as_posix())


def pattern_files(root: Path, pattern: str, exclude: Iterable[str] = ()) -> list[Path]:
    normalized = pattern.replace("\\", "/").rstrip("/")
    if any(char in normalized for char in "*?[]"):
        matches = list(root.glob(normalized))
    else:
        candidate = root / normalized
        matches = [candidate] if candidate.exists() else []
    files: list[Path] = []
    for match in matches:
        if match.is_file():
            files.append(match)
        elif match.is_dir():
            files.extend(child for child in match.rglob("*") if child.is_file())
    exclude_values = list(exclude)
    return unique_files(
        path
        for path in files
        if not any(fnmatch.fnmatch(rel(path, root), item) or fnmatch.fnmatch(path.name, item) for item in exclude_values)
    )


def summarize_files(files: list[Path], root: Path, max_samples: int) -> dict[str, Any]:
    total_bytes = sum(path.stat().st_size for path in files)
    return {
        "count": len(files),
        "total_bytes": total_bytes,
        "total_human": format_bytes(total_bytes),
        "samples": [rel(path, root) for path in files[:max_samples]],
    }


def nearby_baseline_candidates(root: Path, pattern: str, exclude: Iterable[str], max_samples: int) -> dict[str, Any]:
    normalized = pattern.replace("\\", "/").rstrip("/")
    if "results_artifacts/baselines/selectors/" not in normalized or "/stage20/" not in normalized:
        return {}

    pattern_path = Path(normalized)
    selector_name = ""
    parts = pattern_path.parts
    if "selectors" in parts:
        index = parts.index("selectors")
        if index + 1 < len(parts):
            selector_name = parts[index + 1]

    candidate_dirs: list[Path] = [root / pattern_path.parent]
    if selector_name:
        tail = Path("results_artifacts") / "baselines" / "selectors" / selector_name / "stage20"
        candidate_dirs.append(root / "paper" / tail)
        parent = root.parent
        if parent.exists():
            for child in parent.iterdir():
                if child.is_dir():
                    candidate_dirs.append(child / tail)

    name = pattern_path.name
    file_patterns = [name]
    for source, target in (
        ("zoo20-23", "zoo20-20"),
        ("_fast_afgbdt", "_rfast_afgbdt"),
        ("_rfast_afgbdt", "_fast_afgbdt"),
    ):
        if source in name:
            file_patterns.append(name.replace(source, target))

    kind_prefix = "weight_" if name.startswith("weight_") else ""
    if name.endswith("_model_manifest.json"):
        kind_suffix = "*_model_manifest.json"
    elif name.endswith(".pkl"):
        kind_suffix = "*.pkl"
    else:
        kind_suffix = "*"
    method_tokens = [
        token
        for token in ("afgbdt", "fast_afgbdt", "rfast_afgbdt", "TS2Vec", "v7C", "v7M", "zoo20-23", "zoo20-20")
        if token in name
    ]
    if method_tokens:
        file_patterns.append(f"{kind_prefix}*" + "*".join(method_tokens[:3]) + f"*{kind_suffix.lstrip('*')}")
    if "afgbdt" in name:
        file_patterns.append(f"{kind_prefix}*afgbdt*{kind_suffix.lstrip('*')}")
    if "TS2Vec" in name:
        file_patterns.append(f"{kind_prefix}*TS2Vec*{kind_suffix.lstrip('*')}")

    exclude_values = list(exclude)
    searched_dirs: list[str] = []
    found: list[Path] = []
    seen_dirs: set[Path] = set()
    for directory in candidate_dirs:
        directory = directory.resolve()
        if directory in seen_dirs:
            continue
        seen_dirs.add(directory)
        if not directory.is_dir():
            continue
        searched_dirs.append(rel(directory, root))
        for file_pattern in dict.fromkeys(file_patterns):
            found.extend(directory.glob(file_pattern))

    files = unique_files(
        path
        for path in found
        if not any(fnmatch.fnmatch(rel(path, root), item) or fnmatch.fnmatch(path.name, item) for item in exclude_values)
    )
    if not searched_dirs and not files:
        return {}
    return {
        "searched_dirs": searched_dirs[:max_samples],
        "matches": summarize_files(files, root, max_samples),
    }


@dataclass(frozen=True)
class SimpleSpec:
    spec_id: str
    bundle: str
    pattern: str
    target: str
    required: bool = True
    exclude: tuple[str, ...] = ("*.lock",)


@dataclass(frozen=True)
class ZooSpec:
    spec_id: str
    bundle: str
    template: str
    target: str
    required: bool = True
    exclude: tuple[str, ...] = ("*.lock",)


PROFILE_SOURCES = [
    SimpleSpec("profile_chronos", "profile_sources", "Dataset/Repr_data_sourse/c62.tsf", "data/profile_sources/chronos/chronos_profile_source.tsf"),
    SimpleSpec("profile_energy", "profile_sources", "Dataset/Repr_data_sourse/energy_num1w_len992_sd2029_std.npy", "data/profile_sources/moirai_timesfm/domain_energy.npy"),
    SimpleSpec("profile_nature", "profile_sources", "Dataset/Repr_data_sourse/nature_num1w_len992_sd2029_std.npy", "data/profile_sources/moirai_timesfm/domain_nature.npy"),
    SimpleSpec("profile_healthcare", "profile_sources", "Dataset/Repr_data_sourse/healthcare_num1w_len992_sd2029_std.npy", "data/profile_sources/moirai_timesfm/domain_healthcare.npy"),
    SimpleSpec("profile_web_cloudops", "profile_sources", "Dataset/Repr_data_sourse/web_cloudops_num1w_len992_sd2029_std.npy", "data/profile_sources/moirai_timesfm/domain_web_cloudops.npy"),
    SimpleSpec("profile_sales", "profile_sources", "Dataset/Repr_data_sourse/sales_num1w_len992_sd2029_std.npy", "data/profile_sources/moirai_timesfm/domain_sales.npy"),
]

ANCHOR_STEM = "StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n"
POOL_STEM = "c-e-n-h-w-s_x3000_in512_pl480_std_sd2029_awsfirst_pool"
TSROUTER_STAGE20_STEMS = {
    "main": f"zoo20-20_{ANCHOR_STEM}_sf2025-v4C_repr-all_sub0_1.0",
    "fast": f"zoo20-20_{ANCHOR_STEM}_sf2025-v4C_repr-all_sub0_1.0_rfast",
    "profile_probe_m": f"zoo20-20_{ANCHOR_STEM}_sf2025-v4M_repr-all_sub0_1.0",
}
BASELINE_STAGE20_COMPAT_STEMS = {
    "autoforecast": (
        "AutoForecast_Select",
        f"zoo20-23_{ANCHOR_STEM}_sf2025-v7C_repr-all_sub0_1.0_afgbdt",
        f"zoo20-20_{ANCHOR_STEM}_sf2025-v7C_repr-all_sub0_1.0_afgbdt",
    ),
    "autoxpcr": (
        "AutoXPCR_Select",
        f"zoo20-23_{ANCHOR_STEM}_sf2025-v7C_repr-all_sub0_1.0_rfast_afgbdt",
        f"zoo20-20_{ANCHOR_STEM}_sf2025-v7C_repr-all_sub0_1.0_rfast_afgbdt",
    ),
    "simplets": (
        "SimpleTS_Select",
        "zoo20-23_TS2Vec_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v6C_repr-all_sub0_1.0",
        "zoo20-20_TS2Vec_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v6C_repr-all_sub0_1.0",
    ),
}


def step3_specs(prefix: str, bundle: str, source_root: str, target_root: str, stems: dict[str, str]) -> list[SimpleSpec]:
    specs: list[SimpleSpec] = []
    for method, stem in stems.items():
        for suffix, filename in {
            "weight": f"weight_{stem}.pkl",
            "model_repr": f"{stem}.pkl",
            "subset": f"{stem}_subset_assign.pkl",
            "manifest": f"{stem}_model_manifest.json",
        }.items():
            specs.append(
                SimpleSpec(
                    f"{prefix}_{method}_{suffix}",
                    bundle,
                    f"{source_root}/{filename}",
                    f"{target_root}/{filename}",
                )
            )
    return specs


def baseline_compat_step3_specs() -> list[SimpleSpec]:
    specs: list[SimpleSpec] = []
    for method, (folder, source_stem, target_stem) in BASELINE_STAGE20_COMPAT_STEMS.items():
        source_root = f"results_artifacts/baselines/selectors/{folder}/stage20"
        target_root = f"artifacts/baseline_results/results_artifacts/baselines/selectors/{folder}/stage20"
        for suffix, source_filename, target_filename in (
            ("weight", f"weight_{source_stem}.pkl", f"weight_{target_stem}.pkl"),
            ("model_repr", f"{source_stem}.pkl", f"{target_stem}.pkl"),
            ("manifest", f"{source_stem}_model_manifest.json", f"{target_stem}_model_manifest.json"),
        ):
            specs.append(
                SimpleSpec(
                    f"step3_baseline_{method}_{suffix}",
                    "baselines_stage20",
                    f"{source_root}/{source_filename}",
                    f"{target_root}/{target_filename}",
                )
            )
    return specs

TSROUTER_CORE_SIMPLE = [
    SimpleSpec(
        "anchor_pkl",
        "tsrouter_core_stage20",
        f"results_artifacts/TSRouter/Sampled_repr_anchor/{ANCHOR_STEM}.pkl",
        "artifacts/tsrouter_core/results_artifacts/TSRouter/Sampled_repr_anchor/",
    ),
    SimpleSpec(
        "anchor_meta",
        "tsrouter_core_stage20",
        f"results_artifacts/TSRouter/Sampled_repr_anchor/{ANCHOR_STEM}_meta.pkl",
        "artifacts/tsrouter_core/results_artifacts/TSRouter/Sampled_repr_anchor/",
    ),
    SimpleSpec(
        "pool_pkl",
        "tsrouter_core_stage20",
        f"results_artifacts/caches/Sampled_repr_pool/{POOL_STEM}.pkl",
        "artifacts/tsrouter_core/results_artifacts/caches/Sampled_repr_pool/",
    ),
    SimpleSpec(
        "pool_meta",
        "tsrouter_core_stage20",
        f"results_artifacts/caches/Sampled_repr_pool/{POOL_STEM}_meta.pkl",
        "artifacts/tsrouter_core/results_artifacts/caches/Sampled_repr_pool/",
    ),
    SimpleSpec(
        "repr_forward_encoder_all",
        "tsrouter_core_stage20",
        f"results_csv/TSRouter/Repr_forward/{ANCHOR_STEM}_sf2025_all_results.csv",
        "artifacts/tsrouter_core/results_csv/TSRouter/Repr_forward/",
    ),
    SimpleSpec(
        "repr_forward_encoder_per_sample",
        "tsrouter_core_stage20",
        f"results_csv/TSRouter/Repr_forward/{ANCHOR_STEM}_sf2025_per_sample_results.csv",
        "artifacts/tsrouter_core/results_csv/TSRouter/Repr_forward/",
    ),
    SimpleSpec(
        "repr_forward_pool_all",
        "tsrouter_core_stage20",
        f"results_csv/TSRouter/Repr_forward/{POOL_STEM}_sf2025_all_results.csv",
        "artifacts/tsrouter_core/results_csv/TSRouter/Repr_forward/",
    ),
    SimpleSpec(
        "repr_forward_pool_per_sample",
        "tsrouter_core_stage20",
        f"results_csv/TSRouter/Repr_forward/{POOL_STEM}_sf2025_per_sample_results.csv",
        "artifacts/tsrouter_core/results_csv/TSRouter/Repr_forward/",
    ),
    SimpleSpec(
        "step3_insert_timing",
        "tsrouter_core_stage20",
        "results_csv/TSRouter/Model_zoo_repr/step3_insert_timing.csv",
        "artifacts/tsrouter_core/results_csv/TSRouter/Model_zoo_repr/step3_insert_timing.csv",
    ),
] + step3_specs(
    "step3_tsrouter",
    "tsrouter_core_stage20",
    "results_artifacts/TSRouter/Model_zoo_repr/stage20",
    "artifacts/tsrouter_core/results_artifacts/TSRouter/Model_zoo_repr/stage20",
    TSROUTER_STAGE20_STEMS,
)

SELECTOR_RESULTS = [
    ZooSpec(
        "tsrouter_main",
        "tsrouter_core_stage20",
        "results_csv/TSRouter/Selector_results/stage20/zoo{zoo}_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v4C_repr-all_sub0_1.0_w0.5_min_task20_v2_wseven_ss2025_top1-median_res1_all_results.csv",
        "artifacts/tsrouter_core/results_csv/TSRouter/Selector_results/stage20/",
    ),
    ZooSpec(
        "tsrouter_fast",
        "tsrouter_core_stage20",
        "results_csv/TSRouter/Selector_results/stage20/zoo{zoo}_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v4C_repr-all_sub0_1.0_rfast_w0.5_min_task20_v2_wseven_ss2025_top1-median_res1_all_results.csv",
        "artifacts/tsrouter_core/results_csv/TSRouter/Selector_results/stage20/",
    ),
    ZooSpec(
        "autoforecast",
        "tsrouter_core_stage20",
        "results_csv/TSRouter/Selector_results/stage20/zoo{zoo}_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v7C_repr-all_sub0_1.0_afgbdt_w0.5_min_task20_v2_wseven_ss2025_top1-median_res1_all_results.csv",
        "artifacts/tsrouter_core/results_csv/TSRouter/Selector_results/stage20/",
    ),
    ZooSpec(
        "autoxpcr",
        "tsrouter_core_stage20",
        "results_csv/TSRouter/Selector_results/stage20/zoo{zoo}_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v7C_repr-all_sub0_1.0_rfast_afgbdt_w0.5_min_task20_v2_wseven_ss2025_top1-median_res1_all_results.csv",
        "artifacts/tsrouter_core/results_csv/TSRouter/Selector_results/stage20/",
    ),
    ZooSpec(
        "simplets",
        "tsrouter_core_stage20",
        "results_csv/TSRouter/Selector_results/stage20/zoo{zoo}_TS2Vec_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v6C_repr-all_sub0_1.0_w0.5_min_task20_v2_wseven_ss2025_top1-median_res1_all_results.csv",
        "artifacts/tsrouter_core/results_csv/TSRouter/Selector_results/stage20/",
    ),
    ZooSpec(
        "profile_probe_m",
        "tsrouter_core_stage20",
        "results_csv/TSRouter/Selector_results/stage20/zoo{zoo}_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v4M_repr-all_sub0_1.0_w0.5_min_task20_v2_wseven_fb0_ss2025_top1-median_res1_all_results.csv",
        "artifacts/tsrouter_core/results_csv/TSRouter/Selector_results/stage20/",
    ),
    ZooSpec(
        "profile_probe_c",
        "tsrouter_core_stage20",
        "results_csv/TSRouter/Selector_results/stage20/zoo{zoo}_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v4C_repr-all_sub0_1.0_w0.5_min_task20_v2_wseven_fb0_ss2025_top1-median_res1_all_results.csv",
        "artifacts/tsrouter_core/results_csv/TSRouter/Selector_results/stage20/",
    ),
]

BASELINE_ZOO = [
    ZooSpec("baseline_static_best", "baselines_stage20", "results_csv/baselines/vldb/Static_Best/zoo{zoo}_static_best_all_results.csv", "artifacts/baseline_results/results_csv/baselines/vldb/Static_Best/", required=False),
    ZooSpec("baseline_task_oracle", "baselines_stage20", "results_csv/baselines/vldb/Task_Oracle_Best/zoo{zoo}_task_oracle_best*_all_results.csv", "artifacts/baseline_results/results_csv/baselines/vldb/Task_Oracle_Best/", required=False),
    ZooSpec("baseline_metafeature_gbdt", "baselines_stage20", "results_csv/baselines/vldb/MetaFeature_GBDT/zoo{zoo}_metafeature_gbdt_all_results.csv", "artifacts/baseline_results/results_csv/baselines/vldb/MetaFeature_GBDT/", required=False),
    ZooSpec("baseline_metafeature_mlp", "baselines_stage20", "results_csv/baselines/vldb/MetaFeature_MLP/zoo{zoo}_metafeature_mlp_all_results.csv", "artifacts/baseline_results/results_csv/baselines/vldb/MetaFeature_MLP/", required=False),
    ZooSpec("rank_truth", "baselines_stage20", "results_csv/baselines/selectors/Rank_Truth_Select/zoo{zoo}_rank_truth_*.csv", "artifacts/baseline_results/results_csv/baselines/selectors/Rank_Truth_Select/", required=False),
]

BASELINE_SIMPLE = [
    SimpleSpec("task_probe_rank_summary", "baselines_stage20", "results_csv/baselines/selectors/Task_probe_Select/rank_summary.csv", "artifacts/baseline_results/results_csv/baselines/selectors/Task_probe_Select/rank_summary.csv"),
    SimpleSpec("task_probe_forward_summary", "baselines_stage20", "results_csv/baselines/selectors/Task_probe_Select/forward_summary.csv", "artifacts/baseline_results/results_csv/baselines/selectors/Task_probe_Select/forward_summary.csv"),
    SimpleSpec("autoforecast_insert_timing", "baselines_stage20", "results_csv/baselines/selectors/AutoForecast_Select/step3_insert_timing.csv", "artifacts/baseline_results/results_csv/baselines/selectors/AutoForecast_Select/step3_insert_timing.csv"),
    SimpleSpec("autoxpcr_insert_timing", "baselines_stage20", "results_csv/baselines/selectors/AutoXPCR_Select/step3_insert_timing.csv", "artifacts/baseline_results/results_csv/baselines/selectors/AutoXPCR_Select/step3_insert_timing.csv"),
    SimpleSpec("simplets_insert_timing", "baselines_stage20", "results_csv/baselines/selectors/SimpleTS_Select/step3_insert_timing.csv", "artifacts/baseline_results/results_csv/baselines/selectors/SimpleTS_Select/step3_insert_timing.csv"),
] + baseline_compat_step3_specs()

TASK_CACHE = [
    SimpleSpec("task_cache_n20", "task_cache_stage20", "results_artifacts/caches/GE_test_sample/cl512_n20_std_wseven_ss2025.pkl", "artifacts/task_sample_cache/results_artifacts/caches/GE_test_sample/cl512_n20_std_wseven_ss2025.pkl"),
    SimpleSpec("task_cache_n20_meta", "task_cache_stage20", "results_artifacts/caches/GE_test_sample/cl512_n20_std_wseven_ss2025.pkl.meta.json", "artifacts/task_sample_cache/results_artifacts/caches/GE_test_sample/cl512_n20_std_wseven_ss2025.pkl.meta.json"),
]

TABLES_AND_FIGURES = [
    SimpleSpec("channel_meta", "tables_figures_stage20", "Dataset/channel_meta.csv", "artifacts/tables_figures/Dataset/channel_meta.csv"),
    SimpleSpec("channel_meta_with_real_rank", "tables_figures_stage20", "Dataset/channel_meta_with_real_rank.csv", "artifacts/tables_figures/Dataset/channel_meta_with_real_rank.csv"),
    SimpleSpec("dataset_properties", "tables_figures_stage20", "Dataset/dataset_properties.json", "artifacts/tables_figures/Dataset/dataset_properties.json"),
    SimpleSpec("table1", "tables_figures_stage20", "results_csv/TSRouter/vldb/tables/vldb_results_table1_latest_stage*.csv", "artifacts/tables_figures/results_csv/TSRouter/vldb/tables/"),
    SimpleSpec("table2_mase", "tables_figures_stage20", "results_csv/TSRouter/vldb/tables/vldb_results_table2_mase_by_stage*.csv", "artifacts/tables_figures/results_csv/TSRouter/vldb/tables/"),
    SimpleSpec("table2_crps", "tables_figures_stage20", "results_csv/TSRouter/vldb/tables/vldb_results_table2_crps_by_stage*.csv", "artifacts/tables_figures/results_csv/TSRouter/vldb/tables/"),
    SimpleSpec("table3", "tables_figures_stage20", "results_csv/TSRouter/vldb/tables/vldb_results_table3_insert_breakdown_by_stage.csv", "artifacts/tables_figures/results_csv/TSRouter/vldb/tables/"),
    SimpleSpec("table4", "tables_figures_stage20", "results_csv/TSRouter/vldb/tables/vldb_results_table4_route_breakdown_by_stage.csv", "artifacts/tables_figures/results_csv/TSRouter/vldb/tables/"),
    SimpleSpec("table5", "tables_figures_stage20", "results_csv/TSRouter/vldb/tables/vldb_results_table5_combined_overhead_by_stage.csv", "artifacts/tables_figures/results_csv/TSRouter/vldb/tables/"),
    SimpleSpec("table6", "tables_figures_stage20", "results_csv/TSRouter/vldb/tables/vldb_results_table6_*.csv", "artifacts/tables_figures/results_csv/TSRouter/vldb/tables/"),
    SimpleSpec("figure1", "tables_figures_stage20", "figs/vldb_results/stage20/figure1_*", "artifacts/tables_figures/figs/vldb_results/stage20/"),
    SimpleSpec("figure2", "tables_figures_stage20", "figs/vldb_results/stage20/figure2_*", "artifacts/tables_figures/figs/vldb_results/stage20/"),
    SimpleSpec("figure3", "tables_figures_stage20", "figs/vldb_results/stage20/figure3_*", "artifacts/tables_figures/figs/vldb_results/stage20/"),
    SimpleSpec("figure_growth", "tables_figures_stage20", "figs/vldb_results/stage20/figure_stage_overhead_growth_*", "artifacts/tables_figures/figs/vldb_results/stage20/"),
]

EXCLUSION_SCAN_ROOTS = [
    "results_csv/TSRouter/Selector_results/stage20",
    "results_csv/baselines",
    "results_artifacts/caches/GE_test_sample",
    "results_csv/TSRouter/vldb/tables",
    "figs/vldb_results/stage20",
]

EXCLUSION_PATTERNS = [
    "*.lock",
    "*x5000*",
    "*" + "v" + "5" + "*",
    "*" + "auto" + "_cl*",
    "*" + "acl" + "_len*",
    "*cl512_n1_std_wseven_ss2025*",
    "*cl512_n10_std_wseven_ss2025*",
    "*min_task1_v2*",
    "*min_task10_v2*",
    "*ablation*",
]

RESULT_DIR_ALIASES = {
    "timesfm_2_5": "timesfm_2.5",
    "tirex_1_1": "tirex_1.1",
}


def check_simple(spec: SimpleSpec, root: Path, max_samples: int) -> dict[str, Any]:
    files = pattern_files(root, spec.pattern, spec.exclude)
    status = "ready" if files else ("missing_required" if spec.required else "missing_optional")
    result = {
        "id": spec.spec_id,
        "bundle": spec.bundle,
        "required": spec.required,
        "status": status,
        "pattern": spec.pattern,
        "target": spec.target,
        "matches": summarize_files(files, root, max_samples),
    }
    if not files:
        nearby = nearby_baseline_candidates(root, spec.pattern, spec.exclude, max_samples)
        if nearby:
            result["nearby_matches"] = nearby
    return result


def check_zoo(
    spec: ZooSpec,
    root: Path,
    strict_span: str,
    compatible_spans: list[str],
    max_samples: int,
    allow_compatible: bool,
) -> dict[str, Any]:
    strict_pattern = spec.template.format(zoo=strict_span)
    strict_files = pattern_files(root, strict_pattern, spec.exclude)
    compatible: dict[str, Any] = {}
    for span in compatible_spans:
        if span == strict_span:
            continue
        files = pattern_files(root, spec.template.format(zoo=span), spec.exclude)
        compatible[span] = summarize_files(files, root, max_samples)
    compatible_count = sum(item["count"] for item in compatible.values())
    if strict_files:
        status = "ready"
    elif compatible_count and allow_compatible:
        status = "ready_compatible"
    elif compatible_count:
        status = "compatible_only"
    else:
        status = "missing_required" if spec.required else "missing_optional"
    return {
        "id": spec.spec_id,
        "bundle": spec.bundle,
        "required": spec.required,
        "status": status,
        "strict_zoo_span": strict_span,
        "strict_pattern": strict_pattern,
        "target": spec.target,
        "strict_matches": summarize_files(strict_files, root, max_samples),
        "compatible_matches": compatible,
    }


def expected_model_keys() -> list[str]:
    registry = load_yaml(RELEASE_ROOT / "configs" / "model_registry.yaml")
    models = [
        item
        for item in registry.get("models", [])
        if isinstance(item, dict) and int(item.get("stage_id", 9999)) <= int(registry.get("zoo_size", 20))
    ]
    return [str(item["registry_key"]) for item in sorted(models, key=lambda row: int(row.get("stage_id", 0)))]


def check_tsfm(root: Path, max_samples: int) -> dict[str, Any]:
    base = root / "results_csv" / "TSFM" / "cl_512"
    required_models = expected_model_keys()
    model_results = []
    missing = []
    for model in required_models:
        result_dir = RESULT_DIR_ALIASES.get(model, model)
        all_results = base / result_dir / "all_results.csv"
        per_window = base / result_dir / "per_window_results.csv"
        model_ok = all_results.exists() and per_window.exists()
        if not model_ok:
            missing.append(model)
        model_results.append(
            {
                "model": model,
                "result_dir": result_dir,
                "all_results": all_results.exists(),
                "per_window_results": per_window.exists(),
            }
        )
    extra_dirs = []
    if base.exists():
        required_dirs = {RESULT_DIR_ALIASES.get(model, model) for model in required_models}
        extra_dirs = sorted(path.name for path in base.iterdir() if path.is_dir() and path.name not in required_dirs)
    status = "ready" if not missing else "missing_required"
    files = pattern_files(root, "results_csv/TSFM/cl_512/*/all_results.csv") + pattern_files(root, "results_csv/TSFM/cl_512/*/per_window_results.csv")
    return {
        "id": "tsfm_cl512_20_model_subset",
        "bundle": "tsfm_results_stage20",
        "required": True,
        "status": status,
        "base": "results_csv/TSFM/cl_512",
        "required_model_count": len(required_models),
        "missing_models": missing,
        "extra_model_dirs": extra_dirs[:max_samples],
        "model_results": model_results,
        "matches": summarize_files(unique_files(files), root, max_samples),
    }


def check_reconstructed_baseline(spec_id: str, method: str, tsfm_check: dict[str, Any]) -> dict[str, Any]:
    status = "ready" if tsfm_check["status"] == "ready" else "missing_required"
    return {
        "id": spec_id,
        "bundle": "baselines_stage20",
        "required": True,
        "status": status,
        "kind": "derived_from_tsfm",
        "method": method,
        "source_check": tsfm_check["id"],
        "target": "computed by src/cli/vldb_fast_baselines.py from results_csv/TSFM/cl_512",
        "missing_models": list(tsfm_check.get("missing_models", [])),
    }


def check_task_probe_mix_route(root: Path, strict_span: str, max_samples: int) -> dict[str, Any]:
    main_pattern = SELECTOR_RESULTS[0].template.format(zoo=strict_span)
    rank_pattern = "results_csv/baselines/selectors/Task_probe_Select/rank_summary.csv"
    forward_pattern = "results_csv/baselines/selectors/Task_probe_Select/forward_summary.csv"
    deps = {
        "tsrouter_main_selector": summarize_files(pattern_files(root, main_pattern), root, max_samples),
        "rank_summary": summarize_files(pattern_files(root, rank_pattern), root, max_samples),
        "forward_summary": summarize_files(pattern_files(root, forward_pattern), root, max_samples),
    }
    missing = [key for key, value in deps.items() if value["count"] == 0]
    return {
        "id": "baseline_task_probe_from_mix_route",
        "bundle": "baselines_stage20",
        "required": True,
        "status": "ready" if not missing else "missing_required",
        "kind": "derived_from_mix_route",
        "method": "Task-probe",
        "patterns": {
            "tsrouter_main_selector": main_pattern,
            "rank_summary": rank_pattern,
            "forward_summary": forward_pattern,
        },
        "target": "computed from TSRouter main Mix-route selector outputs and Task_probe_Select summaries",
        "missing_dependencies": missing,
        "dependencies": deps,
    }


def scan_exclusions(root: Path, max_samples: int) -> list[dict[str, Any]]:
    rows = []
    for scan_root in EXCLUSION_SCAN_ROOTS:
        base = root / scan_root
        files = list(base.rglob("*")) if base.exists() else []
        for pattern in EXCLUSION_PATTERNS:
            matches = unique_files(path for path in files if fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(rel(path, root), pattern))
            if matches:
                rows.append(
                    {
                        "root": scan_root,
                        "pattern": pattern,
                        "matches": summarize_files(matches, root, max_samples),
                    }
                )
    return rows


def build_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.legacy_root).resolve()
    compatible_spans = [item.strip() for item in str(args.compatible_zoo_spans).split(",") if item.strip()]
    simple_specs = PROFILE_SOURCES + TSROUTER_CORE_SIMPLE + BASELINE_SIMPLE + TASK_CACHE + TABLES_AND_FIGURES
    zoo_specs = SELECTOR_RESULTS + BASELINE_ZOO
    tsfm_check = check_tsfm(root, args.max_samples)
    checks = [tsfm_check]
    checks.extend(
        [
            check_reconstructed_baseline("baseline_random_from_tsfm", "Random", tsfm_check),
            check_reconstructed_baseline("baseline_recent_from_tsfm", "Recent", tsfm_check),
            check_task_probe_mix_route(root, args.strict_zoo_span, args.max_samples),
        ]
    )
    checks.extend(check_simple(spec, root, args.max_samples) for spec in simple_specs)
    checks.extend(
        check_zoo(
            spec,
            root,
            args.strict_zoo_span,
            compatible_spans,
            args.max_samples,
            args.allow_compatible_zoo_span,
        )
        for spec in zoo_specs
    )
    required_bad = [
        item
        for item in checks
        if item.get("required")
        and item.get("status") not in {"ready", "ready_compatible"}
    ]
    compatible_only = [item for item in checks if item.get("status") == "compatible_only"]
    missing_required = [item for item in checks if item.get("status") == "missing_required"]
    exclusions = scan_exclusions(root, args.max_samples)

    def missing_summary(item: dict[str, Any]) -> dict[str, Any]:
        row = {
            "id": item["id"],
            "bundle": item["bundle"],
            "status": item["status"],
            "pattern": item.get("pattern") or item.get("strict_pattern") or item.get("target", ""),
        }
        if "nearby_matches" in item:
            row["nearby_matches"] = item["nearby_matches"]
        return row

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "legacy_root": str(root),
        "release_root": str(RELEASE_ROOT),
        "strict_zoo_span": args.strict_zoo_span,
        "compatible_zoo_spans": compatible_spans,
        "allow_compatible_zoo_span": bool(args.allow_compatible_zoo_span),
        "ok": not required_bad,
        "migration_safe": not required_bad,
        "summary": {
            "check_count": len(checks),
            "missing_required_count": len(missing_required),
            "compatible_only_required_count": len([item for item in compatible_only if item.get("required")]),
            "exclusion_warning_count": len(exclusions),
        },
        "missing_required": [missing_summary(item) for item in missing_required],
        "compatible_only": [
            {"id": item["id"], "bundle": item["bundle"], "strict_pattern": item.get("strict_pattern"), "compatible_matches": item.get("compatible_matches")}
            for item in compatible_only
        ],
        "checks": checks,
        "exclusion_warnings": exclusions,
    }


def md_row(values: Iterable[Any]) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# 服务器迁移快照检查",
        "",
        f"- 源根目录：`{payload['legacy_root']}`",
        f"- 严格 zoo span：`{payload['strict_zoo_span']}`",
        f"- 是否可迁移：`{str(payload['migration_safe']).lower()}`",
        "",
        "## 汇总",
        "",
        md_row(["项目", "数量"]),
        md_row(["---", "---"]),
    ]
    for key, value in payload["summary"].items():
        lines.append(md_row([key, value]))

    lines.extend(["", "## 缺失的必须项", ""])
    if payload["missing_required"]:
        lines.append(md_row(["ID", "Bundle", "状态", "模式或来源"]))
        lines.append(md_row(["---", "---", "---", "---"]))
        for item in payload["missing_required"]:
            lines.append(md_row([item["id"], item["bundle"], item["status"], item["pattern"]]))
    else:
        lines.append("无。")

    lines.extend(["", "## 只有兼容候选的必须项", ""])
    if payload["compatible_only"]:
        lines.append(md_row(["ID", "Bundle", "严格模式", "兼容候选数量"]))
        lines.append(md_row(["---", "---", "---", "---"]))
        for item in payload["compatible_only"]:
            count = sum(value["count"] for value in item["compatible_matches"].values())
            lines.append(md_row([item["id"], item["bundle"], item["strict_pattern"], count]))
    else:
        lines.append("无。")

    lines.extend(["", "## 排除项警告", ""])
    if payload["exclusion_warnings"]:
        lines.append(md_row(["扫描目录", "模式", "命中数", "样例"]))
        lines.append(md_row(["---", "---", "---", "---"]))
        for item in payload["exclusion_warnings"]:
            lines.append(md_row([item["root"], item["pattern"], item["matches"]["count"], "<br>".join(item["matches"]["samples"])]))
    else:
        lines.append("无。")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check source files before TSRouter-VLDB artifact migration.")
    parser.add_argument("--legacy-root", default=".", help="Original TSRouter-v0 workspace root on the server.")
    parser.add_argument("--strict-zoo-span", default="20-20", help="Strict paper-facing zoo span for stage20 filenames.")
    parser.add_argument("--compatible-zoo-spans", default="20-21,20-23", help="Comma-separated compatible spans to report but not accept by default.")
    parser.add_argument("--allow-compatible-zoo-span", action="store_true", help="Treat compatible-only matches as migration-safe.")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--md-out", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_snapshot(args)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.md_out:
        write_markdown(Path(args.md_out), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit("PyYAML is required to check the release contract.") from exc


SCRIPT_PATH = Path(__file__).resolve()
RELEASE_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = RELEASE_ROOT.parent
sys.path.insert(0, str(RELEASE_ROOT / "src"))

from tsrouter_vldb.execution import build_release_command_plan  # noqa: E402


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"expected mapping in {path}")
    return data


def result(name: str, ok: bool, detail: str = "") -> dict[str, Any]:
    payload = {"name": name, "ok": bool(ok)}
    if detail:
        payload["detail"] = detail
    return payload


def argv_pairs(argv: list[str]) -> dict[str, str]:
    pairs: dict[str, str] = {}
    idx = 0
    while idx < len(argv):
        item = argv[idx]
        if item.startswith("--") and idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
            pairs[item] = argv[idx + 1]
            idx += 2
        else:
            idx += 1
    return pairs


def route_plan(variant: str) -> dict[str, Any]:
    args = SimpleNamespace(
        action="run",
        stage=20,
        reuse="results",
        execute=False,
        python_bin=sys.executable,
        workspace_root=str(PROJECT_ROOT),
        variant=variant,
        methods="",
        table="",
        start_stage=None,
        end_stage=None,
        write=False,
    )
    return build_release_command_plan("route", args)["_execution_commands"][0]


def command_plan(command: str, reuse: str, *, variant: str = "main,fast") -> list[dict[str, Any]]:
    args = SimpleNamespace(
        action="run",
        stage=20,
        reuse=reuse,
        execute=False,
        python_bin=sys.executable,
        workspace_root=str(PROJECT_ROOT),
        variant=variant,
        methods="all",
        table="",
        start_stage=None,
        end_stage=None,
        write=False,
        devices="",
        quick_test=False,
    )
    return build_release_command_plan(command, args)["_execution_commands"]


def check_release_profiles(profiles: dict[str, Any]) -> list[dict[str, Any]]:
    main = profiles.get("main_profile")
    fast = profiles.get("fast_profile")
    checks = [
        result("profile:main_defined", isinstance(main, dict)),
        result("profile:fast_defined", isinstance(fast, dict)),
    ]
    if isinstance(fast, dict):
        checks.append(result("profile:fast_inherits_main", fast.get("inherits") == "TSRouter-main"))
        checks.append(result("profile:fast_uses_route_efficiency", fast.get("route_efficiency_mode") is True))
    return checks


def check_route_plans(contract: dict[str, Any]) -> list[dict[str, Any]]:
    main_command = route_plan("main")
    fast_command = route_plan("fast")
    main_args = argv_pairs(list(main_command["argv"]))
    fast_args = argv_pairs(list(fast_command["argv"]))

    policy = contract.get("variant_policy", {})
    allowed_delta = set(policy.get("allowed_main_fast_delta", []))
    default_allowed = {"route_efficiency_mode", "derived_route_id", "derived_route_profile_id"}
    allowed_delta = allowed_delta or default_allowed
    ignored = {"--vldb_route_id", "--vldb_route_profile_id"}

    common_keys = sorted((set(main_args) | set(fast_args)) - ignored - {"--route_efficiency_mode"})
    diffs = [key for key in common_keys if main_args.get(key) != fast_args.get(key)]

    return [
        result("route_plan:main_defined", bool(main_command.get("argv"))),
        result("route_plan:fast_defined", bool(fast_command.get("argv"))),
        result("route_plan:main_uses_standard_route", main_args.get("--route_efficiency_mode") == "False"),
        result("route_plan:fast_uses_fast_route", fast_args.get("--route_efficiency_mode") == "True"),
        result("route_plan:main_fast_delta", not diffs, "route efficiency only" if not diffs else ",".join(diffs)),
        result(
            "variant_policy:declared_delta",
            {"route_efficiency_mode"}.issubset(allowed_delta),
            "route_efficiency_mode",
        ),
    ]


def check_method_source_files() -> list[dict[str, Any]]:
    required = (
        "src/cli/run_model_zoo.py",
        "src/cli/check_selector.py",
        "src/model_zoo/base_model.py",
        "src/selector/TSRouter_Select/sampled_repr_set.py",
        "src/selector/TSRouter_Select/model_zoo_repr.py",
        "src/encoder/base_encoder.py",
        "src/encoder/baseline/RandomTS_encoder.py",
        "src/encoder/baseline/TS2Vec_encoder.py",
        "src/encoder/baseline/random_stats_features.py",
    )
    return [
        result(f"method_source:{path}", (RELEASE_ROOT / path).is_file())
        for path in required
    ]


def check_baseline_scope(profiles: dict[str, Any]) -> list[dict[str, Any]]:
    configured = profiles.get("baseline_methods", {}).get("deployable", [])
    expected = {
        "TSRouter-main",
        "TSRouter-fast",
        "AutoForecast",
        "AutoXPCR",
        "SimpleTS",
        "Profile-probe-M",
        "Random",
        "Recent",
        "Task-probe",
    }
    args = SimpleNamespace(
        action="run",
        stage=20,
        reuse="core",
        execute=False,
        python_bin=sys.executable,
        workspace_root=str(PROJECT_ROOT),
        variant="main,fast",
        methods="all",
        table="",
        start_stage=None,
        end_stage=None,
        write=False,
        devices="",
        quick_test=False,
    )
    plan = build_release_command_plan("baselines", args)
    planned = {
        str(command.get("metadata", {}).get("baseline_method"))
        for command in plan.get("_execution_commands", [])
        if command.get("metadata", {}).get("baseline_method")
    }
    return [
        result("baseline_scope:configured", set(configured) == expected),
        result("baseline_scope:planned", planned == expected - {"TSRouter-main", "TSRouter-fast"}),
    ]


def check_reuse_levels() -> list[dict[str, Any]]:
    profile_reused = command_plan("profile", "route")
    profile_rebuilt = command_plan("profile", "core")
    route_rebuilt = command_plan("route", "route")
    route_args = [argv_pairs(list(item["argv"])) for item in route_rebuilt]
    tsfm_reused = command_plan("tsfm", "core")
    insert_rebuilt = command_plan("insert", "core")
    baseline_reused = command_plan("baselines", "core")
    return [
        result("reuse:route_skips_profile", all(item.get("skip_saved") for item in profile_reused)),
        result("reuse:core_rebuilds_profile", all(not item.get("skip_saved") for item in profile_rebuilt)),
        result("reuse:route_rebuilds_route", all(not item.get("skip_saved") for item in route_rebuilt)),
        result(
            "reuse:route_uses_task_cache",
            all(args.get("--vldb_fast_sample") == "True" for args in route_args),
        ),
        result(
            "reuse:route_uses_cache_only_metadata",
            all(args.get("--route-cache-only") == "True" for args in route_args),
        ),
        result("reuse:core_skips_tsfm", all(item.get("skip_saved") for item in tsfm_reused)),
        result("reuse:core_rebuilds_insert", all(not item.get("skip_saved") for item in insert_rebuilt)),
        result("reuse:core_skips_baselines", all(item.get("skip_saved") for item in baseline_reused)),
    ]


def main() -> int:
    profiles = read_yaml(RELEASE_ROOT / "configs" / "paper_run_profiles.yaml")
    contract = read_yaml(RELEASE_ROOT / "configs" / "execution_contract.yaml")

    checks = []
    checks.extend(check_release_profiles(profiles))
    checks.extend(check_route_plans(contract))
    checks.extend(check_method_source_files())
    checks.extend(check_baseline_scope(profiles))
    checks.extend(check_reuse_levels())
    release_id = (RELEASE_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    pyproject = (RELEASE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    checks.append(result("release:version_file", release_id == "v1.0", release_id))
    checks.append(result("release:package_version", 'version = "1.0"' in pyproject, "1.0"))

    payload = {
        "ok": all(item["ok"] for item in checks),
        "release_root": str(RELEASE_ROOT),
        "checks": checks,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

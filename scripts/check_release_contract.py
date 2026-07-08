#!/usr/bin/env python
from __future__ import annotations

import ast
import json
import re
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

from tsrouter_vldb.legacy import build_release_command_plan  # noqa: E402


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"expected mapping in {path}")
    return data


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_ast_assignment(path: Path, name: str) -> dict[str, Any]:
    tree = ast.parse(read_text(path), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                value = ast.literal_eval(node.value)
                if not isinstance(value, dict):
                    raise TypeError(f"{name} must be a dict")
                return value
    raise KeyError(f"{name} not found in {path}")


def normalize(value: Any) -> Any:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return float(value) if isinstance(value, float) else int(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            if "." in lowered:
                return float(lowered)
            return int(lowered)
        except ValueError:
            return value.strip()
    return value


def result(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def check_main_grid(profile: dict[str, Any], grid: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for key, raw_values in grid.items():
        if key not in profile and key.startswith("repr_") and key.endswith(("_nearest_k", "_distance_power")):
            continue
        if not isinstance(raw_values, list) or len(raw_values) != 1:
            checks.append(result(f"main_grid:{key}", False, "expected a one-item list"))
            continue
        expected = raw_values[0]
        actual = profile.get(key)
        ok = normalize(actual) == normalize(expected)
        checks.append(result(f"main_grid:{key}", ok, f"release={actual!r}, source={expected!r}"))
    return checks


def contains_check(text: str, name: str, needle: str) -> dict[str, Any]:
    return result(name, needle in text, needle)


def regex_check(text: str, name: str, pattern: str) -> dict[str, Any]:
    return result(name, re.search(pattern, text, flags=re.MULTILINE) is not None, pattern)


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
        reuse="all",
        execute=False,
        python_bin=sys.executable,
        legacy_root=str(PROJECT_ROOT),
        variant=variant,
        methods="",
        table="",
        start_stage=None,
        end_stage=None,
        write=False,
    )
    return build_release_command_plan("route", args)["backend_commands"][0]


def check_route_plans(profile: dict[str, Any], contract: dict[str, Any]) -> list[dict[str, Any]]:
    main_command = route_plan("main")
    fast_command = route_plan("fast")
    main_args = argv_pairs(list(main_command["argv"]))
    fast_args = argv_pairs(list(fast_command["argv"]))
    checks = [
        result("route_plan:main_repr_size", main_args.get("--repr_size") == str(profile["repr_size"]), str(main_args.get("--repr_size"))),
        result("route_plan:fast_repr_size", fast_args.get("--repr_size") == str(profile["repr_size"]), str(fast_args.get("--repr_size"))),
        result("route_plan:main_repr_v", main_args.get("--repr_v") == str(profile["repr_v"]), str(main_args.get("--repr_v"))),
        result("route_plan:fast_repr_v", fast_args.get("--repr_v") == str(profile["repr_v"]), str(fast_args.get("--repr_v"))),
        result("route_plan:main_route_efficiency", main_args.get("--route_efficiency_mode") == "False", str(main_args.get("--route_efficiency_mode"))),
        result("route_plan:fast_route_efficiency", fast_args.get("--route_efficiency_mode") == "True", str(fast_args.get("--route_efficiency_mode"))),
    ]
    ignored = {
        "--route_efficiency_mode",
        "--vldb_route_id",
        "--vldb_route_profile_id",
    }
    common_keys = sorted((set(main_args) | set(fast_args)) - ignored)
    diffs = [key for key in common_keys if main_args.get(key) != fast_args.get(key)]
    checks.append(result("route_plan:main_fast_delta", not diffs, "diff_keys=" + ",".join(diffs)))

    paper = contract["paper_results_main"]
    checks.append(
        result(
            "contract:paper_main_repr_size",
            profile["repr_size"] == paper["repr_size"],
            f"release={profile['repr_size']}, contract={paper['repr_size']}",
        )
    )
    checks.append(
        result(
            "contract:paper_main_repr_v",
            profile["repr_v"] == paper["repr_v"],
            f"release={profile['repr_v']}, contract={paper['repr_v']}",
        )
    )
    return checks


def main() -> int:
    profiles = read_yaml(RELEASE_ROOT / "configs" / "paper_run_profiles.yaml")
    contract = read_yaml(RELEASE_ROOT / "configs" / "legacy_run_contract.yaml")
    main_profile = profiles["main_profile"]
    main_grid = extract_ast_assignment(
        PROJECT_ROOT / "src" / "cli" / "vldb_fast_baselines.py",
        "VLDB_RESULTS_MAIN_PARAM_GRID",
    )

    checks = []
    checks.extend(check_main_grid(main_profile, main_grid))
    checks.extend(check_route_plans(main_profile, contract))

    payload = {
        "ok": all(item["ok"] for item in checks),
        "release_root": str(RELEASE_ROOT),
        "project_root": str(PROJECT_ROOT),
        "checks": checks,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

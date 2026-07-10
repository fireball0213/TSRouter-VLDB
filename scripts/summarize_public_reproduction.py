#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


TABLES = [
    ("Table 1: latest-stage overview", "vldb_results_table1_latest_stage.csv"),
    ("Table 2: MASE by stage", "vldb_results_table2_mase_by_stage.csv"),
    ("Table 2: CRPS by stage", "vldb_results_table2_crps_by_stage.csv"),
    ("Table 3: insert breakdown", "vldb_results_table3_insert_breakdown_by_stage.csv"),
    ("Table 4: route breakdown", "vldb_results_table4_route_breakdown_by_stage.csv"),
    ("Table 5: combined overhead", "vldb_results_table5_combined_overhead_by_stage.csv"),
    ("Table 6.1: total overhead growth", "vldb_results_table6_1_total_overhead_growth.csv"),
    ("Table 6.2: P95 overhead growth", "vldb_results_table6_2_p95_overhead_growth.csv"),
    ("Table 6.3: P95 growth points", "vldb_results_table6_3_figure4_p95_growth_points.csv"),
]

DISPLAY_REPLACEMENTS = (
    ("Step" + "2" + "InsertRuntime", "IncomingProfile"),
    ("IncomingProfileRuntime", "IncomingProfile"),
)


def _shorten(value: object, width: int) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    for source, target in DISPLAY_REPLACEMENTS:
        text = text.replace(source, target)
    text = "".join(char if ord(char) < 128 else "*" for char in text)
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."


def _md_row(values: Iterable[object], width: int) -> str:
    return "| " + " | ".join(_shorten(value, width) for value in values) + " |"


def print_markdown_table(rows: list[dict[str, object]], *, max_rows: int, max_cols: int, max_width: int) -> None:
    if not rows:
        print("(empty)")
        return
    columns = list(rows[0].keys())[:max_cols]
    print(_md_row(columns, max_width))
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows[:max_rows]:
        print(_md_row((row.get(column, "") for column in columns), max_width))
    if len(rows) > max_rows:
        print(f"... {len(rows) - max_rows} more rows")
    extra_cols = len(rows[0]) - len(columns)
    if extra_cols > 0:
        print(f"... {extra_cols} more columns")


def load_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def workflow_rows(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = []
    for step in data.get("workflow_results", []):
        old_result_key = "back" + "end_results"
        execution_results = step.get("execution_results", step.get(old_result_key, []))
        rows.append(
            {
                "step": step.get("index"),
                "command": step.get("command"),
                "artifact_reuse": step.get("artifact_backed_reuse"),
                "ops": len(execution_results),
                "all_skipped": bool(execution_results) and all(item.get("skipped") for item in execution_results),
                "wall_elapsed_s": step.get("workflow_wall_seconds"),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a concise TSRouter-VLDB public reproduction summary.")
    parser.add_argument("--workflow-json", required=True)
    parser.add_argument("--tables-dir", default="results_csv/TSRouter/vldb/tables")
    parser.add_argument("--max-rows", type=int, default=8)
    parser.add_argument("--max-cols", type=int, default=12)
    parser.add_argument("--max-width", type=int, default=28)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workflow_path = Path(args.workflow_json)
    tables_dir = Path(args.tables_dir)

    print("\n== Workflow Summary ==")
    if workflow_path.exists():
        print_markdown_table(
            workflow_rows(workflow_path),
            max_rows=args.max_rows,
            max_cols=args.max_cols,
            max_width=args.max_width,
        )
    else:
        print(f"missing workflow log: {workflow_path}")

    print("\n== Result Tables ==")
    print("A leading * marks the highlighted value from the released table.")
    for title, filename in TABLES:
        path = tables_dir / filename
        print(f"\n### {title}")
        print(f"`{path.as_posix()}`")
        if not path_exists(path):
            print("missing")
            continue
        rows = load_csv_rows(path)
        print_markdown_table(rows, max_rows=args.max_rows, max_cols=args.max_cols, max_width=args.max_width)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

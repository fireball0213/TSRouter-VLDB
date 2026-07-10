from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the local GIFT-Eval benchmark layout.")
    parser.add_argument("--root", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configured_root = args.root or os.environ.get("TSROUTER_GIFTEVAL_ROOT", "")
    root = Path(configured_root).expanduser() if configured_root else RELEASE_ROOT / "data" / "gifteval"
    root = root.resolve()

    metadata = RELEASE_ROOT / "data" / "benchmark_metadata" / "channel_meta.csv"
    with metadata.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = sorted({str(row["dataset_name"]) for row in rows if row.get("dataset_name")})
    missing = [name for name in names if not (root / name).is_dir()]
    print(json.dumps({"ok": not missing, "root": str(root), "dataset_count": len(names), "missing": missing}, indent=2))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())

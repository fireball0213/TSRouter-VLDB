from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "src"))

from config.model_zoo_config import Model_zoo_details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official TSFM checkpoints into the public workspace layout.")
    parser.add_argument("--out", required=True, help="Target checkpoints directory.")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--revision", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def model_records(selected: list[str]) -> list[dict[str, str]]:
    available: dict[str, dict[str, str]] = {}
    for family, variants in Model_zoo_details.items():
        for variant, config in variants.items():
            key = f"{family}_{variant}"
            relative = Path(str(config["model_local_path"]))
            if not relative.parts or relative.parts[0] != "checkpoints":
                raise ValueError(f"invalid checkpoint layout for {key}: {relative}")
            available[key] = {
                "model": key,
                "source": str(config["module_name"]),
                "relative_path": Path(*relative.parts[1:]).as_posix(),
            }
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"unknown model identifiers: {', '.join(unknown)}")
    keys = selected or sorted(available)
    return [available[key] for key in keys]


def main() -> int:
    args = parse_args()
    output_root = Path(args.out).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    entries = []

    for record in model_records(args.model):
        destination = output_root / record["relative_path"]
        if destination.is_dir() and any(destination.iterdir()) and not args.force:
            entries.append({**record, "path": str(destination), "status": "reused"})
            continue

        revision = str(args.revision or api.model_info(record["source"]).sha)
        snapshot_download(
            repo_id=record["source"],
            revision=revision,
            local_dir=str(destination),
        )
        entries.append({**record, "path": str(destination), "revision": revision, "status": "downloaded"})

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
    manifest_path = output_root / "checkpoint_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "manifest": str(manifest_path), "entries": entries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPOSITORY = "Salesforce/GiftEval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the public GIFT-Eval benchmark into a reusable local directory."
    )
    parser.add_argument("--out", required=True, help="Directory that will contain GIFT-Eval dataset folders.")
    parser.add_argument("--repo-id", default=DEFAULT_REPOSITORY)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset folder to download; repeat this option. Omit to download the full benchmark.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.out).expanduser().resolve()
    patterns = [f"{name.strip().strip('/')}/**" for name in args.dataset if name.strip()]
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        allow_patterns=patterns or None,
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

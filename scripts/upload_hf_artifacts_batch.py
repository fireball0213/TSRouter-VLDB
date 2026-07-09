#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


METADATA_FILES = ("README.md", "LICENSE", "manifest.json", "checksums.sha256")


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload TSRouter-VLDB Hugging Face artifacts one file at a time.")
    parser.add_argument("--folder", default="TSRouter-VLDB_hf_upload_release")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--force-bundles", action="store_true", help="Upload bundle files even if they already exist.")
    parser.add_argument("--metadata-only", action="store_true", help="Upload README, LICENSE, manifest, and checksums only.")
    parser.add_argument("--bundles-only", action="store_true", help="Upload bundle archives only.")
    parser.add_argument("--include", default="", help="Comma-separated repository paths to upload.")
    parser.add_argument("--use-xet", action="store_true", help="Use Xet upload support instead of classic HTTP/LFS.")
    return parser.parse_args()


def configure_upload_transport(use_xet: bool) -> None:
    if use_xet:
        return
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)


def selected_files(folder: Path, *, metadata_only: bool, bundles_only: bool, include: str = "") -> list[tuple[str, Path]]:
    if metadata_only and bundles_only:
        raise ValueError("--metadata-only and --bundles-only cannot be used together")

    pairs: list[tuple[str, Path]] = []
    if not bundles_only:
        for name in METADATA_FILES:
            path = folder / name
            if not path.is_file():
                raise FileNotFoundError(path)
            pairs.append((name, path))

    if not metadata_only:
        bundle_dir = folder / "bundles"
        if not bundle_dir.is_dir():
            raise FileNotFoundError(bundle_dir)
        bundle_files = sorted(bundle_dir.glob("*.tar.zst"), key=lambda item: (item.stat().st_size, item.name))
        pairs.extend((f"bundles/{path.name}", path) for path in bundle_files)

    requested = {item.strip().replace("\\", "/") for item in include.split(",") if item.strip()}
    if requested:
        pairs = [(path_in_repo, path) for path_in_repo, path in pairs if path_in_repo in requested]
        missing = sorted(requested - {path_in_repo for path_in_repo, _ in pairs})
        if missing:
            raise FileNotFoundError(f"requested upload paths not found in local folder: {missing}")
    return pairs


def main() -> int:
    args = parse_args()
    configure_upload_transport(args.use_xet)

    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install it with: "
            "python -m pip install \"huggingface_hub==0.36.0\" \"hf-xet>=1.4.2,<2.0\""
        ) from exc

    folder = Path(args.folder)
    files = selected_files(
        folder,
        metadata_only=args.metadata_only,
        bundles_only=args.bundles_only,
        include=args.include,
    )
    api = HfApi()

    repo = api.repo_info(args.repo_id, repo_type=args.repo_type)
    print(f"repo: {repo.id}")
    print(f"folder: {folder.resolve()}")
    print(f"xet: {'enabled' if args.use_xet else 'disabled'}")

    remote_files = set(api.list_repo_files(repo_id=args.repo_id, repo_type=args.repo_type))
    total = len(files)

    for index, (path_in_repo, local_path) in enumerate(files, start=1):
        size = local_path.stat().st_size
        is_bundle = path_in_repo.startswith("bundles/")
        if is_bundle and path_in_repo in remote_files and not args.force_bundles:
            print(f"[{index}/{total}] skip existing bundle: {path_in_repo} ({format_bytes(size)})")
            continue

        started = time.time()
        print(f"[{index}/{total}] upload: {path_in_repo} ({format_bytes(size)})")
        sys.stdout.flush()
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=f"Upload {path_in_repo}",
        )
        elapsed = time.time() - started
        print(f"[{index}/{total}] done: {path_in_repo} ({elapsed:.1f}s)")
        remote_files.add(path_in_repo)

    final_files = sorted(api.list_repo_files(repo_id=args.repo_id, repo_type=args.repo_type))
    print("remote files:")
    for path in final_files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

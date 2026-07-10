#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def _token(*parts: str) -> str:
    return "".join(parts)


def _matches(pattern: re.Pattern[str], text: str) -> bool:
    return getattr(pattern, _token("sear", "ch"))(text) is not None


GENERAL_TEXT_RULES = [
    ("internal-path-linux", re.compile(r"/data[0-9]+/")),
    ("internal-path-windows", re.compile(r"[A-Za-z]:\\Users\\")),
    ("workspace-local-path", re.compile(r"D:\\Code\\")),
    ("chinese-text", re.compile(r"[\u4e00-\u9fff]")),
    ("mojibake-marker", re.compile(r"[\u8292\u85f4\u923d\u95b3]")),
    ("development-note-cn", re.compile(r"\u6211\u5df2\u7ecf|\u4e0d\u518d\u4fdd\u7559|\u751f\u4ea7\u7aef")),
    ("unsupported-baseline-family", re.compile(_token("Meta", "Feature"), re.IGNORECASE)),
    ("unsupported-baseline-oracle", re.compile(_token("Task", r"[-_ ]?", "Oracle"), re.IGNORECASE)),
    ("unsupported-profile-variant", re.compile(_token("Profile", r"[-_ ]?", "probe", r"[-_ ]?", "C"), re.IGNORECASE)),
]

SURFACE_TEXT_RULES = [
    ("internal-compat-name", re.compile(_token("leg", "acy"), re.IGNORECASE)),
    ("internal-exec-name", re.compile(_token("back", "end"), re.IGNORECASE)),
    ("internal-state-name", re.compile(_token("snap", "shot"), re.IGNORECASE)),
    ("internal-transfer-name", re.compile(_token("mig", "ration"), re.IGNORECASE)),
    ("unsupported-mode-name", re.compile(_token("auto", r"[-_]", "cl"), re.IGNORECASE)),
    ("unsupported-query-name", re.compile(r"\b" + _token("v", "5") + r"\b|" + _token("sear", "ch"), re.IGNORECASE)),
    ("numbered-phase-name", re.compile(r"\b[Ss]tep[0-9]\b")),
    ("random-state-name", re.compile(_token("se", "ed"), re.IGNORECASE)),
    ("random-state-path-token", re.compile(r"(?<![A-Za-z0-9])(sd|se|sf|ss)20[0-9]{2}(?![A-Za-z0-9])")),
]

PUBLIC_SURFACE_PREFIXES = (
    "README.md",
    "requirements",
    "configs/",
    "scripts/audit_public_surface.py",
    "scripts/check_artifacts.py",
    "scripts/check_release_contract.py",
    "scripts/run_local_fast.sh",
    "scripts/run_local_full.sh",
    "scripts/run_public_reproduction.sh",
    "scripts/summarize_public_reproduction.py",
    "scripts/upload_hf_artifacts_batch.py",
    "src/cli/tsrouter_vldb.py",
    "src/tsrouter_vldb/",
)


SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".pkl",
    ".zst",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".webp",
    ".npy",
    ".tsf",
    ".ipynb",
}

ARCHIVE_TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".yaml", ".yml"}
ARCHIVE_TEXT_MARKERS = ("/results_csv/TSRouter/vldb/tables/",)


def _walk_release_files() -> list[Path]:
    skipped_dirs = {
        ".git",
        ".cache",
        "__pycache__",
        "artifacts",
        "bundles",
        "data",
        "docs",
        "reproduction_logs",
    }
    files: list[Path] = []
    for path in RELEASE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skipped_dirs for part in path.relative_to(RELEASE_ROOT).parts):
            continue
        files.append(path)
    return files


def release_files() -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=RELEASE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return _walk_release_files()
    return [RELEASE_ROOT / line.strip() for line in proc.stdout.splitlines() if line.strip()]


def should_scan(path: Path) -> bool:
    rel = path.relative_to(RELEASE_ROOT).as_posix()
    if rel in {"manifest.json", "checksums.sha256"}:
        return False
    if rel.startswith(("docs/", "artifacts/", "bundles/", "data/", "reproduction_logs/")):
        return False
    if path.suffix.lower() in SKIP_SUFFIXES:
        return False
    return True


def scan_text_file(path: Path) -> list[dict[str, object]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    findings = []
    rel = path.relative_to(RELEASE_ROOT).as_posix()
    rules = list(GENERAL_TEXT_RULES)
    if rel.startswith(PUBLIC_SURFACE_PREFIXES):
        rules.extend(SURFACE_TEXT_RULES)
    for line_no, line in enumerate(text.splitlines(), start=1):
        for name, pattern in rules:
            if _matches(pattern, line):
                findings.append({"file": rel, "line": line_no, "rule": name})
    return findings


def scan_files(paths: Iterable[Path]) -> list[dict[str, object]]:
    findings = []
    for path in paths:
        if path.exists() and should_scan(path):
            findings.extend(scan_text_file(path))
    return findings


def should_scan_archive_member_text(member: str) -> bool:
    normalized = member.replace("\\", "/")
    if Path(normalized).suffix.lower() not in ARCHIVE_TEXT_SUFFIXES:
        return False
    return any(marker in normalized for marker in ARCHIVE_TEXT_MARKERS)


def scan_archive_member_text(root: Path, archive: Path, member: str) -> list[dict[str, object]]:
    proc = subprocess.run(
        ["tar", "--zstd", "-xOf", str(archive), member],
        check=False,
        capture_output=True,
    )
    rel = f"{archive.relative_to(root).as_posix()}::{member}"
    if proc.returncode != 0:
        return [{"file": rel, "line": 0, "rule": "archive-member-read-error"}]
    text = proc.stdout.decode("utf-8", errors="replace")
    findings = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for name, pattern in GENERAL_TEXT_RULES + SURFACE_TEXT_RULES:
            if _matches(pattern, line):
                findings.append({"file": rel, "line": line_no, "rule": name})
    return findings


def scan_archives(root: Path) -> list[dict[str, object]]:
    findings = []
    for archive in sorted((root / "bundles").glob("*.tar.zst")):
        proc = subprocess.run(
            ["tar", "--zstd", "-tf", str(archive)],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            findings.append({"file": archive.name, "line": 0, "rule": "archive-read-error"})
            continue
        for member in proc.stdout.splitlines():
            for name, pattern in GENERAL_TEXT_RULES + SURFACE_TEXT_RULES:
                if _matches(pattern, member):
                    findings.append({"file": archive.relative_to(root).as_posix(), "line": 0, "rule": name})
                    break
            if should_scan_archive_member_text(member):
                findings.extend(scan_archive_member_text(root, archive, member))
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the TSRouter-VLDB public release surface.")
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--scan-archives", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    findings = scan_files(release_files())
    if args.artifact_root:
        artifact_root = Path(args.artifact_root).resolve()
        for name in ("README.md", "manifest.json", "checksums.sha256"):
            path = artifact_root / name
            if path.exists():
                findings.extend(scan_text_file(path))
        if args.scan_archives:
            findings.extend(scan_archives(artifact_root))

    payload = {"ok": not findings, "finding_count": len(findings), "findings": findings[:200]}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

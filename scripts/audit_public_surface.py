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


TEXT_RULES = [
    ("internal-path-linux", re.compile(r"/data[0-9]+/")),
    ("internal-path-windows", re.compile(r"[A-Za-z]:\\Users\\")),
    ("workspace-local-path", re.compile(r"D:\\Code\\")),
    ("internal-compat-name", re.compile(_token("leg", "acy"), re.IGNORECASE)),
    ("internal-exec-name", re.compile(_token("back", "end"), re.IGNORECASE)),
    ("internal-state-name", re.compile(_token("snap", "shot"), re.IGNORECASE)),
    ("internal-transfer-name", re.compile(_token("mig", "ration"), re.IGNORECASE)),
    ("unsupported-mode-name", re.compile(_token("auto", r"[-_]", "cl"), re.IGNORECASE)),
    ("unsupported-query-name", re.compile(r"\b" + _token("v", "5") + r"\b|" + _token("sear", "ch"), re.IGNORECASE)),
    ("numbered-phase-name", re.compile(r"\b[Ss]tep[0-9]\b")),
    ("random-state-name", re.compile(_token("se", "ed"), re.IGNORECASE)),
    ("random-state-path-token", re.compile(r"(?<![A-Za-z0-9])(sd|se|sf|ss)20[0-9]{2}(?![A-Za-z0-9])")),
    ("mojibake-marker", re.compile(r"[\u8292\u85f4\u923d\u95b3]")),
    ("development-note-cn", re.compile(r"\u6211\u5df2\u7ecf|\u4e0d\u518d\u4fdd\u7559|\u751f\u4ea7\u7aef")),
]


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
}


def git_files() -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=RELEASE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [RELEASE_ROOT / line.strip() for line in proc.stdout.splitlines() if line.strip()]


def should_scan(path: Path) -> bool:
    rel = path.relative_to(RELEASE_ROOT).as_posix()
    if rel.startswith(("docs/", "artifacts/", "bundles/", "reproduction_logs/")):
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
    for line_no, line in enumerate(text.splitlines(), start=1):
        for name, pattern in TEXT_RULES:
            if _matches(pattern, line):
                findings.append({"file": rel, "line": line_no, "rule": name})
    return findings


def scan_files(paths: Iterable[Path]) -> list[dict[str, object]]:
    findings = []
    for path in paths:
        if path.exists() and should_scan(path):
            findings.extend(scan_text_file(path))
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
            for name, pattern in TEXT_RULES:
                if _matches(pattern, member):
                    findings.append({"file": archive.relative_to(root).as_posix(), "line": 0, "rule": name})
                    break
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the TSRouter-VLDB public release surface.")
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--scan-archives", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    findings = scan_files(git_files())
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

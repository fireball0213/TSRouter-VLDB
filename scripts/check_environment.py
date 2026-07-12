from __future__ import annotations

import argparse
import json
import platform
import sys
from typing import Any


EXPECTED_PYTHON = "3.11.15"
EXPECTED_TORCH = "2.5.1"
EXPECTED_CUDA = "12.4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report compatibility with the validated TSRouter-VLDB Linux GPU environment.")
    parser.add_argument("--require-gpu", action="store_true", help="Fail when CUDA is unavailable.")
    parser.add_argument("--strict", action="store_true", help="Fail when Python, PyTorch, or CUDA differs from the validated versions.")
    return parser.parse_args()


def check(condition: bool, expected: str, actual: str) -> dict[str, Any]:
    return {"ok": condition, "expected": expected, "actual": actual}


def main() -> int:
    args = parse_args()
    try:
        import torch
    except ModuleNotFoundError:
        payload = {"ok": False, "error": "PyTorch is not installed."}
        print(json.dumps(payload, indent=2))
        return 2

    python_version = platform.python_version()
    torch_version = str(torch.__version__)
    cuda_version = str(torch.version.cuda or "none")
    cuda_available = bool(torch.cuda.is_available())
    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if cuda_available else []

    checks = {
        "python": check(python_version == EXPECTED_PYTHON, EXPECTED_PYTHON, python_version),
        "torch": check(torch_version.startswith(EXPECTED_TORCH), EXPECTED_TORCH, torch_version),
        "cuda_runtime": check(cuda_version == EXPECTED_CUDA, EXPECTED_CUDA, cuda_version),
        "gpu": check(not args.require_gpu or cuda_available, "CUDA available" if args.require_gpu else "optional", str(cuda_available)),
    }
    strict_checks = (checks["python"]["ok"], checks["torch"]["ok"], checks["cuda_runtime"]["ok"])
    payload = {
        "ok": all(strict_checks) if args.strict else checks["gpu"]["ok"],
        "platform": platform.platform(),
        "gpu_count": len(gpu_names),
        "gpu_names": gpu_names,
        "checks": checks,
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

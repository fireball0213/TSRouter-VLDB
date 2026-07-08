#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python TSRouter-VLDB/scripts/check_release_contract.py
python TSRouter-VLDB/src/cli/tsrouter_vldb.py workflow run --mode fast --reuse all "$@"

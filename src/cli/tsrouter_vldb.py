#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path


RELEASE_SRC = Path(__file__).resolve().parents[1]
if str(RELEASE_SRC) not in sys.path:
    sys.path.insert(0, str(RELEASE_SRC))

from tsrouter_vldb.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

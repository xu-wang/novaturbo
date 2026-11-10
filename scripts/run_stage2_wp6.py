#!/usr/bin/env python3
"""Run Stage-II WP6/G3 v2 validation."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.stage2_wp6 import main


if __name__ == "__main__":
    raise SystemExit(main())

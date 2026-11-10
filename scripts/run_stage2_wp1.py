#!/usr/bin/env python3
"""Repository entry point for Stage-II WP1 baseline expansion."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.stage2_wp1 import main


if __name__ == "__main__":
    raise SystemExit(main())


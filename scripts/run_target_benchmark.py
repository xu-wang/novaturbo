#!/usr/bin/env python3
"""Repository entry point for the target-directed initialization benchmark."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.target_benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())

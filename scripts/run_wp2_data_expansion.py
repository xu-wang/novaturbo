#!/usr/bin/env python3
"""CLI entry point for the WP2 data expansion and target-suite build."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.generator_data_expansion import main


if __name__ == "__main__":
    raise SystemExit(main())

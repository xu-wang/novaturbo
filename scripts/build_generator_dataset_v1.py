#!/usr/bin/env python3
"""Command-line entry point for the NovaTurbo WP1 generator dataset."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ai.generator_dataset import main


if __name__ == "__main__":
    raise SystemExit(main())

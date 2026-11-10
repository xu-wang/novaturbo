#!/usr/bin/env python3
"""CLI for S2-WP3 forward consistency and latent repair."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.stage2_wp3 import main


if __name__ == "__main__":
    raise SystemExit(main())

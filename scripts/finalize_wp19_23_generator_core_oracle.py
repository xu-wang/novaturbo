#!/usr/bin/env python3
"""Add reference replay and non-neural controls to the completed oracle run."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.geometry19_generator_oracle_analysis import finalize


if __name__ == "__main__":
    result = finalize(ROOT, ROOT / "config/experiments/wp19_23_generator_core_oracle_v1.yaml")
    print(f"Generator oracle analysis: replay={result['reference_replay']['strict_hits']}/{result['reference_replay']['targets']}; gate={result['gate']['overall']}")


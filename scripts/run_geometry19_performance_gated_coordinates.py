#!/usr/bin/env python3
"""Generate and freeze G0b geometry-only training coordinates."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai.geometry19_performance_gated_coordinates import run, PASS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiments/geometry19_performance_gated_g0b_coordinates_v1.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(ROOT, args.config, output_root=args.output)
    print(json.dumps({k: result[k] for k in ("status", "audit", "rejection_counts", "next_stage")}, indent=2))
    sys.exit(0 if result["status"] == PASS else 2)

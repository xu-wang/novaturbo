#!/usr/bin/env python3
"""Lock G0a performance-gated CFM protocol and 99 existing-data targets."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai.geometry19_performance_gated_protocol import lock_protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiments/geometry19_v2_2_performance_gated_local_cfm_v1.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = lock_protocol(ROOT, args.config, output_root=args.output)
    print(json.dumps({k: result[k] for k in ("status", "roots", "targets", "target_class_counts", "authorization", "next_stage")}, indent=2))

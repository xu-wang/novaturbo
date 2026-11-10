#!/usr/bin/env python3
"""Lock G0c acquisition contract without running V2.2 evaluations."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai.geometry19_performance_gated_physics_protocol import lock_protocol

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config/experiments/geometry19_performance_gated_g0c_physics_v1.yaml")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    r = lock_protocol(ROOT, args.config, output_root=args.output)
    print(json.dumps({k: r[k] for k in ("status", "shards", "rows_per_shard", "context_sha256", "authorization", "next_stage")}, indent=2))

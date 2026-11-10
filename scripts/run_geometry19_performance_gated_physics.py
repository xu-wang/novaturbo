#!/usr/bin/env python3
"""Execute or verify/resume the frozen G0c physics acquisition."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai.geometry19_performance_gated_physics import run

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--authorize-acquisition", action="store_true")
    group.add_argument("--resume", action="store_true")
    args = p.parse_args()
    result = run(ROOT, ROOT / "config/experiments/geometry19_performance_gated_g0c_physics_v1.yaml",
                 authorize=args.authorize_acquisition, resume=args.resume)
    print(json.dumps({k: result[k] for k in ("status", "rows", "valid_train", "valid_validation", "actual_calls", "next_stage")}, indent=2))

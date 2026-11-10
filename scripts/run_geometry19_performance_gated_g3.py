#!/usr/bin/env python3
"""Generate and seal G3 candidates; no physics evaluation is permitted."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai.geometry19_performance_gated_g3 import execute

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true")
    group.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = execute(ROOT, smoke=args.smoke, resume=args.resume, verify_only=args.verify_only)
    print(json.dumps({k: result[k] for k in ("status", "tasks", "raw_slots", "selection_slots", "selected_candidates", "missing_slots", "physics_calls")}, indent=2))

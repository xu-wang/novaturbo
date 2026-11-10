#!/usr/bin/env python3
"""Run the B-arm fixed-total-budget Top-M tuple diagnostic."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.ai.geometry19_tuple_diagnostic import run_tuple_diagnostic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiments/wp19_23_mixed_variable_cfm_v1.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    result = run_tuple_diagnostic(ROOT, args.config, device=args.device)
    print(f"tuple diagnostic: {result['status']}; fixed_per_tuple_gate={result['fixed_per_tuple_gate']}")

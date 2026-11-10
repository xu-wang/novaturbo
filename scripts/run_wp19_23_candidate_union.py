#!/usr/bin/env python3
"""Run the equal-budget baseline+C candidate-union Oracle gate."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.ai.geometry19_candidate_union import run_equal_budget_union

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiments/wp19_23_mixed_variable_cfm_v1.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    result = run_equal_budget_union(ROOT, args.config, device=args.device)
    print(f"candidate union: {result['status']}; reranker_gate={result['gate_for_reranker']}")

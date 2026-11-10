#!/usr/bin/env python3
"""Evaluate Geometry19 mixed-variable CFM arms under direct Oracle protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.geometry19_mixed_cfm_eval import run_mixed_cfm_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiments/wp19_23_mixed_variable_cfm_v1.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--checkpoint-run-kind", default=None)
    parser.add_argument("--checkpoint-state", choices=("best", "last"), default="best")
    args = parser.parse_args()
    summary = run_mixed_cfm_evaluation(
        ROOT, args.config, device=args.device, smoke=args.smoke,
        selected_arms=args.arms, output_subdir=args.output_subdir,
        checkpoint_run_kind=args.checkpoint_run_kind,
        checkpoint_state=args.checkpoint_state,
    )
    print(f"mixed CFM eval: {summary['status']}; targets={summary['target_count']}; physics={summary['physics_calls']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

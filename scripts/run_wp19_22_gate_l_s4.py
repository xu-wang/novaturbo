#!/usr/bin/env python3
"""Generate sparse direction labels and train Gate L S4 proxies."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.geometry19_gate_l_s4 import run_s4_evaluation, run_s4_training


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/experiments/wp19_22_gate_l_gradient_alignment_v1.yaml",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--stage", choices=("train", "evaluate"), default="train")
    args = parser.parse_args()
    if args.stage == "evaluate":
        summary = run_s4_evaluation(ROOT, ROOT / args.config, device=args.device)
        print(
            f"Gate L S4 evaluation: decision={summary['final_evaluation']['decision']}; "
            "V2.1=0"
        )
        return 0
    summary = run_s4_training(ROOT, ROOT / args.config, device=args.device)
    print(
        f"Gate L S4: winner={summary['selection']['winning_arm']}; "
        f"eligible={summary['selection']['eligible_arms']}; "
        f"V2.1={summary['new_v2_1_calls']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

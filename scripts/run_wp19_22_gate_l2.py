#!/usr/bin/env python3
"""Run Gate L2 V2.1 central-gradient training and gated evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.geometry19_gate_l2 import run_gate_l2, run_gate_l2_closure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config/experiments/wp19_22_gate_l2_v21_gradient_v1.yaml"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--stage", choices=("train", "closure"), default="train")
    args = parser.parse_args()
    if args.stage == "closure":
        summary = run_gate_l2_closure(ROOT, ROOT / args.config, device=args.device)
        print(
            f"Gate L2 closure: winner={summary['comparison']['winning_arm']}; "
            f"decision={summary['comparison']['decision']}; "
            f"V2.1={summary['new_v2_1_calls']}"
        )
        return 0
    summary = run_gate_l2(ROOT, ROOT / args.config, device=args.device)
    print(
        f"Gate L2: winner={summary['selection']['winning_arm']}; "
        f"decision={summary['decision']['decision']}; "
        f"V2.1={summary['new_v2_1_calls']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

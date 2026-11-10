#!/usr/bin/env python3
"""Run Gate K2 four-loss frozen-pool and full-closure comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_gate_k2 import run_gate_k2_phase_a


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "config/experiments/wp19_22_gate_k2_full_closure_v1.yaml",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    summary = run_gate_k2_phase_a(
        REPO_ROOT, args.config, device=args.device, output_root=args.output_root
    )
    print(
        f"Gate K2: winner={summary['selection']['winning_arm']}; "
        f"decision={summary['selection']['phase_a_decision']}; "
        f"V2.1={summary['new_v2_1_calls']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

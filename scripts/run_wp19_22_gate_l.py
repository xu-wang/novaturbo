#!/usr/bin/env python3
"""Run the Gate L true-direction surrogate gradient audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.geometry19_gate_l import run_gate_l_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/experiments/wp19_22_gate_l_gradient_alignment_v1.yaml",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    summary = run_gate_l_audit(
        ROOT, ROOT / args.config, device=args.device, output_root=args.output_root
    )
    print(
        f"Gate L: pairs={summary['pair_counts']['retained_pairs']}; "
        f"decision={summary['decision']['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

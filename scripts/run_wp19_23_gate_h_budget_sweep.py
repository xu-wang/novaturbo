#!/usr/bin/env python3
"""Run the Gate H V2.1 budget curve and exhaustive pool audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_gate_h_sweep import run_gate_h_budget_sweep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/wp19_23_gate_h_budget_sweep_v1.yaml",
    )
    parser.add_argument("--gate-h-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    summary = run_gate_h_budget_sweep(
        REPO_ROOT,
        args.config,
        gate_h_root=args.gate_h_root,
        output_root=args.output_root,
    )
    print(
        f"Gate H budget sweep: {summary['status']}; "
        f"pool={summary['gate_h_sweep']['pool_coverage_status']}; "
        f"ranking={summary['gate_h_sweep']['ranking_budget_status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

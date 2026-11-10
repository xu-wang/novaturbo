#!/usr/bin/env python3
"""Run Gate H without retraining the Geometry19 generator."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_gate_h import run_gate_h_campaign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/wp19_23_gate_h_candidate_utilization_v1.yaml",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    summary = run_gate_h_campaign(
        REPO_ROOT,
        args.config,
        device=args.device,
        output_root=args.output_root,
    )
    print(
        f"Gate H: {summary['status']}; targets={summary['target_count']}; "
        f"best={summary['gate_h']['best_arm_by_hits_then_rho']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

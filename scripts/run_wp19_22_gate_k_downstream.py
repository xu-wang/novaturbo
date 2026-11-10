#!/usr/bin/env python3
"""Run the validation-selected Gate K surrogate through Geometry19 B/C/D."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_gate_k import run_gate_k_downstream


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "config/experiments/wp19_22_gate_k_hit_aligned_surrogate_v1.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    summary = run_gate_k_downstream(
        REPO_ROOT, args.config, device=args.device, output_root=args.output_root
    )
    result = summary["gate_k_winner"]
    print(
        f"Gate K downstream: {result['target_hits']}/4 hits; "
        f"median_rho={result['median_best_true_rho']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

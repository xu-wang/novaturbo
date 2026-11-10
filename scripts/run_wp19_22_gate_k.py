#!/usr/bin/env python3
"""Train the Gate K equal-data hit-aligned surrogate ablation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_gate_k import run_gate_k_training


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "config/experiments/wp19_22_gate_k_hit_aligned_surrogate_v1.yaml",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    summary = run_gate_k_training(
        REPO_ROOT, args.config, device=args.device,
        output_root=args.output_root, smoke=args.smoke,
    )
    print(
        f"Gate K {summary['run_kind']}: {summary['status']}; "
        f"winner={summary['selection']['winning_arm']}/seed_{summary['selection']['winning_seed']}"
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

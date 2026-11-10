#!/usr/bin/env python3
"""Run WP19-23E true-physics warm-start NSGA-II."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_nsga2 import run_nsga2_stage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--physics-root", type=Path, default=None)
    parser.add_argument("--refinement-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    summary = run_nsga2_stage(
        REPO_ROOT,
        args.target,
        physics_root=args.physics_root,
        refinement_root=args.refinement_root,
        output_root=args.output_root,
    )
    counts = summary["counts"]
    print(
        f"WP19-23E: {summary['status']}; evaluations="
        f"{counts['logical_physics_evaluations']}, hits={counts['diagnostic_hits']}, "
        f"delivered={counts['delivered_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Run WP19-23D Geometry19 surrogate-gradient refinement."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_refinement import run_refinement_stage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--physics-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    summary = run_refinement_stage(
        REPO_ROOT,
        args.target,
        physics_root=args.physics_root,
        output_root=args.output_root,
        device=args.device,
    )
    counts = summary["counts"]
    print(
        f"WP19-23D: {summary['status']}; seeds={counts['seed_count']}, "
        f"V2.1={counts['true_v2_1_evaluations']}, "
        f"accepted={counts['strictly_accepted_updates']}, next={summary['next_stage']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


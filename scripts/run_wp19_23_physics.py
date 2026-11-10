#!/usr/bin/env python3
"""Run WP19-23C true Geometry19 V2.1 candidate evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_physics import run_true_physics_stage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--sampling", choices=("direct", "dflow"), default="direct")
    args = parser.parse_args()
    summary = run_true_physics_stage(
        REPO_ROOT,
        args.target,
        candidate_root=args.candidate_root,
        output_root=args.output_root,
        sampling_mode=args.sampling,
    )
    counts = summary["counts"]
    print(
        f"WP19-23C: {summary['status']}; "
        f"sampling={summary['sampling_mode']}, "
        f"hits={counts['diagnostic_hits']}, "
        f"next={summary['next_stage']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

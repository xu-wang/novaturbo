#!/usr/bin/env python3
"""Run the 256-candidate WP19-23 pipeline versus cold-start NSGA-II."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_efficiency_comparison import run_efficiency_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/wp19_23_pipeline_vs_cold_nsga2_256_v1.yaml",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = run_efficiency_comparison(
        REPO_ROOT,
        args.config,
        device=args.device,
        output_root=args.output_root,
        resume=args.resume,
    )
    overall = summary["overall"]
    print(
        f"WP19-23 efficiency comparison: {summary['status']}; "
        f"saved={overall['saved_v2_1_calls']:+d} V2.1 calls "
        f"({overall['saved_fraction']:+.2%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

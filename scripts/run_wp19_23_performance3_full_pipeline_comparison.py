#!/usr/bin/env python3
"""Run the evaluation-only S0 versus performance3_log full-flow comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.geometry19_performance3_full_pipeline import run_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/experiments/wp19_23_performance3_full_pipeline_comparison_v1.yaml",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_comparison(
        ROOT,
        ROOT / args.config,
        device=args.device,
        output_root=args.output_root,
        resume=args.resume,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()

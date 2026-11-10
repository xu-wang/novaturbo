#!/usr/bin/env python3
"""Run the target-isolated Geometry19 independent reranker experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.geometry19_reranker import run_reranker_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/experiments/wp19_23_independent_reranker_v1.yaml",
    )
    parser.add_argument("--output-root")
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    summary = run_reranker_experiment(
        ROOT,
        ROOT / args.config,
        output_root=args.output_root,
        device=args.device,
        smoke=args.smoke,
    )
    print(summary["decision"])


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Run one stage of the frozen E1 performance-surrogate ranking experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.geometry19_e1_performance_ranking import (
    evaluate_e1_truth,
    freeze_e1_campaign,
    score_e1_campaign,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "truth", "score"))
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/experiments/wp19_23_e1_performance3_pipeline_ranking_v1.yaml",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.stage == "freeze":
        result = freeze_e1_campaign(
            ROOT,
            args.config,
            output_root=args.output_root,
            device=args.device,
            smoke=args.smoke,
        )
    elif args.stage == "truth":
        result = evaluate_e1_truth(ROOT, args.config, output_root=args.output_root)
    else:
        result = score_e1_campaign(
            ROOT, args.config, output_root=args.output_root, device=args.device
        )
    print(result.get("decision", result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


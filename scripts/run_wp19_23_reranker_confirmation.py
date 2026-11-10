#!/usr/bin/env python3
"""Freeze or evaluate the independent-reranker confirmation campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.geometry19_reranker_confirmation import (
    evaluate_confirmation_campaign,
    freeze_confirmation_campaign,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "evaluate"))
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/experiments/wp19_23_reranker_confirmation_v1.yaml",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.stage == "freeze":
        result = freeze_confirmation_campaign(
            ROOT, args.config, output_root=args.output_root
        )
        print(
            f"Frozen: targets={result['target_count']}; runs={result['run_count']}; "
            f"rows={result['ledger_row_count']}"
        )
    else:
        result = evaluate_confirmation_campaign(
            ROOT, args.config, output_root=args.output_root, device=args.device
        )
        print(result["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


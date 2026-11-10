#!/usr/bin/env python3
"""Train Geometry19 Gumbel and dequantization mixed-variable CFM arms."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.geometry19_mixed_cfm import train_mixed_cfm_arms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiments/wp19_23_mixed_variable_cfm_v1.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--fixed-400", action="store_true")
    parser.add_argument("--snapshot-every", type=int, default=0)
    args = parser.parse_args()
    summaries = train_mixed_cfm_arms(
        ROOT, args.config, smoke=args.smoke, selected_arms=args.arms,
        fixed_400=args.fixed_400,
        snapshot_every=args.snapshot_every,
    )
    for arm, summary in summaries.items():
        print(f"{arm}: {summary['status']}; epochs={summary['training']['epochs_completed']}; integer_legal={summary['validation']['integer_legal']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

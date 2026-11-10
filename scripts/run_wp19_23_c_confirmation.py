#!/usr/bin/env python3
"""Confirm the development-selected C checkpoint on unseen frozen targets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.geometry19_checkpoint_selection import run_confirmation_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiments/wp19_23_mixed_variable_cfm_v1.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    summary = run_confirmation_evaluation(ROOT, args.config, device=args.device)
    print(f"C confirmation: {summary['status']}; gate={summary['gate']}; epoch={summary['selected_epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

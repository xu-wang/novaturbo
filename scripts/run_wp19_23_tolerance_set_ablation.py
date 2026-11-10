#!/usr/bin/env python3
"""Run 19D tolerance-set CFM checkpoint selection and blind confirmation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_tolerance_set_eval import run_tolerance_set_ablation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/wp19_23_tolerance_set_ablation_v1.yaml"),
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    summary = run_tolerance_set_ablation(REPO_ROOT, args.config, device=args.device)
    print(
        f"19D tolerance-set CFM: {summary['status']}; "
        f"targets={summary['confirmation_target_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

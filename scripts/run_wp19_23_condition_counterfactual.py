#!/usr/bin/env python3
"""Run the frozen 19D CFM condition counterfactual diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_condition_counterfactual import run_condition_counterfactual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/wp19_23_condition_counterfactual_v1.yaml"),
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    summary = run_condition_counterfactual(REPO_ROOT, args.config, device=args.device)
    decision = summary["decision"]
    print(
        "19D condition counterfactual: "
        f"{'SUFFICIENT' if decision['condition_utilization_sufficient'] else 'INSUFFICIENT'}; "
        f"next={decision['selected_branch']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

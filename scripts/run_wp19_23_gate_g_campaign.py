#!/usr/bin/env python3
"""Run the frozen WP19-23 multi-target direct/D-Flow campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_campaign import run_equal_budget_campaign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/wp19_23_direct_dflow_equal_budget_campaign_v1.yaml",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    summary = run_equal_budget_campaign(
        REPO_ROOT,
        args.config,
        device=args.device,
        output_root=args.output_root,
    )
    print(
        f"WP19-23 Gate G campaign: {summary['status']}; "
        f"targets={summary['target_count']}; "
        f"formal={summary['gate_g']['formal_gate_status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

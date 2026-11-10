#!/usr/bin/env python3
"""Run NovaTurbo's default high-precision inverse-design delivery chain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.inverse_design_delivery import DEFAULT_CONFIG, run_inverse_design_delivery


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-config",
        type=Path,
        required=True,
        help="YAML file containing one TargetSpec (thrust, TSFC, mass, and classification)",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--disable-surrogate-gradient",
        action="store_true",
        help="Disable the optional forward-surrogate gradient repair stage.",
    )
    parser.add_argument(
        "--disable-nsga2",
        action="store_true",
        help="Disable the optional conditional NSGA-II fallback.",
    )
    args = parser.parse_args()
    output = run_inverse_design_delivery(
        args.config,
        args.target_config,
        repo_root=REPO_ROOT,
        output_dir_override=args.output_dir,
        gradient_enabled_override=(
            False if args.disable_surrogate_gradient else None
        ),
        nsga2_enabled_override=False if args.disable_nsga2 else None,
    )
    print(json.dumps({"output": str(output), "status": "completed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

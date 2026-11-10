#!/usr/bin/env python3
"""Run P2 train-fold-only local Jacobian bootstrap stability estimation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_v2_2_local_chart_features import (  # noqa: E402
    run_local_jacobian_stability,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config/experiments/geometry19_v2_2_local_chart_moe_v1.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run_local_jacobian_stability(REPO_ROOT, args.config, output_root=args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "root_fold_estimates": result["root_fold_estimates"],
                "reliable_root_fold_estimates": result["reliable_root_fold_estimates"],
                "minimum_fold_reliable_fraction": result[
                    "minimum_fold_reliable_fraction"
                ],
                "p3_authorized": result["p3_authorized"],
                "confirmation_content_read": result["confirmation_content_read"],
                "next_authorized_stage": result["next_authorized_stage"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

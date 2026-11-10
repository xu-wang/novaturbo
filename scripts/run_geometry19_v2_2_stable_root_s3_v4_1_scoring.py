#!/usr/bin/env python3
"""Execute the locked S3 v4.1 one-shot five-arm model scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_v2_2_stable_root_s3_v4_1_scoring import (  # noqa: E402
    run_s3_v4_1_model_scoring,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/geometry19_v2_2_stable_root_s3_v4_1_model_scoring_v1.yaml",
    )
    args = parser.parse_args()
    result = run_s3_v4_1_model_scoring(REPO_ROOT, args.config)
    print(
        json.dumps(
            {
                "status": result["status"],
                "scoring_rows": result["scoring_rows"],
                "advancing_arms": result["advancing_arms"],
                "nearest_route_accuracy": result["nearest_route_accuracy"],
                "nearest_vs_oracle_relative_worst_root_gap": result[
                    "nearest_vs_oracle_relative_worst_root_gap"
                ],
                "next_authorized_stage": result["next_authorized_stage"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

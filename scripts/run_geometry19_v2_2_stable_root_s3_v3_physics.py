#!/usr/bin/env python3
"""Seal unified V2.2 labels for the S3 v3 33-root coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_v2_2_stable_root_s3_v3_physics import (  # noqa: E402
    run_s3_v3_physics_acquisition,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/geometry19_v2_2_stable_root_s3_v3_33root_evaluation_v1.yaml",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_s3_v3_physics_acquisition(
        REPO_ROOT, args.config, resume=args.resume
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "counts": result["counts"],
                "evaluation_eligible_fraction_overall": result[
                    "evaluation_eligible_fraction_overall"
                ],
                "minimum_evaluation_eligible_per_root_observed": result[
                    "minimum_evaluation_eligible_per_root_observed"
                ],
                "model_scoring_authorized": result["model_scoring_authorized"],
                "next_authorized_stage": result["next_authorized_stage"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

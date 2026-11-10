#!/usr/bin/env python3
"""Generate boundary-aware stable-root S3 v2 prelabel coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_v2_2_stable_root_s3_prelabel_v2 import (  # noqa: E402
    generate_s3_v2_prelabel_coordinates,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/geometry19_v2_2_stable_root_s3_prelabel_v2.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = generate_s3_v2_prelabel_coordinates(
        REPO_ROOT, args.config, output_root=args.output
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "screening_pairs_considered": result[
                    "screening_pairs_considered"
                ],
                "screening_pairs_rejected": result[
                    "screening_pairs_rejected"
                ],
                "reserve_pairs_selected": result["reserve_pairs_selected"],
                "selected_pairs": result["selected_pairs"],
                "final_prelabel_rows": result["final_prelabel_rows"],
                "physics_labeling_authorized": result[
                    "physics_labeling_authorized"
                ],
                "physics_evaluations_executed": result[
                    "physics_evaluations_executed"
                ],
                "model_predictions_executed": result[
                    "model_predictions_executed"
                ],
                "confirmation_content_read": result["confirmation_content_read"],
                "next_authorized_stage": result["next_authorized_stage"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

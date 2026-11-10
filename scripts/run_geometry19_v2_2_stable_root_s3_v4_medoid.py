#!/usr/bin/env python3
"""Generate S3 v4 medoid/covariance development coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_v2_2_stable_root_s3_v4_medoid import (  # noqa: E402
    generate_s3_v4_medoid_coordinates,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/geometry19_v2_2_stable_root_s3_v4_medoid_covariance_prelabel_v1.yaml",
    )
    args = parser.parse_args()
    result = generate_s3_v4_medoid_coordinates(REPO_ROOT, args.config)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_root_count": result["selected_root_count"],
                "eligible_medoid_anchors": result["eligible_medoid_anchors"],
                "screening_pairs_considered": result["screening_pairs_considered"],
                "screening_pairs_rejected": result["screening_pairs_rejected"],
                "reserve_pairs_selected": result["reserve_pairs_selected"],
                "selected_pairs": result["selected_pairs"],
                "final_prelabel_rows": result["final_prelabel_rows"],
                "future_physics_protocol_authorized": result[
                    "future_physics_protocol_authorized"
                ],
                "next_authorized_stage": result["next_authorized_stage"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

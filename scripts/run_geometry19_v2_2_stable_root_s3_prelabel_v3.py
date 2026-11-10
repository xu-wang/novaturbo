#!/usr/bin/env python3
"""Generate independent S3 v3 coordinates for the frozen 33-root scope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_v2_2_stable_root_s3_prelabel_v3 import (  # noqa: E402
    generate_s3_v3_prelabel_coordinates,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/geometry19_v2_2_stable_root_s3_prelabel_v3_33root.yaml",
    )
    args = parser.parse_args()
    result = generate_s3_v3_prelabel_coordinates(REPO_ROOT, args.config)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_root_count": result["selected_root_count"],
                "excluded_root_count": result["excluded_root_count"],
                "screening_pairs_considered": result["screening_pairs_considered"],
                "screening_pairs_rejected": result["screening_pairs_rejected"],
                "reserve_pairs_selected": result["reserve_pairs_selected"],
                "selected_pairs": result["selected_pairs"],
                "final_prelabel_rows": result["final_prelabel_rows"],
                "physics_labeling_authorized": result["physics_labeling_authorized"],
                "next_authorized_stage": result["next_authorized_stage"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

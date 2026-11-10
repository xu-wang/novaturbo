#!/usr/bin/env python3
"""Complete v4 medoid coordinates for discrete-boundary roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_v2_2_stable_root_s3_v4_1_completion import (  # noqa: E402
    complete_s3_v4_boundary_coordinates,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/geometry19_v2_2_stable_root_s3_v4_1_boundary_completion_v1.yaml",
    )
    args = parser.parse_args()
    result = complete_s3_v4_boundary_coordinates(REPO_ROOT, args.config)
    print(
        json.dumps(
            {
                "status": result["status"],
                "carried_forward_roots": result["carried_forward_roots"],
                "completed_boundary_roots": result["completed_boundary_roots"],
                "completion_rows": result["completion_rows"],
                "combined_rows": result["combined_rows"],
                "reserve_pairs_selected": result["reserve_pairs_selected"],
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

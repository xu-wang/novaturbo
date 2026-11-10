#!/usr/bin/env python3
"""Lock the S3 v4.1 V2.2 development physics-acquisition protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_v2_2_stable_root_s3_v4_1_physics_protocol import (  # noqa: E402
    lock_s3_v4_1_physics_protocol,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/geometry19_v2_2_stable_root_s3_v4_1_physics_acquisition_v1.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = lock_s3_v4_1_physics_protocol(REPO_ROOT, args.config, output_root=args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "coordinate_rows": result["coordinate_rows"],
                "root_count": result["root_count"],
                "physics_acquisition_authorized": result["physics_acquisition_authorized"],
                "model_scoring_authorized_now": result["model_scoring_authorized_now"],
                "next_authorized_stage": result["next_authorized_stage"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

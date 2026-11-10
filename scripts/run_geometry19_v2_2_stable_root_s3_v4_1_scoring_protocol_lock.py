#!/usr/bin/env python3
"""Lock one-shot five-arm scoring for the sealed S3 v4.1 rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_v2_2_stable_root_s3_v4_1_scoring_protocol import (  # noqa: E402
    lock_s3_v4_1_model_scoring_protocol,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/geometry19_v2_2_stable_root_s3_v4_1_model_scoring_v1.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = lock_s3_v4_1_model_scoring_protocol(REPO_ROOT, args.config, output_root=args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "eligible_rows": result["eligible_rows"],
                "eligible_roots": result["eligible_roots"],
                "model_arms": len(result["model_arms"]),
                "model_arrays_loaded": result["model_arrays_loaded"],
                "model_predictions_executed": result["model_predictions_executed"],
                "model_scoring_authorized": result["model_scoring_authorized"],
                "next_authorized_stage": result["next_authorized_stage"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

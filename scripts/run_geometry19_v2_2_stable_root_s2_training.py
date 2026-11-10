#!/usr/bin/env python3
"""Train and lock the five stable-root S2 Ridge arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_v2_2_stable_root_s2_training import run_s2_training  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/geometry19_v2_2_stable_root_s2_training_v1.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run_s2_training(REPO_ROOT, args.config, output_root=args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "models": len(result["models"]),
                "selected_alpha_by_arm": {
                    model["arm_id"]: model["selected_alpha"]
                    for model in result["models"]
                },
                "exact_reload_pass": result["exact_reload_pass"],
                "s3_coordinate_generation_authorized": result[
                    "s3_coordinate_generation_authorized"
                ],
                "next_authorized_stage": result["next_authorized_stage"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

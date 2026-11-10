#!/usr/bin/env python3
"""Lock the stable-root S2 forward and S3 fresh-evaluation protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_v2_2_stable_root_forward_protocol import (  # noqa: E402
    lock_stable_root_forward_protocol,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/geometry19_v2_2_stable_root_forward_fresh_eval_v1.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = lock_stable_root_forward_protocol(
        REPO_ROOT, args.config, output_root=args.output
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "stable_roots": result["stable_roots"],
                "model_arms": len(result["model_arm_ids"]),
                "s2_training_authorized": result["s2_training_authorized"],
                "s3_generation_authorized_now": result["s3_generation_authorized_now"],
                "confirmation_content_read": result["confirmation_content_read"],
                "next_authorized_stage": result["next_authorized_stage"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

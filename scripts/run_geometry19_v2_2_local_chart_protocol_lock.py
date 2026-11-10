#!/usr/bin/env python3
"""Run P0 preregistration and immutable input locking for local-chart MoE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_v2_2_local_chart_protocol import (  # noqa: E402
    run_protocol_input_lock,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/geometry19_v2_2_local_chart_moe_v1.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run_protocol_input_lock(
        REPO_ROOT,
        args.config,
        output_root=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "all_input_hashes_match": result["all_input_hashes_match"],
                "rows": result["fold_audit"]["rows"],
                "global_roots": result["fold_audit"]["global_roots"],
                "confirmation_content_read": result["confirmation_content_read"],
                "next_authorized_stage": result["next_authorized_stage"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

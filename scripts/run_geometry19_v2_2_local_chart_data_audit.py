#!/usr/bin/env python3
"""Run P1 data, root-lineage, and fold-integrity audit for local-chart MoE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_v2_2_local_chart_data_audit import (  # noqa: E402
    run_data_root_fold_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config/experiments/geometry19_v2_2_local_chart_moe_v1.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run_data_root_fold_audit(REPO_ROOT, args.config, output_root=args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": result["row_contract"]["rows"],
                "global_roots": result["root_contract"]["global_roots"],
                "existing_roots": result["root_contract"]["existing_roots"],
                "new_roots": result["root_contract"]["new_roots"],
                "rank_sufficient_roots": result["root_contract"]["sufficient_local_variation_root_count"],
                "rank_insufficient_roots": result["root_contract"]["insufficient_local_variation_root_count"],
                "confirmation_content_read": result["confirmation_content_read"],
                "next_authorized_stage": result["next_authorized_stage"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

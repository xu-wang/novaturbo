#!/usr/bin/env python3
"""Run the post-hoc audit of failed V2.2 corrective confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_v2_2_corrective_audit import run_corrective_failure_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/geometry19_v2_2_corrective_failure_audit_v1.yaml",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    result = run_corrective_failure_audit(
        REPO_ROOT, args.config, device=args.device, output_root=args.output_root
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "rmse_gap": result["generalization_gap"][
                    "validation_to_train_rmse_ratio"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

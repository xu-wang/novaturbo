#!/usr/bin/env python3
"""Train the leakage-safe V2.2 19D-to-three-performance baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_v2_2_forward_baseline import run_v2_2_forward_baseline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/geometry19_v2_2_forward_baseline_v1.yaml",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    summary = run_v2_2_forward_baseline(
        REPO_ROOT,
        args.config,
        smoke=args.smoke,
        device=args.device,
        output_root=args.output_root,
    )
    selected = next(
        run for run in summary["runs"] if run["seed"] == summary["selected_seed"]
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "selected_seed": summary["selected_seed"],
                "test_normalized_rmse": selected["metrics"]["test"]["overall"][
                    "normalized_rmse"
                ],
                "test_rho_p95": selected["metrics"]["test"]["overall"]["rho_p95"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "BASELINE_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

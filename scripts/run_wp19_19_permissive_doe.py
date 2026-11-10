#!/usr/bin/env python3
"""Run the resumable WP19-19 permissive 4096-point DOE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physics.wp19_19_doe import run_wp19_19  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "config/physics/wp19_19_permissive_doe_v1.yaml",
    )
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--preview-budget", type=int, default=None)
    parser.add_argument("--export-budget", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--physics-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_wp19_19(
        args.protocol,
        REPO_ROOT,
        sample_count_override=args.sample_count,
        preview_budget_override=args.preview_budget,
        export_budget_override=args.export_budget,
        workers_override=args.workers,
        physics_only=args.physics_only,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "sample_count": report["summary"]["sample_count"],
                "converged_count": report["summary"]["converged_count"],
                "classification_distribution": report["summary"][
                    "classification_distribution"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] in {"COMPLETE", "PHYSICS_COMPLETE_MESH_PENDING"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

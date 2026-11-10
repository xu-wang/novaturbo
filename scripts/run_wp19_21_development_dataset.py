#!/usr/bin/env python3
"""Generate or resume the WP19-21 Geometry19 development dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physics.wp19_21_dataset import run_wp19_21  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path,
        default=REPO_ROOT / "config/physics/wp19_21_development_dataset_v1.yaml",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--preview-budget", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use 16 samples, 2 previews, two workers, and 4-record shards unless overridden.",
    )
    args = parser.parse_args()
    output = args.output_root
    if args.smoke and output is None:
        output = REPO_ROOT / "data/generated/wp19_21_development_dataset_smoke_v1"
    report = run_wp19_21(
        args.protocol, REPO_ROOT,
        output_root_override=output,
        sample_count_override=args.sample_count or (16 if args.smoke else None),
        preview_budget_override=(
            args.preview_budget if args.preview_budget is not None else (2 if args.smoke else None)
        ),
        workers_override=args.workers or (2 if args.smoke else None),
        shard_size_override=args.shard_size or (4 if args.smoke else None),
        resume=args.resume,
    )
    print(json.dumps({"status": report["status"], **report["summary"]}, sort_keys=True))
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

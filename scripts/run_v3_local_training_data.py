"""Freeze and label the V3 22D center-disjoint local training-data population."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.v3_local_training_data import (  # noqa: E402
    evaluate_v3_local_training_data,
    prepare_v3_local_training_data,
    verify_v3_local_training_coordinates,
    verify_v3_local_training_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate", "verify-coordinates", "verify"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rebuild", action="store_true", help="recreate all coordinates when verifying")
    args = parser.parse_args()
    output = args.output_root or ROOT / "data/validation/v3_22d_local_training_data_v1"
    if args.action == "prepare":
        manifest = prepare_v3_local_training_data(ROOT, output)
    elif args.action == "evaluate":
        manifest = evaluate_v3_local_training_data(ROOT, output, workers=args.workers)
    elif args.action == "verify-coordinates":
        manifest = verify_v3_local_training_coordinates(ROOT, output, rebuild=args.rebuild)
    else:
        manifest = verify_v3_local_training_labels(ROOT, output)
    if args.action in ("prepare", "verify-coordinates"):
        print(f"V3 local coordinates: {manifest['coordinate_count']}; {manifest['split_counts']}")
    else:
        print(
            f"V3 local labels: {manifest['row_count']}; "
            f"converged={manifest['solver_converged_count']}; "
            f"software_candidates={manifest['software_regression_candidate_count']}; "
            f"coverage_gate={manifest['data_coverage_gate_pass']}"
        )


if __name__ == "__main__":
    main()

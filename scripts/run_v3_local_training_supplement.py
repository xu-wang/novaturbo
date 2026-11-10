"""Freeze and evaluate the separate V3 development-only coverage supplement."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.v3_local_training_supplement import (  # noqa: E402
    evaluate_v3_training_supplement,
    prepare_v3_training_supplement,
    verify_v3_training_supplement_coordinates,
    verify_v3_training_supplement_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate", "verify-coordinates", "verify"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output_root or ROOT / "data/validation/v3_22d_local_training_data_dev_supplement_v1"
    if args.action == "prepare":
        manifest = prepare_v3_training_supplement(ROOT, output)
    elif args.action == "evaluate":
        manifest = evaluate_v3_training_supplement(ROOT, output, workers=args.workers)
    elif args.action == "verify-coordinates":
        manifest = verify_v3_training_supplement_coordinates(ROOT, output)
    else:
        manifest = verify_v3_training_supplement_labels(ROOT, output)
    if args.action in ("prepare", "verify-coordinates"):
        print(f"V3 development supplement coordinates: {manifest['coordinate_count']}")
    else:
        print(
            f"V3 supplement labels: {manifest['row_count']}; "
            f"candidates={manifest['software_regression_candidate_count']}; "
            f"combined_coverage_gate={manifest['combined_data_coverage_gate_pass']}"
        )


if __name__ == "__main__":
    main()

"""Freeze L3 center-disjoint local coordinates, then evaluate V3 labels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.v3_local_acquisition import (  # noqa: E402
    evaluate_v3_local_acquisition,
    prepare_v3_local_acquisition,
    verify_v3_local_acquisition_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate", "verify"))
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    if args.action == "prepare":
        manifest = prepare_v3_local_acquisition(ROOT, args.output_root)
        print(f"L3 coordinates frozen: {manifest['coordinate_count']}; {manifest['split_counts']}")
    elif args.action == "evaluate":
        manifest = evaluate_v3_local_acquisition(ROOT, args.output_root)
        print(
            f"L3 V3 labels: {manifest['row_count']}; "
            f"converged={manifest['solver_converged_count']}, "
            f"software_candidates={manifest['software_regression_candidate_count']}, "
            f"engine_feasible={manifest['engine_feasible_count']}"
        )
    else:
        manifest = verify_v3_local_acquisition_labels(ROOT, args.output_root)
        print(f"L3 verified: {manifest['row_count']} labels")


if __name__ == "__main__":
    main()

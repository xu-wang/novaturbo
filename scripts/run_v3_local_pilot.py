"""Prepare prelabel coordinates, then evaluate the V3 local pilot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.v3_local_pilot import (  # noqa: E402
    evaluate_v3_local_pilot,
    prepare_v3_local_pilot,
    verify_v3_local_pilot_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate", "verify"))
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    if args.action == "prepare":
        manifest = prepare_v3_local_pilot(ROOT, args.output_root)
        print(f"V3 local pilot coordinates frozen: {manifest['coordinate_count']}; {manifest['split_counts']}")
    elif args.action == "evaluate":
        manifest = evaluate_v3_local_pilot(ROOT, args.output_root)
        print(
            f"V3 local pilot labeled: {manifest['row_count']}; "
            f"converged={manifest['solver_converged_count']}, "
            f"development_domain={manifest['development_domain_retained_count']}, "
            f"engine_feasible={manifest['engine_feasible_count']}"
        )
    else:
        manifest = verify_v3_local_pilot_labels(ROOT, args.output_root)
        print(f"V3 local pilot verified: {manifest['row_count']} labels")


if __name__ == "__main__":
    main()

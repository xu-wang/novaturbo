"""Build or verify center-disjoint V3 software regression candidate files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.v3_local_candidate_dataset import (  # noqa: E402
    build_v3_local_candidate_dataset,
    verify_v3_local_candidate_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_root or ROOT / "data/validation/v3_22d_local_candidate_dataset_v1"
    if args.action == "build":
        manifest = build_v3_local_candidate_dataset(ROOT, output)
    else:
        manifest = verify_v3_local_candidate_dataset(ROOT, output)
    print(f"V3 software candidates: {manifest['candidate_count']}; {manifest['split_counts']}")


if __name__ == "__main__":
    main()

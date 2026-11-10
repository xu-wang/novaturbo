"""Evaluate frozen V2.2 medoid coordinates under the V3 development backend."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.v3_local_center_audit import build_v3_local_center_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    manifest = build_v3_local_center_audit(ROOT, args.output_root)
    print(
        f"V3 local center audit: {manifest['row_count']} coordinates, "
        f"{manifest['solver_converged_count']} converged, "
        f"{manifest['candidate_count']} diagnostic pilot candidates, "
        f"{manifest['engine_feasible_count']} engine-feasible; "
        f"physics={manifest['identity']['runtime_contract']['physics_bundle_hash']}"
    )


if __name__ == "__main__":
    main()

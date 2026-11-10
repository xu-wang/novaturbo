#!/usr/bin/env python3
"""Run the frozen V2.2 known-manifold five-fold surrogate ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_v2_2_known_manifold_ablation import (  # noqa: E402
    run_known_manifold_ablation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/geometry19_v2_2_known_manifold_ablation_v1.yaml",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run_known_manifold_ablation(
        REPO_ROOT,
        args.config,
        device=args.device,
        output_root=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_variant": result["selected_variant"],
                "selected_vs_legacy_reduction_fraction": result[
                    "selected_vs_legacy_reduction_fraction"
                ],
                "support_gate_pass": result["support_audit"]["gate_pass"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run V2.2 low-mass corrective retraining and fresh confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_v2_2_corrective_retraining import run_corrective_retraining  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/geometry19_v2_2_low_mass_corrective_retraining_v1.yaml",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    result = run_corrective_retraining(
        REPO_ROOT, args.config, device=args.device, output_root=args.output_root
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_seed": result["selected_seed"],
                "overall_rho_p50_reduction": result["confirmation_gate"][
                    "overall_rho_p50_reduction"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

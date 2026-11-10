#!/usr/bin/env python3
"""Run the controlled Geometry19 performance-only surrogate ablation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_performance_surrogate_v2 import run_performance_surrogate_v2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/wp19_22_performance_surrogate_v2_ablation.yaml",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run_performance_surrogate_v2(
        REPO_ROOT,
        args.config,
        smoke=args.smoke,
        device=args.device,
        output_root=args.output_root,
    )
    print(
        f"performance surrogate v2: {result['status']}; "
        f"passed={','.join(result['passed_arms']) or 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

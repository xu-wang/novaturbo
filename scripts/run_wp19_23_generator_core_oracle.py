#!/usr/bin/env python3
"""Run the direct-only Geometry19 generator oracle experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.geometry19_generator_oracle import run_generator_oracle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiments/wp19_23_generator_core_oracle_v1.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    summary = run_generator_oracle(ROOT, args.config, device=args.device, output_root=args.output_root, smoke=args.smoke)
    print(f"Generator oracle: {summary['status']}; targets={summary['target_count']}; physics={summary['physics_task_count']}; gate={summary['gate']['overall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


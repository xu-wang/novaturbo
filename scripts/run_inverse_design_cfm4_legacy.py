#!/usr/bin/env python3
"""Run NovaTurbo's frozen CFM-4 target-directed delivery chain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.inverse_design_delivery_cfm4_legacy import (
    DEFAULT_CONFIG,
    run_inverse_design_delivery,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-config",
        type=Path,
        required=True,
        help="YAML file containing one TargetSpec (thrust, TSFC, mass, and classification)",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = run_inverse_design_delivery(
        args.config,
        args.target_config,
        repo_root=REPO_ROOT,
        output_dir_override=args.output_dir,
    )
    print(json.dumps({"output": str(output), "status": "completed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

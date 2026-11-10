#!/usr/bin/env python3
"""Freeze or verify the canonical active-10D inverse-design baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.inverse_design_baseline import (
    DEFAULT_BASELINE_CONFIG,
    run_inverse_design_baseline,
    verify_inverse_design_baseline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--verify",
        type=Path,
        default=None,
        help="Verify an existing baseline directory instead of creating one",
    )
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_inverse_design_baseline(args.verify)
        print(json.dumps(result, sort_keys=True))
        return 0
    output = run_inverse_design_baseline(
        args.config,
        repo_root=REPO_ROOT,
        output_dir_override=args.output_dir,
    )
    print(json.dumps({"output": str(output), "status": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

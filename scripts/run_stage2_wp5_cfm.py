#!/usr/bin/env python3
"""Run the frozen CFM-default Stage-II WP5 delivery chain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.stage2_wp5 import run_stage2_wp5


DEFAULT_CONFIG = REPO_ROOT / "config" / "stage2" / "s2_wp5_cfm_default.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = run_stage2_wp5(
        args.config,
        repo_root=REPO_ROOT,
        output_dir_override=args.output_dir,
    )
    print(json.dumps({"output": str(output), "status": "completed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

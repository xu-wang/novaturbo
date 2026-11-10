#!/usr/bin/env python3
"""Run the frozen N0 target/scorer compatibility audit."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.geometry19_v2_2_cross_version_compatibility import run_audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/experiments/geometry19_v2_2_cross_version_compatibility_v1.yaml",
    )
    args = parser.parse_args()
    result = run_audit(ROOT, args.config)
    print(json.dumps({
        "status": result["status"],
        "target_calls": result["target_source_audit"]["actual_calls"],
        "target_pass": result["target_source_audit"]["pass"],
        "surrogate_pass": result["surrogate_transport_audit"]["pass"],
        "next_stage": result["next_stage"],
    }, indent=2))

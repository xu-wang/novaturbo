#!/usr/bin/env python3
"""Train tagged Geometry19 forward and feasibility MLP variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_training import run_geometry19_training  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path,
        default=REPO_ROOT / "config/physics/wp19_22_geometry19_surrogate_v1.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = run_geometry19_training(args.protocol, REPO_ROOT, smoke=args.smoke)
    print(json.dumps({
        "status": report["status"],
        "run_kind": report["run_kind"],
        "forward_test_r2": report["forward"]["test_metrics"]["performance_mean_r2"],
        "feasibility_macro_f1": report["feasibility"]["test_metrics"]["macro_f1"],
        "weight_reload_equal": report["weight_reload_equal"],
        "physics_passed": report["closed_loop"]["physics_passed_count"],
        "visualization_passed": report["closed_loop"]["visualization_passed_count"],
    }, sort_keys=True))
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

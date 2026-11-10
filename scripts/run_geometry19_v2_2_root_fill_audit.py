#!/usr/bin/env python3
"""Run the cumulative V2.2 development root-fill audit."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physics.geometry19_v2_2_root_fill_audit import run_root_fill_audit  # noqa: E402


if __name__ == "__main__":
    result = run_root_fill_audit(
        REPO_ROOT,
        REPO_ROOT / "config/physics/geometry19_v2_2_root_fill_audit_v1.yaml",
        REPO_ROOT / "data/validation/v2_2_root_fill_audit_v1",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "root_count_gate_pass": result["root_count_gate_pass"],
                "fold_gate_pass": result["fold_gate_pass"],
                "bridge_coverage_gate_pass": result["bridge_coverage_gate_pass"],
            },
            sort_keys=True,
        )
    )

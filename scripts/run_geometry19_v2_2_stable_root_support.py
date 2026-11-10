#!/usr/bin/env python3
"""Run Geometry19 V2.2 stable-root S0/S1 support audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_v2_2_stable_root_support import (  # noqa: E402
    run_stable_root_support_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/geometry19_v2_2_stable_root_conditional_surrogate_v1.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run_stable_root_support_audit(REPO_ROOT, args.config, output_root=args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "stable_roots": result["stable_roots"],
                "stable_rows": result["stable_rows"],
                "anchored_support_gate_pass": result["anchored_support_gate_pass"],
                "blind_support_gate_pass": result["blind_support_gate_pass"],
                "s2_anchored_mode_authorized": result["s2_anchored_mode_authorized"],
                "s2_blind_mode_authorized": result["s2_blind_mode_authorized"],
                "confirmation_content_read": result["confirmation_content_read"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate development-only bridge-root data and frozen grouped folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physics.geometry19_v2_2_bridge_root_expansion import (  # noqa: E402
    run_bridge_root_expansion,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "config/physics/geometry19_v2_2_bridge_root_expansion_v6.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/validation/v2_2_bridge_root_expansion_v6",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_bridge_root_expansion(
        REPO_ROOT, args.protocol, args.output, resume=args.resume
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "evaluated": result["counts"]["evaluated"],
                "accepted": result["counts"]["accepted_unique"],
                "qualified_roots": {
                    segment: values["qualified_bridge_roots"]
                    for segment, values in result["counts"]["by_segment"].items()
                },
                "fold_gate_pass": result["fold_gate_pass"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "BRIDGE_DEVELOPMENT_GATE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

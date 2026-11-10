#!/usr/bin/env python3
"""Freeze globally root-disjoint V2.2 development and confirmation roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physics.geometry19_v2_2_fresh_root_split import freeze_fresh_root_split  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT
        / "config/physics/geometry19_v2_2_fresh_low_mass_confirmation_split_v1.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/validation/v2_2_fresh_low_mass_confirmation_split_v1",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = freeze_fresh_root_split(
        REPO_ROOT, args.protocol, args.output, resume=args.resume
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "counts": {
                    role: values["rows"] for role, values in result["counts"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0 if result["training_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

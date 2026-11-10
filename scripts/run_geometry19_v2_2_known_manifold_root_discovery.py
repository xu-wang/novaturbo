#!/usr/bin/env python3
"""Discover independent V2.2 development roots inside observed support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physics.geometry19_v2_2_known_manifold_root_discovery import (  # noqa: E402
    run_known_manifold_root_discovery,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT
        / "config/physics/geometry19_v2_2_known_manifold_root_discovery_v1.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "data/validation/v2_2_known_manifold_root_discovery_v1",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_known_manifold_root_discovery(
        REPO_ROOT, args.protocol, args.output, resume=args.resume
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "roots": {
                    segment: values["roots_after"]
                    for segment, values in result["counts"]["by_segment"].items()
                },
                "effective_roots": {
                    segment: values["effective_roots_after"]
                    for segment, values in result["counts"]["by_segment"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0 if result["development_training_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate independent-root V2.2 data for the weak low-mass segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physics.geometry19_v2_2_root_diversity_expansion import (  # noqa: E402
    run_root_diversity_expansion,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT
        / "config/physics/geometry19_v2_2_independent_root_expansion_v1.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "data/validation/v2_2_independent_root_expansion_v1",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_root_diversity_expansion(
        REPO_ROOT, args.protocol, args.output, resume=args.resume
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "evaluated": result["counts"]["evaluated"],
                "accepted": result["counts"]["accepted_unique"],
                "qualified_roots": {
                    segment: values["qualified_novel_roots"]
                    for segment, values in result["counts"]["by_segment"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0 if result["data_expansion_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

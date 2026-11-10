#!/usr/bin/env python3
"""Authorize stable-root S3 from the locked model manifest and JUnit result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_v2_2_stable_root_s2_training import (  # noqa: E402
    authorize_s3_from_junit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/geometry19_v2_2_stable_root_s2_training_v1.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = authorize_s3_from_junit(REPO_ROOT, args.config, output_root=args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

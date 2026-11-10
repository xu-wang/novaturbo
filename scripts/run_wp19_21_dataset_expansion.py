#!/usr/bin/env python3
"""Run or resume the five-campaign WP19-21 expansion to 98,304 samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physics.wp19_21_expansion import run_wp19_21_expansion  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "config/physics/wp19_21_dataset_expansion_98304_v1.yaml",
    )
    args = parser.parse_args()
    report = run_wp19_21_expansion(args.protocol, REPO_ROOT)
    print(json.dumps({"status": report["status"], **report["summary"]}, sort_keys=True))
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

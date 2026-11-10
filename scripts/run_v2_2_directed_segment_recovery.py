#!/usr/bin/env python3
"""Run or resume local-sensitivity recovery of under-covered V2.2 segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physics.geometry19_v2_2_directed_recovery import run_directed_recovery


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--protocol",
    type=Path,
    default=REPO_ROOT
    / "config/physics/geometry19_v2_2_directed_segment_recovery_v1.yaml",
)
parser.add_argument(
    "--output-root",
    type=Path,
    default=REPO_ROOT / "data/validation/v2_2_directed_segment_recovery_v1",
)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

result = run_directed_recovery(
    REPO_ROOT, args.protocol, args.output_root, resume=args.resume
)
print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))

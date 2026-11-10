#!/usr/bin/env python3
"""Run or resume the simplified-V2.2 restricted-domain dataset pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physics.geometry19_v2_2_dataset import run_restricted_pilot


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--protocol",
    type=Path,
    default=REPO_ROOT / "config/physics/geometry19_v2_2_restricted_pilot_v1.yaml",
)
parser.add_argument(
    "--output-root",
    type=Path,
    default=REPO_ROOT / "data/validation/v2_2_restricted_pilot_v1",
)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

result = run_restricted_pilot(
    REPO_ROOT, args.protocol, args.output_root, resume=args.resume
)
print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))

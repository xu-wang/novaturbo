#!/usr/bin/env python3
"""Run the fixed, CPU-only simplified-V2.2 B--F software smoke."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_pipeline_smoke import run_v2_2_development_smoke


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target",
    type=Path,
    default=REPO_ROOT
    / "config/inverse_design/targets/v2_2_development_domain_smoke_v1.yaml",
)
parser.add_argument("--output-root", type=Path, default=None)
parser.add_argument("--maximum-runtime-seconds", type=float, default=120.0)
args = parser.parse_args()

output_root = args.output_root
if output_root is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = REPO_ROOT / "data/inverse_design/v2_2_development_smoke" / stamp

result = run_v2_2_development_smoke(
    REPO_ROOT,
    args.target,
    output_root,
    maximum_runtime_seconds=args.maximum_runtime_seconds,
)
print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))

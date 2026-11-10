#!/usr/bin/env python3
"""Select a C checkpoint using frozen development V2.1 metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.geometry19_checkpoint_selection import run_checkpoint_selection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiments/wp19_23_mixed_variable_cfm_v1.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    summary = run_checkpoint_selection(ROOT, args.config, device=args.device)
    print(f"checkpoint selection: {summary['status']}; epoch={summary['selected']['epoch']}; hit={summary['selected']['hit_rate']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


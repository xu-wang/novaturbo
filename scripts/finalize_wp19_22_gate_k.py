#!/usr/bin/env python3
"""Finalize the Gate K surrogate promotion decision."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_gate_k import finalize_gate_k


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "config/experiments/wp19_22_gate_k_hit_aligned_surrogate_v1.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    summary = finalize_gate_k(REPO_ROOT, args.config, output_root=args.output_root)
    print(f"Gate K decision: {summary['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

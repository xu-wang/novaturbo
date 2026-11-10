#!/usr/bin/env python3
"""Rerank the fully evaluated Gate H candidate pools with the Gate K winner."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_gate_k import run_gate_k_frozen_rerank


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "config/experiments/wp19_22_gate_k_hit_aligned_surrogate_v1.yaml",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    summary = run_gate_k_frozen_rerank(
        REPO_ROOT, args.config, device=args.device, output_root=args.output_root
    )
    print(
        "Gate K frozen rerank: "
        + ", ".join(
            f"{strategy}={value['budgets']['32']['target_hits']}/{value['target_count']}@32"
            for strategy, value in summary["strategies"].items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

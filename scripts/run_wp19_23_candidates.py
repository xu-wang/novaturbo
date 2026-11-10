#!/usr/bin/env python3
"""Run WP19-23B direct or D-Flow fixed-raw candidate generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_candidates import run_candidate_protocol


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--mode", choices=("direct", "dflow"), default="direct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--generator-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    summary = run_candidate_protocol(
        REPO_ROOT,
        args.target,
        mode=args.mode,
        device=args.device,
        output_root=args.output_root,
        generator_checkpoint=args.generator_checkpoint,
    )
    protocol = summary["candidate_protocol"]
    print(
        f"WP19-23B {args.mode}: {summary['status']}; "
        f"{protocol['n_raw']}->{protocol['n_valid']}->{protocol['n_keep']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

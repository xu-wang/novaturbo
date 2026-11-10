#!/usr/bin/env python3
"""Train the WP19-23A Geometry19 GPU-Sinkhorn CFM-4 model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_cfm import train_geometry19_cfm4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/geometry19_cfm4_seed43_v1.yaml"),
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    summary = train_geometry19_cfm4(
        args.config,
        smoke=args.smoke,
        repository_root=REPO_ROOT,
    )
    print(
        f"{summary['work_package']} {summary['run_kind']}: {summary['status']}; "
        f"epochs={summary['training']['epochs_completed']}; "
        f"source_exact={summary['validation']['source_integration_exact']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

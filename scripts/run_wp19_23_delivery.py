#!/usr/bin/env python3
"""Run WP19-23F preview/UI/JSON delivery closure."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_delivery import run_delivery_stage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--nsga2-root", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--storage-root", type=Path, default=None)
    parser.add_argument("--network-architecture-modified", action="store_true")
    parser.add_argument("--export-mesh", "--export-stl", dest="export_mesh", action="store_true")
    args = parser.parse_args()
    summary = run_delivery_stage(
        REPO_ROOT,
        args.target,
        nsga2_root=args.nsga2_root,
        source_root=args.source_root,
        output_root=args.output_root,
        storage_root=args.storage_root,
        network_architecture_modified=args.network_architecture_modified,
        export_mesh=args.export_mesh,
    )
    counts = summary["counts"]
    print(
        f"WP19-23F: {summary['status']}; delivered={counts['delivered_count']}, "
        f"preview={counts['preview_mesh_passed']}, "
        f"export={counts['export_mesh_generated']}, "
        f"json={counts['json_round_trip_passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

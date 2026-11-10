#!/usr/bin/env python3
"""Run the locked V2.1 to V2.2-C4 migration audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physics.wp_v2_2_c5_migration_audit import (
    run_v2_2_c5_migration_audit,
    write_v2_2_c5_migration_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-dataset", type=Path)
    parser.add_argument("--sample-limit", type=int, default=256)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data/validation/v2_2_c5_migration_audit_v1",
    )
    args = parser.parse_args()
    report = run_v2_2_c5_migration_audit(
        args.repository_root,
        source_dataset=args.source_dataset,
        sample_limit=args.sample_limit,
    )
    write_v2_2_c5_migration_artifacts(report, args.output_root)
    print(args.output_root / "Migration_Audit.md")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the frozen WP19-9 single-point validation protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physics.single_point_validation_v2 import (
    run_wp19_9_validation,
    write_wp19_9_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT
        / "config"
        / "physics"
        / "wp19_9_single_point_validation_v1.yaml",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "physics"
        / "wp19_9_single_point_validation_v1.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPO_ROOT
        / "docs"
        / "physics"
        / "WP19_9_SINGLE_POINT_VALIDATION_REPORT.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_wp19_9_validation(args.protocol, REPO_ROOT)
    write_wp19_9_artifacts(report, args.json_output, args.markdown_output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "passed_case_count": report["passed_case_count"],
                "case_count": report["case_count"],
                "json_output": str(args.json_output),
                "markdown_output": str(args.markdown_output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["wp19_9_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

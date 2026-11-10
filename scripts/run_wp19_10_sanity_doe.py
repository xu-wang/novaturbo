#!/usr/bin/env python3
"""Run the frozen WP19-10 small design-of-experiments protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physics.sanity_doe_v2 import run_wp19_10_doe, write_wp19_10_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT
        / "config"
        / "physics"
        / "wp19_10_sanity_doe_v1.yaml",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Override the frozen count for smoke testing.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_wp19_10_doe(
        args.protocol, REPO_ROOT, sample_count_override=args.sample_count
    )
    protocol = __import__("yaml").safe_load(
        args.protocol.read_text(encoding="utf-8")
    )
    artifacts = protocol["artifacts"]
    if args.output_dir is None:
        report_json = REPO_ROOT / artifacts["report_json"]
        sample_csv = REPO_ROOT / artifacts["sample_ledger_csv"]
        report_markdown = REPO_ROOT / artifacts["report_markdown"]
    else:
        report_json = args.output_dir / "wp19_10_sanity_doe.json"
        sample_csv = args.output_dir / "wp19_10_sanity_doe_samples.csv"
        report_markdown = args.output_dir / "WP19_10_SANITY_DOE_REPORT.md"
    write_wp19_10_artifacts(
        result, report_json, sample_csv, report_markdown
    )
    print(
        json.dumps(
            {
                "status": result.report["status"],
                "sample_count": result.report["effective_sample_count"],
                "convergence_rate": result.report["summary"]["convergence_rate"],
                "feasible_given_converged_rate": result.report["summary"][
                    "feasible_given_converged_rate"
                ],
                "dead_dimensions": result.report["sensitivity_audit"][
                    "oat_physical_path"
                ]["dead_dimensions"],
                "report_json": str(report_json),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.report["wp19_10_numerical_audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

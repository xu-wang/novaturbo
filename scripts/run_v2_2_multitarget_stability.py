#!/usr/bin/env python3
"""Run the simplified-V2.2 multi-target software-stability audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_v2_2_stability import run_v2_2_stability_campaign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "config/experiments/geometry19_v2_2_multitarget_software_stability_v1.yaml",
    )
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--evidence-root", type=Path, default=None)
    args = parser.parse_args()
    summary = run_v2_2_stability_campaign(
        REPO_ROOT,
        args.config,
        runtime_root=args.runtime_root,
        evidence_root=args.evidence_root,
    )
    aggregate = summary["aggregate"]
    print(
        f"V2.2 stability: {summary['status']}; "
        f"pipeline={aggregate['pipeline_pass_count']}/{aggregate['target_count']}; "
        "physics-valid-targets="
        f"{aggregate['targets_with_physics_acceptable']}/{aggregate['target_count']}; "
        f"runtime={aggregate['total_elapsed_seconds']:.2f}s"
    )
    print(f"evidence={summary['evidence_root']}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

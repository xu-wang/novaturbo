#!/usr/bin/env python3
"""Run the locked WP-V3-2 HECC shadow validation campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physics.wp_v3_2_validation import (
    run_wp_v3_2_validation,
    write_wp_v3_2_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--config",
        default="config/physics/wp_v3_2_hecc_validation_v1.yaml",
    )
    args = parser.parse_args()
    report = run_wp_v3_2_validation(REPO_ROOT, config_path=args.config)
    config = REPO_ROOT / args.config
    if args.output is None:
        import yaml

        default = yaml.safe_load(config.read_text(encoding="utf-8"))["artifacts"][
            "default_output_root"
        ]
        output = REPO_ROOT / default
    else:
        output = args.output
    write_wp_v3_2_artifacts(report, output)
    print(output.resolve())
    return 0 if report["software_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

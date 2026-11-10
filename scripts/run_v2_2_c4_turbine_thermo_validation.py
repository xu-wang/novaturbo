#!/usr/bin/env python3
"""Run the locked V2.2-C4 turbine variable-property campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physics.wp_v2_2_c4_validation import (
    run_v2_2_c4_validation,
    write_v2_2_c4_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/validation/wp_v2_2_c4_turbine_thermo_v1",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/physics/wp_v2_2_c4_turbine_thermo_validation_v1.yaml"
        ),
    )
    args = parser.parse_args()
    report = run_v2_2_c4_validation(REPO_ROOT, config_path=args.config)
    write_v2_2_c4_artifacts(report, args.output)
    print(args.output.resolve())
    return 0 if report["software_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

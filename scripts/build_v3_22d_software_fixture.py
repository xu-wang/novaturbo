"""Build the isolated, unvalidated 22D V3 software fixture once."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.v3_development_assets import build_v3_fixture_assets  # noqa: E402
from src.physics.v3_development_runtime import (  # noqa: E402
    build_v3_development_physics_runtime,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/validation/v3_22d_software_fixture_v1",
        help="A new directory; an existing fixture is never overwritten",
    )
    args = parser.parse_args()
    runtime = build_v3_development_physics_runtime(ROOT)
    manifest = build_v3_fixture_assets(args.output_root, runtime)
    print(
        f"{args.output_root.resolve()}: {manifest['row_count']} cases, "
        f"{manifest['converged_count']} converged, {manifest['failure_count']} failed; "
        f"schema={manifest['design_schema_hash']} "
        f"physics={manifest['physics_bundle_hash']}"
    )


if __name__ == "__main__":
    main()

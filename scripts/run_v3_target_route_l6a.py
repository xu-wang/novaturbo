"""Freeze or audit the V3 22D target-to-local-center routing protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.v3_target_route_protocol import (  # noqa: E402
    audit_v3_target_route,
    freeze_v3_target_route,
    verify_v3_target_route_complete,
    verify_v3_target_route_freeze,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "audit", "verify-freeze", "verify"))
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_root or ROOT / "data/validation/v3_22d_target_route_l6a_v1"
    if args.action == "freeze":
        result = freeze_v3_target_route(ROOT, output)
        print(f"V3 L6a route protocol frozen: {len(result['route_center_ids'])} development centers")
    elif args.action == "audit":
        audit_v3_target_route(ROOT, output)
        print("V3 L6a cross-center target coverage audit complete")
    elif args.action == "verify-freeze":
        verify_v3_target_route_freeze(ROOT, output)
        print("V3 L6a route protocol verified")
    else:
        verify_v3_target_route_complete(ROOT, output)
        print("V3 L6a route protocol and coverage audit verified")


if __name__ == "__main__":
    main()

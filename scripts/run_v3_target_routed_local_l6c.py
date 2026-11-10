"""Run L6c frozen target routing, local candidate ranking and V3 software evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.v3_target_routed_local_l6c import (  # noqa: E402
    evaluate_v3_routed_local,
    freeze_v3_routed_local,
    prepare_v3_routed_local,
    verify_v3_routed_local_complete,
    verify_v3_routed_local_freeze,
    verify_v3_routed_local_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "prepare", "evaluate", "verify-freeze", "verify-selection", "verify"))
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_root or ROOT / "data/validation/v3_22d_target_routed_local_l6c_v1"
    if args.action == "freeze":
        result = freeze_v3_routed_local(ROOT, output)
        print(f"V3 L6c protocol frozen: {len(result['target_ids'])} targets")
    elif args.action == "prepare":
        result = prepare_v3_routed_local(ROOT, output)
        print(f"V3 L6c selection sealed: {result['selected_slots']} selected candidates")
    elif args.action == "evaluate":
        result = evaluate_v3_routed_local(ROOT, output)
        print(f"V3 L6c software evaluation complete: {result['physics_evaluations']} physics evaluations")
    elif args.action == "verify-freeze":
        verify_v3_routed_local_freeze(ROOT, output)
        print("V3 L6c protocol verified")
    elif args.action == "verify-selection":
        verify_v3_routed_local_selection(ROOT, output)
        print("V3 L6c prephysics candidate selection verified")
    else:
        result = verify_v3_routed_local_complete(ROOT, output)
        print(f"V3 L6c software result verified: {result['physics_evaluations']} physics evaluations")


if __name__ == "__main__":
    main()

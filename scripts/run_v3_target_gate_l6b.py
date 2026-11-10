"""Freeze, train and verify the V3 L6b target-conditioned gate on GPU 6."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.v3_target_gate_l6b import (  # noqa: E402
    fit_v3_target_gate,
    freeze_v3_target_gate,
    verify_v3_target_gate_fit,
    verify_v3_target_gate_freeze,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "fit", "verify-freeze", "verify"))
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_root or ROOT / "data/validation/v3_22d_target_gate_l6b_gpu6_v1"
    if args.action == "freeze":
        freeze_v3_target_gate(ROOT, output)
        print("V3 L6b GPU 6 target-gate protocol frozen")
    elif args.action == "fit":
        result = fit_v3_target_gate(ROOT, output)
        print(f"V3 L6b GPU gate selected: {result['selected_neural_model_id']}; local forward: {result['selected_local_forward_model_id']}")
    elif args.action == "verify-freeze":
        verify_v3_target_gate_freeze(ROOT, output)
        print("V3 L6b GPU 6 target-gate protocol verified")
    else:
        result = verify_v3_target_gate_fit(ROOT, output)
        print(f"V3 L6b target-gate fit verified: {result['selected_neural_model_id']}")


if __name__ == "__main__":
    main()

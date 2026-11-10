"""Freeze, fit and evaluate the isolated V3 22D forward surrogate baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.v3_forward_surrogate_baseline import (  # noqa: E402
    evaluate_v3_forward_holdout,
    fit_v3_forward_surrogate,
    freeze_v3_forward_protocol,
    verify_v3_forward_complete,
    verify_v3_forward_fit,
    verify_v3_forward_protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "fit", "holdout", "verify-lock", "verify-fit", "verify"))
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_root or ROOT / "data/validation/v3_22d_forward_surrogate_baseline_v1"
    if args.action == "freeze":
        result = freeze_v3_forward_protocol(ROOT, output)
        print(f"V3 forward protocol frozen: {len(result['development_center_ids'])} development centers")
    elif args.action == "fit":
        result = fit_v3_forward_surrogate(ROOT, output)
        print(f"V3 forward model selected on validation: {result['selected_model_id']}")
    elif args.action == "holdout":
        result = evaluate_v3_forward_holdout(ROOT, output)
        print(f"V3 forward holdout evaluated: {result['selected_model_id']}")
    elif args.action == "verify-lock":
        verify_v3_forward_protocol(ROOT, output)
        print("V3 forward protocol lock verified")
    elif args.action == "verify-fit":
        result = verify_v3_forward_fit(ROOT, output)
        print(f"V3 forward fit verified: {result['selected_model_id']}")
    else:
        result = verify_v3_forward_complete(ROOT, output)
        print(f"V3 forward experiment verified: {result['selected_model_id']}")


if __name__ == "__main__":
    main()

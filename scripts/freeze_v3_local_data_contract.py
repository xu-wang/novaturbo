"""Freeze or verify the V3 future local-data center split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.v3_local_data_contract import (  # noqa: E402
    freeze_v3_local_data_contract,
    verify_v3_local_data_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify"))
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_root or ROOT / "data/validation/v3_22d_local_data_contract_v1"
    if args.action == "freeze":
        manifest = freeze_v3_local_data_contract(ROOT, output)
    else:
        manifest = verify_v3_local_data_contract(ROOT, output)
    print(f"V3 center split {args.action}: {manifest['center_count']} centers; {manifest['split_counts']}")


if __name__ == "__main__":
    main()

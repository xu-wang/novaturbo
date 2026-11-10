#!/usr/bin/env python3
"""Compare aligned baseline and enhanced WP19-23 method arms."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_method_uplift import compare_method_arms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--enhanced", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = compare_method_arms(args.baseline, args.enhanced, args.output_root)
    value = summary["overall"]
    print(
        f"WP19-23 method uplift: {value['baseline_hits']}/{value['runs']} -> "
        f"{value['enhanced_hits']}/{value['runs']}; "
        f"uplift={value['hit_rate_uplift_percentage_points']:+.2f} pp"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate or verify the committed Geometry19 V2.1 multi-case baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physics.geometry19_baseline_validation import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SUITE_PATH,
    Geometry19BaselineError,
    build_v2_1_baseline_manifest,
    verify_v2_1_baseline,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--write-candidate",
        type=Path,
        help="Write a candidate manifest to a new path instead of verifying.",
    )
    args = parser.parse_args()
    try:
        if args.write_candidate is not None:
            output = args.write_candidate.resolve()
            if output.exists():
                raise Geometry19BaselineError(
                    f"Refusing to overwrite candidate manifest: {output}"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            candidate = build_v2_1_baseline_manifest(
                REPO_ROOT, suite_path=args.suite
            )
            output.write_text(
                json.dumps(candidate, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(output)
            return 0
        report = verify_v2_1_baseline(
            REPO_ROOT,
            suite_path=args.suite,
            manifest_path=args.manifest,
        )
    except Geometry19BaselineError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

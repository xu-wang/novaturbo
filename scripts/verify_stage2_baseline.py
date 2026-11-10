#!/usr/bin/env python3
"""Verify the immutable files that define the Stage-II active-10D baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.stage2_baseline import verify_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "config/stage2/active10d_baseline_v1.yaml",
    )
    parser.add_argument(
        "--skip-git",
        action="store_true",
        help="Skip the source-commit ancestry check.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON output.")
    args = parser.parse_args()

    result = verify_manifest(
        args.manifest,
        repo_root=REPO_ROOT,
        check_git=not args.skip_git,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif result["ok"]:
        print(
            f"PASS {result['baseline_id']}: "
            f"{result['checked_artifacts']} frozen artifacts verified"
        )
    else:
        print(f"FAIL {result['baseline_id']}: {len(result['issues'])} issue(s)")
        for issue in result["issues"]:
            print(f"- {json.dumps(issue, ensure_ascii=False, sort_keys=True)}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


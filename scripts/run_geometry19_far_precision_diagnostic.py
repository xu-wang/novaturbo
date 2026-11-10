#!/usr/bin/env python3
"""Lock the FAR diagnostic without physics, or explicitly execute/resume it."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai.geometry19_far_precision_diagnostic import prepare, execute

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--lock", action="store_true")
    mode.add_argument("--authorize-diagnostic", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = prepare(ROOT) if args.lock else execute(ROOT, authorize=args.authorize_diagnostic, resume=args.resume)
    print(json.dumps({k: result[k] for k in ("status", "version", "budget", "actual_calls", "physics_execution_authorized",
                                            "numerical_issue_resolved", "training_authorized") if k in result}, indent=2))

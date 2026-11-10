#!/usr/bin/env python3
"""Prepare, explicitly authorize, or resume the staged FAR precision migration."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai.geometry19_far_precision_relabel import prepare, execute

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--authorize-regression", action="store_true")
    g.add_argument("--authorize-relabel", action="store_true")
    g.add_argument("--resume", choices=["regression", "relabel"])
    args = p.parse_args()
    result = prepare(ROOT) if args.prepare else execute(ROOT, args.resume or ("regression" if args.authorize_regression else "relabel"), resume=bool(args.resume))
    print(json.dumps({k: result[k] for k in ("status", "actual_calls", "rows", "valid_train", "valid_validation", "training_authorized") if k in result}, indent=2))

#!/usr/bin/env python3
"""Explicit modes: implementation seal, GPU preflight, or authorized training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai.geometry19_performance_gated_training import execute, seal


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=["seal", "preflight", "train"])
    parser.add_argument("--test-report", type=Path)
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    if args.mode == "seal":
        if args.test_report is None: parser.error("seal requires --test-report")
        result = seal(ROOT, args.test_report)
    else:
        result = execute(ROOT, authorized=args.authorize_training, preflight=args.mode == "preflight")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

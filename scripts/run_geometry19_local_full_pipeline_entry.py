#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai.geometry19_local_full_pipeline_entry import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = run(ROOT, resume=args.resume, verify_only=args.verify_only)
    print(json.dumps({"status": result["status"], "all_delivery_pass": result["all_delivery_pass"],
                      "first_hit_stage_counts": result["first_hit_stage_counts"]}, indent=2))

#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.ai.geometry19_performance_gated_g4 import run

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    result=run(ROOT,resume=args.resume,verify_only=args.verify_only)
    print(json.dumps({k:result[k] for k in ('status','tasks','slots','actual_calls','selected_slots','missing_slots')},indent=2))

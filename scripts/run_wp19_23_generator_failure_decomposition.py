#!/usr/bin/env python3
"""Run frozen-generator failure decomposition without improvement experiments."""

import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.ai.geometry19_generator_failure import run_failure_decomposition

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,default=ROOT/"config/experiments/wp19_23_generator_failure_decomposition_v1.yaml"); parser.add_argument("--device",default=None); parser.add_argument("--output-root",type=Path,default=None); parser.add_argument("--smoke",action="store_true"); args=parser.parse_args()
    result=run_failure_decomposition(ROOT,args.config,device=args.device,output_root=args.output_root,smoke=args.smoke)
    print(f"Generator failure decomposition: {result['status']}; targets={result['target_count']}; physics={result['physics_calls']}")


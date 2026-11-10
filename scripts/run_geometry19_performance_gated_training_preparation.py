#!/usr/bin/env python3
"""Prepare G1/G2 immutable data views without model training or physics."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.ai.geometry19_performance_gated_training_preparation import run

if __name__=="__main__":
    result=run(ROOT)
    print(json.dumps({k:result[k] for k in ("status","rows","root_count","condition_dim","physical_evaluations","training_executed")},indent=2))

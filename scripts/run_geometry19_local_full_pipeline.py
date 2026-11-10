#!/usr/bin/env python3
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.ai.geometry19_local_full_pipeline import run
if __name__=='__main__':
    result=run(ROOT)
    print(json.dumps(dict(status=result['status'],all_delivery_pass=result['all_delivery_pass']),indent=2))

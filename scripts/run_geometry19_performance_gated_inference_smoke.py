#!/usr/bin/env python3
"""Run the N1 performance-gated inference adapter GPU smoke."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.geometry19_performance_gated_inference import run_smoke


if __name__ == "__main__":
    result = run_smoke(ROOT, ROOT / "config/experiments/geometry19_v2_2_performance_gated_inference_v1.yaml")
    print(json.dumps({key: result[key] for key in ("status", "tasks", "raw_slots", "valid_slots", "selected_rows", "physics_evaluations", "next_stage")}, indent=2))

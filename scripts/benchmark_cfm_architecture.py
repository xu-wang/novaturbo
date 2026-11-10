#!/usr/bin/env python3
"""Reproducible GPU generation benchmark for a stage-gated CFM architecture."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.generators import load_generator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--ode-steps", type=int, choices=(8, 16, 32, 64))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CFM architecture benchmark")
    device = torch.device("cuda:0")
    generator = load_generator(args.checkpoint, device=str(device))
    original_ode_steps = int(generator.model_config.get("ode_steps", 32))
    if args.ode_steps is not None:
        generator.model_config["ode_steps"] = int(args.ode_steps)
    target = pd.read_csv(args.targets).iloc[0]
    condition_names = generator.normalizer.condition_names
    physical = target[list(condition_names)].to_numpy(dtype=float)
    condition = generator.normalizer.transform_conditions(physical)

    for index in range(args.warmup):
        generator.sample(condition, 128, seed=args.seed + index, space="normalized")
    torch.cuda.synchronize(device)
    timings = []
    nfe = []
    for index in range(args.repeats):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        generated = generator.sample(
            condition, 128, seed=args.seed + 1000 + index, space="normalized"
        )
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - started)
        nfe.append(int(generated.diagnostics.get("nfe", 0)))

    values = np.asarray(timings, dtype=float)
    payload = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "dtype": str(generator.dtype),
        "N_raw": 128,
        "warmup_runs": args.warmup,
        "timed_runs": args.repeats,
        "mean_seconds": float(values.mean()),
        "median_seconds": float(np.median(values)),
        "std_seconds": float(values.std(ddof=0)),
        "min_seconds": float(values.min()),
        "max_seconds": float(values.max()),
        "mean_nfe": float(statistics.mean(nfe)),
        "ode_method": generator.model_config.get("ode_method"),
        "ode_steps": generator.model_config.get("ode_steps"),
        "checkpoint_ode_steps": original_ode_steps,
        "timings_seconds": timings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "timings_seconds"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

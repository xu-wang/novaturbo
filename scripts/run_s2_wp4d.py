#!/usr/bin/env python3
"""Run the frozen S2-WP4D random Gaussian source sensitivity matrix."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.wp4 import run_wp4


DEFAULT_CONFIG = Path(
    "config/experiments/cfm_arch/s2_wp4d_random_source_sensitivity.yaml"
)
FROZEN_SOURCE_SEED_COUNT = 8


def load_matrix(path: Path) -> Dict[str, Any]:
    matrix = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"version", "experiment_id", "fixed", "checkpoints", "source_seeds", "output"}
    missing = required.difference(matrix)
    if missing:
        raise ValueError(f"S2-WP4D matrix is missing keys: {sorted(missing)}")
    fixed = matrix["fixed"]
    if int(fixed["sample_count"]) != 128:
        raise ValueError("S2-WP4D freezes sample_count at 128")
    if str(fixed["sampling_policy"]) != "v2_1_fixed_raw":
        raise ValueError("S2-WP4D requires the V2.1 fixed-raw protocol")
    if int(fixed["ode_steps"]) != 32:
        raise ValueError("S2-WP4D freezes RK4 at 32 steps")
    checkpoint_seeds = sorted(int(seed) for seed in matrix["checkpoints"])
    if checkpoint_seeds != [42, 43]:
        raise ValueError("S2-WP4D requires checkpoint seeds 42 and 43")
    source_seeds = [int(seed) for seed in matrix["source_seeds"]]
    if len(source_seeds) != FROZEN_SOURCE_SEED_COUNT or len(set(source_seeds)) != len(source_seeds):
        raise ValueError("S2-WP4D requires eight unique source seeds")
    return matrix


def build_wp4_config(
    matrix: Dict[str, Any], checkpoint_seed: int, source_seed: int
) -> Dict[str, Any]:
    fixed = matrix["fixed"]
    output_root = Path(matrix["output"]["root"])
    return {
        "version": f"s2_wp4d_checkpoint{checkpoint_seed}_source{source_seed}_v2_1",
        "metrics_config": fixed["metrics_config"],
        "data": {
            "dataset_csv": fixed["dataset_csv"],
            "schema": fixed["schema"],
            "normalizer": fixed["normalizer"],
            "target_suite": fixed["target_suite"],
        },
        "models": {"cfm": matrix["checkpoints"][checkpoint_seed]},
        "runtime": {
            "device": fixed["device"],
            "sample_count": int(fixed["sample_count"]),
            "sample_seed": int(source_seed),
            "maximum_targets_per_type": None,
            "sampling": {"policy": fixed["sampling_policy"]},
            "generator_overrides": {"ode_steps": int(fixed["ode_steps"])},
        },
        "output": {
            "directory": str(
                output_root
                / f"checkpoint{checkpoint_seed}_source{source_seed}_v2_1"
            ),
            "refuse_overwrite": bool(matrix["output"].get("refuse_overwrite", True)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Resume a matrix by retaining already completed immutable run directories.",
    )
    args = parser.parse_args()
    config_path = (REPO_ROOT / args.config).resolve()
    matrix = load_matrix(config_path)
    completed = []
    skipped = []
    with tempfile.TemporaryDirectory(prefix="s2_wp4d_") as temporary:
        temporary_root = Path(temporary)
        for checkpoint_seed in (42, 43):
            for source_seed in matrix["source_seeds"]:
                config = build_wp4_config(matrix, checkpoint_seed, int(source_seed))
                output = REPO_ROOT / config["output"]["directory"]
                if output.exists() and args.skip_existing:
                    skipped.append(str(output.relative_to(REPO_ROOT)))
                    continue
                generated_config = temporary_root / (
                    f"checkpoint{checkpoint_seed}_source{source_seed}.yaml"
                )
                generated_config.write_text(
                    yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
                )
                run_wp4(generated_config, repo_root=REPO_ROOT)
                completed.append(str(output.relative_to(REPO_ROOT)))
    print(json.dumps({"completed": completed, "skipped": skipped}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Run one model/seed partition from a WP3 training configuration."""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.wp3 import run_wp3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=("nflows", "cinn"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = copy.deepcopy(source)
    config["version"] = f"{source['version']}__{args.model}__seed{args.seed}"
    config["runtime"]["seeds"] = [args.seed]
    for name, model_config in config["models"].items():
        model_config["enabled"] = name == args.model
    config["output"] = {"directory": "unused", "refuse_overwrite": True}

    with tempfile.TemporaryDirectory(prefix="novaturbo-wp3-partition-") as temporary:
        path = Path(temporary) / "partition.yaml"
        path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        output = run_wp3(
            path,
            repo_root=REPO_ROOT,
            output_dir_override=args.output_dir.resolve(),
        )
    print(f"WP3 partition complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

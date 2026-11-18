"""Command line interface for the standalone V2.1 capability.

Examples::

    python -m novaturbo_v21 info
    python -m novaturbo_v21 predict --design design.json
    python -m novaturbo_v21 evaluate --design design.json --target target.json
    python -m novaturbo_v21 generate --target target.json --n 256 --seed 20260817
    python -m novaturbo_v21 inverse --target target.json --n 256 --budget 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .api import V21Capability, dumps


def _load(path: str) -> dict:
    if path == "-":
        return json.loads(sys.stdin.read().lstrip("\ufeff"))
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="novaturbo_v21", description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resource-root", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="print contracts and model identities")

    predict = sub.add_parser("predict", help="forward surrogate prediction")
    predict.add_argument("--design", required=True, help="JSON file with the 19 coordinates")

    evaluate = sub.add_parser("evaluate", help="authoritative V2.1 physics evaluation")
    evaluate.add_argument("--design", required=True)
    evaluate.add_argument("--target", default=None)
    evaluate.add_argument("--environment", default=None, help='JSON file, e.g. {"temperature_K": 288.15}')

    generate = sub.add_parser("generate", help="conditional generation from targets")
    generate.add_argument("--target", required=True)
    generate.add_argument("--n", type=int, default=256)
    generate.add_argument("--seed", type=int, default=None)
    generate.add_argument("--guided", action="store_true", help="run D-Flow source guidance")

    inverse = sub.add_parser("inverse", help="generate + rank + V2.1 verification")
    inverse.add_argument("--target", required=True)
    inverse.add_argument("--n", type=int, default=256)
    inverse.add_argument("--seed", type=int, default=None)
    inverse.add_argument("--budget", type=int, default=0, help="max V2.1 calls; 0 = verify every kept candidate")
    inverse.add_argument("--direct", action="store_true", help="skip D-Flow guidance")

    args = parser.parse_args(argv)
    capability = V21Capability(device=args.device, resource_root=args.resource_root)

    if args.command == "info":
        print(dumps(capability.describe()))
    elif args.command == "predict":
        print(dumps(capability.predict(_load(args.design))))
    elif args.command == "evaluate":
        environment = _load(args.environment) if args.environment else None
        target = _load(args.target) if args.target else None
        print(dumps(capability.evaluate(_load(args.design), environment, target)))
    elif args.command == "generate":
        print(dumps(capability.generate(
            _load(args.target), args.n, args.seed, guided=args.guided
        )))
    elif args.command == "inverse":
        print(dumps(capability.inverse_design(
            _load(args.target), n_samples=args.n, seed=args.seed,
            physics_budget=args.budget, guided=not args.direct,
        )))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

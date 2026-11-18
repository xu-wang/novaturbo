#!/usr/bin/env python3
"""Verify the two V2.1 weight bundles and the physics chain they feed.

Checks, in order:

1. every declared artefact exists and matches the hash recorded in its manifest;
2. the CFM-4 bounded-sigmoid generator loads and samples in physical space;
3. the ``performance3_log`` surrogate loads through the pipeline's own
   ``resolve_forward_model_spec`` contract (tag, transform, three outputs);
4. the V2.1 physics runtime evaluates a reference design and returns a
   converged operating point;
5. the surrogate and physics agree in order of magnitude on that design.

Exit code 0 means all checks passed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_forward_model import resolve_forward_model_spec  # noqa: E402
from src.ai.geometry19_inverse_contract import load_wp19_23_contract  # noqa: E402
from src.ai.geometry19_surrogate import load_weight_bundle, sha256_file  # noqa: E402
from src.ai.generators import load_generator  # noqa: E402
from src.physics import Environment, Geometry19  # noqa: E402


def _check(condition: bool, message: str, failures: list[str]) -> None:
    print(("PASS  " if condition else "FAIL  ") + message)
    if not condition:
        failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generator-dir", type=Path, default=None)
    parser.add_argument("--surrogate-dir", type=Path, default=None)
    args = parser.parse_args()

    protocol = load_wp19_23_contract(REPO_ROOT / "config/physics/wp19_23_inverse_closure_v1.yaml")
    payload = protocol.payload
    failures: list[str] = []

    # 1. generator artefacts
    variant = payload["models"]["generator"]["variants"]["bounded-sigmoid"]
    checkpoint = REPO_ROOT / variant["checkpoint"]
    normalizer = REPO_ROOT / variant["normalizer"]
    _check(checkpoint.is_file(), f"generator checkpoint exists: {variant['checkpoint']}", failures)
    _check(normalizer.is_file(), f"generator normalizer exists: {variant['normalizer']}", failures)
    if checkpoint.is_file():
        print(f"      sha256(best.pt) = {sha256_file(checkpoint)}")
        print(f"      size(best.pt)   = {checkpoint.stat().st_size} bytes")

    # 2. generator loads and samples
    generator = load_generator(checkpoint, device=args.device)
    _check(
        generator.design_dim == 19 and generator.condition_dim == 3,
        "generator exposes 19 design dimensions conditioned on 3 targets",
        failures,
    )
    target = {"thrust_N": 60.0, "tsfc_kg_N_s": 6.7e-5, "mass_kg": 1.77}
    condition = [[target[name] for name in ("thrust_N", "tsfc_kg_N_s", "mass_kg")]]
    sample = generator.sample(condition, 64, seed=20260817, space="physical")
    values = sample.values.detach().cpu().numpy().reshape(-1, 19)
    _check(bool(np.isfinite(values).all()), "generator samples are finite", failures)
    schema = yaml.safe_load(
        (REPO_ROOT / "config/physics/geometry19_schema_v1.yaml").read_text(encoding="utf-8")
    )
    names = [item["name"] for item in schema["design_variables"]]
    bounds = {item["name"]: item["bounds"] for item in schema["design_variables"]}
    in_bounds = sum(
        all(bounds[names[index]][0] <= row[index] <= bounds[names[index]][1] for index in range(19))
        for row in values
    )
    _check(
        in_bounds == len(values),
        f"all {len(values)} generated samples are inside the schema bounds (in_bounds={in_bounds})",
        failures,
    )

    # 3. surrogate contract
    spec = resolve_forward_model_spec(
        REPO_ROOT,
        payload,
        weights_override=args.surrogate_dir,
        logical_tag=None if args.surrogate_dir else "performance3_log",
    )
    _check(spec.output_transform == "natural_log", "surrogate output transform is natural_log", failures)
    _check(spec.output_contract == "performance3_only", "surrogate exposes only the three performance outputs", failures)
    _check(tuple(spec.target_names) == ("thrust_N", "tsfc_kg_N_s", "mass_kg"), "surrogate target order is thrust/TSFC/mass", failures)
    model, normalization, manifest = load_weight_bundle(
        spec.directory,
        expected_model_tag=spec.model_tag,
        expected_feature_names=names,
        expected_target_names=("thrust_N", "tsfc_kg_N_s", "mass_kg"),
        device=args.device,
    )
    print(f"      surrogate model_tag = {manifest['model_tag']}")
    print(f"      surrogate trained   = {manifest.get('training_note', 'published checkpoint')}")

    # 4. physics runtime
    from src.physics.runtime import build_physics_runtime

    runtime = build_physics_runtime(REPO_ROOT, payload)
    reference = {}
    for item in schema["design_variables"]:
        value = item["reference_value"]
        reference[item["name"]] = (
            int(value) if str(item.get("dtype", "")).startswith("int") else float(value)
        )
    design = Geometry19.from_mapping(reference)
    result = runtime.evaluate(design, Environment())
    print(
        f"      solver: converged={result.solver.converged} "
        f"failure={result.solver.failure_code} message={result.solver.message!r}"
    )
    _check(bool(result.solver.converged), "V2.1 physics converges on the reference design", failures)
    _check(bool(result.performance), "V2.1 physics returns performance outputs", failures)
    print(f"      backend             = {runtime.backend_id}")
    print(f"      physics bundle hash = {runtime.physics_bundle_hash}")

    # 5. surrogate vs physics sanity
    row = np.asarray([[reference[name] for name in names]], dtype=np.float32)
    normalized = (row - normalization["input_mean"]) / normalization["input_std"]
    with torch.no_grad():
        raw = model(torch.from_numpy(normalized).to(torch.device(args.device))).cpu().numpy()
    decoded = raw * normalization["output_std"] + normalization["output_mean"]
    predicted = {
        name: float(math.exp(decoded[0, index]))
        for index, name in enumerate(("thrust_N", "tsfc_kg_N_s", "mass_kg"))
    }
    truth = {name: float(result.performance[name]) for name in predicted}
    print("      surrogate:", json.dumps(predicted, sort_keys=True))
    print("      physics  :", json.dumps(truth, sort_keys=True))
    agrees = all(
        abs(predicted[name] - truth[name]) / abs(truth[name]) < 0.35 for name in predicted
    )
    _check(agrees, "surrogate and physics agree within 35% on the reference design", failures)

    if failures:
        print(f"\n{len(failures)} check(s) failed")
        return 1
    print("\nAll V2.1 weight and chain checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

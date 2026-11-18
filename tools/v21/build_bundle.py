#!/usr/bin/env python3
"""Build a standalone ``novaturbo_v21`` capability bundle from this repository.

The bundle vendors the transitive ``src.*`` import closure required by the V2.1
(Geometry19, 19-D) physics + generator + forward-surrogate stack, rewrites the
absolute ``src.*`` imports into the bundle namespace, copies the frozen configs
and model weights, and adds a small public API.

Usage::

    python tools/v21/build_bundle.py \
        --output v21_capability \
        [--generator-dir data/generator_models/.../baseline] \
        [--surrogate-dir data/inverse_design/.../seed_43] \
        [--mesh]

The result is self-contained: copy the output directory into another project,
``pip install -r requirements.txt`` and ``from novaturbo_v21 import V21Capability``.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
BUNDLE_NAMESPACE = "novaturbo_v21._vendor.src"

SEED_MODULES = [
    # physics: the V2.1 closed low-order model and its runtime
    "src.physics.runtime",
    "src.physics.geometry_state",
    "src.physics.model_config",
    "src.physics.constraint_profiles",
    "src.physics.engine_match",
    "src.physics.thermo",
    "src.physics.maps",
    "src.physics.mass_model_v2",
    "src.physics.materials",
    "src.physics.nozzle",
    "src.physics.combustor_loworder",
    "src.physics.combustor_map_enhanced",
    "src.physics.compressor_map_enhanced",
    "src.physics.compressor_meanline",
    "src.physics.formal_constraints_v2",
    "src.physics.inlet_map_enhanced",
    "src.physics.nozzle_map_enhanced",
    "src.physics.sanity_doe_v2",
    "src.physics.single_point_validation_v2",
    "src.physics.turbine_map_enhanced",
    "src.physics.turbine_meanline",
    # models: conditional generator + performance surrogate + hit alignment
    "src.ai.generators",
    "src.ai.generator_dataset",
    "src.ai.geometry19_surrogate",
    "src.ai.geometry19_hit_alignment",
    "src.ai.geometry19_forward_model",
    "src.ai.geometry19_inverse_contract",
]

MESH_SEEDS = [
    "src.visualization.geometry19_adapter",
    "src.visualization.geometry19_mesh",
    "src.geometry.assembly",
]

ASSET_DIRS = [
    "config/physics",
]

ASSET_FILES = [
    "config/physics/geometry19_schema_v1.yaml",
    "config/physics/geometry19_calibration_bootstrap_v1.yaml",
    "config/physics/geometry19_calibration_schema_v1.yaml",
    "config/physics/geometry19_constraint_profile_pipeline_debug_v1.yaml",
    "config/physics/geometry19_operating_point_policy_reference_continuation_v1.yaml",
    "config/physics/geometry19_conservation_tolerances_v1.yaml",
    "config/physics/geometry19_acceptance_policy_v1.yaml",
    "config/physics/geometry19_visualization_calibration_v1.yaml",
    "config/physics/geometry19_mesh_profiles_v1.yaml",
    "config/physics/geometry19_constraint_dominance_v1.yaml",
    "config/physics/geometry19_component_maps_v1.yaml",
    "config/physics/geometry19_thermo_variable_property_shadow_v1.yaml",
    "config/physics/wp19_23_inverse_closure_v1.yaml",
]

TRIMMED_INITS = {
    # the full package __init__ pulls every generator adapter (nflows/FrEIA/cVAE);
    # the bundle only needs the CFM-4 adapter used by V2.1.
    "src/ai/generators/__init__.py": (
        '"""Conditional generator adapters bundled for the V2.1 capability."""\n\n'
        "from .base import ConditionalGenerator, GeneratedSamples, load_generator\n"
        "from .torchcfm_adapter import CFMGenerator\n\n"
        '__all__ = ["ConditionalGenerator", "GeneratedSamples", "load_generator", "CFMGenerator"]\n'
    ),
    # the full package __init__ pulls Geometry19DesignStore (filesystem store),
    # which the standalone capability does not need.
    "src/visualization/__init__.py": (
        '"""Visualization adapter subset bundled for the V2.1 capability."""\n\n'
        "from .geometry19_adapter import (\n"
        "    VisualizationAssembly,\n"
        "    VisualizationCalibration,\n"
        "    VisualizationCalibrationError,\n"
        "    VisualizationGeometryError,\n"
        "    geometry19_to_assembly_params,\n"
        ")\n\n"
        "__all__ = [\n"
        '    "VisualizationAssembly",\n'
        '    "VisualizationCalibration",\n'
        '    "VisualizationCalibrationError",\n'
        '    "VisualizationGeometryError",\n'
        '    "geometry19_to_assembly_params",\n'
        "]\n"
    ),
}


def module_path(name: str) -> Path | None:
    relative = Path(*name.split(".")[1:])
    candidate = SRC / relative.with_suffix(".py")
    if candidate.is_file():
        return candidate
    package = SRC / relative / "__init__.py"
    if package.is_file():
        return package
    return None


def imports_of(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    is_package = path.name == "__init__.py"
    package = module if is_package else module.rpartition(".")[0]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src."):
                    names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = (
                    ".".join(parts[: len(parts) - (node.level - 1)])
                    if node.level > 1
                    else package
                )
                target = f"{base}.{node.module}" if node.module else base
            elif node.module and node.module.startswith("src."):
                target = node.module
            else:
                continue
            names.add(target)
            for alias in node.names:
                if alias.name != "*":
                    names.add(f"{target}.{alias.name}")
    return names


def compute_closure(seeds: list[str]) -> set[str]:
    closure: set[str] = set()
    stack = list(seeds)
    while stack:
        name = stack.pop()
        if name in closure:
            continue
        path = module_path(name)
        if path is None:
            continue
        closure.add(name)
        stack.extend(child for child in imports_of(path, name) if child not in closure)
    return closure


def rewrite(text: str) -> str:
    text = re.sub(r"(?m)^(\s*)from src\.", lambda m: f"{m.group(1)}from {BUNDLE_NAMESPACE}.", text)
    text = re.sub(r"(?m)^(\s*)import src\.", lambda m: f"{m.group(1)}import {BUNDLE_NAMESPACE}.", text)
    return text


def copy_module(path: Path, destination_root: Path) -> None:
    target = destination_root / path.relative_to(SRC)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8")
    target.write_text(rewrite(text), encoding="utf-8")


def ensure_package_inits(vendor_src: Path, closure_paths: list[Path]) -> None:
    """Create __init__.py for every vendored package that has none."""
    directories = {path.parent for path in closure_paths}
    for directory in sorted(directories):
        relative = directory.relative_to(SRC)
        package_dir = vendor_src / relative
        if not package_dir.is_dir():
            continue
        init = package_dir / "__init__.py"
        if not init.is_file():
            init.write_text('"""Vendored NovaTurbo module package."""\n', encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "v21_capability")
    parser.add_argument("--generator-dir", type=Path, default=None)
    parser.add_argument("--surrogate-dir", type=Path, default=None)
    parser.add_argument("--mesh", action="store_true", help="also vendor the STL mesh adapter")
    parser.add_argument("--force", action="store_true", help="replace an existing output directory")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"Output exists; pass --force to replace: {output}")
        shutil.rmtree(output)

    protocol = REPO_ROOT / "config/physics/wp19_23_inverse_closure_v1.yaml"
    generator_dir = args.generator_dir or (
        REPO_ROOT
        / "data/generator_models/geometry19/wp19_23_cfm4_bounded_sigmoid_seed43_v1/baseline"
    )
    surrogate_dir = args.surrogate_dir or (
        REPO_ROOT
        / "data/inverse_design/wp19_22_performance_surrogate_v2_ablation/formal/models/performance3_log/seed_43"
    )
    for required in ("best.pt", "normalizer.json"):
        if not (generator_dir / required).is_file():
            raise SystemExit(f"Generator artefact missing: {generator_dir / required}")
    for required in ("weights_best_val.pt", "normalization.npz", "model_manifest.json"):
        if not (surrogate_dir / required).is_file():
            raise SystemExit(f"Surrogate artefact missing: {surrogate_dir / required}")

    seeds = list(SEED_MODULES)
    if args.mesh:
        seeds.extend(MESH_SEEDS)
    closure = compute_closure(seeds)
    paths = sorted(path for path in (module_path(name) for name in closure) if path)
    print(f"vendoring {len(paths)} modules")

    package = output / "novaturbo_v21"
    vendor = package / "_vendor"
    vendor_src = vendor / "src"
    vendor.mkdir(parents=True)
    (vendor / "__init__.py").write_text('"""Vendored NovaTurbo sources."""\n', encoding="utf-8")

    for path in paths:
        copy_module(path, vendor_src)
    for relative, content in TRIMMED_INITS.items():
        if any(path.relative_to(SRC) == Path(relative) for path in paths):
            (vendor_src / relative).write_text(content, encoding="utf-8")
    ensure_package_inits(vendor_src, paths)

    for directory in ASSET_DIRS:
        source_dir = REPO_ROOT / directory
        target_dir = vendor / directory
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
    for relative in ASSET_FILES:
        source = REPO_ROOT / relative
        target = vendor / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    generator_relative = (
        Path("data/generator_models/geometry19")
        / "wp19_23_cfm4_bounded_sigmoid_seed43_v1"
        / "baseline"
    )
    surrogate_relative = (
        Path("data/inverse_design")
        / "wp19_22_performance_surrogate_v2_ablation"
        / "formal/models/performance3_log"
        / "seed_43"
    )
    for relative, source_dir, files in (
        (generator_relative, generator_dir, ("best.pt", "normalizer.json")),
        (surrogate_relative, surrogate_dir,
         ("weights_best_val.pt", "normalization.npz", "model_manifest.json", "training_metrics.json")),
    ):
        for name in files:
            source = source_dir / name
            if source.is_file():
                target = vendor / relative / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    api_source = Path(__file__).resolve().parent / "templates/api.py"
    shutil.copy2(api_source, package / "api.py")
    (package / "__init__.py").write_text(
        '"""Standalone NovaTurbo V2.1 (Geometry19, 19-D) capability."""\n\n'
        "from .api import (\n"
        "    V21Capability,\n"
        "    V21Error,\n"
        "    load_capability,\n"
        ")\n\n"
        '__all__ = ["V21Capability", "V21Error", "load_capability"]\n',
        encoding="utf-8",
    )
    for name in ("__main__.py",):
        shutil.copy2(Path(__file__).resolve().parent / "templates" / name, package / name)
    for name in ("README.md", "requirements.txt", "example_usage.py"):
        shutil.copy2(Path(__file__).resolve().parent / "templates" / name, output / name)
    tests = output / "tests"
    tests.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).resolve().parent / "templates/test_bundle.py", tests / "test_bundle.py")

    manifest = {
        "bundle": "novaturbo_v21",
        "source_repository": str(REPO_ROOT),
        "protocol": str(protocol.relative_to(REPO_ROOT)),
        "modules_vendored": len(paths) + len(TRIMMED_INITS),
        "mesh_enabled": bool(args.mesh),
        "generator_weights": str(generator_relative),
        "surrogate_weights": str(surrogate_relative),
        "assets": ASSET_FILES,
    }
    (output / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"bundle written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

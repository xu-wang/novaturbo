#!/usr/bin/env python3
"""Generate and audit the 1 + 2*19 Geometry19 visualization regression set."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping

import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/novaturbo-matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physics import Geometry19  # noqa: E402
from src.visualization import Geometry19DesignStore  # noqa: E402


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "visualization"
    / "wp19_16_geometry19_regression_20260825"
)
SCHEMA_PATH = REPO_ROOT / "config" / "physics" / "geometry19_schema_v1.yaml"
SOLID_COMPONENTS = {
    "inlet",
    "compressor",
    "combustor",
    "turbine",
    "nozzle",
    "shaft",
    "casing",
    "transitions",
}
LATTICE_COMPONENTS = {"combustor_lattice", "nozzle_lattice"}
COMPONENT_COLORS = {
    "inlet": "#4fc3f7",
    "compressor": "#81c784",
    "combustor": "#ff8a65",
    "turbine": "#ffd54f",
    "nozzle": "#e57373",
    "shaft": "#b0bec5",
    "casing": "#78909c",
    "transitions": "#9575cd",
    "combustor_lattice": "#ff6d00",
    "nozzle_lattice": "#ff3d00",
}
EXPECTED_HARD_FAILURES = {
    "02_compressor_inducer_tip_diameter_m_high": {
        "failure_type": "Geometry19ServiceError",
        "message": "compressor inducer tip must be inside the exit",
        "constraint": "compressor_inducer_order",
        "reason": "Frozen marginal high bound D_c1t=0.100 m exceeds reference D_c2=0.080 m.",
    }
}


@dataclass(frozen=True)
class RegressionCase:
    name: str
    variable_name: str | None
    boundary: str
    design: Geometry19


def load_schema() -> dict[str, Any]:
    return yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))


def build_cases(schema: Mapping[str, Any]) -> list[RegressionCase]:
    variables = schema["design_variables"]
    reference = Geometry19.from_mapping(
        {item["name"]: item["reference_value"] for item in variables}
    )
    cases = [RegressionCase("reference", None, "reference", reference)]
    for item in variables:
        index = int(item["index"])
        name = str(item["name"])
        for boundary, value in (("low", item["bounds"][0]), ("high", item["bounds"][1])):
            case_name = f"{index:02d}_{name}_{boundary}"
            cases.append(
                RegressionCase(
                    case_name,
                    name,
                    boundary,
                    replace(reference, **{name: value}),
                )
            )
    return cases


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def component_hashes(component_directory: Path) -> dict[str, str]:
    return {
        path.stem: sha256_file(path)
        for path in sorted(component_directory.glob("*.stl"))
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _link_components(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(source.glob("*.stl")):
        target = destination / source_path.name
        if target.exists() and sha256_file(target) == sha256_file(source_path):
            continue
        if target.exists():
            target.unlink()
        try:
            os.link(source_path, target)
        except OSError:
            shutil.copy2(source_path, target)


def _load_meshes(component_directory: Path) -> list[tuple[str, Any]]:
    import trimesh

    meshes = []
    for path in sorted(component_directory.glob("*.stl")):
        if path.stem not in COMPONENT_COLORS:
            continue
        mesh = trimesh.load_mesh(path, process=False, force="mesh")
        meshes.append((path.stem, mesh))
    return meshes


def render_front_view(component_directory: Path, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.collections import PolyCollection
    import numpy as np

    figure, axes = plt.subplots(figsize=(12, 5), dpi=120)
    figure.patch.set_facecolor("#0a0e17")
    axes.set_facecolor("#0a0e17")
    for name, mesh in _load_meshes(component_directory):
        triangles = np.asarray(mesh.triangles)
        if len(triangles) > 30000:
            triangles = triangles[np.linspace(0, len(triangles) - 1, 30000, dtype=int)]
        polygons = triangles[:, :, [2, 1]]
        axes.add_collection(
            PolyCollection(
                polygons,
                facecolor=COMPONENT_COLORS.get(name, "#00e5ff"),
                edgecolor="none",
                alpha=0.55 if name in LATTICE_COMPONENTS else 0.82,
                rasterized=True,
            )
        )
    axes.autoscale_view()
    axes.set_aspect("equal", adjustable="box")
    axes.axis("off")
    figure.tight_layout(pad=0.1)
    figure.savefig(output_path, facecolor=figure.get_facecolor(), bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def render_section_view(component_directory: Path, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    import numpy as np

    figure, axes = plt.subplots(figsize=(12, 5), dpi=120)
    figure.patch.set_facecolor("#0a0e17")
    axes.set_facecolor("#0a0e17")
    for name, mesh in _load_meshes(component_directory):
        section = mesh.section(
            plane_origin=np.array([0.0, 0.0, 0.0]),
            plane_normal=np.array([1.0, 0.0, 0.0]),
        )
        if section is None:
            continue
        segments = []
        for line in section.discrete:
            points = np.asarray(line)
            if len(points) >= 2:
                segments.extend(np.stack((points[:-1, [2, 1]], points[1:, [2, 1]]), axis=1))
        if segments:
            axes.add_collection(
                LineCollection(
                    segments,
                    colors=COMPONENT_COLORS.get(name, "#00e5ff"),
                    linewidths=0.8 if name in LATTICE_COMPONENTS else 1.25,
                    alpha=0.9,
                )
            )
    axes.autoscale_view()
    axes.set_aspect("equal", adjustable="box")
    axes.axis("off")
    figure.tight_layout(pad=0.1)
    figure.savefig(output_path, facecolor=figure.get_facecolor(), bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def evaluate_channel(
    case: RegressionCase,
    variable: Mapping[str, Any] | None,
    response: Mapping[str, Any],
    hashes: Mapping[str, str],
    reference_response: Mapping[str, Any],
    reference_hashes: Mapping[str, str],
) -> dict[str, Any]:
    changed = sorted(
        name
        for name in set(hashes) | set(reference_hashes)
        if hashes.get(name) != reference_hashes.get(name)
    )
    if case.variable_name is None:
        checks = {"reference_case": True}
        channel = "reference"
    else:
        component = str(variable["component"])
        if component == "operating_point":
            channel = "overlay"
            checks = {
                "solid_geometry_hash_unchanged": response["geometry_hash"]
                == reference_response["geometry_hash"],
                "lattice_hash_unchanged": response["lattice_hash"]
                == reference_response["lattice_hash"],
                "component_stl_unchanged": not changed,
                "physics_changed": response["physics"]["performance"]
                != reference_response["physics"]["performance"],
            }
        elif component == "lattice":
            channel = "lattice"
            checks = {
                "solid_geometry_hash_unchanged": response["geometry_hash"]
                == reference_response["geometry_hash"],
                "lattice_hash_changed": response["lattice_hash"]
                != reference_response["lattice_hash"],
                "solid_components_unchanged": not (set(changed) & SOLID_COMPONENTS),
                "lattice_component_changed": bool(set(changed) & LATTICE_COMPONENTS),
            }
        else:
            channel = "solid"
            checks = {
                "solid_geometry_hash_changed": response["geometry_hash"]
                != reference_response["geometry_hash"],
                "target_component_changed": component in changed,
            }
    checks["mesh_diagnostics_passed"] = bool(response["mesh_diagnostics"]["passed"])
    return {
        "channel": channel,
        "changed_components": changed,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _integer_measurement_checks(
    case: RegressionCase, response: Mapping[str, Any]
) -> dict[str, bool]:
    measurements = response["mesh_diagnostics"].get("geometry_measurements", {})
    design = case.design.as_mapping()
    return {
        "compressor_blade_count_exact": measurements.get("compressor_blade_count")
        == design["compressor_blade_count"],
        "combustor_injector_count_exact": measurements.get("combustor_injector_count")
        == design["combustor_injector_count"],
        "turbine_blade_count_exact": measurements.get("turbine_blade_count")
        == design["turbine_blade_count"],
    }


def select_cases(cases: list[RegressionCase], names: Iterable[str]) -> list[RegressionCase]:
    requested = set(names)
    if not requested:
        return cases
    selected = [case for case in cases if case.name in requested]
    missing = requested - {case.name for case in selected}
    if missing:
        raise ValueError("Unknown cases: " + ", ".join(sorted(missing)))
    if "reference" not in {case.name for case in selected}:
        selected.insert(0, cases[0])
    return selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    schema = load_schema()
    variables = {item["name"]: item for item in schema["design_variables"]}
    cases = select_cases(build_cases(schema), args.case)
    output_root = args.output_root.resolve()
    storage_root = (args.storage_root or output_root / "_design_store_v3").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    store = Geometry19DesignStore(REPO_ROOT, storage_root)
    records = []
    reference_response = None
    reference_hashes = None

    for ordinal, case in enumerate(cases, start=1):
        print(f"[{ordinal:02d}/{len(cases):02d}] {case.name}", flush=True)
        case_root = output_root / "cases" / case.name
        case_root.mkdir(parents=True, exist_ok=True)
        try:
            response = store.regenerate(
                case.design,
                quality=args.quality,
                constraint_profile=args.constraint_profile,
            )
        except Exception as exc:
            if case.variable_name is None:
                raise RuntimeError("WP19-16 reference case failed") from exc
            expected_rule = EXPECTED_HARD_FAILURES.get(case.name)
            expected = bool(
                expected_rule
                and type(exc).__name__ == expected_rule["failure_type"]
                and expected_rule["message"] in str(exc)
            )
            failure = {
                "case": case.name,
                "variable": case.variable_name,
                "boundary": case.boundary,
                "status": (
                    "expected_numerical_hard_failure"
                    if expected
                    else "unexpected_generation_failure"
                ),
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "expected_failure": expected,
                "expected_failure_rule": expected_rule if expected else None,
                "passed": expected,
            }
            _write_json(case_root / "geometry19.json", case.design.as_mapping())
            _write_json(case_root / "case_result.json", failure)
            records.append(failure)
            label = "EXPECTED HARD FAILURE" if expected else "FAILED"
            print(f"  {label}: {type(exc).__name__}: {exc}", flush=True)
            continue
        source_components = storage_root / response["design_id"] / args.quality
        hashes = component_hashes(source_components)
        if reference_response is None:
            reference_response, reference_hashes = response, hashes

        component_root = case_root / "component_stl"
        _write_json(case_root / "geometry19.json", response["geometry19"])
        _write_json(case_root / "derived_geometry.json", response["derived_geometry"])
        _write_json(case_root / "mesh_diagnostics.json", response["mesh_diagnostics"])
        _write_json(
            case_root / "physics_result.json",
            {
                "physics": response["physics"],
                "constraints": response["constraints"],
                "solver": response["solver"],
                "provenance": {
                    name: response.get(name)
                    for name in (
                        "design_id",
                        "geometry_hash",
                        "lattice_hash",
                        "schema_hash",
                        "calibration_hash",
                        "constraint_profile",
                        "mesh_quality",
                    )
                },
            },
        )
        _link_components(source_components, component_root)
        if not args.skip_images:
            render_front_view(component_root, case_root / "front_view.png")
            render_section_view(component_root, case_root / "section_view.png")

        channel = evaluate_channel(
            case,
            variables.get(case.variable_name),
            response,
            hashes,
            reference_response,
            reference_hashes,
        )
        integer_checks = _integer_measurement_checks(case, response)
        case_passed = channel["passed"] and all(integer_checks.values())
        record = {
            "case": case.name,
            "variable": case.variable_name,
            "boundary": case.boundary,
            "design_id": response["design_id"],
            "geometry_hash": response["geometry_hash"],
            "lattice_hash": response["lattice_hash"],
            "schema_hash": response["schema_hash"],
            "calibration_hash": response["calibration_hash"],
            "constraint_profile": response["constraint_profile"],
            "solver_converged": response["solver"]["converged"],
            "engineering_feasible": response["constraints"]["engineering_feasible"],
            "violated_constraints": response["constraints"]["violated_constraints"],
            "channel_audit": channel,
            "integer_measurement_checks": integer_checks,
            "passed": case_passed,
        }
        _write_json(case_root / "case_result.json", record)
        records.append(record)

    summary = {
        "work_package": "WP19-16",
        "schema_version": schema["schema_version"],
        "quality": args.quality,
        "constraint_profile": args.constraint_profile,
        "case_count": len(records),
        "expected_full_case_count": 39,
        "complete_matrix": len(records) == 39,
        "generated_case_count": sum("design_id" in record for record in records),
        "expected_hard_failure_count": sum(
            record.get("status") == "expected_numerical_hard_failure"
            for record in records
        ),
        "expected_hard_failure_cases": [
            record["case"]
            for record in records
            if record.get("status") == "expected_numerical_hard_failure"
        ],
        "passed_count": sum(record["passed"] for record in records),
        "failed_cases": [record["case"] for record in records if not record["passed"]],
        "passed": all(record["passed"] for record in records),
        "cases": records,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--quality", choices=("preview_mesh", "export_mesh"), default="preview_mesh")
    parser.add_argument(
        "--constraint-profile",
        choices=("pipeline_debug_v1", "engineering_candidate_v1"),
        default="pipeline_debug_v1",
    )
    parser.add_argument("--case", action="append", default=[], help="Run one named case; repeatable")
    parser.add_argument("--skip-images", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args)
    print(
        f"WP19-16: {summary['passed_count']}/{summary['case_count']} passed; "
        f"summary={args.output_root.resolve() / 'summary.json'}",
        flush=True,
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

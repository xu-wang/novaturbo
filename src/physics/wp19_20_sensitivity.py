"""Pure analysis helpers for the WP19-20 19D sensitivity audit."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Iterable, Mapping

import numpy as np

from src.physics.geometry_state import Geometry19


SOLID_COMPONENTS = {
    "inlet", "compressor", "combustor", "turbine", "nozzle",
    "shaft", "casing", "transitions",
}
LATTICE_COMPONENTS = {"combustor_lattice", "nozzle_lattice"}


def build_positive_oat_cases(
    schema: Mapping[str, Any], fraction_of_range: float = 0.05
) -> list[tuple[Mapping[str, Any] | None, Geometry19]]:
    """Return reference plus one deterministic positive perturbation per axis."""
    variables = schema["design_variables"]
    reference = Geometry19.from_mapping(
        {item["name"]: item["reference_value"] for item in variables}
    )
    cases: list[tuple[Mapping[str, Any] | None, Geometry19]] = [(None, reference)]
    for variable in variables:
        low, high = variable["bounds"]
        if variable["dtype"] == "int64":
            value = min(int(high), int(variable["reference_value"]) + 1)
        else:
            value = min(
                float(high),
                float(variable["reference_value"])
                + fraction_of_range * (float(high) - float(low)),
            )
        cases.append((variable, replace(reference, **{variable["name"]: value})))
    return cases


def _changed_numeric_paths(
    baseline: Any, changed: Any, prefix: str = ""
) -> list[str]:
    if isinstance(baseline, Mapping) and isinstance(changed, Mapping):
        paths = []
        for key in sorted(set(baseline) & set(changed)):
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_changed_numeric_paths(baseline[key], changed[key], child))
        return paths
    if (
        isinstance(baseline, (int, float))
        and not isinstance(baseline, bool)
        and isinstance(changed, (int, float))
        and not isinstance(changed, bool)
        and math.isfinite(float(baseline))
        and math.isfinite(float(changed))
        and not math.isclose(float(baseline), float(changed), rel_tol=1.0e-10, abs_tol=1.0e-12)
    ):
        return [prefix]
    return []


def evaluate_oat_channel(
    variable: Mapping[str, Any],
    response: Mapping[str, Any],
    hashes: Mapping[str, str],
    reference_response: Mapping[str, Any],
    reference_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Evaluate mesh/physics/overlay response for one positive OAT case."""
    changed_components = sorted(
        name for name in set(hashes) | set(reference_hashes)
        if hashes.get(name) != reference_hashes.get(name)
    )
    physics_paths = _changed_numeric_paths(
        reference_response["physics"], response["physics"], "physics"
    )
    component = str(variable["component"])
    name = str(variable["name"])
    direction_checks = {
        "request_value_increased": response["geometry19"][name]
        > reference_response["geometry19"][name],
        "derived_design_value_increased": response["derived_geometry"]["design"][name]
        > reference_response["derived_geometry"]["design"][name],
    }
    if component == "operating_point":
        metric = (
            "simulation.thermal.t_max"
            if name == "turbine_inlet_total_temperature_K"
            else "simulation.stress.s_max"
        )
        root, field, key = metric.split(".")
        overlay_increased = response[root][field][key] > reference_response[root][field][key]
        checks = {
            **direction_checks,
            "solid_geometry_hash_unchanged": response["geometry_hash"] == reference_response["geometry_hash"],
            "lattice_hash_unchanged": response["lattice_hash"] == reference_response["lattice_hash"],
            "component_stl_unchanged": not changed_components,
            "overlay_parameter_changed": response["overlay_parameters"][name] != reference_response["overlay_parameters"][name],
            "expected_overlay_metric_increased": overlay_increased,
            "physics_channel_changed": bool(physics_paths),
        }
        channel = "overlay"
    elif component == "lattice":
        checks = {
            **direction_checks,
            "solid_geometry_hash_unchanged": response["geometry_hash"] == reference_response["geometry_hash"],
            "lattice_hash_changed": response["lattice_hash"] != reference_response["lattice_hash"],
            "solid_components_unchanged": not (set(changed_components) & SOLID_COMPONENTS),
            "lattice_component_changed": bool(set(changed_components) & LATTICE_COMPONENTS),
            "physics_channel_changed": bool(physics_paths),
        }
        if name == "lattice_cell_size_m":
            checks["cells_across_region_decreased"] = (
                response["physics"]["formal_diagnostics"]["lattice_cells_across_region"]
                < reference_response["physics"]["formal_diagnostics"]["lattice_cells_across_region"]
            )
        else:
            checks["mass_increased"] = (
                response["physics"]["performance"]["mass_kg"]
                > reference_response["physics"]["performance"]["mass_kg"]
            )
            checks["lattice_strength_ratio_increased"] = (
                response["physics"]["formal_diagnostics"]["lattice_strength_ratio"]
                > reference_response["physics"]["formal_diagnostics"]["lattice_strength_ratio"]
            )
        channel = "lattice"
    else:
        checks = {
            **direction_checks,
            "solid_geometry_hash_changed": response["geometry_hash"] != reference_response["geometry_hash"],
            "target_component_changed": component in changed_components,
            "physics_channel_changed": bool(physics_paths),
        }
        if variable["dtype"] == "int64":
            checks["discrete_mesh_count_exact"] = (
                response["mesh_diagnostics"]["geometry_measurements"].get(name)
                == response["geometry19"][name]
            )
        channel = "solid"
    checks["mesh_diagnostics_passed"] = bool(response["mesh_diagnostics"]["passed"])
    return {
        "channel": channel,
        "changed_components": changed_components,
        "changed_physics_paths": physics_paths,
        "checks": checks,
        "passed": all(checks.values()),
    }


def spearman_sensitivity(
    records: Iterable[Mapping[str, Any]], variable_names: Iterable[str]
) -> dict[str, Any]:
    """Compute rank correlations on converged records without SciPy dependency."""
    converged = [record for record in records if record["physics"]["solver"]["converged"]]
    outputs = {
        "performance.thrust_N": lambda r: r["physics"]["performance"]["thrust_N"],
        "performance.tsfc_kg_N_s": lambda r: r["physics"]["performance"]["tsfc_kg_N_s"],
        "performance.mass_kg": lambda r: r["physics"]["performance"]["mass_kg"],
    }
    constraint_names = sorted({
        name for record in converged for name in record["physics"].get("constraints", {})
    })
    outputs.update({
        f"constraints.{name}": (lambda r, key=name: r["physics"]["constraints"].get(key))
        for name in constraint_names
    })

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranked = np.empty(len(values), dtype=float)
        ranked[order] = np.arange(len(values), dtype=float)
        unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        if len(unique) != len(values):
            sums = np.bincount(inverse, weights=ranked)
            ranked = (sums / counts)[inverse]
        return ranked

    result = {}
    for name in variable_names:
        x = np.asarray([record["geometry19"][name] for record in converged], dtype=float)
        correlations = {}
        for output_name, getter in outputs.items():
            pairs = [(x[index], getter(record)) for index, record in enumerate(converged)]
            pairs = [(a, float(b)) for a, b in pairs if b is not None and math.isfinite(float(b))]
            if len(pairs) < 3:
                continue
            xa, ya = (np.asarray(values, dtype=float) for values in zip(*pairs))
            if np.ptp(xa) == 0.0 or np.ptp(ya) == 0.0:
                continue
            rho = float(np.corrcoef(ranks(xa), ranks(ya))[0, 1])
            correlations[output_name] = rho
        ordered = sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)
        result[name] = {
            "nonconstant_channel_count": len(correlations),
            "max_abs_spearman": abs(ordered[0][1]) if ordered else None,
            "top_channels": [{"channel": key, "rho": value} for key, value in ordered[:8]],
        }
    return {"converged_sample_count": len(converged), "variables": result}

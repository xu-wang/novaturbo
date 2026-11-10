"""Locked V2.2-C2 variable-property combustor shadow campaign."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Mapping

import numpy as np
import yaml

from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.physics.engine_match import Environment, Geometry19
from src.physics.maps import ComponentMap
from src.physics.maps.boundary_probe import probe_axis_boundaries
from src.physics.maps.validation import validate_component_map_package


class V22C2CombustorValidationError(ValueError):
    """Raised when the locked C2 campaign contract is invalid."""


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise V22C2CombustorValidationError(
            f"Cannot load validation config: {path}"
        ) from exc
    required = {
        "validation_id",
        "baseline_profile",
        "enhanced_profile",
        "expected_combustor_map_id",
        "dense_map_scan",
        "boundary_classification",
        "root_sweep",
        "runtime_benchmark",
        "engineering_blockers",
        "artifacts",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise V22C2CombustorValidationError("C2 validation config keys mismatch")
    blockers = payload["engineering_blockers"]
    if not isinstance(blockers, list) or not blockers or not all(
        isinstance(item, str) and item for item in blockers
    ):
        raise V22C2CombustorValidationError("Engineering blockers must be explicit")
    return payload


def _reference_design(context: Any) -> Geometry19:
    return Geometry19.from_mapping(
        {
            item["name"]: item["reference_value"]
            for item in context.physics_runtime.design_schema["design_variables"]
        }
    )


def _maximum_jump(surface: np.ndarray) -> float:
    return float(
        max(
            np.max(np.abs(np.diff(surface, axis=0))),
            np.max(np.abs(np.diff(surface, axis=1))),
        )
    )


def _dense_map_scan(
    component_map: ComponentMap, config: Mapping[str, Any]
) -> dict[str, Any]:
    flux_count = int(config["mass_flux_ratio_samples"])
    phi_count = int(config["equivalence_ratio_samples"])
    efficiency_limit = float(config["max_adjacent_efficiency_jump"])
    pressure_limit = float(config["max_adjacent_pressure_loss_jump"])
    if min(flux_count, phi_count) < 3 or min(efficiency_limit, pressure_limit) <= 0:
        raise V22C2CombustorValidationError("Dense map scan contract is invalid")
    flux_axis = component_map.coordinates["mass_flux_ratio"]
    phi_axis = component_map.coordinates["equivalence_ratio"]
    fluxes = np.linspace(float(flux_axis[0]), float(flux_axis[-1]), flux_count)
    phis = np.linspace(float(phi_axis[0]), float(phi_axis[-1]), phi_count)
    efficiency = np.empty((flux_count, phi_count))
    pressure_loss = np.empty_like(efficiency)
    for i, flux in enumerate(fluxes):
        for j, phi in enumerate(phis):
            result = component_map.evaluate(
                {"mass_flux_ratio": float(flux), "equivalence_ratio": float(phi)}
            )
            efficiency[i, j] = result.values["combustion_efficiency"]
            pressure_loss[i, j] = result.values["total_pressure_loss_fraction"]
    efficiency_jump = _maximum_jump(efficiency)
    pressure_jump = _maximum_jump(pressure_loss)
    physical = bool(
        np.all(efficiency > 0.0)
        and np.all(efficiency <= 1.0)
        and np.all(pressure_loss >= 0.0)
        and np.all(pressure_loss < 1.0)
    )
    return {
        "sample_shape": [flux_count, phi_count],
        "maximum_adjacent_efficiency_jump": efficiency_jump,
        "maximum_adjacent_pressure_loss_jump": pressure_jump,
        "maximum_allowed_efficiency_jump": efficiency_limit,
        "maximum_allowed_pressure_loss_jump": pressure_limit,
        "outputs_physical": physical,
        "pass": (
            physical
            and efficiency_jump <= efficiency_limit
            and pressure_jump <= pressure_limit
        ),
    }


def _boundary_classification(
    component_map: ComponentMap, config: Mapping[str, Any]
) -> dict[str, Any]:
    fraction = float(config["probe_fraction_of_stable_span"])
    if not 0.0 < fraction < 0.1:
        raise V22C2CombustorValidationError("Boundary probe fraction is invalid")
    cases = []
    for flux in component_map.coordinates["mass_flux_ratio"]:
        probe = probe_axis_boundaries(
            component_map,
            fixed_coordinates={"mass_flux_ratio": float(flux)},
            axis_name="equivalence_ratio",
            center_value=0.35,
            minimum_margin_name="minimum_equivalence_ratio",
            maximum_margin_name="maximum_equivalence_ratio",
            fraction=fraction,
        )
        cases.append({"mass_flux_ratio": float(flux), **probe})
    return {"cases": cases, "pass": all(case["pass"] for case in cases)}


def _root_sweep(
    baseline: Any,
    enhanced: Any,
    design: Geometry19,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ratios = tuple(float(value) for value in config["shaft_speed_ratios"])
    jump_limit = float(config["maximum_consecutive_mass_flow_relative_jump"])
    energy_tolerance = float(config["energy_absolute_tolerance_W"])
    if len(ratios) < 3 or min(ratios) <= 0.0 or jump_limit <= 0.0:
        raise V22C2CombustorValidationError("Root sweep contract is invalid")
    records = []
    enhanced_flows = []
    for ratio in ratios:
        trial = replace(design, shaft_speed_rpm=design.shaft_speed_rpm * ratio)
        for context in (baseline, enhanced):
            result = context.physics_runtime.evaluate(trial, Environment())
            combustor = result.components.get("combustor", {})
            record = {
                "profile": context.profile.name,
                "shaft_speed_ratio": ratio,
                "converged": result.solver.converged,
                "flow_root_count": int(result.operating_point.get("flow_root_count", 0)),
                "selected_mass_flow_air_kg_s": result.operating_point.get(
                    "selected_flow_root_mass_flow_air_kg_s"
                ),
                "fuel_air_ratio": result.operating_point.get("fuel_air_ratio"),
                "energy_residual_W": result.conservation.combustor_energy_W,
                "map_in_domain": combustor.get("map_in_domain"),
                "fuel_solution_iterations": combustor.get("fuel_solution_iterations"),
                "lean_stability_margin": combustor.get("lean_stability_margin"),
                "rich_stability_margin": combustor.get("rich_stability_margin"),
                "pressure_decreases": bool(
                    result.solver.converged
                    and result.stations["t4"].total_pressure_Pa
                    < result.stations["t3"].total_pressure_Pa
                ),
            }
            records.append(record)
            if context is enhanced and result.solver.converged:
                enhanced_flows.append(float(record["selected_mass_flow_air_kg_s"]))
    transitions = [
        abs(right - left) / max(left, 1.0e-12)
        for left, right in zip(enhanced_flows[:-1], enhanced_flows[1:], strict=True)
    ]
    enhanced_records = [
        record for record in records if record["profile"] == enhanced.profile.name
    ]
    closure_pass = all(
        record["converged"]
        and record["flow_root_count"] == 1
        and record["map_in_domain"] is True
        and int(record["fuel_solution_iterations"]) > 0
        and float(record["fuel_air_ratio"]) > 0.0
        and abs(float(record["energy_residual_W"])) <= energy_tolerance
        and float(record["lean_stability_margin"]) > 0.0
        and float(record["rich_stability_margin"]) > 0.0
        and record["pressure_decreases"]
        for record in enhanced_records
    )
    trace_pass = (
        len(enhanced_flows) == len(ratios)
        and enhanced_flows == sorted(enhanced_flows)
        and all(value <= jump_limit for value in transitions)
    )
    return {
        "records": records,
        "trace_complete": len(records) == 2 * len(ratios),
        "maximum_mass_flow_relative_jump": max(transitions, default=math.inf),
        "maximum_allowed_mass_flow_relative_jump": jump_limit,
        "local_far_energy_and_pressure_pass": closure_pass,
        "root_trace_pass": trace_pass,
        "pass": closure_pass and trace_pass,
    }


def _runtime_benchmark(
    baseline: Any,
    enhanced: Any,
    design: Geometry19,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ratios = tuple(float(value) for value in config["shaft_speed_ratios"])
    warmups = int(config["warmup_repetitions"])
    repetitions = int(config["timed_repetitions"])
    p50_limit = float(config["max_p50_enhanced_to_baseline_ratio"])
    p95_limit = float(config["max_p95_enhanced_to_baseline_ratio"])
    if len(ratios) < 3 or warmups < 1 or repetitions < 3:
        raise V22C2CombustorValidationError("Runtime benchmark contract is invalid")
    designs = tuple(
        replace(design, shaft_speed_rpm=design.shaft_speed_rpm * ratio)
        for ratio in ratios
    )
    contexts = (baseline, enhanced)
    for context in contexts:
        for _ in range(warmups):
            for trial in designs:
                context.physics_runtime.evaluate(trial, Environment())
    timings = {context.profile.name: [] for context in contexts}
    for repetition in range(repetitions):
        ordered_contexts = contexts if repetition % 2 == 0 else tuple(reversed(contexts))
        ordered_designs = designs if repetition % 2 == 0 else tuple(reversed(designs))
        for trial in ordered_designs:
            for context in ordered_contexts:
                start = time.perf_counter_ns()
                context.physics_runtime.evaluate(trial, Environment())
                timings[context.profile.name].append(
                    (time.perf_counter_ns() - start) / 1.0e6
                )
    percentiles = {
        name: {
            "p50_ms": statistics.median(values),
            "p95_ms": float(np.percentile(np.asarray(values), 95.0)),
        }
        for name, values in timings.items()
    }
    baseline_times = percentiles[baseline.profile.name]
    enhanced_times = percentiles[enhanced.profile.name]
    p50_ratio = enhanced_times["p50_ms"] / baseline_times["p50_ms"]
    p95_ratio = enhanced_times["p95_ms"] / baseline_times["p95_ms"]
    return {
        "timed_evaluation_count_per_profile": repetitions * len(designs),
        "percentile_milliseconds": percentiles,
        "p50_enhanced_to_baseline_ratio": p50_ratio,
        "p95_enhanced_to_baseline_ratio": p95_ratio,
        "maximum_p50_ratio": p50_limit,
        "maximum_p95_ratio": p95_limit,
        "pass": p50_ratio <= p50_limit and p95_ratio <= p95_limit,
    }


def run_v2_2_c2_combustor_validation(
    repository_root: str | Path,
    *,
    config_path: str | Path = "config/physics/wp_v2_2_c2_combustor_validation_v1.yaml",
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved_config = (root / config_path).resolve()
    config = _load_config(resolved_config)
    baseline = build_pipeline_context(root, pipeline_profile=config["baseline_profile"])
    enhanced = build_pipeline_context(root, pipeline_profile=config["enhanced_profile"])
    maps = enhanced.physics_runtime.component_maps
    if set(maps) != {"compressor", "turbine", "inlet", "combustor"}:
        raise V22C2CombustorValidationError("Enhanced profile map selection mismatch")
    combustor_map = maps["combustor"]
    if combustor_map.map_id != config["expected_combustor_map_id"]:
        raise V22C2CombustorValidationError("Unexpected combustor map selection")
    design = _reference_design(enhanced)
    evidence = validate_component_map_package(
        enhanced.physics_runtime.source_paths["combustor_map"]
    )
    dense = _dense_map_scan(combustor_map, config["dense_map_scan"])
    boundaries = _boundary_classification(
        combustor_map, config["boundary_classification"]
    )
    roots = _root_sweep(baseline, enhanced, design, config["root_sweep"])
    runtime = _runtime_benchmark(
        baseline, enhanced, design, config["runtime_benchmark"]
    )
    gates = {
        "synthetic_map_structural_shadow_ready": bool(evidence["shadow_ready"]),
        "dense_map_continuity": bool(dense["pass"]),
        "stability_boundary_classification": bool(boundaries["pass"]),
        "local_far_energy_and_root_trace": bool(roots["pass"]),
        "p50_p95_runtime_ratio": bool(runtime["pass"]),
    }
    return {
        "validation_id": config["validation_id"],
        "config_sha256": sha256(resolved_config.read_bytes()).hexdigest(),
        "baseline_context": baseline.contract_mapping(),
        "enhanced_context": enhanced.contract_mapping(),
        "map_validation": evidence,
        "dense_map_scan": dense,
        "boundary_classification": boundaries,
        "root_sweep": roots,
        "runtime_benchmark": runtime,
        "software_gates": gates,
        "software_gate_pass": all(gates.values()),
        "engineering_ready": False,
        "engineering_blockers": list(config["engineering_blockers"]),
    }


def render_v2_2_c2_combustor_report(report: Mapping[str, Any]) -> str:
    runtime = report["runtime_benchmark"]
    lines = [
        f"# {report['validation_id']}",
        "",
        f"- software gate: **{'PASS' if report['software_gate_pass'] else 'FAIL'}**",
        f"- engineering ready: **{str(report['engineering_ready']).lower()}**",
        f"- P50 runtime ratio: `{runtime['p50_enhanced_to_baseline_ratio']:.3f}x`",
        f"- P95 runtime ratio: `{runtime['p95_enhanced_to_baseline_ratio']:.3f}x`",
        "",
        "## Software gates",
        "",
    ]
    lines.extend(
        f"- [{'x' if passed else ' '}] {name}"
        for name, passed in report["software_gates"].items()
    )
    lines.extend(["", "## Engineering blockers", ""])
    lines.extend(f"- {item}" for item in report["engineering_blockers"])
    return "\n".join(lines) + "\n"


def write_v2_2_c2_combustor_artifacts(
    report: Mapping[str, Any], output_root: str | Path
) -> None:
    output = Path(output_root).resolve()
    if output.exists():
        raise V22C2CombustorValidationError(
            f"Refusing to overwrite validation output: {output}"
        )
    output.mkdir(parents=True)
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "Validation_Report.md").write_text(
        render_v2_2_c2_combustor_report(report), encoding="utf-8"
    )

"""Locked V2.2-C3 variable-property convergent-nozzle campaign."""

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
from src.physics.combustor_map_enhanced import solve_combustor_map_enhanced
from src.physics.compressor_map_enhanced import solve_compressor_map_enhanced
from src.physics.engine_match import Environment, Geometry19
from src.physics.geometry_state import derive_geometry_state
from src.physics.inlet_map_enhanced import solve_inlet_map_enhanced
from src.physics.maps import ComponentMap
from src.physics.maps.boundary_probe import probe_axis_boundaries
from src.physics.maps.validation import validate_component_map_package
from src.physics.nozzle_map_enhanced import solve_nozzle_map_enhanced
from src.physics.turbine_map_enhanced import solve_turbine_map_enhanced


class V22C3NozzleValidationError(ValueError):
    """Raised when the locked C3 campaign contract is invalid."""


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise V22C3NozzleValidationError(
            f"Cannot load validation config: {path}"
        ) from exc
    required = {
        "validation_id",
        "baseline_profile",
        "enhanced_profile",
        "expected_nozzle_map_id",
        "dense_map_scan",
        "boundary_classification",
        "critical_transition",
        "root_sweep",
        "runtime_benchmark",
        "engineering_blockers",
        "artifacts",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise V22C3NozzleValidationError("C3 validation config keys mismatch")
    blockers = payload["engineering_blockers"]
    if not isinstance(blockers, list) or not blockers or not all(
        isinstance(item, str) and item for item in blockers
    ):
        raise V22C3NozzleValidationError("Engineering blockers must be explicit")
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
    pressure_count = int(config["inlet_pressure_ratio_samples"])
    reynolds_count = int(config["reynolds_ratio_samples"])
    recovery_limit = float(config["max_adjacent_recovery_jump"])
    discharge_limit = float(config["max_adjacent_discharge_jump"])
    if min(pressure_count, reynolds_count) < 3:
        raise V22C3NozzleValidationError("Dense map scan contract is invalid")
    pressure_axis = component_map.coordinates["inlet_pressure_ratio"]
    reynolds_axis = component_map.coordinates["reynolds_ratio"]
    pressures = np.linspace(
        float(pressure_axis[0]), float(pressure_axis[-1]), pressure_count
    )
    reynolds_values = np.linspace(
        float(reynolds_axis[0]), float(reynolds_axis[-1]), reynolds_count
    )
    recovery = np.empty((pressure_count, reynolds_count))
    discharge = np.empty_like(recovery)
    for i, pressure in enumerate(pressures):
        for j, reynolds in enumerate(reynolds_values):
            result = component_map.evaluate(
                {
                    "inlet_pressure_ratio": float(pressure),
                    "reynolds_ratio": float(reynolds),
                }
            )
            recovery[i, j] = result.values["total_pressure_recovery"]
            discharge[i, j] = result.values["discharge_coefficient"]
    recovery_jump = _maximum_jump(recovery)
    discharge_jump = _maximum_jump(discharge)
    physical = bool(
        np.all(recovery > 0.0)
        and np.all(recovery <= 1.0)
        and np.all(discharge > 0.0)
        and np.all(discharge <= 1.0)
    )
    return {
        "sample_shape": [pressure_count, reynolds_count],
        "maximum_adjacent_recovery_jump": recovery_jump,
        "maximum_adjacent_discharge_jump": discharge_jump,
        "maximum_allowed_recovery_jump": recovery_limit,
        "maximum_allowed_discharge_jump": discharge_limit,
        "outputs_physical": physical,
        "pass": (
            physical
            and recovery_jump <= recovery_limit
            and discharge_jump <= discharge_limit
        ),
    }


def _boundary_classification(
    component_map: ComponentMap, config: Mapping[str, Any]
) -> dict[str, Any]:
    fraction = float(config["probe_fraction_of_reynolds_span"])
    if not 0.0 < fraction < 0.1:
        raise V22C3NozzleValidationError("Boundary probe fraction is invalid")
    cases = []
    for pressure in component_map.coordinates["inlet_pressure_ratio"]:
        probe = probe_axis_boundaries(
            component_map,
            fixed_coordinates={"inlet_pressure_ratio": float(pressure)},
            axis_name="reynolds_ratio",
            center_value=1.0,
            minimum_margin_name="minimum_reynolds_ratio",
            maximum_margin_name="maximum_reynolds_ratio",
            fraction=fraction,
        )
        cases.append(
            {
                "inlet_pressure_ratio": float(pressure),
                **probe,
            }
        )
    return {"cases": cases, "pass": all(case["pass"] for case in cases)}


def _reference_turbine(context: Any, design: Geometry19, mass_flow_air: float):
    runtime = context.physics_runtime
    geometry = derive_geometry_state(design, runtime.model_config)
    inlet, _, _ = solve_inlet_map_enhanced(
        geometry,
        Environment(),
        runtime.model_config,
        component_map=runtime.component_maps["inlet"],
        thermo=runtime.thermo_model,
    )
    compressor = solve_compressor_map_enhanced(
        geometry,
        mass_flow_air,
        inlet,
        runtime.model_config,
        component_map=runtime.component_maps["compressor"],
        thermo=runtime.thermo_model,
    )
    combustor = solve_combustor_map_enhanced(
        geometry,
        mass_flow_air,
        compressor.outlet,
        runtime.model_config,
        component_map=runtime.component_maps["combustor"],
        thermo=runtime.thermo_model,
    )
    turbine = solve_turbine_map_enhanced(
        geometry,
        combustor,
        compressor.shaft_power_W,
        runtime.model_config,
        component_map=runtime.component_maps["turbine"],
    )
    return geometry, turbine


def _critical_transition(
    context: Any,
    design: Geometry19,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    minimum = float(config["ambient_pressure_min_Pa"])
    maximum = float(config["ambient_pressure_max_Pa"])
    count = int(config["pressure_samples"])
    jump_limit = float(config["max_branch_transition_capacity_relative_jump"])
    enthalpy_tolerance = float(config["enthalpy_absolute_tolerance_J_kg"])
    if not 0.0 < minimum < maximum or count < 5:
        raise V22C3NozzleValidationError("Critical-transition contract is invalid")
    matched = context.physics_runtime.evaluate(design, Environment())
    geometry, turbine = _reference_turbine(
        context, design, float(matched.operating_point["mass_flow_air_kg_s"])
    )
    records = []
    for pressure in np.linspace(minimum, maximum, count):
        result = solve_nozzle_map_enhanced(
            geometry,
            turbine,
            Environment(pressure_Pa=float(pressure)),
            context.physics_runtime.model_config,
            component_map=context.physics_runtime.component_maps["nozzle"],
            thermo=context.physics_runtime.thermo_model,
        )
        records.append(
            {
                "ambient_pressure_Pa": float(pressure),
                "choked": result.choked,
                "exit_mach": result.exit.mach,
                "capacity_kg_s": result.mass_flow_capacity_kg_s,
                "pressure_thrust_N": result.pressure_thrust_N,
                "enthalpy_residual_J_kg": result.exit_enthalpy_residual_J_kg,
            }
        )
    jumps = [
        abs(right["capacity_kg_s"] - left["capacity_kg_s"])
        / max(left["capacity_kg_s"], 1.0e-12)
        for left, right in zip(records[:-1], records[1:], strict=True)
    ]
    transition_indices = [
        index
        for index, (left, right) in enumerate(
            zip(records[:-1], records[1:], strict=True)
        )
        if left["choked"] != right["choked"]
    ]
    branch_changes = len(transition_indices)
    transition_jumps = [jumps[index] for index in transition_indices]
    physical = all(
        0.0 < record["exit_mach"] <= 1.0 + 1.0e-8
        and abs(record["enthalpy_residual_J_kg"]) <= enthalpy_tolerance
        and record["pressure_thrust_N"] >= 0.0
        for record in records
    )
    return {
        "sample_count": count,
        "choked_count": sum(record["choked"] for record in records),
        "unchoked_count": sum(not record["choked"] for record in records),
        "branch_change_count": branch_changes,
        "maximum_adjacent_capacity_relative_jump": max(jumps),
        "maximum_branch_transition_capacity_relative_jump": max(
            transition_jumps, default=math.inf
        ),
        "maximum_allowed_branch_transition_capacity_relative_jump": jump_limit,
        "physical_state_and_enthalpy_pass": physical,
        "pass": (
            physical
            and branch_changes == 1
            and any(record["choked"] for record in records)
            and any(not record["choked"] for record in records)
            and max(transition_jumps, default=math.inf) <= jump_limit
        ),
    }


def _root_sweep(
    baseline: Any,
    enhanced: Any,
    design: Geometry19,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ratios = tuple(float(value) for value in config["shaft_speed_ratios"])
    jump_limit = float(config["maximum_consecutive_mass_flow_relative_jump"])
    records = []
    enhanced_flows = []
    for ratio in ratios:
        trial = replace(design, shaft_speed_rpm=design.shaft_speed_rpm * ratio)
        for context in (baseline, enhanced):
            result = context.physics_runtime.evaluate(trial, Environment())
            nozzle = result.components.get("nozzle", {})
            record = {
                "profile": context.profile.name,
                "shaft_speed_ratio": ratio,
                "converged": result.solver.converged,
                "flow_root_count": int(result.operating_point.get("flow_root_count", 0)),
                "selected_mass_flow_air_kg_s": result.operating_point.get(
                    "selected_flow_root_mass_flow_air_kg_s"
                ),
                "map_in_domain": nozzle.get("map_in_domain"),
                "enthalpy_residual_J_kg": nozzle.get("exit_enthalpy_residual_J_kg"),
            }
            records.append(record)
            if context is enhanced and result.solver.converged:
                enhanced_flows.append(float(record["selected_mass_flow_air_kg_s"]))
    jumps = [
        abs(right - left) / max(left, 1.0e-12)
        for left, right in zip(enhanced_flows[:-1], enhanced_flows[1:], strict=True)
    ]
    enhanced_records = [
        record for record in records if record["profile"] == enhanced.profile.name
    ]
    state_pass = all(
        record["converged"]
        and record["flow_root_count"] == 1
        and record["map_in_domain"] is True
        and abs(float(record["enthalpy_residual_J_kg"])) <= 1.0e-8
        for record in enhanced_records
    )
    trace_pass = (
        len(enhanced_flows) == len(ratios)
        and enhanced_flows == sorted(enhanced_flows)
        and all(jump <= jump_limit for jump in jumps)
    )
    return {
        "records": records,
        "trace_complete": len(records) == 2 * len(ratios),
        "maximum_mass_flow_relative_jump": max(jumps, default=math.inf),
        "state_pass": state_pass,
        "root_trace_pass": trace_pass,
        "pass": state_pass and trace_pass,
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


def run_v2_2_c3_nozzle_validation(
    repository_root: str | Path,
    *,
    config_path: str | Path = "config/physics/wp_v2_2_c3_nozzle_validation_v1.yaml",
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved_config = (root / config_path).resolve()
    config = _load_config(resolved_config)
    baseline = build_pipeline_context(root, pipeline_profile=config["baseline_profile"])
    enhanced = build_pipeline_context(root, pipeline_profile=config["enhanced_profile"])
    maps = enhanced.physics_runtime.component_maps
    if set(maps) != {"compressor", "turbine", "inlet", "combustor", "nozzle"}:
        raise V22C3NozzleValidationError("Enhanced profile map selection mismatch")
    nozzle_map = maps["nozzle"]
    if nozzle_map.map_id != config["expected_nozzle_map_id"]:
        raise V22C3NozzleValidationError("Unexpected nozzle map selection")
    design = _reference_design(enhanced)
    evidence = validate_component_map_package(
        enhanced.physics_runtime.source_paths["nozzle_map"]
    )
    dense = _dense_map_scan(nozzle_map, config["dense_map_scan"])
    boundaries = _boundary_classification(
        nozzle_map, config["boundary_classification"]
    )
    transition = _critical_transition(enhanced, design, config["critical_transition"])
    roots = _root_sweep(baseline, enhanced, design, config["root_sweep"])
    runtime = _runtime_benchmark(
        baseline, enhanced, design, config["runtime_benchmark"]
    )
    gates = {
        "synthetic_map_structural_shadow_ready": bool(evidence["shadow_ready"]),
        "dense_map_continuity": bool(dense["pass"]),
        "reynolds_boundary_classification": bool(boundaries["pass"]),
        "critical_transition_continuity": bool(transition["pass"]),
        "whole_engine_root_trace": bool(roots["pass"]),
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
        "critical_transition": transition,
        "root_sweep": roots,
        "runtime_benchmark": runtime,
        "software_gates": gates,
        "software_gate_pass": all(gates.values()),
        "engineering_ready": False,
        "engineering_blockers": list(config["engineering_blockers"]),
    }


def render_v2_2_c3_nozzle_report(report: Mapping[str, Any]) -> str:
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


def write_v2_2_c3_nozzle_artifacts(
    report: Mapping[str, Any], output_root: str | Path
) -> None:
    output = Path(output_root).resolve()
    if output.exists():
        raise V22C3NozzleValidationError(
            f"Refusing to overwrite validation output: {output}"
        )
    output.mkdir(parents=True)
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "Validation_Report.md").write_text(
        render_v2_2_c3_nozzle_report(report), encoding="utf-8"
    )

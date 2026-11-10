"""Locked V2.2-C4 turbine hot-side variable-property campaign."""

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
from src.physics.nozzle_map_enhanced import solve_nozzle_map_enhanced
from src.physics.turbine_map_enhanced import solve_turbine_map_enhanced


class V22C4ValidationError(ValueError):
    """Raised when the locked C4 validation contract is invalid."""


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise V22C4ValidationError(f"Cannot load validation config: {path}") from exc
    required = {
        "validation_id",
        "baseline_profile",
        "enhanced_profile",
        "expected_thermo_model_id",
        "c3_reference_isolation",
        "hot_side_closure",
        "root_sweep",
        "runtime_benchmark",
        "engineering_blockers",
        "artifacts",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise V22C4ValidationError("C4 validation config keys mismatch")
    blockers = payload["engineering_blockers"]
    if not isinstance(blockers, list) or not blockers or not all(
        isinstance(item, str) and item for item in blockers
    ):
        raise V22C4ValidationError("Engineering blockers must be explicit")
    return payload


def _reference_design(context: Any) -> Geometry19:
    return Geometry19.from_mapping(
        {
            item["name"]: item["reference_value"]
            for item in context.physics_runtime.design_schema["design_variables"]
        }
    )


def _c3_reference_isolation(
    baseline: Any, design: Geometry19, config: Mapping[str, Any]
) -> dict[str, Any]:
    result = baseline.physics_runtime.evaluate(design, Environment())
    tolerance = float(config["relative_tolerance"])
    actual = {
        "mass_flow_air_kg_s": result.operating_point.get("mass_flow_air_kg_s"),
        "thrust_N": result.performance.get("thrust_N"),
        "tsfc_kg_N_s": result.performance.get("tsfc_kg_N_s"),
    }
    expected = {
        name: float(config[name])
        for name in ("mass_flow_air_kg_s", "thrust_N", "tsfc_kg_N_s")
    }
    relative_errors = {
        name: abs(float(actual[name]) - target) / abs(target)
        for name, target in expected.items()
    }
    passed = (
        result.solver.converged
        and result.operating_point.get("flow_root_count") == 1
        and "thermo_model_id" not in result.components["turbine"]
        and all(error <= tolerance for error in relative_errors.values())
    )
    return {
        "actual": actual,
        "expected": expected,
        "relative_errors": relative_errors,
        "relative_tolerance": tolerance,
        "pass": passed,
    }


def _direct_hot_side_handoff(
    context: Any,
    design: Geometry19,
    closure: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = context.physics_runtime
    matched = runtime.evaluate(design, Environment())
    mass_flow = float(matched.operating_point["mass_flow_air_kg_s"])
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
        mass_flow,
        inlet,
        runtime.model_config,
        component_map=runtime.component_maps["compressor"],
        thermo=runtime.thermo_model,
    )
    combustor = solve_combustor_map_enhanced(
        geometry,
        mass_flow,
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
        thermo=runtime.thermo_model,
    )
    nozzle = solve_nozzle_map_enhanced(
        geometry,
        turbine,
        Environment(),
        runtime.model_config,
        component_map=runtime.component_maps["nozzle"],
        thermo=runtime.thermo_model,
    )
    enthalpy_limit = float(closure["enthalpy_absolute_tolerance_J_kg"])
    shaft_limit = float(closure["shaft_power_absolute_tolerance_W"])
    station_equal = nozzle.turbine_outlet == turbine.outlet
    thermo_equal = turbine.thermo_model_id == nozzle.thermo_model_id
    passed = (
        station_equal
        and thermo_equal
        and abs(float(turbine.outlet_enthalpy_residual_J_kg)) <= enthalpy_limit
        and abs(float(turbine.isentropic_efficiency_residual_J_kg)) <= enthalpy_limit
        and abs(nozzle.exit_enthalpy_residual_J_kg) <= enthalpy_limit
        and abs(turbine.shaft_power_residual_W) <= shaft_limit
        and turbine.outlet.total_temperature_K < turbine.inlet.total_temperature_K
        and turbine.outlet.total_pressure_Pa < turbine.inlet.total_pressure_Pa
    )
    return {
        "turbine_to_nozzle_station_equal": station_equal,
        "turbine_to_nozzle_thermo_model_equal": thermo_equal,
        "thermo_model_id": turbine.thermo_model_id,
        "turbine_outlet_enthalpy_residual_J_kg": (
            turbine.outlet_enthalpy_residual_J_kg
        ),
        "turbine_isentropic_efficiency_residual_J_kg": (
            turbine.isentropic_efficiency_residual_J_kg
        ),
        "nozzle_exit_enthalpy_residual_J_kg": nozzle.exit_enthalpy_residual_J_kg,
        "shaft_power_residual_W": turbine.shaft_power_residual_W,
        "pass": passed,
    }


def _root_sweep(
    baseline: Any,
    enhanced: Any,
    design: Geometry19,
    config: Mapping[str, Any],
    thermo_model_id: str,
) -> dict[str, Any]:
    ratios = tuple(float(value) for value in config["shaft_speed_ratios"])
    environments = tuple(Environment(**item) for item in config["environments"])
    jump_limit = float(config["maximum_consecutive_mass_flow_relative_jump"])
    records = []
    enhanced_flows_by_environment: list[list[float]] = []
    for environment_index, environment in enumerate(environments):
        enhanced_flows = []
        for ratio in ratios:
            trial = replace(design, shaft_speed_rpm=design.shaft_speed_rpm * ratio)
            for context in (baseline, enhanced):
                result = context.physics_runtime.evaluate(trial, environment)
                turbine = result.components.get("turbine", {})
                nozzle = result.components.get("nozzle", {})
                record = {
                    "profile": context.profile.name,
                    "environment_index": environment_index,
                    "shaft_speed_ratio": ratio,
                    "converged": result.solver.converged,
                    "root_count": int(result.operating_point.get("flow_root_count", 0)),
                    "mass_flow_air_kg_s": result.operating_point.get(
                        "mass_flow_air_kg_s"
                    ),
                    "turbine_map_in_domain": turbine.get("map_in_domain"),
                    "nozzle_map_in_domain": nozzle.get("map_in_domain"),
                    "thermo_model_id": turbine.get("thermo_model_id"),
                    "outlet_enthalpy_residual_J_kg": turbine.get(
                        "outlet_enthalpy_residual_J_kg"
                    ),
                    "isentropic_efficiency_residual_J_kg": turbine.get(
                        "isentropic_efficiency_residual_J_kg"
                    ),
                }
                records.append(record)
                if context is enhanced and result.solver.converged:
                    enhanced_flows.append(float(record["mass_flow_air_kg_s"]))
        enhanced_flows_by_environment.append(enhanced_flows)
    enhanced_records = [
        record for record in records if record["profile"] == enhanced.profile.name
    ]
    state_pass = all(
        record["converged"]
        and record["root_count"] == 1
        and record["turbine_map_in_domain"] is True
        and record["nozzle_map_in_domain"] is True
        and record["thermo_model_id"] == thermo_model_id
        and abs(float(record["outlet_enthalpy_residual_J_kg"])) <= 1.0e-8
        and abs(float(record["isentropic_efficiency_residual_J_kg"])) <= 1.0e-8
        for record in enhanced_records
    )
    jumps = [
        abs(right - left) / max(left, 1.0e-12)
        for flows in enhanced_flows_by_environment
        for left, right in zip(flows[:-1], flows[1:], strict=True)
    ]
    trace_pass = all(
        len(flows) == len(ratios) and flows == sorted(flows)
        for flows in enhanced_flows_by_environment
    ) and all(jump <= jump_limit for jump in jumps)
    return {
        "records": records,
        "environment_count": len(environments),
        "speed_count_per_environment": len(ratios),
        "enhanced_case_count": len(enhanced_records),
        "maximum_mass_flow_relative_jump": max(jumps, default=math.inf),
        "state_pass": state_pass,
        "root_trace_pass": trace_pass,
        "pass": state_pass and trace_pass,
    }


def _runtime_benchmark(
    baseline: Any, enhanced: Any, design: Geometry19, config: Mapping[str, Any]
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
        ordered = contexts if repetition % 2 == 0 else tuple(reversed(contexts))
        trials = designs if repetition % 2 == 0 else tuple(reversed(designs))
        for trial in trials:
            for context in ordered:
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


def run_v2_2_c4_validation(
    repository_root: str | Path,
    *,
    config_path: str | Path = (
        "config/physics/wp_v2_2_c4_turbine_thermo_validation_v1.yaml"
    ),
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved_config = (root / config_path).resolve()
    config = _load_config(resolved_config)
    baseline = build_pipeline_context(root, pipeline_profile=config["baseline_profile"])
    enhanced = build_pipeline_context(root, pipeline_profile=config["enhanced_profile"])
    design = _reference_design(enhanced)
    isolation = _c3_reference_isolation(
        baseline, design, config["c3_reference_isolation"]
    )
    handoff = _direct_hot_side_handoff(
        enhanced, design, config["hot_side_closure"]
    )
    roots = _root_sweep(
        baseline,
        enhanced,
        design,
        config["root_sweep"],
        str(config["expected_thermo_model_id"]),
    )
    runtime = _runtime_benchmark(
        baseline, enhanced, design, config["runtime_benchmark"]
    )
    gates = {
        "c3_fixed_property_path_isolated": bool(isolation["pass"]),
        "hot_side_enthalpy_entropy_closure": bool(handoff["pass"]),
        "speed_environment_root_trace": bool(roots["pass"]),
        "p50_p95_runtime_ratio": bool(runtime["pass"]),
    }
    return {
        "validation_id": config["validation_id"],
        "config_sha256": sha256(resolved_config.read_bytes()).hexdigest(),
        "baseline_context": baseline.contract_mapping(),
        "enhanced_context": enhanced.contract_mapping(),
        "c3_reference_isolation": isolation,
        "hot_side_handoff": handoff,
        "root_sweep": roots,
        "runtime_benchmark": runtime,
        "software_gates": gates,
        "software_gate_pass": all(gates.values()),
        "engineering_ready": False,
        "engineering_blockers": list(config["engineering_blockers"]),
    }


def render_v2_2_c4_report(report: Mapping[str, Any]) -> str:
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


def write_v2_2_c4_artifacts(
    report: Mapping[str, Any], output_root: str | Path
) -> None:
    output = Path(output_root).resolve()
    if output.exists():
        raise V22C4ValidationError(
            f"Refusing to overwrite validation output: {output}"
        )
    output.mkdir(parents=True)
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "Validation_Report.md").write_text(
        render_v2_2_c4_report(report), encoding="utf-8"
    )

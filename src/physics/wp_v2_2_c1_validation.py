"""Locked V2.2-C1 variable-property inlet shadow validation campaign."""

from __future__ import annotations

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


class V22C1InletValidationError(ValueError):
    """Raised when the locked V2.2-C1 campaign contract is invalid."""


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise V22C1InletValidationError(
            f"Cannot load validation config: {path}"
        ) from exc
    required = {
        "validation_id",
        "baseline_profile",
        "enhanced_profile",
        "expected_inlet_map_id",
        "dense_map_scan",
        "boundary_classification",
        "operating_sweep",
        "runtime_benchmark",
        "engineering_blockers",
        "artifacts",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise V22C1InletValidationError("V2.2-C1 validation config keys mismatch")
    blockers = payload["engineering_blockers"]
    if not isinstance(blockers, list) or not blockers or not all(
        isinstance(item, str) and item for item in blockers
    ):
        raise V22C1InletValidationError("Engineering blockers must be explicit")
    return payload


def _reference_design(context: Any) -> Geometry19:
    return Geometry19.from_mapping(
        {
            item["name"]: item["reference_value"]
            for item in context.physics_runtime.design_schema["design_variables"]
        }
    )


def _maximum_adjacent_jump(surface: np.ndarray) -> float:
    return float(
        max(
            np.max(np.abs(np.diff(surface, axis=0))),
            np.max(np.abs(np.diff(surface, axis=1))),
        )
    )


def _dense_map_scan(
    component_map: ComponentMap, config: Mapping[str, Any]
) -> dict[str, Any]:
    mach_count = int(config["flight_mach_samples"])
    area_count = int(config["capture_area_ratio_samples"])
    recovery_limit = float(config["max_adjacent_pressure_recovery_jump"])
    discharge_limit = float(config["max_adjacent_discharge_coefficient_jump"])
    if min(mach_count, area_count) < 3 or min(recovery_limit, discharge_limit) <= 0:
        raise V22C1InletValidationError("Dense map scan contract is invalid")
    mach_axis = component_map.coordinates["flight_mach"]
    area_axis = component_map.coordinates["capture_area_ratio"]
    machs = np.linspace(float(mach_axis[0]), float(mach_axis[-1]), mach_count)
    areas = np.linspace(float(area_axis[0]), float(area_axis[-1]), area_count)
    recovery = np.empty((mach_count, area_count))
    discharge = np.empty_like(recovery)
    for i, mach in enumerate(machs):
        for j, area in enumerate(areas):
            evaluated = component_map.evaluate(
                {"flight_mach": float(mach), "capture_area_ratio": float(area)}
            )
            recovery[i, j] = evaluated.values["total_pressure_recovery"]
            discharge[i, j] = evaluated.values["discharge_coefficient"]
    recovery_jump = _maximum_adjacent_jump(recovery)
    discharge_jump = _maximum_adjacent_jump(discharge)
    physical = bool(
        np.all(recovery > 0.0)
        and np.all(recovery <= 1.0)
        and np.all(discharge > 0.0)
        and np.all(discharge <= 1.0)
    )
    return {
        "sample_shape": [mach_count, area_count],
        "maximum_adjacent_pressure_recovery_jump": recovery_jump,
        "maximum_adjacent_discharge_coefficient_jump": discharge_jump,
        "maximum_allowed_pressure_recovery_jump": recovery_limit,
        "maximum_allowed_discharge_coefficient_jump": discharge_limit,
        "outputs_in_zero_one": physical,
        "pass": (
            physical
            and recovery_jump <= recovery_limit
            and discharge_jump <= discharge_limit
        ),
    }


def _boundary_classification(
    component_map: ComponentMap, config: Mapping[str, Any]
) -> dict[str, Any]:
    fraction = float(config["probe_fraction_of_capture_span"])
    if not 0.0 < fraction < 0.1:
        raise V22C1InletValidationError("Boundary probe fraction is invalid")
    cases = []
    for mach in component_map.coordinates["flight_mach"]:
        probe = probe_axis_boundaries(
            component_map,
            fixed_coordinates={"flight_mach": float(mach)},
            axis_name="capture_area_ratio",
            center_value=0.8,
            minimum_margin_name="minimum_capture_area_ratio",
            maximum_margin_name="maximum_capture_area_ratio",
            fraction=fraction,
        )
        cases.append({"flight_mach": float(mach), **probe})
    return {"cases": cases, "pass": all(case["pass"] for case in cases)}


def _operating_sweep(
    baseline: Any,
    enhanced: Any,
    design: Geometry19,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    machs = tuple(float(value) for value in config["flight_mach_values"])
    jump_limit = float(config["maximum_consecutive_mass_flow_relative_jump"])
    enthalpy_tolerance = float(config["total_enthalpy_absolute_tolerance_J_kg"])
    if len(machs) < 3 or min(machs) < 0.0 or jump_limit <= 0.0:
        raise V22C1InletValidationError("Operating sweep contract is invalid")
    records = []
    enhanced_flows = []
    enhanced_recoveries = []
    for mach in machs:
        environment = Environment(mach=mach)
        for context in (baseline, enhanced):
            result = context.physics_runtime.evaluate(design, environment)
            inlet = result.components.get("inlet", {})
            record = {
                "profile": context.profile.name,
                "flight_mach": mach,
                "converged": result.solver.converged,
                "flow_root_count": int(result.operating_point.get("flow_root_count", 0)),
                "selected_mass_flow_air_kg_s": result.operating_point.get(
                    "selected_flow_root_mass_flow_air_kg_s"
                ),
                "map_in_domain": inlet.get("map_in_domain"),
                "total_pressure_recovery": inlet.get("total_pressure_recovery"),
                "total_enthalpy_residual_J_kg": inlet.get(
                    "total_enthalpy_residual_J_kg"
                ),
                "freestream_total_pressure_Pa": inlet.get(
                    "freestream_total_pressure_Pa"
                ),
                "station_2_total_pressure_Pa": (
                    result.stations["t2"].total_pressure_Pa
                    if result.solver.converged
                    else None
                ),
            }
            records.append(record)
            if context is enhanced and result.solver.converged:
                enhanced_flows.append(float(record["selected_mass_flow_air_kg_s"]))
                enhanced_recoveries.append(float(record["total_pressure_recovery"]))
    transitions = [
        abs(right - left) / max(left, 1.0e-12)
        for left, right in zip(enhanced_flows[:-1], enhanced_flows[1:], strict=True)
    ]
    enhanced_records = [
        item for item in records if item["profile"] == enhanced.profile.name
    ]
    state_pass = all(
        item["converged"]
        and item["flow_root_count"] == 1
        and item["map_in_domain"] is True
        and abs(float(item["total_enthalpy_residual_J_kg"])) <= enthalpy_tolerance
        and float(item["station_2_total_pressure_Pa"])
        <= float(item["freestream_total_pressure_Pa"])
        for item in enhanced_records
    )
    trend_pass = (
        len(enhanced_flows) == len(machs)
        and enhanced_flows == sorted(enhanced_flows)
        and enhanced_recoveries == sorted(enhanced_recoveries, reverse=True)
        and all(value <= jump_limit for value in transitions)
    )
    return {
        "records": records,
        "trace_complete": len(records) == 2 * len(machs),
        "maximum_mass_flow_relative_jump": max(transitions, default=math.inf),
        "maximum_allowed_mass_flow_relative_jump": jump_limit,
        "state_and_enthalpy_closure_pass": state_pass,
        "monotone_trends_pass": trend_pass,
        "pass": state_pass and trend_pass,
    }


def _runtime_benchmark(
    baseline: Any,
    enhanced: Any,
    design: Geometry19,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    machs = tuple(float(value) for value in config["flight_mach_values"])
    warmups = int(config["warmup_repetitions"])
    repetitions = int(config["timed_repetitions"])
    p50_limit = float(config["max_p50_enhanced_to_baseline_ratio"])
    p95_limit = float(config["max_p95_enhanced_to_baseline_ratio"])
    if len(machs) < 3 or warmups < 1 or repetitions < 3:
        raise V22C1InletValidationError("Runtime benchmark contract is invalid")
    contexts = (baseline, enhanced)
    for context in contexts:
        for _ in range(warmups):
            for mach in machs:
                context.physics_runtime.evaluate(design, Environment(mach=mach))
    timings = {context.profile.name: [] for context in contexts}
    for repetition in range(repetitions):
        ordered_contexts = contexts if repetition % 2 == 0 else tuple(reversed(contexts))
        ordered_machs = machs if repetition % 2 == 0 else tuple(reversed(machs))
        for mach in ordered_machs:
            for context in ordered_contexts:
                start = time.perf_counter_ns()
                context.physics_runtime.evaluate(design, Environment(mach=mach))
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
        "timed_evaluation_count_per_profile": repetitions * len(machs),
        "percentile_milliseconds": percentiles,
        "p50_enhanced_to_baseline_ratio": p50_ratio,
        "p95_enhanced_to_baseline_ratio": p95_ratio,
        "maximum_p50_ratio": p50_limit,
        "maximum_p95_ratio": p95_limit,
        "pass": p50_ratio <= p50_limit and p95_ratio <= p95_limit,
    }


def run_v2_2_c1_inlet_validation(
    repository_root: str | Path,
    *,
    config_path: str | Path = "config/physics/wp_v2_2_c1_inlet_validation_v1.yaml",
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved_config = (root / config_path).resolve()
    config = _load_config(resolved_config)
    baseline = build_pipeline_context(root, pipeline_profile=config["baseline_profile"])
    enhanced = build_pipeline_context(root, pipeline_profile=config["enhanced_profile"])
    maps = enhanced.physics_runtime.component_maps
    if set(maps) != {"compressor", "turbine", "inlet"}:
        raise V22C1InletValidationError("Enhanced profile map selection mismatch")
    inlet_map = maps["inlet"]
    if inlet_map.map_id != config["expected_inlet_map_id"]:
        raise V22C1InletValidationError("Unexpected inlet map selection")
    design = _reference_design(enhanced)
    evidence = validate_component_map_package(
        enhanced.physics_runtime.source_paths["inlet_map"]
    )
    dense = _dense_map_scan(inlet_map, config["dense_map_scan"])
    boundaries = _boundary_classification(
        inlet_map, config["boundary_classification"]
    )
    operating = _operating_sweep(
        baseline, enhanced, design, config["operating_sweep"]
    )
    runtime = _runtime_benchmark(
        baseline, enhanced, design, config["runtime_benchmark"]
    )
    gates = {
        "synthetic_map_structural_shadow_ready": bool(evidence["shadow_ready"]),
        "dense_map_continuity": bool(dense["pass"]),
        "capture_boundary_classification": bool(boundaries["pass"]),
        "operating_state_and_enthalpy_closure": bool(operating["pass"]),
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
        "operating_sweep": operating,
        "runtime_benchmark": runtime,
        "software_gates": gates,
        "software_gate_pass": all(gates.values()),
        "engineering_ready": False,
        "engineering_blockers": list(config["engineering_blockers"]),
    }


def render_v2_2_c1_inlet_report(report: Mapping[str, Any]) -> str:
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


def write_v2_2_c1_inlet_artifacts(
    report: Mapping[str, Any], output_root: str | Path
) -> None:
    output = Path(output_root).resolve()
    if output.exists():
        raise V22C1InletValidationError(
            f"Refusing to overwrite validation output: {output}"
        )
    output.mkdir(parents=True)
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "Validation_Report.md").write_text(
        render_v2_2_c1_inlet_report(report), encoding="utf-8"
    )

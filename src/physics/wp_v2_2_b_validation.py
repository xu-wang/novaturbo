"""Locked V2.2-B compressor+turbine shadow validation campaign."""

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


class V22BTurbineValidationError(ValueError):
    """Raised when the locked V2.2-B campaign contract is invalid."""


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise V22BTurbineValidationError(
            f"Cannot load validation config: {path}"
        ) from exc
    required = {
        "validation_id",
        "baseline_profile",
        "enhanced_profile",
        "expected_turbine_map_id",
        "dense_map_scan",
        "boundary_classification",
        "root_sweep",
        "runtime_benchmark",
        "engineering_blockers",
        "artifacts",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise V22BTurbineValidationError("V2.2-B validation config keys mismatch")
    blockers = payload["engineering_blockers"]
    if not isinstance(blockers, list) or not blockers or not all(
        isinstance(item, str) and item for item in blockers
    ):
        raise V22BTurbineValidationError("Engineering blockers must be explicit")
    return payload


def _reference_design(context: Any) -> Geometry19:
    return Geometry19.from_mapping(
        {
            item["name"]: item["reference_value"]
            for item in context.physics_runtime.design_schema["design_variables"]
        }
    )


def _dense_map_scan(
    component_map: ComponentMap, config: Mapping[str, Any]
) -> dict[str, Any]:
    speed_count = int(config["corrected_speed_samples"])
    flux_count = int(config["corrected_mass_flux_samples"])
    threshold = float(config["max_adjacent_efficiency_jump"])
    if speed_count < 3 or flux_count < 3 or threshold <= 0.0:
        raise V22BTurbineValidationError("Dense map scan contract is invalid")
    speed_axis = component_map.coordinates["corrected_speed_ratio"]
    flux_axis = component_map.coordinates["corrected_mass_flux_ratio"]
    speeds = np.linspace(float(speed_axis[0]), float(speed_axis[-1]), speed_count)
    fluxes = np.linspace(float(flux_axis[0]), float(flux_axis[-1]), flux_count)
    efficiency = np.empty((speed_count, flux_count))
    for i, speed in enumerate(speeds):
        for j, flux in enumerate(fluxes):
            result = component_map.evaluate(
                {
                    "corrected_speed_ratio": float(speed),
                    "corrected_mass_flux_ratio": float(flux),
                }
            )
            efficiency[i, j] = result.values["total_to_total_efficiency"]
    maximum_jump = float(
        max(
            np.max(np.abs(np.diff(efficiency, axis=0))),
            np.max(np.abs(np.diff(efficiency, axis=1))),
        )
    )
    finite = bool(np.all(np.isfinite(efficiency)))
    physical = bool(np.all(efficiency > 0.0) and np.all(efficiency < 1.0))
    return {
        "sample_shape": [speed_count, flux_count],
        "maximum_adjacent_efficiency_jump": maximum_jump,
        "maximum_allowed_jump": threshold,
        "finite": finite,
        "efficiency_strictly_between_zero_and_one": physical,
        "pass": finite and physical and maximum_jump <= threshold,
    }


def _boundary_classification(
    component_map: ComponentMap, config: Mapping[str, Any]
) -> dict[str, Any]:
    fraction = float(config["probe_fraction_of_capacity_span"])
    if not 0.0 < fraction < 0.1:
        raise V22BTurbineValidationError(
            "Boundary probe fraction must be in (0, 0.1)"
        )
    cases = []
    for speed in component_map.coordinates["corrected_speed_ratio"]:
        probe = probe_axis_boundaries(
            component_map,
            fixed_coordinates={"corrected_speed_ratio": float(speed)},
            axis_name="corrected_mass_flux_ratio",
            center_value=1.0,
            minimum_margin_name="minimum_corrected_mass_flux_ratio",
            maximum_margin_name="maximum_corrected_mass_flux_ratio",
            fraction=fraction,
        )
        cases.append(
            {
                "corrected_speed_ratio": float(speed),
                **probe,
            }
        )
    return {"cases": cases, "pass": all(case["pass"] for case in cases)}


def _result_record(profile: str, ratio: float, result: Any) -> dict[str, Any]:
    failure = result.solver.failure_code
    turbine = result.components.get("turbine", {})
    t4 = result.stations.get("t4")
    t5 = result.stations.get("t5")
    return {
        "profile": profile,
        "shaft_speed_ratio": ratio,
        "converged": result.solver.converged,
        "failure_code": failure.value if failure is not None else None,
        "flow_root_count": int(result.operating_point.get("flow_root_count", 0)),
        "selected_mass_flow_air_kg_s": result.operating_point.get(
            "selected_flow_root_mass_flow_air_kg_s"
        ),
        "shaft_power_residual_W": result.conservation.shaft_power_W,
        "temperature_decreases": bool(
            t4 is not None
            and t5 is not None
            and t5.total_temperature_K < t4.total_temperature_K
        ),
        "pressure_decreases": bool(
            t4 is not None
            and t5 is not None
            and t5.total_pressure_Pa < t4.total_pressure_Pa
        ),
        "map_id": turbine.get("map_id"),
        "map_in_domain": turbine.get("map_in_domain"),
        "minimum_capacity_margin": turbine.get("minimum_capacity_margin"),
        "maximum_capacity_margin": turbine.get("maximum_capacity_margin"),
    }


def _root_sweep(
    baseline_context: Any,
    enhanced_context: Any,
    design: Geometry19,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ratios = tuple(float(value) for value in config["shaft_speed_ratios"])
    minimum_converged = float(config["minimum_enhanced_converged_fraction"])
    jump_limit = float(config["max_consecutive_mass_flow_relative_jump"])
    shaft_tolerance = float(config["shaft_power_absolute_tolerance_W"])
    if (
        len(ratios) < 3
        or any(value <= 0.0 for value in ratios)
        or not 0.0 < minimum_converged <= 1.0
        or jump_limit <= 0.0
        or shaft_tolerance < 0.0
    ):
        raise V22BTurbineValidationError("Root sweep contract is invalid")
    records = []
    for ratio in ratios:
        trial = replace(design, shaft_speed_rpm=design.shaft_speed_rpm * ratio)
        for context in (baseline_context, enhanced_context):
            records.append(
                _result_record(
                    context.profile.name,
                    ratio,
                    context.physics_runtime.evaluate(trial, Environment()),
                )
            )
    enhanced = [
        item for item in records if item["profile"] == enhanced_context.profile.name
    ]
    converged_fraction = sum(item["converged"] for item in enhanced) / len(enhanced)
    unexplained_jumps = 0
    transitions = []
    for left, right in zip(enhanced[:-1], enhanced[1:], strict=True):
        relative_jump = None
        if left["converged"] and right["converged"]:
            left_flow = float(left["selected_mass_flow_air_kg_s"])
            right_flow = float(right["selected_mass_flow_air_kg_s"])
            relative_jump = abs(right_flow - left_flow) / max(left_flow, 1.0e-12)
        unexplained = relative_jump is not None and relative_jump > jump_limit
        unexplained_jumps += int(unexplained)
        transitions.append(
            {
                "from_speed_ratio": left["shaft_speed_ratio"],
                "to_speed_ratio": right["shaft_speed_ratio"],
                "selected_mass_flow_relative_jump": relative_jump,
                "unexplained_jump": unexplained,
            }
        )
    state_pass = all(
        item["converged"]
        and item["flow_root_count"] == 1
        and item["temperature_decreases"]
        and item["pressure_decreases"]
        and abs(float(item["shaft_power_residual_W"])) <= shaft_tolerance
        and item["map_in_domain"] is True
        for item in enhanced
    )
    return {
        "records": records,
        "transitions": transitions,
        "trace_complete": len(records) == 2 * len(ratios),
        "enhanced_converged_fraction": converged_fraction,
        "minimum_enhanced_converged_fraction": minimum_converged,
        "unexplained_jump_count": unexplained_jumps,
        "max_consecutive_mass_flow_relative_jump": jump_limit,
        "shaft_power_absolute_tolerance_W": shaft_tolerance,
        "state_direction_and_shaft_balance_pass": state_pass,
        "pass": (
            converged_fraction >= minimum_converged
            and unexplained_jumps == 0
            and state_pass
        ),
    }


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values), percentile))


def _runtime_benchmark(
    baseline_context: Any,
    enhanced_context: Any,
    design: Geometry19,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ratios = tuple(float(value) for value in config["shaft_speed_ratios"])
    warmups = int(config["warmup_repetitions"])
    repetitions = int(config["timed_repetitions"])
    p50_limit = float(config["max_p50_enhanced_to_baseline_ratio"])
    p95_limit = float(config["max_p95_enhanced_to_baseline_ratio"])
    if (
        len(ratios) < 3
        or warmups < 1
        or repetitions < 3
        or p50_limit <= 0.0
        or p95_limit <= 0.0
    ):
        raise V22BTurbineValidationError("Runtime benchmark contract is invalid")
    designs = tuple(
        replace(design, shaft_speed_rpm=design.shaft_speed_rpm * ratio)
        for ratio in ratios
    )
    contexts = (baseline_context, enhanced_context)
    for context in contexts:
        for _ in range(warmups):
            for trial in designs:
                context.physics_runtime.evaluate(trial, Environment())
    timings = {context.profile.name: [] for context in contexts}
    for repetition in range(repetitions):
        ordered_contexts = (
            contexts if repetition % 2 == 0 else tuple(reversed(contexts))
        )
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
            "p95_ms": _percentile(values, 95.0),
        }
        for name, values in timings.items()
    }
    baseline = percentiles[baseline_context.profile.name]
    enhanced = percentiles[enhanced_context.profile.name]
    p50_ratio = enhanced["p50_ms"] / baseline["p50_ms"]
    p95_ratio = enhanced["p95_ms"] / baseline["p95_ms"]
    return {
        "warmup_repetitions": warmups,
        "timed_repetitions": repetitions,
        "timed_evaluation_count_per_profile": repetitions * len(designs),
        "percentile_milliseconds": percentiles,
        "p50_enhanced_to_baseline_ratio": p50_ratio,
        "p95_enhanced_to_baseline_ratio": p95_ratio,
        "maximum_p50_ratio": p50_limit,
        "maximum_p95_ratio": p95_limit,
        "pass": (
            math.isfinite(p50_ratio)
            and math.isfinite(p95_ratio)
            and p50_ratio <= p50_limit
            and p95_ratio <= p95_limit
        ),
    }


def run_v2_2_b_turbine_validation(
    repository_root: str | Path,
    *,
    config_path: str | Path = "config/physics/wp_v2_2_b_turbine_validation_v1.yaml",
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved_config = (root / config_path).resolve()
    config = _load_config(resolved_config)
    baseline = build_pipeline_context(root, pipeline_profile=config["baseline_profile"])
    enhanced = build_pipeline_context(root, pipeline_profile=config["enhanced_profile"])
    maps = enhanced.physics_runtime.component_maps
    if set(maps) != {"compressor", "turbine"}:
        raise V22BTurbineValidationError(
            "Enhanced profile must select compressor and turbine maps"
        )
    turbine_map = maps["turbine"]
    if turbine_map.map_id != config["expected_turbine_map_id"]:
        raise V22BTurbineValidationError("Unexpected turbine map selection")
    design = _reference_design(baseline)
    evidence = validate_component_map_package(
        enhanced.physics_runtime.source_paths["turbine_map"]
    )
    dense_scan = _dense_map_scan(turbine_map, config["dense_map_scan"])
    boundaries = _boundary_classification(
        turbine_map, config["boundary_classification"]
    )
    roots = _root_sweep(baseline, enhanced, design, config["root_sweep"])
    runtime = _runtime_benchmark(
        baseline, enhanced, design, config["runtime_benchmark"]
    )
    gates = {
        "synthetic_map_structural_shadow_ready": bool(evidence["shadow_ready"]),
        "dense_map_continuity": bool(dense_scan["pass"]),
        "capacity_boundary_classification": bool(boundaries["pass"]),
        "root_state_and_shaft_balance": bool(roots["pass"]),
        "p50_p95_runtime_ratio": bool(runtime["pass"]),
    }
    return {
        "validation_id": config["validation_id"],
        "config_sha256": sha256(resolved_config.read_bytes()).hexdigest(),
        "baseline_context": baseline.contract_mapping(),
        "enhanced_context": enhanced.contract_mapping(),
        "map_validation": evidence,
        "dense_map_scan": dense_scan,
        "boundary_classification": boundaries,
        "root_sweep": roots,
        "runtime_benchmark": runtime,
        "software_gates": gates,
        "software_gate_pass": all(gates.values()),
        "engineering_ready": False,
        "engineering_blockers": list(config["engineering_blockers"]),
    }


def render_v2_2_b_turbine_report(report: Mapping[str, Any]) -> str:
    runtime = report["runtime_benchmark"]
    lines = [
        f"# {report['validation_id']}",
        "",
        f"- software gate: **{'PASS' if report['software_gate_pass'] else 'FAIL'}**",
        f"- engineering ready: **{str(report['engineering_ready']).lower()}**",
        f"- P50 runtime ratio: `{runtime['p50_enhanced_to_baseline_ratio']:.3f}x`",
        f"- P95 runtime ratio: `{runtime['p95_enhanced_to_baseline_ratio']:.3f}x`",
        f"- unexplained root jumps: `{report['root_sweep']['unexplained_jump_count']}`",
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


def write_v2_2_b_turbine_artifacts(
    report: Mapping[str, Any], output_root: str | Path
) -> None:
    output = Path(output_root).resolve()
    if output.exists():
        raise V22BTurbineValidationError(
            f"Refusing to overwrite validation output: {output}"
        )
    output.mkdir(parents=True)
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "Validation_Report.md").write_text(
        render_v2_2_b_turbine_report(report), encoding="utf-8"
    )

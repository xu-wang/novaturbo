"""Locked WP-V3-2 compressor-map validation campaign."""

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
from src.physics.maps.validation import validate_component_map_package


class WPV32ValidationError(ValueError):
    """Raised when the locked validation contract cannot be executed."""


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise WPV32ValidationError(f"Cannot load validation config: {path}") from exc
    required = {
        "validation_id",
        "baseline_profile",
        "enhanced_profile",
        "expected_map_id",
        "dense_map_scan",
        "boundary_classification",
        "root_sweep",
        "runtime_benchmark",
        "artifacts",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise WPV32ValidationError("WP-V3-2 validation config keys mismatch")
    return payload


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


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
    flow_count = int(config["flow_coefficient_samples"])
    if speed_count < 3 or flow_count < 3:
        raise WPV32ValidationError("Dense map scan requires at least three samples per axis")
    speed_axis = component_map.coordinates["corrected_speed_ratio"]
    flow_axis = component_map.coordinates["flow_coefficient_ratio"]
    speeds = np.linspace(float(speed_axis[0]), float(speed_axis[-1]), speed_count)
    flows = np.linspace(float(flow_axis[0]), float(flow_axis[-1]), flow_count)
    surfaces = {
        name: np.empty((speed_count, flow_count))
        for name in component_map.output_names
    }
    for i, speed in enumerate(speeds):
        for j, flow in enumerate(flows):
            result = component_map.evaluate(
                {
                    "corrected_speed_ratio": float(speed),
                    "flow_coefficient_ratio": float(flow),
                }
            )
            for name in component_map.output_names:
                surfaces[name][i, j] = result.values[name]

    def maximum_adjacent_jump(surface: np.ndarray) -> float:
        return float(
            max(
                np.max(np.abs(np.diff(surface, axis=0))),
                np.max(np.abs(np.diff(surface, axis=1))),
            )
        )

    work_jump = maximum_adjacent_jump(surfaces["work_coefficient"])
    efficiency_jump = maximum_adjacent_jump(surfaces["total_to_total_efficiency"])
    thresholds = {
        "work_coefficient": float(config["max_adjacent_work_coefficient_jump"]),
        "total_to_total_efficiency": float(config["max_adjacent_efficiency_jump"]),
    }
    return {
        "sample_shape": [speed_count, flow_count],
        "maximum_adjacent_jump": {
            "work_coefficient": work_jump,
            "total_to_total_efficiency": efficiency_jump,
        },
        "thresholds": thresholds,
        "finite": all(
            bool(np.all(np.isfinite(surface))) for surface in surfaces.values()
        ),
        "pass": work_jump <= thresholds["work_coefficient"]
        and efficiency_jump <= thresholds["total_to_total_efficiency"],
    }


def _boundary_classification(
    component_map: ComponentMap, config: Mapping[str, Any]
) -> dict[str, Any]:
    fraction = float(config["probe_fraction_of_stable_span"])
    if not 0.0 < fraction < 0.1:
        raise WPV32ValidationError("Boundary probe fraction must be in (0, 0.1)")
    cases = []
    for speed in component_map.coordinates["corrected_speed_ratio"]:
        query = {
            "corrected_speed_ratio": float(speed),
            "flow_coefficient_ratio": 1.0,
        }
        boundaries = component_map.evaluate(query).margins
        surge = float(boundaries["surge_flow_coefficient_ratio"])
        choke = float(boundaries["choke_flow_coefficient_ratio"])
        span = choke - surge
        probes = {
            "below_surge": surge - fraction * span,
            "inside_near_surge": surge + fraction * span,
            "inside_near_choke": choke - fraction * span,
            "above_choke": choke + fraction * span,
        }
        classified = {
            name: {
                "flow_coefficient_ratio": value,
                "surge_margin": (value - surge) / span,
                "choke_margin": (choke - value) / span,
            }
            for name, value in probes.items()
        }
        passed = (
            classified["below_surge"]["surge_margin"] < 0.0
            and classified["inside_near_surge"]["surge_margin"] > 0.0
            and classified["inside_near_choke"]["choke_margin"] > 0.0
            and classified["above_choke"]["choke_margin"] < 0.0
        )
        cases.append(
            {
                "corrected_speed_ratio": float(speed),
                "pass": passed,
                "probes": classified,
            }
        )
    return {"cases": cases, "pass": all(case["pass"] for case in cases)}


def _result_record(profile: str, speed_ratio: float, result: Any) -> dict[str, Any]:
    failure = result.solver.failure_code
    compressor = result.components.get("compressor", {})
    return {
        "profile": profile,
        "shaft_speed_ratio": speed_ratio,
        "converged": result.solver.converged,
        "failure_code": failure.value if failure is not None else None,
        "flow_root_count": int(result.operating_point["flow_root_count"]),
        "selected_flow_root_index": result.operating_point["selected_flow_root_index"],
        "selected_mass_flow_air_kg_s": result.operating_point.get(
            "selected_flow_root_mass_flow_air_kg_s"
        ),
        "map_id": compressor.get("map_id"),
        "map_in_domain": compressor.get("map_in_domain"),
        "surge_margin": compressor.get("surge_margin"),
        "choke_margin": compressor.get("choke_margin"),
    }


def _root_sweep(
    baseline_context: Any,
    enhanced_context: Any,
    design: Geometry19,
    environment: Environment,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ratios = tuple(float(value) for value in config["shaft_speed_ratios"])
    if len(ratios) < 3 or any(value <= 0.0 for value in ratios):
        raise WPV32ValidationError("Root sweep requires at least three positive ratios")
    records = []
    for ratio in ratios:
        trial = replace(design, shaft_speed_rpm=design.shaft_speed_rpm * ratio)
        for context in (baseline_context, enhanced_context):
            records.append(
                _result_record(
                    context.profile.name,
                    ratio,
                    context.physics_runtime.evaluate(trial, environment),
                )
            )
    enhanced = [
        record for record in records if record["profile"] == enhanced_context.profile.name
    ]
    converged_fraction = sum(record["converged"] for record in enhanced) / len(enhanced)
    minimum_converged = float(config["minimum_enhanced_converged_fraction"])
    if not 0.0 < minimum_converged <= 1.0:
        raise WPV32ValidationError(
            "Minimum enhanced convergence fraction must be in (0, 1]"
        )
    transitions = []
    limit = float(config["max_consecutive_mass_flow_relative_jump"])
    unexplained_jumps = 0
    for left, right in zip(enhanced[:-1], enhanced[1:]):
        relative_jump = None
        unexplained = False
        if left["converged"] and right["converged"]:
            left_flow = float(left["selected_mass_flow_air_kg_s"])
            right_flow = float(right["selected_mass_flow_air_kg_s"])
            relative_jump = abs(right_flow - left_flow) / max(abs(left_flow), 1.0e-12)
            unexplained = (
                relative_jump > limit
                and left["flow_root_count"] == right["flow_root_count"]
            )
        unexplained_jumps += int(unexplained)
        transitions.append(
            {
                "from_speed_ratio": left["shaft_speed_ratio"],
                "to_speed_ratio": right["shaft_speed_ratio"],
                "convergence_changed": left["converged"] != right["converged"],
                "root_count_changed": left["flow_root_count"] != right["flow_root_count"],
                "selected_mass_flow_relative_jump": relative_jump,
                "unexplained_jump": unexplained,
            }
        )
    return {
        "records": records,
        "transitions": transitions,
        "unexplained_jump_count": unexplained_jumps,
        "enhanced_converged_fraction": converged_fraction,
        "minimum_enhanced_converged_fraction": minimum_converged,
        "max_consecutive_mass_flow_relative_jump": limit,
        "trace_complete": len(records) == 2 * len(ratios),
        "pass": (
            unexplained_jumps == 0
            and converged_fraction >= minimum_converged
            and all(record["flow_root_count"] <= 1 for record in enhanced)
        ),
    }


def _runtime_benchmark(
    baseline_context: Any,
    enhanced_context: Any,
    design: Geometry19,
    environment: Environment,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    warmups = int(config["warmup_repetitions"])
    repetitions = int(config["timed_repetitions"])
    limit = float(config["max_enhanced_to_baseline_ratio"])
    speed_ratios = tuple(float(value) for value in config["shaft_speed_ratios"])
    if (
        warmups < 1
        or repetitions < 3
        or limit <= 0.0
        or len(speed_ratios) < 3
        or any(value <= 0.0 for value in speed_ratios)
    ):
        raise WPV32ValidationError("Runtime benchmark contract is invalid")
    contexts = (baseline_context, enhanced_context)
    designs = tuple(
        replace(design, shaft_speed_rpm=design.shaft_speed_rpm * ratio)
        for ratio in speed_ratios
    )
    for context in contexts:
        for _ in range(warmups):
            for trial in designs:
                context.physics_runtime.evaluate(trial, environment)
    timings: dict[str, list[float]] = {context.profile.name: [] for context in contexts}
    # Alternate profile order per repetition to reduce systematic thermal/load drift.
    for repetition in range(repetitions):
        ordered = contexts if repetition % 2 == 0 else tuple(reversed(contexts))
        ordered_designs = designs if repetition % 2 == 0 else tuple(reversed(designs))
        for trial in ordered_designs:
            for context in ordered:
                start = time.perf_counter_ns()
                context.physics_runtime.evaluate(trial, environment)
                timings[context.profile.name].append(
                    (time.perf_counter_ns() - start) / 1.0e6
                )
    medians = {name: statistics.median(values) for name, values in timings.items()}
    ratio = (
        medians[enhanced_context.profile.name]
        / medians[baseline_context.profile.name]
    )
    return {
        "warmup_repetitions": warmups,
        "timed_repetitions": repetitions,
        "shaft_speed_ratios": list(speed_ratios),
        "timed_evaluation_count_per_profile": repetitions * len(designs),
        "median_milliseconds": medians,
        "enhanced_to_baseline_ratio": ratio,
        "maximum_allowed_ratio": limit,
        "pass": math.isfinite(ratio) and ratio <= limit,
    }


def run_wp_v3_2_validation(
    repository_root: str | Path,
    *,
    config_path: str | Path = "config/physics/wp_v3_2_hecc_validation_v1.yaml",
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved_config = (root / config_path).resolve()
    config = _load_config(resolved_config)
    baseline = build_pipeline_context(root, pipeline_profile=config["baseline_profile"])
    enhanced = build_pipeline_context(root, pipeline_profile=config["enhanced_profile"])
    maps = enhanced.physics_runtime.component_maps
    if set(maps) != {"compressor"}:
        raise WPV32ValidationError(
            "Enhanced profile must contain exactly one compressor map"
        )
    component_map = maps["compressor"]
    if component_map.map_id != config["expected_map_id"]:
        raise WPV32ValidationError("Enhanced profile selected an unexpected compressor map")
    design = _reference_design(baseline)
    environment = Environment()
    evidence = validate_component_map_package(
        enhanced.physics_runtime.source_paths["compressor_map"]
    )
    dense_scan = _dense_map_scan(component_map, config["dense_map_scan"])
    boundaries = _boundary_classification(
        component_map, config["boundary_classification"]
    )
    roots = _root_sweep(baseline, enhanced, design, environment, config["root_sweep"])
    runtime = _runtime_benchmark(
        baseline, enhanced, design, environment, config["runtime_benchmark"]
    )
    gates = {
        "source_evidence_shadow_ready": bool(evidence["shadow_ready"]),
        "dense_map_continuity": bool(dense_scan["pass"]),
        "boundary_classification": bool(boundaries["pass"]),
        "root_branch_traceability": bool(roots["pass"]),
        "runtime_ratio": bool(runtime["pass"]),
    }
    return {
        "validation_id": config["validation_id"],
        "config_sha256": _sha256_file(resolved_config),
        "baseline_context": baseline.contract_mapping(),
        "enhanced_context": enhanced.contract_mapping(),
        "map_validation": evidence,
        "dense_map_scan": dense_scan,
        "boundary_classification": boundaries,
        "root_sweep": roots,
        "runtime_benchmark": runtime,
        "software_gate_pass": all(gates.values()),
        "software_gates": gates,
        "engineering_ready": False,
        "engineering_blockers": evidence["evidence_validation"]["engineering_blockers"],
    }


def render_wp_v3_2_report(report: Mapping[str, Any]) -> str:
    runtime = report["runtime_benchmark"]
    scan = report["dense_map_scan"]
    lines = [
        f"# {report['validation_id']}",
        "",
        f"- software gate: **{'PASS' if report['software_gate_pass'] else 'FAIL'}**",
        f"- engineering ready: **{str(report['engineering_ready']).lower()}**",
        (
            f"- runtime ratio: `{runtime['enhanced_to_baseline_ratio']:.3f}x` "
            f"(limit `{runtime['maximum_allowed_ratio']:.3f}x`)"
        ),
        f"- dense scan: `{scan['sample_shape'][0]} x {scan['sample_shape'][1]}`",
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
    lines.extend(f"- {blocker}" for blocker in report["engineering_blockers"])
    return "\n".join(lines) + "\n"


def write_wp_v3_2_artifacts(report: Mapping[str, Any], output_root: str | Path) -> None:
    output = Path(output_root).resolve()
    if output.exists():
        raise WPV32ValidationError(f"Refusing to overwrite validation output: {output}")
    output.mkdir(parents=True)
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "Validation_Report.md").write_text(
        render_wp_v3_2_report(report), encoding="utf-8"
    )

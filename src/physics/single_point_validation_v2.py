"""Deterministic WP19-9 validation for the Geometry19 V2.1 model."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.physics.engine_match import (
    EngineResult,
    Environment,
    NozzleExitState,
    StationState,
    solve_design_point,
)
from src.physics.geometry_state import Geometry19
from src.physics.model_config import ModelConfig


class WP19ValidationError(ValueError):
    """Raised when the WP19-9 protocol itself is malformed."""


_STATION_NAMES = ("t2", "t3", "t4", "t5", "nozzle_total", "exit")
_COMPONENT_NAMES = ("compressor", "combustor", "turbine", "nozzle")


def _load_yaml_mapping(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise WP19ValidationError(f"Cannot load {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise WP19ValidationError(f"{label} must contain a mapping")
    return payload, sha256(raw).hexdigest()


def _require_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WP19ValidationError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise WP19ValidationError(f"{name} must be finite")
    return converted


def _load_tolerances(protocol: Mapping[str, Any]) -> dict[str, float]:
    raw = protocol.get("tolerances")
    if not isinstance(raw, Mapping):
        raise WP19ValidationError("Protocol tolerances must be a mapping")
    required = (
        "combustor_energy_abs_W",
        "shaft_power_abs_W",
        "nozzle_temperature_abs_K",
        "nozzle_pressure_abs_Pa",
        "nozzle_velocity_abs_m_s",
        "choked_mach_abs",
    )
    tolerances = {
        name: _require_finite_number(raw.get(name), f"tolerances.{name}")
        for name in required
    }
    if any(value <= 0.0 for value in tolerances.values()):
        raise WP19ValidationError("Every WP19-9 tolerance must be positive")
    return tolerances


def _reference_design(schema: Mapping[str, Any]) -> Geometry19:
    variables = schema.get("design_variables")
    if not isinstance(variables, list) or len(variables) != 19:
        raise WP19ValidationError("Design schema must define 19 variables")
    try:
        values = {
            str(variable["name"]): variable["reference_value"]
            for variable in variables
        }
    except (KeyError, TypeError) as exc:
        raise WP19ValidationError(
            "Every design variable requires a name and reference value"
        ) from exc
    return Geometry19.from_mapping(values)


def _case_design(reference: Geometry19, case: Mapping[str, Any]) -> Geometry19:
    overrides = case.get("design_overrides", {})
    if not isinstance(overrides, Mapping):
        raise WP19ValidationError("case.design_overrides must be a mapping")
    values = reference.as_mapping()
    unknown = sorted(set(overrides) - set(values))
    if unknown:
        raise WP19ValidationError(f"Unknown design overrides: {unknown}")
    values.update(overrides)
    return Geometry19.from_mapping(values)


def _case_environment(case: Mapping[str, Any]) -> Environment:
    raw = case.get("environment")
    if not isinstance(raw, Mapping):
        raise WP19ValidationError("case.environment must be a mapping")
    required = ("temperature_K", "pressure_Pa", "mach")
    values = {
        name: _require_finite_number(raw.get(name), f"environment.{name}")
        for name in required
    }
    return Environment(**values)


def _expectations(case: Mapping[str, Any]) -> dict[str, bool]:
    raw = case.get("expected")
    if not isinstance(raw, Mapping):
        raise WP19ValidationError("case.expected must be a mapping")
    required = ("converged", "nozzle_choked", "feasible")
    if any(not isinstance(raw.get(name), bool) for name in required):
        raise WP19ValidationError(
            "Expected converged, nozzle_choked, and feasible values must be boolean"
        )
    return {name: bool(raw[name]) for name in required}


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, Sequence):
        return all(_all_finite(item) for item in value)
    return False


def _station_records(result: EngineResult) -> dict[str, dict[str, float]]:
    if not result.solver.converged:
        return {}
    mdot_air = float(result.operating_point["mass_flow_air_kg_s"])
    mdot_gas = float(result.operating_point["mass_flow_gas_kg_s"])
    records: dict[str, dict[str, float]] = {}
    for name in ("t2", "t3", "t4", "t5", "nozzle_total"):
        state = result.stations[name]
        if not isinstance(state, StationState):
            raise WP19ValidationError(f"Station {name} has the wrong state type")
        records[name] = {
            "total_temperature_K": state.total_temperature_K,
            "total_pressure_Pa": state.total_pressure_Pa,
            "mass_flow_kg_s": mdot_air if name in ("t2", "t3") else mdot_gas,
        }
    exit_state = result.stations["exit"]
    if not isinstance(exit_state, NozzleExitState):
        raise WP19ValidationError("Nozzle exit has the wrong state type")
    records["exit"] = {
        "static_temperature_K": exit_state.static_temperature_K,
        "static_pressure_Pa": exit_state.static_pressure_Pa,
        "mach": exit_state.mach,
        "velocity_m_s": exit_state.velocity_m_s,
        "mass_flow_kg_s": mdot_gas,
    }
    return records


def _component_records(result: EngineResult) -> dict[str, dict[str, float | bool]]:
    if not result.solver.converged:
        return {}
    t3 = result.stations["t3"]
    t4 = result.stations["t4"]
    t5 = result.stations["t5"]
    nozzle_total = result.stations["nozzle_total"]
    if not all(
        isinstance(state, StationState) for state in (t3, t4, t5, nozzle_total)
    ):
        raise WP19ValidationError("Component pressure-ratio stations are invalid")
    compressor = result.components["compressor"]
    combustor = result.components["combustor"]
    turbine = result.components["turbine"]
    nozzle = result.components["nozzle"]
    return {
        "compressor": {
            "pressure_ratio": float(compressor["pressure_ratio"]),
            "total_to_total_efficiency": float(
                compressor["total_to_total_efficiency"]
            ),
            "flow_coefficient_ref": float(compressor["flow_coefficient_ref"]),
            "loading_coefficient": float(compressor["loading_coefficient"]),
        },
        "combustor": {
            "total_pressure_ratio": t4.total_pressure_Pa / t3.total_pressure_Pa,
            "pressure_loss_fraction": float(combustor["pressure_loss_fraction"]),
            "primary_residence_time_s": float(
                combustor["primary_residence_time_s"]
            ),
            "equivalence_ratio": float(combustor["equivalence_ratio"]),
        },
        "turbine": {
            "total_pressure_ratio": t5.total_pressure_Pa / t4.total_pressure_Pa,
            "total_to_total_efficiency": float(
                turbine["total_to_total_efficiency"]
            ),
            "flow_coefficient": float(turbine["flow_coefficient"]),
            "loading_coefficient": float(turbine["loading_coefficient"]),
            "solidity": float(turbine["solidity"]),
            "corrected_mass_flux_kg_s_m2": float(
                turbine["corrected_mass_flux_kg_s_m2"]
            ),
        },
        "nozzle": {
            "total_pressure_ratio": (
                nozzle_total.total_pressure_Pa / t5.total_pressure_Pa
            ),
            "total_pressure_recovery": float(nozzle["total_pressure_recovery"]),
            "choked": bool(nozzle["choked"]),
        },
    }


def _conservation_check(
    result: EngineResult,
    config: ModelConfig,
    tolerances: Mapping[str, float],
) -> dict[str, Any]:
    residuals = result.conservation
    flow_tolerance = float(config.solver_candidate["flow_residual_tolerance"])
    flow_epsilon = float(
        config.solver_candidate["mass_flow_residual_epsilon_kg_s"]
    )
    mdot_gas = float(result.operating_point["mass_flow_gas_kg_s"])
    raw_flow_tolerance = flow_tolerance * max(mdot_gas, flow_epsilon)
    values = {
        "combustor_energy_W": residuals.combustor_energy_W,
        "shaft_power_W": residuals.shaft_power_W,
        "mass_flow_kg_s": residuals.mass_flow_kg_s,
        "normalized_flow": residuals.normalized_flow,
    }
    limits = {
        "combustor_energy_W": tolerances["combustor_energy_abs_W"],
        "shaft_power_W": tolerances["shaft_power_abs_W"],
        "mass_flow_kg_s": raw_flow_tolerance,
        "normalized_flow": flow_tolerance,
    }
    checks = {
        name: math.isfinite(float(values[name]))
        and abs(float(values[name])) <= limit
        for name, limit in limits.items()
    }
    return {
        "values": values,
        "absolute_tolerances": limits,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _nozzle_state_check(
    result: EngineResult,
    config: ModelConfig,
    tolerances: Mapping[str, float],
) -> dict[str, Any]:
    nozzle_total = result.stations["nozzle_total"]
    exit_state = result.stations["exit"]
    if not isinstance(nozzle_total, StationState) or not isinstance(
        exit_state, NozzleExitState
    ):
        raise WP19ValidationError("Nozzle state types are invalid")
    gamma = float(config.require("gamma_hot"))
    gas_constant = float(config.require("gas_constant_hot_J_kg_K"))
    temperature_factor = 1.0 + 0.5 * (gamma - 1.0) * exit_state.mach**2
    reconstructed_total_temperature = (
        exit_state.static_temperature_K * temperature_factor
    )
    reconstructed_total_pressure = exit_state.static_pressure_Pa * (
        temperature_factor ** (gamma / (gamma - 1.0))
    )
    reconstructed_velocity = exit_state.mach * math.sqrt(
        gamma * gas_constant * exit_state.static_temperature_K
    )
    temperature_residual = (
        nozzle_total.total_temperature_K - reconstructed_total_temperature
    )
    pressure_residual = (
        nozzle_total.total_pressure_Pa - reconstructed_total_pressure
    )
    velocity_residual = exit_state.velocity_m_s - reconstructed_velocity
    choked = bool(result.components["nozzle"]["choked"])
    choked_mach_residual = abs(exit_state.mach - 1.0) if choked else None
    checks = {
        "total_temperature_reconstruction": abs(temperature_residual)
        <= tolerances["nozzle_temperature_abs_K"],
        "total_pressure_reconstruction": abs(pressure_residual)
        <= tolerances["nozzle_pressure_abs_Pa"],
        "velocity_reconstruction": abs(velocity_residual)
        <= tolerances["nozzle_velocity_abs_m_s"],
        "reported_velocity_consistency": abs(
            result.conservation.nozzle_velocity_m_s
        )
        <= tolerances["nozzle_velocity_abs_m_s"],
        "choked_mach_equals_one": (
            choked_mach_residual is None
            or choked_mach_residual <= tolerances["choked_mach_abs"]
        ),
    }
    return {
        "choked": choked,
        "exit_mach": exit_state.mach,
        "temperature_reconstruction_residual_K": temperature_residual,
        "pressure_reconstruction_residual_Pa": pressure_residual,
        "velocity_reconstruction_residual_m_s": velocity_residual,
        "reported_velocity_consistency_residual_m_s": (
            result.conservation.nozzle_velocity_m_s
        ),
        "choked_mach_residual": choked_mach_residual,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _constraint_check(result: EngineResult) -> dict[str, Any]:
    grouped = {
        **result.thermodynamic_constraints,
        **result.assembly_constraints,
        **result.structural_constraints,
        **result.thermal_constraints,
        **result.manufacturing_constraints,
    }
    violations = {
        name: value
        for name, value in result.constraints.items()
        if not math.isfinite(float(value)) or float(value) > 0.0
    }
    formal_feasible = not violations
    return {
        "constraint_count": len(result.constraints),
        "violated_count": len(violations),
        "violations": violations,
        "group_union_matches": grouped == result.constraints,
        "feasible_matches_residuals": result.feasible == formal_feasible,
        "passed": grouped == result.constraints
        and result.feasible == formal_feasible,
    }


def _validate_case(
    case: Mapping[str, Any],
    reference: Geometry19,
    config: ModelConfig,
    tolerances: Mapping[str, float],
) -> dict[str, Any]:
    case_id = case.get("id")
    if not isinstance(case_id, str) or not case_id:
        raise WP19ValidationError("Every validation case requires a nonempty id")
    design = _case_design(reference, case)
    environment = _case_environment(case)
    expected = _expectations(case)
    result = solve_design_point(design, environment, config)
    failure_code = (
        result.solver.failure_code.value if result.solver.failure_code else None
    )
    actual_choked = bool(result.components.get("nozzle", {}).get("choked", False))
    expectation_checks = {
        "converged": result.solver.converged == expected["converged"],
        "nozzle_choked": actual_choked == expected["nozzle_choked"],
        "feasible": result.feasible == expected["feasible"],
    }
    if not result.solver.converged:
        return {
            "id": case_id,
            "description": str(case.get("description", "")),
            "design_overrides": dict(case.get("design_overrides", {})),
            "environment": vars(environment),
            "expected": expected,
            "solver": {
                "converged": False,
                "failure_code": failure_code,
                "message": result.solver.message,
                "iterations": result.solver.iterations,
                "evaluations": result.solver.evaluations,
            },
            "expectation_checks": expectation_checks,
            "passed": False,
        }

    stations = _station_records(result)
    components = _component_records(result)
    conservation = _conservation_check(result, config, tolerances)
    nozzle_state = _nozzle_state_check(result, config, tolerances)
    constraints = _constraint_check(result)
    station_complete = (
        tuple(stations) == _STATION_NAMES
        and tuple(components) == _COMPONENT_NAMES
        and _all_finite(stations)
        and _all_finite(components)
    )
    soft_infeasible_semantics = (
        result.feasible
        or (
            failure_code is None
            and result.solver.converged
            and bool(constraints["violations"])
        )
    )
    checks = {
        "expectations": all(expectation_checks.values()),
        "station_and_component_completeness": station_complete,
        "conservation": conservation["passed"],
        "nozzle_state_consistency": nozzle_state["passed"],
        "constraint_partition_and_feasibility": constraints["passed"],
        "soft_infeasible_semantics": soft_infeasible_semantics,
    }
    return {
        "id": case_id,
        "description": str(case.get("description", "")),
        "design_overrides": dict(case.get("design_overrides", {})),
        "environment": vars(environment),
        "expected": expected,
        "solver": {
            "converged": result.solver.converged,
            "failure_code": failure_code,
            "message": result.solver.message,
            "iterations": result.solver.iterations,
            "evaluations": result.solver.evaluations,
            "flow_residual": result.solver.flow_residual,
        },
        "classification": {
            "feasible": result.feasible,
            "feasibility_complete": result.feasibility_complete,
            "feasibility_scope": result.feasibility_scope,
        },
        "operating_point": dict(result.operating_point),
        "stations": stations,
        "components": components,
        "performance": dict(result.performance),
        "mass_breakdown_kg": dict(result.mass_breakdown_kg),
        "conservation": conservation,
        "nozzle_state_consistency": nozzle_state,
        "constraints": constraints,
        "expectation_checks": expectation_checks,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_wp19_9_validation(
    protocol_path: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the frozen WP19-9 protocol and return a strict-JSON report."""
    protocol_source = Path(protocol_path).resolve()
    protocol, protocol_hash = _load_yaml_mapping(protocol_source, "protocol")
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else protocol_source.parents[2]
    )
    schema_path = root / str(protocol.get("design_schema", ""))
    calibration_path = root / str(protocol.get("calibration_pack", ""))
    schema, schema_hash = _load_yaml_mapping(schema_path, "design schema")
    config = ModelConfig.from_yaml(calibration_path)
    if protocol.get("physics_spec_version") != config.physics_spec_version:
        raise WP19ValidationError(
            "Protocol and calibration physics specification versions differ"
        )
    if schema.get("schema_version") != config.design_schema_version:
        raise WP19ValidationError(
            "Design schema and calibration design schema versions differ"
        )
    tolerances = _load_tolerances(protocol)
    raw_cases = protocol.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise WP19ValidationError("Protocol must define at least one case")
    case_ids = [case.get("id") for case in raw_cases if isinstance(case, Mapping)]
    if len(case_ids) != len(raw_cases) or len(case_ids) != len(set(case_ids)):
        raise WP19ValidationError("Validation case ids must be unique")
    expected_choked = {
        _expectations(case)["nozzle_choked"] for case in raw_cases
    }
    if expected_choked != {False, True}:
        raise WP19ValidationError(
            "WP19-9 requires both subsonic and choked nozzle cases"
        )
    reference = _reference_design(schema)
    cases = [
        _validate_case(case, reference, config, tolerances)
        for case in raw_cases
    ]
    gate_passed = all(case["passed"] for case in cases)
    report = {
        "protocol_version": protocol.get("protocol_version"),
        "status": "PASS" if gate_passed else "FAIL",
        "wp19_9_gate_passed": gate_passed,
        "wp19_10_entry_authorized": gate_passed,
        "formal_dataset_generation_ready": (
            config.formal_dataset_generation_ready
        ),
        "calibration_status": config.status,
        "case_count": len(cases),
        "passed_case_count": sum(bool(case["passed"]) for case in cases),
        "tolerances": {
            **tolerances,
            "flow_normalized_abs": float(
                config.solver_candidate["flow_residual_tolerance"]
            ),
        },
        "provenance": {
            "physics_spec_version": config.physics_spec_version,
            "design_schema_version": config.design_schema_version,
            "calibration_pack_version": config.pack_version,
            "calibration_pack_hash": config.calibration_pack_hash,
            "calibration_source_file_hash": config.source_file_hash,
            "design_schema_file_hash": schema_hash,
            "validation_protocol_file_hash": protocol_hash,
        },
        "cases": cases,
    }
    if not _all_finite(report):
        raise WP19ValidationError("WP19-9 report contains non-finite data")
    return report


def render_wp19_9_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise human-readable companion to the JSON artifact."""

    def formatted(value: Any) -> str:
        if value is None:
            return "—"
        return f"{float(value):.6e}"

    lines = [
        "# WP19-9 单点守恒验证报告",
        "",
        f"- 协议：`{report['protocol_version']}`",
        f"- 结论：`{report['status']}`",
        f"- 通过设计点：{report['passed_case_count']}/{report['case_count']}",
        f"- 标定状态：`{report['calibration_status']}`",
        "- 正式数据生成许可："
        f"`{str(report['formal_dataset_generation_ready']).lower()}`",
        "",
        "## 代表点结果",
        "",
        "| 设计点 | 收敛 | 喷管堵塞 | 正式可行 | 数值门禁 |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        lines.append(
            "| {id} | {converged} | {choked} | {feasible} | {passed} |".format(
                id=case["id"],
                converged=case["solver"]["converged"],
                choked=case.get("nozzle_state_consistency", {}).get(
                    "choked", False
                ),
                feasible=case.get("classification", {}).get("feasible", False),
                passed=case["passed"],
            )
        )
    lines.extend(
        [
            "",
            "## 逐站状态",
            "",
            "表中出口站使用静态量，其余站使用总量。",
            "",
            "| 设计点 | 站位 | 温度/K | 压力/Pa | 质量流量/(kg/s) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for case in report["cases"]:
        for station_name, station in case.get("stations", {}).items():
            temperature = station.get(
                "total_temperature_K", station.get("static_temperature_K")
            )
            pressure = station.get(
                "total_pressure_Pa", station.get("static_pressure_Pa")
            )
            lines.append(
                "| {case_id} | {station} | {temperature} | {pressure} | "
                "{mass_flow} |".format(
                    case_id=case["id"],
                    station=station_name,
                    temperature=formatted(temperature),
                    pressure=formatted(pressure),
                    mass_flow=formatted(station.get("mass_flow_kg_s")),
                )
            )
    lines.extend(
        [
            "",
            "## 部件闭合量",
            "",
            "| 设计点 | 部件 | 压比 | 效率 | 流量系数 | 负荷系数 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for case in report["cases"]:
        for component_name, component in case.get("components", {}).items():
            pressure_ratio = component.get(
                "pressure_ratio", component.get("total_pressure_ratio")
            )
            efficiency = component.get("total_to_total_efficiency")
            flow_coefficient = component.get(
                "flow_coefficient_ref", component.get("flow_coefficient")
            )
            lines.append(
                "| {case_id} | {component_name} | {pressure_ratio} | "
                "{efficiency} | {flow_coefficient} | {loading} |".format(
                    case_id=case["id"],
                    component_name=component_name,
                    pressure_ratio=formatted(pressure_ratio),
                    efficiency=formatted(efficiency),
                    flow_coefficient=formatted(flow_coefficient),
                    loading=formatted(component.get("loading_coefficient")),
                )
            )
    lines.extend(
        [
            "",
            "## 守恒残差",
            "",
            "| 设计点 | 燃烧能量/W | 轴功/W | 质量流量/(kg/s) | 归一化流量 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for case in report["cases"]:
        values = case.get("conservation", {}).get("values", {})
        lines.append(
            "| {id} | {energy:.6e} | {shaft:.6e} | {mass:.6e} | "
            "{flow:.6e} |".format(
                id=case["id"],
                energy=float(values.get("combustor_energy_W", math.nan)),
                shaft=float(values.get("shaft_power_W", math.nan)),
                mass=float(values.get("mass_flow_kg_s", math.nan)),
                flow=float(values.get("normalized_flow", math.nan)),
            )
        )
    lines.extend(
        [
            "",
            "## 喷管状态一致性残差",
            "",
            "| 设计点 | 总温重构/K | 总压重构/Pa | 速度重构/(m/s) | 堵塞Mach误差 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for case in report["cases"]:
        nozzle = case.get("nozzle_state_consistency", {})
        lines.append(
            "| {id} | {temperature} | {pressure} | {velocity} | "
            "{mach} |".format(
                id=case["id"],
                temperature=formatted(
                    nozzle.get("temperature_reconstruction_residual_K")
                ),
                pressure=formatted(
                    nozzle.get("pressure_reconstruction_residual_Pa")
                ),
                velocity=formatted(
                    nozzle.get("velocity_reconstruction_residual_m_s")
                ),
                mach=formatted(nozzle.get("choked_mach_residual")),
            )
        )
    lines.extend(
        [
            "",
            "## 当前软不可行项",
            "",
            "这些超限不影响 WP19-9 守恒门禁，但必须在标定阶段处理。",
            "",
            "| 设计点 | 超限约束 |",
            "|---|---|",
        ]
    )
    for case in report["cases"]:
        violations = case.get("constraints", {}).get("violations", {})
        names = ", ".join(f"`{name}`" for name in violations) or "无"
        lines.append(f"| {case['id']} | {names} |")
    lines.extend(
        [
            "",
            "## Gate 解释",
            "",
            "WP19-9 通过仅说明代表点上的逐站状态、守恒、喷管一致性以及软不可行语义正确。",
            "标定包仍为 bootstrap candidate，且 `formal_dataset_generation_ready=false`；",
            "因此可以进入 WP19-10 小规模 sanity DOE，但不能启动正式大规模数据生成。",
            "",
        ]
    )
    return "\n".join(lines)


def write_wp19_9_artifacts(
    report: Mapping[str, Any],
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write deterministic JSON and Markdown validation artifacts."""
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_target.write_text(
        render_wp19_9_markdown(report),
        encoding="utf-8",
    )

"""Read-only component-chain diagnostics for V2.2 mass-flow root scans."""

from __future__ import annotations

from collections import Counter
from functools import partial
import math
from typing import Any

import numpy as np

from src.ai.geometry19_pipeline_context import PipelineContext
from src.physics.combustor_map_enhanced import solve_combustor_map_enhanced
from src.physics.compressor_map_enhanced import solve_compressor_map_enhanced
from src.physics.engine_match import Environment, FailureCode, Geometry19, RootSearchConfig
from src.physics.geometry_state import derive_geometry_state
from src.physics.inlet_map_enhanced import solve_inlet_map_enhanced
from src.physics.maps import MapPolicy
from src.physics.nozzle_map_enhanced import solve_nozzle_map_enhanced
from src.physics.turbine_map_enhanced import solve_turbine_map_enhanced


class FlowScanDiagnosticError(ValueError):
    """Raised when flow diagnostics are requested for an incompatible backend."""


def _failure_code(error: Exception) -> str:
    raw = getattr(error, "failure_code", FailureCode.ROOT_SOLVER_FAILED)
    return raw.value if isinstance(raw, FailureCode) else str(raw)


def _map_snapshot(component: Any) -> dict[str, Any]:
    values = {"map_in_domain": bool(getattr(component, "map_in_domain", True))}
    for name in (
        "surge_margin",
        "choke_margin",
        "lean_stability_margin",
        "rich_stability_margin",
        "minimum_capacity_margin",
        "maximum_capacity_margin",
        "minimum_reynolds_margin",
        "maximum_reynolds_margin",
    ):
        if hasattr(component, name):
            values[name] = float(getattr(component, name))
    return values


def _normalized_domain_distance(component_map: Any, query: dict[str, float]) -> float:
    outside = []
    for name in component_map.coordinate_names:
        axis = component_map.coordinates[name]
        span = float(axis[-1] - axis[0])
        value = query[name]
        outside.append(max(float(axis[0]) - value, value - float(axis[-1]), 0.0) / span)
    return math.sqrt(sum(value * value for value in outside))


def diagnose_v2_2_c4_flow_scan(
    context: PipelineContext,
    design: Geometry19,
    environment: Environment | None = None,
) -> dict[str, Any]:
    """Replay the configured root scan while preserving first-component failures."""

    runtime = context.physics_runtime
    required_maps = {"compressor", "turbine", "inlet", "combustor", "nozzle"}
    if runtime.model_version != "geometry19_map_enhanced_v2_2_full_variable_thermo_shadow":
        raise FlowScanDiagnosticError("Flow diagnostics require the C4 backend")
    if runtime.thermo_model is None or set(runtime.component_maps) != required_maps:
        raise FlowScanDiagnosticError("C4 diagnostic dependencies are incomplete")
    env = environment or Environment()
    config = runtime.model_config
    geometry = derive_geometry_state(design, config)
    maps = runtime.component_maps
    thermo = runtime.thermo_model
    inlet_solver = partial(
        solve_inlet_map_enhanced,
        component_map=maps["inlet"],
        thermo=thermo,
        map_policy=MapPolicy.DIAGNOSTIC,
    )
    compressor_solver = partial(
        solve_compressor_map_enhanced,
        component_map=maps["compressor"],
        thermo=thermo,
        map_policy=MapPolicy.DIAGNOSTIC,
    )
    combustor_solver = partial(
        solve_combustor_map_enhanced,
        component_map=maps["combustor"],
        thermo=thermo,
        map_policy=MapPolicy.DIAGNOSTIC,
    )
    turbine_solver = partial(
        solve_turbine_map_enhanced,
        component_map=maps["turbine"],
        thermo=thermo,
        map_policy=MapPolicy.DIAGNOSTIC,
    )
    nozzle_solver = partial(
        solve_nozzle_map_enhanced,
        component_map=maps["nozzle"],
        thermo=thermo,
        map_policy=MapPolicy.DIAGNOSTIC,
    )
    inlet, _, inlet_capacity = inlet_solver(geometry, env, config)
    lower = float(config.solver_candidate["mass_flow_lower_fraction"]) * inlet_capacity
    upper = float(config.solver_candidate["mass_flow_upper_fraction"]) * inlet_capacity
    search = RootSearchConfig.from_model_config(config)
    scan_points = np.unique(
        np.concatenate(
            (
                np.linspace(lower, upper, search.linear_scan_points),
                np.geomspace(lower, upper, search.logarithmic_scan_points),
            )
        )
    )
    failure_components: Counter[str] = Counter()
    failure_codes: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    compressor_queries = []
    valid_trials: list[dict[str, Any]] = []
    for scan_index, raw_mass_flow in enumerate(scan_points):
        mass_flow = float(raw_mass_flow)
        chain: dict[str, Any] = {}
        compressor_map = maps["compressor"]
        reference = compressor_map.metadata["reference_state"]
        rho_ref = inlet.total_pressure_Pa / (
            thermo.gas_constant_air_J_kg_K * inlet.total_temperature_K
        )
        meridional_velocity = mass_flow / (
            rho_ref * geometry.compressor_exit_effective_area_m2
        )
        compressor_query = {
            "corrected_speed_ratio": (
                geometry.design.shaft_speed_rpm
                * math.sqrt(
                    float(reference["temperature_K"])
                    / inlet.total_temperature_K
                )
                / float(reference["corrected_speed_design_rpm"])
            ),
            "flow_coefficient_ratio": (
                meridional_velocity
                / geometry.compressor_tip_speed_m_s
                / float(reference["phi_design"])
            ),
        }
        query_distance = _normalized_domain_distance(
            compressor_map, compressor_query
        )
        compressor_queries.append({**compressor_query, "domain_distance": query_distance})
        try:
            chain["compressor"] = compressor_solver(
                geometry, mass_flow, inlet, config
            )
        except (ArithmeticError, ValueError) as exc:
            failure_components["compressor"] += 1
            code = _failure_code(exc)
            failure_codes[code] += 1
            reason = (
                "compressor_query_outside_diagnostic_domain"
                if query_distance > compressor_map.diagnostic_extrapolation_limit
                else f"compressor_{code.lower()}"
            )
            failure_reasons[reason] += 1
            continue
        try:
            chain["combustor"] = combustor_solver(
                geometry, mass_flow, chain["compressor"].outlet, config
            )
        except (ArithmeticError, ValueError) as exc:
            failure_components["combustor"] += 1
            code = _failure_code(exc)
            failure_codes[code] += 1
            failure_reasons[f"combustor_{code.lower()}"] += 1
            continue
        try:
            chain["turbine"] = turbine_solver(
                geometry,
                chain["combustor"],
                chain["compressor"].shaft_power_W,
                config,
            )
        except (ArithmeticError, ValueError) as exc:
            failure_components["turbine"] += 1
            code = _failure_code(exc)
            failure_codes[code] += 1
            failure_reasons[f"turbine_{code.lower()}"] += 1
            continue
        try:
            chain["nozzle"] = nozzle_solver(
                geometry, chain["turbine"], env, config
            )
        except (ArithmeticError, ValueError) as exc:
            failure_components["nozzle"] += 1
            code = _failure_code(exc)
            failure_codes[code] += 1
            failure_reasons[f"nozzle_{code.lower()}"] += 1
            continue
        residual = float(chain["nozzle"].normalized_flow_residual)
        if not math.isfinite(residual):
            failure_components["nozzle"] += 1
            failure_codes[FailureCode.NONFINITE_STATE.value] += 1
            failure_reasons["nozzle_nonfinite_residual"] += 1
            continue
        valid_trials.append(
            {
                "scan_index": scan_index,
                "mass_flow_air_kg_s": mass_flow,
                "normalized_flow_residual": residual,
                "components": {
                    name: _map_snapshot(value) for name, value in chain.items()
                },
            }
        )

    sign_change_brackets = 0
    for left, right in zip(valid_trials[:-1], valid_trials[1:]):
        if (
            right["scan_index"] == left["scan_index"] + 1
            and left["normalized_flow_residual"]
            * right["normalized_flow_residual"]
            < 0.0
        ):
            sign_change_brackets += 1
    residuals = [row["normalized_flow_residual"] for row in valid_trials]
    return {
        "scan_point_count": len(scan_points),
        "mass_flow_search_min_kg_s": lower,
        "mass_flow_search_max_kg_s": upper,
        "valid_trial_count": len(valid_trials),
        "failure_trial_count": len(scan_points) - len(valid_trials),
        "first_failure_component_counts": dict(sorted(failure_components.items())),
        "failure_code_counts": dict(sorted(failure_codes.items())),
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
        "compressor_query_envelope": {
            name: {
                "minimum": min(row[name] for row in compressor_queries),
                "maximum": max(row[name] for row in compressor_queries),
            }
            for name in (
                "corrected_speed_ratio",
                "flow_coefficient_ratio",
                "domain_distance",
            )
        },
        "negative_residual_count": sum(value < 0.0 for value in residuals),
        "positive_residual_count": sum(value > 0.0 for value in residuals),
        "minimum_normalized_flow_residual": min(residuals) if residuals else None,
        "maximum_normalized_flow_residual": max(residuals) if residuals else None,
        "closest_absolute_normalized_flow_residual": (
            min(abs(value) for value in residuals) if residuals else None
        ),
        "sign_change_bracket_count": sign_change_brackets,
        "valid_trials": valid_trials,
    }

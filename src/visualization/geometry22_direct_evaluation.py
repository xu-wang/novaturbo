"""Single-design V3 development physics evaluation for the interactive UI."""
from __future__ import annotations

import math
from pathlib import Path

from src.physics.constraint_profiles import ConstraintProfile, assess_constraint_profile
from src.physics.geometry22_design import Geometry22Design, geometry22_schema_metadata
from src.physics.v3_development_runtime import build_v3_development_physics_runtime


DIAGNOSTIC_PROFILE = "config/physics/geometry19_constraint_profile_pipeline_debug_v1.yaml"
OUTPUTS = ("thrust_N", "tsfc_kg_N_s", "mass_kg")


def finite(value):
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def evaluate_geometry22_design(repository_root: str | Path, coordinates: dict) -> dict:
    """Evaluate exactly the submitted 22D coordinates, without inverse-design search."""
    if not isinstance(coordinates, dict):
        raise ValueError("Provide a 22D design object")
    design = Geometry22Design.from_mapping(coordinates)
    root = Path(repository_root).resolve()
    runtime = build_v3_development_physics_runtime(root)
    result = runtime.evaluate(design)
    engine = result.engine_result
    profile = ConstraintProfile.from_yaml(root / DIAGNOSTIC_PROFILE)
    diagnostic = assess_constraint_profile(
        engine, profile, conservation_tolerances=runtime.base_runtime.conservation_tolerances
    ).as_mapping()
    performance = ({name: finite(engine.performance[name]) for name in OUTPUTS}
                   if engine.solver.converged and all(name in engine.performance for name in OUTPUTS) else None)
    constraints = {name: finite(value) for name, value in sorted(engine.constraints.items())}
    compressor = engine.components.get("compressor", {})
    combustor = engine.components.get("combustor", {})
    return {
        "mode": "v3_single_design_development_evaluation",
        "design_id": result.design_id,
        "geometry22": design.as_mapping(),
        "schema": geometry22_schema_metadata(),
        "physics_contract": runtime.contract_mapping(),
        "solver": {
            "converged": bool(engine.solver.converged),
            "failure_code": engine.solver.failure_code.value if engine.solver.failure_code else None,
            "message": engine.solver.message,
            "flow_residual": finite(engine.solver.flow_residual),
            "iterations": engine.solver.iterations,
            "evaluations": engine.solver.evaluations,
        },
        "performance": performance,
        "diagnostic": diagnostic,
        "feasible": bool(engine.feasible),
        "feasibility_complete": bool(engine.feasibility_complete),
        "feasibility_scope": engine.feasibility_scope,
        "constraints": constraints,
        "v3_physics": {
            "diffuser_to_combustor_area_residual": finite(
                engine.assembly_constraints.get("diffuser_to_combustor_area")
            ),
            "combustor_primary_residence_time_s": finite(
                combustor.get("primary_residence_time_s")
            ),
            "compressor_total_to_total_efficiency": finite(
                compressor.get("total_to_total_efficiency")
            ),
            "compressor_pressure_ratio": finite(compressor.get("pressure_ratio")),
        },
        "physics_evaluation_count": 1,
        "physics_validation_status": "not_completed",
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }

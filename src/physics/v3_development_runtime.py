"""Isolated 22D V3 development backend built on the frozen V2.2 component chain."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.physics.compressor_meanline import CompressorModelError
from src.physics.engine_match import EngineResult, Environment, FailureCode, StationState
from src.physics.geometry22_design import (
    Geometry22Design,
    SCHEMA_PATH,
    geometry22_schema_hash,
)
from src.physics.runtime import (
    PhysicsRuntime,
    V22ShadowPhysicsBackend,
    build_physics_runtime,
)


MODEL_VERSION = "geometry22_map_enhanced_v3_development_v1"
BACKEND_ID = "geometry22_v3_development_backend"
CORRECTION_PATH = "config/physics/geometry22_v3_development_correction_v1.yaml"
R0_CONTRACT_PATH = "docs/physics/V3_R0_22D_DEVELOPMENT_CONTRACT_20260923.yaml"


class V3DevelopmentRuntimeError(ValueError):
    """V3 development contract or design identity failure."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_hash(payload: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class V3DevelopmentResult:
    """Complete 22D identity around an internal 19D-compatible engine result."""

    design: Geometry22Design
    engine_result: EngineResult
    physics_bundle_hash: str
    design_schema_hash: str
    correction_hash: str
    input_parameters: Mapping[str, float]
    backend_id: str = BACKEND_ID
    physics_model_version: str = MODEL_VERSION
    physics_validation_status: str = "not_completed"
    formal_use_allowed: bool = False
    deployment_allowed: bool = False

    @property
    def design_id(self) -> str:
        return self.design.design_id()

    @property
    def geometry19_projection(self):
        return self.design.geometry19_projection()


@dataclass(frozen=True)
class V3DevelopmentPhysicsRuntime:
    repository_root: Path
    base_runtime: PhysicsRuntime
    design_schema_hash: str
    r0_contract_hash: str
    correction_hash: str
    code_hash: str
    correction_reference_ratio: float
    correction_slope: float
    physics_bundle_hash: str
    backend_id: str = BACKEND_ID
    model_version: str = MODEL_VERSION

    def contract_mapping(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "physics_model_version": self.model_version,
            "design_schema_version": "geometry22_schema_v3_development_v1",
            "design_schema_hash": self.design_schema_hash,
            "r0_contract_hash": self.r0_contract_hash,
            "correction_hash": self.correction_hash,
            "code_hash": self.code_hash,
            "base_physics_bundle_hash": self.base_runtime.physics_bundle_hash,
            "physics_bundle_hash": self.physics_bundle_hash,
            "physics_validation_status": "not_completed",
            "formal_use_allowed": False,
            "deployment_allowed": False,
        }

    def evaluate(
        self,
        design: Geometry22Design,
        environment: Environment | None = None,
    ) -> V3DevelopmentResult:
        if not isinstance(design, Geometry22Design):
            raise V3DevelopmentRuntimeError("V3 requires an explicit Geometry22Design")
        if geometry22_schema_hash() != self.design_schema_hash:
            raise V3DevelopmentRuntimeError("V3 design schema changed after runtime construction")
        if _sha256(self.repository_root / CORRECTION_PATH) != self.correction_hash:
            raise V3DevelopmentRuntimeError("V3 correction config changed after runtime construction")
        if _sha256(self.repository_root / R0_CONTRACT_PATH) != self.r0_contract_hash:
            raise V3DevelopmentRuntimeError("V3 R0 contract changed after runtime construction")
        params = design.component_parameters()
        config = self.base_runtime.model_config.with_overrides(
            parameter_values={
                "diffuser_area_ratio": params["diffuser_area_ratio"],
                "combustor_primary_air_fraction": params["combustor_primary_air_fraction"],
            }
        )
        ratio = params["compressor_tip_clearance_ratio"]
        base_backend = self.base_runtime.backend
        if not isinstance(base_backend, V22ShadowPhysicsBackend):
            raise V3DevelopmentRuntimeError("V3 base backend is not the V2.2 component chain")
        thermo = base_backend.thermo_model

        def v3_solver(x19, env, effective_config, **solvers):
            base_compressor_solver = solvers["compressor_solver"]

            def v3_compressor_solver(geometry, mass_flow, inlet, component_config):
                baseline = base_compressor_solver(
                    geometry, mass_flow, inlet, component_config
                )
                efficiency = baseline.total_to_total_efficiency - self.correction_slope * (
                    ratio - self.correction_reference_ratio
                )
                if not math.isfinite(efficiency) or not 0.0 < efficiency < 1.0:
                    raise CompressorModelError(
                        FailureCode.COMPRESSOR_EFFICIENCY_INVALID,
                        "V3 development clearance correction produced invalid efficiency",
                    )
                try:
                    inlet_enthalpy = thermo.h_air(inlet.total_temperature_K)
                    isentropic_temperature = thermo.air_temperature_from_h(
                        inlet_enthalpy + efficiency * baseline.specific_work_J_kg
                    )
                    pressure_ratio = thermo.air_isentropic_pressure_ratio(
                        inlet.total_temperature_K, isentropic_temperature
                    )
                except (ArithmeticError, ValueError) as exc:
                    raise CompressorModelError(
                        FailureCode.COMPRESSOR_EFFICIENCY_INVALID,
                        "V3 development clearance correction failed",
                    ) from exc
                if not math.isfinite(pressure_ratio) or pressure_ratio <= 0.0:
                    raise CompressorModelError(
                        FailureCode.COMPRESSOR_EFFICIENCY_INVALID,
                        "V3 development clearance correction produced invalid pressure ratio",
                    )
                residuals = dict(baseline.calibration_domain_residuals)
                residuals["compressor_efficiency_min"] = (
                    float(component_config.require("compressor_efficiency_min"))
                    - efficiency
                )
                return replace(
                    baseline,
                    outlet=StationState(
                        baseline.outlet.total_temperature_K,
                        pressure_ratio * inlet.total_pressure_Pa,
                    ),
                    total_to_total_efficiency=efficiency,
                    pressure_ratio=pressure_ratio,
                    calibration_domain_residuals=residuals,
                    calibration_domain_feasible=all(
                        value <= 0.0 for value in residuals.values()
                    ),
                )

            solvers["compressor_solver"] = v3_compressor_solver
            return base_backend.solver(x19, env, effective_config, **solvers)

        selected_backend = replace(base_backend, solver=v3_solver)
        base_result = selected_backend.evaluate(
            design.geometry19_projection(),
            environment or Environment(),
            config,
            self.base_runtime.operating_point_policy,
        )
        components = {name: dict(values) for name, values in base_result.components.items()}
        if "compressor" in components:
            components["compressor"].update(
                {
                    "tip_clearance_ratio": ratio,
                    "tip_clearance_reference_ratio": self.correction_reference_ratio,
                    "tip_clearance_efficiency_slope": self.correction_slope,
                    "tip_clearance_correction_status": "software_fixture_unvalidated",
                }
            )
        if "combustor" in components:
            components["combustor"]["primary_air_fraction_design"] = params[
                "combustor_primary_air_fraction"
            ]
        result = replace(
            base_result,
            physics_model_version=self.model_version,
            components=components,
            formal_dataset_generation_ready=False,
        )
        return V3DevelopmentResult(
            design=design,
            engine_result=result,
            physics_bundle_hash=self.physics_bundle_hash,
            design_schema_hash=self.design_schema_hash,
            correction_hash=self.correction_hash,
            input_parameters=params,
        )


def build_v3_development_physics_runtime(
    repository_root: str | Path,
) -> V3DevelopmentPhysicsRuntime:
    """Resolve the V2.2 component base through the shared physics builder."""

    root = Path(repository_root).resolve()
    if root != SCHEMA_PATH.parents[2].resolve():
        raise V3DevelopmentRuntimeError("V3 schema repository differs from runtime repository")
    config_path = root / CORRECTION_PATH
    contract_path = root / R0_CONTRACT_PATH
    correction = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    r0_contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if correction.get("model_version") != MODEL_VERSION:
        raise V3DevelopmentRuntimeError("V3 model version mismatch")
    if correction.get("status") != "software_development_fixture_unvalidated":
        raise V3DevelopmentRuntimeError("V3 correction status mismatch")
    if correction.get("development_runtime_allowed") is not True:
        raise V3DevelopmentRuntimeError("V3 development runtime is not enabled by correction config")
    if correction.get("formal_use_allowed") is not False or correction.get("deployment_allowed") is not False:
        raise V3DevelopmentRuntimeError("V3 development correction cannot enable formal use")
    base_runtime = build_physics_runtime(root, correction["base_runtime"])
    if not isinstance(base_runtime.backend, V22ShadowPhysicsBackend):
        raise V3DevelopmentRuntimeError("V3 requires the V2.2 full component backend")
    if set(base_runtime.component_maps) != {
        "compressor", "turbine", "inlet", "combustor", "nozzle"
    }:
        raise V3DevelopmentRuntimeError("V3 requires the full V2.2 component map set")
    if r0_contract.get("target_dimension") != 22:
        raise V3DevelopmentRuntimeError("V3 R0 contract dimension mismatch")
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    if r0_contract.get("status") != "development_contract_frozen_physics_unvalidated":
        raise V3DevelopmentRuntimeError("V3 R0 contract status mismatch")
    if schema.get("runtime_enabled") is not False:
        raise V3DevelopmentRuntimeError("V3 R1 schema must remain development-only")
    if schema.get("single_design_development_runtime_available") is not True:
        raise V3DevelopmentRuntimeError("V3 single-design development runtime is disabled")
    for field, decision in zip(
        schema["design_variables"][19:],
        r0_contract["design_variables_append"],
        strict=True,
    ):
        if (
            field["index"] != decision["index"]
            or field["name"] != decision["name"]
            or field["bounds"] != decision["prototype_bounds"]
            or field["reference_value"] != decision["prototype_reference_value"]
        ):
            raise V3DevelopmentRuntimeError("V3 schema differs from R0 variable contract")
    tip = correction["compressor_tip_clearance"]
    reference = float(tip["reference_ratio"])
    slope = float(tip["efficiency_slope_per_unit_ratio"])
    bounds = r0_contract["design_variables_append"][2]["prototype_bounds"]
    if not (math.isfinite(slope) and slope > 0.0 and bounds[0] <= reference <= bounds[1]):
        raise V3DevelopmentRuntimeError("V3 clearance correction is outside its contract")
    if reference != r0_contract["design_variables_append"][2]["prototype_reference_value"]:
        raise V3DevelopmentRuntimeError("V3 clearance reference differs from R0")
    code_hash = _bundle_hash(
        {
            relative: _sha256(root / relative)
            for relative in (
                "src/physics/v3_development_runtime.py",
                "src/physics/geometry22_design.py",
            )
        }
    )
    identity = {
        "base_physics_bundle_hash": base_runtime.physics_bundle_hash,
        "design_schema_hash": geometry22_schema_hash(),
        "r0_contract_hash": _sha256(contract_path),
        "correction_hash": _sha256(config_path),
        "code_hash": code_hash,
        "model_version": MODEL_VERSION,
    }
    return V3DevelopmentPhysicsRuntime(
        repository_root=root,
        base_runtime=base_runtime,
        design_schema_hash=identity["design_schema_hash"],
        r0_contract_hash=identity["r0_contract_hash"],
        correction_hash=identity["correction_hash"],
        code_hash=code_hash,
        correction_reference_ratio=reference,
        correction_slope=slope,
        physics_bundle_hash=_bundle_hash(identity),
    )

"""Single construction and evaluation boundary for Geometry19 physics.

The runtime deliberately owns all repository-backed physics configuration.
Component solvers receive validated objects and never resolve configuration
paths themselves. V2.1 and the non-deliverable V2.2 shadow backend share this
same construction boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

import yaml

from src.physics.constraint_profiles import ConstraintProfile
from src.physics.engine_match import (
    EngineResult,
    Environment,
    Geometry19,
    OperatingPointSelectionPolicy,
    solve_design_point,
)
from src.physics.model_config import ModelConfig
from src.physics.maps import (
    ComponentMap,
    ComponentMapRegistry,
    ComponentMapRegistryError,
    MapPolicy,
    load_component_map_registry,
)
from src.physics.thermo import ThermoModel, load_variable_property_thermo


PhysicsSolver = Callable[..., EngineResult]


class PhysicsRuntimeError(ValueError):
    """Raised when a physics runtime cannot be built from one frozen contract."""


class PhysicsBackend(Protocol):
    """Backend contract hidden behind the single runtime evaluation entry."""

    backend_id: str
    model_version: str

    def evaluate(
        self,
        design: Geometry19,
        environment: Environment,
        model_config: ModelConfig,
        operating_point_policy: OperatingPointSelectionPolicy,
    ) -> EngineResult: ...


@dataclass(frozen=True)
class V21PhysicsBackend:
    """Compatibility adapter around the frozen Geometry19 V2.1 solver."""

    solver: PhysicsSolver = solve_design_point
    backend_id: str = "geometry19_v2_1_backend"
    model_version: str = "geometry19_closed_v2_1"

    def evaluate(
        self,
        design: Geometry19,
        environment: Environment,
        model_config: ModelConfig,
        operating_point_policy: OperatingPointSelectionPolicy,
    ) -> EngineResult:
        return self.solver(
            design,
            environment,
            model_config,
            operating_point_policy=operating_point_policy,
        )


@dataclass(frozen=True)
class V22ShadowPhysicsBackend:
    """One-root shadow backend with registered component-map closures."""

    compressor_map: ComponentMap
    thermo_model: ThermoModel
    turbine_map: ComponentMap | None = None
    inlet_map: ComponentMap | None = None
    combustor_map: ComponentMap | None = None
    nozzle_map: ComponentMap | None = None
    variable_property_turbine: bool = False
    solver: PhysicsSolver = solve_design_point
    backend_id: str = "geometry19_v2_2_shadow_backend"
    model_version: str = "geometry19_map_enhanced_v2_2_shadow"

    def evaluate(
        self,
        design: Geometry19,
        environment: Environment,
        model_config: ModelConfig,
        operating_point_policy: OperatingPointSelectionPolicy,
    ) -> EngineResult:
        from src.physics.compressor_map_enhanced import (
            solve_compressor_map_enhanced,
        )

        compressor_solver = partial(
            solve_compressor_map_enhanced,
            component_map=self.compressor_map,
            thermo=self.thermo_model,
            map_policy=MapPolicy.DIAGNOSTIC,
        )
        turbine_solver = None
        if self.turbine_map is not None:
            from src.physics.turbine_map_enhanced import (
                solve_turbine_map_enhanced,
            )

            turbine_solver = partial(
                solve_turbine_map_enhanced,
                component_map=self.turbine_map,
                thermo=(self.thermo_model if self.variable_property_turbine else None),
                map_policy=MapPolicy.DIAGNOSTIC,
            )
        inlet_solver = None
        if self.inlet_map is not None:
            from src.physics.inlet_map_enhanced import solve_inlet_map_enhanced

            inlet_solver = partial(
                solve_inlet_map_enhanced,
                component_map=self.inlet_map,
                thermo=self.thermo_model,
                map_policy=MapPolicy.DIAGNOSTIC,
            )
        combustor_solver = None
        if self.combustor_map is not None:
            from src.physics.combustor_map_enhanced import (
                solve_combustor_map_enhanced,
            )

            combustor_solver = partial(
                solve_combustor_map_enhanced,
                component_map=self.combustor_map,
                thermo=self.thermo_model,
                map_policy=MapPolicy.DIAGNOSTIC,
            )
        nozzle_solver = None
        if self.nozzle_map is not None:
            from src.physics.nozzle_map_enhanced import solve_nozzle_map_enhanced

            nozzle_solver = partial(
                solve_nozzle_map_enhanced,
                component_map=self.nozzle_map,
                thermo=self.thermo_model,
                map_policy=MapPolicy.DIAGNOSTIC,
            )
        result = self.solver(
            design,
            environment,
            model_config,
            operating_point_policy=operating_point_policy,
            inlet_solver=inlet_solver,
            compressor_solver=compressor_solver,
            combustor_solver=combustor_solver,
            turbine_solver=turbine_solver,
            nozzle_solver=nozzle_solver,
        )
        return replace(result, physics_model_version=self.model_version)


@dataclass(frozen=True)
class PhysicsRuntime:
    """Fully resolved immutable physics dependencies for one pipeline run."""

    repository_root: Path
    backend: PhysicsBackend
    model_config: ModelConfig
    design_schema: Mapping[str, Any]
    constraint_profile: ConstraintProfile
    operating_point_policy: OperatingPointSelectionPolicy
    conservation_tolerances: Mapping[str, Any]
    thermo_model: ThermoModel | None
    component_map_registry: ComponentMapRegistry | None
    component_maps: Mapping[str, ComponentMap]
    source_paths: Mapping[str, Path]
    provenance_hashes: Mapping[str, str]
    physics_bundle_hash: str

    @property
    def backend_id(self) -> str:
        return self.backend.backend_id

    @property
    def model_version(self) -> str:
        return self.backend.model_version

    def evaluate(
        self,
        design: Geometry19,
        environment: Environment | None = None,
    ) -> EngineResult:
        """Evaluate through the only solver entry exposed to pipeline stages."""

        return self.backend.evaluate(
            design,
            environment or Environment(),
            self.model_config,
            self.operating_point_policy,
        )

    def contract_mapping(self) -> dict[str, Any]:
        """Return the stable runtime identity stored in pipeline manifests."""

        component_map_contracts = {}
        for name, component_map in self.component_maps.items():
            contract = {
                "map_id": component_map.map_id,
                "map_version": component_map.map_version,
                "source_hash": component_map.source_hash,
            }
            if self.component_map_registry is not None:
                registration = self.component_map_registry.registration(
                    name, component_map.map_id
                )
                contract.update(
                    {
                        "usage_level": registration.usage_level,
                        "formal_use_allowed": registration.formal_use_allowed,
                    }
                )
            component_map_contracts[name] = contract
        mapping = {
            "backend_id": self.backend_id,
            "physics_model_version": self.model_version,
            "design_schema_version": str(self.design_schema["schema_version"]),
            "constraint_profile": self.constraint_profile.profile_version,
            "operating_point_policy": self.operating_point_policy.value,
            "physics_bundle_hash": self.physics_bundle_hash,
            "provenance_hashes": dict(self.provenance_hashes),
            "thermo_model_id": (
                self.thermo_model.model_id if self.thermo_model is not None else None
            ),
            "component_maps": component_map_contracts,
        }
        if self.component_map_registry is not None:
            mapping["component_map_registry"] = {
                "registry_version": self.component_map_registry.registry_version,
                "source_hash": self.component_map_registry.source_hash,
            }
        return mapping


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PhysicsRuntimeError(f"Cannot load {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise PhysicsRuntimeError(f"{label} must be a YAML mapping: {path}")
    return payload


def _resolve_contract_path(
    root: Path, value: Any, label: str, *, allow_directory: bool = False
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PhysicsRuntimeError(f"contracts.{label} must be a non-empty path")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PhysicsRuntimeError(f"contracts.{label} escapes the repository root") from exc
    exists = path.is_dir() if allow_directory else path.is_file()
    if not exists:
        raise PhysicsRuntimeError(f"contracts.{label} does not exist: {path}")
    return path


def build_physics_runtime(
    repository_root: str | Path,
    resolved_contract: Mapping[str, Any],
    *,
    solver: PhysicsSolver | None = None,
) -> PhysicsRuntime:
    """Build one registered physics backend from an already parsed protocol."""

    root = Path(repository_root).resolve()
    contracts = resolved_contract.get("contracts")
    if not isinstance(contracts, Mapping):
        raise PhysicsRuntimeError("Pipeline protocol has no contracts mapping")

    physics_section = resolved_contract.get("physics_evaluation")
    if not isinstance(physics_section, Mapping):
        raise PhysicsRuntimeError("Pipeline protocol has no physics_evaluation mapping")
    model_version = str(physics_section.get("model_tag", ""))
    supported_versions = {
        "geometry19_closed_v2_1",
        "geometry19_map_enhanced_v2_2_shadow",
        "geometry19_map_enhanced_v2_2_comp_turb_shadow",
        "geometry19_map_enhanced_v2_2_comp_turb_inlet_shadow",
        "geometry19_map_enhanced_v2_2_comp_turb_inlet_comb_shadow",
        "geometry19_map_enhanced_v2_2_full_component_shadow",
        "geometry19_map_enhanced_v2_2_full_variable_thermo_shadow",
    }
    if model_version not in supported_versions:
        raise PhysicsRuntimeError(f"Unsupported physics model version: {model_version!r}")

    required = [
        "design_schema",
        "calibration_pack",
        "acceptance_policy",
        "constraint_profile",
        "operating_point_policy",
        "conservation_tolerances",
    ]
    if model_version != "geometry19_closed_v2_1":
        required.extend(("thermo_model", "component_map_registry"))
    source_paths = {
        name: _resolve_contract_path(
            root,
            contracts.get(name),
            name,
            allow_directory=name.endswith("_map"),
        )
        for name in required
    }
    design_schema = _load_yaml_mapping(source_paths["design_schema"], "design schema")
    conservation_tolerances = _load_yaml_mapping(
        source_paths["conservation_tolerances"], "conservation tolerances"
    )
    operating_point = _load_yaml_mapping(
        source_paths["operating_point_policy"], "operating-point policy"
    )
    try:
        operating_point_policy = OperatingPointSelectionPolicy(
            str(operating_point["solver_policy"])
        )
    except (KeyError, ValueError) as exc:
        raise PhysicsRuntimeError("Unsupported operating-point solver policy") from exc

    model_config = ModelConfig.from_yaml(source_paths["calibration_pack"])
    constraint_profile = ConstraintProfile.from_yaml(
        source_paths["constraint_profile"]
    )
    expected_calibration_version = (
        model_version
        if model_version == "geometry19_closed_v2_1"
        else str(physics_section.get("base_calibration_model_version", ""))
    )
    if model_config.physics_spec_version != expected_calibration_version:
        raise PhysicsRuntimeError(
            "Calibration physics_spec_version differs from the backend calibration contract"
        )
    schema_version = str(design_schema.get("schema_version", ""))
    if schema_version != model_config.design_schema_version:
        raise PhysicsRuntimeError(
            "Design schema version differs from the calibration-pack contract"
        )

    resolved_solver = solver or solve_design_point
    thermo_model: ThermoModel | None = None
    component_map_registry: ComponentMapRegistry | None = None
    component_maps: dict[str, ComponentMap] = {}
    if model_version == "geometry19_closed_v2_1":
        backend: PhysicsBackend = V21PhysicsBackend(solver=resolved_solver)
    else:
        thermo_model = load_variable_property_thermo(source_paths["thermo_model"])
        try:
            component_map_registry = load_component_map_registry(
                root, source_paths["component_map_registry"]
            )
            component_maps.update(
                component_map_registry.resolve(
                    physics_section.get("component_maps", {})
                )
            )
        except ComponentMapRegistryError as exc:
            raise PhysicsRuntimeError(
                f"Cannot resolve component maps: {exc}"
            ) from exc
        expected_components = {
            "geometry19_map_enhanced_v2_2_shadow": {"compressor"},
            "geometry19_map_enhanced_v2_2_comp_turb_shadow": {
                "compressor",
                "turbine",
            },
            "geometry19_map_enhanced_v2_2_comp_turb_inlet_shadow": {
                "compressor",
                "turbine",
                "inlet",
            },
            "geometry19_map_enhanced_v2_2_comp_turb_inlet_comb_shadow": {
                "compressor",
                "turbine",
                "inlet",
                "combustor",
            },
            "geometry19_map_enhanced_v2_2_full_component_shadow": {
                "compressor",
                "turbine",
                "inlet",
                "combustor",
                "nozzle",
            },
            "geometry19_map_enhanced_v2_2_full_variable_thermo_shadow": {
                "compressor",
                "turbine",
                "inlet",
                "combustor",
                "nozzle",
            },
        }[model_version]
        if set(component_maps) != expected_components:
            raise PhysicsRuntimeError(
                "V2.2 shadow component-map selection mismatch"
            )
        compressor_map = component_maps["compressor"]
        source_paths["compressor_map"] = component_map_registry.registration(
            "compressor", compressor_map.map_id
        ).package_path
        turbine_map = component_maps.get("turbine")
        if turbine_map is not None:
            source_paths["turbine_map"] = component_map_registry.registration(
                "turbine", turbine_map.map_id
            ).package_path
        inlet_map = component_maps.get("inlet")
        if inlet_map is not None:
            source_paths["inlet_map"] = component_map_registry.registration(
                "inlet", inlet_map.map_id
            ).package_path
        combustor_map = component_maps.get("combustor")
        if combustor_map is not None:
            source_paths["combustor_map"] = component_map_registry.registration(
                "combustor", combustor_map.map_id
            ).package_path
        nozzle_map = component_maps.get("nozzle")
        if nozzle_map is not None:
            source_paths["nozzle_map"] = component_map_registry.registration(
                "nozzle", nozzle_map.map_id
            ).package_path
        backend = V22ShadowPhysicsBackend(
            compressor_map=compressor_map,
            thermo_model=thermo_model,
            turbine_map=turbine_map,
            inlet_map=inlet_map,
            combustor_map=combustor_map,
            nozzle_map=nozzle_map,
            variable_property_turbine=(
                model_version
                == "geometry19_map_enhanced_v2_2_full_variable_thermo_shadow"
            ),
            solver=resolved_solver,
            backend_id=(
                "geometry19_v2_2_full_variable_thermo_shadow_backend"
                if model_version
                == "geometry19_map_enhanced_v2_2_full_variable_thermo_shadow"
                else "geometry19_v2_2_full_component_shadow_backend"
                if nozzle_map is not None
                else (
                    "geometry19_v2_2_comp_turb_inlet_comb_shadow_backend"
                    if combustor_map is not None
                    else (
                        "geometry19_v2_2_comp_turb_inlet_shadow_backend"
                        if inlet_map is not None
                        else (
                            "geometry19_v2_2_comp_turb_shadow_backend"
                            if turbine_map is not None
                            else "geometry19_v2_2_shadow_backend"
                        )
                    )
                )
            ),
            model_version=model_version,
        )
    physics_source_files = [
        "src/physics/engine_match.py",
        "src/physics/geometry_state.py",
        "src/physics/compressor_meanline.py",
        "src/physics/combustor_loworder.py",
        "src/physics/turbine_meanline.py",
        "src/physics/nozzle.py",
        "src/physics/mass_model_v2.py",
        "src/physics/formal_constraints_v2.py",
        "src/physics/materials.py",
        "src/physics/runtime.py",
    ]
    if model_version != "geometry19_closed_v2_1":
        physics_source_files.extend(
            (
                "src/physics/thermo.py",
                "src/physics/maps/__init__.py",
                "src/physics/maps/base.py",
                "src/physics/maps/registry.py",
                "src/physics/compressor_map_enhanced.py",
            )
        )
    if model_version == "geometry19_map_enhanced_v2_2_comp_turb_shadow":
        physics_source_files.append("src/physics/turbine_map_enhanced.py")
    if model_version == "geometry19_map_enhanced_v2_2_comp_turb_inlet_shadow":
        physics_source_files.extend(
            (
                "src/physics/turbine_map_enhanced.py",
                "src/physics/inlet_map_enhanced.py",
            )
        )
    if model_version == "geometry19_map_enhanced_v2_2_comp_turb_inlet_comb_shadow":
        physics_source_files.extend(
            (
                "src/physics/turbine_map_enhanced.py",
                "src/physics/inlet_map_enhanced.py",
                "src/physics/combustor_map_enhanced.py",
            )
        )
    if model_version in {
        "geometry19_map_enhanced_v2_2_full_component_shadow",
        "geometry19_map_enhanced_v2_2_full_variable_thermo_shadow",
    }:
        physics_source_files.extend(
            (
                "src/physics/turbine_map_enhanced.py",
                "src/physics/inlet_map_enhanced.py",
                "src/physics/combustor_map_enhanced.py",
                "src/physics/nozzle_map_enhanced.py",
            )
        )
    physics_code_hash = _canonical_hash(
        {relative: _sha256_file(root / relative) for relative in physics_source_files}
    )
    solver_binding_hash = _canonical_hash(
        {
            "module": str(getattr(resolved_solver, "__module__", "unknown")),
            "qualname": str(getattr(resolved_solver, "__qualname__", "unknown")),
        }
    )
    provenance = {
        "design_schema_hash": _sha256_file(source_paths["design_schema"]),
        "calibration_source_hash": _sha256_file(source_paths["calibration_pack"]),
        "calibration_hash": model_config.calibration_pack_hash,
        "acceptance_policy_hash": _sha256_file(source_paths["acceptance_policy"]),
        "constraint_profile_hash": constraint_profile.source_hash,
        "operating_point_policy_hash": _sha256_file(
            source_paths["operating_point_policy"]
        ),
        "conservation_tolerances_hash": _sha256_file(
            source_paths["conservation_tolerances"]
        ),
        "physics_code_hash": physics_code_hash,
        "solver_binding_hash": solver_binding_hash,
    }
    if thermo_model is not None:
        provenance["thermo_model_hash"] = thermo_model.source_hash
    if component_map_registry is not None:
        provenance["component_map_registry_hash"] = (
            component_map_registry.source_hash
        )
    provenance.update(
        {
            f"{name}_map_hash": component_map.source_hash
            for name, component_map in component_maps.items()
        }
    )
    bundle_identity = {
        "backend_id": backend.backend_id,
        "physics_model_version": backend.model_version,
        "design_schema_version": schema_version,
        **provenance,
    }
    return PhysicsRuntime(
        repository_root=root,
        backend=backend,
        model_config=model_config,
        design_schema=MappingProxyType(design_schema),
        constraint_profile=constraint_profile,
        operating_point_policy=operating_point_policy,
        conservation_tolerances=MappingProxyType(conservation_tolerances),
        thermo_model=thermo_model,
        component_map_registry=component_map_registry,
        component_maps=MappingProxyType(component_maps),
        source_paths=MappingProxyType(source_paths),
        provenance_hashes=MappingProxyType(provenance),
        physics_bundle_hash=_canonical_hash(bundle_identity),
    )

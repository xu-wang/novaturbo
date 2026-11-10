"""Profile-driven Geometry19 validity labels without solver-side mutation.

The V2.1 solver remains the single source of physical results.  Profiles only
decide whether the immutable result is accepted for a downstream workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from src.physics.engine_match import EngineResult


class ConstraintProfileError(ValueError):
    """Raised when a Geometry19 constraint-profile contract is invalid."""


@dataclass(frozen=True)
class ConstraintProfile:
    """Immutable acceptance policy loaded from a versioned YAML profile."""

    profile_version: str
    description: str
    require_numerical_valid: bool
    require_physics_domain_valid: bool
    require_engineering_feasible: bool
    source_hash: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConstraintProfile":
        source_path = Path(path)
        try:
            source_bytes = source_path.read_bytes()
            payload = yaml.safe_load(source_bytes)
        except (OSError, yaml.YAMLError) as exc:
            raise ConstraintProfileError(
                f"Cannot load constraint profile: {source_path}"
            ) from exc
        if not isinstance(payload, dict):
            raise ConstraintProfileError("Constraint profile must be a mapping")

        expected_top_level = {
            "profile_version",
            "description",
            "acceptance",
            "constraint_partition",
        }
        if set(payload) != expected_top_level:
            raise ConstraintProfileError(
                "Constraint profile keys mismatch; "
                f"expected={sorted(expected_top_level)}, got={sorted(payload)}"
            )
        version = payload["profile_version"]
        description = payload["description"]
        if not isinstance(version, str) or not version.strip():
            raise ConstraintProfileError("profile_version must be non-empty")
        if source_path.stem != version:
            raise ConstraintProfileError(
                "Profile filename must match profile_version"
            )
        if not isinstance(description, str) or not description.strip():
            raise ConstraintProfileError("description must be non-empty")

        acceptance = payload["acceptance"]
        expected_acceptance = {
            "require_numerical_valid",
            "require_physics_domain_valid",
            "require_engineering_feasible",
        }
        if not isinstance(acceptance, dict) or set(acceptance) != expected_acceptance:
            raise ConstraintProfileError(
                "acceptance must define exactly the three validity requirements"
            )
        if any(type(acceptance[name]) is not bool for name in expected_acceptance):
            raise ConstraintProfileError("All acceptance requirements must be boolean")

        partition = payload["constraint_partition"]
        expected_partition = {
            "numerical_hard",
            "physics_domain",
            "engineering_soft",
        }
        if not isinstance(partition, dict) or set(partition) != expected_partition:
            raise ConstraintProfileError(
                "constraint_partition must define numerical_hard, "
                "physics_domain, and engineering_soft"
            )
        for category, entries in partition.items():
            if (
                not isinstance(entries, list)
                or not entries
                or any(not isinstance(item, str) or not item.strip() for item in entries)
            ):
                raise ConstraintProfileError(
                    f"constraint_partition.{category} must be a non-empty string list"
                )

        return cls(
            profile_version=version,
            description=description,
            require_numerical_valid=acceptance["require_numerical_valid"],
            require_physics_domain_valid=acceptance[
                "require_physics_domain_valid"
            ],
            require_engineering_feasible=acceptance[
                "require_engineering_feasible"
            ],
            source_hash=hashlib.sha256(source_bytes).hexdigest(),
        )


@dataclass(frozen=True)
class ConstraintAssessment:
    """The four required WP19-11 labels plus acceptance provenance."""

    constraint_profile: str
    constraint_profile_hash: str
    numerically_valid: bool
    physics_domain_valid: bool
    engineering_feasible: bool
    violated_constraints: tuple[str, ...]
    accepted: bool

    def as_mapping(self) -> dict[str, Any]:
        return {
            "constraint_profile": self.constraint_profile,
            "constraint_profile_hash": self.constraint_profile_hash,
            "numerically_valid": self.numerically_valid,
            "physics_domain_valid": self.physics_domain_valid,
            "engineering_feasible": self.engineering_feasible,
            "violated_constraints": list(self.violated_constraints),
            "accepted": self.accepted,
        }


def _finite_positive_outputs(result: "EngineResult") -> bool:
    required = ("thrust_N", "tsfc_kg_N_s", "mass_kg")
    return all(
        name in result.performance
        and math.isfinite(float(result.performance[name]))
        and float(result.performance[name]) > 0.0
        for name in required
    )


def _conservation_violations(
    result: "EngineResult",
    tolerances: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    conservation = result.conservation
    values = {
        "combustor_energy": float(conservation.combustor_energy_W),
        "shaft_power": float(conservation.shaft_power_W),
        "mass_flow": float(conservation.mass_flow_kg_s),
        "normalized_flow": float(conservation.normalized_flow),
        "nozzle_velocity": float(conservation.nozzle_velocity_m_s),
    }
    violations = [name + ".nonfinite" for name, value in values.items() if not math.isfinite(value)]
    if violations or tolerances is None:
        return tuple(violations)

    required = {
        "combustor_energy_absolute_W",
        "shaft_power_absolute_W",
        "normalized_flow_absolute",
        "nozzle_velocity_absolute_m_s",
        "mass_flow_rule",
    }
    if set(tolerances) - {"profile_version", "description"} != required:
        raise ConstraintProfileError("Conservation tolerance keys mismatch")
    if tolerances["mass_flow_rule"] != (
        "abs(residual_kg_s) <= normalized_flow_absolute * "
        "max(abs(mass_flow_gas_kg_s), 1.0e-12)"
    ):
        raise ConstraintProfileError("Unsupported mass-flow conservation rule")
    limits = {
        "combustor_energy": float(tolerances["combustor_energy_absolute_W"]),
        "shaft_power": float(tolerances["shaft_power_absolute_W"]),
        "normalized_flow": float(tolerances["normalized_flow_absolute"]),
        "nozzle_velocity": float(tolerances["nozzle_velocity_absolute_m_s"]),
    }
    if any(not math.isfinite(value) or value < 0.0 for value in limits.values()):
        raise ConstraintProfileError("Conservation tolerances must be finite and non-negative")
    mass_flow_gas = float(result.operating_point.get("mass_flow_gas_kg_s", math.nan))
    if not math.isfinite(mass_flow_gas):
        violations.append("mass_flow.reference_nonfinite")
    else:
        limits["mass_flow"] = limits["normalized_flow"] * max(abs(mass_flow_gas), 1.0e-12)
    violations.extend(
        name for name, limit in limits.items() if abs(values[name]) > limit
    )
    return tuple(sorted(violations))


def _is_physics_domain_residual(name: str) -> bool:
    """Return whether a thermodynamic residual defines model applicability."""

    return "_map_" in name or name.endswith("_enthalpy_closure")


def _constraint_groups(
    result: "EngineResult",
) -> Mapping[str, Mapping[str, float]]:
    return {
        "thermodynamic": result.thermodynamic_constraints,
        "assembly": result.assembly_constraints,
        "structural": result.structural_constraints,
        "thermal": result.thermal_constraints,
        "manufacturing": result.manufacturing_constraints,
    }


def assess_constraint_profile(
    result: "EngineResult",
    profile: ConstraintProfile,
    *,
    conservation_tolerances: Mapping[str, Any] | None = None,
) -> ConstraintAssessment:
    """Classify an existing result; never rerun or mutate its physics.

    A successful V2.1 solve establishes numerical validity.  Physics-domain
    validity additionally requires the complete post-cycle state and finite
    thermodynamic residuals.  All residual limits remain visible as
    engineering labels; the selected profile only controls final acceptance.
    """
    if not isinstance(profile, ConstraintProfile):
        raise ConstraintProfileError("profile must be a ConstraintProfile")

    conservation_violations = _conservation_violations(
        result, conservation_tolerances
    )
    numerically_valid = (
        result.solver.converged
        and result.solver.failure_code is None
        and _finite_positive_outputs(result)
        and not conservation_violations
    )
    domain_values = {
        name: float(value)
        for name, value in result.thermodynamic_constraints.items()
        if _is_physics_domain_residual(name)
    }
    thermodynamic_values = {
        name: float(value)
        for name, value in result.thermodynamic_constraints.items()
    }
    physics_domain_valid = (
        numerically_valid
        and result.feasibility_complete
        and bool(result.thermodynamic_constraints)
        and all(math.isfinite(value) for value in thermodynamic_values.values())
        and all(math.isfinite(value) and value <= 0.0 for value in domain_values.values())
    )

    violations: list[str] = []
    if not numerically_valid:
        failure = result.solver.failure_code
        code = failure.value if failure is not None else "INVALID_RESULT"
        violations.append(f"numerical_hard.solver.{code}")
        if result.solver.converged:
            violations.extend(
                f"numerical_hard.conservation.{name}"
                for name in conservation_violations
            )
    if numerically_valid and not physics_domain_valid:
        if not result.feasibility_complete or not result.thermodynamic_constraints:
            violations.append("physics_domain.incomplete_state")
        for name, value in thermodynamic_values.items():
            if not math.isfinite(value):
                violations.append(f"physics_domain.thermodynamic.{name}.nonfinite")
        for name, value in domain_values.items():
            if math.isfinite(value) and value > 0.0:
                violations.append(f"physics_domain.thermodynamic.{name}")

    engineering_values: list[float] = []
    for group_name, residuals in _constraint_groups(result).items():
        for name, raw_value in residuals.items():
            if group_name == "thermodynamic" and _is_physics_domain_residual(name):
                continue
            value = float(raw_value)
            engineering_values.append(value)
            if not math.isfinite(value):
                violations.append(f"engineering_soft.{group_name}.{name}.nonfinite")
            elif value > 0.0:
                violations.append(f"engineering_soft.{group_name}.{name}")
    engineering_feasible = (
        physics_domain_valid
        and bool(engineering_values)
        and all(math.isfinite(value) and value <= 0.0 for value in engineering_values)
    )

    accepted = (
        (numerically_valid or not profile.require_numerical_valid)
        and (
            physics_domain_valid
            or not profile.require_physics_domain_valid
        )
        and (
            engineering_feasible
            or not profile.require_engineering_feasible
        )
    )
    return ConstraintAssessment(
        constraint_profile=profile.profile_version,
        constraint_profile_hash=profile.source_hash,
        numerically_valid=numerically_valid,
        physics_domain_valid=physics_domain_valid,
        engineering_feasible=engineering_feasible,
        violated_constraints=tuple(sorted(violations)),
        accepted=accepted,
    )

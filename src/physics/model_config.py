"""Strict, immutable configuration loading for the Geometry19 physics model."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


class CalibrationPackError(ValueError):
    """Raised when a calibration pack cannot satisfy the runtime contract."""


def _freeze(value: Any) -> Any:
    """Recursively convert containers to immutable runtime representations."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    """Convert immutable runtime containers to canonical JSON-compatible data."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _plain(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _require_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationPackError(f"{field_name} must be a non-empty string")
    return value


def _load_contract(_source_path: Path) -> Mapping[str, Any]:
    # The executable contract is repository-owned. A calibration pack cannot
    # weaken validation by placing a modified schema next to itself.
    contract_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "physics"
        / "geometry19_calibration_schema_v1.yaml"
    )
    try:
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CalibrationPackError(
            f"Cannot load calibration schema contract: {contract_path}"
        ) from exc
    if not isinstance(contract, dict):
        raise CalibrationPackError("Calibration schema contract must be a mapping")
    return contract


def _validate_solver(
    raw: Any, contract: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise CalibrationPackError("solver_candidate must be a mapping")
    field_contracts = contract.get("solver_candidate_contract")
    if not isinstance(field_contracts, dict) or not field_contracts:
        raise CalibrationPackError(
            "Calibration schema has no solver_candidate_contract"
        )
    required = set(field_contracts)
    missing = sorted(required - raw.keys())
    extra = sorted(raw.keys() - required)
    if missing or extra:
        raise CalibrationPackError(
            f"solver_candidate fields mismatch; missing={missing}, extra={extra}"
        )
    for name, field_contract in field_contracts.items():
        if not isinstance(field_contract, dict):
            raise CalibrationPackError(
                f"Solver contract for {name} must be a mapping"
            )
        value = raw[name]
        field_type = field_contract.get("type")
        if field_type == "integer":
            if type(value) is not int:
                raise CalibrationPackError(f"{name} must be an integer")
        elif field_type == "number":
            if not _is_number(value) or not math.isfinite(float(value)):
                raise CalibrationPackError(f"{name} must be a finite number")
        elif field_type == "boolean":
            if type(value) is not bool:
                raise CalibrationPackError(f"{name} must be a boolean")
        else:
            raise CalibrationPackError(
                f"Solver contract for {name} has unsupported type {field_type!r}"
            )
        if "minimum" in field_contract and value < field_contract["minimum"]:
            raise CalibrationPackError(
                f"{name} is below solver minimum {field_contract['minimum']}"
            )
        if "maximum" in field_contract and value > field_contract["maximum"]:
            raise CalibrationPackError(
                f"{name} exceeds solver maximum {field_contract['maximum']}"
            )
        if "const" in field_contract and value != field_contract["const"]:
            if name == "reject_multiple_roots":
                raise CalibrationPackError(
                    "V2.1 freezes reject_multiple_roots=true; false is unsupported"
                )
            raise CalibrationPackError(
                f"{name} must equal {field_contract['const']!r}"
            )
    lower = float(raw["mass_flow_lower_fraction"])
    upper = float(raw["mass_flow_upper_fraction"])
    if not lower < upper:
        raise CalibrationPackError(
            "mass-flow fractions must satisfy lower < upper"
        )
    if raw["flow_residual_tolerance"] > raw["tangency_refine_threshold"]:
        raise CalibrationPackError(
            "flow_residual_tolerance must not exceed tangency_refine_threshold"
        )
    return dict(raw)


def _validate_reference_states(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict) or not raw:
        raise CalibrationPackError("reference_states must be a non-empty mapping")
    result: dict[str, dict[str, Any]] = {}
    for state_name, state_values in raw.items():
        _require_nonempty_string(state_name, "reference state name")
        if not isinstance(state_values, dict):
            raise CalibrationPackError(
                f"Reference state {state_name!r} must be a mapping"
            )
        result[state_name] = dict(state_values)
    try:
        corrected = result["turbine_corrected_mass_flux"]
        temperature = corrected["temperature_K"]
        pressure = corrected["pressure_Pa"]
        unit = corrected["unit"]
    except KeyError as exc:
        raise CalibrationPackError(
            "turbine_corrected_mass_flux reference state is incomplete"
        ) from exc
    if unit != "kg/(s*m^2)":
        raise CalibrationPackError(
            "turbine_corrected_mass_flux unit must be kg/(s*m^2)"
        )
    for name, value in (("temperature_K", temperature), ("pressure_Pa", pressure)):
        if not _is_number(value) or not math.isfinite(float(value)) or value <= 0:
            raise CalibrationPackError(
                f"turbine_corrected_mass_flux.{name} must be finite and positive"
            )
    return result


def _validate_pack_payload(
    payload: Any, contract: Mapping[str, Any]
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, tuple[str, ...]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    if not isinstance(payload, dict):
        raise CalibrationPackError("Calibration pack must be a YAML mapping")
    required_top = set(contract.get("required_top_level_fields", ()))
    missing_top = sorted(required_top - payload.keys())
    if missing_top:
        raise CalibrationPackError(
            f"Calibration pack is missing top-level fields: {missing_top}"
        )
    for name in (
        "pack_version",
        "calibration_schema_version",
        "physics_spec_version",
        "design_schema_version",
        "status",
        "notes",
    ):
        _require_nonempty_string(payload.get(name), name)
    if payload["calibration_schema_version"] != contract.get("schema_version"):
        raise CalibrationPackError("calibration_schema_version does not match contract")
    allowed_status = set(contract.get("allowed_status", ()))
    if payload["status"] not in allowed_status:
        raise CalibrationPackError(
            f"Unsupported calibration status: {payload['status']!r}"
        )
    ready = payload["formal_dataset_generation_ready"]
    if type(ready) is not bool:
        raise CalibrationPackError(
            "formal_dataset_generation_ready must be a boolean"
        )
    if (payload["status"] == "frozen") != ready:
        raise CalibrationPackError(
            "Only frozen packs may be formal-ready, and frozen packs must be ready"
        )

    references = _validate_reference_states(payload["reference_states"])
    raw_groups = payload["parameter_groups"]
    if not isinstance(raw_groups, dict) or not raw_groups:
        raise CalibrationPackError("parameter_groups must be a non-empty mapping")
    required_groups = set(contract.get("required_parameter_groups", ()))
    missing_groups = sorted(required_groups - raw_groups.keys())
    if missing_groups:
        raise CalibrationPackError(
            f"Calibration pack is missing parameter groups: {missing_groups}"
        )
    required_fields = set(contract.get("required_parameter_fields", ()))
    allowed_sources = set(contract.get("allowed_source_types", ()))
    values: dict[str, Any] = {}
    records: dict[str, dict[str, Any]] = {}
    groups: dict[str, tuple[str, ...]] = {}
    for group_name, group_records in raw_groups.items():
        _require_nonempty_string(group_name, "parameter group name")
        if not isinstance(group_records, list) or not group_records:
            raise CalibrationPackError(
                f"Parameter group {group_name!r} must be a non-empty list"
            )
        names: list[str] = []
        for record in group_records:
            if not isinstance(record, dict):
                raise CalibrationPackError(
                    f"Parameter group {group_name!r} contains a non-mapping record"
                )
            missing_record = sorted(required_fields - record.keys())
            if missing_record:
                raise CalibrationPackError(
                    f"Parameter record in {group_name!r} is missing: {missing_record}"
                )
            name = _require_nonempty_string(record.get("name"), "parameter name")
            if name in values:
                raise CalibrationPackError(f"Duplicate calibration parameter: {name}")
            for field_name in (
                "unit",
                "source_type",
                "source_reference",
                "physical_meaning",
                "calibration_target",
            ):
                _require_nonempty_string(record.get(field_name), f"{name}.{field_name}")
            if record["source_type"] not in allowed_sources:
                raise CalibrationPackError(
                    f"{name}.source_type is not allowed: {record['source_type']!r}"
                )
            valid_range = record.get("valid_range")
            if not isinstance(valid_range, (list, tuple)) or len(valid_range) != 2:
                raise CalibrationPackError(
                    f"{name}.valid_range must contain two values"
                )
            value = record.get("value")
            lower, upper = valid_range
            if type(value) is bool:
                if type(lower) is not bool or type(upper) is not bool:
                    raise CalibrationPackError(
                        f"{name} boolean range must contain booleans"
                    )
                if value not in valid_range:
                    raise CalibrationPackError(f"{name} is outside valid_range")
            elif _is_number(value):
                if not _is_number(lower) or not _is_number(upper):
                    raise CalibrationPackError(f"{name} numeric range is invalid")
                numeric = (float(lower), float(value), float(upper))
                if not all(math.isfinite(item) for item in numeric):
                    raise CalibrationPackError(f"{name} must contain finite numbers")
                if not lower <= value <= upper:
                    raise CalibrationPackError(f"{name} is outside valid_range")
            elif isinstance(value, str):
                if value not in valid_range:
                    raise CalibrationPackError(f"{name} is outside valid_range")
            else:
                raise CalibrationPackError(f"{name}.value has an unsupported type")
            values[name] = value
            records[name] = dict(record)
            names.append(name)
        groups[group_name] = tuple(names)

    corrected_names = (
        "turbine_corrected_mass_flux_min",
        "turbine_corrected_mass_flux_max",
    )
    for name in corrected_names:
        if name not in records or records[name]["unit"] != "kg/(s*m^2)":
            raise CalibrationPackError(f"{name} must use kg/(s*m^2)")
    if values[corrected_names[0]] >= values[corrected_names[1]]:
        raise CalibrationPackError("Corrected mass-flux limits are not ordered")
    for density_prefix in contract.get("rules", {}).get(
        "material_density_prefixes", ()
    ):
        matches = [name for name in records if name.startswith(density_prefix)]
        if len(matches) != 1 or records[matches[0]]["unit"] != "kg/m^3":
            raise CalibrationPackError(
                f"Material density {density_prefix!r} must have one kg/m^3 record"
            )

    solver = _validate_solver(payload["solver_candidate"], contract)
    return values, records, groups, references, solver


@dataclass(frozen=True)
class ModelConfig:
    """Validated immutable view of the effective calibration configuration."""

    pack_version: str
    calibration_schema_version: str
    physics_spec_version: str
    design_schema_version: str
    status: str
    formal_dataset_generation_ready: bool
    notes: str
    reference_states: Mapping[str, Mapping[str, Any]]
    parameter_values: Mapping[str, Any]
    parameter_records: Mapping[str, Mapping[str, Any]]
    parameter_groups: Mapping[str, tuple[str, ...]]
    solver_candidate: Mapping[str, Any]
    source_file_hash: str
    source_path: Path
    calibration_pack_hash: str = field(init=False)

    def __post_init__(self) -> None:
        records = {
            name: dict(record) for name, record in self.parameter_records.items()
        }
        values = dict(self.parameter_values)
        if set(values) != set(records):
            raise CalibrationPackError(
                "parameter_values and parameter_records must have identical names"
            )
        for name, record in records.items():
            if record.get("name") != name:
                raise CalibrationPackError(
                    f"Parameter record key/name mismatch for {name!r}"
                )
        grouped_names = tuple(
            name for names in self.parameter_groups.values() for name in names
        )
        if len(grouped_names) != len(set(grouped_names)):
            raise CalibrationPackError("parameter_groups contain duplicate names")
        if set(grouped_names) != set(records):
            raise CalibrationPackError(
                "parameter_groups must reference every parameter exactly once"
            )
        # dataclasses.replace() is supported as an override path: synchronize the
        # record values and run the complete pack validator on the effective data.
        for name, value in values.items():
            records[name]["value"] = value
        grouped_records = {
            group_name: [records[name] for name in names]
            for group_name, names in self.parameter_groups.items()
        }
        effective_payload = {
            "pack_version": self.pack_version,
            "calibration_schema_version": self.calibration_schema_version,
            "physics_spec_version": self.physics_spec_version,
            "design_schema_version": self.design_schema_version,
            "status": self.status,
            "formal_dataset_generation_ready": self.formal_dataset_generation_ready,
            "notes": self.notes,
            "reference_states": _plain(self.reference_states),
            "parameter_groups": grouped_records,
            "solver_candidate": _plain(self.solver_candidate),
        }
        contract = _load_contract(Path(self.source_path))
        values, records, groups, references, solver = _validate_pack_payload(
            effective_payload, contract
        )
        if (
            not isinstance(self.source_file_hash, str)
            or len(self.source_file_hash) != 64
        ):
            raise CalibrationPackError("source_file_hash must be a SHA-256 digest")
        frozen_references = _freeze(references)
        frozen_values = _freeze(values)
        frozen_records = _freeze(records)
        frozen_groups = _freeze(groups)
        frozen_solver = _freeze(solver)
        object.__setattr__(self, "reference_states", frozen_references)
        object.__setattr__(self, "parameter_values", frozen_values)
        object.__setattr__(self, "parameter_records", frozen_records)
        object.__setattr__(self, "parameter_groups", frozen_groups)
        object.__setattr__(self, "solver_candidate", frozen_solver)
        effective = {
            "pack_version": self.pack_version,
            "calibration_schema_version": self.calibration_schema_version,
            "physics_spec_version": self.physics_spec_version,
            "design_schema_version": self.design_schema_version,
            "status": self.status,
            "formal_dataset_generation_ready": self.formal_dataset_generation_ready,
            "notes": self.notes,
            "reference_states": frozen_references,
            "parameter_records": frozen_records,
            "parameter_groups": frozen_groups,
            "solver_candidate": frozen_solver,
        }
        object.__setattr__(self, "calibration_pack_hash", _canonical_hash(effective))

    def require(self, name: str) -> Any:
        try:
            return self.parameter_values[name]
        except KeyError as exc:
            raise CalibrationPackError(
                f"Calibration parameter {name!r} is required but missing"
            ) from exc

    def with_overrides(
        self,
        *,
        parameter_values: Mapping[str, Any] | None = None,
        solver_candidate: Mapping[str, Any] | None = None,
    ) -> "ModelConfig":
        """Create a validated, re-hashed effective configuration."""
        from dataclasses import replace

        values = dict(self.parameter_values)
        if parameter_values:
            unknown = sorted(set(parameter_values) - set(values))
            if unknown:
                raise CalibrationPackError(f"Unknown calibration overrides: {unknown}")
            values.update(parameter_values)
        solver = dict(self.solver_candidate)
        if solver_candidate:
            unknown = sorted(set(solver_candidate) - set(solver))
            if unknown:
                raise CalibrationPackError(f"Unknown solver overrides: {unknown}")
            solver.update(solver_candidate)
        return replace(self, parameter_values=values, solver_candidate=solver)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        source_path = Path(path).resolve()
        try:
            raw_bytes = source_path.read_bytes()
            payload = yaml.safe_load(raw_bytes)
        except (OSError, yaml.YAMLError) as exc:
            raise CalibrationPackError(
                f"Cannot load calibration pack: {source_path}"
            ) from exc
        contract = _load_contract(source_path)
        values, records, groups, references, solver = _validate_pack_payload(
            payload, contract
        )
        assert isinstance(payload, dict)
        return cls(
            pack_version=payload["pack_version"],
            calibration_schema_version=payload["calibration_schema_version"],
            physics_spec_version=payload["physics_spec_version"],
            design_schema_version=payload["design_schema_version"],
            status=payload["status"],
            formal_dataset_generation_ready=payload[
                "formal_dataset_generation_ready"
            ],
            notes=payload["notes"],
            reference_states=references,
            parameter_values=values,
            parameter_records=records,
            parameter_groups=groups,
            solver_candidate=solver,
            source_file_hash=sha256(raw_bytes).hexdigest(),
            source_path=source_path,
        )


def load_bootstrap_model_config(repo_root: str | Path) -> ModelConfig:
    path = (
        Path(repo_root)
        / "config"
        / "physics"
        / "geometry19_calibration_bootstrap_v1.yaml"
    )
    return ModelConfig.from_yaml(path)

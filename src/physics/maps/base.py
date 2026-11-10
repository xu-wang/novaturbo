"""Strict metadata, interpolation, domain, and provenance contracts for maps."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import yaml


class MapMetadataError(ValueError):
    """Raised when a component-map package does not satisfy its contract."""


class MapDomainError(ValueError):
    """Raised when a query is forbidden by the selected domain policy."""


class MapPolicy(str, Enum):
    DIAGNOSTIC = "diagnostic"
    ENGINEERING_CANDIDATE = "engineering_candidate"
    FORMAL = "formal"


@dataclass(frozen=True)
class MapEvaluation:
    values: Mapping[str, float]
    margins: Mapping[str, float]
    in_domain: bool
    extrapolation_distance: float
    uncertainty: Mapping[str, float]
    map_id: str
    map_version: str
    source_hash: str


@dataclass(frozen=True)
class ComponentMap:
    """Immutable two-coordinate rectilinear map with explicit extrapolation."""

    map_id: str
    map_version: str
    component_type: str
    coordinate_names: tuple[str, str]
    output_names: tuple[str, ...]
    coordinates: Mapping[str, np.ndarray]
    outputs: Mapping[str, np.ndarray]
    boundaries: Mapping[str, np.ndarray]
    uncertainty: Mapping[str, float]
    diagnostic_extrapolation_limit: float
    formal_use_allowed: bool
    source_hash: str
    metadata: Mapping[str, Any]

    def evaluate(
        self,
        query: Mapping[str, float],
        *,
        policy: MapPolicy | str = MapPolicy.DIAGNOSTIC,
    ) -> MapEvaluation:
        selected_policy = MapPolicy(policy)
        if set(query) != set(self.coordinate_names):
            raise MapDomainError(
                f"Map query coordinates must be exactly {self.coordinate_names}"
            )
        values = tuple(float(query[name]) for name in self.coordinate_names)
        if not all(math.isfinite(value) for value in values):
            raise MapDomainError("Map query coordinates must be finite")
        axes = tuple(self.coordinates[name] for name in self.coordinate_names)
        outside = []
        bounded_values = []
        for value, axis in zip(values, axes, strict=True):
            span = float(axis[-1] - axis[0])
            tolerance = 32.0 * np.finfo(np.float64).eps * max(
                1.0, abs(float(axis[0])), abs(float(axis[-1]))
            )
            if value < axis[0] - tolerance:
                outside.append(float((axis[0] - value) / span))
                bounded_values.append(value)
            elif value > axis[-1] + tolerance:
                outside.append(float((value - axis[-1]) / span))
                bounded_values.append(value)
            else:
                outside.append(0.0)
                bounded_values.append(float(np.clip(value, axis[0], axis[-1])))
        values = tuple(bounded_values)
        distance = math.sqrt(sum(item * item for item in outside))
        in_domain = distance == 0.0
        if selected_policy is MapPolicy.FORMAL and not self.formal_use_allowed:
            raise MapDomainError(f"Map {self.map_id} is not approved for formal use")
        if not in_domain and (
            selected_policy is not MapPolicy.DIAGNOSTIC
            or distance > self.diagnostic_extrapolation_limit
        ):
            raise MapDomainError(
                f"Map query is outside the {selected_policy.value} domain; "
                f"normalized_distance={distance:.6g}"
            )

        interpolated = {
            name: _bilinear(axes[0], axes[1], surface, values[0], values[1])
            for name, surface in self.outputs.items()
        }
        boundary_values = {
            name: _linear(axes[0], line, values[0])
            for name, line in self.boundaries.items()
        }
        inflation = 1.0 + 4.0 * distance
        return MapEvaluation(
            values=MappingProxyType(interpolated),
            margins=MappingProxyType(boundary_values),
            in_domain=in_domain,
            extrapolation_distance=distance,
            uncertainty=MappingProxyType(
                {name: value * inflation for name, value in self.uncertainty.items()}
            ),
            map_id=self.map_id,
            map_version=self.map_version,
            source_hash=self.source_hash,
        )


def _bracket(axis: np.ndarray, value: float) -> tuple[int, int, float]:
    if value <= axis[0]:
        lower, upper = 0, 1
    elif value >= axis[-1]:
        lower, upper = len(axis) - 2, len(axis) - 1
    else:
        upper = int(np.searchsorted(axis, value, side="right"))
        lower = upper - 1
    fraction = float((value - axis[lower]) / (axis[upper] - axis[lower]))
    return lower, upper, fraction


def _linear(axis: np.ndarray, line: np.ndarray, value: float) -> float:
    lower, upper, fraction = _bracket(axis, value)
    return float(line[lower] + fraction * (line[upper] - line[lower]))


def _bilinear(
    axis0: np.ndarray,
    axis1: np.ndarray,
    surface: np.ndarray,
    value0: float,
    value1: float,
) -> float:
    i0, i1, tx = _bracket(axis0, value0)
    j0, j1, ty = _bracket(axis1, value1)
    low = surface[i0, j0] + ty * (surface[i0, j1] - surface[i0, j0])
    high = surface[i1, j0] + ty * (surface[i1, j1] - surface[i1, j0])
    return float(low + tx * (high - low))


def _read_mapping(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        source = path.read_bytes()
        payload = yaml.safe_load(source)
    except (OSError, yaml.YAMLError) as exc:
        raise MapMetadataError(f"Cannot load {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise MapMetadataError(f"{label} must be a YAML mapping")
    return source, payload


def _string(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise MapMetadataError(f"{name} must be a non-empty string")
    return value


def _array(value: Any, label: str, *, ndim: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MapMetadataError(f"{label} must be numeric") from exc
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise MapMetadataError(f"{label} must be a finite {ndim}D array")
    array.setflags(write=False)
    return array


def load_component_map(package_path: str | Path) -> ComponentMap:
    """Load and validate ``metadata.yaml + data.yaml`` as one hashed package."""

    package = Path(package_path).resolve()
    metadata_bytes, metadata = _read_mapping(package / "metadata.yaml", "map metadata")
    data_bytes, data = _read_mapping(package / "data.yaml", "map data")
    required = {
        "map_id",
        "map_version",
        "component_type",
        "formal_use_allowed",
        "source",
        "coordinates",
        "outputs",
        "boundaries",
        "reference_state",
        "interpolation",
        "domain_policy",
        "uncertainty",
        "applicability",
    }
    if set(metadata) != required:
        raise MapMetadataError(
            f"Map metadata keys mismatch; missing={sorted(required - set(metadata))}, "
            f"extra={sorted(set(metadata) - required)}"
        )
    map_id = _string(metadata, "map_id")
    map_version = _string(metadata, "map_version")
    component_type = _string(metadata, "component_type")
    if package.name != map_id:
        raise MapMetadataError("Map package directory must match map_id")
    if type(metadata["formal_use_allowed"]) is not bool:
        raise MapMetadataError("formal_use_allowed must be boolean")
    source = metadata["source"]
    required_source = {
        "type",
        "reference",
        "license",
        "validation_status",
    }
    optional_source = {
        "document_id",
        "report_url",
        "source_location",
        "local_data_file",
        "local_data_sha256",
        "extraction_method",
        "archive_url",
        "archive_access_status",
    }
    if (
        not isinstance(source, dict)
        or not required_source.issubset(source)
        or not set(source).issubset(required_source | optional_source)
        or any(not isinstance(value, str) or not value for value in source.values())
    ):
        raise MapMetadataError("Map source metadata contract mismatch")
    local_source = source.get("local_data_file")
    local_hash = source.get("local_data_sha256")
    if (local_source is None) != (local_hash is None):
        raise MapMetadataError(
            "source.local_data_file and source.local_data_sha256 must appear together"
        )
    if local_source is not None:
        source_path = (package / local_source).resolve()
        try:
            source_path.relative_to(package)
        except ValueError as exc:
            raise MapMetadataError("Map local source file escapes its package") from exc
        if not source_path.is_file():
            raise MapMetadataError(f"Map local source file does not exist: {source_path}")
        actual_hash = sha256(source_path.read_bytes()).hexdigest()
        if actual_hash != local_hash:
            raise MapMetadataError(
                "Map local source hash mismatch; source data may have changed"
            )
    reference_state = metadata["reference_state"]
    turbomachinery_reference = {
        "temperature_K",
        "pressure_Pa",
        "corrected_speed_design_rpm",
        "total_or_static",
        "mass_flow_basis",
        "area_definition",
        "diameter_definition",
        "speed_definition",
    }
    component_reference_fields = {
        "compressor": (
            turbomachinery_reference | {"phi_design"},
            {"temperature_K", "pressure_Pa", "corrected_speed_design_rpm", "phi_design"},
        ),
        "turbine": (
            turbomachinery_reference
            | {"corrected_mass_flux_design_kg_s_m2"},
            {
                "temperature_K",
                "pressure_Pa",
                "corrected_speed_design_rpm",
                "corrected_mass_flux_design_kg_s_m2",
            },
        ),
        "inlet": (
            {
                "temperature_K",
                "pressure_Pa",
                "capture_area_ratio_design",
                "total_or_static",
                "mass_flow_basis",
                "area_definition",
                "diameter_definition",
            },
            {"temperature_K", "pressure_Pa", "capture_area_ratio_design"},
        ),
        "combustor": (
            {
                "temperature_K",
                "pressure_Pa",
                "mass_flux_design_kg_s_m2",
                "total_or_static",
                "mass_flow_basis",
                "area_definition",
                "temperature_definition",
            },
            {"temperature_K", "pressure_Pa", "mass_flux_design_kg_s_m2"},
        ),
        "nozzle": (
            {
                "temperature_K",
                "pressure_Pa",
                "reynolds_design",
                "total_or_static",
                "mass_flow_basis",
                "area_definition",
                "diameter_definition",
            },
            {"temperature_K", "pressure_Pa", "reynolds_design"},
        ),
    }
    try:
        expected_reference, numeric_reference_fields = component_reference_fields[
            component_type
        ]
    except KeyError as exc:
        raise MapMetadataError(
            f"Unsupported component map reference state: {component_type}"
        ) from exc
    if not isinstance(reference_state, dict) or set(reference_state) != expected_reference:
        raise MapMetadataError("Map reference_state contract mismatch")
    for name in numeric_reference_fields:
        try:
            value = float(reference_state[name])
        except (TypeError, ValueError) as exc:
            raise MapMetadataError(f"reference_state.{name} must be numeric") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise MapMetadataError(f"reference_state.{name} must be positive")
    for name in expected_reference - numeric_reference_fields:
        if not isinstance(reference_state[name], str) or not reference_state[name]:
            raise MapMetadataError(f"reference_state.{name} must be non-empty")
    applicability = metadata["applicability"]
    if not isinstance(applicability, dict) or set(applicability) != {
        "geometry_family",
        "scaling_basis",
        "excluded_uses",
    }:
        raise MapMetadataError("Map applicability contract mismatch")
    if (
        not isinstance(applicability["geometry_family"], str)
        or not isinstance(applicability["scaling_basis"], str)
        or not isinstance(applicability["excluded_uses"], list)
        or not applicability["excluded_uses"]
    ):
        raise MapMetadataError("Map applicability values are invalid")
    coordinate_specs = metadata["coordinates"]
    output_specs = metadata["outputs"]
    boundary_specs = metadata["boundaries"]
    if not isinstance(coordinate_specs, list) or len(coordinate_specs) != 2:
        raise MapMetadataError("Exactly two coordinate definitions are required")
    if not isinstance(output_specs, list) or not output_specs:
        raise MapMetadataError("At least one output definition is required")
    if not isinstance(boundary_specs, list) or not boundary_specs:
        raise MapMetadataError("At least one boundary definition is required")
    for label, specs in (
        ("coordinates", coordinate_specs),
        ("outputs", output_specs),
        ("boundaries", boundary_specs),
    ):
        if any(
            not isinstance(spec, dict)
            or set(spec) != {"name", "definition", "unit"}
            or any(not isinstance(spec[key], str) or not spec[key] for key in spec)
            for spec in specs
        ):
            raise MapMetadataError(f"{label} definitions require name, definition, unit")
    coordinate_names = tuple(spec["name"] for spec in coordinate_specs)
    output_names = tuple(spec["name"] for spec in output_specs)
    boundary_names = tuple(spec["name"] for spec in boundary_specs)
    if len(set(coordinate_names + output_names + boundary_names)) != (
        len(coordinate_names) + len(output_names) + len(boundary_names)
    ):
        raise MapMetadataError("Map coordinate/output/boundary names must be unique")
    if set(data) != {"coordinates", "outputs", "boundaries"}:
        raise MapMetadataError("Map data must contain coordinates, outputs, boundaries")
    if set(data["coordinates"]) != set(coordinate_names):
        raise MapMetadataError("Map data coordinate names differ from metadata")
    if set(data["outputs"]) != set(output_names):
        raise MapMetadataError("Map data output names differ from metadata")
    if set(data["boundaries"]) != set(boundary_names):
        raise MapMetadataError("Map data boundary names differ from metadata")
    coordinates = {
        name: _array(data["coordinates"][name], f"coordinates.{name}", ndim=1)
        for name in coordinate_names
    }
    if any(len(axis) < 2 or np.any(np.diff(axis) <= 0.0) for axis in coordinates.values()):
        raise MapMetadataError("Coordinate axes must be strictly increasing with >=2 points")
    shape = tuple(len(coordinates[name]) for name in coordinate_names)
    outputs = {
        name: _array(data["outputs"][name], f"outputs.{name}", ndim=2)
        for name in output_names
    }
    if any(array.shape != shape for array in outputs.values()):
        raise MapMetadataError(f"Output surfaces must have shape {shape}")
    boundaries = {
        name: _array(data["boundaries"][name], f"boundaries.{name}", ndim=1)
        for name in boundary_names
    }
    if any(len(line) != shape[0] for line in boundaries.values()):
        raise MapMetadataError("Boundary lines must follow the first coordinate axis")
    uncertainty = metadata["uncertainty"]
    if not isinstance(uncertainty, dict) or set(uncertainty) != set(output_names):
        raise MapMetadataError("Uncertainty must define every map output")
    uncertainty_values = {name: float(uncertainty[name]) for name in output_names}
    if any(not math.isfinite(value) or value < 0.0 for value in uncertainty_values.values()):
        raise MapMetadataError("Map uncertainties must be finite and nonnegative")
    interpolation = metadata["interpolation"]
    if interpolation != {"method": "bilinear_v1", "boundary_method": "linear_v1"}:
        raise MapMetadataError("Only bilinear_v1/linear_v1 is supported")
    domain_policy = metadata["domain_policy"]
    if not isinstance(domain_policy, dict) or set(domain_policy) != {
        "diagnostic_extrapolation_limit"
    }:
        raise MapMetadataError("domain_policy contract mismatch")
    extrapolation_limit = float(domain_policy["diagnostic_extrapolation_limit"])
    if not math.isfinite(extrapolation_limit) or not 0.0 <= extrapolation_limit <= 0.25:
        raise MapMetadataError("Diagnostic extrapolation limit must be in [0, 0.25]")
    source_hash = sha256(metadata_bytes + b"\0" + data_bytes).hexdigest()
    return ComponentMap(
        map_id=map_id,
        map_version=map_version,
        component_type=component_type,
        coordinate_names=coordinate_names,
        output_names=output_names,
        coordinates=MappingProxyType(coordinates),
        outputs=MappingProxyType(outputs),
        boundaries=MappingProxyType(boundaries),
        uncertainty=MappingProxyType(uncertainty_values),
        diagnostic_extrapolation_limit=extrapolation_limit,
        formal_use_allowed=metadata["formal_use_allowed"],
        source_hash=source_hash,
        metadata=MappingProxyType(metadata),
    )

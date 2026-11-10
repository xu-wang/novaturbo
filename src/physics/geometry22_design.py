"""Schema-driven V3 development coordinates, without enabling a physics runtime."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from src.physics.geometry_state import Geometry19


SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "config/physics/geometry22_schema_v3_development_v1.yaml"
)


class Geometry22DesignError(ValueError):
    """Invalid 22D development coordinate data."""


@lru_cache(maxsize=1)
def _schema_fields() -> tuple[dict[str, object], ...]:
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    if schema.get("schema_version") != "geometry22_schema_v3_development_v1":
        raise Geometry22DesignError("Unexpected 22D development schema version")
    fields = tuple(schema["design_variables"])
    if len(fields) != 22 or tuple(row["index"] for row in fields) != tuple(range(1, 23)):
        raise Geometry22DesignError("22D schema must contain ordered indices 1 through 22")
    if tuple(row["name"] for row in fields[:19]) != Geometry19.coordinate_names():
        raise Geometry22DesignError("22D schema does not preserve the Geometry19 prefix")
    return fields


def geometry22_schema_hash() -> str:
    """Hash the exact versioned schema bytes used by this adapter."""

    return hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest()


def geometry22_schema_metadata() -> dict[str, object]:
    """Return versioned UI/report metadata without claiming runtime support."""

    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    return {
        "schema_version": schema["schema_version"],
        "schema_hash": geometry22_schema_hash(),
        "runtime_enabled": bool(schema["runtime_enabled"]),
        "single_design_development_runtime_available": bool(
            schema["single_design_development_runtime_available"]
        ),
        "physics_validation_status": "not_completed",
        "design_variables": schema["design_variables"],
        "geometry_projection_dimension": 19,
        "prototype_dimension": 22,
    }


def _coordinate_hash(schema_hash: str, values: Sequence[float | int]) -> str:
    encoded = json.dumps(
        {
            "schema_hash": schema_hash,
            "coordinates": [
                int(value) if isinstance(value, Integral) else float(value).hex()
                for value in values
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Geometry22Design:
    """Validated 22D development design; the physical backend is still pending."""

    coordinates: tuple[float | int, ...]

    def __post_init__(self) -> None:
        fields = _schema_fields()
        if len(self.coordinates) != len(fields):
            raise Geometry22DesignError(
                f"Geometry22 requires 22 coordinates, received {len(self.coordinates)}"
            )
        for field, value in zip(fields, self.coordinates, strict=True):
            name = str(field["name"])
            if field["dtype"] == "int64":
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise Geometry22DesignError(f"{name} requires an integer value")
            elif isinstance(value, bool) or not isinstance(value, Real):
                raise Geometry22DesignError(f"{name} requires a real value")
            if not math.isfinite(float(value)):
                raise Geometry22DesignError(f"{name} must be finite")
            lower, upper = field["bounds"]
            if not lower <= value <= upper:
                raise Geometry22DesignError(
                    f"{name}={value!r} is outside [{lower}, {upper}]"
                )

    @classmethod
    def coordinate_names(cls) -> tuple[str, ...]:
        return tuple(str(row["name"]) for row in _schema_fields())

    @classmethod
    def from_sequence(cls, values: Sequence[float | int]) -> "Geometry22Design":
        return cls(tuple(values))

    @classmethod
    def from_mapping(cls, values: Mapping[str, float | int]) -> "Geometry22Design":
        names = cls.coordinate_names()
        missing = sorted(set(names) - set(values))
        extra = sorted(set(values) - set(names))
        if missing or extra:
            raise Geometry22DesignError(
                f"Geometry22 mapping mismatch; missing={missing}, extra={extra}"
            )
        return cls(tuple(values[name] for name in names))

    def as_mapping(self) -> dict[str, float | int]:
        return dict(zip(self.coordinate_names(), self.coordinates, strict=True))

    def geometry19_projection(self) -> Geometry19:
        """Return only the frozen geometric prefix for existing preview adapters."""

        return Geometry19.from_sequence(self.coordinates[:19])

    def component_parameters(self) -> dict[str, float]:
        """Return appended V3 parameters without treating them as STL geometry."""

        return {
            name: float(value)
            for name, value in zip(
                self.coordinate_names()[19:], self.coordinates[19:], strict=True
            )
        }

    def design_id(self) -> str:
        """Identify the complete 22D design, including schema bytes."""

        return _coordinate_hash(geometry22_schema_hash(), self.coordinates)

    def geometry19_projection_hash(self) -> str:
        """Identify the 19D projection, distinct from a published mesh ID."""

        return _coordinate_hash("geometry19_projection_v1", self.coordinates[:19])

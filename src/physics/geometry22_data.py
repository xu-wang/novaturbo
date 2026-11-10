"""V3 22D development normalization and isolated design records."""

from __future__ import annotations

import json
import math
from numbers import Real
import os
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import yaml

from src.physics.geometry22_design import (
    Geometry22Design,
    Geometry22DesignError,
    SCHEMA_PATH,
    geometry22_schema_hash,
)


class Geometry22DataError(ValueError):
    """A 22D normalization or artifact integrity failure."""


class Geometry22Normalizer:
    """Fixed schema-bound transform; no fitted 19D model state is reused."""

    def __init__(self) -> None:
        schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.schema_version = str(schema["schema_version"])
        self.schema_hash = geometry22_schema_hash()
        self.fields = tuple(schema["design_variables"])
        if len(self.fields) != 22:
            raise Geometry22DataError("Expected 22D development schema")

    def transform(self, design: Geometry22Design) -> tuple[float, ...]:
        return tuple(
            2.0 * (float(value) - float(field["bounds"][0]))
            / (float(field["bounds"][1]) - float(field["bounds"][0]))
            - 1.0
            for field, value in zip(self.fields, design.coordinates, strict=True)
        )

    def inverse(self, values: Sequence[float], *, round_integers: bool = False) -> Geometry22Design:
        if len(values) != 22:
            raise Geometry22DataError("Normalized design must contain 22 coordinates")
        physical: list[float | int] = []
        for field, value in zip(self.fields, values, strict=True):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise Geometry22DataError(f"Invalid normalized value for {field['name']}")
            value = float(value)
            if not math.isfinite(value) or value < -1.0 - 1e-10 or value > 1.0 + 1e-10:
                raise Geometry22DataError(f"Normalized value for {field['name']} is outside [-1, 1]")
            lower, upper = (float(bound) for bound in field["bounds"])
            restored = lower + (value + 1.0) * 0.5 * (upper - lower)
            if field["dtype"] == "int64":
                rounded = round(restored)
                if not round_integers and abs(restored - rounded) > 1e-9:
                    raise Geometry22DataError(
                        f"{field['name']} requires explicit integer rounding"
                    )
                physical.append(int(rounded))
            else:
                physical.append(min(max(restored, lower), upper))
        try:
            return Geometry22Design.from_sequence(physical)
        except Geometry22DesignError as exc:
            raise Geometry22DataError(str(exc)) from exc


class Geometry22DesignStore:
    """Append-only 22D design records; no physics or mesh publication."""

    def __init__(self, storage_root: str | Path) -> None:
        self.storage_root = Path(storage_root).resolve()

    def _record_path(self, design_id: str) -> Path:
        if len(design_id) != 64 or any(ch not in "0123456789abcdef" for ch in design_id):
            raise Geometry22DataError("Invalid 22D design ID")
        return self.storage_root / "geometry22_schema_v3_development_v1" / design_id / "design.json"

    def _payload(self, design: Geometry22Design) -> dict[str, object]:
        return {
            "record_version": "geometry22_development_design_record_v1",
            "design_id": design.design_id(),
            "design_schema_version": "geometry22_schema_v3_development_v1",
            "design_schema_hash": geometry22_schema_hash(),
            "design_dimension": 22,
            "design": design.as_mapping(),
            "geometry19_projection": design.geometry19_projection().as_mapping(),
            "geometry19_projection_hash": design.geometry19_projection_hash(),
            "geometry_asset_id": None,
            "component_parameters": design.component_parameters(),
            "physics_validation_status": "not_completed",
            "formal_use_allowed": False,
            "deployment_allowed": False,
        }

    def publish(self, design: Geometry22Design) -> dict[str, object]:
        payload = self._payload(design)
        target = self._record_path(design.design_id())
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return self.read(design.design_id())
        temporary = target.with_name(f".{uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            try:
                os.link(temporary, target)
            except FileExistsError:
                existing = self.read(design.design_id())
                if existing != payload:
                    raise Geometry22DataError("Existing 22D design record differs")
        finally:
            temporary.unlink(missing_ok=True)
        return self.read(design.design_id())

    def read(self, design_id: str) -> dict[str, object]:
        path = self._record_path(design_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            design = Geometry22Design.from_mapping(payload["design"])
        except (OSError, ValueError, KeyError, TypeError, Geometry22DesignError) as exc:
            raise Geometry22DataError("Cannot read valid 22D design record") from exc
        if design.design_id() != design_id or payload != self._payload(design):
            raise Geometry22DataError("22D design record identity or content mismatch")
        return payload

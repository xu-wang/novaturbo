"""Versioned target-domain contract for the simplified V2.2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


DOMAIN_VERSION = "geometry19_v2_2_development_target_domain_v1"
AXES = ("thrust_N", "tsfc_kg_N_s", "mass_kg")


class Geometry19TargetDomainError(ValueError):
    """Raised when the target-domain contract or a target is invalid."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TargetDomainAssessment:
    inside: bool
    status: str
    normalized_distance: float
    nearest_segment: str
    segment_fraction: float

    def as_mapping(self) -> dict[str, Any]:
        return {
            "inside": self.inside,
            "status": self.status,
            "normalized_distance": self.normalized_distance,
            "nearest_segment": self.nearest_segment,
            "segment_fraction": self.segment_fraction,
        }


@dataclass(frozen=True)
class Geometry19TargetDomain:
    domain_version: str
    source_path: Path
    source_hash: str
    nodes: Mapping[str, tuple[float, float, float]]
    segments: tuple[tuple[str, str, str], ...]
    log_scales: tuple[float, float, float]
    maximum_normalized_distance: float
    out_of_domain_status: str

    def _scaled_log_point(self, target: Mapping[str, Any]) -> tuple[float, ...]:
        if not set(AXES) <= set(target):
            raise Geometry19TargetDomainError("Target is missing required performance axes")
        values = tuple(float(target[name]) for name in AXES)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise Geometry19TargetDomainError("Target axes must be finite and positive")
        return tuple(math.log(value) / scale for value, scale in zip(values, self.log_scales))

    @staticmethod
    def _segment_distance(
        point: tuple[float, ...],
        start: tuple[float, ...],
        end: tuple[float, ...],
    ) -> tuple[float, float]:
        def distance(fraction: float) -> float:
            return max(
                abs(value - (left + fraction * (right - left)))
                for value, left, right in zip(point, start, end)
            )

        lower = 0.0
        upper = 1.0
        for _ in range(96):
            left = lower + (upper - lower) / 3.0
            right = upper - (upper - lower) / 3.0
            if distance(left) <= distance(right):
                upper = right
            else:
                lower = left
        candidates = (0.0, 1.0, (lower + upper) / 2.0)
        return min((distance(value), value) for value in candidates)

    def assess(self, target: Mapping[str, Any]) -> TargetDomainAssessment:
        point = self._scaled_log_point(target)
        best_distance = math.inf
        best_segment = ""
        best_fraction = 0.0
        for segment_id, start_name, end_name in self.segments:
            start = tuple(
                math.log(value) / scale
                for value, scale in zip(self.nodes[start_name], self.log_scales)
            )
            end = tuple(
                math.log(value) / scale
                for value, scale in zip(self.nodes[end_name], self.log_scales)
            )
            distance, fraction = self._segment_distance(point, start, end)
            if distance < best_distance:
                best_distance = distance
                best_segment = segment_id
                best_fraction = fraction
        inside = best_distance <= self.maximum_normalized_distance + 1.0e-12
        return TargetDomainAssessment(
            inside=inside,
            status=("INSIDE_V2_2_DEVELOPMENT_TARGET_DOMAIN" if inside else self.out_of_domain_status),
            normalized_distance=best_distance,
            nearest_segment=best_segment,
            segment_fraction=best_fraction,
        )


def load_geometry19_target_domain(
    repository_root: str | Path,
    source_path: str | Path = "config/physics/geometry19_v2_2_development_target_domain_v1.yaml",
) -> Geometry19TargetDomain:
    """Load the domain, validate evidence hashes, and return an immutable assessor."""

    root = Path(repository_root).resolve()
    source = Path(source_path)
    source = source.resolve() if source.is_absolute() else (root / source).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("domain_version") != DOMAIN_VERSION:
        raise Geometry19TargetDomainError("Target-domain version mismatch")
    if payload.get("formal_use_allowed") is not False:
        raise Geometry19TargetDomainError("Target domain must remain non-formal")
    if payload.get("engineering_validated") is not False:
        raise Geometry19TargetDomainError("Target domain is not engineering validated")
    if payload.get("interpolation_only") is not True or payload.get("extrapolation_allowed") is not False:
        raise Geometry19TargetDomainError("Target domain must remain interpolation-only")
    axes = payload.get("axes")
    if not isinstance(axes, dict) or tuple(axes) != AXES:
        raise Geometry19TargetDomainError("Target-domain axes or order differ")
    tolerances = tuple(float(axes[name]["relative_tolerance"]) for name in AXES)
    if any(not math.isfinite(value) or value <= 0.0 for value in tolerances):
        raise Geometry19TargetDomainError("Target-domain tolerances are invalid")
    nodes_payload = payload.get("nodes")
    if not isinstance(nodes_payload, dict) or len(nodes_payload) < 2:
        raise Geometry19TargetDomainError("Target-domain nodes are missing")
    nodes: dict[str, tuple[float, float, float]] = {}
    for name, raw in nodes_payload.items():
        if not isinstance(raw, list) or len(raw) != len(AXES):
            raise Geometry19TargetDomainError(f"Invalid target-domain node: {name}")
        values = tuple(float(value) for value in raw)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise Geometry19TargetDomainError(f"Invalid target-domain node: {name}")
        nodes[str(name)] = values
    segments_payload = payload.get("segments")
    if not isinstance(segments_payload, list) or not segments_payload:
        raise Geometry19TargetDomainError("Target-domain segments are missing")
    segments: list[tuple[str, str, str]] = []
    segment_ids: set[str] = set()
    for raw in segments_payload:
        segment_id = str(raw.get("id", ""))
        start = str(raw.get("start", ""))
        end = str(raw.get("end", ""))
        if not segment_id or segment_id in segment_ids or start not in nodes or end not in nodes or start == end:
            raise Geometry19TargetDomainError("Invalid or duplicate target-domain segment")
        segment_ids.add(segment_id)
        segments.append((segment_id, start, end))
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise Geometry19TargetDomainError("Target-domain evidence is missing")
    for record in evidence.values():
        evidence_path = (root / str(record["path"])).resolve()
        if root not in evidence_path.parents or not evidence_path.is_file():
            raise Geometry19TargetDomainError("Target-domain evidence path is invalid")
        if _sha256_file(evidence_path) != record.get("sha256"):
            raise Geometry19TargetDomainError("Target-domain evidence hash mismatch")
        summary = json.loads(evidence_path.read_text(encoding="utf-8"))
        aggregate = summary.get("aggregate", {})
        if (
            int(aggregate.get("pipeline_pass_count", -1)) != int(record["pipeline_pass"])
            or int(aggregate.get("target_count", -1)) != int(record["target_count"])
            or int(aggregate.get("targets_with_physics_acceptable", -1))
            != int(record["targets_with_physics_acceptable"])
        ):
            raise Geometry19TargetDomainError("Target-domain evidence counts differ")
    maximum_distance = float(payload.get("maximum_normalized_distance", 0.0))
    if not math.isfinite(maximum_distance) or maximum_distance <= 0.0:
        raise Geometry19TargetDomainError("Target-domain distance threshold is invalid")
    out_status = str(payload.get("out_of_domain", {}).get("status", ""))
    if out_status != "OUTSIDE_V2_2_DEVELOPMENT_TARGET_DOMAIN":
        raise Geometry19TargetDomainError("Target-domain rejection status differs")
    return Geometry19TargetDomain(
        domain_version=DOMAIN_VERSION,
        source_path=source,
        source_hash=_sha256_file(source),
        nodes=MappingProxyType(nodes),
        segments=tuple(segments),
        log_scales=tuple(math.log1p(value) for value in tolerances),
        maximum_normalized_distance=maximum_distance,
        out_of_domain_status=out_status,
    )

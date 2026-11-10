"""Frozen constraint dominance, delivery ranking, and disposition for WP19-23."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from src.ai.geometry19_hit_alignment import evaluation_hit_key


class Geometry19SelectionError(ValueError):
    """Raised when a candidate violates the frozen selection contract."""


@dataclass(frozen=True)
class ConstraintDominanceContract:
    contract_version: str
    source_path: Path
    source_hash: str
    scales: Mapping[str, float]
    hard_failure_penalty: float
    normalized_component_cap: float
    zero_tolerance: float
    disposition_codes: Mapping[str, str]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConstraintDominanceContract":
        source = Path(path).resolve()
        source_bytes = source.read_bytes()
        payload = yaml.safe_load(source_bytes)
        if not isinstance(payload, Mapping):
            raise Geometry19SelectionError("Constraint dominance contract must be a mapping")
        supported_versions = {
            "geometry19_constraint_dominance_v1": "frozen_wp19_23_development_v1",
            "geometry19_constraint_dominance_v2_2_c4_v1": (
                "v2_2_c4_shadow_integration"
            ),
        }
        version = payload.get("contract_version")
        if version not in supported_versions:
            raise Geometry19SelectionError("Constraint dominance contract version mismatch")
        if payload.get("status") != supported_versions[version]:
            raise Geometry19SelectionError("Constraint dominance contract status mismatch")
        if (
            payload.get("feasibility_rule")
            != "physics_acceptable_and_all_frozen_residuals_nonpositive"
            or payload.get("infeasible_comparison")
            != "lower_total_normalized_violation"
            or payload.get("feasible_comparison") != "three_axis_pareto"
            or payload.get("aggregation") != "capped_positive_part_l1_sum"
        ):
            raise Geometry19SelectionError("Constraint dominance policy mismatch")
        nested = payload.get("normalization_scales")
        if not isinstance(nested, Mapping):
            raise Geometry19SelectionError("Constraint normalization scales are missing")
        scales: dict[str, float] = {}
        for group, values in nested.items():
            if not isinstance(values, Mapping) or not values:
                raise Geometry19SelectionError(f"Invalid normalization group: {group}")
            for name, raw_scale in values.items():
                scale = float(raw_scale)
                if not math.isfinite(scale) or scale <= 0.0:
                    raise Geometry19SelectionError(
                        f"Invalid constraint normalization scale: {group}.{name}"
                    )
                scales[f"{group}.{name}"] = scale
        penalty = float(payload["hard_failure_penalty"])
        cap = float(payload["normalized_component_cap"])
        tolerance = float(payload["zero_tolerance"])
        if (
            not math.isfinite(penalty)
            or not math.isfinite(cap)
            or not math.isfinite(tolerance)
            or penalty <= len(scales) * cap
            or cap <= 0.0
            or tolerance < 0.0
        ):
            raise Geometry19SelectionError("Constraint dominance numeric contract is invalid")
        dispositions = payload.get("disposition_codes")
        expected_dispositions = {
            "delivered_hit",
            "delivered_approximation",
            "invalid_physics",
            "duplicate_design",
            "hit_not_selected",
            "nonhit_when_hit_exists",
            "forwarded_refinement",
            "forwarded_nsga2",
            "ranked_below_limit",
        }
        if not isinstance(dispositions, Mapping) or set(dispositions) != expected_dispositions:
            raise Geometry19SelectionError("Candidate disposition code contract mismatch")
        return cls(
            contract_version=str(payload["contract_version"]),
            source_path=source,
            source_hash=sha256(source_bytes).hexdigest(),
            scales=scales,
            hard_failure_penalty=penalty,
            normalized_component_cap=cap,
            zero_tolerance=tolerance,
            disposition_codes={str(key): str(value) for key, value in dispositions.items()},
        )


def assess_constraint_dominance(
    evaluation: Mapping[str, Any], contract: ConstraintDominanceContract
) -> dict[str, Any]:
    """Return a unit-normalized Deb constraint state for one true-physics result."""

    groups = evaluation.get("constraint_groups")
    assessment = evaluation.get("constraint_assessment")
    hard_valid = bool(
        evaluation.get("physics_acceptable")
        and isinstance(assessment, Mapping)
        and assessment.get("numerically_valid")
        and assessment.get("physics_domain_valid")
    )
    if not hard_valid or not isinstance(groups, Mapping):
        return {
            "contract_version": contract.contract_version,
            "constraint_feasible": False,
            "hard_valid": False,
            "violation_count": 1,
            "total_normalized_violation": contract.hard_failure_penalty,
            "maximum_normalized_violation": contract.hard_failure_penalty,
            "minimum_normalized_margin": -contract.hard_failure_penalty,
            "normalized_violations": {
                "numerical_or_physics_domain": contract.hard_failure_penalty
            },
        }

    flattened: dict[str, float] = {}
    for group, values in groups.items():
        if not isinstance(values, Mapping):
            raise Geometry19SelectionError(f"Constraint group is not a mapping: {group}")
        for name, raw_value in values.items():
            flattened[f"{group}.{name}"] = float(raw_value)
    missing = sorted(set(contract.scales) - set(flattened))
    extra = sorted(set(flattened) - set(contract.scales))
    if missing or extra:
        raise Geometry19SelectionError(
            f"Frozen constraint set mismatch; missing={missing}, extra={extra}"
        )

    normalized_violations: dict[str, float] = {}
    normalized_margins: list[float] = []
    for name in sorted(contract.scales):
        value = flattened[name]
        if not math.isfinite(value):
            normalized = contract.normalized_component_cap
            margin = -normalized
        else:
            scaled = value / contract.scales[name]
            normalized = min(max(scaled, 0.0), contract.normalized_component_cap)
            margin = -scaled
        normalized_margins.append(margin)
        if normalized > contract.zero_tolerance:
            normalized_violations[name] = normalized
    total = float(sum(normalized_violations.values()))
    maximum = float(max(normalized_violations.values(), default=0.0))
    feasible = not normalized_violations
    return {
        "contract_version": contract.contract_version,
        "constraint_feasible": feasible,
        "hard_valid": True,
        "violation_count": len(normalized_violations),
        "total_normalized_violation": total,
        "maximum_normalized_violation": maximum,
        "minimum_normalized_margin": float(min(normalized_margins)),
        "normalized_violations": normalized_violations,
    }


def constraint_state(evaluation: Mapping[str, Any]) -> tuple[bool, float]:
    state = evaluation.get("constraint_dominance")
    if isinstance(state, Mapping):
        return bool(state.get("constraint_feasible")), float(
            state.get("total_normalized_violation", math.inf)
        )
    acceptable = bool(evaluation.get("physics_acceptable"))
    return acceptable, 0.0 if acceptable else math.inf


def candidate_sort_key(evaluation: Mapping[str, Any]) -> tuple[Any, ...]:
    feasible, violation = constraint_state(evaluation)
    total_error = evaluation.get("total_true_error")
    total = float(total_error) if total_error is not None else math.inf
    errors = evaluation.get("relative_errors")
    maximum_axis = (
        max(float(value) for value in errors.values())
        if isinstance(errors, Mapping) and errors
        else math.inf
    )
    return (
        not bool(evaluation.get("development_diagnostic_hit")),
        *evaluation_hit_key(evaluation),
        not feasible,
        violation,
        evaluation.get("root_classification") != "CONVERGED_UNIQUE",
        total,
        maximum_axis,
        str(evaluation.get("evaluation_id", "")),
    )


def _normalized_geometry(
    evaluation: Mapping[str, Any], fields: Sequence[Mapping[str, Any]]
) -> np.ndarray | None:
    geometry = evaluation.get("geometry19")
    if not isinstance(geometry, Mapping):
        return None
    lower = np.asarray([item["bounds"][0] for item in fields], dtype=float)
    upper = np.asarray([item["bounds"][1] for item in fields], dtype=float)
    values = np.asarray([geometry[item["name"]] for item in fields], dtype=float)
    return 2.0 * (values - lower) / (upper - lower) - 1.0


def rank_delivery_candidates(
    rows: Sequence[Mapping[str, Any]],
    maximum: int,
    fields: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate and rank C/D/E candidates with one deterministic contract."""

    if maximum <= 0:
        return []
    unique: dict[str, dict[str, Any]] = {}
    for source in rows:
        if not source.get("physics_acceptable") or source.get("total_true_error") is None:
            continue
        row = dict(source)
        design_hash = str(row.get("design_hash") or row.get("evaluation_id"))
        previous = unique.get(design_hash)
        if previous is None or candidate_sort_key(row) < candidate_sort_key(previous):
            unique[design_hash] = row
    pool = list(unique.values())
    if any(row.get("development_diagnostic_hit") for row in pool):
        pool = [row for row in pool if row.get("development_diagnostic_hit")]
    if not pool:
        return []
    pool.sort(key=candidate_sort_key)
    selected = [pool.pop(0)]
    while pool and len(selected) < maximum:
        selected_vectors = [
            vector
            for vector in (_normalized_geometry(row, fields) for row in selected)
            if vector is not None
        ]

        def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
            base = candidate_sort_key(row)
            vector = _normalized_geometry(row, fields)
            distance = (
                min(float(np.linalg.norm(vector - prior)) for prior in selected_vectors)
                if vector is not None and selected_vectors
                else 0.0
            )
            return (*base[:6], -distance, base[6])

        chosen = min(pool, key=key)
        pool.remove(chosen)
        selected.append(chosen)
    return selected


def build_disposition_rows(
    rows: Sequence[Mapping[str, Any]],
    delivered: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    next_stage: str | None,
    contract: ConstraintDominanceContract,
) -> list[dict[str, Any]]:
    """Assign one frozen disposition reason to every evaluated candidate."""

    delivery_rank = {
        str(row["evaluation_id"]): rank for rank, row in enumerate(delivered, start=1)
    }
    delivered_ids = set(delivery_rank)
    any_hit = any(bool(row.get("development_diagnostic_hit")) for row in rows)
    dispositions: list[dict[str, Any]] = []
    representatives: dict[str, str] = {}
    for source in rows:
        if not source.get("design_hash"):
            continue
        design_hash = str(source["design_hash"])
        prior_id = representatives.get(design_hash)
        if prior_id is None:
            representatives[design_hash] = str(source.get("evaluation_id"))
            continue
        prior = next(row for row in rows if str(row.get("evaluation_id")) == prior_id)
        if candidate_sort_key(source) < candidate_sort_key(prior):
            representatives[design_hash] = str(source.get("evaluation_id"))
    codes = contract.disposition_codes
    for source in sorted(rows, key=lambda row: str(row.get("evaluation_id", ""))):
        row = dict(source)
        evaluation_id = str(row.get("evaluation_id"))
        design_hash = str(row.get("design_hash")) if row.get("design_hash") else None
        if evaluation_id in delivered_ids:
            reason = (
                codes["delivered_hit"]
                if row.get("development_diagnostic_hit")
                else codes["delivered_approximation"]
            )
        elif not row.get("physics_acceptable"):
            reason = codes["invalid_physics"]
        elif design_hash and representatives.get(design_hash) != evaluation_id:
            reason = codes["duplicate_design"]
        elif row.get("development_diagnostic_hit"):
            reason = codes["hit_not_selected"]
        elif any_hit:
            reason = codes["nonhit_when_hit_exists"]
        elif next_stage == "WP19-23D":
            reason = codes["forwarded_refinement"]
        elif next_stage == "WP19-23E":
            reason = codes["forwarded_nsga2"]
        else:
            reason = codes["ranked_below_limit"]
        disposition = {
            "contract_version": contract.contract_version,
            "stage": stage,
            "evaluation_id": evaluation_id,
            "design_hash": design_hash,
            "delivery_rank": delivery_rank.get(evaluation_id),
            "disposition_reason": reason,
            "physics_acceptable": bool(row.get("physics_acceptable")),
            "development_diagnostic_hit": bool(row.get("development_diagnostic_hit")),
            "constraint_dominance": row.get("constraint_dominance"),
        }
        dispositions.append(disposition)
    return dispositions


def attach_dispositions(
    rows: Sequence[dict[str, Any]], dispositions: Sequence[Mapping[str, Any]]
) -> None:
    by_id = {str(row["evaluation_id"]): dict(row) for row in dispositions}
    for row in rows:
        value = by_id.get(str(row.get("evaluation_id")))
        if value is not None:
            row["candidate_disposition"] = value

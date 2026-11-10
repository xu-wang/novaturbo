"""Frozen WP19-23 constraint dominance, ranking, and disposition tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.ai.geometry19_nsga2 import nondominated_fronts
from src.ai.geometry19_selection import (
    ConstraintDominanceContract,
    Geometry19SelectionError,
    assess_constraint_dominance,
    build_disposition_rows,
    rank_delivery_candidates,
    candidate_sort_key,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    REPO_ROOT / "config/physics/geometry19_constraint_dominance_v1.yaml"
)


def _groups(contract: ConstraintDominanceContract) -> dict[str, dict[str, float]]:
    groups: dict[str, dict[str, float]] = {}
    for full_name, scale in contract.scales.items():
        group, name = full_name.split(".", 1)
        groups.setdefault(group, {})[name] = -scale
    return groups


def _evaluation(
    contract: ConstraintDominanceContract,
    identifier: str,
    error: float,
    *,
    hit: bool = False,
    violation: float = 0.0,
) -> dict:
    schema = yaml.safe_load(
        (REPO_ROOT / "config/physics/geometry19_schema_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    geometry = {
        item["name"]: item["reference_value"] for item in schema["design_variables"]
    }
    groups = _groups(contract)
    groups["structural"]["compressor_tip_speed_limit"] = 500.0 * violation
    row = {
        "evaluation_id": identifier,
        "design_hash": identifier,
        "geometry19": geometry,
        "physics_acceptable": True,
        "development_diagnostic_hit": hit,
        "root_classification": "CONVERGED_UNIQUE",
        "relative_errors": {
            "thrust_N": error,
            "tsfc_kg_N_s": error,
            "mass_kg": error,
        },
        "total_true_error": error,
        "constraint_assessment": {
            "numerically_valid": True,
            "physics_domain_valid": True,
        },
        "constraint_groups": groups,
    }
    row["constraint_dominance"] = assess_constraint_dominance(row, contract)
    return row


def test_normalized_constraint_violation_uses_frozen_scales() -> None:
    contract = ConstraintDominanceContract.from_yaml(CONTRACT_PATH)
    feasible = _evaluation(contract, "feasible", 0.2)
    infeasible = _evaluation(contract, "infeasible", 0.1, violation=0.2)
    assert feasible["constraint_dominance"]["constraint_feasible"] is True
    state = infeasible["constraint_dominance"]
    assert state["constraint_feasible"] is False
    assert state["violation_count"] == 1
    assert state["total_normalized_violation"] == pytest.approx(0.2)
    assert state["normalized_violations"] == {
        "structural.compressor_tip_speed_limit": pytest.approx(0.2)
    }


def test_constraint_evaluation_rejects_changed_residual_set() -> None:
    contract = ConstraintDominanceContract.from_yaml(CONTRACT_PATH)
    row = _evaluation(contract, "changed", 0.1)
    del row["constraint_groups"]["thermal"]["turbine_metal_temperature"]
    with pytest.raises(Geometry19SelectionError, match="Frozen constraint set mismatch"):
        assess_constraint_dominance(row, contract)


def test_deb_constraint_dominance_precedes_three_axis_objectives() -> None:
    objectives = np.asarray(
        [
            [0.9, 0.9, 0.9],
            [0.01, 0.01, 0.01],
            [0.02, 0.02, 0.02],
        ]
    )
    feasible = np.asarray([True, False, False])
    violations = np.asarray([0.0, 0.2, 0.1])
    assert nondominated_fronts(objectives, violations, feasible) == [[0], [2], [1]]


def test_cde_delivery_ranking_and_disposition_share_one_contract() -> None:
    contract = ConstraintDominanceContract.from_yaml(CONTRACT_PATH)
    fields = yaml.safe_load(
        (REPO_ROOT / "config/physics/geometry19_schema_v1.yaml").read_text(
            encoding="utf-8"
        )
    )["design_variables"]
    infeasible_better_error = _evaluation(
        contract, "infeasible", 0.01, hit=True, violation=0.1
    )
    feasible = _evaluation(contract, "feasible", 0.02, hit=True)
    nonhit = _evaluation(contract, "nonhit", 0.001, hit=False)
    delivered = rank_delivery_candidates(
        [infeasible_better_error, feasible, nonhit], 1, fields
    )
    assert [row["evaluation_id"] for row in delivered] == ["feasible"]
    dispositions = build_disposition_rows(
        [infeasible_better_error, feasible, nonhit],
        delivered,
        stage="WP19-23E",
        next_stage=None,
        contract=contract,
    )
    reasons = {row["evaluation_id"]: row["disposition_reason"] for row in dispositions}
    assert reasons == {
        "feasible": "DELIVERED_DIAGNOSTIC_HIT",
        "infeasible": "NOT_DELIVERED_DELIVERY_LIMIT",
        "nonhit": "NOT_DELIVERED_DIAGNOSTIC_HIT_PREFERRED",
    }


def test_hit_alignment_precedes_engineering_violation_in_candidate_sort() -> None:
    contract = ConstraintDominanceContract.from_yaml(CONTRACT_PATH)
    fewer_failed_axes = _evaluation(contract, "fewer", 0.2, violation=0.2)
    more_failed_axes = _evaluation(contract, "more", 0.1)
    fewer_failed_axes["hit_alignment"] = {
        "failed_axis_count": 1,
        "rho": 1.4,
        "sum_axis_ratios": 2.0,
    }
    more_failed_axes["hit_alignment"] = {
        "failed_axis_count": 2,
        "rho": 1.1,
        "sum_axis_ratios": 2.1,
    }
    assert candidate_sort_key(fewer_failed_axes) < candidate_sort_key(more_failed_axes)

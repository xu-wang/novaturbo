"""Contract tests for the frozen-form Geometry19 acceptance policy."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    REPO_ROOT / "config" / "physics" / "geometry19_acceptance_policy_v1.yaml"
)
RATIONALE_PATH = (
    REPO_ROOT / "docs" / "physics" / "geometry19_acceptance_policy_v1.md"
)


def _policy():
    return yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))


def test_acceptance_policy_is_machine_readable_and_documented():
    assert POLICY_PATH.is_file()
    assert RATIONALE_PATH.is_file()
    policy = _policy()
    assert policy["policy_version"] == "geometry19_acceptance_policy_v1"
    assert policy["physics_model_version"] == "geometry19_closed_v2_1"


def test_hybrid_rule_form_is_frozen_but_thresholds_are_pending():
    policy = _policy()
    assert policy["status"] == "rule_frozen_thresholds_pending"
    assert policy["formal_hit_rate_ready"] is False
    gate = policy["formal_hit_gate"]
    assert gate["per_output_rule"] == (
        "abs(actual-target) <= "
        "max(absolute_tolerance, relative_tolerance*abs(target))"
    )
    assert gate["aggregate_error_may_compensate_failed_output"] is False
    assert all(
        item["threshold_status"] == "pending_single_point_and_small_doe"
        for item in policy["outputs"]
    )


def test_hit_requires_three_outputs_convergence_and_physical_feasibility():
    policy = _policy()
    assert [item["name"] for item in policy["outputs"]] == [
        "thrust_N",
        "tsfc_kg_N_s",
        "mass_kg",
    ]
    gate = policy["formal_hit_gate"]
    assert gate["require_exactly_one_operating_point"] is True
    assert gate["require_solver_converged"] is True
    assert gate["require_all_constraints_nonpositive"] is True
    assert gate["require_finite_positive_required_outputs"] is True
    assert policy["capability_diagnostic"][
        "may_silently_widen_engineering_acceptance"
    ] is False

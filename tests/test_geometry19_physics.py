"""WP19-23C true-error and paired D-Flow acceptance tests."""

from __future__ import annotations

import math

import pytest

from src.ai.geometry19_inverse_contract import (
    DiagnosticAcceptance,
    Geometry19InverseTarget,
)
from src.ai.geometry19_physics import choose_paired_variant, true_error_metrics


def _target() -> Geometry19InverseTarget:
    return Geometry19InverseTarget(
        target_contract_version="wp19_23_target_v1",
        target_id="test",
        target_type="ID",
        expected_status="KNOWN_FEASIBLE",
        thrust_N=100.0,
        tsfc_kg_N_s=2.0e-5,
        mass_kg=2.0,
        acceptance_profile="wp19_23_development_diagnostic_v1",
    )


def _acceptance() -> DiagnosticAcceptance:
    return DiagnosticAcceptance(
        relative_tolerances={
            "thrust_N": 0.03,
            "tsfc_kg_N_s": 0.01,
            "mass_kg": 0.02,
        },
        epsilons={
            "thrust_N": 1.0e-12,
            "tsfc_kg_N_s": 1.0e-15,
            "mass_kg": 1.0e-15,
        },
        formal_hit_rate_ready=False,
    )


def _evaluation(variant: str, error: float | None, acceptable: bool = True) -> dict:
    row = {
        "evaluation_id": f"{variant}:7",
        "candidate_id": 7,
        "variant": variant,
        "physics_acceptable": acceptable,
        "total_true_error": error,
        "development_diagnostic_hit": False,
    }
    if acceptable and error is not None:
        row["hit_alignment"] = {
            "version": "geometry19_hit_alignment_v1",
            "axis_ratios": {
                "thrust_N": error,
                "tsfc_kg_N_s": error,
                "mass_kg": error,
            },
            "failed_axis_count": 1,
            "rho": error,
            "sum_axis_ratios": 3.0 * error,
            "strict_hit": False,
        }
    return row


def test_total_true_error_is_equal_mean_but_hit_remains_per_axis() -> None:
    errors, total = true_error_metrics(
        _acceptance(),
        _target(),
        {"thrust_N": 102.0, "tsfc_kg_N_s": 2.04e-5, "mass_kg": 1.98},
    )
    assert errors == pytest.approx(
        {"thrust_N": 0.02, "tsfc_kg_N_s": 0.02, "mass_kg": 0.01}
    )
    assert total == pytest.approx(0.05 / 3.0)
    assert not _acceptance().is_diagnostic_hit(
        _target(),
        {"thrust_N": 102.0, "tsfc_kg_N_s": 2.04e-5, "mass_kg": 1.98},
        solver_converged=True,
        numerical_hard_constraints_passed=True,
    )


def test_dflow_is_accepted_only_for_hit_aligned_improvement() -> None:
    improved = choose_paired_variant(
        _evaluation("paired_direct", 0.2), _evaluation("dflow", 0.1)
    )
    equal = choose_paired_variant(
        _evaluation("paired_direct", 0.1), _evaluation("dflow", 0.1)
    )
    worse = choose_paired_variant(
        _evaluation("paired_direct", 0.1), _evaluation("dflow", 0.2)
    )
    assert improved["selected_variant"] == "dflow"
    assert improved["acceptance_reason"] == "DFLOW_HIT_ALIGNED_LEXICOGRAPHIC_IMPROVEMENT"
    assert equal["selected_variant"] == "paired_direct"
    assert worse["selected_variant"] == "paired_direct"


def test_fewer_failed_axes_precedes_lower_legacy_mean_error() -> None:
    direct = _evaluation("paired_direct", 0.01)
    dflow = _evaluation("dflow", 0.2)
    direct["hit_alignment"].update(failed_axis_count=2, rho=1.1, sum_axis_ratios=1.5)
    dflow["hit_alignment"].update(failed_axis_count=1, rho=1.4, sum_axis_ratios=2.0)
    selected = choose_paired_variant(direct, dflow)
    assert selected["selected_variant"] == "dflow"


def test_invalid_dflow_never_replaces_valid_direct() -> None:
    selected = choose_paired_variant(
        _evaluation("paired_direct", 0.1),
        _evaluation("dflow", None, acceptable=False),
    )
    assert selected["selected_variant"] == "paired_direct"
    assert selected["acceptance_reason"] == "DIRECT_RETAINED_DFLOW_PHYSICS_INVALID"


def test_valid_dflow_can_rescue_invalid_paired_direct() -> None:
    selected = choose_paired_variant(
        _evaluation("paired_direct", None, acceptable=False),
        _evaluation("dflow", 0.3),
    )
    assert selected["selected_variant"] == "dflow"
    assert selected["direct_total_true_error"] is None
    assert math.isclose(selected["dflow_total_true_error"], 0.3)

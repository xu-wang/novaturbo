from __future__ import annotations

import numpy as np

from src.ai.geometry19_e1_performance_ranking import (
    _predicted_rho,
    _stored_acceptable,
    _target_id,
)


def test_predicted_rho_matches_tolerance_axis_maximum() -> None:
    prediction = np.asarray([[105.0, 97.0, 102.0]])
    target = np.asarray([[100.0, 100.0, 100.0]])
    rho = _predicted_rho(prediction, target, np.asarray([0.05, 0.03, 0.04]))
    assert np.allclose(rho, [1.0])


def test_stored_acceptable_requires_solver_conservation_and_assessment() -> None:
    row = {
        "solver_status": {"converged": True, "conservation_passed": True},
        "constraint_values": {"assessment": {"accepted": True}},
        "physics_outputs": {"performance": {"thrust_N": 1.0}},
    }
    assert _stored_acceptable(row)
    row["solver_status"]["conservation_passed"] = False
    assert not _stored_acceptable(row)


def test_target_id_supports_both_suite_formats() -> None:
    assert _target_id({"target_id": "a"}) == "a"
    assert _target_id({"target": {"target_id": "b"}}) == "b"

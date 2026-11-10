"""Full-19D Pareto preference selection tests."""

import numpy as np
import pandas as pd

from src.ai.preference import objective_scaling, select_reference_directions


def test_reference_direction_selection_is_deterministic_and_balanced():
    objectives = np.asarray(
        [
            [0.0, 0.9],
            [0.2, 0.5],
            [0.5, 0.2],
            [0.9, 0.0],
            [0.1, 0.1],
        ]
    )
    feasible = np.asarray([True, True, True, True, False])
    selected, ledger = select_reference_directions(
        objectives,
        feasible,
        2,
        [[0.8, 0.2], [0.2, 0.8]],
        np.zeros(2),
        np.ones(2),
    )

    assert selected == [2, 1]
    assert ledger["direction_index"].tolist() == [0, 1]
    assert not set(selected).intersection({4})


def test_objective_scaling_uses_valid_rows_and_internal_signs():
    reference = pd.DataFrame(
        {
            "thrust": [10.0, 20.0, 1000.0],
            "mass": [3.0, 2.0, -10.0],
            "is_valid": [True, True, False],
        }
    )
    lower, upper = objective_scaling(
        reference, ["thrust", "mass"], [True, False]
    )

    np.testing.assert_allclose(lower, [-20.0, 2.0])
    np.testing.assert_allclose(upper, [-10.0, 3.0])

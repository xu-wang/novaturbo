import numpy as np

from src.ai.condition_support import ConditionSupportModel
from src.ai.bounded_flow_experiments import _stable_seed


def test_condition_support_flags_extreme_but_not_calibration_center():
    rng = np.random.RandomState(7)
    train = rng.normal(size=(200, 3))
    calibration = rng.normal(size=(80, 3))
    model = ConditionSupportModel.fit(
        train, calibration, neighbor_count=5, quantile=0.95
    )

    assert model.diagnose(np.zeros(3))["condition_support_status"] == "in_support"
    extreme = model.diagnose(np.asarray([-2.12, 10.62, -0.57]))
    assert extreme["condition_support_status"] == "out_of_support"
    assert extreme["condition_magnitude_out_of_support"] is True


def test_paired_experiment_seed_ignores_arm_label():
    first = _stable_seed(100, "nflows", 42, "target-a")
    second = _stable_seed(100, "nflows", 42, "target-a")
    assert first == second

from src.ai.geometry19_generator_oracle_analysis import _rho


def test_rho_is_maximum_tolerance_normalized_axis_error():
    target = {"thrust_N": 100.0, "tsfc_kg_N_s": 1.0, "mass_kg": 10.0}
    actual = {"thrust_N": 102.0, "tsfc_kg_N_s": 1.01, "mass_kg": 10.5}
    tolerances = {"thrust_N": 0.02, "tsfc_kg_N_s": 0.02, "mass_kg": 0.10}
    assert abs(_rho(actual, target, tolerances) - 1.0) < 1e-12

"""Contract tests for the N0 FAR-precision compatibility audit."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.ai.geometry19_performance_gated_protocol import ProtocolError
from src.ai.geometry19_v2_2_cross_version_compatibility import validate_protocol


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/experiments/geometry19_v2_2_cross_version_compatibility_v1.yaml"


def test_protocol_locks_runtime_scope_budget_and_gates() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_protocol(config)
    for mutation in (
        lambda value: value["runtime"].update(model_version="geometry19_map_enhanced_v2_2_full_variable_thermo_shadow"),
        lambda value: value["evaluation"].update(target_source_calls=98),
        lambda value: value["gates"]["target_sources"].update(maximum_target_rho=1.0),
        lambda value: value["authorization"].update(candidate_generation_authorized=True),
    ):
        changed = deepcopy(config)
        mutation(changed)
        with pytest.raises(ProtocolError):
            validate_protocol(changed)

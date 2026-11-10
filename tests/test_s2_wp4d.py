from pathlib import Path

import pytest
import yaml

from scripts.run_s2_wp4d import build_wp4_config, load_matrix
from src.ai.wp4 import _target_seed


MATRIX_PATH = Path(
    "config/experiments/cfm_arch/s2_wp4d_random_source_sensitivity.yaml"
)


def test_wp4d_frozen_matrix_builds_sixteen_isolated_runs():
    matrix = load_matrix(MATRIX_PATH)
    configs = [
        build_wp4_config(matrix, checkpoint_seed, source_seed)
        for checkpoint_seed in (42, 43)
        for source_seed in matrix["source_seeds"]
    ]
    assert len(configs) == 16
    assert len({config["output"]["directory"] for config in configs}) == 16
    assert {config["runtime"]["sample_seed"] for config in configs} == set(
        matrix["source_seeds"]
    )
    assert {config["runtime"]["generator_overrides"]["ode_steps"] for config in configs} == {32}
    assert {config["runtime"]["sample_count"] for config in configs} == {128}
    assert {config["runtime"]["sampling"]["policy"] for config in configs} == {
        "v2_1_fixed_raw"
    }


def test_wp4d_rejects_non_frozen_source_seed_count(tmp_path):
    matrix = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
    matrix["source_seeds"] = [1, 2]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(matrix), encoding="utf-8")
    with pytest.raises(ValueError, match="eight unique source seeds"):
        load_matrix(path)


def test_wp4d_source_seed_is_independent_of_checkpoint_seed():
    matrix = load_matrix(MATRIX_PATH)
    target_id = "frozen-target"
    for source_seed in (20260812, 20260819):
        # Both checkpoints are configured under the same public model name, so
        # their source draw is paired exactly for a fixed source seed + target.
        checkpoint42 = build_wp4_config(matrix, 42, source_seed)
        checkpoint43 = build_wp4_config(matrix, 43, source_seed)
        assert checkpoint42["models"]["cfm"] != checkpoint43["models"]["cfm"]
        assert checkpoint42["runtime"]["sample_seed"] == checkpoint43["runtime"][
            "sample_seed"
        ]
        assert _target_seed(
            checkpoint42["runtime"]["sample_seed"], "cfm", target_id
        ) == _target_seed(
            checkpoint43["runtime"]["sample_seed"], "cfm", target_id
        )
    assert _target_seed(20260812, "cfm", target_id) != _target_seed(
        20260819, "cfm", target_id
    )

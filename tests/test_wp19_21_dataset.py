"""WP19-21 development dataset contract and resume tests."""

from pathlib import Path

import pytest

from src.physics.wp19_21_dataset import WP1921Error, run_wp19_21


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / "config/physics/wp19_21_development_dataset_v1.yaml"


def test_small_dataset_is_normalized_and_resumable(tmp_path):
    output = tmp_path / "dataset"
    first = run_wp19_21(
        PROTOCOL, REPO_ROOT, output_root_override=output,
        sample_count_override=4, preview_budget_override=0,
        workers_override=1, shard_size_override=2,
    )
    second = run_wp19_21(
        PROTOCOL, REPO_ROOT, output_root_override=output,
        sample_count_override=4, preview_budget_override=0,
        workers_override=1, shard_size_override=2, resume=True,
    )
    assert first["status"] == second["status"] == "COMPLETE"
    assert first["summary"]["dataset_record_count"] == 4
    records = (output / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 4
    source = __import__("json").loads(records[0])
    assert len(source["x_19"]) == 19
    assert len(source["control_3"]) == 3
    assert "physics_outputs" in source
    assert "constraint_values" in source
    assert source["provenance"]["formal_dataset_generation_ready"] is False
    assert source["provenance"]["constraint_profile"] == (
        "geometry19_constraint_profile_pipeline_debug_v1"
    )
    assert Path(second["artifacts"]["resume_contract"]).name == "resume_contract.json"


def test_resume_rejects_changed_effective_contract(tmp_path):
    output = tmp_path / "dataset"
    run_wp19_21(
        PROTOCOL, REPO_ROOT, output_root_override=output,
        sample_count_override=2, preview_budget_override=0,
        workers_override=1, shard_size_override=2,
    )
    with pytest.raises(WP1921Error, match="different resume contract"):
        run_wp19_21(
            PROTOCOL, REPO_ROOT, output_root_override=output,
            sample_count_override=3, preview_budget_override=0,
            workers_override=1, shard_size_override=2, resume=True,
        )

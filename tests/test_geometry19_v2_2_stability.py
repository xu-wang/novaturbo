"""Simplified-V2.2 multi-target software-stability campaign tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.ai.geometry19_v2_2_stability import (
    Geometry19V22StabilityError,
    aggregate_v2_2_stability,
    load_v2_2_stability_config,
    render_v2_2_stability_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "config/experiments/geometry19_v2_2_multitarget_software_stability_v1.yaml"
)


def test_stability_config_freezes_six_unknown_cpu_targets() -> None:
    config = load_v2_2_stability_config(CONFIG_PATH)

    assert len(config["targets"]) == 6
    assert config["pipeline_profile"] == "v2_2_development"
    assert config["execution"]["device"] == "cpu"
    assert config["execution"]["raw_candidate_count_per_target"] == 256
    assert all(target["expected_status"] == "UNKNOWN" for target in config["targets"])
    assert config["training_enabled"] is False
    assert config["doe_generation_enabled"] is False


@pytest.mark.parametrize(
    "mutation",
    ("formal", "training", "too_few_targets", "known_feasible"),
)
def test_stability_config_rejects_scope_expansion(
    tmp_path: Path, mutation: str
) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if mutation == "formal":
        config["formal_use_allowed"] = True
    elif mutation == "training":
        config["training_enabled"] = True
    elif mutation == "too_few_targets":
        config["targets"] = config["targets"][:4]
    else:
        config["targets"][0]["expected_status"] = "KNOWN_FEASIBLE"
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(Geometry19V22StabilityError):
        load_v2_2_stability_config(path)


def _rows() -> list[dict]:
    return [
        {
            "label": f"target_{index}",
            "status": "PASS",
            "artifact_contract_status": "PASS",
            "elapsed_seconds": 20.0 + index,
            "has_physics_domain_valid": True,
            "c_physics_acceptable": 2,
            "physics_evaluations": 292,
            "best_rho": 10.0 + index,
            "formal_use_allowed": False,
            "target_hit": False,
            "completed_stages": ["B", "C", "D", "E", "F"],
            "n_valid": 32,
            "failure_stage": None,
        }
        for index in range(6)
    ]


def test_stability_aggregate_applies_software_only_gate() -> None:
    config = load_v2_2_stability_config(CONFIG_PATH)
    rows = _rows()

    aggregate = aggregate_v2_2_stability(rows, config)

    assert aggregate["software_stability_ready"] is True
    assert aggregate["pipeline_pass_rate"] == 1.0
    assert aggregate["targets_with_physics_acceptable"] == 6
    assert aggregate["median_best_rho"] == 12.5
    report = render_v2_2_stability_report(
        {"generated_at_utc": "now", "aggregate": aggregate, "targets": rows}
    )
    assert "不评价实验精度" in report
    assert "不训练代理/CFM" in report


def test_stability_gate_fails_when_two_targets_crash() -> None:
    config = load_v2_2_stability_config(CONFIG_PATH)
    rows = _rows()
    failed = deepcopy(rows)
    for row in failed[:2]:
        row.update(
            {
                "status": "FAILED",
                "artifact_contract_status": "FAILED",
                "has_physics_domain_valid": False,
            }
        )

    aggregate = aggregate_v2_2_stability(failed, config)

    assert aggregate["pipeline_pass_rate"] == pytest.approx(4.0 / 6.0)
    assert aggregate["software_stability_ready"] is False
    assert aggregate["checks"]["pipeline_pass_rate"] is False

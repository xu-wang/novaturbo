"""Boundary and reporting tests for the WP19-23 Gate G campaign."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import src.ai.geometry19_campaign as campaign
from src.ai.geometry19_campaign import (
    Geometry19CampaignError,
    _percentile,
    common_physics_budget,
    render_gate_g_report,
)
from src.ai.geometry19_physics import Geometry19PhysicsError, validate_evaluation_budget


def test_common_budget_is_the_smaller_frozen_power_of_two() -> None:
    assert common_physics_budget({"direct": 64, "dflow": 32}) == 32
    assert common_physics_budget({"direct": 0, "dflow": 64}) == 0
    with pytest.raises(Geometry19CampaignError, match="power of two"):
        common_physics_budget({"direct": 48, "dflow": 64})


def test_empty_and_partial_metric_sets_have_explicit_missing_semantics() -> None:
    assert _percentile([None, None], 50) is None
    assert _percentile([None, 0.1, 0.2], 50) == pytest.approx(0.15)


@pytest.mark.parametrize("value", [True, 0, -2, 3, 48])
def test_offline_physics_budget_rejects_boundary_contract_violations(value: int) -> None:
    with pytest.raises(Geometry19PhysicsError, match="positive power of two"):
        validate_evaluation_budget(value, 64)
    assert validate_evaluation_budget(None, 64) == 64
    with pytest.raises(Geometry19PhysicsError, match="cannot exceed"):
        validate_evaluation_budget(128, 64)


def test_gate_g_report_cannot_promote_formal_or_default_model() -> None:
    aggregate = {
        mode: {
            "targets": 2,
            "target_hits": 1,
            "physics_evaluations": 64,
            "physics_acceptable": 60,
            "median_n_valid": 100.0,
            "median_n_keep": 64.0,
            "median_best_total_true_error": 0.01,
            "p95_best_total_true_error": 0.02,
            "elapsed_seconds": 3.0,
        }
        for mode in ("direct", "dflow")
    }
    report = render_gate_g_report(
        {
            "generated_at_utc": "2026-08-28T00:00:00+00:00",
            "target_count": 2,
            "raw_candidate_count": 128,
            "aggregate": aggregate,
            "gate_g": {
                "development_status": "PASS",
                "equal_budget_status": "PASS",
                "provenance_status": "PASS",
            },
        }
    )
    assert "formal Gate：**DEFERRED**" in report
    assert "默认模型替换：**DEFERRED / NO ACTION**" in report


def test_campaign_integrates_both_modes_with_common_lower_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(__file__).resolve().parents[1]
    frozen = yaml.safe_load(
        (
            repository
            / "config/experiments/wp19_23_direct_dflow_equal_budget_campaign_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    frozen["targets"] = frozen["targets"][:2]
    frozen["budgets"]["raw_candidates_per_target_per_mode"] = 256
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(yaml.safe_dump(frozen), encoding="utf-8")
    physics_calls: list[tuple[str, int]] = []

    def candidate_mock(*args, **kwargs):
        mode = kwargs["mode"]
        output = Path(kwargs["output_root"])
        output.mkdir(parents=True)
        n_keep = 64 if mode == "direct" else 32
        target_id = yaml.safe_load(Path(args[1]).read_text(encoding="utf-8"))["target_id"]
        raw = []
        for candidate_id in range(256):
            row = {
                "target_id": target_id,
                "candidate_id": candidate_id,
                "source_id": candidate_id,
                "source_seed": 20260817,
                "model_tag": "test",
                "checkpoint_hash": "a" * 64,
                "model_hashes": {"generator": {"checkpoint_sha256": "a" * 64}},
                "source_initial": [float(candidate_id)],
                "sampling_mode": mode,
                "surrogate_before": None,
                "surrogate_after": None,
                "source_displacement": None,
                "normalized_design_displacement": None,
            }
            if mode == "dflow":
                row.update(
                    {
                        "surrogate_before": {"objective": 2.0},
                        "surrogate_after": {"objective": 1.0},
                        "source_displacement": {"linf": 0.1},
                        "normalized_design_displacement": {"linf": 0.1},
                    }
                )
            raw.append(row)
        (output / "raw_candidates.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in raw), encoding="utf-8"
        )
        summary = {
            "candidate_protocol": {
                "n_raw": 256,
                "n_valid": n_keep + 3,
                "n_keep": n_keep,
            },
            "hashes": {"initial_sources_sha256": "same-source"},
        }
        (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    def physics_mock(*args, **kwargs):
        mode = kwargs["sampling_mode"]
        budget = kwargs["evaluation_budget"]
        physics_calls.append((mode, budget))
        output = Path(kwargs["output_root"])
        output.mkdir(parents=True)
        rows = [
            {
                "relative_errors": {
                    "thrust_N": 0.1,
                    "tsfc_kg_N_s": 0.2,
                    "mass_kg": 0.05,
                }
            }
            for _ in range(budget)
        ]
        (output / "physics_forward_ledger.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        summary = {
            "counts": {
                "selected_candidate_count": budget,
                "physics_acceptable": budget,
                "diagnostic_hits": 0,
            },
            "best_approximation": {"total_true_error": 0.1},
        }
        (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(campaign, "run_candidate_protocol", candidate_mock)
    monkeypatch.setattr(campaign, "run_true_physics_stage", physics_mock)
    runtime = tmp_path / "runtime"
    summary = campaign.run_equal_budget_campaign(
        repository, config_path, device="cpu", output_root=runtime
    )
    assert physics_calls == [("direct", 32), ("dflow", 32)] * 2
    assert summary["aggregate"]["direct"]["physics_evaluations"] == 64
    assert summary["aggregate"]["dflow"]["physics_evaluations"] == 64
    assert summary["gate_g"]["formal_gate_status"].startswith("DEFERRED")
    assert summary["gate_g"]["default_model_replacement"] == "DEFERRED_NO_ACTION"
    assert (runtime / "campaign_manifest.json").is_file()
    assert (runtime / "Gate_G_summary.md").is_file()

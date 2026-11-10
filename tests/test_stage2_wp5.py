from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
import yaml

from src.ai.stage2_wp5 import (
    DELIVERY_MODES,
    _accept_updates,
    load_stage2_wp5_config,
    refinement_plan,
    run_stage2_wp5,
    surrogate_gradient_refinement,
)


class _IdentitySurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, design):
        return design[:, :1] * self.scale


def _config():
    return {
        "version": "test",
        "data": {},
        "target": {},
        "model": {},
        "generation": {"candidate_protocol": "v2_1_fixed_raw", "n_raw": 128},
        "delivery": {
            "mode": "generator_auto_refinement",
            "minimum_hit_count": 1,
            "maximum_delivered_candidates": 4,
        },
        "gradient": {
            "candidate_count": 2,
            "steps": 8,
            "learning_rate": 0.1,
            "validation_interval": 2,
            "maximum_0d_evaluations": 8,
            "trust_region_linf": 0.5,
        },
        "latent_repair": {
            "candidate_count": 2,
            "maximum_0d_evaluations": 2,
        },
        "nsga2": {
            "enabled": True,
            "seed": 1,
            "population_size": 4,
            "n_generations": 1,
            "objectives": ["target_weighted_relative_error"],
            "maximize": [False],
        },
        "runtime": {},
        "output": {},
    }


def test_wp5_config_freezes_v21_candidate_count(tmp_path):
    config = _config()
    config["generation"]["n_raw"] = 127
    path = tmp_path / "wp5.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="n_raw=128"):
        load_stage2_wp5_config(path)


def test_extra_objectives_require_nsga_capable_mode(tmp_path):
    config = _config()
    config["delivery"]["mode"] = "generator_gradient_refinement"
    config["delivery"]["extra_engineering_objectives"] = True
    path = tmp_path / "wp5.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="NSGA-II-capable"):
        load_stage2_wp5_config(path)


def test_refinement_plans_are_explicit_and_latent_safe():
    assert set(DELIVERY_MODES) == {
        "generator_only",
        "generator_screened",
        "generator_latent_repair",
        "generator_gradient_refinement",
        "generator_nsga2_refinement",
        "generator_auto_refinement",
    }
    assert refinement_plan("generator_auto_refinement", supports_latent=True) == (
        "latent_repair",
        "design_gradient",
        "nsga2",
    )
    assert refinement_plan("generator_auto_refinement", supports_latent=False) == (
        "design_gradient",
        "nsga2",
    )
    with pytest.raises(ValueError, match="repairable latent"):
        refinement_plan("generator_latent_repair", supports_latent=False)


def test_surrogate_gradient_accepts_only_true_validation_improvements():
    surrogate = _IdentitySurrogate()
    validations = []

    def validate(values, candidate_ids, step):
        validations.append(step)
        errors = np.abs(values[:, 0])
        return (
            pd.DataFrame(
                {
                    "candidate_id": candidate_ids,
                    "forward_valid": True,
                    "total_error": errors,
                    "within_tolerance": errors <= 0.05,
                }
            ),
            len(candidate_ids),
        )

    result = surrogate_gradient_refinement(
        np.asarray([[0.8], [-0.7]], dtype=float),
        [11, 19],
        [0.8, 0.7],
        np.asarray([0.0]),
        surrogate,
        {
            "steps": 8,
            "learning_rate": 0.2,
            "validation_interval": 2,
            "maximum_0d_evaluations": 8,
            "trust_region_linf": 0.8,
            "prior_weight": 0.0,
            "gradient_clip_norm": 1.0,
            "minimum_hit_count": 1,
        },
        validate,
    )

    assert validations
    assert result.diagnostics["actual_0d_evaluations"] <= 8
    assert result.improved_mask.all()
    assert np.all(result.best_actual_errors < np.asarray([0.8, 0.7]))
    assert set(result.accepted_forward["candidate_id"]) == {11, 19}


def test_accept_updates_preserves_immutable_candidate_id():
    forward = {7: {"candidate_id": 7, "forward_valid": True, "total_error": 0.5}}
    normalized = {7: np.asarray([0.4])}
    accepted = _accept_updates(
        forward,
        normalized,
        [7],
        np.asarray([[0.2]]),
        pd.DataFrame(
            {
                "candidate_id": [7],
                "forward_valid": [True],
                "total_error": [0.2],
                "within_tolerance": [False],
            }
        ),
    )

    assert accepted == 1
    assert forward[7]["candidate_id"] == 7


def test_repository_wp5_config_is_valid():
    repo_root = Path(__file__).resolve().parents[1]
    config = load_stage2_wp5_config(
        repo_root / "config" / "stage2" / "s2_wp5_cfm_default.yaml"
    )
    assert config["delivery"]["mode"] == "generator_auto_refinement"
    assert config["generation"]["n_raw"] == 128
    assert config["model"]["model_id"] == "cfm"
    assert config["model"]["checkpoint_path"].endswith("cfm/seed_43/best.pt")


def test_default_cfm_auto_mode_stops_after_direct_hits(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    config = copy.deepcopy(
        load_stage2_wp5_config(
            repo_root / "config" / "stage2" / "s2_wp5_cfm_default.yaml"
        )
    )
    config["runtime"]["device"] = "cpu"
    config["output"] = {"directory": str(tmp_path / "unused"), "refuse_overwrite": True}
    config_path = tmp_path / "wp5.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr("src.ai.stage2_wp5.collect_environment", lambda: {})
    monkeypatch.setattr("src.ai.stage2_wp5.collect_git_state", lambda *_args: {})
    monkeypatch.setattr("src.ai.stage2_wp5.collect_source_manifest", lambda *_args: [])

    output = run_stage2_wp5(
        config_path,
        repo_root=repo_root,
        output_dir_override=tmp_path / "output",
    )

    generation = json.loads(
        (output / "generation" / "metrics.json").read_text(encoding="utf-8")
    )
    decision = json.loads(
        (output / "delivery" / "decision.json").read_text(encoding="utf-8")
    )
    assert generation["candidate_protocol"] == "v2_1_fixed_raw"
    assert generation["N_raw"] == 128
    assert generation["N_keep"] == 64
    assert decision["direct_hit_count"] > 0
    assert decision["planned_actions"] == ["design_gradient", "nsga2"]
    assert decision["executed_actions"] == []
    assert decision["stop_reason"] == "direct_minimum_hit_count_reached"
    assert decision["cost"]["total_0d"] == 64

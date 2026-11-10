from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn

from src.ai.diag_cfm_strategy import (
    _full_budget_gradient_config,
    load_diag_cfm_strategy_config,
)
from src.ai.stage2_wp5 import surrogate_gradient_refinement


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
        "generation": {
            "candidate_protocol": "v2_1_fixed_raw",
            "n_raw": 128,
            "diag_prefilter": {
                "enabled": True,
                "pool_size": 512,
                "reranker_checkpoint": "diag.pt",
                "score": "reverse_condition_mse",
            },
        },
        "delivery": {
            "mode": "generator_auto_refinement",
            "minimum_hit_count": 1,
            "maximum_delivered_candidates": 10,
            "always_run_gradient": True,
        },
        "gradient": {
            "candidate_count": 2,
            "steps": 8,
            "learning_rate": 0.2,
            "validation_interval": 2,
            "maximum_0d_evaluations": 8,
            "trust_region_linf": 0.8,
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


def test_strategy_config_requires_complete_diag_prefilter(tmp_path):
    config = _config()
    config["generation"]["diag_prefilter"]["pool_size"] = 127
    path = tmp_path / "strategy.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="pool_size"):
        load_diag_cfm_strategy_config(path)


def test_strategy_runs_full_gradient_budget_despite_existing_hits():
    surrogate = _IdentitySurrogate()
    validation_steps = []

    def validate(values, candidate_ids, step):
        validation_steps.append(step)
        errors = np.abs(values[:, 0])
        return (
            pd.DataFrame(
                {
                    "candidate_id": candidate_ids,
                    "forward_valid": True,
                    "total_error": errors,
                    "within_tolerance": True,
                }
            ),
            len(candidate_ids),
        )

    raw = _config()["gradient"]
    result = surrogate_gradient_refinement(
        np.asarray([[0.8], [-0.7]], dtype=float),
        [11, 19],
        [0.8, 0.7],
        np.asarray([0.0]),
        surrogate,
        _full_budget_gradient_config(raw, existing_hits=10, candidate_count=2),
        validate,
    )

    assert validation_steps == [2, 4, 6, 8]
    assert result.diagnostics["actual_0d_evaluations"] == 8
    assert result.diagnostics["termination_reason"] == "steps_completed"

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
import yaml

from src.ai.stage2_wp3 import (
    ForwardConsistencySurrogate,
    aggregate_wp3_results,
    consistency_scores,
    load_stage2_wp3_config,
    repair_latents,
    select_by_consistency,
)


def _minimal_config():
    return {
        "version": "test",
        "data": {},
        "surrogate": {},
        "model_sets": [
            {
                "training_seed": 42,
                "models": [{"model_id": "nflows", "checkpoint_path": "a.pt"}],
            }
        ],
        "experiments": {
            "baseline": {"experiment_id": "raw"},
            "screening": {"experiment_id": "screen"},
            "repair": {
                "experiment_id": "repair",
                "steps": 4,
                "learning_rate": 0.1,
            },
        },
        "runtime": {
            "candidate_pool_size": 8,
            "physical_evaluation_count": 4,
        },
        "output": {},
    }


def test_config_requires_unique_independent_experiment_ids(tmp_path):
    config = _minimal_config()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert load_stage2_wp3_config(path)["version"] == "test"

    config["experiments"]["repair"]["experiment_id"] = "screen"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="unique experiment_id"):
        load_stage2_wp3_config(path)


def test_consistency_scoring_and_selection_are_weighted_and_stable():
    predicted = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 0.0]])
    score = consistency_scores(predicted, torch.zeros(2), weights=[3.0, 1.0])
    torch.testing.assert_close(score, torch.tensor([0.75, 1.0, 0.75]))
    selected = select_by_consistency(score.numpy(), 2)
    assert selected.tolist() == [0, 2]


def test_consistency_selection_ranks_nonfinite_scores_last():
    selected = select_by_consistency(np.asarray([np.nan, 0.2, np.inf, 0.1]), 3)
    assert selected.tolist() == [3, 1, 0]


class _IdentityGenerator(nn.Module):
    generator_name = "identity"
    supports_latent_repair = True
    design_dim = 2

    def decode_latent(self, latent, condition):
        del condition
        return latent


def test_latent_repair_reduces_frozen_forward_consistency():
    generator = _IdentityGenerator()
    surrogate = ForwardConsistencySurrogate(2, 1, [])
    linear = surrogate.network[0]
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
        linear.bias.zero_()
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)
    initial = torch.tensor([[[1.0, 0.2], [-0.8, -0.1]]])
    target = torch.tensor([[0.0]])
    _, repaired, diagnostics = repair_latents(
        generator,
        surrogate,
        initial,
        target,
        {
            "steps": 30,
            "learning_rate": 0.1,
            "prior_weight": 0.0,
            "boundary_weight": 0.0,
        },
    )
    assert diagnostics["after_consistency_mean"] < diagnostics["before_consistency_mean"]
    assert repaired[..., 0].abs().mean() < initial[..., 0].abs().mean()


def test_latent_repair_skips_numerically_unsafe_candidates():
    generator = _IdentityGenerator()
    surrogate = ForwardConsistencySurrogate(2, 1, [])
    with torch.no_grad():
        surrogate.network[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
        surrogate.network[0].bias.zero_()
    initial = torch.tensor([[[1.0, 0.0], [1.0e20, 0.0]]])
    _, repaired, diagnostics = repair_latents(
        generator,
        surrogate,
        initial,
        torch.tensor([[0.0]]),
        {
            "steps": 2,
            "learning_rate": 0.1,
            "prior_weight": 0.0,
            "boundary_weight": 0.0,
            "maximum_initial_abs_design": 10.0,
            "gradient_clip_norm": 1.0,
        },
    )
    assert diagnostics["optimized_candidate_count"] == 1
    assert diagnostics["skipped_nonfinite_count"] == 1
    assert repaired[0, 1, 0] == initial[0, 1, 0]


def test_aggregate_keeps_target_types_and_pairs_each_changed_arm():
    rows = []
    for target_type in ("ID", "Interpolation"):
        for method, error in (("baseline", 0.3), ("screening", 0.2), ("repair", 0.1)):
            rows.append(
                {
                    "model": "nflows",
                    "training_seed": 42,
                    "target_id": target_type,
                    "target_type": target_type,
                    "method": method,
                    "candidate_count": 10,
                    "forward_valid_count": 8,
                    "best_total_error": error,
                    "best_within_tolerance": error < 0.25,
                    "evaluations_to_first_hit": 2 if error < 0.25 else np.nan,
                    "mean_nearest_neighbour_distance": 0.5,
                }
            )
    summary, paired = aggregate_wp3_results(pd.DataFrame(rows))
    assert len(summary) == 6
    assert len(paired) == 4
    assert set(paired["target_type"]) == {"ID", "Interpolation"}
    assert paired["best_error_win_rate"].eq(1.0).all()

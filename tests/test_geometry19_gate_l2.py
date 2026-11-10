import torch

from src.ai.geometry19_gate_l2 import (
    build_negative_proposals,
    closure_comparison_key,
    v21_gradient_loss,
)


def test_negative_proposal_is_exact_reflection_and_keeps_integer() -> None:
    fields = [
        {"name": "x", "dtype": "float64", "bounds": [0.0, 1.0]},
        {"name": "z", "dtype": "int64", "bounds": [1, 3]},
    ]
    rows = [
        {
            "pair_id": "p",
            "split": "train",
            "campaign_id": "c",
            "sample_id": 1,
            "prior_geometry": {"x": 0.5, "z": 2},
            "proposed_geometry": {"x": 0.6, "z": 2},
            "prior_performance": {},
            "proposed_performance": {"thrust_N": 1.0},
            "physics_acceptable": True,
        }
    ]
    proposals, counts = build_negative_proposals(rows, fields)
    assert counts["negative_evaluations_requested"] == 1
    assert proposals[0]["proposed_geometry"] == {"x": 0.4, "z": 2}


def test_v21_gradient_loss_backpropagates_physical_gradient_label() -> None:
    model = torch.nn.Linear(2, 3, bias=False)
    center = torch.zeros((2, 2))
    direction = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    true_jvp = torch.full((2, 3), 2.0)
    loss = v21_gradient_loss(
        model, center, direction, true_jvp, torch.ones((2, 3)), torch.ones(3),
        huber_delta=1.0, cosine_weight=0.25, create_graph=True,
    )
    loss.backward()
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad).all()
    assert float(model.weight.grad.abs().sum()) > 0.0


def test_closure_comparison_prioritizes_target_hits() -> None:
    def result(hits: int, rho: float) -> dict:
        return {
            "arm_id": str(hits),
            "final_target_hits": hits,
            "c_target_hits": hits,
            "median_best_true_rho": rho,
            "total_v2_1_calls": 200,
            "refinement": {"rho_improved_rate": 0.5},
        }

    assert closure_comparison_key(result(3, 2.0)) < closure_comparison_key(result(2, 0.5))

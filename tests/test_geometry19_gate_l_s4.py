import numpy as np
import torch

from src.ai.geometry19_gate_l_s4 import directional_loss, propose_direction_rows


class LinearModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x[:, :3], torch.zeros((len(x), 1))), dim=1)


def test_directional_loss_is_zero_for_matching_secant() -> None:
    model = LinearModel()
    x0 = torch.zeros((2, 3))
    x1 = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, -1.0]])
    denominator = torch.ones((2, 3))
    loss = directional_loss(
        model, x0, x1, x1 - x0, denominator, torch.ones(4),
        huber_delta=1.0, cosine_weight=0.25,
    )
    assert float(loss) < 1.0e-7


def test_proposals_keep_integer_coordinates_fixed() -> None:
    record = {
        "campaign_id": "c",
        "sample_id": 1,
        "x_19": {"x": 0.5, "z": 2},
        "physics_outputs": {
            "performance": {"thrust_N": 1.0, "tsfc_kg_N_s": 2.0, "mass_kg": 3.0}
        },
    }
    splits = {
        name: {"records": [record]}
        for name in ("train", "validation", "test")
    }
    fields = [
        {"name": "x", "dtype": "float64", "bounds": [0.0, 1.0]},
        {"name": "z", "dtype": "int64", "bounds": [1, 3]},
    ]
    rows = propose_direction_rows(
        splits, fields, {"train": 1, "validation": 1, "test": 1},
        normalized_step=0.025, seed=7,
    )
    assert len(rows) == 3
    assert all(row["proposed_geometry"]["z"] == 2 for row in rows)
    assert all(np.isclose(row["normalized_direction_norm"], 0.025) for row in rows)

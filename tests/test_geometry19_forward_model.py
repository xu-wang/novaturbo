from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.ai.geometry19_forward_model import (
    Geometry19ForwardModelError,
    attach_forward_spec,
    decode_forward_outputs,
    resolve_forward_model_spec,
)
from src.ai.geometry19_inverse_contract import load_wp19_23_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "config/physics/wp19_23_inverse_closure_v1.yaml"


def test_default_and_s0_tags_resolve_declared_responsibilities() -> None:
    protocol = load_wp19_23_contract(PROTOCOL_PATH).payload
    default = resolve_forward_model_spec(REPO_ROOT, protocol)
    assert default.logical_tag == "performance3_log"
    assert default.output_transform == "natural_log"
    assert default.target_names == ("thrust_N", "tsfc_kg_N_s", "mass_kg")
    assert default.use_constraint_loss is False
    assert default.use_feasibility_loss is False

    s0 = resolve_forward_model_spec(REPO_ROOT, protocol, logical_tag="s0")
    assert s0.logical_tag == "s0"
    assert s0.output_transform == "identity"
    assert len(s0.target_names) == 48
    assert s0.use_constraint_loss is True
    assert s0.use_feasibility_loss is True


def test_log_output_decoding_uses_runtime_contract() -> None:
    protocol = load_wp19_23_contract(PROTOCOL_PATH).payload
    spec = resolve_forward_model_spec(REPO_ROOT, protocol)
    physical = torch.tensor([[10.0, 0.25, 3.0]], dtype=torch.float64)
    mean = np.log(np.asarray([5.0, 0.5, 1.5]))
    std = np.asarray([2.0, 0.5, 0.25])
    normalized = (torch.log(physical) - torch.tensor(mean)) / torch.tensor(std)
    stats = attach_forward_spec(
        {"output_mean": mean, "output_std": std}, spec
    )
    assert torch.allclose(decode_forward_outputs(normalized, stats), physical)


def test_unknown_forward_tag_is_rejected() -> None:
    protocol = load_wp19_23_contract(PROTOCOL_PATH).payload
    with pytest.raises(Geometry19ForwardModelError, match="Unsupported forward model tag"):
        resolve_forward_model_spec(REPO_ROOT, protocol, logical_tag="unknown")

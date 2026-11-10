"""Synthetic GPU-6 tests only; never train the sealed production dataset."""
from copy import deepcopy
from pathlib import Path
import inspect
import json

import pytest
import torch

from src.ai import geometry19_performance_gated_training as runner
from src.ai.geometry19_performance_gated_models import PerformanceGate, LocalCFM, PooledCFM
from src.ai.geometry19_performance_gated_protocol import ProtocolError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def physical_gpu6():
    manifest = runner.cuda_manifest()
    assert manifest["visible_devices"] == "6" and manifest["logical_device"] == "cuda:0"
    runner.set_determinism(42)
    return manifest


@pytest.fixture(autouse=True)
def no_physics_or_surrogate(monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail("No physics or surrogate-array loading in model tests")
    monkeypatch.setattr("src.physics.runtime.PhysicsRuntime.evaluate", forbidden)
    monkeypatch.setattr("numpy.load", forbidden)


def synthetic():
    runner.set_determinism(42)
    device = "cuda:0"
    roots = torch.arange(33, device=device).repeat_interleave(2)
    data = {"condition": torch.randn(66, 3, device=device), "root": roots,
            "local_cfm": torch.randn(66, 16, device=device)*.01,
            "pooled_cfm": torch.rand(66, 16, device=device),
            "integers": torch.stack([torch.randint(n, (66,), device=device) for n in [3, 4, 5]], dim=1),
            "weight": torch.ones(66, device=device)}
    bank = {"epsilon": torch.randn(66, 16, device=device), "time": torch.rand(66, device=device)}
    return data, bank


@pytest.mark.parametrize("arm", ["gate", "local_cfm", "pooled_cfm"])
def test_shapes_finite_loss_and_gradients(arm):
    data, bank = synthetic()
    model = runner.build_model(arm, [3, 4, 5]).cuda()
    losses = runner.row_losses(model, arm, data, **bank)
    assert losses.shape == (66,) and bool(torch.isfinite(losses).all())
    losses.mean().backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    if arm == "local_cfm":
        assert len(model.heads) == 33
        assert all(float(h[0].weight.grad.abs().sum()) > 0 for h in model.heads)


def test_only_three_performance_conditions():
    data, bank = synthetic()
    gate, local, pooled = PerformanceGate().cuda(), LocalCFM().cuda(), PooledCFM([3, 4, 5]).cuda()
    bad = torch.randn(66, 4, device="cuda:0")
    with pytest.raises(ValueError): gate(bad)
    with pytest.raises(ValueError): local(data["local_cfm"], bank["time"], bad, data["root"])
    with pytest.raises(ValueError): pooled.integer_logits(bad)
    assert list(inspect.signature(pooled.forward).parameters) == ["state", "time", "condition"]
    assert list(inspect.signature(pooled.integer_logits).parameters) == ["condition"]
    assert not any(isinstance(m, torch.nn.Embedding) for m in local.modules())


def test_local_dispatch_is_head_selection_not_condition():
    data, bank = synthetic()
    model = LocalCFM().cuda()
    hidden = model.trunk(data["local_cfm"], bank["time"], data["condition"])
    result = model(data["local_cfm"], bank["time"], data["condition"], data["root"])
    for k in range(33):
        mask = data["root"] == k
        assert torch.allclose(result[mask], model.heads[k](hidden[mask]), atol=1e-7)
    invalid = data["root"].clone()
    invalid[0] = 33
    with pytest.raises(ValueError): model(data["local_cfm"], bank["time"], data["condition"], invalid)


def test_flow_pair_endpoints_and_pooled_loss():
    data, bank = synthetic()
    for time, expected in [(torch.zeros(66, device="cuda:0"), bank["epsilon"]),
                           (torch.ones(66, device="cuda:0"), data["pooled_cfm"])]:
        state, velocity = runner.flow_pair(data["pooled_cfm"], bank["epsilon"], time)
        assert torch.equal(state, expected)
        assert torch.equal(velocity, data["pooled_cfm"]-bank["epsilon"])
    model = PooledCFM([3, 4, 5]).cuda()
    x, velocity = runner.flow_pair(data["pooled_cfm"], **bank)
    mse = (model(x, bank["time"], data["condition"])-velocity).square().mean(dim=1)
    ce = torch.stack([torch.nn.functional.cross_entropy(p, data["integers"][:, k], reduction="none")
                      for k, p in enumerate(model.integer_logits(data["condition"]))], dim=1).mean(dim=1)
    assert torch.equal(runner.row_losses(model, "pooled_cfm", data, **bank), mse+ce)


def test_balanced_sampler_and_shared_stream():
    roots = torch.cat([torch.full((i+1,), i, dtype=torch.long) for i in range(33)])
    settings = {"rows_per_root_per_batch": 16, "steps_per_epoch": 32}
    local = list(runner.batches(roots, "local_cfm", torch.Generator().manual_seed(10042), settings))
    pooled = list(runner.batches(roots, "pooled_cfm", torch.Generator().manual_seed(10042), settings))
    assert len(local) == 32
    for a, b in zip(local, pooled):
        assert torch.equal(a, b) and len(a) == 528
        assert torch.equal(torch.bincount(roots[a]), torch.full((33,), 16))
    gate = torch.cat(list(runner.batches(roots, "gate", torch.Generator().manual_seed(10042), {"batch_size": 256})))
    assert torch.equal(gate.sort().values, torch.arange(len(roots)))


def test_root_equal_mean_and_nonfinite_selection():
    roots = torch.cat([torch.full((i+1,), i, device="cuda:0", dtype=torch.long) for i in range(33)])
    assert float(runner.root_equal_mean(roots.float(), roots)) == pytest.approx(16.)
    with pytest.raises(ProtocolError): runner.Selection().update(1, float("nan"), {"min_delta": .1, "minimum_epochs": 2, "patience": 2})


def test_checkpoint_tie_and_significant_improvement_are_separate():
    state = runner.Selection()
    settings = {"min_delta": .1, "minimum_epochs": 3, "patience": 2}
    assert state.update(1, 1., settings) == (True, False)
    assert state.update(2, .99, settings) == (True, False)
    assert state.update(3, .99, settings) == (False, True)
    assert state.best_epoch == 2 and state.anchor == 1.


@pytest.mark.parametrize("arm", ["gate", "local_cfm", "pooled_cfm"])
def test_synthetic_two_epoch_runner_reload_and_reproducibility(tmp_path, arm):
    data, bank = synthetic()
    frozen_bank = {k: v.clone() for k, v in bank.items()}
    settings = {"learning_rate": .001, "weight_decay": .0001, "maximum_epochs": 2,
                "minimum_epochs": 2, "patience": 2, "min_delta": 1e-5,
                "batch_size": 33, "rows_per_root_per_batch": 1, "steps_per_epoch": 2, "gradient_clip_norm": 10}
    identity = {"arm": arm, "seed": 42, "synthetic_only": True}
    summaries = []
    for name in ("a", "b"):
        runner.set_determinism(42)
        model = runner.build_model(arm, [3, 4, 5]).cuda()
        result = runner.fit(model, arm, data, data, bank, settings, 42, tmp_path/name, identity)
        assert result["epochs"] == 2
        assert result["reload_validation_objective"] == result["best_validation_objective"]
        summaries.append(result)
    assert summaries[0] == summaries[1]
    assert all(torch.equal(bank[k], frozen_bank[k]) for k in bank)
    with pytest.raises(FileExistsError): runner.fit(model, arm, data, data, bank, settings, 42, tmp_path/"a", identity)


def test_authorization_and_gpu_fail_closed(monkeypatch):
    with pytest.raises(ProtocolError): runner.execute(ROOT)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ProtocolError): runner.cuda_manifest()


def test_real_data_verification_without_training():
    config, base, protocol, decoders, train, validation = runner.verified_data(ROOT)
    assert len(train) == 12153 and len(validation) == 3110
    assert len(decoders["experts"]) == 33 and protocol["parent_models"]["condition_dim"] == 3
    assert len(runner.category_counts(decoders)) == 3
    for seed in [42, 43, 44]:
        bank = runner.read_rows(base/f"Validation_Noise_seed{seed}.jsonl")
        assert [r["design_hash"] for r in bank] == [r["design_hash"] for r in validation]
    assert config["runtime"]["arms"] == ["gate", "local_cfm", "pooled_cfm"]

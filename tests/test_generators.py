"""Tests for the WP3 unified conditional generator adapters."""

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import torch
import yaml

from src.ai.dataset import generate_dataset
from src.ai.generator_dataset import run_wp1
from src.ai.generators import (
    CFMGenerator,
    CINNGenerator,
    CVAEGenerator,
    NFlowsGenerator,
    load_generator,
)
from src.ai.generators.data import build_tail_sampling_weights
from src.ai.wp3 import load_and_validate_lock, run_upstream_smoke, run_wp3


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "config" / "generator_v1_schema.yaml"
LOCK_PATH = REPO_ROOT / "config" / "third_party_models.lock.yaml"


MODEL_CONFIGS = {
    "cvae": {
        "enabled": True,
        "hidden_dims": [16, 16],
        "latent_dim": 3,
        "beta": 0.001,
        "learning_rate": 0.001,
    },
    "nflows": {
        "enabled": True,
        "num_layers": 2,
        "hidden_features": 16,
        "num_blocks": 1,
        "learning_rate": 0.001,
    },
    "cinn": {
        "enabled": True,
        "num_blocks": 2,
        "hidden_features": 16,
        "affine_clamping": 2.0,
        "learning_rate": 0.001,
    },
    "cfm": {
        "enabled": True,
        "hidden_dims": [16, 16],
        "sigma": 0.0,
        "ode_method": "rk4",
        "ode_steps": 2,
        "ode_rtol": 1e-5,
        "ode_atol": 1e-5,
        "learning_rate": 0.001,
    },
}


RUNTIME = {
    "seed": 7,
    "device": "cpu",
    "epochs": 1,
    "patience": 1,
    "gradient_clip_norm": 10.0,
}


@pytest.mark.parametrize(
    ("generator_class", "name"),
    [
        (CVAEGenerator, "cvae"),
        (NFlowsGenerator, "nflows"),
        (CINNGenerator, "cinn"),
        (CFMGenerator, "cfm"),
    ],
)
def test_generators_share_shape_seed_and_checkpoint_contract(
    generator_class, name, tmp_path
):
    model = generator_class(MODEL_CONFIGS[name], RUNTIME)
    condition = torch.tensor([0.1, -0.2])
    first = model.sample(condition, 6, seed=19)
    second = model.sample(condition, 6, seed=19)
    assert first.space == "normalized"
    assert first.values.shape == (6, 5)
    assert torch.equal(first.values, second.values)
    assert torch.isfinite(first.values).all()

    checkpoint = tmp_path / f"{name}.pt"
    model.save(checkpoint)
    restored = load_generator(checkpoint, device="cpu")
    after = restored.sample(condition, 6, seed=19)
    assert torch.equal(first.values, after.values)


@pytest.mark.parametrize(
    ("generator_class", "name"),
    [
        (CVAEGenerator, "cvae"),
        (NFlowsGenerator, "nflows"),
        (CINNGenerator, "cinn"),
        (CFMGenerator, "cfm"),
    ],
)
def test_generators_support_active10d_with_three_conditions(
    generator_class, name, tmp_path
):
    config = {
        **MODEL_CONFIGS[name],
        "design_dim": 10,
        "condition_dim": 3,
    }
    model = generator_class(config, RUNTIME, schema_version="generator_active10d_v1")
    condition = torch.tensor([0.1, -0.2, 0.3])
    generated = model.sample(condition, 4, seed=23)
    assert generated.values.shape == (4, 10)
    assert torch.isfinite(generated.values).all()

    checkpoint = tmp_path / f"{name}-10d.pt"
    model.save(checkpoint)
    restored = load_generator(checkpoint, device="cpu")
    after = restored.sample(condition, 4, seed=23)
    assert torch.equal(generated.values, after.values)


def test_flow_log_prob_and_cinn_inverse_are_finite():
    x = torch.randn(8, 5)
    condition = torch.randn(8, 2)
    nflows = NFlowsGenerator(MODEL_CONFIGS["nflows"], RUNTIME)
    assert torch.isfinite(nflows.log_prob(x, condition)).all()

    cinn = CINNGenerator(MODEL_CONFIGS["cinn"], RUNTIME)
    z, _ = cinn.forward_transform(x, condition)
    restored, _ = cinn.inverse_transform(z, condition)
    torch.testing.assert_close(restored, x, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(cinn.log_prob(x, condition)).all()


@pytest.mark.parametrize(
    ("generator_class", "name"),
    [(NFlowsGenerator, "nflows"), (CINNGenerator, "cinn")],
)
def test_flow_condition_softclip_and_bounded_output_survive_extreme_context(
    generator_class, name, tmp_path
):
    config = {
        **MODEL_CONFIGS[name],
        "condition_stabilization": {
            "enabled": True,
            "kind": "softclip",
            "limit": 3.0,
        },
        "bounded_output": {"enabled": True, "epsilon": 1.0e-4},
    }
    model = generator_class(config, RUNTIME)
    condition = torch.tensor([-2.12, 10.62])
    transformed = model._condition_for_model(condition)
    assert float(transformed.abs().max()) < 3.0
    generated = model.sample(condition, 16, seed=31)
    assert torch.isfinite(generated.values).all()
    assert float(generated.values.abs().max()) < 1.0
    assert generated.diagnostics["condition_raw_max_abs"] == pytest.approx(10.62)
    assert generated.diagnostics["condition_model_max_abs"] < 3.0

    checkpoint = tmp_path / f"{name}-stable.pt"
    model.save(checkpoint)
    restored = load_generator(checkpoint, device="cpu")
    after = restored.sample(condition, 16, seed=31)
    torch.testing.assert_close(after.values, generated.values)


@pytest.mark.parametrize(
    ("generator_class", "name"),
    [(NFlowsGenerator, "nflows"), (CINNGenerator, "cinn")],
)
def test_flow_adapters_expose_differentiable_latent_decode(generator_class, name):
    model = generator_class(MODEL_CONFIGS[name], RUNTIME)
    condition = torch.tensor([0.1, -0.2])
    latent = model.sample_latent(condition, 5, seed=29).requires_grad_(True)
    decoded = model.decode_latent(latent, condition)
    assert decoded.shape == (1, 5, 5)
    decoded.square().mean().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()


def test_fit_records_scheduler_and_converged_early_stopping():
    samples = [
        {
            "x": torch.zeros(5),
            "condition": torch.zeros(2),
            "sample_id": f"sample-{index}",
        }
        for index in range(8)
    ]
    loader = torch.utils.data.DataLoader(samples, batch_size=4, shuffle=False)
    runtime = {
        "seed": 7,
        "device": "cpu",
        "epochs": 6,
        "minimum_epochs": 2,
        "patience": 2,
        "min_delta": 0.0,
        "gradient_clip_norm": 10.0,
        "lr_scheduler": {
            "enabled": True,
            "patience": 0,
            "factor": 0.5,
            "min_lr": 1.0e-6,
        },
    }
    model_config = {**MODEL_CONFIGS["cvae"], "learning_rate": 0.0}
    model = CVAEGenerator(model_config, runtime)
    metadata = model.fit(loader, loader)
    assert metadata["early_stopped"] is True
    assert metadata["termination_reason"] == "early_stopping"
    assert metadata["epochs_completed"] == 3
    assert metadata["minimum_epochs"] == 2
    assert all("learning_rate_after_scheduler" in row for row in metadata["history"])


def test_tail_sampling_weights_emphasize_extreme_conditions():
    conditions = torch.tensor(
        [[0.0, 0.0], [0.2, -0.1], [0.5, 0.4], [3.0, -2.0]],
        dtype=torch.float32,
    )
    weights = build_tail_sampling_weights(
        conditions, {"quantile": 0.75, "maximum_weight": 5.0, "power": 2.0}
    )

    assert weights.shape == (4,)
    assert weights[-1] == pytest.approx(5.0)
    assert torch.all(weights >= 1.0)


def test_lock_and_upstream_core_smoke_pass():
    lock = load_and_validate_lock(LOCK_PATH, REPO_ROOT)
    assert lock["models"]["nflows"]["commit"] == (
        "64b856c081e5f07521b32be99da262e8338fbfe8"
    )
    config = {"models": MODEL_CONFIGS}
    smoke = run_upstream_smoke(config)
    assert smoke["status"].tolist() == ["passed"] * 4
    assert smoke.loc[smoke["model"] == "cinn", "inverse_max_error"].iloc[0] < 1e-5


def _build_small_wp1_data(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    generate_dataset(180, output_dir=str(raw_dir), seed=31, verbose=False)
    config = {
        "dataset_version": "wp3_test_dataset_v1",
        "schema": str(SCHEMA_PATH),
        "source": {
            "csv": str(raw_dir / "dataset_180.csv"),
            "source_id": "test/dataset_180.csv",
        },
        "output": {
            "directory": "unused",
            "split_csv": "unused.csv",
            "refuse_overwrite": True,
        },
        "split": {
            "seed": 32,
            "train_fraction": 0.8,
            "val_fraction": 0.1,
            "test_fraction": 0.1,
        },
        "parity": {
            "thrust_absolute_tolerance": 1e-10,
            "thrust_relative_tolerance": 1e-10,
            "tsfc_absolute_tolerance": 1e-12,
            "tsfc_relative_tolerance": 1e-10,
        },
    }
    path = tmp_path / "wp1.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return run_wp1(
        path, repo_root=REPO_ROOT, output_dir_override=tmp_path / "generator_data"
    )


def test_wp3_small_end_to_end_trains_all_adapters(tmp_path):
    data_dir = _build_small_wp1_data(tmp_path)
    target_path = tmp_path / "targets.csv"
    pd.DataFrame(
        [{"target_id": "test-target", "thrust_N": 50.0, "tsfc_kg_N_s": 3e-5}]
    ).to_csv(target_path, index=False)
    output_dir = tmp_path / "wp3"
    config = {
        "version": "wp3_test_v1",
        "lockfile": str(LOCK_PATH),
        "data": {
            "dataset_csv": str(data_dir / "generator_dataset_v1_valid.csv"),
            "schema": str(SCHEMA_PATH),
            "normalizer": str(data_dir / "normalizer_v1.json"),
            "target_suite": str(target_path),
            "batch_size": 32,
            "num_workers": 0,
            "subset_seed": 44,
            "max_train_samples": 100,
            "max_val_samples": 16,
        },
        "runtime": {
            "seeds": [7],
            "device": "cpu",
            "epochs": 1,
            "patience": 1,
            "gradient_clip_norm": 10.0,
            "checkpoint_sample_count": 8,
        },
        "models": MODEL_CONFIGS,
        "output": {"directory": str(output_dir), "refuse_overwrite": True},
    }
    config_path = tmp_path / "wp3.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    result = run_wp3(
        config_path, repo_root=REPO_ROOT, output_dir_override=output_dir
    )
    summary = pd.read_csv(result / "seed_summary.csv")
    assert set(summary["model"]) == {"cvae", "nflows", "cinn", "cfm"}
    assert summary["checkpoint_reproducible"].all()
    assert summary["sample_finite"].all()
    assert (result / "wp3_manifest.yaml").is_file()
    assert (result / "upstream_smoke.csv").is_file()
    for model in summary["model"]:
        assert (result / model / "seed_7" / "best.pt").is_file()
        assert (result / model / "seed_7" / "last.pt").is_file()

    with pytest.raises(FileExistsError):
        run_wp3(
            config_path, repo_root=REPO_ROOT, output_dir_override=output_dir
        )


def test_wp3_cli_is_directly_executable():
    completed = subprocess.run(
        [sys.executable, "scripts/run_wp3_generators.py", "--help"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr
    assert "open-source conditional generator" in completed.stdout

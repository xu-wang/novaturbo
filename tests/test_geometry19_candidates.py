"""WP19-23B fixed-raw filtering and D-Flow contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import src.ai.geometry19_candidates as candidates
from src.ai.geometry19_candidates import (
    CandidateBatch,
    filter_candidate_batch,
    run_candidate_protocol,
    run_paired_candidate_audit,
    sample_device_invariant_sources,
)
from src.physics import ModelConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_source_sampling_is_fixed_by_cpu_stream_before_device_transfer() -> None:
    generator = SimpleNamespace(
        design_dim=19,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    actual = sample_device_invariant_sources(generator, 256, 20260817)
    torch_generator = torch.Generator(device="cpu").manual_seed(20260817)
    expected = torch.randn(256, 19, generator=torch_generator, dtype=generator.dtype)
    assert actual.device.type == "cpu"
    assert torch.equal(actual, expected)


def _reference_normalized(schema: dict) -> np.ndarray:
    values = []
    for item in schema["design_variables"]:
        lower, upper = item["bounds"]
        reference = item["reference_value"]
        values.append(2.0 * (reference - lower) / (upper - lower) - 1.0)
    return np.asarray(values, dtype=float)


def test_filter_preserves_256_and_power_of_two_floor() -> None:
    schema = yaml.safe_load(
        (REPO_ROOT / "config/physics/geometry19_schema_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = ModelConfig.from_yaml(
        REPO_ROOT / "config/physics/geometry19_calibration_bootstrap_v1.yaml"
    )
    reference = _reference_normalized(schema)
    matrix = np.repeat(reference[None, :], 256, axis=0)
    # Make 31 unique and valid, leave one exact duplicate, and make 224 out of bounds.
    for index in range(31):
        matrix[index, 0] += index * 1.0e-5
    matrix[31] = matrix[0]
    matrix[32:, 0] = 1.1
    batch = CandidateBatch(
        mode="direct",
        candidate_ids=np.arange(256),
        initial_sources=np.zeros((256, 19)),
        direct_normalized=matrix.copy(),
        active_normalized=matrix,
        refined_sources=None,
        dflow_history=(),
    )
    rows, summary = filter_candidate_batch(batch, schema, config)
    assert summary["n_raw"] == 256
    assert summary["n_valid"] == 31
    assert summary["n_keep"] == 16
    assert summary["failure_counts"] == {
        "VALID": 31,
        "DUPLICATE_CANDIDATE": 1,
        "OUT_OF_SCHEMA_BOUNDS": 224,
    }
    assert [row["candidate_id"] for row in rows if row["kept"]] == list(range(16))


def test_integer_coordinates_are_deterministically_mapped() -> None:
    schema = yaml.safe_load(
        (REPO_ROOT / "config/physics/geometry19_schema_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = ModelConfig.from_yaml(
        REPO_ROOT / "config/physics/geometry19_calibration_bootstrap_v1.yaml"
    )
    reference = _reference_normalized(schema)
    matrix = np.repeat(reference[None, :], 256, axis=0)
    for index in range(256):
        matrix[index, 0] += index * 1.0e-6
    batch = CandidateBatch(
        mode="dflow",
        candidate_ids=np.arange(256),
        initial_sources=np.zeros((256, 19)),
        direct_normalized=matrix,
        active_normalized=matrix,
        refined_sources=np.zeros((256, 19)),
        dflow_history=(),
    )
    rows, summary = filter_candidate_batch(batch, schema, config)
    integer_names = {
        item["name"] for item in schema["design_variables"] if item["dtype"] == "int64"
    }
    assert summary["n_keep"] == 256
    for row in rows:
        assert all(isinstance(row["geometry19"][name], int) for name in integer_names)


def test_discrete_coordinates_are_unconditionally_projected_to_nearest_integer() -> None:
    schema = yaml.safe_load(
        (REPO_ROOT / "config/physics/geometry19_schema_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = ModelConfig.from_yaml(
        REPO_ROOT / "config/physics/geometry19_calibration_bootstrap_v1.yaml"
    )
    reference = _reference_normalized(schema)
    matrix = np.repeat(reference[None, :], 256, axis=0)
    blade = schema["design_variables"][5]
    lower, upper = blade["bounds"]
    for row_index, offset in enumerate((0.25, 0.250001)):
        physical = blade["reference_value"] + offset
        matrix[row_index, 5] = 2.0 * (physical - lower) / (upper - lower) - 1.0
    matrix[1, 0] += 1.0e-5
    batch = CandidateBatch(
        mode="direct",
        candidate_ids=np.arange(256),
        initial_sources=np.zeros((256, 19)),
        direct_normalized=matrix.copy(),
        active_normalized=matrix,
        refined_sources=None,
        dflow_history=(),
    )
    rows, summary = filter_candidate_batch(batch, schema, config)
    assert rows[0]["valid"] is True
    assert rows[0]["geometry19"]["compressor_blade_count"] == 12
    assert rows[1]["valid"] is True
    assert rows[1]["geometry19"]["compressor_blade_count"] == 12
    assert summary["discrete_mapping"] == {
        "policy": "unconditional_nearest_integer",
        "absolute_tolerance_count": None,
        "boundary_inclusive": True,
        "tie_break": "numpy_rint_nearest_even",
    }


def test_explicit_quarter_count_tolerance_remains_available_for_audits() -> None:
    schema = yaml.safe_load(
        (REPO_ROOT / "config/physics/geometry19_schema_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = ModelConfig.from_yaml(
        REPO_ROOT / "config/physics/geometry19_calibration_bootstrap_v1.yaml"
    )
    matrix = np.repeat(_reference_normalized(schema)[None, :], 256, axis=0)
    blade = schema["design_variables"][5]
    lower, upper = blade["bounds"]
    physical = blade["reference_value"] + 0.250001
    matrix[0, 5] = 2.0 * (physical - lower) / (upper - lower) - 1.0
    batch = CandidateBatch(
        mode="direct",
        candidate_ids=np.arange(256),
        initial_sources=np.zeros((256, 19)),
        direct_normalized=matrix.copy(),
        active_normalized=matrix,
        refined_sources=None,
        dflow_history=(),
    )
    rows, summary = filter_candidate_batch(
        batch, schema, config, discrete_tolerance_count=0.25
    )
    assert rows[0]["failure_code"] == "INVALID_DISCRETE_COORDINATE"
    assert summary["discrete_mapping"]["policy"] == (
        "nearest_integer_with_frozen_absolute_tolerance"
    )


@pytest.mark.parametrize("mode", ["direct", "dflow"])
def test_production_raw_rows_include_provenance_dflow_metrics_and_model_hashes(
    tmp_path: Path, monkeypatch, mode: str
) -> None:
    schema = yaml.safe_load(
        (REPO_ROOT / "config/physics/geometry19_schema_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    matrix = np.repeat(_reference_normalized(schema)[None, :], 256, axis=0)
    for index in range(256):
        matrix[index, 0] += index * 1.0e-6
    metric = {
        "surrogate_before": {"objective": 2.0},
        "surrogate_after": {"objective": 1.0},
        "source_displacement": {"linf": 0.1, "l2": 0.2, "rms": 0.03},
        "normalized_design_displacement": {
            "linf": 0.01,
            "l2": 0.02,
            "rms": 0.003,
        },
    }

    def generate(*args, **kwargs):
        dflow = kwargs["mode"] == "dflow"
        return CandidateBatch(
            mode=kwargs["mode"],
            candidate_ids=np.arange(256),
            initial_sources=np.zeros((256, 19)),
            direct_normalized=matrix.copy(),
            active_normalized=matrix,
            refined_sources=np.ones((256, 19)) if dflow else None,
            dflow_history=(),
            dflow_candidate_metrics=(
                tuple(dict(metric) for _ in range(256)) if dflow else ()
            ),
            dflow_model_hashes=(
                {
                    "forward": {"model_tag": "forward", "bundle_sha256": "f" * 64},
                    "feasibility": {
                        "model_tag": "feasibility",
                        "bundle_sha256": "e" * 64,
                    },
                }
                if dflow
                else {}
            ),
            generator_model_tag="test_geometry19_generator",
        )

    monkeypatch.setattr(candidates, "generate_candidate_batch", generate)
    output = tmp_path / "production"
    summary = run_candidate_protocol(
        REPO_ROOT,
        REPO_ROOT / "config/inverse_design/targets/wp19_23_smoke_test0_v1.yaml",
        mode=mode,
        device="cpu",
        output_root=output,
        generator_checkpoint=REPO_ROOT
        / "config/physics/wp19_23_inverse_closure_v1.yaml",
    )
    raw = json.loads((output / "raw_candidates.jsonl").read_text().splitlines()[0])
    required = {
        "target_id",
        "candidate_id",
        "source_id",
        "source_seed",
        "model_tag",
        "checkpoint_hash",
        "sampling_mode",
        "x_19_raw",
    }
    assert required <= set(raw)
    assert raw["sampling_mode"] == mode
    if mode == "dflow":
        assert raw["surrogate_before"]["objective"] == 2.0
        assert raw["surrogate_after"]["objective"] == 1.0
        assert raw["source_displacement"]["linf"] == 0.1
        assert raw["normalized_design_displacement"]["linf"] == 0.01
        assert set(raw["model_hashes"]) == {
            "generator",
            "forward",
            "feasibility",
            "ranking_forward",
        }
        assert summary["hashes"]["forward_checkpoint_hash"] == "f" * 64
        assert summary["hashes"]["feasibility_checkpoint_hash"] == "e" * 64
    else:
        assert raw["surrogate_before"] is None
        assert raw["surrogate_after"] is None
        assert raw["source_displacement"] is None
        assert raw["normalized_design_displacement"] is None
        assert set(raw["model_hashes"]) == {"generator", "ranking_forward"}
        assert summary["hashes"]["forward_checkpoint_hash"] == raw["model_hashes"][
            "ranking_forward"
        ]["bundle_sha256"]
        assert summary["hashes"]["feasibility_checkpoint_hash"] is None
    assert summary["candidate_protocol"]["constraint_eligible_sort"] == (
        "predicted_rho_ascending_then_candidate_id"
    )
    assert summary["candidate_protocol"]["surrogate_hit_alignment"][
        "ranked_candidates"
    ] == 256
    assert summary["development_overrides"]["forward_model_tag"] == (
        "performance3_log"
    )


def test_offline_paired_audit_generates_once_and_keeps_one_candidate_id_set(
    tmp_path: Path, monkeypatch
) -> None:
    schema = yaml.safe_load(
        (REPO_ROOT / "config/physics/geometry19_schema_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    reference = _reference_normalized(schema)
    direct = np.repeat(reference[None, :], 256, axis=0)
    dflow = direct.copy()
    for index in range(256):
        direct[index, 0] += index * 1.0e-6
        dflow[index, 0] += index * 2.0e-6
    calls = []

    def generate(*args, **kwargs):
        calls.append(kwargs["mode"])
        return CandidateBatch(
            mode="dflow",
            candidate_ids=np.arange(256),
            initial_sources=np.zeros((256, 19)),
            direct_normalized=direct,
            active_normalized=dflow,
            refined_sources=np.ones((256, 19)),
            dflow_history=(),
            generator_model_tag="test_geometry19_generator",
        )

    monkeypatch.setattr(candidates, "generate_candidate_batch", generate)
    output = tmp_path / "paired"
    summary = run_paired_candidate_audit(
        REPO_ROOT,
        REPO_ROOT / "config/inverse_design/targets/wp19_23_smoke_test0_v1.yaml",
        device="cpu",
        output_root=output,
    )
    assert calls == ["dflow"]
    assert summary["generator_call_count"] == 1
    assert summary["sampling_mode"] == "paired"
    assert summary["candidate_protocol"]["n_keep"] == 256
    pair_rows = [
        json.loads(line)
        for line in (output / "paired_candidate_ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(pair_rows) == 256
    assert all(row["paired_kept"] for row in pair_rows)
    raw = json.loads((output / "raw_candidates.jsonl").read_text().splitlines()[0])
    assert set(raw["x_19_raw"]) == {"direct", "dflow"}

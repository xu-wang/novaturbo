"""Tests for development-only V2.2 bridge-root expansion."""

from __future__ import annotations

from collections import Counter
import numpy as np
import json
from hashlib import sha256
from pathlib import Path

from src.physics.geometry19_v2_2_bridge_root_expansion import (
    _assign_development_folds,
    _select_bridge_roots,
    _select_nullspace_roots,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bridge_selection_is_deterministic_and_separated() -> None:
    centers = {
        f"root-{index}": np.asarray(value, dtype=float)
        for index, value in enumerate(
            ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0])
        )
    }
    selected, candidates = _select_bridge_roots(
        centers,
        segment_id="segment-a",
        interpolation_fractions=[0.35, 0.50, 0.65],
        bridge_root_count=2,
        minimum_endpoint_l2=0.9,
        minimum_l2_to_existing=0.2,
        minimum_l2_between_bridges=0.2,
    )
    assert candidates
    assert len(selected) == 2
    assert len({item["root_seed_id"] for item in selected}) == 2
    assert all(item["minimum_l2_to_existing"] >= 0.2 for item in selected)
    assert all(item["minimum_l2_at_selection"] >= 0.2 for item in selected)


def test_development_folds_keep_global_roots_together() -> None:
    variables = [
        {"name": "x", "bounds": [0.0, 1.0], "dtype": "float64"},
        {"name": "y", "bounds": [0.0, 1.0], "dtype": "float64"},
    ]
    rows = []
    for root_index in range(15):
        for segment in ("a", "b"):
            rows.append(
                {
                    "root_seed_id": f"root-{root_index:02d}",
                    "assigned_segment": segment,
                    "geometry19": {
                        "x": root_index / 15.0,
                        "y": (root_index % 5) / 5.0,
                    },
                }
            )
    assignments, summaries, metadata = _assign_development_folds(
        rows,
        variables,
        ("a", "b"),
        fold_count=5,
        distance_strata=3,
    )
    assert len(assignments) == 15
    assert len(summaries) == 5
    assert metadata["fold_count"] == 5
    assert all(
        summary["validation_roots_by_target_segment"] == {"a": 3, "b": 3}
        for summary in summaries
    )


def test_nullspace_selection_moves_away_without_linear_performance_drift() -> None:
    class Domain:
        log_scales = (1.0, 1.0, 1.0)

        @staticmethod
        def _scaled_log_point(performance):
            return tuple(np.log(performance[name]) for name in ("thrust_N", "tsfc_kg_N_s", "mass_kg"))

    variables = [
        {"name": f"x{index}", "bounds": [0.0, 1.0], "dtype": "float64"}
        for index in range(5)
    ]
    rows_by_root = {}
    centers = {}
    for root_index in range(3):
        rows = []
        for sample_index in range(8):
            design = {
                f"x{column}": 0.2 + 0.2 * root_index + 0.002 * sample_index * (column + 1)
                for column in range(5)
            }
            rows.append(
                {
                    "root_seed_id": f"root-{root_index}",
                    "design_hash": f"design-{root_index}-{sample_index}",
                    "geometry19": design,
                    "performance": {
                        "thrust_N": np.exp(design["x0"]),
                        "tsfc_kg_N_s": np.exp(design["x1"]),
                        "mass_kg": np.exp(design["x2"]),
                    },
                    "assigned_segment_assessment": {"normalized_distance": 0.0},
                }
            )
        rows_by_root[f"root-{root_index}"] = rows
        centers[f"root-{root_index}"] = np.mean(
            [[row["geometry19"][f"x{i}"] for i in range(5)] for row in rows], axis=0
        )
    selected, candidates = _select_nullspace_roots(
        centers,
        rows_by_root,
        variables,
        Domain(),
        dataset_version="test-v1",
        segment_id="a",
        bridge_root_count=2,
        directions_per_parent=4,
        nullspace_step_l2=[0.15],
        maximum_observations_per_root=8,
        ridge=1.0e-6,
        inverse_ridge=1.0e-6,
        minimum_l2_to_existing=0.08,
        minimum_l2_between_bridges=0.08,
        predicted_performance_drift_penalty=1.0,
        random_seed=42,
    )
    assert candidates
    assert len(selected) == 2
    assert all(item["minimum_l2_at_selection"] >= 0.08 for item in selected)


def test_final_root_fill_dataset_is_qualified_and_confirmation_free() -> None:
    evidence = REPO_ROOT / "data/validation/v2_2_bridge_root_expansion_v6"
    manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    development_path = evidence / "development_dataset.jsonl"
    qualified_path = evidence / "qualified_bridge_roots.jsonl"
    folds_path = evidence / "development_root_folds.json"
    development = [json.loads(line) for line in development_path.read_text().splitlines()]
    qualified = [json.loads(line) for line in qualified_path.read_text().splitlines()]
    confirmation = [
        json.loads(line)
        for line in (
            REPO_ROOT
            / "data/validation/v2_2_fresh_low_mass_confirmation_split_v1/splits/confirmation.jsonl"
        ).read_text().splitlines()
    ]
    assert manifest["counts"]["qualified_accepted_unique"] == 372
    assert manifest["counts"]["development_combined_unique"] == 3489
    assert manifest["counts"]["development_roots_by_target_segment"] == {
        "low_mass_low_to_mid": 27,
        "low_mass_mid_to_high": 27,
    }
    assert manifest["fold_gate_pass"] is True
    assert manifest["hashes"]["development_dataset_hash"] == sha256(
        development_path.read_bytes()
    ).hexdigest()
    assert manifest["hashes"]["qualified_bridge_roots_hash"] == sha256(
        qualified_path.read_bytes()
    ).hexdigest()
    assert manifest["hashes"]["development_root_folds_hash"] == sha256(
        folds_path.read_bytes()
    ).hexdigest()
    assert not (
        {row["root_seed_id"] for row in development}
        & {row["root_seed_id"] for row in confirmation}
    )
    counts = Counter(row["root_seed_id"] for row in qualified)
    assert len(counts) == 8
    assert min(counts.values()) >= 24
